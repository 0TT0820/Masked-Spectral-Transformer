"""SHERLOC-specific preprocessing trials for MST fine-tuning.

The goal is to test whether conservative SHERLOC denoising improves
fine-tuning and external validation. The preprocessing is intentionally
restricted to intensity-domain operations that do not translate Raman band
centers:

- mask SHERLOC values below 800 cm-1;
- optional Hampel despiking for isolated outliers;
- optional small-window Savitzky-Golay smoothing;
- optional mild high-frequency damping by median replacement;
- recompute first derivative after denoising.

Reference-domain spectra are kept with the original preprocessing so this
script also reports whether SHERLOC fine-tuning degrades reference-domain test
performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_model_comparison import (
    GRID,
    RamanDataset,
    class_weights,
    predict_torch,
    preprocess_spectrum,
)

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from run_review_updated_training_v2 import (
    METADATA_V2,
    SEED,
    build_arrays,
    load_partial_mst_state,
    load_v2_metadata,
    make_mst_from_checkpoint_config,
    reference_split,
    sherloc_split,
)


ROOT = Path(__file__).resolve().parents[1]


PREPROCESSING_VARIANTS = {
    "raw_poly": {
        "baseline": "poly",
        "hampel_window": 0,
        "hampel_sigma": 0.0,
        "hampel_min_abs": 0.15,
        "savgol_window": 0,
        "savgol_polyorder": 2,
    },
    "despike_poly": {
        "baseline": "poly",
        "hampel_window": 9,
        "hampel_sigma": 6.0,
        "hampel_min_abs": 0.15,
        "savgol_window": 0,
        "savgol_polyorder": 2,
    },
    "despike_sg11_poly": {
        "baseline": "poly",
        "hampel_window": 9,
        "hampel_sigma": 6.0,
        "hampel_min_abs": 0.15,
        "savgol_window": 11,
        "savgol_polyorder": 2,
    },
    "despike_sg21_poly": {
        "baseline": "poly",
        "hampel_window": 9,
        "hampel_sigma": 6.0,
        "hampel_min_abs": 0.15,
        "savgol_window": 21,
        "savgol_polyorder": 2,
    },
    "despike_sg11_asls": {
        "baseline": "asls",
        "hampel_window": 9,
        "hampel_sigma": 6.0,
        "hampel_min_abs": 0.15,
        "savgol_window": 11,
        "savgol_polyorder": 2,
    },
}


def metrics_for(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    half = window // 2
    padded = np.pad(y, (half, half), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(padded[i : i + window])
    return out


def hampel_despike(y: np.ndarray, window: int, sigma: float, min_abs: float) -> tuple[np.ndarray, int]:
    if window <= 1 or sigma <= 0 or len(y) < window:
        return y.copy(), 0
    med = rolling_median(y, window)
    abs_dev = np.abs(y - med)
    local_mad = rolling_median(abs_dev, window)
    threshold = sigma * 1.4826 * np.maximum(local_mad, 1e-6)
    spike = (abs_dev > threshold) & (abs_dev > min_abs)
    out = y.copy()
    out[spike] = med[spike]
    return out, int(np.sum(spike))


def recompute_features(intensity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    intensity = np.maximum(intensity, 0.0)
    if np.any(valid):
        maxv = float(np.max(intensity[valid]))
        if maxv > 0:
            intensity = intensity / (maxv + 1e-12)
    intensity[~valid] = 0.0
    deriv = np.gradient(intensity, GRID).astype(np.float32)
    if np.any(valid):
        max_abs = float(np.max(np.abs(deriv[valid])))
        if max_abs > 1e-9:
            deriv = deriv / max_abs
    deriv[~valid] = 0.0
    valid_channel = valid.astype(np.float32)
    return np.stack([intensity.astype(np.float32), deriv.astype(np.float32), valid_channel], axis=-1)


def preprocess_sherloc(path: str, variant: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    x, _, mask = preprocess_spectrum(path, baseline=variant["baseline"], smooth=False)
    low = GRID < 800.0
    mask[low] = True
    valid = ~mask
    intensity = x[:, 0].copy()
    stats = {"spikes_replaced": 0}
    if np.any(valid):
        valid_idx = np.where(valid)[0]
        y = intensity[valid_idx].copy()
        y, n_spikes = hampel_despike(
            y,
            int(variant["hampel_window"]),
            float(variant["hampel_sigma"]),
            float(variant["hampel_min_abs"]),
        )
        stats["spikes_replaced"] = n_spikes
        if int(variant["savgol_window"]) > 0 and savgol_filter is not None:
            window = int(variant["savgol_window"])
            if window % 2 == 0:
                window += 1
            if len(y) > window:
                y = savgol_filter(y, window_length=window, polyorder=int(variant["savgol_polyorder"]))
        intensity[valid_idx] = y
    x = recompute_features(intensity, valid)
    return x, mask, stats


def build_sherloc_arrays(df: pd.DataFrame, cache_dir: Path, name: str, variant_name: str, refresh: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_{variant_name}.npz"
    stats_path = cache_dir / f"{name}_{variant_name}_preprocessing_stats.csv"
    if cache_path.exists() and stats_path.exists() and not refresh:
        cache = np.load(cache_path)
        return cache["features"], cache["masks"], pd.read_csv(stats_path)
    variant = PREPROCESSING_VARIANTS[variant_name]
    xs = []
    masks = []
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        x, mask, stats = preprocess_sherloc(str(row["file_path"]), variant)
        xs.append(x)
        masks.append(mask)
        rows.append({"spectrum_id": row["spectrum_id"], "variant": variant_name, **stats})
        if i == 1 or i % 100 == 0 or i == len(df):
            print(f"Preprocessed {name} {variant_name}: {i}/{len(df)}", flush=True)
    np.savez_compressed(cache_path, features=np.stack(xs).astype(np.float32), masks=np.stack(masks).astype(bool), grid=GRID)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    return np.stack(xs).astype(np.float32), np.stack(masks).astype(bool), stats_df


def make_model_pair(base_checkpoint: Path, base_classes: list[str], all_classes: list[str], device: torch.device):
    model, base_state, inferred = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(model, base_state, base_classes, all_classes)
    model.to(device)
    return model, inferred


def fine_tune_variant(
    base_checkpoint: Path,
    base_classes: list[str],
    all_classes: list[str],
    ft_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_x: np.ndarray,
    reference_masks: np.ndarray,
    reference_y: np.ndarray,
    cache_dir: Path,
    out_dir: Path,
    variant_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    refresh: bool,
) -> dict:
    variant_dir = out_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    x_ft, masks_ft, ft_stats = build_sherloc_arrays(ft_df, cache_dir, "sherloc_finetune_pool", variant_name, refresh)
    x_ext, masks_ext, ext_stats = build_sherloc_arrays(ext_df, cache_dir, "sherloc_external_validation", variant_name, refresh)

    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_ft = ft_df["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    y_ext = ext_df["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    train_idx, val_idx = train_test_split(
        np.arange(len(ft_df)),
        test_size=0.2,
        random_state=SEED,
        stratify=y_ft if np.min(np.bincount(y_ft)) >= 2 else None,
    )

    model, inferred = make_model_pair(base_checkpoint, base_classes, all_classes, device)
    for p in model.parameters():
        p.requires_grad = False
    for module in [model.norm, model.head]:
        for p in module.parameters():
            p.requires_grad = True

    train_loader = DataLoader(RamanDataset(x_ft[train_idx], masks_ft[train_idx], y_ft[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RamanDataset(x_ft[val_idx], masks_ft[val_idx], y_ft[val_idx]), batch_size=batch_size, shuffle=False)
    ext_loader = DataLoader(RamanDataset(x_ext, masks_ext, y_ext), batch_size=batch_size, shuffle=False)
    ref_loader = DataLoader(RamanDataset(reference_x, reference_masks, reference_y), batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_ft[train_idx], len(all_classes)).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, shifts, mask, y in train_loader:
            x, shifts, mask, y = x.to(device), shifts.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, shifts, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probs, val_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(val_true, np.argmax(val_probs, axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "internal_sherloc_val_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), variant_dir / "mst_sherloc_preprocessed_finetuned.pth")
    pd.DataFrame(history).to_csv(variant_dir / "history.csv", index=False, encoding="utf-8-sig")

    ext_probs, _ = predict_torch(model, ext_loader, device)
    ref_probs, _ = predict_torch(model, ref_loader, device)
    ext_metrics = metrics_for(y_ext, ext_probs)
    ref_metrics = metrics_for(reference_y, ref_probs)
    result = {
        "variant": variant_name,
        "baseline": PREPROCESSING_VARIANTS[variant_name]["baseline"],
        "hampel_window": PREPROCESSING_VARIANTS[variant_name]["hampel_window"],
        "savgol_window": PREPROCESSING_VARIANTS[variant_name]["savgol_window"],
        "fine_tune_mode": "norm_and_head_only",
        "lr": lr,
        "epochs": epochs,
        "inferred_base_config": json.dumps(inferred),
        "mean_spikes_replaced_finetune": float(ft_stats["spikes_replaced"].mean()),
        "mean_spikes_replaced_external": float(ext_stats["spikes_replaced"].mean()),
        "best_internal_sherloc_val_macro_f1": best_val,
        "external_accuracy": ext_metrics["accuracy"],
        "external_macro_f1": ext_metrics["macro_f1"],
        "external_weighted_f1": ext_metrics["weighted_f1"],
        "reference_test_accuracy_after": ref_metrics["accuracy"],
        "reference_test_macro_f1_after": ref_metrics["macro_f1"],
        "reference_test_weighted_f1_after": ref_metrics["weighted_f1"],
    }
    pd.DataFrame([result]).to_csv(variant_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pred_df = ext_df.copy()
    pred_df["true_label"] = ext_df["model_label"].to_numpy()
    pred_df["prediction"] = [all_classes[i] for i in np.argmax(ext_probs, axis=1)]
    pred_df["confidence"] = np.max(ext_probs, axis=1)
    pred_df.to_csv(variant_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHERLOC denoising/preprocessing fine-tuning trials.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--variants", nargs="+", default=list(PREPROCESSING_VARIANTS))
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or (args.run_dir / "sherloc_preprocessing_trials")
    cache_dir = args.cache_dir or (args.run_dir / "_sherloc_preprocessing_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_summary = json.loads((args.run_dir / "sherloc_finetune" / "sherloc_finetune_v2_summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(ft_summary["base_checkpoint"])
    base_classes = ft_summary["base_classes"]
    all_classes = ft_summary["all_classes_after_sherloc"]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    full = load_v2_metadata(args.metadata_file)
    ref = reference_split(full)
    ft, ext = sherloc_split(full)
    ref_test = ref[ref["split_main"].eq("test")].copy().reset_index(drop=True)
    ref_test = ref_test[ref_test["model_label"].isin(class_to_idx)].copy().reset_index(drop=True)
    reference_y = ref_test["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    reference_x, reference_masks = build_arrays(ref_test, cache_dir, "reference_test_original", False, "poly")

    before_model, _ = make_model_pair(base_checkpoint, base_classes, all_classes, device)
    ref_loader = DataLoader(RamanDataset(reference_x, reference_masks, reference_y), batch_size=args.batch_size, shuffle=False)
    before_ref_probs, _ = predict_torch(before_model, ref_loader, device)
    before_ref_metrics = metrics_for(reference_y, before_ref_probs)

    rows = []
    for variant in args.variants:
        if variant not in PREPROCESSING_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        print(f"Running SHERLOC preprocessing variant: {variant}", flush=True)
        rows.append(
            fine_tune_variant(
                base_checkpoint,
                base_classes,
                all_classes,
                ft,
                ext,
                ref_test,
                reference_x,
                reference_masks,
                reference_y,
                cache_dir,
                out_dir,
                variant,
                args.epochs,
                args.lr,
                args.batch_size,
                device,
                args.refresh_cache,
            )
        )
        pd.DataFrame(rows).to_csv(out_dir / "sherloc_preprocessing_trial_summary.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary["reference_test_accuracy_before"] = before_ref_metrics["accuracy"]
    summary["reference_test_macro_f1_before"] = before_ref_metrics["macro_f1"]
    summary["reference_test_weighted_f1_before"] = before_ref_metrics["weighted_f1"]
    summary.to_csv(out_dir / "sherloc_preprocessing_trial_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "run_dir": str(args.run_dir),
        "base_checkpoint": str(base_checkpoint),
        "variants": {k: PREPROCESSING_VARIANTS[k] for k in args.variants},
        "fine_tune_mode": "norm_and_head_only",
        "sherloc_valid_range_cm-1": [800.0, 4000.0],
        "band_center_shift_applied": False,
        "reference_test_before_finetune": before_ref_metrics,
        "device": str(device),
    }
    (out_dir / "sherloc_preprocessing_trial_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.sort_values("external_macro_f1", ascending=False).to_string(index=False))
    print(f"Saved SHERLOC preprocessing trials to: {out_dir}")


if __name__ == "__main__":
    main()
