"""SHERLOC-domain adaptation experiments for Band-aware MST.

This script keeps the reference-domain Band-aware MST checkpoint fixed and
tests explicit SHERLOC fine-tuning strategies. The goal is to improve SHERLOC
macro-F1 with methods that are scientifically defensible for imbalanced,
noisy, in-situ Raman spectra:

- class-balanced sampling for the SHERLOC fine-tuning split;
- weighted cross entropy versus focal loss;
- optional last-block fine-tuning while preserving most spectral encoder
  weights;
- reporting both the full reference-label prediction space and the
  SHERLOC-domain candidate-label prediction space.

The script intentionally writes a separate result directory so these variants
can be reported as explicit method-improvement experiments, not as a silent
replacement of the original MST.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

from run_band_aware_mst_experiments import (
    BandAwareMST,
    load_partial_band_state,
    make_pooled_sherloc,
    split_indices,
)
from run_review_updated_training_v2 import load_v2_metadata, reference_split
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays
from train_model_comparison import RamanDataset, class_weights, predict_torch


@dataclass(frozen=True)
class AdaptationTrial:
    name: str
    mode: str
    loss: str
    lr: float
    sampler: str
    gamma: float = 1.5
    label_smoothing: float = 0.0


TRIALS = [
    AdaptationTrial("ce_lastblock_standard_sampling", "last_block_norm_attn_head", "ce", 5e-5, "standard"),
    AdaptationTrial("ce_lastblock_balanced_sampling", "last_block_norm_attn_head", "ce", 5e-5, "balanced"),
    AdaptationTrial("focal15_lastblock_balanced_sampling", "last_block_norm_attn_head", "focal", 5e-5, "balanced", 1.5),
    AdaptationTrial("focal20_lastblock_balanced_sampling", "last_block_norm_attn_head", "focal", 5e-5, "balanced", 2.0),
    AdaptationTrial("ce_smooth005_lastblock_balanced_sampling", "last_block_norm_attn_head", "ce", 5e-5, "balanced", label_smoothing=0.05),
    AdaptationTrial("focal15_all_low_lr_balanced_sampling", "all", "focal", 1e-5, "balanced", 1.5),
]


def set_trainable(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "last_block_norm_attn_head":
        modules = [model.encoder.layers[-1], model.norm, model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "norm_attn_head":
        modules = [model.norm, model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "all":
        modules = [model]
        extra = []
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
    for p in extra:
        p.requires_grad = True


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 1.5) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_loss(trial: AdaptationTrial, y_train: np.ndarray, num_classes: int, device: torch.device) -> nn.Module:
    weights = class_weights(y_train, num_classes).to(device)
    if trial.loss == "focal":
        return FocalLoss(weights, gamma=trial.gamma)
    if trial.loss == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=trial.label_smoothing)
    raise ValueError(f"Unknown loss: {trial.loss}")


def make_train_loader(x: np.ndarray, masks: np.ndarray, y: np.ndarray, batch_size: int, sampler: str) -> DataLoader:
    ds = RamanDataset(x, masks, y)
    if sampler == "standard":
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
    if sampler == "balanced":
        counts = np.bincount(y, minlength=int(np.max(y)) + 1).astype(np.float64)
        sample_weights = 1.0 / np.maximum(counts[y], 1.0)
        weighted_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(y),
            replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=weighted_sampler)
    raise ValueError(f"Unknown sampler: {sampler}")


def restrict_probs(probs: np.ndarray, allowed_idx: list[int]) -> np.ndarray:
    out = np.zeros_like(probs)
    out[:, allowed_idx] = probs[:, allowed_idx]
    denom = out.sum(axis=1, keepdims=True)
    zero = denom[:, 0] <= 0
    if np.any(zero):
        out[zero, allowed_idx] = 1.0 / len(allowed_idx)
        denom = out.sum(axis=1, keepdims=True)
    return out / denom


def metric_row(y_true: np.ndarray, probs: np.ndarray, labels: list[str], prediction_space: str) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "prediction_space": prediction_space,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "present_label_macro_f1": float(
            f1_score(y_true, pred, labels=np.unique(y_true), average="macro", zero_division=0)
        ),
        "predicted_labels": ";".join(sorted({labels[i] for i in np.unique(pred)})),
    }


def confidence_sweep(y_true: np.ndarray, probs: np.ndarray, labels: list[str], base_row: dict) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for threshold in np.linspace(0.0, 0.95, 20):
        keep = conf >= threshold
        row = {
            **base_row,
            "threshold": float(threshold),
            "coverage": float(np.mean(keep)),
            "accepted_n": int(np.sum(keep)),
        }
        if np.any(keep):
            row.update(
                {
                    "accuracy_on_accepted": float(accuracy_score(y_true[keep], pred[keep])),
                    "macro_f1_on_accepted": float(
                        f1_score(y_true[keep], pred[keep], average="macro", zero_division=0)
                    ),
                    "present_label_macro_f1_on_accepted": float(
                        f1_score(y_true[keep], pred[keep], labels=np.unique(y_true), average="macro", zero_division=0)
                    ),
                    "predicted_labels_on_accepted": ";".join(sorted({labels[i] for i in np.unique(pred[keep])})),
                }
            )
        else:
            row.update(
                {
                    "accuracy_on_accepted": np.nan,
                    "macro_f1_on_accepted": np.nan,
                    "present_label_macro_f1_on_accepted": np.nan,
                    "predicted_labels_on_accepted": "",
                }
            )
        rows.append(row)
    return rows


def train_one(
    checkpoint: Path,
    config: dict,
    base_classes: list[str],
    all_classes: list[str],
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trial: AdaptationTrial,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = BandAwareMST(
        num_classes=len(all_classes),
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        patch_size=int(config["patch_size"]),
        saliency_init=float(config.get("saliency_init", 1.0)),
    )
    load_partial_band_state(model, checkpoint, base_classes, all_classes)
    set_trainable(model, trial.mode)
    model.to(device)

    train_loader = make_train_loader(x[train_idx], masks[train_idx], y[train_idx], batch_size, trial.sampler)
    val_loader = DataLoader(RamanDataset(x[val_idx], masks[val_idx], y[val_idx]), batch_size=batch_size, shuffle=False)
    loss_fn = make_loss(trial, y[train_idx], len(all_classes), device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=trial.lr, weight_decay=1e-4)

    best_state = None
    best_val_macro = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xx, shifts, mask, yy in train_loader:
            xx, shifts, mask, yy = xx.to(device), shifts.to(device), mask.to(device), yy.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xx, shifts, mask), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probs, yy_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(yy_true, np.argmax(probs, axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    probs, yy_true = predict_torch(model, val_loader, device)
    present_train_idx = sorted(np.unique(y[train_idx]).tolist())
    full = metric_row(yy_true, probs, all_classes, "all_reference_classes")
    restricted_probs = restrict_probs(probs, present_train_idx)
    restricted = metric_row(yy_true, restricted_probs, all_classes, "sherloc_train_present_classes")
    base = {
        "seed": seed,
        "trial": trial.name,
        "mode": trial.mode,
        "loss": trial.loss,
        "gamma": trial.gamma,
        "label_smoothing": trial.label_smoothing,
        "lr": trial.lr,
        "sampler": trial.sampler,
        "epochs": epochs,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "best_val_macro_f1_during_training": best_val_macro,
        "train_present_labels": ";".join(all_classes[i] for i in present_train_idx),
    }
    rows = [{**base, **full}, {**base, **restricted}]
    sweep_rows = []
    for prediction_space, p in [
        ("all_reference_classes", probs),
        ("sherloc_train_present_classes", restricted_probs),
    ]:
        sweep_rows.extend(confidence_sweep(yy_true, p, all_classes, {**base, "prediction_space": prediction_space}))

    stem = f"seed{seed}_{trial.name}"
    pd.DataFrame(history).to_csv(out_dir / f"{stem}_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), out_dir / f"{stem}.pth")
    pred = np.argmax(restricted_probs, axis=1)
    report = classification_report(yy_true, pred, labels=np.arange(len(all_classes)), target_names=all_classes, zero_division=0)
    (out_dir / f"{stem}_restricted_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(yy_true, pred, labels=np.arange(len(all_classes)))
    pd.DataFrame(cm, index=all_classes, columns=all_classes).to_csv(
        out_dir / f"{stem}_restricted_confusion_matrix.csv", encoding="utf-8-sig"
    )
    return {"rows": rows}, sweep_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHERLOC-domain adaptation strategies for Band-aware MST.")
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--band-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--variant", default="despike_sg11_asls")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.band_run_dir / "band_aware_mst_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["best_checkpoint"])
    config = manifest["best_config"]

    ref = reference_split(load_v2_metadata(args.metadata_file))
    encoder = LabelEncoder()
    ref["label_id"] = encoder.fit_transform(ref["model_label"])
    base_classes = list(encoder.classes_)

    pooled = make_pooled_sherloc(args.metadata_file)
    all_classes = sorted(set(base_classes) | set(pooled["model_label"].unique()))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y = pooled["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    pooled.to_csv(args.out_dir / "pooled_sherloc_labeled_samples.csv", index=False, encoding="utf-8-sig")
    pd.Series(pooled["model_label"]).value_counts().rename_axis("model_label").reset_index(name="count").to_csv(
        args.out_dir / "pooled_sherloc_class_counts.csv", index=False, encoding="utf-8-sig"
    )

    x, masks, stats = build_sherloc_arrays(
        pooled,
        args.band_run_dir / "_sherloc_preprocessing_cache",
        "band_aware_sherloc_pooled_labeled",
        args.variant,
        args.refresh_cache,
    )
    stats.to_csv(args.out_dir / "pooled_sherloc_preprocessing_stats.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []
    sweep_rows = []
    split_rows = []
    for seed in args.seeds:
        train_idx, val_idx, split_note = split_indices(y, seed, 0.2)
        split_rows.append(
            {
                "seed": seed,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "split_note": split_note,
                "train_label_counts": json.dumps(pd.Series(y[train_idx]).value_counts().to_dict(), ensure_ascii=False),
                "val_label_counts": json.dumps(pd.Series(y[val_idx]).value_counts().to_dict(), ensure_ascii=False),
            }
        )
        for trial in TRIALS:
            print(f"SHERLOC adaptation seed={seed}, trial={trial.name}", flush=True)
            result, sweeps = train_one(
                checkpoint,
                config,
                base_classes,
                all_classes,
                x,
                masks,
                y,
                train_idx,
                val_idx,
                trial,
                seed,
                args.epochs,
                args.batch_size,
                device,
                args.out_dir,
            )
            summary_rows.extend(result["rows"])
            sweep_rows.extend(sweeps)
            pd.DataFrame(summary_rows).to_csv(args.out_dir / "sherloc_adaptation_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(sweep_rows).to_csv(args.out_dir / "sherloc_adaptation_threshold_sweep.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(split_rows).to_csv(args.out_dir / "pooled_sherloc_random_splits.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    aggregate = (
        summary.groupby(["trial", "prediction_space"])[
            ["accuracy", "macro_f1", "weighted_f1", "present_label_macro_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(args.out_dir / "sherloc_adaptation_aggregate.csv", index=False, encoding="utf-8-sig")
    sweep = pd.DataFrame(sweep_rows)
    key_thresholds = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95])
    key = sweep[sweep["threshold"].map(lambda v: bool(np.any(np.isclose(float(v), key_thresholds))))]
    key.to_csv(args.out_dir / "sherloc_adaptation_key_thresholds.csv", index=False, encoding="utf-8-sig")
    best_threshold = (
        sweep[sweep["coverage"] >= 0.75]
        .sort_values("macro_f1_on_accepted", ascending=False)
        .groupby(["trial", "prediction_space"], as_index=False)
        .first()
    )
    best_threshold.to_csv(args.out_dir / "sherloc_adaptation_best_threshold_min75coverage.csv", index=False, encoding="utf-8-sig")
    print(aggregate.to_string(index=False))
    print(f"Saved SHERLOC adaptation experiments to: {args.out_dir}")


if __name__ == "__main__":
    main()
