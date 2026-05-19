"""Reviewer-oriented experiments on the updated v2 training database.

This script uses the consolidated metadata table built by
``build_training_database_v2.py``. It keeps the reviewer-requested comparison
focused on chemometric baselines (PCA-SVM and PLS-DA), an optimized 1D-CNN, a
standard Transformer, and the proposed MST. Tree ensembles are intentionally
not included in the main comparison because they were not requested by the
reviewer.

Protocol:
1. Train/validate/test on the supervised reference rows with ``split_v2`` equal
   to train/val/test.
2. Use a shared Raman-aware augmentation set for all model families.
3. Select hyperparameters within each model family by validation macro-F1.
4. Fine-tune the selected MST on labeled SHERLOC in-situ DUV spectra and
   evaluate on the held-out SHERLOC external-validation rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from train_model_comparison import (
    AUGMENTATION_PROTOCOL,
    GRID,
    PLSDA,
    RamanDataset,
    StandardTransformer,
    MaskedSpectralTransformer,
    augment_raman_features,
    class_weights,
    evaluate_arrays,
    fix_seed,
    flatten_features,
    predict_torch,
    preprocess_spectrum,
    train_torch_model,
)
from run_model_selection import TunedCNN


ROOT = Path(__file__).resolve().parents[1]
METADATA_V2 = ROOT / "data" / "metadata" / "metadata_training_database_v2_all_sources.csv"
OUT_DIR = ROOT / "results" / "review_updated_training_v2"
SEED = 2024
EXCLUDED_LABELS = {"Halides"}
PHYLL_LABELS = {"Clay", "Mica", "Serpentine"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def harmonize_label(value) -> str:
    label = str(value).strip()
    if label in PHYLL_LABELS:
        return "Phyllosilicates"
    if label == "Other Silicate":
        return "Other Silicates"
    return label


def resolve_path(path_value: str) -> Path:
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    candidate = ROOT.parent / path
    if candidate.exists():
        return candidate
    return path


def load_v2_metadata(metadata_file: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_file)
    df["model_label"] = df["label_category_final"].map(harmonize_label)
    df["supervised_label_usable_v2"] = df["supervised_label_usable_v2"].map(as_bool)
    df["duv_library_include"] = df["duv_library_include"].map(as_bool)
    df["resolved_file_path"] = df["file_path"].map(lambda p: str(resolve_path(str(p))))
    df["resolved_file_exists"] = df["resolved_file_path"].map(lambda p: Path(p).exists())
    df = df[df["resolved_file_exists"]].copy()
    df = df[df["supervised_label_usable_v2"]].copy()
    df = df[df["model_label"].notna() & df["model_label"].astype(str).str.len().gt(0)].copy()
    df = df[~df["model_label"].isin(EXCLUDED_LABELS)].copy()
    return df.reset_index(drop=True)


def reference_split(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df["split_v2"].isin(["train", "val", "test"])].copy()
    ref["split_main"] = ref["split_v2"]
    ref["source_type"] = ref["source_type_normalized"]
    ref["file_path"] = ref["resolved_file_path"]
    ref["parent_group"] = ref["parent_group"].fillna(ref["spectrum_id"])
    return ref.reset_index(drop=True)


def sherloc_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ft = df[df["split_v2"].eq("sherloc_finetune_pool")].copy()
    ext = df[df["split_v2"].eq("sherloc_external_validation")].copy()
    for sub in (ft, ext):
        sub["source_type"] = sub["source_type_normalized"]
        sub["file_path"] = sub["resolved_file_path"]
        sub["parent_group"] = sub["parent_group"].fillna(sub["spectrum_id"])
    return ft.reset_index(drop=True), ext.reset_index(drop=True)


def build_arrays(df: pd.DataFrame, cache_dir: Path, name: str, refresh: bool, baseline: str) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_baseline-{baseline}.npz"
    if cache_path.exists() and not refresh:
        cache = np.load(cache_path)
        return cache["features"], cache["masks"]

    features = []
    masks = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            log(f"Preprocessing {name}: {i}/{total}")
        x, _, mask = preprocess_spectrum(str(row["file_path"]), baseline=baseline, smooth=False)
        if str(row.get("source_domain", "")).startswith("sherloc"):
            low = GRID < 800.0
            x[low, :] = 0.0
            mask[low] = True
        features.append(x)
        masks.append(mask)

    np.savez_compressed(
        cache_path,
        features=np.stack(features).astype(np.float32),
        masks=np.stack(masks).astype(bool),
        grid=GRID.astype(np.float32),
    )
    return np.stack(features).astype(np.float32), np.stack(masks).astype(bool)


def make_balanced_augmented_train(
    x_train: np.ndarray,
    masks_train: np.ndarray,
    y_train: np.ndarray,
    min_per_class: int,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    xs = [x_train]
    ms = [masks_train]
    ys = [y_train]
    rows = []
    for cls in sorted(np.unique(y_train)):
        idx = np.where(y_train == cls)[0]
        target_n = min(max(len(idx), min_per_class), max_per_class)
        need = max(0, target_n - len(idx))
        aug_x = []
        aug_m = []
        for _ in range(need):
            src = int(rng.choice(idx))
            aug_x.append(augment_raman_features(x_train[src].copy(), masks_train[src].copy()))
            aug_m.append(masks_train[src].copy())
        if aug_x:
            xs.append(np.stack(aug_x).astype(np.float32))
            ms.append(np.stack(aug_m).astype(bool))
            ys.append(np.full(len(aug_x), cls, dtype=np.int64))
        rows.append(
            {
                "label_id": int(cls),
                "original_train_count": int(len(idx)),
                "augmented_count": int(need),
                "final_train_count": int(len(idx) + need),
            }
        )
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ms).astype(bool),
        np.concatenate(ys).astype(np.int64),
        pd.DataFrame(rows),
    )


def sklearn_val_test(model, xt, yt, xv, yv, xs, ys) -> tuple[float, np.ndarray]:
    model.fit(xt, yt)
    val_probs = model.predict_proba(xv)
    val_pred = np.argmax(val_probs, axis=1)
    val_macro = float(f1_score(yv, val_pred, average="macro", zero_division=0))
    return val_macro, model.predict_proba(xs)


def run_reviewer_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for stride, n_pca, c_value, gamma in itertools.product([4, 8], [40, 80], [1.0, 10.0], ["scale", 0.001]):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_use = min(n_pca, xt.shape[0] - 1, xt.shape[1])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_use, random_state=SEED)),
                ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)),
            ]
        )
        log(f"PCA-SVM trial: stride={stride}, pca={n_use}, C={c_value}, gamma={gamma}")
        val_macro, test_probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs, y_test)
        rows.append({"model": "pca_svm", "params": json.dumps({"stride": stride, "pca": n_use, "C": c_value, "gamma": gamma}), "val_macro_f1": val_macro, "test_probs": test_probs})

    for stride, n_comp in itertools.product([4, 8], [4, 8, 12]):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_use = min(n_comp, len(classes) - 1, xt.shape[0] - 1, xt.shape[1])
        model = PLSDA(n_components=n_use)
        log(f"PLS-DA trial: stride={stride}, components={n_use}")
        val_macro, test_probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs, y_test)
        rows.append({"model": "pls_da", "params": json.dumps({"stride": stride, "components": n_use}), "val_macro_f1": val_macro, "test_probs": test_probs})

    compact = [{k: v for k, v in row.items() if k != "test_probs"} for row in rows]
    pd.DataFrame(compact).to_csv(out_dir / "chemometric_hyperparameter_trials.csv", index=False, encoding="utf-8-sig")
    return rows


def best_by_model(rows: list[dict]) -> list[dict]:
    compact = pd.DataFrame([{k: v for k, v in row.items() if k != "test_probs"} for row in rows])
    selected = []
    for model, sub in compact.groupby("model"):
        idx = sub["val_macro_f1"].astype(float).idxmax()
        selected.append(rows[int(idx)])
    return selected


def infer_mst_config_from_state(state: dict) -> dict:
    d_model = int(state["head.weight"].shape[1])
    patch_size = int(state["patch.proj.weight"].shape[-1])
    layer_ids = set()
    for key in state:
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    layers = max(layer_ids) + 1 if layer_ids else 3
    return {"d_model": d_model, "layers": layers, "patch_size": patch_size}


def make_mst_from_checkpoint_config(checkpoint: Path, num_classes: int) -> tuple[MaskedSpectralTransformer, dict, dict]:
    state = torch.load(checkpoint, map_location="cpu")
    config = infer_mst_config_from_state(state)
    model = MaskedSpectralTransformer(num_classes=num_classes, **config)
    return model, state, config


def load_partial_mst_state(
    model: MaskedSpectralTransformer,
    state: dict,
    old_classes: list[str],
    new_classes: list[str],
) -> None:
    model_state = model.state_dict()
    transferable = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape and not k.startswith("head.")}
    model_state.update(transferable)
    if "head.weight" in state and "head.bias" in state:
        old_index = {c: i for i, c in enumerate(old_classes)}
        for new_i, cls in enumerate(new_classes):
            old_i = old_index.get(cls)
            if old_i is not None:
                model_state["head.weight"][new_i] = state["head.weight"][old_i]
                model_state["head.bias"][new_i] = state["head.bias"][old_i]
    model.load_state_dict(model_state)


def fine_tune_mst(
    base_checkpoint: Path,
    base_classes: list[str],
    all_classes: list[str],
    ft_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    cache_dir: Path,
    out_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    baseline: str,
    refresh: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_ft, masks_ft = build_arrays(ft_df, cache_dir, "sherloc_finetune_pool", refresh, baseline)
    x_ext, masks_ext = build_arrays(ext_df, cache_dir, "sherloc_external_validation", refresh, baseline)
    encoder = {c: i for i, c in enumerate(all_classes)}
    y_ft = ft_df["model_label"].map(encoder).to_numpy(dtype=np.int64)
    y_ext = ext_df["model_label"].map(encoder).to_numpy(dtype=np.int64)

    train_idx, val_idx = train_test_split(
        np.arange(len(ft_df)),
        test_size=0.2,
        random_state=SEED,
        stratify=y_ft if np.min(np.bincount(y_ft)) >= 2 else None,
    )

    model, base_state, inferred_config = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(model, base_state, base_classes, all_classes)
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.encoder.layers[-1], model.norm, model.head]:
        for param in module.parameters():
            param.requires_grad = True

    train_ds = RamanDataset(x_ft[train_idx], masks_ft[train_idx], y_ft[train_idx], augment=False)
    val_ds = RamanDataset(x_ft[val_idx], masks_ft[val_idx], y_ft[val_idx], augment=False)
    ext_ds = RamanDataset(x_ext, masks_ext, y_ext, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    ext_loader = DataLoader(ext_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds.y, len(all_classes)).to(device))
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
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "mst_sherloc_finetuned_v2.pth")
    pd.DataFrame(history).to_csv(out_dir / "mst_sherloc_finetune_history.csv", index=False, encoding="utf-8-sig")
    ext_probs, ext_true = predict_torch(model, ext_loader, device)
    metrics = evaluate_arrays(ext_true, ext_probs, all_classes, out_dir / "mst_sherloc_external_validation")

    zero_model, zero_state, _ = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(zero_model, zero_state, base_classes, all_classes)
    zero_model.to(device)
    zero_probs, _ = predict_torch(zero_model, ext_loader, device)
    zero_metrics = evaluate_arrays(ext_true, zero_probs, all_classes, out_dir / "mst_zero_shot_external_validation")

    pred = np.argmax(ext_probs, axis=1)
    zero_pred = np.argmax(zero_probs, axis=1)
    ext_rows = ext_df.copy()
    ext_rows["true_label"] = [all_classes[i] for i in ext_true]
    ext_rows["zero_shot_prediction"] = [all_classes[i] for i in zero_pred]
    ext_rows["zero_shot_confidence"] = np.max(zero_probs, axis=1)
    ext_rows["finetuned_prediction"] = [all_classes[i] for i in pred]
    ext_rows["finetuned_confidence"] = np.max(ext_probs, axis=1)
    ext_rows.to_csv(out_dir / "sherloc_external_validation_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_checkpoint": str(base_checkpoint),
        "inferred_base_mst_config": inferred_config,
        "base_classes": base_classes,
        "all_classes_after_sherloc": all_classes,
        "fine_tune_pool_rows": int(len(ft_df)),
        "fine_tune_train_rows": int(len(train_idx)),
        "fine_tune_validation_rows": int(len(val_idx)),
        "external_validation_rows": int(len(ext_df)),
        "fine_tune_mode": "last_transformer_block_norm_and_head",
        "sherloc_valid_range_cm-1": [800.0, 4000.0],
        "augmentation_on_sherloc": False,
        "best_internal_sherloc_val_macro_f1": best_val,
        "zero_shot_external_accuracy": zero_metrics["accuracy"],
        "zero_shot_external_macro_f1": zero_metrics["macro_f1"],
        "finetuned_external_accuracy": metrics["accuracy"],
        "finetuned_external_macro_f1": metrics["macro_f1"],
    }
    (out_dir / "sherloc_finetune_v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dir / "sherloc_finetune_v2_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Updated reviewer experiments using training database v2.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.out_dir / f"updated_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "_cache"
    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")

    full = load_v2_metadata(args.metadata_file)
    ref = reference_split(full)
    ft, ext = sherloc_split(full)
    ref.to_csv(run_dir / "reference_supervised_samples.csv", index=False, encoding="utf-8-sig")
    ft.to_csv(run_dir / "sherloc_finetune_pool_samples.csv", index=False, encoding="utf-8-sig")
    ext.to_csv(run_dir / "sherloc_external_validation_samples.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(ref["model_label"], ref["split_main"]).to_csv(run_dir / "reference_class_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(full["source_type_normalized"], full["split_v2"]).to_csv(run_dir / "all_source_by_split_v2.csv", encoding="utf-8-sig")

    encoder = LabelEncoder()
    ref["label_id"] = encoder.fit_transform(ref["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_arrays(ref, cache_dir, "reference_supervised_v2", args.refresh_cache, args.baseline)
    y = ref["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(ref["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(ref["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(ref["split_main"].eq("test").to_numpy())[0]
    x_train_aug, masks_train_aug, y_train_aug, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(run_dir / "shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")

    log(f"Reference train/val/test rows: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    log(f"Shared augmented train rows: {len(y_train_aug)}")
    baseline_rows = run_reviewer_baselines(
        x_train_aug, y_train_aug, x[val_idx], y[val_idx], x[test_idx], y[test_idx], classes, run_dir / "chemometric"
    )

    selected = []
    for row in best_by_model(baseline_rows):
        metrics = evaluate_arrays(y[test_idx], row["test_probs"], classes, run_dir / f"{row['model']}.selected.test")
        selected.append({"model": row["model"], "params": row["params"], "val_macro_f1": row["val_macro_f1"], **metrics})

    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)

    torch_trials = [
        ("cnn", TunedCNN(num_classes=len(classes), dropout=0.25), {"lr": 1e-3, "dropout": 0.25}),
        ("cnn", TunedCNN(num_classes=len(classes), dropout=0.40), {"lr": 3e-4, "dropout": 0.40}),
        ("standard_transformer", StandardTransformer(num_classes=len(classes), d_model=96, layers=3, patch_size=8), {"lr": 1e-4, "d_model": 96, "layers": 3, "patch_size": 8}),
        ("standard_transformer", StandardTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=8), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=96, layers=3, patch_size=8), {"lr": 1e-4, "d_model": 96, "layers": 3, "patch_size": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=8), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=4), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 4}),
    ]
    torch_rows = []
    for i, (name, model, params) in enumerate(torch_trials, start=1):
        trial_name = f"{name}_trial{i}"
        log(f"Training {trial_name}: {params}")
        metrics = train_torch_model(
            trial_name,
            model,
            train_ds,
            val_ds,
            test_ds,
            classes,
            run_dir / "torch",
            args.epochs,
            args.batch_size,
            params["lr"],
            device,
            augment=False,
        )
        torch_rows.append({"model": name, "trial": trial_name, "params": json.dumps(params), **metrics})
    torch_trials_df = pd.DataFrame(torch_rows)
    torch_trials_df.to_csv(run_dir / "torch_hyperparameter_trials.csv", index=False, encoding="utf-8-sig")
    best_mst_checkpoint = None
    for model_name, sub in torch_trials_df.groupby("model"):
        idx = sub["best_val_macro_f1"].astype(float).idxmax()
        row = torch_trials_df.loc[idx].to_dict()
        selected.append(
            {
                "model": model_name,
                "params": row["params"],
                "val_macro_f1": row["best_val_macro_f1"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
            }
        )
        if model_name == "mst":
            best_mst_checkpoint = run_dir / "torch" / f"{row['trial']}.pth"

    selected_df = pd.DataFrame(selected).sort_values("macro_f1", ascending=False)
    selected_df.to_csv(run_dir / "selected_model_test_summary.csv", index=False, encoding="utf-8-sig")

    sherloc_classes = sorted(set(classes) | set(ft["model_label"].unique()) | set(ext["model_label"].unique()))
    ft_summary = {}
    if best_mst_checkpoint is not None and len(ft) > 0 and len(ext) > 0:
        log("Fine-tuning selected MST on updated SHERLOC DUV pool")
        ft_summary = fine_tune_mst(
            best_mst_checkpoint,
            classes,
            sherloc_classes,
            ft,
            ext,
            cache_dir,
            run_dir / "sherloc_finetune",
            args.finetune_epochs,
            1e-4,
            args.batch_size,
            device,
            args.baseline,
            args.refresh_cache,
        )

    manifest = {
        "metadata_file": str(args.metadata_file),
        "reference_protocol": "supervised rows with split_v2 train/val/test; SHERLOC fine-tune and external-validation rows held out from base comparison",
        "reviewer_requested_models": ["pca_svm", "pls_da", "cnn", "standard_transformer", "mst"],
        "excluded_from_main_comparison": ["random_forest", "extra_trees"],
        "excluded_labels": sorted(EXCLUDED_LABELS),
        "label_harmonization": {"Clay/Mica/Serpentine": "Phyllosilicates", "Other Silicate": "Other Silicates"},
        "classes": classes,
        "sherloc_classes_after_head_expansion": sherloc_classes,
        "reference_rows": int(len(ref)),
        "sherloc_finetune_rows": int(len(ft)),
        "sherloc_external_validation_rows": int(len(ext)),
        "shared_training_augmentation": {
            "min_per_class": args.min_per_class,
            "max_per_class": args.max_per_class,
            "original_train_rows": int(len(train_idx)),
            "augmented_train_rows": int(len(y_train_aug)),
            "applied_to_all_base_models": True,
            "band_positions_shifted": False,
            "protocol": AUGMENTATION_PROTOCOL,
        },
        "torch_epochs": args.epochs,
        "sherloc_finetune_epochs": args.finetune_epochs,
        "device": str(device),
        "finetune_summary": ft_summary,
        "seed": SEED,
    }
    (run_dir / "review_updated_training_v2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(selected_df.to_string(index=False))
    print(f"Saved updated reviewer experiment to: {run_dir}")


if __name__ == "__main__":
    main()
