"""Hyperparameter search for Raman mineral-classification benchmarks.

The goal is a fair model comparison:
- one shared preprocessed split;
- one shared physics-aware augmented training set for every model;
- validation-set model selection;
- one final test-set report from the selected configuration.
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from train_model_comparison import (
    GRID,
    METADATA_FILE,
    OUT_DIR,
    PLSDA,
    RamanDataset,
    StandardTransformer,
    MaskedSpectralTransformer,
    augment_raman_features,
    build_cache,
    evaluate_arrays,
    flatten_features,
    load_metadata,
    predict_torch,
    train_torch_model,
    fix_seed,
)


SEED = 2024


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ms, axis=0).astype(bool),
        np.concatenate(ys, axis=0).astype(np.int64),
        pd.DataFrame(rows),
    )


class TunedCNN(nn.Module):
    def __init__(self, num_classes: int, in_chans: int = 3, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=21, padding=10),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=15, padding=7),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 160, kernel_size=11, padding=5),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(160, 224, kernel_size=7, padding=3),
            nn.BatchNorm1d(224),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(224, num_classes))

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.permute(0, 2, 1)))


def sklearn_val_test(
    model,
    x_train_flat: np.ndarray,
    y_train: np.ndarray,
    x_val_flat: np.ndarray,
    y_val: np.ndarray,
    x_test_flat: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    model.fit(x_train_flat, y_train)
    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(x_val_flat)
        val_pred = np.argmax(val_probs, axis=1)
    else:
        val_pred = model.predict(x_val_flat)
    val_macro = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    test_probs = model.predict_proba(x_test_flat)
    test_pred = np.argmax(test_probs, axis=1)
    test_macro = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
    return val_macro, test_macro, test_probs


def run_sklearn_search(
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

    # PCA-SVM grid. PCA dimensionality is capped by sample and feature counts.
    for stride, n_pca, c_value, gamma in itertools.product(
        [4, 8],
        [40, 80],
        [1.0, 10.0],
        ["scale", 0.001],
    ):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_comp = min(n_pca, xt.shape[0] - 1, xt.shape[1])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_comp, random_state=SEED)),
                ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)),
            ]
        )
        log(f"sklearn trial pca_svm stride={stride} pca={n_comp} C={c_value} gamma={gamma}")
        val_macro, test_macro, probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs, y_test)
        rows.append(
            {
                "model": "pca_svm",
                "params": json.dumps({"stride": stride, "pca": n_comp, "C": c_value, "gamma": gamma}),
                "val_macro_f1": val_macro,
                "test_macro_f1": test_macro,
                "test_probs": probs,
            }
        )

    for stride, n_comp in itertools.product([4, 8], [4, 8, 12]):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_use = min(n_comp, len(classes) - 1, xt.shape[0] - 1, xt.shape[1])
        model = PLSDA(n_components=n_use)
        log(f"sklearn trial pls_da stride={stride} components={n_use}")
        val_macro, test_macro, probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs, y_test)
        rows.append(
            {
                "model": "pls_da",
                "params": json.dumps({"stride": stride, "components": n_use}),
                "val_macro_f1": val_macro,
                "test_macro_f1": test_macro,
                "test_probs": probs,
            }
        )

    for model_name, estimator_cls in [("random_forest", RandomForestClassifier), ("extra_trees", ExtraTreesClassifier)]:
        for n_estimators, max_depth, min_leaf, max_features in itertools.product(
            [500],
            [None, 24],
            [1, 2],
            ["sqrt"],
        ):
            xt, xv, xs = flatten_features(x_train, 4), flatten_features(x_val, 4), flatten_features(x_test, 4)
            model = estimator_cls(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                max_features=max_features,
                class_weight="balanced" if model_name == "random_forest" else "balanced",
                random_state=SEED,
                n_jobs=1,
            )
            log(f"sklearn trial {model_name} depth={max_depth} leaf={min_leaf} max_features={max_features}")
            val_macro, test_macro, probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs, y_test)
            rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(
                        {
                            "stride": 4,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_features,
                        }
                    ),
                    "val_macro_f1": val_macro,
                    "test_macro_f1": test_macro,
                    "test_probs": probs,
                }
            )

    compact = [{k: v for k, v in row.items() if k != "test_probs"} for row in rows]
    pd.DataFrame(compact).sort_values(["model", "val_macro_f1"], ascending=[True, False]).to_csv(
        out_dir / "sklearn_hyperparameter_trials.csv", index=False, encoding="utf-8-sig"
    )
    return rows


def best_rows_by_model(rows: list[dict]) -> list[dict]:
    best = []
    for model, sub in pd.DataFrame([{k: v for k, v in r.items() if k != "test_probs"} for r in rows]).groupby("model"):
        idx = sub["val_macro_f1"].astype(float).idxmax()
        best.append(rows[int(idx)])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for Raman model comparison.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "model_selection")
    parser.add_argument("--label-scheme", choices=["curated", "original_major"], default="curated")
    parser.add_argument("--baseline", choices=["none", "poly", "asls"], default="poly")
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.out_dir / f"{args.label_scheme}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")

    df = load_metadata(args.label_scheme, include_review_required=False, metadata_file=args.metadata_file)
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["model_label"])
    classes = list(encoder.classes_)
    cache_path = build_cache(df, args.out_dir / "_cache", args.refresh_cache, args.baseline, False, args.label_scheme)
    cache = np.load(cache_path)
    x, masks, y = cache["features"], cache["masks"], df["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]

    x_train_aug, masks_train_aug, y_train_aug, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(run_dir / "shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")

    x_val, masks_val, y_val = x[val_idx], masks[val_idx], y[val_idx]
    x_test, masks_test, y_test = x[test_idx], masks[test_idx], y[test_idx]

    log(f"Original train rows: {len(train_idx)}; shared augmented train rows: {len(y_train_aug)}")
    log("Searching sklearn/chemometric baselines")
    rows = run_sklearn_search(x_train_aug, y_train_aug, x_val, y_val, x_test, y_test, classes, run_dir / "sklearn")

    selected = []
    for row in best_rows_by_model(rows):
        metrics = evaluate_arrays(y_test, row["test_probs"], classes, run_dir / f"{row['model']}.selected.test")
        selected.append(
            {
                "model": row["model"],
                "params": row["params"],
                "selection_metric": "validation_macro_f1",
                "val_macro_f1": row["val_macro_f1"],
                **metrics,
            }
        )

    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x_val, masks_val, y_val, augment=False)
    test_ds = RamanDataset(x_test, masks_test, y_test, augment=False)

    torch_trials = [
        ("cnn", TunedCNN(num_classes=len(classes), dropout=0.25), {"lr": 1e-3, "dropout": 0.25}),
        ("cnn", TunedCNN(num_classes=len(classes), dropout=0.40), {"lr": 3e-4, "dropout": 0.40}),
        ("standard_transformer", StandardTransformer(num_classes=len(classes), d_model=96, layers=3, patch_size=8), {"lr": 1e-4, "d_model": 96, "layers": 3, "patch": 8}),
        ("standard_transformer", StandardTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=8), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=96, layers=3, patch_size=8), {"lr": 1e-4, "d_model": 96, "layers": 3, "patch": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=8), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 8}),
        ("mst", MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=4), {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 4}),
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
    for model, sub in torch_trials_df.groupby("model"):
        idx = sub["best_val_macro_f1"].astype(float).idxmax()
        row = torch_trials_df.loc[idx].to_dict()
        selected.append(
            {
                "model": model,
                "params": row["params"],
                "selection_metric": "validation_macro_f1",
                "val_macro_f1": row["best_val_macro_f1"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
            }
        )

    selected_df = pd.DataFrame(selected).sort_values("macro_f1", ascending=False)
    selected_df.to_csv(run_dir / "selected_model_test_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "metadata_file": str(args.metadata_file),
        "label_scheme": args.label_scheme,
        "classes": classes,
        "selection_rule": "best validation macro-F1 within each model family; report selected test metrics",
        "shared_training_augmentation": {
            "min_per_class": args.min_per_class,
            "max_per_class": args.max_per_class,
            "original_train_rows": int(len(train_idx)),
            "augmented_train_rows": int(len(y_train_aug)),
            "applied_to_all_models": True,
            "band_positions_shifted": False,
        },
        "epochs_for_torch_trials": args.epochs,
        "batch_size": args.batch_size,
        "seed": SEED,
    }
    (run_dir / "model_selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(selected_df.to_string(index=False))
    print(f"Saved model-selection results to: {run_dir}")


if __name__ == "__main__":
    main()
