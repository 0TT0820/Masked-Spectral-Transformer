"""Focused MST tuning around the best-performing lightweight configuration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from train_model_comparison import (
    METADATA_FILE,
    MaskedSpectralTransformer,
    RamanDataset,
    build_cache,
    load_metadata,
    train_torch_model,
    fix_seed,
)
from run_model_selection import make_balanced_augmented_train


SEED = 2024
OUT_DIR = Path("results/mst_focused_tuning")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = OUT_DIR / f"curated_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata("curated", include_review_required=False, metadata_file=METADATA_FILE)
    enc = LabelEncoder()
    df["label_id"] = enc.fit_transform(df["model_label"])
    classes = list(enc.classes_)
    cache_path = build_cache(df, OUT_DIR / "_cache", False, "poly", False, "curated")
    cache = np.load(cache_path)
    x, masks, y = cache["features"], cache["masks"], df["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]

    x_train, masks_train, y_train, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], 200, 260
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(run_dir / "shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")

    train_ds = RamanDataset(x_train, masks_train, y_train, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)

    trials = [
        {"d_model": 96, "layers": 3, "patch_size": 8, "lr": 1e-4, "epochs": 100},
        {"d_model": 96, "layers": 3, "patch_size": 8, "lr": 5e-5, "epochs": 100},
        {"d_model": 96, "layers": 3, "patch_size": 8, "lr": 2e-4, "epochs": 100},
        {"d_model": 96, "layers": 4, "patch_size": 8, "lr": 1e-4, "epochs": 100},
        {"d_model": 128, "layers": 3, "patch_size": 8, "lr": 1e-4, "epochs": 100},
        {"d_model": 64, "layers": 3, "patch_size": 8, "lr": 1e-4, "epochs": 100},
    ]

    rows = []
    for i, params in enumerate(trials, start=1):
        name = f"mst_focused_trial{i}"
        log(f"Training {name}: {params}")
        model = MaskedSpectralTransformer(
            num_classes=len(classes),
            d_model=params["d_model"],
            layers=params["layers"],
            patch_size=params["patch_size"],
        )
        metrics = train_torch_model(
            name,
            model,
            train_ds,
            val_ds,
            test_ds,
            classes,
            run_dir / "torch",
            params["epochs"],
            24,
            params["lr"],
            device,
            augment=False,
        )
        rows.append({"trial": name, "params": json.dumps(params), **metrics})

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.to_csv(run_dir / "mst_focused_trials.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"Saved MST focused tuning to: {run_dir}")


if __name__ == "__main__":
    main()
