"""Additional MST-focused trials for the updated reviewer v2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from train_model_comparison import MaskedSpectralTransformer, RamanDataset, fix_seed, train_torch_model
from run_review_updated_training_v2 import (
    SEED,
    build_arrays,
    load_v2_metadata,
    make_balanced_augmented_train,
    reference_split,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA_V2 = ROOT / "data" / "metadata" / "metadata_training_database_v2_all_sources.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extra MST trials for an existing updated v2 protocol.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "review_updated_training_v2" / "_cache")
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = reference_split(load_v2_metadata(args.metadata_file))
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_arrays(df, args.cache_dir, "reference_supervised_v2", args.refresh_cache, args.baseline)
    y = df["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]
    x_train_aug, masks_train_aug, y_train_aug, _ = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )

    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)

    trials = [
        ("mst_extra_patch4_lr1e4_d96_l3", {"d_model": 96, "layers": 3, "patch_size": 4, "lr": 1e-4}),
        ("mst_extra_patch4_lr1e4_d128_l4", {"d_model": 128, "layers": 4, "patch_size": 4, "lr": 1e-4}),
        ("mst_extra_patch4_lr5e5_d160_l4", {"d_model": 160, "layers": 4, "patch_size": 4, "lr": 5e-5}),
        ("mst_extra_patch8_lr1e4_d128_l4", {"d_model": 128, "layers": 4, "patch_size": 8, "lr": 1e-4}),
    ]
    rows = []
    for name, params in trials:
        print(f"Training {name}: {params}", flush=True)
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
            args.out_dir,
            args.epochs,
            args.batch_size,
            params["lr"],
            device,
            augment=False,
        )
        rows.append({"trial": name, "params": json.dumps(params), **metrics})
        pd.DataFrame(rows).to_csv(args.out_dir / "mst_extra_trials.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
