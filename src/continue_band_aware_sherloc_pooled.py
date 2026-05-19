"""Continue Band-aware MST pooled SHERLOC validation for additional seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from run_band_aware_mst_experiments import (
    make_pooled_sherloc,
    split_indices,
    train_sherloc_split,
)
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays
from run_review_updated_training_v2 import load_v2_metadata, reference_split
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue Band-aware MST SHERLOC pooled validation.")
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--band-run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--modes", nargs="+", default=["norm_attn_head", "last_block_norm_attn_head"])
    args = parser.parse_args()

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
    y = pooled["model_label"].map(class_to_idx).to_numpy(dtype="int64")

    out_dir = args.band_run_dir / "sherloc_pooled_random_validation"
    cache_dir = args.band_run_dir / "_sherloc_preprocessing_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, masks, stats = build_sherloc_arrays(
        pooled,
        cache_dir,
        "band_aware_sherloc_pooled_labeled",
        "despike_sg11_asls",
        False,
    )
    stats.to_csv(out_dir / "pooled_sherloc_preprocessing_stats.csv", index=False, encoding="utf-8-sig")

    summary_path = out_dir / "band_aware_pooled_random_validation_summary.csv"
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
        done = {(int(r["seed"]), r["mode"]) for r in rows}
    else:
        rows = []
        done = set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        train_idx, val_idx, split_note = split_indices(y, seed, 0.2)
        for mode in args.modes:
            if (seed, mode) in done:
                continue
            print(f"Continuing BandAwareMST pooled SHERLOC seed={seed}, mode={mode}", flush=True)
            res = train_sherloc_split(
                checkpoint,
                config,
                base_classes,
                all_classes,
                x,
                masks,
                y,
                train_idx,
                val_idx,
                mode,
                args.epochs,
                args.lr,
                args.batch_size,
                device,
                out_dir,
                f"seed{seed}",
            )
            res["seed"] = seed
            res["split_note"] = split_note
            rows.append(res)
            pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    aggregate = (
        summary.groupby("mode")[["accuracy", "macro_f1", "weighted_f1", "present_label_macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(out_dir / "band_aware_pooled_random_validation_aggregate.csv", index=False, encoding="utf-8-sig")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
