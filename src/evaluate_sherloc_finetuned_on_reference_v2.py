"""Evaluate the SHERLOC-fine-tuned MST on non-SHERLOC reference data.

This diagnostic checks whether SHERLOC fine-tuning causes loss of performance
on the original reference-domain train/validation/test splits. It reports
metrics before and after fine-tuning for all reference rows and for each source
type separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from train_model_comparison import RamanDataset, predict_torch
from run_review_updated_training_v2 import (
    METADATA_V2,
    build_arrays,
    load_partial_mst_state,
    load_v2_metadata,
    make_mst_from_checkpoint_config,
    reference_split,
)


ROOT = Path(__file__).resolve().parents[1]


def metrics_for(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }


def predict_model(model, x: np.ndarray, masks: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int):
    ds = RamanDataset(x, masks, y, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return predict_torch(model, loader, device)


def evaluate_subset(
    df: pd.DataFrame,
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        idx = np.where(df["split_main"].eq(split).to_numpy())[0]
        if len(idx) == 0:
            continue
        before = metrics_for(y[idx], probs_before[idx])
        after = metrics_for(y[idx], probs_after[idx])
        rows.append({"subset": split, "source_type": "all_reference", "phase": "before_sherloc_finetune", **before})
        rows.append({"subset": split, "source_type": "all_reference", "phase": "after_sherloc_finetune", **after})

        for source, sub_idx_values in df.iloc[idx].groupby("source_type_normalized").groups.items():
            sub_idx = np.array(list(sub_idx_values), dtype=int)
            before = metrics_for(y[sub_idx], probs_before[sub_idx])
            after = metrics_for(y[sub_idx], probs_after[sub_idx])
            rows.append({"subset": split, "source_type": source, "phase": "before_sherloc_finetune", **before})
            rows.append({"subset": split, "source_type": source, "phase": "after_sherloc_finetune", **after})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "reference_metrics_before_after_sherloc_finetune.csv", index=False, encoding="utf-8-sig")

    pred_before = np.argmax(probs_before, axis=1)
    pred_after = np.argmax(probs_after, axis=1)
    pred_df = df.copy()
    pred_df["true_label_id"] = y
    pred_df["before_prediction_id"] = pred_before
    pred_df["before_confidence"] = np.max(probs_before, axis=1)
    pred_df["after_prediction_id"] = pred_after
    pred_df["after_confidence"] = np.max(probs_after, axis=1)
    pred_df["changed_after_finetune"] = pred_before != pred_after
    pred_df.to_csv(out_dir / "reference_predictions_before_after_sherloc_finetune.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHERLOC-fine-tuned MST on reference data.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "review_updated_training_v2" / "_cache")
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.run_dir / "reference_after_sherloc_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_summary = json.loads((args.run_dir / "sherloc_finetune" / "sherloc_finetune_v2_summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(ft_summary["base_checkpoint"])
    fine_checkpoint = args.run_dir / "sherloc_finetune" / "mst_sherloc_finetuned_v2.pth"
    base_classes = ft_summary["base_classes"]
    all_classes = ft_summary["all_classes_after_sherloc"]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    df = reference_split(load_v2_metadata(args.metadata_file))
    df = df[df["model_label"].isin(class_to_idx)].copy().reset_index(drop=True)
    y = df["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    x, masks = build_arrays(df, args.cache_dir, "reference_supervised_v2", args.refresh_cache, args.baseline)

    before_model, before_state, _ = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(before_model, before_state, base_classes, all_classes)
    before_model.to(device)

    after_model, _, _ = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    after_model.load_state_dict(torch.load(fine_checkpoint, map_location="cpu"))
    after_model.to(device)

    probs_before, _ = predict_model(before_model, x, masks, y, device, args.batch_size)
    probs_after, _ = predict_model(after_model, x, masks, y, device, args.batch_size)

    result = evaluate_subset(df, x, masks, y, probs_before, probs_after, out_dir)
    summary = {
        "run_dir": str(args.run_dir),
        "base_checkpoint": str(base_checkpoint),
        "fine_tuned_checkpoint": str(fine_checkpoint),
        "reference_rows": int(len(df)),
        "device": str(device),
    }
    (out_dir / "reference_finetune_evaluation_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(f"Saved reference-domain fine-tune diagnostic to: {out_dir}")


if __name__ == "__main__":
    main()
