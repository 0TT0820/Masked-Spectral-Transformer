"""Confidence/rejection operating points for SHERLOC external predictions."""

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
    load_partial_mst_state,
    load_v2_metadata,
    make_mst_from_checkpoint_config,
    sherloc_split,
)
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays


CANDIDATE_SETS = {
    "all_classes": None,
    "sherloc_plausible": ["Carbonate", "Olivine", "Other Silicates", "Perchlorate", "Phosphate", "Pyroxene", "Sulfate"],
    "external_target_prior_sulfate_phosphate": ["Phosphate", "Sulfate"],
}


def restrict_probs(probs: np.ndarray, allowed_idx: list[int] | None) -> np.ndarray:
    if allowed_idx is None:
        return probs.copy()
    out = np.zeros_like(probs)
    out[:, allowed_idx] = probs[:, allowed_idx]
    denom = out.sum(axis=1, keepdims=True)
    zero = denom[:, 0] <= 0
    if np.any(zero):
        out[zero, allowed_idx] = 1.0 / len(allowed_idx)
        denom = out.sum(axis=1, keepdims=True)
    return out / denom


def sweep(y_true: np.ndarray, probs: np.ndarray, labels: list[str], candidate_set: str) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for threshold in np.linspace(0.0, 0.95, 20):
        keep = conf >= threshold
        row = {
            "candidate_set": candidate_set,
            "threshold": float(threshold),
            "coverage": float(np.mean(keep)),
            "accepted_n": int(np.sum(keep)),
            "rejected_n": int(np.sum(~keep)),
        }
        if np.any(keep):
            row.update(
                {
                    "accuracy_on_accepted": float(accuracy_score(y_true[keep], pred[keep])),
                    "macro_f1_union_on_accepted": float(f1_score(y_true[keep], pred[keep], average="macro", zero_division=0)),
                    "macro_f1_true_labels_on_accepted": float(
                        f1_score(y_true[keep], pred[keep], labels=np.unique(y_true), average="macro", zero_division=0)
                    ),
                    "predicted_labels_on_accepted": ";".join(sorted({labels[i] for i in np.unique(pred[keep])})),
                }
            )
        else:
            row.update(
                {
                    "accuracy_on_accepted": np.nan,
                    "macro_f1_union_on_accepted": np.nan,
                    "macro_f1_true_labels_on_accepted": np.nan,
                    "predicted_labels_on_accepted": "",
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SHERLOC confidence/rejection operating points.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", default="despike_sg11_asls")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    ft_summary = json.loads((args.run_dir / "sherloc_finetune" / "sherloc_finetune_v2_summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(ft_summary["base_checkpoint"])
    base_classes = ft_summary["base_classes"]
    all_classes = ft_summary["all_classes_after_sherloc"]
    checkpoint = args.checkpoint or (args.run_dir / "sherloc_adaptation_strategies" / f"zero_shot_{args.variant}.pth")
    if not checkpoint.exists():
        checkpoint = base_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    _, ext = sherloc_split(load_v2_metadata(args.metadata_file))
    x_ext, masks_ext, _ = build_sherloc_arrays(ext, args.run_dir / "_sherloc_preprocessing_cache", "sherloc_external_validation", args.variant, args.refresh_cache)
    y_ext = ext["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)

    model, state, _ = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(model, state, base_classes, all_classes)
    if checkpoint != base_checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    probs, _ = predict_torch(model, DataLoader(RamanDataset(x_ext, masks_ext, y_ext), batch_size=args.batch_size), device)

    rows = []
    pred_rows = []
    for name, candidates in CANDIDATE_SETS.items():
        allowed = None if candidates is None else [class_to_idx[c] for c in candidates if c in class_to_idx]
        p = restrict_probs(probs, allowed)
        rows.extend(sweep(y_ext, p, all_classes, name))
        pred = np.argmax(p, axis=1)
        conf = np.max(p, axis=1)
        for i, (_, row) in enumerate(ext.iterrows()):
            pred_rows.append(
                {
                    "candidate_set": name,
                    "spectrum_id": row["spectrum_id"],
                    "true_label": row["model_label"],
                    "prediction": all_classes[int(pred[i])],
                    "confidence": float(conf[i]),
                    "correct": bool(pred[i] == y_ext[i]),
                }
            )
    out_dir = args.run_dir / "sherloc_operating_points"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "sherloc_external_threshold_sweep_by_candidate_set.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out_dir / "sherloc_external_predictions_by_candidate_set.csv", index=False, encoding="utf-8-sig")
    key = pd.DataFrame(rows)
    key_thresholds = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95])
    key = key[key["threshold"].map(lambda v: bool(np.any(np.isclose(float(v), key_thresholds))))]
    key.to_csv(out_dir / "sherloc_external_key_operating_points.csv", index=False, encoding="utf-8-sig")
    best = (
        pd.DataFrame(rows)
        .sort_values("macro_f1_union_on_accepted", ascending=False)
        .groupby("candidate_set", as_index=False)
        .first()
    )
    best.to_csv(out_dir / "sherloc_external_best_operating_points_by_candidate_set.csv", index=False, encoding="utf-8-sig")
    print(key.sort_values(["candidate_set", "threshold"]).to_string(index=False))
    print(f"Saved SHERLOC operating-point analysis to: {out_dir}")


if __name__ == "__main__":
    main()
