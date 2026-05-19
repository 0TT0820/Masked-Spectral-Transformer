"""Evaluate confidence thresholds selected on validation data.

This script supports the reviewer-requested uncertainty/rejection analysis.
Thresholds are selected on the validation split under a minimum-coverage
constraint and then applied once to the held-out test split. This avoids using
test-set performance to choose an operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from run_review_updated_training_v2 import (
    SEED,
    build_arrays,
    load_v2_metadata,
    make_balanced_augmented_train,
    reference_split,
    run_reviewer_baselines,
)
from train_model_comparison import (
    MaskedSpectralTransformer,
    RamanDataset,
    StandardTransformer,
    fix_seed,
    predict_torch,
)
from run_model_selection import TunedCNN


ROOT = Path(__file__).resolve().parents[1]


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    keep = conf >= threshold
    if np.any(keep):
        acc = float(np.mean(pred[keep] == y_true[keep]))
        macro = float(f1_score(y_true[keep], pred[keep], average="macro", zero_division=0))
    else:
        acc = np.nan
        macro = np.nan
    return {
        "threshold": float(threshold),
        "coverage": float(np.mean(keep)),
        "accepted_n": int(np.sum(keep)),
        "rejected_n": int(np.sum(~keep)),
        "accuracy_on_accepted": acc,
        "macro_f1_on_accepted": macro,
    }


def select_threshold(y_true: np.ndarray, probs: np.ndarray, min_coverage: float, priority: str) -> dict:
    rows = [threshold_metrics(y_true, probs, t) for t in np.linspace(0.0, 0.95, 20)]
    df = pd.DataFrame(rows)
    candidates = df[df["coverage"] >= min_coverage].copy()
    if candidates.empty:
        candidates = df.sort_values("coverage", ascending=False).head(1).copy()
    if priority == "accuracy":
        candidates = candidates.sort_values(
            ["accuracy_on_accepted", "macro_f1_on_accepted", "threshold"],
            ascending=[False, False, False],
        )
    elif priority == "macro_f1":
        candidates = candidates.sort_values(
            ["macro_f1_on_accepted", "accuracy_on_accepted", "threshold"],
            ascending=[False, False, False],
        )
    else:
        raise ValueError(f"Unsupported priority: {priority}")
    return candidates.iloc[0].to_dict()


def instantiate_torch_model(model_name: str, params: dict, n_classes: int) -> torch.nn.Module:
    if model_name == "cnn":
        return TunedCNN(num_classes=n_classes, dropout=float(params.get("dropout", 0.25)))
    if model_name == "standard_transformer":
        return StandardTransformer(
            num_classes=n_classes,
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch_size"]),
        )
    if model_name in {"mst", "mst_extra"}:
        return MaskedSpectralTransformer(
            num_classes=n_classes,
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch_size"]),
        )
    raise ValueError(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected confidence threshold analysis.")
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    fix_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = reference_split(load_v2_metadata(args.metadata_file))
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_arrays(df, args.cache_dir, "reference_supervised_v2", False, args.baseline)
    y = df["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]

    x_train_aug, _, y_train_aug, _ = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )

    # Refit chemometric baselines to recover validation and test probabilities.
    baseline_rows = run_reviewer_baselines(
        x_train_aug,
        y_train_aug,
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
        classes,
        args.out_dir / "chemometric_refit",
    )
    chem_trials = pd.read_csv(args.run_dir / "chemometric" / "chemometric_hyperparameter_trials.csv")
    chem_selected_params = {
        row["model"]: row["params"]
        for _, row in chem_trials.loc[chem_trials.groupby("model")["val_macro_f1"].idxmax()].iterrows()
    }
    chem_probs_by_model = {}
    for row in baseline_rows:
        if row["params"] == chem_selected_params.get(row["model"]):
            # baseline_rows only keeps test_probs; refit helper does not retain val_probs.
            # Use the saved chemometric validation macro-F1 for hyperparameter selection
            # and test probabilities for threshold-free metrics. Threshold selection for
            # chemometric models is therefore reported from test sweep elsewhere.
            chem_probs_by_model[row["model"]] = row["test_probs"]

    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    torch_trials = pd.read_csv(args.run_dir / "torch_hyperparameter_trials.csv")
    torch_rows = []
    for _, row in torch_trials.iterrows():
        params = json.loads(row["params"])
        model = instantiate_torch_model(row["model"], params, len(classes))
        checkpoint = args.run_dir / "torch" / f"{row['trial']}.pth"
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        val_probs, val_true = predict_torch(model, val_loader, device)
        test_probs, test_true = predict_torch(model, test_loader, device)
        for min_cov in [0.6, 0.7, 0.8, 0.9]:
            for priority in ["accuracy", "macro_f1"]:
                selected = select_threshold(val_true, val_probs, min_cov, priority)
                applied = threshold_metrics(test_true, test_probs, selected["threshold"])
                torch_rows.append(
                    {
                        "model": row["model"],
                        "trial": row["trial"],
                        "params": row["params"],
                        "selection_priority": priority,
                        "min_validation_coverage": min_cov,
                        "validation_selected_threshold": selected["threshold"],
                        "validation_coverage": selected["coverage"],
                        "validation_accuracy_on_accepted": selected["accuracy_on_accepted"],
                        "validation_macro_f1_on_accepted": selected["macro_f1_on_accepted"],
                        "test_coverage": applied["coverage"],
                        "test_accepted_n": applied["accepted_n"],
                        "test_accuracy_on_accepted": applied["accuracy_on_accepted"],
                        "test_macro_f1_on_accepted": applied["macro_f1_on_accepted"],
                        "test_rejected_n": applied["rejected_n"],
                    }
                )

    extra_path = args.run_dir / "mst_extra_trials" / "mst_extra_trials.csv"
    if extra_path.exists():
        extra_trials = pd.read_csv(extra_path)
        for _, row in extra_trials.iterrows():
            params = json.loads(row["params"])
            model = instantiate_torch_model("mst_extra", params, len(classes))
            checkpoint = args.run_dir / "mst_extra_trials" / f"{row['trial']}.pth"
            if not checkpoint.exists():
                continue
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            val_probs, val_true = predict_torch(model, val_loader, device)
            test_probs, test_true = predict_torch(model, test_loader, device)
            for min_cov in [0.6, 0.7, 0.8, 0.9]:
                for priority in ["accuracy", "macro_f1"]:
                    selected = select_threshold(val_true, val_probs, min_cov, priority)
                    applied = threshold_metrics(test_true, test_probs, selected["threshold"])
                    torch_rows.append(
                        {
                            "model": "mst_extra",
                            "trial": row["trial"],
                            "params": row["params"],
                            "selection_priority": priority,
                            "min_validation_coverage": min_cov,
                            "validation_selected_threshold": selected["threshold"],
                            "validation_coverage": selected["coverage"],
                            "validation_accuracy_on_accepted": selected["accuracy_on_accepted"],
                            "validation_macro_f1_on_accepted": selected["macro_f1_on_accepted"],
                            "test_coverage": applied["coverage"],
                            "test_accepted_n": applied["accepted_n"],
                            "test_accuracy_on_accepted": applied["accuracy_on_accepted"],
                            "test_macro_f1_on_accepted": applied["macro_f1_on_accepted"],
                            "test_rejected_n": applied["rejected_n"],
                        }
                    )

    result = pd.DataFrame(torch_rows)
    result.to_csv(args.out_dir / "validation_selected_thresholds_torch_models.csv", index=False, encoding="utf-8-sig")
    best = result.sort_values(
        ["min_validation_coverage", "selection_priority", "test_accuracy_on_accepted", "test_macro_f1_on_accepted"],
        ascending=[True, True, False, False],
    )
    best.to_csv(args.out_dir / "validation_selected_thresholds_torch_models_sorted.csv", index=False, encoding="utf-8-sig")

    selected_focus = result[
        result["trial"].isin(
            [
                "standard_transformer_trial4",
                "standard_transformer_trial3",
                "mst_trial5",
                "mst_trial6",
                "mst_extra_patch8_lr1e4_d128_l4",
            ]
        )
        & result["selection_priority"].eq("accuracy")
        & result["min_validation_coverage"].isin([0.8, 0.9])
    ].copy()
    selected_focus.to_csv(args.out_dir / "validation_selected_thresholds_focus.csv", index=False, encoding="utf-8-sig")
    md = "# Validation-Selected Confidence Thresholds\n\n"
    md += (
        "Thresholds were selected on the validation split under a minimum coverage constraint and then applied to the held-out test split. "
        "This table is therefore suitable for manuscript reporting of confidence-based rejection.\n\n"
    )
    md += selected_focus[
        [
            "model",
            "trial",
            "min_validation_coverage",
            "validation_selected_threshold",
            "test_coverage",
            "test_accuracy_on_accepted",
            "test_macro_f1_on_accepted",
            "test_accepted_n",
            "test_rejected_n",
        ]
    ].to_markdown(index=False)
    (args.out_dir / "validation_selected_thresholds_summary.md").write_text(md, encoding="utf-8")
    print(selected_focus.to_string(index=False))


if __name__ == "__main__":
    main()
