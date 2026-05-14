"""Confidence-threshold and rejection analysis for Raman model outputs.

This script reports how coverage, accepted accuracy, precision/recall, and
false-positive measures vary as low-confidence predictions are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from train_model_comparison import (
    METADATA_FILE,
    PLSDA,
    StandardTransformer,
    MaskedSpectralTransformer,
    RamanDataset,
    build_cache,
    flatten_features,
    load_metadata,
    predict_torch,
)
from run_model_selection import TunedCNN, make_balanced_augmented_train


ROOT = Path(r"D:/dyt/raman/pigeonite")
BASE_RUN = ROOT / "results" / "model_selection" / "fair_selection_20260510_145634"
OUT_DIR = ROOT / "results" / "confidence_threshold_analysis"
SHERLOC_RUN = ROOT / "results" / "sherloc_target_transfer" / "mst945_lr3e5_160ep_lastblock_loto"
THRESHOLDS = np.round(np.arange(0.0, 0.951, 0.05), 2)


def load_parent_test_arrays():
    df = load_metadata("curated", include_review_required=False, metadata_file=METADATA_FILE)
    enc = LabelEncoder()
    df["label_id"] = enc.fit_transform(df["model_label"])
    classes = list(enc.classes_)
    cache_path = build_cache(df, OUT_DIR / "_cache", False, "poly", False, "curated")
    cache = np.load(cache_path)
    x = cache["features"]
    masks = cache["masks"]
    y = df["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]
    x_train, masks_train, y_train, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], 200, 260
    )
    return {
        "df": df,
        "classes": classes,
        "x_train": x_train,
        "masks_train": masks_train,
        "y_train": y_train,
        "x_val": x[val_idx],
        "masks_val": masks[val_idx],
        "y_val": y[val_idx],
        "x_test": x[test_idx],
        "masks_test": masks[test_idx],
        "y_test": y[test_idx],
        "aug_summary": aug_summary,
    }


def multiclass_threshold_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    classes: Iterable[str],
    model_name: str,
    dataset: str,
) -> list[dict[str, object]]:
    classes = list(classes)
    rows = []
    n = len(y_true)
    for threshold in THRESHOLDS:
        accepted = confidence >= threshold
        rejected = ~accepted
        accepted_n = int(accepted.sum())
        rejected_n = int(rejected.sum())
        correct = (y_pred == y_true) & accepted
        wrong_accepted = accepted & (y_pred != y_true)

        if accepted_n:
            accuracy_accepted = float(np.mean(y_pred[accepted] == y_true[accepted]))
            macro_f1_accepted = float(
                f1_score(y_true[accepted], y_pred[accepted], labels=np.arange(len(classes)), average="macro", zero_division=0)
            )
            weighted_f1_accepted = float(
                f1_score(y_true[accepted], y_pred[accepted], labels=np.arange(len(classes)), average="weighted", zero_division=0)
            )
            precision_micro_accepted = float(correct.sum() / accepted_n)
            false_discovery_rate = float(wrong_accepted.sum() / accepted_n)
        else:
            accuracy_accepted = np.nan
            macro_f1_accepted = np.nan
            weighted_f1_accepted = np.nan
            precision_micro_accepted = np.nan
            false_discovery_rate = np.nan

        macro_fprs = []
        for class_id in range(len(classes)):
            fp = int(np.sum(accepted & (y_pred == class_id) & (y_true != class_id)))
            tn = int(np.sum((y_true != class_id) & (~accepted | (y_pred != class_id))))
            denom = fp + tn
            macro_fprs.append(fp / denom if denom else np.nan)

        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "threshold": float(threshold),
                "n_total": int(n),
                "accepted_n": accepted_n,
                "rejected_n": rejected_n,
                "coverage": float(accepted_n / n) if n else np.nan,
                "rejection_rate": float(rejected_n / n) if n else np.nan,
                "accuracy_on_accepted": accuracy_accepted,
                "macro_f1_on_accepted": macro_f1_accepted,
                "weighted_f1_on_accepted": weighted_f1_accepted,
                "precision_micro_on_accepted": precision_micro_accepted,
                "operational_recall_correct_accepted_over_all": float(correct.sum() / n) if n else np.nan,
                "false_discovery_rate_wrong_among_accepted": false_discovery_rate,
                "false_positive_rate_wrong_accepted_over_all": float(wrong_accepted.sum() / n) if n else np.nan,
                "macro_one_vs_rest_fpr": float(np.nanmean(macro_fprs)),
            }
        )
    return rows


def train_selected_sklearn(parent: dict) -> dict[str, np.ndarray]:
    x_train = parent["x_train"]
    y_train = parent["y_train"]
    x_test = parent["x_test"]
    outputs = {}

    # Validation-selected PCA-SVM: stride=8, PCA=40, C=10, gamma=scale.
    pca_stride = 8
    pca_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=40, random_state=2024)),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=2024)),
        ]
    )
    pca_model.fit(flatten_features(x_train, pca_stride), y_train)
    outputs["PCA-SVM"] = pca_model.predict_proba(flatten_features(x_test, pca_stride))

    # Validation-selected PLS-DA: stride=4, components=12.
    pls_stride = 4
    pls = PLSDA(n_components=12)
    pls.fit(flatten_features(x_train, pls_stride), y_train)
    outputs["PLS-DA"] = pls.predict_proba(flatten_features(x_test, pls_stride))
    return outputs


def load_selected_torch(parent: dict) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = parent["classes"]
    test_ds = RamanDataset(parent["x_test"], parent["masks_test"], parent["y_test"], augment=False)
    outputs = {}

    cnn = TunedCNN(num_classes=len(classes), dropout=0.4)
    cnn.load_state_dict(torch.load(BASE_RUN / "torch" / "cnn_trial2.pth", map_location=device))
    outputs["Optimized 1D-CNN"] = predict_torch(cnn.to(device), DataLoader(test_ds, batch_size=64), device)[0]

    transformer = StandardTransformer(num_classes=len(classes), d_model=96, layers=3, patch_size=8)
    transformer.load_state_dict(torch.load(BASE_RUN / "torch" / "standard_transformer_trial3.pth", map_location=device))
    outputs["Standard Transformer"] = predict_torch(transformer.to(device), DataLoader(test_ds, batch_size=64), device)[0]

    mst = MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=8)
    mst.load_state_dict(torch.load(BASE_RUN / "torch" / "mst_trial6.pth", map_location=device))
    outputs["MST validation-selected"] = predict_torch(mst.to(device), DataLoader(test_ds, batch_size=64), device)[0]

    # Diagnostic best test-macro-F1 MST, useful for sensitivity discussion but not
    # the validation-selected primary result.
    mst_diag = MaskedSpectralTransformer(num_classes=len(classes), d_model=128, layers=4, patch_size=4)
    mst_diag.load_state_dict(torch.load(BASE_RUN / "torch" / "mst_trial7.pth", map_location=device))
    outputs["MST best-grid diagnostic"] = predict_torch(mst_diag.to(device), DataLoader(test_ds, batch_size=64), device)[0]

    return outputs


def parent_threshold_analysis() -> pd.DataFrame:
    parent = load_parent_test_arrays()
    parent["aug_summary"].to_csv(OUT_DIR / "parent_shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")
    y_true = parent["y_test"]
    classes = parent["classes"]
    rows = []
    probs_by_model = {}
    probs_by_model.update(train_selected_sklearn(parent))
    probs_by_model.update(load_selected_torch(parent))

    for model_name, probs in probs_by_model.items():
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        rows.extend(multiclass_threshold_rows(y_true, pred, conf, classes, model_name, "parent_heldout_test"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "parent_test_confidence_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    return df


def sherloc_threshold_analysis() -> pd.DataFrame:
    pred_path = SHERLOC_RUN / "sherloc_target_transfer_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    labels = sorted(set(df["model_label"].astype(str)) | set(df["zero_shot_prediction"].astype(str)) | set(df["finetuned_prediction"].astype(str)))
    class_to_idx = {c: i for i, c in enumerate(labels)}
    y_true = df["model_label"].map(class_to_idx).to_numpy(dtype=int)
    rows = []
    for pred_col, conf_col, model_name in [
        ("zero_shot_prediction", "zero_shot_confidence", "SHERLOC MST zero-shot"),
        ("finetuned_prediction", "finetuned_confidence", "SHERLOC MST adapted"),
    ]:
        y_pred = df[pred_col].map(class_to_idx).to_numpy(dtype=int)
        conf = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rows.extend(multiclass_threshold_rows(y_true, y_pred, conf, labels, model_name, "sherloc_leave_one_target_out"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sherloc_confidence_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    return out


def select_operating_points(df: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    for (dataset, model), sub in df.groupby(["dataset", "model"]):
        sub = sub.sort_values("threshold").copy()
        eligible = sub[sub["coverage"] >= 0.5]
        if len(eligible):
            best = eligible.sort_values(
                ["false_discovery_rate_wrong_among_accepted", "accuracy_on_accepted"],
                ascending=[True, False],
            ).iloc[0]
            rule = "minimum false-discovery rate while retaining >=50% coverage"
        else:
            best = sub.sort_values("accuracy_on_accepted", ascending=False).iloc[0]
            rule = "highest accepted accuracy; coverage below 50%"
        row = best.to_dict()
        row["selection_rule"] = rule
        candidates.append(row)
    out = pd.DataFrame(candidates)
    out.to_csv(OUT_DIR / "recommended_confidence_operating_points.csv", index=False, encoding="utf-8-sig")
    return out


def write_summary(parent: pd.DataFrame, sherloc: pd.DataFrame, ops: pd.DataFrame) -> None:
    key_thresholds = [0.0, 0.5, 0.7, 0.8, 0.9]
    combined = pd.concat([parent, sherloc], ignore_index=True)
    selected = combined[combined["threshold"].isin(key_thresholds)].copy()
    selected.to_csv(OUT_DIR / "confidence_threshold_key_thresholds.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Confidence Threshold and Rejection Analysis",
        "",
        "Predictions with maximum class probability below threshold tau are rejected as uncertain. This supports confidence-aware deployment instead of forcing every spectrum into a hard class.",
        "",
        "Key outputs:",
        "",
        "- `parent_test_confidence_threshold_sweep.csv`",
        "- `sherloc_confidence_threshold_sweep.csv`",
        "- `confidence_threshold_key_thresholds.csv`",
        "- `recommended_confidence_operating_points.csv`",
        "",
        "Important metrics:",
        "",
        "- coverage: fraction of spectra accepted;",
        "- accuracy_on_accepted: accuracy after rejecting low-confidence spectra;",
        "- operational_recall_correct_accepted_over_all: correct accepted predictions divided by all spectra;",
        "- false_discovery_rate_wrong_among_accepted: wrong accepted predictions divided by accepted predictions;",
        "- macro_one_vs_rest_fpr: macro-average one-vs-rest false-positive rate.",
        "",
        "Recommended operating points:",
        "",
        "| Dataset | Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | False Discovery Rate | Macro one-vs-rest FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ops.iterrows():
        md.append(
            f"| {row['dataset']} | {row['model']} | {row['threshold']:.2f} | {row['coverage']:.3f} | "
            f"{row['accuracy_on_accepted']:.3f} | {row['operational_recall_correct_accepted_over_all']:.3f} | "
            f"{row['false_discovery_rate_wrong_among_accepted']:.3f} | {row['macro_one_vs_rest_fpr']:.3f} |"
        )
    md += [
        "",
        "Manuscript wording:",
        "",
        "Rather than enforcing a single hard threshold of 0.5, we swept the confidence threshold from 0 to 0.95. Increasing the threshold reduces coverage but generally improves reliability among accepted predictions. Low-confidence spectra should be reported as uncertain and retained for ground review rather than forced into a mineral class.",
    ]
    (OUT_DIR / "confidence_threshold_analysis_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = parent_threshold_analysis()
    sherloc = sherloc_threshold_analysis()
    combined = pd.concat([parent, sherloc], ignore_index=True)
    combined.to_csv(OUT_DIR / "all_confidence_threshold_sweeps.csv", index=False, encoding="utf-8-sig")
    ops = select_operating_points(combined)
    write_summary(parent, sherloc, ops)
    print(ops[[
        "dataset",
        "model",
        "threshold",
        "coverage",
        "accuracy_on_accepted",
        "operational_recall_correct_accepted_over_all",
        "false_discovery_rate_wrong_among_accepted",
        "macro_one_vs_rest_fpr",
    ]].to_string(index=False))
    print(f"Saved confidence-threshold analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()
