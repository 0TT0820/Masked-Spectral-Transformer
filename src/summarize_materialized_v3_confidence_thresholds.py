"""Summarize reviewer-requested confidence-threshold analyses.

The script combines confidence-threshold scans from two evaluation settings:

1. Reference-domain held-out test spectra generated from the materialized
   Raman-aware augmented training dataset.
2. Pooled labeled SHERLOC in-situ random-split validation spectra.

Only the reviewer-requested model families are included in the final tables:
PCA-SVM, PLS-DA, 1D-CNN, Standard Transformer, and MST.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "results" / "materialized_augmented_v3_model_selection" / "curated_20260519_104300"
SHERLOC_DIR = ROOT / "results" / "sherloc_in_situ_model_comparison_v3"
OUT_DIR = ROOT / "results" / "confidence_threshold_materialized_v3"

KEY_THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9]

REFERENCE_THRESHOLD_FILES = {
    "pca_svm": REFERENCE_DIR / "pca_svm.selected.threshold_sweep.csv",
    "pls_da": REFERENCE_DIR / "pls_da.selected.threshold_sweep.csv",
    "cnn": REFERENCE_DIR / "torch" / "cnn_trial2.threshold_sweep.csv",
    "standard_transformer": REFERENCE_DIR / "torch" / "standard_transformer_trial4.threshold_sweep.csv",
    "mst": REFERENCE_DIR / "torch" / "mst_trial5.threshold_sweep.csv",
}

MODEL_DISPLAY = {
    "pca_svm": "PCA-SVM",
    "pls_da": "PLS-DA",
    "cnn": "1D-CNN",
    "standard_transformer": "Standard Transformer",
    "mst": "MST",
}


def nearest_threshold_rows(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        idx = (df["threshold"].astype(float) - threshold).abs().idxmin()
        rows.append(df.loc[idx].copy())
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_reference_thresholds() -> pd.DataFrame:
    frames = []
    for model, path in REFERENCE_THRESHOLD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        key = nearest_threshold_rows(df, KEY_THRESHOLDS)
        key["dataset"] = "reference_heldout_test"
        key["model"] = model
        key["model_display"] = MODEL_DISPLAY[model]
        key["n_total"] = key["accepted_n"] + key["rejected_n"]
        key["accuracy_on_accepted"] = key["accuracy_on_accepted"]
        key["macro_f1_on_accepted"] = key["macro_f1_on_accepted"]
        key["operational_recall_correct_accepted_over_all"] = key["coverage"] * key["accuracy_on_accepted"]
        key["false_discovery_rate_wrong_among_accepted"] = 1.0 - key["accuracy_on_accepted"]
        key["false_positive_rate_wrong_accepted_over_all"] = key["coverage"] * (
            1.0 - key["accuracy_on_accepted"]
        )
        frames.append(key)
    cols = [
        "dataset",
        "model",
        "model_display",
        "threshold",
        "n_total",
        "accepted_n",
        "rejected_n",
        "coverage",
        "accuracy_on_accepted",
        "macro_f1_on_accepted",
        "operational_recall_correct_accepted_over_all",
        "false_discovery_rate_wrong_among_accepted",
        "false_positive_rate_wrong_accepted_over_all",
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def summarize_sherloc_thresholds() -> pd.DataFrame:
    path = SHERLOC_DIR / "sherloc_in_situ_key_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["model"].isin(MODEL_DISPLAY)].copy()
    df["model_display"] = df["model"].map(MODEL_DISPLAY)
    df = df.rename(columns={"macro_f1_on_accepted_present_labels": "macro_f1_on_accepted"})
    cols = [
        "dataset",
        "model",
        "model_display",
        "threshold",
        "n_total",
        "accepted_n",
        "rejected_n",
        "coverage",
        "accuracy_on_accepted",
        "macro_f1_on_accepted",
        "operational_recall_correct_accepted_over_all",
        "false_discovery_rate_wrong_among_accepted",
        "false_positive_rate_wrong_accepted_over_all",
        "macro_one_vs_rest_fpr",
    ]
    return df[cols].sort_values(["model", "threshold"]).reset_index(drop=True)


def write_markdown(reference: pd.DataFrame, sherloc: pd.DataFrame, combined: pd.DataFrame) -> None:
    model_summary = pd.read_csv(REFERENCE_DIR / "reviewer_requested_model_test_summary.csv")
    sherloc_summary = pd.read_csv(SHERLOC_DIR / "sherloc_in_situ_model_comparison_aggregate.csv", header=[0, 1])
    sherloc_summary.columns = [
        str(col[0])
        if str(col[0]) == "model"
        else "_".join([str(part) for part in col if str(part) != "nan"]).strip("_")
        for col in sherloc_summary.columns.to_flat_index()
    ]

    lines = [
        "# Confidence-Threshold and Model-Comparison Summary",
        "",
        "This directory contains reviewer-requested operating-threshold summaries for the final materialized",
        "augmentation benchmark and for pooled labeled SHERLOC in-situ random-split validation.",
        "",
        "## Reference-Domain Held-Out Test",
        "",
        model_summary.to_markdown(index=False),
        "",
        "## SHERLOC In-Situ Pooled Random-Split Validation",
        "",
        "The SHERLOC table pools labeled in-situ spectra and reports repeated random-split validation over",
        "three seeds. This is a within-domain adaptation validation, not an independent target-transfer test.",
        "",
        sherloc_summary.to_markdown(index=False),
        "",
        "## Key Confidence Thresholds",
        "",
        "Key thresholds are 0.0, 0.5, 0.7, 0.8, and 0.9. Coverage is the fraction of spectra accepted",
        "after rejection of low-confidence predictions. Accuracy and macro-F1 are recomputed only on",
        "accepted predictions. Operational recall is the fraction of all spectra that are both accepted",
        "and correctly classified.",
        "",
        "### Reference Test",
        "",
        reference.to_markdown(index=False),
        "",
        "### SHERLOC In-Situ Validation",
        "",
        sherloc.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `reference_test_key_thresholds_requested_models.csv`: key thresholds for the reference held-out test.",
        "- `sherloc_in_situ_key_thresholds_requested_models.csv`: key thresholds for pooled SHERLOC validation.",
        "- `combined_key_thresholds_requested_models.csv`: combined reference and SHERLOC key thresholds.",
        "- `confidence_threshold_summary.md`: this human-readable summary.",
        "",
    ]
    (OUT_DIR / "confidence_threshold_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = summarize_reference_thresholds()
    sherloc = summarize_sherloc_thresholds()
    combined = pd.concat([reference, sherloc], ignore_index=True, sort=False)

    reference.to_csv(OUT_DIR / "reference_test_key_thresholds_requested_models.csv", index=False, encoding="utf-8-sig")
    sherloc.to_csv(OUT_DIR / "sherloc_in_situ_key_thresholds_requested_models.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUT_DIR / "combined_key_thresholds_requested_models.csv", index=False, encoding="utf-8-sig")
    write_markdown(reference, sherloc, combined)
    print(f"Wrote threshold summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
