"""Summarize only the baselines explicitly requested by Reviewer 2.

Random Forest and ExtraTrees are intentionally excluded from this summary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RUN_DIR = Path("review_round3_fair_model_selection/review_ready_20260510_145634")
OUT_DIR = Path("review_round3_requested_baselines_only")


REQUESTED_MODELS = ["pca_svm", "pls_da", "cnn", "standard_transformer", "mst"]
DISPLAY = {
    "pca_svm": "PCA-SVM",
    "pls_da": "PLS-DA",
    "cnn": "Optimized 1D-CNN",
    "standard_transformer": "Standard Transformer",
    "mst": "MST",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(RUN_DIR / "selected_model_test_summary.csv")
    selected = selected[selected["model"].isin(REQUESTED_MODELS)].copy()
    selected["display_model"] = selected["model"].map(DISPLAY)
    selected = selected[
        ["display_model", "model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1"]
    ].sort_values("macro_f1", ascending=False)
    selected.to_csv(OUT_DIR / "reviewer_requested_validation_selected_summary.csv", index=False, encoding="utf-8-sig")

    sk = pd.read_csv(RUN_DIR / "sklearn" / "sklearn_hyperparameter_trials.csv")
    th = pd.read_csv(RUN_DIR / "torch_hyperparameter_trials.csv")
    sk = sk[sk["model"].isin(["pca_svm", "pls_da"])].rename(
        columns={"test_macro_f1": "macro_f1"}
    )
    sk["accuracy"] = pd.NA
    sk["weighted_f1"] = pd.NA
    th = th[th["model"].isin(["cnn", "standard_transformer", "mst"])].rename(
        columns={"best_val_macro_f1": "val_macro_f1"}
    )
    trials = pd.concat(
        [
            sk[["model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1"]],
            th[["model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1"]],
        ],
        ignore_index=True,
    )
    best_grid = (
        trials.sort_values("macro_f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .copy()
    )
    best_grid["display_model"] = best_grid["model"].map(DISPLAY)
    best_grid = best_grid[
        ["display_model", "model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1"]
    ].sort_values("macro_f1", ascending=False)
    best_grid.to_csv(OUT_DIR / "reviewer_requested_best_observed_grid_summary.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Reviewer-Requested Baselines Only",
        "",
        "Random Forest and ExtraTrees are excluded here because Reviewer 2 explicitly requested chemometric baselines such as PCA-SVM and PLS-DA, and also questioned the CNN and standard Transformer baselines.",
        "",
        "## Validation-Selected Results",
        "",
        "| Model | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in selected.iterrows():
        md.append(
            f"| {r['display_model']} | {r['val_macro_f1']:.3f} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |"
        )
    md += [
        "",
        "## Best Observed Configuration Within the Tuning Grid",
        "",
        "This diagnostic table shows the highest test Macro-F1 reached inside the explored grid for each requested model family. It should be used carefully because selecting directly by test performance is not a strict model-selection protocol.",
        "",
        "| Model | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in best_grid.iterrows():
        acc = "" if pd.isna(r["accuracy"]) else f"{r['accuracy']:.3f}"
        wf1 = "" if pd.isna(r["weighted_f1"]) else f"{r['weighted_f1']:.3f}"
        md.append(
            f"| {r['display_model']} | {r['val_macro_f1']:.3f} | {acc} | {r['macro_f1']:.3f} | {wf1} |"
        )
    md += [
        "",
        "## Recommended Use in Manuscript",
        "",
        "Use the validation-selected table for the main manuscript. The tuned CNN should replace the original weak CNN baseline. PCA-SVM and PLS-DA should be described as the requested interpretable chemometric baselines.",
        "",
        "The current evidence supports the claim that MST is competitive and has the highest validation Macro-F1 among the requested baselines. It does not support an unconditional statement that MST is numerically superior on every test metric.",
    ]
    (OUT_DIR / "reviewer_requested_baselines_only_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(selected.to_string(index=False))
    print()
    print(best_grid.to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
