"""Summarize confidence-threshold analysis for all benchmark models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IN_DIR = Path("results/confidence_threshold_analysis")
OUT = IN_DIR / "model_confidence_summary.md"


ORDER = [
    "PCA-SVM",
    "PLS-DA",
    "Optimized 1D-CNN",
    "Standard Transformer",
    "MST validation-selected",
    "MST best-grid diagnostic",
]


def fmt(v: float) -> str:
    return "" if pd.isna(v) else f"{float(v):.3f}"


def main() -> None:
    parent = pd.read_csv(IN_DIR / "parent_test_confidence_threshold_sweep.csv")
    ops = pd.read_csv(IN_DIR / "recommended_confidence_operating_points.csv")
    parent_ops = ops[ops["dataset"].eq("parent_heldout_test")].copy()
    parent_ops["order"] = parent_ops["model"].map({m: i for i, m in enumerate(ORDER)})
    parent_ops = parent_ops.sort_values("order")

    key = parent[parent["threshold"].isin([0.0, 0.5, 0.7, 0.8, 0.9])].copy()
    key = key[key["model"].isin(ORDER)]
    key["order"] = key["model"].map({m: i for i, m in enumerate(ORDER)})
    key = key.sort_values(["order", "threshold"])

    key.to_csv(IN_DIR / "parent_test_key_thresholds_all_requested_models.csv", index=False, encoding="utf-8-sig")
    parent_ops.to_csv(IN_DIR / "parent_test_recommended_operating_points_all_requested_models.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Confidence Thresholds for Benchmark Models",
        "",
        "This summary covers the parent held-out test set for all benchmark model families. SHERLOC leave-one-target-out confidence analysis is currently available for MST because the completed SHERLOC fine-tuning protocol was implemented for MST.",
        "",
        "## Recommended Operating Points",
        "",
        "| Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | Wrong Accepted Rate | Macro one-vs-rest FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in parent_ops.iterrows():
        md.append(
            f"| {r['model']} | {fmt(r['threshold'])} | {fmt(r['coverage'])} | {fmt(r['accuracy_on_accepted'])} | "
            f"{fmt(r['operational_recall_correct_accepted_over_all'])} | {fmt(r['false_discovery_rate_wrong_among_accepted'])} | {fmt(r['macro_one_vs_rest_fpr'])} |"
        )

    md += [
        "",
        "## Key Thresholds",
        "",
        "| Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | Wrong Accepted Rate | Macro one-vs-rest FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in key.iterrows():
        md.append(
            f"| {r['model']} | {fmt(r['threshold'])} | {fmt(r['coverage'])} | {fmt(r['accuracy_on_accepted'])} | "
            f"{fmt(r['operational_recall_correct_accepted_over_all'])} | {fmt(r['false_discovery_rate_wrong_among_accepted'])} | {fmt(r['macro_one_vs_rest_fpr'])} |"
        )
    md += [
        "",
        "## Note",
        "",
        "For the manuscript, the main confidence-threshold table can include all parent-test models, while the SHERLOC operational rejection analysis should be reported for the SHERLOC-adapted MST protocol. If needed, analogous SHERLOC adaptation experiments can be run for the CNN and standard Transformer, but PCA-SVM/PLS-DA adaptation would require refitting rather than neural fine-tuning.",
    ]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(parent_ops[[
        "model",
        "threshold",
        "coverage",
        "accuracy_on_accepted",
        "operational_recall_correct_accepted_over_all",
        "false_discovery_rate_wrong_among_accepted",
        "macro_one_vs_rest_fpr",
    ]].to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
