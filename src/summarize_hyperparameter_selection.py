"""Create a reviewer-ready hyperparameter selection table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path("review_round3_fair_model_selection/review_ready_20260510_145634")
OUT_DIR = Path("review_round3_requested_baselines_only")


DISPLAY = {
    "pca_svm": "PCA-SVM",
    "pls_da": "PLS-DA",
    "cnn": "Optimized 1D-CNN",
    "standard_transformer": "Standard Transformer",
    "mst": "MST",
}

SEARCH_SPACE = {
    "pca_svm": "stride {4, 8}; PCA components {40, 80}; SVM C {1, 10}; gamma {scale, 0.001}",
    "pls_da": "stride {4, 8}; latent components {4, 8, 12}",
    "cnn": "learning rate {1e-3, 3e-4}; dropout {0.25, 0.40}; tuned 4-block 1D-CNN",
    "standard_transformer": "learning rate {1e-4, 3e-5}; d_model {96, 128}; layers {3, 4}; patch size 8",
    "mst": "learning rate {1e-4, 3e-5}; d_model {96, 128}; layers {3, 4}; patch size {8, 4}",
}


def param_text(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    return "; ".join(f"{k}={v}" for k, v in obj.items())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(OUT_DIR / "reviewer_requested_validation_selected_summary.csv")
    rows = []
    for _, r in selected.iterrows():
        model = r["model"]
        rows.append(
            {
                "Model": DISPLAY[model],
                "Tuning search space": SEARCH_SPACE[model],
                "Selection criterion": "highest validation macro-F1",
                "Selected hyperparameters": param_text(r["params"]),
                "Validation macro-F1": round(float(r["val_macro_f1"]), 3),
                "Test accuracy": round(float(r["accuracy"]), 3),
                "Test macro-F1": round(float(r["macro_f1"]), 3),
                "Test weighted-F1": round(float(r["weighted_f1"]), 3),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reviewer_requested_hyperparameter_selection_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Hyperparameter Selection Summary",
        "",
        "## Why validation macro-F1 matters",
        "",
        "Validation macro-F1 is the model-selection criterion, not an extra final-performance metric. It is used to choose hyperparameters without looking at the independent test set. This is especially important for this dataset because the mineral classes are imbalanced; macro-F1 weights minority classes equally rather than allowing abundant classes to dominate the choice.",
        "",
        "A high validation macro-F1 means that MST was the preferred model family under the pre-specified validation-based selection rule. It does not by itself prove that MST has the highest final test score on every metric. Therefore, the manuscript should report both validation-selected hyperparameters and independent test performance.",
        "",
        "## Reviewer-requested baseline tuning",
        "",
        "| Model | Search space | Selected hyperparameters | Val Macro-F1 | Test Acc. | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['Model']} | {row['Tuning search space']} | {row['Selected hyperparameters']} | "
            f"{row['Validation macro-F1']:.3f} | {row['Test accuracy']:.3f} | {row['Test macro-F1']:.3f} | {row['Test weighted-F1']:.3f} |"
        )
    md += [
        "",
        "## Manuscript wording recommendation",
        "",
        "Use wording such as: 'Hyperparameters for all baseline models were selected on the validation split using macro-F1, and the final selected models were evaluated once on the held-out test split.'",
        "",
        "Avoid wording such as: 'MST is superior because it has the highest validation macro-F1.' The correct claim is narrower: 'MST achieved the highest validation macro-F1 among the reviewer-requested baselines and competitive independent test performance, while retaining the architectural advantages required for parameter-efficient SHERLOC adaptation.'",
    ]
    (OUT_DIR / "hyperparameter_selection_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
