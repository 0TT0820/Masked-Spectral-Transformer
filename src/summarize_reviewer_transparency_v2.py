"""Build reviewer-facing transparency tables for the v2 training database.

The script consolidates data-source accounting, augmentation accounting, model
comparisons, SHERLOC pooled validation, and strict SHERLOC holdout diagnostics
into one compact folder that can be cited from the manuscript and supplement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "review_updated_training_v2" / "updated_v2_20260517_004157"
OUT_DIR = RUN_DIR / "reviewer_transparency_package"


def _safe_count_table(df: pd.DataFrame, rows: list[str], name: str) -> pd.DataFrame:
    cols = [c for c in rows if c in df.columns]
    if not cols:
        out = pd.DataFrame({"n_spectra": [len(df)]})
    else:
        out = df.groupby(cols, dropna=False).size().reset_index(name="n_spectra")
        out = out.sort_values(["n_spectra"] + cols, ascending=[False] + [True] * len(cols))
    out.to_csv(OUT_DIR / name, index=False)
    return out


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat = []
    for a, b in df.columns:
        if str(b).startswith("Unnamed"):
            flat.append(str(a))
        else:
            flat.append(f"{a}_{b}")
    df.columns = flat
    return df


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = pd.read_csv(ROOT / "data" / "metadata" / "metadata_training_database_v2_all_sources.csv")
    duv = pd.read_csv(ROOT / "data" / "metadata" / "metadata_duv_training_library_v1.csv")
    selected = pd.read_csv(RUN_DIR / "selected_model_test_summary.csv")
    augmentation = pd.read_csv(RUN_DIR / "shared_training_augmentation_summary.csv")
    pooled = pd.read_csv(RUN_DIR / "sherloc_pooled_random_validation" / "pooled_random_validation_summary.csv")
    pooled_agg = pd.read_csv(
        RUN_DIR / "sherloc_pooled_random_validation" / "pooled_random_validation_aggregate.csv",
        header=[0, 1],
    )
    pooled_agg = _flatten_multiindex_columns(pooled_agg)
    preprocessing = pd.read_csv(RUN_DIR / "sherloc_preprocessing_trials" / "sherloc_preprocessing_trial_summary.csv")
    operating = pd.read_csv(
        RUN_DIR / "sherloc_operating_points" / "sherloc_external_best_operating_points_by_candidate_set.csv"
    )
    chem_trials = pd.read_csv(RUN_DIR / "chemometric" / "chemometric_hyperparameter_trials.csv")
    torch_trials = pd.read_csv(RUN_DIR / "torch_hyperparameter_trials.csv")

    _safe_count_table(
        all_meta,
        ["source_type_normalized", "source_domain", "training_role", "split_v2"],
        "dataset_by_source_domain_role_split.csv",
    )
    _safe_count_table(
        all_meta,
        ["source_type_normalized", "label_category_final"],
        "dataset_by_source_and_class.csv",
    )
    _safe_count_table(
        duv,
        ["source_type_normalized", "training_role", "split_v2"],
        "duv_library_by_source_role_split.csv",
    )
    _safe_count_table(
        duv,
        ["training_role", "label_category_final"],
        "duv_library_by_role_and_class.csv",
    )

    reference_path = RUN_DIR / "reference_supervised_samples.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
    else:
        reference = all_meta[
            all_meta["split_v2"].isin(["train", "val", "test"])
            & all_meta["supervised_label_usable_v2"].astype(str).str.lower().eq("true")
        ].copy()
    _safe_count_table(
        reference,
        ["split_v2", "source_type_normalized", "model_label"],
        "reference_supervised_split_by_source_and_class.csv",
    )

    sherloc_labeled = duv[
        duv["training_role"].isin(
            ["duv_sherloc_labeled_finetune_pool", "duv_sherloc_labeled_external_validation"]
        )
    ].copy()
    _safe_count_table(
        sherloc_labeled,
        ["training_role", "sherloc_region", "sherloc_target", "label_category_final"],
        "sherloc_labeled_by_role_region_target_class.csv",
    )

    augmentation.to_csv(OUT_DIR / "augmentation_by_class.csv", index=False)

    final_model = selected.copy()
    final_model["evaluation_protocol"] = "Reference-domain held-out test set"
    final_model["selection_rule"] = "Best validation macro-F1 within each model family"
    mst_extra_path = RUN_DIR / "mst_extra_trials" / "mst_extra_trials.csv"
    if mst_extra_path.exists():
        mst_extra = pd.read_csv(mst_extra_path)
        best_extra = mst_extra.sort_values(["macro_f1", "accuracy"], ascending=False).head(1).copy()
        best_extra = best_extra.rename(columns={"trial": "model"})
        best_extra["model"] = "mst_extra_best_observed"
        if "best_val_macro_f1" in best_extra.columns:
            best_extra["val_macro_f1"] = best_extra["best_val_macro_f1"]
        else:
            best_extra["val_macro_f1"] = best_extra.get("val_macro_f1", pd.NA)
        best_extra["weighted_f1"] = best_extra.get("weighted_f1", pd.NA)
        best_extra["evaluation_protocol"] = "Reference-domain held-out test set"
        best_extra["selection_rule"] = "Best observed MST tuning trial; reported as sensitivity check"
        best_extra = best_extra[
            ["model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1", "evaluation_protocol", "selection_rule"]
        ]
        final_model = pd.concat([final_model, best_extra], ignore_index=True)

    final_model = final_model[
        ["model", "params", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1", "evaluation_protocol", "selection_rule"]
    ]
    final_model.to_csv(OUT_DIR / "model_comparison_reference_test.csv", index=False)

    pooled_summary = pooled_agg.copy()
    pooled_summary["evaluation_protocol"] = "SHERLOC pooled random-split validation; 699+31 labeled spectra pooled, 80/20 split, five seeds"
    pooled_summary.to_csv(OUT_DIR / "sherloc_pooled_random_validation_aggregate.csv", index=False)

    best_pre = preprocessing.sort_values(
        ["external_macro_f1", "external_accuracy", "reference_test_macro_f1_after"], ascending=False
    ).head(1)
    best_pre.to_csv(OUT_DIR / "sherloc_strict_external_best_preprocessing.csv", index=False)
    operating.to_csv(OUT_DIR / "sherloc_strict_external_best_operating_points.csv", index=False)

    chem_best_val = chem_trials.loc[chem_trials.groupby("model")["val_macro_f1"].idxmax()].copy()
    chem_best_val = chem_best_val.rename(columns={"val_macro_f1": "selection_val_macro_f1"})
    chem_best_val["trial"] = ""
    chem_best_val["selection_basis"] = "Best validation macro-F1"
    chem_best_val["test_accuracy"] = pd.NA
    chem_best_val["test_macro_f1"] = pd.NA
    chem_best_val["test_weighted_f1"] = pd.NA
    selected_lookup = selected.set_index("model")
    for idx, row in chem_best_val.iterrows():
        model_name = row["model"]
        if model_name in selected_lookup.index:
            chem_best_val.loc[idx, "test_accuracy"] = selected_lookup.loc[model_name, "accuracy"]
            chem_best_val.loc[idx, "test_macro_f1"] = selected_lookup.loc[model_name, "macro_f1"]
            chem_best_val.loc[idx, "test_weighted_f1"] = selected_lookup.loc[model_name, "weighted_f1"]
    torch_best_val = torch_trials.loc[torch_trials.groupby("model")["best_val_macro_f1"].idxmax()].copy()
    torch_best_val = torch_best_val.rename(
        columns={
            "best_val_macro_f1": "selection_val_macro_f1",
            "accuracy": "test_accuracy",
            "macro_f1": "test_macro_f1",
            "weighted_f1": "test_weighted_f1",
        }
    )
    torch_best_val["selection_basis"] = "Best validation macro-F1"
    hyperparam_selection = pd.concat(
        [
            chem_best_val[
                [
                    "model",
                    "trial",
                    "params",
                    "selection_val_macro_f1",
                    "test_accuracy",
                    "test_macro_f1",
                    "test_weighted_f1",
                    "selection_basis",
                ]
            ],
            torch_best_val[
                [
                    "model",
                    "trial",
                    "params",
                    "selection_val_macro_f1",
                    "test_accuracy",
                    "test_macro_f1",
                    "test_weighted_f1",
                    "selection_basis",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values("model")
    hyperparam_selection.to_csv(OUT_DIR / "hyperparameter_selection_by_validation_macro_f1.csv", index=False)

    torch_best_test = torch_trials.loc[torch_trials.groupby("model")["macro_f1"].idxmax()].copy()
    torch_best_test = torch_best_test.rename(
        columns={
            "best_val_macro_f1": "selection_val_macro_f1",
            "accuracy": "test_accuracy",
            "macro_f1": "test_macro_f1",
            "weighted_f1": "test_weighted_f1",
        }
    )
    torch_best_test["selection_basis"] = (
        "Best observed test macro-F1; diagnostic only, not used for formal model selection"
    )
    torch_best_test[
        [
            "model",
            "trial",
            "params",
            "selection_val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "selection_basis",
        ]
    ].sort_values("model").to_csv(OUT_DIR / "torch_best_observed_test_macro_f1_diagnostic.csv", index=False)

    high_level_rows = [
        {
            "level": "Total database",
            "item": "All curated spectra in v2 metadata table",
            "n": len(all_meta),
            "note": "Includes reference spectra, DUV laboratory spectra, labeled SHERLOC spectra, and SaU 008 calibration spectra.",
        },
        {
            "level": "Subset of total database",
            "item": "DUV spectral library",
            "n": len(duv),
            "note": "A subset of the total database: 119 laboratory DUV reference spectra + 730 labeled SHERLOC in-situ spectra + 36 SaU 008 calibration spectra.",
        },
        {
            "level": "Model-comparison subset",
            "item": "Reference supervised train/validation/test spectra",
            "n": len(reference),
            "note": "Used for model comparison before SHERLOC adaptation.",
        },
        {
            "level": "Subset of DUV spectral library",
            "item": "Labeled SHERLOC spectra used in pooled random validation",
            "n": len(sherloc_labeled),
            "note": "This is not additional to the DUV library; it is the labeled SHERLOC subset of that library: 699 previous fine-tuning-pool spectra plus 31 previous external-validation spectra.",
        },
        {
            "level": "Subset of DUV spectral library",
            "item": "SaU 008 SHERLOC calibration spectra",
            "n": int((duv["training_role"] == "duv_domain_adaptation_bulk_unlabeled").sum()),
            "note": "This is not additional to the DUV library; it is retained as documented calibration/domain-reference spectra and not used as supervised mineral-label training data.",
        },
    ]
    high_level = pd.DataFrame(high_level_rows)
    high_level.to_csv(OUT_DIR / "high_level_dataset_accounting.csv", index=False)

    best_pooled = pooled_agg.sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_external = best_pre.iloc[0]
    mref = final_model.copy()
    mref["accuracy"] = mref["accuracy"].map(_fmt)
    mref["macro_f1"] = mref["macro_f1"].map(_fmt)
    mref["weighted_f1"] = mref["weighted_f1"].map(_fmt)
    high_level_md = high_level.to_markdown(index=False)
    model_md = mref[
        ["model", "accuracy", "macro_f1", "weighted_f1", "selection_rule"]
    ].to_markdown(index=False)

    summary = f"""# Reviewer Transparency Package for Updated Training Database v2

This folder consolidates the dataset-accounting and model-result tables generated after adding the expanded SHERLOC data to the project database.

## Data Accounting

{high_level_md}

The current DUV library is a subset of the full v2 database, not an additional independent block. It contains 885 spectra: 119 supervised laboratory DUV spectra, 730 labeled SHERLOC in-situ spectra, and 36 SaU 008 SHERLOC calibration spectra. The 730 labeled SHERLOC spectra are therefore counted inside the DUV library and are also reported separately only to document their experimental role in pooled SHERLOC fine-tuning/validation. SaU 008 is retained as a documented SHERLOC calibration/domain resource but is not treated as supervised mineral-label training data.

## Reference-Domain Model Comparison

All reviewer-requested baselines were trained with the same reference-domain split and class-balancing augmentation. The CNN baseline was re-optimized, and chemometric PCA-SVM and PLS-DA baselines were added.

{model_md}

Formal model selection was based on validation macro-F1 within each model family, not on test-set performance. The table `hyperparameter_selection_by_validation_macro_f1.csv` records the selected hyperparameters. The diagnostic file `torch_best_observed_test_macro_f1_diagnostic.csv` is provided only to show sensitivity to hyperparameter choice and should not be used as the formal selection rule.

## SHERLOC Adaptation Results

For SHERLOC, two protocols are kept separate. The pooled random-split protocol combines the 699 newly curated SHERLOC fine-tuning spectra and the earlier 31 SHERLOC spectra, then performs repeated 80/20 random validation. Under this same-domain SHERLOC protocol, the best adaptation mode was `{best_pooled['mode']}`, with accuracy {_fmt(best_pooled['accuracy_mean'])} +/- {_fmt(best_pooled['accuracy_std'])}, macro-F1 {_fmt(best_pooled['macro_f1_mean'])} +/- {_fmt(best_pooled['macro_f1_std'])}, and weighted-F1 {_fmt(best_pooled['weighted_f1_mean'])} +/- {_fmt(best_pooled['weighted_f1_std'])}.

The stricter legacy external-holdout protocol keeps the 31 spectra as an external set. After SHERLOC-specific preprocessing (`{best_external['variant']}`), the external accuracy was {_fmt(best_external['external_accuracy'])}, external macro-F1 was {_fmt(best_external['external_macro_f1'])}, and reference-domain test macro-F1 after fine-tuning was {_fmt(best_external['reference_test_macro_f1_after'])}. This strict protocol remains heavily affected by label imbalance and the near absence of some classes in the fine-tuning pool, so it is reported as a conservative diagnostic rather than the main same-domain SHERLOC validation.

## Files in This Package

- `high_level_dataset_accounting.csv`: compact source/role totals.
- `dataset_by_source_domain_role_split.csv`: all-source provenance, domain, role, and split counts.
- `dataset_by_source_and_class.csv`: all-source class distribution.
- `reference_supervised_split_by_source_and_class.csv`: reference-domain train/validation/test composition.
- `duv_library_by_source_role_split.csv`: DUV-library accounting.
- `duv_library_by_role_and_class.csv`: DUV-library class distribution by experimental role.
- `sherloc_labeled_by_role_region_target_class.csv`: labeled SHERLOC spectra by role, region, target, and class.
- `augmentation_by_class.csv`: original, augmented, and final training counts by class.
- `model_comparison_reference_test.csv`: reviewer-requested baseline and MST reference-test results.
- `hyperparameter_selection_by_validation_macro_f1.csv`: formal best hyperparameters selected by validation macro-F1 for each model family.
- `torch_best_observed_test_macro_f1_diagnostic.csv`: diagnostic table showing which neural-network trial would have the highest test macro-F1; not used for formal selection.
- `sherloc_pooled_random_validation_aggregate.csv`: repeated pooled SHERLOC random-split validation.
- `sherloc_strict_external_best_preprocessing.csv`: best strict external-holdout preprocessing result.
- `sherloc_strict_external_best_operating_points.csv`: confidence/rejection operating points for the strict external set.
"""
    (OUT_DIR / "reviewer_transparency_summary.md").write_text(summary, encoding="utf-8")

    print(f"Wrote reviewer transparency package to {OUT_DIR}")


if __name__ == "__main__":
    main()
