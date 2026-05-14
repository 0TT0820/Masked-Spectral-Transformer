# Dataset and Experiment Map

This file maps the repository files to the manuscript sections and reviewer
comments they support. It is intended as the first stop for reviewers who want
to inspect data provenance, augmentation lineage, model comparisons, SHERLOC
fine-tuning, and confidence-threshold analyses.

## Quick Navigation

| Topic | Manuscript / Review Context | Main Data Files | Scripts | Result Files |
|---|---|---|---|---|
| Parent 945-spectrum dataset | Dataset construction; Reviewer 1 data transparency | `data/metadata/metadata_parent_945.csv`; `data/spectra/parent/`; `data/overview/parent_by_source_type.csv`; `data/overview/parent_by_source_and_category.csv` | `src/build_review_data_inventory.py` | `data/overview/review_data_inventory/` |
| RRUFF provenance enrichment | Reviewer 1 concern on spectrum source, chemistry, locality, and reference traceability | `data/metadata/rruff_official_header_metadata.csv`; `data/metadata/metadata_parent_945_rruff_enriched.csv`; `data/metadata/rruff_official_header_missing.csv` | `src/fetch_rruff_metadata.py`; `src/enrich_metadata_from_rruff_headers.py` | `data/metadata/rruff_header_enrichment_summary.json` |
| Reproducible Raman-aware augmentation | Reviewer 1 comment 5; Reviewer 2 detailed comment 4 | `data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv`; `data/materialized_augmented_review_ready_v1/spectra/`; `data/materialized_augmented_review_ready_v1/augmentation_protocol.json`; `docs/Reviewer2_Comment4_augmentation_parameters.csv` | `src/build_materialized_augmented_dataset.py`; `src/train_review_comparison.py` | `data/materialized_augmented_review_ready_v1/overview_tables/`; `results/model_comparison_materialized_augmented.csv` |
| Reviewer-requested model baselines | Reviewer 2 major comment 1; Reviewer 2 detailed comment 6 | `data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv` | `src/run_review_model_selection.py`; `src/summarize_reviewer_requested_baselines.py`; `src/summarize_hyperparameter_selection.py` | `results/reviewer_requested_baselines/`; `results/model_comparison_materialized_augmented.csv` |
| Hyperparameter selection | Reviewer 2 request for fair CNN and baseline optimization | Same as baseline comparison | `src/run_review_model_selection.py`; `src/summarize_hyperparameter_selection.py` | `results/reviewer_requested_baselines/reviewer_requested_hyperparameter_selection_table.csv`; `results/reviewer_requested_baselines/hyperparameter_selection_summary.md` |
| SHERLOC region fine-tuning data | Reviewer 2 detailed comment 1 and 2; clarify that final SHERLOC tests use in-situ adaptation, not strict zero-shot transfer | `data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv`; `data/metadata/metadata_sherloc_region_points_only.csv`; `data/metadata/sherloc_region_detail_to_ss_mapping.csv`; `data/overview/sherloc_regions/` | `src/build_sherloc_region_dataset.py`; `src/run_sherloc_finetune_protocol.py`; `src/run_sherloc_target_transfer.py` | `results/sherloc_finetune/` |
| Confidence threshold and rejection analysis | Reviewer 1 line 595 and line 606 comments on threshold >0.5, precision, recall, and false-positive rate | Model prediction outputs produced by comparison scripts | `src/run_confidence_threshold_analysis.py`; `src/summarize_all_requested_confidence_thresholds.py` | `results/confidence_threshold_analysis/` |
| MST-focused tuning | Follow-up model selection after reviewer-requested baselines | Same materialized augmented dataset | `src/run_mst_focused_tuning.py` | `results/mst_focused_tuning/` |

## Data Files by Role

### Original Parent Spectra

Use these files to inspect the original 945-spectrum dataset before reviewer
updates:

```text
data/metadata/metadata_parent_945.csv
data/metadata/metadata_parent_945_rruff_enriched.csv
data/spectra/parent/
data/overview/parent_by_source_type.csv
data/overview/parent_by_source_and_category.csv
data/overview/parent_by_split_and_category.csv
```

These files support manuscript text describing the source composition of the
parent dataset and the reviewer response on spectrum-level provenance.

### Materialized Augmented Dataset

Use these files to reproduce the reviewer-ready augmented dataset:

```text
data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv
data/materialized_augmented_review_ready_v1/augmentation_protocol.json
data/materialized_augmented_review_ready_v1/spectra/
data/materialized_augmented_review_ready_v1/overview_tables/lineage_manifest.csv
data/materialized_augmented_review_ready_v1/overview_tables/split_by_class_and_augmentation.csv
```

Each spectrum CSV stores point-wise model inputs:

```text
raman_shift_cm-1
intensity_normalized
first_derivative_normalized
valid_mask
```

Only training spectra are augmented. Validation and test spectra remain
unaugmented originals. Raman band centers are not shifted in the final
reviewer-ready augmentation protocol.

### SHERLOC Region Dataset

Use these files to inspect the in-situ SHERLOC fine-tuning dataset:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
data/metadata/metadata_sherloc_region_points_only.csv
data/metadata/sherloc_region_detail_to_ss_mapping.csv
data/metadata/sherloc_region_point_extraction_manifest.csv
data/metadata/sherloc_region_table1_training_summary.csv
data/overview/sherloc_regions/
```

Only spreadsheet points with explicit mineral labels are included. Unlabeled
points are treated as background/noise and excluded from supervised training.
If one point has two accepted mineral assignments, it is represented by two
label rows with the same spectrum source.

## Result Files by Reviewer Comment

### Reviewer 2 Major Comment 1: Baselines

Primary files:

```text
results/reviewer_requested_baselines/reviewer_requested_validation_selected_summary.csv
results/reviewer_requested_baselines/reviewer_requested_hyperparameter_selection_table.csv
results/reviewer_requested_baselines/hyperparameter_selection_summary.md
results/model_comparison_materialized_augmented.csv
```

These files support the response that PCA-SVM, PLS-DA, 1D-CNN, Standard
Transformer, and MST were compared under explicit hyperparameter selection.

### Reviewer 2 Detailed Comment 4: Augmentation Reproducibility

Primary files:

```text
docs/Reviewer2_Comment4_augmentation_parameters.csv
docs/Reviewer2_Comment4_augmentation_protocol.json
data/materialized_augmented_review_ready_v1/augmentation_protocol.json
data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv
data/materialized_augmented_review_ready_v1/overview_tables/lineage_manifest.csv
```

These files support the response that every augmented spectrum is materialized,
has a parent spectrum ID, has a random seed, and has explicit augmentation
parameters.

### Reviewer 2 Detailed Comment 1: SHERLOC Workflow Consistency

Primary files:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
data/metadata/sherloc_region_detail_to_ss_mapping.csv
results/sherloc_finetune/sherloc_finetune_summary.csv
results/sherloc_finetune/sherloc_target_summary.csv
results/sherloc_finetune/sherloc_loso_predictions.csv
```

These files support the clarified protocol: the final SHERLOC experiment is
treated as transfer after SHERLOC in-situ fine-tuning/adaptation, not as strict
zero-shot Earth-to-Mars transfer.

### Reviewer 1 Confidence Threshold Comments

Primary files:

```text
results/confidence_threshold_analysis/parent_test_key_thresholds_all_requested_models.csv
results/confidence_threshold_analysis/parent_test_recommended_operating_points_all_requested_models.csv
results/confidence_threshold_analysis/all_confidence_threshold_sweeps.csv
results/confidence_threshold_analysis/reviewer_requested_models_confidence_summary.md
```

These files support discussion of accuracy, macro-F1, precision, recall,
false-positive rate, and coverage under different confidence thresholds.

## Recommended Manuscript Tables

For the main manuscript, cite compact tables:

```text
data/overview/parent_by_source_type.csv
data/overview/parent_by_source_and_category.csv
data/materialized_augmented_review_ready_v1/overview_tables/split_by_class_and_augmentation.csv
results/reviewer_requested_baselines/reviewer_requested_validation_selected_summary.csv
results/confidence_threshold_analysis/parent_test_key_thresholds_all_requested_models.csv
results/sherloc_finetune/sherloc_finetune_summary.csv
```

For supporting information, cite detailed tables:

```text
data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv
data/materialized_augmented_review_ready_v1/overview_tables/lineage_manifest.csv
data/metadata/sherloc_region_point_extraction_manifest.csv
results/reviewer_requested_baselines/reviewer_requested_hyperparameter_selection_table.csv
results/confidence_threshold_analysis/all_confidence_threshold_sweeps.csv
```

## Reproducibility Commands

See `docs/reproducibility.md` for full commands. The most relevant entry
points are:

```bash
python src/build_materialized_augmented_dataset.py
python src/run_review_model_selection.py
python src/build_sherloc_region_dataset.py
python src/run_sherloc_finetune_protocol.py
python src/run_confidence_threshold_analysis.py
```
