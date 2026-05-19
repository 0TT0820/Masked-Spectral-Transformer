# Project Data and Results Guide

This guide explains how the repository is organized as an open, reproducible
research project. It describes the raw data, curated metadata, augmentation
products, model-training scripts, benchmark results, and SHERLOC transfer
experiments.

## Repository Workflow

The project is organized around a reproducible workflow:

1. Curate parent Raman spectra and spectrum-level metadata.
2. Enrich public database records with source provenance.
3. Build deterministic train/validation/test splits.
4. Generate Raman-aware augmented training spectra with full lineage.
5. Train and compare chemometric, convolutional, Transformer, and MST models.
6. Adapt models to labeled SHERLOC in-situ spectra.
7. Evaluate predictions under confidence-threshold and rejection settings.

## Main Data Products

| Data product | Purpose | Location |
|---|---|---|
| Parent spectra | Original non-compressed Raman spectra used as the curated parent dataset | `data/spectra/parent/` |
| Parent metadata | Spectrum-level labels, source fields, split labels, quality-control fields, and file paths | `data/metadata/metadata_parent_945.csv` |
| RRUFF-enriched metadata | Parent metadata enriched with official RRUFF header fields such as mineral name, chemistry, locality, and URL | `data/metadata/metadata_parent_945_rruff_enriched.csv` |
| RRUFF header table | Parsed official RRUFF header metadata | `data/metadata/rruff_official_header_metadata.csv` |
| Dataset overview tables | Source, class, excitation, split, and quality-control summaries | `data/overview/` |
| Materialized augmented dataset | Preprocessed original train/validation/test spectra plus deterministic training-only augmented spectra with one CSV per spectrum and full parent lineage | `data/augmented_spectra_v3/` |
| SHERLOC region dataset | Labeled point-level SHERLOC spectra for in-situ adaptation and transfer evaluation | `data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv` |
| SHERLOC PDS provenance | Product-level PDS identifiers, label URLs, observation times, and upstream raw/intermediate LIDs for SHERLOC spectra | `data/metadata/sherloc_pds_product_provenance.csv` |
| SHERLOC spectrum-PDS crosswalk | One-row-per-spectrum link from local point spectra to PDS products and labels | `data/metadata/sherloc_spectrum_to_pds_crosswalk.csv` |
| SHERLOC SaU 008 calibration target | PDS-traceable SHERLOC DUV spectra from the Mars meteorite SaU 008 calibration target; included for domain adaptation/manual review, not closed-set supervised labels | `data/metadata/metadata_sherloc_sau008_calibration_mean_spectra.csv` |
| Montpezat/Alfalfa weak candidates | PDS-traceable region-level spectra with scan-level weak labels; not default supervised training rows | `data/metadata/metadata_sherloc_montpezat_alfalfa_weak_candidates.csv` |
| All-source training database v2 | Reorganized metadata table after adding labeled SHERLOC DUV rows and SaU 008 domain-adaptation rows | `data/metadata/metadata_training_database_v2_all_sources.csv` |
| DUV spectral library v1 | DUV-only library combining laboratory DUV spectra, labeled SHERLOC in-situ spectra, and SaU 008 calibration-target spectra with explicit training roles | `data/metadata/metadata_duv_training_library_v1.csv` |
| Caltech/JPL SHERLOC-analog audit | Import audit for the 62-mineral SHERLOC-analog library; documented but not imported because machine-readable spectra were not available in the located public supplement | `data/metadata/metadata_caltech_sherloc_duv_62min_import_audit.csv` |

## Original Parent Dataset

The parent dataset contains 945 Raman spectra. The main metadata table is:

```text
data/metadata/metadata_parent_945.csv
```

The original spectrum files are stored individually rather than as a compressed
archive:

```text
data/spectra/parent/
```

Useful summary tables:

```text
data/overview/parent_by_source_type.csv
data/overview/parent_by_source_and_category.csv
data/overview/parent_by_excitation_and_source.csv
data/overview/parent_by_split_and_category.csv
data/overview/parent_provenance_inventory.csv
```

These files are the starting point for all downstream experiments.

## Public Database Provenance

RRUFF-derived spectra are documented with official header fields in:

```text
data/metadata/rruff_official_header_metadata.csv
data/metadata/metadata_parent_945_rruff_enriched.csv
data/metadata/rruff_official_header_missing.csv
```

The enrichment scripts are:

```text
src/fetch_rruff_metadata.py
src/enrich_metadata_from_rruff_headers.py
```

The enriched fields include official mineral name, ideal chemistry, measured
chemistry where available, locality, sample owner/source, identification
status, and official RRUFF URL.

## Augmented Training Dataset

The current materialized augmented dataset is:

```text
data/augmented_spectra_v3/
```

Important files:

```text
data/augmented_spectra_v3/metadata_augmented_training.csv
data/augmented_spectra_v3/augmentation_summary.json
data/augmented_spectra_v3/split_by_class_and_augmentation.csv
data/augmented_spectra_v3/source_by_class_and_augmentation.csv
data/augmented_spectra_v3/original_spectra/
data/augmented_spectra_v3/spectra/
```

Each spectrum CSV contains:

```text
raman_shift_cm-1
intensity_normalized
first_derivative_normalized
valid_mask
```

Only training spectra are augmented. Validation and test spectra remain
original spectra. The final augmentation protocol does not shift Raman band
centers; it applies bounded intensity, baseline, broadening, and noise
transformations while preserving mineral-diagnostic band positions.

To regenerate the materialized dataset:

```bash
python src/build_materialized_augmented_dataset.py \
  --metadata-file data/metadata/metadata_parent_945.csv \
  --out-dir data/materialized_augmented_v1 \
  --min-train-per-class 200 \
  --baseline poly
```

The current v3 materialized dataset is generated with:

```bash
python src/augment_raman_dataset.py \
  --metadata data/metadata/metadata_parent_945.csv \
  --out-dir data/augmented_spectra_v3 \
  --target-per-class 200 \
  --seed 2024
```

Additional augmentation documentation:

```text
docs/augmentation_rationale.md
docs/augmentation_parameters.csv
docs/augmentation_protocol.json
```

## SHERLOC In-Situ Adaptation Dataset

The repository includes a training-ready SHERLOC region table:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
```

Supporting files:

```text
data/metadata/metadata_sherloc_region_points_only.csv
data/metadata/sherloc_region_detail_to_ss_mapping.csv
data/metadata/sherloc_pds_product_provenance.csv
data/metadata/sherloc_spectrum_to_pds_crosswalk.csv
data/metadata/sherloc_region_point_extraction_manifest.csv
data/metadata/sherloc_region_table1_training_summary.csv
data/overview/sherloc_regions/
```

Only labeled point spectra are included. Unlabeled points are treated as
background/noise and are excluded from supervised training. If a point has two
accepted mineral labels, it appears as two label records with the same spectrum
source.

The dataset construction script is:

```text
src/build_sherloc_region_dataset.py
src/build_sherloc_pds_provenance.py
src/build_sherloc_montpezat_alfalfa_candidates.py
```

Montpezat and Alfalfa are kept in a separate weak-label candidate table because
Corpolongo et al. (2023) report scan-level detections for those targets without
a point-level label workbook. The extracted PDS region spectra are traceable and
useful for review, visualization, and possible weak-label/domain-adaptation
tests, but are marked `sherloc_training_label_usable=False` in the default
supervised protocol.

## Current All-Source and DUV Training Database

The current consolidated training metadata table is:

```text
data/metadata/metadata_training_database_v2_all_sources.csv
```

The DUV-only spectral library is:

```text
data/metadata/metadata_duv_training_library_v1.csv
```

This DUV library includes laboratory DUV spectra, labeled SHERLOC in-situ Mars
2020 spectra, and SHERLOC SaU 008 calibration-target spectra. The table uses
explicit role fields so that these data are not accidentally mixed in an
uncontrolled way:

```text
source_type_normalized
source_domain
training_role
supervised_label_usable_v2
duv_library_include
split_v2
```

The key count tables are:

```text
data/overview/training_database_v2/all_sources_counts.csv
data/overview/training_database_v2/duv_library_counts.csv
data/overview/training_database_v2/duv_source_by_label.csv
data/overview/training_database_v2/all_source_by_training_role.csv
```

The current DUV library contains 885 spectra: 119 laboratory DUV reference
spectra, 730 labeled SHERLOC in-situ Mars 2020 spectra, and 36 SHERLOC SaU 008
calibration-target spectra. Of these, 849 rows have supervised mineral labels
and 36 SaU 008 rows are reserved for unlabeled domain adaptation or manual
review because the PDS RRS products do not provide point-level mineral labels.

To rebuild these files:

```bash
python src/build_sherloc_sau008_calibration_dataset.py
python src/build_training_database_v2.py
```

Additional documentation is provided in:

```text
docs/training_database_v2.md
docs/sherloc_sau008_calibration_data.md
docs/caltech_sherloc_duv_62min_import_status.md
```

## Model Training Code

Core training and experiment scripts:

```text
src/train_model_comparison.py
src/run_model_selection.py
src/run_mst_focused_tuning.py
src/run_sherloc_finetune_protocol.py
src/run_sherloc_target_transfer.py
src/run_confidence_threshold_analysis.py
src/run_review_updated_training_v2.py
src/run_mst_extra_v2.py
src/run_sherloc_preprocessing_trials_v2.py
src/run_sherloc_adaptation_strategies_v2.py
src/analyze_sherloc_operating_points_v2.py
src/run_sherloc_pooled_random_validation_v2.py
src/run_sherloc_in_situ_model_comparison_v3.py
src/summarize_materialized_v3_confidence_thresholds.py
```

The implemented model families include:

- PCA-SVM
- PLS-DA
- 1D-CNN
- Standard Transformer
- Masked Spectral Transformer (MST)

The main comparison script supports fixed splits, optional augmentation, and
exported classification reports, confusion matrices, per-class metrics, and
threshold sweeps.

## Benchmark Result Directories

| Result directory | Contents |
|---|---|
| `results/model_comparison/` | Original model comparison summaries |
| `results/model_comparison_materialized_augmented.csv` | Final comparison on the materialized augmented dataset |
| `results/model_benchmarks/` | Hyperparameter selection and validation-selected baseline summaries |
| `results/materialized_augmented_v3_model_selection/` | Final model selection on the v3 materialized augmented dataset |
| `results/mst_focused_tuning/` | MST-focused tuning tables, histories, reports, and threshold sweeps without large weight files |
| `results/sherloc_finetune/` | SHERLOC fine-tuning, leave-one-region/target predictions, and transfer summaries |
| `results/sherloc_in_situ_model_comparison_v3/` | Pooled labeled SHERLOC in-situ random-split model comparison and threshold scans |
| `results/confidence_threshold_analysis/` | Accuracy, macro-F1, precision, recall, false-positive rate, and coverage across confidence thresholds |
| `results/confidence_threshold_materialized_v3/` | Final key confidence thresholds for the reference benchmark and pooled SHERLOC in-situ validation |
| `results/review_updated_training_v2/` | Reviewer-oriented rerun on the updated v2 all-source database, DUV library, and SHERLOC fine-tuning pool |

## Recommended Files for Reuse

For reproducing the main benchmark:

```text
data/augmented_spectra_v3/metadata_augmented_training.csv
src/augment_raman_dataset.py
src/run_model_selection.py
results/materialized_augmented_v3_model_selection/curated_20260519_104300/reviewer_requested_model_test_summary.csv
```

For inspecting data provenance:

```text
data/metadata/metadata_parent_945_rruff_enriched.csv
data/metadata/rruff_official_header_metadata.csv
data/overview/parent_provenance_inventory.csv
data/augmented_spectra_v3/metadata_augmented_training.csv
```

For SHERLOC adaptation:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
results/sherloc_finetune/
results/sherloc_in_situ_model_comparison_v3/
```

For confidence-aware deployment studies:

```text
results/confidence_threshold_analysis/all_confidence_threshold_sweeps.csv
results/confidence_threshold_analysis/parent_test_key_thresholds_all_requested_models.csv
results/confidence_threshold_analysis/parent_test_recommended_operating_points_all_requested_models.csv
results/confidence_threshold_materialized_v3/combined_key_thresholds_requested_models.csv
results/confidence_threshold_materialized_v3/confidence_threshold_summary.md
```

## Reproducibility Entry Points

See `docs/reproducibility.md` for full commands. Common entry points are:

```bash
python src/build_materialized_augmented_dataset.py
python src/run_model_selection.py
python src/build_sherloc_region_dataset.py
python src/build_sherloc_sau008_calibration_dataset.py
python src/build_training_database_v2.py
python src/run_sherloc_finetune_protocol.py
python src/run_confidence_threshold_analysis.py
python src/run_review_updated_training_v2.py
python src/run_sherloc_preprocessing_trials_v2.py --run-dir results/review_updated_training_v2/<run_id>
python src/run_sherloc_adaptation_strategies_v2.py --run-dir results/review_updated_training_v2/<run_id>
python src/analyze_sherloc_operating_points_v2.py --run-dir results/review_updated_training_v2/<run_id>
python src/run_sherloc_pooled_random_validation_v2.py --run-dir results/review_updated_training_v2/<run_id>
python src/run_sherloc_in_situ_model_comparison_v3.py
python src/summarize_materialized_v3_confidence_thresholds.py
```
