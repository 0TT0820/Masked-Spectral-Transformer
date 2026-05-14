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
| Materialized augmented dataset | Model-ready original and augmented spectra with one CSV per spectrum and full parent lineage | `data/materialized_augmented_v1/` |
| SHERLOC region dataset | Labeled point-level SHERLOC spectra for in-situ adaptation and transfer evaluation | `data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv` |

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

The materialized augmented dataset is:

```text
data/materialized_augmented_v1/
```

Important files:

```text
data/materialized_augmented_v1/metadata_materialized_augmented.csv
data/materialized_augmented_v1/augmentation_protocol.json
data/materialized_augmented_v1/overview_tables/lineage_manifest.csv
data/materialized_augmented_v1/overview_tables/split_by_class_and_augmentation.csv
data/materialized_augmented_v1/spectra/
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
| `results/mst_focused_tuning/` | MST-focused tuning tables, histories, reports, and threshold sweeps without large weight files |
| `results/sherloc_finetune/` | SHERLOC fine-tuning, leave-one-region/target predictions, and transfer summaries |
| `results/confidence_threshold_analysis/` | Accuracy, macro-F1, precision, recall, false-positive rate, and coverage across confidence thresholds |

## Recommended Files for Reuse

For reproducing the main benchmark:

```text
data/materialized_augmented_v1/metadata_materialized_augmented.csv
src/train_model_comparison.py
results/model_comparison_materialized_augmented.csv
results/model_benchmarks/hyperparameter_selection_table.csv
```

For inspecting data provenance:

```text
data/metadata/metadata_parent_945_rruff_enriched.csv
data/metadata/rruff_official_header_metadata.csv
data/overview/parent_provenance_inventory.csv
data/materialized_augmented_v1/overview_tables/lineage_manifest.csv
```

For SHERLOC adaptation:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
results/sherloc_finetune/
```

For confidence-aware deployment studies:

```text
results/confidence_threshold_analysis/all_confidence_threshold_sweeps.csv
results/confidence_threshold_analysis/parent_test_key_thresholds_all_requested_models.csv
results/confidence_threshold_analysis/parent_test_recommended_operating_points_all_requested_models.csv
```

## Reproducibility Entry Points

See `docs/reproducibility.md` for full commands. Common entry points are:

```bash
python src/build_materialized_augmented_dataset.py
python src/run_model_selection.py
python src/build_sherloc_region_dataset.py
python src/run_sherloc_finetune_protocol.py
python src/run_confidence_threshold_analysis.py
```
