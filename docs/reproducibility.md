# Reproducibility

## Main Comparison

The main comparison table was produced with:

```bash
python src/train_review_comparison.py --models pca_svm pls_da random_forest cnn standard_transformer mst --epochs 180 --batch-size 16 --lr 1e-4 --baseline poly --chemometric-stride 8 --no-augment
```

For a faster baseline-only check:

```bash
python src/train_review_comparison.py --models pca_svm pls_da random_forest --baseline poly --chemometric-stride 8 --no-augment
```

For a smoke test:

```bash
python src/train_review_comparison.py --models pca_svm pls_da --baseline poly --chemometric-stride 8
```

## Reviewer-Requested Baselines

Reviewer 2 requested interpretable chemometric baselines and a fairer CNN
baseline. The reviewer-facing comparison therefore includes PCA-SVM, PLS-DA,
1D-CNN, Standard Transformer, and MST, with validation-based hyperparameter
selection before test reporting.

The archived summaries are:

```text
results/reviewer_requested_baselines/reviewer_requested_hyperparameter_selection_table.csv
results/reviewer_requested_baselines/reviewer_requested_validation_selected_summary.csv
results/reviewer_requested_baselines/reviewer_requested_best_observed_grid_summary.csv
results/reviewer_requested_baselines/hyperparameter_selection_summary.md
```

To rerun the same family of experiments:

```bash
python src/run_review_model_selection.py
python src/summarize_reviewer_requested_baselines.py
python src/summarize_hyperparameter_selection.py
```

## Materialized Augmented Dataset

The reviewer-ready augmentation is deterministic and materialized as one CSV
per spectrum. Validation and test spectra remain original spectra; augmentation
is applied only to the training split.

```bash
python src/build_materialized_augmented_dataset.py \
  --metadata-file data/metadata/metadata_parent_945.csv \
  --out-dir data/materialized_augmented_review_ready_v1 \
  --min-train-per-class 200 \
  --baseline poly
```

The resulting master metadata table is:

```text
data/materialized_augmented_review_ready_v1/metadata_review_ready_materialized_augmented.csv
```

Each spectrum file contains the model-ready point-wise values:

- `raman_shift_cm-1`
- `intensity_normalized`
- `first_derivative_normalized`
- `valid_mask`

## SHERLOC Fine-Tuning Protocol

SHERLOC region spectra extracted from Dourbes, Garde/Bellegarde, Guillaumes,
and Quartier are summarized in:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
data/overview/sherloc_regions/
```

The fine-tuning and target-transfer summaries are archived in:

```text
results/sherloc_finetune/
```

To rerun the protocol:

```bash
python src/build_sherloc_region_dataset.py
python src/run_sherloc_finetune_protocol.py
```

## Confidence Thresholds

Reviewer 1 requested precision, recall, and false-positive-rate behavior as a
function of confidence threshold. Reviewer-requested threshold scans are
archived in:

```text
results/confidence_threshold_analysis/
```

To rerun the threshold analysis:

```bash
python src/run_confidence_threshold_analysis.py
python src/summarize_all_requested_confidence_thresholds.py
```

The key table for manuscript reporting is:

```text
results/confidence_threshold_analysis/parent_test_key_thresholds_all_requested_models.csv
```

## Current Best Summary

The current model-comparison summary is:

```text
results/model_comparison/best_by_model_summary.csv
```

The best current MST setting uses:

- label scheme: `review_ready`
- baseline correction: `poly`
- train-time augmentation: disabled
- learning rate: `1e-4`
- epochs: `180`
- batch size: `16`

## Randomness

Scripts use a fixed seed of `2024` unless otherwise specified. Deep-learning results may still vary slightly across hardware, CUDA versions, and PyTorch versions.

## Computational Requirements

Chemometric baselines run in seconds on a CPU. CNN, Standard Transformer, and MST are faster on a CUDA GPU. The patch-token Transformer implementation reduces the original 4100-point sequence into shorter spectral tokens to keep training practical while retaining physical wavenumber information.

Large trained weights are intentionally not committed. The repository contains
training histories, classification reports, confusion matrices, threshold
tables, and run manifests sufficient to evaluate and rerun the reported
experiments.
