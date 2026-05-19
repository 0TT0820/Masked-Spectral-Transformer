# Reproducibility

## Main Comparison

The main comparison table was produced with:

```bash
python src/train_model_comparison.py --models pca_svm pls_da random_forest cnn standard_transformer mst --epochs 180 --batch-size 16 --lr 1e-4 --baseline poly --chemometric-stride 8 --no-augment
```

For a faster baseline-only check:

```bash
python src/train_model_comparison.py --models pca_svm pls_da random_forest --baseline poly --chemometric-stride 8 --no-augment
```

For a smoke test:

```bash
python src/train_model_comparison.py --models pca_svm pls_da --baseline poly --chemometric-stride 8
```

## Model Benchmark Suite

The benchmark suite includes PCA-SVM, PLS-DA, 1D-CNN, Standard Transformer,
and MST. Hyperparameters are selected on the validation split before test
reporting.

The archived summaries are:

```text
results/model_benchmarks/hyperparameter_selection_table.csv
results/model_benchmarks/validation_selected_summary.csv
results/model_benchmarks/best_observed_grid_summary.csv
results/model_benchmarks/hyperparameter_selection_summary.md
```

To rerun the current materialized-augmentation benchmark:

```bash
python src/augment_raman_dataset.py \
  --metadata data/metadata/metadata_parent_945.csv \
  --out-dir data/augmented_spectra_v3 \
  --target-per-class 200 \
  --seed 2024

python src/run_model_selection.py \
  --metadata-file data/augmented_spectra_v3/metadata_augmented_training.csv \
  --out-dir results/materialized_augmented_v3_model_selection \
  --baseline none \
  --min-per-class 200 \
  --max-per-class 260 \
  --epochs 120 \
  --batch-size 24 \
  --refresh-cache

python src/summarize_model_benchmarks.py
python src/summarize_hyperparameter_selection.py
```

## Materialized Augmented Dataset

The final augmentation dataset is deterministic and materialized as one CSV
per spectrum. Validation and test spectra remain original spectra; augmentation
is applied only to the training split.

```bash
python src/augment_raman_dataset.py \
  --metadata data/metadata/metadata_parent_945.csv \
  --out-dir data/augmented_spectra_v3 \
  --target-per-class 200 \
  --seed 2024
```

The resulting master metadata table is:

```text
data/augmented_spectra_v3/metadata_augmented_training.csv
```

Each spectrum file contains point-wise Raman-shift and normalized-intensity
values. Original validation and test spectra are preprocessed and materialized
without stochastic augmentation, while training spectra include both the
preprocessed parent spectra and deterministic augmented derivatives.

The current materialized dataset contains 897 original spectra and 1,970
augmented training spectra, for 2,867 rows in the combined metadata table.

## SHERLOC In-Situ Model Comparison

The pooled labeled SHERLOC in-situ validation compares the same
reviewer-requested model families as the reference benchmark: PCA-SVM, PLS-DA,
1D-CNN, Standard Transformer, and MST. It uses repeated random splits over the
pooled labeled SHERLOC spectra and should be interpreted as within-domain
SHERLOC adaptation validation, not as independent target-transfer validation.

```bash
python src/run_sherloc_in_situ_model_comparison_v3.py \
  --metadata-file data/metadata/metadata_training_database_v2_all_sources.csv \
  --out-dir results/sherloc_in_situ_model_comparison_v3 \
  --variant despike_sg11_asls \
  --seeds 2024 2025 2026 \
  --epochs 60 \
  --batch-size 32
```

The key outputs are:

```text
results/sherloc_in_situ_model_comparison_v3/sherloc_in_situ_model_comparison_aggregate.csv
results/sherloc_in_situ_model_comparison_v3/sherloc_in_situ_key_thresholds.csv
results/sherloc_in_situ_model_comparison_v3/sherloc_in_situ_validation_predictions.csv
```

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

Confidence-threshold scans report precision, recall, false-positive rate,
accuracy, macro-F1, and coverage as a function of accepted prediction
confidence. The archived tables are in:

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

For the final materialized-augmentation benchmark and the pooled SHERLOC
in-situ validation, run:

```bash
python src/summarize_materialized_v3_confidence_thresholds.py
```

The combined key-threshold tables are:

```text
results/confidence_threshold_materialized_v3/reference_test_key_thresholds_requested_models.csv
results/confidence_threshold_materialized_v3/sherloc_in_situ_key_thresholds_requested_models.csv
results/confidence_threshold_materialized_v3/combined_key_thresholds_requested_models.csv
results/confidence_threshold_materialized_v3/confidence_threshold_summary.md
```

## Current Best Summary

The current model-comparison summary is:

```text
results/model_comparison/best_by_model_summary.csv
```

The best current MST setting uses:

- label scheme: `curated`
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
