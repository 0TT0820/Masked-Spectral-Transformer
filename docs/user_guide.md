# User Guide

## Main Scripts

### `src/train_review_comparison.py`

Runs model comparisons on the parent Raman dataset.

Supported models:

- `pca_svm`
- `pls_da`
- `random_forest`
- `cnn`
- `standard_transformer`
- `mst`

Common options:

```text
--label-scheme review_ready|original_major
--include-review-required
--models MODEL [MODEL ...]
--epochs INTEGER
--batch-size INTEGER
--lr FLOAT
--baseline none|poly|asls
--smooth
--augment / --no-augment
--chemometric-stride INTEGER
--out-dir PATH
```

Expected outputs:

```text
model_comparison_summary.csv
experiment_manifest.json
experiment_samples.csv
*.classification_report.csv
*.confusion_matrix.csv
*.per_class.csv
*.threshold_sweep.csv
*.history.csv
*.pth
```

### `src/augment_raman_dataset.py`

Generates Raman-aware augmented spectra with parent-level lineage.

Common options:

```text
--metadata PATH
--out-dir PATH
--target-per-class INTEGER
--seed INTEGER
--include-review-required
--dry-run
```

Expected outputs:

```text
data/augmented_spectra/augmented_lineage.csv
data/augmented_spectra/augmentation_summary.json
data/augmented_spectra/spectra/*.csv
```

## Label Schemes

`review_ready` is recommended for manuscript revision. It removes Halides from supervised Raman-active classes and harmonizes Clay, Mica, and Serpentine into `Phyllosilicates`.

`original_major` keeps the original major categories and is intended only for comparison with older experiments.

## Confidence Thresholds

Every classifier writes a threshold sweep table. This table reports accepted coverage, accuracy on accepted spectra, macro-F1 on accepted spectra, and rejected count for thresholds from 0 to 0.95. It supports uncertainty-aware reporting instead of a fixed arbitrary confidence threshold.
