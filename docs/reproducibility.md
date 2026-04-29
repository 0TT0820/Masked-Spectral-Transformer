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
