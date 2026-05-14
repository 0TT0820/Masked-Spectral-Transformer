# Benchmark Model Summary

This summary focuses on the core open benchmark model families: PCA-SVM,
PLS-DA, 1D-CNN, Standard Transformer, and MST.

## Validation-Selected Results

| Model | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|
| Standard Transformer | 0.784 | 0.774 | 0.642 | 0.767 |
| MST | 0.811 | 0.744 | 0.641 | 0.746 |
| PCA-SVM | 0.801 | 0.729 | 0.621 | 0.724 |
| Optimized 1D-CNN | 0.687 | 0.669 | 0.552 | 0.660 |
| PLS-DA | 0.492 | 0.519 | 0.367 | 0.491 |

## Best Observed Configuration Within the Tuning Grid

This diagnostic table shows the highest test Macro-F1 reached inside the explored grid for each model family. It should be used carefully because selecting directly by test performance is not a strict model-selection protocol.

| Model | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|
| PCA-SVM | 0.765 |  | 0.702 |  |
| MST | 0.725 | 0.767 | 0.695 | 0.763 |
| Standard Transformer | 0.759 | 0.759 | 0.686 | 0.744 |
| Optimized 1D-CNN | 0.687 | 0.669 | 0.552 | 0.660 |
| PLS-DA | 0.476 |  | 0.385 |  |

## Recommended Use

Use the validation-selected table for the main manuscript. The tuned CNN should replace the original weak CNN baseline. PCA-SVM and PLS-DA should be described as the requested interpretable chemometric baselines.

The current evidence supports the claim that MST is competitive and has the highest validation Macro-F1 among the benchmark models. It does not support an unconditional statement that MST is numerically superior on every test metric.
