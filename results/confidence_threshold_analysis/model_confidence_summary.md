# Confidence Thresholds for Benchmark Models

This summary covers the parent held-out test set for all benchmark model families. SHERLOC leave-one-target-out confidence analysis is currently available for MST because the completed SHERLOC fine-tuning protocol was implemented for MST.

## Recommended Operating Points

| Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | Wrong Accepted Rate | Macro one-vs-rest FPR |
|---|---:|---:|---:|---:|---:|---:|
| PCA-SVM | 0.800 | 0.519 | 0.942 | 0.489 | 0.058 | 0.003 |
| PLS-DA | 0.200 | 0.835 | 0.595 | 0.496 | 0.405 | 0.028 |
| Optimized 1D-CNN | 0.900 | 0.541 | 0.847 | 0.459 | 0.153 | 0.007 |
| Standard Transformer | 0.950 | 0.842 | 0.857 | 0.722 | 0.143 | 0.010 |
| MST validation-selected | 0.900 | 0.850 | 0.841 | 0.714 | 0.159 | 0.012 |
| MST best-grid diagnostic | 0.950 | 0.820 | 0.872 | 0.714 | 0.128 | 0.009 |

## Key Thresholds

| Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | Wrong Accepted Rate | Macro one-vs-rest FPR |
|---|---:|---:|---:|---:|---:|---:|
| PCA-SVM | 0.000 | 1.000 | 0.759 | 0.759 | 0.241 | 0.021 |
| PCA-SVM | 0.500 | 0.820 | 0.817 | 0.669 | 0.183 | 0.013 |
| PCA-SVM | 0.700 | 0.684 | 0.868 | 0.594 | 0.132 | 0.008 |
| PCA-SVM | 0.800 | 0.519 | 0.942 | 0.489 | 0.058 | 0.003 |
| PCA-SVM | 0.900 | 0.368 | 0.980 | 0.361 | 0.020 | 0.001 |
| PLS-DA | 0.000 | 1.000 | 0.534 | 0.534 | 0.466 | 0.039 |
| PLS-DA | 0.500 | 0.068 | 1.000 | 0.068 | 0.000 | 0.000 |
| PLS-DA | 0.700 | 0.023 | 1.000 | 0.023 | 0.000 | 0.000 |
| PLS-DA | 0.800 | 0.000 |  | 0.000 |  | 0.000 |
| PLS-DA | 0.900 | 0.000 |  | 0.000 |  | 0.000 |
| Optimized 1D-CNN | 0.000 | 1.000 | 0.669 | 0.669 | 0.331 | 0.028 |
| Optimized 1D-CNN | 0.500 | 0.895 | 0.706 | 0.632 | 0.294 | 0.023 |
| Optimized 1D-CNN | 0.700 | 0.744 | 0.758 | 0.564 | 0.242 | 0.015 |
| Optimized 1D-CNN | 0.800 | 0.669 | 0.809 | 0.541 | 0.191 | 0.011 |
| Optimized 1D-CNN | 0.900 | 0.541 | 0.847 | 0.459 | 0.153 | 0.007 |
| Standard Transformer | 0.000 | 1.000 | 0.774 | 0.774 | 0.226 | 0.019 |
| Standard Transformer | 0.500 | 0.970 | 0.791 | 0.767 | 0.209 | 0.017 |
| Standard Transformer | 0.700 | 0.932 | 0.815 | 0.759 | 0.185 | 0.015 |
| Standard Transformer | 0.800 | 0.917 | 0.828 | 0.759 | 0.172 | 0.013 |
| Standard Transformer | 0.900 | 0.887 | 0.831 | 0.737 | 0.169 | 0.013 |
| MST validation-selected | 0.000 | 1.000 | 0.744 | 0.744 | 0.256 | 0.022 |
| MST validation-selected | 0.500 | 0.977 | 0.762 | 0.744 | 0.238 | 0.020 |
| MST validation-selected | 0.700 | 0.925 | 0.789 | 0.729 | 0.211 | 0.017 |
| MST validation-selected | 0.800 | 0.895 | 0.807 | 0.722 | 0.193 | 0.015 |
| MST validation-selected | 0.900 | 0.850 | 0.841 | 0.714 | 0.159 | 0.012 |
| MST best-grid diagnostic | 0.000 | 1.000 | 0.767 | 0.767 | 0.233 | 0.020 |
| MST best-grid diagnostic | 0.500 | 0.970 | 0.791 | 0.767 | 0.209 | 0.017 |
| MST best-grid diagnostic | 0.700 | 0.917 | 0.803 | 0.737 | 0.197 | 0.015 |
| MST best-grid diagnostic | 0.800 | 0.910 | 0.810 | 0.737 | 0.190 | 0.015 |
| MST best-grid diagnostic | 0.900 | 0.865 | 0.843 | 0.729 | 0.157 | 0.012 |

## Note

For the manuscript, the main confidence-threshold table can include all parent-test models, while the SHERLOC operational rejection analysis should be reported for the SHERLOC-adapted MST protocol. If needed, analogous SHERLOC adaptation experiments can be run for the CNN and standard Transformer, but PCA-SVM/PLS-DA adaptation would require refitting rather than neural fine-tuning.
