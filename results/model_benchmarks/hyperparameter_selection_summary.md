# Hyperparameter Selection Summary

## Why validation macro-F1 matters

Validation macro-F1 is the model-selection criterion, not an extra final-performance metric. It is used to choose hyperparameters without looking at the independent test set. This is especially important for this dataset because the mineral classes are imbalanced; macro-F1 weights minority classes equally rather than allowing abundant classes to dominate the choice.

A high validation macro-F1 means that MST was the preferred model family under the pre-specified validation-based selection rule. It does not by itself prove that MST has the highest final test score on every metric. Therefore, the manuscript should report both validation-selected hyperparameters and independent test performance.

## Benchmark Model Tuning

| Model | Search space | Selected hyperparameters | Val Macro-F1 | Test Acc. | Test Macro-F1 | Test Weighted-F1 |
|---|---|---|---:|---:|---:|---:|
| Standard Transformer | learning rate {1e-4, 3e-5}; d_model {96, 128}; layers {3, 4}; patch size 8 | lr=0.0001; d_model=96; layers=3; patch=8 | 0.784 | 0.774 | 0.642 | 0.767 |
| MST | learning rate {1e-4, 3e-5}; d_model {96, 128}; layers {3, 4}; patch size {8, 4} | lr=3e-05; d_model=128; layers=4; patch=8 | 0.811 | 0.744 | 0.641 | 0.746 |
| PCA-SVM | stride {4, 8}; PCA components {40, 80}; SVM C {1, 10}; gamma {scale, 0.001} | stride=8; pca=40; C=10.0; gamma=scale | 0.801 | 0.729 | 0.621 | 0.724 |
| Optimized 1D-CNN | learning rate {1e-3, 3e-4}; dropout {0.25, 0.40}; tuned 4-block 1D-CNN | lr=0.0003; dropout=0.4 | 0.687 | 0.669 | 0.552 | 0.660 |
| PLS-DA | stride {4, 8}; latent components {4, 8, 12} | stride=4; components=12 | 0.492 | 0.519 | 0.367 | 0.491 |

## Manuscript wording recommendation

Use wording such as: 'Hyperparameters for all baseline models were selected on the validation split using macro-F1, and the final selected models were evaluated once on the held-out test split.'

Avoid wording such as: 'MST is superior because it has the highest validation macro-F1.' The appropriate claim is narrower: 'MST achieved the highest validation macro-F1 among the benchmark models and competitive independent test performance, while retaining the architectural advantages required for parameter-efficient SHERLOC adaptation.'
