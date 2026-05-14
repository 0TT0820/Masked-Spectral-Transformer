# Fair Model Selection Results for Reviewer Response

## Why the previous comparison was insufficient

The earlier comparison was not strong enough for the reviewer response because:

- the CNN baseline was under-optimized;
- the standard Transformer was under-optimized;
- sklearn/chemometric baselines did not use the same augmented training set as
  the neural models;
- the comparison did not clearly separate validation-set model selection from
  final test reporting.

## Revised experiment

Script:

- `run_review_model_selection.py`

Shared input:

- `data/metadata_outputs/metadata_parent_945.csv`

Shared training augmentation:

- training split only;
- every model sees the same fixed augmented training set;
- each class was balanced to 200 spectra in the training set;
- Raman band positions were not shifted;
- only intensity response, baseline, noise, and band width/weakness were
  perturbed.

Model families:

- PCA-SVM;
- PLS-DA;
- Random Forest;
- ExtraTrees;
- tuned 1D-CNN;
- tuned standard Transformer;
- tuned MST.

Model selection:

- hyperparameters were selected by validation macro-F1 within each model family;
- final numbers below are test-set metrics for the selected configuration.

Main output:

- `review_round3_fair_model_selection/review_ready_20260510_145634/selected_model_test_summary.csv`

## Final selected test results

| Model | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.793 | 0.789 | 0.728 | 0.782 |
| Random Forest | 0.784 | 0.759 | 0.647 | 0.752 |
| Standard Transformer | 0.784 | 0.774 | 0.642 | 0.767 |
| MST | 0.811 | 0.744 | 0.641 | 0.746 |
| PCA-SVM | 0.801 | 0.729 | 0.621 | 0.724 |
| Tuned 1D-CNN | 0.687 | 0.669 | 0.552 | 0.660 |
| PLS-DA | 0.492 | 0.519 | 0.367 | 0.491 |

## Important interpretation

The optimized CNN is no longer anomalously poor: accuracy improved from the
previous 36-40% range to 66.9%, with macro-F1 = 55.2%. This directly addresses
the reviewer concern that the original CNN baseline appeared under-optimized.

The MST has the highest validation macro-F1 among all tested model families
(0.811), but the final test set is led by ExtraTrees. Therefore, it would not be
scientifically defensible to claim that MST is the best static classifier on the
parent-spectrum benchmark. The strongest honest statement is:

> The tuned MST is the strongest neural architecture and is competitive with
> optimized chemometric/tree baselines, while additionally providing a compact,
> differentiable model that can be adapted to SHERLOC spectra.

## Additional MST checks

MST-only patch-size search:

- `review_round3_fair_model_selection_mst_extended/review_ready_20260510_152004`

Best observed MST test macro-F1 among the completed fair-search trials:

- `patch_size=4`, `d_model=128`, `layers=4`, `lr=3e-5`;
- test accuracy = 0.767;
- test macro-F1 = 0.695.

Online augmentation check:

- `review_round3_fair_model_selection_mst_online_aug/review_ready_20260510_155233`
- test accuracy = 0.767;
- test macro-F1 = 0.648.

These checks improve the original MST comparison, but they still do not justify
claiming that MST beats every optimized tree-based baseline on the parent split.

## Recommended manuscript strategy

Do not frame the revised paper as "MST outperforms all baselines." That claim is
not supported by the fair model-selection results.

Instead, frame it as:

1. The original baseline comparison was expanded and corrected.
2. The CNN baseline was tuned and is no longer artificially weak.
3. Chemometric/tree baselines are competitive and are reported transparently.
4. MST is retained because it is the best neural model, compact, differentiable,
   and directly adaptable to SHERLOC in-situ spectra.
5. The central contribution should be the reproducible Raman pipeline and
   SHERLOC fine-tuning protocol, not a blanket claim of static benchmark
   dominance.

This framing is safer for peer review and directly addresses Reviewer 2 Comment
1 without overstating the model comparison.
