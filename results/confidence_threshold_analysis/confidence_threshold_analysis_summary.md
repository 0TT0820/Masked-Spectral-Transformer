# Confidence Threshold and Rejection Analysis

Predictions with maximum class probability below threshold tau are rejected as uncertain. This supports confidence-aware deployment instead of forcing every spectrum into a hard class.

Key outputs:

- `parent_test_confidence_threshold_sweep.csv`
- `sherloc_confidence_threshold_sweep.csv`
- `confidence_threshold_key_thresholds.csv`
- `recommended_confidence_operating_points.csv`

Important metrics:

- coverage: fraction of spectra accepted;
- accuracy_on_accepted: accuracy after rejecting low-confidence spectra;
- operational_recall_correct_accepted_over_all: correct accepted predictions divided by all spectra;
- false_discovery_rate_wrong_among_accepted: wrong accepted predictions divided by accepted predictions;
- macro_one_vs_rest_fpr: macro-average one-vs-rest false-positive rate.

Recommended operating points:

| Dataset | Model | Threshold | Coverage | Accepted Accuracy | Operational Recall | False Discovery Rate | Macro one-vs-rest FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| parent_heldout_test | MST best-grid diagnostic | 0.95 | 0.820 | 0.872 | 0.714 | 0.128 | 0.009 |
| parent_heldout_test | MST validation-selected | 0.90 | 0.850 | 0.841 | 0.714 | 0.159 | 0.012 |
| parent_heldout_test | Optimized 1D-CNN | 0.90 | 0.541 | 0.847 | 0.459 | 0.153 | 0.007 |
| parent_heldout_test | PCA-SVM | 0.80 | 0.519 | 0.942 | 0.489 | 0.058 | 0.003 |
| parent_heldout_test | PLS-DA | 0.20 | 0.835 | 0.595 | 0.496 | 0.405 | 0.028 |
| parent_heldout_test | Standard Transformer | 0.95 | 0.842 | 0.857 | 0.722 | 0.143 | 0.010 |
| sherloc_leave_one_target_out | SHERLOC MST adapted | 0.75 | 0.527 | 0.853 | 0.450 | 0.147 | 0.017 |
| sherloc_leave_one_target_out | SHERLOC MST zero-shot | 0.75 | 0.601 | 0.639 | 0.384 | 0.361 | 0.045 |

Manuscript wording:

Rather than enforcing a single hard threshold of 0.5, we swept the confidence threshold from 0 to 0.95. Increasing the threshold reduces coverage but generally improves reliability among accepted predictions. Low-confidence spectra should be reported as uncertain and retained for ground review rather than forced into a mineral class.
