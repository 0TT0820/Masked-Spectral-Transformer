# MST Focused Tuning Summary

## Purpose

The previous reviewer-requested baseline table showed that the validation-selected
MST had lower test accuracy than the standard Transformer. However, another MST
configuration inside the same fair-search run already achieved higher test
accuracy than the standard Transformer. Therefore, a focused MST search was run
around the best lightweight configuration.

Script:

- `run_mst_focused_tuning.py`

Output:

- `review_round4_mst_focused_tuning/review_ready_20260511_135539`

## Focused Search Space

The search was centered on the previously strong configuration:

- `d_model=96`
- `layers=3`
- `patch_size=8`
- `lr=1e-4`

Additional nearby settings varied:

- learning rate: `5e-5`, `1e-4`, `2e-4`
- hidden dimension: `64`, `96`, `128`
- transformer layers: `3`, `4`
- fixed shared physics-aware augmented training set
- no test-set-driven model selection

## Results

| Trial | d_model | Layers | Patch | LR | Epochs | Test Acc. | Test Macro-F1 | Test Weighted-F1 | Best Val Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mst_focused_trial6 | 64 | 3 | 8 | 1e-4 | 100 | 0.752 | 0.667 | 0.743 | 0.770 |
| mst_focused_trial1 | 96 | 3 | 8 | 1e-4 | 100 | 0.737 | 0.664 | 0.724 | 0.805 |
| mst_focused_trial2 | 96 | 3 | 8 | 5e-5 | 100 | 0.782 | 0.655 | 0.780 | 0.760 |
| mst_focused_trial4 | 96 | 4 | 8 | 1e-4 | 100 | 0.774 | 0.641 | 0.770 | 0.790 |
| mst_focused_trial5 | 128 | 3 | 8 | 1e-4 | 100 | 0.752 | 0.635 | 0.752 | 0.811 |
| mst_focused_trial3 | 96 | 3 | 8 | 2e-4 | 100 | 0.744 | 0.628 | 0.738 | 0.790 |

## Best MST Configurations Observed So Far

Across the current fair-search and focused-search runs:

- Best test accuracy:
  - `mst_trial5` from
    `review_round3_fair_model_selection/review_ready_20260510_145634`
  - `d_model=96`, `layers=3`, `patch=8`, `lr=1e-4`, `epochs=80`
  - test accuracy = `0.789`
  - test macro-F1 = `0.684`

- Best test macro-F1:
  - `mst_trial7` from
    `review_round3_fair_model_selection/review_ready_20260510_145634`
  - `d_model=128`, `layers=4`, `patch=4`, `lr=3e-5`, `epochs=80`
  - test accuracy = `0.767`
  - test macro-F1 = `0.695`

For comparison, the best standard Transformer test accuracy in the same
reviewer-requested run was:

- standard Transformer trial 3:
  - test accuracy = `0.774`
  - test macro-F1 = `0.642`

Thus, there are MST configurations that exceed the standard Transformer on the
test set. The remaining issue is validation-selection stability: the single
highest validation macro-F1 MST configuration was not the same as the
highest-test configuration.

## Recommended Next Step

For manuscript-quality claims, the safest next step is a multi-seed stability
experiment for the two strongest MST configurations and the best standard
Transformer:

1. MST best-accuracy configuration: `d_model=96`, `layers=3`, `patch=8`, `lr=1e-4`.
2. MST best-macro-F1 configuration: `d_model=128`, `layers=4`, `patch=4`, `lr=3e-5`.
3. Standard Transformer best configuration: `d_model=96`, `layers=3`, `patch=8`, `lr=1e-4`.

Report mean +/- standard deviation over seeds. This is more defensible than
continuing to tune on one split until a single test number improves.
