# Band-Aware MLROD v3 Experiment Summary

Run directory: `results\band_aware_mlrod_v3_spots\band_aware_mlrod_v3_grid0-4000_n4001_20260616_131911`

Grid: 0-4000 cm-1, 4001 points.
Selected benchmark: 8633 train, 1934 validation, 1933 test spectra.

Validation-best screening candidate: `band_multiscale_mst_patch8_d128_ce`.
Manuscript main reported MST configuration: `band_multiscale_mst_patch8_d128_ce`.

## Reference + MLROD Candidate Models

| model | validation_macro_f1 | combined_test_accuracy | combined_test_macro_f1 | curated_reference_test_accuracy | curated_reference_test_macro_f1 | mlrod_test_accuracy | mlrod_test_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| band_multiscale_mst_patch8_d128_ce | 0.898 | 0.976 | 0.785 | 0.767 | 0.615 | 0.991 | 0.991 |

## SHERLOC Fine-Tuning of Manuscript Main Candidate

| sherloc_rows | train_n | validation_n | split_note | finetune_mode | zero_shot_accuracy | zero_shot_macro_f1 | finetuned_accuracy | finetuned_macro_f1 | best_finetune_validation_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 730 | 584 | 146 | stratified_for_labels_with_n>=2_singletons_forced_to_train | last_block_pool_head | 0.726 | 0.240 | 0.877 | 0.693 | 0.693 |
