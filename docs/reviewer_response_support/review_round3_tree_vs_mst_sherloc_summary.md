# Tree Ensembles Versus MST on SHERLOC Adaptation

## Tree-ensemble adaptation protocol

Script:

- `run_tree_sherloc_transfer.py`

Output:

- `review_round3_tree_sherloc_transfer/review_ready_20260511_121710`

Protocol:

- The parent 945-spectrum training split was physically augmented and balanced
  to 2,600 spectra, exactly as in the fair model-selection experiment.
- SHERLOC spectra were not augmented.
- For each held-out SHERLOC target, Random Forest and ExtraTrees were refit on:
  parent augmented train set + SHERLOC rows from the other three targets.
- The held-out target was used only for evaluation.
- Three Perchlorate rows were excluded because the current 945-spectrum base
  class set does not contain Perchlorate.

## Overall SHERLOC results

| Model | Adaptation method | Zero-shot label accuracy | Adapted label accuracy | Zero-shot macro-F1 | Adapted macro-F1 | Zero-shot point any-match | Adapted point any-match |
|---|---|---:|---:|---:|---:|---:|---:|
| MST | parameter fine-tuning | 0.553 | 0.710 | 0.276 | 0.431 | 0.588 | 0.754 |
| Random Forest | refit with SHERLOC training targets | 0.717 | 0.777 | 0.375 | 0.485 | 0.762 | 0.826 |
| ExtraTrees | refit with SHERLOC training targets | 0.731 | 0.744 | 0.435 | 0.451 | 0.777 | 0.791 |

MST output source:

- `review_round2_sherloc_target_transfer/mst945_lr3e5_160ep_lastblock_loto`

Tree output source:

- `review_round3_tree_sherloc_transfer/review_ready_20260511_121710`

## By held-out target

### Random Forest

| Held-out target | Zero-shot label acc. | Adapted label acc. | Zero-shot point any-match | Adapted point any-match |
|---|---:|---:|---:|---:|
| Dourbes | 0.531 | 0.750 | 0.591 | 0.835 |
| Garde | 0.571 | 0.589 | 0.633 | 0.652 |
| Guillaumes | 0.615 | 0.846 | 0.615 | 0.846 |
| Quartier | 0.850 | 0.871 | 0.875 | 0.897 |

### ExtraTrees

| Held-out target | Zero-shot label acc. | Adapted label acc. | Zero-shot point any-match | Adapted point any-match |
|---|---:|---:|---:|---:|
| Dourbes | 0.578 | 0.758 | 0.643 | 0.843 |
| Garde | 0.589 | 0.497 | 0.652 | 0.551 |
| Guillaumes | 0.538 | 0.769 | 0.538 | 0.769 |
| Quartier | 0.855 | 0.853 | 0.881 | 0.878 |

### MST

| Held-out target | Zero-shot label acc. | Adapted label acc. | Zero-shot point any-match | Adapted point any-match |
|---|---:|---:|---:|---:|
| Dourbes | 0.297 | 0.555 | 0.330 | 0.617 |
| Garde | 0.166 | 0.463 | 0.184 | 0.513 |
| Guillaumes | 0.308 | 0.923 | 0.308 | 0.923 |
| Quartier | 0.826 | 0.868 | 0.851 | 0.894 |

## Interpretation

The tree ensembles are strong not only on the static parent-spectrum benchmark
but also after SHERLOC-domain refitting. Random Forest gives the best overall
adapted SHERLOC score in this experiment.

Therefore, the manuscript should not claim that MST is the highest-performing
model after SHERLOC adaptation. A defensible revised claim is:

> Tree ensembles provide strong static and refit baselines. MST is retained as a
> compact neural architecture that supports parameter-efficient fine-tuning,
> differentiable representation learning, and future onboard adaptation, but the
> revised results transparently show that optimized tree ensembles are highly
> competitive.

This framing is safer for the reviewer response because it acknowledges the
strong baselines instead of forcing MST to appear best.
