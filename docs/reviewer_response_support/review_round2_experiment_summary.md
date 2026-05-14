# Reviewer-Oriented Training and SHERLOC Fine-Tuning Summary

## Experiment Design

### Stage 1: Closed-set parent-spectrum training

- Input metadata: `data/metadata_outputs/metadata_parent_945.csv`.
- After QC filtering and review-ready label harmonization, 897 spectra were used.
- Labels were reduced to 13 model classes: Borate, Carbonate, Glass,
  K-Feldspar, Olivine, Other Silicates, Oxides/Hydroxides, Phosphate,
  Phyllosilicates, Plagioclase, Pyroxene, Silica Phase, and Sulfate.
- Training-time augmentation was applied only to the training split. The
  augmentation does not translate Raman band positions. It changes relative
  intensity response, residual baseline, noise, and band width/weakness.
- Baselines were expanded to address Reviewer 2 Comment 1:
  PCA-SVM, PLS-DA, Random Forest, 1D-CNN, standard Transformer, and MST.

### Stage 2: SHERLOC target-level adaptation

- Input metadata:
  `data/metadata_outputs/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv`.
- Only SHERLOC point labels that map unambiguously to the current 945-spectrum
  closed-set model were used.
- 699 SHERLOC labels were Table 1 training-ready; 696 were usable by the
  current base model. The three excluded labels were Perchlorate because the
  original 945-spectrum parent dataset does not contain a Perchlorate class.
- No SHERLOC spectra were augmented during fine-tuning.
- Evaluation used leave-one-target-out transfer: the MST was fine-tuned on three
  SHERLOC targets and tested on the held-out target.
- Fine-tuning mode: last transformer block + normalization layer + classifier
  head, with L2-SP regularization to reduce catastrophic forgetting.

## Stage 1 Results

Main baseline run:
`review_round2_base_945_augmented/review_ready_20260509_152910/model_comparison_summary.csv`

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Random Forest | 0.782 | 0.709 | 0.763 |
| PCA-SVM | 0.699 | 0.621 | 0.690 |
| MST | 0.759 | 0.621 | 0.755 |
| PLS-DA | 0.556 | 0.396 | 0.468 |
| 1D-CNN | 0.406 | 0.344 | 0.377 |
| Standard Transformer | 0.353 | 0.237 | 0.355 |

MST tuning run:
`review_round2_base_945_augmented_tuned_lr3e5/review_ready_20260509_153216/model_comparison_summary.csv`

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| MST, 160 epochs | 0.774 | 0.638 | 0.770 |

Interpretation: Random Forest remains the strongest conventional baseline on
the fixed parent split, while MST is the strongest neural architecture and
approaches Random Forest accuracy with a deployable transformer architecture.
This should be written conservatively: MST is not claimed to dominate every
chemometric baseline, but it provides competitive accuracy with a compact
end-to-end model that can be adapted to SHERLOC spectra.

## Stage 2 Results

Best SHERLOC target-transfer run:
`review_round2_sherloc_target_transfer/mst945_lr3e5_160ep_lastblock_loto`

Overall:

| Protocol | Label-row accuracy | Macro-F1 | Point any-match accuracy |
|---|---:|---:|---:|
| Zero-shot MST | 0.553 | 0.276 | 0.588 |
| SHERLOC-adapted MST | 0.710 | 0.431 | 0.754 |

By held-out target:

| Held-out target | Label rows | Unique points | Zero-shot acc. | Fine-tuned acc. | Zero-shot point any-match | Fine-tuned point any-match |
|---|---:|---:|---:|---:|---:|---:|
| Dourbes | 128 | 115 | 0.297 | 0.555 | 0.330 | 0.617 |
| Garde | 175 | 158 | 0.166 | 0.463 | 0.184 | 0.513 |
| Guillaumes | 13 | 13 | 0.308 | 0.923 | 0.308 | 0.923 |
| Quartier | 380 | 369 | 0.826 | 0.868 | 0.851 | 0.894 |

Interpretation: The SHERLOC adaptation experiment directly addresses the
reviewer concern about whether the model is zero-shot or fine-tuned. The revised
protocol is explicitly fine-tuned transfer, not strict zero-shot Earth-to-Mars
prediction. Fine-tuning improves target-level transfer from 55.3% to 71.0% by
label-row accuracy and from 58.8% to 75.4% by point-level any-match accuracy.

## Recommended Manuscript Framing

- Replace broad claims such as "Earth-trained, Mars-applied" with
  "Earth-pretrained, SHERLOC-adapted, Mars-applied".
- State that chemometric baselines were included because PCA-SVM and PLS-DA are
  standard in Raman/spectroscopy classification.
- Report that Random Forest and PCA-SVM are competitive, and emphasize MST's
  value as a compact, differentiable, fine-tunable architecture rather than as a
  universally superior classifier.
- Clarify that SHERLOC spectra were not augmented during fine-tuning; only the
  parent training split used physics-aware augmentation.
- Explain that ambiguous multi-mineral salt labels were retained for
  traceability but not silently merged into closed-set training classes.
