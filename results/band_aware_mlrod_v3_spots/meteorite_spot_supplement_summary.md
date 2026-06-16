# Meteorite Spot Supplement Sensitivity Run

This run adds `data/minerals_100spots_wide.csv` to the v3 metadata as a
training-only supplement. The source file contains 800 one-row-per-spot spectra:
100 spots for each of clinopyroxene, orthopyroxene, olivine, plagioclase,
gypsum, hematite, quartz, and serpentine. The file's own `source` field is
`synthetic_teaching_only`; these rows are therefore kept transparent and should
not be described as measured meteorite spectra.

## Metadata Update

| Source group | N |
|---|---:|
| MLROD Raman open dataset | 74,961 |
| Synthetic meteorite mineral spot spectra | 800 |
| RRUFF database | 791 |
| SHERLOC in-situ Mars 2020 | 730 |
| Lab-acquired DUV spectra | 119 |
| SHERLOC calibration target Mars meteorite SaU 008 | 36 |
| Martian meteorite spectra | 7 |

Total compact metadata rows after the supplement: 77,444. The 800 supplement
rows are assigned to `split_v3=train` only.

## Main-MST Sensitivity Result

Run directory:

```text
results/band_aware_mlrod_v3_spots/band_aware_mlrod_v3_grid0-4000_n4001_20260616_131911/
```

Command:

```bash
python src/run_band_aware_mlrod_v3_experiments.py \
  --metadata-file data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv \
  --out-dir results/band_aware_mlrod_v3_spots \
  --grid-min 0 --grid-max 4000 --grid-points 4001 \
  --trial-set main \
  --main-model band_multiscale_mst_patch8_d128_ce \
  --run-sherloc \
  --epochs 45 \
  --finetune-epochs 60 \
  --batch-size 32
```

Selected benchmark size: 8,633 train, 1,934 validation, and 1,933 test spectra.
The supplement contributes 800 train spectra and no validation/test spectra.

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Reference+MLROD measured test spectra | 0.976 | 0.785 |
| SHERLOC zero-shot validation | 0.726 | 0.240 |
| SHERLOC fine-tuned validation | 0.877 | 0.693 |

Interpretation: the supplement improves SHERLOC zero-shot accuracy relative to
the measured-data final run, but it lowers the fine-tuned SHERLOC macro-F1. It
is therefore best reported as a transparent training-supplement sensitivity
experiment rather than as a replacement for the measured-data manuscript result.
