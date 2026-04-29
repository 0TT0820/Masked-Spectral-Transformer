# Tutorial 2: Generate a Traceable Augmented Dataset

Run:

```bash
python src/augment_raman_dataset.py --target-per-class 200 --seed 2024
```

Inspect:

```text
data/augmented_spectra/augmented_lineage.csv
```

Each row links one augmented spectrum to its parent spectrum and stores the exact augmentation parameters in JSON format.

The augmentation does not move Raman band positions. This is intentional because diagnostic band positions are physically meaningful and should not be randomly shifted for a fixed mineral phase.
