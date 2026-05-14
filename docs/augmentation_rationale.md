# Raman-Aware Data Augmentation Rationale

## Physical Basis

For a given crystalline mineral phase, Raman band positions are controlled primarily by vibrational modes of the crystal structure. Under comparable measurement conditions, diagnostic band positions should remain relatively stable. Therefore, the augmentation policy in this repository does **not** randomly shift band positions.

The augmentation script instead simulates effects that are common in Raman spectroscopy and planetary analog measurements:

- relative intensity variation caused by crystal orientation, grain size, focus, optical coupling, excitation wavelength, and detector response;
- weak residual fluorescence or background curvature after baseline correction;
- read noise and shot-noise-like perturbations at conservative normalized levels;
- mild symmetric broadening or smoothing that preserves band centers;
- partial attenuation of weak bands to mimic low signal-to-noise or mixed-pixel effects.

## What Is Not Augmented

The following transforms are intentionally avoided in the default workflow:

- large random band shifts;
- class-changing band creation or deletion;
- arbitrary warping of the wavenumber axis;
- synthetic spectra whose parent spectrum cannot be identified.

This design reduces the risk that overly aggressive augmentation creates non-physical spectra or teaches the model artifacts rather than mineralogical variability.

## Reproducible Augmentation

Run:

```bash
python src/augment_raman_dataset.py --target-per-class 200 --seed 2024
```

The script writes:

```text
data/augmented_spectra/augmented_lineage.csv
data/augmented_spectra/augmentation_summary.json
data/augmented_spectra/spectra/*.csv
```

Every generated spectrum records:

- `augmented_id`
- `parent_spectrum_id`
- `parent_group`
- `model_label`
- `source_type`
- `split_main`
- `random_seed`
- `augmentation_parameters_json`

## Legacy Augmentation Limitation

The file `data/metadata/metadata_augmented_lineage_current_legacy.csv` documents an older augmented manifest. Most legacy augmented filenames encode only mineral species, not exact parent-spectrum identifiers. These legacy records are retained for transparency but should not be treated as fully parent-traceable. For final claims, regenerate augmented spectra with `src/augment_raman_dataset.py`.
