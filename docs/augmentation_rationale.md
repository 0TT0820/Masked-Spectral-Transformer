# Raman-Aware Data Augmentation Rationale

## Physical Basis

For a given crystalline mineral phase, Raman band positions are controlled primarily by vibrational modes of the crystal structure. Under comparable measurement conditions, diagnostic band positions should remain relatively stable. Therefore, the augmentation policy in this repository does **not** randomly shift band positions.

The augmentation script instead simulates effects that are common in Raman spectroscopy and planetary analog measurements:

- relative intensity variation caused by crystal orientation, grain size, focus, optical coupling, excitation wavelength, and detector response;
- weak residual fluorescence or background curvature after baseline correction;
- Gaussian read-noise-like perturbations at conservative normalized levels;
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

The exact operation probabilities and parameter ranges are:

| Operation | Probability | Parameter range | Constraint |
| --- | ---: | --- | --- |
| Gamma intensity response | 0.70 | gamma = 0.75-1.35 | Changes normalized intensity only |
| Band-envelope intensity perturbation | 0.20 | amplitude = -0.08 to 0.08; sigma = 4-10 cm^-1; center shift = 0 cm^-1 | Perturbs relative band strength without moving band centers |
| Residual baseline | 0.50 | second-order polynomial over [-1, 1]; coefficient std = [0.015, 0.020, 0.015] | Simulates small residual continuum after baseline correction |
| Gaussian read noise | 0.80 | sigma = 0.005-0.025 after normalization | Valid spectral range only |
| Symmetric broadening | 0.35 | kernel = [0.08, 0.18, 0.48, 0.18, 0.08]; alpha = 0.25-0.65 | Symmetric kernel preserves band centers |
| Weak-band attenuation | 0.25 | 1-3 windows; half-width = 8-35 grid points; attenuation = 0.75-0.95 | No new bands and no band shifts |

Candidate Raman bands are detected on the baseline-corrected, max-normalized
intensity using minimum height 0.05, minimum prominence 0.03, minimum separation
8 cm^-1, and a maximum of 12 bands per spectrum. The detected bands are used
only as fixed anchors for local intensity perturbation.

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
