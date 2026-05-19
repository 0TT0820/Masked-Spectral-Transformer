"""Export reviewer-ready preprocessing and augmentation protocol tables.

The training code already uses conservative Raman-aware augmentation that does
not shift Raman band centers. This utility materializes the exact parameter
ranges, trigger probabilities, band-selection rules, and physical constraints
used by the code into machine-readable tables for the repository and
Supporting Information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_model_comparison import AUGMENTATION_PROTOCOL, GRID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "augmentation_reproducibility"


def operation_rows() -> list[dict]:
    transforms = AUGMENTATION_PROTOCOL["transforms"]
    rows: list[dict] = []
    rows.append(
        {
            "operation": "gamma_intensity_response",
            "trigger_probability": transforms["gamma_intensity_response"]["probability"],
            "parameter_range": "gamma uniformly sampled from 0.75-1.35",
            "physical_purpose": "Simulates relative intensity-response changes from laser power, focusing, grain orientation, and instrument response.",
            "physical_constraint": "Only normalized intensity is changed; Raman-shift axis and valid spectral mask are unchanged.",
        }
    )
    rows.append(
        {
            "operation": "band_envelope_intensity_perturbation",
            "trigger_probability": transforms["band_envelope_intensity_perturbation"]["probability"],
            "parameter_range": "Gaussian envelope amplitude uniformly sampled from -0.08 to 0.08; sigma 4-10 cm-1; center shift fixed at 0 cm-1",
            "physical_purpose": "Perturbs relative band strength while preserving diagnostic band centers.",
            "physical_constraint": "Band centers are detected from the parent spectrum and are not translated.",
        }
    )
    rows.append(
        {
            "operation": "residual_baseline",
            "trigger_probability": transforms["residual_baseline"]["probability"],
            "parameter_range": "Second-order polynomial over normalized coordinate [-1,1]; coefficients drawn from N(0, [0.015, 0.020, 0.015])",
            "physical_purpose": "Simulates residual continuum/background after baseline correction.",
            "physical_constraint": "Added baseline is small relative to unit-normalized intensity and followed by non-negative clipping and renormalization.",
        }
    )
    rows.append(
        {
            "operation": "gaussian_read_noise",
            "trigger_probability": transforms["gaussian_read_noise"]["probability"],
            "parameter_range": "Gaussian noise sigma uniformly sampled from 0.005-0.025 after normalization",
            "physical_purpose": "Simulates read-noise-like intensity perturbations at a conservative normalized scale.",
            "physical_constraint": "Noise is added only within valid spectral coverage; invalid/padded regions remain masked.",
        }
    )
    rows.append(
        {
            "operation": "symmetric_broadening",
            "trigger_probability": transforms["symmetric_broadening"]["probability"],
            "parameter_range": "Symmetric kernel [0.08, 0.18, 0.48, 0.18, 0.08]; mixing alpha uniformly sampled from 0.25-0.65",
            "physical_purpose": "Simulates mild resolution-related broadening/denoising without asymmetric band displacement.",
            "physical_constraint": "Symmetric kernel preserves band center positions; no unsharp-mask sharpening is applied.",
        }
    )
    rows.append(
        {
            "operation": "weak_band_attenuation",
            "trigger_probability": transforms["weak_band_attenuation"]["probability"],
            "parameter_range": "1-3 windows; half-width 8-35 grid points; attenuation factor 0.75-0.95",
            "physical_purpose": "Simulates weak or partially obscured bands caused by low signal-to-noise, grain heterogeneity, or mixing.",
            "physical_constraint": "Only local intensity is attenuated; no artificial new bands or band shifts are introduced.",
        }
    )
    return rows


def band_detection_rows() -> list[dict]:
    bd = AUGMENTATION_PROTOCOL["band_detection"]
    grid_step = float(GRID[1] - GRID[0]) if len(GRID) > 1 else 1.0
    return [
        {
            "item": "input_signal",
            "value": bd["signal"],
            "note": "The same preprocessed normalized intensity channel used by model training is used for band detection.",
        },
        {
            "item": "minimum_height",
            "value": bd["minimum_height"],
            "note": "Height threshold after max-normalization.",
        },
        {
            "item": "minimum_prominence",
            "value": bd["minimum_prominence"],
            "note": "Prominence threshold after max-normalization.",
        },
        {
            "item": "minimum_distance_cm-1",
            "value": bd["minimum_distance_cm-1"],
            "note": f"Equivalent to approximately {round(bd['minimum_distance_cm-1'] / grid_step)} grid points on the {grid_step:.3f} cm-1 grid.",
        },
        {
            "item": "maximum_bands_per_spectrum",
            "value": bd["maximum_bands_per_spectrum"],
            "note": "If more bands are detected, the most prominent bands are retained.",
        },
        {
            "item": "fallback",
            "value": bd["fallback"],
            "note": "Used only if scipy.signal.find_peaks is unavailable.",
        },
    ]


def preprocessing_rows() -> list[dict]:
    return [
        {
            "step": "common_grid",
            "parameter": f"0-4000 cm-1 grid with {len(GRID)} points",
            "purpose": "Allows common tensor input while preserving a valid-mask channel for variable spectral coverage.",
        },
        {
            "step": "baseline_correction",
            "parameter": "Second-order lower-envelope polynomial by default; ASLS option for SHERLOC preprocessing trials",
            "purpose": "Reduces smooth continuum background without treating Raman bands as baseline.",
        },
        {
            "step": "normalization",
            "parameter": "Non-negative clipping followed by unit-maximum normalization within valid spectral coverage",
            "purpose": "Reduces absolute intensity differences across instruments and acquisition settings.",
        },
        {
            "step": "derivative_channel",
            "parameter": "First derivative recomputed after preprocessing and after augmentation",
            "purpose": "Provides local band-shape information while retaining intensity channel.",
        },
        {
            "step": "valid_mask",
            "parameter": "Invalid or padded regions remain zeroed and masked",
            "purpose": "Prevents padding or outside-coverage regions from influencing the model.",
        },
    ]


def markdown_text() -> str:
    protocol = json.dumps(AUGMENTATION_PROTOCOL, indent=2, ensure_ascii=False)
    return f"""# Raman preprocessing and augmentation protocol

This directory documents the exact preprocessing and augmentation protocol used
for the reference-domain training experiments. The augmentation is applied only
to the reference-domain training split after train/validation/test separation.
Validation, test, SHERLOC in-situ, and calibration-target spectra are not
augmented.

Most importantly, Raman band centers are not shifted. The earlier qualitative
description using a ±10 cm-1 example has been removed from the manuscript
because band positions are diagnostic for mineral phase identification.

## Machine-readable protocol

```json
{protocol}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export augmentation protocol documentation tables.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "augmentation_protocol.json").write_text(
        json.dumps(AUGMENTATION_PROTOCOL, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(preprocessing_rows()).to_csv(out_dir / "preprocessing_protocol.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(band_detection_rows()).to_csv(out_dir / "band_detection_rules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(operation_rows()).to_csv(out_dir / "augmentation_operations.csv", index=False, encoding="utf-8-sig")
    (out_dir / "augmentation_protocol_summary.md").write_text(markdown_text(), encoding="utf-8")
    print(f"Saved augmentation protocol documentation to: {out_dir}")


if __name__ == "__main__":
    main()
