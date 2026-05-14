from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "metadata" / "metadata_parent_945.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "augmented_spectra"


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    shift = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    intensity = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(shift) & np.isfinite(intensity)
    shift = shift[valid]
    intensity = intensity[valid]
    order = np.argsort(shift)
    return shift[order], intensity[order]


def lower_envelope_baseline(shift: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    if len(shift) < 8:
        return np.zeros_like(intensity)
    cutoff = np.percentile(intensity, 35)
    low = intensity <= cutoff
    if np.sum(low) < 4:
        low = np.ones_like(intensity, dtype=bool)
    coef = np.polyfit(shift[low], intensity[low], deg=2)
    return np.polyval(coef, shift)


def augment_without_peak_shift(shift: np.ndarray, intensity: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply Raman-aware augmentation while preserving band positions."""
    y = intensity.astype(np.float64).copy()
    y = y - lower_envelope_baseline(shift, y)
    y = np.maximum(y, 0.0)
    max_y = float(np.max(y)) if np.max(y) > 0 else 1.0
    y = y / max_y

    params: dict[str, object] = {"band_position_shift_cm-1": 0.0}

    gamma = float(rng.uniform(0.75, 1.35))
    y = np.power(np.clip(y, 0.0, 1.0), gamma)
    params["intensity_gamma"] = gamma

    baseline_scale = float(rng.uniform(0.0, 0.04))
    xv = np.linspace(-1.0, 1.0, len(y))
    baseline_coefficients = rng.normal(0.0, [baseline_scale, baseline_scale, baseline_scale / 2.0])
    y = y + baseline_coefficients[0] + baseline_coefficients[1] * xv + baseline_coefficients[2] * xv * xv
    params["baseline_polynomial_coefficients"] = [float(v) for v in baseline_coefficients]

    noise_sigma = float(rng.uniform(0.003, 0.025))
    y = y + rng.normal(0.0, noise_sigma, len(y))
    params["gaussian_noise_sigma_normalized"] = noise_sigma

    if rng.random() < 0.35 and len(y) > 7:
        kernel = np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float64)
        smoothed = np.convolve(y, kernel, mode="same")
        alpha = float(rng.uniform(0.2, 0.65))
        y = (1.0 - alpha) * y + alpha * smoothed
        params["symmetric_broadening_alpha"] = alpha
    else:
        params["symmetric_broadening_alpha"] = 0.0

    attenuation_windows = []
    if rng.random() < 0.25 and len(y) > 30:
        for _ in range(int(rng.integers(1, 4))):
            center = int(rng.integers(0, len(y)))
            half_width = int(rng.integers(8, 35))
            factor = float(rng.uniform(0.75, 0.95))
            lo = max(0, center - half_width)
            hi = min(len(y), center + half_width)
            y[lo:hi] *= factor
            attenuation_windows.append({"index_start": lo, "index_stop": hi, "factor": factor})
    params["weak_band_attenuation_windows"] = attenuation_windows

    y = np.maximum(y, 0.0)
    max_y = float(np.max(y)) if np.max(y) > 0 else 1.0
    y = y / max_y
    return y, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Raman-aware augmented spectra with parent-level lineage.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--include-qc-required", dest="include_review_required", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    spectra_dir = out_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["split_main"].isin(["train", "val", "test"])].copy()
    metadata = metadata[~metadata["major_category"].eq("Halides")].copy()
    metadata["model_label"] = metadata["major_category"].replace(
        {"Clay": "Phyllosilicates", "Mica": "Phyllosilicates", "Serpentine": "Phyllosilicates"}
    )
    if not args.include_review_required:
        keep = ~metadata["qc_status"].eq("review_required") | metadata["model_label"].eq("Phyllosilicates")
        metadata = metadata[keep].copy()

    rng = np.random.default_rng(args.seed)
    records = []
    class_counts = metadata["model_label"].value_counts().to_dict()
    for label, count in sorted(class_counts.items()):
        needed = max(0, args.target_per_class - int(count))
        pool = metadata[metadata["model_label"].eq(label)].copy()
        if pool.empty:
            continue
        for aug_index in range(needed):
            row = pool.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            parent_path = Path(str(row["file_path"]))
            if not parent_path.is_absolute():
                parent_path = ROOT / parent_path
            shift, intensity = read_spectrum(parent_path)
            augmented, params = augment_without_peak_shift(shift, intensity, rng)
            aug_id = f"{row['spectrum_id']}__AUG_{aug_index:04d}"
            out_file = spectra_dir / f"{aug_id}.csv"
            if not args.dry_run:
                pd.DataFrame({"wavenumber_cm-1": shift, "intensity": augmented}).to_csv(out_file, index=False)
            records.append(
                {
                    "augmented_id": aug_id,
                    "parent_spectrum_id": row["spectrum_id"],
                    "parent_group": row["parent_group"],
                    "model_label": label,
                    "major_category": row["major_category"],
                    "subtype_label": row["subtype_label"],
                    "source_type": row["source_type"],
                    "split_main": row["split_main"],
                    "repository_file": str(out_file.relative_to(ROOT)),
                    "random_seed": args.seed,
                    "augmentation_parameters_json": json.dumps(params, sort_keys=True),
                }
            )

    lineage = pd.DataFrame(records)
    lineage.to_csv(out_dir / "augmented_lineage.csv", index=False, encoding="utf-8-sig")
    summary = {
        "target_per_class": args.target_per_class,
        "seed": args.seed,
        "generated_spectra": int(len(lineage)),
        "augmentation_policy": "Band positions are not shifted. Augmentation only modifies intensity response, residual baseline, noise, symmetric broadening, and weak-band attenuation.",
    }
    (out_dir / "augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
