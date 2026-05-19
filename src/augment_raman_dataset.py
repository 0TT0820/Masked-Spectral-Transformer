from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "metadata" / "metadata_parent_945.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "augmented_spectra"

AUGMENTATION_PROTOCOL = {
    "scope": "training split only; validation and test spectra are never augmented",
    "physical_principle": "Raman band centers are not translated because band positions are diagnostic for a mineral phase.",
    "band_detection": {
        "signal": "baseline-corrected, max-normalized intensity within the valid spectral range",
        "minimum_height": 0.05,
        "minimum_prominence": 0.03,
        "minimum_distance_cm-1": 8.0,
        "maximum_bands_per_spectrum": 12,
        "fallback": "if scipy.signal.find_peaks is unavailable, local maxima satisfying the same height threshold are used",
    },
    "transforms": {
        "gamma_intensity_response": {"probability": 0.70, "gamma_range": [0.75, 1.35]},
        "band_envelope_intensity_perturbation": {
            "probability": 0.20,
            "amplitude_range": [-0.08, 0.08],
            "sigma_cm-1_range": [4.0, 10.0],
            "center_shift_cm-1": 0.0,
        },
        "residual_baseline": {
            "probability": 0.50,
            "polynomial_order": 2,
            "coefficient_std": [0.015, 0.020, 0.015],
        },
        "gaussian_read_noise": {"probability": 0.80, "sigma_range_after_normalization": [0.005, 0.025]},
        "symmetric_broadening": {
            "probability": 0.35,
            "kernel": [0.08, 0.18, 0.48, 0.18, 0.08],
            "mixing_alpha_range": [0.25, 0.65],
        },
        "weak_band_attenuation": {
            "probability": 0.25,
            "windows_per_spectrum": [1, 3],
            "half_width_points_range": [8, 35],
            "attenuation_factor_range": [0.75, 0.95],
        },
    },
    "constraints": [
        "no wavenumber-axis translation or interpolation jitter is applied during augmentation",
        "negative intensities are clipped to zero after perturbation",
        "each augmented spectrum is renormalized to unit maximum within the valid range",
    ],
}


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    shift = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    intensity = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(shift) & np.isfinite(intensity)
    shift = shift[valid]
    intensity = intensity[valid]
    order = np.argsort(shift)
    return shift[order], intensity[order]


def preprocess_parent_intensity(shift: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    y = intensity.astype(np.float64).copy()
    y = y - lower_envelope_baseline(shift, y)
    y = np.maximum(y, 0.0)
    max_y = float(np.max(y)) if np.max(y) > 0 else 1.0
    return y / max_y


def write_two_column_spectrum(path: Path, shift: np.ndarray, intensity: np.ndarray) -> None:
    pd.DataFrame({"wavenumber_cm-1": shift, "intensity": intensity}).to_csv(path, index=False)


def resolve_spectrum_path(file_path: str, spectrum_id: str) -> Path:
    """Resolve metadata paths for both raw and materialized repository filenames."""
    path = Path(str(file_path))
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path

    candidates = [path.with_name(f"{spectrum_id}__{path.name}")]
    if path.name.endswith(".rruff.csv"):
        normalized_name = path.name.replace(".rruff.csv", ".csv")
        candidates.append(path.with_name(normalized_name))
        candidates.append(path.with_name(f"{spectrum_id}__{normalized_name}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    prefixed_matches = sorted(path.parent.glob(f"{spectrum_id}__*.csv"))
    if len(prefixed_matches) == 1:
        return prefixed_matches[0]
    searched = ", ".join(str(candidate) for candidate in [path, *candidates])
    raise FileNotFoundError(f"Could not find spectrum file for {spectrum_id}. Searched: {searched}")


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def lower_envelope_baseline(shift: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    if len(shift) < 8:
        return np.zeros_like(intensity)
    cutoff = np.percentile(intensity, 35)
    low = intensity <= cutoff
    if np.sum(low) < 4:
        low = np.ones_like(intensity, dtype=bool)
    coef = np.polyfit(shift[low], intensity[low], deg=2)
    return np.polyval(coef, shift)


def detect_raman_bands(shift: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Detect local Raman band centers used as fixed anchors for intensity perturbation."""
    protocol = AUGMENTATION_PROTOCOL["band_detection"]
    if len(shift) < 3 or np.max(intensity) <= 0:
        return np.array([], dtype=np.int64)

    step = float(np.median(np.diff(shift))) if len(shift) > 1 else 1.0
    min_distance = max(1, int(round(protocol["minimum_distance_cm-1"] / max(step, 1e-6))))

    if find_peaks is not None:
        peaks, props = find_peaks(
            intensity,
            height=protocol["minimum_height"],
            prominence=protocol["minimum_prominence"],
            distance=min_distance,
        )
        prominences = props.get("prominences", np.ones(len(peaks)))
    else:
        local = np.flatnonzero(
            (intensity[1:-1] > intensity[:-2])
            & (intensity[1:-1] >= intensity[2:])
            & (intensity[1:-1] >= protocol["minimum_height"])
        ) + 1
        peaks = local
        prominences = intensity[peaks] if len(peaks) else np.array([], dtype=np.float64)

    if len(peaks) > protocol["maximum_bands_per_spectrum"]:
        order = np.argsort(prominences)[::-1][: protocol["maximum_bands_per_spectrum"]]
        peaks = peaks[order]
    return np.asarray(np.sort(peaks), dtype=np.int64)


def augment_without_band_shift(shift: np.ndarray, intensity: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply Raman-aware augmentation while preserving band positions."""
    transforms = AUGMENTATION_PROTOCOL["transforms"]
    y = intensity.astype(np.float64).copy()
    y = y - lower_envelope_baseline(shift, y)
    y = np.maximum(y, 0.0)
    max_y = float(np.max(y)) if np.max(y) > 0 else 1.0
    y = y / max_y

    band_indices = detect_raman_bands(shift, y)
    params: dict[str, object] = {
        "band_position_shift_cm-1": 0.0,
        "detected_band_count": int(len(band_indices)),
        "detected_band_positions_cm-1": [float(shift[idx]) for idx in band_indices],
    }

    gamma_cfg = transforms["gamma_intensity_response"]
    if rng.random() < gamma_cfg["probability"]:
        gamma = float(rng.uniform(*gamma_cfg["gamma_range"]))
        y = np.power(np.clip(y, 0.0, 1.0), gamma)
        params["intensity_gamma"] = gamma
    else:
        params["intensity_gamma"] = 1.0

    band_cfg = transforms["band_envelope_intensity_perturbation"]
    band_envelopes = []
    if len(band_indices) and rng.random() < band_cfg["probability"]:
        for idx in band_indices:
            amplitude = float(rng.uniform(*band_cfg["amplitude_range"]))
            sigma = float(rng.uniform(*band_cfg["sigma_cm-1_range"]))
            envelope = np.exp(-0.5 * ((shift - shift[idx]) / sigma) ** 2)
            y = y + amplitude * envelope
            band_envelopes.append({"center_cm-1": float(shift[idx]), "amplitude": amplitude, "sigma_cm-1": sigma})
    params["band_envelope_intensity_perturbations"] = band_envelopes

    baseline_cfg = transforms["residual_baseline"]
    if rng.random() < baseline_cfg["probability"]:
        xv = np.linspace(-1.0, 1.0, len(y))
        baseline_coefficients = rng.normal(0.0, baseline_cfg["coefficient_std"])
        y = y + baseline_coefficients[0] + baseline_coefficients[1] * xv + baseline_coefficients[2] * xv * xv
        params["baseline_polynomial_coefficients"] = [float(v) for v in baseline_coefficients]
    else:
        params["baseline_polynomial_coefficients"] = [0.0, 0.0, 0.0]

    noise_cfg = transforms["gaussian_read_noise"]
    if rng.random() < noise_cfg["probability"]:
        noise_sigma = float(rng.uniform(*noise_cfg["sigma_range_after_normalization"]))
        y = y + rng.normal(0.0, noise_sigma, len(y))
        params["gaussian_noise_sigma_normalized"] = noise_sigma
    else:
        params["gaussian_noise_sigma_normalized"] = 0.0

    broadening_cfg = transforms["symmetric_broadening"]
    if rng.random() < broadening_cfg["probability"] and len(y) > 7:
        kernel = np.array(broadening_cfg["kernel"], dtype=np.float64)
        smoothed = np.convolve(y, kernel, mode="same")
        alpha = float(rng.uniform(*broadening_cfg["mixing_alpha_range"]))
        y = (1.0 - alpha) * y + alpha * smoothed
        params["symmetric_broadening_alpha"] = alpha
    else:
        params["symmetric_broadening_alpha"] = 0.0

    attenuation_cfg = transforms["weak_band_attenuation"]
    attenuation_windows = []
    if rng.random() < attenuation_cfg["probability"] and len(y) > 30:
        min_windows, max_windows = attenuation_cfg["windows_per_spectrum"]
        for _ in range(int(rng.integers(min_windows, max_windows + 1))):
            center = int(rng.integers(0, len(y)))
            min_half_width, max_half_width = attenuation_cfg["half_width_points_range"]
            half_width = int(rng.integers(min_half_width, max_half_width + 1))
            factor = float(rng.uniform(*attenuation_cfg["attenuation_factor_range"]))
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
    parser.add_argument("--split", default="train", help="Split to augment; default is train.")
    parser.add_argument("--combined-metadata-name", default="metadata_augmented_training.csv")
    parser.add_argument("--include-qc-required", dest="include_review_required", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    spectra_dir = out_dir / "spectra"
    original_dir = out_dir / "original_spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    metadata_all = pd.read_csv(metadata_path)
    metadata_all = metadata_all[metadata_all["split_main"].isin(["train", "val", "test"])].copy()
    metadata_all = metadata_all[~metadata_all["major_category"].eq("Halides")].copy()
    metadata_all["model_label"] = metadata_all["major_category"].replace(
        {"Clay": "Phyllosilicates", "Mica": "Phyllosilicates", "Serpentine": "Phyllosilicates"}
    )
    if not args.include_review_required:
        keep = ~metadata_all["qc_status"].eq("review_required") | metadata_all["model_label"].eq("Phyllosilicates")
        metadata_all = metadata_all[keep].copy()

    original_records = []
    missing_originals = []
    for _, row in metadata_all.iterrows():
        out_row = row.to_dict()
        try:
            resolved = resolve_spectrum_path(str(row["file_path"]), str(row["spectrum_id"]))
            shift, intensity = read_spectrum(resolved)
            normalized = preprocess_parent_intensity(shift, intensity)
            original_file = original_dir / f"{row['spectrum_id']}__original.csv"
            if not args.dry_run:
                write_two_column_spectrum(original_file, shift, normalized)
            out_row["raw_file_path"] = repository_relative(resolved)
            out_row["file_path"] = repository_relative(original_file)
            out_row["file_exists"] = True
        except FileNotFoundError as exc:
            missing_originals.append({"spectrum_id": row["spectrum_id"], "error": str(exc)})
            out_row["file_exists"] = False
        out_row.update(
            {
                "is_augmented": "no",
                "parent_spectrum_id": row["spectrum_id"],
                "augmentation_seed": "",
                "augmentation_parameters_json": "",
                "repository_file": out_row["file_path"],
            }
        )
        original_records.append(out_row)
    if missing_originals:
        pd.DataFrame(missing_originals).to_csv(out_dir / "missing_original_spectra.csv", index=False, encoding="utf-8-sig")

    metadata = metadata_all[metadata_all["split_main"].eq(args.split)].copy()

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
            parent_path = resolve_spectrum_path(str(row["file_path"]), str(row["spectrum_id"]))
            shift, intensity = read_spectrum(parent_path)
            augmented, params = augment_without_band_shift(shift, intensity, rng)
            aug_id = f"{row['spectrum_id']}__AUG_{aug_index:04d}"
            out_file = spectra_dir / f"{aug_id}.csv"
            if not args.dry_run:
                write_two_column_spectrum(out_file, shift, augmented)
            record = row.to_dict()
            record.update(
                {
                    "augmented_id": aug_id,
                    "spectrum_id": aug_id,
                    "file_name_clean": aug_id,
                    "parent_spectrum_id": row["spectrum_id"],
                    "parent_group": row["parent_group"],
                    "model_label": label,
                    "major_category": row["major_category"],
                    "subtype_label": row["subtype_label"],
                    "source_type": f"Raman-aware augmentation of {row['source_type']}",
                    "source_note": f"Augmented from parent spectrum {row['spectrum_id']}; parameters logged in augmentation_parameters_json.",
                    "split_main": args.split,
                    "file_path": repository_relative(out_file),
                    "repository_file": repository_relative(out_file),
                    "file_exists": True,
                    "is_augmented": "yes",
                    "augmentation_seed": args.seed,
                    "random_seed": args.seed,
                    "augmentation_parameters_json": json.dumps(params, sort_keys=True),
                }
            )
            records.append(record)

    lineage = pd.DataFrame(records)
    lineage.to_csv(out_dir / "augmented_lineage.csv", index=False, encoding="utf-8-sig")
    combined = pd.DataFrame(original_records + records)
    combined.to_csv(out_dir / args.combined_metadata_name, index=False, encoding="utf-8-sig")
    combined.groupby(["split_main", "model_label", "is_augmented"]).size().reset_index(name="n_spectra").to_csv(
        out_dir / "split_by_class_and_augmentation.csv", index=False, encoding="utf-8-sig"
    )
    combined.groupby(["source_type", "model_label", "is_augmented"]).size().reset_index(name="n_spectra").to_csv(
        out_dir / "source_by_class_and_augmentation.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "target_per_class": args.target_per_class,
        "seed": args.seed,
        "split_augmented": args.split,
        "original_spectra": int(len(original_records)),
        "generated_spectra": int(len(lineage)),
        "combined_metadata_rows": int(len(combined)),
        "augmentation_policy": "Band positions are not shifted. Augmentation only modifies intensity response, band-envelope intensity, residual baseline, Gaussian read-noise-like perturbations, symmetric broadening, and weak-band attenuation.",
        "augmentation_protocol": AUGMENTATION_PROTOCOL,
    }
    (out_dir / "augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
