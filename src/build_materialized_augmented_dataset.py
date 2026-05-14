from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from train_review_comparison import (
    AUGMENTATION_PROTOCOL,
    GRID,
    PHYLL_LABELS,
    find_peaks,
    load_metadata,
    preprocess_spectrum,
)


ROOT = Path(r"d:/dyt/raman/pigeonite")
DEFAULT_METADATA = ROOT / "data" / "metadata_outputs" / "metadata_parent_945.csv"
DEFAULT_OUT = ROOT / "data" / "materialized_augmented_review_ready"
SEED = 2024


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def stable_seed(parent_id: str, replicate: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{parent_id}|{replicate}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_label(row: pd.Series) -> str:
    label = str(row["major_category"])
    if label in PHYLL_LABELS:
        return "Phyllosilicates"
    return label


def detect_bands(intensity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 3:
        return np.array([], dtype=int)
    y = intensity[valid_idx]
    grid_step = float(np.median(np.diff(GRID))) if len(GRID) > 1 else 1.0
    min_distance = max(1, int(round(8.0 / grid_step)))
    if find_peaks is not None:
        peaks_local, props = find_peaks(y, height=0.05, prominence=0.03, distance=min_distance)
        if len(peaks_local) > 12 and "prominences" in props:
            keep = np.argsort(props["prominences"])[-12:]
            peaks_local = peaks_local[keep]
    else:
        candidate = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] >= 0.05))[0] + 1
        peaks_local = candidate[::min_distance][:12]
    return valid_idx[np.asarray(peaks_local, dtype=int)]


def materialized_augment(intensity: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    y = intensity.copy()
    params: dict[str, object] = {"applied": [], "detected_band_centers_cm-1": []}

    band_idx = detect_bands(y, valid)
    params["detected_band_centers_cm-1"] = [round(float(GRID[i]), 3) for i in band_idx]

    if rng.random() < 0.70:
        gamma = float(rng.uniform(0.75, 1.35))
        y[valid] = np.power(np.clip(y[valid], 0.0, 1.0), gamma)
        params["applied"].append("gamma_intensity_response")
        params["gamma"] = gamma

    if len(band_idx) and rng.random() < 0.20:
        events = []
        for center_idx in band_idx:
            sigma_cm = float(rng.uniform(4.0, 10.0))
            amp = float(rng.uniform(-0.08, 0.08))
            envelope = np.exp(-0.5 * ((GRID - GRID[center_idx]) / sigma_cm) ** 2).astype(np.float32)
            y[valid] = y[valid] * (1.0 + amp * envelope[valid])
            events.append(
                {
                    "center_cm-1": round(float(GRID[center_idx]), 3),
                    "amplitude": amp,
                    "sigma_cm-1": sigma_cm,
                    "center_shift_cm-1": 0.0,
                }
            )
        params["applied"].append("band_envelope_intensity_perturbation")
        params["band_envelope_events"] = events

    if rng.random() < 0.50:
        xv = np.linspace(-1.0, 1.0, int(np.sum(valid)), dtype=np.float32)
        coef = rng.normal(0.0, [0.015, 0.020, 0.015]).astype(np.float32)
        baseline = coef[0] + coef[1] * xv + coef[2] * xv * xv
        y[valid] = y[valid] + baseline
        params["applied"].append("residual_baseline")
        params["baseline_coefficients"] = [float(v) for v in coef]

    if rng.random() < 0.80:
        sigma = float(rng.uniform(0.005, 0.025))
        y[valid] = y[valid] + rng.normal(0.0, sigma, int(np.sum(valid))).astype(np.float32)
        params["applied"].append("gaussian_read_noise")
        params["noise_sigma"] = sigma

    if rng.random() < 0.35:
        kernel = np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float32)
        smoothed = np.convolve(y, kernel, mode="same")
        alpha = float(rng.uniform(0.25, 0.65))
        y[valid] = (1.0 - alpha) * y[valid] + alpha * smoothed[valid]
        params["applied"].append("symmetric_broadening")
        params["broadening_alpha"] = alpha
        params["broadening_kernel"] = [float(v) for v in kernel]

    if rng.random() < 0.25:
        valid_idx = np.where(valid)[0]
        n_windows = int(rng.integers(1, 4))
        windows = []
        for _ in range(n_windows):
            center = int(rng.choice(valid_idx))
            width = int(rng.integers(8, 36))
            factor = float(rng.uniform(0.75, 0.95))
            lo = max(int(valid_idx[0]), center - width)
            hi = min(int(valid_idx[-1]) + 1, center + width)
            y[lo:hi] *= factor
            windows.append(
                {
                    "center_cm-1": round(float(GRID[center]), 3),
                    "half_width_points": width,
                    "attenuation_factor": factor,
                }
            )
        params["applied"].append("weak_band_attenuation")
        params["attenuation_windows"] = windows

    y[~valid] = 0.0
    y = np.maximum(y, 0.0)
    maxv = float(np.max(y[valid])) if np.any(valid) else 0.0
    if maxv > 0:
        y = y / (maxv + 1e-12)
    return y.astype(np.float32), params


def write_spectrum(path: Path, intensity: np.ndarray, valid: np.ndarray) -> None:
    derivative = np.gradient(intensity, GRID).astype(np.float32)
    if np.any(valid):
        max_abs = float(np.max(np.abs(derivative[valid])))
        if max_abs > 1e-9:
            derivative = derivative / max_abs
    derivative[~valid] = 0.0
    out = pd.DataFrame(
        {
            "raman_shift_cm-1": GRID.astype(np.float32),
            "intensity_normalized": intensity.astype(np.float32),
            "first_derivative_normalized": derivative.astype(np.float32),
            "valid_mask": valid.astype(np.int8),
        }
    )
    out.to_csv(path, index=False, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    spectra_dir = out_dir / "spectra"
    tables_dir = out_dir / "overview_tables"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata("review_ready", False, args.metadata_file)
    df["model_label"] = df.apply(model_label, axis=1)
    df = df[df["split_main"].isin(["train", "val", "test"])].copy()

    rows = []
    log(f"Writing canonical original spectra: {len(df)}")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == len(df):
            log(f"Original spectrum {i}/{len(df)}")
        features, _, key_padding = preprocess_spectrum(str(row["file_path"]), baseline=args.baseline, smooth=False)
        valid = ~key_padding
        intensity = features[:, 0]
        spectrum_id = str(row["spectrum_id"])
        fname = f"{spectrum_id}__original.csv"
        out_path = spectra_dir / fname
        write_spectrum(out_path, intensity, valid)
        out_row = row.to_dict()
        out_row.update(
            {
                "materialized_spectrum_id": spectrum_id,
                "parent_spectrum_id": spectrum_id,
                "is_augmented": "no",
                "augmentation_replicate": 0,
                "augmentation_seed": "",
                "augmentation_parameters_json": "",
                "materialized_file_path": str(out_path),
                "file_path": str(out_path),
                "materialized_file_sha256": sha256_file(out_path),
                "model_label": row["model_label"],
                "preprocessing_planned": "Already materialized on common 0-4000 cm-1 grid; use baseline none for model comparison",
            }
        )
        rows.append(out_row)

    train = df[df["split_main"].eq("train")].copy()
    class_counts = train["model_label"].value_counts().to_dict()
    aug_count = 0
    log("Writing deterministic augmented training spectra")
    for label, count in sorted(class_counts.items()):
        needed = max(0, args.min_train_per_class - int(count))
        if needed == 0:
            continue
        parents = train[train["model_label"].eq(label)].reset_index(drop=True)
        for rep in range(needed):
            parent = parents.iloc[rep % len(parents)]
            parent_id = str(parent["spectrum_id"])
            seed = stable_seed(parent_id, rep + 1, args.seed)
            rng = np.random.default_rng(seed)
            features, _, key_padding = preprocess_spectrum(str(parent["file_path"]), baseline=args.baseline, smooth=False)
            valid = ~key_padding
            intensity, params = materialized_augment(features[:, 0], valid, rng)
            mat_id = f"AUG_{label.replace('/', '_').replace(' ', '_')}_{rep + 1:04d}_from_{parent_id}"
            fname = f"{mat_id}.csv"
            out_path = spectra_dir / fname
            write_spectrum(out_path, intensity, valid)
            out_row = parent.to_dict()
            out_row.update(
                {
                    "materialized_spectrum_id": mat_id,
                    "parent_spectrum_id": parent_id,
                    "is_augmented": "yes",
                    "augmentation_replicate": rep + 1,
                    "augmentation_seed": seed,
                    "augmentation_parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "materialized_file_path": str(out_path),
                    "file_path": str(out_path),
                    "file_name_clean": mat_id,
                    "spectrum_id": mat_id,
                    "materialized_file_sha256": sha256_file(out_path),
                    "augmentation_used": "yes",
                    "split_main": "train",
                    "model_label": label,
                    "source_type": f"Deterministic augmentation of {parent.get('source_type', '')}",
                    "source_note": f"Augmented from parent spectrum {parent_id}; parameters logged in augmentation_parameters_json",
                    "preprocessing_planned": "Already materialized on common 0-4000 cm-1 grid; use baseline none for model comparison",
                }
            )
            rows.append(out_row)
            aug_count += 1
    materialized = pd.DataFrame(rows)

    metadata_out = out_dir / "metadata_review_ready_materialized_augmented.csv"
    materialized.to_csv(metadata_out, index=False, encoding="utf-8-sig")
    materialized.to_csv(ROOT / "data" / "metadata_outputs" / "metadata_review_ready_materialized_augmented.csv", index=False, encoding="utf-8-sig")

    protocol = dict(AUGMENTATION_PROTOCOL)
    protocol["materialized_dataset"] = {
        "metadata_file": str(metadata_out),
        "spectra_directory": str(spectra_dir),
        "min_train_per_class": args.min_train_per_class,
        "baseline": args.baseline,
        "seed": args.seed,
        "original_spectra": int(len(df)),
        "augmented_training_spectra": int(aug_count),
        "total_materialized_spectra": int(len(materialized)),
    }
    (out_dir / "augmentation_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    materialized.groupby(["split_main", "model_label", "is_augmented"]).size().reset_index(name="n_spectra").to_csv(
        tables_dir / "split_by_class_and_augmentation.csv", index=False, encoding="utf-8-sig"
    )
    materialized.groupby(["source_type", "model_label", "is_augmented"]).size().reset_index(name="n_spectra").to_csv(
        tables_dir / "source_by_class_and_augmentation.csv", index=False, encoding="utf-8-sig"
    )
    materialized[[
        "materialized_spectrum_id",
        "parent_spectrum_id",
        "model_label",
        "subtype_label",
        "source_type",
        "source_id",
        "split_main",
        "is_augmented",
        "materialized_file_path",
        "materialized_file_sha256",
        "augmentation_seed",
        "augmentation_parameters_json",
    ]].to_csv(tables_dir / "lineage_manifest.csv", index=False, encoding="utf-8-sig")

    log(f"Wrote metadata: {metadata_out}")
    log(f"Wrote spectra directory: {spectra_dir}")
    log(f"Original spectra: {len(df)}; augmented training spectra: {aug_count}; total: {len(materialized)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize deterministic Raman augmentation for reviewer-ready reproduction.")
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-train-per-class", type=int, default=200)
    parser.add_argument("--baseline", choices=["none", "poly", "asls"], default="poly")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
