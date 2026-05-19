"""Apply reproducible Raman spectral quality control to the v2 database.

The goal is not to erase provenance, but to separate the archived data table
from the subset used for closed-set supervised model comparison. Each spectrum
receives numerical QC metrics and a decision:

- main_training_keep: acceptable for the main supervised comparison.
- manual_review_keep_in_metadata: retained for provenance but excluded from the
  main comparison until manually checked.
- exclude_from_closed_set: retained in metadata but excluded from closed-set
  mineral classification.

The rules are intentionally conservative for Raman spectroscopy: mineral band
positions are preserved, halides are not treated as diagnostic Raman classes,
and spectra dominated by continuum/noise are excluded or sent to review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, savgol_filter
except Exception:  # pragma: no cover - optional dependency fallback
    find_peaks = None
    savgol_filter = None


ROOT = Path(__file__).resolve().parents[1]
META_IN = ROOT / "data" / "metadata" / "metadata_training_database_v2_all_sources.csv"
META_OUT = ROOT / "data" / "metadata" / "metadata_training_database_v2_qc_filtered.csv"
META_BALANCED_OUT = ROOT / "data" / "metadata" / "metadata_training_database_v2_qc_balanced.csv"
META_HARD_OUT = ROOT / "data" / "metadata" / "metadata_training_database_v2_qc_hard_excluded.csv"
QC_DIR = ROOT / "data" / "overview" / "spectral_quality_qc_v2"
REPORT_OUT = ROOT / "docs" / "spectral_quality_qc_v2.md"

HALIDE_LABELS = {"Halides"}
LOW_CONFIDENCE_RRUFF_PHRASES = (
    "not yet confirmed",
    "determined only by raman spectroscopy",
    "determined only by raman",
)
PHYLLOSILICATE_LABELS = {"Clay", "Mica", "Serpentine", "Phyllosilicates"}


def resolve_path(value: str) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    candidate = ROOT.parent / path
    if candidate.exists():
        return candidate
    return path


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = pd.read_csv(path)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)
    if data.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    shift = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    inten = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(shift) & np.isfinite(inten)
    shift = shift[ok]
    inten = inten[ok]
    if len(shift) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    uniq, idx = np.unique(shift, return_index=True)
    inten = inten[idx]
    order = np.argsort(uniq)
    return uniq[order], inten[order]


def lower_envelope_poly_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 12:
        return np.zeros_like(y)
    degree = 2
    cutoff = np.percentile(y, 35)
    low = y <= cutoff
    if np.sum(low) < degree + 2:
        low = np.ones_like(y, dtype=bool)
    try:
        coef = np.polyfit(x[low], y[low], deg=degree)
        return np.polyval(coef, x)
    except Exception:
        return np.zeros_like(y)


def robust_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)) * 1.4826)


def local_peak_count(y: np.ndarray, min_distance_points: int) -> int:
    if len(y) < 3:
        return 0
    left = y[1:-1] > y[:-2]
    right = y[1:-1] >= y[2:]
    height = y[1:-1] >= 0.05
    candidates = np.where(left & right & height)[0] + 1
    if len(candidates) == 0:
        return 0
    selected = []
    for idx in candidates[np.argsort(y[candidates])[::-1]]:
        if all(abs(idx - old) >= min_distance_points for old in selected):
            selected.append(idx)
    return len(selected)


def spectrum_metrics(path: Path) -> dict:
    shift, raw = read_spectrum(path)
    if len(shift) < 2:
        return {
            "qc_file_readable": False,
            "qc_finite_points": int(len(shift)),
            "qc_spectral_min_cm1": np.nan,
            "qc_spectral_max_cm1": np.nan,
            "qc_spectral_span_cm1": np.nan,
            "qc_median_step_cm1": np.nan,
            "qc_snr_robust": 0.0,
            "qc_band_count": 0,
            "qc_top_band_prominence": 0.0,
            "qc_continuum_fraction": 1.0,
            "qc_spike_fraction": 1.0,
        }

    finite_points = len(shift)
    span = float(np.max(shift) - np.min(shift))
    steps = np.diff(shift)
    median_step = float(np.median(np.abs(steps))) if len(steps) else np.nan

    raw = raw.astype(float)
    raw_dynamic = float(np.percentile(raw, 99) - np.percentile(raw, 1))
    diff_noise = robust_mad(np.diff(raw)) / np.sqrt(2.0) if len(raw) > 2 else 0.0
    snr = raw_dynamic / (diff_noise + 1e-12)

    baseline = lower_envelope_poly_baseline(shift, raw)
    corrected = raw - baseline
    if savgol_filter is not None and len(corrected) >= 21:
        window = 21 if len(corrected) % 2 else 20
        window = max(5, window)
        if window % 2 == 0:
            window += 1
        if window < len(corrected):
            corrected = savgol_filter(corrected, window_length=window, polyorder=2)
    corrected = np.maximum(corrected, 0.0)
    max_corr = float(np.max(corrected)) if len(corrected) else 0.0
    norm = corrected / (max_corr + 1e-12) if max_corr > 0 else corrected

    distance_points = max(1, int(round(8.0 / max(median_step, 1e-6)))) if np.isfinite(median_step) else 5
    if find_peaks is not None and len(norm) > 3:
        peaks, props = find_peaks(norm, height=0.05, prominence=0.03, distance=distance_points)
        band_count = int(len(peaks))
        top_prom = float(np.max(props.get("prominences", [0.0]))) if band_count else 0.0
    else:
        band_count = int(local_peak_count(norm, distance_points))
        top_prom = float(np.max(norm)) if band_count else 0.0

    baseline_dynamic = float(np.percentile(baseline, 95) - np.percentile(baseline, 5))
    continuum_fraction = baseline_dynamic / (abs(raw_dynamic) + baseline_dynamic + 1e-12)
    diff = np.diff(raw)
    diff_mad = robust_mad(diff)
    if diff_mad <= 1e-12:
        spike_fraction = 0.0
    else:
        spike_fraction = float(np.mean(np.abs(diff - np.median(diff)) > 10.0 * diff_mad))

    return {
        "qc_file_readable": True,
        "qc_finite_points": int(finite_points),
        "qc_spectral_min_cm1": float(np.min(shift)),
        "qc_spectral_max_cm1": float(np.max(shift)),
        "qc_spectral_span_cm1": span,
        "qc_median_step_cm1": median_step,
        "qc_snr_robust": float(snr),
        "qc_band_count": int(band_count),
        "qc_top_band_prominence": float(top_prom),
        "qc_continuum_fraction": float(continuum_fraction),
        "qc_spike_fraction": float(spike_fraction),
    }


def low_confidence_rruff_status(status: object) -> bool:
    text = str(status).strip().lower()
    if not text or text == "nan":
        return False
    return any(phrase in text for phrase in LOW_CONFIDENCE_RRUFF_PHRASES)


def decide(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    label = str(row.get("label_category_final", "")).strip()
    source = str(row.get("source_type_normalized", "")).strip()
    role = str(row.get("training_role", "")).strip()

    if label in HALIDE_LABELS:
        return "exclude_from_closed_set", "halide_class_not_used_as_closed_set_raman_class"
    if not bool(row.get("qc_file_readable", False)):
        return "exclude_from_closed_set", "spectrum_file_unreadable_or_empty"
    if row.get("qc_finite_points", 0) < 50:
        return "exclude_from_closed_set", "too_few_finite_points"
    if row.get("qc_spectral_span_cm1", 0) < 300:
        return "exclude_from_closed_set", "spectral_range_too_narrow"
    if row.get("qc_snr_robust", 0) < 3 and row.get("qc_band_count", 0) == 0:
        return "exclude_from_closed_set", "flat_or_noise_dominated_no_detectable_raman_bands"
    if row.get("qc_spike_fraction", 0) > 0.08:
        return "manual_review_keep_in_metadata", "high_cosmic_spike_or_discontinuity_fraction"

    if low_confidence_rruff_status(row.get("rruff_status")):
        reasons.append("rruff_identification_status_low_confidence")
    if source == "RRUFF database" and not bool(row.get("rruff_metadata_match", False)):
        reasons.append("rruff_header_metadata_not_matched")
    if row.get("qc_continuum_fraction", 0) > 0.85 and row.get("qc_band_count", 0) < 2:
        reasons.append("continuum_or_fluorescence_dominated_weak_raman_bands")
    if label in PHYLLOSILICATE_LABELS and row.get("qc_band_count", 0) == 0:
        reasons.append("phyllosilicate_without_detected_diagnostic_bands")
    if "sherloc" in role.lower():
        # SHERLOC spectra are noisy by nature; retain labeled spectra unless
        # they fail hard numerical checks, but record their metrics.
        reasons = [r for r in reasons if not r.startswith("continuum")]

    if reasons:
        return "manual_review_keep_in_metadata", ";".join(reasons)
    return "main_training_keep", "passes_provenance_and_spectral_quality_checks"


def main() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_IN)
    metrics = []
    for _, row in meta.iterrows():
        path = resolve_path(str(row.get("file_path", "")))
        metrics.append(spectrum_metrics(path))
    metrics_df = pd.DataFrame(metrics)
    out = pd.concat([meta.reset_index(drop=True), metrics_df], axis=1)

    decisions = out.apply(decide, axis=1, result_type="expand")
    out["raman_qc_decision"] = decisions[0]
    out["raman_qc_reason"] = decisions[1]
    out["supervised_label_usable_before_qc"] = out["supervised_label_usable_v2"]

    reference_mask = out["split_v2"].isin(["train", "val", "test"])
    exclude_mask = reference_mask & out["raman_qc_decision"].ne("main_training_keep")
    out.loc[exclude_mask, "supervised_label_usable_v2"] = False
    out.loc[exclude_mask, "split_v2"] = "reference_qc_excluded"

    out.to_csv(META_OUT, index=False, encoding="utf-8-sig")

    original_split = meta["split_v2"].reset_index(drop=True)
    reference_mask_original = original_split.isin(["train", "val", "test"])
    reasons = out["raman_qc_reason"].fillna("")
    hard_exclude = reasons.str.contains(
        "halide|unreadable|too_few|spectral_range_too_narrow|flat_or_noise",
        case=False,
        regex=True,
    )
    balanced_exclude = hard_exclude | reasons.str.contains(
        "rruff_identification_status_low_confidence|phyllosilicate_without_detected|continuum_or_fluorescence",
        case=False,
        regex=True,
    )

    hard_out = out.copy()
    hard_out["split_v2"] = original_split
    hard_out["supervised_label_usable_v2"] = hard_out["supervised_label_usable_before_qc"]
    hard_ref_exclude = reference_mask_original & hard_exclude
    hard_out.loc[hard_ref_exclude, "supervised_label_usable_v2"] = False
    hard_out.loc[hard_ref_exclude, "split_v2"] = "reference_qc_excluded"
    hard_out.to_csv(META_HARD_OUT, index=False, encoding="utf-8-sig")

    balanced_out = out.copy()
    balanced_out["split_v2"] = original_split
    balanced_out["supervised_label_usable_v2"] = balanced_out["supervised_label_usable_before_qc"]
    balanced_ref_exclude = reference_mask_original & balanced_exclude
    balanced_out.loc[balanced_ref_exclude, "supervised_label_usable_v2"] = False
    balanced_out.loc[balanced_ref_exclude, "split_v2"] = "reference_qc_excluded"
    balanced_out["raman_qc_training_policy"] = np.where(
        hard_exclude,
        "hard_excluded",
        np.where(
            balanced_exclude,
            "excluded_from_main_training_balanced_policy",
            "eligible_for_main_training_balanced_policy",
        ),
    )
    balanced_out.to_csv(META_BALANCED_OUT, index=False, encoding="utf-8-sig")
    out[
        [
            "spectrum_id",
            "source_type_normalized",
            "training_role",
            "split_v2",
            "label_category_final",
            "mineral_species_final",
            "raman_qc_decision",
            "raman_qc_reason",
            "qc_snr_robust",
            "qc_band_count",
            "qc_top_band_prominence",
            "qc_continuum_fraction",
            "qc_spike_fraction",
            "rruff_status",
            "file_path",
        ]
    ].to_csv(QC_DIR / "spectrum_level_qc_decisions.csv", index=False, encoding="utf-8-sig")

    out.groupby(["source_type_normalized", "raman_qc_decision"], dropna=False).size().reset_index(
        name="n_spectra"
    ).to_csv(QC_DIR / "qc_decision_by_source.csv", index=False, encoding="utf-8-sig")
    out.groupby(["label_category_final", "raman_qc_decision"], dropna=False).size().reset_index(
        name="n_spectra"
    ).to_csv(QC_DIR / "qc_decision_by_class.csv", index=False, encoding="utf-8-sig")
    out.groupby(["split_v2", "raman_qc_decision"], dropna=False).size().reset_index(
        name="n_spectra"
    ).to_csv(QC_DIR / "qc_decision_by_split.csv", index=False, encoding="utf-8-sig")

    before_reference = int(reference_mask.sum())
    after_reference = int((out["split_v2"].isin(["train", "val", "test"]) & out["supervised_label_usable_v2"].astype(str).str.lower().isin(["true", "1"])).sum())
    hard_reference = int(
        (
            hard_out["split_v2"].isin(["train", "val", "test"])
            & hard_out["supervised_label_usable_v2"].astype(str).str.lower().isin(["true", "1"])
        ).sum()
    )
    balanced_reference = int(
        (
            balanced_out["split_v2"].isin(["train", "val", "test"])
            & balanced_out["supervised_label_usable_v2"].astype(str).str.lower().isin(["true", "1"])
        ).sum()
    )
    decision_table = out["raman_qc_decision"].value_counts().rename_axis("decision").reset_index(name="n_spectra")
    source_table = (
        out.groupby(["source_type_normalized", "raman_qc_decision"], dropna=False).size().reset_index(name="n_spectra")
    )

    report = f"""# Raman Spectral Quality Control v2

This QC layer was added to address reviewer concerns about spectral provenance,
RRUFF quality heterogeneity, fluorescence-dominated spectra, weak Raman bands,
and the physical unsuitability of halides as a closed-set Raman class.

The original all-source metadata table is retained unchanged. The filtered table
`data/metadata/metadata_training_database_v2_qc_filtered.csv` adds numerical
QC metrics and marks spectra excluded from the main closed-set supervised
comparison as `reference_qc_excluded`.

## QC Rules

1. Hard exclusions: unreadable files, fewer than 50 finite points, spectral span
   below 300 cm-1, flat/noise-dominated spectra with no detected Raman bands,
   and halides.
2. Manual-review flags: RRUFF records with low-confidence identification status,
   unmatched RRUFF header metadata, high spike/discontinuity fraction, or spectra
   dominated by broad continuum with too few Raman bands.
3. SHERLOC spectra are retained unless they fail hard numerical checks, because
   rover DUV spectra are expected to be noisier than laboratory references.
4. The main model-comparison subset uses only spectra marked
   `main_training_keep`.

## Summary

- Reference candidate spectra before QC: {before_reference}
- Reference spectra after hard exclusions only: {hard_reference}
- Reference spectra after balanced QC policy: {balanced_reference}
- Reference spectra after strict QC policy: {after_reference}

The balanced policy is recommended for the main reviewer experiment because it
removes physically problematic classes and low-confidence/continuum-dominated
reference spectra while avoiding an overly aggressive automatic exclusion of
spectra that only show spike-like discontinuities. Those spike-flagged spectra
remain visible in the spectrum-level QC table and should be inspected manually
before final archival release.

### Decisions

{decision_table.to_markdown(index=False)}

### Decisions by Source

{source_table.to_markdown(index=False)}
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"Wrote QC-filtered metadata: {META_OUT}")
    print(f"Wrote QC report: {REPORT_OUT}")
    print(decision_table.to_string(index=False))


if __name__ == "__main__":
    main()
