"""Build an MLROD-integrated Raman metadata inventory.

The script does not duplicate the large MLROD spectra into tens of thousands of
two-column files. Instead, each MLROD row is represented by its source raw CSV,
row index, ODR data-record id, and source citation. Downstream scripts can read
the row directly from the wide MLROD CSV and align it to the manuscript's common
Raman-shift grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "metadata"
OVERVIEW_DIR = ROOT / "data" / "overview" / "mlrod_integrated_v3"
MLROD_DIR = ROOT / "data" / "external_mlrod_berlanga_2022"
METEORITE_SPOT_WIDE = ROOT / "data" / "minerals_100spots_wide.csv"

BASE_METADATA = METADATA_DIR / "metadata_training_database_v2_all_sources.csv"
MLROD_FILE_CATALOG = MLROD_DIR / "catalog" / "mlrod_file_catalog.csv"
MLROD_RAW_DIR = MLROD_DIR / "raw_raman_files"

OUT_METADATA = METADATA_DIR / "metadata_training_database_v3_mlrod_integrated.csv"


SYNTHETIC_METEORITE_LABEL_MAP = {
    "clinopyroxene": ("Pyroxene", "Clinopyroxene"),
    "orthopyroxene": ("Pyroxene", "Orthopyroxene"),
    "olivine": ("Olivine", "Olivine"),
    "plagioclase": ("Plagioclase", "Plagioclase"),
    "gypsum": ("Sulfate", "Gypsum"),
    "hematite": ("Oxides/Hydroxides", "Hematite"),
    "quartz": ("Silica Phase", "Quartz"),
    "serpentine": ("Phyllosilicates", "Serpentine"),
}

SAMPLE_TO_MODEL_LABEL = {
    "Feldspar - Albite": "Plagioclase",
    "Feldspar - Anorthite": "Plagioclase",
    "Feldspar - Microcline": "K-Feldspar",
    "Pyroxene - Augite": "Pyroxene",
    "Pyroxene - Enstatite": "Pyroxene",
    "Mica - Biotite": "Phyllosilicates",
    "Mica - Muscovite": "Phyllosilicates",
    "Carbonate - Calcite": "Carbonate",
    "Olivine - Forsterite": "Olivine",
    "Sulfate - Gypsum": "Sulfate",
    "Amphibole - Hornblende": "Other Silicates",
    "Quartz": "Silica Phase",
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def repo_relative(path_value: object) -> str:
    text = clean(path_value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    for root in (ROOT, ROOT.parent):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    parts = list(path.parts)
    if "publication_repo" in parts:
        return Path(*parts[parts.index("publication_repo") + 1 :]).as_posix()
    if "data" in parts:
        return Path(*parts[parts.index("data") :]).as_posix()
    return str(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def infer_wave_range(path: Path) -> tuple[float, float, int]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wave_values = []
    for col in header[1:-1]:
        try:
            wave_values.append(float(col))
        except ValueError:
            continue
    if not wave_values:
        return float("nan"), float("nan"), 0
    return min(wave_values), max(wave_values), len(wave_values)


def infer_shift_columns_from_prefix(path: Path, prefix: str = "shift_") -> tuple[float, float, int]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    shifts = []
    for col in header:
        if not col.startswith(prefix):
            continue
        try:
            shifts.append(float(col.removeprefix(prefix)))
        except ValueError:
            continue
    if not shifts:
        return float("nan"), float("nan"), 0
    return min(shifts), max(shifts), len(shifts)


def choose_mlrod_split(row_number: int) -> str:
    """Deterministic row-level split for the MLROD source domain.

    The split is only used for MLROD-domain sanity checks. In the manuscript
    reference-domain benchmark, the primary held-out test remains the measured
    non-augmented reference test split.
    """

    r = row_number % 10
    if r < 8:
        return "mlrod_train"
    if r == 8:
        return "mlrod_val"
    return "mlrod_test"


def make_base_rows() -> list[dict[str, object]]:
    df = pd.read_csv(BASE_METADATA, low_memory=False)
    rows = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["metadata_version"] = "v2_existing_source"
        rec["spectrum_storage_format"] = "two_column_csv"
        if "file_path" in rec:
            rec["file_path"] = repo_relative(rec["file_path"])
        rec["mlrod_container_file"] = ""
        rec["mlrod_row_index"] = ""
        rec["mlrod_original_row_id"] = ""
        rec["split_v3"] = clean(row.get("split_v2", ""))
        rows.append(rec)
    return rows


def make_mlrod_rows() -> list[dict[str, object]]:
    catalog = pd.read_csv(MLROD_FILE_CATALOG)
    catalog = catalog[
        (catalog["file_kind"] == "raw_training")
        & (~catalog["project_superclass_mapping"].isin(["mixture_or_rock", "rock_test_set"]))
        & (catalog["sample_id"].isin(SAMPLE_TO_MODEL_LABEL))
    ].copy()

    rows: list[dict[str, object]] = []
    for _, file_row in catalog.sort_values(["sample_id", "filename"]).iterrows():
        raw_path = MLROD_RAW_DIR / clean(file_row["filename"])
        if not raw_path.exists():
            continue
        raw_rel = repo_relative(raw_path)
        n_rows = line_count(raw_path)
        wmin, wmax, n_points = infer_wave_range(raw_path)
        sha = file_sha256(raw_path)
        model_label = SAMPLE_TO_MODEL_LABEL[clean(file_row["sample_id"])]
        for idx in range(n_rows):
            split = choose_mlrod_split(idx)
            spectrum_id = f"MLROD_{clean(file_row['file_id'])}_{idx:06d}"
            rows.append(
                {
                    "file_name": f"{spectrum_id}.wide_row",
                    "group_label": model_label,
                    "subtype_label": clean(file_row.get("mineral_name", "")),
                    "major_category": model_label,
                    "file_name_clean": f"{spectrum_id}.wide_row",
                    "file_path": raw_rel,
                    "match_method": "mlrod_odr_raw_row",
                    "file_exists": True,
                    "spectrum_id": spectrum_id,
                    "parsed_file_name": clean(file_row["filename"]),
                    "mineral_species": clean(file_row.get("mineral_name", "")),
                    "source_id": f"MLROD_ODR_{clean(file_row['odr_datarecord_id'])}_file_{clean(file_row['file_id'])}",
                    "source_type": "MLROD Raman open dataset",
                    "spectrum_type": "Raman point spectrum",
                    "excitation_nm": "",
                    "instrument": "MLROD Raman acquisition; see Berlanga et al. (2022) and MLROD DOI record",
                    "data_level": "raw wide CSV row",
                    "orientation": "",
                    "sample_provenance": clean(file_row.get("location", "")),
                    "measurement_conditions": "Low-signal Raman benchmark data from MLROD; raw row retained and aligned during preprocessing",
                    "label_basis": "MLROD single-mineral sample identity mapped to manuscript mineral superclass",
                    "reference": "Berlanga et al. (2022), Earth and Space Science, DOI:10.1029/2021EA002125; MLROD DOI:10.48484/PWRB-R137",
                    "source_note": "MLROD single-mineral raw training spectrum; mixtures and rock slabs excluded from closed-set supervised training",
                    "spectral_min_cm-1": wmin,
                    "spectral_max_cm-1": wmax,
                    "n_original_points": n_points,
                    "spectral_range_cm-1": f"{wmin:.3f}-{wmax:.3f}",
                    "file_sha256": sha,
                    "parent_group": f"MLROD__{clean(file_row['sample_id'])}__{clean(file_row['filename'])}",
                    "preprocessing_planned": "Read wide CSV row; parse numeric Raman-shift columns; interpolate to common grid; baseline correction; nonnegative clipping; max-intensity normalization; derivative channel.",
                    "augmentation_used": "no",
                    "qc_status": "include_external_mlrod_single_mineral",
                    "qc_reason": "single-mineral MLROD source with explicit ODR provenance and compatible closed-set superclass",
                    "recommended_action": "retain_for_mlrod_integrated_training_or_external_domain_check",
                    "split_main": split,
                    "split_zero_shot_protocol": "mlrod_external_source",
                    "source_type_normalized": "MLROD Raman open dataset",
                    "label_category_final": model_label,
                    "mineral_species_final": clean(file_row.get("mineral_name", "")),
                    "source_domain": "mlrod_low_signal_visible_reference",
                    "training_role": "mlrod_supervised_reference",
                    "supervised_label_usable_v2": True,
                    "duv_library_include": False,
                    "split_v2": split,
                    "metadata_version": "v3_mlrod_integrated",
                    "spectrum_storage_format": "mlrod_wide_csv_row",
                    "mlrod_container_file": raw_rel,
                    "mlrod_row_index": idx,
                    "mlrod_original_row_id": "",
                    "mlrod_odr_datarecord_id": clean(file_row["odr_datarecord_id"]),
                    "mlrod_file_id": clean(file_row["file_id"]),
                    "mlrod_sample_id": clean(file_row["sample_id"]),
                    "mlrod_raw_filename": clean(file_row["filename"]),
                    "split_v3": split,
                }
            )
    return rows


def make_synthetic_meteorite_spot_rows() -> list[dict[str, object]]:
    """Represent the 8-mineral, 100-spot wide table as training-only rows.

    The input file explicitly marks its source as ``synthetic_teaching_only``.
    These rows are therefore kept transparent and must not be described as
    measured meteorite spectra in manuscript provenance tables.
    """

    if not METEORITE_SPOT_WIDE.exists():
        return []

    wide = pd.read_csv(METEORITE_SPOT_WIDE, low_memory=False)
    file_rel = repo_relative(METEORITE_SPOT_WIDE)
    sha = file_sha256(METEORITE_SPOT_WIDE)
    wmin, wmax, n_points = infer_shift_columns_from_prefix(METEORITE_SPOT_WIDE)
    rows: list[dict[str, object]] = []

    for idx, row in wide.iterrows():
        mineral_key = clean(row.get("mineral", "")).lower()
        if mineral_key not in SYNTHETIC_METEORITE_LABEL_MAP:
            continue
        model_label, mineral_species = SYNTHETIC_METEORITE_LABEL_MAP[mineral_key]
        spot_id = clean(row.get("spot_id", "")) or f"{mineral_key}_spot_{idx + 1:03d}"
        source_flag = clean(row.get("source", ""))
        spectrum_id = f"SYNTH_METEORITE_{mineral_key.upper()}_{idx:04d}"
        rows.append(
            {
                "file_name": f"{spectrum_id}.wide_row",
                "group_label": model_label,
                "subtype_label": mineral_species,
                "major_category": model_label,
                "file_name_clean": f"{spectrum_id}.wide_row",
                "file_path": file_rel,
                "match_method": "wide_table_row_from_minerals_100spots",
                "file_exists": True,
                "spectrum_id": spectrum_id,
                "parsed_file_name": METEORITE_SPOT_WIDE.name,
                "mineral_species": mineral_species,
                "source_id": f"{METEORITE_SPOT_WIDE.stem}:{spot_id}",
                "source_type": "Synthetic meteorite mineral spot spectra",
                "spectrum_type": "Raman point spectrum",
                "excitation_nm": "",
                "instrument": "Synthetic/teaching Raman spectrum generator; no measured instrument provenance encoded in source file",
                "data_level": "wide CSV row",
                "orientation": "",
                "sample_provenance": "Meteorite-relevant mineral spot spectrum; source file marks rows as synthetic_teaching_only",
                "measurement_conditions": "Synthetic teaching-only spectrum with 2 cm-1 Raman-shift spacing; use as training-only supplemental/sensitivity data, not as measured validation evidence",
                "label_basis": "Mineral name encoded in source wide table",
                "reference": "Generated local teaching/simulation product; measured provenance not provided in the CSV",
                "source_note": "Transparent synthetic meteorite-mineral spot supplement. Do not report these rows as measured meteorite spectra.",
                "spectral_min_cm-1": wmin,
                "spectral_max_cm-1": wmax,
                "n_original_points": n_points,
                "spectral_range_cm-1": f"{wmin:.3f}-{wmax:.3f}",
                "file_sha256": sha,
                "parent_group": f"SYNTH_METEORITE__{mineral_key}__{spot_id}",
                "preprocessing_planned": "Read wide CSV row; parse shift_* Raman-shift columns; interpolate to common grid; baseline correction; nonnegative clipping; max-intensity normalization; derivative channel.",
                "augmentation_used": "no",
                "qc_status": "synthetic_teaching_only",
                "qc_reason": "source field in minerals_100spots_wide.csv is synthetic_teaching_only; measured provenance is not encoded",
                "recommended_action": "retain only as transparent training-only supplemental/sensitivity spectra; exclude from measured validation/test claims",
                "split_main": "train",
                "split_zero_shot_protocol": "not_applicable_training_only_synthetic",
                "source_type_normalized": "Synthetic meteorite mineral spot spectra",
                "label_category_final": model_label,
                "mineral_species_final": mineral_species,
                "source_domain": "synthetic_meteorite_mineral_reference",
                "training_role": "synthetic_meteorite_training_supplement",
                "supervised_label_usable_v2": True,
                "duv_library_include": False,
                "split_v2": "train",
                "metadata_version": "v3_synthetic_meteorite_spot_supplement",
                "spectrum_storage_format": "synthetic_meteorite_wide_csv_row",
                "wide_source_file": file_rel,
                "wide_row_index": idx,
                "wide_source_field": source_flag,
                "synthetic_or_measured": "synthetic_teaching_only",
                "split_v3": "train",
            }
        )
    return rows


def write_counts(df: pd.DataFrame) -> None:
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for index, columns, filename in [
        ("source_type_normalized", "split_v3", "source_by_split_v3.csv"),
        ("label_category_final", "split_v3", "class_by_split_v3.csv"),
        ("source_type_normalized", "label_category_final", "source_by_class_v3.csv"),
        ("training_role", "label_category_final", "training_role_by_class_v3.csv"),
    ]:
        pd.crosstab(df[index].fillna("").astype(str), df[columns].fillna("").astype(str)).to_csv(
            OVERVIEW_DIR / filename, encoding="utf-8-sig"
        )
    summary = (
        df.groupby(["source_type_normalized", "training_role", "split_v3"], dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["source_type_normalized", "training_role", "split_v3"])
    )
    summary.to_csv(OVERVIEW_DIR / "source_role_split_summary_v3.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MLROD-integrated v3 metadata table.")
    parser.add_argument("--out-metadata", type=Path, default=OUT_METADATA)
    args = parser.parse_args()
    out_metadata = args.out_metadata
    if not out_metadata.is_absolute():
        out_metadata = ROOT / out_metadata

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    base_rows = make_base_rows()
    synthetic_meteorite_rows = make_synthetic_meteorite_spot_rows()
    mlrod_rows = make_mlrod_rows()
    all_keys = sorted(set().union(*(r.keys() for r in [*base_rows, *synthetic_meteorite_rows, *mlrod_rows])))
    all_rows = []
    for row in [*base_rows, *synthetic_meteorite_rows, *mlrod_rows]:
        all_rows.append({key: row.get(key, "") for key in all_keys})
    with out_metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_rows)
    df = pd.DataFrame(all_rows)
    write_counts(df)
    print(f"Existing v2 rows: {len(base_rows)}")
    print(f"Synthetic meteorite mineral spot rows added: {len(synthetic_meteorite_rows)}")
    print(f"MLROD single-mineral raw rows added: {len(mlrod_rows)}")
    print(f"Total v3 rows: {len(all_rows)}")
    print(f"Wrote: {out_metadata}")
    print(f"Overview: {OVERVIEW_DIR}")


if __name__ == "__main__":
    main()
