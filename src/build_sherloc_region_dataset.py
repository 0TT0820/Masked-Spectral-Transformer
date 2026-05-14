"""Build point-level SHERLOC spectra from Corpolongo et al. region files.

This script links each region standard-label workbook sheet to the matching
SHERLOC all-points Raman CSV, then exports one spectrum CSV per labeled point
and appends those spectra to the parent metadata table.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MARS_ROOT = Path(r"D:\dyt\raman\metadata\mars\mars")
PARENT_METADATA = PROJECT_ROOT / "data" / "metadata_outputs" / "metadata_parent_945.csv"
OUTPUT_SPECTRA_DIR = PROJECT_ROOT / "data" / "sherloc_corpolongo_region_spectra"
OUTPUT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata_outputs"


REFERENCE = (
    "Corpolongo, A., et al. (2023). SHERLOC Raman mineral class detections "
    "of the Mars 2020 crater floor campaign. Journal of Geophysical Research: "
    "Planets, 128, e2022JE007455."
)


LABEL_MAP = {
    # raw SHERLOC label: (metadata major_category, Table 1 superclass, candidates, status, training usable)
    "carbonate": (
        "Carbonate",
        "Carbonate",
        "Carbonate",
        "supported_table1_closed_set",
        True,
    ),
    "olivine": (
        "Olivine",
        "Olivine",
        "Olivine",
        "supported_table1_closed_set",
        True,
    ),
    "pyroxene": (
        "Pyroxene",
        "Pyroxene",
        "Pyroxene",
        "supported_table1_closed_set",
        True,
    ),
    "silicate": (
        "Other Silicate",
        "Other Silicate",
        "Other Silicate",
        "broad_sherloc_silicate_mapped_to_table1_other_silicate",
        True,
    ),
    "sulfate": (
        "Sulfate",
        "Sulfate",
        "Sulfate",
        "supported_table1_closed_set",
        True,
    ),
    "na perchlorate": (
        "Perchlorate",
        "Perchlorate",
        "Perchlorate",
        "species_level_sherloc_label_mapped_to_table1_perchlorate",
        True,
    ),
    "perchlorate/chlorate": (
        "Perchlorate/Chlorate",
        "",
        "Perchlorate; Chlorate",
        "ambiguous_sherloc_salt_label_not_directly_usable_for_closed_set_training",
        False,
    ),
    "perchlorate/phospate": (
        "Perchlorate/Phosphate",
        "",
        "Perchlorate; Phosphate",
        "ambiguous_sherloc_label_spelled_phospate_not_directly_usable_for_closed_set_training",
        False,
    ),
    "perchlorate/phosphate": (
        "Perchlorate/Phosphate",
        "",
        "Perchlorate; Phosphate",
        "ambiguous_sherloc_label_not_directly_usable_for_closed_set_training",
        False,
    ),
}


@dataclass(frozen=True)
class SheetScanLink:
    region: str
    target: str
    sheet_name: str
    scan_name: str
    ss_file: Path
    mapping_basis: str


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return re.sub(r"_+", "_", value).strip("_")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_region_dirs() -> dict[str, Path]:
    wanted = {"dourbes", "garde", "guillaumes", "quartier"}
    found: dict[str, Path] = {}
    for path in MARS_ROOT.rglob("*"):
        if path.is_dir() and path.name.lower() in wanted:
            found[path.name.lower()] = path
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"Missing SHERLOC region folders: {sorted(missing)}")
    return found


def sort_ss_files(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, int, int, str]:
        name = path.name.lower()
        sol = int(re.search(r"ss__([0-9]+)_", name).group(1))
        srlc = int(re.search(r"srlc([0-9]+)w", name).group(1))
        wavelength_order = 0 if "w108" in name else 1
        return sol, srlc, wavelength_order, name

    return sorted(paths, key=key)


def parse_ss_filename(path: Path) -> dict[str, object]:
    name = path.name.lower()
    sol_match = re.search(r"ss__([0-9]+)_", name)
    sclk_match = re.search(r"ss__[0-9]+_([0-9]+)_([0-9]+)rrs__", name)
    srlc_match = re.search(r"srlc([0-9]+)w", name)
    subscan_match = re.search(r"w(108|208)", name)
    return {
        "ss_sol": int(sol_match.group(1)) if sol_match else "",
        "ss_sclk": f"{sclk_match.group(1)}.{sclk_match.group(2)}" if sclk_match else "",
        "ss_srlc_sequence": srlc_match.group(1) if srlc_match else "",
        "ss_subscan": subscan_match.group(1) if subscan_match else "",
    }


def excel_path(region_dir: Path) -> Path:
    candidates = [p for p in region_dir.glob("标准数据*.xlsx") if not p.name.startswith("~$")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one standard workbook in {region_dir}, found {candidates}")
    return candidates[0]


def scan_name_for(region: str, sheet: str) -> str:
    normalized = sheet.lower()
    if region == "dourbes":
        return f"0269_Dourbes Detail_500_{normalized.split('_')[-1].split('.')[0]}"
    if region == "garde":
        return f"0208_Garde Detail_500_{normalized.split('_')[-1].split('.')[0]}"
    if region == "guillaumes":
        return "0162_Guillaumes HDR_250_1"
    if region == "quartier":
        if normalized.startswith("q293"):
            return "0293_Quartier HDR_500_1"
        return f"0304_Quartier Detail_500_{normalized.split('_')[-1].split('.')[0]}"
    return sheet


def build_mapping(region_dirs: dict[str, Path]) -> tuple[list[SheetScanLink], list[dict[str, str]]]:
    links: list[SheetScanLink] = []
    rows: list[dict[str, str]] = []
    for region, region_dir in region_dirs.items():
        workbook = excel_path(region_dir)
        sheets = pd.ExcelFile(workbook).sheet_names
        ss_files = sort_ss_files(list(region_dir.glob("ss__*Raman.csv")))
        if len(sheets) != len(ss_files):
            raise ValueError(
                f"{region}: workbook has {len(sheets)} sheets but {len(ss_files)} Raman files"
            )
        for sheet, ss_file in zip(sheets, ss_files):
            target = {
                "dourbes": "Dourbes",
                "garde": "Garde",
                "guillaumes": "Guillaumes",
                "quartier": "Quartier",
            }[region]
            scan_name = scan_name_for(region, sheet)
            basis = (
                "Order-matched by Corpolongo et al. (2023) scan list and local "
                "file naming: sheets follow scan/detail order; Raman CSVs sorted "
                "by sol, SRLC sequence, and w108/w208 subscan."
            )
            links.append(SheetScanLink(region, target, sheet, scan_name, ss_file, basis))
            rows.append(
                {
                    "region": region,
                    "target": target,
                    "standard_workbook": str(workbook),
                    "sheet_name": sheet,
                    "scan_name": scan_name,
                    "ss_raman_file": str(ss_file),
                    **parse_ss_filename(ss_file),
                    "mapping_basis": basis,
                    "reference": REFERENCE,
                }
            )
    return links, rows


def normalize_label(raw: object) -> tuple[str, str, str, str, bool]:
    raw_text = str(raw).strip()
    key = raw_text.lower()
    major, table1, candidates, status, usable = LABEL_MAP.get(
        key,
        (
            raw_text.title(),
            "",
            "",
            "unmapped_manual_review_required",
            False,
        ),
    )
    return raw_text, major, table1, candidates, status, usable


def export_point_spectra(links: list[SheetScanLink]) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    counter = 0

    for link in links:
        workbook = excel_path(link.ss_file.parent)
        labels = pd.read_excel(workbook, sheet_name=link.sheet_name)
        spectrum_table = pd.read_csv(link.ss_file)
        shift_col = spectrum_table.columns[0]
        mineral_cols = [c for c in labels.columns if str(c).lower().startswith("minerall_type")]

        for _, label_row in labels.iterrows():
            point_name = str(label_row["Point名称"]).strip()
            if point_name not in spectrum_table.columns:
                manifest_rows.append(
                    {
                        "status": "missing_point_column",
                        "region": link.region,
                        "sheet_name": link.sheet_name,
                        "point_name": point_name,
                        "ss_raman_file": str(link.ss_file),
                    }
                )
                continue

            point_spectrum = spectrum_table[[shift_col, point_name]].copy()
            point_spectrum.columns = ["raman_shift_cm-1", "intensity"]
            point_spectrum = point_spectrum.dropna()
            point_spectrum["raman_shift_cm-1"] = pd.to_numeric(
                point_spectrum["raman_shift_cm-1"], errors="coerce"
            )
            point_spectrum["intensity"] = pd.to_numeric(point_spectrum["intensity"], errors="coerce")
            point_spectrum = point_spectrum.dropna().sort_values("raman_shift_cm-1")

            for label_slot, mineral_col in enumerate(mineral_cols, start=1):
                value = label_row[mineral_col]
                if pd.isna(value) or not str(value).strip():
                    continue
                (
                    raw_label,
                    major_category,
                    table1_superclass,
                    table1_candidates,
                    label_status,
                    training_usable,
                ) = normalize_label(value)
                file_stem = (
                    f"SHERLOC_{link.region}_{sanitize(link.sheet_name)}_{sanitize(point_name)}_"
                    f"label{label_slot}_{sanitize(major_category)}"
                )
                out_csv = OUTPUT_SPECTRA_DIR / f"{file_stem}.csv"
                point_spectrum.to_csv(out_csv, index=False)

                spectrum_id = f"SHERLOC_REGION_{counter:04d}"
                counter += 1
                row = {
                    "file_name": file_stem,
                    "group_label": major_category,
                    "subtype_label": raw_label,
                    "major_category": major_category,
                    "file_name_clean": file_stem,
                    "file_path": str(out_csv),
                    "match_method": "sherloc_region_point_extraction",
                    "file_exists": True,
                    "spectrum_id": spectrum_id,
                    "parsed_file_name": out_csv.name,
                    "mineral_species": raw_label,
                    "source_id": link.ss_file.stem,
                    "source_type": "SHERLOC in-situ Mars 2020",
                    "spectrum_type": "DUV Raman point spectrum",
                    "excitation_nm": 248.6,
                    "instrument": "SHERLOC, Perseverance rover",
                    "data_level": "point spectrum exported from all-points Raman CSV",
                    "orientation": "in-situ rover scan point",
                    "sample_provenance": f"Mars 2020 target {link.target}; region folder {link.region}",
                    "measurement_conditions": (
                        "In-situ SHERLOC deep-UV Raman; point intensity column extracted from "
                        "local all-points Raman CSV; see Corpolongo et al. (2023)."
                    ),
                    "label_basis": (
                        f"Point-level mineral class from {workbook.name}, sheet {link.sheet_name}, "
                        f"column {mineral_col}; two/three mineral assignments are retained as "
                        "separate metadata rows."
                    ),
                    "reference": REFERENCE,
                    "source_note": link.mapping_basis,
                    "spectral_min_cm-1": float(point_spectrum["raman_shift_cm-1"].min()),
                    "spectral_max_cm-1": float(point_spectrum["raman_shift_cm-1"].max()),
                    "n_original_points": int(len(point_spectrum)),
                    "spectral_range_cm-1": (
                        f"{point_spectrum['raman_shift_cm-1'].min():.1f}-"
                        f"{point_spectrum['raman_shift_cm-1'].max():.1f}"
                    ),
                    "file_sha256": file_sha256(out_csv),
                    "parent_group": f"SHERLOC_{link.target}_{link.sheet_name}",
                    "preprocessing_planned": (
                        "Mask non-SHERLOC Raman region below 800 cm-1; baseline correction; "
                        "max-intensity normalization; first derivative channel for MST."
                    ),
                    "augmentation_used": "no",
                    "qc_status": "include_pending_manual_review",
                    "qc_reason": label_status,
                    "recommended_action": (
                        "Use rows marked sherloc_training_label_usable=True for closed-set "
                        "fine-tuning/validation; ambiguous salt/phosphate labels should be "
                        "handled explicitly and not silently merged."
                    ),
                    "split_main": "sherloc_candidate_pool",
                    "split_zero_shot_protocol": "sherloc_finetune_candidate_not_augmented",
                    "sherloc_region": link.region,
                    "sherloc_target": link.target,
                    "sherloc_scan_name": link.scan_name,
                    "sherloc_sheet_name": link.sheet_name,
                    "sherloc_point_name": point_name,
                    "sherloc_label_column": mineral_col,
                    "sherloc_label_status": label_status,
                    "paper_table1_superclass": table1_superclass,
                    "paper_table1_superclass_candidates": table1_candidates,
                    "sherloc_training_label_usable": training_usable,
                    "sherloc_source_raman_csv": str(link.ss_file),
                    "sherloc_standard_workbook": str(workbook),
                }
                metadata_rows.append(row)
                manifest_rows.append(
                    {
                        "status": "exported",
                        "spectrum_id": spectrum_id,
                        "region": link.region,
                        "target": link.target,
                        "sheet_name": link.sheet_name,
                        "scan_name": link.scan_name,
                        "point_name": point_name,
                        "label_column": mineral_col,
                        "raw_label": raw_label,
                        "major_category": major_category,
                        "paper_table1_superclass": table1_superclass,
                        "paper_table1_superclass_candidates": table1_candidates,
                        "label_status": label_status,
                        "training_label_usable": training_usable,
                        "spectrum_file": str(out_csv),
                        "ss_raman_file": str(link.ss_file),
                    }
                )

    return pd.DataFrame(metadata_rows), pd.DataFrame(manifest_rows)


def append_to_parent(new_metadata: pd.DataFrame) -> pd.DataFrame:
    parent = pd.read_csv(PARENT_METADATA)
    for col in new_metadata.columns:
        if col not in parent.columns:
            parent[col] = ""
    for col in parent.columns:
        if col not in new_metadata.columns:
            new_metadata[col] = ""
    return pd.concat([parent, new_metadata[parent.columns]], ignore_index=True)


def main() -> None:
    OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    region_dirs = find_region_dirs()
    links, mapping_rows = build_mapping(region_dirs)
    mapping = pd.DataFrame(mapping_rows)
    new_metadata, extraction_manifest = export_point_spectra(links)
    combined = append_to_parent(new_metadata)
    training_ready = combined[
        (combined.get("sherloc_training_label_usable", "").astype(str).str.lower() != "false")
    ].copy()

    mapping_path = OUTPUT_METADATA_DIR / "sherloc_region_detail_to_ss_mapping.csv"
    manifest_path = OUTPUT_METADATA_DIR / "sherloc_region_point_extraction_manifest.csv"
    new_metadata_path = OUTPUT_METADATA_DIR / "metadata_sherloc_region_points_only.csv"
    combined_path = OUTPUT_METADATA_DIR / "metadata_parent_945_plus_sherloc_regions.csv"
    training_ready_path = OUTPUT_METADATA_DIR / "metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv"
    summary_path = OUTPUT_METADATA_DIR / "sherloc_region_dataset_summary.csv"
    table1_summary_path = OUTPUT_METADATA_DIR / "sherloc_region_table1_training_summary.csv"

    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    extraction_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    new_metadata.to_csv(new_metadata_path, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    training_ready.to_csv(training_ready_path, index=False, encoding="utf-8-sig")

    summary = (
        new_metadata.groupby(["sherloc_region", "major_category", "paper_table1_superclass", "sherloc_label_status", "sherloc_training_label_usable"])
        .size()
        .reset_index(name="n_point_label_rows")
        .sort_values(["sherloc_region", "major_category", "paper_table1_superclass", "sherloc_label_status"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    table1_summary = (
        new_metadata[new_metadata["sherloc_training_label_usable"] == True]
        .groupby(["paper_table1_superclass", "sherloc_region"])
        .size()
        .reset_index(name="n_training_usable_point_label_rows")
        .sort_values(["paper_table1_superclass", "sherloc_region"])
    )
    table1_summary.to_csv(table1_summary_path, index=False, encoding="utf-8-sig")

    print(f"Mapped sheets/scans: {len(mapping)}")
    print(f"Exported point-label spectra rows: {len(new_metadata)}")
    print(f"Combined metadata rows: {len(combined)}")
    print(f"Combined training-ready rows: {len(training_ready)}")
    print(f"Wrote: {mapping_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {new_metadata_path}")
    print(f"Wrote: {combined_path}")
    print(f"Wrote: {training_ready_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {table1_summary_path}")


if __name__ == "__main__":
    main()
