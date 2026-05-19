"""Build traceable weak-label SHERLOC candidate spectra for Montpezat/Alfalfa.

The Corpolongo et al. (2023) paper reports scan-level mineral detections for
Montpezat and Alfalfa, but the public article does not provide a point-by-point
label workbook equivalent to the local Dourbes/Garde/Guillaumes/Quartier files.
This script therefore extracts only the PDS region-level representative spectra
from the HDR products and marks them as weak-label candidates, not default
closed-set fine-tuning labels.
"""

from __future__ import annotations

import csv
import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PDS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "urn-nasa-pds-mars2020_sherloc/data_processed"
)
PDS_COLLECTION_LID = "urn:nasa:pds:mars2020_sherloc:data_processed"
PDS_BUNDLE_DOI = "10.17189/1522643"
EXCITATION_NM = 248.6

OUT_SPECTRA = ROOT / "data" / "sherloc_montpezat_alfalfa_candidates"
OUT_METADATA = ROOT / "data" / "metadata"

REFERENCE = (
    "Corpolongo, A., et al. (2023). SHERLOC Raman mineral class detections "
    "of the Mars 2020 crater floor campaign. Journal of Geophysical Research: "
    "Planets, 128, e2022JE007455."
)


@dataclass(frozen=True)
class ProductSpec:
    target: str
    scan_name: str
    product_stem: str
    label_summary: str
    candidate_label: str
    label_status: str
    paper_context: str

    @property
    def sol(self) -> int:
        return int(self.product_stem.split("_")[2])

    @property
    def sol_dir(self) -> str:
        return f"sol_{self.sol:05d}"

    @property
    def csv_url(self) -> str:
        return f"{PDS_BASE_URL}/{self.sol_dir}/{self.product_stem}.csv"

    @property
    def label_url(self) -> str:
        return f"{PDS_BASE_URL}/{self.sol_dir}/{self.product_stem}.xml"

    @property
    def lid(self) -> str:
        return f"{PDS_COLLECTION_LID}:{self.product_stem}"


PRODUCTS = [
    ProductSpec(
        target="Montpezat",
        scan_name="0349_Montpezat HDR_500_1",
        product_stem="ss__0349_0697954375_450rrs__0092982srlc16000w1__cgnj02",
        label_summary=(
            "Montpezat HDR_500_1: silicate, carbonate, and perchlorate-or-"
            "phosphate signatures were reported; 17 single spectra displayed "
            "silicate signatures, four of them high-confidence; three carbonate "
            "and three perchlorate/phosphate spectra were observed."
        ),
        candidate_label="mixed_silicate_carbonate_perchlorate_or_phosphate",
        label_status="weak_scan_level_label_not_point_level",
        paper_context=(
            "The paper reports scan-level detections and representative mean "
            "spectra, not a point-level mineral-label table for this target."
        ),
    ),
    ProductSpec(
        target="Alfalfa",
        scan_name="0370_Alfalfa HDR_500_1",
        product_stem="ss__0370_0699816583_410rrs__0110108srlc16000w1__cgnj02",
        label_summary=(
            "Alfalfa HDR_500_1: spectra were reported as predominantly "
            "silicate, with minor carbonate and perchlorate-or-phosphate; "
            "44 of 100 spectra were silicate detections, 30 high-confidence."
        ),
        candidate_label="predominantly_silicate_with_minor_carbonate_perchlorate_or_phosphate",
        label_status="weak_scan_level_label_not_point_level",
        paper_context=(
            "The paper reports scan-level detections and representative mean "
            "spectra, not a point-level mineral-label table for this target."
        ),
    ),
]


PROVENANCE_ONLY_PRODUCTS = [
    ("Montpezat", "0349_Montpezat Survey_15_1 or HDR component", "ss__0349_0697951251_495rrs__0092982srlc11360w108cgnj02"),
    ("Montpezat", "0349_Montpezat Survey_15_1 or HDR component", "ss__0349_0697951900_355rrs__0092982srlc11360w208cgnj02"),
    ("Montpezat", "0349_Montpezat Survey_15_1 or HDR component", "ss__0349_0697952686_570rrs__0092982srlc11420w108cgnj07"),
    ("Montpezat", "0349_Montpezat Survey_15_1 or HDR component", "ss__0349_0697953530_075rrs__0092982srlc11420b108zpzj02"),
    ("Alfalfa", "0370_Alfalfa Survey_15_1 or HDR component", "ss__0370_0699813450_405rrs__0110108srlc11360w108cgnj02"),
    ("Alfalfa", "0370_Alfalfa Survey_15_1 or HDR component", "ss__0370_0699814099_215rrs__0110108srlc11360w208cgnj02"),
    ("Alfalfa", "0370_Alfalfa Survey_15_1 or HDR component", "ss__0370_0699814879_265rrs__0110108srlc11420w108cgnj07"),
    ("Alfalfa", "0370_Alfalfa Survey_15_1 or HDR component", "ss__0370_0699815741_520rrs__0110108srlc11420b108zpzj02"),
]


def sanitize(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]+", "_", text)).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def label_metadata(label_url: str) -> dict[str, str]:
    xml_text = fetch_text(label_url)
    root = ET.fromstring(xml_text)
    meta = {
        "pds_start_date_time": "",
        "pds_stop_date_time": "",
        "pds_release_number": "",
        "pds_records_by_table": "",
        "pds_table_names": "",
    }
    table_names: list[str] = []
    records: list[str] = []
    for elem in root.iter():
        name = local_name(elem.tag)
        text = (elem.text or "").strip()
        if name == "start_date_time":
            meta["pds_start_date_time"] = text
        elif name == "stop_date_time":
            meta["pds_stop_date_time"] = text
        elif name == "release_number":
            meta["pds_release_number"] = text
        elif name == "Table_Delimited":
            for child in elem:
                child_name = local_name(child.tag)
                child_text = (child.text or "").strip()
                if child_name == "name":
                    table_names.append(child_text)
                elif child_name == "records":
                    records.append(child_text)
    meta["pds_records_by_table"] = "|".join(records)
    meta["pds_table_names"] = "|".join(table_names)
    return meta


def parse_pds_csv_sections(text: str) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = next(csv.reader([line]))
        if len(parsed) == 1:
            token = parsed[0].strip()
            normalized = token.replace("_", " ").replace(":", "").strip()
            if normalized.upper().startswith("WAVELENGTH"):
                current = "WAVELENGTH (NM)"
                sections[current] = []
                continue
            region_match = re.search(r"REGION\s*([123])", normalized, re.IGNORECASE)
            if "SPECTRA" in normalized.upper() and region_match:
                current = f"LASER-NORMALIZED SPECTRA REGION {region_match.group(1)}"
                sections[current] = []
                continue
        if line.endswith(":") and "," not in line:
            current = line[:-1].strip()
            sections[current] = []
            continue
        if current:
            sections[current].append(parsed)
    return sections


def wavelengths_to_shift(wavelength_nm: list[float]) -> list[float]:
    laser = 1e7 / EXCITATION_NM
    return [laser - (1e7 / wl) for wl in wavelength_nm]


def extract_region_spectra(product: ProductSpec) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    text = fetch_text(product.csv_url)
    sections = parse_pds_csv_sections(text)
    if "WAVELENGTH (NM)" not in sections:
        raise ValueError(f"No wavelength section in {product.product_stem}")
    wavelength_rows = sections["WAVELENGTH (NM)"]
    wavelengths = [float(v) for v in wavelength_rows[1] if v.strip()]
    shifts = wavelengths_to_shift(wavelengths)

    rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, str]] = []
    pds_meta = label_metadata(product.label_url)

    for region_number in (1, 2, 3):
        section_name = f"LASER-NORMALIZED SPECTRA REGION {region_number}"
        if section_name not in sections:
            continue
        table = sections[section_name]
        for row_index, value_row in enumerate(table[1:]):
            values = [float(v) if v.strip() else float("nan") for v in value_row]
            spectrum = (
                pd.DataFrame({"raman_shift_cm-1": shifts[: len(values)], "intensity": values})
                .dropna()
                .sort_values("raman_shift_cm-1")
            )
            spectrum = spectrum[spectrum["raman_shift_cm-1"] > 0].copy()
            file_stem = (
                f"SHERLOC_{sanitize(product.target)}_{sanitize(product.scan_name)}_"
                f"R{region_number}_row{row_index:02d}_weak_candidate"
            )
            out_path = OUT_SPECTRA / f"{file_stem}.csv"
            spectrum.to_csv(out_path, index=False)
            dominant = spectrum.loc[spectrum["intensity"].idxmax()]
            spectrum_id = f"SHERLOC_WEAK_{product.target.upper()}_R{region_number}_{row_index:02d}"
            row = {
                "file_name": file_stem,
                "group_label": product.candidate_label,
                "subtype_label": product.candidate_label,
                "major_category": product.candidate_label,
                "file_name_clean": file_stem,
                "file_path": str(out_path),
                "match_method": "pds_region_spectrum_weak_label_extraction",
                "file_exists": True,
                "spectrum_id": spectrum_id,
                "parsed_file_name": out_path.name,
                "mineral_species": product.candidate_label,
                "source_id": product.product_stem,
                "source_type": "SHERLOC in-situ Mars 2020",
                "spectrum_type": "DUV Raman PDS region-level representative spectrum",
                "excitation_nm": EXCITATION_NM,
                "instrument": "SHERLOC, Perseverance rover",
                "data_level": "PDS processed RRS laser-normalized region spectrum",
                "orientation": "in-situ rover scan region spectrum",
                "sample_provenance": f"Mars 2020 target {product.target}; {product.scan_name}",
                "measurement_conditions": (
                    "In-situ SHERLOC deep-UV Raman; PDS processed RRS product; "
                    "laser-normalized spectra in public PDS4 CSV table."
                ),
                "label_basis": (
                    "Weak scan-level candidate label from Corpolongo et al. (2023); "
                    "not a point-level mineral assignment. Use only for exploratory "
                    "semi-supervised/domain-adaptation analyses unless a point-label "
                    "workbook is supplied."
                ),
                "reference": REFERENCE,
                "source_note": product.label_summary,
                "spectral_min_cm-1": float(spectrum["raman_shift_cm-1"].min()),
                "spectral_max_cm-1": float(spectrum["raman_shift_cm-1"].max()),
                "n_original_points": int(len(spectrum)),
                "spectral_range_cm-1": (
                    f"{spectrum['raman_shift_cm-1'].min():.1f}-"
                    f"{spectrum['raman_shift_cm-1'].max():.1f}"
                ),
                "file_sha256": sha256_file(out_path),
                "parent_group": f"SHERLOC_{product.target}_{product.scan_name}",
                "preprocessing_planned": (
                    "Mask non-SHERLOC Raman region below 800 cm-1; baseline correction; "
                    "max-intensity normalization; first derivative channel for MST."
                ),
                "augmentation_used": "no",
                "qc_status": "candidate_only",
                "qc_reason": product.label_status,
                "recommended_action": (
                    "Do not include in the default closed-set fine-tuning table; add to "
                    "training only after point-level labels are independently curated."
                ),
                "split_main": "sherloc_weak_label_candidate_pool",
                "split_zero_shot_protocol": "sherloc_candidate_not_default_training",
                "sherloc_region": product.target.lower(),
                "sherloc_target": product.target,
                "sherloc_scan_name": product.scan_name,
                "sherloc_sheet_name": "",
                "sherloc_point_name": f"R{region_number}_row{row_index}",
                "sherloc_label_column": "scan_level_summary",
                "sherloc_label_status": product.label_status,
                "paper_table1_superclass": "",
                "paper_table1_superclass_candidates": (
                    "Other Silicate; Carbonate; Perchlorate; Phosphate"
                ),
                "sherloc_training_label_usable": False,
                "sherloc_source_raman_csv": product.csv_url,
                "sherloc_standard_workbook": "",
                "pds_product_lid": product.lid,
                "pds_csv_url": product.csv_url,
                "pds_label_url": product.label_url,
                "pds_bundle_doi": PDS_BUNDLE_DOI,
                "pds_table_name": section_name,
                "pds_table_row_index": row_index,
                "dominant_band_cm-1": float(dominant["raman_shift_cm-1"]),
                "dominant_band_intensity": float(dominant["intensity"]),
                "paper_context": product.paper_context,
                **pds_meta,
            }
            rows.append(row)
            provenance_rows.append(
                {
                    "target": product.target,
                    "scan_name": product.scan_name,
                    "product_stem": product.product_stem,
                    "pds_product_lid": product.lid,
                    "pds_csv_url": product.csv_url,
                    "pds_label_url": product.label_url,
                    "extracted_spectrum_id": spectrum_id,
                    "pds_table_name": section_name,
                    "pds_table_row_index": str(row_index),
                    "candidate_label": product.candidate_label,
                    "label_status": product.label_status,
                    "dominant_band_cm-1": f"{float(dominant['raman_shift_cm-1']):.2f}",
                    **pds_meta,
                }
            )
    return rows, provenance_rows


def provenance_only_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for target, scan_name, stem in PROVENANCE_ONLY_PRODUCTS:
        spec = ProductSpec(
            target=target,
            scan_name=scan_name,
            product_stem=stem,
            label_summary="PDS point-spectra component product; no point-level label table available here.",
            candidate_label="unlabeled_pds_point_spectra_component",
            label_status="no_point_level_label_available",
            paper_context="Listed for traceability; not extracted into default training labels.",
        )
        try:
            pds_meta = label_metadata(spec.label_url)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            pds_meta = {
                "pds_start_date_time": "",
                "pds_stop_date_time": "",
                "pds_release_number": "",
                "pds_records_by_table": "",
                "pds_table_names": "",
            }
            status = f"label_fetch_failed: {exc}"
        rows.append(
            {
                "target": target,
                "scan_name": scan_name,
                "product_stem": stem,
                "pds_product_lid": spec.lid,
                "pds_csv_url": spec.csv_url,
                "pds_label_url": spec.label_url,
                "extracted_spectrum_id": "",
                "pds_table_name": "",
                "pds_table_row_index": "",
                "candidate_label": spec.candidate_label,
                "label_status": spec.label_status,
                "dominant_band_cm-1": "",
                "provenance_status": status,
                **pds_meta,
            }
        )
    return rows


def append_candidate_reference(existing_path: Path, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(existing_path)
    for col in candidate_rows.columns:
        if col not in existing.columns:
            existing[col] = ""
    for col in existing.columns:
        if col not in candidate_rows.columns:
            candidate_rows[col] = ""
    return pd.concat([existing, candidate_rows[existing.columns]], ignore_index=True)


def main() -> None:
    OUT_SPECTRA.mkdir(parents=True, exist_ok=True)
    OUT_METADATA.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, str]] = []
    for product in PRODUCTS:
        rows, prov = extract_region_spectra(product)
        all_rows.extend(rows)
        provenance_rows.extend(prov)

    provenance_rows.extend(provenance_only_rows())

    candidate_metadata = pd.DataFrame(all_rows)
    provenance = pd.DataFrame(provenance_rows)
    candidate_path = OUT_METADATA / "metadata_sherloc_montpezat_alfalfa_weak_candidates.csv"
    provenance_path = OUT_METADATA / "sherloc_montpezat_alfalfa_pds_products.csv"
    combined_reference_path = (
        OUT_METADATA
        / "metadata_parent_945_plus_sherloc_regions_with_montpezat_alfalfa_candidates.csv"
    )

    candidate_metadata.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    provenance.to_csv(provenance_path, index=False, encoding="utf-8-sig")

    existing_path = OUT_METADATA / "metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv"
    if existing_path.exists():
        combined = append_candidate_reference(existing_path, candidate_metadata)
        combined.to_csv(combined_reference_path, index=False, encoding="utf-8-sig")

    summary = (
        candidate_metadata.groupby(["sherloc_target", "sherloc_scan_name", "sherloc_label_status"])
        .size()
        .reset_index(name="n_candidate_region_spectra")
    )
    summary_path = OUT_METADATA / "sherloc_montpezat_alfalfa_candidate_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    readme_path = OUT_METADATA / "sherloc_montpezat_alfalfa_candidate_readme.md"
    readme_path.write_text(
        "# Montpezat and Alfalfa SHERLOC candidate spectra\n\n"
        "This table adds PDS-traceable candidate spectra for Montpezat and Alfalfa. "
        "They are not included in the default closed-set fine-tuning table because "
        "the available publication text provides scan-level mineral detections, not "
        "a point-by-point label workbook. Rows are marked "
        "`sherloc_training_label_usable=False` and should be used only for "
        "exploratory weak-label or domain-adaptation analyses until point-level "
        "labels are curated.\n\n"
        "Primary sources: NASA PDS Mars 2020 SHERLOC bundle "
        "(DOI 10.17189/1522643) and Corpolongo et al. (2023), "
        "JGR: Planets, e2022JE007455.\n",
        encoding="utf-8",
    )

    print(f"Candidate weak-label spectra: {len(candidate_metadata)}")
    print(f"Product provenance rows: {len(provenance)}")
    print(f"Wrote: {candidate_path}")
    print(f"Wrote: {provenance_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {readme_path}")
    if existing_path.exists():
        print(f"Wrote: {combined_reference_path}")


if __name__ == "__main__":
    main()
