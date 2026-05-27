"""Build a local catalog for the Berlanga et al. 2022 MLROD data source.

The script parses ODR search/result JSON saved from the public MLROD page,
extracts sample-level metadata and downloadable file ids, and compares MLROD
mineral labels with the current project label inventory.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MLROD_DIR = ROOT / "data" / "external_mlrod_berlanga_2022"
RECORD_DIR = MLROD_DIR / "odr_records"
OUT_DIR = MLROD_DIR / "catalog"
CURRENT_METADATA = ROOT / "data" / "metadata" / "metadata_training_database_v2_qc_filtered.csv"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_size_mb(text: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(Mb|Kb|Gb)", text, flags=re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "kb":
        return value / 1024.0
    if unit == "gb":
        return value * 1024.0
    return value


def infer_file_kind(filename: str, context: str) -> str:
    name = filename.lower()
    ctx = context.lower()
    if name.startswith("l_cwt") or "continuous wavelet" in ctx:
        return "cwt_processed"
    if "average" in ctx or name.startswith("average") or name.startswith("avg"):
        return "average_raman_spectrum"
    if "test" in name or "labeled test" in ctx:
        return "raw_labeled_test"
    if "train" in name or "training" in ctx:
        return "raw_training"
    if "xrd" in ctx or name.endswith(".xy") or "xrd" in name:
        return "xrd"
    if "xrf" in ctx or "xrf" in name:
        return "xrf"
    return "other"


def map_mlrod_to_project_class(mineral_name: str, mineral_group: str, sample_type: str) -> tuple[str, str]:
    name = mineral_name.lower()
    group = mineral_group.lower()
    if "mixture" in sample_type.lower() or "+" in mineral_name or "+" in mineral_group:
        return "mixture_or_rock", "Not directly used as single-label mineral-superclass training data"
    if "gabbro" in name or "granite" in name or "rock" in sample_type.lower():
        return "rock_test_set", "Useful as external robustness test, not single-mineral superclass label"
    if "albite" in name or "anorthite" in name:
        return "Tectosilicate", "Maps to feldspar/plagioclase-type tectosilicate superclass"
    if "microcline" in name:
        return "Tectosilicate", "Maps to K-feldspar/tectosilicate superclass"
    if "augite" in name or "enstatite" in name or "pyroxene" in group:
        return "Pyroxene", "Direct overlap with pyroxene superclass"
    if "forsterite" in name or "olivine" in group:
        return "Olivine", "Direct overlap with olivine superclass"
    if "calcite" in name or "carbonate" in group:
        return "Carbonate", "Direct overlap with carbonate superclass"
    if "gypsum" in name or "sulfate" in group:
        return "Sulfate", "Direct overlap with sulfate superclass"
    if "biotite" in name or "muscovite" in name or "mica" in group:
        return "Phyllosilicate", "Mica maps to the phyllosilicate/clay-related silicate group"
    if "hornblende" in name or "amphibole" in group:
        return "Other Silicate", "Amphibole is a chain silicate but was not a primary class in all current splits"
    if "quartz" in name or "silicate" in group:
        return "Silica Phase", "Quartz maps to silica phase"
    return "unmapped", "Manual review required"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_path = MLROD_DIR / "odr_rows.json"
    with rows_path.open("r", encoding="utf-8") as f:
        rows_json = json.load(f)

    sample_rows: list[dict[str, object]] = []
    for row in rows_json["data"]:
        project_class, overlap_note = map_mlrod_to_project_class(row[3], row[4], row[5])
        sample_rows.append(
            {
                "odr_datarecord_id": row[0],
                "odr_sort_index": row[1],
                "sample_id": row[2],
                "mineral_name": row[3],
                "mineral_group": row[4],
                "sample_type": row[5],
                "location": row[6],
                "number_of_spectra": int(row[7]),
                "data_set": row[8],
                "project_superclass_mapping": project_class,
                "overlap_note": overlap_note,
                "dataset_citation": "Berlanga, Williams, and Temiquel (2022), MLROD, DOI:10.48484/PWRB-R137",
                "paper_citation": "Berlanga et al. (2022), Earth and Space Science, DOI:10.1029/2021EA002125",
            }
        )

    write_csv(
        OUT_DIR / "mlrod_sample_catalog.csv",
        sample_rows,
        [
            "odr_datarecord_id",
            "odr_sort_index",
            "sample_id",
            "mineral_name",
            "mineral_group",
            "sample_type",
            "location",
            "number_of_spectra",
            "data_set",
            "project_superclass_mapping",
            "overlap_note",
            "dataset_citation",
            "paper_citation",
        ],
    )

    file_rows: list[dict[str, object]] = []
    file_pattern = re.compile(
        r"<div id=\\?\"File_(?P<file_id>\d+)\\?\".*?<a\s+class=\\?\"ODRFileDownload\\?\"\s+title=\\?\"(?P<title>[^\\\"]+)\\?\"\s+rel=\\?\"(?P<rel>\d+)\\?\"[^>]*>(?P<label>.*?)</a>",
        flags=re.S,
    )
    for sample in sample_rows:
        record_id = str(sample["odr_datarecord_id"])
        record_path = RECORD_DIR / f"record_{record_id}.json"
        if not record_path.exists():
            continue
        with record_path.open("r", encoding="utf-8") as f:
            record_json = json.load(f)
        record_html = record_json.get("d", {}).get("html", "")
        for m in file_pattern.finditer(record_html):
            start = max(0, m.start() - 1200)
            context = clean_text(record_html[start : m.start()])
            filename = html.unescape(m.group("title"))
            file_rows.append(
                {
                    "odr_datarecord_id": record_id,
                    "sample_id": sample["sample_id"],
                    "mineral_name": sample["mineral_name"],
                    "mineral_group": sample["mineral_group"],
                    "project_superclass_mapping": sample["project_superclass_mapping"],
                    "data_set": sample["data_set"],
                    "file_id": m.group("rel"),
                    "filename": filename,
                    "file_kind": infer_file_kind(filename, context),
                    "approx_size_mb": parse_size_mb(context),
                    "download_url": f"https://www.odr.io/view/downloadfile/{m.group('rel')}",
                    "context_excerpt": context[-500:],
                }
            )

        graph_files = re.findall(
            r"data_files\[\d+\]\s*=\s*\{\s*\\?\"url\\?\":\s*file_url\s*\+\s*file,\s*\\?\"legend\\?\":\s*\\?\"([^\\\"]+)\\?\".*?\\?\"file_id\\?\":\s*(\d+)",
            record_html,
            flags=re.S,
        )
        for legend, file_id in graph_files:
            if any(r["file_id"] == file_id for r in file_rows if r["odr_datarecord_id"] == record_id):
                continue
            file_rows.append(
                {
                    "odr_datarecord_id": record_id,
                    "sample_id": sample["sample_id"],
                    "mineral_name": sample["mineral_name"],
                    "mineral_group": sample["mineral_group"],
                    "project_superclass_mapping": sample["project_superclass_mapping"],
                    "data_set": sample["data_set"],
                    "file_id": file_id,
                    "filename": f"average_raman_spectrum_{legend}.csv",
                    "file_kind": "average_raman_spectrum",
                    "approx_size_mb": None,
                    "download_url": f"https://www.odr.io/view/downloadfile/{file_id}",
                    "context_excerpt": "ODR graph plugin file for Average Raman Spectrum",
                }
            )

    write_csv(
        OUT_DIR / "mlrod_file_catalog.csv",
        file_rows,
        [
            "odr_datarecord_id",
            "sample_id",
            "mineral_name",
            "mineral_group",
            "project_superclass_mapping",
            "data_set",
            "file_id",
            "filename",
            "file_kind",
            "approx_size_mb",
            "download_url",
            "context_excerpt",
        ],
    )

    current_rows = read_csv_dicts(CURRENT_METADATA)
    current_species = {
        (r.get("mineral_species_final") or r.get("mineral_species") or "").strip().lower()
        for r in current_rows
        if (r.get("supervised_label_usable_v2") or "").strip().lower() == "true"
    }
    current_classes = Counter(
        (r.get("label_category_final") or r.get("major_category") or "").strip()
        for r in current_rows
        if (r.get("supervised_label_usable_v2") or "").strip().lower() == "true"
    )

    overlap_rows: list[dict[str, object]] = []
    for sample in sample_rows:
        mineral_tokens = [x.strip().lower() for x in re.split(r"\+|/", str(sample["mineral_name"])) if x.strip()]
        exact_species_overlap = [x for x in mineral_tokens if x in current_species]
        project_class = str(sample["project_superclass_mapping"])
        class_count = current_classes.get(project_class, 0)
        overlap_rows.append(
            {
                "mlrod_sample_id": sample["sample_id"],
                "mlrod_mineral_name": sample["mineral_name"],
                "mlrod_mineral_group": sample["mineral_group"],
                "mlrod_type": sample["sample_type"],
                "mlrod_number_of_spectra": sample["number_of_spectra"],
                "mapped_project_superclass": project_class,
                "current_project_class_count": class_count,
                "exact_species_overlap_in_current_qc_dataset": "; ".join(exact_species_overlap),
                "recommended_use": (
                    "candidate_training_or_external_validation"
                    if project_class not in {"mixture_or_rock", "rock_test_set", "unmapped"}
                    else "external_robustness_only"
                ),
                "note": sample["overlap_note"],
            }
        )

    write_csv(
        OUT_DIR / "mlrod_overlap_with_current_project.csv",
        overlap_rows,
        [
            "mlrod_sample_id",
            "mlrod_mineral_name",
            "mlrod_mineral_group",
            "mlrod_type",
            "mlrod_number_of_spectra",
            "mapped_project_superclass",
            "current_project_class_count",
            "exact_species_overlap_in_current_qc_dataset",
            "recommended_use",
            "note",
        ],
    )

    summary_rows = []
    by_class = defaultdict(lambda: {"samples": 0, "spectra": 0, "files": 0, "raw_files": 0})
    for sample in sample_rows:
        item = by_class[str(sample["project_superclass_mapping"])]
        item["samples"] += 1
        item["spectra"] += int(sample["number_of_spectra"])
    for file_row in file_rows:
        item = by_class[str(file_row["project_superclass_mapping"])]
        item["files"] += 1
        if str(file_row["file_kind"]).startswith("raw"):
            item["raw_files"] += 1
    for klass, counts in sorted(by_class.items()):
        summary_rows.append({"mapped_project_superclass": klass, **counts})
    write_csv(
        OUT_DIR / "mlrod_overlap_summary_by_project_class.csv",
        summary_rows,
        ["mapped_project_superclass", "samples", "spectra", "files", "raw_files"],
    )

    try:
        import pandas as pd

        xlsx_path = OUT_DIR / "mlrod_catalog_and_overlap.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(sample_rows).to_excel(writer, sheet_name="sample_catalog", index=False)
            pd.DataFrame(file_rows).to_excel(writer, sheet_name="file_catalog", index=False)
            pd.DataFrame(overlap_rows).to_excel(writer, sheet_name="overlap_with_project", index=False)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_by_class", index=False)
    except Exception as exc:  # pragma: no cover
        print(f"Workbook export skipped: {exc}")

    total_spectra = sum(int(r["number_of_spectra"]) for r in sample_rows)
    print(f"MLROD samples: {len(sample_rows)}")
    print(f"MLROD spectra reported by ODR: {total_spectra}")
    print(f"MLROD downloadable files catalogued: {len(file_rows)}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
