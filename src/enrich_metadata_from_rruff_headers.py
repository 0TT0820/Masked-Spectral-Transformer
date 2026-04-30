from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "metadata" / "metadata_parent_945.csv"
DEFAULT_OUT = ROOT / "data" / "metadata" / "metadata_parent_945_rruff_enriched.csv"
DEFAULT_RRUFF_TABLE = ROOT / "data" / "metadata" / "rruff_official_header_metadata.csv"
SEARCH_DIRS = [
    ROOT / "data" / "spectra" / "parent",
    ROOT / "source_rruff_header",
]


HEADER_MAP = {
    "NAMES": "rruff_official_name",
    "RRUFFID": "rruff_official_id",
    "IDEAL CHEMISTRY": "rruff_ideal_chemistry",
    "LOCALITY": "rruff_locality",
    "OWNER": "rruff_owner",
    "SOURCE": "rruff_source",
    "DESCRIPTION": "rruff_description",
    "STATUS": "rruff_status",
    "URL": "rruff_url",
    "MEASURED CHEMISTRY": "rruff_measured_chemistry",
}


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def normalize_formula(value: str) -> str:
    value = compact(value)
    value = value.replace("_", "")
    value = value.replace("ІВ=", "Σ=")
    return value


def parse_rruff_txt(path: Path) -> dict[str, str]:
    parsed = {"rruff_header_file": str(path)}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("##"):
                continue
            if "=" not in line:
                continue
            key, value = line[2:].split("=", 1)
            key = key.strip().upper()
            out_key = HEADER_MAP.get(key)
            if not out_key:
                continue
            parsed[out_key] = compact(value)

    for key in ["rruff_ideal_chemistry", "rruff_measured_chemistry"]:
        if parsed.get(key):
            parsed[key] = normalize_formula(parsed[key])
    if parsed.get("rruff_url") and not parsed["rruff_url"].startswith("http"):
        parsed["rruff_url"] = "https://" + parsed["rruff_url"].lstrip("/")
    parsed["rruff_header_parse_status"] = "ok" if parsed.get("rruff_official_id") else "missing_rruff_id"
    return parsed


def build_header_index(search_dirs: list[Path]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_stem: dict[str, dict[str, str]] = {}
    by_id: dict[str, dict[str, str]] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*.txt"):
            record = parse_rruff_txt(path)
            if record.get("rruff_header_parse_status") != "ok":
                continue
            stem = path.stem.lower()
            rruff_id = record.get("rruff_official_id", "").strip()
            by_stem.setdefault(stem, record)
            if rruff_id:
                by_id.setdefault(rruff_id, record)
    return by_stem, by_id


def find_header(row: pd.Series, by_stem: dict[str, dict[str, str]], by_id: dict[str, dict[str, str]]) -> dict[str, str]:
    file_name_clean = str(row.get("file_name_clean", "")).strip().lower()
    if file_name_clean.endswith(".csv") or file_name_clean.endswith(".txt"):
        file_name_clean = Path(file_name_clean).stem
    source_id = str(row.get("source_id", "")).strip()
    if file_name_clean in by_stem:
        return dict(by_stem[file_name_clean])
    if source_id in by_id:
        return dict(by_id[source_id])
    return {"rruff_header_parse_status": "not_found"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge official RRUFF header metadata from downloaded .txt spectra into the parent metadata CSV."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rruff-table", type=Path, default=DEFAULT_RRUFF_TABLE)
    parser.add_argument("--update-source", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, encoding="utf-8-sig")
    existing_rruff_cols = [col for col in metadata.columns if col.startswith("rruff_")]
    if existing_rruff_cols:
        metadata = metadata.drop(columns=existing_rruff_cols)
    if "rruff_metadata_match" in metadata.columns:
        metadata = metadata.drop(columns=["rruff_metadata_match"])
    by_stem, by_id = build_header_index(SEARCH_DIRS)

    records = []
    for _, row in metadata.iterrows():
        if row.get("source_type") == "RRUFF database":
            record = find_header(row, by_stem, by_id)
        else:
            record = {"rruff_header_parse_status": "not_applicable"}
        records.append(record)
    header_df = pd.DataFrame(records)

    enriched = pd.concat([metadata.reset_index(drop=True), header_df.reset_index(drop=True)], axis=1)
    if "rruff_official_id" in enriched.columns:
        source_text = enriched["source_id"].astype(str)
        official_text = enriched["rruff_official_id"].fillna("").astype(str)
        enriched["rruff_metadata_match"] = [
            bool(source == official or (official and source.startswith(official + ".")))
            for source, official in zip(source_text, official_text)
        ]
        enriched["rruff_metadata_match_basis"] = "source_id_exact_or_decimal_variant_of_official_rruff_id"
    else:
        enriched["rruff_metadata_match"] = False
        enriched["rruff_metadata_match_basis"] = ""

    rruff_cols = [
        "source_id",
        "file_name_clean",
        "major_category",
        "subtype_label",
        "source_type",
        "excitation_nm",
        "data_level",
        "orientation",
        "rruff_official_id",
        "rruff_official_name",
        "rruff_ideal_chemistry",
        "rruff_measured_chemistry",
        "rruff_locality",
        "rruff_source",
        "rruff_owner",
        "rruff_description",
        "rruff_status",
        "rruff_url",
        "rruff_header_file",
        "rruff_header_parse_status",
        "rruff_metadata_match",
        "rruff_metadata_match_basis",
    ]
    for col in rruff_cols:
        if col not in enriched.columns:
            enriched[col] = ""
    rruff_table = enriched[enriched["source_type"].eq("RRUFF database")][rruff_cols].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False, encoding="utf-8-sig")
    rruff_table.to_csv(args.rruff_table, index=False, encoding="utf-8-sig")
    if args.update_source:
        enriched.to_csv(args.metadata, index=False, encoding="utf-8-sig")

    summary = {
        "metadata": str(args.metadata),
        "enriched_out": str(args.out),
        "rruff_table": str(args.rruff_table),
        "rruff_rows": int(enriched["source_type"].eq("RRUFF database").sum()),
        "rruff_headers_ok": int(
            (
                (enriched["source_type"].eq("RRUFF database"))
                & (enriched["rruff_header_parse_status"].eq("ok"))
            ).sum()
        ),
        "rruff_headers_not_found": int(
            (
                (enriched["source_type"].eq("RRUFF database"))
                & (enriched["rruff_header_parse_status"].eq("not_found"))
            ).sum()
        ),
        "unique_rruff_ids_with_headers": int(rruff_table["rruff_official_id"].dropna().nunique()),
    }
    (args.out.parent / "rruff_header_enrichment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
