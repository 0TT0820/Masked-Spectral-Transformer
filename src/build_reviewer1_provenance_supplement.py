"""Build reviewer-facing spectrum provenance supplementary tables.

The provenance workbook is generated from the final v3 compact metadata used in
the revised experiments. It is intentionally conservative: public records,
sample identifiers, and instrument fields are copied from the curated metadata,
while unavailable fields are marked with a controlled phrase rather than being
invented.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata" / "metadata_training_database_v3_compact.csv"
FINAL_RUN_DIR = ROOT / "results" / "band_aware_mlrod_v3" / "band_aware_mlrod_v3_grid0-4000_n4001_20260527_192236"
FINAL_AUGMENTATION = FINAL_RUN_DIR / "shared_training_augmentation_summary.csv"
OUT_DIR = ROOT / "data" / "metadata" / "reviewer1_provenance_supplement"
DOCS_DIR = ROOT / "docs"

CONTROLLED_MISSING = "not reported in source record"


def read_current_metadata() -> pd.DataFrame:
    if not METADATA.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA}")
    return pd.read_csv(METADATA, low_memory=False)


def as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def first_existing(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return as_text(df[col])
    return pd.Series([default] * len(df), index=df.index)


def coalesce_text(*values: object, default: str = CONTROLLED_MISSING) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "na", "<na>"}:
            return text
    return default


def infer_source_domain(source_type: str) -> str:
    if "MLROD" in source_type:
        return "public Raman reference domain"
    if "RRUFF" in source_type:
        return "public visible/NIR Raman reference domain"
    if "Lab-acquired DUV" in source_type:
        return "laboratory DUV reference domain"
    if "SHERLOC in-situ" in source_type:
        return "Mars 2020 SHERLOC in-situ DUV domain"
    if "SHERLOC calibration" in source_type:
        return "Mars 2020 SHERLOC calibration DUV domain"
    if "meteorite" in source_type:
        return "extraterrestrial meteorite Raman reference domain"
    return "source domain not classified"


def source_url_or_product(row: pd.Series) -> str:
    st = str(row.get("source_type_normalized", row.get("source_type", "")))
    if "RRUFF" in st:
        return coalesce_text(row.get("rruff_url", ""))
    if "MLROD" in st:
        return "Berlanga et al. (2022), https://doi.org/10.1029/2021EA002125; MLROD dataset record https://doi.org/10.48484/PWRB-R137"
    if "SHERLOC" in st:
        return coalesce_text(
            row.get("source_id", ""),
            row.get("sherloc_scan_name", ""),
            default="NASA PDS Mars 2020 SHERLOC product-level record; target/region/point identifiers are listed in this row",
        )
    if "Lab-acquired DUV" in st:
        return coalesce_text(row.get("source_id", ""), default="author-measured DUV spectrum; sample and instrument fields are listed in this row")
    if "meteorite" in st:
        return coalesce_text(row.get("source_id", ""), default="meteorite Raman reference record listed in source metadata")
    return CONTROLLED_MISSING


def provenance_confidence(row: pd.Series) -> str:
    st = str(row.get("source_type_normalized", row.get("source_type", "")))
    qc_status = str(row.get("qc_status", ""))
    if "MLROD" in st:
        return "public_dataset_record"
    if "RRUFF" in st and coalesce_text(row.get("rruff_official_id", ""), default=""):
        return "verified_public_database_record" if "review_required" not in qc_status else "public_database_record_qc_flagged"
    if "SHERLOC in-situ" in st:
        return "mission_in_situ_record_with_label_crosswalk"
    if "SHERLOC calibration" in st:
        return "mission_calibration_record_no_closed_set_label"
    if "Lab-acquired DUV" in st:
        return "author_measured_record"
    if "meteorite" in st:
        return "meteorite_reference_record"
    return "provenance_not_classified"


def supervised_use(row: pd.Series) -> str:
    split = str(row.get("split_v3", ""))
    st = str(row.get("source_type_normalized", row.get("source_type", "")))
    action = str(row.get("recommended_action", ""))
    if split in {"train", "val", "test", "mlrod_train", "mlrod_val", "mlrod_test"}:
        return "included_in_reference_domain_benchmark"
    if split == "sherloc_finetune_pool":
        return "sherloc_domain_fine_tuning_pool_not_augmented"
    if split == "sherloc_external_validation":
        return "sherloc_domain_validation_only_not_augmented"
    if split == "domain_adaptation_only":
        return "domain_reference_only_not_closed_set_training"
    if "exclude" in action.lower():
        return "metadata_only_excluded_from_closed_set_training"
    if "SHERLOC" in st:
        return "sherloc_domain_record_review_before_supervised_use"
    return "metadata_only_review_before_supervised_use"


def build_inventory(df: pd.DataFrame) -> pd.DataFrame:
    source_type = first_existing(df, ["source_type_normalized", "source_type"]).replace("", CONTROLLED_MISSING)
    source_domain = source_type.map(infer_source_domain)
    inventory = pd.DataFrame(
        {
            "spectrum_id": first_existing(df, ["spectrum_id", "file_name"]).replace("", CONTROLLED_MISSING),
            "file_name": first_existing(df, ["file_name"]).replace("", CONTROLLED_MISSING),
            "storage_format": first_existing(df, ["spectrum_storage_format"]).replace("", CONTROLLED_MISSING),
            "mineral_superclass": first_existing(df, ["label_category_final", "model_label", "major_category"]).replace("", CONTROLLED_MISSING),
            "mineral_species": first_existing(df, ["mineral_species_final", "mineral_species", "subtype_label"]).replace("", CONTROLLED_MISSING),
            "source_type": source_type,
            "source_domain": source_domain,
            "training_role": first_existing(df, ["training_role"]).replace("", CONTROLLED_MISSING),
            "split_v3": first_existing(df, ["split_v3"]).replace("", CONTROLLED_MISSING),
            "zero_shot_protocol_role": first_existing(df, ["split_zero_shot_protocol"]).replace("", CONTROLLED_MISSING),
            "source_identifier": first_existing(df, ["source_id", "rruff_official_id", "mlrod_odr_datarecord_id"]).replace("", CONTROLLED_MISSING),
            "instrument_or_database": first_existing(df, ["instrument"]).replace("", CONTROLLED_MISSING),
            "excitation_nm": first_existing(df, ["excitation_nm"]).replace("", CONTROLLED_MISSING),
            "spectral_min_cm-1": first_existing(df, ["spectral_min_cm-1"]).replace("", CONTROLLED_MISSING),
            "spectral_max_cm-1": first_existing(df, ["spectral_max_cm-1"]).replace("", CONTROLLED_MISSING),
            "n_original_points": first_existing(df, ["n_original_points"]).replace("", CONTROLLED_MISSING),
            "qc_status": first_existing(df, ["qc_status"]).replace("", CONTROLLED_MISSING),
            "qc_reason": first_existing(df, ["qc_reason"]).replace("", CONTROLLED_MISSING),
            "recommended_action": first_existing(df, ["recommended_action"]).replace("", CONTROLLED_MISSING),
            "supervised_label_usable": first_existing(df, ["supervised_label_usable_v2"]).replace("", CONTROLLED_MISSING),
            "duv_library_include": first_existing(df, ["duv_library_include"]).replace("", CONTROLLED_MISSING),
            "rruff_official_id": first_existing(df, ["rruff_official_id"]).replace("", CONTROLLED_MISSING),
            "rruff_official_name": first_existing(df, ["rruff_official_name"]).replace("", CONTROLLED_MISSING),
            "rruff_url": first_existing(df, ["rruff_url"]).replace("", CONTROLLED_MISSING),
            "rruff_status": first_existing(df, ["rruff_status"]).replace("", CONTROLLED_MISSING),
            "mlrod_datarecord_id": first_existing(df, ["mlrod_odr_datarecord_id"]).replace("", CONTROLLED_MISSING),
            "mlrod_file_id": first_existing(df, ["mlrod_file_id"]).replace("", CONTROLLED_MISSING),
            "mlrod_sample_id": first_existing(df, ["mlrod_sample_id"]).replace("", CONTROLLED_MISSING),
            "mlrod_raw_filename": first_existing(df, ["mlrod_raw_filename"]).replace("", CONTROLLED_MISSING),
            "mlrod_row_index": first_existing(df, ["mlrod_row_index"]).replace("", CONTROLLED_MISSING),
            "sherloc_region": first_existing(df, ["sherloc_region"]).replace("", CONTROLLED_MISSING),
            "sherloc_target": first_existing(df, ["sherloc_target"]).replace("", CONTROLLED_MISSING),
            "sherloc_scan_name": first_existing(df, ["sherloc_scan_name"]).replace("", CONTROLLED_MISSING),
            "sherloc_point_name": first_existing(df, ["sherloc_point_name"]).replace("", CONTROLLED_MISSING),
            "source_url_or_product": [source_url_or_product(row) for _, row in df.iterrows()],
            "provenance_confidence": [provenance_confidence(row) for _, row in df.iterrows()],
            "supervised_use_recommendation": [supervised_use(row) for _, row in df.iterrows()],
        }
    )
    return inventory


def build_rruff(df: pd.DataFrame) -> pd.DataFrame:
    mask = first_existing(df, ["source_type_normalized", "source_type"]).str.contains("RRUFF", na=False)
    r = df[mask].copy()
    cols = [
        "spectrum_id",
        "file_name",
        "rruff_official_id",
        "rruff_official_name",
        "rruff_url",
        "excitation_nm",
        "rruff_status",
        "label_category_final",
        "split_v3",
        "qc_status",
        "qc_reason",
        "recommended_action",
    ]
    out = pd.DataFrame()
    for col in cols:
        out[col] = first_existing(r, [col]).replace("", CONTROLLED_MISSING)
    out["provenance_confidence"] = [provenance_confidence(row) for _, row in r.iterrows()]
    return out


def build_summaries(inventory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source_summary = (
        inventory.groupby(["source_type", "source_domain", "training_role", "split_v3"], dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["source_type", "training_role", "split_v3"])
    )
    source_by_class = pd.crosstab(inventory["source_type"], inventory["mineral_superclass"]).reset_index()
    qc_summary = (
        inventory.groupby(["source_type", "qc_status", "qc_reason", "supervised_use_recommendation"], dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["source_type", "qc_status", "n_spectra"], ascending=[True, True, False])
    )
    split_summary = (
        inventory.groupby(["split_v3", "source_type", "mineral_superclass"], dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["split_v3", "source_type", "mineral_superclass"])
    )
    field_dictionary = pd.DataFrame(
        [
            ("spectrum_id", "Unique spectrum identifier used in the revised metadata."),
            ("source_type", "Five retained source families plus MLROD public Raman spectra used in the final expanded benchmark."),
            ("source_domain", "Experimental domain inferred from the source type."),
            ("training_role", "Role assigned before splitting, e.g., reference benchmark, SHERLOC fine-tuning, or domain reference."),
            ("split_v3", "Final split used in the v3 experiments."),
            ("zero_shot_protocol_role", "Whether the spectrum belongs to the Earth/reference pool, SHERLOC candidate pool, external SHERLOC test, or MLROD external source."),
            ("source_url_or_product", "RRUFF URL, DOI/PDS/source identifier, or controlled missing phrase."),
            ("qc_status", "Final quality-control flag used in dataset construction."),
            ("recommended_action", "Final recommendation for inclusion, exclusion, or adaptation-only use."),
            ("provenance_confidence", "Controlled provenance-confidence label."),
            ("supervised_use_recommendation", "How the row is used in supervised training, SHERLOC fine-tuning, validation, or metadata-only records."),
        ],
        columns=["field", "description"],
    )
    return {
        "source_summary": source_summary,
        "source_by_class": source_by_class,
        "qc_summary": qc_summary,
        "split_summary": split_summary,
        "field_dictionary": field_dictionary,
    }


def build_augmentation_summary() -> pd.DataFrame:
    if not FINAL_AUGMENTATION.exists():
        return pd.DataFrame({"note": [f"Final augmentation summary not found: {FINAL_AUGMENTATION.relative_to(ROOT)}"]})
    aug = pd.read_csv(FINAL_AUGMENTATION)
    aug = aug.rename(columns={"class": "mineral_superclass"})
    aug["augmentation_scope"] = "reference-domain training split only"
    aug["validation_and_test_policy"] = "measured spectra only; no augmented validation or test spectra"
    aug["note"] = "Materialized Raman-aware augmentation used only to balance minority classes in training."
    return aug[
        [
            "mineral_superclass",
            "original_train_count",
            "augmented_count",
            "final_train_count",
            "augmentation_scope",
            "validation_and_test_policy",
            "note",
        ]
    ]


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = read_current_metadata()
    inventory = build_inventory(df)
    rruff = build_rruff(df)
    summaries = build_summaries(inventory)
    augmentation = build_augmentation_summary()

    tables = {
        "v3_P1_full_spectrum_provenance_inventory.csv": inventory,
        "v3_P2_source_task_summary.csv": summaries["source_summary"],
        "v3_P3_source_by_class.csv": summaries["source_by_class"],
        "v3_P4_rruff_official_metadata_qc.csv": rruff,
        "v3_P5_quality_control_summary.csv": summaries["qc_summary"],
        "v3_P6_split_by_source_and_class.csv": summaries["split_summary"],
        "v3_P7_training_only_augmentation_summary.csv": augmentation,
        "v3_P8_field_dictionary.csv": summaries["field_dictionary"],
    }
    for name, table in tables.items():
        table.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")

    workbook = OUT_DIR / "Reviewer1_Comment1_Spectrum_Provenance_Supplement_v3.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet_name, table in [
            ("P1_full_inventory", inventory),
            ("P2_source_task_summary", summaries["source_summary"]),
            ("P3_source_by_class", summaries["source_by_class"]),
            ("P4_rruff_metadata_qc", rruff),
            ("P5_qc_summary", summaries["qc_summary"]),
            ("P6_split_summary", summaries["split_summary"]),
            ("P7_augmentation_summary", augmentation),
            ("P8_field_dictionary", summaries["field_dictionary"]),
        ]:
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.book[sheet_name[:31]]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = min(60, max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells) + 2)
                ws.column_dimensions[col_cells[0].column_letter].width = max(12, max_len)

    readme = DOCS_DIR / "reviewer1_comment1_provenance_supplement.md"
    readme.write_text(
        "\n".join(
            [
                "# Reviewer 1 Comment 1: Spectrum Provenance Supplement",
                "",
                "This document describes the reviewer-facing provenance workbook generated from the final v3 compact metadata.",
                f"Metadata source: `{METADATA.relative_to(ROOT)}`.",
                "",
                "The workbook uses P1-P8 sheet labels to avoid conflict with the manuscript's numbered supporting-information tables.",
                "Unavailable fields are encoded as `not reported in source record`; no supplier names, product IDs, or measurement conditions are invented.",
                "",
                "Main output workbook:",
                f"- `{workbook.relative_to(ROOT)}`",
                "",
                "CSV exports:",
            ]
            + [f"- `{(OUT_DIR / name).relative_to(ROOT)}`" for name in tables]
            + [
                "",
                "Reviewer-facing use:",
                "- P1 is the complete per-spectrum provenance inventory.",
                "- P2-P3 summarize source roles and source-by-mineral-class counts.",
                "- P4 lists RRUFF official identifiers, URLs, excitation wavelength, status, and QC decisions.",
                "- P5-P6 summarize quality control and final train/validation/test/fine-tuning split composition.",
                "- P7 documents training-only materialized augmentation; validation and test spectra remain measured spectra.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote workbook: {workbook}")
    print(f"Wrote readme: {readme}")


if __name__ == "__main__":
    write_outputs()
