from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "metadata"
PARENT_FILE = METADATA_DIR / "metadata_parent_945.csv"
OUT_DIR = ROOT / "data" / "overview" / "review_data_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_REVIEW_CLASS = 200

RRUFF_URL = "https://www.rruff.net/about/"
RRUFF_RAMAN_URL = "https://rruff.info/about/about_raman.php"
PDS_SHERLOC_URL = "https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm"
HOLLIS_2022_URL = "https://doi.org/10.1016/j.icarus.2022.115179"
WANG_2024_URL = "https://doi.org/10.1016/j.vibspec.2024.103745"

REFERENCE_BY_SOURCE = {
    "RRUFF database": (
        "RRUFF Project; Lafuente et al. (2015), The power of databases: the RRUFF project",
        RRUFF_URL,
    ),
    "Lab-acquired DUV spectra": (
        "This study; laboratory DUV Raman spectra acquired from commercial/analogue mineral samples",
        "",
    ),
    "Martian meteorite spectra": (
        "This study; laboratory DUV Raman spectra from Tissint and NWA 10153 polished meteorite sections",
        "",
    ),
    "SHERLOC in-situ spectra": (
        "NASA PDS Mars 2020 SHERLOC archive; Razzell Hollis et al. (2022); Wang et al. (2024)",
        PDS_SHERLOC_URL,
    ),
}

METEORITE_SPOT_MAP = {
    "kuang": {
        "phase": "Ilmenite",
        "meteorite_sample": "Tissint or NWA 10153",
        "sample_assignment_status": "manual_confirmation_required",
        "note": "Opaque/oxide phase spectrum; assign exact meteorite section and spot ID from TIMA/BSE map notebook.",
    },
    "kuang2": {
        "phase": "Ilmenite",
        "meteorite_sample": "Tissint or NWA 10153",
        "sample_assignment_status": "manual_confirmation_required",
        "note": "Second opaque/oxide phase spectrum; assign exact meteorite section and spot ID from TIMA/BSE map notebook.",
    },
    "py": {
        "phase": "Augite/pyroxene",
        "meteorite_sample": "Tissint or NWA 10153",
        "sample_assignment_status": "manual_confirmation_required",
        "note": "Pyroxene spectrum; exact meteorite source must be confirmed from raw acquisition log.",
    },
    "pl-g": {
        "phase": "Basaltic glass or maskelynite-related feldspathic glass",
        "meteorite_sample": "Tissint or NWA 10153",
        "sample_assignment_status": "manual_confirmation_required",
        "note": "Feldspathic glass spectrum; exact meteorite source must be confirmed from raw acquisition log.",
    },
}


def review_label(category: str) -> str:
    category = str(category).strip()
    if category in {"Clay", "Mica", "Serpentine"}:
        return "Phyllosilicates"
    return category


def review_status(row: pd.Series) -> str:
    if row["source_type"] == "SHERLOC in-situ spectra":
        return "external_sherloc"
    if row["major_category"] == "Halides":
        return "excluded_halide_review"
    return "included_review_ready"


def extract_sherloc_product_group(source_id: str) -> str:
    text = str(source_id)
    match = re.match(r"(ss__\d{4}_[^_]+_[^_]+__[^_]+)", text)
    if match:
        return match.group(1)
    return text.split("_Region", 1)[0]


def sherloc_protocol_role(product_group: str) -> str:
    if "0186" in product_group:
        return "candidate_transfer_finetune_group"
    if "0304" in product_group:
        return "candidate_final_validation_group"
    return "external_validation_unassigned"


def source_reference_name(source_type: str) -> str:
    return REFERENCE_BY_SOURCE.get(source_type, ("Manual review required", ""))[0]


def source_reference_url(source_type: str) -> str:
    return REFERENCE_BY_SOURCE.get(source_type, ("", ""))[1]


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    parent = pd.read_csv(PARENT_FILE, encoding="utf-8-sig")
    parent["review_label"] = parent["major_category"].map(review_label)
    parent["review_status"] = parent.apply(review_status, axis=1)
    parent["source_reference_name"] = parent["source_type"].map(source_reference_name)
    parent["source_reference_url"] = parent["source_type"].map(source_reference_url)
    parent["sherloc_product_group"] = ""
    sherloc_mask = parent["source_type"].eq("SHERLOC in-situ spectra")
    parent.loc[sherloc_mask, "sherloc_product_group"] = parent.loc[sherloc_mask, "source_id"].map(
        extract_sherloc_product_group
    )
    parent["recommended_zero_shot_role"] = parent["split_main"].replace(
        {
            "train": "earth_train_parent",
            "val": "earth_validation_parent",
            "test": "earth_test_parent",
            "external_sherloc": "external_sherloc_final_validation",
        }
    )
    parent["candidate_transfer_role"] = parent["recommended_zero_shot_role"]
    parent.loc[sherloc_mask, "candidate_transfer_role"] = parent.loc[
        sherloc_mask, "sherloc_product_group"
    ].map(sherloc_protocol_role)
    parent["protocol_note"] = ""
    parent.loc[sherloc_mask, "protocol_note"] = (
        "Use as external validation in the primary zero-shot protocol. "
        "If a SHERLOC fine-tuning experiment is retained, report product-group split explicitly."
    )

    for stem, info in METEORITE_SPOT_MAP.items():
        mask = parent["file_name_clean"].astype(str).str.lower().eq(stem)
        for key, value in info.items():
            parent.loc[mask, key] = value

    curated_cols = [
        "spectrum_id",
        "file_name_clean",
        "major_category",
        "review_label",
        "subtype_label",
        "review_status",
        "source_type",
        "source_id",
        "source_reference_name",
        "source_reference_url",
        "excitation_nm",
        "instrument",
        "data_level",
        "sample_provenance",
        "meteorite_sample",
        "sample_assignment_status",
        "phase",
        "sherloc_product_group",
        "recommended_zero_shot_role",
        "candidate_transfer_role",
        "split_main",
        "split_zero_shot_protocol",
        "qc_status",
        "qc_reason",
        "file_sha256",
        "file_path",
    ]
    for col in curated_cols:
        if col not in parent.columns:
            parent[col] = ""
    write_csv(parent[curated_cols], "spectrum_level_provenance_review.csv")

    source_split = (
        parent.groupby(["source_type", "review_status", "split_main", "review_label"], dropna=False)
        .size()
        .reset_index(name="spectrum_count")
        .sort_values(["source_type", "split_main", "review_label"])
    )
    write_csv(source_split, "source_split_class_matrix.csv")

    raw_by_source_category = (
        parent.groupby(["source_type", "major_category"], dropna=False)
        .size()
        .reset_index(name="raw_parent_count")
        .sort_values(["source_type", "major_category"])
    )
    write_csv(raw_by_source_category, "raw_parent_by_source_and_category.csv")

    included = parent[parent["review_status"].eq("included_review_ready")].copy()
    train = included[included["split_main"].eq("train")].copy()
    val = included[included["split_main"].eq("val")].copy()
    test = included[included["split_main"].eq("test")].copy()
    external = parent[parent["review_status"].eq("external_sherloc")].copy()

    labels = sorted(set(included["review_label"]) | set(external["review_label"]))
    flow_rows = []
    for label in labels:
        train_count = int((train["review_label"] == label).sum())
        val_count = int((val["review_label"] == label).sum())
        test_count = int((test["review_label"] == label).sum())
        external_count = int((external["review_label"] == label).sum())
        generated = max(0, TARGET_PER_REVIEW_CLASS - train_count) if train_count > 0 else 0
        effective_train = train_count + generated
        flow_rows.append(
            {
                "review_label": label,
                "raw_parent_total": int((parent["review_label"] == label).sum()),
                "included_parent_total": int((included["review_label"] == label).sum()),
                "earth_train_parent": train_count,
                "earth_validation_parent": val_count,
                "earth_test_parent": test_count,
                "external_sherloc_parent": external_count,
                "augmentation_target_per_class": TARGET_PER_REVIEW_CLASS if train_count > 0 else 0,
                "generated_train_spectra_required": generated,
                "effective_train_after_reproducible_augmentation": effective_train,
                "augmentation_policy": (
                    "augment_train_only; preserve Raman band positions; perturb intensity, baseline, noise, and band width"
                    if generated
                    else "no augmentation required or no earth-train parent spectra"
                ),
            }
        )
    flow = pd.DataFrame(flow_rows).sort_values("review_label")
    write_csv(flow, "dataset_flow_by_review_class.csv")

    excluded = parent[parent["review_status"].eq("excluded_halide_review")]
    excluded_summary = (
        excluded.groupby(["major_category", "subtype_label", "source_type"], dropna=False)
        .size()
        .reset_index(name="excluded_parent_count")
        .sort_values(["major_category", "subtype_label", "source_type"])
    )
    write_csv(excluded_summary, "excluded_halide_inventory.csv")

    sherloc_summary = (
        external.groupby(["sherloc_product_group", "candidate_transfer_role", "review_label", "subtype_label"], dropna=False)
        .size()
        .reset_index(name="spectrum_count")
        .sort_values(["sherloc_product_group", "review_label", "subtype_label"])
    )
    sherloc_summary["target_name"] = "resolve_from_PDS_product_label"
    sherloc_summary["literature_context"] = (
        "Bellegarde/Dourbes context from Razzell Hollis et al. (2022); "
        "Quartier/Bellegarde sulfate context from Wang et al. (2024). "
        "Assign exact target after checking PDS product label."
    )
    write_csv(sherloc_summary, "sherloc_product_group_summary.csv")

    meteorite = parent[parent["source_type"].eq("Martian meteorite spectra")].copy()
    meteorite_cols = [
        "spectrum_id",
        "file_name_clean",
        "major_category",
        "subtype_label",
        "phase",
        "meteorite_sample",
        "sample_assignment_status",
        "note",
        "source_id",
        "file_sha256",
    ]
    for col in meteorite_cols:
        if col not in meteorite.columns:
            meteorite[col] = ""
    write_csv(meteorite[meteorite_cols], "meteorite_spot_inventory.csv")

    stage_rows = [
        {"stage": "all_parent_spectra", "spectrum_count": int(len(parent)), "description": "All raw parent spectra before review-ready filtering."},
        {"stage": "rruff_parent_spectra", "spectrum_count": int((parent["source_type"] == "RRUFF database").sum()), "description": "Visible/NIR Raman spectra from RRUFF; source IDs retained."},
        {"stage": "lab_duv_parent_spectra", "spectrum_count": int((parent["source_type"] == "Lab-acquired DUV spectra").sum()), "description": "Laboratory DUV spectra from analogue/commercial minerals."},
        {"stage": "meteorite_parent_spectra", "spectrum_count": int((parent["source_type"] == "Martian meteorite spectra").sum()), "description": "Laboratory DUV spectra from Tissint/NWA 10153; exact spot-level sample assignment must be confirmed."},
        {"stage": "sherloc_external_spectra", "spectrum_count": int(len(external)), "description": "Mars in-situ SHERLOC spectra; kept out of Earth-domain training for primary protocol."},
        {"stage": "excluded_halide_parent_spectra", "spectrum_count": int(len(excluded)), "description": "Halide spectra excluded from review-ready supervised classification because pure halides are weak/non-Raman-active and require separate treatment."},
        {"stage": "review_ready_earth_train_parent", "spectrum_count": int(len(train)), "description": "Parent spectra used for Earth-domain model training before augmentation."},
        {"stage": "review_ready_earth_validation_parent", "spectrum_count": int(len(val)), "description": "Parent spectra used for validation before augmentation."},
        {"stage": "review_ready_earth_test_parent", "spectrum_count": int(len(test)), "description": "Parent spectra used for final Earth-domain test before SHERLOC external evaluation."},
        {"stage": "review_ready_effective_train_after_augmentation", "spectrum_count": int(flow["effective_train_after_reproducible_augmentation"].sum()), "description": "Effective train set if every included class with train parents is augmented to the target count."},
    ]
    stage_summary = pd.DataFrame(stage_rows)
    write_csv(stage_summary, "dataset_stage_summary.csv")

    workbook = OUT_DIR / "review_data_inventory_tables.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        stage_summary.to_excel(writer, sheet_name="stage_summary", index=False)
        flow.to_excel(writer, sheet_name="class_flow", index=False)
        source_split.to_excel(writer, sheet_name="source_split_class", index=False)
        raw_by_source_category.to_excel(writer, sheet_name="raw_source_category", index=False)
        sherloc_summary.to_excel(writer, sheet_name="sherloc_groups", index=False)
        meteorite[meteorite_cols].to_excel(writer, sheet_name="meteorite_spots", index=False)
        excluded_summary.to_excel(writer, sheet_name="excluded_halides", index=False)

    report = OUT_DIR / "data_transparency_report.md"
    report.write_text(
        f"""# Review Data Transparency Report

Generated from `{PARENT_FILE.relative_to(ROOT)}`.

## Primary Recommendation

Use a clean zero-shot/transfer-separated protocol in the revised manuscript:

1. Earth-domain training, validation, and test use only RRUFF, laboratory DUV analogue spectra, and confirmed meteorite spectra.
2. SHERLOC in-situ spectra are kept out of the primary training set and reported as external Mars-domain validation.
3. If a SHERLOC fine-tuning experiment is retained, it must be reported as a separate transfer-learning experiment using the product-group split in `sherloc_product_group_summary.csv`.
4. Halides are excluded from the review-ready supervised classification table because the reviewer is correct that pure crystalline halite is weak/non-Raman-active under standard Raman conditions. Halide/chloride detection should be discussed as a separate operational/non-detection or uncertainty problem.

## Dataset Counts

- Parent spectra: {len(parent)}
- RRUFF parent spectra: {(parent["source_type"] == "RRUFF database").sum()}
- Laboratory DUV parent spectra: {(parent["source_type"] == "Lab-acquired DUV spectra").sum()}
- Martian meteorite parent spectra: {(parent["source_type"] == "Martian meteorite spectra").sum()}
- SHERLOC external spectra: {len(external)}
- Review-ready Earth train parent spectra: {len(train)}
- Review-ready Earth validation parent spectra: {len(val)}
- Review-ready Earth test parent spectra: {len(test)}
- Effective train spectra after reproducible class balancing to {TARGET_PER_REVIEW_CLASS} per class: {int(flow["effective_train_after_reproducible_augmentation"].sum())}

## Files

- `spectrum_level_provenance_review.csv`: one row per parent spectrum with source, role, QC, hash, and reference fields.
- `dataset_flow_by_review_class.csv`: raw parent counts, split counts, external SHERLOC counts, and reproducible augmentation counts by class.
- `source_split_class_matrix.csv`: source x split x class matrix.
- `sherloc_product_group_summary.csv`: SHERLOC product groups and candidate transfer roles.
- `meteorite_spot_inventory.csv`: Tissint/NWA 10153 spectra requiring final spot-level assignment.
- `excluded_halide_inventory.csv`: halide spectra removed from the review-ready classification protocol.

## External Source Notes

- RRUFF: the RRUFF project provides mineral Raman/IR/XRD/chemistry data and asks users to cite the RRUFF Project and external contributors where applicable. See {RRUFF_URL}.
- RRUFF Raman limitations: RRUFF Raman data include visible red/green laser excitation; fluorescence behaviour is wavelength dependent, so DUV transfer must be explicitly described. See {RRUFF_RAMAN_URL}.
- SHERLOC archive: NASA PDS provides raw, intermediate, and processed spectroscopy collections for Mars 2020 SHERLOC. See {PDS_SHERLOC_URL}.
- Bellegarde/Dourbes context: Razzell Hollis et al. (2022), Icarus, DOI {HOLLIS_2022_URL}.
- Quartier/Bellegarde sulfate context: Wang et al. (2024), Vibrational Spectroscopy, DOI {WANG_2024_URL}.

## Manual Items Still Required

- Assign each meteorite spectrum (`kuang`, `kuang2`, `py`, `pl-g`) to Tissint or NWA 10153 using the original acquisition log or TIMA/BSE spot map.
- Resolve exact SHERLOC target names for each `ss__...` product group using PDS labels or Analyst's Notebook metadata.
- Complete supplier, lot, and preparation metadata for the laboratory DUV analogue minerals.
""",
        encoding="utf-8",
    )

    summary = {
        "parent_spectra": int(len(parent)),
        "source_counts": parent["source_type"].value_counts().to_dict(),
        "review_status_counts": parent["review_status"].value_counts().to_dict(),
        "stage_summary_file": str((OUT_DIR / "dataset_stage_summary.csv").relative_to(ROOT)),
        "class_flow_file": str((OUT_DIR / "dataset_flow_by_review_class.csv").relative_to(ROOT)),
        "workbook": str(workbook.relative_to(ROOT)),
    }
    (OUT_DIR / "review_data_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
