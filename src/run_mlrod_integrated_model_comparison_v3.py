"""MLROD-integrated Raman model comparison and SHERLOC fine-tuning.

This reviewer-oriented experiment merges the curated manuscript reference
database with the single-mineral raw Raman spectra from MLROD
(Berlanga et al., 2022; DOI:10.1029/2021EA002125; dataset DOI:
10.48484/PWRB-R137). The full MLROD inventory is kept in metadata, while this
script samples a reproducible, class-balanced subset for model training so that
the large external source does not dominate the smaller RRUFF, DUV, and
meteorite sources.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

import train_model_comparison as tmc
from train_model_comparison import (
    AUGMENTATION_PROTOCOL,
    GRID,
    PLSDA,
    RamanDataset,
    StandardTransformer,
    MaskedSpectralTransformer,
    augment_raman_features,
    class_weights,
    evaluate_arrays,
    fix_seed,
    flatten_features,
    preprocess_spectrum,
    predict_torch,
    remove_polynomial_baseline,
    run_asls,
    train_torch_model,
)
from run_model_selection import TunedCNN


ROOT = Path(__file__).resolve().parents[1]
METADATA_V3 = ROOT / "data" / "metadata" / "metadata_training_database_v3_mlrod_integrated.csv"
OUT_DIR = ROOT / "results" / "mlrod_integrated_training_v3"
SEED = 2024
EXCLUDED_LABELS = {"Halides"}
PHYLL_LABELS = {"Clay", "Mica", "Serpentine"}
MLROD_SPLIT_MAP = {"mlrod_train": "train", "mlrod_val": "val", "mlrod_test": "test"}
MLROD_FORMAT = "mlrod_wide_csv_row"
SYNTHETIC_METEORITE_FORMAT = "synthetic_meteorite_wide_csv_row"
WIDE_ROW_FORMATS = {MLROD_FORMAT, SYNTHETIC_METEORITE_FORMAT}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def configure_grid(grid_min: float, grid_max: float, grid_points: int) -> None:
    """Configure the shared Raman-shift grid used by imported preprocessing code."""

    global GRID
    GRID = np.linspace(float(grid_min), float(grid_max), int(grid_points), dtype=np.float32)
    tmc.GRID = GRID
    tmc.GRID_POINTS = int(grid_points)


def grid_tag() -> str:
    return f"grid{float(GRID[0]):.0f}-{float(GRID[-1]):.0f}_n{len(GRID)}"


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def harmonize_label(value: object) -> str:
    label = str(value).strip()
    if label in PHYLL_LABELS:
        return "Phyllosilicates"
    if label == "Other Silicate":
        return "Other Silicates"
    return label


def resolve_path(path_value: object) -> Path:
    text = str(path_value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return Path("")
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
    parts = list(path.parts)
    if "publication_repo" in parts:
        suffix = Path(*parts[parts.index("publication_repo") + 1 :])
        candidates.append(ROOT / suffix)
    if "data" in parts:
        suffix = Path(*parts[parts.index("data") :])
        candidates.append(ROOT / suffix)
    candidates.append(ROOT.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_v3_metadata(metadata_file: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_file, low_memory=False)
    df["model_label"] = df["label_category_final"].map(harmonize_label)
    df["resolved_file_path"] = df["file_path"].map(resolve_path).map(str)
    if "mlrod_container_file" in df.columns:
        df["mlrod_container_file"] = df["mlrod_container_file"].map(lambda p: str(resolve_path(p)) if str(p).strip() else "")
    df["resolved_file_exists"] = df["resolved_file_path"].map(lambda p: Path(p).exists())
    if "supervised_label_usable_v2" in df.columns:
        df["supervised_label_usable_v2"] = df["supervised_label_usable_v2"].map(as_bool)
    else:
        df["supervised_label_usable_v2"] = True
    df = df[df["resolved_file_exists"]].copy()
    df = df[df["model_label"].notna() & df["model_label"].astype(str).str.len().gt(0)].copy()
    df = df[~df["model_label"].isin(EXCLUDED_LABELS)].copy()
    return df.reset_index(drop=True)


def build_reference_and_mlrod_pool(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["split_v3"].isin(["train", "val", "test"]) & df["supervised_label_usable_v2"] & ~df["spectrum_storage_format"].eq(MLROD_FORMAT)].copy()
    base["split_model"] = base["split_v3"]

    mlrod = df[df["spectrum_storage_format"].eq(MLROD_FORMAT) & df["split_v3"].isin(MLROD_SPLIT_MAP)].copy()
    mlrod["split_model"] = mlrod["split_v3"].map(MLROD_SPLIT_MAP)

    pool = pd.concat([base, mlrod], ignore_index=True)
    pool["source_family"] = np.select(
        [
            pool["spectrum_storage_format"].eq(MLROD_FORMAT),
            pool["spectrum_storage_format"].eq(SYNTHETIC_METEORITE_FORMAT),
        ],
        ["MLROD", "synthetic_meteorite"],
        default="curated_reference",
    )
    pool["parent_group"] = pool["parent_group"].fillna(pool["spectrum_id"])
    pool["resolved_file_path"] = pool["resolved_file_path"].fillna(pool["file_path"])
    return pool.reset_index(drop=True)


def sample_mlrod_by_class(pool: pd.DataFrame, train_cap: int, val_cap: int, test_cap: int) -> pd.DataFrame:
    base = pool[~pool["source_family"].eq("MLROD")].copy()
    mlrod = pool[pool["source_family"].eq("MLROD")].copy()
    caps = {"train": train_cap, "val": val_cap, "test": test_cap}
    sampled_parts = [base]
    for split, cap in caps.items():
        sub = mlrod[mlrod["split_model"].eq(split)]
        for _, group in sub.groupby("model_label", sort=True):
            if cap is None or cap < 0 or len(group) <= cap:
                sampled_parts.append(group)
            else:
                sampled_parts.append(group.sample(n=cap, random_state=SEED))
    selected = pd.concat(sampled_parts, ignore_index=True)
    return selected.sort_values(["split_model", "model_label", "source_family", "spectrum_id"]).reset_index(drop=True)


def preprocess_vector(shift: np.ndarray, intensity: np.ndarray, baseline: str) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(shift) & np.isfinite(intensity)
    shift = shift[ok].astype(np.float64)
    intensity = intensity[ok].astype(np.float64)
    if len(shift) < 2:
        empty = np.zeros_like(GRID, dtype=np.float32)
        return np.stack([empty, empty, empty], axis=-1), np.ones_like(GRID, dtype=bool)

    unique_shift, unique_idx = np.unique(shift, return_index=True)
    intensity = intensity[unique_idx]
    order = np.argsort(unique_shift)
    unique_shift = unique_shift[order]
    intensity = intensity[order]

    valid_mask = (GRID >= float(np.min(unique_shift))) & (GRID <= float(np.max(unique_shift)))
    interp = np.interp(GRID, unique_shift, intensity, left=0.0, right=0.0).astype(np.float64)
    if baseline == "asls" and len(interp) > 3:
        interp = run_asls(interp)
    elif baseline == "poly":
        interp = remove_polynomial_baseline(interp, valid_mask)

    interp = np.maximum(interp, 0.0)
    maxv = float(np.max(interp[valid_mask])) if np.any(valid_mask) else float(np.max(interp))
    if maxv <= 0:
        maxv = 1.0
    norm_intensity = (interp / (maxv + 1e-12)).astype(np.float32)
    norm_intensity[~valid_mask] = 0.0
    deriv = np.gradient(norm_intensity, GRID).astype(np.float32)
    max_abs = float(np.max(np.abs(deriv[valid_mask]))) if np.any(valid_mask) else float(np.max(np.abs(deriv)))
    if max_abs > 1e-9:
        deriv = deriv / max_abs
    deriv[~valid_mask] = 0.0
    features = np.stack([norm_intensity, deriv, valid_mask.astype(np.float32)], axis=-1).astype(np.float32)
    return features, ~valid_mask


def load_mlrod_group(path: Path, row_indices: list[int], baseline: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    wave_cols = df.columns[1:-1]
    shifts = np.asarray([float(c) for c in wave_cols], dtype=np.float64)
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row_idx in row_indices:
        values = pd.to_numeric(df.iloc[int(row_idx), 1:-1], errors="coerce").to_numpy(dtype=np.float64)
        rows[int(row_idx)] = preprocess_vector(shifts, values, baseline)
    return rows


def parse_shift_prefixed_columns(columns: pd.Index | list[str], prefix: str = "shift_") -> tuple[list[str], np.ndarray]:
    shift_cols = []
    shifts = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        try:
            shifts.append(float(text.removeprefix(prefix)))
            shift_cols.append(text)
        except ValueError:
            continue
    order = np.argsort(np.asarray(shifts, dtype=np.float64))
    ordered_cols = [shift_cols[int(i)] for i in order]
    ordered_shifts = np.asarray([shifts[int(i)] for i in order], dtype=np.float64)
    return ordered_cols, ordered_shifts


def load_synthetic_meteorite_group(path: Path, row_indices: list[int], baseline: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    shift_cols, shifts = parse_shift_prefixed_columns(df.columns)
    if len(shift_cols) == 0:
        raise ValueError(f"No shift_* columns found in {path}")
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row_idx in row_indices:
        values = pd.to_numeric(df.loc[int(row_idx), shift_cols], errors="coerce").to_numpy(dtype=np.float64)
        rows[int(row_idx)] = preprocess_vector(shifts, values, baseline)
    return rows


def build_feature_arrays(samples: pd.DataFrame, cache_dir: Path, name: str, baseline: str, refresh: bool) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_{grid_tag()}_baseline-{baseline}_n{len(samples)}.npz"
    if cache_path.exists() and not refresh:
        cache = np.load(cache_path)
        return cache["features"], cache["masks"]

    features = np.zeros((len(samples), len(GRID), 3), dtype=np.float32)
    masks = np.ones((len(samples), len(GRID)), dtype=bool)

    normal = samples[~samples["spectrum_storage_format"].isin(WIDE_ROW_FORMATS)]
    total_normal = len(normal)
    for count, (idx, row) in enumerate(normal.iterrows(), start=1):
        if count == 1 or count % 100 == 0 or count == total_normal:
            log(f"Preprocessing two-column spectra for {name}: {count}/{total_normal}")
        x, _, mask = preprocess_spectrum(str(row["resolved_file_path"]), baseline=baseline, smooth=False)
        if str(row.get("source_domain", "")).startswith("sherloc"):
            low = GRID < 800.0
            x[low, :] = 0.0
            mask[low] = True
        features[idx] = x
        masks[idx] = mask

    synthetic = samples[samples["spectrum_storage_format"].eq(SYNTHETIC_METEORITE_FORMAT)]
    if len(synthetic) > 0:
        grouped = synthetic.groupby("resolved_file_path", sort=True)
        for group_i, (file_path, group) in enumerate(grouped, start=1):
            path = Path(str(file_path))
            row_indices = [int(float(v)) for v in group["wide_row_index"].tolist()]
            log(f"Preprocessing synthetic meteorite wide file {group_i}/{len(grouped)}: {path.name} ({len(row_indices)} selected rows)")
            processed = load_synthetic_meteorite_group(path, row_indices, baseline)
            for idx, row in group.iterrows():
                row_idx = int(float(row["wide_row_index"]))
                x, mask = processed[row_idx]
                features[idx] = x
                masks[idx] = mask

    mlrod = samples[samples["spectrum_storage_format"].eq(MLROD_FORMAT)]
    if len(mlrod) > 0:
        grouped = mlrod.groupby("mlrod_container_file", sort=True)
        for group_i, (file_path, group) in enumerate(grouped, start=1):
            path = Path(str(file_path))
            row_indices = [int(v) for v in group["mlrod_row_index"].tolist()]
            log(f"Preprocessing MLROD file {group_i}/{len(grouped)}: {path.name} ({len(row_indices)} selected rows)")
            processed = load_mlrod_group(path, row_indices, baseline)
            for idx, row in group.iterrows():
                row_idx = int(row["mlrod_row_index"])
                x, mask = processed[row_idx]
                features[idx] = x
                masks[idx] = mask

    np.savez_compressed(cache_path, features=features, masks=masks, grid=GRID.astype(np.float32))
    return features, masks


def make_balanced_augmented_train(
    x_train: np.ndarray,
    masks_train: np.ndarray,
    y_train: np.ndarray,
    min_per_class: int,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    xs = [x_train]
    ms = [masks_train]
    ys = [y_train]
    rows = []
    for cls in sorted(np.unique(y_train)):
        idx = np.where(y_train == cls)[0]
        target_n = min(max(len(idx), min_per_class), max_per_class)
        need = max(0, target_n - len(idx))
        aug_x = []
        aug_m = []
        for _ in range(need):
            src = int(rng.choice(idx))
            aug_x.append(augment_raman_features(x_train[src].copy(), masks_train[src].copy()))
            aug_m.append(masks_train[src].copy())
        if aug_x:
            xs.append(np.stack(aug_x).astype(np.float32))
            ms.append(np.stack(aug_m).astype(bool))
            ys.append(np.full(len(aug_x), cls, dtype=np.int64))
        rows.append(
            {
                "label_id": int(cls),
                "original_train_count": int(len(idx)),
                "augmented_count": int(need),
                "final_train_count": int(len(idx) + need),
            }
        )
    return np.concatenate(xs), np.concatenate(ms), np.concatenate(ys), pd.DataFrame(rows)


def sklearn_val_test(model, xt, yt, xv, yv, xs) -> tuple[float, np.ndarray]:
    model.fit(xt, yt)
    val_probs = model.predict_proba(xv)
    val_pred = np.argmax(val_probs, axis=1)
    val_macro = float(f1_score(yv, val_pred, average="macro", zero_division=0))
    return val_macro, model.predict_proba(xs)


def run_chemometric_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    out_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for stride, n_pca, c_value, gamma in itertools.product([8, 16], [40, 80, 120], [1.0, 10.0], ["scale", 0.001]):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_use = min(n_pca, xt.shape[0] - 1, xt.shape[1])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_use, random_state=SEED)),
                ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma, class_weight="balanced", probability=True, random_state=SEED)),
            ]
        )
        log(f"PCA-SVM trial: stride={stride}, pca={n_use}, C={c_value}, gamma={gamma}")
        val_macro, probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs)
        rows.append({"model": "pca_svm", "params": json.dumps({"stride": stride, "pca": n_use, "C": c_value, "gamma": gamma}), "val_macro_f1": val_macro, "test_probs": probs})

    for stride, n_comp in itertools.product([8, 16], [4, 8, 12]):
        xt, xv, xs = flatten_features(x_train, stride), flatten_features(x_val, stride), flatten_features(x_test, stride)
        n_use = min(n_comp, len(np.unique(y_train)) - 1, xt.shape[0] - 1, xt.shape[1])
        if n_use < 2:
            continue
        model = PLSDA(n_components=n_use)
        log(f"PLS-DA trial: stride={stride}, components={n_use}")
        val_macro, probs = sklearn_val_test(model, xt, y_train, xv, y_val, xs)
        rows.append({"model": "pls_da", "params": json.dumps({"stride": stride, "components": n_use}), "val_macro_f1": val_macro, "test_probs": probs})

    pd.DataFrame([{k: v for k, v in row.items() if k != "test_probs"} for row in rows]).to_csv(
        out_dir / "chemometric_hyperparameter_trials.csv", index=False, encoding="utf-8-sig"
    )
    return rows


def best_by_model(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    compact = pd.DataFrame([{k: v for k, v in row.items() if k != "test_probs"} for row in rows])
    selected = []
    for model, sub in compact.groupby("model"):
        idx = sub["val_macro_f1"].astype(float).idxmax()
        selected.append(rows[int(idx)])
    return selected


def make_torch_model(model_key: str, n_classes: int, params: dict[str, object]) -> nn.Module:
    if model_key == "cnn":
        return TunedCNN(num_classes=n_classes, dropout=float(params["dropout"]))
    if model_key == "standard_transformer":
        return StandardTransformer(
            num_classes=n_classes,
            seq_len=len(GRID),
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch_size"]),
        )
    if model_key == "mst":
        return MaskedSpectralTransformer(
            num_classes=n_classes,
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch_size"]),
        )
    raise ValueError(model_key)


def partial_load_mst(model: MaskedSpectralTransformer, state: dict, old_classes: list[str], new_classes: list[str]) -> None:
    model_state = model.state_dict()
    transferable = {
        key: value
        for key, value in state.items()
        if key in model_state and value.shape == model_state[key].shape and not key.startswith("head.")
    }
    model_state.update(transferable)
    old_index = {label: i for i, label in enumerate(old_classes)}
    if "head.weight" in state and "head.bias" in state:
        for new_i, label in enumerate(new_classes):
            old_i = old_index.get(label)
            if old_i is not None:
                model_state["head.weight"][new_i] = state["head.weight"][old_i]
                model_state["head.bias"][new_i] = state["head.bias"][old_i]
    model.load_state_dict(model_state)


def split_sherloc_indices(y: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray, str]:
    counts = pd.Series(y).value_counts()
    singleton_labels = set(counts[counts < 2].index.tolist())
    singleton_idx = np.asarray([i for i, yy in enumerate(y) if yy in singleton_labels], dtype=int)
    normal_idx = np.asarray([i for i, yy in enumerate(y) if yy not in singleton_labels], dtype=int)
    if len(normal_idx) == 0:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=SEED)
        return np.sort(train_idx), np.sort(val_idx), "random_no_stratification"
    train_norm, val_idx = train_test_split(
        normal_idx,
        test_size=val_fraction,
        random_state=SEED,
        stratify=y[normal_idx],
    )
    return np.sort(np.concatenate([train_norm, singleton_idx])), np.sort(val_idx), "stratified_for_labels_with_n>=2_singletons_forced_to_train"


def fine_tune_mst_on_pooled_sherloc(
    base_checkpoint: Path,
    mst_params: dict[str, object],
    base_classes: list[str],
    sherloc_df: pd.DataFrame,
    cache_dir: Path,
    out_dir: Path,
    baseline: str,
    refresh_cache: bool,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sherloc_df = sherloc_df.copy().reset_index(drop=True)
    sherloc_df["split_model"] = "sherloc_pool"
    sherloc_df["source_family"] = "SHERLOC"
    x, masks = build_feature_arrays(sherloc_df, cache_dir, "sherloc_pooled_v3", baseline, refresh_cache)
    all_classes = sorted(set(base_classes) | set(sherloc_df["model_label"].unique()))
    label_to_id = {label: i for i, label in enumerate(all_classes)}
    y = sherloc_df["model_label"].map(label_to_id).to_numpy(dtype=np.int64)
    train_idx, val_idx, split_note = split_sherloc_indices(y, 0.2)

    model = MaskedSpectralTransformer(num_classes=len(all_classes), **{k: int(v) for k, v in mst_params.items() if k in {"d_model", "layers", "patch_size"}})
    state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    partial_load_mst(model, state, base_classes, all_classes)

    zero_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    zero_model = MaskedSpectralTransformer(num_classes=len(all_classes), **{k: int(v) for k, v in mst_params.items() if k in {"d_model", "layers", "patch_size"}})
    partial_load_mst(zero_model, state, base_classes, all_classes)
    zero_model.to(device)
    zero_probs, zero_true = predict_torch(zero_model, DataLoader(zero_ds, batch_size=batch_size, shuffle=False), device)
    zero_metrics = evaluate_arrays(zero_true, zero_probs, all_classes, out_dir / "mst_zero_shot_sherloc_pooled_validation")

    for param in model.parameters():
        param.requires_grad = False
    for module in [model.encoder.layers[-1], model.norm, model.head]:
        for param in module.parameters():
            param.requires_grad = True

    train_ds = RamanDataset(x[train_idx], masks[train_idx], y[train_idx], augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds.y, len(all_classes)).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, shifts, mask, yb in train_loader:
            xb, shifts, mask, yb = xb.to(device), shifts.to(device), mask.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, shifts, mask), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probs, val_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(val_true, np.argmax(val_probs, axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "mst_mlrod_base_sherloc_pooled_finetuned.pth")
    pd.DataFrame(history).to_csv(out_dir / "mst_sherloc_pooled_finetune_history.csv", index=False, encoding="utf-8-sig")
    ft_probs, ft_true = predict_torch(model, val_loader, device)
    ft_metrics = evaluate_arrays(ft_true, ft_probs, all_classes, out_dir / "mst_finetuned_sherloc_pooled_validation")

    pred_rows = sherloc_df.iloc[val_idx].copy()
    pred_rows["true_label"] = [all_classes[i] for i in ft_true]
    pred_rows["zero_shot_prediction"] = [all_classes[i] for i in np.argmax(zero_probs, axis=1)]
    pred_rows["zero_shot_confidence"] = np.max(zero_probs, axis=1)
    pred_rows["finetuned_prediction"] = [all_classes[i] for i in np.argmax(ft_probs, axis=1)]
    pred_rows["finetuned_confidence"] = np.max(ft_probs, axis=1)
    pred_rows.to_csv(out_dir / "sherloc_pooled_validation_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "sherloc_rows_total": int(len(sherloc_df)),
        "sherloc_train_rows": int(len(train_idx)),
        "sherloc_validation_rows": int(len(val_idx)),
        "split_note": split_note,
        "base_checkpoint": str(base_checkpoint),
        "fine_tune_mode": "last_transformer_block_norm_and_head",
        "sherloc_valid_range_cm-1": [float(max(800.0, GRID[0])), float(GRID[-1])],
        "zero_shot_accuracy": zero_metrics["accuracy"],
        "zero_shot_macro_f1": zero_metrics["macro_f1"],
        "finetuned_accuracy": ft_metrics["accuracy"],
        "finetuned_macro_f1": ft_metrics["macro_f1"],
        "best_internal_finetune_macro_f1": best_val,
        "classes_after_head_expansion": all_classes,
    }
    (out_dir / "sherloc_pooled_finetune_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dir / "sherloc_pooled_finetune_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def summarize_selected_model(
    model_name: str,
    params: str,
    val_macro: float,
    probs: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    test_meta: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = evaluate_arrays(y_test, probs, classes, out_dir / f"{model_name}.combined_test")
    mlrod_mask = test_meta["source_family"].eq("MLROD").to_numpy()
    ref_mask = ~mlrod_mask
    ref = evaluate_arrays(y_test[ref_mask], probs[ref_mask], classes, out_dir / f"{model_name}.curated_reference_test") if np.any(ref_mask) else {}
    mlrod = evaluate_arrays(y_test[mlrod_mask], probs[mlrod_mask], classes, out_dir / f"{model_name}.mlrod_test") if np.any(mlrod_mask) else {}
    return {
        "model": model_name,
        "params": params,
        "validation_macro_f1": float(val_macro),
        "combined_test_accuracy": combined.get("accuracy"),
        "combined_test_macro_f1": combined.get("macro_f1"),
        "curated_reference_test_accuracy": ref.get("accuracy"),
        "curated_reference_test_macro_f1": ref.get("macro_f1"),
        "mlrod_test_accuracy": mlrod.get("accuracy"),
        "mlrod_test_macro_f1": mlrod.get("macro_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLROD-integrated Raman model comparison.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V3)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--grid-min", type=float, default=0.0)
    parser.add_argument("--grid-max", type=float, default=4000.0)
    parser.add_argument("--grid-points", type=int, default=4100)
    parser.add_argument("--mlrod-train-per-class", type=int, default=800)
    parser.add_argument("--mlrod-val-per-class", type=int, default=200)
    parser.add_argument("--mlrod-test-per-class", type=int, default=200)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    configure_grid(args.grid_min, args.grid_max, args.grid_points)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.out_dir / f"mlrod_v3_{grid_tag()}_{time.strftime('%Y%m%d_%H%M%S')}"
    cache_dir = args.out_dir / "_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")

    full = load_v3_metadata(args.metadata_file)
    pool = build_reference_and_mlrod_pool(full)
    selected = sample_mlrod_by_class(pool, args.mlrod_train_per_class, args.mlrod_val_per_class, args.mlrod_test_per_class)
    selected.to_csv(run_dir / "selected_reference_plus_mlrod_samples.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(selected["model_label"], selected["split_model"]).to_csv(run_dir / "selected_class_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(selected["source_type_normalized"], selected["split_model"]).to_csv(run_dir / "selected_source_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(pool["source_type_normalized"], pool["split_model"]).to_csv(run_dir / "full_available_source_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(pool["model_label"], pool["split_model"]).to_csv(run_dir / "full_available_class_by_split.csv", encoding="utf-8-sig")

    encoder = LabelEncoder()
    selected["label_id"] = encoder.fit_transform(selected["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_feature_arrays(selected, cache_dir, "reference_plus_mlrod_selected", args.baseline, args.refresh_cache)
    y = selected["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(selected["split_model"].eq("train").to_numpy())[0]
    val_idx = np.where(selected["split_model"].eq("val").to_numpy())[0]
    test_idx = np.where(selected["split_model"].eq("test").to_numpy())[0]
    log(f"Selected train/val/test rows: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    x_train_aug, masks_train_aug, y_train_aug, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(run_dir / "shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")

    selected_rows: list[dict[str, object]] = []
    chem_rows = run_chemometric_search(
        x_train_aug,
        y_train_aug,
        x[val_idx],
        y[val_idx],
        x[test_idx],
        run_dir / "chemometric",
    )
    test_meta = selected.iloc[test_idx].reset_index(drop=True)
    for row in best_by_model(chem_rows):
        selected_rows.append(
            summarize_selected_model(
                str(row["model"]),
                str(row["params"]),
                float(row["val_macro_f1"]),
                row["test_probs"],
                y[test_idx],
                classes,
                test_meta,
                run_dir / "selected_model_reports",
            )
        )

    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)

    torch_trials = [
        ("cnn", {"lr": 1e-3, "dropout": 0.25}),
        ("cnn", {"lr": 3e-4, "dropout": 0.40}),
        ("standard_transformer", {"lr": 1e-4, "d_model": 96, "layers": 3, "patch_size": 8}),
        ("standard_transformer", {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 8}),
        ("mst", {"lr": 1e-4, "d_model": 96, "layers": 3, "patch_size": 8}),
        ("mst", {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 8}),
        ("mst", {"lr": 3e-5, "d_model": 128, "layers": 4, "patch_size": 4}),
    ]
    torch_rows = []
    best_mst_checkpoint: Path | None = None
    best_mst_params: dict[str, object] | None = None
    best_mst_val = -1.0
    for trial_i, (model_name, params) in enumerate(torch_trials, start=1):
        trial_name = f"{model_name}_trial{trial_i}"
        log(f"Training {trial_name}: {params}")
        model = make_torch_model(model_name, len(classes), params)
        metrics = train_torch_model(
            trial_name,
            model,
            train_ds,
            val_ds,
            test_ds,
            classes,
            run_dir / "torch",
            args.epochs,
            args.batch_size,
            float(params["lr"]),
            device,
            augment=False,
        )
        checkpoint = run_dir / "torch" / f"{trial_name}.pth"
        reloaded = make_torch_model(model_name, len(classes), params).to(device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        probs, _ = predict_torch(reloaded, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False), device)
        row = {"model": model_name, "trial": trial_name, "params": json.dumps(params), "checkpoint": str(checkpoint), **metrics}
        torch_rows.append(row)
        if model_name == "mst":
            if float(metrics["best_val_macro_f1"]) > best_mst_val:
                best_mst_val = float(metrics["best_val_macro_f1"])
                best_mst_checkpoint = checkpoint
                best_mst_params = params
        selected_rows.append(
            summarize_selected_model(
                trial_name,
                json.dumps(params),
                float(metrics["best_val_macro_f1"]),
                probs,
                y[test_idx],
                classes,
                test_meta,
                run_dir / "all_torch_trial_reports",
            )
        )

    torch_df = pd.DataFrame(torch_rows)
    torch_df.to_csv(run_dir / "torch_hyperparameter_trials.csv", index=False, encoding="utf-8-sig")

    final_rows = []
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(run_dir / "all_trial_test_summary.csv", index=False, encoding="utf-8-sig")
    for model_key, sub in selected_df.assign(base_model=selected_df["model"].str.replace(r"_trial.*", "", regex=True)).groupby("base_model"):
        idx = sub["validation_macro_f1"].astype(float).idxmax()
        final_rows.append(selected_df.loc[idx].to_dict())
    final_df = pd.DataFrame(final_rows).sort_values("combined_test_macro_f1", ascending=False)
    final_df.to_csv(run_dir / "selected_model_test_summary.csv", index=False, encoding="utf-8-sig")

    sherloc = full[
        full["source_type_normalized"].eq("SHERLOC in-situ Mars 2020")
        & full["split_v3"].isin(["sherloc_finetune_pool", "sherloc_external_validation"])
    ].copy()
    sherloc_summary = {}
    if best_mst_checkpoint is not None and best_mst_params is not None and len(sherloc) > 0:
        log("Fine-tuning selected MLROD-integrated MST on pooled SHERLOC in-situ spectra")
        sherloc_summary = fine_tune_mst_on_pooled_sherloc(
            best_mst_checkpoint,
            best_mst_params,
            classes,
            sherloc,
            cache_dir,
            run_dir / "sherloc_pooled_finetune",
            args.baseline,
            args.refresh_cache,
            args.finetune_epochs,
            args.batch_size,
            device,
        )

    manifest = {
        "metadata_file": str(args.metadata_file),
        "full_reference_plus_mlrod_rows_available": int(len(pool)),
        "selected_reference_plus_mlrod_rows": int(len(selected)),
        "mlrod_sampling_caps_per_class": {
            "train": args.mlrod_train_per_class,
            "val": args.mlrod_val_per_class,
            "test": args.mlrod_test_per_class,
            "negative_value_means_use_all_available_rows": True,
        },
        "classes": classes,
        "grid": {"min_cm-1": float(GRID[0]), "max_cm-1": float(GRID[-1]), "points": int(len(GRID))},
        "preprocessing": {
            "axis_alignment": f"Each spectrum is interpolated to the common {float(GRID[0]):.1f}-{float(GRID[-1]):.1f} cm-1 grid; out-of-coverage regions are zeroed and masked.",
            "mlrod_raw_format": "Wide CSV rows with Raman-shift column headers are parsed row-wise and aligned directly from the source file.",
            "baseline": args.baseline,
            "normalization": "Nonnegative max normalization within the valid spectral range, followed by first-derivative channel computation.",
            "sherloc_low_region": "For SHERLOC spectra, values below 800 cm-1 are zeroed and masked.",
        },
        "augmentation": {
            "training_split_only": True,
            "validation_and_test_augmented": False,
            "min_per_class": args.min_per_class,
            "max_per_class_for_augmentation_target": args.max_per_class,
            "protocol": AUGMENTATION_PROTOCOL,
        },
        "reviewer_requested_models": ["pca_svm", "pls_da", "cnn", "standard_transformer", "mst"],
        "torch_epochs": args.epochs,
        "sherloc_finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": SEED,
        "sherloc_finetune_summary": sherloc_summary,
    }
    (run_dir / "mlrod_integrated_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(final_df.to_string(index=False))
    print(f"Saved MLROD-integrated experiment to: {run_dir}")


if __name__ == "__main__":
    main()
