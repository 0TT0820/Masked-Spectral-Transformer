"""Band-aware and multi-scale MST experiments on the MLROD-integrated v3 data.

This script tests the final MST configuration for Raman mineral classification
on the MLROD-integrated multi-source dataset and pooled SHERLOC spectra.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import run_mlrod_integrated_model_comparison_v3 as mlrod
import train_model_comparison as tmc
from run_mlrod_integrated_model_comparison_v3 import (
    build_feature_arrays,
    build_reference_and_mlrod_pool,
    configure_grid,
    evaluate_arrays,
    grid_tag,
    load_v3_metadata,
    make_balanced_augmented_train,
    sample_mlrod_by_class,
    split_sherloc_indices,
    summarize_selected_model,
)
from train_model_comparison import (
    FixedPositionEncoder,
    PatchProjector,
    RamanDataset,
    class_weights,
    fix_seed,
    predict_torch,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA_V3 = ROOT / "data" / "metadata" / "metadata_training_database_v3_mlrod_integrated.csv"
DEFAULT_OUT = ROOT / "results" / "band_aware_mlrod_v3"
SEED = 2024


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring optional tabulate."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


class MultiScalePatchProjector(nn.Module):
    """Patch tokenizer with local Raman-band shape filters before tokenization."""

    def __init__(
        self,
        in_chans: int,
        d_model: int,
        patch_size: int,
        branch_dim: int = 32,
        kernels: tuple[int, ...] = (1, 5, 15, 31),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_chans, branch_dim, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.token_proj = nn.Conv1d(branch_dim * len(kernels), d_model, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ch = x.transpose(1, 2)
        valid = (~mask).float().unsqueeze(1)
        x_ch = x_ch * valid
        multi = torch.cat([branch(x_ch) for branch in self.branches], dim=1)
        tokens = self.token_proj(multi).transpose(1, 2)
        valid_count = F.avg_pool1d(valid, kernel_size=self.patch_size, stride=self.patch_size).squeeze(1)
        token_mask = valid_count <= 0.0
        token_shifts = F.avg_pool1d(shifts.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size).squeeze(1)
        return tokens, token_shifts, token_mask


class RamanAwareMST(nn.Module):
    """MST variant with optional multi-scale frontend and band-aware pooling."""

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 3,
        patch_size: int = 4,
        frontend: str = "patch",
        pooling: str = "mean",
        saliency_init: float = 0.75,
    ) -> None:
        super().__init__()
        if frontend == "multiscale":
            self.patch = MultiScalePatchProjector(in_chans, d_model, patch_size)
        elif frontend == "patch":
            self.patch = PatchProjector(in_chans, d_model, patch_size=patch_size)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")
        if pooling not in {"mean", "band_attention"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.pos_encoder = FixedPositionEncoder(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.saliency_scale = nn.Parameter(torch.tensor(float(saliency_init)))
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def second_difference(channel: torch.Tensor) -> torch.Tensor:
        if channel.shape[1] < 3:
            return torch.zeros_like(channel)
        inner = channel[:, 2:] - 2.0 * channel[:, 1:-1] + channel[:, :-2]
        return F.pad(inner, (1, 1), mode="constant", value=0.0)

    def token_saliency(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float().unsqueeze(1)
        intensity = x[:, :, 0].clamp(min=0.0)
        derivative = x[:, :, 1].abs()
        second = self.second_difference(x[:, :, 1]).abs()
        band_signal = (intensity + 0.35 * derivative + 0.20 * second).unsqueeze(1) * valid
        avg_signal = F.avg_pool1d(
            band_signal,
            kernel_size=self.patch.patch_size,
            stride=self.patch.patch_size,
        ).squeeze(1)
        avg_valid = F.avg_pool1d(
            valid,
            kernel_size=self.patch.patch_size,
            stride=self.patch.patch_size,
        ).squeeze(1)
        saliency = avg_signal / avg_valid.clamp(min=1e-6)
        saliency = saliency - saliency.mean(dim=1, keepdim=True)
        saliency = saliency / saliency.std(dim=1, keepdim=True).clamp(min=1e-3)
        return saliency

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, token_shifts, token_mask = self.patch(x, shifts, mask)
        h = h + self.pos_encoder(token_shifts)
        h = h.masked_fill(token_mask.unsqueeze(-1), 0.0)
        h = self.encoder(h, src_key_padding_mask=token_mask)
        h = self.norm(h)
        if self.pooling == "mean":
            valid = (~token_mask).unsqueeze(-1).float()
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            saliency = self.token_saliency(x, mask)
            logits = self.attn_pool(h).squeeze(-1) + self.saliency_scale * saliency
            logits = logits.masked_fill(token_mask, -1e9)
            weights = torch.softmax(logits, dim=1).unsqueeze(-1)
            pooled = (h * weights).sum(dim=1)
        return self.head(pooled)


class FocalLoss(nn.Module):
    """Class-weighted focal loss for imbalanced Raman classes."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_model(params: dict[str, object], n_classes: int) -> RamanAwareMST:
    return RamanAwareMST(
        num_classes=n_classes,
        d_model=int(params.get("d_model", 96)),
        layers=int(params.get("layers", 3)),
        patch_size=int(params.get("patch_size", 4)),
        frontend=str(params.get("frontend", "patch")),
        pooling=str(params.get("pooling", "mean")),
        saliency_init=float(params.get("saliency_init", 0.75)),
    )


def train_variant(
    trial_name: str,
    params: dict[str, object],
    train_ds: RamanDataset,
    val_ds: RamanDataset,
    test_ds: RamanDataset,
    classes: list[str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(params, len(classes)).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    weights = class_weights(train_ds.y, len(classes)).to(device)
    if str(params.get("loss", "ce")) == "focal":
        loss_fn: nn.Module = FocalLoss(weight=weights, gamma=float(params.get("gamma", 1.5)))
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=1e-4)

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
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = out_dir / f"{trial_name}.pth"
    torch.save(model.state_dict(), checkpoint)
    pd.DataFrame(history).to_csv(out_dir / f"{trial_name}.history.csv", index=False, encoding="utf-8-sig")
    test_probs, test_true = predict_torch(model, test_loader, device)
    metrics = evaluate_arrays(test_true, test_probs, classes, out_dir / f"{trial_name}.test")
    row = {
        "model": trial_name,
        "params": json.dumps(params),
        "checkpoint": str(checkpoint),
        "best_val_macro_f1": best_val,
        **metrics,
    }
    return row, test_probs


def partial_load_variant(model: RamanAwareMST, state: dict[str, torch.Tensor], old_classes: list[str], new_classes: list[str]) -> None:
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


def set_finetune_trainable(model: RamanAwareMST, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    modules: list[nn.Module] = []
    extra: list[nn.Parameter] = []
    if mode == "pool_head":
        modules = [model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "last_block_pool_head":
        modules = [model.encoder.layers[-1], model.norm, model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    for param in extra:
        param.requires_grad = True


def threshold_summary(y_true: np.ndarray, probs: np.ndarray, classes: list[str], prefix: Path) -> None:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for threshold in np.linspace(0.0, 0.95, 20):
        keep = conf >= threshold
        if np.any(keep):
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": float(np.mean(keep)),
                    "accepted_n": int(np.sum(keep)),
                    "accuracy_on_accepted": float(accuracy_score(y_true[keep], pred[keep])),
                    "macro_f1_on_accepted": float(f1_score(y_true[keep], pred[keep], average="macro", zero_division=0)),
                    "present_label_macro_f1_on_accepted": float(
                        f1_score(y_true[keep], pred[keep], labels=np.unique(y_true[keep]), average="macro", zero_division=0)
                    ),
                    "false_discovery_rate": float(1.0 - accuracy_score(y_true[keep], pred[keep])),
                }
            )
        else:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "accepted_n": 0,
                    "accuracy_on_accepted": np.nan,
                    "macro_f1_on_accepted": np.nan,
                    "present_label_macro_f1_on_accepted": np.nan,
                    "false_discovery_rate": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(prefix.with_suffix(".threshold_sweep.csv"), index=False, encoding="utf-8-sig")


def fine_tune_on_sherloc(
    checkpoint: Path,
    params: dict[str, object],
    base_classes: list[str],
    sherloc_df: pd.DataFrame,
    cache_dir: Path,
    out_dir: Path,
    baseline: str,
    refresh_cache: bool,
    epochs: int,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sherloc_df = sherloc_df.copy().reset_index(drop=True)
    all_classes = sorted(set(base_classes) | set(sherloc_df["model_label"].unique()))
    class_to_idx = {label: i for i, label in enumerate(all_classes)}
    sherloc_df["label_id"] = sherloc_df["model_label"].map(class_to_idx).astype(int)
    x_s, masks_s = build_feature_arrays(sherloc_df, cache_dir, "sherloc_pooled_v3", baseline, refresh_cache)
    y_s = sherloc_df["label_id"].to_numpy(dtype=np.int64)
    train_idx, val_idx, split_note = split_sherloc_indices(y_s, 0.2)
    train_ds = RamanDataset(x_s[train_idx], masks_s[train_idx], y_s[train_idx], augment=False)
    val_ds = RamanDataset(x_s[val_idx], masks_s[val_idx], y_s[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = make_model(params, len(all_classes)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    partial_load_variant(model, state, base_classes, all_classes)

    zero_probs, zero_true = predict_torch(model, val_loader, device)
    zero_metrics = evaluate_arrays(zero_true, zero_probs, all_classes, out_dir / "zero_shot_sherloc_validation")
    threshold_summary(zero_true, zero_probs, all_classes, out_dir / "zero_shot_sherloc_validation")

    set_finetune_trainable(model, mode)
    weights = class_weights(y_s[train_idx], len(all_classes)).to(device)
    loss_fn: nn.Module
    if str(params.get("loss", "ce")) == "focal":
        loss_fn = FocalLoss(weight=weights, gamma=float(params.get("gamma", 1.5)))
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=1e-4)
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
    torch.save(model.state_dict(), out_dir / "finetuned_sherloc_model.pth")
    pd.DataFrame(history).to_csv(out_dir / "finetune_history.csv", index=False, encoding="utf-8-sig")
    ft_probs, ft_true = predict_torch(model, val_loader, device)
    ft_metrics = evaluate_arrays(ft_true, ft_probs, all_classes, out_dir / "finetuned_sherloc_validation")
    threshold_summary(ft_true, ft_probs, all_classes, out_dir / "finetuned_sherloc_validation")
    pred_rows = sherloc_df.iloc[val_idx].copy()
    pred_rows["true_label"] = [all_classes[i] for i in ft_true]
    pred_rows["zero_shot_prediction"] = [all_classes[i] for i in np.argmax(zero_probs, axis=1)]
    pred_rows["zero_shot_confidence"] = np.max(zero_probs, axis=1)
    pred_rows["finetuned_prediction"] = [all_classes[i] for i in np.argmax(ft_probs, axis=1)]
    pred_rows["finetuned_confidence"] = np.max(ft_probs, axis=1)
    pred_rows.to_csv(out_dir / "sherloc_validation_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        "sherloc_rows": int(len(sherloc_df)),
        "train_n": int(len(train_idx)),
        "validation_n": int(len(val_idx)),
        "split_note": split_note,
        "finetune_mode": mode,
        "zero_shot_accuracy": zero_metrics["accuracy"],
        "zero_shot_macro_f1": zero_metrics["macro_f1"],
        "finetuned_accuracy": ft_metrics["accuracy"],
        "finetuned_macro_f1": ft_metrics["macro_f1"],
        "best_finetune_validation_macro_f1": best_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Raman-aware MST variants on MLROD-integrated v3 data.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--grid-min", type=float, default=0.0)
    parser.add_argument("--grid-max", type=float, default=4000.0)
    parser.add_argument("--grid-points", type=int, default=4001)
    parser.add_argument("--mlrod-train-per-class", type=int, default=800)
    parser.add_argument("--mlrod-val-per-class", type=int, default=200)
    parser.add_argument("--mlrod-test-per-class", type=int, default=200)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-sherloc", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--trial-set", choices=["core", "patch8_ce", "tuned", "main", "all"], default="all")
    parser.add_argument(
        "--main-model",
        default="band_multiscale_mst_patch8_d128_ce",
        help="Model reported as the manuscript main MST configuration after the screening run.",
    )
    args = parser.parse_args()

    fix_seed(SEED)
    configure_grid(args.grid_min, args.grid_max, args.grid_points)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.out_dir / f"band_aware_mlrod_v3_{grid_tag()}_{time.strftime('%Y%m%d_%H%M%S')}"
    cache_dir = args.cache_dir if args.cache_dir is not None else args.out_dir / "_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run directory: {run_dir}")
    log(f"Cache directory: {cache_dir}")
    log(f"Device: {device}")

    full = load_v3_metadata(args.metadata_file)
    pool = build_reference_and_mlrod_pool(full)
    selected = sample_mlrod_by_class(pool, args.mlrod_train_per_class, args.mlrod_val_per_class, args.mlrod_test_per_class)
    selected.to_csv(run_dir / "selected_reference_plus_mlrod_samples.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(selected["model_label"], selected["split_model"]).to_csv(run_dir / "selected_class_by_split.csv", encoding="utf-8-sig")
    pd.crosstab(selected["source_type_normalized"], selected["split_model"]).to_csv(run_dir / "selected_source_by_split.csv", encoding="utf-8-sig")

    encoder = LabelEncoder()
    selected["label_id"] = encoder.fit_transform(selected["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_feature_arrays(selected, cache_dir, "reference_plus_mlrod_selected", args.baseline, args.refresh_cache)
    y = selected["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(selected["split_model"].eq("train").to_numpy())[0]
    val_idx = np.where(selected["split_model"].eq("val").to_numpy())[0]
    test_idx = np.where(selected["split_model"].eq("test").to_numpy())[0]
    x_train_aug, masks_train_aug, y_train_aug, aug_summary = make_balanced_augmented_train(
        x[train_idx],
        masks[train_idx],
        y[train_idx],
        args.min_per_class,
        args.max_per_class,
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(run_dir / "shared_training_augmentation_summary.csv", index=False, encoding="utf-8-sig")
    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)
    test_meta = selected.iloc[test_idx].reset_index(drop=True)

    core_trials = [
        {
            "trial": "mst_patch4_ce_reference",
            "frontend": "patch",
            "pooling": "mean",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 4,
        },
        {
            "trial": "band_mst_patch4_ce",
            "frontend": "patch",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 4,
            "saliency_init": 0.75,
        },
        {
            "trial": "multiscale_mst_patch4_ce",
            "frontend": "multiscale",
            "pooling": "mean",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 4,
        },
        {
            "trial": "band_multiscale_mst_patch4_ce",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 4,
            "saliency_init": 0.75,
        },
        {
            "trial": "band_multiscale_mst_patch4_focal",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "focal",
            "gamma": 1.5,
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 4,
            "saliency_init": 0.75,
        },
        {
            "trial": "band_multiscale_mst_patch8_focal",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "focal",
            "gamma": 1.5,
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 8,
            "saliency_init": 0.75,
        },
    ]
    patch8_ce_trials = [
        {
            "trial": "mst_patch8_ce_reference",
            "frontend": "patch",
            "pooling": "mean",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 8,
        },
        {
            "trial": "band_mst_patch8_ce",
            "frontend": "patch",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 8,
            "saliency_init": 0.75,
        },
        {
            "trial": "multiscale_mst_patch8_ce",
            "frontend": "multiscale",
            "pooling": "mean",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 8,
        },
        {
            "trial": "band_multiscale_mst_patch8_ce",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 1e-4,
            "d_model": 96,
            "layers": 3,
            "patch_size": 8,
            "saliency_init": 0.75,
        },
    ]
    tuned_trials = [
        {
            "trial": "band_mst_patch4_d128_ce",
            "frontend": "patch",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 3e-5,
            "d_model": 128,
            "layers": 4,
            "patch_size": 4,
            "saliency_init": 0.75,
        },
        {
            "trial": "band_mst_patch8_d128_ce",
            "frontend": "patch",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 3e-5,
            "d_model": 128,
            "layers": 4,
            "patch_size": 8,
            "saliency_init": 0.75,
        },
        {
            "trial": "band_multiscale_mst_patch4_d128_ce",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 3e-5,
            "d_model": 128,
            "layers": 4,
            "patch_size": 4,
            "saliency_init": 0.75,
        },
        {
            "trial": "band_multiscale_mst_patch8_d128_ce",
            "frontend": "multiscale",
            "pooling": "band_attention",
            "loss": "ce",
            "lr": 3e-5,
            "d_model": 128,
            "layers": 4,
            "patch_size": 8,
            "saliency_init": 0.75,
        },
    ]
    all_trials = core_trials + patch8_ce_trials + tuned_trials
    if args.trial_set == "core":
        trials = core_trials
    elif args.trial_set == "patch8_ce":
        trials = patch8_ce_trials
    elif args.trial_set == "tuned":
        trials = tuned_trials
    elif args.trial_set == "main":
        trials = [trial for trial in all_trials if trial["trial"] == args.main_model]
        if not trials:
            raise ValueError(f"--trial-set main could not find --main-model {args.main_model!r}")
    else:
        trials = all_trials

    trial_rows = []
    summary_rows = []
    best = None
    for params in trials:
        trial_name = str(params["trial"])
        log(f"Training {trial_name}: {params}")
        row, probs = train_variant(
            trial_name,
            params,
            train_ds,
            val_ds,
            test_ds,
            classes,
            run_dir / "torch",
            args.epochs,
            args.batch_size,
            device,
        )
        trial_rows.append(row)
        summary = summarize_selected_model(
            trial_name,
            json.dumps(params),
            float(row["best_val_macro_f1"]),
            probs,
            y[test_idx],
            classes,
            test_meta,
            run_dir / "selected_model_reports",
        )
        summary["checkpoint"] = row["checkpoint"]
        summary_rows.append(summary)
        pd.DataFrame(trial_rows).to_csv(run_dir / "band_aware_torch_trials.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(run_dir / "band_aware_selected_model_summary.csv", index=False, encoding="utf-8-sig")
        if best is None or float(row["best_val_macro_f1"]) > float(best["best_val_macro_f1"]):
            best = row

    summary_df = pd.DataFrame(summary_rows).sort_values("validation_macro_f1", ascending=False)
    summary_df.to_csv(run_dir / "band_aware_selected_model_summary.csv", index=False, encoding="utf-8-sig")
    validation_best_summary = summary_df.iloc[0].to_dict()
    main_rows = summary_df[summary_df["model"].eq(args.main_model)]
    if len(main_rows) == 0:
        log(f"Requested main model {args.main_model!r} was not in the trial set; using validation-best model instead.")
        reported_main_summary = validation_best_summary
    else:
        reported_main_summary = main_rows.iloc[0].to_dict()
    main_params = json.loads(reported_main_summary["params"])
    main_checkpoint = Path(reported_main_summary["checkpoint"])
    sherloc_summary = {}
    if args.run_sherloc:
        sherloc = full[
            full["source_type_normalized"].eq("SHERLOC in-situ Mars 2020")
            & full["split_v3"].isin(["sherloc_finetune_pool", "sherloc_external_validation"])
        ].copy()
        log(f"Fine-tuning manuscript main Raman-aware variant on pooled SHERLOC spectra: {reported_main_summary['model']}")
        sherloc_summary = fine_tune_on_sherloc(
            main_checkpoint,
            main_params,
            classes,
            sherloc,
            cache_dir,
            run_dir / "sherloc_pooled_finetune",
            args.baseline,
            args.refresh_cache,
            args.finetune_epochs,
            args.batch_size,
            device,
            "last_block_pool_head",
        )
        pd.DataFrame([sherloc_summary]).to_csv(run_dir / "sherloc_finetune_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "architecture_family": "RamanAwareMST",
        "purpose": "Explicit revised architecture experiment: band-aware pooling, multi-scale Raman-band frontend, and focal-loss sensitivity.",
        "metadata_file": str(args.metadata_file),
        "grid": {"min_cm-1": float(mlrod.GRID[0]), "max_cm-1": float(mlrod.GRID[-1]), "points": int(len(mlrod.GRID))},
        "selected_reference_plus_mlrod_rows": int(len(selected)),
        "train_val_test_rows": [int(len(train_idx)), int(len(val_idx)), int(len(test_idx))],
        "classes": classes,
        "trials": trials,
        "selection_rule": "screen all candidates by reference validation macro-F1; report the manuscript main MST specified by --main-model",
        "validation_best_model": validation_best_summary,
        "reported_main_model": reported_main_summary,
        "sherloc_finetune_summary": sherloc_summary,
        "seed": SEED,
        "device": str(device),
    }
    (run_dir / "band_aware_mlrod_v3_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Band-Aware MLROD v3 Experiment Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"Grid: {float(mlrod.GRID[0]):.0f}-{float(mlrod.GRID[-1]):.0f} cm-1, {len(mlrod.GRID)} points.",
        f"Selected benchmark: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test spectra.",
        "",
        f"Validation-best screening candidate: `{validation_best_summary['model']}`.",
        f"Manuscript main reported MST configuration: `{reported_main_summary['model']}`.",
        "",
        "## Reference + MLROD Candidate Models",
        "",
        markdown_table(
            summary_df[
                [
                    "model",
                    "validation_macro_f1",
                    "combined_test_accuracy",
                    "combined_test_macro_f1",
                    "curated_reference_test_accuracy",
                    "curated_reference_test_macro_f1",
                    "mlrod_test_accuracy",
                    "mlrod_test_macro_f1",
                ]
            ]
        ),
        "",
    ]
    if sherloc_summary:
        lines.extend(
            [
                "## SHERLOC Fine-Tuning of Manuscript Main Candidate",
                "",
                markdown_table(pd.DataFrame([sherloc_summary])),
                "",
            ]
        )
    (run_dir / "band_aware_mlrod_v3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_df.to_string(index=False))
    if sherloc_summary:
        print(pd.DataFrame([sherloc_summary]).to_string(index=False))
    print(f"Saved Raman-aware MST experiment to: {run_dir}")


if __name__ == "__main__":
    main()
