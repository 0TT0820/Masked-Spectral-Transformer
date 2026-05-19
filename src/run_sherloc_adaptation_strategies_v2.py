"""Adaptation strategies to improve SHERLOC external macro-F1.

This script tests conservative post-processing and fine-tuning choices for the
small, imbalanced SHERLOC external set. The key issue is that the fine-tuning
pool does not contain all external labels, especially Phosphate. Updating the
full classification head with ordinary cross-entropy can suppress classes that
are absent from the SHERLOC fine-tuning pool.

Strategies tested:
- zero-shot MST with SHERLOC preprocessing;
- norm/head fine-tuning;
- norm-only fine-tuning, preserving the original classifier;
- head-row-masked fine-tuning, where only rows for labels present in the
  SHERLOC fine-tuning pool are updated;
- optional prediction-space restriction to SHERLOC-plausible labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_model_comparison import RamanDataset, class_weights, predict_torch
from run_review_updated_training_v2 import (
    METADATA_V2,
    SEED,
    load_partial_mst_state,
    load_v2_metadata,
    make_mst_from_checkpoint_config,
    reference_split,
    sherloc_split,
)
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays


PLAUSIBLE_SHERLOC_LABELS = {
    "Carbonate",
    "Olivine",
    "Other Silicates",
    "Perchlorate",
    "Phosphate",
    "Pyroxene",
    "Sulfate",
}


def metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str], prefix: str) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}_macro_f1_union": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        f"{prefix}_macro_f1_true_labels": float(
            f1_score(y_true, pred, labels=np.unique(y_true), average="macro", zero_division=0)
        ),
        f"{prefix}_weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        f"{prefix}_predicted_labels": ";".join(sorted({labels[i] for i in np.unique(pred)})),
    }


def restrict_probs(probs: np.ndarray, allowed_idx: list[int]) -> np.ndarray:
    out = np.zeros_like(probs)
    out[:, allowed_idx] = probs[:, allowed_idx]
    denom = out.sum(axis=1, keepdims=True)
    # If every allowed logit underflows to zero, keep uniform over allowed labels.
    zero = denom[:, 0] <= 0
    if np.any(zero):
        out[zero, allowed_idx] = 1.0 / len(allowed_idx)
        denom = out.sum(axis=1, keepdims=True)
    return out / denom


def make_model(base_checkpoint: Path, base_classes: list[str], all_classes: list[str], device: torch.device):
    model, state, config = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(model, state, base_classes, all_classes)
    model.to(device)
    return model, config


def set_trainable(model: nn.Module, strategy: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if strategy == "zero_shot":
        return
    if strategy == "norm_only":
        modules = [model.norm]
    elif strategy in {"norm_head", "head_row_masked"}:
        modules = [model.norm, model.head]
    elif strategy == "last_block_norm_head":
        modules = [model.encoder.layers[-1], model.norm, model.head]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def apply_head_row_mask(model: nn.Module, allowed_update_idx: set[int]) -> None:
    if model.head.weight.grad is not None:
        mask = torch.zeros_like(model.head.weight.grad)
        rows = torch.tensor(sorted(allowed_update_idx), device=mask.device, dtype=torch.long)
        mask[rows, :] = 1.0
        model.head.weight.grad.mul_(mask)
    if model.head.bias.grad is not None:
        mask_b = torch.zeros_like(model.head.bias.grad)
        rows = torch.tensor(sorted(allowed_update_idx), device=mask_b.device, dtype=torch.long)
        mask_b[rows] = 1.0
        model.head.bias.grad.mul_(mask_b)


def train_strategy(
    model: nn.Module,
    strategy: str,
    x_ft: np.ndarray,
    masks_ft: np.ndarray,
    y_ft: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_count: int,
    present_idx: set[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[nn.Module, float]:
    if strategy == "zero_shot":
        return model, np.nan
    set_trainable(model, strategy)
    train_loader = DataLoader(RamanDataset(x_ft[train_idx], masks_ft[train_idx], y_ft[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RamanDataset(x_ft[val_idx], masks_ft[val_idx], y_ft[val_idx]), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_ft[train_idx], class_count).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for x, shifts, mask, y in train_loader:
            x, shifts, mask, y = x.to(device), shifts.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, shifts, mask), y)
            loss.backward()
            if strategy == "head_row_masked":
                apply_head_row_mask(model, present_idx)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_probs, val_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(val_true, np.argmax(val_probs, axis=1), average="macro", zero_division=0))
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Try SHERLOC adaptation strategies to improve macro-F1.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--variant", default="despike_sg11_asls")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.run_dir / "sherloc_adaptation_strategies"
    cache_dir = args.run_dir / "_sherloc_preprocessing_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_summary = json.loads((args.run_dir / "sherloc_finetune" / "sherloc_finetune_v2_summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(ft_summary["base_checkpoint"])
    base_classes = ft_summary["base_classes"]
    all_classes = ft_summary["all_classes_after_sherloc"]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    full = load_v2_metadata(args.metadata_file)
    ft, ext = sherloc_split(full)
    x_ft, masks_ft, _ = build_sherloc_arrays(ft, cache_dir, "sherloc_finetune_pool", args.variant, args.refresh_cache)
    x_ext, masks_ext, _ = build_sherloc_arrays(ext, cache_dir, "sherloc_external_validation", args.variant, args.refresh_cache)
    y_ft = ft["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    y_ext = ext["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
    train_idx, val_idx = train_test_split(
        np.arange(len(ft)),
        test_size=0.2,
        random_state=SEED,
        stratify=y_ft if np.min(np.bincount(y_ft)) >= 2 else None,
    )
    present_idx = set(np.unique(y_ft).tolist())
    plausible_idx = [class_to_idx[c] for c in all_classes if c in PLAUSIBLE_SHERLOC_LABELS]
    present_plus_external_truth_idx = sorted(set(present_idx) | set(np.unique(y_ext).tolist()))

    ext_loader = DataLoader(RamanDataset(x_ext, masks_ext, y_ext), batch_size=args.batch_size, shuffle=False)
    rows = []
    strategies = ["zero_shot", "norm_only", "norm_head", "head_row_masked", "last_block_norm_head"]
    for strategy in strategies:
        model, config = make_model(base_checkpoint, base_classes, all_classes, device)
        model, best_val = train_strategy(
            model,
            strategy,
            x_ft,
            masks_ft,
            y_ft,
            train_idx,
            val_idx,
            len(all_classes),
            present_idx,
            device,
            args.epochs,
            args.lr,
            args.batch_size,
        )
        probs, _ = predict_torch(model, ext_loader, device)
        row = {
            "strategy": strategy,
            "variant": args.variant,
            "best_internal_val_macro_f1": best_val,
            "base_config": json.dumps(config),
            "prediction_space": "all_classes",
            **metrics(y_ext, probs, all_classes, "external"),
        }
        rows.append(row)
        for name, allowed in [
            ("plausible_sherloc_labels", plausible_idx),
            ("finetune_present_plus_external_truth", present_plus_external_truth_idx),
        ]:
            restricted = restrict_probs(probs, allowed)
            rows.append(
                {
                    "strategy": strategy,
                    "variant": args.variant,
                    "best_internal_val_macro_f1": best_val,
                    "base_config": json.dumps(config),
                    "prediction_space": name,
                    **metrics(y_ext, restricted, all_classes, "external"),
                }
            )
        torch.save(model.state_dict(), out_dir / f"{strategy}_{args.variant}.pth")
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "sherloc_adaptation_strategy_summary.csv", index=False, encoding="utf-8-sig")

    pred_note = {
        "variant": args.variant,
        "strategies": strategies,
        "plausible_sherloc_labels": sorted(PLAUSIBLE_SHERLOC_LABELS),
        "finetune_present_labels": sorted({all_classes[i] for i in present_idx}),
        "external_true_labels": sorted({all_classes[i] for i in np.unique(y_ext)}),
        "caution": "The finetune_present_plus_external_truth prediction space uses external labels for diagnostic upper-bound analysis and should not be reported as a deployment result.",
    }
    (out_dir / "sherloc_adaptation_strategy_manifest.json").write_text(json.dumps(pred_note, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result.sort_values("external_macro_f1_union", ascending=False).to_string(index=False))
    print(f"Saved SHERLOC adaptation strategies to: {out_dir}")


if __name__ == "__main__":
    main()
