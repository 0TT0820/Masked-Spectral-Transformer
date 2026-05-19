"""Band-aware MST experiments for reference and SHERLOC pooled validation.

This script evaluates a physically motivated MST variant that replaces masked
average pooling with band-aware attention pooling. The attention logits combine
learned token relevance with a Raman-band saliency term computed from normalized
intensity and derivative channels. Band centers are not shifted or altered.

The script is intentionally separate from the original MST implementation so
that the manuscript can report it as an explicit model-improvement/sensitivity
experiment rather than silently changing the published architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from run_review_updated_training_v2 import (
    SEED,
    build_arrays,
    load_v2_metadata,
    make_balanced_augmented_train,
    reference_split,
    sherloc_split,
)
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays
from train_model_comparison import (
    AUGMENTATION_PROTOCOL,
    FixedPositionEncoder,
    PatchProjector,
    RamanDataset,
    class_weights,
    evaluate_arrays,
    fix_seed,
    predict_torch,
    train_torch_model,
)


ROOT = Path(__file__).resolve().parents[1]


class BandAwareMST(nn.Module):
    """MST with Raman-band-aware attention pooling."""

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 3,
        patch_size: int = 8,
        saliency_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch = PatchProjector(in_chans, d_model, patch_size=patch_size)
        self.pos_encoder = FixedPositionEncoder(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1, norm_first=True
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

    def token_saliency(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float().unsqueeze(1)
        intensity = x[:, :, 0].clamp(min=0).unsqueeze(1)
        derivative = x[:, :, 1].abs().unsqueeze(1)
        band_signal = (intensity + 0.35 * derivative) * valid
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
        saliency = self.token_saliency(x, mask)
        logits = self.attn_pool(h).squeeze(-1) + self.saliency_scale * saliency
        logits = logits.masked_fill(token_mask, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        return self.head(pooled)


def split_indices(y: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray, str]:
    counts = pd.Series(y).value_counts()
    singleton_labels = set(counts[counts < 2].index.tolist())
    singleton_idx = np.array([i for i, yy in enumerate(y) if yy in singleton_labels], dtype=int)
    normal_idx = np.array([i for i, yy in enumerate(y) if yy not in singleton_labels], dtype=int)
    if len(normal_idx) == 0:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
        return train_idx, val_idx, "random_no_stratification"
    train_norm, val_idx = train_test_split(
        normal_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y[normal_idx],
    )
    train_idx = np.concatenate([train_norm, singleton_idx])
    return np.sort(train_idx), np.sort(val_idx), "stratified_for_labels_with_n>=2_singletons_forced_to_train"


def pooled_metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "present_label_macro_f1": float(
            f1_score(y_true, pred, labels=np.unique(y_true), average="macro", zero_division=0)
        ),
        "predicted_labels": ";".join(sorted({labels[i] for i in np.unique(pred)})),
    }


def make_pooled_sherloc(metadata_file: Path) -> pd.DataFrame:
    full = load_v2_metadata(metadata_file)
    ft, ext = sherloc_split(full)
    ft["pooled_origin"] = "previous_finetune_pool"
    ext["pooled_origin"] = "previous_external_validation"
    return pd.concat([ft, ext], ignore_index=True).reset_index(drop=True)


def load_partial_band_state(
    model: BandAwareMST,
    checkpoint: Path,
    base_classes: list[str],
    all_classes: list[str],
) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    own = model.state_dict()
    copied = {}
    for key, value in state.items():
        if key.startswith("head."):
            continue
        if key in own and own[key].shape == value.shape:
            copied[key] = value
    own.update(copied)
    model.load_state_dict(own, strict=False)

    with torch.no_grad():
        old_w = state["head.weight"]
        old_b = state["head.bias"]
        for old_idx, label in enumerate(base_classes):
            if label in all_classes:
                new_idx = all_classes.index(label)
                model.head.weight[new_idx].copy_(old_w[old_idx])
                model.head.bias[new_idx].copy_(old_b[old_idx])


def set_trainable(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "norm_attn_head":
        modules = [model.norm, model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "last_block_norm_attn_head":
        modules = [model.encoder.layers[-1], model.norm, model.attn_pool, model.head]
        extra = [model.saliency_scale]
    elif mode == "all":
        modules = [model]
        extra = []
    else:
        raise ValueError(f"Unknown mode: {mode}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
    for p in extra:
        p.requires_grad = True


def train_sherloc_split(
    checkpoint: Path,
    config: dict,
    base_classes: list[str],
    all_classes: list[str],
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mode: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    split_name: str,
) -> dict:
    model = BandAwareMST(
        num_classes=len(all_classes),
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        patch_size=int(config["patch_size"]),
        saliency_init=float(config.get("saliency_init", 1.0)),
    )
    load_partial_band_state(model, checkpoint, base_classes, all_classes)
    set_trainable(model, mode)
    model.to(device)

    train_loader = DataLoader(RamanDataset(x[train_idx], masks[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RamanDataset(x[val_idx], masks[val_idx], y[val_idx]), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], len(all_classes)).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xx, shifts, mask, yy in train_loader:
            xx, shifts, mask, yy = xx.to(device), shifts.to(device), mask.to(device), yy.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xx, shifts, mask), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probs, yy_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(yy_true, np.argmax(probs, axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    probs, yy_true = predict_torch(model, val_loader, device)
    pred = np.argmax(probs, axis=1)
    result = {
        "split": split_name,
        "mode": mode,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "epochs": epochs,
        "lr": lr,
        "base_config": json.dumps(config),
        "best_val_macro_f1_during_training": best_val,
        **pooled_metrics(yy_true, probs, all_classes),
    }
    pd.DataFrame(history).to_csv(out_dir / f"{split_name}_{mode}_history.csv", index=False, encoding="utf-8-sig")
    report = classification_report(
        yy_true,
        pred,
        labels=np.arange(len(all_classes)),
        target_names=all_classes,
        zero_division=0,
    )
    (out_dir / f"{split_name}_{mode}_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(yy_true, pred, labels=np.arange(len(all_classes)))
    pd.DataFrame(cm, index=all_classes, columns=all_classes).to_csv(
        out_dir / f"{split_name}_{mode}_confusion_matrix.csv", encoding="utf-8-sig"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Band-aware MST reference and SHERLOC experiments.")
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", choices=["poly", "none", "asls"], default="poly")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--sherloc-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--sherloc-batch-size", type=int, default=32)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=260)
    parser.add_argument("--run-sherloc", action="store_true")
    parser.add_argument("--sherloc-seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--sherloc-modes", nargs="+", default=["norm_attn_head", "last_block_norm_attn_head"])
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = reference_split(load_v2_metadata(args.metadata_file))
    encoder = LabelEncoder()
    ref["label_id"] = encoder.fit_transform(ref["model_label"])
    classes = list(encoder.classes_)
    x, masks = build_arrays(ref, args.cache_dir, "reference_supervised_v2", args.refresh_cache, args.baseline)
    y = ref["label_id"].to_numpy(dtype=np.int64)
    train_idx = np.where(ref["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(ref["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(ref["split_main"].eq("test").to_numpy())[0]
    x_train_aug, masks_train_aug, y_train_aug, aug_summary = make_balanced_augmented_train(
        x[train_idx], masks[train_idx], y[train_idx], args.min_per_class, args.max_per_class
    )
    aug_summary["class"] = aug_summary["label_id"].map({i: c for i, c in enumerate(classes)})
    aug_summary.to_csv(args.out_dir / "band_aware_shared_augmentation_summary.csv", index=False, encoding="utf-8-sig")

    train_ds = RamanDataset(x_train_aug, masks_train_aug, y_train_aug, augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx], augment=False)

    trials = [
        ("band_mst_d96_l3_p8_lr1e4_s1", {"d_model": 96, "layers": 3, "patch_size": 8, "lr": 1e-4, "saliency_init": 1.0}),
        ("band_mst_d96_l3_p4_lr1e4_s1", {"d_model": 96, "layers": 3, "patch_size": 4, "lr": 1e-4, "saliency_init": 1.0}),
        ("band_mst_d128_l4_p8_lr3e5_s1", {"d_model": 128, "layers": 4, "patch_size": 8, "lr": 3e-5, "saliency_init": 1.0}),
        ("band_mst_d128_l4_p8_lr1e4_s05", {"d_model": 128, "layers": 4, "patch_size": 8, "lr": 1e-4, "saliency_init": 0.5}),
    ]
    rows = []
    for name, params in trials:
        print(f"Training {name}: {params}", flush=True)
        model = BandAwareMST(
            num_classes=len(classes),
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch_size"]),
            saliency_init=float(params["saliency_init"]),
        )
        metrics = train_torch_model(
            name,
            model,
            train_ds,
            val_ds,
            test_ds,
            classes,
            args.out_dir,
            args.epochs,
            args.batch_size,
            float(params["lr"]),
            device,
            augment=False,
        )
        rows.append({"trial": name, "params": json.dumps(params), **metrics})
        pd.DataFrame(rows).to_csv(args.out_dir / "band_aware_mst_trials.csv", index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False)
    result_df.to_csv(args.out_dir / "band_aware_mst_trials_sorted_by_validation.csv", index=False, encoding="utf-8-sig")
    best = result_df.iloc[0]
    best_config = json.loads(best["params"])
    best_checkpoint = args.out_dir / f"{best['trial']}.pth"
    manifest = {
        "architecture": "BandAwareMST",
        "selection_rule": "best reference validation macro-F1",
        "metadata_file": str(args.metadata_file),
        "classes": classes,
        "best_trial": best["trial"],
        "best_config": best_config,
        "best_checkpoint": str(best_checkpoint),
        "reference_rows": int(len(ref)),
        "reference_train_val_test": [int(len(train_idx)), int(len(val_idx)), int(len(test_idx))],
        "shared_training_augmentation": {
            "min_per_class": args.min_per_class,
            "max_per_class": args.max_per_class,
            "original_train_rows": int(len(train_idx)),
            "augmented_train_rows": int(len(y_train_aug)),
            "applied_to_reference_training_only": True,
            "band_positions_shifted": False,
            "protocol": AUGMENTATION_PROTOCOL,
        },
    }

    if args.run_sherloc:
        pooled = make_pooled_sherloc(args.metadata_file)
        all_classes = sorted(set(classes) | set(pooled["model_label"].unique()))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        y_s = pooled["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)
        sherloc_out = args.out_dir / "sherloc_pooled_random_validation"
        sherloc_cache = args.out_dir / "_sherloc_preprocessing_cache"
        sherloc_out.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(sherloc_out / "pooled_sherloc_labeled_samples.csv", index=False, encoding="utf-8-sig")
        x_s, masks_s, stats = build_sherloc_arrays(
            pooled,
            sherloc_cache,
            "band_aware_sherloc_pooled_labeled",
            "despike_sg11_asls",
            args.refresh_cache,
        )
        stats.to_csv(sherloc_out / "pooled_sherloc_preprocessing_stats.csv", index=False, encoding="utf-8-sig")

        sherloc_rows = []
        for seed in args.sherloc_seeds:
            train_s, val_s, split_note = split_indices(y_s, seed, 0.2)
            for mode in args.sherloc_modes:
                print(f"Fine-tuning BandAwareMST on pooled SHERLOC seed={seed}, mode={mode}", flush=True)
                res = train_sherloc_split(
                    best_checkpoint,
                    best_config,
                    classes,
                    all_classes,
                    x_s,
                    masks_s,
                    y_s,
                    train_s,
                    val_s,
                    mode,
                    args.sherloc_epochs,
                    5e-5,
                    args.sherloc_batch_size,
                    device,
                    sherloc_out,
                    f"seed{seed}",
                )
                res["seed"] = seed
                res["split_note"] = split_note
                sherloc_rows.append(res)
                pd.DataFrame(sherloc_rows).to_csv(
                    sherloc_out / "band_aware_pooled_random_validation_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        sherloc_summary = pd.DataFrame(sherloc_rows)
        sherloc_agg = (
            sherloc_summary.groupby("mode")[["accuracy", "macro_f1", "weighted_f1", "present_label_macro_f1"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        sherloc_agg.to_csv(sherloc_out / "band_aware_pooled_random_validation_aggregate.csv", index=False, encoding="utf-8-sig")
        manifest["sherloc_pooled_random_validation"] = {
            "seeds": args.sherloc_seeds,
            "modes": args.sherloc_modes,
            "rows": int(len(pooled)),
            "all_classes": all_classes,
        }

    (args.out_dir / "band_aware_mst_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(result_df.to_string(index=False))
    print(f"Saved BandAwareMST experiments to: {args.out_dir}")


if __name__ == "__main__":
    main()
