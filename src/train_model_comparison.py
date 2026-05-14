from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"d:/dyt/raman/pigeonite")
METADATA_FILE = ROOT / "data" / "metadata" / "metadata_parent_945.csv"
OUT_DIR = ROOT / "results" / "model_comparison_runs"

SEED = 2024
MAX_SHIFT = 4000.0
GRID_POINTS = 4100
GRID = np.linspace(0.0, MAX_SHIFT, GRID_POINTS, dtype=np.float32)

AUGMENTATION_PROTOCOL = {
    "scope": "training split only; validation and test spectra are never augmented",
    "physical_principle": "Raman band centers are not translated because band positions are diagnostic for a mineral phase.",
    "band_detection": {
        "signal": "baseline-corrected, max-normalized intensity within the valid spectral range",
        "minimum_height": 0.05,
        "minimum_prominence": 0.03,
        "minimum_distance_cm-1": 8.0,
        "maximum_bands_per_spectrum": 12,
        "fallback": "if scipy.signal.find_peaks is unavailable, local maxima satisfying the same height threshold are used",
    },
    "transforms": {
        "gamma_intensity_response": {"probability": 0.70, "gamma_range": [0.75, 1.35]},
        "band_envelope_intensity_perturbation": {
            "probability": 0.20,
            "amplitude_range": [-0.08, 0.08],
            "sigma_cm-1_range": [4.0, 10.0],
            "center_shift_cm-1": 0.0,
        },
        "residual_baseline": {
            "probability": 0.50,
            "polynomial_order": 2,
            "coefficient_std": [0.015, 0.020, 0.015],
        },
        "gaussian_read_noise": {"probability": 0.80, "sigma_range_after_normalization": [0.005, 0.025]},
        "symmetric_broadening": {
            "probability": 0.35,
            "kernel": [0.08, 0.18, 0.48, 0.18, 0.08],
            "mixing_alpha_range": [0.25, 0.65],
        },
        "weak_band_attenuation": {
            "probability": 0.25,
            "windows_per_spectrum": [1, 3],
            "half_width_points_range": [8, 35],
            "attenuation_factor_range": [0.75, 0.95],
        },
    },
    "constraints": [
        "no wavenumber-axis translation or interpolation jitter is applied during augmentation",
        "regions outside original spectral coverage remain zeroed and masked",
        "negative intensities are clipped to zero after perturbation",
        "each augmented spectrum is renormalized to unit maximum within the valid range",
        "first-derivative features are recomputed after intensity augmentation",
    ],
}

EXCLUDE_QC_STATUSES = {"external_domain"}
EXCLUDE_LABELS_CURATED = {"Halides"}
PHYLL_LABELS = {"Clay", "Mica", "Serpentine"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_asls(intensity: np.ndarray) -> np.ndarray:
    try:
        from pybaselines.whittaker import asls
    except ImportError:
        return intensity
    try:
        base, _ = asls(intensity, lam=1e7, p=0.01)
        return intensity - base
    except Exception:
        return intensity


def fix_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    shift = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    inten = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(shift) & np.isfinite(inten)
    shift = shift[ok]
    inten = inten[ok]
    if len(shift) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    unique_shift, unique_idx = np.unique(shift, return_index=True)
    inten = inten[unique_idx]
    order = np.argsort(unique_shift)
    return unique_shift[order], inten[order]


def remove_polynomial_baseline(intensity: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not np.any(valid_mask):
        return intensity
    x = GRID[valid_mask].astype(np.float64)
    y = intensity[valid_mask].astype(np.float64)
    if len(x) < 8:
        return intensity
    # Fit the lower envelope to avoid treating Raman bands as baseline.
    cutoff = np.percentile(y, 35)
    low = y <= cutoff
    if np.sum(low) < 4:
        low = np.ones_like(y, dtype=bool)
    coef = np.polyfit(x[low], y[low], deg=2)
    corrected = intensity - np.polyval(coef, GRID)
    return corrected


def preprocess_spectrum(path: str, baseline: str = "poly", smooth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shift, inten = read_spectrum(path)
    if len(shift) < 2:
        empty = np.zeros_like(GRID, dtype=np.float32)
        invalid = np.ones_like(GRID, dtype=bool)
        return np.stack([empty, empty, empty], axis=-1), GRID.copy(), invalid

    valid_mask = (GRID >= float(np.min(shift))) & (GRID <= float(np.max(shift)))
    interp = np.interp(GRID, shift, inten, left=0.0, right=0.0).astype(np.float64)

    if baseline == "asls" and len(interp) > 3:
        interp = run_asls(interp)
    elif baseline == "poly":
        interp = remove_polynomial_baseline(interp, valid_mask)

    if smooth and savgol_filter is not None and len(interp) > 21:
        interp = savgol_filter(interp, window_length=21, polyorder=2)

    interp = np.maximum(interp, 0.0)
    maxv = float(np.max(interp[valid_mask])) if np.any(valid_mask) else float(np.max(interp))
    if maxv <= 0:
        maxv = 1.0
    intensity = (interp / (maxv + 1e-12)).astype(np.float32)
    intensity[~valid_mask] = 0.0

    if np.all(np.diff(GRID) > 0):
        deriv = np.gradient(intensity, GRID).astype(np.float32)
    else:
        deriv = np.zeros_like(intensity, dtype=np.float32)
    max_abs_deriv = float(np.max(np.abs(deriv[valid_mask]))) if np.any(valid_mask) else float(np.max(np.abs(deriv)))
    if max_abs_deriv > 1e-9:
        deriv = deriv / max_abs_deriv
    deriv[~valid_mask] = 0.0

    valid_channel = valid_mask.astype(np.float32)
    features = np.stack([intensity, deriv, valid_channel], axis=-1).astype(np.float32)
    key_padding_mask = ~valid_mask
    return features, GRID.copy(), key_padding_mask


def make_label(row: pd.Series, label_scheme: str) -> str:
    label = str(row["major_category"])
    if label_scheme == "curated" and label in PHYLL_LABELS:
        return "Phyllosilicates"
    return label


def load_metadata(label_scheme: str, include_review_required: bool, metadata_file: Path = METADATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(metadata_file)
    df = df[df["file_exists"].astype(bool)].copy()
    df = df[~df["qc_status"].isin(EXCLUDE_QC_STATUSES)].copy()

    if label_scheme == "curated":
        df = df[~df["major_category"].isin(EXCLUDE_LABELS_CURATED)].copy()
        if not include_review_required:
            # Keep phyllosilicates because their labels are harmonized below.
            mask_review = df["qc_status"].eq("review_required") & ~df["major_category"].isin(PHYLL_LABELS)
            df = df[~mask_review].copy()
    elif not include_review_required:
        df = df[~df["qc_status"].eq("review_required")].copy()

    df["model_label"] = df.apply(lambda r: make_label(r, label_scheme), axis=1)
    df = df[df["split_main"].isin(["train", "val", "test"])].copy()
    return df.reset_index(drop=True)


def build_cache(df: pd.DataFrame, cache_dir: Path, refresh: bool, baseline: str, smooth: bool, label_scheme: str) -> Path:
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_name = f"preprocessed_{label_scheme}_baseline-{baseline}_smooth-{int(smooth)}"
    cache_path = cache_dir / f"{cache_name}.npz"
    meta_path = cache_dir / f"{cache_name}.csv"
    if cache_path.exists() and meta_path.exists() and not refresh:
        return cache_path

    features = []
    masks = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            log(f"Preprocessing spectrum {i}/{total}")
        x, _, mask = preprocess_spectrum(str(row["file_path"]), baseline=baseline, smooth=smooth)
        features.append(x)
        masks.append(mask)

    np.savez_compressed(
        cache_path,
        features=np.stack(features, axis=0).astype(np.float32),
        masks=np.stack(masks, axis=0).astype(bool),
        grid=GRID.astype(np.float32),
    )
    df[["spectrum_id", "file_name_clean", "model_label", "split_main", "parent_group", "source_type"]].to_csv(
        meta_path, index=False, encoding="utf-8-sig"
    )
    return cache_path


class RamanDataset(Dataset):
    def __init__(self, x: np.ndarray, masks: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = x.astype(np.float32)
        self.masks = masks.astype(bool)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.grid = np.broadcast_to(GRID.reshape(1, -1), (len(self.x), GRID_POINTS)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx].copy()
        mask = self.masks[idx].copy()
        if self.augment:
            x = augment_raman_features(x, mask)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.grid[idx]),
            torch.from_numpy(mask),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def augment_raman_features(x: np.ndarray, key_padding_mask: np.ndarray) -> np.ndarray:
    """Train-time Raman augmentation without shifting band positions.

    Raman band positions are treated as physically stable for a given mineral
    phase. The transforms below only change intensity response, fluorescence-like
    baseline, noise level, and band width/weakness.
    """
    valid = ~key_padding_mask
    if not np.any(valid):
        return x

    intensity = x[:, 0].copy()
    derivative = x[:, 1].copy()

    # Relative intensity response changes across instruments/excitation wavelengths.
    if np.random.rand() < 0.7:
        gamma = np.random.uniform(0.75, 1.35)
        intensity[valid] = np.power(np.clip(intensity[valid], 0.0, 1.0), gamma)

    # Local band-envelope perturbation changes relative band strength but not band center.
    if np.random.rand() < 0.2:
        valid_idx = np.where(valid)[0]
        y_valid = intensity[valid_idx]
        grid_step = float(np.median(np.diff(GRID))) if len(GRID) > 1 else 1.0
        min_distance = max(1, int(round(8.0 / grid_step)))
        if find_peaks is not None:
            peaks_local, props = find_peaks(y_valid, height=0.05, prominence=0.03, distance=min_distance)
            if len(peaks_local) > 12 and "prominences" in props:
                keep = np.argsort(props["prominences"])[-12:]
                peaks_local = peaks_local[keep]
        else:
            candidate = np.where((y_valid[1:-1] > y_valid[:-2]) & (y_valid[1:-1] >= y_valid[2:]) & (y_valid[1:-1] >= 0.05))[0] + 1
            peaks_local = candidate[::min_distance][:12]
        for peak_local in peaks_local:
            center_idx = valid_idx[int(peak_local)]
            sigma_cm = np.random.uniform(4.0, 10.0)
            amp = np.random.uniform(-0.08, 0.08)
            envelope = np.exp(-0.5 * ((GRID - GRID[center_idx]) / sigma_cm) ** 2).astype(np.float32)
            intensity[valid] = intensity[valid] * (1.0 + amp * envelope[valid])

    # Fluorescence/background residual after correction. No band displacement.
    if np.random.rand() < 0.5:
        xv = np.linspace(-1.0, 1.0, int(np.sum(valid)), dtype=np.float32)
        coef = np.random.normal(0.0, [0.015, 0.02, 0.015]).astype(np.float32)
        baseline = coef[0] + coef[1] * xv + coef[2] * xv * xv
        intensity[valid] = intensity[valid] + baseline

    # Shot/read noise at a conservative normalized scale.
    if np.random.rand() < 0.8:
        sigma = np.random.uniform(0.005, 0.025)
        intensity[valid] = intensity[valid] + np.random.normal(0.0, sigma, int(np.sum(valid))).astype(np.float32)

    # Mild broadening/denoising with symmetric kernels preserves band centers.
    if np.random.rand() < 0.35:
        kernel = np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float32)
        smoothed = np.convolve(intensity, kernel, mode="same")
        alpha = np.random.uniform(0.25, 0.65)
        intensity[valid] = (1.0 - alpha) * intensity[valid] + alpha * smoothed[valid]

    # Weak-band/dropout simulation: attenuate small random windows, not translate them.
    if np.random.rand() < 0.25:
        valid_idx = np.where(valid)[0]
        for _ in range(np.random.randint(1, 4)):
            center = np.random.choice(valid_idx)
            width = np.random.randint(8, 35)
            lo = max(valid_idx[0], center - width)
            hi = min(valid_idx[-1] + 1, center + width)
            intensity[lo:hi] *= np.random.uniform(0.75, 0.95)

    intensity[~valid] = 0.0
    intensity = np.maximum(intensity, 0.0)
    maxv = float(np.max(intensity[valid]))
    if maxv > 0:
        intensity = intensity / (maxv + 1e-12)
    derivative = np.gradient(intensity, GRID).astype(np.float32)
    max_abs = float(np.max(np.abs(derivative[valid])))
    if max_abs > 1e-9:
        derivative = derivative / max_abs
    derivative[~valid] = 0.0

    out = x.copy()
    out[:, 0] = intensity.astype(np.float32)
    out[:, 1] = derivative.astype(np.float32)
    return out


class FixedPositionEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, shifts: torch.Tensor) -> torch.Tensor:
        angles = shifts.unsqueeze(-1) * self.div_term
        pos = torch.zeros(*shifts.shape, self.div_term.shape[-1] * 2, device=shifts.device, dtype=shifts.dtype)
        pos[:, :, 0::2] = torch.sin(angles)
        pos[:, :, 1::2] = torch.cos(angles)
        return pos


class PatchProjector(nn.Module):
    def __init__(self, in_chans: int, d_model: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, d_model, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, C] -> tokens: [B, T, D]
        x_ch = x.transpose(1, 2)
        tokens = self.proj(x_ch).transpose(1, 2)
        valid = (~mask).float().unsqueeze(1)
        valid_count = F.avg_pool1d(valid, kernel_size=self.patch_size, stride=self.patch_size).squeeze(1)
        token_mask = valid_count <= 0.0
        token_shifts = F.avg_pool1d(shifts.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size).squeeze(1)
        return tokens, token_shifts, token_mask


class MaskedSpectralTransformer(nn.Module):
    def __init__(self, num_classes: int, in_chans: int = 3, d_model: int = 96, nhead: int = 4, layers: int = 3, patch_size: int = 8):
        super().__init__()
        self.patch = PatchProjector(in_chans, d_model, patch_size=patch_size)
        self.pos_encoder = FixedPositionEncoder(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, token_shifts, token_mask = self.patch(x, shifts, mask)
        h = h + self.pos_encoder(token_shifts)
        h = h.masked_fill(token_mask.unsqueeze(-1), 0.0)
        h = self.encoder(h, src_key_padding_mask=token_mask)
        valid = (~token_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(self.norm(pooled))


class StandardTransformer(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = GRID_POINTS, in_chans: int = 3, d_model: int = 96, nhead: int = 4, layers: int = 3, patch_size: int = 8):
        super().__init__()
        self.patch = PatchProjector(in_chans, d_model, patch_size=patch_size)
        token_len = seq_len // patch_size
        self.index_pos = nn.Embedding(token_len, d_model)
        self.register_buffer("idx", torch.arange(token_len))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _, token_mask = self.patch(x, shifts, mask)
        h = h + self.index_pos(self.idx[: h.shape[1]])
        h = h.masked_fill(token_mask.unsqueeze(-1), 0.0)
        h = self.encoder(h, src_key_padding_mask=token_mask)
        valid = (~token_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(self.norm(pooled))


class RamanCNN(nn.Module):
    def __init__(self, num_classes: int, in_chans: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 192, kernel_size=7, padding=3),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(192, num_classes))

    def forward(self, x: torch.Tensor, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)
        return self.head(self.features(h))


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_arrays(y_true: np.ndarray, probs: np.ndarray, classes: Iterable[str], out_prefix: Path) -> dict:
    classes = list(classes)
    pred = np.argmax(probs, axis=1)
    label_ids = np.arange(len(classes))
    report = classification_report(
        y_true, pred, labels=label_ids, target_names=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_true, pred, labels=label_ids, target_names=classes, zero_division=0)
    out_prefix.with_suffix(".classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report).T.to_csv(out_prefix.with_suffix(".classification_report.csv"), encoding="utf-8-sig")

    cm = confusion_matrix(y_true, pred, labels=label_ids)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_prefix.with_suffix(".confusion_matrix.csv"), encoding="utf-8-sig")

    threshold_rows = []
    conf = np.max(probs, axis=1)
    correct = pred == y_true
    for threshold in np.linspace(0.0, 0.95, 20):
        keep = conf >= threshold
        coverage = float(np.mean(keep))
        if np.any(keep):
            acc = float(np.mean(correct[keep]))
            macro_f1 = float(f1_score(y_true[keep], pred[keep], average="macro", zero_division=0))
        else:
            acc = np.nan
            macro_f1 = np.nan
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accepted_n": int(np.sum(keep)),
                "accuracy_on_accepted": acc,
                "macro_f1_on_accepted": macro_f1,
                "rejected_n": int(np.sum(~keep)),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(out_prefix.with_suffix(".threshold_sweep.csv"), index=False, encoding="utf-8-sig")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(classes)), zero_division=0
    )
    pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(out_prefix.with_suffix(".per_class.csv"), index=False, encoding="utf-8-sig")

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }


def train_torch_model(
    model_name: str,
    model: nn.Module,
    train_ds: RamanDataset,
    val_ds: RamanDataset,
    test_ds: RamanDataset,
    classes: list[str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    augment: bool,
) -> dict:
    out_dir.mkdir(exist_ok=True, parents=True)
    train_ds.augment = augment
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds.y, len(classes)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, shifts, mask, y in train_loader:
            x, shifts, mask, y = x.to(device), shifts.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, shifts, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_probs, val_true = predict_torch(model, val_loader, device)
        val_macro = f1_score(val_true, np.argmax(val_probs, axis=1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_macro_f1": float(val_macro)})
        if val_macro > best_val:
            best_val = float(val_macro)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / f"{model_name}.pth")
    pd.DataFrame(history).to_csv(out_dir / f"{model_name}.history.csv", index=False, encoding="utf-8-sig")

    test_probs, test_true = predict_torch(model, test_loader, device)
    metrics = evaluate_arrays(test_true, test_probs, classes, out_dir / f"{model_name}.test")
    metrics["best_val_macro_f1"] = best_val
    return metrics


def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    true = []
    with torch.no_grad():
        for x, shifts, mask, y in loader:
            logits = model(x.to(device), shifts.to(device), mask.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            true.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(true, axis=0)


def flatten_features(x: np.ndarray, stride: int = 4) -> np.ndarray:
    # Use intensity and derivative. The valid-mask channel is intentionally omitted
    # from chemometric baselines to keep them closer to conventional Raman features.
    stride = max(1, int(stride))
    return x[:, ::stride, :2].reshape(x.shape[0], -1)


class PLSDA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model = PLSRegression(n_components=n_components)
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        y_bin = label_binarize(y, classes=self.classes_)
        if y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y_bin)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred = self.model.predict(self.scaler.transform(x))
        pred = np.maximum(pred, 0.0)
        denom = pred.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        return pred / denom


def run_sklearn_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    out_dir: Path,
    models: list[str],
) -> dict[str, dict]:
    out_dir.mkdir(exist_ok=True, parents=True)
    results = {}
    n_classes = len(classes)
    max_pca = max(2, min(50, x_train.shape[0] - 1, x_train.shape[1]))
    max_pls = max(2, min(20, n_classes - 1, x_train.shape[0] - 1, x_train.shape[1]))

    sklearn_models = {}
    if "pca_svm" in models:
        sklearn_models["pca_svm"] = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=max_pca, random_state=SEED)),
                ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
            ]
        )
    if "pls_da" in models:
        sklearn_models["pls_da"] = PLSDA(n_components=max_pls)
    if "random_forest" in models:
        sklearn_models["random_forest"] = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
            max_features="sqrt",
        )

    for name, model in sklearn_models.items():
        start = time.time()
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)
        metrics = evaluate_arrays(y_test, probs, classes, out_dir / f"{name}.test")
        metrics["train_seconds"] = float(time.time() - start)
        results[name] = metrics
    return results


def save_experiment_manifest(args: argparse.Namespace, df: pd.DataFrame, classes: list[str], out_dir: Path) -> None:
    manifest = {
        "metadata_file": str(args.metadata_file),
        "label_scheme": args.label_scheme,
        "include_review_required": args.include_review_required,
        "grid": {"min_cm-1": 0.0, "max_cm-1": MAX_SHIFT, "points": GRID_POINTS},
        "preprocessing": {
            "baseline": args.baseline,
            "smooth_savgol": bool(args.smooth),
            "normalization": "nonnegative max normalization within valid spectral range",
            "features": ["intensity", "first_derivative", "valid_mask_channel_for_deep_models"],
            "masking": "regions outside each spectrum original wavenumber coverage are zeroed and masked for transformer pooling",
        },
        "classes": classes,
        "split_counts": df["split_main"].value_counts().to_dict(),
        "class_by_split": pd.crosstab(df["model_label"], df["split_main"]).to_dict(),
        "models": args.models,
        "chemometric_stride": args.chemometric_stride,
        "train_time_augmentation": {
            "enabled": bool(args.augment),
            "protocol": AUGMENTATION_PROTOCOL,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": SEED,
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_dir / "experiment_samples.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raman model comparison experiment.")
    parser.add_argument("--label-scheme", choices=["curated", "original_major"], default="curated")
    parser.add_argument("--include-qc-required", dest="include_review_required", action="store_true")
    parser.add_argument("--models", nargs="+", default=["pca_svm", "pls_da", "cnn", "standard_transformer", "mst"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--baseline", choices=["none", "poly", "asls"], default="poly")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--chemometric-stride", type=int, default=4)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.out_dir / f"{args.label_scheme}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True, parents=True)
    log(f"Run directory: {run_dir}")
    log(f"Using device: {device}")

    df = load_metadata(args.label_scheme, args.include_review_required, args.metadata_file)
    log(f"Loaded {len(df)} spectra after QC/label filtering")
    classes = sorted(df["model_label"].unique())
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["model_label"])
    classes = list(label_encoder.classes_)
    log(f"Classes ({len(classes)}): {classes}")

    save_experiment_manifest(args, df, classes, run_dir)
    log("Building/loading preprocessing cache")
    cache_path = build_cache(df, args.out_dir / "_cache", args.refresh_cache, args.baseline, args.smooth, args.label_scheme)
    log(f"Cache ready: {cache_path}")
    cache = np.load(cache_path)
    x = cache["features"]
    masks = cache["masks"]
    y = df["label_id"].to_numpy(dtype=np.int64)

    train_idx = np.where(df["split_main"].eq("train").to_numpy())[0]
    val_idx = np.where(df["split_main"].eq("val").to_numpy())[0]
    test_idx = np.where(df["split_main"].eq("test").to_numpy())[0]

    train_ds = RamanDataset(x[train_idx], masks[train_idx], y[train_idx], augment=args.augment)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx])
    test_ds = RamanDataset(x[test_idx], masks[test_idx], y[test_idx])

    results: dict[str, dict] = {}
    sklearn_names = [m for m in args.models if m in {"pca_svm", "pls_da", "random_forest"}]
    if sklearn_names:
        log(f"Training sklearn baselines: {sklearn_names}")
        results.update(
            run_sklearn_models(
                flatten_features(x[train_idx], args.chemometric_stride),
                y[train_idx],
                flatten_features(x[test_idx], args.chemometric_stride),
                y[test_idx],
                classes,
                run_dir / "sklearn",
                sklearn_names,
            )
        )

    torch_models = {}
    if "cnn" in args.models:
        torch_models["cnn"] = RamanCNN(num_classes=len(classes))
    if "standard_transformer" in args.models:
        torch_models["standard_transformer"] = StandardTransformer(num_classes=len(classes))
    if "mst" in args.models:
        torch_models["mst"] = MaskedSpectralTransformer(num_classes=len(classes))

    for name, model in torch_models.items():
        log(f"Training torch model: {name}")
        start = time.time()
        metrics = train_torch_model(
            name,
            model,
            train_ds,
            val_ds,
            test_ds,
            classes,
            run_dir / "torch",
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            args.augment,
        )
        metrics["train_seconds"] = float(time.time() - start)
        results[name] = metrics

    if results:
        pd.DataFrame(results).T.sort_values("macro_f1", ascending=False).to_csv(
            run_dir / "model_comparison_summary.csv", encoding="utf-8-sig"
        )
        print(pd.DataFrame(results).T.sort_values("macro_f1", ascending=False).to_string())
    else:
        (run_dir / "model_comparison_summary.csv").write_text("model,accuracy,macro_f1,weighted_f1\n", encoding="utf-8")
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
