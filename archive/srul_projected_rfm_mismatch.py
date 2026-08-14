"""
Projected-RFM geometry-mismatch experiment for SRUL.

Purpose
-------
This experiment tests whether a spherical Riemannian prior is well matched to
an autoencoder whose latent tokens are *not* constrained to a sphere.

The experiment reuses a trained Euclidean spatial autoencoder. For each latent
 token h[b,:,i,j], it separates radius and direction:

    r = ||h||_2,        u = h / ||h||_2.

A conditional Riemannian Flow Matching (RFM) prior is trained only on u. At
sampling time, the generated direction is multiplied by a class-conditional,
spatial mean radius estimated from the training set before the Euclidean
decoder is applied. This gives the direction-only spherical prior a generous,
but still generation-available, radius estimate.

The script also performs a radius-information diagnostic before training the
prior:

1. Original reconstruction:       D(h)
2. Mean-radius reconstruction:    D(r_bar[y,i,j] * h/||h||)
3. Unit-radius reconstruction:    D(h/||h||)

If replacing the sample-specific radius damages reconstruction, the Euclidean
autoencoder uses radial information and a direction-only RFM prior is
structurally mismatched.

Recommended comparison
----------------------
- Matched spherical:  spherical autoencoder + RFM (existing SRUL result)
- Matched Euclidean:  Euclidean autoencoder + Euclidean FM (existing result)
- Mismatched:         Euclidean autoencoder + projected direction-only RFM

Required helper file
--------------------
Put this script beside:
    srul_medmnist_conditional_experiment.py

The helper contains the shared CIFAR-10 loader, spatial architecture,
product-of-spheres geometry, metrics, and conditional vector field.

Example (Google Colab)
----------------------
python srul_projected_rfm_mismatch.py \
  --out-dir /content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10_projected_rfm \
  --euclidean-ae-checkpoint /content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/checkpoints/euclidean_fm_autoencoder_final.pt \
  --previous-metrics-csv /content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/generation_metrics.csv \
  --prior-epochs 120 --guidance-scale 2.0 --resume --amp
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.utils import save_image


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def save_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def atomic_torch_save(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def load_model_state(path: Path, device: torch.device, prefer_ema: bool = False):
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if prefer_ema and "ema" in obj:
        return obj["ema"]
    if "model" in obj:
        return obj["model"]
    return obj


def import_base_module():
    """Import the shared project implementation from /content or this folder."""
    module_name = "srul_medmnist_conditional_experiment"
    candidates = [
        Path(__file__).resolve().parent,
        Path("/content"),
        Path.cwd(),
    ]
    for directory in candidates:
        directory_str = str(directory)
        if directory_str not in sys.path:
            sys.path.insert(0, directory_str)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import srul_medmnist_conditional_experiment. "
            "Put srul_medmnist_conditional_experiment.py in the same Colab "
            "directory as this script."
        ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Config:
    dataset: str = "cifar10"
    data_root: str = "/content/data"
    out_dir: str = "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10_projected_rfm"
    seed: int = 0
    num_classes: int = 10
    train_samples: Optional[int] = None
    test_samples: Optional[int] = 10000

    batch_size: int = 128
    metric_batch_size: int = 128
    num_workers: int = 2
    base_channels: int = 96
    latent_channels: int = 32
    latent_size: int = 4

    prior_epochs: int = 120
    prior_lr: float = 2e-4
    prior_width: int = 256
    prior_depth: int = 6
    time_dim: int = 128
    time_sampling: str = "logit_normal"
    label_drop_prob: float = 0.10
    ema_decay: float = 0.999
    guidance_scale: float = 2.0
    sample_steps: int = 100
    direction_noise: float = 0.0

    radius_mode: str = "class_token_mean"
    radius_max_samples: int = 50000

    metric_samples: int = 10000
    pr_samples: int = 5000
    reconstruction_metric_samples: int = 5000
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5

    euclidean_ae_checkpoint: str = ""
    previous_metrics_csv: Optional[str] = None

    checkpoint_every: int = 5
    resume: bool = True
    amp: bool = True
    skip_heavy_metrics: bool = False
    eval_only: bool = False


# -----------------------------------------------------------------------------
# Dataset and Euclidean autoencoder
# -----------------------------------------------------------------------------


class EuclideanAutoencoder(nn.Module):
    """Same spatial encoder/decoder as SRUL, without token normalization."""

    def __init__(self, base, latent_channels: int, base_channels: int):
        super().__init__()
        self.encoder = base.SpatialEncoder(latent_channels, base_channels)
        self.decoder = base.SpatialDecoder(latent_channels, base_channels)
        self.latent_channels = int(latent_channels)
        self.latent_size = 4

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def make_loaders(base, config: Config, device: torch.device):
    if config.dataset == "fake32":
        train_n = config.train_samples or 96
        test_n = config.test_samples or 48
        generator = torch.Generator().manual_seed(config.seed)
        train_x = torch.rand(train_n, 3, 32, 32, generator=generator) * 2 - 1
        train_y = torch.arange(train_n) % config.num_classes
        test_x = torch.rand(test_n, 3, 32, 32, generator=generator) * 2 - 1
        test_y = torch.arange(test_n) % config.num_classes
        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
    else:
        train_set, test_set = base.make_datasets(
            name=config.dataset,
            root=config.data_root,
            train_samples=config.train_samples,
            test_samples=config.test_samples,
            seed=config.seed,
        )

    train_loader, test_loader = base.make_loaders(
        train_set=train_set,
        test_set=test_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    return train_set, test_set, train_loader, test_loader


# -----------------------------------------------------------------------------
# Radius statistics and diagnostic reconstructions
# -----------------------------------------------------------------------------


@dataclass
class RadiusStats:
    class_token_mean: torch.Tensor  # [K,H,W]
    class_token_std: torch.Tensor   # [K,H,W]
    global_token_mean: torch.Tensor # [H,W]
    global_token_std: torch.Tensor  # [H,W]
    class_counts: torch.Tensor      # [K]
    global_mean: float
    global_std: float
    global_cv: float
    mean_class_token_cv: float

    def cpu_dict(self) -> Dict[str, object]:
        return {
            "class_token_mean": self.class_token_mean.cpu(),
            "class_token_std": self.class_token_std.cpu(),
            "global_token_mean": self.global_token_mean.cpu(),
            "global_token_std": self.global_token_std.cpu(),
            "class_counts": self.class_counts.cpu(),
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "global_cv": self.global_cv,
            "mean_class_token_cv": self.mean_class_token_cv,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> "RadiusStats":
        return cls(
            class_token_mean=obj["class_token_mean"].cpu(),
            class_token_std=obj["class_token_std"].cpu(),
            global_token_mean=obj["global_token_mean"].cpu(),
            global_token_std=obj["global_token_std"].cpu(),
            class_counts=obj["class_counts"].cpu(),
            global_mean=float(obj["global_mean"]),
            global_std=float(obj["global_std"]),
            global_cv=float(obj["global_cv"]),
            mean_class_token_cv=float(obj["mean_class_token_cv"]),
        )

    def radius_map(self, labels: torch.Tensor, mode: str) -> torch.Tensor:
        labels = labels.long().cpu()
        if mode == "class_token_mean":
            return self.class_token_mean[labels]
        if mode == "global_token_mean":
            return self.global_token_mean[None].expand(labels.numel(), -1, -1)
        if mode == "unit":
            h, w = self.global_token_mean.shape
            return torch.ones(labels.numel(), h, w)
        raise ValueError(f"Unknown radius mode: {mode}")


@torch.no_grad()
def estimate_radius_stats(
    ae: EuclideanAutoencoder,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> RadiusStats:
    ae.eval()
    k = config.num_classes
    h = config.latent_size
    w = config.latent_size

    class_sum = torch.zeros(k, h, w, dtype=torch.float64)
    class_sq = torch.zeros_like(class_sum)
    class_counts = torch.zeros(k, dtype=torch.float64)
    global_sum = torch.zeros(h, w, dtype=torch.float64)
    global_sq = torch.zeros_like(global_sum)
    global_count = 0
    scalar_sum = 0.0
    scalar_sq = 0.0
    scalar_count = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().view(-1)
        radii = ae.encode(images).float().norm(dim=1).cpu().double()  # [B,H,W]

        remaining = config.radius_max_samples - seen
        if remaining <= 0:
            break
        radii = radii[:remaining]
        labels = labels[:remaining]
        batch = radii.size(0)
        seen += batch

        global_sum += radii.sum(dim=0)
        global_sq += radii.pow(2).sum(dim=0)
        global_count += batch
        scalar_sum += float(radii.sum().item())
        scalar_sq += float(radii.pow(2).sum().item())
        scalar_count += int(radii.numel())

        for class_id in range(k):
            mask = labels == class_id
            if mask.any():
                values = radii[mask]
                class_sum[class_id] += values.sum(dim=0)
                class_sq[class_id] += values.pow(2).sum(dim=0)
                class_counts[class_id] += int(mask.sum().item())

        if seen >= config.radius_max_samples:
            break

    if global_count == 0:
        raise RuntimeError("No latent radii were collected.")

    global_mean_map = global_sum / global_count
    global_var_map = (global_sq / global_count - global_mean_map.pow(2)).clamp_min(0)
    global_std_map = global_var_map.sqrt()

    class_mean = torch.empty_like(class_sum)
    class_std = torch.empty_like(class_sum)
    for class_id in range(k):
        if class_counts[class_id] > 0:
            count = class_counts[class_id]
            mean = class_sum[class_id] / count
            var = (class_sq[class_id] / count - mean.pow(2)).clamp_min(0)
            class_mean[class_id] = mean
            class_std[class_id] = var.sqrt()
        else:
            class_mean[class_id] = global_mean_map
            class_std[class_id] = global_std_map

    scalar_mean = scalar_sum / max(scalar_count, 1)
    scalar_var = max(scalar_sq / max(scalar_count, 1) - scalar_mean**2, 0.0)
    scalar_std = math.sqrt(scalar_var)
    scalar_cv = scalar_std / max(abs(scalar_mean), 1e-8)

    token_cv = class_std / class_mean.abs().clamp_min(1e-8)
    weights = class_counts[:, None, None].expand_as(token_cv)
    mean_class_token_cv = float(
        (token_cv * weights).sum().item() / weights.sum().clamp_min(1.0).item()
    )

    return RadiusStats(
        class_token_mean=class_mean.float(),
        class_token_std=class_std.float(),
        global_token_mean=global_mean_map.float(),
        global_token_std=global_std_map.float(),
        class_counts=class_counts.float(),
        global_mean=float(scalar_mean),
        global_std=float(scalar_std),
        global_cv=float(scalar_cv),
        mean_class_token_cv=float(mean_class_token_cv),
    )


def basic_reconstruction_metrics(real: torch.Tensor, recon: torch.Tensor, base) -> Dict[str, float]:
    mse = float(F.mse_loss(recon, real).item())
    psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))
    out = {"mse": mse, "psnr": psnr}
    if getattr(base, "structural_similarity_index_measure", None) is not None:
        out["ssim"] = float(
            base.structural_similarity_index_measure(
                recon, real, data_range=2.0
            ).item()
        )
    return out


def add_distribution_metrics(
    row: Dict[str, float],
    real: torch.Tensor,
    recon: torch.Tensor,
    base,
    config: Config,
    device: torch.device,
) -> None:
    if config.skip_heavy_metrics:
        return
    real_uint8 = base.to_uint8(real)
    recon_uint8 = base.to_uint8(recon)
    row.update(
        base.compute_fid_kid(
            real_uint8,
            recon_uint8,
            device=device,
            batch_size=config.metric_batch_size,
        )
    )
    n = min(config.pr_samples, real_uint8.size(0), recon_uint8.size(0))
    real_features = base.extract_resnet_features(
        real_uint8[:n], device=device, batch_size=config.metric_batch_size
    )
    recon_features = base.extract_resnet_features(
        recon_uint8[:n], device=device, batch_size=config.metric_batch_size
    )
    row.update(
        base.feature_precision_recall(
            real_features,
            recon_features,
            device=device,
            nearest_k=config.pr_nearest_k,
            chunk_size=config.pr_chunk_size,
        )
    )


@torch.no_grad()
def evaluate_radius_reconstruction(
    ae: EuclideanAutoencoder,
    radius_stats: RadiusStats,
    loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    figure_path: Path,
) -> Dict[str, Dict[str, float]]:
    ae.eval()
    real_list: List[torch.Tensor] = []
    original_list: List[torch.Tensor] = []
    mean_radius_list: List[torch.Tensor] = []
    unit_radius_list: List[torch.Tensor] = []
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().view(-1)
        h = ae.encode(images)
        directions = base.normalize_tokens(h)
        radius_map = radius_stats.radius_map(labels, config.radius_mode).to(device)

        original = ae.decode(h)
        mean_radius = ae.decode(directions * radius_map[:, None])
        unit_radius = ae.decode(directions)

        remaining = config.reconstruction_metric_samples - count
        if remaining <= 0:
            break
        real_list.append(images[:remaining].cpu())
        original_list.append(original[:remaining].cpu())
        mean_radius_list.append(mean_radius[:remaining].cpu())
        unit_radius_list.append(unit_radius[:remaining].cpu())
        count += min(images.size(0), remaining)
        if count >= config.reconstruction_metric_samples:
            break

    real = torch.cat(real_list, dim=0)
    original = torch.cat(original_list, dim=0)
    mean_radius = torch.cat(mean_radius_list, dim=0)
    unit_radius = torch.cat(unit_radius_list, dim=0)

    rows: Dict[str, Dict[str, float]] = {}
    for name, recon in [
        ("original_euclidean_latent", original),
        ("class_mean_radius_direction", mean_radius),
        ("unit_radius_direction", unit_radius),
    ]:
        row = basic_reconstruction_metrics(real, recon, base)
        add_distribution_metrics(row, real, recon, base, config, device)
        row["num_samples"] = float(real.size(0))
        rows[name] = row

    n = min(12, real.size(0))
    grid = torch.cat(
        [real[:n], original[:n], mean_radius[:n], unit_radius[:n]], dim=0
    )
    save_image((grid + 1.0) * 0.5, figure_path, nrow=n)
    return rows


# -----------------------------------------------------------------------------
# Projected RFM prior
# -----------------------------------------------------------------------------


def apply_label_dropout(labels: torch.Tensor, null_label: int, probability: float):
    out = labels.clone().long()
    mask = torch.rand(labels.size(0), device=labels.device) < probability
    out[mask] = null_label
    return out, mask


def projected_rfm_objective(
    field: nn.Module,
    ae: EuclideanAutoencoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    base,
    config: Config,
):
    with torch.no_grad():
        h = ae.encode(images)
        target = base.normalize_tokens(h)
        if config.direction_noise > 0:
            target = base.tangent_noise_tokens(
                target, config.direction_noise, True
            )

    source = base.sample_uniform_product_spheres(
        images.size(0),
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        images.device,
    )
    t = base.sample_time(images.size(0), images.device, config.time_sampling)
    zt, omega = base.slerp_tokens(target, source, t)
    target_velocity = base.slerp_velocity_tokens(target, source, t, omega)
    target_velocity = base.tangent_projection_tokens(zt, target_velocity)

    training_labels, drop_mask = apply_label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    raw = field(zt, t, training_labels)
    prediction = base.tangent_projection_tokens(zt, raw)
    loss = (prediction - target_velocity).pow(2).mean()

    norms = base.token_norms(zt)
    radial_fraction = base.radial_velocity_fraction_tokens(zt, raw)
    return loss, {
        "loss": float(loss.detach().item()),
        "mean_path_norm": float(norms.mean().detach().item()),
        "mean_abs_norm_error": float((norms - 1.0).abs().mean().detach().item()),
        "mean_omega": float(omega.mean().detach().item()),
        "raw_radial_fraction": float(radial_fraction.detach().item()),
        "label_drop_fraction": float(drop_mask.float().mean().detach().item()),
    }


def train_projected_rfm(
    field: nn.Module,
    ae: EuclideanAutoencoder,
    loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    run_dir: Path,
):
    checkpoints = ensure_dir(run_dir / "checkpoints")
    logs = ensure_dir(run_dir / "logs")
    latest = checkpoints / "projected_rfm_latest.pt"
    final = checkpoints / "projected_rfm_final.pt"
    log_path = logs / "projected_rfm.csv"

    optimizer = torch.optim.AdamW(
        field.parameters(), lr=config.prior_lr, weight_decay=1e-4
    )
    scaler = base.make_grad_scaler(device, config.amp)
    ema = base.ExponentialMovingAverage(field, decay=config.ema_decay)
    start_epoch = 1
    rows: List[Dict[str, float]] = []

    if config.resume and latest.exists():
        obj = torch.load(latest, map_location=device)
        field.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        if obj.get("scaler") is not None:
            scaler.load_state_dict(obj["scaler"])
        if obj.get("ema") is not None:
            ema.load_state_dict(obj["ema"])
        start_epoch = int(obj["epoch"]) + 1
        rows = list(obj.get("logs", []))
        print(f"[Projected-RFM] resumed from epoch {start_epoch - 1}")
    elif config.resume and final.exists():
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj["model"] if "model" in obj else obj)
        if isinstance(obj, dict) and "ema" in obj:
            ema.load_state_dict(obj["ema"])
        start_epoch = config.prior_epochs + 1
        print("[Projected-RFM] final checkpoint found")

    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    if not config.eval_only:
        field.train()
        for epoch in range(start_epoch, config.prior_epochs + 1):
            started = time.time()
            sums: Dict[str, float] = {}
            count = 0
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().view(-1)
                optimizer.zero_grad(set_to_none=True)
                with base.autocast_context(device, config.amp):
                    loss, stats = projected_rfm_objective(
                        field, ae, images, labels, base, config
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(field)

                batch = images.size(0)
                count += batch
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + value * batch

            row = {key: value / max(count, 1) for key, value in sums.items()}
            row.update({"epoch": float(epoch), "seconds": time.time() - started})
            rows.append(row)
            save_csv(rows, log_path)
            print(
                f"[Projected-RFM] {epoch:03d}/{config.prior_epochs} "
                f"loss={row['loss']:.6f} "
                f"path_norm={row['mean_path_norm']:.4f}"
            )

            if epoch % config.checkpoint_every == 0 or epoch == config.prior_epochs:
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model": field.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "logs": rows,
                        "config": asdict(config),
                    },
                    latest,
                )

        atomic_torch_save(
            {
                "model": field.state_dict(),
                "ema": ema.state_dict(),
                "config": asdict(config),
            },
            final,
        )
    else:
        if not final.exists():
            raise FileNotFoundError(f"Missing projected RFM checkpoint: {final}")
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj["model"] if "model" in obj else obj)
        if isinstance(obj, dict) and "ema" in obj:
            ema.load_state_dict(obj["ema"])

    field.load_state_dict(ema.state_dict())
    field.eval()
    return rows


@torch.no_grad()
def cfg_velocity(
    field: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    null = torch.full_like(labels, field.null_label)
    unconditional = field(z, t, null)
    conditional = field(z, t, labels)
    return unconditional + scale * (conditional - unconditional)


@torch.no_grad()
def sample_projected_rfm_directions(
    field: nn.Module,
    labels: torch.Tensor,
    base,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = labels.long().cpu()
    all_latents: List[torch.Tensor] = []
    norm_paths: List[np.ndarray] = []
    radial_paths: List[np.ndarray] = []

    for start in range(0, labels.numel(), config.metric_batch_size):
        batch_labels = labels[start : start + config.metric_batch_size].to(device)
        z = base.sample_uniform_product_spheres(
            batch_labels.numel(),
            config.latent_channels,
            config.latent_size,
            config.latent_size,
            device,
        )
        local_norms = [float(base.token_norms(z).mean().item())]
        local_radials: List[float] = []
        dt = -1.0 / config.sample_steps

        for index in range(config.sample_steps, 0, -1):
            t = torch.full(
                (batch_labels.numel(),),
                index / config.sample_steps,
                device=device,
            )
            raw = cfg_velocity(
                field, z, t, batch_labels, config.guidance_scale
            )
            local_radials.append(
                float(base.radial_velocity_fraction_tokens(z, raw).item())
            )
            tangent = base.tangent_projection_tokens(z, raw)
            z = base.exp_map_tokens(z, dt * tangent)
            local_norms.append(float(base.token_norms(z).mean().item()))

        all_latents.append(z.cpu())
        norm_paths.append(np.asarray(local_norms, dtype=np.float64))
        radial_paths.append(np.asarray(local_radials, dtype=np.float64))

    directions = torch.cat(all_latents, dim=0)
    mean_norm_path = np.stack(norm_paths).mean(axis=0)
    mean_radial_path = np.stack(radial_paths).mean(axis=0)
    final_norms = base.token_norms(directions)
    geometry = {
        "final_mean_norm": float(final_norms.mean().item()),
        "final_std_norm": float(final_norms.std().item()),
        "min_mean_path_norm": float(mean_norm_path.min()),
        "mean_abs_path_norm_error": float(np.abs(mean_norm_path - 1.0).mean()),
        "mean_raw_radial_fraction": float(mean_radial_path.mean()),
    }
    return directions, geometry


@torch.no_grad()
def decode_projected_directions(
    ae: EuclideanAutoencoder,
    directions: torch.Tensor,
    labels: torch.Tensor,
    radius_stats: RadiusStats,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    radius_map = radius_stats.radius_map(labels, config.radius_mode)
    images: List[torch.Tensor] = []
    for start in range(0, directions.size(0), config.metric_batch_size):
        u = directions[start : start + config.metric_batch_size].to(device)
        r = radius_map[start : start + config.metric_batch_size].to(device)
        h = u * r[:, None]
        images.append(ae.decode(h).cpu())
    return torch.cat(images, dim=0)


def evaluate_generation(
    field: nn.Module,
    ae: EuclideanAutoencoder,
    radius_stats: RadiusStats,
    test_loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, float]:
    metric_count = config.metric_samples
    real = base.collect_real_images(test_loader, metric_count)
    labels = base.collect_real_labels(test_loader, metric_count)
    metric_count = min(real.size(0), labels.numel())
    real = real[:metric_count]
    labels = labels[:metric_count]

    directions, geometry = sample_projected_rfm_directions(
        field, labels, base, config, device
    )
    fake = decode_projected_directions(
        ae, directions, labels, radius_stats, config, device
    )

    base.save_class_balanced_grid(
        fake,
        labels,
        ensure_dir(run_dir / "samples") / "euclidean_ae_projected_rfm.png",
        num_classes=config.num_classes,
        samples_per_class=8,
    )

    row: Dict[str, float] = {
        "method": "euclidean_ae_projected_rfm",
        "seed": float(config.seed),
        "guidance_scale": float(config.guidance_scale),
        "radius_mode": config.radius_mode,
        "radius_global_cv": float(radius_stats.global_cv),
        **geometry,
    }

    if not config.skip_heavy_metrics:
        real_uint8 = base.to_uint8(real)
        fake_uint8 = base.to_uint8(fake)
        row.update(
            base.compute_fid_kid(
                real_uint8,
                fake_uint8,
                device=device,
                batch_size=config.metric_batch_size,
            )
        )
        n = min(config.pr_samples, real_uint8.size(0), fake_uint8.size(0))
        real_features = base.extract_resnet_features(
            real_uint8[:n], device=device, batch_size=config.metric_batch_size
        )
        fake_features = base.extract_resnet_features(
            fake_uint8[:n], device=device, batch_size=config.metric_batch_size
        )
        row.update(
            base.feature_precision_recall(
                real_features,
                fake_features,
                device=device,
                nearest_k=config.pr_nearest_k,
                chunk_size=config.pr_chunk_size,
            )
        )
    return row


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------


def build_comparison_table(
    new_row: Mapping[str, object],
    previous_metrics_csv: Optional[str],
    output_path: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if previous_metrics_csv:
        previous = read_csv(Path(previous_metrics_csv))
        wanted = {"srul_rfm", "euclidean_fm"}
        for row in previous:
            if row.get("method") in wanted:
                renamed = dict(row)
                renamed["comparison_role"] = (
                    "matched_spherical" if row.get("method") == "srul_rfm"
                    else "matched_euclidean"
                )
                rows.append(renamed)
    mismatch = dict(new_row)
    mismatch["comparison_role"] = "mismatched_direction_only_rfm"
    rows.append(mismatch)
    save_csv(rows, output_path)
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(config: Config) -> None:
    set_seed(config.seed)
    device = get_device()
    base = import_base_module()

    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "samples")
    ensure_dir(run_dir / "figures")
    save_json(asdict(config), run_dir / "config.json")

    print("=" * 80)
    print("Euclidean autoencoder + projected spherical RFM mismatch experiment")
    print("Device:", device)
    print("Run directory:", run_dir)
    print("Radius restoration:", config.radius_mode)
    print("=" * 80)

    _, _, train_loader, test_loader = make_loaders(base, config, device)

    ae = EuclideanAutoencoder(
        base,
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
    ).to(device)
    checkpoint = Path(config.euclidean_ae_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Euclidean autoencoder checkpoint not found: {checkpoint}"
        )
    ae.load_state_dict(load_model_state(checkpoint, device))
    ae.eval()
    print("Loaded Euclidean autoencoder:", checkpoint)

    stats_path = run_dir / "radius_stats.pt"
    if config.resume and stats_path.exists():
        radius_stats = RadiusStats.from_dict(torch.load(stats_path, map_location="cpu"))
        print("Loaded saved radius statistics.")
    else:
        radius_stats = estimate_radius_stats(ae, train_loader, config, device)
        atomic_torch_save(radius_stats.cpu_dict(), stats_path)

    radius_summary = {
        "global_mean_radius": radius_stats.global_mean,
        "global_std_radius": radius_stats.global_std,
        "global_radius_cv": radius_stats.global_cv,
        "mean_class_token_cv": radius_stats.mean_class_token_cv,
        "class_counts": radius_stats.class_counts.tolist(),
        "interpretation": (
            "A larger coefficient of variation means the Euclidean encoder "
            "uses more variable token radii."
        ),
    }
    save_json(radius_summary, run_dir / "radius_diagnostics.json")
    print(
        "Radius CV: global=%.4f, class-token mean=%.4f"
        % (radius_stats.global_cv, radius_stats.mean_class_token_cv)
    )

    recon_metrics = evaluate_radius_reconstruction(
        ae,
        radius_stats,
        test_loader,
        base,
        config,
        device,
        run_dir / "figures" / "radius_reconstruction_diagnostic.png",
    )
    save_json(recon_metrics, run_dir / "radius_reconstruction_metrics.json")
    print("Radius reconstruction diagnostic complete.")

    field = base.SpatialVectorField(
        latent_channels=config.latent_channels,
        width=config.prior_width,
        depth=config.prior_depth,
        time_dim=config.time_dim,
        num_classes=config.num_classes,
    ).to(device)
    train_rows = train_projected_rfm(
        field,
        ae,
        train_loader,
        base,
        config,
        device,
        run_dir,
    )

    generation_row = evaluate_generation(
        field,
        ae,
        radius_stats,
        test_loader,
        base,
        config,
        device,
        run_dir,
    )
    save_csv([generation_row], run_dir / "generation_metrics.csv")

    comparison_rows = build_comparison_table(
        generation_row,
        config.previous_metrics_csv,
        run_dir / "matched_vs_mismatched_comparison.csv",
    )

    summary = {
        "radius_diagnostics": radius_summary,
        "radius_reconstruction": recon_metrics,
        "projected_rfm_generation": generation_row,
        "comparison_rows": comparison_rows,
        "final_training_row": train_rows[-1] if train_rows else None,
    }
    save_json(summary, run_dir / "summary.json")

    print("\nCompleted.")
    print("New generation result:")
    print(json.dumps(generation_row, indent=2))
    print("\nSaved comparison table:")
    print(run_dir / "matched_vs_mismatched_comparison.csv")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def optional_count(value: str) -> Optional[int]:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Test a spherical RFM prior on a non-spherical autoencoder."
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake32"])
    parser.add_argument("--data-root", default="/content/data")
    parser.add_argument(
        "--out-dir",
        default="/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10_projected_rfm",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-samples", type=optional_count, default=None)
    parser.add_argument("--test-samples", type=optional_count, default=10000)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--latent-channels", type=int, default=32)

    parser.add_argument("--prior-epochs", type=int, default=120)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--prior-width", type=int, default=256)
    parser.add_argument("--prior-depth", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument(
        "--time-sampling", choices=["uniform", "logit_normal"], default="logit_normal"
    )
    parser.add_argument("--label-drop-prob", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--direction-noise", type=float, default=0.0)

    parser.add_argument(
        "--radius-mode",
        choices=["class_token_mean", "global_token_mean", "unit"],
        default="class_token_mean",
    )
    parser.add_argument("--radius-max-samples", type=int, default=50000)

    parser.add_argument("--metric-samples", type=int, default=10000)
    parser.add_argument("--pr-samples", type=int, default=5000)
    parser.add_argument("--reconstruction-metric-samples", type=int, default=5000)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)

    parser.add_argument("--euclidean-ae-checkpoint", required=True)
    parser.add_argument("--previous-metrics-csv", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    run(parse_args())
