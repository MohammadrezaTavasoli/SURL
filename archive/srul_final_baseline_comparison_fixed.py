"""
Matched final baselines for the SRUL course project.

This script adds the comparisons that are missing from the current report:

1. SRUL-RFM: existing spatial spherical autoencoder + conditional RFM prior.
   The script normally evaluates already-trained SRUL checkpoints so the main
   model does not need to be trained again.
2. Euclidean-FM: the same spatial encoder/decoder capacity, but without token
   normalization. Latents are standardized with training-set statistics and a
   conditional Euclidean flow-matching prior is trained in that space.
3. LDM: a matched KL autoencoder with the same spatial bottleneck and a
   conditional latent DDPM prior. This is a compact LDM implemented in the
   project codebase, rather than a comparison to published numbers from a
   different architecture/training budget.

The script supports CIFAR-10, PathMNIST, and CelebA-64. It intentionally imports
existing project modules for datasets, geometry helpers, metrics, and the shared
convolutional architecture. Put the relevant base script beside this file:

  CIFAR-10 / PathMNIST:
      srul_medmnist_conditional_experiment.py
  CelebA-64:
      srul_celeba64_conditional_experiment.py

Example (CIFAR-10):

python srul_final_baseline_comparison.py \
  --dataset cifar10 \
  --out-dir /content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10 \
  --data-root /content/data \
  --num-classes 10 --image-size 32 \
  --train-samples 0 --test-samples 10000 \
  --srul-ae-checkpoint /content/drive/MyDrive/SRUL_CIFAR10/spatial_v1/seed_0/checkpoints/autoencoder_final.pt \
  --srul-prior-checkpoint /content/drive/MyDrive/SRUL_CIFAR10/conditional_v1/seed_0/checkpoints/cond_rfm_final.pt \
  --methods srul_rfm euclidean_fm ldm \
  --ae-epochs 60 --prior-epochs 120 --guidance-scale 2.0 \
  --resume --amp

Quick synthetic smoke test:

python srul_final_baseline_comparison.py \
  --dataset fake32 --out-dir /tmp/srul_final_compare \
  --num-classes 4 --image-size 32 --train-samples 96 --test-samples 48 \
  --methods euclidean_fm ldm --ae-epochs 1 --prior-epochs 1 \
  --batch-size 16 --metric-samples 32 --pr-samples 24 \
  --recon-metric-samples 32 --base-channels 16 --latent-channels 8 \
  --prior-width 32 --prior-depth 2 --diffusion-steps 40 \
  --sample-steps 8 --skip-heavy-metrics
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


# -----------------------------------------------------------------------------
# General utilities
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


def atomic_torch_save(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def load_checkpoint_state(
    path: Path,
    device: torch.device,
    prefer_ema: bool = False,
) -> Mapping[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint: {path}")
    if prefer_ema and "ema" in obj:
        return obj["ema"]
    if "model" in obj:
        return obj["model"]
    return obj


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            value = value.detach()
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(
                    value, alpha=1.0 - self.decay
                )
            else:
                self.shadow[key].copy_(value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.shadow = {key: value.detach().clone() for key, value in state.items()}


# -----------------------------------------------------------------------------
# Configuration and base-module loading
# -----------------------------------------------------------------------------


@dataclass
class Config:
    dataset: str = "cifar10"
    data_root: str = "./data"
    out_dir: str = "./SRUL_Final_Comparisons"
    seed: int = 0
    image_size: int = 32
    num_classes: int = 10
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None

    batch_size: int = 128
    metric_batch_size: int = 128
    num_workers: int = 2
    base_channels: int = 96
    latent_channels: int = 32

    ae_epochs: int = 60
    ae_lr: float = 2e-4
    lambda_clean: float = 1.0
    lambda_noisy: float = 0.5
    lambda_edge: float = 0.15
    lambda_latent: float = 0.05
    lambda_lpips: float = 0.10
    euclidean_noise: float = 0.15
    kl_weight: float = 1e-6

    prior_epochs: int = 120
    prior_lr: float = 2e-4
    prior_width: int = 256
    prior_depth: int = 6
    time_dim: int = 128
    time_sampling: str = "logit_normal"
    label_drop_prob: float = 0.10
    ema_decay: float = 0.999
    guidance_scale: float = 2.0

    diffusion_steps: int = 1000
    sample_steps: int = 100
    methods: Tuple[str, ...] = ("srul_rfm", "euclidean_fm", "ldm")

    metric_samples: int = 5000
    pr_samples: int = 5000
    recon_metric_samples: int = 5000
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5

    srul_ae_checkpoint: Optional[str] = None
    srul_prior_checkpoint: Optional[str] = None

    checkpoint_every: int = 5
    resume: bool = False
    amp: bool = False
    compute_lpips: bool = False
    skip_heavy_metrics: bool = False

    # CelebA cache arguments, ignored for 32x32 datasets.
    hf_dataset: str = "flwrlabs/celeba"
    celeba_attribute: str = "Smiling"
    hf_shuffle_buffer: int = 10000


def import_base_module(dataset: str):
    if dataset in {"celeba64", "fake64"}:
        module_name = "srul_celeba64_conditional_experiment"
    else:
        module_name = "srul_medmnist_conditional_experiment"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import {module_name}. Put {module_name}.py in the "
            "same Colab directory as this script."
        ) from exc


def make_datasets_and_loaders(base, config: Config, device: torch.device):
    if config.dataset == "fake32":
        dataset_name = "fake"
    elif config.dataset == "fake64":
        dataset_name = "fake64"
    else:
        dataset_name = config.dataset

    if config.dataset in {"celeba64", "fake64"}:
        train_set, test_set = base.make_datasets(
            name=dataset_name,
            root=config.data_root,
            train_samples=config.train_samples,
            test_samples=config.test_samples,
            seed=config.seed,
            image_size=config.image_size,
            hf_dataset=config.hf_dataset,
            celeba_attribute=config.celeba_attribute,
            hf_shuffle_buffer=config.hf_shuffle_buffer,
        )
    else:
        train_set, test_set = base.make_datasets(
            name=dataset_name,
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
# Matched autoencoders
# -----------------------------------------------------------------------------


class EuclideanAutoencoder(nn.Module):
    """Same spatial architecture as SRUL, but no token normalization."""

    def __init__(self, base, latent_channels: int, base_channels: int, image_size: int):
        super().__init__()
        self.encoder = base.SpatialEncoder(latent_channels, base_channels)
        self.decoder = base.SpatialDecoder(latent_channels, base_channels)
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // 8

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class KLAutoencoder(nn.Module):
    """Matched KL autoencoder used as the first stage of the LDM baseline."""

    def __init__(self, base, latent_channels: int, base_channels: int, image_size: int):
        super().__init__()
        self.encoder = base.SpatialEncoder(latent_channels * 2, base_channels)
        self.decoder = base.SpatialDecoder(latent_channels, base_channels)
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // 8

    def posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(x)
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(-12.0, 8.0)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.posterior(x)[0]

    def encode_sample(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.posterior(x)
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def cpu_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_dict(cls, obj: Mapping[str, torch.Tensor]) -> "LatentStats":
        return cls(mean=obj["mean"].cpu(), std=obj["std"].cpu())


def relative_gaussian_noise(z: torch.Tensor, sigma: float) -> torch.Tensor:
    # Noise magnitude follows each sample's current latent RMS, avoiding an
    # arbitrary dependency on an unconstrained Euclidean scale.
    rms = z.detach().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-4)
    return z + sigma * rms * torch.randn_like(z)


def token_cosine_after_normalization(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, dim=1, eps=1e-8)
    b_n = F.normalize(b, dim=1, eps=1e-8)
    return (a_n * b_n).sum(dim=1).mean()


def get_perceptual(base, config: Config, device: torch.device):
    if not config.compute_lpips or config.lambda_lpips <= 0:
        return None
    return base.LPIPSLoss(device)


def euclidean_ae_objective(
    base,
    ae: EuclideanAutoencoder,
    images: torch.Tensor,
    config: Config,
    perceptual: Optional[nn.Module],
):
    z = ae.encode(images)
    z_noisy = relative_gaussian_noise(z, config.euclidean_noise)
    clean = ae.decode(z)
    noisy = ae.decode(z_noisy)

    clean_l1 = F.l1_loss(clean, images)
    noisy_l1 = F.l1_loss(noisy, images)
    edges = base.edge_loss(clean, images)
    reencoded = ae.encode(noisy)
    latent_consistency = 1.0 - token_cosine_after_normalization(
        z.detach(), reencoded
    )
    perceptual_loss = torch.zeros((), device=images.device)
    if perceptual is not None:
        perceptual_loss = perceptual(clean, images)

    loss = (
        config.lambda_clean * clean_l1
        + config.lambda_noisy * noisy_l1
        + config.lambda_edge * edges
        + config.lambda_latent * latent_consistency
        + config.lambda_lpips * perceptual_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "clean_l1": float(clean_l1.detach()),
        "noisy_l1": float(noisy_l1.detach()),
        "edge": float(edges.detach()),
        "latent": float(latent_consistency.detach()),
        "lpips": float(perceptual_loss.detach()),
    }


def kl_ae_objective(
    base,
    ae: KLAutoencoder,
    images: torch.Tensor,
    config: Config,
    perceptual: Optional[nn.Module],
):
    mean, logvar = ae.posterior(images)
    z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    recon = ae.decode(z)

    recon_l1 = F.l1_loss(recon, images)
    edges = base.edge_loss(recon, images)
    kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()
    perceptual_loss = torch.zeros((), device=images.device)
    if perceptual is not None:
        perceptual_loss = perceptual(recon, images)

    loss = (
        config.lambda_clean * recon_l1
        + config.lambda_edge * edges
        + config.lambda_lpips * perceptual_loss
        + config.kl_weight * kl
    )
    return loss, {
        "loss": float(loss.detach()),
        "recon_l1": float(recon_l1.detach()),
        "edge": float(edges.detach()),
        "lpips": float(perceptual_loss.detach()),
        "kl": float(kl.detach()),
    }


def train_autoencoder(
    name: str,
    base,
    ae: nn.Module,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    run_dir: Path,
) -> List[Dict[str, float]]:
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_path = run_dir / "logs" / f"{name}_autoencoder.csv"
    latest = checkpoint_dir / f"{name}_autoencoder_latest.pt"
    final = checkpoint_dir / f"{name}_autoencoder_final.pt"

    optimizer = torch.optim.AdamW(ae.parameters(), lr=config.ae_lr, weight_decay=1e-4)
    scaler = make_grad_scaler(device, config.amp)
    perceptual = get_perceptual(base, config, device)
    start_epoch = 0
    history: List[Dict[str, float]] = []

    if config.resume and latest.exists():
        obj = torch.load(latest, map_location=device)
        ae.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        start_epoch = int(obj["epoch"])
        history = list(obj.get("history", []))
        print(f"[{name} AE] resumed from epoch {start_epoch}")
    elif config.resume and final.exists():
        ae.load_state_dict(load_checkpoint_state(final, device))
        start_epoch = config.ae_epochs
        print(f"[{name} AE] final checkpoint found")

    ae.train()
    for epoch in range(start_epoch, config.ae_epochs):
        totals: Dict[str, float] = {}
        batches = 0
        start = time.time()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.amp):
                if name == "euclidean_fm":
                    loss, stats = euclidean_ae_objective(
                        base, ae, images, config, perceptual
                    )
                elif name == "ldm":
                    loss, stats = kl_ae_objective(
                        base, ae, images, config, perceptual
                    )
                else:
                    raise ValueError(name)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1

        row = {key: value / max(batches, 1) for key, value in totals.items()}
        row.update({"epoch": epoch + 1, "seconds": time.time() - start})
        history.append(row)
        print(
            f"[{name} AE] {epoch + 1:03d}/{config.ae_epochs} "
            f"loss={row['loss']:.5f}"
        )
        save_csv(history, log_path)

        if (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.ae_epochs:
            atomic_torch_save(
                {
                    "model": ae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                    "config": asdict(config),
                },
                latest,
            )

    atomic_torch_save(ae.state_dict(), final)
    return history


@torch.no_grad()
def estimate_latent_stats(
    name: str,
    ae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 20000,
) -> LatentStats:
    ae.eval()
    sum_c = None
    sum_sq_c = None
    count = 0
    for images, _ in loader:
        images = images.to(device)
        if name == "euclidean_fm":
            z = ae.encode(images)
        elif name == "ldm":
            z = ae.encode_sample(images)
        else:
            raise ValueError(name)
        # Channel-wise moments over batch and spatial positions.
        local_sum = z.sum(dim=(0, 2, 3), keepdim=True)
        local_sq = z.pow(2).sum(dim=(0, 2, 3), keepdim=True)
        elements = z.size(0) * z.size(2) * z.size(3)
        sum_c = local_sum if sum_c is None else sum_c + local_sum
        sum_sq_c = local_sq if sum_sq_c is None else sum_sq_c + local_sq
        count += elements
        if count >= max_samples * z.size(2) * z.size(3):
            break
    mean = sum_c / max(count, 1)
    variance = sum_sq_c / max(count, 1) - mean.pow(2)
    std = variance.clamp_min(1e-6).sqrt()
    return LatentStats(mean=mean.cpu(), std=std.cpu())


# -----------------------------------------------------------------------------
# Prior training
# -----------------------------------------------------------------------------


def sample_time(batch: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch, device=device)
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, device=device))
    raise ValueError(mode)


def apply_label_dropout(
    labels: torch.Tensor,
    null_label: int,
    probability: float,
) -> torch.Tensor:
    dropped = labels.clone().long()
    mask = torch.rand(labels.size(0), device=labels.device) < probability
    dropped[mask] = null_label
    return dropped


def euclidean_fm_objective(
    field: nn.Module,
    ae: EuclideanAutoencoder,
    stats: LatentStats,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
):
    with torch.no_grad():
        target = stats.standardize(ae.encode(images))
    source = torch.randn_like(target)
    t = sample_time(images.size(0), images.device, config.time_sampling)
    view = t[:, None, None, None]
    path = (1.0 - view) * target + view * source
    velocity = source - target
    training_labels = apply_label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    prediction = field(path, t, training_labels)
    loss = F.mse_loss(prediction, velocity)
    return loss, {
        "loss": float(loss.detach()),
        "path_rms": float(path.detach().pow(2).mean().sqrt()),
    }


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def ldm_objective(
    field: nn.Module,
    ae: KLAutoencoder,
    stats: LatentStats,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
    alpha_bars: torch.Tensor,
):
    with torch.no_grad():
        target = stats.standardize(ae.encode_sample(images))
    indices = torch.randint(
        0, config.diffusion_steps, (images.size(0),), device=images.device
    )
    alpha = alpha_bars[indices].view(-1, 1, 1, 1)
    noise = torch.randn_like(target)
    noisy = alpha.sqrt() * target + (1.0 - alpha).sqrt() * noise
    t = indices.float() / max(config.diffusion_steps - 1, 1)
    training_labels = apply_label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    prediction = field(noisy, t, training_labels)
    loss = F.mse_loss(prediction, noise)
    return loss, {"loss": float(loss.detach())}


def train_prior(
    name: str,
    base,
    field: nn.Module,
    ae: nn.Module,
    stats: LatentStats,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    run_dir: Path,
):
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_path = run_dir / "logs" / f"{name}_prior.csv"
    latest = checkpoint_dir / f"{name}_prior_latest.pt"
    final = checkpoint_dir / f"{name}_prior_final.pt"

    optimizer = torch.optim.AdamW(field.parameters(), lr=config.prior_lr, weight_decay=1e-4)
    scaler = make_grad_scaler(device, config.amp)
    ema = EMA(field, config.ema_decay)
    start_epoch = 0
    history: List[Dict[str, float]] = []

    if config.resume and latest.exists():
        obj = torch.load(latest, map_location=device)
        field.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        if "ema" in obj:
            ema.load_state_dict(obj["ema"])
        start_epoch = int(obj["epoch"])
        history = list(obj.get("history", []))
        print(f"[{name}] resumed from epoch {start_epoch}")
    elif config.resume and final.exists():
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj["model"] if "model" in obj else obj)
        if isinstance(obj, dict) and "ema" in obj:
            ema.load_state_dict(obj["ema"])
        start_epoch = config.prior_epochs
        print(f"[{name}] final checkpoint found")

    ae.eval()
    field.train()
    alpha_bars = torch.cumprod(
        1.0 - cosine_beta_schedule(config.diffusion_steps).to(device), dim=0
    )

    for epoch in range(start_epoch, config.prior_epochs):
        totals: Dict[str, float] = {}
        batches = 0
        start = time.time()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().view(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.amp):
                if name == "euclidean_fm":
                    loss, row = euclidean_fm_objective(
                        field, ae, stats, images, labels, config
                    )
                elif name == "ldm":
                    loss, row = ldm_objective(
                        field, ae, stats, images, labels, config, alpha_bars
                    )
                else:
                    raise ValueError(name)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(field)
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1

        row = {key: value / max(batches, 1) for key, value in totals.items()}
        row.update({"epoch": epoch + 1, "seconds": time.time() - start})
        history.append(row)
        print(
            f"[{name}] {epoch + 1:03d}/{config.prior_epochs} "
            f"loss={row['loss']:.6f}"
        )
        save_csv(history, log_path)

        if (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.prior_epochs:
            atomic_torch_save(
                {
                    "model": field.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                    "latent_stats": stats.cpu_dict(),
                    "config": asdict(config),
                },
                latest,
            )

    atomic_torch_save(
        {
            "model": field.state_dict(),
            "ema": ema.state_dict(),
            "latent_stats": stats.cpu_dict(),
            "config": asdict(config),
        },
        final,
    )
    return history, ema


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


@torch.no_grad()
def cfg_prediction(
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
def sample_euclidean_fm_latents(
    field: nn.Module,
    labels: torch.Tensor,
    latent_shape: Tuple[int, int, int],
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    field.eval()
    z = torch.randn((labels.size(0),) + latent_shape, device=device)
    dt = -1.0 / config.sample_steps
    for index in range(config.sample_steps, 0, -1):
        t = torch.full(
            (labels.size(0),), index / config.sample_steps, device=device
        )
        velocity = cfg_prediction(
            field, z, t, labels, config.guidance_scale
        )
        z = z + dt * velocity
    return z


@torch.no_grad()
def sample_ldm_latents(
    field: nn.Module,
    labels: torch.Tensor,
    latent_shape: Tuple[int, int, int],
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    field.eval()
    z = torch.randn((labels.size(0),) + latent_shape, device=device)
    betas = cosine_beta_schedule(config.diffusion_steps).to(device)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    # Descending, unique DDIM schedule.
    indices = torch.linspace(
        config.diffusion_steps - 1,
        0,
        config.sample_steps,
        device=device,
    ).round().long()
    indices = torch.unique_consecutive(indices)

    for step, index in enumerate(indices):
        batch = labels.size(0)
        t = torch.full(
            (batch,), index.item() / max(config.diffusion_steps - 1, 1), device=device
        )
        eps = cfg_prediction(field, z, t, labels, config.guidance_scale)
        alpha_t = alpha_bars[index]
        if step + 1 < indices.numel():
            prev_index = indices[step + 1]
            alpha_prev = alpha_bars[prev_index]
        else:
            alpha_prev = torch.tensor(1.0, device=device)
        pred_x0 = (z - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt().clamp_min(1e-6)
        z = alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * eps
    return z


@torch.no_grad()
def instantiate_spherical_ae(base, config: Config, device: torch.device):
    try:
        ae = base.SpatialSphericalAutoencoder(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            image_size=config.image_size,
        )
    except TypeError:
        ae = base.SpatialSphericalAutoencoder(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
        )
    return ae.to(device)


@torch.no_grad()
def sample_srul_latents(
    base,
    field: nn.Module,
    labels: torch.Tensor,
    latent_shape: Tuple[int, int, int],
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    channels, height, width = latent_shape
    z = base.sample_uniform_product_spheres(
        labels.size(0), channels, height, width, device
    )
    dt = -1.0 / config.sample_steps
    for index in range(config.sample_steps, 0, -1):
        t = torch.full(
            (labels.size(0),), index / config.sample_steps, device=device
        )
        raw = cfg_prediction(field, z, t, labels, config.guidance_scale)
        tangent = base.tangent_projection_tokens(z, raw)
        z = base.exp_map_tokens(z, dt * tangent)
    return z


@torch.no_grad()
def collect_test_images_and_labels(
    loader: DataLoader,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    images_out = []
    labels_out = []
    count = 0
    for images, labels in loader:
        remaining = max_samples - count
        images_out.append(images[:remaining].cpu())
        labels_out.append(labels[:remaining].long().view(-1).cpu())
        count += min(images.size(0), remaining)
        if count >= max_samples:
            break
    return torch.cat(images_out), torch.cat(labels_out)


@torch.no_grad()
def generate_images(
    name: str,
    base,
    ae: nn.Module,
    field: nn.Module,
    stats: Optional[LatentStats],
    labels: torch.Tensor,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ae.eval()
    field.eval()
    outputs = []
    latents = []
    batch_size = config.metric_batch_size
    latent_shape = (
        config.latent_channels,
        config.image_size // 8,
        config.image_size // 8,
    )
    for start in range(0, labels.numel(), batch_size):
        y = labels[start : start + batch_size].to(device)
        if name == "euclidean_fm":
            standardized = sample_euclidean_fm_latents(
                field, y, latent_shape, config, device
            )
            z = stats.unstandardize(standardized)
        elif name == "ldm":
            standardized = sample_ldm_latents(
                field, y, latent_shape, config, device
            )
            z = stats.unstandardize(standardized)
        elif name == "srul_rfm":
            z = sample_srul_latents(
                base, field, y, latent_shape, config, device
            )
        else:
            raise ValueError(name)
        images = ae.decode(z)
        outputs.append(base.to_uint8(images).cpu())
        latents.append(z.flatten(1).float().cpu())
    return torch.cat(outputs), torch.cat(latents)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_autoencoder(
    name: str,
    base,
    ae: nn.Module,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> Dict[str, float]:
    ae.eval()
    real_list = []
    recon_list = []
    mse_total = 0.0
    pixel_count = 0
    sample_count = 0
    for images, _ in loader:
        images = images.to(device)
        if name == "euclidean_fm":
            recon = ae.decode(ae.encode(images))
        elif name == "ldm":
            recon = ae.decode(ae.encode_mean(images))
        elif name == "srul_rfm":
            recon = ae.decode(ae.encode(images))
        else:
            raise ValueError(name)
        remaining = config.recon_metric_samples - sample_count
        if remaining <= 0:
            break
        images = images[:remaining]
        recon = recon[:remaining]
        mse_total += F.mse_loss(recon, images, reduction="sum").item()
        pixel_count += images.numel()
        sample_count += images.size(0)
        real_list.append(base.to_uint8(images).cpu())
        recon_list.append(base.to_uint8(recon).cpu())
        if sample_count >= config.recon_metric_samples:
            break
    real = torch.cat(real_list)[: config.recon_metric_samples]
    recon = torch.cat(recon_list)[: config.recon_metric_samples]
    metrics = {"mse": mse_total / max(pixel_count, 1), "num_samples": float(sample_count)}
    metrics["psnr"] = 10.0 * math.log10(4.0 / max(metrics["mse"], 1e-12))
    if not config.skip_heavy_metrics:
        metrics.update(
            base.compute_fid_kid(
                real, recon, device=device, batch_size=config.metric_batch_size
            )
        )
        n = min(config.pr_samples, real.size(0))
        real_features = base.extract_resnet_features(
            real[:n], device, config.metric_batch_size
        )
        recon_features = base.extract_resnet_features(
            recon[:n], device, config.metric_batch_size
        )
        metrics.update(
            base.feature_precision_recall(
                real_features,
                recon_features,
                device=device,
                nearest_k=config.pr_nearest_k,
                chunk_size=config.pr_chunk_size,
            )
        )
    return metrics


def create_field(base, config: Config, device: torch.device):
    return base.SpatialVectorField(
        latent_channels=config.latent_channels,
        width=config.prior_width,
        depth=config.prior_depth,
        time_dim=config.time_dim,
        num_classes=config.num_classes,
    ).to(device)


def load_srul_models(base, config: Config, device: torch.device):
    if not config.srul_ae_checkpoint or not config.srul_prior_checkpoint:
        raise ValueError(
            "srul_rfm requires --srul-ae-checkpoint and --srul-prior-checkpoint"
        )
    ae = instantiate_spherical_ae(base, config, device)
    ae.load_state_dict(
        load_checkpoint_state(Path(config.srul_ae_checkpoint), device)
    )
    field = create_field(base, config, device)
    field.load_state_dict(
        load_checkpoint_state(
            Path(config.srul_prior_checkpoint), device, prefer_ema=True
        )
    )
    ae.eval()
    field.eval()
    return ae, field


def train_or_load_baseline(
    name: str,
    base,
    config: Config,
    train_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
):
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    if name == "euclidean_fm":
        ae = EuclideanAutoencoder(
            base, config.latent_channels, config.base_channels, config.image_size
        ).to(device)
    elif name == "ldm":
        ae = KLAutoencoder(
            base, config.latent_channels, config.base_channels, config.image_size
        ).to(device)
    else:
        raise ValueError(name)

    final_ae = checkpoint_dir / f"{name}_autoencoder_final.pt"
    if config.resume and final_ae.exists():
        ae.load_state_dict(load_checkpoint_state(final_ae, device))
        print(f"[{name} AE] loaded final checkpoint")
    else:
        train_autoencoder(name, base, ae, train_loader, config, device, run_dir)

    stats_path = checkpoint_dir / f"{name}_latent_stats.pt"
    if config.resume and stats_path.exists():
        stats = LatentStats.from_dict(torch.load(stats_path, map_location="cpu"))
    else:
        stats = estimate_latent_stats(name, ae, train_loader, device)
        atomic_torch_save(stats.cpu_dict(), stats_path)

    field = create_field(base, config, device)
    final_prior = checkpoint_dir / f"{name}_prior_final.pt"
    if config.resume and final_prior.exists():
        obj = torch.load(final_prior, map_location=device)
        state = obj.get("ema", obj.get("model", obj))
        field.load_state_dict(state)
        if "latent_stats" in obj:
            stats = LatentStats.from_dict(obj["latent_stats"])
        print(f"[{name}] loaded final prior")
    else:
        _, ema = train_prior(
            name, base, field, ae, stats, train_loader, config, device, run_dir
        )
        field.load_state_dict(ema.state_dict())
    return ae, field, stats


def save_sample_grid(
    images_uint8: torch.Tensor,
    path: Path,
    image_size: int,
    count: int = 64,
):
    images = images_uint8[:count].float() / 127.5 - 1.0
    nrow = int(math.sqrt(min(count, images.size(0))))
    save_image(images, path, nrow=max(nrow, 1), normalize=True, value_range=(-1, 1))


def run(config: Config) -> None:
    set_seed(config.seed)
    device = get_device()
    base = import_base_module(config.dataset)
    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "samples")
    save_json(asdict(config), run_dir / "config.json")

    print("=" * 80)
    print("SRUL final matched comparison")
    print("Dataset:", config.dataset)
    print("Methods:", list(config.methods))
    print("Device:", device)
    print("Run directory:", run_dir)
    print("=" * 80)

    _, _, train_loader, test_loader = make_datasets_and_loaders(
        base, config, device
    )
    real_float, test_labels = collect_test_images_and_labels(
        test_loader,
        max(config.metric_samples, config.recon_metric_samples, config.pr_samples),
    )
    real_uint8 = base.to_uint8(real_float)
    metric_labels = test_labels[: config.metric_samples]

    results = []
    recon_results = {}

    for method in config.methods:
        print("\n" + "-" * 80)
        print("Preparing:", method)
        if method == "srul_rfm":
            ae, field = load_srul_models(base, config, device)
            stats = None
        elif method in {"euclidean_fm", "ldm"}:
            ae, field, stats = train_or_load_baseline(
                method, base, config, train_loader, device, run_dir
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        recon_metrics = evaluate_autoencoder(
            method, base, ae, test_loader, config, device
        )
        recon_results[method] = recon_metrics
        save_json(recon_results, run_dir / "reconstruction_metrics.json")

        fake_uint8, fake_latents = generate_images(
            method,
            base,
            ae,
            field,
            stats,
            metric_labels,
            config,
            device,
        )
        save_sample_grid(
            fake_uint8,
            run_dir / "samples" / f"{method}.png",
            config.image_size,
        )

        row: Dict[str, object] = {
            "method": method,
            "seed": config.seed,
            "guidance_scale": config.guidance_scale,
            "metric_samples": min(real_uint8.size(0), fake_uint8.size(0)),
        }
        if not config.skip_heavy_metrics:
            n = min(config.metric_samples, real_uint8.size(0), fake_uint8.size(0))
            row.update(
                base.compute_fid_kid(
                    real_uint8[:n],
                    fake_uint8[:n],
                    device=device,
                    batch_size=config.metric_batch_size,
                )
            )
            p = min(config.pr_samples, n)
            real_features = base.extract_resnet_features(
                real_uint8[:p], device, config.metric_batch_size
            )
            fake_features = base.extract_resnet_features(
                fake_uint8[:p], device, config.metric_batch_size
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
        results.append(row)
        save_csv(results, run_dir / "generation_metrics.csv")
        print(row)

    save_json(
        {
            "config": asdict(config),
            "reconstruction_metrics": recon_results,
            "generation_metrics": results,
        },
        run_dir / "summary.json",
    )
    print("\nFinished. Results saved to", run_dir)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=[
        "cifar10", "pathmnist", "bloodmnist", "celeba64", "fake32", "fake64"
    ])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./SRUL_Final_Comparisons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--ae-epochs", type=int, default=60)
    parser.add_argument("--ae-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--lambda-noisy", type=float, default=0.5)
    parser.add_argument("--lambda-edge", type=float, default=0.15)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lambda-lpips", type=float, default=0.10)
    parser.add_argument("--euclidean-noise", type=float, default=0.15)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--prior-epochs", type=int, default=120)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--prior-width", type=int, default=256)
    parser.add_argument("--prior-depth", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--time-sampling", default="logit_normal", choices=["uniform", "logit_normal"])
    parser.add_argument("--label-drop-prob", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=["srul_rfm", "euclidean_fm", "ldm"])
    parser.add_argument("--metric-samples", type=int, default=5000)
    parser.add_argument("--pr-samples", type=int, default=5000)
    parser.add_argument("--recon-metric-samples", type=int, default=5000)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)
    parser.add_argument("--srul-ae-checkpoint")
    parser.add_argument("--srul-prior-checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--hf-dataset", default="flwrlabs/celeba")
    parser.add_argument("--celeba-attribute", default="Smiling")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10000)
    args = parser.parse_args()

    train_samples = None if args.train_samples <= 0 else args.train_samples
    test_samples = None if args.test_samples <= 0 else args.test_samples
    return Config(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        image_size=args.image_size,
        num_classes=args.num_classes,
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        lambda_clean=args.lambda_clean,
        lambda_noisy=args.lambda_noisy,
        lambda_edge=args.lambda_edge,
        lambda_latent=args.lambda_latent,
        lambda_lpips=args.lambda_lpips,
        euclidean_noise=args.euclidean_noise,
        kl_weight=args.kl_weight,
        prior_epochs=args.prior_epochs,
        prior_lr=args.prior_lr,
        prior_width=args.prior_width,
        prior_depth=args.prior_depth,
        time_dim=args.time_dim,
        time_sampling=args.time_sampling,
        label_drop_prob=args.label_drop_prob,
        ema_decay=args.ema_decay,
        guidance_scale=args.guidance_scale,
        diffusion_steps=args.diffusion_steps,
        sample_steps=args.sample_steps,
        methods=tuple(args.methods),
        metric_samples=args.metric_samples,
        pr_samples=args.pr_samples,
        recon_metric_samples=args.recon_metric_samples,
        pr_chunk_size=args.pr_chunk_size,
        pr_nearest_k=args.pr_nearest_k,
        srul_ae_checkpoint=args.srul_ae_checkpoint,
        srul_prior_checkpoint=args.srul_prior_checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        amp=args.amp,
        compute_lpips=args.compute_lpips,
        skip_heavy_metrics=args.skip_heavy_metrics,
        hf_dataset=args.hf_dataset,
        celeba_attribute=args.celeba_attribute,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
    )


if __name__ == "__main__":
    run(parse_args())
