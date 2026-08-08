"""
SRUL CIFAR-10 controlled experiment for Google Colab.

This script trains one shared spherical autoencoder and compares five
latent generative priors on the SAME target latent distribution:

  1) gaussian_fm : Gaussian source + spherical target + Euclidean linear path
  2) chord       : spherical endpoints + Euclidean linear path
  3) rfm         : SLERP geodesic + tangent velocity, no Jacobi weight
  4) srul        : SLERP geodesic + tangent velocity + Jacobi weight
  5) latent_ddpm : Gaussian latent diffusion baseline (x0-prediction)

The latent DDPM is an LDM-style PRIOR baseline using the same spherical
encoder/decoder. It is not a full reproduction of the original LDM paper.
Keeping the autoencoder fixed isolates the effect of the prior/noising rule.

Recommended Colab pilot command:

python srul_cifar_next_experiment.py \
  --out-dir /content/drive/MyDrive/SRUL_CIFAR_pilot \
  --methods gaussian_fm chord rfm srul latent_ddpm \
  --train-samples 20000 --ae-epochs 20 --prior-epochs 30 \
  --latent-dim 128 --batch-size 256 --metric-samples 5000 \
  --pr-samples 2000 --sample-steps 50 --sigma-enc 0.20 \
  --seed 0 --amp --resume

For a quick code smoke test:

python srul_cifar_next_experiment.py \
  --dataset fake --train-samples 1024 --test-samples 256 \
  --ae-epochs 1 --prior-epochs 1 --metric-samples 128 \
  --pr-samples 64 --batch-size 64 --latent-dim 32 \
  --methods gaussian_fm chord rfm srul latent_ddpm --skip-heavy-metrics

Outputs are written under <out-dir>/seed_<seed>/.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import make_grid, save_image

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    structural_similarity_index_measure = None
    FrechetInceptionDistance = None
    KernelInceptionDistance = None

try:
    import lpips  # optional
except Exception:
    lpips = None


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is useful for debugging. Benchmark remains enabled for speed
    # only when deterministic algorithms are not explicitly requested.
    torch.backends.cudnn.deterministic = False
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


def atomic_torch_save(obj: object, path: Path) -> None:
    """Write a checkpoint through a temporary file to reduce corruption risk."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(data: Mapping[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_rows_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class Denormalize:
    """Convert images from [-1, 1] to [0, 1]."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5


def make_datasets(
    name: str,
    root: str,
    train_samples: Optional[int],
    test_samples: Optional[int],
    seed: int,
) -> Tuple[Dataset, Dataset]:
    name = name.lower()
    if name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set: Dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )
        test_set: Dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )
    elif name == "fake":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = datasets.FakeData(
            size=max(train_samples or 1024, 1024),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=0,
        )
        test_set = datasets.FakeData(
            size=max(test_samples or 256, 256),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=100000,
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    generator = torch.Generator().manual_seed(seed)
    if train_samples is not None and train_samples < len(train_set):
        ids = torch.randperm(len(train_set), generator=generator)[:train_samples].tolist()
        train_set = Subset(train_set, ids)
    if test_samples is not None and test_samples < len(test_set):
        ids = torch.randperm(len(test_set), generator=generator)[:test_samples].tolist()
        test_set = Subset(test_set, ids)
    return train_set, test_set


def make_loaders(
    train_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Autoencoder architecture
# -----------------------------------------------------------------------------


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConvEncoder(nn.Module):
    """CIFAR-10 encoder: 3x32x32 -> latent vector."""

    def __init__(self, latent_dim: int, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            ResBlock2d(base),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # 16x16
            ResBlock2d(base * 2),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),  # 8x8
            ResBlock2d(base * 4),
            nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1),  # 4x4
            ResBlock2d(base * 4),
        )
        self.out_norm = nn.GroupNorm(8, base * 4)
        self.fc = nn.Linear(base * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.out_norm(self.net(x)))
        return self.fc(h.flatten(1))


class ConvDecoder(nn.Module):
    """CIFAR-10 decoder: latent vector -> 3x32x32."""

    def __init__(self, latent_dim: int, base: int = 64):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(latent_dim, base * 4 * 4 * 4)
        self.blocks = nn.Sequential(
            ResBlock2d(base * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),  # 8x8
            ResBlock2d(base * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),  # 16x16
            ResBlock2d(base * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),  # 32x32
            ResBlock2d(base),
        )
        self.norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.base * 4, 4, 4)
        h = self.blocks(h)
        return torch.tanh(self.out(F.silu(self.norm(h))))


class SphericalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, base=base_channels)
        self.decoder = ConvDecoder(latent_dim, base=base_channels)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_to_sphere(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# -----------------------------------------------------------------------------
# Time-conditioned latent prior architecture
# -----------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=t.device, dtype=t.dtype
        ) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        args = t[:, None] * frequencies[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentResidualBlock(nn.Module):
    def __init__(self, width: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.time = nn.Linear(time_dim, width * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = self.fc1(x) + self.time(temb)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return h + x


class LatentVectorField(nn.Module):
    """Shared architecture for flow velocity or DDPM noise prediction."""

    def __init__(
        self,
        latent_dim: int,
        width: int = 512,
        depth: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.input = nn.Linear(latent_dim, width)
        self.blocks = nn.ModuleList(
            [LatentResidualBlock(width, time_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t.float())
        h = self.input(z)
        for block in self.blocks:
            h = block(h, temb)
        return self.output(F.silu(self.norm(h)))


# -----------------------------------------------------------------------------
# Sphere geometry
# -----------------------------------------------------------------------------


def normalize_to_sphere(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return z / z.norm(dim=1, keepdim=True).clamp_min(eps)


def tangent_projection(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return u - (u * z).sum(dim=1, keepdim=True) * z


def tangent_noise(
    z: torch.Tensor,
    sigma: float,
    dimension_scaled: bool = True,
) -> torch.Tensor:
    """Sample tangent Gaussian noise.

    When dimension_scaled=True, divide by sqrt(d-1), so sigma approximately
    controls the RMS geodesic angle independently of latent dimension.
    """
    xi = torch.randn_like(z)
    u = tangent_projection(z, xi)
    if dimension_scaled:
        u = u / math.sqrt(max(z.size(1) - 1, 1))
    return exp_map_sphere(z, sigma * u)


def exp_map_sphere(z: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = tangent_projection(z, v)
    norm = v.norm(dim=1, keepdim=True)
    direction = v / norm.clamp_min(eps)
    out = torch.cos(norm) * z + torch.sin(norm) * direction
    tiny = norm.squeeze(1) < eps
    if tiny.any():
        out[tiny] = normalize_to_sphere(z[tiny] + v[tiny])
    return normalize_to_sphere(out)


def sample_uniform_sphere(
    batch_size: int, latent_dim: int, device: torch.device
) -> torch.Tensor:
    return normalize_to_sphere(torch.randn(batch_size, latent_dim, device=device))


def slerp(
    x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SLERP with x at t=0 and eps at t=1."""
    dot = (x * eps).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    safe = sin_omega.abs() > 1e-5

    a = torch.empty_like(omega)
    b = torch.empty_like(omega)
    a[safe] = torch.sin((1.0 - t[safe]) * omega[safe]) / sin_omega[safe]
    b[safe] = torch.sin(t[safe] * omega[safe]) / sin_omega[safe]
    # Rare nearly-collinear fallback.
    a[~safe] = 1.0 - t[~safe]
    b[~safe] = t[~safe]
    zt = a[:, None] * x + b[:, None] * eps
    return normalize_to_sphere(zt), omega


def slerp_velocity(
    x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, omega: torch.Tensor
) -> torch.Tensor:
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    scale = omega / sin_omega
    velocity = scale[:, None] * (
        torch.cos(t * omega)[:, None] * eps
        - torch.cos((1.0 - t) * omega)[:, None] * x
    )
    # Numerical projection guarantees tangency at z_t when used in the caller.
    return velocity


def sinc_sq(x: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(x)
    mask = x.abs() > 1e-4
    result[mask] = (torch.sin(x[mask]) / x[mask]).pow(2)
    # Taylor expansion: sinc(x)^2 = 1 - x^2/3 + O(x^4)
    result[~mask] = 1.0 - x[~mask].pow(2) / 3.0
    return result


def radial_velocity_fraction(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    numerator = (z * v).sum(dim=1).pow(2)
    denominator = z.pow(2).sum(dim=1) * v.pow(2).sum(dim=1) + 1e-8
    return (numerator / denominator).mean()


# -----------------------------------------------------------------------------
# Autoencoder loss and evaluation
# -----------------------------------------------------------------------------


def autoencoder_loss(
    ae: SphericalAutoencoder,
    x: torch.Tensor,
    sigma_enc: float,
    lambda_noisy: float,
    lambda_latent: float,
    dimension_scaled_noise: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z_clean = ae.encode(x)
    z_noisy = tangent_noise(
        z_clean, sigma_enc, dimension_scaled=dimension_scaled_noise
    )
    clean = ae.decode(z_clean)
    noisy = ae.decode(z_noisy)

    clean_l1 = F.l1_loss(clean, x)
    noisy_l1 = F.l1_loss(noisy, x)
    # Re-encode without detaching decoder output. The clean target is detached
    # so the loss does not collapse both representations together.
    reencoded = ae.encode(noisy)
    latent_consistency = 1.0 - F.cosine_similarity(
        reencoded, z_clean.detach(), dim=1
    ).mean()

    loss = clean_l1 + lambda_noisy * noisy_l1 + lambda_latent * latent_consistency
    return loss, {
        "loss": float(loss.detach().item()),
        "clean_l1": float(clean_l1.detach().item()),
        "noisy_l1": float(noisy_l1.detach().item()),
        "latent_consistency": float(latent_consistency.detach().item()),
    }


@torch.no_grad()
def reconstruction_metrics(
    ae: SphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    sigma_enc: float,
    max_samples: int,
    dimension_scaled_noise: bool,
    compute_lpips: bool,
) -> Dict[str, float]:
    ae.eval()
    total = 0
    sums: MutableMapping[str, float] = {
        "clean_mse": 0.0,
        "noisy_mse": 0.0,
        "clean_psnr": 0.0,
        "noisy_psnr": 0.0,
        "clean_ssim": 0.0,
        "noisy_ssim": 0.0,
    }

    lpips_model = None
    if compute_lpips:
        if lpips is None:
            print("[warning] lpips is not installed; LPIPS evaluation skipped.")
        else:
            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            sums["clean_lpips"] = 0.0
            sums["noisy_lpips"] = 0.0

    for x, _ in loader:
        if total >= max_samples:
            break
        x = x.to(device)
        remaining = max_samples - total
        x = x[:remaining]
        z = ae.encode(x)
        z_noisy = tangent_noise(z, sigma_enc, dimension_scaled_noise)
        clean = ae.decode(z)
        noisy = ae.decode(z_noisy)

        x01 = (x + 1.0) * 0.5
        clean01 = ((clean + 1.0) * 0.5).clamp(0, 1)
        noisy01 = ((noisy + 1.0) * 0.5).clamp(0, 1)

        b = x.size(0)
        clean_mse = F.mse_loss(clean01, x01, reduction="none").flatten(1).mean(1)
        noisy_mse = F.mse_loss(noisy01, x01, reduction="none").flatten(1).mean(1)
        clean_psnr = 10.0 * torch.log10(1.0 / clean_mse.clamp_min(1e-10))
        noisy_psnr = 10.0 * torch.log10(1.0 / noisy_mse.clamp_min(1e-10))

        sums["clean_mse"] += clean_mse.sum().item()
        sums["noisy_mse"] += noisy_mse.sum().item()
        sums["clean_psnr"] += clean_psnr.sum().item()
        sums["noisy_psnr"] += noisy_psnr.sum().item()

        if structural_similarity_index_measure is not None:
            # TorchMetrics SSIM returns a batch aggregate; weight it by batch.
            sums["clean_ssim"] += float(
                structural_similarity_index_measure(clean01, x01, data_range=1.0).item()
            ) * b
            sums["noisy_ssim"] += float(
                structural_similarity_index_measure(noisy01, x01, data_range=1.0).item()
            ) * b
        else:
            sums["clean_ssim"] += float("nan")
            sums["noisy_ssim"] += float("nan")

        if lpips_model is not None:
            sums["clean_lpips"] += float(lpips_model(clean, x).sum().item())
            sums["noisy_lpips"] += float(lpips_model(noisy, x).sum().item())
        total += b

    return {key: value / max(total, 1) for key, value in sums.items()} | {
        "num_samples": float(total)
    }


@torch.no_grad()
def sigma_distortion_study(
    ae: SphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    sigmas: Sequence[float],
    max_samples: int,
    dimension_scaled_noise: bool,
) -> List[Dict[str, float]]:
    ae.eval()
    batches: List[torch.Tensor] = []
    count = 0
    for x, _ in loader:
        if count >= max_samples:
            break
        remaining = max_samples - count
        x = x[:remaining].to(device)
        batches.append(x)
        count += x.size(0)
    images = torch.cat(batches, dim=0)
    z = ae.encode(images)
    rows = []
    for sigma in sigmas:
        zn = tangent_noise(z, sigma, dimension_scaled_noise)
        recon = ae.decode(zn)
        mse = F.mse_loss((recon + 1) * 0.5, (images + 1) * 0.5).item()
        cosine = F.cosine_similarity(z, zn, dim=1).mean().item()
        angle = torch.acos((z * zn).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)).mean().item()
        rows.append(
            {
                "sigma_enc": float(sigma),
                "reconstruction_mse": float(mse),
                "mean_cosine": float(cosine),
                "mean_geodesic_angle": float(angle),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Flow-matching objectives
# -----------------------------------------------------------------------------


def sample_time(batch: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch, device=device)
    if mode == "logit_normal":
        raw = torch.randn(batch, device=device)
        return torch.sigmoid(raw)
    raise ValueError(f"Unknown time sampling mode: {mode}")


def flow_matching_loss(
    method: str,
    field: LatentVectorField,
    ae: SphericalAutoencoder,
    images: torch.Tensor,
    sigma_enc: float,
    dimension_scaled_noise: bool,
    time_sampling: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Matched loss for chord, RFM, or SRUL.

    All methods use the same target z0 and the same sphere source epsilon.
    Only the source distribution, probability path, tangent constraint, and Jacobi weight differ.
    """
    with torch.no_grad():
        z_clean = ae.encode(images)
        z0 = tangent_noise(z_clean, sigma_enc, dimension_scaled_noise)

    batch = images.size(0)
    eps = sample_uniform_sphere(batch, ae.latent_dim, images.device)
    t = sample_time(batch, images.device, time_sampling)

    if method == "gaussian_fm":
        gaussian_source = torch.randn_like(z0)
        zt = (1.0 - t)[:, None] * z0 + t[:, None] * gaussian_source
        target = gaussian_source - z0
        raw_pred = field(zt, t)
        prediction = raw_pred
        weights = torch.ones(batch, device=images.device)
        source_direction = normalize_to_sphere(gaussian_source)
        omega = torch.acos(
            (z0 * source_direction).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
        )
    elif method == "chord":
        zt = (1.0 - t)[:, None] * z0 + t[:, None] * eps
        target = eps - z0
        raw_pred = field(zt, t)
        prediction = raw_pred
        weights = torch.ones(batch, device=images.device)
        omega = torch.acos((z0 * eps).sum(1).clamp(-1 + 1e-6, 1 - 1e-6))
    elif method in {"rfm", "srul"}:
        zt, omega = slerp(z0, eps, t)
        target = slerp_velocity(z0, eps, t, omega)
        target = tangent_projection(zt, target)
        raw_pred = field(zt, t)
        prediction = tangent_projection(zt, raw_pred)
        weights = (
            sinc_sq((1.0 - t) * omega)
            if method == "srul"
            else torch.ones(batch, device=images.device)
        )
    else:
        raise ValueError(f"Unsupported flow method: {method}")

    # Per-coordinate mean makes loss scales comparable across latent dimensions.
    per_sample = (prediction - target).pow(2).mean(dim=1)
    loss = (weights * per_sample).mean()

    return loss, {
        "loss": float(loss.detach().item()),
        "mean_path_norm": float(zt.norm(dim=1).mean().detach().item()),
        "mean_abs_norm_error": float((zt.norm(dim=1) - 1.0).abs().mean().detach().item()),
        "mean_omega": float(omega.mean().detach().item()),
        "mean_weight": float(weights.mean().detach().item()),
        "raw_radial_fraction": float(radial_velocity_fraction(zt, raw_pred).detach().item()),
    }


# -----------------------------------------------------------------------------
# Latent DDPM baseline
# -----------------------------------------------------------------------------


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def create(cls, timesteps: int, device: torch.device) -> "DiffusionSchedule":
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def ddpm_loss(
    denoiser: LatentVectorField,
    ae: SphericalAutoencoder,
    images: torch.Tensor,
    sigma_enc: float,
    dimension_scaled_noise: bool,
    schedule: DiffusionSchedule,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        z = ae.encode(images)
        z0 = tangent_noise(z, sigma_enc, dimension_scaled_noise)
    batch = images.size(0)
    indices = torch.randint(0, schedule.alpha_bars.numel(), (batch,), device=images.device)
    alpha_bar = schedule.alpha_bars[indices]
    noise = torch.randn_like(z0)
    zt = alpha_bar.sqrt()[:, None] * z0 + (1.0 - alpha_bar).sqrt()[:, None] * noise
    t = indices.float() / max(schedule.alpha_bars.numel() - 1, 1)
    # x0-prediction is used here because the target latent lies on a sphere.
    # Normalizing the prediction makes the final support explicit while the
    # forward corruption is still the standard Gaussian diffusion process.
    predicted_x0 = normalize_to_sphere(denoiser(zt, t))
    loss = F.mse_loss(predicted_x0, z0)
    return loss, {
        "loss": float(loss.detach().item()),
        "mean_path_norm": float(zt.norm(dim=1).mean().detach().item()),
        "mean_abs_norm_error": float((zt.norm(dim=1) - 1.0).abs().mean().detach().item()),
    }


# -----------------------------------------------------------------------------
# Checkpointed training
# -----------------------------------------------------------------------------


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


def make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def train_autoencoder(
    ae: SphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    sigma_enc: float,
    lambda_noisy: float,
    lambda_latent: float,
    dimension_scaled_noise: bool,
    checkpoint_path: Path,
    checkpoint_every: int,
    amp: bool,
    resume: bool,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    scaler = make_scaler(device, amp)
    start_epoch = 1
    logs: List[Dict[str, float]] = []

    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ae.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        logs = list(state.get("logs", []))
        print(f"[AE] resumed from epoch {start_epoch - 1}")

    ae.train()
    for epoch in range(start_epoch, epochs + 1):
        running: MutableMapping[str, float] = {
            "loss": 0.0,
            "clean_l1": 0.0,
            "noisy_l1": 0.0,
            "latent_consistency": 0.0,
        }
        seen = 0
        began = time.time()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                loss, stats = autoencoder_loss(
                    ae,
                    images,
                    sigma_enc,
                    lambda_noisy,
                    lambda_latent,
                    dimension_scaled_noise,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            b = images.size(0)
            for key in running:
                running[key] += stats[key] * b
            seen += b

        row = {key: value / max(seen, 1) for key, value in running.items()}
        row.update({"epoch": epoch, "seconds": time.time() - began})
        logs.append(row)
        print(
            f"[AE] {epoch:03d}/{epochs} "
            f"loss={row['loss']:.4f} clean={row['clean_l1']:.4f} "
            f"noisy={row['noisy_l1']:.4f} lat={row['latent_consistency']:.4f}"
        )

        if epoch % checkpoint_every == 0 or epoch == epochs:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model": ae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "logs": logs,
                },
                checkpoint_path,
            )
    return logs


def train_prior(
    method: str,
    model: LatentVectorField,
    ae: SphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    sigma_enc: float,
    dimension_scaled_noise: bool,
    time_sampling: str,
    diffusion_steps: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    amp: bool,
    resume: bool,
) -> List[Dict[str, float]]:
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    ae.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = make_scaler(device, amp)
    schedule = (
        DiffusionSchedule.create(diffusion_steps, device)
        if method == "latent_ddpm"
        else None
    )
    start_epoch = 1
    logs: List[Dict[str, float]] = []

    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        logs = list(state.get("logs", []))
        print(f"[{method}] resumed from epoch {start_epoch - 1}")

    model.train()
    for epoch in range(start_epoch, epochs + 1):
        running: MutableMapping[str, float] = {}
        seen = 0
        began = time.time()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                if method == "latent_ddpm":
                    assert schedule is not None
                    loss, stats = ddpm_loss(
                        model,
                        ae,
                        images,
                        sigma_enc,
                        dimension_scaled_noise,
                        schedule,
                    )
                else:
                    loss, stats = flow_matching_loss(
                        method,
                        model,
                        ae,
                        images,
                        sigma_enc,
                        dimension_scaled_noise,
                        time_sampling,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            b = images.size(0)
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + value * b
            seen += b

        row = {key: value / max(seen, 1) for key, value in running.items()}
        row.update({"epoch": epoch, "seconds": time.time() - began})
        logs.append(row)
        print(
            f"[{method}] {epoch:03d}/{epochs} loss={row['loss']:.5f} "
            f"path_norm={row.get('mean_path_norm', float('nan')):.4f}"
        )

        if epoch % checkpoint_every == 0 or epoch == epochs:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "logs": logs,
                },
                checkpoint_path,
            )
    return logs


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_flow(
    method: str,
    field: LatentVectorField,
    ae: SphericalAutoencoder,
    num_samples: int,
    steps: int,
    device: torch.device,
    batch_size: int,
    project_final_for_decoder: bool,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    field.eval()
    ae.eval()
    all_images: List[torch.Tensor] = []
    all_final_norms: List[torch.Tensor] = []
    aggregate_norm_path = torch.zeros(steps + 1, device=device)
    aggregate_radial_path = torch.zeros(steps, device=device)
    batches = 0

    for start in range(0, num_samples, batch_size):
        b = min(batch_size, num_samples - start)
        z = (
            torch.randn(b, ae.latent_dim, device=device)
            if method == "gaussian_fm"
            else sample_uniform_sphere(b, ae.latent_dim, device)
        )
        norm_path = [z.norm(dim=1).mean()]
        radial_path = []

        for index in range(steps, 0, -1):
            t = torch.full((b,), index / steps, device=device)
            raw_velocity = field(z, t)
            radial_path.append(radial_velocity_fraction(z, raw_velocity))
            dt = -1.0 / steps
            if method in {"gaussian_fm", "chord"}:
                z = z + dt * raw_velocity
            elif method in {"rfm", "srul"}:
                velocity = tangent_projection(z, raw_velocity)
                z = exp_map_sphere(z, dt * velocity)
            else:
                raise ValueError(method)
            norm_path.append(z.norm(dim=1).mean())

        raw_final = z
        decode_z = (
            normalize_to_sphere(raw_final)
            if project_final_for_decoder and method in {"gaussian_fm", "chord"}
            else raw_final
        )
        images = ae.decode(decode_z)
        all_images.append(images.cpu())
        all_final_norms.append(raw_final.norm(dim=1).cpu())
        aggregate_norm_path += torch.stack(norm_path)
        aggregate_radial_path += torch.stack(radial_path)
        batches += 1

    images = torch.cat(all_images, dim=0)
    final_norms = torch.cat(all_final_norms)
    norm_path = (aggregate_norm_path / max(batches, 1)).cpu().tolist()
    radial_path = (aggregate_radial_path / max(batches, 1)).cpu().tolist()
    geometry = {
        "final_mean_norm": float(final_norms.mean().item()),
        "final_std_norm": float(final_norms.std().item()),
        "mean_abs_final_norm_error": float((final_norms - 1).abs().mean().item()),
        "min_mean_path_norm": float(min(norm_path)),
        "mean_abs_path_norm_error": float(np.mean(np.abs(np.asarray(norm_path) - 1.0))),
        "mean_raw_radial_fraction": float(np.mean(radial_path)),
        "norm_path": norm_path,
        "radial_fraction_path": radial_path,
        "final_projection_for_decoder": bool(
            project_final_for_decoder and method in {"gaussian_fm", "chord"}
        ),
    }
    return images, geometry


@torch.no_grad()
def sample_ddim(
    denoiser: LatentVectorField,
    ae: SphericalAutoencoder,
    num_samples: int,
    sample_steps: int,
    diffusion_steps: int,
    device: torch.device,
    batch_size: int,
    project_final_for_decoder: bool,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    denoiser.eval()
    ae.eval()
    schedule = DiffusionSchedule.create(diffusion_steps, device)
    all_images: List[torch.Tensor] = []
    all_final_norms: List[torch.Tensor] = []
    step_indices = torch.linspace(
        diffusion_steps - 1, 0, sample_steps, device=device
    ).round().long().unique_consecutive()
    aggregate_norm_path = torch.zeros(len(step_indices) + 1, device=device)
    batches = 0

    for start in range(0, num_samples, batch_size):
        b = min(batch_size, num_samples - start)
        z = torch.randn(b, ae.latent_dim, device=device)
        norm_path = [z.norm(dim=1).mean()]

        for i, current in enumerate(step_indices):
            current_i = int(current.item())
            previous_i = (
                int(step_indices[i + 1].item()) if i + 1 < len(step_indices) else -1
            )
            alpha_bar = schedule.alpha_bars[current_i]
            t = torch.full(
                (b,), current_i / max(diffusion_steps - 1, 1), device=device
            )
            predicted_x0 = normalize_to_sphere(denoiser(z, t))
            predicted_noise = (
                z - torch.sqrt(alpha_bar) * predicted_x0
            ) / torch.sqrt(1.0 - alpha_bar).clamp_min(1e-6)
            # Mild clipping avoids a single poor early prediction creating an
            # extreme DDIM trajectory in short proof-of-concept runs.
            predicted_noise = predicted_noise.clamp(-5.0, 5.0)
            if previous_i < 0:
                z = predicted_x0
            else:
                previous_alpha_bar = schedule.alpha_bars[previous_i]
                # Deterministic DDIM update (eta=0).
                z = (
                    torch.sqrt(previous_alpha_bar) * predicted_x0
                    + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
                )
            norm_path.append(z.norm(dim=1).mean())

        raw_final = z
        decode_z = normalize_to_sphere(raw_final) if project_final_for_decoder else raw_final
        images = ae.decode(decode_z)
        all_images.append(images.cpu())
        all_final_norms.append(raw_final.norm(dim=1).cpu())
        aggregate_norm_path += torch.stack(norm_path)
        batches += 1

    images = torch.cat(all_images, dim=0)
    final_norms = torch.cat(all_final_norms)
    norm_path = (aggregate_norm_path / max(batches, 1)).cpu().tolist()
    geometry = {
        "final_mean_norm": float(final_norms.mean().item()),
        "final_std_norm": float(final_norms.std().item()),
        "mean_abs_final_norm_error": float((final_norms - 1).abs().mean().item()),
        "min_mean_path_norm": float(min(norm_path)),
        "mean_abs_path_norm_error": float(np.mean(np.abs(np.asarray(norm_path) - 1.0))),
        "mean_raw_radial_fraction": float("nan"),
        "norm_path": norm_path,
        "final_projection_for_decoder": bool(project_final_for_decoder),
    }
    return images, geometry


# -----------------------------------------------------------------------------
# Image metrics
# -----------------------------------------------------------------------------


def to_uint8(images_minus1_1: torch.Tensor) -> torch.Tensor:
    return (((images_minus1_1.clamp(-1, 1) + 1.0) * 127.5).round()).to(torch.uint8)


@torch.no_grad()
def collect_real_uint8(loader: DataLoader, max_samples: int) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    count = 0
    for x, _ in loader:
        if count >= max_samples:
            break
        remaining = max_samples - count
        x = x[:remaining]
        collected.append(to_uint8(x))
        count += x.size(0)
    return torch.cat(collected, dim=0)


def compute_fid_kid(
    real_uint8: torch.Tensor,
    fake_uint8: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if FrechetInceptionDistance is None or KernelInceptionDistance is None:
        raise RuntimeError(
            "torchmetrics image dependencies are unavailable. In Colab run: "
            "pip install 'torchmetrics[image]' torch-fidelity"
        )
    n = min(real_uint8.size(0), fake_uint8.size(0))
    real_uint8 = real_uint8[:n]
    fake_uint8 = fake_uint8[:n]
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    subset_size = max(10, min(1000, n // 2))
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=min(50, max(5, n // subset_size)),
        subset_size=subset_size,
        normalize=False,
    ).to(device)

    for start in range(0, n, batch_size):
        real = real_uint8[start : start + batch_size].to(device)
        fake = fake_uint8[start : start + batch_size].to(device)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        kid.update(real, real=True)
        kid.update(fake, real=False)
    fid_value = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    return {
        "fid": fid_value,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "metric_samples": float(n),
    }


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        self.model = model.eval()
        self.register_buffer(
            "mean", torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
        )

    def forward(self, uint8_images: torch.Tensor) -> torch.Tensor:
        x = uint8_images.float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)


@torch.no_grad()
def extract_resnet_features(
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    extractor = ResNet18FeatureExtractor().to(device).eval()
    features: List[torch.Tensor] = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start : start + batch_size].to(device)
        features.append(extractor(batch).float().cpu())
    return torch.cat(features, dim=0)


def kth_neighbor_radii(
    features: torch.Tensor,
    k: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    features = features.to(device)
    radii: List[torch.Tensor] = []
    n = features.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = torch.cdist(features[start:end], features)
        local_rows = torch.arange(end - start, device=device)
        global_cols = torch.arange(start, end, device=device)
        distances[local_rows, global_cols] = float("inf")
        radius = distances.kthvalue(k, dim=1).values
        radii.append(radius.cpu())
    return torch.cat(radii, dim=0)


def fraction_inside_manifold(
    query: torch.Tensor,
    support: torch.Tensor,
    support_radii: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> float:
    support = support.to(device)
    support_radii = support_radii.to(device)
    accepted = 0
    total = query.size(0)
    for start in range(0, total, chunk_size):
        q = query[start : start + chunk_size].to(device)
        distances = torch.cdist(q, support)
        inside = (distances <= support_radii[None, :]).any(dim=1)
        accepted += int(inside.sum().item())
    return accepted / max(total, 1)


def feature_precision_recall(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    device: torch.device,
    nearest_k: int = 5,
    chunk_size: int = 256,
) -> Dict[str, float]:
    """Improved precision/recall style manifold estimate.

    Features come from ImageNet-pretrained ResNet-18. Therefore these values
    should be labeled feature precision/recall, not treated as a universal
    ground-truth metric.
    """
    n = min(real_features.size(0), fake_features.size(0))
    real = real_features[:n].float()
    fake = fake_features[:n].float()
    k = min(nearest_k, n - 1)
    real_radii = kth_neighbor_radii(real, k, device, chunk_size)
    fake_radii = kth_neighbor_radii(fake, k, device, chunk_size)
    precision = fraction_inside_manifold(fake, real, real_radii, device, chunk_size)
    recall = fraction_inside_manifold(real, fake, fake_radii, device, chunk_size)
    return {
        "feature_precision": float(precision),
        "feature_recall": float(recall),
        "pr_samples": float(n),
        "pr_nearest_k": float(k),
    }


# -----------------------------------------------------------------------------
# Plots and visual outputs
# -----------------------------------------------------------------------------


def save_reconstruction_grid(
    ae: SphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    sigma_enc: float,
    dimension_scaled_noise: bool,
    path: Path,
) -> None:
    ae.eval()
    images, _ = next(iter(loader))
    images = images[:16].to(device)
    with torch.no_grad():
        z = ae.encode(images)
        zn = tangent_noise(z, sigma_enc, dimension_scaled_noise)
        clean = ae.decode(z)
        noisy = ae.decode(zn)
    grid = torch.cat([images, clean, noisy], dim=0)
    save_image((grid + 1.0) * 0.5, path, nrow=16)


def save_method_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    save_image((images[:64] + 1.0) * 0.5, path, nrow=nrow)


def plot_training_curves(
    ae_logs: Sequence[Mapping[str, float]],
    prior_logs: Mapping[str, Sequence[Mapping[str, float]]],
    path: Path,
) -> None:
    if plt is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if ae_logs:
        axes[0].plot([r["epoch"] for r in ae_logs], [r["loss"] for r in ae_logs], label="total")
        axes[0].plot([r["epoch"] for r in ae_logs], [r["clean_l1"] for r in ae_logs], label="clean L1")
        axes[0].plot([r["epoch"] for r in ae_logs], [r["noisy_l1"] for r in ae_logs], label="noisy L1")
        axes[0].set_title("Autoencoder training")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

    for method, rows in prior_logs.items():
        if not rows:
            continue
        losses = np.asarray([float(r["loss"]) for r in rows])
        normalized = losses / max(losses[0], 1e-12)
        axes[1].plot([r["epoch"] for r in rows], normalized, label=method)
    axes[1].set_title("Prior loss / first-epoch loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Normalized loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sigma_study(rows: Sequence[Mapping[str, float]], path: Path) -> None:
    if plt is None or not rows:
        return
    sigmas = [float(r["sigma_enc"]) for r in rows]
    mses = [float(r["reconstruction_mse"]) for r in rows]
    angles = [float(r["mean_geodesic_angle"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
    ax1.plot(sigmas, mses, marker="o", label="Reconstruction MSE")
    ax1.set_xlabel(r"$\sigma_{enc}$")
    ax1.set_ylabel("Reconstruction MSE")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(sigmas, angles, marker="s", linestyle="--", label="Mean angle")
    ax2.set_ylabel("Mean geodesic angle (rad)")
    fig.suptitle("Tangent-noise distortion study")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_norm_paths(
    geometry: Mapping[str, Mapping[str, object]], path: Path
) -> None:
    if plt is None:
        return
    plt.figure(figsize=(6.0, 4.0))
    for method, metrics in geometry.items():
        path_values = metrics.get("norm_path")
        if isinstance(path_values, list):
            plt.plot(path_values, label=method)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Sampling step")
    plt.ylabel(r"Mean $\|z_t\|_2$")
    plt.title("Latent norm during sampling")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_metric_bars(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if plt is None or not rows:
        return
    methods = [str(r["method"]) for r in rows]
    fid_values = [float(r.get("fid", np.nan)) for r in rows]
    precision = [float(r.get("feature_precision", np.nan)) for r in rows]
    recall = [float(r.get("feature_recall", np.nan)) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].bar(methods, fid_values)
    axes[0].set_title("FID ↓")
    axes[1].bar(methods, precision)
    axes[1].set_title("Feature precision ↑")
    axes[2].bar(methods, recall)
    axes[2].set_title("Feature recall ↑")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    dataset: str
    data_root: str
    out_dir: str
    seed: int
    methods: List[str]
    train_samples: Optional[int]
    test_samples: Optional[int]
    latent_dim: int
    base_channels: int
    batch_size: int
    num_workers: int
    ae_epochs: int
    prior_epochs: int
    ae_lr: float
    prior_lr: float
    sigma_enc: float
    sigma_values: List[float]
    lambda_noisy: float
    lambda_latent: float
    dimension_scaled_noise: bool
    prior_width: int
    prior_depth: int
    time_dim: int
    time_sampling: str
    diffusion_steps: int
    sample_steps: int
    metric_samples: int
    pr_samples: int
    recon_metric_samples: int
    metric_batch_size: int
    pr_batch_size: int
    pr_chunk_size: int
    pr_nearest_k: int
    checkpoint_every: int
    amp: bool
    resume: bool
    skip_heavy_metrics: bool
    compute_lpips: bool
    project_chord_final: bool
    project_ddpm_final: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled SRUL vs chord/RFM/latent-DDPM experiment on CIFAR-10."
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gaussian_fm", "chord", "rfm", "srul", "latent_ddpm"],
        choices=["gaussian_fm", "chord", "rfm", "srul", "latent_ddpm"],
    )
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=30)
    parser.add_argument("--ae-lr", type=float, default=2e-4)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--sigma-enc", type=float, default=0.20)
    parser.add_argument(
        "--sigma-values",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
    )
    parser.add_argument("--lambda-noisy", type=float, default=1.0)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument(
        "--raw-tangent-noise",
        action="store_true",
        help="Do not divide tangent Gaussian by sqrt(d-1).",
    )
    parser.add_argument("--prior-width", type=int, default=512)
    parser.add_argument("--prior-depth", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument(
        "--time-sampling", default="uniform", choices=["uniform", "logit_normal"]
    )
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--metric-samples", type=int, default=5000)
    parser.add_argument("--pr-samples", type=int, default=2000)
    parser.add_argument("--recon-metric-samples", type=int, default=2000)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--pr-batch-size", type=int, default=128)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument(
        "--no-project-chord-final",
        action="store_true",
        help="Decode raw chord output instead of final sphere projection.",
    )
    parser.add_argument(
        "--no-project-ddpm-final",
        action="store_true",
        help="Decode raw DDPM output instead of final sphere projection.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    train_samples = None if args.train_samples <= 0 else args.train_samples
    test_samples = None if args.test_samples <= 0 else args.test_samples
    return ExperimentConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        methods=list(dict.fromkeys(args.methods)),
        train_samples=train_samples,
        test_samples=test_samples,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ae_epochs=args.ae_epochs,
        prior_epochs=args.prior_epochs,
        ae_lr=args.ae_lr,
        prior_lr=args.prior_lr,
        sigma_enc=args.sigma_enc,
        sigma_values=list(args.sigma_values),
        lambda_noisy=args.lambda_noisy,
        lambda_latent=args.lambda_latent,
        dimension_scaled_noise=not args.raw_tangent_noise,
        prior_width=args.prior_width,
        prior_depth=args.prior_depth,
        time_dim=args.time_dim,
        time_sampling=args.time_sampling,
        diffusion_steps=args.diffusion_steps,
        sample_steps=args.sample_steps,
        metric_samples=args.metric_samples,
        pr_samples=args.pr_samples,
        recon_metric_samples=args.recon_metric_samples,
        metric_batch_size=args.metric_batch_size,
        pr_batch_size=args.pr_batch_size,
        pr_chunk_size=args.pr_chunk_size,
        pr_nearest_k=args.pr_nearest_k,
        checkpoint_every=args.checkpoint_every,
        amp=args.amp,
        resume=args.resume,
        skip_heavy_metrics=args.skip_heavy_metrics,
        compute_lpips=args.compute_lpips,
        project_chord_final=not args.no_project_chord_final,
        project_ddpm_final=not args.no_project_ddpm_final,
    )


def run_experiment(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    checkpoints = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    figures = ensure_dir(run_dir / "figures")
    samples_dir = ensure_dir(run_dir / "samples")
    save_json(asdict(config), run_dir / "config.json")

    print("=" * 80)
    print("SRUL CIFAR-10 experiment")
    print("Device:", device)
    print("Run directory:", run_dir)
    print("Methods:", config.methods)
    print("=" * 80)

    train_set, test_set = make_datasets(
        config.dataset,
        config.data_root,
        config.train_samples,
        config.test_samples,
        config.seed,
    )
    train_loader, test_loader = make_loaders(
        train_set,
        test_set,
        config.batch_size,
        config.num_workers,
        device,
    )

    ae = SphericalAutoencoder(
        config.latent_dim, base_channels=config.base_channels
    ).to(device)
    ae_checkpoint = checkpoints / "autoencoder_latest.pt"
    ae_logs = train_autoencoder(
        ae,
        train_loader,
        device,
        config.ae_epochs,
        config.ae_lr,
        config.sigma_enc,
        config.lambda_noisy,
        config.lambda_latent,
        config.dimension_scaled_noise,
        ae_checkpoint,
        config.checkpoint_every,
        config.amp,
        config.resume,
    )
    save_rows_csv(ae_logs, logs_dir / "autoencoder.csv")
    atomic_torch_save(ae.state_dict(), checkpoints / "autoencoder_final.pt")

    recon_metrics = reconstruction_metrics(
        ae,
        test_loader,
        device,
        config.sigma_enc,
        min(config.recon_metric_samples, len(test_set)),
        config.dimension_scaled_noise,
        config.compute_lpips,
    )
    save_json(recon_metrics, run_dir / "reconstruction_metrics.json")
    save_reconstruction_grid(
        ae,
        test_loader,
        device,
        config.sigma_enc,
        config.dimension_scaled_noise,
        figures / "reconstructions.png",
    )

    sigma_rows = sigma_distortion_study(
        ae,
        test_loader,
        device,
        config.sigma_values,
        min(config.recon_metric_samples, len(test_set)),
        config.dimension_scaled_noise,
    )
    save_rows_csv(sigma_rows, logs_dir / "sigma_distortion.csv")
    plot_sigma_study(sigma_rows, figures / "sigma_distortion.png")

    prior_models: Dict[str, LatentVectorField] = {}
    prior_logs: Dict[str, List[Dict[str, float]]] = {}
    for method in config.methods:
        print("\n" + "-" * 80)
        print(f"Training prior: {method}")
        model = LatentVectorField(
            latent_dim=config.latent_dim,
            width=config.prior_width,
            depth=config.prior_depth,
            time_dim=config.time_dim,
        ).to(device)
        checkpoint_path = checkpoints / f"{method}_latest.pt"
        logs = train_prior(
            method,
            model,
            ae,
            train_loader,
            device,
            config.prior_epochs,
            config.prior_lr,
            config.sigma_enc,
            config.dimension_scaled_noise,
            config.time_sampling,
            config.diffusion_steps,
            checkpoint_path,
            config.checkpoint_every,
            config.amp,
            config.resume,
        )
        prior_models[method] = model
        prior_logs[method] = logs
        save_rows_csv(logs, logs_dir / f"{method}.csv")
        atomic_torch_save(model.state_dict(), checkpoints / f"{method}_final.pt")

    plot_training_curves(ae_logs, prior_logs, figures / "training_curves.png")

    requested_real = min(config.metric_samples, len(test_set))
    real_uint8 = collect_real_uint8(test_loader, requested_real)
    pr_real_uint8 = real_uint8[: min(config.pr_samples, real_uint8.size(0))]
    real_pr_features: Optional[torch.Tensor] = None
    if not config.skip_heavy_metrics:
        print("Extracting real ResNet-18 features for feature precision/recall...")
        real_pr_features = extract_resnet_features(
            pr_real_uint8, device, config.pr_batch_size
        )

    metric_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    geometry_full: Dict[str, Dict[str, object]] = {}

    for method in config.methods:
        print("\n" + "-" * 80)
        print(f"Sampling and evaluating: {method}")
        model = prior_models[method]
        if method == "latent_ddpm":
            generated, geometry = sample_ddim(
                model,
                ae,
                requested_real,
                config.sample_steps,
                config.diffusion_steps,
                device,
                config.metric_batch_size,
                config.project_ddpm_final,
            )
        else:
            generated, geometry = sample_flow(
                method,
                model,
                ae,
                requested_real,
                config.sample_steps,
                device,
                config.metric_batch_size,
                config.project_chord_final,
            )
        geometry_full[method] = geometry
        save_method_grid(generated, samples_dir / f"{method}.png")
        fake_uint8 = to_uint8(generated)

        row: Dict[str, object] = {"method": method, "seed": config.seed}
        row.update(
            {
                key: value
                for key, value in geometry.items()
                if not isinstance(value, list)
            }
        )
        geometry_rows.append(dict(row))

        if config.skip_heavy_metrics:
            row.update(
                {
                    "fid": float("nan"),
                    "kid_mean": float("nan"),
                    "kid_std": float("nan"),
                    "feature_precision": float("nan"),
                    "feature_recall": float("nan"),
                }
            )
        else:
            print("Computing FID/KID...")
            row.update(
                compute_fid_kid(
                    real_uint8,
                    fake_uint8,
                    device,
                    config.metric_batch_size,
                )
            )
            print("Computing ResNet feature precision/recall...")
            assert real_pr_features is not None
            fake_pr_features = extract_resnet_features(
                fake_uint8[: real_pr_features.size(0)],
                device,
                config.pr_batch_size,
            )
            row.update(
                feature_precision_recall(
                    real_pr_features,
                    fake_pr_features,
                    device,
                    nearest_k=config.pr_nearest_k,
                    chunk_size=config.pr_chunk_size,
                )
            )
        metric_rows.append(row)
        print(json.dumps(row, indent=2))

    save_rows_csv(metric_rows, run_dir / "generation_metrics.csv")
    save_rows_csv(geometry_rows, run_dir / "geometry_metrics.csv")
    save_json(
        {method: metrics for method, metrics in geometry_full.items()},
        run_dir / "geometry_paths.json",
    )
    plot_norm_paths(geometry_full, figures / "sampling_norm_paths.png")
    plot_metric_bars(metric_rows, figures / "generation_metrics.png")

    summary = {
        "config": asdict(config),
        "reconstruction_metrics": recon_metrics,
        "sigma_distortion": sigma_rows,
        "generation_metrics": metric_rows,
        "geometry_metrics": geometry_rows,
    }
    save_json(summary, run_dir / "summary.json")

    print("\nExperiment complete.")
    print("Results:", run_dir)
    print("Main files:")
    print(" - generation_metrics.csv")
    print(" - geometry_metrics.csv")
    print(" - reconstruction_metrics.json")
    print(" - figures/")
    print(" - samples/")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
