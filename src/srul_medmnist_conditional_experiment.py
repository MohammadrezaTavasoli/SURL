"""Train and evaluate class-conditional Riemannian Flow Matching on CIFAR-10 or MedMNIST spatial spherical latents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import save_image

try:
    import medmnist
except Exception:
    medmnist = None

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
    import lpips
except Exception:
    lpips = None


# =============================================================================
# Configuration and utility helpers
# =============================================================================


@dataclass
class ExperimentConfig:
    dataset: str = "pathmnist"
    data_root: str = "./data"
    out_dir: str = "./SRUL_PathMNIST_conditional"
    seed: int = 0

    train_samples: Optional[int] = None
    test_samples: Optional[int] = 10000
    batch_size: int = 128
    num_workers: int = 2
    num_classes: int = 9

    base_channels: int = 96
    latent_channels: int = 32
    latent_size: int = 4  # fixed by the encoder architecture for 32x32 images

    ae_epochs: int = 60
    ae_lr: float = 2e-4
    lambda_clean: float = 1.0
    lambda_noisy: float = 0.5
    lambda_edge: float = 0.15
    lambda_latent: float = 0.05
    lambda_lpips: float = 0.10
    sigma_enc: float = 0.15
    dimension_scaled_noise: bool = True
    ae_checkpoint: Optional[str] = None
    skip_ae_training: bool = False

    prior_epochs: int = 120
    prior_lr: float = 2e-4
    prior_width: int = 256
    prior_depth: int = 6
    time_dim: int = 128
    time_sampling: str = "logit_normal"
    methods: Tuple[str, ...] = ("cond_rfm", "cond_srul")

    # Conditional SRUL-v2 settings.
    label_drop_prob: float = 0.10
    jacobi_alpha: float = 0.25
    ema_decay: float = 0.999
    guidance_scales: Tuple[float, ...] = (1.0, 1.5, 2.0)
    grid_samples_per_class: int = 8

    sample_steps: int = 100
    metric_samples: int = 10000
    pr_samples: int = 5000
    recon_metric_samples: int = 5000
    metric_batch_size: int = 128
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5

    checkpoint_every: int = 5
    resume: bool = False
    amp: bool = False
    ae_only: bool = False
    eval_only: bool = False
    skip_heavy_metrics: bool = False
    compute_lpips: bool = False


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


def atomic_torch_save(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


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


def load_state_dict_safely(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint structure: {path}")


def load_eval_state_dict(
    path: Path,
    device: torch.device,
    prefer_ema: bool = True,
) -> Mapping[str, torch.Tensor]:
    """Load EMA weights when available, otherwise load the raw model."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        if prefer_ema and "ema" in obj:
            return obj["ema"]
        if "model" in obj:
            return obj["model"]
        return obj
    raise ValueError(f"Unsupported checkpoint structure: {path}")


class ExponentialMovingAverage:
    """EMA over a module state_dict, including parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1).")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for key, value in current.items():
            value = value.detach()
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(
                    value, alpha=1.0 - self.decay
                )
            else:
                self.shadow[key].copy_(value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.shadow = {
            key: value.detach().clone()
            for key, value in state.items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()


# =============================================================================
# Dataset
# =============================================================================


def _single_label(target: object) -> int:
    """Convert MedMNIST's shape-(1,) label into a Python integer."""
    array = np.asarray(target).reshape(-1)
    if array.size != 1:
        raise ValueError(
            "This experiment requires a single-label classification dataset. "
            f"Received target shape {np.asarray(target).shape}."
        )
    return int(array[0])


def make_datasets(
    name: str,
    root: str,
    train_samples: Optional[int],
    test_samples: Optional[int],
    seed: int,
) -> Tuple[Dataset, Dataset]:
    # MedMNIST requires an explicitly supplied root directory to already
    # exist. Creating it here also makes the script robust in fresh Colab
    # runtimes, where /content/data may not yet be present.
    dataset_root = Path(root).expanduser()
    dataset_root.mkdir(parents=True, exist_ok=True)
    root = str(dataset_root)

    # The architecture expects RGB 32x32 inputs. PathMNIST and BloodMNIST are
    # distributed at 28x28 by default, so they are resized slightly to 32x32
    # to keep every model component unchanged.
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    if name == "cifar10":
        train_set: Dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        test_set: Dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
    elif name in {"pathmnist", "bloodmnist"}:
        if medmnist is None:
            raise RuntimeError(
                f"{name} requires the official MedMNIST package. "
                "Install it in Colab with: pip install medmnist"
            )
        dataset_class = (
            medmnist.PathMNIST if name == "pathmnist" else medmnist.BloodMNIST
        )
        train_set = dataset_class(
            split="train",
            root=root,
            download=True,
            size=28,
            transform=transform,
            target_transform=_single_label,
        )
        test_set = dataset_class(
            split="test",
            root=root,
            download=True,
            size=28,
            transform=transform,
            target_transform=_single_label,
        )
    elif name == "fake":
        train_size = train_samples or 1024
        test_size = test_samples or 256
        train_set = datasets.FakeData(
            size=train_size,
            image_size=(3, 32, 32),
            num_classes=9,
            transform=transform,
            random_offset=0,
        )
        test_set = datasets.FakeData(
            size=test_size,
            image_size=(3, 32, 32),
            num_classes=9,
            transform=transform,
            random_offset=100000,
        )
        return train_set, test_set
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    generator = torch.Generator().manual_seed(seed)

    if train_samples is not None and train_samples > 0 and train_samples < len(train_set):
        indices = torch.randperm(len(train_set), generator=generator)[:train_samples].tolist()
        train_set = Subset(train_set, indices)

    if test_samples is not None and test_samples > 0 and test_samples < len(test_set):
        indices = torch.randperm(len(test_set), generator=generator)[:test_samples].tolist()
        test_set = Subset(test_set, indices)

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
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


# =============================================================================
# Product-of-spheres geometry
# =============================================================================


def normalize_tokens(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each spatial token across the channel dimension.

    Input shape: [B, C, H, W]
    Every z[b, :, i, j] lies on S^{C-1} after normalization.
    """
    norm = z.pow(2).sum(dim=1, keepdim=True).sqrt()
    return z / norm.clamp_min(eps)


def tangent_projection_tokens(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    coefficient = (z * u).sum(dim=1, keepdim=True)
    return u - coefficient * z


def exp_map_tokens(z: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = tangent_projection_tokens(z, v)
    norm = v.pow(2).sum(dim=1, keepdim=True).sqrt()
    direction = v / norm.clamp_min(eps)
    out = torch.cos(norm) * z + torch.sin(norm) * direction
    tiny = norm < eps
    if tiny.any():
        fallback = normalize_tokens(z + v)
        out = torch.where(tiny.expand_as(out), fallback, out)
    return normalize_tokens(out)


def tangent_noise_tokens(
    z: torch.Tensor,
    sigma: float,
    dimension_scaled: bool = True,
) -> torch.Tensor:
    xi = torch.randn_like(z)
    u = tangent_projection_tokens(z, xi)
    if dimension_scaled:
        u = u / math.sqrt(max(z.size(1) - 1, 1))
    return exp_map_tokens(z, sigma * u)


def sample_uniform_product_spheres(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    return normalize_tokens(
        torch.randn(batch_size, channels, height, width, device=device)
    )


def slerp_tokens(
    x: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenwise SLERP. x is at t=0 and eps is at t=1."""
    dot = (x * eps).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # [B,H,W]
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    t_map = t[:, None, None]
    safe = sin_omega.abs() > 1e-5

    a = torch.where(
        safe,
        torch.sin((1.0 - t_map) * omega) / sin_omega.clamp_min(1e-6),
        1.0 - t_map,
    )
    b = torch.where(
        safe,
        torch.sin(t_map * omega) / sin_omega.clamp_min(1e-6),
        t_map,
    )
    zt = a[:, None] * x + b[:, None] * eps
    return normalize_tokens(zt), omega


def slerp_velocity_tokens(
    x: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    t_map = t[:, None, None]
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    scale = omega / sin_omega
    velocity = scale[:, None] * (
        torch.cos(t_map * omega)[:, None] * eps
        - torch.cos((1.0 - t_map) * omega)[:, None] * x
    )
    return velocity


def sinc_sq(x: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(x)
    mask = x.abs() > 1e-4
    result[mask] = (torch.sin(x[mask]) / x[mask]).pow(2)
    result[~mask] = 1.0 - x[~mask].pow(2) / 3.0
    return result


def token_norms(z: torch.Tensor) -> torch.Tensor:
    return z.pow(2).sum(dim=1).sqrt()  # [B,H,W]


def radial_velocity_fraction_tokens(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    numerator = (z * v).sum(dim=1).pow(2)
    denominator = z.pow(2).sum(dim=1) * v.pow(2).sum(dim=1) + 1e-8
    return (numerator / denominator).mean()


def mean_token_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1).mean()


def mean_token_geodesic_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(dot).mean()


# =============================================================================
# Autoencoder
# =============================================================================


def valid_groups(channels: int, target: int = 8) -> int:
    groups = min(target, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ResBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = valid_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SpatialEncoder(nn.Module):
    """3x32x32 -> Cx4x4 token grid."""

    def __init__(self, latent_channels: int, base: int = 96):
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
            nn.GroupNorm(valid_groups(base * 4), base * 4),
            nn.SiLU(),
            nn.Conv2d(base * 4, latent_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialDecoder(nn.Module):
    """Cx4x4 token grid -> 3x32x32."""

    def __init__(self, latent_channels: int, base: int = 96):
        super().__init__()
        self.input = nn.Conv2d(latent_channels, base * 4, 1)
        self.net = nn.Sequential(
            ResBlock2d(base * 4),
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
            nn.GroupNorm(valid_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.input(z))


class SpatialSphericalAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 32, base_channels: int = 96):
        super().__init__()
        self.encoder = SpatialEncoder(latent_channels, base_channels)
        self.decoder = SpatialDecoder(latent_channels, base_channels)
        self.latent_channels = latent_channels
        self.latent_size = 4

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_tokens(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class LPIPSLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        if lpips is None:
            raise RuntimeError("Install lpips with: pip install lpips")
        self.model = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Upsampling makes the perceptual network more stable on 32x32 inputs.
        x64 = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        y64 = F.interpolate(y, size=(64, 64), mode="bilinear", align_corners=False)
        return self.model(x64, y64).mean()


def edge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]
    dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


def autoencoder_objective(
    ae: SpatialSphericalAutoencoder,
    images: torch.Tensor,
    config: ExperimentConfig,
    perceptual: Optional[nn.Module],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z = ae.encode(images)
    z_noisy = tangent_noise_tokens(
        z, config.sigma_enc, config.dimension_scaled_noise
    )
    clean = ae.decode(z)
    noisy = ae.decode(z_noisy)

    clean_l1 = F.l1_loss(clean, images)
    noisy_l1 = F.l1_loss(noisy, images)
    edges = edge_loss(clean, images)

    # Re-encoding prevents many latent tokens from becoming decoder-equivalent.
    reencoded = ae.encode(noisy)
    latent_consistency = 1.0 - mean_token_cosine(z.detach(), reencoded)

    perceptual_loss = torch.zeros((), device=images.device)
    if perceptual is not None and config.lambda_lpips > 0:
        perceptual_loss = perceptual(clean, images)

    total = (
        config.lambda_clean * clean_l1
        + config.lambda_noisy * noisy_l1
        + config.lambda_edge * edges
        + config.lambda_latent * latent_consistency
        + config.lambda_lpips * perceptual_loss
    )

    return total, {
        "loss": float(total.detach().item()),
        "clean_l1": float(clean_l1.detach().item()),
        "noisy_l1": float(noisy_l1.detach().item()),
        "edge": float(edges.detach().item()),
        "latent": float(latent_consistency.detach().item()),
        "lpips": float(perceptual_loss.detach().item()),
    }


# =============================================================================
# Spatial time-conditioned vector field
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=t.device, dtype=t.dtype
        ) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        arguments = t[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat(
            [torch.sin(arguments), torch.cos(arguments)], dim=1
        )
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class SpatialPriorBlock(nn.Module):
    def __init__(self, width: int, conditioning_dim: int):
        super().__init__()
        groups = valid_groups(width)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conditioning = nn.Linear(conditioning_dim, width * 2)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        scale, shift = self.conditioning(conditioning).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SpatialVectorField(nn.Module):
    """Class-conditional spatial vector field with a null CFG label."""

    def __init__(
        self,
        latent_channels: int,
        width: int = 256,
        depth: int = 6,
        time_dim: int = 128,
        num_classes: int = 9,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.null_label = self.num_classes

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.label_embed = nn.Embedding(self.num_classes + 1, time_dim)
        nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)

        self.conditioning_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [SpatialPriorBlock(width, time_dim) for _ in range(depth)]
        )
        self.norm = nn.GroupNorm(valid_groups(width), width)
        self.output = nn.Conv2d(width, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is None:
            labels = torch.full(
                (z.size(0),),
                self.null_label,
                dtype=torch.long,
                device=z.device,
            )
        labels = labels.long().view(-1)
        if labels.numel() != z.size(0):
            raise ValueError("One label is required per latent sample.")

        conditioning = self.time_embed(t.float()) + self.label_embed(labels)
        conditioning = self.conditioning_mlp(conditioning)

        h = self.input(z)
        for block in self.blocks:
            h = block(h, conditioning)
        return self.output(F.silu(self.norm(h)))

def sample_time(batch_size: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch_size, device=device)
    if mode == "logit_normal":
        raw = torch.randn(batch_size, device=device)
        return torch.sigmoid(raw)
    raise ValueError(f"Unknown time sampling mode: {mode}")


def flow_prior_objective(
    method: str,
    field: SpatialVectorField,
    ae: SpatialSphericalAutoencoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conditional RFM/SRUL objective with classifier-free label dropout."""
    if method not in {"cond_rfm", "cond_srul"}:
        raise ValueError(f"Unsupported conditional method: {method}")

    with torch.no_grad():
        z_clean = ae.encode(images)
        target = tangent_noise_tokens(
            z_clean, config.sigma_enc, config.dimension_scaled_noise
        )

    batch = images.size(0)
    source = sample_uniform_product_spheres(
        batch,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        images.device,
    )
    t = sample_time(batch, images.device, config.time_sampling)

    zt, omega = slerp_tokens(target, source, t)
    target_velocity = slerp_velocity_tokens(target, source, t, omega)
    target_velocity = tangent_projection_tokens(zt, target_velocity)

    jacobi = sinc_sq((1.0 - t[:, None, None]) * omega)
    alpha = config.jacobi_alpha if method == "cond_srul" else 0.0
    weight = (1.0 - alpha) + alpha * jacobi

    labels = labels.long().to(images.device)
    drop_mask = torch.rand(batch, device=images.device) < config.label_drop_prob
    training_labels = labels.clone()
    training_labels[drop_mask] = field.null_label

    raw_prediction = field(zt, t, training_labels)
    prediction = tangent_projection_tokens(zt, raw_prediction)

    # Mean squared error per token, then soft Jacobi weighting tokenwise.
    squared_error = (prediction - target_velocity).pow(2).mean(dim=1)
    loss = (weight * squared_error).mean()

    norms = token_norms(zt)
    radial_fraction = radial_velocity_fraction_tokens(zt, raw_prediction)

    return loss, {
        "loss": float(loss.detach().item()),
        "mean_path_norm": float(norms.mean().detach().item()),
        "mean_abs_norm_error": float((norms - 1.0).abs().mean().detach().item()),
        "mean_omega": float(omega.mean().detach().item()),
        "mean_weight": float(weight.mean().detach().item()),
        "raw_radial_fraction": float(radial_fraction.detach().item()),
        "label_drop_fraction": float(drop_mask.float().mean().detach().item()),
        "jacobi_alpha": float(alpha),
    }


# =============================================================================
# Training and checkpointing
# =============================================================================


def train_autoencoder(
    ae: SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    checkpoints: Path,
    logs_dir: Path,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(ae.parameters(), lr=config.ae_lr, weight_decay=1e-4)
    scaler = make_grad_scaler(device, config.amp)
    latest = checkpoints / "autoencoder_latest.pt"
    final = checkpoints / "autoencoder_final.pt"
    log_path = logs_dir / "autoencoder.csv"

    start_epoch = 1
    rows: List[Dict[str, float]] = []

    if config.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        ae.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        rows = list(checkpoint.get("logs", []))
        print(f"[AE] resumed from epoch {start_epoch - 1}")
    elif config.eval_only and final.exists():
        ae.load_state_dict(load_state_dict_safely(final, device))
        return rows

    perceptual: Optional[nn.Module] = None
    if config.lambda_lpips > 0 and config.compute_lpips:
        perceptual = LPIPSLoss(device)

    ae.train()
    for epoch in range(start_epoch, config.ae_epochs + 1):
        start = time.time()
        sums = {
            "loss": 0.0,
            "clean_l1": 0.0,
            "noisy_l1": 0.0,
            "edge": 0.0,
            "latent": 0.0,
            "lpips": 0.0,
        }
        count = 0

        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, config.amp):
                loss, stats = autoencoder_objective(ae, images, config, perceptual)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = images.size(0)
            count += batch
            for key in sums:
                sums[key] += stats[key] * batch

        row = {key: value / max(count, 1) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["seconds"] = float(time.time() - start)
        rows.append(row)
        save_csv(rows, log_path)

        print(
            f"[AE] {epoch:03d}/{config.ae_epochs} "
            f"loss={row['loss']:.4f} clean={row['clean_l1']:.4f} "
            f"noisy={row['noisy_l1']:.4f} edge={row['edge']:.4f} "
            f"latent={row['latent']:.4f} lpips={row['lpips']:.4f}"
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.ae_epochs:
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model": ae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "logs": rows,
                    "config": asdict(config),
                },
                latest,
            )

    atomic_torch_save(ae.state_dict(), final)
    return rows


def train_prior(
    method: str,
    field: SpatialVectorField,
    ae: SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    checkpoints: Path,
    logs_dir: Path,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        field.parameters(), lr=config.prior_lr, weight_decay=1e-4
    )
    scaler = make_grad_scaler(device, config.amp)
    ema = ExponentialMovingAverage(field, decay=config.ema_decay)

    latest = checkpoints / f"{method}_latest.pt"
    final = checkpoints / f"{method}_final.pt"
    log_path = logs_dir / f"{method}.csv"

    start_epoch = 1
    rows: List[Dict[str, float]] = []

    if config.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        field.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        else:
            ema = ExponentialMovingAverage(field, decay=config.ema_decay)
        start_epoch = int(checkpoint["epoch"]) + 1
        rows = list(checkpoint.get("logs", []))
        print(f"[{method}] resumed from epoch {start_epoch - 1}")
    elif config.eval_only and final.exists():
        field.load_state_dict(load_eval_state_dict(final, device, prefer_ema=True))
        return rows

    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    field.train()
    for epoch in range(start_epoch, config.prior_epochs + 1):
        start = time.time()
        sums = {
            "loss": 0.0,
            "mean_path_norm": 0.0,
            "mean_abs_norm_error": 0.0,
            "mean_omega": 0.0,
            "mean_weight": 0.0,
            "raw_radial_fraction": 0.0,
            "label_drop_fraction": 0.0,
            "jacobi_alpha": 0.0,
        }
        count = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, config.amp):
                loss, stats = flow_prior_objective(
                    method, field, ae, images, labels, config
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(field)

            batch = images.size(0)
            count += batch
            for key in sums:
                sums[key] += stats[key] * batch

        row = {key: value / max(count, 1) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["seconds"] = float(time.time() - start)
        rows.append(row)
        save_csv(rows, log_path)

        print(
            f"[{method}] {epoch:03d}/{config.prior_epochs} "
            f"loss={row['loss']:.5f} path_norm={row['mean_path_norm']:.4f} "
            f"radial={row['raw_radial_fraction']:.4f} "
            f"drop={row['label_drop_fraction']:.3f} "
            f"alpha={row['jacobi_alpha']:.2f}"
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

    # Keep both raw and EMA states. Evaluation loads EMA by default.
    atomic_torch_save(
        {
            "model": field.state_dict(),
            "ema": ema.state_dict(),
            "config": asdict(config),
        },
        final,
    )
    return rows


# =============================================================================
# Sampling
# =============================================================================


def balanced_class_labels(
    num_samples: int,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return labels grouped by class with counts differing by at most one."""
    base = num_samples // num_classes
    remainder = num_samples % num_classes
    parts = []
    for class_id in range(num_classes):
        count = base + (1 if class_id < remainder else 0)
        if count > 0:
            parts.append(torch.full((count,), class_id, dtype=torch.long))
    labels = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long)
    if device is not None:
        labels = labels.to(device)
    return labels


@torch.no_grad()
def sample_flow_latents(
    method: str,
    field: SpatialVectorField,
    config: ExperimentConfig,
    device: torch.device,
    labels: torch.Tensor,
    guidance_scale: float,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Conditional sampling with classifier-free guidance."""
    if method not in {"cond_rfm", "cond_srul"}:
        raise ValueError(method)
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    field.eval()
    labels = labels.long().cpu()
    num_samples = labels.numel()

    all_latents: List[torch.Tensor] = []
    norm_paths: List[np.ndarray] = []
    radial_paths: List[np.ndarray] = []
    guidance_delta_paths: List[np.ndarray] = []

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        batch_labels = labels[start:stop].to(device)
        batch = batch_labels.numel()

        z = sample_uniform_product_spheres(
            batch,
            config.latent_channels,
            config.latent_size,
            config.latent_size,
            device,
        )
        null_labels = torch.full_like(batch_labels, field.null_label)

        local_norm_path = [float(token_norms(z).mean().item())]
        local_radial_path: List[float] = []
        local_guidance_delta: List[float] = []

        for index in range(config.sample_steps, 0, -1):
            t = torch.full(
                (batch,), index / config.sample_steps, device=device
            )
            conditional = field(z, t, batch_labels)
            unconditional = field(z, t, null_labels)
            raw_velocity = unconditional + guidance_scale * (
                conditional - unconditional
            )

            local_radial_path.append(
                float(radial_velocity_fraction_tokens(z, raw_velocity).item())
            )
            delta = (conditional - unconditional).pow(2).mean().sqrt()
            local_guidance_delta.append(float(delta.item()))

            velocity = tangent_projection_tokens(z, raw_velocity)
            dt = -1.0 / config.sample_steps
            z = exp_map_tokens(z, dt * velocity)
            local_norm_path.append(float(token_norms(z).mean().item()))

        all_latents.append(z.cpu())
        norm_paths.append(np.asarray(local_norm_path, dtype=np.float64))
        radial_paths.append(np.asarray(local_radial_path, dtype=np.float64))
        guidance_delta_paths.append(
            np.asarray(local_guidance_delta, dtype=np.float64)
        )

    latents = torch.cat(all_latents, dim=0)
    mean_norm_path = np.stack(norm_paths).mean(axis=0)
    mean_radial_path = np.stack(radial_paths).mean(axis=0)
    mean_guidance_delta_path = np.stack(guidance_delta_paths).mean(axis=0)
    final_norms = token_norms(latents)

    geometry = {
        "method": method,
        "guidance_scale": float(guidance_scale),
        "final_mean_norm": float(final_norms.mean().item()),
        "final_std_norm": float(final_norms.std().item()),
        "mean_abs_final_norm_error": float((final_norms - 1.0).abs().mean().item()),
        "min_mean_path_norm": float(mean_norm_path.min()),
        "mean_abs_path_norm_error": float(np.abs(mean_norm_path - 1.0).mean()),
        "mean_raw_radial_fraction": float(mean_radial_path.mean()),
        "mean_conditioning_delta": float(mean_guidance_delta_path.mean()),
        "norm_path": mean_norm_path.tolist(),
        "radial_path": mean_radial_path.tolist(),
        "conditioning_delta_path": mean_guidance_delta_path.tolist(),
        "final_projection_for_decoder": False,
    }
    return latents, geometry


@torch.no_grad()
def decode_latents(
    ae: SpatialSphericalAutoencoder,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    ae.eval()
    images: List[torch.Tensor] = []
    for start in range(0, latents.size(0), batch_size):
        z = latents[start : start + batch_size].to(device)
        images.append(ae.decode(z).cpu())
    return torch.cat(images, dim=0)


# =============================================================================
# Metrics
# =============================================================================


def to_uint8(images_minus1_1: torch.Tensor) -> torch.Tensor:
    return (((images_minus1_1.clamp(-1, 1) + 1.0) * 127.5).round()).to(torch.uint8)


@torch.no_grad()
def collect_real_images(loader: DataLoader, max_samples: int) -> torch.Tensor:
    images: List[torch.Tensor] = []
    count = 0
    for batch, _ in loader:
        remaining = max_samples - count
        batch = batch[:remaining]
        images.append(batch.cpu())
        count += batch.size(0)
        if count >= max_samples:
            break
    return torch.cat(images, dim=0)



def collect_real_labels(loader: DataLoader, max_samples: int) -> torch.Tensor:
    """Collect labels in the same order as collect_real_images.

    Sampling conditional generations with these labels matches the empirical
    class proportions of the evaluation set. This matters for PathMNIST,
    whose tissue classes are not assumed to be perfectly balanced.
    """
    labels_out: List[torch.Tensor] = []
    count = 0
    for _, labels in loader:
        labels = labels.long().view(-1)
        remaining = max_samples - count
        labels = labels[:remaining]
        labels_out.append(labels.cpu())
        count += labels.numel()
        if count >= max_samples:
            break
    return torch.cat(labels_out, dim=0)


def compute_fid_kid(
    real_uint8: torch.Tensor,
    fake_uint8: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if FrechetInceptionDistance is None or KernelInceptionDistance is None:
        raise RuntimeError(
            "Install image metrics in Colab with: "
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
        radii.append(distances.kthvalue(k, dim=1).values.cpu())
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


@torch.no_grad()
def collect_real_latents(
    ae: SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> torch.Tensor:
    ae.eval()
    latents: List[torch.Tensor] = []
    count = 0
    for images, _ in loader:
        images = images.to(device)
        z = ae.encode(images)
        remaining = max_samples - count
        z = z[:remaining]
        latents.append(z.flatten(1).cpu())
        count += z.size(0)
        if count >= max_samples:
            break
    return torch.cat(latents, dim=0)


@torch.no_grad()
def evaluate_reconstruction(
    ae: SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    ae.eval()
    real: List[torch.Tensor] = []
    clean: List[torch.Tensor] = []
    noisy: List[torch.Tensor] = []
    count = 0

    for images, _ in loader:
        images = images.to(device)
        z = ae.encode(images)
        z_noisy = tangent_noise_tokens(
            z, config.sigma_enc, config.dimension_scaled_noise
        )
        clean_recon = ae.decode(z)
        noisy_recon = ae.decode(z_noisy)

        remaining = config.recon_metric_samples - count
        real.append(images[:remaining].cpu())
        clean.append(clean_recon[:remaining].cpu())
        noisy.append(noisy_recon[:remaining].cpu())
        count += min(images.size(0), remaining)
        if count >= config.recon_metric_samples:
            break

    real_images = torch.cat(real, dim=0)
    clean_images = torch.cat(clean, dim=0)
    noisy_images = torch.cat(noisy, dim=0)

    clean_mse = F.mse_loss(clean_images, real_images).item()
    noisy_mse = F.mse_loss(noisy_images, real_images).item()
    clean_psnr = 10.0 * math.log10(4.0 / max(clean_mse, 1e-12))
    noisy_psnr = 10.0 * math.log10(4.0 / max(noisy_mse, 1e-12))

    metrics = {
        "clean_mse": float(clean_mse),
        "clean_psnr": float(clean_psnr),
        "noisy_mse": float(noisy_mse),
        "noisy_psnr": float(noisy_psnr),
        "num_samples": float(real_images.size(0)),
    }

    if structural_similarity_index_measure is not None:
        metrics["clean_ssim"] = float(
            structural_similarity_index_measure(
                clean_images, real_images, data_range=2.0
            ).item()
        )
        metrics["noisy_ssim"] = float(
            structural_similarity_index_measure(
                noisy_images, real_images, data_range=2.0
            ).item()
        )

    return metrics, real_images, clean_images, noisy_images


# =============================================================================
# Figures
# =============================================================================


def save_reconstruction_grid(
    real: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    path: Path,
) -> None:
    n = min(16, real.size(0))
    grid = torch.cat([real[:n], clean[:n], noisy[:n]], dim=0)
    save_image((grid + 1.0) * 0.5, path, nrow=n)


def save_method_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    save_image((images[:64] + 1.0) * 0.5, path, nrow=nrow)


def save_class_balanced_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    num_classes: int,
    samples_per_class: int = 8,
) -> None:
    """Save one row per class, preserving a clear class-by-class layout."""
    selected: List[torch.Tensor] = []
    labels = labels.cpu()
    for class_id in range(num_classes):
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if indices.numel() < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_id}: "
                f"need {samples_per_class}, found {indices.numel()}"
            )
        selected.append(images[indices[:samples_per_class]])
    grid = torch.cat(selected, dim=0)
    save_image((grid + 1.0) * 0.5, path, nrow=samples_per_class)


def plot_training_curves(
    ae_rows: Sequence[Mapping[str, float]],
    prior_rows: Mapping[str, Sequence[Mapping[str, float]]],
    path: Path,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if ae_rows:
        axes[0].plot([r["epoch"] for r in ae_rows], [r["loss"] for r in ae_rows], label="total")
        axes[0].plot([r["epoch"] for r in ae_rows], [r["clean_l1"] for r in ae_rows], label="clean L1")
        axes[0].plot([r["epoch"] for r in ae_rows], [r["noisy_l1"] for r in ae_rows], label="noisy L1")
        axes[0].set_title("Spatial autoencoder")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

    for method, rows in prior_rows.items():
        if not rows:
            continue
        losses = np.asarray([float(r["loss"]) for r in rows])
        axes[1].plot(
            [r["epoch"] for r in rows],
            losses / max(losses[0], 1e-12),
            label=method,
        )
    axes[1].set_title("Prior loss / first epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Normalized loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_norm_paths(geometry: Mapping[str, Mapping[str, object]], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(6.0, 4.0))
    for method, record in geometry.items():
        values = record.get("norm_path")
        if isinstance(values, list):
            plt.plot(values, label=method)
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Sampling step")
    plt.ylabel("Mean token norm")
    plt.title("Product-of-spheres norm during sampling")
    plt.grid(alpha=0.25)
    plt.legend()
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
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# =============================================================================
# Experiment orchestration
# =============================================================================


def run_experiment(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    device = get_device()

    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    checkpoints = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    samples_dir = ensure_dir(run_dir / "samples")
    figures_dir = ensure_dir(run_dir / "figures")

    save_json(asdict(config), run_dir / "config.json")

    print("=" * 80)
    print("Class-conditional SRUL-v2 on spatial product-of-spheres latents")
    print("Device:", device)
    print("Run directory:", run_dir)
    print(
        "Latent:",
        f"{config.latent_channels} x {config.latent_size} x {config.latent_size}",
    )
    print("Methods:", list(config.methods))
    print("Guidance scales:", list(config.guidance_scales))
    print("Soft Jacobi alpha:", config.jacobi_alpha)
    print("EMA decay:", config.ema_decay)
    print("Label-drop probability:", config.label_drop_prob)
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

    ae = SpatialSphericalAutoencoder(
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
    ).to(device)

    ae_logs: List[Dict[str, float]] = []
    local_ae_final = checkpoints / "autoencoder_final.pt"

    if config.ae_checkpoint is not None:
        source = Path(config.ae_checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"External AE checkpoint not found: {source}")
        ae.load_state_dict(load_state_dict_safely(source, device), strict=True)
        save_json(
            {"external_autoencoder_checkpoint": str(source)},
            run_dir / "autoencoder_source.json",
        )
        print("Loaded external spatial autoencoder:", source)
    elif config.skip_ae_training:
        if not local_ae_final.exists():
            raise FileNotFoundError(
                "--skip-ae-training was requested, but no local "
                f"checkpoint exists at {local_ae_final}. Supply "
                "--ae-checkpoint or train the autoencoder first."
            )
        ae.load_state_dict(load_state_dict_safely(local_ae_final, device))
        print("Loaded local spatial autoencoder:", local_ae_final)
    else:
        ae_logs = train_autoencoder(
            ae, train_loader, device, config, checkpoints, logs_dir
        )
        ae.load_state_dict(load_state_dict_safely(local_ae_final, device))

    ae.eval()

    reconstruction_metrics, real_recon, clean_recon, noisy_recon = (
        evaluate_reconstruction(ae, test_loader, device, config)
    )
    save_reconstruction_grid(
        real_recon,
        clean_recon,
        noisy_recon,
        figures_dir / "reconstructions.png",
    )

    if not config.skip_heavy_metrics:
        recon_n = min(config.recon_metric_samples, real_recon.size(0))
        real_recon_uint8 = to_uint8(real_recon[:recon_n])
        clean_recon_uint8 = to_uint8(clean_recon[:recon_n])
        real_recon_features = extract_resnet_features(
            real_recon_uint8, device, config.metric_batch_size
        )
        clean_recon_features = extract_resnet_features(
            clean_recon_uint8, device, config.metric_batch_size
        )
        reconstruction_metrics.update(
            feature_precision_recall(
                real_recon_features,
                clean_recon_features,
                device,
                config.pr_nearest_k,
                config.pr_chunk_size,
            )
        )
        reconstruction_metrics.update(
            {
                f"reconstruction_{key}": value
                for key, value in compute_fid_kid(
                    real_recon_uint8,
                    clean_recon_uint8,
                    device,
                    config.metric_batch_size,
                ).items()
            }
        )

    save_json(reconstruction_metrics, run_dir / "reconstruction_metrics.json")
    print("Reconstruction metrics:")
    print(json.dumps(reconstruction_metrics, indent=2))

    # Keep the uniform-product-of-spheres diagnostic for context.
    uniform_count = min(config.pr_samples, len(test_set))
    real_latents_uniform = collect_real_latents(
        ae, test_loader, device, uniform_count
    )
    uniform_latents_4d = sample_uniform_product_spheres(
        uniform_count,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        device,
    ).cpu()
    uniform_record: Dict[str, object] = {"method": "uniform_product_sphere"}
    uniform_record.update(
        feature_precision_recall(
            real_latents_uniform,
            uniform_latents_4d.flatten(1),
            device,
            config.pr_nearest_k,
            config.pr_chunk_size,
        )
    )
    uniform_images = decode_latents(
        ae, uniform_latents_4d, device, config.metric_batch_size
    )
    save_method_grid(
        uniform_images,
        samples_dir / "uniform_product_sphere.png",
    )

    if not config.skip_heavy_metrics:
        real_uniform = collect_real_images(test_loader, uniform_count)
        real_uniform_uint8 = to_uint8(real_uniform)
        uniform_uint8 = to_uint8(uniform_images)
        real_uniform_features = extract_resnet_features(
            real_uniform_uint8, device, config.metric_batch_size
        )
        uniform_features = extract_resnet_features(
            uniform_uint8, device, config.metric_batch_size
        )
        image_pr = feature_precision_recall(
            real_uniform_features,
            uniform_features,
            device,
            config.pr_nearest_k,
            config.pr_chunk_size,
        )
        uniform_record.update(
            {f"image_{key}": value for key, value in image_pr.items()}
        )
        uniform_record.update(
            compute_fid_kid(
                real_uniform_uint8,
                uniform_uint8,
                device,
                config.metric_batch_size,
            )
        )
    save_json(uniform_record, run_dir / "uniform_baseline.json")

    if config.ae_only:
        plot_training_curves(ae_logs, {}, figures_dir / "training_curves.png")
        print("AE-only run complete.")
        return

    prior_logs: Dict[str, List[Dict[str, float]]] = {}
    fields: Dict[str, SpatialVectorField] = {}

    for method in config.methods:
        print("\n" + "-" * 80)
        print("Training conditional prior:", method)
        field = SpatialVectorField(
            latent_channels=config.latent_channels,
            width=config.prior_width,
            depth=config.prior_depth,
            time_dim=config.time_dim,
            num_classes=config.num_classes,
        ).to(device)
        rows = train_prior(
            method,
            field,
            ae,
            train_loader,
            device,
            config,
            checkpoints,
            logs_dir,
        )
        field.load_state_dict(
            load_eval_state_dict(
                checkpoints / f"{method}_final.pt",
                device,
                prefer_ema=True,
            ),
            strict=True,
        )
        field.eval()
        fields[method] = field
        prior_logs[method] = rows
        print(f"[{method}] loaded EMA weights for evaluation.")

    metric_count = min(config.metric_samples, len(test_set))
    pr_count = min(config.pr_samples, metric_count)
    real_images = collect_real_images(test_loader, metric_count)
    real_uint8 = to_uint8(real_images)

    real_image_features: Optional[torch.Tensor] = None
    if not config.skip_heavy_metrics:
        print("Extracting real ResNet-18 features...")
        real_image_features = extract_resnet_features(
            real_uint8[:pr_count], device, config.metric_batch_size
        )

    real_latents_pr = collect_real_latents(ae, test_loader, device, pr_count)
    sample_labels = collect_real_labels(test_loader, metric_count)

    generation_rows: List[Dict[str, object]] = []
    geometry_records: Dict[str, Dict[str, object]] = {}

    for method, field in fields.items():
        for guidance_scale in config.guidance_scales:
            variant = f"{method}_cfg{guidance_scale:g}"
            safe_variant = variant.replace(".", "p")

            print("\n" + "-" * 80)
            print("Sampling and evaluating:", variant)

            # Reuse the same initial random latent sequence for every method
            # and guidance scale. This makes the ablation less noisy.
            sampling_seed = 100_000 + config.seed
            torch.manual_seed(sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_seed)

            latents, geometry = sample_flow_latents(
                method=method,
                field=field,
                config=config,
                device=device,
                labels=sample_labels,
                guidance_scale=guidance_scale,
                batch_size=config.metric_batch_size,
            )
            images = decode_latents(
                ae, latents, device, config.metric_batch_size
            )

            class_counts = torch.bincount(
                sample_labels.long().cpu(), minlength=config.num_classes
            )
            available_per_class = int(class_counts.min().item())
            grid_per_class = min(
                config.grid_samples_per_class,
                available_per_class,
            )
            if grid_per_class >= 1:
                save_class_balanced_grid(
                    images,
                    sample_labels,
                    samples_dir / f"{safe_variant}.png",
                    num_classes=config.num_classes,
                    samples_per_class=grid_per_class,
                )
            else:
                # A very small smoke-test subset may omit one or more classes.
                save_method_grid(images, samples_dir / f"{safe_variant}.png")

            row: Dict[str, object] = {
                key: value
                for key, value in geometry.items()
                if key not in {
                    "norm_path",
                    "radial_path",
                    "conditioning_delta_path",
                }
            }
            row.update(
                {
                    "method": variant,
                    "base_method": method,
                    "guidance_scale": float(guidance_scale),
                    "seed": config.seed,
                    "sampling_seed": sampling_seed,
                    "ema_decay": config.ema_decay,
                    "label_drop_prob": config.label_drop_prob,
                    "jacobi_alpha": (
                        config.jacobi_alpha if method == "cond_srul" else 0.0
                    ),
                }
            )

            latent_pr = feature_precision_recall(
                real_latents_pr,
                latents[:pr_count].flatten(1),
                device,
                config.pr_nearest_k,
                config.pr_chunk_size,
            )
            row.update({f"latent_{key}": value for key, value in latent_pr.items()})

            if not config.skip_heavy_metrics:
                fake_uint8 = to_uint8(images)
                row.update(
                    compute_fid_kid(
                        real_uint8,
                        fake_uint8,
                        device,
                        config.metric_batch_size,
                    )
                )
                assert real_image_features is not None
                fake_features = extract_resnet_features(
                    fake_uint8[:pr_count], device, config.metric_batch_size
                )
                row.update(
                    feature_precision_recall(
                        real_image_features,
                        fake_features,
                        device,
                        config.pr_nearest_k,
                        config.pr_chunk_size,
                    )
                )

            generation_rows.append(row)
            geometry_records[variant] = geometry
            print(json.dumps(row, indent=2))

    save_csv(generation_rows, run_dir / "generation_metrics.csv")
    save_json(geometry_records, run_dir / "geometry_paths.json")
    save_csv(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {
                    "norm_path",
                    "radial_path",
                    "conditioning_delta_path",
                }
            }
            for record in geometry_records.values()
        ],
        run_dir / "geometry_metrics.csv",
    )

    summary = {
        "config": asdict(config),
        "reconstruction_metrics": reconstruction_metrics,
        "uniform_baseline": uniform_record,
        "generation_metrics": generation_rows,
        "geometry": geometry_records,
    }
    save_json(summary, run_dir / "summary.json")

    plot_training_curves(
        ae_logs, prior_logs, figures_dir / "training_curves.png"
    )
    plot_norm_paths(
        geometry_records, figures_dir / "sampling_norm_paths.png"
    )
    if not config.skip_heavy_metrics:
        plot_metric_bars(
            generation_rows, figures_dir / "generation_metrics.png"
        )

    print("\nConditional experiment complete.")
    print("Results saved to:", run_dir)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Class-conditional SRUL-v2 on CIFAR-10, PathMNIST, or BloodMNIST with "
            "spatial product-of-spheres latents, EMA, soft Jacobi "
            "weighting, and CFG."
        )
    )
    parser.add_argument(
        "--dataset", choices=["pathmnist", "bloodmnist", "cifar10", "fake"], default="pathmnist"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./SRUL_MedMNIST_conditional")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=9)

    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--latent-channels", type=int, default=32)

    parser.add_argument("--ae-epochs", type=int, default=60)
    parser.add_argument("--ae-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--lambda-noisy", type=float, default=0.5)
    parser.add_argument("--lambda-edge", type=float, default=0.15)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lambda-lpips", type=float, default=0.10)
    parser.add_argument("--sigma-enc", type=float, default=0.15)
    parser.add_argument(
        "--ae-checkpoint",
        default=None,
        help=(
            "Optional existing spatial autoencoder checkpoint. This is the "
            "recommended way to reuse spatial_v1 without retraining."
        ),
    )
    parser.add_argument(
        "--skip-ae-training",
        action="store_true",
        help="Load --ae-checkpoint or a local autoencoder_final.pt.",
    )

    parser.add_argument("--prior-epochs", type=int, default=120)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--prior-width", type=int, default=256)
    parser.add_argument("--prior-depth", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument(
        "--time-sampling",
        choices=["uniform", "logit_normal"],
        default="logit_normal",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cond_rfm", "cond_srul"],
        default=["cond_rfm", "cond_srul"],
    )

    parser.add_argument("--label-drop-prob", type=float, default=0.10)
    parser.add_argument("--jacobi-alpha", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
    )
    parser.add_argument("--grid-samples-per-class", type=int, default=8)

    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--metric-samples", type=int, default=10000)
    parser.add_argument("--pr-samples", type=int, default=5000)
    parser.add_argument("--recon-metric-samples", type=int, default=5000)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)

    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ae-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument(
        "--no-dimension-scaled-noise",
        action="store_true",
        help="Do not divide tokenwise tangent Gaussian noise by sqrt(C-1).",
    )

    args = parser.parse_args()
    train_samples = args.train_samples if args.train_samples > 0 else None
    test_samples = args.test_samples if args.test_samples > 0 else None

    # Avoid accidental label-embedding mismatches when switching datasets.
    if args.dataset == "pathmnist":
        args.num_classes = 9
    elif args.dataset == "bloodmnist":
        args.num_classes = 8
    elif args.dataset == "cifar10":
        args.num_classes = 10

    if not 0.0 <= args.label_drop_prob < 1.0:
        parser.error("--label-drop-prob must be in [0,1).")
    if not 0.0 <= args.jacobi_alpha <= 1.0:
        parser.error("--jacobi-alpha must be in [0,1].")
    if not 0.0 < args.ema_decay < 1.0:
        parser.error("--ema-decay must be in (0,1).")
    if any(scale < 0 for scale in args.guidance_scales):
        parser.error("All --guidance-scales must be non-negative.")

    return ExperimentConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        lambda_clean=args.lambda_clean,
        lambda_noisy=args.lambda_noisy,
        lambda_edge=args.lambda_edge,
        lambda_latent=args.lambda_latent,
        lambda_lpips=args.lambda_lpips,
        sigma_enc=args.sigma_enc,
        dimension_scaled_noise=not args.no_dimension_scaled_noise,
        ae_checkpoint=args.ae_checkpoint,
        skip_ae_training=args.skip_ae_training,
        prior_epochs=args.prior_epochs,
        prior_lr=args.prior_lr,
        prior_width=args.prior_width,
        prior_depth=args.prior_depth,
        time_dim=args.time_dim,
        time_sampling=args.time_sampling,
        methods=tuple(args.methods),
        label_drop_prob=args.label_drop_prob,
        jacobi_alpha=args.jacobi_alpha,
        ema_decay=args.ema_decay,
        guidance_scales=tuple(args.guidance_scales),
        grid_samples_per_class=args.grid_samples_per_class,
        sample_steps=args.sample_steps,
        metric_samples=args.metric_samples,
        pr_samples=args.pr_samples,
        recon_metric_samples=args.recon_metric_samples,
        metric_batch_size=args.metric_batch_size,
        pr_chunk_size=args.pr_chunk_size,
        pr_nearest_k=args.pr_nearest_k,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        amp=args.amp,
        ae_only=args.ae_only,
        eval_only=args.eval_only,
        skip_heavy_metrics=args.skip_heavy_metrics,
        compute_lpips=args.compute_lpips,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
