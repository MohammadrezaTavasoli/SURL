"""Compare standard Flow Matching and projected Riemannian Flow Matching on the same trained VAE latent space."""

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
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image


# =============================================================================
# Utilities
# =============================================================================


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_torch_save(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def load_model_state(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def import_base_module():
    module_name = "srul_medmnist_conditional_experiment"
    candidates = [Path(__file__).resolve().parent, Path("/content"), Path.cwd()]
    for directory in candidates:
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import srul_medmnist_conditional_experiment. "
            "Place srul_medmnist_conditional_experiment.py beside this script."
        ) from exc


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    dataset: str = "cifar10"
    data_root: str = "/content/data"
    out_dir: str = (
        "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10_vae_prior_geometry"
    )
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

    methods: Tuple[str, ...] = ("vae_euclidean_fm", "vae_projected_rfm")
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

    radius_mode: str = "class_token_mean"
    stats_max_samples: int = 50000

    metric_samples: int = 10000
    pr_samples: int = 5000
    reconstruction_metric_samples: int = 5000
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5
    grid_samples_per_class: int = 8

    vae_checkpoint: str = ""
    vae_latent_stats_checkpoint: Optional[str] = None
    previous_metrics_csv: Optional[str] = None

    checkpoint_every: int = 5
    resume: bool = True
    amp: bool = True
    skip_heavy_metrics: bool = False
    eval_only: bool = False


# =============================================================================
# Dataset and VAE
# =============================================================================


class KLAutoencoder(nn.Module):
    """Spatial KL autoencoder used by the compact LDM baseline."""

    def __init__(self, base, latent_channels: int, base_channels: int):
        super().__init__()
        self.encoder = base.SpatialEncoder(latent_channels * 2, base_channels)
        self.decoder = base.SpatialDecoder(latent_channels, base_channels)
        self.latent_channels = int(latent_channels)
        self.latent_size = 4

    def posterior(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(images)
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(-12.0, 8.0)

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        return self.posterior(images)[0]

    def encode_sample(self, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.posterior(images)
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


def make_loaders(base, config: Config, device: torch.device):
    if config.dataset == "fake32":
        train_count = config.train_samples or 96
        test_count = config.test_samples or 48
        generator = torch.Generator().manual_seed(config.seed)
        train_images = torch.rand(
            train_count, 3, 32, 32, generator=generator
        ) * 2 - 1
        train_labels = torch.arange(train_count) % config.num_classes
        test_images = torch.rand(
            test_count, 3, 32, 32, generator=generator
        ) * 2 - 1
        test_labels = torch.arange(test_count) % config.num_classes
        train_set = TensorDataset(train_images, train_labels)
        test_set = TensorDataset(test_images, test_labels)
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


# =============================================================================
# Latent and radius statistics
# =============================================================================


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor

    def standardize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean.to(latent.device)) / self.std.to(latent.device)

    def unstandardize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std.to(latent.device) + self.mean.to(latent.device)

    def cpu_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_dict(cls, obj: Mapping[str, torch.Tensor]) -> "LatentStats":
        return cls(mean=obj["mean"].cpu(), std=obj["std"].cpu())


@dataclass
class RadiusStats:
    class_token_mean: torch.Tensor
    class_token_std: torch.Tensor
    global_token_mean: torch.Tensor
    global_token_std: torch.Tensor
    class_counts: torch.Tensor
    global_mean: float
    global_std: float
    global_cv: float
    mean_class_token_cv: float
    mean_posterior_std: float
    mean_logvar: float

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
            "mean_posterior_std": self.mean_posterior_std,
            "mean_logvar": self.mean_logvar,
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
            mean_posterior_std=float(obj.get("mean_posterior_std", float("nan"))),
            mean_logvar=float(obj.get("mean_logvar", float("nan"))),
        )

    def radius_map(self, labels: torch.Tensor, mode: str) -> torch.Tensor:
        labels_cpu = labels.long().cpu()
        if mode == "class_token_mean":
            return self.class_token_mean[labels_cpu]
        if mode == "global_token_mean":
            return self.global_token_mean[None].expand(labels_cpu.numel(), -1, -1)
        if mode == "unit":
            height, width = self.global_token_mean.shape
            return torch.ones(labels_cpu.numel(), height, width)
        raise ValueError(f"Unknown radius mode: {mode}")


@torch.no_grad()
def estimate_latent_stats(
    ae: KLAutoencoder,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> LatentStats:
    ae.eval()
    sum_channels = None
    sum_square_channels = None
    count = 0
    seen_images = 0

    for images, _ in loader:
        images = images.to(device)
        latent = ae.encode_sample(images).float()
        remaining = config.stats_max_samples - seen_images
        if remaining <= 0:
            break
        latent = latent[:remaining]
        seen_images += latent.size(0)

        local_sum = latent.sum(dim=(0, 2, 3), keepdim=True).double().cpu()
        local_square = latent.pow(2).sum(dim=(0, 2, 3), keepdim=True).double().cpu()
        elements = latent.size(0) * latent.size(2) * latent.size(3)
        sum_channels = local_sum if sum_channels is None else sum_channels + local_sum
        sum_square_channels = (
            local_square
            if sum_square_channels is None
            else sum_square_channels + local_square
        )
        count += elements
        if seen_images >= config.stats_max_samples:
            break

    if count == 0:
        raise RuntimeError("No VAE latent samples were collected.")
    mean = sum_channels / count
    variance = sum_square_channels / count - mean.pow(2)
    std = variance.clamp_min(1e-6).sqrt()
    return LatentStats(mean=mean.float(), std=std.float())


@torch.no_grad()
def estimate_radius_stats(
    ae: KLAutoencoder,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> RadiusStats:
    ae.eval()
    classes = config.num_classes
    height = config.latent_size
    width = config.latent_size

    class_sum = torch.zeros(classes, height, width, dtype=torch.float64)
    class_square = torch.zeros_like(class_sum)
    class_counts = torch.zeros(classes, dtype=torch.float64)
    global_sum = torch.zeros(height, width, dtype=torch.float64)
    global_square = torch.zeros_like(global_sum)
    global_count = 0
    scalar_sum = 0.0
    scalar_square = 0.0
    scalar_count = 0
    posterior_std_sum = 0.0
    posterior_std_count = 0
    logvar_sum = 0.0
    logvar_count = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().view(-1)
        mean, logvar = ae.posterior(images)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        radii = latent.float().norm(dim=1).cpu().double()

        remaining = config.stats_max_samples - seen
        if remaining <= 0:
            break
        radii = radii[:remaining]
        labels = labels[:remaining]
        logvar = logvar[:remaining]
        batch = radii.size(0)
        seen += batch

        global_sum += radii.sum(dim=0)
        global_square += radii.pow(2).sum(dim=0)
        global_count += batch
        scalar_sum += float(radii.sum().item())
        scalar_square += float(radii.pow(2).sum().item())
        scalar_count += int(radii.numel())

        posterior_std = torch.exp(0.5 * logvar.float())
        posterior_std_sum += float(posterior_std.sum().item())
        posterior_std_count += int(posterior_std.numel())
        logvar_sum += float(logvar.float().sum().item())
        logvar_count += int(logvar.numel())

        for class_id in range(classes):
            mask = labels == class_id
            if mask.any():
                values = radii[mask]
                class_sum[class_id] += values.sum(dim=0)
                class_square[class_id] += values.pow(2).sum(dim=0)
                class_counts[class_id] += int(mask.sum().item())

        if seen >= config.stats_max_samples:
            break

    if global_count == 0:
        raise RuntimeError("No VAE radii were collected.")

    global_mean_map = global_sum / global_count
    global_variance_map = (
        global_square / global_count - global_mean_map.pow(2)
    ).clamp_min(0)
    global_std_map = global_variance_map.sqrt()

    class_mean = torch.empty_like(class_sum)
    class_std = torch.empty_like(class_sum)
    for class_id in range(classes):
        if class_counts[class_id] > 0:
            count = class_counts[class_id]
            mean = class_sum[class_id] / count
            variance = (class_square[class_id] / count - mean.pow(2)).clamp_min(0)
            class_mean[class_id] = mean
            class_std[class_id] = variance.sqrt()
        else:
            class_mean[class_id] = global_mean_map
            class_std[class_id] = global_std_map

    scalar_mean = scalar_sum / max(scalar_count, 1)
    scalar_variance = max(
        scalar_square / max(scalar_count, 1) - scalar_mean**2, 0.0
    )
    scalar_std = math.sqrt(scalar_variance)
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
        mean_posterior_std=posterior_std_sum / max(posterior_std_count, 1),
        mean_logvar=logvar_sum / max(logvar_count, 1),
    )


# =============================================================================
# Reconstruction diagnostics
# =============================================================================


def basic_reconstruction_metrics(
    real: torch.Tensor, reconstruction: torch.Tensor, base
) -> Dict[str, float]:
    mse = float(F.mse_loss(reconstruction, real).item())
    psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))
    output = {"mse": mse, "psnr": psnr}
    if getattr(base, "structural_similarity_index_measure", None) is not None:
        output["ssim"] = float(
            base.structural_similarity_index_measure(
                reconstruction, real, data_range=2.0
            ).item()
        )
    return output


def add_distribution_metrics(
    row: Dict[str, float],
    real: torch.Tensor,
    reconstruction: torch.Tensor,
    base,
    config: Config,
    device: torch.device,
) -> None:
    if config.skip_heavy_metrics:
        return
    real_uint8 = base.to_uint8(real)
    reconstruction_uint8 = base.to_uint8(reconstruction)
    row.update(
        base.compute_fid_kid(
            real_uint8,
            reconstruction_uint8,
            device=device,
            batch_size=config.metric_batch_size,
        )
    )
    count = min(config.pr_samples, real_uint8.size(0), reconstruction_uint8.size(0))
    real_features = base.extract_resnet_features(
        real_uint8[:count], device=device, batch_size=config.metric_batch_size
    )
    reconstruction_features = base.extract_resnet_features(
        reconstruction_uint8[:count],
        device=device,
        batch_size=config.metric_batch_size,
    )
    row.update(
        base.feature_precision_recall(
            real_features,
            reconstruction_features,
            device=device,
            nearest_k=config.pr_nearest_k,
            chunk_size=config.pr_chunk_size,
        )
    )


@torch.no_grad()
def evaluate_vae_reconstruction_support(
    ae: KLAutoencoder,
    radius_stats: RadiusStats,
    loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    figure_path: Path,
) -> Dict[str, Dict[str, float]]:
    ae.eval()
    real_list: List[torch.Tensor] = []
    posterior_mean_list: List[torch.Tensor] = []
    posterior_sample_list: List[torch.Tensor] = []
    mean_radius_list: List[torch.Tensor] = []
    unit_radius_list: List[torch.Tensor] = []
    count = 0

    # Keep the diagnostic reproducible without changing the global training seed.
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(config.seed + 314159)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().view(-1)
        mean, logvar = ae.posterior(images)
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        direction = base.normalize_tokens(sample)
        radius_map = radius_stats.radius_map(labels, config.radius_mode).to(device)

        posterior_mean = ae.decode(mean)
        posterior_sample = ae.decode(sample)
        mean_radius = ae.decode(direction * radius_map[:, None])
        unit_radius = ae.decode(direction)

        remaining = config.reconstruction_metric_samples - count
        if remaining <= 0:
            break
        real_list.append(images[:remaining].cpu())
        posterior_mean_list.append(posterior_mean[:remaining].cpu())
        posterior_sample_list.append(posterior_sample[:remaining].cpu())
        mean_radius_list.append(mean_radius[:remaining].cpu())
        unit_radius_list.append(unit_radius[:remaining].cpu())
        count += min(images.size(0), remaining)
        if count >= config.reconstruction_metric_samples:
            break

    torch.random.set_rng_state(generator_state)

    real = torch.cat(real_list, dim=0)
    reconstructions = {
        "posterior_mean": torch.cat(posterior_mean_list, dim=0),
        "posterior_sample": torch.cat(posterior_sample_list, dim=0),
        "class_mean_radius_direction": torch.cat(mean_radius_list, dim=0),
        "unit_radius_direction": torch.cat(unit_radius_list, dim=0),
    }

    rows: Dict[str, Dict[str, float]] = {}
    for name, reconstruction in reconstructions.items():
        row = basic_reconstruction_metrics(real, reconstruction, base)
        add_distribution_metrics(row, real, reconstruction, base, config, device)
        row["num_samples"] = float(real.size(0))
        rows[name] = row

    example_count = min(10, real.size(0))
    grid = torch.cat(
        [
            real[:example_count],
            reconstructions["posterior_mean"][:example_count],
            reconstructions["posterior_sample"][:example_count],
            reconstructions["class_mean_radius_direction"][:example_count],
            reconstructions["unit_radius_direction"][:example_count],
        ],
        dim=0,
    )
    save_image((grid + 1.0) * 0.5, figure_path, nrow=example_count)
    return rows


# =============================================================================
# Prior training
# =============================================================================


def sample_time(batch: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch, device=device)
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, device=device))
    raise ValueError(mode)


def apply_label_dropout(
    labels: torch.Tensor, null_label: int, probability: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = labels.clone().long()
    mask = torch.rand(labels.size(0), device=labels.device) < probability
    output[mask] = null_label
    return output, mask


def vae_euclidean_fm_objective(
    field: nn.Module,
    ae: KLAutoencoder,
    latent_stats: LatentStats,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
):
    with torch.no_grad():
        target = latent_stats.standardize(ae.encode_sample(images))
    source = torch.randn_like(target)
    time_values = sample_time(images.size(0), images.device, config.time_sampling)
    view = time_values[:, None, None, None]
    path = (1.0 - view) * target + view * source
    target_velocity = source - target
    training_labels, drop_mask = apply_label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    prediction = field(path, time_values, training_labels)
    loss = F.mse_loss(prediction, target_velocity)
    return loss, {
        "loss": float(loss.detach().item()),
        "path_rms": float(path.detach().pow(2).mean().sqrt().item()),
        "label_drop_fraction": float(drop_mask.float().mean().item()),
    }


def vae_projected_rfm_objective(
    field: nn.Module,
    ae: KLAutoencoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    base,
    config: Config,
):
    with torch.no_grad():
        target = base.normalize_tokens(ae.encode_sample(images))

    source = base.sample_uniform_product_spheres(
        images.size(0),
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        images.device,
    )
    time_values = base.sample_time(
        images.size(0), images.device, config.time_sampling
    )
    path, omega = base.slerp_tokens(target, source, time_values)
    target_velocity = base.slerp_velocity_tokens(
        target, source, time_values, omega
    )
    target_velocity = base.tangent_projection_tokens(path, target_velocity)

    training_labels, drop_mask = apply_label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    raw_prediction = field(path, time_values, training_labels)
    prediction = base.tangent_projection_tokens(path, raw_prediction)
    loss = (prediction - target_velocity).pow(2).mean()

    norms = base.token_norms(path)
    radial_fraction = base.radial_velocity_fraction_tokens(path, raw_prediction)
    return loss, {
        "loss": float(loss.detach().item()),
        "mean_path_norm": float(norms.mean().detach().item()),
        "mean_abs_norm_error": float((norms - 1.0).abs().mean().detach().item()),
        "mean_omega": float(omega.mean().detach().item()),
        "raw_radial_fraction": float(radial_fraction.detach().item()),
        "label_drop_fraction": float(drop_mask.float().mean().detach().item()),
    }


def train_prior(
    method: str,
    field: nn.Module,
    ae: KLAutoencoder,
    latent_stats: LatentStats,
    loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    run_dir: Path,
):
    checkpoints = ensure_dir(run_dir / "checkpoints")
    logs = ensure_dir(run_dir / "logs")
    latest = checkpoints / f"{method}_latest.pt"
    final = checkpoints / f"{method}_final.pt"
    log_path = logs / f"{method}.csv"

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
        print(f"[{method}] resumed from epoch {start_epoch - 1}")
    elif config.resume and final.exists():
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj["model"] if "model" in obj else obj)
        if isinstance(obj, dict) and "ema" in obj:
            ema.load_state_dict(obj["ema"])
        start_epoch = config.prior_epochs + 1
        print(f"[{method}] final checkpoint found")

    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    if not config.eval_only:
        field.train()
        for epoch in range(start_epoch, config.prior_epochs + 1):
            started = time.time()
            sums: Dict[str, float] = {}
            sample_count = 0
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().view(-1)
                optimizer.zero_grad(set_to_none=True)
                with base.autocast_context(device, config.amp):
                    if method == "vae_euclidean_fm":
                        loss, stats = vae_euclidean_fm_objective(
                            field, ae, latent_stats, images, labels, config
                        )
                    elif method == "vae_projected_rfm":
                        loss, stats = vae_projected_rfm_objective(
                            field, ae, images, labels, base, config
                        )
                    else:
                        raise ValueError(method)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(field)

                batch = images.size(0)
                sample_count += batch
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + value * batch

            row = {
                key: value / max(sample_count, 1) for key, value in sums.items()
            }
            row.update({"epoch": float(epoch), "seconds": time.time() - started})
            rows.append(row)
            save_csv(rows, log_path)
            print(
                f"[{method}] {epoch:03d}/{config.prior_epochs} "
                f"loss={row['loss']:.6f}"
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
            raise FileNotFoundError(f"Missing prior checkpoint: {final}")
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj["model"] if "model" in obj else obj)
        if isinstance(obj, dict) and "ema" in obj:
            ema.load_state_dict(obj["ema"])

    ema.copy_to(field)
    field.eval()
    return rows


# =============================================================================
# Sampling and evaluation
# =============================================================================


@torch.no_grad()
def cfg_prediction(
    field: nn.Module,
    latent: torch.Tensor,
    time_values: torch.Tensor,
    labels: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    null_labels = torch.full_like(labels, field.null_label)
    unconditional = field(latent, time_values, null_labels)
    conditional = field(latent, time_values, labels)
    return unconditional + scale * (conditional - unconditional)


@torch.no_grad()
def sample_vae_euclidean_fm(
    field: nn.Module,
    labels: torch.Tensor,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    latent = torch.randn(
        labels.size(0),
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        device=device,
    )
    step_size = -1.0 / config.sample_steps
    for index in range(config.sample_steps, 0, -1):
        time_values = torch.full(
            (labels.size(0),), index / config.sample_steps, device=device
        )
        velocity = cfg_prediction(
            field, latent, time_values, labels, config.guidance_scale
        )
        latent = latent + step_size * velocity
    return latent


@torch.no_grad()
def sample_vae_projected_rfm(
    field: nn.Module,
    labels: torch.Tensor,
    base,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    directions = base.sample_uniform_product_spheres(
        labels.size(0),
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        device,
    )
    norm_path = [float(base.token_norms(directions).mean().item())]
    radial_values: List[float] = []
    step_size = -1.0 / config.sample_steps

    for index in range(config.sample_steps, 0, -1):
        time_values = torch.full(
            (labels.size(0),), index / config.sample_steps, device=device
        )
        raw_velocity = cfg_prediction(
            field, directions, time_values, labels, config.guidance_scale
        )
        radial_values.append(
            float(base.radial_velocity_fraction_tokens(directions, raw_velocity).item())
        )
        tangent_velocity = base.tangent_projection_tokens(
            directions, raw_velocity
        )
        directions = base.exp_map_tokens(
            directions, step_size * tangent_velocity
        )
        norm_path.append(float(base.token_norms(directions).mean().item()))

    final_norms = base.token_norms(directions)
    norm_array = np.asarray(norm_path)
    geometry = {
        "final_mean_norm": float(final_norms.mean().item()),
        "final_std_norm": float(final_norms.std().item()),
        "min_mean_path_norm": float(norm_array.min()),
        "mean_abs_path_norm_error": float(np.abs(norm_array - 1.0).mean()),
        "mean_raw_radial_fraction": float(np.mean(radial_values)),
    }
    return directions, geometry


@torch.no_grad()
def decode_in_batches(
    ae: KLAutoencoder,
    latent: torch.Tensor,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    images: List[torch.Tensor] = []
    for start in range(0, latent.size(0), config.metric_batch_size):
        batch = latent[start : start + config.metric_batch_size].to(device)
        images.append(ae.decode(batch).cpu())
    return torch.cat(images, dim=0)


def evaluate_generation_method(
    method: str,
    field: nn.Module,
    ae: KLAutoencoder,
    latent_stats: LatentStats,
    radius_stats: RadiusStats,
    test_loader: DataLoader,
    base,
    config: Config,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    real = base.collect_real_images(test_loader, config.metric_samples)
    labels = base.collect_real_labels(test_loader, config.metric_samples)
    count = min(real.size(0), labels.numel())
    real = real[:count]
    labels = labels[:count]

    geometry: Dict[str, float] = {}
    if method == "vae_euclidean_fm":
        standardized = sample_vae_euclidean_fm(
            field, labels.to(device), config, device
        ).cpu()
        latent = latent_stats.unstandardize(standardized)
        fake = decode_in_batches(ae, latent, config, device)
        role = "matched_vae_euclidean"
    elif method == "vae_projected_rfm":
        directions, geometry = sample_vae_projected_rfm(
            field, labels.to(device), base, config, device
        )
        radius_map = radius_stats.radius_map(labels, config.radius_mode)
        latent = directions.cpu() * radius_map[:, None]
        fake = decode_in_batches(ae, latent, config, device)
        role = "mismatched_vae_direction_only"
    else:
        raise ValueError(method)

    base.save_class_balanced_grid(
        fake,
        labels,
        ensure_dir(run_dir / "samples") / f"{method}.png",
        num_classes=config.num_classes,
        samples_per_class=min(
            config.grid_samples_per_class,
            max(1, count // max(config.num_classes, 1)),
        ),
    )

    row: Dict[str, object] = {
        "method": method,
        "comparison_role": role,
        "seed": config.seed,
        "guidance_scale": config.guidance_scale,
        "metric_samples": float(count),
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
        pr_count = min(config.pr_samples, real_uint8.size(0), fake_uint8.size(0))
        real_features = base.extract_resnet_features(
            real_uint8[:pr_count],
            device=device,
            batch_size=config.metric_batch_size,
        )
        fake_features = base.extract_resnet_features(
            fake_uint8[:pr_count],
            device=device,
            batch_size=config.metric_batch_size,
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


def build_comparison_table(
    new_rows: Sequence[Mapping[str, object]],
    previous_metrics_csv: Optional[str],
    output_path: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if previous_metrics_csv:
        previous = read_csv(Path(previous_metrics_csv))
        role_map = {
            "srul_rfm": "matched_spherical_srul",
            "ldm": "vae_ddpm_reference",
            "euclidean_fm": "deterministic_euclidean_reference",
        }
        for row in previous:
            method = row.get("method")
            if method in role_map:
                copied = dict(row)
                copied["comparison_role"] = role_map[method]
                rows.append(copied)
    rows.extend(dict(row) for row in new_rows)
    save_csv(rows, output_path)
    return rows


# =============================================================================
# Main
# =============================================================================


def run(config: Config) -> None:
    set_seed(config.seed)
    device = get_device()
    base = import_base_module()

    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    for folder in ["checkpoints", "logs", "samples", "figures"]:
        ensure_dir(run_dir / folder)
    save_json(asdict(config), run_dir / "config.json")

    print("=" * 80)
    print("VAE prior-geometry comparison")
    print("Device:", device)
    print("Run directory:", run_dir)
    print("Methods:", list(config.methods))
    print("=" * 80)

    _, _, train_loader, test_loader = make_loaders(base, config, device)

    ae = KLAutoencoder(
        base,
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
    ).to(device)
    checkpoint = Path(config.vae_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")
    ae.load_state_dict(load_model_state(checkpoint, device))
    ae.eval()
    print("Loaded VAE:", checkpoint)

    latent_stats_path = run_dir / "vae_latent_stats.pt"
    external_stats = (
        Path(config.vae_latent_stats_checkpoint)
        if config.vae_latent_stats_checkpoint
        else None
    )

    if config.resume and external_stats is not None and external_stats.exists():
        latent_stats = LatentStats.from_dict(
            torch.load(external_stats, map_location="cpu")
        )
        print("Loaded external VAE latent statistics:", external_stats)
    elif config.resume and latent_stats_path.exists():
        latent_stats = LatentStats.from_dict(
            torch.load(latent_stats_path, map_location="cpu")
        )
        print("Loaded saved VAE latent statistics.")
    else:
        latent_stats = estimate_latent_stats(ae, train_loader, config, device)
        atomic_torch_save(latent_stats.cpu_dict(), latent_stats_path)

    radius_stats_path = run_dir / "vae_radius_stats.pt"
    if config.resume and radius_stats_path.exists():
        radius_stats = RadiusStats.from_dict(
            torch.load(radius_stats_path, map_location="cpu")
        )
        print("Loaded saved VAE radius statistics.")
    else:
        radius_stats = estimate_radius_stats(ae, train_loader, config, device)
        atomic_torch_save(radius_stats.cpu_dict(), radius_stats_path)

    radius_summary = {
        "global_mean_radius": radius_stats.global_mean,
        "global_std_radius": radius_stats.global_std,
        "global_radius_cv": radius_stats.global_cv,
        "mean_class_token_cv": radius_stats.mean_class_token_cv,
        "mean_posterior_std": radius_stats.mean_posterior_std,
        "mean_logvar": radius_stats.mean_logvar,
        "class_counts": radius_stats.class_counts.tolist(),
    }
    save_json(radius_summary, run_dir / "vae_radius_diagnostics.json")
    print(
        "VAE radius CV=%.4f, mean posterior std=%.4f"
        % (radius_stats.global_cv, radius_stats.mean_posterior_std)
    )

    reconstruction_rows = evaluate_vae_reconstruction_support(
        ae,
        radius_stats,
        test_loader,
        base,
        config,
        device,
        run_dir / "figures" / "vae_support_reconstruction_diagnostic.png",
    )
    save_json(
        reconstruction_rows,
        run_dir / "vae_support_reconstruction_metrics.json",
    )

    fields: Dict[str, nn.Module] = {}
    histories: Dict[str, List[Dict[str, float]]] = {}
    generation_rows: List[Dict[str, object]] = []

    for method in config.methods:
        field = base.SpatialVectorField(
            latent_channels=config.latent_channels,
            width=config.prior_width,
            depth=config.prior_depth,
            time_dim=config.time_dim,
            num_classes=config.num_classes,
        ).to(device)
        histories[method] = train_prior(
            method,
            field,
            ae,
            latent_stats,
            train_loader,
            base,
            config,
            device,
            run_dir,
        )
        fields[method] = field
        generation_rows.append(
            evaluate_generation_method(
                method,
                field,
                ae,
                latent_stats,
                radius_stats,
                test_loader,
                base,
                config,
                device,
                run_dir,
            )
        )

    save_csv(generation_rows, run_dir / "generation_metrics.csv")
    comparison_rows = build_comparison_table(
        generation_rows,
        config.previous_metrics_csv,
        run_dir / "vae_prior_geometry_comparison.csv",
    )

    summary = {
        "radius_diagnostics": radius_summary,
        "reconstruction_diagnostics": reconstruction_rows,
        "generation_metrics": generation_rows,
        "comparison_rows": comparison_rows,
        "final_training_rows": {
            method: rows[-1] if rows else None
            for method, rows in histories.items()
        },
    }
    save_json(summary, run_dir / "summary.json")

    print("\nCompleted.")
    print(json.dumps(generation_rows, indent=2))
    print("Comparison table:", run_dir / "vae_prior_geometry_comparison.csv")


# =============================================================================
# CLI
# =============================================================================


def optional_count(value: str) -> Optional[int]:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard FM and direction-only spherical RFM on one "
            "fixed KL/VAE autoencoder."
        )
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake32"])
    parser.add_argument("--data-root", default="/content/data")
    parser.add_argument(
        "--out-dir",
        default=(
            "/content/drive/MyDrive/SRUL_Final_Comparisons/"
            "CIFAR10_vae_prior_geometry"
        ),
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

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vae_euclidean_fm", "vae_projected_rfm"],
        default=["vae_euclidean_fm", "vae_projected_rfm"],
    )
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

    parser.add_argument(
        "--radius-mode",
        choices=["class_token_mean", "global_token_mean", "unit"],
        default="class_token_mean",
    )
    parser.add_argument("--stats-max-samples", type=int, default=50000)

    parser.add_argument("--metric-samples", type=int, default=10000)
    parser.add_argument("--pr-samples", type=int, default=5000)
    parser.add_argument("--reconstruction-metric-samples", type=int, default=5000)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)
    parser.add_argument("--grid-samples-per-class", type=int, default=8)

    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--vae-latent-stats-checkpoint", default=None)
    parser.add_argument("--previous-metrics-csv", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    arguments = parser.parse_args()
    values = vars(arguments)
    values["methods"] = tuple(values["methods"])
    return Config(**values)


if __name__ == "__main__":
    run(parse_args())
