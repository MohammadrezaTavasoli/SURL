#!/usr/bin/env python3
"""Train the spatial spherical autoencoder with small and large tangent perturbations to encourage broad token coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    import srul_cifar_conditional_spatial_experiment as base
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Place srul_cifar_conditional_spatial_experiment.py in the same "
        "directory as this script."
    ) from exc


@dataclass
class CoverageConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    out_dir: str = "./SRUL_CIFAR10_spherical_coverage"
    seed: int = 0

    train_samples: Optional[int] = None
    test_samples: Optional[int] = 10000
    batch_size: int = 128
    metric_batch_size: int = 128
    num_workers: int = 2
    num_classes: int = 10

    base_channels: int = 96
    latent_channels: int = 32
    latent_size: int = 4

    init_ae_checkpoint: Optional[str] = None
    ae_epochs: int = 30
    ae_lr: float = 5e-5

    # Product-manifold angle jitter for broad spherical coverage.
    alpha_max_deg: float = 80.0
    alpha_mix_low_deg: float = 80.0
    alpha_mix_high_deg: float = 85.0
    alpha_mix_prob: float = 0.10
    small_scale_max: float = 0.50

    # Loss weights for reconstruction, consistency, and latent alignment.
    lambda_pix_recon_l1: float = 1.0
    lambda_pix_recon_lpips: float = 1.0
    lambda_pix_cons_l1: float = 0.5
    lambda_pix_cons_lpips: float = 0.5
    lambda_lat_cons: float = 0.10
    lambda_edge: float = 0.10
    lambda_clean_anchor: float = 0.25

    # Final SRUL prior still uses the selected small information-control noise.
    sigma_enc: float = 0.05
    prior_epochs: int = 80
    prior_lr: float = 2e-4
    prior_width: int = 256
    prior_depth: int = 6
    time_dim: int = 128
    time_sampling: str = "logit_normal"
    label_drop_prob: float = 0.10
    ema_decay: float = 0.999
    guidance_scales: Tuple[float, ...] = (1.0, 2.0)
    sample_steps: int = 100

    metric_samples: int = 10000
    pr_samples: int = 5000
    recon_metric_samples: int = 5000
    uniformity_samples: int = 10000
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5
    grid_samples_per_class: int = 8

    checkpoint_every: int = 5
    compute_lpips: bool = True
    skip_heavy_metrics: bool = False
    ae_only: bool = False
    resume: bool = False
    amp: bool = False
    baseline_generation_csv: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def product_tangent_unit(z: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Project xi tokenwise, then normalize the complete tangent grid per image.

    The product-of-spheres metric is the sum of tokenwise squared norms, so the
    flattened CxHxW norm is the natural product-manifold tangent norm.
    """
    tangent = base.tangent_projection_tokens(z, xi)
    norm = tangent.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return tangent / norm.clamp_min(1e-8)


def sample_coverage_pair(
    z: torch.Tensor,
    config: CoverageConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return large/small perturbations sharing one tangent direction.

    alpha is a total product-manifold angle.  This avoids rotating every token
    independently by 80 degrees, which would make the total product distance
    grow by sqrt(number_of_tokens).
    """
    batch = z.size(0)
    u = product_tangent_unit(z, torch.randn_like(z))

    alpha_max = math.radians(config.alpha_max_deg)
    alpha_mix_low = math.radians(config.alpha_mix_low_deg)
    alpha_mix_high = math.radians(config.alpha_mix_high_deg)

    alpha = torch.rand(batch, device=z.device, dtype=z.dtype) * alpha_max
    if config.alpha_mix_prob > 0:
        mix_mask = torch.rand(batch, device=z.device) < config.alpha_mix_prob
        mixed = alpha_mix_low + torch.rand(
            batch, device=z.device, dtype=z.dtype
        ) * max(alpha_mix_high - alpha_mix_low, 0.0)
        alpha = torch.where(mix_mask, mixed, alpha)
    else:
        mix_mask = torch.zeros(batch, dtype=torch.bool, device=z.device)

    small_scale = torch.rand(batch, device=z.device, dtype=z.dtype)
    small_scale = small_scale * config.small_scale_max
    alpha_small = alpha * small_scale

    shape = (-1, 1, 1, 1)
    z_large = base.exp_map_tokens(z, alpha.view(shape) * u)
    z_small = base.exp_map_tokens(z, alpha_small.view(shape) * u)

    return z_large, z_small, {
        "alpha_large": alpha,
        "alpha_small": alpha_small,
        "mix_mask": mix_mask.float(),
    }


def smooth_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(a, b)


def sphere_style_objective(
    ae: base.SpatialSphericalAutoencoder,
    images: torch.Tensor,
    config: CoverageConfig,
    perceptual: Optional[nn.Module],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z = ae.encode(images)
    z_large, z_small, jitter = sample_coverage_pair(z, config)

    clean = ae.decode(z)
    recon_small = ae.decode(z_small)
    recon_large = ae.decode(z_large)

    pix_recon_l1 = smooth_l1(recon_small, images)
    clean_anchor = smooth_l1(clean, images)
    edges = base.edge_loss(recon_small, images)

    # Large-noise decode should agree with the small-noise target.  The target
    # branch is detached so the small-noise reconstruction acts as the target.
    small_target = recon_small.detach()
    pix_cons_l1 = smooth_l1(recon_large, small_target)

    reencoded_large = ae.encode(recon_large)
    lat_cons = 1.0 - base.mean_token_cosine(z.detach(), reencoded_large)

    pix_recon_lpips = torch.zeros((), device=images.device)
    pix_cons_lpips = torch.zeros((), device=images.device)
    if perceptual is not None:
        pix_recon_lpips = perceptual(recon_small, images)
        pix_cons_lpips = perceptual(recon_large, small_target)

    total = (
        config.lambda_pix_recon_l1 * pix_recon_l1
        + config.lambda_pix_recon_lpips * pix_recon_lpips
        + config.lambda_pix_cons_l1 * pix_cons_l1
        + config.lambda_pix_cons_lpips * pix_cons_lpips
        + config.lambda_lat_cons * lat_cons
        + config.lambda_edge * edges
        + config.lambda_clean_anchor * clean_anchor
    )

    return total, {
        "loss": float(total.detach().item()),
        "pix_recon_l1": float(pix_recon_l1.detach().item()),
        "pix_recon_lpips": float(pix_recon_lpips.detach().item()),
        "pix_cons_l1": float(pix_cons_l1.detach().item()),
        "pix_cons_lpips": float(pix_cons_lpips.detach().item()),
        "lat_cons": float(lat_cons.detach().item()),
        "edge": float(edges.detach().item()),
        "clean_anchor": float(clean_anchor.detach().item()),
        "alpha_large_deg": float(
            torch.rad2deg(jitter["alpha_large"]).mean().detach().item()
        ),
        "alpha_small_deg": float(
            torch.rad2deg(jitter["alpha_small"]).mean().detach().item()
        ),
        "mix_fraction": float(jitter["mix_mask"].mean().detach().item()),
    }


def train_coverage_autoencoder(
    ae: base.SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: CoverageConfig,
    checkpoints: Path,
    logs_dir: Path,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(ae.parameters(), lr=config.ae_lr, weight_decay=1e-4)
    scaler = base.make_grad_scaler(device, config.amp)
    latest = checkpoints / "coverage_autoencoder_latest.pt"
    final = checkpoints / "coverage_autoencoder_final.pt"
    log_path = logs_dir / "coverage_autoencoder.csv"

    start_epoch = 1
    rows: List[Dict[str, float]] = []
    if config.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        ae.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        rows = list(checkpoint.get("logs", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[UNIFORM-AE] resumed from epoch {start_epoch - 1}")

    perceptual: Optional[nn.Module] = None
    if config.compute_lpips and (
        config.lambda_pix_recon_lpips > 0 or config.lambda_pix_cons_lpips > 0
    ):
        perceptual = base.LPIPSLoss(device)

    keys = [
        "loss",
        "pix_recon_l1",
        "pix_recon_lpips",
        "pix_cons_l1",
        "pix_cons_lpips",
        "lat_cons",
        "edge",
        "clean_anchor",
        "alpha_large_deg",
        "alpha_small_deg",
        "mix_fraction",
    ]

    ae.train()
    for epoch in range(start_epoch, config.ae_epochs + 1):
        start = time.time()
        sums = {key: 0.0 for key in keys}
        count = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with base.autocast_context(device, config.amp):
                loss, stats = sphere_style_objective(ae, images, config, perceptual)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = images.size(0)
            count += batch
            for key in keys:
                sums[key] += stats[key] * batch

        row = {key: value / max(count, 1) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["seconds"] = float(time.time() - start)
        rows.append(row)
        base.save_csv(rows, log_path)
        print(
            f"[UNIFORM-AE] {epoch:03d}/{config.ae_epochs} "
            f"loss={row['loss']:.4f} recon={row['pix_recon_l1']:.4f} "
            f"pixcon={row['pix_cons_l1']:.4f} lat={row['lat_cons']:.4f} "
            f"alpha={row['alpha_large_deg']:.1f}deg"
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.ae_epochs:
            base.atomic_torch_save(
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

    base.atomic_torch_save(ae.state_dict(), final)
    return rows


@torch.no_grad()
def collect_uniformity_diagnostics(
    ae: base.SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_images: int,
    num_classes: int,
) -> Dict[str, float]:
    """Moment diagnostics for token marginals and class-conditional marginals."""
    ae.eval()
    token_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    count = 0
    for images, labels in loader:
        images = images.to(device)
        z = ae.encode(images)  # [B,C,H,W]
        remaining = max_images - count
        z = z[:remaining]
        labels = labels[:remaining]
        tokens = z.permute(0, 2, 3, 1).reshape(-1, z.size(1)).cpu()
        expanded_labels = labels[:, None, None].expand(
            labels.size(0), z.size(2), z.size(3)
        ).reshape(-1)
        token_chunks.append(tokens)
        label_chunks.append(expanded_labels.cpu())
        count += z.size(0)
        if count >= max_images:
            break

    tokens = torch.cat(token_chunks, dim=0).float()
    labels = torch.cat(label_chunks, dim=0).long()
    c = tokens.size(1)
    eye = torch.eye(c) / float(c)

    mean = tokens.mean(dim=0)
    second = tokens.T @ tokens / float(tokens.size(0))
    covariance_deviation = torch.linalg.matrix_norm(second - eye, ord="fro")
    off_diag = second - torch.diag(torch.diag(second))

    # Pairwise cosine diagnostic on a bounded subset.
    pair_n = min(4096, tokens.size(0))
    perm = torch.randperm(tokens.size(0))[:pair_n]
    paired = tokens[perm]
    paired2 = paired[torch.randperm(pair_n)]
    pair_cos = (paired * paired2).sum(dim=1)

    per_class_mean_norm: List[float] = []
    per_class_cov_dev: List[float] = []
    for class_id in range(num_classes):
        cls = tokens[labels == class_id]
        if cls.size(0) < c:
            continue
        cls_mean = cls.mean(dim=0)
        cls_second = cls.T @ cls / float(cls.size(0))
        per_class_mean_norm.append(float(cls_mean.norm().item()))
        per_class_cov_dev.append(
            float(torch.linalg.matrix_norm(cls_second - eye, ord="fro").item())
        )

    return {
        "num_images": float(count),
        "num_tokens": float(tokens.size(0)),
        "token_mean_norm": float(mean.norm().item()),
        "second_moment_frobenius_deviation": float(covariance_deviation.item()),
        "mean_abs_off_diagonal_second_moment": float(off_diag.abs().mean().item()),
        "pairwise_cosine_mean": float(pair_cos.mean().item()),
        "pairwise_cosine_std": float(pair_cos.std(unbiased=False).item()),
        "class_conditional_mean_norm_avg": float(np.mean(per_class_mean_norm)),
        "class_conditional_mean_norm_max": float(np.max(per_class_mean_norm)),
        "class_conditional_second_moment_deviation_avg": float(
            np.mean(per_class_cov_dev)
        ),
    }


def make_base_config(config: CoverageConfig) -> base.ExperimentConfig:
    return base.ExperimentConfig(
        dataset=config.dataset,
        data_root=config.data_root,
        out_dir=config.out_dir,
        seed=config.seed,
        train_samples=config.train_samples,
        test_samples=config.test_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
        ae_epochs=config.ae_epochs,
        ae_lr=config.ae_lr,
        lambda_clean=1.0,
        lambda_noisy=0.5,
        lambda_edge=0.15,
        lambda_latent=0.05,
        lambda_lpips=0.10,
        sigma_enc=config.sigma_enc,
        dimension_scaled_noise=True,
        prior_epochs=config.prior_epochs,
        prior_lr=config.prior_lr,
        prior_width=config.prior_width,
        prior_depth=config.prior_depth,
        time_dim=config.time_dim,
        time_sampling=config.time_sampling,
        methods=("cond_rfm",),
        label_drop_prob=config.label_drop_prob,
        jacobi_alpha=0.0,
        ema_decay=config.ema_decay,
        guidance_scales=config.guidance_scales,
        grid_samples_per_class=config.grid_samples_per_class,
        sample_steps=config.sample_steps,
        metric_samples=config.metric_samples,
        pr_samples=config.pr_samples,
        recon_metric_samples=config.recon_metric_samples,
        metric_batch_size=config.metric_batch_size,
        pr_chunk_size=config.pr_chunk_size,
        pr_nearest_k=config.pr_nearest_k,
        checkpoint_every=config.checkpoint_every,
        resume=config.resume,
        amp=config.amp,
        ae_only=config.ae_only,
        eval_only=False,
        skip_heavy_metrics=config.skip_heavy_metrics,
        compute_lpips=config.compute_lpips,
    )


@torch.no_grad()
def evaluate_reconstruction_full(
    ae: base.SpatialSphericalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    base_config: base.ExperimentConfig,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    metrics, real, clean, noisy = base.evaluate_reconstruction(
        ae, loader, device, base_config
    )
    if not base_config.skip_heavy_metrics:
        n = min(base_config.recon_metric_samples, real.size(0))
        real_uint8 = base.to_uint8(real[:n])
        clean_uint8 = base.to_uint8(clean[:n])
        real_features = base.extract_resnet_features(
            real_uint8, device, base_config.metric_batch_size
        )
        clean_features = base.extract_resnet_features(
            clean_uint8, device, base_config.metric_batch_size
        )
        metrics.update(
            base.feature_precision_recall(
                real_features,
                clean_features,
                device,
                base_config.pr_nearest_k,
                base_config.pr_chunk_size,
            )
        )
        metrics.update(
            {
                f"reconstruction_{key}": value
                for key, value in base.compute_fid_kid(
                    real_uint8, clean_uint8, device, base_config.metric_batch_size
                ).items()
            }
        )
    return metrics, real, clean, noisy


@torch.no_grad()
def evaluate_direct_uniform_decode(
    ae: base.SpatialSphericalAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    config: CoverageConfig,
) -> Dict[str, float]:
    count = min(config.metric_samples, len(test_loader.dataset))
    latents = base.sample_uniform_product_spheres(
        count,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        device,
    ).cpu()
    images = base.decode_latents(ae, latents, device, config.metric_batch_size)
    record: Dict[str, float] = {"num_samples": float(count)}
    if not config.skip_heavy_metrics:
        real = base.collect_real_images(test_loader, count)
        real_uint8 = base.to_uint8(real)
        fake_uint8 = base.to_uint8(images)
        record.update(
            base.compute_fid_kid(real_uint8, fake_uint8, device, config.metric_batch_size)
        )
        pr_n = min(config.pr_samples, count)
        real_features = base.extract_resnet_features(
            real_uint8[:pr_n], device, config.metric_batch_size
        )
        fake_features = base.extract_resnet_features(
            fake_uint8[:pr_n], device, config.metric_batch_size
        )
        record.update(
            base.feature_precision_recall(
                real_features,
                fake_features,
                device,
                config.pr_nearest_k,
                config.pr_chunk_size,
            )
        )
    return record



def plot_uniform_training_curves(
    ae_rows: Sequence[Mapping[str, float]],
    prior_rows: Sequence[Mapping[str, float]],
    path: Path,
) -> None:
    plt = getattr(base, "plt", None)
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if ae_rows:
        axes[0].plot([r["epoch"] for r in ae_rows], [r["loss"] for r in ae_rows], label="total")
        axes[0].plot([r["epoch"] for r in ae_rows], [r["pix_recon_l1"] for r in ae_rows], label="reconstruction")
        axes[0].plot([r["epoch"] for r in ae_rows], [r["pix_cons_l1"] for r in ae_rows], label="pixel consistency")
        axes[0].plot([r["epoch"] for r in ae_rows], [r["lat_cons"] for r in ae_rows], label="latent consistency")
        axes[0].set_title("Spherical coverage autoencoder")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
    if prior_rows:
        losses = np.asarray([float(r["loss"]) for r in prior_rows])
        axes[1].plot([r["epoch"] for r in prior_rows], losses / max(losses[0], 1e-12), label="conditional RFM")
        axes[1].set_title("Prior loss / first epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Normalized loss")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def read_baseline_generation(path: Optional[str]) -> List[Dict[str, object]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline generation CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config: CoverageConfig) -> None:
    set_seed(config.seed)
    device = base.get_device()
    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    checkpoints = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    samples_dir = ensure_dir(run_dir / "samples")
    figures_dir = ensure_dir(run_dir / "figures")
    base.save_json(asdict(config), run_dir / "config.json")

    print("=" * 88)
    print("SRUL spherical coverage training")
    print("Device:", device)
    print("Run directory:", run_dir)
    print(
        "Large product-manifold angle jitter:",
        f"[0, {config.alpha_max_deg:g}] deg + "
        f"{config.alpha_mix_prob:g} mix in "
        f"[{config.alpha_mix_low_deg:g}, {config.alpha_mix_high_deg:g}] deg",
    )
    print("=" * 88)

    train_set, test_set = base.make_datasets(
        config.dataset,
        config.data_root,
        config.train_samples,
        config.test_samples,
        config.seed,
    )
    train_loader, test_loader = base.make_loaders(
        train_set,
        test_set,
        config.batch_size,
        config.num_workers,
        device,
    )
    base_config = make_base_config(config)

    ae = base.SpatialSphericalAutoencoder(
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
    ).to(device)

    baseline_results: Dict[str, object] = {}
    if config.init_ae_checkpoint:
        source = Path(config.init_ae_checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"Initial autoencoder checkpoint not found: {source}")
        ae.load_state_dict(base.load_state_dict_safely(source, device), strict=True)
        print("Loaded initial SRUL autoencoder:", source)
        baseline_uniformity = collect_uniformity_diagnostics(
            ae,
            train_loader,
            device,
            config.uniformity_samples,
            config.num_classes,
        )
        baseline_recon, _, _, _ = evaluate_reconstruction_full(
            ae, test_loader, device, base_config
        )
        baseline_direct = evaluate_direct_uniform_decode(
            ae, test_loader, device, config
        )
        baseline_results = {
            "checkpoint": str(source),
            "uniformity": baseline_uniformity,
            "reconstruction": baseline_recon,
            "direct_uniform_decode": baseline_direct,
        }
        base.save_json(baseline_results, run_dir / "baseline_autoencoder_metrics.json")

    ae_logs = train_coverage_autoencoder(
        ae, train_loader, device, config, checkpoints, logs_dir
    )
    ae.load_state_dict(
        base.load_state_dict_safely(
            checkpoints / "coverage_autoencoder_final.pt", device
        )
    )
    ae.eval()

    uniformity = collect_uniformity_diagnostics(
        ae,
        train_loader,
        device,
        config.uniformity_samples,
        config.num_classes,
    )
    reconstruction, real_recon, clean_recon, noisy_recon = evaluate_reconstruction_full(
        ae, test_loader, device, base_config
    )
    direct_uniform = evaluate_direct_uniform_decode(ae, test_loader, device, config)
    base.save_json(uniformity, run_dir / "uniformity_metrics.json")
    base.save_json(reconstruction, run_dir / "reconstruction_metrics.json")
    base.save_json(direct_uniform, run_dir / "direct_uniform_decode_metrics.json")
    base.save_reconstruction_grid(
        real_recon,
        clean_recon,
        noisy_recon,
        figures_dir / "reconstructions.png",
    )

    comparison_rows: List[Dict[str, object]] = []
    if baseline_results:
        b_uni = baseline_results["uniformity"]
        b_rec = baseline_results["reconstruction"]
        b_direct = baseline_results["direct_uniform_decode"]
        assert isinstance(b_uni, Mapping)
        assert isinstance(b_rec, Mapping)
        assert isinstance(b_direct, Mapping)
        comparison_rows.append(
            {
                "model": "base_srul_autoencoder",
                "token_mean_norm": b_uni.get("token_mean_norm"),
                "second_moment_deviation": b_uni.get(
                    "second_moment_frobenius_deviation"
                ),
                "class_mean_norm_avg": b_uni.get(
                    "class_conditional_mean_norm_avg"
                ),
                "reconstruction_fid": b_rec.get("reconstruction_fid"),
                "direct_uniform_fid": b_direct.get("fid"),
            }
        )
    comparison_rows.append(
        {
            "model": "coverage_trained_autoencoder",
            "token_mean_norm": uniformity.get("token_mean_norm"),
            "second_moment_deviation": uniformity.get(
                "second_moment_frobenius_deviation"
            ),
            "class_mean_norm_avg": uniformity.get(
                "class_conditional_mean_norm_avg"
            ),
            "reconstruction_fid": reconstruction.get("reconstruction_fid"),
            "direct_uniform_fid": direct_uniform.get("fid"),
        }
    )

    generation_rows: List[Dict[str, object]] = []
    if not config.ae_only:
        field = base.SpatialVectorField(
            latent_channels=config.latent_channels,
            width=config.prior_width,
            depth=config.prior_depth,
            time_dim=config.time_dim,
            num_classes=config.num_classes,
        ).to(device)
        prior_logs = base.train_prior(
            "cond_rfm",
            field,
            ae,
            train_loader,
            device,
            base_config,
            checkpoints,
            logs_dir,
        )
        field.load_state_dict(
            base.load_eval_state_dict(
                checkpoints / "cond_rfm_final.pt", device, prefer_ema=True
            ),
            strict=True,
        )
        field.eval()

        metric_count = min(config.metric_samples, len(test_set))
        pr_count = min(config.pr_samples, metric_count)
        real_images = base.collect_real_images(test_loader, metric_count)
        real_uint8 = base.to_uint8(real_images)
        real_features = None
        if not config.skip_heavy_metrics:
            real_features = base.extract_resnet_features(
                real_uint8[:pr_count], device, config.metric_batch_size
            )
        real_latents = base.collect_real_latents(ae, test_loader, device, pr_count)
        labels = base.balanced_class_labels(metric_count, config.num_classes)

        for guidance_scale in config.guidance_scales:
            torch.manual_seed(100_000 + config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(100_000 + config.seed)
            latents, geometry = base.sample_flow_latents(
                method="cond_rfm",
                field=field,
                config=base_config,
                device=device,
                labels=labels,
                guidance_scale=guidance_scale,
                batch_size=config.metric_batch_size,
            )
            images = base.decode_latents(ae, latents, device, config.metric_batch_size)
            safe = f"coverage_rfm_cfg{guidance_scale:g}".replace(".", "p")
            base.save_class_balanced_grid(
                images,
                labels,
                samples_dir / f"{safe}.png",
                num_classes=config.num_classes,
                samples_per_class=min(
                    config.grid_samples_per_class,
                    metric_count // config.num_classes,
                ),
            )
            row: Dict[str, object] = {
                "method": safe,
                "guidance_scale": float(guidance_scale),
                "seed": config.seed,
            }
            row.update(
                {
                    key: value
                    for key, value in geometry.items()
                    if not isinstance(value, list)
                }
            )
            row.update(
                {
                    f"latent_{key}": value
                    for key, value in base.feature_precision_recall(
                        real_latents,
                        latents[:pr_count].flatten(1),
                        device,
                        config.pr_nearest_k,
                        config.pr_chunk_size,
                    ).items()
                }
            )
            if not config.skip_heavy_metrics:
                fake_uint8 = base.to_uint8(images)
                row.update(
                    base.compute_fid_kid(
                        real_uint8, fake_uint8, device, config.metric_batch_size
                    )
                )
                assert real_features is not None
                fake_features = base.extract_resnet_features(
                    fake_uint8[:pr_count], device, config.metric_batch_size
                )
                row.update(
                    base.feature_precision_recall(
                        real_features,
                        fake_features,
                        device,
                        config.pr_nearest_k,
                        config.pr_chunk_size,
                    )
                )
            generation_rows.append(row)
            print(json.dumps(row, indent=2))

        base.save_csv(generation_rows, run_dir / "generation_metrics.csv")
        plot_uniform_training_curves(
            ae_logs,
            prior_logs,
            figures_dir / "training_curves.png",
        )

    baseline_generation = read_baseline_generation(config.baseline_generation_csv)
    if baseline_generation:
        for row in baseline_generation:
            if "cond_rfm" in str(row.get("method", "")):
                comparison_rows.append(
                    {
                        "model": f"baseline_{row.get('method')}",
                        "generation_fid": row.get("fid"),
                        "generation_precision": row.get("feature_precision"),
                        "generation_recall": row.get("feature_recall"),
                    }
                )
    for row in generation_rows:
        comparison_rows.append(
            {
                "model": row.get("method"),
                "generation_fid": row.get("fid"),
                "generation_precision": row.get("feature_precision"),
                "generation_recall": row.get("feature_recall"),
            }
        )

    base.save_csv(comparison_rows, run_dir / "coverage_comparison.csv")
    summary = {
        "config": asdict(config),
        "baseline": baseline_results,
        "coverage_trained": {
            "uniformity": uniformity,
            "reconstruction": reconstruction,
            "direct_uniform_decode": direct_uniform,
            "generation": generation_rows,
        },
        "interpretation": (
            "Lower token-mean and second-moment deviations indicate more "
            "uniform token marginals. Direct-uniform FID tests whether this "
            "also improves decoding of factorized uniform samples. RFM FID "
            "tests whether the remaining joint spatial structure is easier "
            "or harder to model."
        ),
    }
    base.save_json(summary, run_dir / "summary.json")
    print("\nExperiment complete. Results saved to:", run_dir)


def parse_args() -> CoverageConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train the SRUL product-of-spheres autoencoder with small and "
            "large tangent perturbations, then train a conditional RFM prior."
        )
    )
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./SRUL_CIFAR10_spherical_coverage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--init-ae-checkpoint")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-lr", type=float, default=5e-5)

    parser.add_argument("--alpha-max-deg", type=float, default=80.0)
    parser.add_argument("--alpha-mix-low-deg", type=float, default=80.0)
    parser.add_argument("--alpha-mix-high-deg", type=float, default=85.0)
    parser.add_argument("--alpha-mix-prob", type=float, default=0.10)
    parser.add_argument("--small-scale-max", type=float, default=0.50)

    parser.add_argument("--lambda-pix-recon-l1", type=float, default=1.0)
    parser.add_argument("--lambda-pix-recon-lpips", type=float, default=1.0)
    parser.add_argument("--lambda-pix-cons-l1", type=float, default=0.5)
    parser.add_argument("--lambda-pix-cons-lpips", type=float, default=0.5)
    parser.add_argument("--lambda-lat-cons", type=float, default=0.10)
    parser.add_argument("--lambda-edge", type=float, default=0.10)
    parser.add_argument("--lambda-clean-anchor", type=float, default=0.25)

    parser.add_argument("--sigma-enc", type=float, default=0.05)
    parser.add_argument("--prior-epochs", type=int, default=80)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--prior-width", type=int, default=256)
    parser.add_argument("--prior-depth", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument(
        "--time-sampling", choices=["uniform", "logit_normal"], default="logit_normal"
    )
    parser.add_argument("--label-drop-prob", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--guidance-scales", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--sample-steps", type=int, default=100)

    parser.add_argument("--metric-samples", type=int, default=10000)
    parser.add_argument("--pr-samples", type=int, default=5000)
    parser.add_argument("--recon-metric-samples", type=int, default=5000)
    parser.add_argument("--uniformity-samples", type=int, default=10000)
    parser.add_argument("--pr-chunk-size", type=int, default=256)
    parser.add_argument("--pr-nearest-k", type=int, default=5)
    parser.add_argument("--grid-samples-per-class", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--baseline-generation-csv")

    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--skip-heavy-metrics", action="store_true")
    parser.add_argument("--ae-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    train_samples = args.train_samples if args.train_samples > 0 else None
    test_samples = args.test_samples if args.test_samples > 0 else None
    return CoverageConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        init_ae_checkpoint=args.init_ae_checkpoint,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        alpha_max_deg=args.alpha_max_deg,
        alpha_mix_low_deg=args.alpha_mix_low_deg,
        alpha_mix_high_deg=args.alpha_mix_high_deg,
        alpha_mix_prob=args.alpha_mix_prob,
        small_scale_max=args.small_scale_max,
        lambda_pix_recon_l1=args.lambda_pix_recon_l1,
        lambda_pix_recon_lpips=args.lambda_pix_recon_lpips,
        lambda_pix_cons_l1=args.lambda_pix_cons_l1,
        lambda_pix_cons_lpips=args.lambda_pix_cons_lpips,
        lambda_lat_cons=args.lambda_lat_cons,
        lambda_edge=args.lambda_edge,
        lambda_clean_anchor=args.lambda_clean_anchor,
        sigma_enc=args.sigma_enc,
        prior_epochs=args.prior_epochs,
        prior_lr=args.prior_lr,
        prior_width=args.prior_width,
        prior_depth=args.prior_depth,
        time_dim=args.time_dim,
        time_sampling=args.time_sampling,
        label_drop_prob=args.label_drop_prob,
        ema_decay=args.ema_decay,
        guidance_scales=tuple(args.guidance_scales),
        sample_steps=args.sample_steps,
        metric_samples=args.metric_samples,
        pr_samples=args.pr_samples,
        recon_metric_samples=args.recon_metric_samples,
        uniformity_samples=args.uniformity_samples,
        pr_chunk_size=args.pr_chunk_size,
        pr_nearest_k=args.pr_nearest_k,
        grid_samples_per_class=args.grid_samples_per_class,
        checkpoint_every=args.checkpoint_every,
        compute_lpips=args.compute_lpips,
        skip_heavy_metrics=args.skip_heavy_metrics,
        ae_only=args.ae_only,
        resume=args.resume,
        amp=args.amp,
        baseline_generation_csv=args.baseline_generation_csv,
    )


if __name__ == "__main__":
    run(parse_args())
