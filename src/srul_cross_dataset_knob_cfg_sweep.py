"""Evaluate tangent-noise and classifier-free-guidance settings on a trained spatial spherical autoencoder."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def device_of_choice() -> torch.device:
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
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            value = value.detach()
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.detach().clone() for k, v in state.items()}


@dataclass
class Config:
    dataset: str = "cifar10"
    data_root: str = "./data"
    out_dir: str = "./SRUL_Sigma_Sweep"
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
    ae_checkpoint: str = ""

    sigma_values: Tuple[float, ...] = (0.05, 0.15, 0.30)
    dimension_scaled_noise: bool = True
    prior_epochs: int = 80
    prior_lr: float = 2e-4
    prior_width: int = 256
    prior_depth: int = 6
    time_dim: int = 128
    time_sampling: str = "logit_normal"
    label_drop_prob: float = 0.10
    ema_decay: float = 0.999
    guidance_scales: Tuple[float, ...] = (1.0, 1.5, 2.0)
    sample_steps: int = 100

    metric_samples: int = 5000
    pr_samples: int = 5000
    recon_metric_samples: int = 5000
    pr_chunk_size: int = 256
    pr_nearest_k: int = 5

    checkpoint_every: int = 5
    resume: bool = False
    amp: bool = False
    skip_heavy_metrics: bool = False

    hf_dataset: str = "flwrlabs/celeba"
    celeba_attribute: str = "Smiling"
    hf_shuffle_buffer: int = 10000


def import_base(dataset: str):
    module_name = (
        "srul_celeba64_conditional_experiment"
        if dataset in {"celeba64", "fake64"}
        else "srul_medmnist_conditional_experiment"
    )
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Put {module_name}.py beside this script before running."
        ) from exc


def make_data(base, config: Config, device: torch.device):
    name = "fake64" if config.dataset == "fake64" else (
        "fake" if config.dataset == "fake32" else config.dataset
    )
    if config.dataset in {"celeba64", "fake64"}:
        train_set, test_set = base.make_datasets(
            name=name,
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
            name=name,
            root=config.data_root,
            train_samples=config.train_samples,
            test_samples=config.test_samples,
            seed=config.seed,
        )
    return base.make_loaders(
        train_set, test_set, config.batch_size, config.num_workers, device
    )


def instantiate_ae(base, config: Config, device: torch.device):
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
    checkpoint = torch.load(config.ae_checkpoint, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    ae.load_state_dict(state)
    return ae.to(device).eval()


def create_field(base, config: Config, device: torch.device):
    return base.SpatialVectorField(
        latent_channels=config.latent_channels,
        width=config.prior_width,
        depth=config.prior_depth,
        time_dim=config.time_dim,
        num_classes=config.num_classes,
    ).to(device)


def sample_time(batch: int, device: torch.device, mode: str):
    if mode == "uniform":
        return torch.rand(batch, device=device)
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, device=device))
    raise ValueError(mode)


def label_dropout(labels: torch.Tensor, null_label: int, probability: float):
    labels = labels.clone().long()
    mask = torch.rand(labels.size(0), device=labels.device) < probability
    labels[mask] = null_label
    return labels


def rfm_loss(base, field, ae, images, labels, sigma: float, config: Config):
    with torch.no_grad():
        clean = ae.encode(images)
        target = base.tangent_noise_tokens(
            clean, sigma, config.dimension_scaled_noise
        )
    batch = images.size(0)
    source = base.sample_uniform_product_spheres(
        batch,
        config.latent_channels,
        config.image_size // 8,
        config.image_size // 8,
        images.device,
    )
    t = sample_time(batch, images.device, config.time_sampling)
    path, omega = base.slerp_tokens(target, source, t)
    velocity = base.slerp_velocity_tokens(target, source, t, omega)
    train_labels = label_dropout(
        labels, field.null_label, config.label_drop_prob
    )
    prediction = field(path, t, train_labels)
    prediction = base.tangent_projection_tokens(path, prediction)
    loss = F.mse_loss(prediction, velocity)
    radial = (
        (field(path, t, train_labels) * path).sum(dim=1).pow(2)
        / (field(path, t, train_labels).pow(2).sum(dim=1).clamp_min(1e-8))
    ).mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_omega": float(omega.detach().mean()),
        "raw_radial_fraction": float(radial.detach()),
    }


def sigma_tag(value: float) -> str:
    return str(value).replace(".", "p")


def train_prior(base, ae, sigma: float, loader, config: Config, device, run_dir):
    tag = sigma_tag(sigma)
    field = create_field(base, config, device)
    optimizer = torch.optim.AdamW(field.parameters(), lr=config.prior_lr, weight_decay=1e-4)
    scaler = grad_scaler(device, config.amp)
    ema = EMA(field, config.ema_decay)
    latest = run_dir / "checkpoints" / f"rfm_sigma_{tag}_latest.pt"
    final = run_dir / "checkpoints" / f"rfm_sigma_{tag}_final.pt"
    history: List[Dict[str, float]] = []
    start_epoch = 0

    if config.resume and latest.exists():
        obj = torch.load(latest, map_location=device)
        field.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        ema.load_state_dict(obj["ema"])
        history = list(obj.get("history", []))
        start_epoch = int(obj["epoch"])
        print(f"[sigma={sigma}] resumed from epoch {start_epoch}")
    elif config.resume and final.exists():
        obj = torch.load(final, map_location=device)
        field.load_state_dict(obj.get("ema", obj.get("model", obj)))
        return field, list(obj.get("history", []))

    ae.eval()
    for epoch in range(start_epoch, config.prior_epochs):
        field.train()
        totals: Dict[str, float] = {}
        batches = 0
        start = time.time()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().view(-1)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.amp):
                loss, row = rfm_loss(
                    base, field, ae, images, labels, sigma, config
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(field)
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        row = {k: v / max(batches, 1) for k, v in totals.items()}
        row.update({"epoch": epoch + 1, "seconds": time.time() - start})
        history.append(row)
        print(
            f"[sigma={sigma}] {epoch + 1:03d}/{config.prior_epochs} "
            f"loss={row['loss']:.6f}"
        )
        save_csv(history, run_dir / "logs" / f"rfm_sigma_{tag}.csv")
        if (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.prior_epochs:
            atomic_save(
                {
                    "model": field.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                    "sigma_enc": sigma,
                    "config": asdict(config),
                },
                latest,
            )
    atomic_save(
        {
            "model": field.state_dict(),
            "ema": ema.state_dict(),
            "history": history,
            "sigma_enc": sigma,
            "config": asdict(config),
        },
        final,
    )
    field.load_state_dict(ema.state_dict())
    return field, history


@torch.no_grad()
def cfg_prediction(field, z, t, labels, scale: float):
    null = torch.full_like(labels, field.null_label)
    uncond = field(z, t, null)
    cond = field(z, t, labels)
    return uncond + scale * (cond - uncond)


@torch.no_grad()
def generate(base, ae, field, labels, guidance_scale: float, config: Config, device):
    outputs = []
    latents = []
    latent_size = config.image_size // 8
    for start in range(0, labels.numel(), config.metric_batch_size):
        y = labels[start : start + config.metric_batch_size].to(device)
        z = base.sample_uniform_product_spheres(
            y.size(0),
            config.latent_channels,
            latent_size,
            latent_size,
            device,
        )
        dt = -1.0 / config.sample_steps
        for index in range(config.sample_steps, 0, -1):
            t = torch.full(
                (y.size(0),), index / config.sample_steps, device=device
            )
            raw = cfg_prediction(
                field, z, t, y, guidance_scale
            )
            tangent = base.tangent_projection_tokens(z, raw)
            z = base.exp_map_tokens(z, dt * tangent)
        outputs.append(base.to_uint8(ae.decode(z)).cpu())
        latents.append(z.flatten(1).cpu())
    return torch.cat(outputs), torch.cat(latents)


@torch.no_grad()
def collect_test(loader, count: int):
    images_out, labels_out = [], []
    total = 0
    for images, labels in loader:
        remaining = count - total
        images_out.append(images[:remaining].cpu())
        labels_out.append(labels[:remaining].long().view(-1).cpu())
        total += min(images.size(0), remaining)
        if total >= count:
            break
    return torch.cat(images_out), torch.cat(labels_out)


@torch.no_grad()
def reconstruction_metrics(base, ae, loader, sigma: float, config: Config, device):
    real, noisy_recon = [], []
    mse_sum = 0.0
    pixels = 0
    samples = 0
    for images, _ in loader:
        images = images.to(device)
        z = ae.encode(images)
        z_noisy = base.tangent_noise_tokens(
            z, sigma, config.dimension_scaled_noise
        )
        recon = ae.decode(z_noisy)
        remaining = config.recon_metric_samples - samples
        images = images[:remaining]
        recon = recon[:remaining]
        mse_sum += F.mse_loss(recon, images, reduction="sum").item()
        pixels += images.numel()
        samples += images.size(0)
        real.append(base.to_uint8(images).cpu())
        noisy_recon.append(base.to_uint8(recon).cpu())
        if samples >= config.recon_metric_samples:
            break
    real = torch.cat(real)[: config.recon_metric_samples]
    recon = torch.cat(noisy_recon)[: config.recon_metric_samples]
    mse = mse_sum / max(pixels, 1)
    result = {
        "reconstruction_mse": mse,
        "reconstruction_psnr": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
    }
    if not config.skip_heavy_metrics:
        result.update(
            {
                "reconstruction_" + key: value
                for key, value in base.compute_fid_kid(
                    real, recon, device, config.metric_batch_size
                ).items()
            }
        )
        n = min(config.pr_samples, real.size(0))
        real_features = base.extract_resnet_features(
            real[:n], device, config.metric_batch_size
        )
        recon_features = base.extract_resnet_features(
            recon[:n], device, config.metric_batch_size
        )
        pr = base.feature_precision_recall(
            real_features,
            recon_features,
            device,
            config.pr_nearest_k,
            config.pr_chunk_size,
        )
        result.update({"reconstruction_" + key: value for key, value in pr.items()})
    return result


def save_grid(images_uint8: torch.Tensor, path: Path):
    images = images_uint8[:64].float() / 127.5 - 1.0
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


def run(config: Config):
    if not config.ae_checkpoint:
        raise ValueError("--ae-checkpoint is required")
    if not config.guidance_scales:
        raise ValueError("At least one --guidance-scales value is required")
    if any(scale < 0 for scale in config.guidance_scales):
        raise ValueError("Guidance scales must be non-negative")

    set_seed(config.seed)
    device = device_of_choice()
    base = import_base(config.dataset)
    run_dir = ensure_dir(Path(config.out_dir) / f"seed_{config.seed}")
    for folder in ["checkpoints", "logs", "samples"]:
        ensure_dir(run_dir / folder)
    save_json(asdict(config), run_dir / "config.json")

    train_loader, test_loader = make_data(base, config, device)
    ae = instantiate_ae(base, config, device)
    max_test = max(
        config.metric_samples,
        config.pr_samples,
        config.recon_metric_samples,
    )
    real_float, labels = collect_test(test_loader, max_test)
    real_uint8 = base.to_uint8(real_float)

    real_features = None
    if not config.skip_heavy_metrics:
        pr_count = min(config.pr_samples, real_uint8.size(0))
        real_features = base.extract_resnet_features(
            real_uint8[:pr_count], device, config.metric_batch_size
        )

    rows: List[Dict[str, object]] = []
    for sigma in config.sigma_values:
        print("\n" + "=" * 80)
        print("Training prior for sigma_enc =", sigma)
        recon = reconstruction_metrics(
            base, ae, test_loader, sigma, config, device
        )
        field, history = train_prior(
            base, ae, sigma, train_loader, config, device, run_dir
        )

        for guidance_scale in config.guidance_scales:
            print("-" * 80)
            print(
                f"Evaluating sigma_enc={sigma}, CFG={guidance_scale}"
            )
            # Reuse the same source-noise sequence for all settings so the
            # comparison is less affected by sampling noise.
            sampling_seed = 100_000 + config.seed
            torch.manual_seed(sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_seed)

            fake, _ = generate(
                base,
                ae,
                field,
                labels[: config.metric_samples],
                guidance_scale,
                config,
                device,
            )
            cfg_tag = str(guidance_scale).replace(".", "p")
            save_grid(
                fake,
                run_dir
                / "samples"
                / f"sigma_{sigma_tag(sigma)}_cfg_{cfg_tag}.png",
            )

            row: Dict[str, object] = {
                "dataset": config.dataset,
                "sigma_enc": sigma,
                "seed": config.seed,
                "sampling_seed": sampling_seed,
                "guidance_scale": float(guidance_scale),
                "prior_final_loss": (
                    history[-1]["loss"] if history else float("nan")
                ),
            }
            row.update(recon)

            if not config.skip_heavy_metrics:
                n = min(
                    config.metric_samples,
                    real_uint8.size(0),
                    fake.size(0),
                )
                row.update(
                    base.compute_fid_kid(
                        real_uint8[:n],
                        fake[:n],
                        device,
                        config.metric_batch_size,
                    )
                )
                p = min(config.pr_samples, n)
                fake_features = base.extract_resnet_features(
                    fake[:p], device, config.metric_batch_size
                )
                assert real_features is not None
                row.update(
                    base.feature_precision_recall(
                        real_features[:p],
                        fake_features,
                        device,
                        config.pr_nearest_k,
                        config.pr_chunk_size,
                    )
                )

            rows.append(row)
            save_csv(rows, run_dir / "knob_cfg_sweep_metrics.csv")
            # Compatibility name for older notebooks.
            save_csv(rows, run_dir / "sigma_sweep_metrics.csv")
            print(row)

    save_json(
        {"config": asdict(config), "results": rows},
        run_dir / "summary.json",
    )
    print("Finished. Results saved to", run_dir)

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=[
        "cifar10", "pathmnist", "bloodmnist", "celeba64", "fake32", "fake64"
    ])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-dir", default="./SRUL_Sigma_Sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--test-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=96)
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--ae-checkpoint", required=True)
    p.add_argument("--sigma-values", nargs="+", type=float, default=[0.05, 0.15, 0.30])
    p.add_argument("--prior-epochs", type=int, default=80)
    p.add_argument("--prior-lr", type=float, default=2e-4)
    p.add_argument("--prior-width", type=int, default=256)
    p.add_argument("--prior-depth", type=int, default=6)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--time-sampling", default="logit_normal", choices=["uniform", "logit_normal"])
    p.add_argument("--label-drop-prob", type=float, default=0.10)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--guidance-scales", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--metric-samples", type=int, default=5000)
    p.add_argument("--pr-samples", type=int, default=5000)
    p.add_argument("--recon-metric-samples", type=int, default=5000)
    p.add_argument("--pr-chunk-size", type=int, default=256)
    p.add_argument("--pr-nearest-k", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip-heavy-metrics", action="store_true")
    p.add_argument("--hf-dataset", default="flwrlabs/celeba")
    p.add_argument("--celeba-attribute", default="Smiling")
    p.add_argument("--hf-shuffle-buffer", type=int, default=10000)
    a = p.parse_args()
    return Config(
        dataset=a.dataset,
        data_root=a.data_root,
        out_dir=a.out_dir,
        seed=a.seed,
        image_size=a.image_size,
        num_classes=a.num_classes,
        train_samples=None if a.train_samples <= 0 else a.train_samples,
        test_samples=None if a.test_samples <= 0 else a.test_samples,
        batch_size=a.batch_size,
        metric_batch_size=a.metric_batch_size,
        num_workers=a.num_workers,
        base_channels=a.base_channels,
        latent_channels=a.latent_channels,
        ae_checkpoint=a.ae_checkpoint,
        sigma_values=tuple(a.sigma_values),
        prior_epochs=a.prior_epochs,
        prior_lr=a.prior_lr,
        prior_width=a.prior_width,
        prior_depth=a.prior_depth,
        time_dim=a.time_dim,
        time_sampling=a.time_sampling,
        label_drop_prob=a.label_drop_prob,
        ema_decay=a.ema_decay,
        guidance_scales=tuple(a.guidance_scales),
        sample_steps=a.sample_steps,
        metric_samples=a.metric_samples,
        pr_samples=a.pr_samples,
        recon_metric_samples=a.recon_metric_samples,
        pr_chunk_size=a.pr_chunk_size,
        pr_nearest_k=a.pr_nearest_k,
        checkpoint_every=a.checkpoint_every,
        resume=a.resume,
        amp=a.amp,
        skip_heavy_metrics=a.skip_heavy_metrics,
        hf_dataset=a.hf_dataset,
        celeba_attribute=a.celeba_attribute,
        hf_shuffle_buffer=a.hf_shuffle_buffer,
    )


if __name__ == "__main__":
    run(parse_args())
