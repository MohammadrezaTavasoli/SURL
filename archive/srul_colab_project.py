"""
SRUL proof-of-concept for Google Colab / local Python.

Goal: demonstrate the main mechanisms of Spherical Riemannian Unit-Norm Latents (SRUL):
1) exact spherical latent via normalization;
2) tangent-noise bitrate knob sigma_enc;
3) geodesic flow-matching prior on the sphere;
4) controlled comparison with a Euclidean chord prior.

Recommended Colab run:
    python srul_colab_project.py --dataset fashionmnist --latent-dim 16 --ae-epochs 8 --prior-epochs 15 --baseline-epochs 10

Outputs saved to --out-dir:
    geometry_norm_collapse.png
    reconstructions.png
    srul_samples.png
    euclidean_samples.png
    rate_distortion.png / rate_distortion.txt
    prior_loss_curves.png
    norm_stats.txt
    srul_model.pt
"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Model blocks
# -----------------------------

class Encoder(nn.Module):
    """Small MLP encoder for 28x28 grayscale images."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Small MLP decoder from latent to image."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PriorField(nn.Module):
    """Time-conditioned vector field v_theta(z,t)."""
    def __init__(self, latent_dim: int, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=1))


@dataclass
class AEOutputs:
    z_clean: torch.Tensor
    z_noisy: torch.Tensor
    recon_clean: torch.Tensor
    recon_noisy: torch.Tensor


class SRULModel(nn.Module):
    """Encoder + decoder + two prior fields for controlled comparison."""
    def __init__(self, input_dim: int, latent_dim: int, prior_width: int = 512):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.srul_prior = PriorField(latent_dim, width=prior_width)
        self.euclidean_prior = PriorField(latent_dim, width=prior_width)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode_to_sphere(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return normalize_to_sphere(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_ae(self, x: torch.Tensor, sigma_enc: float) -> AEOutputs:
        z_clean = self.encode_to_sphere(x)
        z_noisy = tangent_noisy_latent(z_clean, sigma_enc)
        recon_clean = self.decode(z_clean)
        recon_noisy = self.decode(z_noisy)
        return AEOutputs(z_clean, z_noisy, recon_clean, recon_noisy)


# -----------------------------
# Sphere geometry helpers
# -----------------------------

def normalize_to_sphere(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project vectors onto the unit sphere."""
    return h / h.norm(dim=1, keepdim=True).clamp_min(eps)


def tangent_projection(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project vector u to tangent space at unit-sphere point z.
    Tangent space is {u : <u,z>=0}.
    """
    return u - (u * z).sum(dim=1, keepdim=True) * z


def exp_map_sphere(z: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map on unit sphere.
    It moves from point z along tangent vector v while staying on the sphere.
    """
    v_norm = v.norm(dim=1, keepdim=True)
    direction = v / v_norm.clamp_min(eps)
    mapped = torch.cos(v_norm) * z + torch.sin(v_norm) * direction
    # fallback when v is extremely small
    small = (v_norm.squeeze(1) < eps)
    if small.any():
        mapped[small] = normalize_to_sphere(z[small] + v[small])
    return normalize_to_sphere(mapped)


def tangent_noisy_latent(z_clean: torch.Tensor, sigma_enc: float) -> torch.Tensor:
    """SRUL encoder noise: tangent Gaussian + exponential map."""
    xi = torch.randn_like(z_clean)
    u = tangent_projection(z_clean, xi)
    return exp_map_sphere(z_clean, sigma_enc * u)


def sample_uniform_sphere(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    eps = torch.randn(batch_size, latent_dim, device=device)
    return normalize_to_sphere(eps)


def slerp(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spherical linear interpolation between z0 and eps.
    t=0 -> z0, t=1 -> eps.
    Returns point z_t and angle omega.
    """
    dot = (z0 * eps).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    if t.ndim == 2:
        t = t.squeeze(1)
    c0 = torch.sin((1 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega
    zt = c0.unsqueeze(1) * z0 + c1.unsqueeze(1) * eps
    return normalize_to_sphere(zt), omega


def slerp_velocity(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Derivative d/dt SLERP(z0, eps; t)."""
    if t.ndim == 2:
        t = t.squeeze(1)
    scale = omega / torch.sin(omega).clamp_min(1e-6)
    term = (
        torch.cos(t * omega).unsqueeze(1) * eps
        - torch.cos((1 - t) * omega).unsqueeze(1) * z0
    )
    return scale.unsqueeze(1) * term


def sinc_sq(x: torch.Tensor) -> torch.Tensor:
    """Stable sinc(x)^2."""
    out = torch.ones_like(x)
    mask = x.abs() > 1e-6
    out[mask] = (torch.sin(x[mask]) / x[mask]) ** 2
    return out


# -----------------------------
# Losses
# -----------------------------

def autoencoder_loss(model: SRULModel, x: torch.Tensor, sigma_enc: float,
                     lambda_con: float, lambda_lat: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model.forward_ae(x, sigma_enc)
    loss_recon = F.mse_loss(out.recon_clean, x)
    # smooth decoding under small tangent perturbation
    loss_cons = F.mse_loss(out.recon_noisy, out.recon_clean.detach())
    # latent self-consistency after decode/re-encode
    reenc = model.encode_to_sphere(out.recon_noisy.detach())
    loss_lat = 1.0 - F.cosine_similarity(out.z_clean.detach(), reenc, dim=1).mean()
    loss = loss_recon + lambda_con * loss_cons + lambda_lat * loss_lat
    return loss, {
        "ae_total": float(loss.item()),
        "recon": float(loss_recon.item()),
        "cons": float(loss_cons.item()),
        "lat": float(loss_lat.item()),
    }


def srul_prior_loss(model: SRULModel, x: torch.Tensor, sigma_enc: float,
                    use_jacobi: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RJF-style geodesic flow-matching loss on the sphere."""
    with torch.no_grad():
        z_clean = model.encode_to_sphere(x)
        z0 = tangent_noisy_latent(z_clean, sigma_enc)
    batch = x.size(0)
    eps = sample_uniform_sphere(batch, model.latent_dim, x.device)
    t = torch.rand(batch, device=x.device)
    zt, omega = slerp(z0, eps, t)
    target_vel = slerp_velocity(z0, eps, t, omega)
    pred = model.srul_prior(zt, t)
    pred_tan = tangent_projection(zt, pred)
    sq_err = (pred_tan - target_vel).pow(2).sum(dim=1, keepdim=True)
    weight = sinc_sq((1 - t) * omega).unsqueeze(1) if use_jacobi else torch.ones_like(sq_err)
    loss = (weight * sq_err).mean()
    return loss, {
        "srul_prior": float(loss.item()),
        "avg_omega": float(omega.mean().item()),
        "avg_weight": float(weight.mean().item()),
    }


def euclidean_prior_loss(model: SRULModel, x: torch.Tensor, sigma_enc: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Euclidean chord baseline.
    Same sphere endpoints, but path is linear chord: zt=(1-t)z0 + t eps.
    """
    with torch.no_grad():
        z_clean = model.encode_to_sphere(x)
        z0 = tangent_noisy_latent(z_clean, sigma_enc)
    batch = x.size(0)
    eps = sample_uniform_sphere(batch, model.latent_dim, x.device)
    t = torch.rand(batch, device=x.device)
    zt = (1 - t).unsqueeze(1) * z0 + t.unsqueeze(1) * eps
    target_vel = eps - z0
    pred = model.euclidean_prior(zt, t)
    loss = F.mse_loss(pred, target_vel)
    return loss, {"euclidean_prior": float(loss.item()), "avg_norm_zt": float(zt.norm(dim=1).mean().item())}


# -----------------------------
# Training loops
# -----------------------------

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def train_autoencoder(model: SRULModel, loader: DataLoader, device: torch.device, epochs: int,
                      sigma_enc: float, lr: float, lambda_con: float, lambda_lat: float) -> List[Dict[str, float]]:
    set_requires_grad(model.encoder, True)
    set_requires_grad(model.decoder, True)
    set_requires_grad(model.srul_prior, False)
    set_requires_grad(model.euclidean_prior, False)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)
    logs = []
    model.train()
    for epoch in range(1, epochs + 1):
        agg = {"ae_total": 0.0, "recon": 0.0, "cons": 0.0, "lat": 0.0}
        n = 0
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            opt.zero_grad()
            loss, stats = autoencoder_loss(model, x, sigma_enc, lambda_con, lambda_lat)
            loss.backward()
            opt.step()
            b = x.size(0)
            for k in agg:
                agg[k] += stats[k] * b
            n += b
        row = {k: v / n for k, v in agg.items()}
        row["epoch"] = epoch
        logs.append(row)
        print(f"[AE] {epoch:03d}/{epochs} total={row['ae_total']:.4f} recon={row['recon']:.4f} cons={row['cons']:.4f} lat={row['lat']:.4f}")
    return logs


def train_srul_prior(model: SRULModel, loader: DataLoader, device: torch.device, epochs: int,
                     sigma_enc: float, lr: float, use_jacobi: bool = True) -> List[Dict[str, float]]:
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.decoder, False)
    set_requires_grad(model.srul_prior, True)
    set_requires_grad(model.euclidean_prior, False)
    opt = torch.optim.Adam(model.srul_prior.parameters(), lr=lr)
    logs = []
    model.train()
    for epoch in range(1, epochs + 1):
        agg = {"srul_prior": 0.0, "avg_omega": 0.0, "avg_weight": 0.0}
        n = 0
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            opt.zero_grad()
            loss, stats = srul_prior_loss(model, x, sigma_enc, use_jacobi=use_jacobi)
            loss.backward()
            opt.step()
            b = x.size(0)
            for k in agg:
                agg[k] += stats[k] * b
            n += b
        row = {k: v / n for k, v in agg.items()}
        row["epoch"] = epoch
        logs.append(row)
        print(f"[SRUL PRIOR] {epoch:03d}/{epochs} loss={row['srul_prior']:.4f} omega={row['avg_omega']:.4f} weight={row['avg_weight']:.4f}")
    return logs


def train_euclidean_prior(model: SRULModel, loader: DataLoader, device: torch.device, epochs: int,
                          sigma_enc: float, lr: float) -> List[Dict[str, float]]:
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.decoder, False)
    set_requires_grad(model.srul_prior, False)
    set_requires_grad(model.euclidean_prior, True)
    opt = torch.optim.Adam(model.euclidean_prior.parameters(), lr=lr)
    logs = []
    model.train()
    for epoch in range(1, epochs + 1):
        agg = {"euclidean_prior": 0.0, "avg_norm_zt": 0.0}
        n = 0
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            opt.zero_grad()
            loss, stats = euclidean_prior_loss(model, x, sigma_enc)
            loss.backward()
            opt.step()
            b = x.size(0)
            for k in agg:
                agg[k] += stats[k] * b
            n += b
        row = {k: v / n for k, v in agg.items()}
        row["epoch"] = epoch
        logs.append(row)
        print(f"[EUCLIDEAN PRIOR] {epoch:03d}/{epochs} loss={row['euclidean_prior']:.4f} chord_norm={row['avg_norm_zt']:.4f}")
    return logs


# -----------------------------
# Sampling and evaluation
# -----------------------------

@torch.no_grad()
def sample_srul(model: SRULModel, device: torch.device, num_samples: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Sample using geodesic exp-map integration (stays on sphere)."""
    model.eval()
    z = sample_uniform_sphere(num_samples, model.latent_dim, device)
    norm_path = [float(z.norm(dim=1).mean().item())]
    for i in range(steps, 0, -1):
        t_val = torch.full((num_samples,), i / steps, device=device)
        pred = model.srul_prior(z, t_val)
        pred_tan = tangent_projection(z, pred)
        z = exp_map_sphere(z, (-1.0 / steps) * pred_tan)
        norm_path.append(float(z.norm(dim=1).mean().item()))
    x_hat = model.decode(z)
    return x_hat, z, norm_path


@torch.no_grad()
def sample_euclidean(model: SRULModel, device: torch.device, num_samples: int, steps: int,
                     normalize_final_for_decoder: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Sample with Euclidean Euler updates (can drift off sphere)."""
    model.eval()
    z = sample_uniform_sphere(num_samples, model.latent_dim, device)
    norm_path = [float(z.norm(dim=1).mean().item())]
    for i in range(steps, 0, -1):
        t_val = torch.full((num_samples,), i / steps, device=device)
        pred = model.euclidean_prior(z, t_val)
        z = z + (-1.0 / steps) * pred
        norm_path.append(float(z.norm(dim=1).mean().item()))
    z_dec = normalize_to_sphere(z) if normalize_final_for_decoder else z
    x_hat = model.decode(z_dec)
    return x_hat, z, norm_path


@torch.no_grad()
def evaluate_rate_distortion(model: SRULModel, loader: DataLoader, device: torch.device,
                             sigma_values: List[float]) -> List[Tuple[float, float]]:
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device).view(x.size(0), -1)
    out = []
    for sigma in sigma_values:
        ae = model.forward_ae(x, sigma)
        mse = F.mse_loss(ae.recon_noisy, x).item()
        out.append((sigma, mse))
    return out


@torch.no_grad()
def save_reconstructions(model: SRULModel, loader: DataLoader, device: torch.device, out_dir: str,
                         sigma_enc: float, shape: Tuple[int, int, int]) -> None:
    model.eval()
    c, h, w = shape
    x, _ = next(iter(loader))
    x = x.to(device)[:16]
    xf = x.view(x.size(0), -1)
    ae = model.forward_ae(xf, sigma_enc)
    grid = torch.cat([
        x,
        ae.recon_clean.view(-1, c, h, w)[:16],
        ae.recon_noisy.view(-1, c, h, w)[:16],
    ], dim=0)
    save_image(grid, os.path.join(out_dir, "reconstructions.png"), nrow=16)


def write_logs(out_dir: str, logs: Dict[str, List[Dict[str, float]]]) -> None:
    for name, rows in logs.items():
        if not rows:
            continue
        path = os.path.join(out_dir, f"{name}.csv")
        keys = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")


def plot_outputs(out_dir: str, rd: List[Tuple[float, float]], srul_logs: List[Dict[str, float]],
                 euc_logs: List[Dict[str, float]], srul_norm_path: List[float], euc_norm_path: List[float]) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    # Rate-distortion
    sigmas, mses = zip(*rd)
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(sigmas, mses, marker="o")
    plt.xlabel(r"$\sigma_{enc}$")
    plt.ylabel("Reconstruction MSE")
    plt.title("Rate-distortion controlled by tangent noise")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rate_distortion.png"), dpi=180)
    plt.close()

    # Prior losses
    plt.figure(figsize=(5.0, 3.4))
    if srul_logs:
        plt.plot([r["epoch"] for r in srul_logs], [r["srul_prior"] for r in srul_logs], marker="o", label="SRUL geodesic prior")
    if euc_logs:
        plt.plot([r["epoch"] for r in euc_logs], [r["euclidean_prior"] for r in euc_logs], marker="s", label="Euclidean chord prior")
    plt.xlabel("Epoch")
    plt.ylabel("Prior loss")
    plt.title("Prior training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prior_loss_curves.png"), dpi=180)
    plt.close()

    # Sampling norm paths
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(srul_norm_path, label="SRUL exp-map sampling")
    plt.plot(euc_norm_path, label="Euclidean Euler sampling")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="k")
    plt.xlabel("Integration step")
    plt.ylabel(r"Mean $\|z\|_2$")
    plt.title("Norm preservation during sampling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sampling_norm_paths.png"), dpi=180)
    plt.close()


def plot_geometry_norm_collapse(out_dir: str, latent_dim: int = 32, n_pairs: int = 4096) -> None:
    """Analytic toy check: Euclidean chord norm collapses, SLERP stays at 1."""
    if plt is None:
        return
    device = torch.device("cpu")
    z0 = sample_uniform_sphere(n_pairs, latent_dim, device)
    eps = sample_uniform_sphere(n_pairs, latent_dim, device)
    ts = torch.linspace(0, 1, 101)
    chord_norm = []
    slerp_norm = []
    for t in ts:
        t_batch = torch.full((n_pairs,), float(t))
        chord = (1 - t) * z0 + t * eps
        zt, _ = slerp(z0, eps, t_batch)
        chord_norm.append(float(chord.norm(dim=1).mean().item()))
        slerp_norm.append(float(zt.norm(dim=1).mean().item()))
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(ts.numpy(), chord_norm, label="Euclidean chord")
    plt.plot(ts.numpy(), slerp_norm, label="SLERP geodesic")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="k")
    plt.xlabel("t")
    plt.ylabel(r"Mean $\|z_t\|_2$")
    plt.title("Chord path leaves the sphere; SLERP stays on it")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "geometry_norm_collapse.png"), dpi=180)
    plt.close()


# -----------------------------
# Data / CLI
# -----------------------------

def get_dataset(name: str, root: str, max_train_samples: Optional[int] = None):
    transform = transforms.ToTensor()
    name = name.lower()
    if name == "mnist":
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        shape = (1, 28, 28)
    elif name in {"fashionmnist", "fashion-mnist", "fmnist"}:
        dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        shape = (1, 28, 28)
    else:
        raise ValueError("This proof-of-concept supports mnist or fashionmnist.")
    if max_train_samples is not None and max_train_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_train_samples)))
    return dataset, shape


def parse_args():
    p = argparse.ArgumentParser(description="SRUL proof-of-concept for Colab.")
    p.add_argument("--dataset", type=str, default="fashionmnist", choices=["mnist", "fashionmnist", "fashion-mnist", "fmnist"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./srul_course_runs")
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ae-epochs", type=int, default=8)
    p.add_argument("--prior-epochs", type=int, default=15)
    p.add_argument("--baseline-epochs", type=int, default=10)
    p.add_argument("--prior-width", type=int, default=512)
    p.add_argument("--lr-ae", type=float, default=1e-3)
    p.add_argument("--lr-prior", type=float, default=1e-3)
    p.add_argument("--sigma-enc", type=float, default=0.20)
    p.add_argument("--lambda-con", type=float, default=0.5)
    p.add_argument("--lambda-lat", type=float, default=0.05)
    p.add_argument("--sample-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-euclidean-baseline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print("Device:", device)
    print("Output directory:", os.path.abspath(args.out_dir))
    plot_geometry_norm_collapse(args.out_dir, latent_dim=args.latent_dim)

    dataset, shape = get_dataset(args.dataset, args.data_root, args.max_train_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    input_dim = shape[0] * shape[1] * shape[2]
    model = SRULModel(input_dim=input_dim, latent_dim=args.latent_dim, prior_width=args.prior_width).to(device)

    print("\nStage 1: spherical autoencoder")
    ae_logs = train_autoencoder(model, loader, device, args.ae_epochs, args.sigma_enc, args.lr_ae, args.lambda_con, args.lambda_lat)

    print("\nStage 2: SRUL geodesic prior")
    srul_logs = train_srul_prior(model, loader, device, args.prior_epochs, args.sigma_enc, args.lr_prior, use_jacobi=True)

    euc_logs = []
    if not args.no_euclidean_baseline:
        print("\nStage 3: Euclidean chord prior baseline")
        euc_logs = train_euclidean_prior(model, loader, device, args.baseline_epochs, args.sigma_enc, args.lr_prior)

    print("\nSaving visualizations and metrics")
    save_reconstructions(model, loader, device, args.out_dir, args.sigma_enc, shape)
    srul_samples, srul_z, srul_norm_path = sample_srul(model, device, 64, args.sample_steps)
    save_image(srul_samples.view(-1, *shape), os.path.join(args.out_dir, "srul_samples.png"), nrow=8)
    if not args.no_euclidean_baseline:
        euc_samples, euc_z, euc_norm_path = sample_euclidean(model, device, 64, args.sample_steps, normalize_final_for_decoder=False)
        save_image(euc_samples.view(-1, *shape).clamp(0, 1), os.path.join(args.out_dir, "euclidean_samples.png"), nrow=8)
    else:
        euc_z = torch.empty_like(srul_z)
        euc_norm_path = []

    sigma_values = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    rd = evaluate_rate_distortion(model, loader, device, sigma_values)
    with open(os.path.join(args.out_dir, "rate_distortion.txt"), "w", encoding="utf-8") as f:
        for sigma, mse in rd:
            f.write(f"sigma={sigma:.3f}, recon_mse={mse:.6f}\n")
            print(f"sigma={sigma:.3f}, recon_mse={mse:.6f}")

    with open(os.path.join(args.out_dir, "norm_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"srul_sample_mean_norm={srul_z.norm(dim=1).mean().item():.6f}\n")
        f.write(f"srul_sample_std_norm={srul_z.norm(dim=1).std().item():.6f}\n")
        if not args.no_euclidean_baseline:
            f.write(f"euclidean_sample_mean_norm={euc_z.norm(dim=1).mean().item():.6f}\n")
            f.write(f"euclidean_sample_std_norm={euc_z.norm(dim=1).std().item():.6f}\n")

    write_logs(args.out_dir, {"ae_logs": ae_logs, "srul_prior_logs": srul_logs, "euclidean_prior_logs": euc_logs})
    plot_outputs(args.out_dir, rd, srul_logs, euc_logs, srul_norm_path, euc_norm_path)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "srul_model.pt"))
    print("Done. Check output directory for images and metrics.")


if __name__ == "__main__":
    main()
