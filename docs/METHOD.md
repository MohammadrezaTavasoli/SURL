# Method

## Design principle

SRUL co-designs the latent support and the prior. The encoder is trained to produce a spatial grid of unit-norm tokens, and RFM is trained on that same product-of-spheres support.

This avoids two mismatches:

- **path mismatch:** spherical endpoints are connected by a straight chord that enters the sphere;
- **support mismatch:** a direction-only spherical prior is attached after a VAE has learned to use variable token radii.

## Spatial spherical autoencoder

For a feature tensor `h = E(x)`, every spatial token is normalized over channels:

```math
z_{:,i,j}=\frac{h_{:,i,j}}{\|h_{:,i,j}\|_2}\in S^{C-1}.
```

The full latent support is

```math
\mathcal M=(S^{C-1})^{HW}.
```

The main 32x32 model uses 32 channels and a 4x4 token grid. CelebA-64 uses an 8x8 grid.

## Approximate uniform token coverage

Unit normalization alone can leave token directions concentrated in a small angular region. During autoencoder training, SRUL creates a small and a larger tangent rotation along the same random direction. The small branch reconstructs the image; the larger branch is trained to decode consistently and to re-encode toward the clean latent direction.

This makes a wider spherical neighborhood decodable and encourages each token marginal to cover its sphere approximately uniformly:

```math
q_\phi(z_{i,j})\approx \operatorname{Unif}(S^{C-1}).
```

The complete token grid can still contain spatial dependencies. RFM learns those remaining joint dependencies.

## Information-control knob

A Gaussian vector is projected onto the tangent space,

```math
u=\Pi_z(\xi)=\xi-\langle\xi,z\rangle z,
```

and the token is rotated with the exponential map:

```math
z_\sigma=\operatorname{Exp}_z(\sigma_{\mathrm{enc}}u).
```

`sigma_enc` controls how strongly image-specific directions are perturbed. It is a controlled proxy for retained information, not an exact bitrate.

## Riemannian Flow Matching

The source is factorized uniform spherical noise. The target is the encoded latent distribution. Each token follows a SLERP path:

```math
z_t=\frac{\sin((1-t)\Omega)}{\sin\Omega}z_0+
\frac{\sin(t\Omega)}{\sin\Omega}z_1.
```

The network predicts a velocity, which is projected onto the tangent space. The RFM loss matches this projected prediction to the exact derivative of the geodesic path.

At inference, sampling starts from spherical noise and applies tangent-projected exponential-map updates. Every intermediate token therefore remains unit norm.

## Conditional sampling

Labels condition the velocity network. Classifier-free guidance combines a null-label and conditional prediction:

```math
v_{\mathrm{CFG}}=v_\varnothing+s(v_y-v_\varnothing).
```

The combined velocity is projected before each exponential-map update.

## Network architecture

Main CIFAR-10 implementation:

- encoder: `3 -> 96 -> 192 -> 384 -> 32`, spatial size `32 -> 16 -> 8 -> 4`;
- token grid: `32 x 4 x 4`;
- decoder: `32 -> 384 -> 192 -> 96 -> 3`;
- RFM prior: input/output 32 channels, hidden width 256, six conditioned residual blocks;
- time and class embedding: 128 dimensions;
- sampler: 100 exponential-map steps.

See [`MATHEMATICAL_DETAILS.md`](MATHEMATICAL_DETAILS.md) for the loss interpretation, SLERP proof, exponential-map proof, and CFG/Bayes connection.
