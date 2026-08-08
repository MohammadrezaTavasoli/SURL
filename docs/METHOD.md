# Method

## 1. Main design principle

An autoencoder does not have to produce Gaussian latents. Its architecture and normalization determine the latent geometry. Normalization can make token norms nearly constant, so direction carries most of the information.

Spherical Riemannian Unit-Norm Latents (SRUL) trains a spatial spherical autoencoder and an RFM prior on the same product-of-spheres support. This avoids two mismatches:

- **path mismatch:** a straight chord leaves a spherical support;
- **support mismatch:** a direction-only spherical prior is attached after a VAE has learned to use variable token radii.

## 2. Pipeline overview

![SRUL pipeline](../assets/srul_pipeline.png)

SRUL is trained in two stages. First, the autoencoder learns spatial unit-norm tokens and a decoder that is stable under tangent perturbations. Second, RFM learns the distribution of the encoded token grids. Sampling transports spherical noise with tangent-projected velocities and exponential-map updates.

## 3. Spatial product-of-spheres representation

The encoder produces `B x C x H x W` tokens. Each spatial token is normalized over its channel dimension:

```math
z_{:,i,j}=\frac{h_{:,i,j}}{\|h_{:,i,j}\|_2} \in S^{C-1}.
```

The full latent space is

```math
\mathcal M=(S^{C-1})^{HW}.
```

The reported experiments use `C=32`, a `4x4` grid for 32x32 images, and an `8x8` grid for CelebA-64. The normalization is part of autoencoder training; it is not added after training.

## 4. Tangent-noise control

Gaussian noise is projected onto the tangent space:

```math
u=\Pi_z(\xi)=\xi-\langle\xi,z\rangle z.
```

The perturbed latent is created with the exponential map:

```math
z_\sigma=\operatorname{Exp}_z(\sigma_{\mathrm{enc}}u).
```

This changes direction without changing radius. The sweep over `sigma_enc` is implemented in `src/srul_sigma_knob_sweep.py`.

## 5. Riemannian Flow Matching

The source and target lie on the same product-of-spheres support. Their token-wise path is SLERP:

```math
z_t=\frac{\sin((1-t)\Omega)}{\sin\Omega}z_\sigma
+\frac{\sin(t\Omega)}{\sin\Omega}\epsilon.
```

The network predicts a velocity that is projected onto the tangent space before the velocity-matching loss is computed. Sampling uses the same projection followed by exponential-map integration, so every token stays on its sphere.

## 6. Conditional extension

Class or attribute labels are optional inputs to the velocity field. Classifier-free guidance combines conditional and null-label predictions during sampling. The guided velocity is projected before every exponential-map step, so conditioning does not change the latent support.

## 7. VAE support-mismatch diagnostic

For a VAE token, write

```math
z=r u,\qquad r=\|z\|_2,\qquad u=z/\|z\|_2.
```

The same trained VAE encoder and decoder are used for both variants. **VAE + standard FM** models the complete Euclidean token `z`. **VAE + projected RFM** models only `u`, restores a class/token mean radius, and applies the same decoder.

The experiment measures token-radius variation and compares reconstruction from:

- the posterior mean;
- a posterior sample;
- direction with class/token mean radius;
- direction with unit radius.

It then compares SRUL, VAE + standard FM, and VAE + projected RFM using FID, precision, and recall. Full metrics, including KID, remain in the result CSV files. The deterministic-autoencoder version is retained in `archive/` as supplementary evidence.
