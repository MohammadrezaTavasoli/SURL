# Method

## 1. Why use spherical tokens?

An autoencoder does not guarantee a Gaussian latent distribution. Its geometry depends on the architecture and training objective. Encoders that use LayerNorm, feature normalization, or explicit L2 normalization can produce features with small radial variation and information carried mainly by direction. Transformer autoencoders are one common example, but not the only case. Ignoring the learned affine scale and shift, a LayerNorm token has approximately zero mean and unit variance, so its squared norm is close to the channel dimension. If these features are modeled with a Gaussian source and a linear Euclidean path, the model introduces radial variation and intermediate states outside the fixed-radius shell. SRUL uses token-wise L2 normalization to make the spherical geometry exact and Riemannian Flow Matching to keep the transport path on that geometry.

## 2. Pipeline overview

![SRUL pipeline](../assets/srul_pipeline.png)

SRUL is trained in two connected stages. First, an autoencoder learns a spatial product-of-spheres representation and a decoder that is stable under tangent perturbations. Second, an RFM prior learns the nonuniform distribution of encoded token grids. During sampling, the prior transports spherical noise with tangent-projected velocities and exponential-map steps, and the decoder maps the sampled tokens to an image.

## 3. Spatial product-of-spheres representation

The encoder produces `B x C x H x W` tokens. Every spatial token is normalized over its channel dimension:

```math
z_{:,i,j}=\frac{h_{:,i,j}}{\|h_{:,i,j}\|_2} \in S^{C-1}.
```

The full latent space is

```math
\mathcal M=(S^{C-1})^{HW}.
```

The final experiments use `C=32`, a `4x4` grid for 32x32 images, and an `8x8` grid for CelebA-64.

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

The source and target lie on the same product-of-spheres manifold. Their tokenwise probability path is SLERP:

```math
z_t=\frac{\sin((1-t)\Omega)}{\sin\Omega}z_\sigma
+\frac{\sin(t\Omega)}{\sin\Omega}\epsilon.
```

The network predicts a velocity, which is projected onto the tangent space before the velocity-matching loss is computed. Sampling uses the same projection followed by exponential-map integration, so every token remains on its sphere.

## 6. Conditional extension

Class or attribute labels are optional inputs to the velocity field. Classifier-free guidance combines conditional and null-label predictions during sampling. The guided field is projected before every exponential-map step, so conditioning does not change the core geometry.
