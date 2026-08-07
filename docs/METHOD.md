# Method

## 1. Why study spherical latent geometry?

An autoencoder does not guarantee a Gaussian latent space. The geometry is learned by the encoder and can depend strongly on normalization. When token norms vary little, the representation lies close to a hypersphere and most variation is carried by direction. Transformer autoencoders with LayerNorm are one common example, while explicit L2 normalization can create the same structure in other encoders.

If a Euclidean prior is applied without accounting for this geometry, its probability path can enter regions that the encoder does not produce. SRUL makes the spherical structure exact through token-wise normalization and provides a controlled test: the encoder and decoder are held fixed while chord and geodesic transport are compared.

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
