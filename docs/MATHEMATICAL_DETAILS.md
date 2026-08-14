# Mathematical details behind SRUL

The final report keeps the main five pages concise and places the longer explanations in an appendix. This note mirrors those derivations for repository readers.

## 1. Autoencoder losses

The clean and noisy `L1` terms compare pixels and preserve color and spatial content. The edge loss compares image gradients and helps preserve boundaries.

### Perceptual loss

LPIPS passes the real and reconstructed images through a frozen pretrained feature network and compares normalized feature maps:

```math
\mathcal L_{\mathrm{LPIPS}}(x,\hat x)
=\sum_\ell \frac{1}{H_\ell W_\ell}
\sum_p \|w_\ell\odot(\bar f_\ell(x)_p-\bar f_\ell(\hat x)_p)\|_2^2.
```

Pixel loss asks whether corresponding pixels are close. LPIPS asks whether the images have similar learned visual features, such as edges, texture, and larger shape patterns.

### Latent consistency

For a tangent-perturbed token grid `z_sigma`, decode and encode again:

```math
\tilde z=E_\phi(D_\psi(z_\sigma)).
```

The loss is the average cosine distance from the clean unit-norm tokens:

```math
\mathcal L_{\mathrm{lat}}
=1-\frac{1}{BHW}\sum_{b,i,j}z_{b,i,j}^{\top}\tilde z_{b,i,j}.
```

Because the tokens have unit norm, the dot product is the cosine of their angle. The loss encourages a noisy decode-and-reencode cycle to return to the same clean token directions.

## 2. Tangent noise as information control

For a clean unit-norm token `z`, sample Gaussian noise and project it onto the tangent space:

```math
u=\xi-\langle \xi,z\rangle z.
```

The perturbed token is

```math
z_\sigma=\operatorname{Exp}_z(\sigma_{\mathrm{enc}}u).
```

The scale `sigma_enc` changes the angular distance from the clean token while preserving unit norm. It therefore acts as an explicit information-control knob: small values preserve more sample-specific detail, while larger values create a stronger bottleneck. It is a controlled proxy for retained information, not an exact bitrate in bits. In the CIFAR-10 sweep, larger values worsened both reconstruction and generation, so `0.05` was the best tested value.

## 3. Standard linear Flow Matching

For source noise `epsilon` and target latent `z`, standard linear Flow Matching uses

```math
x_t=(1-t)\epsilon+t z,
\qquad u_t=z-\epsilon.
```

The neural field is trained by velocity regression:

```math
\mathcal L_{\mathrm{FM}}
=\mathbb E\|v_\theta(x_t,t)-u_t\|_2^2.
```

At inference, a sample is drawn from the source and the learned ODE is integrated.

## 4. Why SLERP is geodesic

For unit endpoints `a` and `b`, let

```math
\Omega=\arccos(a^\top b),
\qquad q=\frac{b-\cos\Omega\,a}{\sin\Omega}.
```

Then `q` is unit length and orthogonal to `a`. SLERP can be rewritten as

```math
\gamma(t)=\cos(t\Omega)a+\sin(t\Omega)q.
```

Therefore

```math
\|\gamma(t)\|_2^2=\cos^2(t\Omega)+\sin^2(t\Omega)=1.
```

Its speed is constant and

```math
\ddot\gamma(t)=-\Omega^2\gamma(t).
```

The acceleration is purely normal to the sphere, so the tangential acceleration is zero. This is the geodesic equation.

## 5. RFM velocity loss

Differentiating SLERP gives the exact tangent target velocity:

```math
u_t=\frac{\Omega}{\sin\Omega}
[-\cos((1-t)\Omega)a+\cos(t\Omega)b].
```

The neural output is projected onto the tangent space:

```math
\Pi_z(v)=v-(z^\top v)z.
```

SRUL trains

```math
\mathcal L_{\mathrm{RFM}}
=\mathbb E\|\Pi_{z_t}v_\theta-u_t\|_2^2.
```

## 6. Exponential-map sampling

An ordinary Euler step leaves the sphere even for a tangent velocity:

```math
\|z+\Delta t v\|_2^2=1+\Delta t^2\|v\|_2^2>1.
```

The exponential map performs a spherical rotation:

```math
\operatorname{Exp}_z(\Delta t v)
=\cos(\Delta t\|v\|_2)z
+\sin(\Delta t\|v\|_2)\frac{v}{\|v\|_2}.
```

The two directions are orthogonal, so the new token remains unit norm. Sampling repeatedly predicts a velocity, optionally applies CFG, projects to the tangent space, and takes an exponential-map step.

## 7. Classifier-free guidance and Bayes rule

Bayes rule gives the score identity

```math
\nabla_z\log p_t(z\mid y)
=\nabla_z\log p_t(z)+\nabla_z\log p_t(y\mid z).
```

Classical classifier guidance estimates the label term with a separate classifier. CFG instead trains one generator both with labels and with a null label. In the SRUL velocity parameterization, sampling uses

```math
v_{\mathrm{CFG}}=v_\varnothing+s(v_y-v_\varnothing).
```

For `s=1`, this is the ordinary conditional field. For larger `s`, the label-specific difference is amplified. Samples usually become more class-typical, which can improve FID and precision, but probability becomes concentrated on fewer modes and recall can fall. This is an empirical trade-off, not a theorem.
