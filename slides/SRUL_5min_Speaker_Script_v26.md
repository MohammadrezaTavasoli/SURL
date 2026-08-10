# Learning Spatial Unit-Norm Latents with Riemannian Flow Matching - 5-minute speaker script

## Slide 1 - Motivation and question (about 50 seconds)

Latent generative models usually start from a simple prior, often Gaussian. But an autoencoder does not have to produce Gaussian latents. Architecture and normalization can make latent-token norms nearly constant, so the important information is mainly in direction. Transformer features with LayerNorm are one example, and our encoder makes this structure exact by normalizing every spatial token.

This gives the main question of the project: does the prior use the same support and geometry as the autoencoder? A mismatch can happen in two ways. The prior may be defined on a different support, or its transport path may leave the support between its endpoints. SRUL learns unit-norm spatial tokens, uses tangent noise as an information-control knob, and trains an RFM prior on the same product of spheres.

## Slide 2 - SRUL pipeline and key equations (about 85 seconds)

The first stage learns the representation. The encoder produces a spatial feature map, and every token is normalized as

\[
z_{ij}=\frac{h_{ij}}{\|h_{ij}\|_2}.
\]

The decoder is trained from both clean and tangent-perturbed tokens. The tangent-noise scale controls how much image-specific information remains while every token stays on the sphere. Pixel and edge losses preserve local content, LPIPS compares visual features, and latent consistency asks a noisy decode-and-reencode cycle to return near the clean token direction.

The second stage learns the prior. Spherical noise and an encoded latent are connected by SLERP. SLERP is a constant-speed rotation, so every intermediate token remains at unit norm. The RFM loss matches the network's tangent-projected velocity to the exact derivative of this geodesic:

\[
\mathcal L_{\mathrm{RFM}}=
\mathbb E\|\Pi_{z_t}v_\theta-u_t\|_2^2.
\]

At inference, the real endpoint is unknown. We start from spherical noise and repeatedly apply the learned velocity through an exponential-map update. This replaces an ordinary Euler step and keeps every token on its sphere before the final latent grid is decoded.

## Slide 3 - Path and support matching (about 85 seconds)

The first controlled test isolates the path. We keep the same spherical autoencoder and change only transport. The chord enters the sphere and reaches a minimum token norm of 0.705. Geodesic RFM remains at norm one and improves FID from 67.44 to 63.84, with better precision and recall.

The second test studies support matching. SRUL reaches FID 49.42. We then keep the same trained VAE encoder and decoder and change only the prior treatment. Standard linear Flow Matching models the complete VAE latent and reaches 50.70. Projected RFM keeps only directions, restores a mean radius, and reaches 56.04.

This happens because the VAE was not trained on spherical support and its decoder can use radius information. SRUL instead imposes unit-norm support during representation learning, so its decoder and RFM prior are designed for the same space.

## Slide 4 - Information control and classifier-free guidance (about 55 seconds)

These controls act at different stages.

The tangent-noise scale is our explicit information-control knob. It changes the representation through

\[
z_\sigma=\operatorname{Exp}_z(\sigma_{\mathrm{enc}}u).
\]

The knob is a controlled proxy for retained information rather than an exact bitrate in bits. The sweep shows that 0.05 gives the best tested reconstruction and generation FID. Larger perturbations remove too much useful image information.

Classifier-free guidance changes conditional sampling. One network learns both conditional and null-label velocity fields, and sampling uses

\[
v_{\mathrm{CFG}}=v_\varnothing+s(v_y-v_\varnothing).
\]

The difference is the label-specific part of the motion. Increasing the scale improves FID from 62.44 to 49.39 and raises precision, but recall drops because probability becomes more concentrated on class-typical modes.

## Slide 5 - Evaluation settings and conclusion (about 35 seconds)

CIFAR-10 is used for controlled geometry, support-matching, information-control, and guidance studies. PathMNIST checks transfer to texture-dominated histopathology using the same four-by-four token grid. CelebA-64 tests a larger image size and an eight-by-eight token grid.

The main conclusion is simple: the autoencoder support and prior geometry should be designed together. SRUL learns spatial unit-norm tokens, controls retained information with tangent noise, and models that exact support with geodesic Riemannian Flow Matching.

## Optional Q&A pointers

- Detailed derivations of standard Flow Matching, the SLERP geodesic proof, exponential-map norm preservation, autoencoder losses, and the Bayes motivation for CFG are included in the report appendix.
- The work is a controlled geometry study; it is not presented as a state-of-the-art FID claim.
