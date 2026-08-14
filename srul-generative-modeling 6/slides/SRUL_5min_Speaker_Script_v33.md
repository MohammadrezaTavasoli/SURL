# Learning Spatial Unit-Norm Latents with Riemannian Flow Matching - 5-minute speaker script

## Slide 1 - Motivation and question (about 45 seconds)

Latent generative models often start from a Gaussian prior, but an autoencoder does not have to produce Gaussian latents. Architecture and normalization can make token norms nearly constant, so much of the information is carried by direction. SRUL studies the design rule that the prior should match the support learned by the autoencoder.

Our encoder produces a spatial grid of unit-norm tokens. Tangent perturbations control retained information and encourage the decoder to work over broader spherical neighborhoods. We then train Riemannian Flow Matching on the same product-of-spheres support. The experiments use CIFAR-10, PathMNIST, and CelebA-64.

## Slide 2 - Pipeline and architecture (about 85 seconds)

For CIFAR-10, the CNN encoder maps a three-channel 32-by-32 image through channel widths 96, 192, and 384 to a 32-channel, 4-by-4 latent grid. Every 32-dimensional token is normalized independently, so the complete latent is a product of 16 spheres. The CNN decoder reverses the channel progression and upsamples the grid back to the image.

During autoencoder training, small tangent rotations act as the information-control mechanism. A larger rotation along the same direction is also decoded consistently, which encourages broader and approximately uniform coverage of each token sphere. The grid can still keep dependencies between tokens.

The prior is a class-conditioned convolutional residual network. It projects 32 channels to width 256, uses six conditioned residual blocks with 128-dimensional time and class embeddings, and returns a 32-channel velocity field. RFM trains this network on token-wise SLERP paths. The loss matches its tangent-projected prediction to the exact geodesic velocity.

At sampling time, we start from spherical noise, use 100 exponential-map updates, and decode the final latent grid. The exponential map keeps every token at unit norm.

## Slide 3 - Path and support matching (about 80 seconds)

The first controlled test keeps the spherical autoencoder fixed and changes only the path. A Euclidean chord goes through the sphere interior and reaches a minimum mean norm of 0.705. Geodesic RFM stays at norm one and improves FID from 67.44 to 63.84, with better precision and recall.

The second test studies support matching. SRUL reaches FID 49.42. We keep the same trained VAE encoder and decoder and compare two priors. Standard linear Flow Matching models the complete VAE latent and reaches 50.70. Projected RFM keeps only directions, restores a mean radius, and reaches 56.04. The result supports learning spherical support before applying a direction-only spherical prior.

## Slide 4 - Two controlled ablations (about 60 seconds)

These are two separate experiments, not one joint sweep.

For the information-control study, classifier-free guidance is fixed at two while only the tangent-noise scale changes. On CIFAR-10, increasing the scale from 0.05 to 0.30 worsens generation FID from 48.71 to 58.16 and also worsens reconstruction. The same fixed-guidance slice shows that 0.30 is too destructive on PathMNIST and CelebA-64 as well.

For the guidance study, the tangent-noise scale is fixed at 0.15 while only the CFG scale changes. CIFAR-10 benefits strongly from larger guidance. PathMNIST reaches its best FID around 1.5, and CelebA changes little. In general, stronger guidance tends to reduce recall because it concentrates samples on more class-typical regions.

## Slide 5 - Transfer across image settings (about 40 seconds)

To show transfer without mixing the two ablations, every row uses the same reference setting: tangent-noise scale 0.15 and CFG scale two. The same pipeline works on CIFAR-10 natural objects, PathMNIST histopathology textures, and CelebA-64 faces. CelebA uses an 8-by-8 latent grid instead of 4-by-4.

The main conclusion is that the autoencoder support and prior geometry should be designed together. SRUL learns unit-norm spatial tokens, controls retained information through tangent perturbations, and models the same support with geodesic Riemannian Flow Matching.

## Optional Q&A pointers

- The report appendix gives the full SLERP proof, RFM target velocity, exponential-map norm proof, autoencoder-loss explanations, and the Bayes motivation for classifier-free guidance.
- The appendix reports the cross-dataset sweeps as two controlled slices: noise with guidance fixed, and guidance with noise fixed.
- The full 3-by-3 PathMNIST and CelebA-64 grids remain in the repository.
