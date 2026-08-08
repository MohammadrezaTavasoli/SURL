# SRUL - 5-minute speaker script

## Slide 1 - Motivation and main idea (about 55 seconds)

Latent generative models often start from a Gaussian prior, but an autoencoder does not have to produce Gaussian latents. Its architecture and normalization decide the latent geometry. For example, normalized Transformer features can have nearly constant norm, and we can create the same structure explicitly by normalizing each spatial token.

This leads to the main question of the project: does the prior match the support learned by the autoencoder? A mismatch can happen because the prior uses the wrong support, or because its path leaves the support between the source and target. SRUL addresses both points by learning spherical spatial tokens and an RFM prior on the same product of spheres.

## Slide 2 - SRUL pipeline (about 70 seconds)

SRUL has two training stages.

First, the encoder produces a spatial token grid. Each token is normalized independently, so the latent lies on a product of spheres. We add controlled tangent noise and train the decoder from clean and perturbed tokens using reconstruction, consistency, edge, and perceptual losses.

Second, RFM learns the distribution of the encoded token grids. The source is spherical noise, SLERP gives the geodesic path, and the network predicts a tangent velocity. During sampling, each velocity is projected to the tangent space and integrated with an exponential-map step. The final token grid is decoded into an image. Class conditioning and classifier-free guidance are optional extensions.

## Slide 3 - Why the autoencoder and prior must match (about 90 seconds)

The first test isolates path mismatch. We use the same spherical autoencoder and change only the transport path. The straight chord enters the sphere and reaches a minimum token norm of 0.705. SRUL follows the geodesic, stays at norm one, and improves FID from 67.44 to 63.84. Precision and recall also improve.

The second test studies support mismatch. SRUL reaches FID 49.42. We then keep the same trained VAE and compare two priors. Standard FM models the full VAE latent and reaches FID 50.70. Projected RFM keeps only token directions, restores a mean radius, and reaches FID 56.04.

The VAE token radii are not constant, so a direction-only projection discards part of the representation. SRUL avoids this post-hoc mismatch by learning spherical support before its RFM prior is trained.

## Slide 4 - Tangent noise and guidance (about 45 seconds)

These two controls affect different parts of the system.

The tangent-noise scale changes the encoded representation. In our sweep, 0.05 gives the best reconstruction and generation FID. Larger noise removes too much image information.

Classifier-free guidance changes conditional sampling. Increasing the guidance scale from one to two improves FID from 62.44 to 49.39 and increases precision, while recall decreases. This is the expected fidelity-diversity trade-off.

## Slide 5 - Three evaluation settings and conclusion (about 40 seconds)

CIFAR-10 is used for the controlled geometry, support-matching, noise, and guidance studies. PathMNIST tests transfer to histopathology textures using the same four-by-four token grid. CelebA-64 increases the image size to 64 by 64 and the latent grid to eight by eight.

The main result is that latent support and prior geometry should be designed together. SRUL learns spherical spatial tokens and models that same support with geodesic RFM dynamics.
