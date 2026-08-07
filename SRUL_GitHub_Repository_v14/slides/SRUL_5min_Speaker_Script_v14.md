# SRUL - 5-minute speaker script

## Slide 1 - Problem and motivation (about 50 seconds)

Autoencoders are often paired with Gaussian priors and Euclidean latent dynamics, but their learned latent geometry depends on the architecture, normalization, and training objective. Normalized encoders can produce nearly fixed-radius features, so much of the information is carried by direction. A Transformer autoencoder with LayerNorm is one important example, not a requirement.

In this setting, a Gaussian source adds radial variation that the encoder does not use, and ordinary linear interpolation forms a chord through the sphere interior. These states are not produced by the spherical encoder. SRUL instead makes the spherical representation exact with token-wise normalization and learns a geodesic RFM prior that remains on the same manifold.

The main question is therefore simple: for spherical tokens, should the prior follow a chord or the geodesic?

## Slide 2 - Complete SRUL pipeline (about 80 seconds)

This slide shows the full pipeline.

In the first stage, the encoder maps an image to a spatial grid. Every token is normalized independently, so the latent lies on a product of spheres. We then apply a controlled tangent perturbation. The decoder is trained with reconstruction, consistency, edge, and perceptual losses, so nearby points on the manifold remain meaningful.

In the second stage, an encoded token grid is paired with spherical source noise. SLERP gives an intermediate geodesic state and its target velocity. The RFM network predicts the velocity, and its output is projected onto the tangent space before the velocity-matching loss is computed.

At sampling time, we start from spherical noise and integrate the trained tangent field with exponential-map steps. This produces a new token grid, which is decoded into an image. Class labels and classifier-free guidance are optional conditional inputs; they do not change the core geometry.

## Slide 3 - Main controlled experiment (about 65 seconds)

This is the main geometry experiment. The two models use the same spherical encoder, decoder, latent size, source distribution, network capacity, and training budget. Only the transport path changes.

For nearly orthogonal endpoints, the midpoint of a chord has norm about one over square root of two, or 0.707. The measured minimum norm is 0.705. SRUL follows the geodesic and keeps every token at norm one.

The geometric correction also improves the image metrics. FID decreases from 67.44 to 63.84, KID decreases, and both precision and recall improve. This supports the focused conclusion that a spherical representation is better matched by on-manifold transport than by a Euclidean chord.

## Slide 4 - Two independent controls (about 55 seconds)

The two ablations control different parts of the system.

The tangent-noise scale changes the encoded representation. In the tested range, increasing sigma-encoder makes both reconstruction and generation worse. The best tested value is 0.05.

Classifier-free guidance changes conditional sampling. Increasing the guidance scale from one to two improves FID from 62.44 to 49.39 and raises precision, while recall decreases. This is a fidelity-diversity trade-off: stronger guidance gives more class-specific samples, but covers less of the distribution.

## Slide 5 - Results across three evaluation settings (about 50 seconds)

This table summarizes the selected SRUL configuration in each setting.

CIFAR-10 is the controlled geometry and sampling study. With the selected tangent-noise and guidance values, reconstruction FID is 17.53 and generation FID is 48.71.

PathMNIST uses the same four-by-four token design on histopathology textures. It reaches generation FID 29.36, precision 0.829, and recall 0.309.

CelebA-64 increases the latent grid to eight-by-eight for sixty-four-pixel faces. It reaches reconstruction FID 9.08 and generation FID 41.22.

Together, these experiments show that the same spatial spherical representation and RFM prior can be used for different visual structures and resolutions.
