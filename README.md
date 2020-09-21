# Graduation Thesis
Given a 3D point-cloud, reconstruct a surface that interpolates or approximates the 3D points.

The reconstructed surface is defined implicitly in the form {(x,y,z) \in R^3: f(x,y,z) = 0}.

## List of model

### No constrain

Trained without any constrain

Syntax: filename_noconstrain.traineddata (e.g. circle_noconstrain.trainneddata, riderr_noconstrain.traineddata)

### Uniform

Trained with uniform distribution

Syntax: filename_uniform.traineddata

### Uniform - Gaussian

Trained with the average of a uniform distribution and a sum of Gaussians centered at X with standard deviation equal to the distance to the k-th nearest neighbor(k = 50) as the distribution.

Syntax: filename_uniform_distribution.traineddata
