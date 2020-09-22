# Graduation Thesis
Given a 3D point-cloud, reconstruct a surface that interpolates or approximates the 3D points.

The reconstructed surface is defined implicitly in the form {(x,y,z) \in R^3: f(x,y,z) = 0}.

## List of model

### No constraint

Trained without any constraint

Syntax: filename_noconstraint.traineddata (e.g. circle_noconstrain.trainneddata, riderr_noconstrain.traineddata)

### Uniform

Trained with uniform distribution

Syntax: filename_uniform.traineddata

### Dense Uniform

Traned with uniform distribution in which the sample points are denser in the middle

Syntax: filename_dense_uniform.traineddata

### Gaussian

Trained with Gaussian distribution

Syntax: filename_gaussian.traineddata

### Uniform - Gaussian

Trained with the average of a uniform distribution and a sum of Gaussians centered at X with standard deviation equal to the distance to the k-th nearest neighbor(k = 50) as the distribution.

Syntax: filename_uniform_distribution.traineddata
