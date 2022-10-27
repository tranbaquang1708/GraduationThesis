# Unsupervised Learning of Signed Distance Functions from 3D Point-Clouds

Approximating the signed distance function from point-clouds by solving the p-Poisson problem via a deep neural network.

The [dissertation](https://github.com/tranbaquang1708/GraduationThesis/blob/master/shape_reconstruction_dissertation.pdf) is included in this repository.

## Reproducing experiments

You can try reproducing the experiment in 1D directly with [example_1d.ipynb](https://github.com/tranbaquang1708/GraduationThesis/blob/15540ceda87f5199f384cf4f58da1d0d8e96c453/example_1d.ipynb).

For 2D and 3D experiments, the training datasets need to be placed in `dataset/2d` or `dataset/3d`, then, the experiments can be reproduced by running the notebook `train2d.ipynb` or `train3d.ipynb`. Before you start the training, please change the path to the training dataset as well as the path to save the training results (the loss values and the trained model). We provided some simple datasets for both 2D and 3D in the branch [dataset3d](https://github.com/tranbaquang1708/GraduationThesis/tree/dataset3d).

## Acknowledgments

We would like to express our gratitude to the Stanford Computer Graphics Laboratory for their generosity in distributing their 3D models.
