import os
import numpy as np
import torch
import torch.nn.functional as f

# --------------------------------------------------------------------------------
# DATA SET OPERATION

# Read text file and output dataset tensor and normal_vectors tensor
def read_txt2(filename, device):
  onsurface_points = np.zeros((0,2))
  shifted_points = np.zeros((0,2)) # onsurface_points left shifted by 1
  first_point = np.zeros((1,2))
  last_point = np.zeros((1,2))

  with open(filename, 'r') as f:
    for c in range(int(f.readline())):
      num_of_verticles = int(f.readline())

      first_point[0] = np.loadtxt(f, max_rows=1)
      middle_points = np.loadtxt(f, max_rows=num_of_verticles-2)
      last_point[0] = np.loadtxt(f, max_rows=1)

      # Onsuface points order: first_point,middle_points,last_point
      # Shifted points order:  middle_points,last_point,first_point
      onsurface_points = np.concatenate((onsurface_points, first_point))    # Onsurface: first_point
      onsurface_points = np.concatenate((onsurface_points, middle_points))  # Onsurface: middle_points
      shifted_points = np.concatenate((shifted_points, middle_points))      # Shifted: middle_points
      # Remove the last point if it is the same as the first point
      if np.not_equal(last_point,first_point).any():
        onsurface_points = np.concatenate((onsurface_points,last_point))    # Onsurface: last_point
        shifted_points = np.concatenate((shifted_points,last_point))        # Shifted: last_point
      shifted_points = np.concatenate((shifted_points,first_point))         # Shifted: first_point

  # Vector of 2 consecutive points
  vectors = shifted_points - onsurface_points
  # Getting normal vectors
  norm = np.linalg.norm(vectors, axis=1)
  normal_vectors = np.ones_like(vectors)

  normal_vectors[:,0] = np.divide(-vectors[:,1],norm)
  normal_vectors[:,1] = np.divide(vectors[:,0],norm)

  d = torch.from_numpy(onsurface_points).float().to(device)
  d.requires_grad = True
  n = torch.from_numpy(normal_vectors).float().to(device)
  n.requires_grad = True

  return d,n

# Sample points on a circle
def circle_dataset(device):
	# Points
  num_on_points = 100
  num_points = 3 * num_on_points
  radius = 1.0
  thetas = np.arange(0.0, 2.0*np.pi, 2.0*np.pi/float(num_on_points))
  d = np.zeros((num_on_points,2))
  d[:,0] = radius*np.cos(thetas)
  d[:,1] = radius*np.sin(thetas)
  d = torch.from_numpy(d).float().to(device)
  d.requires_grad = True

  # Normal vectors
  d_shifted = torch.roll(d, 1, 0)
  n = d_shifted - d
  n = f.normalize(n, p=2, dim=1)

  return d,n