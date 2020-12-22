import torch
import math
import random
from scipy import spatial
from modules import Visualization

# Uniform distribution
def uniform(points, dist_size, device='cpu'):
  # xmin = torch.min(points[:,0]).item()
  # xmax = torch.max(points[:,0]).item()
  # ymin = torch.min(points[:,1]).item()
  # ymax = torch.max(points[:,1]).item()

  # # dist_min = torch.ones(points.size()).to(device)
  # dist_min = torch.ones((dist_size, points[0].shape[0])).to(device)
  # dist_min[:,0] = dist_min[:,0] * xmin
  # dist_min[:,1] = dist_min[:,1] * ymin

  # # dist_max = torch.ones(points.size()).to(device)
  # dist_max = torch.ones((dist_size, points[0].shape[0])).to(device)
  # dist_max[:,0] = dist_max[:,0] * xmax
  # dist_max[:,1] = dist_max[:,1] * ymax

  # if points[0].shape[0] == 3:
  #   zmin = torch.min(points[:,2]).item()
  #   zmax = torch.max(points[:,2]).item()
  #   dist_min[:,2] = dist_min[:,2] * zmin
  #   dist_max[:,2] = dist_max[:,2] * zmax

  # uniform = torch.distributions.uniform.Uniform(dist_min, dist_max)
  # dist = uniform.sample().to(device)

  global_sigma = 1.8
  dist = (torch.rand(dist_size, points.shape[1]) * 2 - 1) * global_sigma 

  return dist

def dense_uniform(points, dist_size):
  half1 = int(points.shape[0]/2)
  half2 = points.shape[0] - half1

  xmin = torch.min(points[:,0]).item()
  xmax = torch.max(points[:,0]).item()
  ymin = torch.min(points[:,1]).item()
  ymax = torch.max(points[:,1]).item()
  dx = xmax - xmin
  dy = ymax - ymin

  dist_min1 = torch.ones((half1, 2)).to(device)
  dist_min1[:,0] = dist_min1[:,0] * (xmin + 0.25*dx)
  dist_min1[:,1] = dist_min1[:,1] * (ymin + 0.25*dy)

  dist_max1 = torch.ones((half1, 2)).to(device)
  dist_max1[:,0] = dist_max1[:,0] * (xmax - 0.25*dx)
  dist_max1[:,1] = dist_max1[:,1] * (ymax - 0.25*dy)

  dist_min2 = torch.ones((half1, 2)).to(device)
  dist_min2[:,0] = dist_min2[:,0] * xmin
  dist_min2[:,1] = dist_min2[:,1] * ymin

  dist_max2 = torch.ones((half1, 2)).to(device)
  dist_max2[:,0] = dist_max2[:,0] * xmax
  dist_max2[:,1] = dist_max2[:,1] * ymax

  dist_min = torch.cat((dist_min1, dist_min2))
  dist_max = torch.cat((dist_max1, dist_max2))

  uniform = torch.distributions.uniform.Uniform(dist_min, dist_max)
  dist = uniform.sample()

  return dist

def find_kth_closest_distance(points, k):
  tree = spatial.KDTree(points.detach().cpu().numpy())
  d_50 = tree.query(points.detach().cpu().numpy(), k=k)[0][:,-1]
  stddvt = torch.from_numpy(d_50).reshape((points.size()[0], 1))
  
  return stddvt.type(torch.FloatTensor)

def gaussian(points, stddvt):
  dist = points + (torch.randn_like(points, dtype=torch.float32) * stddvt)
  return dist

# The average of a uniform distribution and a sum of Gaussians centered 
# at X with standard deviation equal to the distance to the k-th nearest 
# neighbor (we used k = 50)
def uniform_gaussian(points, dist_size, sample_ranges, stddvt):
  # g = gaussian(points, stddvt)
  # u = uniform(points, dist_size // 8)#.to(g.device)
  u_x = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[0], sample_ranges[1])
  u_y = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[2], sample_ranges[3])
  if points.shape[1] == 2:
    u = torch.cat((u_x, u_y), dim=-1)
  else:
    u_z = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[4], sample_ranges[5])
    u = torch.cat((u_x, u_y, u_z), dim=-1)
  
  u = u.to(points.device)
  # u = (torch.rand(dist_size//8, points.shape[1], device=points.device) * 2 - 1) * sample_ranges

  g = points + (torch.randn_like(points) * stddvt)

  return torch.cat((u,g))