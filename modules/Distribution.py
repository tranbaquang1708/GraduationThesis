import torch
import math
from modules import Distance

# Uniform distribution
def uniform_distribution(points, device):
  xmin = torch.min(points[:,0]).item()
  xmax = torch.max(points[:,0]).item()
  ymin = torch.min(points[:,1]).item()
  ymax = torch.max(points[:,1]).item()

  dist_min = torch.ones(points.size()).to(device)
  dist_min[:,0] = dist_min[:,0] * xmin
  dist_min[:,1] = dist_min[:,1] * ymin

  dist_max = torch.ones(points.size()).to(device)
  dist_max[:,0] = dist_max[:,0] * xmax
  dist_max[:,1] = dist_max[:,1] * ymax

  uniform = torch.distributions.uniform.Uniform(dist_min, dist_max)
  dist = uniform.sample().to(device)

  return dist

def gaussian_kth(points, device):
  # Standart deviation
  k = 50
  d = Distance.pdist_squareform(points, Distance.euclidean_distance)
  d.to(device)
  d_50 = d[:].topk(50, largest=False)
  std = d_50.values[:,-1].reshape((points.size()[0], 1))
  std = torch.cat((std, std), dim=1)
  std.to(device)

  # Mean
  mean = points.mean(dim=0)

  # Gaussian
  g = 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((points - mean) / (2 * std)) ** 2)
  g = g.sum(dim=0)
  g.to(device)

  return g

# The average of a uniform distribution and a sum of Gaussians centered 
# at X with standard deviation equal to the distance to the k-th nearest 
# neighbor (we used k = 50)
def uniform_gaussian_distribution2(points, device):
  u = uniform_distribution(points, device)
  g = gaussian_kth(points, device)

  return (u+g)/2