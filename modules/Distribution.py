import torch

# Uniform distribution
def uniform_distribution(model, batch_points, device):
  xmin = torch.min(batch_points[:,0]).item()
  xmax = torch.max(batch_points[:,0]).item()
  ymin = torch.min(batch_points[:,1]).item()
  ymax = torch.max(batch_points[:,1]).item()

  dist_min = torch.ones(batch_points.size()).to(device)
  dist_min[:,0] = dist_min[:,0] * xmin
  dist_min[:,1] = dist_min[:,1] * ymin

  dist_max = torch.ones(batch_points.size()).to(device)
  dist_max[:,0] = dist_max[:,0] * xmax
  dist_max[:,1] = dist_max[:,1] * ymax

  uniform = torch.distributions.uniform.Uniform(dist_min, dist_max)
  dist = uniform.sample().to(device)

  return dist