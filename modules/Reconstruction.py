# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import sys
import os
from scipy.spatial import KDTree
from modules import Visualization, Utils


def train(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  print('Setting up')

  geo_coeff = 10.
  normal_coeff = 1.
  constraint_coeff = 0.1

  loss_checkpoint_freq = 100
  plot_freq = 500

  # Siren
  # off_coeff = 0.2

  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)

  if data.shape[1] == 3 or data.shape[1] == 2:
    dim = 1
  elif data.shape[1] == 5 or data.shape[1] == 4:
    dim = 2
  else:
    dim = 3
  

  print('Getting sampling range')
  sample_ranges = Utils.get_sample_ranges(data[:,0:3])
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = data.shape[0]


  # # Get dstandart deviation for Gaussian
  # stddvt = Utils.find_kth_closest_d(data[:,0:3], k=50)
  full_distribution = Utils.uniform_far(data[:, 0:dim].detach().cpu().numpy(), batch_size, sample_ranges, dim=dim, device=device)
  Visualization.scatter_plot(full_distribution.detach().cpu().numpy())

  # Train network
  print()
  print('Training')
  for i in range(start+1, start+num_epochs+1):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:

      point_batch = batch[:,0:dim]
      normal_batch = batch[:,dim:2*dim]
      # Gaussian
      # stddvt_batch = batch[:,2*dim].view(batch.shape[0],1)

      # Distribution

      # Uniform-Gaussian
      # distribution = Utils.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt_batch)
      # Uniform
      # distribution = Utils.uniform(batch_size, sample_ranges, dim=dim, device=device)
      # Uniform_far
      indices = random.sample(range(full_distribution.shape[0]), batch_size*2)
      distribution = full_distribution[indices]

      # Change to train mode
      model.train()
      

      optimizer.zero_grad()

      # Compute loss
      # TotalLoss = f(x) + (grad(f)-normals) + constraint

      # Forward pass
      f_data = model(point_batch)
      f_dist = model(distribution)

      # f(x)
      geo_loss = f_data.abs().mean()
      # geo_loss = (f_data**2).mean()

      # grad(f)-normals
      normal_grad = Utils.compute_grad(point_batch, f_data)
      normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      # normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      # normal_loss = torch.tensor([0])

      # Constraint
      #Eikonal
      # constraint_grad = Utils.compute_grad(distribution, f_dist)
      # constraint = ((constraint_grad.norm(2, dim=-1) - 1) ** 2).mean()
      # all_grad = torch.cat((normal_grad, constraint_grad), dim=0)
      # constraint = ((all_grad.norm(2, dim=-1) - 1) ** 2).mean()

      # Laplacian
      constraint_grad = Utils.compute_laplacian(distribution, f_dist)
      constraint = ((constraint_grad - 1)**2).mean()

      # Variational
      # constraint_grad = Utils.compute_grad(distribution, f_dist)
      # constraint = 0.5 * (constraint_grad.norm(2, dim=-1))**2 - f_dist.abs()
      # constraint = constraint.mean().abs()
      
      # Siren
      
      # alpha = 100.0
      # off_surface = torch.exp(-alpha * f_dist.abs()).mean()
      off_surface = torch.tensor([0])

      
      # loss = geo_coeff * geo_loss + constraint_coeff * constraint + off_coeff * off_surface

      loss = geo_coeff * geo_loss + normal_coeff * normal_loss + constraint_coeff * constraint #+ off_coeff * off_surface

      
      loss.backward()
      optimizer.step()
      scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())
      
      # Siren
      print('Off-surface constraint: ', off_surface.item())
      
      print()

    if (i+1) % plot_freq == 0:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                tucker_normalized=False, func_eval=False,
                                device=device)


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)


  model.eval()
    

  return model, optimizer, scheduler