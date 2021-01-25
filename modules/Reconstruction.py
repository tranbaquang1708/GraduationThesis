# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import sys
import os
from modules import Visualization, Utils


def train(num_iters, model, optimizer, scheduler, batch_size, points, normal_vectors, loss_output_path=None, device='cpu'):
  print('Setting up')

  lb = 1.0
  tau = 0.1

  loss_checkpoint_freq = 100


  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)


  # Standard deviation for Gaussian
  print('Getting distance to 50th closest neighbor')
  stddvt = Utils.find_kth_closest_d(points, 50)
  

  print('Getting sampling range')
  sample_ranges = Utils.get_sample_ranges(points)
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = points.shape[0]


  # Train network
  print()
  print('Training')
  for i in range(start+1, start+num_iters+1):

    # Seperate data into batches
    index_batch = np.random.choice(points.shape[0], batch_size, False)
    point_batch = points[index_batch]


    # Distribution
    distribution = Utils.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt[index_batch])


    # Change to train mode
    model.train()
    

    # Compute loss
    # TotalLoss = f(x) + (grad(f)-normals) + constraint

    # Forward pass
    f_data = model(point_batch)

    # f(x)
    geo_loss = (f_data.abs()).mean()

    # grad(f)-normals
    normal_grad = Utils.compute_grad(point_batch, f_data)
    grad_loss = ((normal_grad - normal_vectors[index_batch])).norm(2, dim=-1).mean()

    # Constraint
    constraint_grad = Utils.compute_grad(distribution, model(distribution))
    constraint = ((constraint_grad.norm(2, dim=-1) - 1) ** 2).mean()
    

    loss = geo_loss + lb * grad_loss + tau * constraint
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), grad_loss.item(), constraint.item()]], axis=0)
      print('Iteration:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', grad_loss.item(), '  Constraint:', constraint.item())
      print()


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)


  model.eval()
    

  return model, optimizer, scheduler