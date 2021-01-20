# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import random
import sys
import os
from modules import Visualization, Operation


# Train with data passed
def train(num_epoches, model, optimizer, scheduler, p, batch_size, data, loss_output_path=None, device='cpu'):
  print('Setting up')

  lb = 1.
  tau = 0.1

  loss = 0
  loss_checkpoint_freq = 20


  # Getting dimension
  if data.shape[1] == 7:
    dim = 3
  else:
    dim = 2


  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(loss_output_path)
  

  print('Getting sampling range')
  sample_ranges = Operation.get_sample_ranges(data[:,0:dim])
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = data.shape[0]


  # Train network
  print()
  print('Training')
  for i in range(start+1, start+num_epoches+1):

    # Seperate data into mini-batches
    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


    for batch in batches:
      point_batch = batch[:, 0:dim]

      # Distribution
      distribution = Operation.uniform_gaussian(point_batch, sample_ranges, batch[:, -1])
      # Visualization.scatter_plot(distribution.detach().cpu().numpy())


      # Change to train mode
      model.train()

      
      optimizer.zero_grad()

      
      # Compute loss
      # TotalLoss = f(x) + (grad(f)-normals) + constraint

      # Forward pass
      f_data = model(point_batch)

      # f(x)
      geo_loss = (f_data.abs()).mean()

      # grad(f)-normals
      normal_grad = Operation.compute_grad(point_batch, f_data)
      grad_loss = ((normal_grad - batch[:, dim:2*dim]).abs()).norm(2, dim=1).mean()

      # Constraint
      constraint_grad = Operation.laplacian(model, distribution, p=p)
      constraint = ((constraint_grad + 1.)**2).mean()

      loss = geo_loss + lb * grad_loss + tau * constraint
      

      loss.backward(retain_graph=True)
      optimizer.step()
      scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), grad_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', grad_loss.item(), '  Constraint:', constraint.item())
      print()


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)
    

  return model, optimizer, scheduler
