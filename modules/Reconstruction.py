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
  normal_coeff = 10.
  constraint_coeff = 1.
  # off_mag = 0.25
  off_coeff = 0.5
  

  loss_checkpoint_freq = 100
  plot_freq = 500
  # coeff_adjust_freq = 2500

  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)
  # constraint_coeff = min(0.1 * (start//coeff_adjust_freq + 1), 10.)

  if data.shape[1] == 3 or data.shape[1] == 2:
    dim = 1
  elif data.shape[1] == 5 or data.shape[1] == 4:
    dim = 2
  else:
    dim = 3
  

  print('Getting sampling range')
  sample_ranges = Utils.get_sample_ranges(data[:,0:3])
  

  # print('Getting off surface points')
  # points_in, points_out = Utils.get_off_surface_points(data[:,0:dim], data[:,dim:2*dim], off_mag)
  # data = torch.cat((data, points_in, points_out), dim=-1)
  # print('Inside: ', str(torch.count_nonzero(points_in).item()), ' points. Outside: ', str(torch.count_nonzero(points_out).item()), ' points.')


   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = data.shape[0]


  # # Get dstandart deviation for Gaussian
  # full_distribution = Utils.uniform_far(data[:, 0:dim].detach().cpu().numpy(), batch_size, sample_ranges, dim=dim, device=device)
  # Visualization.scatter_plot(full_distribution.detach().cpu().numpy())

  # Train network
  print()
  print('Training')
  for i in range(start, start+num_epochs):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:

      point_batch = batch[:,0:dim]
      normal_batch = batch[:,dim:2*dim]
      # Gaussian
      stddvt_batch = batch[:,2*dim].view(batch.shape[0],1)

      # Distribution

      # Uniform-Gaussian
      # distribution = Utils.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt_batch)
      # Uniform
      distribution = Utils.uniform(batch_size, sample_ranges, dim=dim, device=device)
      # Uniform_far
      # indices = random.sample(range(full_distribution.shape[0]), batch_size*2)
      # distribution = full_distribution[indices]
      # Gaussian
      # distribution = Utils.gaussian(point_batch, stddvt_batch)

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
      normal_grad = Utils.compute_grad(f_data, point_batch)
      # normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      # normal_loss = torch.tensor([0]).to(device)

      # Constraint
      #Eikonal
      # constraint_grad = Utils.compute_grad(f_dist, distribution)
      # constraint = ((constraint_grad.norm(2, dim=-1) - 1) ** 2).mean()
      # all_grad = torch.cat((normal_grad, constraint_grad), dim=0)
      # constraint = ((all_grad.norm(2, dim=-1) - 1) ** 2).mean()

      # Laplacian
      constraint_grad = Utils.compute_laplacian(f_dist, distribution, p=2)
      constraint = ((constraint_grad - 1).abs()).mean()

      # Variational
      # constraint_grad = Utils.compute_grad(distribution, f_dist)
      # constraint = 0.5 * (constraint_grad.norm(2, dim=-1))**2 - f_dist.abs()
      # constraint = constraint.mean().abs()
      
      # exp
      alpha = 100.0
      off_surface = torch.exp(-alpha * f_dist.abs()).mean()
      # off_surface = torch.tensor([0])
      #
      # in_i = batch[:,-2]==1
      # points_in = point_batch[in_i] - normal_batch[in_i]
      # off_loss_in = (model(points_in) + off_mag).abs().mean()
      # out_i = batch[:,-1]==1
      # points_out = point_batch[out_i] + normal_batch[out_i]
      # off_loss_out = (model(points_out) - off_mag).abs().mean()
      # off_loss = off_loss_in + off_loss_out

      
      # loss = geo_coeff * geo_loss + constraint_coeff * constraint + off_coeff * off_surface
      loss = geo_coeff * geo_loss + normal_coeff * normal_loss + constraint_coeff * constraint + off_coeff * off_surface
      # loss = geo_coeff * geo_loss + normal_coeff * normal_loss + constraint_coeff * constraint + off_coeff * off_loss

      
      loss.backward()
      optimizer.step()
      scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())

      # print('Constraints coefficient: ', str(constraint_coeff))
      
      print('Off-surface constraint: ', off_surface.item())
      
      print()

    if (i+1) % plot_freq == 0:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                tucker_normalized=False, func_eval=False,
                                device=device)

    # if i % coeff_adjust_freq == 0:
    #   constraint_coeff = min(constraint_coeff * 2, 10.)


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)


  model.eval()
    

  return model, optimizer, scheduler

def train_boundary(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  geo_coeff = 1.
  normal_coeff = 1.
  off_coeff = 0.1

  loss_checkpoint_freq = 100
  plot_freq = 500

  loss_value, start = Utils.load_loss_values(loss_output_path)

  if data.shape[1] == 3 or data.shape[1] == 2:
    dim = 1
  elif data.shape[1] == 5 or data.shape[1] == 4:
    dim = 2
  else:
    dim = 3

  sample_ranges = Utils.get_sample_ranges(data[:,0:3])

  if batch_size is None:
    batch_size = data.shape[0]

  print()
  print('Training')
  for i in range(start, start+num_epochs):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:
      point_batch = batch[:,0:dim]
      normal_batch = batch[:,dim:2*dim]

      distribution = Utils.uniform(batch_size, sample_ranges, dim=dim, device=device)

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
      normal_grad = Utils.compute_grad(f_data, point_batch)
      # normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      # normal_loss = torch.tensor([0]).to(device)


      constraint = torch.tensor([0]).to(device)

      
      # exp
      alpha = 100.0
      off_surface = torch.exp(-alpha * f_dist.abs()).mean()

      
      # loss = geo_coeff * geo_loss + normal_coeff * normal_loss
      loss = geo_coeff * geo_loss + normal_coeff * normal_loss + off_coeff * off_surface
      
      loss.backward()
      optimizer.step()
      scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())

      # print('Constraints coefficient: ', str(constraint_coeff))
      
      # print('Off-surface constraint: ', off_loss.item())
      
      print()

    if (i+1) % plot_freq == 0:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                tucker_normalized=False, func_eval=False,
                                device=device)

    # if i % coeff_adjust_freq == 0:
    #   constraint_coeff = min(constraint_coeff * 2, 10.)


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)


  model.eval()
    

  return model, optimizer, scheduler

def train_constraint(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  print('Setting up')

  # off_coeff = 0.5
  

  loss_checkpoint_freq = 100
  plot_freq = 500
  # coeff_adjust_freq = 2500

  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)
  # constraint_coeff = min(0.1 * (start//coeff_adjust_freq + 1), 10.)

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
    print(batch_size)


  # # Get dstandart deviation for Gaussian
  # full_distribution = Utils.uniform_far(data[:, 0:dim].detach().cpu().numpy(), batch_size, sample_ranges, dim=dim, device=device)
  # Visualization.scatter_plot(full_distribution.detach().cpu().numpy())

  # Train network
  print()
  print('Training')
  for i in range(start, start+num_epochs):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:

      point_batch = batch[:,0:dim]
      # normal_batch = batch[:,dim:2*dim]
      # Gaussian
      # stddvt_batch = batch[:,2*dim].view(batch.shape[0],1)

      # Distribution

      # Uniform-Gaussian
      # distribution = Utils.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt_batch)
      # Uniform
      distribution = Utils.uniform(batch_size, sample_ranges, dim=dim, device=device)
      # Uniform_far
      # indices = random.sample(range(full_distribution.shape[0]), batch_size*2)
      # distribution = full_distribution[indices]
      # Gaussian
      # distribution = Utils.gaussian(point_batch, stddvt_batch)

      distribution = torch.cat((distribution, point_batch))
      # print(distribution.shape)

      # Change to train mode
      model.train()
      

      optimizer.zero_grad()

      # Compute loss
      # TotalLoss = f(x) + (grad(f)-normals) + constraint

      # Forward pass
      # f_data = model(point_batch)
      f_dist = model(distribution)

      # f(x)
      # geo_loss = f_data.abs().mean()
      # geo_loss = (f_data**2).mean()
      geo_loss = torch.tensor([0])

      # grad(f)-normals
      # normal_grad = Utils.compute_grad(f_data, point_batch)
      # normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      # normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      normal_loss = torch.tensor([0])

      # Constraint
      #Eikonal
      # constraint_grad = Utils.compute_grad(f_dist, distribution)
      # constraint = ((constraint_grad.norm(2, dim=-1) - 1) ** 2).mean()
      # all_grad = torch.cat((normal_grad, constraint_grad), dim=0)
      # constraint = ((all_grad.norm(2, dim=-1) - 1) ** 2).mean()

      # Laplacian
      constraint_grad = Utils.compute_laplacian(f_dist, distribution, p=2)
      constraint = ((constraint_grad - 1).abs()).mean()

      # Variational
      # constraint_grad = Utils.compute_grad(distribution, f_dist)
      # constraint = 0.5 * (constraint_grad.norm(2, dim=-1))**2 - f_dist.abs()
      # constraint = constraint.mean().abs()
      
      # exp
      # alpha = 100.0
      # off_surface = torch.exp(-alpha * f_dist.abs()).mean()

      loss = constraint

      
      loss.backward()
      optimizer.step()
      scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())
      
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
    

  return model, optimizer, scheduler, loss_value