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
import matplotlib.pyplot as plt


def train(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  print('Setting up')

  # Define parameters
  geo_coeff = 30.
  # normal_coeff = 10.
  constraint_coeff = 1.
  off_coeff = 0.5

  if data.shape[1] == 3 or data.shape[1] == 2:
    dim = 1
    loss_checkpoint_freq = 100
    plot_freq = 500
  elif data.shape[1] == 5 or data.shape[1] == 4:
    dim = 2
    loss_checkpoint_freq = 100
    plot_freq = 500
  else:
    dim = 3
    loss_checkpoint_freq = 10

  if batch_size is None:
    batch_size = data.shape[0]

  sample_ranges = Utils.get_sample_ranges(data[:, 0:dim])

  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)



  # Train network
  print()
  print('Training')
  for i in range(start, start+num_epochs):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:

      point_batch = batch[:,0:dim]
      normal_batch = batch[:,dim:2*dim]
      stddvt_batch = batch[:,2*dim].view(batch.shape[0],1)

      # Distribution
      distribution = Utils.uniform_gaussian(point_batch, sample_ranges, stddvt_batch)
      # distribution = Utils.uniform(sample_ranges, dim=dim, device=device)
      # distribution = Utils.gaussian(point_batch, stddvt_batch)
      # distribution = torch.cat((distribution, point_batch))

      model.train()

      optimizer.zero_grad()

      # Compute loss
      # TotalLoss = f(x) + (grad(f)-normals) + constraint

      # Forward pass
      f_data = model(point_batch)
      f_dist = model(distribution)

      # f(x)
      geo_loss = f_data.abs().mean()

      # grad(f)-normals
      # normal_grad = Utils.compute_grad(f_data, point_batch)
      # normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      # normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      normal_loss = torch.tensor([0]).to(device)

      # Laplacian as constraint
      constraint_grad = Utils.compute_laplacian(f_dist, distribution, p=10)
      constraint = ((constraint_grad - 1).abs()).mean()
      
      # Off surface
      alpha = 100.0
      off_surface = torch.exp(-alpha * f_dist.abs()).mean()
      
      # loss = geo_coeff * geo_loss + constraint_coeff * constraint + off_coeff * off_surface
      # loss = geo_coeff * geo_loss + normal_coeff * normal_loss + constraint_coeff * constraint + off_coeff * off_surface
      # loss = geo_coeff * geo_loss + normal_coeff * normal_loss + constraint_coeff * constraint
      loss = geo_coeff * geo_loss + constraint_coeff * constraint + off_coeff * off_surface
      
      loss.backward()
      optimizer.step()
      scheduler.step()



    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item(), off_surface.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())
      print('Off-surface constraint: ', off_surface.item())
      print()

    if dim==2 and (i+1) % plot_freq == 0:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                surface=False,
                                additional=False,
                                device=device)

  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1:])

  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)

  model.eval()
    
  return model, optimizer, scheduler



def train_mesh(num_epochs, model, optimizer, scheduler, data, triangles, batch_size=None, loss_output_path=None, device='cpu'):
  print('Setting up')

  # Define parameters
  geo_coeff = 100.
  normal_coeff = 0.
  constraint_coeff = 1.
  off_coeff = 1.

  if data.shape[1] == 2:
    dim = 1
    loss_checkpoint_freq = 100
    plot_freq = 500
  elif data.shape[1] == 3:
    dim = 2
    loss_checkpoint_freq = 100
    plot_freq = 500
  else:
    dim = 3
    loss_checkpoint_freq = 10
    plot_freq = None

  if batch_size is None:
    batch_size = data.shape[0]

  # Get loss values and number of iteration in last training
  loss_value, start = Utils.load_loss_values(loss_output_path)

  sample_ranges = Utils.get_sample_ranges(data[:, 0:2])



  # Train network
  print()
  print('Training')
  for i in range(start, start+num_epochs):

    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for batch in batches:
      
      is_boundary = batch[:,-1]==1
      points = batch[:,0:2]
      point_batch = points[is_boundary]
      distribution = points#[~is_boundary]

      model.train()
      optimizer.zero_grad()

      # Compute loss
      # TotalLoss = f(x) + (grad(f)-normals) + constraint

      # Forward pass
      f_data = model(point_batch)
      f_dist = model(distribution)

      # f(x)
      geo_loss = f_data.abs().mean()
      normal_loss = torch.tensor([0]).to(device)

      # Laplacian as constraint
      constraint_grad = Utils.compute_laplacian(f_dist, distribution, p=2)
      constraint = ((constraint_grad - 1).abs()).mean()

      # Off surface
      alpha = 100.0
      off_loss = torch.exp(-alpha * f_dist.abs()).mean()

      # loss = geo_coeff * geo_loss + constraint_coeff * constraint
      # loss = geo_coeff * geo_loss + off_coeff * off_loss
      loss = geo_coeff * geo_loss + constraint_coeff * constraint + off_coeff * off_loss
      
      loss.backward()
      optimizer.step()
      scheduler.step()



    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())
      print('Off-surface constraint: ', off_loss.item())
      print()

    if dim==2 and (i+1) % plot_freq == 0:
      Visualization.trimesh_visualize2(model, data, triangles,
                                      scatter=False, vecfield=False, surface=True, filled_contour=True, 
                                      additional=False, func_eval=False,
                                      device=device)


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])

  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)

  model.eval()
    
  return model, optimizer, scheduler



def train_boundary(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  # Define parameters
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



  # Training
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

      # grad(f)-normals
      normal_grad = Utils.compute_grad(f_data, point_batch)
      normal_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()
      # normal_loss = (1 - (F.cosine_similarity(normal_grad, normal_batch, dim=-1)).abs()).mean()
      # normal_loss = torch.tensor([0]).to(device)

      constraint = torch.tensor([0]).to(device)
      
      # Off surface
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
      print('Off-surface constraint: ', off_surface.item())
      
      print()

    if (i+1) % plot_freq == 0 and dim==2:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                additional=False, func_eval=False,
                                device=device)


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)


  model.eval()
    

  return model, optimizer, scheduler

def train_constraint(num_epochs, model, optimizer, scheduler, data, batch_size=None, loss_output_path=None, device='cpu'):
  print('Setting up')

  # Define parameters
  off_coeff = 0.
  loss_checkpoint_freq = 100
  plot_freq = 500

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

  # Train network
  print()
  print('Training')
  for i in range(start, start+num_epochs):

    distribution = Utils.uniform(batch_size, sample_ranges, dim=dim, device=device)

    # Change to train mode
    model.train()
    
    optimizer.zero_grad()

    # Compute loss
    # TotalLoss = f(x) + (grad(f)-normals) + constraint

    # Forward pass
    f_dist = model(distribution)

    # f(x)
    geo_loss = torch.tensor([0])

    # grad(f)-normals
    normal_loss = torch.tensor([0])

    # Laplacian as constraint
    constraint_grad = Utils.compute_laplacian(f_dist, distribution, p=2)
    constraint = ((constraint_grad - 1).abs()).mean()
        
    # Off surface
    alpha = 100.0
    off_surface = torch.exp(-alpha * f_dist.abs()).mean()

    loss = constraint + off_coeff * off_surface
    
    loss.backward()
    optimizer.step()
    scheduler.step()



    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i, loss.item(), geo_loss.item(), normal_loss.item(), constraint.item(), off_surface.item()]], axis=0)
      print('Epoch:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', normal_loss.item(), '  Constraint:', constraint.item())
      print('Off-surface constraint: ', off_surface.item())      
      print()

    if (i+1) % plot_freq == 0:
      Visualization.visualize2(model, data,
                                scatter=False, vecfield=False,
                                additional=False, func_eval=False,
                                device=device)

  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])

  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)

  model.eval()
    
  return model, optimizer, scheduler, loss_value