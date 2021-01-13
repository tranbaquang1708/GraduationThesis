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
from modules import Distance, Visualization, Operation


def train(num_iters, model, optimizer, scheduler, p=2, batch_size=None, data_file=None, points=None, normal_vectors=None, loss_output_path=None, device='cpu'):
  if data_file is not None:
    model = train_file(num_iters, model, optimizer, batch_size, data_file, loss_output_path, device)
  if points is not None:
    model, optimizer, scheduler = train_data(num_iters, model, optimizer, scheduler, p, batch_size, points, normal_vectors, loss_output_path, device)

  return model, optimizer, scheduler


# Train with data passed
def train_data(num_iters, model, optimizer, scheduler, p, batch_size, points, normal_vectors, loss_output_path=None, device='cpu'):
  print('Setting up')

  lb = 1.0
  tau = 0.5

  loss = 0
  loss_checkpoint_freq = 100
  plot_freq = 100

  # lr_adjust_interval = 2000
  # lr_initial = 0.005
  # lr_factor = 0.5
  # lr_min = 5.0e-6

  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(loss_output_path)

  # Standard deviation for Gaussian
  print('Getting distance to 50th closest neighbor')
  stddvt = Operation.find_kth_closest_d(points, 50)
  

  print('Getting sampling range')
  sample_ranges = Operation.get_sample_ranges(points)
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = points.shape[0]


  # Train network
  print()
  print('Training')
  for i in range(start, start+num_iters):

    # Seperate data into batches
    # index_batches = list(Operation.chunks(random.sample(range(points.shape[0]), points.shape[0]), batch_size))
    # for index_batch in index_batches:
    index_batch = np.random.choice(points.shape[0], batch_size, False)
    point_batch = points[index_batch]
    # normal_batch = normal_vectors[index_batch]


    # Distribution
    # distribution = Operation.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt[index_batch])
    distribution = Operation.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt[index_batch])
    # Visualization.scatter_plot(distribution.detach().cpu().numpy())

    # Change to train mode
    model.train()

    
    optimizer.zero_grad()

    
    # Compute loss
    # TotalLoss = f(x) + (grad(f)-normals) + constraint

    # Forward pass
    f_data = model(point_batch)
    # f_sample = model(distribution)

    # f(x)
    # geo_loss = model(points[index_batch]).abs().mean()
    geo_loss = (f_data.abs()).mean()

    # print(f_data-model(point_batch))

    # grad(f)-normals
    normal_grad = Operation.compute_grad(point_batch, f_data)
    grad_loss = ((normal_grad - normal_vectors[index_batch]).abs()).norm(2, dim=1).mean()

    # Constraint
    constraint_grad = Operation.laplacian(model, distribution, p=p)
    constraint = ((constraint_grad - 1.)**2).mean()

    loss = geo_loss + lb * grad_loss + tau * constraint
    

    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item(), geo_loss.item(), grad_loss.item(), tau * constraint.item()]], axis=0)
      # print((constraint_grad < 0).sum())
      print('Iteration:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', grad_loss.item(), '  Constraint:', constraint.item())
      print()


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)
    
  # Change to eval mode
  # model.eval()

  return model, optimizer, scheduler
  

# Train with file path passed
def train_file(num_iters, model, loss_function, batch_size, data_file, output_path=None, device='cpu'):
  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(output_path)
  
  # Get total number of lines
  num_of_lines = Operation.get_num_of_lines(data_file)
  
  # Train model
  for i in range(start, start+num_iters):
    # points, normal_vectors = Operation.read_txt3_to_batch(data_file, batch_size, num_of_lines, device)

    result =  model(points)
    loss = loss_function.irg_loss(model, result, points, normal_vectors, device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # Adjust learning rate
    if (i) % lr_adjust_interval == 0:
      factor = 0.0625
      for i_param, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = np.maximum(param_group["lr"] * factor, 5.0e-6)

    # Print loss value
    if (i+1)%500 == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item()]], axis=0)
      print("Step " + str(i+1) + ":")
      print(loss)

  if num_iters == 1:
    print(loss)

  # Plot line graph of loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,:-4])

  # Save loss value
  if output_path is not None:
    np.save(output_path, loss_value)
    
  return model


def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  Visualization.loss_graph(loss_value[:,0], loss_value[:,-4:])
