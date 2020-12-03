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
from modules import Distance, Visualization, Distribution, Operation


def train(num_iters, model, optimizer, batch_size=None, data_file=None, points=None, normal_vectors=None, loss_output_path=None, device='cpu'):
  if data_file is not None:
    model = train_file(num_iters, model, optimizer, batch_size, data_file, loss_output_path, device)
  if points is not None:
    model = train_data(num_iters, model, optimizer, batch_size, points, normal_vectors, loss_output_path, device)

  return model


# Train with data passed
def train_data(num_iters, model, optimizer, batch_size, points, normal_vectors, loss_output_path=None, device='cpu'):
  print('Setting up')

  lb = 1.0
  tau = 0.1

  loss = 0
  loss_checkpoint_freq = 100
  plot_freq = 100

  lr_adjust_interval = 2000
  lr_initial = 0.001
  lr_factor = 0.25
  lr_min = 1.0e-7

  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(loss_output_path)

  # Standard deviation for Gaussian
  print('Getting distances to 50th neighbor')
  stddvt = Distribution.find_kth_closest_distance(points, 50).to(device)

  print('Getting sample range')
  sample_range = (points.abs().max() + 1) * 1.1

  # Grid for plot result
  if points.shape[1] == 2:
    print('Creating grid')
    xx, yy = Visualization.grid_from_torch(points, resx=50, resy=50, device=device)
  # else:
  #   xx, yy, zz = Visualization.grid_from_torch(points, resx=50, resy=50, resz=50, device=device)
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = points.shape[0]


  # Change to train mode
  # model.train()

  # Train network
  print()
  print('Training')
  for i in range(start, start+num_iters):
    

    # Change to train mode
    model.train()


    # Seperate data into batches
    # index_batches = list(Operation.chunks(random.sample(range(points.shape[0]), points.shape[0]), batch_size))
    # for index_batch in index_batches:
    index_batch = np.random.choice(points.shape[0], batch_size, False)
    point_batch = points[index_batch]


    # Distribution
    distribution = Distribution.uniform_gaussian(point_batch, len(index_batch), sample_range, stddvt[index_batch]).to(device)
    # Visualization.scatter_plot(distribution.detach().cpu().numpy())
    

    # Adjust learning rate
    if (i) % lr_adjust_interval == 0:
      for i_param, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = np.maximum(lr_initial * (lr_factor ** (i // lr_adjust_interval)) , lr_min)

    
    # Compute loss
    # TotalLoss = f(x) + (grad(f)-normals) + constraint

    # f(x)
    # geo_loss = model(points[index_batch]).abs().mean()
    geo_loss = model(point_batch).abs().mean()

    # grad(f)-normals
    g = Operation.compute_grad(point_batch, model)
    grad_loss = lb * (g - normal_vectors[index_batch]).norm(2, dim=1).mean()

    # Constraint
    # Eikonal term: grad(f)-1
    g = Operation.compute_grad(distribution, model)
    constraint = tau * (((g.norm(2, dim=1) - 1))**2).mean()

    loss = geo_loss + grad_loss + constraint
    

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    
    # Plot results
    if (i+1) % plot_freq == 0:
      # model = model.eval()
      # 2d plot
      if (points.shape[1] == 2):
        z = Visualization.nn_sampling(model, xx, yy, device=device)
        try:
          plt.figure(figsize=(12,3))
          plt.subplot(1, 2, 1)
          h = plt.contour(xx.detach().cpu(),yy.detach().cpu(), z.detach().cpu(), levels=[0.0], colors='c')

          plt.subplot(1, 2, 2)
          hf = plt.contourf(xx.detach().cpu(),yy.detach().cpu(), z.detach().cpu())
          plt.show()
        except:
          print('No result')

      # else: # 3D plot
      #   z = Visualization.nn_sampling(model, xx, yy, zz, g_norm_output_path=None, device=device)
      #   try:
      #     verts, faces, normals, values = measure.marching_cubes_lewiner(z.detach().cpu().numpy(), 0)
      #     fig = plt.figure(figsize=(5,5))
      #     ax_surface = fig.add_subplot(111, projection='3d')
      #     mesh = Poly3DCollection(verts[faces])
      #     mesh.set_edgecolor('k')
      #     ax_surface.add_collection3d(mesh)
      #     ax_surface.set_xlim(0, 50)
      #     ax_surface.set_ylim(0, 50)
      #     ax_surface.set_zlim(0, 50)
      #     plt.tight_layout()
      #     plt.show()
      #   except:
          # print('No result')


    # Store loss values into numpy array
    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item()]], axis=0)
      print('Iteration:', i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])
      print('Surface loss:' , geo_loss.item(), '  Normal loss:', grad_loss.item(), '  Constraint:', constraint.item())
      print()


  # Draw figure for loss values
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])


  # Save numpy array of loss values
  if loss_output_path is not None:
    np.save(loss_output_path, loss_value)
    
  # Change to eval mode
  # model.eval()

  return model, optimizer
  

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
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])

  # Save loss value
  if output_path is not None:
    np.save(output_path, loss_value)
    
  return model


def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])
