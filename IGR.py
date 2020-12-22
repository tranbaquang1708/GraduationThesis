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


def train(num_iters, model, optimizer, scheduler, batch_size=None, data_file=None, points=None, normal_vectors=None, loss_output_path=None, device='cpu'):
  if data_file is not None:
    model = train_file(num_iters, model, optimizer, batch_size, data_file, loss_output_path, device)
  if points is not None:
    model = train_data(num_iters, model, optimizer, scheduler, batch_size, points, normal_vectors, loss_output_path, device)

  return model


# Train with data passed
def train_data(num_iters, model, optimizer, scheduler, batch_size, points, normal_vectors, loss_output_path=None, device='cpu'):
  print('Setting up')

  lb = 1.0
  tau = 0.1

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
  print('Getting distances to 50th neighbor')
  stddvt = Distribution.find_kth_closest_distance(points, 50).to(device)

  print('Getting sampling range')
  # sample_ranges = []
  # sample_ranges.append(points[:,0].min())
  # sample_ranges.append(points[:,0].max())
  # sample_ranges.append(points[:,1].min())
  # sample_ranges.append(points[:,1].max())
  
  xmin = points[:,0].min()
  xmax = points[:,0].max()
  ymin = points[:,1].min()
  ymax = points[:,1].max()
  dx = xmax - xmin
  dy = ymax - ymin
  dz = 0
  
  if points.shape[1] == 3:
    # sample_ranges.append(points[:,2].min())
    # sample_ranges.append(points[:,2].max())
    zmin = points[:,2].min()
    zmax = points[:,2].max()
    dz = zmax - zmin

  ed = 0.1 * torch.sqrt(dx*dx + dy*dy + dz*dz)

  sample_ranges = torch.tensor([xmin-ed, xmax+ed, ymin-ed, ymax+ed, zmin-ed, zmax+ed], device=device) * 1.5


  # Grid for plot result
  if points.shape[1] == 2:
    print('Creating grid')
    xx, yy = Visualization.grid_from_torch(points, resx=50, resy=50, device=device)
  # else:
  #   xx, yy, zz = Visualization.grid_from_torch(points, resx=50, resy=50, resz=50, device=device)
  
   
  # No batch_size value passed, use all data every iteration
  if batch_size is None:
    batch_size = points.shape[0]

  # Get learning rate schedule
  # schedules = []
  # lr_schedules = 

  # Change to train mode
  # model.train()


  # Train network
  print()
  print('Training')
  for i in range(start, start+num_iters):

    # Seperate data into batches
    # index_batches = list(Operation.chunks(random.sample(range(points.shape[0]), points.shape[0]), batch_size))
    # for index_batch in index_batches:
    index_batch = np.random.choice(points.shape[0], batch_size, False)
    point_batch = points[index_batch]
    normal_batch = normal_vectors[index_batch]


    # Distribution
    distribution = Distribution.uniform_gaussian(point_batch, batch_size, sample_ranges, stddvt[index_batch]).to(device)
    # Visualization.scatter_plot(distribution.detach().cpu().numpy())

    # Change to train mode
    model.train()
    

    # Adjust learning rate
    # if (i) % lr_adjust_interval == 0:
    #   for i_param, param_group in enumerate(optimizer.param_groups):
    #     param_group["lr"] = np.maximum(lr_initial * (lr_factor ** (i // lr_adjust_interval)) , lr_min)

    
    optimizer.zero_grad()

    
    # Compute loss
    # TotalLoss = f(x) + (grad(f)-normals) + constraint

    # Forward pass
    f_data = model(point_batch)
    f_sample = model(distribution)

    # f(x)
    # geo_loss = model(points[index_batch]).abs().mean()
    geo_loss = (f_data.abs()).mean()

    # print(f_data-model(point_batch))

    # grad(f)-normals
    normal_grad = Operation.compute_grad(point_batch, f_data)
    grad_loss = ((normal_grad - normal_batch).abs()).norm(2, dim=1).mean()

    # Constraint
    # Eikonal term: grad(f)-1
    constraint_grad = Operation.compute_grad(distribution, f_sample)
    constraint = ((constraint_grad.norm(2, dim=-1) - 1) ** 2).mean()

    loss = geo_loss + lb * grad_loss + tau * constraint
    

    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()


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
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])

  # Save loss value
  if output_path is not None:
    np.save(output_path, loss_value)
    
  return model


def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])
