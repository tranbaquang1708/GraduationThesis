# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys
import os
from modules import Distance, Visualization, Distribution, Operation

# Neural network model
class IGRPerceptron(nn.Module):
  def __init__(self, dimension):
    # Neural network layers
    super(IGRPerceptron, self).__init__()
    # self.fc0 = nn.Linear(dimension, 512)
    # torch.nn.init.constant_(lin.bias, 0.0)
    # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

    # self.fc1 = nn.Linear(512, 512)
    # self.fc2 = nn.Linear(512, 512)
    # self.fc3 = nn.Linear(512, 512 - dimension) # Skip connection
    # # self.fc3 = nn.Linear(512, 512) # v1.6 No skip connection
    # self.fc4 = nn.Linear(512, 512)
    # self.fc5 = nn.Linear(512, 512)
    # self.fc6 = nn.Linear(512, 512)
    # self.fc7 = nn.Linear(512, 512)
    # self.fc_last = nn.Linear(512, 1)
    # self.activation = nn.Softplus(beta=100)
    d_in = dimension
    dims = [512, 512, 512, 512, 512, 512, 512, 512]
    beta = 100
    skip_in = [4]
    radius_init = 1

    dims = [d_in] + dims + [1]

    self.num_layers = len(dims)
    self.skip_in = skip_in

    for layer in range(0, self.num_layers - 1):

      if layer + 1 in skip_in:
        out_dim = dims[layer + 1] - d_in
      else:
        out_dim = dims[layer + 1]

      lin = nn.Linear(dims[layer], out_dim)

      if layer == self.num_layers - 2:

        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
        torch.nn.init.constant_(lin.bias, -radius_init)
      else:
        torch.nn.init.constant_(lin.bias, 0.0)

        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

      setattr(self, "lin" + str(layer), lin)

    self.activation = nn.Softplus(beta=beta)

  def forward(self, input):
    # out = self.fc0(x)
    # out = self.activation(out)
    # out = self.fc1(out)
    # out = self.activation(out)
    # out = self.fc2(out)
    # out = self.activation(out)
    # out = self.fc3(out)
    # out = self.activation(out)
    # out = torch.cat((out, x), 1) / np.sqrt(2) # v1.7 Skip connection
    # out = self.fc4(out)
    # out = self.activation(out)
    # out = self.fc5(out)
    # out = self.activation(out)
    # out = self.fc6(out)
    # out = self.activation(out)
    # out = self.fc7(out)
    # out = self.activation(out)
    # out = self.fc_last(out)
    # return out
    x = input

    for layer in range(0, self.num_layers - 1):

      lin = getattr(self, "lin" + str(layer))

      if layer in self.skip_in:
        x = torch.cat([x, input], -1) / np.sqrt(2)

      x = lin(x)

      if layer < self.num_layers - 2:
        x = self.activation(x)

    return x

class RBFLossFunction:
  def rbf_loss(self, model, result, batch_points, batch_normal_vectors, device):
    geo_loss = torch.mean(torch.abs(result))

def train(num_iters, model, optimizer, loss_function, batch_size=None, data_file=None, points=None, normal_vectors=None, output_path=None, device='cpu'):
  if data_file is not None:
    model = train_file(num_iters, model, optimizer, loss_function, batch_size, data_file, output_path, device)
  if points is not None:
    model = train_data(num_iters, model, optimizer, loss_function, points, normal_vectors, output_path, device)

  return model

# Train with data passed
def train_data(num_iters, model, optimizer, loss_function, points, normal_vectors, output_path=None, device='cpu'):    
  # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(output_path)

  loss = 0
  lr_adjust_interval = 2000
  loss_checkpoint_freq = 50
  plot_freq = 50

  xx, yy = Visualization.grid_from_torch(points[:,0], points[:,1], resx=50, resy=50, device=device)

  for i in range(start, start+num_iters):
    result =  model(points)
    loss = loss_function.compute_loss(model, result, points, normal_vectors, device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # Visualization.scatter_plot(points.detach())
    
    # Adjust learning rate
    if (i) % lr_adjust_interval == 0:
      factor = 0.5
      for i_param, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = np.maximum(param_group["lr"] * factor, 5.0e-6)


    if (i+1) % loss_checkpoint_freq == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item()]], axis=0)
      print("Epoch:", i+1, '  Loss:', loss.item(), '  Learning rate:', optimizer.param_groups[0]["lr"])

    if (i+1) % plot_freq == 0:
      z = Visualization.nn_sampling(model, xx, yy, device=device)
      h_object = plt.contour(xx.detach().cpu(),yy.detach().cpu(), z.detach().cpu(), levels=[0.0], colors='c')
      h_object.ax.axis('equal')
      plt.title('Contour Plot')
      plt.show()

    if num_iters == 1:
      print(loss)

  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])

  # Save loss value
  if output_path is not None:
    np.save(output_path, loss_value)
    
  return model, optimizer
  
# Train with file path passed
def train_file(num_iters, model, loss_function, batch_size, data_file, output_path=None, device='cpu'):
  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(output_path)
  
  # Get total number of lines
  num_of_lines = Operation.get_num_of_lines(data_file)
  
  # Train model
  for i in range(start, start+num_iters):
    # points, normal_vectors = Operation.read_txt3_to_batch(f, batch_size, line_indices, device)
    points, normal_vectors = Operation.read_txt3_to_batch(data_file, batch_size, num_of_lines, device)
    result =  model(points)
    loss = loss_function.irg_loss(model, result, points, normal_vectors, device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

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

# Save trained data
def save_model(path, model, optimizer):
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
  }, path)

# Load trained data
def load_model(path, dimension=3, device='cpu'):
  model = IGRPerceptron(dimension).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
  try:
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print('Model loaded')
  except:
    print('No model found. New model created')

  return model, optimizer

def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])
