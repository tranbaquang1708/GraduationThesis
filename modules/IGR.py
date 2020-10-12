# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
from modules import Distance, Visualization, Distribution, Operation

# Neural network model
class IGRPerceptron(nn.Module):
  def __init__(self, dimension):
    # Neural network layers
    super(IGRPerceptron, self).__init__()
    self.fc0 = nn.Linear(dimension, 512)
    self.fc1 = nn.Linear(512, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 512 - dimension)
    self.fc4 = nn.Linear(512, 512)
    self.fc5 = nn.Linear(512, 512)
    self.fc6 = nn.Linear(512, 512)
    self.fc_last = nn.Linear(512, 1)
    self.activation = nn.Softplus()

  def forward(self, x):
    out = self.fc0(x)
    out = self.activation(out)
    out = self.fc1(out)
    out = self.activation(out)
    out = self.fc2(out)
    out = self.activation(out)
    out = self.fc3(out)
    out = self.activation(out)
    out = torch.cat((out, x), 1) # Skip connection
    out = self.fc4(out)
    out = self.activation(out)
    out = self.fc5(out)
    out = self.activation(out)
    out = self.fc6(out)
    out = self.activation(out)
    out = self.fc_last(out)
    return out

class LossFunction:
  def __init__(self, tau=None, ld=None, distribution=None):
    if tau is None:
      self.tau = 1
    else:
      self.tau = tau

    if ld is None:
      self.ld = 0.01
    else:
      self.ld = ld

    self.distribution = distribution

  def eval_distribution(self, model, batch_points, device='cpu'):
    d = self.distribution(batch_points, device)
    # Visualization.scatter_plot(d.detach().cpu().numpy())
    # print(d)
    x = torch.autograd.Variable(d, requires_grad=True)
    x.to(device)
    f = model(x)
    g = torch.autograd.grad(outputs=f, inputs=x, 
                    grad_outputs=torch.ones(f.size()).to(device), 
                    create_graph=True, retain_graph=True, 
                    only_inputs=True)[0]

    return (((g.norm(2, dim=1) - 1))**2).mean()
  
  # Compute loss
  def irg_loss(self, model, result, batch_points, batch_normal_vectors, device):
    geo_loss = torch.mean(torch.abs(result))

    x = torch.autograd.Variable(batch_points, requires_grad=True)
    x = x.to(device)
    f = model(x)
    g = torch.autograd.grad(outputs=f, inputs=x, 
                      grad_outputs=torch.ones(f.size()).to(device), 
                      create_graph=True, retain_graph=True, 
                      only_inputs=True)[0]
    grad_loss = (g - batch_normal_vectors).norm(2, dim=1).mean()

    if self.distribution is None:
      constrain = 0
    else:
      constrain = self.eval_distribution(model, batch_points, device)

    return geo_loss + self.tau*grad_loss + self.ld * constrain

def train(num_iters, model, loss_function, batch_size=None, data_file=None, points=None, normal_vectors=None, output_path=None, device='cpu'):
  if data_file is not None:
    model = train_file(num_iters, model, loss_function, batch_size, data_file, output_path, device)
  if points is not None:
    model = train_data(num_iters, model, loss_function, points, normal_vectors, output_path, device)

  return model

# Train with data passed
def train_data(num_iters, model, loss_function, points, normal_vectors, output_path=None, device='cpu'):    
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(output_path)

  for i in range(start, start+num_iters):
    result =  model(points)
    loss = loss_function.irg_loss(model, result, points, normal_vectors, device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
      # Visualization.scatter_plot(points.detach())

    if (i+1)%500 == 0:
      loss_value = np.append(loss_value, [[i+1, loss.item()]], axis=0)
      print("Step " + str(i+1) + ":")
      print(loss)

    if num_iters == 1:
      print(loss)

  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])

  # Save loss value
  if output_path is not None:
    np.save(output_path, loss_value)
    
  return model
  
# Train with file path passed
def train_file(num_iters, model, loss_function, batch_size, data_file, output_path=None, device='cpu'):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  # Get loss values and number of iteration in last training
  loss_value, start = Operation.load_loss_values(output_path)

  # Get index of line
  # line_indices = Operation.line_indexing(data_file)
  
  # Get total number of lines
  num_of_lines = Operation.get_num_of_lines(data_file)
  
  # Train model
  # with open(data_file, 'r') as f:
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
def save_model(path, model):
  torch.save(model.state_dict(), path)

# Load trained data
def load_model(path, dimension=3, device='cpu'):
  model = IGRPerceptron(dimension)
  model.to(device)
  try:
    model.load_state_dict(torch.load(path))
    model.eval()
    print('Model loaded')
  except:
    print('No model found. New model created')

  return model

def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  Visualization.loss_graph(loss_value[:,0], loss_value[:,1])
