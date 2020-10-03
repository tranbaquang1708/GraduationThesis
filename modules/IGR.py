# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
from modules import Distance, Visualization, Distribution

# Neural network model
class IGRPerceptron(nn.Module):
  def __init__(self, dimension):
    # Neural network layers
    super(IGRPerceptron, self).__init__()
    self.fc1 = nn.Linear(dimension, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 512)
    self.fc4 = nn.Linear(512, 512)
    self.fc5 = nn.Linear(512, 512)
    self.fc6 = nn.Linear(512, 512)
    self.fc7 = nn.Linear(512, 512)
    self.fc_last = nn.Linear(512, 1)
    self.activation = nn.Softplus()

  def forward(self, x):
    out = self.fc1(x)
    out = self.activation(out)
    out = self.fc2(out)
    out = self.activation(out)
    out = self.fc3(out)
    out = self.activation(out)
    out = self.fc4(out)
    out = self.activation(out)
    out = self.fc5(out)
    out = self.activation(out)
    # out = self.fc6(out)
    # out = self.activation(out)
    # out = self.fc7(out)
    # out = self.activation(out)
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

def to_batch(dataset, batch_size):
  dataset_size = len(dataset)
  num_of_batch = dataset_size // batch_size
  # print(num_of_batch)
  indices = torch.randperm(dataset_size)
  # print(indices)
  batches_indices = []

  if dataset_size < batch_size:
    batches_indices.append(indices)
  else:
    for i in range(0, num_of_batch):
      batches_indices.append(indices[i*batch_size:(i+1)*batch_size])

    if dataset_size % batch_size != 0:
      batches_indices.append(indices[-batch_size:])

  # print(batches_indices)
  return batches_indices

def train(dataset, normal_vectors, num_epochs, batch_size, device, loss_function, model=None):
  if model is None:
    model = IGRPerceptron(dataset[0].shape[0])
    model = model.to(device)
    
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  loss = 999.0

  loss_i = []
  loss_value = []

  for i in range(num_epochs):
    # batches_indices = to_batch(dataset, batch_size)
    # for batch_indices in batches_indices:
    #   batch_points = dataset[batch_indices]
    #   batch_points.to(device)
    #   # print(batch_points)
    #   # batch_points.requires_grad = True
    #   batch_normal_vectors = normal_vectors[batch_indices]
    #   batch_normal_vectors.to(device)
    #   # batch_normal_vectors.requires_grad = True
    batch_points = dataset
    batch_points.to(device)
    batch_normal_vectors = normal_vectors
    batch_normal_vectors.to(device)
    result =  model(batch_points)
    loss = loss_function.irg_loss(model, result, batch_points, batch_normal_vectors, device)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
      # Visualization.scatter_plot(batch_points.detach())

    if (i+1)%500 == 0:
      loss_i.append(i+1)
      loss_value.append(loss.item())
      print("Step " + str(i+1) + ":")
      print(loss)

    if num_epochs == 1:
      print(loss)

  Visualization.loss_graph(loss_i, loss_value)
    
  return model
  
# Save trained data
def save_model(path, model):
  torch.save(model.state_dict(), path)

# Load trained data
def load_model(path, dimension=3, device='cpu'):
  if os.path.isfile(path):
    model = IGRPerceptron(dimension)
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print('Model loaded')

    return model

  print('No model found')
  return None

