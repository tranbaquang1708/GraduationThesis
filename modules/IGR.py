# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# Neural network model
class IGRPerceptron(nn.Module):
  def __init__(self):
    super(IGRPerceptron, self).__init__()
    self.fc1 = nn.Linear(2, 512)
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

# Compute Eikonal term
def eikonal(nn, distribution):
  out = nn(distribution).sum().backward()
  return F.normalize(distribution.grad)

# Compute gradient of function with respect to the input
def compute_grad(result, dataset, normal_vectors):
  data = dataset.detach().clone()
  data.requires_grad = True
  g = torch.autograd.grad(result.sum(), dataset, retain_graph=True)[0].data
  out = torch.norm(g - normal_vectors, dim=1)
  return torch.reshape(out, (out.shape[0], 1))

# Compute loss
# loss = \sum_I (w^T*x_i)^2 + 0.1 * (||w||^2 - 1)^2
def irg_loss(nn, result, data, normal_vectors, tau=1):
  loss_out = torch.mean(torch.abs(result))
  loss_grad = torch.mean(tau * compute_grad(result, data, normal_vectors))
  return loss_out + loss_grad

def irg_loss2(nn, result, dataset, normal_vectors, device, tau=1):
  geo_loss = torch.mean(torch.abs(result))

  x = torch.autograd.Variable(dataset, requires_grad=True)
  x = x.to(device)
  f = nn(x)
  g = torch.autograd.grad(outputs=f, inputs=x, 
                    grad_outputs=torch.ones(f.size()).to(device), 
                    create_graph=True, retain_graph=True, 
                    only_inputs=True)[0]
  grad_loss = (g - normal_vectors).norm(2, dim=1).mean()

  return geo_loss + tau*grad_loss

def train(dataset, normal_vectors, num_epochs, device, model=None):
  if model is None:
    model = IGRPerceptron()
    model = model.to(device)
    
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  for i in range(num_epochs): 
    result =  model(dataset)
    loss = irg_loss2(model, result, dataset, normal_vectors, device, tau=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%500 == 0:
      print("Step " + str(i+1) + ":")
      print(loss)
    
  return model

# Save trained data
def save_model(path, model):
  torch.save(model.state_dict(), path)

# Load trained data
def load_model(path, device):
  if os.path.isfile(path):
    model = IGRPerceptron()
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

  return None

# Make a grid
def grid(X, Y, device):
  xmin = torch.min(X).item()
  xmax = torch.max(X).item()
  ymin = torch.min(Y).item()
  ymax = torch.max(Y).item()

  dx = xmax - xmin
  dy = ymax - ymin

  resx = 50
  resy = 50

  ed = 0.1*math.sqrt(dx*dx+dy*dy)

  x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
  y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))

  xx, yy = torch.meshgrid(x, y)
  return xx.to(device), yy.to(device)

# Sampling the function on the grid
def sampling(nn, xx, yy):
  dimg = (xx.shape[0])**2
  z = torch.empty((0,1))
  tt = torch.stack((xx, yy), axis=2)
  tt = torch.reshape(tt, (dimg,2))
  z = nn(tt)
  print(torch.reshape(z, (50,50)))

  return torch.reshape(z, (50,50))

# Read text file and output dataset tensor and normal_vectors tensor
def read_txt2(filename, device):
  onsurface_points = np.zeros((0,2))
  shifted_points = np.zeros((0,2)) # onsurface_points left shifted by 1
  first_point = np.zeros((1,2))
  last_point = np.zeros((1,2))

  with open(filename, 'r') as f:
    for c in range(int(f.readline())):
      num_of_verticles = int(f.readline())

      first_point[0] = np.loadtxt(f, max_rows=1)
      middle_points = np.loadtxt(f, max_rows=num_of_verticles-2)
      last_point[0] = np.loadtxt(f, max_rows=1)

      # Onsuface points order: first_point,middle_points,last_point
      # Shifted points order:  middle_points,last_point,first_point
      onsurface_points = np.concatenate((onsurface_points, first_point))    # Onsurface: first_point
      onsurface_points = np.concatenate((onsurface_points, middle_points))  # Onsurface: middle_points
      shifted_points = np.concatenate((shifted_points, middle_points))      # Shifted: middle_points
      # Remove the last point if it is the same as the first point
      if np.not_equal(last_point,first_point).any():
        onsurface_points = np.concatenate((onsurface_points,last_point))    # Onsurface: last_point
        shifted_points = np.concatenate((shifted_points,last_point))        # Shifted: last_point
      shifted_points = np.concatenate((shifted_points,first_point))         # Shifted: first_point

  # Vector of 2 consecutive points
  vectors = shifted_points - onsurface_points
  # Getting normal vectors
  norm = np.linalg.norm(vectors, axis=1)
  normal_vectors = np.ones_like(vectors)

  normal_vectors[:,0] = np.divide(-vectors[:,1],norm)
  normal_vectors[:,1] = np.divide(vectors[:,0],norm)

  d = torch.from_numpy(onsurface_points).float().to(device)
  d.requires_grad = True
  n = torch.from_numpy(normal_vectors).float().to(device)
  n.requires_grad = True

  return d,n