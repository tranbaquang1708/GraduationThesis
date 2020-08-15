# Surface recontruction using Implicit Geometric Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
    # out = torch.flatten(out, 1)
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
  # out = out.backward()
  # print(distribution.grad)
  # print(out)
  # print(nn(distribution).sum())
  # print(distribution.grad)
  return F.normalize(distribution.grad)

# Compute gradient of function with respect to the input
def compute_grad(nn, dataset, normal_vectors):
  data = dataset.detach().clone()
  data.requires_grad = True
  out = nn(data).mean().backward()
  g = F.normalize(data.grad)
  out = torch.norm(g - normal_vectors, dim=1)
  return torch.reshape(out, (out.shape[0], 1))

# Compute loss
# loss = \sum_I (w^T*x_i)^2 + 0.1 * (||w||^2 - 1)^2
def irg_loss(nn, result, data, normal_vectors, tau=1):
  # print(nn.device)
  # print(eikonal(nn, distribution))
  # w^T*x_1 = w_x1 * x_1 + w_y1 * y_1 + w_x2 * x_1 + w_y2 * y_1 + ... + w_xn * x_1 + w_yn * y_1
  # return torch.sum(torch.square(torch.sum(torch.sum(w*x[:,None], dim=1), dim=1))) + 0.1*torch.square(torch.square(torch.norm(w)-1))
  # print(result.shape)
  # w^t * x_1 = w_1 * x_1 + w_2 * x_1 + ... + w_n * x_1 
  # d = torch.sqrt(torch.sum(torch.square(x - x[:, None]), dim=2)) #pdist squareform
  # return torch.sum(torch.square(torch.sum(w*d, dim=0))) + 0.1*torch.square(torch.square(torch.norm(w)-1))
  # loss = torch.mean(torch.abs(nn(input)) + tau * compute_grad(nn, input, normal_vectors)) + 0.1*(torch.norm(eikonal(nn, distribution)) -1)**2
  loss = torch.mean(torch.abs(result) + tau * compute_grad(nn, data, normal_vectors))
  return loss

def train(dataset, normal_vectors, num_epochs, device):
  igr_nn = IGRPerceptron()
  # if not next(my_nn.parameters()).is_cuda:
  igr_nn = igr_nn.to(device)
  # next(my_nn.parameters()).to(device)
  optimizer = torch.optim.Adam(igr_nn.parameters(), lr=0.0001)

  for i in range(num_epochs): 
    result =  igr_nn(dataset)
    loss = irg_loss(igr_nn, result, dataset, normal_vectors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%500 == 0:
      print("Step " + str(i+1) + ":")
      print(loss)
    
  return igr_nn

# Make a grid
def grid(X, Y, device):
  xmin = torch.min(X).item()
  xmax = torch.max(X).item()
  ymin = torch.min(Y).item()
  ymax = torch.max(Y).item()

  # print(xmax)
  # print(xmax.item())

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
  # print(*dimg**2)
  # print(dimg[0])
  # print(*xx.shape)
  # print(xx.shape)
  # z = torch.zeros(dimg)
  z = torch.empty((0,1))
  # print(z.shape)
  # g = np.concatenate((xx, yy), axis=1)
  # print(xx.shape)
  # t = w[None, 0] * xx
  # print(t)
  tt = torch.stack((xx, yy), axis=2)
  tt = torch.reshape(tt, (dimg,2))
  # tt1 = torch.reshape(tt, (-1,))
  # print(tt)
  # print(tt1.shape)
  # print(tt1)
  # print(tt)
  # for k in range(int(dimw[0])):
  #   z += w[k] * np.sqrt((xx - x[k,0])**2 + (yy - x[k,1])**2)
    # z += np.sqrt(np.square(w[i] * (xx-dataset) + np.square(w[i] * yy))
    # z += np.square(w[i,0] * (xx-dataset.numpy()[i,0]) + w[i,1]*(yy-dataset.numpy()[i,1]))
  # z = np.square(w[:,0]*xx[:] + w[:,1]*yy[:])
  # z = np.sum(np.sum(np.square(w*g[:, None]), axis=1), axis=1)
  # print(t.shape)

  # i = 0
  # while True:
  #   if (i+200>=len(tt)):
  #     out = nn(tt[i:len(tt), :])
  #     z = torch.cat((z,out))
  #     break

  #   out = nn(tt[i:i+200, :])
  #   z = torch.cat((z,out))
  #   i = i+200

  z = nn(tt)
  # print(z)
  # print(z.shape)
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