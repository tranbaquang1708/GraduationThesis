import torch
import numpy
from modules import Visualization, Distribution
from scipy import spatial

class IGRLoss:
  def __init__(self, points=None, distribution=None):
    self.tau = 1
    self.ld = 0.01
    self.distribution = distribution
    if points is not None:
      self.stddvt = Distribution.find_kth_closest_distance(points, 50)

  def eval_distribution(self, model, batch_points, device='cpu'):
    d = self.distribution(batch_points, batch_points.shape[0], self.stddvt, device)

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
  def compute_loss(self, model, result, batch_points, batch_normal_vectors, device):
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

class RBFLoss:
  def __init__(self, distribution=None):
    self.offset = 0.1
    self.distribution = distribution

  def compute_loss(self, model, result, batch_points, batch_normal_vectors, device):
    onsurface_loss = (result.abs()).mean()
    
    tree = spatial.KDTree(batch_points.detach().cpu().numpy())
    
    out_points = batch_points + self.offset*batch_normal_vectors
    closest_index = tree.query(out_points.detach().cpu().numpy(), k=1)[1]
    out_points = out_points[numpy.where((closest_index == numpy.arange(0, out_points.shape[0])) == True)]
    outside_loss = ((model(out_points) - self.offset).abs()).mean()

    in_points = batch_points - self.offset*batch_normal_vectors
    closest_index = tree.query(in_points.detach().cpu().numpy(), k=1)[1]
    in_points = in_points[numpy.where((closest_index == numpy.arange(0, in_points.shape[0])) == True)]
    inside_loss = ((model(in_points) + self.offset).abs()).mean()

    # p = torch.cat((in_points, out_points))
    # Visualization.scatter_plot(p.detach().cpu().numpy())

    # inside_loss = 0
    # outside_loss = 0
    return onsurface_loss + inside_loss + outside_loss

class IGRBFLoss:
  def __init__(self, distribution=None):
    self.offset = 0.01
    self.tau = 1
    self.ld = 0.01
    self.distribution = distribution

  def eval_distribution(self, model, batch_points, device='cpu'):

    d = self.distribution

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
  def compute_loss(self, model, result, batch_points, batch_normal_vectors, device):
    geo_loss = torch.mean(torch.abs(result))

    tree = spatial.KDTree(batch_points.detach().cpu().numpy())
    out_points = batch_points + self.offset*batch_normal_vectors
    closest_index = tree.query(out_points.detach().cpu().numpy(), k=1)[1]
    out_points = out_points[numpy.where((closest_index == numpy.arange(0, out_points.shape[0])) == True)]
    outside_loss = ((model(out_points) - self.offset).abs()).mean()

    in_points = batch_points - self.offset*batch_normal_vectors
    closest_index = tree.query(in_points.detach().cpu().numpy(), k=1)[1]
    in_points = in_points[numpy.where((closest_index == numpy.arange(0, in_points.shape[0])) == True)]
    inside_loss = ((model(in_points) + self.offset).abs()).mean()

    x = torch.autograd.Variable(batch_points, requires_grad=True)
    x = x.to(device)
    f = model(x)
    g = torch.autograd.grad(outputs=f, inputs=x, 
                      grad_outputs=torch.ones(f.size()).to(device), 
                      create_graph=True, retain_graph=True, 
                      only_inputs=True)[0]
    grad_loss = (g - batch_normal_vectors).norm(2, dim=1).mean()
    # grad_loss = 0

    if self.distribution is None:
      constrain = 0
    else:
      constrain = self.eval_distribution(model, batch_points, device)

    return geo_loss + outside_loss + inside_loss + self.tau*grad_loss + self.ld * constrain