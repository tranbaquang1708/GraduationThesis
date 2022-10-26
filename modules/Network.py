import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import re

# Neural network model
class Implicit(nn.Module):
  def __init__(self, dimension):
    super().__init__()

    d_in = dimension
    dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ] # 8 layers
    # dims = [ 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512 ] # 12 layers
    # dims = [ 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512 ] # 16 layers
    # dims = [ 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512 ] # 18 layers
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
    # self.activation = torch.cos

  def forward(self, inputs):
    x = inputs

    for layer in range(0, self.num_layers - 1):

      lin = getattr(self, "lin" + str(layer))

      if layer in self.skip_in:
        x = torch.cat([x, inputs], -1) / np.sqrt(2)

      x = lin(x)

      if layer < self.num_layers - 2:
        x = self.activation(x)

    return x



class ImplicitBoundary(nn.Module):
  def __init__(self, dimension):
    super().__init__()

    d_in = dimension
    # dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
    # dims = [256, 256, 256, 256, 256, 256, 256, 256]
    dims = [256, 256, 256, 256]
    # dims = [ 512, 512, 512, 512 ]
    beta = 100
    skip_in = []
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
    # self.activation = torch.cos

  def forward(self, inputs):
    x = inputs

    for layer in range(0, self.num_layers - 1):

      lin = getattr(self, "lin" + str(layer))

      if layer in self.skip_in:
        x = torch.cat([x, inputs], -1) / np.sqrt(2)

      x = lin(x)

      if layer < self.num_layers - 2:
        x = self.activation(x)

    return x

class ImplicitConstraint(nn.Module):
  def __init__(self, modelf, dimension):
    super().__init__()

    d_in = dimension
    dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
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
    # self.activation = torch.cos

    self.modelf = modelf
    for param in self.modelf.parameters():
      param.requires_grad = False

  def forward(self, inputs):
    x = inputs

    for layer in range(0, self.num_layers - 1):

      lin = getattr(self, "lin" + str(layer))

      if layer in self.skip_in:
        x = torch.cat([x, inputs], -1) / np.sqrt(2)

      x = lin(x)

      if layer < self.num_layers - 2:
        x = self.activation(x)

    return x * torch.tanh(10 * self.modelf(inputs))
  


class SineLayer(nn.Module):    
  def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=30):
      super().__init__()
      self.omega_0 = omega_0
      self.is_first = is_first
      
      self.in_features = in_features
      self.linear = nn.Linear(in_features, out_features, bias=bias)
      
      self.init_weights()
  
  def init_weights(self):
    with torch.no_grad():
      if self.is_first:
        self.linear.weight.uniform_(-1 / self.in_features, 
                                      1 / self.in_features)      
      else:
        self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                      np.sqrt(6 / self.in_features) / self.omega_0)
      
  def forward(self, input):
    return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
  def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                first_omega_0=30, hidden_omega_0=30.):
    super().__init__()
    
    self.net = []
    self.net.append(SineLayer(in_features, hidden_features, 
                              is_first=True, omega_0=first_omega_0))

    for i in range(hidden_layers):
      self.net.append(SineLayer(hidden_features, hidden_features, 
                                is_first=False, omega_0=hidden_omega_0))

    if outermost_linear:
      final_linear = nn.Linear(hidden_features, out_features)
        
      with torch.no_grad():
          final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                        np.sqrt(6 / hidden_features) / hidden_omega_0)
            
      self.net.append(final_linear)
    else:
      self.net.append(SineLayer(hidden_features, out_features, 
                                is_first=False, omega_0=hidden_omega_0))
    
    self.net = nn.Sequential(*self.net)
  
  def forward(self, coords):
    output = self.net(coords)
    return output



# Save model, optimizer and scheduler
def save_model(path, model, optimizer, scheduler):
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
  }, path)

# Load model, optimizer and scheduler
def load_model(path, dimension=3, device='cpu'):
  model = Implicit(dimension).to(device)
  # model = Siren(in_features=dimension, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device)
  # model = SingleBVPNet(type='sine', in_features=dimension)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  scheduler_steps = []
  for i in range(6):
    scheduler_steps.append(2000 * (i+1))
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.5)

  try:
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.eval()
    print('Model loaded')
  except Exception as e:
    print('No model found. New model created')
    print()
    print(e)

  return model, optimizer, scheduler

def load_model_f(path, dimension=3, device='cpu'):
  model = ImplicitBoundary(dimension).to(device)
  # model = Siren(in_features=dimension, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device)
  # model = SingleBVPNet(type='sine', in_features=dimension)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  scheduler_steps = []
  for i in range(6):
    scheduler_steps.append(2000 * (i+1))
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.5)

  try:
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.eval()
    print('Model loaded')
  except Exception as e:
    print('No model found. New model created')
    print()
    print(e)

  return model, optimizer, scheduler

def load_model_g(path, modelf, dimension=3, device='cpu'):
  model = ImplicitConstraint(modelf, dimension).to(device)
  # model = Siren(in_features=dimension, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device)
  # model = SingleBVPNet(type='sine', in_features=dimension)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  scheduler_steps = []
  for i in range(6):
    scheduler_steps.append(2000 * (i+1))
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.5)

  try:
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.eval()
    print('Model loaded')
  except Exception as e:
    print('No model found. New model created')
    print()
    print(e)

  return model, optimizer, scheduler