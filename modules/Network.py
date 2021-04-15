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
    dims = [256, 256, 256, 256, 256, 256, 256, 256]
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

def get_subdict(dictionary, key=None):
  if dictionary is None:
    return None
  if (key is None) or (key == ''):
    return dictionary
  key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
  return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
    in dictionary.items() if key_re.match(k) is not None)

class MetaModule(nn.Module):
    # """
    # Base class for PyTorch meta-learning modules. These modules accept an
    # additional argument `params` in their `forward` method.
    # Notes
    # -----
    # Objects inherited from `MetaModule` are fully compatible with PyTorch
    # modules from `torch.nn.Module`. The argument `params` is a dictionary of
    # tensors, with full support of the computation graph (for differentiation).
    # """
  def meta_named_parameters(self, prefix='', recurse=True):
    gen = self._named_members(
        lambda module: module._parameters.items()
        if isinstance(module, MetaModule) else [],
        prefix=prefix, recurse=recurse)
    for elem in gen:
      yield elem

  def meta_parameters(self, recurse=True):
    for name, param in self.meta_named_parameters(recurse=recurse):
      yield param

class MetaSequential(nn.Sequential, MetaModule):
  __doc__ = nn.Sequential.__doc__

  def forward(self, input, params=None):
    for name, module in self._modules.items():
      if isinstance(module, MetaModule):
        input = module(input, params=get_subdict(params, name))
      elif isinstance(module, nn.Module):
        input = module(input)
      else:
        raise TypeError('The module must be either a torch module '
            '(inheriting from `nn.Module`), or a `MetaModule`. '
            'Got type: `{0}`'.format(type(module)))
    return input

class BatchLinear(nn.Linear, MetaModule):
  # '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
  # hypernetwork.'''
  __doc__ = nn.Linear.__doc__

  def forward(self, input, params=None):
    if params is None:
      params = OrderedDict(self.named_parameters())

    bias = params.get('bias', None)
    bias = bias.cuda()
    weight = params['weight'].cuda()

    output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
    output += bias.unsqueeze(-2)
    return output

class Sine(nn.Module):
  def __init(self):
    super().__init__()

  def forward(self, input):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
    return torch.sin(30 * input)

class FCBlock(MetaModule):
    # '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    # Can be used just as a normal neural network though, as well.
    # '''

  def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                outermost_linear=False, nonlinearity='relu', weight_init=None):
    super().__init__()

    self.first_layer_init = None

    # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
    # special first-layer initialization scheme
    nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                      'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                      'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                      'tanh':(nn.Tanh(), init_weights_xavier, None),
                      'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                      'softplus':(nn.Softplus(), init_weights_normal, None),
                      'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

    nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

    if weight_init is not None:  # Overwrite weight init if passed
      self.weight_init = weight_init
    else:
      self.weight_init = nl_weight_init

    self.net = []
    self.net.append(MetaSequential(
        BatchLinear(in_features, hidden_features), nl
    ))

    for i in range(num_hidden_layers):
      self.net.append(MetaSequential(
          BatchLinear(hidden_features, hidden_features), nl
      ))

    if outermost_linear:
      self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
    else:
      self.net.append(MetaSequential(
          BatchLinear(hidden_features, out_features), nl
      ))

    self.net = MetaSequential(*self.net)
    if self.weight_init is not None:
      self.net.apply(self.weight_init)

    if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
      self.net[0].apply(first_layer_init)

  def forward(self, coords, params=None, **kwargs):
    if params is None:
      params = OrderedDict(self.named_parameters())

    output = self.net(coords, params=get_subdict(params, 'net'))
    return output

  def forward_with_activations(self, coords, params=None, retain_grad=False):
    '''Returns not only model output, but also intermediate activations.'''
    if params is None:
      params = OrderedDict(self.named_parameters())

    activations = OrderedDict()

    x = coords.clone().detach().requires_grad_(True)
    activations['input'] = x
    for i, layer in enumerate(self.net):
      subdict = get_subdict(params, 'net.%d' % i)
      for j, sublayer in enumerate(layer):
        if isinstance(sublayer, BatchLinear):
          x = sublayer(x, params=get_subdict(subdict, '%d' % j))
        else:
          x = sublayer(x)

        if retain_grad:
          x.retain_grad()
        activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
    return activations


class SingleBVPNet(MetaModule):
    # '''A canonical representation network for a BVP.'''

  def __init__(self, out_features=1, type='sine', in_features=2,
                mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
    super().__init__()
    self.mode = mode

    if self.mode == 'rbf':
      self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
      in_features = kwargs.get('rbf_centers', 1024)
    elif self.mode == 'nerf':
      self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                  sidelength=kwargs.get('sidelength', None),
                                                  fn_samples=kwargs.get('fn_samples', None),
                                                  use_nyquist=kwargs.get('use_nyquist', True))
      in_features = self.positional_encoding.out_dim

    # self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
    #                                             downsample=kwargs.get('downsample', False))
    self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                        hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
    print(self)

  def forward(self, model_input, params=None):
    if params is None:
      params = OrderedDict(self.named_parameters())

    # Enables us to compute gradients w.r.t. coordinates
    # coords_org = model_input['coords'].clone().detach().requires_grad_(True)
    # coords_org = model_input['coords'].clone().detach().requires_grad_(True)
    # coords = coords_org
    coords = model_input

    # various input processing methods for different applications
    # if self.image_downsampling.downsample:
    #   coords = self.image_downsampling(coords)
    if self.mode == 'rbf':
      coords = self.rbf_layer(coords)
    elif self.mode == 'nerf':
      coords = self.positional_encoding(coords)

    output = self.net(coords, get_subdict(params, 'net'))
    # return {'model_in': coords_org, 'model_out': output}
    return output

  def forward_with_activations(self, model_input):
    '''Returns not only model output, but also intermediate activations.'''
    coords = model_input['coords'].clone().detach().requires_grad_(True)
    activations = self.net.forward_with_activations(coords)
    return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

def init_weights_normal(m):
  if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, 'weight'):
      nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_xavier(m):
  if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)

def init_weights_selu(m):
  if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, 'weight'):
      num_input = m.weight.size(-1)
      nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
def init_weights_elu(m):
  if type(m) == BatchLinear or type(m) == nn.Linear:
    if hasattr(m, 'weight'):
      num_input = m.weight.size(-1)
      nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def sine_init(m):
  with torch.no_grad():
    if hasattr(m, 'weight'):
      num_input = m.weight.size(-1)
      # See supplement Sec. 1.5 for discussion of factor 30
      m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
  with torch.no_grad():
      if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        m.weight.uniform_(-1 / num_input, 1 / num_input)


# Save model and optimizer
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
  except:
    print('No model found. New model created')

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
  except:
    print('No model found. New model created')

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
  except:
    print('No model found. New model created')

  return model, optimizer, scheduler