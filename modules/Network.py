import torch
import torch.nn as nn
import numpy as np

# Neural network model
class Implicit(nn.Module):
  def __init__(self, dimension):
    super(Implicit, self).__init__()

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
    x = input

    for layer in range(0, self.num_layers - 1):

      lin = getattr(self, "lin" + str(layer))

      if layer in self.skip_in:
        x = torch.cat([x, input], -1) / np.sqrt(2)

      x = lin(x)

      if layer < self.num_layers - 2:
        x = self.activation(x)

    return x

# Save model and optimizer
def save_model(path, model, optimizer, scheduler):
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
  }, path)


# Load model and optimizer
def load_model(path, dimension=3, device='cpu'):
  model = Implicit(dimension).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  # Scheduler
  scheduler_steps = []
  for i in range(4):
    scheduler_steps.append(2000 * (i+1))
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.25)

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