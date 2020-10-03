import torch

def pdist_squareform(in_tensor, function=None):
  if function is None:
    function = Distance.euclidean
  size = in_tensor.size()
  x = in_tensor.unsqueeze(0).expand(size[0],size[0],size[1])
  y = x.transpose(0,1)
  return function(x,y)

def euclidean(tensor1, tensor2):
  return torch.norm(tensor1-tensor2, dim=-1)