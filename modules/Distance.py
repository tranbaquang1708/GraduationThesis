import torch

def pdist_squareform(in_tensor, function=None):
  if function is None:
    function = Distance.euclidean_distance
  size0 = in_tensor.size()[0]
  x = in_tensor.unsqueeze(0).expand(size0,size0,2)
  y = x.transpose(0,1)
  return function(x,y)

def euclidean_distance(tensor1, tensor2):
  return torch.norm(tensor1-tensor2, dim=-1)