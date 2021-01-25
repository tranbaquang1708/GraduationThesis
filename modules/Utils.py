import os
import numpy as np
import torch
import torch.nn.functional as f
import random
from scipy.spatial import KDTree
from modules import Visualization

#--------------------------------------------------------------------
# Read file


# Read text file and output dataset tensor and normal_vectors tensor
def read_txt2(filename, device='cpu'):
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

# Read from file, remove some points and output dataset tensor and normal_vectors tensor
# p: the proportion of points taken, value range [0,1]
def read_txt_omit2(filename, p='1', device='cpu'):
  onsurface_points = np.zeros((0,2))
  shifted_points = np.zeros((0,2)) # onsurface_points left shifted by 1
  first_point = np.zeros((1,2))
  last_point = np.zeros((1,2))
  middle_points = np.zeros((0,2))
  next_point = np.zeros((1,2))

  with open(filename, 'r') as f:
    for c in range(int(f.readline())):
      num_of_verticles = int(f.readline())

      first_point[0] = np.loadtxt(f, max_rows=1)
      for i in range(num_of_verticles-2):
        next_point[0] = np.loadtxt(f, max_rows=1)
        if p > random.uniform(0.0, 1.0):
          middle_points = np.append(middle_points, next_point, axis=0)
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

# Sample points on a circle
def circle_dataset(device='cpu'):
	# Points
  num_on_points = 100
  num_points = 3 * num_on_points
  radius = 1.0
  thetas = np.arange(0.0, 2.0*np.pi, 2.0*np.pi/float(num_on_points))
  d = np.zeros((num_on_points,2))
  d[:,0] = radius*np.cos(thetas)
  d[:,1] = radius*np.sin(thetas)
  d = torch.from_numpy(d).float().to(device)
  d.requires_grad = True

  # Normal vectors
  d_shifted = torch.roll(d, -1, 0)
  v = d_shifted - d
  v = f.normalize(v, p=2, dim=1)
  n = torch.zeros_like(v)
  n[:,0] = -v[:,1]
  n[:,1] = v[:,0]

  return d.to(device),n.to(device)

## 3D
def read_txt3(filename, device='cpu'):
  with open(filename, 'r') as f:
    raw_data = np.loadtxt(f)
  onsurface_points, vectors = np.hsplit(raw_data, 2)
  norm = np.linalg.norm(vectors, axis=0)
  normal_vectors = vectors/norm
  
  d = torch.from_numpy(onsurface_points).float().to(device)
  d.requires_grad = True
  n = torch.from_numpy(normal_vectors).float().to(device)
  n.requires_grad = True
  
  return d, n


#---------------------------------------------
# Loss value

# Get loss values of previous training
def load_loss_values(filename):
  try:
    loss_value = np.load(filename)
    print('Loss values loaded')
    start = int(loss_value[-1,0])
  except:
    loss_value = np.empty([0,5])
    print('No previous loss value found.')
    start = -1

  return loss_value, start

  
#-----------------------------------------------------
# Write to file

def save_vtk(filename, tt, resx, resy, resz, z):
  # Create .vtk file
  # Only work for 3D
  # INPUT
  #   tt is the flattened grid
  #   z is the value of distance function at each point in the grid
  #   subx, suby, subz is the size of each vertex

  field_title = 'DENSITY'

  with open(filename, 'w') as f:
    f.write('# vtk DataFile Version 3.0\n')
    f.write('vtk output\n')
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_GRID\n')
    f.write('DIMENSIONS ' + str(resx) + ' ' + str(resy) + ' ' + str(resz) +'\n')
    f.write('POINTS ' + str(resx*resy*resz) + ' double\n')

    np.savetxt(f, tt.detach().cpu().numpy())
    
    f.write('\n\n')

    f.write('POINT_DATA ' + str(resx*resy*resz) + '\n')
    f.write('SCALARS ' + field_title + ' double' + '\n')
    f.write('LOOKUP_TABLE default\n')

    np.savetxt(f, z.detach().cpu().numpy())
    f.write('\n')


#------------------------------------------------
# Compute grad
def compute_grad(inputs, outputs):
  g = torch.autograd.grad(outputs=outputs,
                          inputs=inputs, 
                          grad_outputs=torch.ones_like(outputs, requires_grad=False, device=outputs.device), 
                          create_graph=True,
                          retain_graph=True, 
                          only_inputs=True)[0][:, -3:]

  return g

#-----------------------------------------------------------------
# Sample range
def get_sample_ranges(points):
  xmin = points[:,0].min()
  xmax = points[:,0].max()
  ymin = points[:,1].min()
  ymax = points[:,1].max()
  dx = xmax - xmin
  dy = ymax - ymin
  
  if points.shape[1] == 3:
    zmin = points[:,2].min()
    zmax = points[:,2].max()
    dz = zmax - zmin
    ed = 0.15 * torch.sqrt(dx*dx + dy*dy + dz*dz)
    sample_ranges = torch.tensor([xmin-ed, xmax+ed, ymin-ed, ymax+ed, zmin-ed, zmax+ed], device=points.device) * 1.5
  else:
    ed = 0.15 * torch.sqrt(dx*dx + dy*dy)
    sample_ranges = torch.tensor([xmin-ed, xmax+ed, ymin-ed, ymax+ed], device=points.device) * 1.5

  return sample_ranges


#--------------------------------------------------------------------
# Closest neighbor
def find_kth_closest_d(inputs, k):
  tree = KDTree(inputs.detach().cpu().numpy())
  d_k = tree.query(inputs.detach().cpu().numpy(), k=k+1)[0][:,-1]
  stddvt = torch.from_numpy(d_k).reshape((inputs.size()[0], 1)).type(torch.FloatTensor)

  return stddvt.to(inputs.device)


#----------------------------------------------------------------------
# Distribution
def uniform_gaussian(points, dist_size, sample_ranges, stddvt):
  u_x = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[0], sample_ranges[1])
  u_y = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[2], sample_ranges[3])
  if points.shape[1] == 2:
    u = torch.cat((u_x, u_y), dim=-1)
  else:
    u_z = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[4], sample_ranges[5])
    u = torch.cat((u_x, u_y, u_z), dim=-1)
  
  u = u.to(points.device)


  g = points + (torch.randn_like(points) * stddvt)

  return torch.cat((u,g))