import os
import numpy as np
import torch
import torch.nn.functional as f
import random
from modules import Visualization
# --------------------------------------------------------------------------------
# DATA SET OPERATION

## 2D

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

  normal_vectors[:,0] = np.divide(vectors[:,1],norm)
  normal_vectors[:,1] = np.divide(-vectors[:,0],norm)

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

# def read_txt3_to_batch(file_pointer, batch_size, line_indices, device='cpu'):
#   points = np.zeros((0,3))
#   normal_vectors = np.zeros((0,3))
#   print(line_indices)

#   # with open(filename, 'r') as f:
#   chosen = sorted(random.sample(line_indices, batch_size))
#   # print(chosen)

#   for offset in line_indices:
#     print(offset)
#     file_pointer.seek(offset)
#     line = np.loadtxt(file_pointer, max_rows=1)
#     # line = file_pointer.readline()
#     # print(line)
#     # line = np.fromstring(file_pointer.readline(), sep=' ')
#     # line = np.fromstring(line, sep=' ')
#     print(line)
#     point, normal_vector = np.hsplit(line, 2)
#     print('next')
#     points = np.append(points, [point], axis=0)
#     normal_vectors = np.append(normal_vectors, [normal_vector], axis=0)

#   d = torch.from_numpy(points).float().to(device)
#   d.requires_grad = True
#   n = torch.from_numpy(normal_vectors).float().to(device)
#   n.requires_grad = True
  
#   return d, n 

def read_txt3_to_batch(data_file, batch_size, num_of_lines, device):
  points = np.zeros((0,3))
  normal_vectors = np.zeros((0,3))

  # Uniformly sample random points
  chosen = sorted(random.sample(range(num_of_lines), batch_size))
  
  with open(data_file, 'r') as f:
    for i, line in enumerate(f):
      if i in chosen:
        line = np.fromstring(line, sep=' ')
        point, normal_vector = np.hsplit(line, 2)
        points = np.append(points, [point], axis=0)
        normal_vectors = np.append(normal_vectors, [normal_vector], axis=0)
        if i == chosen[-1]:
          break

  # print(points)
  d = torch.from_numpy(points).float().to(device)
  d.requires_grad = True
  n = torch.from_numpy(normal_vectors).float().to(device)
  n.requires_grad = True
  
  return d, n 

# Read through file and create line indices
def line_indexing(filename):
  s = [0]
  with open(filename, 'r') as f:
    line_indices = [s.append(s[0]+len(n)) or s.pop(0) for n in f]

  return line_indices

def get_num_of_lines(data_file):
  with open(data_file) as f:
    for i, l in enumerate(f):
      pass

  return i + 1

#---------------------------------------------
# Loss value

# Get loss values of previous training
def load_loss_values(filename):
  try:
    loss_value = np.load(filename)
    print('Loss values loaded')
    start = int(loss_value[-1,0])
  except:
    loss_value = np.empty([0,2])
    print('No previous loss value found.')
    start = 0

  return loss_value, start


# Long list to list of lists
def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

#------------------------------------------------
# Compute grad
def compute_grad(points, model):
  var = torch.autograd.Variable(points, requires_grad=True).to(points.device)
  outputs = model(var)
  g = torch.autograd.grad(outputs=outputs,
                          inputs=var, 
                          grad_outputs=torch.ones(outputs.size()).to(points.device), 
                          create_graph=True,
                          retain_graph=True, 
                          only_inputs=True)[0]

  return g

#-----------------------------------------------------
# Write to file

def save_vtk(filename, tt, subx, suby, subz, z):
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
    f.write('DIMENSIONS ' + str(subx) + ' ' + str(suby) + ' ' + str(subz) +'\n')
    f.write('POINTS ' + str(subx*suby*subz) + ' double\n')

    np.savetxt(f, tt.detach().cpu().numpy())
    
    f.write('\n\n')

    f.write('POINT_DATA ' + str(subx*suby*subz) + '\n')
    f.write('SCALARS ' + field_title + ' double' + '\n')
    f.write('LOOKUP_TABLE default\n')

    np.savetxt(f, z.detach().cpu().numpy())
    f.write('\n')
