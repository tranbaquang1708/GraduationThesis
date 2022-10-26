import os
import numpy as np
import torch
import torch.nn.functional as f
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from modules import Visualization

#--------------------------------------------------------------------
# Read file


# Read text file and output dataset tensor and normal_vectors tensor
def read_txt2(filename, k_distance=50, rescale=None, device='cpu'):
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

  if rescale is not None:
    d_mean = onsurface_points.mean(axis=0)
    onsurface_points = (onsurface_points - d_mean)# * 10
    onsurface_points = rescale * onsurface_points / np.max(np.abs(onsurface_points))

  d = torch.from_numpy(onsurface_points).float().to(device)
  n = torch.from_numpy(normal_vectors).float().to(device)

  data = torch.cat((d,n), dim=-1)

  if k_distance is not None:
    stddvt = find_kth_closest_d(d, k_distance)
    data = torch.cat((data,stddvt), dim=-1)
    
  data.requires_grad = True
  
  return data

# Read from file, remove some points and output dataset tensor and normal_vectors tensor
# p: the proportion of points taken, value range [0,1]
def read_txt_omit2(filename, k_distance=50, p='1', device='cpu'):
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

  normal_vectors[:,0] = np.divide(vectors[:,1],norm)
  normal_vectors[:,1] = np.divide(-vectors[:,0],norm)

  d = torch.from_numpy(onsurface_points).float().to(device)
  n = torch.from_numpy(normal_vectors).float().to(device)

  data = torch.cat((d,n), dim=-1)

  if k_distance is not None:
    stddvt = find_kth_closest_d(d, k_distance)
    data = torch.cat((data,stddvt), dim=-1)
    
  data.requires_grad = True
  
  return data



def read_mesh2(filename, rescale=None, device='cpu'):
  with open(filename, 'r') as f:
    line1 = np.loadtxt(f, max_rows=1)

    raw_data = np.loadtxt(f)
    vertices = raw_data[:,[1,2]]
    att = raw_data[:,-1].reshape(raw_data.shape[0],1)

    if rescale is not None:
      d_mean = vertices.mean(axis=0)
      vertices = (vertices - d_mean)# * 10
      vertices = rescale * vertices / np.max(np.abs(vertices))

    data = np.concatenate((vertices, att), axis=1)
    data = torch.from_numpy(data).float().to(device)
    data.requires_grad = True

    # print(data)

  return data

def read_mesh2_boundary_only(filename, k_distance=None, rescale=None, device='cpu'):
  with open(filename, 'r') as f:
    line1 = np.loadtxt(f, max_rows=1)

    raw_data = np.loadtxt(f)
    full_vertices = raw_data[:,[1,2]]
    is_boundary = raw_data[:,-1]==1
    vertices = full_vertices[is_boundary]

    if rescale is not None:
      d_mean = vertices.mean(axis=0)
      vertices = (vertices - d_mean)# * 10
      vertices = rescale * vertices / np.max(np.abs(vertices))

    data = torch.from_numpy(vertices).float().to(device)
    data.requires_grad = True

    if k_distance is not None:
      stddvt = find_kth_closest_d(data, k_distance)
      data = torch.cat((data,stddvt), dim=-1)

    print(data.shape)

  return data

def read_triangle(filename, device='cpu'):
  with open(filename, 'r') as f:
    line1 = np.loadtxt(f, max_rows=1)

    raw_data = np.loadtxt(f, dtype='int')
    triangles = raw_data[:, 1:]

    if triangles.min() == 1:
      triangles = triangles -1
      print('Change starting index to 0')

  return triangles

def read_poly(filename, device='cpu'):
  with open(filename, 'r') as f:
    line1 = np.loadtxt(f, max_rows=1)
    dim = int(line1[1])
    line2 = np.loadtxt(f, max_rows=1)
    nseg = int(line2[0])
    raw_data = np.loadtxt(f, dtype='int', max_rows=nseg)
    segments = raw_data[:, 1:dim+1]
    if segments.min() == 1:
      segments = segments-1

  return segments


# Sample points on a circle
def circle_dataset(radius=1.0, k_distance=None, device='cpu'):
	# Points
  num_on_points = 100
  num_points = 3 * num_on_points
  thetas = np.arange(0.0, 2.0*np.pi, 2.0*np.pi/float(num_on_points))
  d = np.zeros((num_on_points,2))
  d[:,0] = radius*np.cos(thetas)
  d[:,1] = radius*np.sin(thetas)
  d = torch.from_numpy(d).float().to(device)

  # Normal vectors
  d_shifted = torch.roll(d, -1, 0)
  v = d_shifted - d
  v = f.normalize(v, p=2, dim=1)
  n = torch.zeros_like(v)
  n[:,0] = v[:,1]
  n[:,1] = -v[:,0]

  data = torch.cat((d,n), dim=-1)

  if k_distance is not None:
    stddvt = find_kth_closest_d(d, k_distance)
    data = torch.cat((data,stddvt), dim=-1)
    
  data.requires_grad = True
  
  return data


def annulus_dataset(radiuses=[1.0, 2.0], k_distance=None, device='cpu'):
  with torch.no_grad():
    radiuses.sort()
    if len(radiuses) % 2 == 0:
      reverse_normal = True
    else:
      reverse_normal = False
    data = torch.zeros(0,4).to(device)
    for r in radiuses:
      c = circle_dataset(radius=r, k_distance=None, device=device)
      if reverse_normal:
        c[:,-2:] = -c[:,-2:]
      reverse_normal = not reverse_normal
      data = torch.cat((data,c))
  
  if k_distance is not None:
    stddvt = find_kth_closest_d(data, k_distance)
    data = torch.cat((data,stddvt), dim=-1)

  data.requires_grad = True

  return data


# 3D
def read_txt3(filename, k_distance=None, device='cpu'):
  with open(filename, 'r') as f:
    raw_data = np.loadtxt(f)
  onsurface_points, vectors = np.hsplit(raw_data, 2)


  d_mean = onsurface_points.mean(axis=0)
  onsurface_points = (onsurface_points - d_mean) 
  onsurface_points = onsurface_points / np.max(np.abs(onsurface_points))

  norm = np.linalg.norm(vectors, axis=0)
  normal_vectors = vectors/norm

  d = torch.from_numpy(onsurface_points).float().to(device)
  n = torch.from_numpy(normal_vectors).float().to(device)

  data = torch.cat((d,n), dim=-1)

  if k_distance is not None:
    stddvt = find_kth_closest_d(d, k_distance)
    data = torch.cat((data,stddvt), dim=-1)

  data.requires_grad_()
  
  return data


#---------------------------------------------
# Loss value

# Get loss values of previous training
def load_loss_values(filename):
  try:
    loss_value = np.load(filename)
    print('Loss values loaded')
    start = int(loss_value[-1,0])
  except Exception as e:
    loss_value = np.empty([0,6])
    start = 0
    print('No previous loss value found.')
    print(e)

  return loss_value, start

  
#-----------------------------------------------------
# Write to file

def save_vtk(filename, tt, resx, resy, resz, z):
  # Create .vtk file
  # INPUT
  #   tt is the flattened grid
  #   z is the value of distance function at each point in the grid
  #   subx, suby, subz is the size of each vertex
  if tt.shape[1] == 2:
    tt = torch.cat((tt, torch.zeros(tt.shape[0],1).to(tt.device)), dim=-1)

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
# Derivatives
def compute_grad(outputs, inputs):
  g = torch.autograd.grad(outputs=outputs,
                          inputs=inputs, 
                          grad_outputs=torch.ones_like(outputs, requires_grad=False, device=outputs.device), 
                          create_graph=True,
                          retain_graph=True, 
                          only_inputs=True)[0][:, -3:]

  return g

def compute_laplacian(outputs, inputs, p=2):
  g = compute_grad(outputs, inputs)
  g = ((g.norm(2, dim=-1).view(g.shape[0],1))**(p-2))  * g

  div = 0.
  for i in range(g.shape[-1]):
    div += torch.autograd.grad(g[..., i], inputs, grad_outputs=torch.ones_like(g[..., i]), create_graph=True)[0][..., i:i+1]

  return div

def tucker_normalize(outputs, inputs):
  g_norm = compute_grad(outputs, inputs).norm(2, dim=-1).view(inputs.shape[0], 1)  
  denom = torch.sqrt(g_norm**2 + 2*(outputs.abs())) + g_norm

  return (2*outputs)/denom


#-----------------------------------------------------------------
# Sample range
def get_sample_ranges(points):
  ed = 0.2
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
    sample_ranges = torch.tensor([xmin-ed, xmax+ed, ymin-ed, ymax+ed, zmin-ed, zmax+ed], device=points.device)
  else:
    ed = 0.075 * torch.sqrt(dx*dx + dy*dy)
    sample_ranges = torch.tensor([xmin-ed, xmax+ed, ymin-ed, ymax+ed], device=points.device)

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
def uniform(sample_ranges, dim=3, device='cpu'):
  dx = sample_ranges[1]-sample_ranges[0]
  dy = sample_ranges[3]-sample_ranges[2]

  if dim == 2:
    dist_size = 16 * dx * dy
    dist_size = int(torch.max(torch.ceil(dist_size), torch.tensor([200]).to(device)))
    
    u_x = torch.FloatTensor(dist_size, 1).uniform_(sample_ranges[0], sample_ranges[1])
    u_y = torch.FloatTensor(dist_size, 1).uniform_(sample_ranges[2], sample_ranges[3])

    u = torch.cat((u_x, u_y), dim=-1)
  else:
    dz = sample_ranges[5]-sample_ranges[4]
    dist_size= dx * dy * dz
    dist_size = int(torch.max(torch.ceil(dist_size), torch.tensor([2048]).to(device)))
    
    u_x = torch.FloatTensor(dist_size, 1).uniform_(sample_ranges[0], sample_ranges[1])
    u_y = torch.FloatTensor(dist_size, 1).uniform_(sample_ranges[2], sample_ranges[3])
    u_z = torch.FloatTensor(dist_size, 1).uniform_(sample_ranges[4], sample_ranges[5])
    
    u = torch.cat((u_x, u_y, u_z), dim=-1)

  u.requires_grad = True

  return u.to(device)

def gaussian(points, stddvt):
  return points + (torch.randn_like(points) * stddvt)

def uniform_gaussian(points, sample_ranges, stddvt):
  dist_size = points.shape[0]
  u_x = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[0], sample_ranges[1])
  u_y = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[2], sample_ranges[3])
  if points.shape[1] == 2:
    u = torch.cat((u_x, u_y), dim=-1)
  else:
    u_z = torch.FloatTensor(dist_size//8, 1).uniform_(sample_ranges[4], sample_ranges[5])
    u = torch.cat((u_x, u_y, u_z), dim=-1)
  
  u = u.to(points.device)
  u.requires_grad = True


  g = points + (torch.randn_like(points) * stddvt)

  distribution = torch.cat([u,g])

  return distribution

#-----------------------------------------------------
#Distance to polygon
def dis_point2poly(p, nodes, segments):
  d = torch.tensor([float("inf")]).to(p.device)
  dim = p.shape[0]
  
  for segment in segments:
    v1 = nodes[segment[0], 0:dim]
    v2 = nodes[segment[1], 0:dim]
    nd = dis_to_segment(p, v1, v2)
    d = torch.min(d, nd)

  return d

def dis_to_segment(p, v1, v2):
  param = torch.dot((p-v1),(v2-v1)) / (torch.linalg.norm(v2-v1)**2)
  if param < 0:
    return torch.linalg.norm(v1-p) # Euclidean distance
  elif param > 1:
    return torch.linalg.norm(v2-p)
  else:
    return torch.linalg.norm(p - (v1+param*(v2-v1)))
