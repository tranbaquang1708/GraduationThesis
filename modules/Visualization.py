import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import math
import numpy as np

#-------------------------------------------------------
# PLOT

# Visualization for 2D dataset
def visualize2(dataset, normal_vectors, xx, yy, z, scatter=True, vecfield=True, surface=True, filled_contour=True):
  # Visualize for 2D
  # Scatter plot for points
  if scatter:
    h_points = plt.scatter(dataset[:,0], dataset[:,1], s=2)
    plt.show()
  # Plot vector field
  if vecfield:
    h_vector = plt.quiver(dataset[:,0], dataset[:,1], normal_vectors[:,0], normal_vectors[:,1])
    h_vector.ax.axis('equal')
    plt.show()
  # Plot generated surface
  if surface:
    h_object = plt.contour(xx,yy, z, levels=[0.0], colors='c')
    h_object.ax.axis('equal')
    plt.show()
  # Plot filled contour
  if filled_contour:
    hf = plt.contourf(xx,yy,z)
    hf.ax.axis('equal')
    plt.show()

def scatter_plot(points):
  h_points = plt.scatter(points[:,0], points[:,1], s=2)
  plt.show()

def loss_graph(i, value):
  plt.plot(i, value)
  plt.xlabel('Epoch')
  plt.ylabel('Loss value')
  plt.show()

# Visualization for 3D dataset
def visualize3(dataset, normal_vectors, z, scatter=True, vecfield=True, surface=True):
  if scatter:
    fig = plt.figure(figsize=(10,10))
    ax_points = fig.add_subplot(111, projection='3d')
    ax_points.scatter(dataset[:,0], dataset[:,1], dataset[:,2])
    plt.show()
  if vecfield:
    fig = plt.figure(figsize=(10,10))
    ax_vecfield = fig.add_subplot(111, projection='3d')
    ax_vecfield.quiver(dataset[:,0], dataset[:,1], dataset[:,2], 
                       normal_vectors[:,0], normal_vectors[:,1], normal_vectors[:,2])
    plt.show()
  if surface:
    verts, faces, normals, values = measure.marching_cubes_lewiner(z, 0)
    fig = plt.figure(figsize=(10,10))
    ax_surface = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax_surface.add_collection3d(mesh)
    ax_surface.set_xlim(0, 50)
    ax_surface.set_ylim(0, 50)
    ax_surface.set_zlim(0, 50)
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------
# GRID

# Create a grid from torch tensot
def grid_from_torch(X, Y, Z=None, resx=50, resy=50, resz=50, device='cpu'):
  xmin = torch.min(X).item()
  xmax = torch.max(X).item()
  ymin = torch.min(Y).item()
  ymax = torch.max(Y).item()

  dx = xmax - xmin
  dy = ymax - ymin

  if Z is None: # 2D case
    ed = 0.1*math.sqrt(dx*dx+dy*dy)

    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))

    xx, yy = torch.meshgrid(x, y)

    return xx.to(device), yy.to(device)
  else: # 3D case
    zmin = torch.min(Z).item()
    zmax = torch.max(Z).item()

    dz = zmax - zmin

    ed = 0.1 * math.sqrt(dx*dx + dy*dy + dz*dz)

    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))
    z = torch.arange(zmin-ed, zmax+ed, step=(dz+2*ed)/float(resz))

    xx, yy, zz = torch.meshgrid(x, y, z)

    return xx.to(device), yy.to(device), zz.to(device)

#-------------------------------------------------------------------------
# SAMPLING

# Neural Network as function
def nn_sampling(model, xx, yy, zz=None, g_norm_output_path=None, device='cpu'):
  # Evaluate function on each grid point
  resx = xx.shape[0]
  resy = yy.shape[0]
  if zz is None:
    dimg = resx * resy
    tt = torch.stack((xx, yy), axis=2)
    tt = torch.reshape(tt, (dimg,2))
  else:
    resz = zz.shape[0]
    dimg = resx * resy * resz
    tt = torch.stack((xx, yy, zz), axis=3)
    tt = torch.reshape(tt, (dimg,3))
  
  z = model(tt)
  if zz is None:
    print(torch.reshape(z, (resx,resy))) 
  else: 
    print(torch.reshape(z, (resx,resy, resz)))

  # Compute grad on each grid point
  x = torch.autograd.Variable(tt, requires_grad=True)
  x = x.to(device)
  f = model(x)
  g = torch.autograd.grad(outputs=f, inputs=x, 
                      grad_outputs=torch.ones(f.size()).to(device), 
                      create_graph=True, retain_graph=True, 
                      only_inputs=True)[0]

  print()
  print("Grad on each grid point")
  g = torch.norm(g, dim=-1)

  min_g = torch.min(g).item()
  max_g = torch.max(g).item()
  print("Minimum value: " + str(min_g))
  print("Maximum value: " + str(max_g))

  if g_norm_output_path is not None:
    np.savetxt(g_norm_output_path, g.detach().cpu())
    print("Norm of gradient saved")

  if zz is None:
    return torch.reshape(z, (resx,resy))
  else:
    return torch.reshape(z, (resx, resy, resz))