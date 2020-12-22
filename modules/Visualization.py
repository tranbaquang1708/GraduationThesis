import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import math
import numpy as np
from modules import Operation

#-------------------------------------------------------
# PLOT

# Scatter plot
def scatter_plot(points):
  if points.shape[1] == 2:
    h_points = plt.scatter(points[:,0], points[:,1], s=2)
    plt.show()
  else:
    fig = plt.figure(figsize=(10,10))
    ax_points = fig.add_subplot(111, projection='3d')
    ax_points.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()

# Line graph for loss value
def loss_graph(i, value):
  plt.plot(i, value)
  plt.xlabel('Epoch')
  plt.ylabel('Loss value')
  plt.title('Loss Value')
  plt.show()

# Visualization for 2D dataset
def visualize2(dataset, normal_vectors, xx, yy, z, scatter=True, vecfield=True, surface=True, filled_contour=True):
  # Visualize for 2D
  # Scatter plot for points
  if scatter:
    plt.figure(figsize=(8,4))
    h_points = plt.scatter(dataset[:,0], dataset[:,1], s=1)
    plt.title('Point-cloud')
    plt.show()
  # Plot vector field
  if vecfield:
    plt.figure(figsize=(8,4))
    h_vector = plt.quiver(dataset[:,0], dataset[:,1], normal_vectors[:,0], normal_vectors[:,1])
    h_vector.ax.axis('equal')
    plt.title('Normal Vectors')
    plt.show()
  # Plot generated surface
  if surface:
    plt.figure(figsize=(8,4))
    h_object = plt.contour(xx,yy, z, levels=[0.0], colors='c')
    h_object.ax.axis('equal')
    plt.title('Contour Plot')
    plt.show()
  # Plot filled contour
  if filled_contour:
    plt.figure(figsize=(8,4))
    hf = plt.contourf(xx,yy,z)
    hf.ax.axis('equal')
    plt.title('Filled Contour Plot')
    plt.show()

# Visualization for 3D dataset
def visualize3(dataset, normal_vectors, z, scatter=True, vecfield=True, surface=True):
  if scatter:
    fig = plt.figure(figsize=(10,10))
    ax_points = fig.add_subplot(111, projection='3d')
    ax_points.scatter(dataset[:,0], dataset[:,1], dataset[:,2], s=1)
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
def grid_from_torch(points, resx=50, resy=50, resz=50, device='cpu'):
  dims = points.shape[1]

  xmin = torch.min(points[:,0]).item()
  xmax = torch.max(points[:,0]).item()
  ymin = torch.min(points[:,1]).item()
  ymax = torch.max(points[:,1]).item()

  dx = xmax - xmin
  dy = ymax - ymin

  if dims == 2: # 2D case
    ed = 0.1*math.sqrt(dx*dx+dy*dy)

    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))

    xx, yy = torch.meshgrid(x, y)

    return xx.to(device), yy.to(device)
  else: # 3D case
    zmin = torch.min(points[:,2]).item()
    zmax = torch.max(points[:,2]).item()

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
def nn_sampling(model, xx, yy, zz=None, vtk_output_path=None, g_norm_output_path=None, device='cpu'):
  # Evaluate function on each grid point
  resx = xx.shape[0]
  resy = yy.shape[0]
  if zz is None:
    dimg = resx * resy
    tt = torch.stack((xx, yy), dim=-1).reshape(dimg,2)
  else:
    resz = zz.shape[0]
    dimg = resx * resy * resz
    tt = torch.stack((xx, yy, zz), dim=-1).reshape(dimg,3)
  
  z = model(tt)

  # Save z value
  if vtk_output_path is not None:
    Operation.save_vtk(vtk_output_path, tt, resx, resy, resz, z)
    print("VTK file saved")

  if zz is None:
    z = torch.reshape(z, (resx,resy))
  else: 
    z = torch.reshape(z, (resx,resy, resz))

  if g_norm_output_path is not None:
    # Compute grad on each grid point
    x = torch.autograd.Variable(tt, requires_grad=True)
    x = x.to(device)
    f = model(x)
    g = torch.autograd.grad(outputs=f, inputs=x, 
                        grad_outputs=torch.ones(f.size()).to(device), 
                        create_graph=True, retain_graph=True, 
                        only_inputs=True)[0]

    np.savetxt(g_norm_output_path, g.detach().cpu())
    print("Norm of gradient saved")

  return z