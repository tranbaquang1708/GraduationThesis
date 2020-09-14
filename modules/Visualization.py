import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import math

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

def scatter_plot(dataset):
  h_points = plt.scatter(dataset[:,0], dataset[:,1], s=2)
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
def grid_from_torch(X, Y, device):
  xmin = torch.min(X).item()
  xmax = torch.max(X).item()
  ymin = torch.min(Y).item()
  ymax = torch.max(Y).item()

  dx = xmax - xmin
  dy = ymax - ymin

  resx = 50
  resy = 50

  ed = 0.1*math.sqrt(dx*dx+dy*dy)

  x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
  y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))

  xx, yy = torch.meshgrid(x, y)
  return xx.to(device), yy.to(device)

#-------------------------------------------------------------------------
# SAMPLING

# Neural Network as function
def nn_sampling(nn, xx, yy):
  dimg = (xx.shape[0])**2
  z = torch.empty((0,1))
  tt = torch.stack((xx, yy), axis=2)
  tt = torch.reshape(tt, (dimg,2))
  z = nn(tt)
  print(torch.reshape(z, (50,50)))

  return torch.reshape(z, (50,50))