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
def grid_from_torch(X, Y, resx=50, resy=50, device='cpu'):
  xmin = torch.min(X).item()
  xmax = torch.max(X).item()
  ymin = torch.min(Y).item()
  ymax = torch.max(Y).item()

  dx = xmax - xmin
  dy = ymax - ymin

  # resx = 50
  # resy = 100

  ed = 0.1*math.sqrt(dx*dx+dy*dy)

  x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
  y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))

  xx, yy = torch.meshgrid(x, y)

  return xx.to(device), yy.to(device)

#-------------------------------------------------------------------------
# SAMPLING

# Neural Network as function
def nn_sampling(model, xx, yy, device='cpu'):
  # Evaluate function on each grid point
  resx = xx.shape[0]
  resy = yy.shape[1]
  dimg = resx * resy
  z = torch.empty((0,1))
  tt = torch.stack((xx, yy), axis=2)
  # print(tt.size())
  tt = torch.reshape(tt, (dimg,2))
  z = model(tt)
  print("Value of function on each grid point")
  print(torch.reshape(z, (resx,resy)))

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
  print(torch.reshape(g, (resx, resy)))
  # print("Number of elements outside the range [-1.3, 1.3]: " + str((g>1.3).sum() + (g<-1.3).sum()))
  print("Minimum value: " + str(torch.min(g).item()))
  print("Maximum value: " + str(torch.max(g).item()))
  # print("Number of elements outside [0.7,1.3]: " + str(((g<0.7).sum() + (g>1.3).sum()).item()) + "/" + str(g.shape[0]))

  return torch.reshape(z, (resx,resy))