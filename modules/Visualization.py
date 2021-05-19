import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import math
import numpy as np
from modules import Utils


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
def loss_graph(i, values):
  plt.figure(figsize=(12,10))
  plt.subplot(2, 2, 1)
  plt.plot(i, values[:, 0], label='Total')
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.title('Total Loss Value')

  plt.subplot(2, 2, 2)
  plt.plot(i, values[:, 1])
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.title('Geo loss')

  plt.subplot(2, 2, 3)
  plt.plot(i, values[:, 2])
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.title('Normal Loss')

  plt.subplot(2, 2, 4)
  plt.plot(i, values[:, 3], label='Constraint')
  plt.xlabel('Iterations')
  plt.ylabel('Value')
  plt.title('Constraint Loss')

  plt.show()

  try:
    plt.plot(i, values[:, 4], label='Constraint')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Off-surface Loss')
    plt.show()
  except:
    print()

def show_loss_figure(loss_path):
  loss_value = np.load(loss_path)
  print(loss_value)
  loss_graph(loss_value[:,0], loss_value[:,1:])



# Plot constraint and Tucker Normalization
def plot_additional(model, xx, yy, zz=None):
  resx = xx.shape[0]
  resy = yy.shape[1]

  if zz is None:
    dimg = resx * resy
    tt = torch.stack((xx, yy), dim=-1).reshape(dimg,2)
  else:
    resz = zz.shape[2]
    dimg = resx * resy * resz
    tt = torch.stack((xx, yy, zz), dim=-1).reshape(dimg,3)

  tt.requires_grad = True

  # Plot Tucker Normalization
  # z = Utils.tucker_normalize(model(tt), tt)
  # z = torch.reshape(z, (resx,resy))
  # plt.figure(figsize=(8,4))
  # hf = plt.contourf(xx.detach().cpu().numpy(),yy.detach().cpu().numpy(),z.detach().cpu().numpy())
  # plt.colorbar(ticks=[0])
  # hf.ax.axis('equal')
  # plt.title('Tucker Normalized')
  # plt.show()

  lap = Utils.compute_laplacian(model(tt), tt, p=2)

  # Plot Laplacian on the grid
  plt.figure(figsize=(8,4))
  lap_grid = torch.reshape(lap, (resx,resy))
  h_lap_ctf = plt.contourf(xx.detach().cpu(),yy.detach().cpu(),lap_grid.detach().cpu())
  plt.colorbar(ticks=[lap.min().item(), 0., 1., lap.max().item()])
  h_lap_ctf.ax.axis('equal')
  plt.title('Laplacian Filled Contour Plot')
  plt.show()
  
  # Plot histograms of Laplacian values on the grid
  plt.hist(lap.detach().cpu().numpy())
  plt.title('Laplacian')
  plt.show()
  plt.hist(lap[lap.abs()<2].detach().cpu().numpy())
  plt.show()



# Visualization for 2D dataset
def plot2(dataset, normal_vectors, xx, yy, z, scatter, vecfield, surface, filled_contour, func_eval):
  # Points
  if scatter:
    plt.figure(figsize=(8,4))
    h_points = plt.scatter(dataset[:,0], dataset[:,1], s=1)
    plt.title('Point-cloud')
    plt.show()
  # Vector field
  if vecfield:
    plt.figure(figsize=(8,4))
    h_vector = plt.quiver(dataset[:,0], dataset[:,1], normal_vectors[:,0], normal_vectors[:,1])
    h_vector.ax.axis('equal')
    plt.title('Normal Vectors')
    plt.show()
  # Surface
  if surface:
    plt.figure(figsize=(8,4))
    h_object = plt.contour(xx,yy, z, levels=[0.])
    h_object.ax.axis('equal')
    plt.title('Contour Plot')
    plt.show()
  # Filled contour
  if filled_contour:
    plt.figure(figsize=(8,4))
    hf = plt.contourf(xx,yy,z)
    plt.colorbar(ticks=[z.min(), 0., z.max()])
    hf.ax.axis('equal')
    h_o = plt.contour(xx,yy, z, levels=[0.], colors='black')
    h_o.ax.axis('equal')
    plt.title('Filled Contour Plot')
    plt.show()
  if func_eval == True:
    plt.figure(figsize=(8,4))
    h_object = plt.contour(xx, yy, z, levels=[0.])
    h_object.ax.axis('equal')
    h_points = plt.scatter(dataset[:,0], dataset[:,1], s=2)
    plt.show()

def visualize2(model, data, resx=64, resy=64, 
                constraint_output_path=None, vtk_output_path=None,
                scatter=True, vecfield=True, surface=True, filled_contour=True, 
                additional=True, func_eval=True,
                device='cpu'):
  xx, yy = grid_from_torch(data[:,0:2], resx, resy, device=device)
  z = nn_sampling(model, xx, yy, zz=None,
                  constraint_output_path=constraint_output_path,
                  vtk_output_path = vtk_output_path,
                  device=device)

  plot2(data[:,0:2].detach().cpu().numpy(), data[:,2:4].detach().cpu().numpy(), 
        xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), z.detach().cpu().numpy(), 
        scatter=scatter, vecfield=vecfield, surface=surface, filled_contour=filled_contour, func_eval=func_eval)

  if additional == True:
    plot_additional(model, xx, yy)



def trimesh_visualize2(model, data, triangles, resx=64, resy=64, 
                constraint_output_path=None, vtk_output_path=None,
                scatter=True, vecfield=True, surface=True, filled_contour=True, 
                additional=True, func_eval=True,
                device='cpu'):
  
  is_boundary = data[:,-1]==1
  points = data[:,0:2]
  boundary_points = points[is_boundary]
  inside_points = points[~is_boundary]
  
  z = model(points).flatten()

  if surface:
    ht = plt.tricontour(data[:,0].detach().cpu(), data[:,1].detach().cpu(), triangles, z.detach().cpu(), levels=[0])
    plt.show()

  if filled_contour:
    htf = plt.tricontourf(data[:,0].detach().cpu(), data[:,1].detach().cpu(), triangles, z.detach().cpu())
    plt.colorbar(ticks=[z.min().item(), 0., z.max().item()])
    plt.show()

  if func_eval:
    ht = plt.tricontour(data[:,0].detach().cpu(), data[:,1].detach().cpu(), triangles, z.detach().cpu(), levels=[0])
    plt.scatter(boundary_points[:,0].detach().cpu(), boundary_points[:,1].detach().cpu(), s=1)
    plt.show()

  if additional:
    lap = Utils.compute_laplacian(model(points), points, p=2).flatten()
    plt.title('Laplacian')
    hltf = plt.tricontourf(data[:,0].detach().cpu(), data[:,1].detach().cpu(), triangles, lap.detach().cpu())
    plt.colorbar(ticks=[lap.min().item(), -1., 0., 1., lap.max().item()])
    plt.show()

    plt.hist(lap.detach().cpu().numpy())
    plt.show()

    plt.hist(lap[lap.abs()<2].detach().cpu().numpy())
    plt.show()



# Visualization for 3D dataset
def plot3(dataset, normal_vectors, z, scatter=True, vecfield=True, surface=True):
  if scatter:
    fig = plt.figure(figsize=(10,10))
    ax_points = fig.add_subplot(111, projection='3d')
    ax_points.scatter(dataset[:,0], dataset[:,1], dataset[:,2], s=1)
    plt.show()
  if vecfield:
    fig = plt.figure(figsize=(10,10))
    ax_vecfield = fig.add_subplot(111, projection='3d')
    rand_i = np.random.randint(len(dataset), size=500)
    ax_vecfield.quiver(dataset[rand_i,0], dataset[rand_i,1], dataset[rand_i,2], 
                       normal_vectors[rand_i,0], normal_vectors[rand_i,1], normal_vectors[rand_i,2])
    plt.show()
  if surface:
    verts, faces, normals, values = measure.marching_cubes_lewiner(z, 0)
    fig = plt.figure(figsize=(10,10))
    ax_surface = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax_surface.add_collection3d(mesh)
    ax_surface.set_xlim(0, z.shape[0])
    ax_surface.set_ylim(0, z.shape[1])
    ax_surface.set_zlim(0, z.shape[2])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10,10))
    ax_surface = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax_surface.add_collection3d(mesh)
    ax_surface.set_xlim(0, z.shape[0])
    ax_surface.set_ylim(0, z.shape[1])
    ax_surface.set_zlim(0, z.shape[2])
    plt.tight_layout()
    ax_surface.view_init(90,45)
    plt.show()

    fig = plt.figure(figsize=(10,10))
    ax_surface = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax_surface.add_collection3d(mesh)
    ax_surface.set_xlim(0, z.shape[0])
    ax_surface.set_ylim(0, z.shape[1])
    ax_surface.set_zlim(0, z.shape[2])
    plt.tight_layout()
    ax_surface.view_init(45,90)
    plt.show()

def visualize3(model, data, resx, resy, resz,
                constraint_output_path=None, vtk_output_path=None,
                scatter=True, vecfield=True, surface=True,
                additional=True,
                device='cpu'):
  xx, yy, zz = grid_from_torch(data, resx, resy, resz, device)
  z = nn_sampling(model, xx, yy, zz=zz,
                  constraint_output_path=constraint_output_path,
                  vtk_output_path = vtk_output_path,
                  device=device)
  plot3(data[:,0:3].detach().cpu().numpy(), data[:,3:6].detach().cpu().numpy(), 
        z.detach().cpu().numpy(), 
        scatter=scatter, vecfield=vecfield, surface=surface)

  if additional == True:
    plot_additional(model, xx, yy, zz)

#----------------------------------------------------------------------
# GRID

# Create a grid from torch tensot
def grid_from_torch(points, resx=32, resy=32, resz=32, device='cpu'):
  dims = points.shape[1]

  xmin = torch.min(points[:,0]).item()
  xmax = torch.max(points[:,0]).item()
  ymin = torch.min(points[:,1]).item()
  ymax = torch.max(points[:,1]).item()

  dx = xmax - xmin
  dy = ymax - ymin

  if dims == 2: # 2D case
    ed = 0.04 * math.sqrt(dx*dx+dy*dy)

    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))
    
    xx, yy = torch.meshgrid(x, y)

    return xx.to(device), yy.to(device)
  else: # 3D case
    zmin = torch.min(points[:,2]).item()
    zmax = torch.max(points[:,2]).item()

    dz = zmax - zmin

    ed = 0.04 * math.sqrt(dx*dx + dy*dy + dz*dz)

    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))
    z = torch.arange(zmin-ed, zmax+ed, step=(dz+2*ed)/float(resz))

    xx, yy, zz = torch.meshgrid(x, y, z)

    return xx.to(device), yy.to(device), zz.to(device)



#-------------------------------------------------------------------------
# Evaluate function on the grid

# Neural Network as function
def nn_sampling(model, xx, yy, zz=None, vtk_output_path=None, constraint_output_path=None, device='cpu'):
  # Evaluate function on each grid point
  resx = xx.shape[0]
  resy = yy.shape[1]
  if zz is None:
    resz = 1
    dimg = resx * resy
    tt = torch.stack((xx, yy), dim=-1).reshape(dimg,2)
  else:
    resz = zz.shape[2]
    dimg = resx * resy * resz
    tt = torch.stack((xx, yy, zz), dim=-1).reshape(dimg,3)
  
  z = model(tt)



  # Save z value
  if vtk_output_path is not None:
    Utils.save_vtk(vtk_output_path, tt, resx, resy, resz, z)
    print("VTK file saved")

  if constraint_output_path is not None:
    # Compute grad on each grid point
    g = Utils.compute_grad(z, tt)

    np.savetxt(g_norm_output_path, g.detach().cpu())
    print("Norm of gradient saved")



  if zz is None:
    z = torch.reshape(z, (resx,resy))
  else: 
    z = torch.reshape(z, (resx,resy, resz))

  return z