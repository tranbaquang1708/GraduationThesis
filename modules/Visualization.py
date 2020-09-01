import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize2(dataset, normal_vectors, xx, yy, z, scatter=True, vecfield=True, surface=True, offsurface=True, filled_contour=True):
  # Visualize for 2D
  num_on_points = int(len(dataset)/3)
  # Scatter plot for points
  if scatter:
    h_points = plt.scatter(dataset[:,0], dataset[:,1], s=2)
    plt.show()
  # Plot vector field
  if vecfield:
    h_vector = plt.quiver(dataset[0:num_on_points,0], dataset[0:num_on_points,1], normal_vectors[:,0], normal_vectors[:,1])
    h_vector.ax.axis('equal')
    plt.show()
  # Plot generated surface
  if surface:
    h_object = plt.contour(xx,yy, z, levels=[0.0], colors='c')
    h_object.ax.axis('equal')
    plt.show()
  # Plot generated surface for on surface and off surface points
  if offsurface:
    h_with_offsurface = plt.contour(xx,yy, z, levels=[-0.1, 0.0, 0.1])
    h_with_offsurface.ax.axis('equal')
    plt.show()
  # Plot filled contour
  if filled_contour:
    hf = plt.contourf(xx,yy,z)
    hf.ax.axis('equal')
    plt.show()

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