# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
from numpy import *
from scipy.spatial.distance import squareform,pdist


def solve_w_func(dataset, offset=1, func=None):
  # Default RBF: Linear
  if func == None:
    func = Rbf.linear
  # A: the matrix made of evaluating the basis functions at each point from the data-set
  r = squareform(pdist(dataset)) # Pair-wise norm
  A = func(r)
  # b: the vector of the function values at the point from the data-set
  num_points = len(dataset)
  num_on_points = int(num_points/3)
  b = np.zeros(num_points);
  b[num_on_points:2*num_on_points] = -offset
  b[2*num_on_points:3*num_on_points] = offset
  return scipy.linalg.solve(A,b) # Solve linear equation A.x = b


def sampling(dataset, xx, yy, w, func=None):
  # Default RBF: Linear
  if func == None:
    func = Rbf.linear
  dimw = w.shape
  dimg = xx.shape
  z = np.zeros(dimg)
  for k in range(int(dimw[0])):
    z += w[k] * func(np.sqrt((xx - dataset[k,0])**2 + (yy - dataset[k,1])**2))
  return z


def linear(r):
  return r


def thin_plate(r):
  return (r*r*ma.log(r)).filled(0)


def gen_from_txt(filename, dim=3, offset=0.01):
  onsurface_points = np.zeros((0,dim))
  shifted_points = np.zeros((0,dim)) # onsurface_points left shifted by 1
  first_point = np.zeros((1,dim))
  last_point = np.zeros((1,dim))

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
      else:
        num_of_verticles[c] = num_of_verticles[c]-1
      shifted_points = np.concatenate((shifted_points,first_point))         # shifted_first_point

  # Vector of 2 consecutive points
  vectors = shifted_points - onsurface_points
  # Getting normal vectors
  norm = np.linalg.norm(vectors, axis=1)
  normal_vectors = np.ones_like(vectors)

  if dim == 2:
    normal_vectors[:,0] = np.divide(-vectors[:,1],norm)
    normal_vectors[:,1] = np.divide(vectors[:,0],norm)

  # Generate off surface point
  inside_points = onsurface_points - offset*normal_vectors
  outside_points = onsurface_points + offset*normal_vectors
  # First num_on_points point: On surface points
  # Next num_on_points points: Points inside circle
  # Last num_on_points points: Point outside circle
  dataset = np.concatenate((onsurface_points, inside_points, outside_points), axis=0)
  return dataset,normal_vectors

def meshgrid(dataset):
  xmin = np.min(dataset[:,0])
  xmax = np.max(dataset[:,0])
  ymin = np.min(dataset[:,1])
  ymax = np.max(dataset[:,1])
  dx = xmax - xmin
  dy = ymax - ymin
  # Slightly increase the grid by ed along each x and y direction
  ed = 0.1*np.sqrt(dx*dx+dy*dy)
  # Grid resolution
  resx = 50
  resy = 50
  x = np.arange(xmin-ed, xmax+ed, (dx+2*ed)/float(resx))
  y = np.arange(ymin-ed, ymax+ed, (dy+2*ed)/float(resy))
  return np.meshgrid(x, y, sparse=False)


def visualize(dataset, normal_vectors, xx, yy, z, scatter=True, vecfield=True, surface=True, offsurface=True, filled_contour=True):
  num_on_points = int(len(dataset)/3)
  # Scatter plot for points
  if scatter:
    h_points_on = plt.scatter(dataset[0:num_on_points,0], dataset[0:num_on_points,1], s=2)
    h_points_in = plt.scatter(dataset[num_on_points:(2*num_on_points),0], dataset[num_on_points:(2*num_on_points),1], s=2)
    h_points_out = plt.scatter(dataset[(2*num_on_points):(3*num_on_points),0], dataset[(2*num_on_points):(3*num_on_points),1], s=2)
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