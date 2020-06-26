# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
from numpy import *
from scipy.spatial.distance import squareform,pdist

class Rbf(object):
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

  def sampling(xx, yy, w, func=None):
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