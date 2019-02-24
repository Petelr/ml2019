import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
from mpl_toolkits.mplot3d import Axes3D

x          = np.linspace(-5,5,100)
y          = np.linspace(-5,5,100)
X,Y        = np.meshgrid(x,y)

pos        = np.zeros(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y

mean       = [1,1]
cov        = [[2,1],[-1,1]]
var        = multivariate_normal(mean, cov)
fig0       = plt.figure(figsize=(8,8))
c_fig      = fig0.add_subplot(111)
c_fig.contourf(X,Y,var.pdf(pos))
fig0.show()

fig1       = plt.figure(figsize=(8,8))
surf_fig   = fig1.add_subplot(111,projection='3d')
surf_fig.plot_surface(X,Y,var.pdf(pos))
fig1.show()