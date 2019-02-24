import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X,Y = np.meshgrid(x,y)
S = np.meshgrid(x,y)

pos = np.zeros(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
var = multivariate_normal([1,1], [[2,-1],[-1,1]])
plt.contourf(X,Y,var.pdf(pos))
plt.show()