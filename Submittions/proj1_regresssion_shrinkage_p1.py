import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# generating x_i
x1 = np.random.uniform(low=-10,high=10,size=200)
x2 = np.random.uniform(low=-10,high=10,size=200)
x3 = np.random.uniform(low=-10,high=10,size=200)
x = np.array([x1,x2,x3])
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(x[0],x[1],x[2],c='r')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# shuffle the data for picking up training and testing data
# this X.T is 200 rows and 3 columns
np.random.shuffle(x.T)
# setting up other parameters
w = np.array([-0.8, 2.1, 1.5])[None]
b = 10
# loc=>mean scale=>standard diviation
eps = np.random.normal(loc=0, scale=np.sqrt(10) ,size=200)
# adding noise
y = np.dot(w,x)+b+eps

# splitting data into two parts
x_train = x.T[0:100] # its shape is (100,3)
x_test  = x.T[100:200]
# spliting y
y_train = y.T[0:100] # its shape is (100,1)
y_test  = y.T[100:200]