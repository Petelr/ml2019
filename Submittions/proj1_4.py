import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

mean       = [2,5]
cov        = [[1,-0.1],[-0.1,1]]
X          = np.random.multivariate_normal(mean,cov,size=10)
X_hat      = np.empty(X.shape)
X_hat[:,0] = (X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
X_hat[:,1] = (X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])

fig1       = plt.figure()
plot1      = fig1.add_subplot(111)
plot1.scatter(X[:,0],X[:,1], c='blue')
fig1.show()

fig2       = plt.figure()
plot2      = fig2.add_subplot(111)
plot2.scatter(X_hat[:,0],X_hat[:,1], c='red')
fig2.show()
