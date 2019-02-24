import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

# mean and cov for the first dataset
mean1 = [0,0]
cov1  = [[1,0.1],[0.1,1]]

# mean and cov for the second dataset
mean2 = [1,1]
cov2  = [[1,-0.1],[-0.1,1]]

# initializing sample points
data1            = np.random.multivariate_normal(mean1,cov1,100)
data2            = np.random.multivariate_normal(mean2,cov2,100)
# Plotting
fig_gen_syn_data = plt.figure()
plot_gsd         = fig_gen_syn_data.add_subplot(111)
plot_gsd.scatter(data1[:,0],data1[:,1],c='blue')
plot_gsd.scatter(data2[:,0],data2[:,1],c='red')
fig_gen_syn_data.show()
