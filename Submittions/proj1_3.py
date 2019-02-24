import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

uniform_var = np.random.uniform(-5,5,100)
target_val = 0.1 * (uniform_var**3) + 3
noise = np.random.normal(size=100)
noisy_obs = target_val + noise
# plotting
fig_noisy_data = plt.figure()
plot = fig_noisy_data.add_subplot(111)
plot.scatter(uniform_var,target_val,c='blue')
plot.scatter(uniform_var,noisy_obs,c='red')
fig_noisy_data.show()
