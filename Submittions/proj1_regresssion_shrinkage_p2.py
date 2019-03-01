import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# generating x_i
x1 = np.random.uniform(low=-10,high=10,size=200)
x2 = np.random.uniform(low=-10,high=10,size=200)
x3 = np.random.uniform(low=-10,high=10,size=200)
x = np.array([x1,x2,x3])

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

# standardizing test and training data
x_train = (x_train-np.mean(x_train))/np.std(x_train)
# y_train = (y_train-np.mean(y_train))/np.std(y_train)
x_test = (x_test-np.mean(x_test))/np.std(x_test)
# y_test = (y_test-np.mean(y_test))/np.std(y_test)
x_train.shape, y_train.shape, len(x_train.T)

# setting up the lambda test array
lambda_array = np.arange(0,10,0.1)

def rigid_regression(x_train, y_train, _lambda):
    ones    = np.ones(len(x_train))[None].T
    x_train = np.column_stack((ones, x_train))
    
    lambda_identity = _lambda*np.identity(len(x_train.T))
    the_inverse     = np.linalg.inv(np.dot(x_train.T,x_train) + lambda_identity)
    w_hat           = np.dot(np.dot(the_inverse, x_train.T),y_train)
    
    return w_hat

# Apply Rigid Regression
w_hat = []
for i in lambda_array:
    w_temp = rigid_regression(x_train,y_train,i)[None]
    w_hat = np.append(w_hat, w_temp.T)

w_hat = w_hat.reshape(len(lambda_array),4) # reshaping

# print(w_hat.shape)           
b_hat = w_hat[:,0]
w_hat = np.delete(w_hat,0,1).T
w_hat.shape, x_test.shape

# Calculate estimation of y
y_hat = np.empty((100,100))
for i in range(100):
    y_hat[i] = np.dot(w_hat.T[i], x_test.T) + b_hat

# y_hat = np.dot(w_hat.T, x_test.T) + b_hat
# y_hat = np.dot(x_test, w_hat) + b_hat

# y_hat.shape,np.dot(w_hat.T, x_test.T).shape
# y_hat[0] for the first lambda
y_test.shape, y_hat.shape

# Calculate SSE
sse = np.zeros((100,1))
for i in range(100):
    sse[i] = np.sum(np.power((y_test.T-y_hat[i]),2))

sse = np.sum(sse,axis=1)
sse.shape

# First plot
# Plotting SSE verses lambda
figure1 = plt.figure(1)
plot1   = figure1.add_subplot(111)
plot1.set_title("Figure 1. SSE verses Lambda.")
plot1.plot(lambda_array, sse)
plt.xlabel("lambda")
plt.ylabel("SSE")

# Second plot
# plotting w verses lambda
figure2 = plt.figure(2)
plot2   = figure2.add_subplot(111)
plot2.set_title("Figure 2. Estimations of w verses lambda")
plot2.plot(lambda_array, w_hat[0],c="blue", label="w1")
plot2.plot(lambda_array, w_hat[1],c="orange", label="w2")
plot2.plot(lambda_array, w_hat[2],c="black", label="w3")
plt.legend()
plt.xlabel("lambda")
plt.ylabel("w")
plt.show()

print("The estimation of b is ",np.mean(b_hat))