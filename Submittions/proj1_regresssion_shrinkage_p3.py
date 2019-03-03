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

def ols(x_train,y_train):
    # adding a column of ones to x_train
    ones        = np.ones(len(x_train))[None].T
    x_train     = np.column_stack((ones, x_train))
    # appling OLS to the training data
    the_inverse = np.linalg.inv(np.dot(x_train.T,x_train))
    temp_w      = np.dot(np.dot(the_inverse, x_train.T), y_train)
    
    return temp_w
    
temp_w    = ols(x_train,y_train)
bias      = temp_w[0]
initial_w = np.delete(temp_w,0,0)
# initial_w.shape, temp_w
print("The bias is ",float(bias))

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def lasso_regression(init_w, x, y,_lambda):
    w_est    = init_w
    Ge       = np.sign(w_est)
    first_it = True
    while(sum(np.abs(w_est))>=(1/_lambda)):
        if (first_it):
            first_it = False
            # defining H and f matrix from "Standard Form"
            H     = matrix(2*np.dot(x_train.T, x_train))      # 3x3
            f     = matrix(-2*np.dot(x_train.T, y_train))     # 3x1
            A     = matrix(np.sign(initial_w.T))
            h     = matrix((1/_lambda)*np.ones(len(Ge.T)))
            sol   = solvers.qp(H,f,A,h)
            w_est = sol['x']
        else:
            Ge = np.hstack((Ge, np.sign(w_est)))
            # defining H and f matrix from "Standard Form"
            H     = matrix(2*np.dot(x_train.T, x_train))      # 3x3
            f     = matrix(-2*np.dot(x_train.T, y_train))     # 3x1
            A     = matrix(Ge.T)
            h     = matrix((1/_lambda)*np.ones(len(Ge.T)))
            sol   = solvers.qp(H,f,A,h)
            w_est = np.array(sol['x'])
        
    return w_est

# setting up the lambda test array
lam = np.arange(0.1,10.1,0.1)
lam_plot = np.arange(0.1,10.2,0.1)
ans = initial_w
for item in lam:
    # x:100x3   y:100x1    initial_w:3x1
    temp_ans = lasso_regression(initial_w, x_train, y_train, item)
    ans = np.append(ans,temp_ans)
# print("Regression Completed!")

# r = np.append(initial_w.T, ans)
ss = ans.reshape(101,3)
fig = plt.figure(1)
pl = fig.add_subplot(111)
pl.plot(lam_plot,ss[:,0],label="w1")
pl.plot(lam_plot,ss[:,1],label="w2")
pl.plot(lam_plot,ss[:,2],label="w3")
pl.legend()
plt.xlabel("lambda")
plt.ylabel("w")
plt.title("Figure 3. Lasso Estimations of w verses lambda")
# plt.show()
# Calculating SSE

w_hat = ans[0:300].reshape(100,3)
y_hat = np.empty((100,100))
for i in range(100):
    y_hat[i] = np.dot(w_hat[i], x_test.T) + bias
    
sse = np.zeros((100,1))
for i in range(100):
    sse[i] = np.sum(np.power((y_test.T-y_hat[i]),2))

sse = np.sum(sse,axis=1)
sse.shape

figure4 = plt.figure(2)
plot4   = figure4.add_subplot(111)
plot4.set_title("Figure 4. LASSO SSE verses Lambda.")
plot4.plot(lam, sse)
print("im here")
plt.xlabel("lambda")
plt.ylabel("SSE")
plt.show()
