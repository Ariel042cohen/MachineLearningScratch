# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:13:18 2021

@author: ariel
"""

import numpy as np

def cost(X, Y, theta):
    cost = 1/2 * np.sum((X@theta - Y)**2)
    return cost

def normaleq(X, Y):
    theta = np.linalg.inv(X.T@X) @ X.T@Y
    return theta

"""
1. 
"""

"""
a
"""

X = np.array([(31,22,40,26),(22, 21, 37, 25)]).T
print (X)
Y = np.array([(2,3,8,12)]).T

print (Y)
theta = normaleq(X, Y)
print (theta)
print (cost(X, Y, theta))

"""
b
"""
X_b = np.column_stack((np.ones(4),X))
print (X_b)

print (Y)
theta = normaleq(X_b, Y)
print (theta)

print (cost(X_b, Y, theta))

"""
c
"""
feature = ((X[:,0] - X[:,1]) **2)
X_c = np.column_stack((X, feature))
print (X_c)

print (Y)
theta = normaleq(X_c, Y)
print (theta)

print (cost(X_c, Y, theta))


"""
d
"""
X_d = np.column_stack((np.ones(4),X_c))
print (X_d)

print (Y)
theta = normaleq(X_d, Y)
print (theta)

print (cost(X_d, Y, theta))

"""
Gradient descent
"""

def gradient_descent(X, Y, theta, alpha, num_iters):
    m = X.shape[0];
    
    for x in range(num_iters):
        H = X@theta
        T = 1/m * ((H - Y).T@X)
        T = T.T
        theta = theta - (alpha * T)
        print (cost(X, Y, theta))
    return theta

def gradient_descent_momentum(X, Y, theta, alpha, momentum, num_iters):
    m = X.shape[0];
    change = 0
    
    for x in range(num_iters):
        H = X@theta
        T = 1/m * ((H - Y).T@X)
        T = T.T
        change = alpha * T + momentum * change
        theta = theta - change
        print (cost(X, Y, theta))
    return theta

# when I did np.log(X) the first value of X is zero so I didnt know what to do with log(0), I just putted 0
X = np.array([(1,1,1),(0,0,0.301), (0,1,4)]).T
Y = np.array([(1,3,7)]).T
Y = Y.reshape((3, 1))

theta = np.array([(2,2,0)]).T

theta = gradient_descent(X,Y,theta, 0.1, 1000)
print ('1 - 200 iterations ')
print (theta)

#theta = gradient_descent(X,Y,theta, 1, 200)
#print ('0.1 - 200 iterations ')
#print (theta)

#theta = gradient_descent(X,Y,theta, 0.01, 200)
#print ('0.01 - 100 iterations ')
#print (theta)

#theta = gradient_descent_momentum(X,Y,theta, 0.1,0.9, 200)
#print ('0.1 - 200 iterations ')
#print (theta)




