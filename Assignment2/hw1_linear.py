# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
import matplotlib.pyplot as plt

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(np.dot(X, W) + b)
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1 / (1 + np.exp(-a))
    

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    pred = predict(X, W, b)
    loss = np.sum(np.square(y - pred))
    gradient = (y - pred)*pred*(1-pred)
    #print(gradient.shape, X.shape)
    gradient = gradient.reshape(gradient.shape[0],1)
    #print(gradient.shape)
    gradient_weight = np.mean((-2*X)*gradient, axis=0)
    #print(gradient_weight.shape)
    gradient_bias = np.mean(-2*gradient, axis=0)
    #print(gradient_bias.shape)

    return loss, gradient_weight, gradient_bias


def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
    loss_total = []
    for i in range(num_iters):
        loss, g_w, g_b = l2loss(X,y,W,b)
        W = W - eta*g_w
        b = b - eta*g_b
        loss_total.append(loss)

    plt.plot(range(num_iters), loss_total)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.show()
    return W, b



 