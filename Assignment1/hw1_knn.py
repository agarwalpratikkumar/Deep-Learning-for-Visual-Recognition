# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    return np.array([[np.sqrt(np.sum(np.power((i-j),2))) for i in X] for j in Y])
 

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    pred_labels = np.zeros(len(dists))
    for i,v in enumerate(dists):
        nearest_neighbours = labels[np.argsort(v)]
        k_nearest_neighbour = nearest_neighbours[:k]  # [:k] means get a slice from the start to index k. Here [0]
        neighbour, count = np.unique(k_nearest_neighbour, return_counts=True)
        pred_labels[i] = neighbour[np.argmax(count)]

    return pred_labels
     