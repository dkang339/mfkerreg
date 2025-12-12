import numpy as np
from scipy.spatial.distance import cdist

'''
Stores kernel functions
Reference: 
    kernels: https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
'''

def expkernel(Z1,Z2,sigma=0.1):
    '''
    exponential kernel function
    inputs:
        Z1: input 2D array (N1,d_in)
        Z2: input 2D array (N2,d_in)
    sigma: bandwidth parameter (scalar)

    outputs:
        K: kernel matrix (N1,N2)
    '''
    dist = cdist(Z1,Z2,'sqeuclidean') # dist[i][j] = distance btw Z1[i] and Z2[j]
    K = np.exp(-dist/sigma)

    return K

def ardmatern32(Z1,Z2,sigma=0.1):
    '''
    ARD Matern 3/2 kernel function
    Note: ignored sigma_f since it's canceled out in the evaluation

    inputs:
        Z1: input 2D array (N1,d_in)
        Z2: input 2D array (N2,d_in)
    sigma: bandwidth parameter (scalar) if d_in=1 with single output
        or (d_in,) if ard kernel with d_in inputs and sigle output
        or (d_in,r) if ard kernel with d_in inputs and multi-output

    outputs:
        K: kernel matrix (N1,N2)
    '''
    Z1 = np.atleast_2d(Z1)
    Z2 = np.atleast_2d(Z2)
    d = Z1.shape[1] # number of inputs (d_in)

    # sigma = np.asarray(sigma).reshape(-1)
    sigma = np.asarray(sigma)
    if sigma.ndim == 0 or (sigma.ndim == 1 and sigma.size in (1, d)): # if single output
        sigma = np.atleast_1d(sigma)
        if sigma.size == 1:
            sigma = np.repeat(sigma.item(),d) # use the same sigma across the inputs (d_in,)    
        dist = cdist(Z1,Z2,'seuclidean',V=sigma**2)
        K = (1 + np.sqrt(3) * dist)*np.exp(-np.sqrt(3) * dist)
        return K # (N1,N2)
    
    
    elif sigma.ndim == 2: # if multi-output
        r = sigma.shape[1] # number of outputs (r)
        N1, N2 = Z1.shape[0], Z2.shape[0] # number of points in Z1 and Z2
        Kall = np.empty((r, N1, N2)) # initialize kernel matrix (r,N1,N2)
        for i in range(r): # loop over r different outputs
            V=sigma[:,i]**2 #(d_in,)
            dist = cdist(Z1,Z2,'seuclidean',V=V) # (N1,N2)
            K = (1 + np.sqrt(3) * dist)*np.exp(-np.sqrt(3) * dist)
            Kall[i] = K
        return Kall #(r,N1,N2)
