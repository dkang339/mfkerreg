import numpy as np
from kernels import *
from scipy.optimize import minimize
from types import SimpleNamespace

'''
Kernel regression (w/o regularization) or Nadaraya-Watson
Reference: 
    kernels: https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
    cdist: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
'''



def eval_kr(z,Z,Y,sigma,kernel=ardmatern32):
    '''
    inputs:
        z: test data (n,d_in)
        Z: training data (N,d_in)
        Y: (N,r) output data

    outputs:
        Yhat: evaluated output at z (n,r) 
    '''
    numer = kernel(Z,z,sigma).T@Y # (n,r)

    denom_temp = kernel(Z,z,sigma) # (N,n)
    denom = np.sum(denom_temp,axis=0) # (n,)

    npts = len(Z[:,0]) # number of training points (N)
    eps = npts * np.finfo(float).eps # the same value with GPR code
    Yhat = numer/ (denom[:,None] + eps) # (n,r)

    return Yhat


def eval_mfkr(z,Z,Y,sigma,m,alpha,kernel=ardmatern32):
    '''
    inputs:
        z: test data (n,d_in)
        Z: training data (N_all,d_in)
            N_all: total number of training samples across all fidelities
        Y: (N,r,nf) output data
            nf: number of fidelities
        sigma: kernel bandwidth parameter (SimpleNamespace)

    outputs:
        Yhat: evaluated output at z (n,r) 
    '''
    
    nY = len(Y[0,0,:]) # number of fidelities
    d_in = len(Z[0,:]) # input dimension

    # get high fidelity mean estimates
    Z0 = Z[:m[0],:] # get high fidelity (m[0],d_in)
    Y0 = Y[:m[0],:,0] # get high fidelity output (m[0],r)
    sigma0 = getattr(sigma,'sigma0') # get high fidelity sigma 
    Yhat = eval_kr(z,Z0,Y0,sigma0,kernel) # (n,r)

    for i in range(1,nY):
        Zi = Z[:m[i],:] # get m_i inputs (m[i],d_in)
        Zim = Z[:m[i-1],:] # get m_i-1 inputs (m[i-1],d_in)
        sigmai = getattr(sigma, f'sigma{i}')

        Yhat1 = eval_kr(z,Zi,Y[:m[i],:,i],sigmai[:d_in],kernel) # (m[i],r)
        Yhat2 = eval_kr(z,Zim,Y[:m[i-1],:,i],sigmai[d_in:],kernel) # (m[i],r)

        Yhat += alpha[i] * (Yhat1 - Yhat2) # (n,r)

    return Yhat

def loocv_err(sigma,Z,Y,kernel=ardmatern32):
    '''
    compute leave-one-out cross-validation error

    inputs:
        Z: training data (N,d_in)
        Y: output data (N,r)
        sigma: bandwidth parameter (scalar)

    outputs:
        err: leave-one-out cross-validation error (scalar)

    '''

    K = kernel(Z,Z,sigma)
    if K.ndim != 2:
        raise ValueError("loocv_err expects a kernel matrix with shape (N, N).")

    Kdiag = np.diag(K)
    numer = K @ Y - Kdiag[:,None] * Y
    denom = np.sum(K,axis=1) - Kdiag

    N = len(Z) # number of training samples
    eps = N * np.finfo(float).eps
    Yhat = numer / (denom[:,None] + eps)
    err = np.mean(np.sum((Y - Yhat)**2,axis=1))

    return err


def find_sigma(Z,Y,kernel=ardmatern32,ard=True):
    '''
    Find the optimal kernel parameter
    inputs:
        Z: training data (N,d_in)
        Y: output data (N,r)
    outputs:
        sigma: optimal bandwidth parameter (scalar) or (d_in,)
            Note: if ard=True, then one sigma for each input feature (d_in,)
    '''

    if ard==True:
        # if ARD, one sigma for each input feature
        x0 = np.full(Z.shape[1],0.1)
        bounds = [(1e-3,1e3)] * Z.shape[1]

    else:
        x0 = 0.1
        bounds = [(1e-3,1e3)]
    
    res = minimize(
        lambda x: loocv_err(x,Z,Y,kernel),
        x0=x0,
        bounds=bounds
        )

    return res.x if ard else res.x[0] # optimal sigma

def find_mfsigma(Z,Y,m,kernel=ardmatern32,ard=True):
    '''
    Find the optimal kernel parameter
    inputs:
        Z: training data (N_all,d_in)
            N_all: total number of training samples across all fidelities
        Y: (N,r,nf) output data
            nf: number of fidelities
    outputs:
        sigma: optimal bandwidth parameter (scalar) or (d_in,) in SimpleNamespace
            Note: if ard=True, then one sigma for each input feature (d_in,)
    '''
    sigma = SimpleNamespace()

    nY = len(Y[0,0,:]) # number of fidelities

    # get kernel parameter for high fidelity
    if ard==True:
        # if ARD, one sigma for each input feature
        x0 = np.full(Z.shape[1],0.1)
        bounds = [(1e-3,1e3)] * Z.shape[1]

    else:
        x0 = 0.1
        bounds = [(1e-3,1e3)]
    
    res = minimize(
        lambda x: loocv_err(x,Z[:m[0],:],Y[:m[0],:,0],kernel),
        x0=x0,
        bounds=bounds
        )
    
    sigma.sigma0 = res.x if ard else res.x[0]


    for i in range(1,nY):

        Zi = Z[:m[i],:] # get m_i inputs (m[i],d_in)
        Zim = Z[:m[i-1],:] # get m_i-1 inputs (m[i-1],d_in)
        
        sol = []
        res = minimize(
            lambda x: loocv_err(x,Zi,Y[:m[i],:,i],kernel),
            x0=x0,
            bounds=bounds
            )
        sol.append(res.x if ard else res.x[0])

        res = minimize(
            lambda x: loocv_err(x,Zim,Y[:m[i-1],:,i],kernel),
            x0=x0,
            bounds=bounds
            )
        sol.append(res.x if ard else res.x[0])
        setattr(sigma, f'sigma{i}',np.concatenate(sol))

    return sigma # optimal sigma
