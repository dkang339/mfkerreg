'''
Preprocess exponential example data.
'''

from numpy.polynomial.legendre import leggauss
import sys
from pathlib import Path
import numpy as np
import os
import h5py
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / '../src'))
data_dir = '../../data/forrest'
os.makedirs(data_dir, exist_ok=True)

d = 2
a, b, c = 20, 0.2, 2*np.pi
f1 = lambda x: -a * np.exp(-b*np.sqrt(1/d * np.sum(x**2, axis=1))) - np.exp(1/d * np.sum(np.cos(c*x), axis=1)) + a + np.exp(1) # high fidelity
f2 = lambda x: -a * np.exp(-0.9*b*np.sqrt(1/d * np.sum(x**2, axis=1))) - np.exp(1/d * np.sum(np.sin(c*x), axis=1)) + a + np.exp(1) + 0.1* x[:,0] # low fidelity

l, u = -32.768, 32.768 # input domain
def stats(npts=100):
    ''''
    compute stats for MFMC estimator using gaussian quadrature
    outputs:
        sigma: standard deviation of each fidelity per mode (nf,)
        rho: correlation coefficient per mode (nf,)
    '''

    nf = 2  # number of fidelities

    x, w = leggauss(npts) # draw GQ points and weights
    x = (u-l)/2 * x + (u+l)/2 # rescale to [l,u]
    w = (u-l)/2 * w

    # gnerate tensor grids for f(x1,x2,x3)
    X1, X2 = np.meshgrid(x,x, indexing='ij') # (npts, npts)
    X = np.stack([X1.ravel(), X2.ravel()], axis=1) # (npts**2, 2)
    W = np.outer(w,w).ravel() # (npts**2,)

    y1 = f1(X).ravel() # (npts**2,)
    y2 = f2(X).ravel() # (npts**2,)

    mean1 = np.sum(W* y1) / (u-l)**2 # note: denom is for pdf
    mean2 = np.sum(W* y2) / (u-l)**2

    var1 = np.sum(W * (y1 - mean1)**2) / (u-l)**2 # scalar
    var2 = np.sum(W * (y2 - mean2)**2) / (u-l)**2 # scalar
    sigma = np.sqrt([var1, var2]) # (nf,)
    cov12 = np.sum(W * (y1 - mean1)*(y2 - mean2)) / (u-l)**2 # scalar

    # get correlation coefficient between highfi and lowfis
    rho = np.zeros(nf+1) # (nf+1,)
    rho[0] = 1.0
    rho[1] = cov12 / (sigma[0]*sigma[1])
    print('sigma:', sigma, 'rho:', rho)
    np.savez(f"{data_dir}/stats_forrest.npz", sigma=sigma, rho=rho)


def main():
    stats()

if __name__ == "__main__":
    main()


