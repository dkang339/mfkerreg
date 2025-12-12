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
data_dir = '../../data/exponential'
os.makedirs(data_dir, exist_ok=True)

f1 = lambda x: np.exp(x) # high fidelity
f2 = lambda x: 0.9*np.exp(0.5*x) # low fidelity
a, b = 0, 5 # input domain

def stats(npts=50):
    ''''
    compute stats for MFMC estimator using gaussian quadrature
    outputs:
        sigma: standard deviation of each fidelity per mode (nf,)
        rho: correlation coefficient per mode (nf,)
    '''

    nf = 2  # number of fidelities

    x, w = leggauss(npts) # draw GQ points and weights
    X = (b-a)/2 * x + (b+a)/2 # rescale to [a, b]
    W = (b-a)/2 * w

    y1 = f1(X).ravel() # (npts,)
    y2 = f2(X).ravel() # (npts,)

    mean1 = np.sum(W* y1) / 5 # note: denom is for pdf
    mean2 = np.sum(W* y2) / 5

    var1 = np.sum(W * (y1 - mean1)**2) / 5 # scalar
    var2 = np.sum(W * (y2 - mean2)**2) / 5 # scalar
    sigma = np.sqrt([var1, var2]) # (nf,)
    cov12 = np.sum(W * (y1 - mean1)*(y2 - mean2)) / 5 # scalar

    # get correlation coefficient between highfi and lowfis
    rho = np.zeros(nf+1) # (nf+1,)
    rho[0] = 1.0
    rho[1] = cov12 / (sigma[0]*sigma[1])
    print('sigma:', sigma, 'rho:', rho)
    np.savez(f"{data_dir}/stats_exp.npz", sigma=sigma, rho=rho)

def save_data(fun, save_path, n_data=int(1e6), rid=42):
    '''
    This function saves Ishigami function data in h5 format.

    inputs:
    - fun: function handle
    - save_path: path to save the processed h5 file (string)
    - n_data: number of data to be generated
    '''

    rng = np.random.default_rng(rid)

    x = rng.uniform(a, b, (int(n_data), 1)) # (n_data, d_in=1)
    y = fun(x).reshape(-1,1) # (n_data,1)

    # save necessary data
    with h5py.File(save_path, "w") as f:
        f.create_dataset("input", data=x, compression="gzip", compression_opts=9)
        f.create_dataset("output", data=y, compression="gzip", compression_opts=9)


def main():
    save_data(f1, f'{data_dir}/highfi.h5', n_data=int(1e4))
    save_data(f2, f'{data_dir}/lowfi1.h5', n_data=int(1e4))
    save_data(f1, f'{data_dir}/test.h5', n_data=int(1e3), rid=100)
    stats()

if __name__ == "__main__":
    main()


