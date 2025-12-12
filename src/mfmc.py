import numpy as np
import scipy.linalg as la
import sys
from pathlib import Path
import h5py

current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get src directory

sys.path.append(str(root_dir / 'src'))


nf = 2  # number of fidelities

def alloc(sigma, rho, w, p):
    '''
    allocate samples and weights for MFMC estimator
    inputs:
        sigma: standard deviation of each fidelity per mode (nf,)
        rho: correlation coefficient per mode (nf,)
    outputs:
        alpha: weights (scalar)
        m: number of samples for each fidelity (nf,)
    '''

    m = np.zeros(nf)
    temp = rho[:-1]*sigma[0]/sigma[:]
    alpha = temp[1]
    temp = (rho[:-1]**2 - rho[1:]**2)
    const = np.sqrt(w[0]*temp/(w*(1-rho[1]**2))) # (nf,)
    m[0] = p/(w.transpose() @ const);
    m[1] = m[0]*const[1:];

    m = np.floor(m)
    m = m.astype(int)

    return alpha, m, const