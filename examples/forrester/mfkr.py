'''
Run MF kernel regeression on a forrester example and compare 
with SF kernel regression
'''

import h5py
from types import SimpleNamespace
from joblib import parallel_backend, Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
import os
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get code directory
sys.path.append(str(root_dir / '..'))
sys.path.append(str(root_dir / '..' / 'src'))
from kr import *
from mfmc import alloc

plt_dir = f'plots/kr'
npy_dir = f'results/kr'
os.makedirs(plt_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)


def run_find_mfsigma(k):
    Xk = X[:,:,k] # (m[-1], 1)

    Yk = np.zeros((m[-1],d_out,nf))
    Yk[:m[0],:,0] = f.f1(Xk[:m[0]]).reshape(-1,1) # high fidelity output (m[0], d_out, 1)
    Yk[:m[1],:,1] = f.f2(Xk[:m[1]]).reshape(-1,1) # low fidelity output (m[1], d_out, 1)

    return find_mfsigma(Xk, Yk, m, kernel=ardmatern32, ard=True)

# --- set up
d = 2
a, b, c = 20, 0.2, 2*np.pi
l, u = -32.768, 32.768
f = SimpleNamespace()
f.f1 = lambda x: -a * np.exp(-b*np.sqrt(1/d * np.sum(x**2, axis=1))) - np.exp(1/d * np.sum(np.cos(c*x), axis=1)) + a + np.exp(1) # high fidelity
f.f2 = lambda x: -a * np.exp(-0.9*b*np.sqrt(1/d * np.sum(x**2, axis=1))) - np.exp(1/d * np.sum(np.sin(c*x), axis=1)) + a + np.exp(1) + 0.1* x[:,0] # low fidelity
nf = len(vars(f)) # number of fidelities
r = 1 # number of sample replicates
n_test = 900 # number of test samples
w = np.array([1, 0.001]).T # cost per each fidelity (nf,)
p = [50]
mse = np.zeros((len(p),r)) # initialize mean squared error
mse_mf = np.zeros((len(p),r)) # initialize mean squared error
alpha = np.zeros(nf,)
rho = np.zeros(nf+1)
dat = np.load("../../data/forrest/stats_forrest.npz", allow_pickle=True)
std, rho = dat['sigma'], dat['rho']
print('sigma:', std, 'rho:', rho)
model = {
    "w": w,
    "nf": nf,
    "rep": r,
    "p": p,
    "std": std,
    "rho": rho
}

# --- generate data
sq_n_test = int(np.sqrt(n_test))
x1 = np.linspace(l, u, sq_n_test)
x2 = np.linspace(l, u, sq_n_test)
X1, X2 = np.meshgrid(x1, x2, indexing="ij")
Xtest = np.stack([X1.ravel(), X2.ravel()], axis=1) # (n_test, 2)
ytest = f.f1(Xtest) # (n_test,)
ylow = f.f2(Xtest) # (n_test,)
d_out = 1 # output dimension (1,)


start = time.time()

for i in range(len(p)):

    alpha_temp, m, _ = alloc(std, rho, w, p[i])
    print(f"budget: {p[i]}, # samples: {m}, alpha: {alpha_temp}")
    alpha[1] = alpha_temp

    X = np.random.uniform(l, u, (m[-1], 2, r)) # (m[-1], 2, r)
    Y = np.zeros((m[-1],d_out,nf)) # initialize output data (m[-1],d_out,nf), note: no replicates

    X_sf = X[:p[i],:] # (n[i], 2, r)
    y = f.f1(X_sf) # (n[i],d_out)

    '''
    kernel regression
    '''
    stime1 = time.time()
    with parallel_backend('threading'):
        sigmas = Parallel(n_jobs=-1)(
            delayed(find_sigma)(X_sf[:,:,k],y[:,k].reshape(-1,1),
                                kernel=ardmatern32,ard=True)
            for k in range(r)
            )
    # print("sigmas: ", sigmas)
    
    ypred = np.zeros((n_test, r)) # initialize prediction (n_test, r)
    for j in range(r):
        ypred[:,j] = eval_kr(
            Xtest,X_sf[:,:,j],y[:,j].reshape(-1,1),sigmas[j],
            kernel=ardmatern32
            ).squeeze() # (n_test, 1)
        err = np.mean((ytest.squeeze() - ypred[:,j])**2) # (scalar)
        mse[i,j] = err # mean squared error (scalar)
    etime1 = time.time()
    print('Kernel regression time: ', etime1-stime1)

    '''
    MF kernel regression
    '''
    stime2 = time.time()
    with parallel_backend('threading'):
        sigma = Parallel(n_jobs=-1)(
            delayed(run_find_mfsigma)(k)
            for k in range(r)
            )

    ypred_mf = np.zeros((n_test, r)) # initialize prediction (n_test, r)
    for k in range(r):
        Xk = X[:,:,k] # (m[-1], 2)
        Yk = np.zeros((m[-1],d_out,nf))
        Yk[:m[0],:,0] = f.f1(Xk[:m[0]]).reshape(-1, 1) # high fidelity output (m[0], d_out, 1)
        Yk[:m[1],:,1] = f.f2(Xk[:m[1]]).reshape(-1, 1) # low fidelity output (m[1], d_out, 1)

        ypred_mf[:,k] = eval_mfkr(
            Xtest,Xk,Yk,sigma[k],m,alpha,
            kernel=ardmatern32
            ).squeeze() # (n_test, 1)
        err = np.mean((ytest.squeeze() - ypred_mf[:,k])**2) # (scalar)
        mse_mf[i,k] = err # mean squared error (scalar)
    etime1 = time.time()
    print('MF Kernel regression time: ', etime1-stime1)

    etime2 = time.time()

mean_mse = np.mean(mse,axis=1) # mean over replicates (len(n),)
mean_mse_mf = np.mean(mse_mf,axis=1) # mean over replicates (len(n),)
std_mse = np.std(mse,axis=1) # std over replicates (len(n),)
std_mse_mf = np.std(mse_mf,axis=1) # std over replicates (len(n),)

end = time.time()
print('Total time: ', end-start)

np.savez(f"{npy_dir}/mfkr_forrest.npz", 
         model=model, 
         mse_sf=mse, mse_mf=mse_mf, mean_mse_sf=mean_mse, mean_mse_mf=mean_mse_mf, 
         std_mse_sf=std_mse, std_mse_mf=std_mse_mf, 
         Xtest=Xtest.squeeze(), ytest=ytest.squeeze(), 
         ypred_sf=ypred, ypred_mf=ypred_mf )