'''
Run MF kernel regeression on a exponential example and compare 
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
    Xk = X[:,k].reshape(-1,1) # (m[-1], 1)

    Yk = np.zeros((m[-1],d_out,nf))
    Yk[:m[0],:,0] = f.f1(Xk[:m[0]]) # high fidelity output (m[0], d_out, 1)
    Yk[:m[1],:,1] = f.f2(Xk[:m[1]]) # low fidelity output (m[1], d_out, 1)

    return find_mfsigma(Xk, Yk, m, kernel=ardmatern32, ard=True)

# --- set up
f = SimpleNamespace()
f.f1 = lambda x: np.exp(x) # highfi function
f.f2 = lambda x: 0.9*np.sqrt(f.f1(x)) # lowfi function
nf = len(vars(f)) # number of fidelities
r = 50 # number of sample replicates
n_test = 100 # number of test samples
w = np.array([1, 0.001]).T # cost per each fidelity (nf,)
p = [100]
mse = np.zeros((len(p),r)) # initialize mean squared error
mse_mf = np.zeros((len(p),r)) # initialize mean squared error
alpha = np.zeros(nf,)
rho = np.zeros(nf+1)
dat = np.load("../../data/exponential/stats_exp.npz", allow_pickle=True)
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
Xtest = np.linspace(0, 5, n_test).reshape(-1, 1) # (n_test, 1)
ytest = f.f1(Xtest) # true function (n_test, 1))
d_out = len(ytest[0,:]) # output dimension (1,)


start = time.time()

for i in range(len(p)):

    alpha_temp, m, _ = alloc(std, rho, w, p[i])
    alpha[1] = alpha_temp

    X = np.random.uniform(0, 5, (m[-1],r)) # (m[-1], r)
    Y = np.zeros((m[-1],d_out,nf)) # initialize output data (m[-1],d_out,nf), note: no replicates

    X_sf = X[:p[i],:] # (n[i], r)
    y = f.f1(X_sf) # (n[i],d_out)

    '''
    kernel regression
    '''
    stime1 = time.time()
    with parallel_backend('loky', inner_max_num_threads=1):
        sigmas = Parallel(n_jobs=-1)(
            delayed(find_sigma)(X_sf[:,k].reshape(-1,1),y[:,k].reshape(-1,1),
                                kernel=ardmatern32,ard=True)
            for k in range(r)
            )
    # print("sigmas: ", sigmas)
    
    ypred = np.zeros((n_test, r)) # initialize prediction (n_test, r)
    for j in range(r):
        ypred[:,j] = eval_kr(
            Xtest,X_sf[:,j].reshape(-1,1),y[:,j].reshape(-1,1),sigmas[j],
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
    with parallel_backend('loky', inner_max_num_threads=1):
        sigma = Parallel(n_jobs=-1)(
            delayed(run_find_mfsigma)(k)
            for k in range(r)
            )

    ypred_mf = np.zeros((n_test, r)) # initialize prediction (n_test, r)
    for k in range(r):
        Xk = X[:,k].reshape(-1,1) # (m[-1], 1)
        Yk = np.zeros((m[-1],d_out,nf))
        Yk[:m[0],0,0] = f.f1(Xk[:m[0]]).squeeze() # high fidelity output (m[0], d_out, 1)
        Yk[:m[1],0,1] = f.f2(Xk[:m[1]]).squeeze() # low fidelity output (m[1], d_out, 1)

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


'''
plots
'''
# plot prediction
plt.figure()
ax = plt.gca()
ax.plot(Xtest, ytest, color='black', label='True function')

for k in range(r):
    if k == 0:
        ax.plot(Xtest[:,0], ypred[:,k], color='magenta', label='Single fidelity', alpha=0.5)
        ax.plot(Xtest[:,0], ypred_mf[:,k], color='blue', label=f'Multifidelity', alpha=0.5)
    else:
        ax.plot(Xtest[:,0], ypred[:,k], color='magenta', alpha=0.5)
        ax.plot(Xtest[:,0], ypred_mf[:,k], color='blue', alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(f"{plt_dir}/mfkr_pred.png")



np.savez(f"{npy_dir}/mfkr_exp.npz",
         model=model,
         mse_sf=mse,
         mse_mf=mse_mf,
         mean_mse_sf=mean_mse,
         mean_mse_mf=mean_mse_mf,
         std_mse_sf=std_mse,
         std_mse_mf=std_mse_mf,
         Xtest=Xtest.squeeze(),
         ytest=ytest.squeeze(),
         ypred_sf=ypred,
         ypred_mf=ypred_mf
         )

end = time.time()