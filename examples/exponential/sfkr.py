'''
Run kernel regeression on a toy example
'''
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
import os
current_dir = Path(__file__).parent # get current directory
root_dir = current_dir.parent.resolve() # get code directory
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'src'))
from aux import cleanup_loky
cleanup_loky

from kr import *
plt_dir = f'plots'
npy_dir = f'results'
os.makedirs(os.path.join(plt_dir), exist_ok=True)
os.makedirs(os.path.join(npy_dir), exist_ok=True)

# --- set up
r = 100 # number of sample replicates
n_test = 200 # number of test samples
n = [10, 20, 30, 50, 100, 200, 400]
mse = np.zeros((len(n),r)) # initialize mean squared error

# --- generate data
np.random.seed(0)
# Xtest = np.linspace(0, 10, n_test).reshape(-1, 1) # (n_test, 1)
# ytrue = np.sin(np.linspace(0, 10, n_test)).reshape(-1,1) # true function (n_test, 1))

f = lambda x: np.exp(x) # highfi function
Xtest = np.linspace(0, 5, n_test).reshape(-1, 1) # (n_test, 1)
ytrue = f(Xtest) # true function (n_test, 1))


'''
kernel regression
'''
ypred = np.zeros((len(n), r, n_test)) # initialize prediction (len(n), r, n_test)
start = time.time()
for i in range(len(n)):
    # X = np.random.uniform(0, 10, (n[i],r)) # (n, r)
    # y = np.sin(X) + np.random.normal(0, 0.5, size=X.shape) # (n, r)
    X = np.random.uniform(0, 5, (n[i],r)) # (n, r)
    y = f(X) # (n, r)

    # --- kernel regression training ---
    opt_s = time.time()
    sigmas = Parallel(n_jobs=-1)(
        delayed(find_sigma)(X[:,k].reshape(-1,1),y[:,k].reshape(-1,1))
        for k in range(r)
        )
    opt_e = time.time()
    # print("sigmas: ", sigmas)
    # print(f"kernel para opt time for {r} replicates:", opt_e - opt_s)
    
    for j in range(r):
        ypred[i,j,:] = eval_kr(
            Xtest,X[:,j].reshape(-1,1),y[:,j].reshape(-1,1),sigmas[j]
            ).ravel() # (n_test,)
        err = np.sum((ytrue.ravel() - ypred)**2,axis=0) # (scalar)
        mse[i,j] = np.mean(err) # mean squared error (scalar)
    
mean_mse = np.mean(mse,axis=1) # average over replicates (len(n),)

end = time.time()
print(f"total time:", end - start)


'''
plots
'''
# plot prediction
  # plot 
plt.figure()
# plt.scatter(X[:,-1], y[:,-1], color='c', label='data', alpha=0.5)
for i in range(r):
    plt.plot(Xtest, ypred[1,i,:], color='magenta', linestyle = 'solid', alpha=0.3)
plt.plot(Xtest, ytrue, color='black', label='true function')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(f"{plt_dir}/kr_pred.png")


# plot mse (compared to truth)
plt.figure()
plt.loglog(n, mean_mse, marker='o')
plt.xlabel("number of data")
plt.ylabel(f"mean squared error ({r} replicates)")
plt.savefig(f"{plt_dir}/kr_mse.png")