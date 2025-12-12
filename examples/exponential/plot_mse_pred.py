import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

current_dir = Path(__file__).parent
root_dir = current_dir.resolve()
plt_dir = f'{root_dir}/plots/kr'
npy_dir = f'{root_dir}/results/kr'
os.makedirs(plt_dir, exist_ok=True)


# TODO: choose to run mfkr or load saved results
import mfkr

# --- load data ---
data = np.load(f"{npy_dir}/mfkr_exp.npz", allow_pickle=True)
Xtest = data["Xtest"]
ytest = data["ytest"]
ypred_sf = data["ypred_sf"]
ypred_mf = data["ypred_mf"]
x = Xtest
model = data["model"].item()
p = model["p"]
r = model["rep"]
mse_sf = data["mse_sf"]
mse_mf = data["mse_mf"]
mean_mse_sf = data["mean_mse_sf"]
mean_mse_mf = data["mean_mse_mf"]
std_mse_sf = data["std_mse_sf"]
std_mse_mf = data["std_mse_mf"]
ypred_sf = data["ypred_sf"]
ypred_mf = data["ypred_mf"]

'''
plots
'''

# plot mse (compared to truth)
plt.figure()
ax = plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
plt_kr, = ax.plot(p, mean_mse_sf,':o', color='r', label='Single fidelity')
# ax.plot(p, mean_mse_sf - std_mse_sf, linestyle=":", linewidth=0.8,
#         color=plt_kr.get_color(), alpha=0.6)
# ax.plot(p, mean_mse_sf + std_mse_sf, linestyle=":", linewidth=0.8,
#         color=plt_kr.get_color(), alpha=0.6)
ax.fill_between(
    p,
    mean_mse_sf - std_mse_sf,
    mean_mse_sf + std_mse_sf,
    color=plt_kr.get_color(),
    alpha=0.05
)

plt_mfkr, = ax.plot(p, mean_mse_mf, marker='o', color='b', label=f'Multifidelity')
ax.fill_between(
    p,
    mean_mse_mf - std_mse_mf,
    mean_mse_mf + std_mse_mf,
    color=plt_mfkr.get_color(),
    alpha=0.3
)
ax.legend()
ax.set_xlabel("Computational budget")
ax.set_ylabel(f"Mean squared error ({r} replicates)")
plt.savefig(f"{plt_dir}/mfkr_mse.png",dpi=600,bbox_inches='tight') # save figure


plt.figure()
ax = plt.gca()
ax.plot(x, ytest, color='black', label='True function')

for k in range(r):
    if k == 0:
        ax.plot(x, ypred_sf[:,k], color='magenta', label='Single fidelity', alpha=0.5)
        ax.plot(x, ypred_mf[:,k], color='blue', label=f'Multifidelity', alpha=0.5)
    else:
        ax.plot(x, ypred_sf[:,k], color='magenta', alpha=0.5)
        ax.plot(x, ypred_mf[:,k], color='blue', alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(f"{plt_dir}/mfkr_pred.png",dpi=600,bbox_inches='tight')
