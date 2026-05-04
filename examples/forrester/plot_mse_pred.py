import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

current_dir = Path(__file__).parent
root_dir = current_dir.resolve()
plt_dir = f'{root_dir}/plots'
npy_dir = f'{root_dir}/results'
os.makedirs(plt_dir, exist_ok=True)


# TODO: choose to run mfkr or load saved results
# import mfkr

# --- load data ---
data = np.load(f"{npy_dir}/mfkr_forrest.npz", allow_pickle=True)
Xtest = data["Xtest"]
ytest = data["ytest"]
ypred_sf = data["ypred_sf"]
ypred_mf = data["ypred_mf"]
x1 = Xtest[:, 0]
x2 = Xtest[:, 1]
model = data["model"].item()
p = model["p"]
r = model["rep"]
rho = model["rho"]
sigma = model["std"]
mse_sf = data["mse_sf"]
mse_mf = data["mse_mf"]
mean_mse_sf = data["mean_mse_sf"]
mean_mse_mf = data["mean_mse_mf"]
std_mse_sf = data["std_mse_sf"]
std_mse_mf = data["std_mse_mf"]
ypred_sf = data["ypred_sf"]
ypred_mf = data["ypred_mf"]

print('rho:', rho)
print('sigma:', sigma)
print('mean mse sf:', mean_mse_sf)
print('mean mse mf:', mean_mse_mf)

'''
plots
'''
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20, 7)) 
titles = ["High fidelity function", "Single fidelity KR", "Multifidelity KR"]

# get colorbar range
cmin = min(ytest.min(), ypred_sf[:,0].min(), ypred_mf[:,0].min())
cmax = max(ytest.max(), ypred_sf[:,0].max(), ypred_mf[:,0].max())

zmin = cmin - 0.1 * abs(cmax - cmin)
zmax = cmax + 0.1 * abs(cmax - cmin)

# true
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
surf1 = ax1.plot_trisurf(Xtest[:,0], Xtest[:,1], ytest,
                         cmap="turbo", linewidth=0.2, antialiased=True,
                         vmin=cmin, vmax=cmax)
ax1.set_title(titles[0])
ax1.view_init(elev=25, azim=-45)
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$y$")
# ax1.set_zlim(zmin, zmax)

# low fidelity
# ax2 = fig.add_subplot(1, 3, 2, projection="3d")
# surf2 = ax2.plot_trisurf(Xtest[:,0], Xtest[:,1], ylow,
#                          cmap="turbo", linewidth=0.2, antialiased=True)
# ax2.set_title("Low fidelity function")
# ax2.view_init(elev=25, azim=-45)
# ax2.set_xlabel("$x_1$")
# ax2.set_ylabel("$x_2$")
# ax2.set_zlabel("$y$")

# single fidelity
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
surf2 = ax2.plot_trisurf(Xtest[:,0], Xtest[:,1], ypred_sf[:,0],
                         cmap="turbo", linewidth=0.2, antialiased=True,
                         vmin=cmin, vmax=cmax)
ax2.set_title(titles[1])
ax2.view_init(elev=25, azim=-45)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("$y$")
# ax2.set_zlim(zmin, zmax)

# multifidelity
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
surf3 = ax3.plot_trisurf(Xtest[:,0], Xtest[:,1], ypred_mf[:,0],
                         cmap="turbo", linewidth=0.2, antialiased=True,
                         vmin=cmin, vmax=cmax)
ax3.set_title(titles[2])
ax3.view_init(elev=25, azim=-45)
ax3.set_xlabel("$x_1$")
ax3.set_ylabel("$x_2$")
ax3.set_zlabel("$y$")
# ax3.set_zlim(zmin, zmax)

# Shared colorbar for all 3 surfaces
plt.subplots_adjust(wspace=0.15, right=0.88, left=0.05, top=0.93, bottom=0.05)
fig.colorbar(surf1, ax=[ax1, ax2, ax3],
             shrink=0.8, aspect=18, pad=0.08, label="$y$")

plt.savefig(f"{plt_dir}/mfkr_pred.png", dpi=600)




# plot mse (compared to truth)
plt.figure()
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt_kr, = ax.plot(p, mean_mse_sf,':o', color='r', label='Single fidelity')
# ax.plot(p, mean_mse - std_mse, linestyle=":", linewidth=0.8,
#         color=plt_kr.get_color(), alpha=0.6)
# ax.plot(p, mean_mse + std_mse, linestyle=":", linewidth=0.8,
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