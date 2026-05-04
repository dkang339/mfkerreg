import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path

# ------------------------------------------------------------
# Directory setup
# ------------------------------------------------------------
current_dir = Path(__file__).parent
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir / '..'))
sys.path.append(str(root_dir / '..' / 'src'))

plt_dir = f'plots'
os.makedirs(plt_dir, exist_ok=True)

# ------------------------------------------------------------
# Define high- and low-fidelity functions
# ------------------------------------------------------------
d = 2
a, b, c = 20, 0.2, 2*np.pi
f1 = lambda x: -a * np.exp(-b*np.sqrt(1/d * np.sum(x**2, axis=1))) \
               - np.exp(1/d * np.sum(np.cos(c*x), axis=1)) + a + np.exp(1)
f2 = lambda x: -a * np.exp(-0.9*b*np.sqrt(1/d * np.sum(x**2, axis=1))) \
               - np.exp(1/d * np.sum(np.sin(c*x), axis=1)) + a + np.exp(1) + 0.1*x[:,0]

l, u = -32.768, 32.768  # domain

# ------------------------------------------------------------
# Generate mesh grid for plotting
# ------------------------------------------------------------
N = 100
x1 = np.linspace(l, u, N)
x2 = np.linspace(l, u, N)
X1, X2 = np.meshgrid(x1, x2, indexing="ij")
Xtest = np.stack([X1.ravel(), X2.ravel()], axis=1)

Y1 = f1(Xtest).reshape(N, N)
Y2 = f2(Xtest).reshape(N, N)

# global range for both plots
zmin, zmax = min(Y1.min(), Y2.min()), max(Y1.max(), Y2.max())

# ------------------------------------------------------------
# Plot setup
# ------------------------------------------------------------
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(14, 6))

# High fidelity
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf1 = ax1.plot_surface(X1, X2, Y1, cmap="turbo", linewidth=0, antialiased=True,
                         vmin=zmin, vmax=zmax)
ax1.set_title("High fidelity function")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$f^{(1)}(x)$")
ax1.set_zlim(zmin, zmax)
ax1.view_init(elev=25, azim=-45)

# Low fidelity
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf2 = ax2.plot_surface(X1, X2, Y2, cmap="turbo", linewidth=0, antialiased=True,
                         vmin=zmin, vmax=zmax)
ax2.set_title("Low fidelity function")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("$f^{(2)}(x)$")
ax2.set_zlim(zmin, zmax)
ax2.view_init(elev=25, azim=-45)

# ------------------------------------------------------------
# Adjust layout and add colorbar (no overlap)
# ------------------------------------------------------------
fig.subplots_adjust(left=0.05, right=0.83, wspace=0.2, top=0.9, bottom=0.05)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(surf1, cax=cbar_ax)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
save_path = Path(plt_dir) / "functions.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')