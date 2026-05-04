import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

current_dir = Path(__file__).parent
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir / '..'))
sys.path.append(str(root_dir / '..' / 'src'))
print(str(root_dir / '..'))

plt_dir = f'plots'
os.makedirs(plt_dir, exist_ok=True)

def f_high(z):
    return np.exp(z)

def f_low(z):
    return 0.9 * np.exp(0.5 * z)

z = np.linspace(0, 5, 200)
y_high = f_high(z)
y_low = f_low(z)


plt.figure(figsize=(8, 5))
plt.plot(z, y_high, label=r'High fidelity $f^{(1)}(x)=e^x$', color='tab:red', linewidth=2)
plt.plot(z, y_low, label=r'Low fidelity $f^{(2)}(x)=0.9e^{0.5x}$', color='tab:blue', linestyle='--', linewidth=2)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
save_path = Path(plt_dir) / "functions.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')