"""
Steinhardt Bond-Orientational Order Parameters (BOOPs) in 2D and on the sphere.

Demonstrates: compute_2d_boops, compute_2sphere_boops

Test patterns: triangular lattice, square lattice, Poisson, icosahedron, octahedron
"""

import rust
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

outdir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(outdir, exist_ok=True)

# --- 2D patterns ---

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def triangular_lattice_2d(nx, ny, noise=0.0, seed=0):
    """Triangular lattice tiling [0,1)^2 with periodic boundary conditions."""
    np.random.seed(seed)
    pts = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5 * (j % 2)) / nx
            y = j / ny
            pts.append([x, y])
    pts = np.array(pts)
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise / nx
        pts %= 1.0
    return pts

def square_lattice_2d(n, noise=0.0, seed=1):
    """Square lattice in [0,1)^2. A small noise breaks the Delaunay degeneracy."""
    np.random.seed(seed)
    x = np.linspace(0.5/n, 1 - 0.5/n, n)
    xx, yy = np.meshgrid(x, x)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise / n
        pts %= 1.0
    return pts

boxsize = 1.0

# --- 1. |psi_6| and |psi_4| for 2D patterns ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

patterns = [
    (poisson_2d(500), 'Poisson'),
    (triangular_lattice_2d(25, 25, noise=0.05), 'Noisy triangular'),
    (triangular_lattice_2d(25, 25, noise=0.30), 'Disordered triangular'),
]

for col, (pts, label) in enumerate(patterns):
    boops = rust.compute_2d_boops(pts, np.array([4, 6]), boxsize, True)
    psi_mod = np.sqrt(boops[:, :, 0]**2 + boops[:, :, 1]**2)

    axes[0, col].hist(psi_mod[:, 1], bins=40, range=(0, 1), density=True,
                       color='steelblue', edgecolor='white')
    axes[0, col].set_xlabel('|psi_6|')
    axes[0, col].set_title(f'{label} — |psi_6|, mean={psi_mod[:,1].mean():.3f}')

    axes[1, col].hist(psi_mod[:, 0], bins=40, range=(0, 1), density=True,
                       color='coral', edgecolor='white')
    axes[1, col].set_xlabel('|psi_4|')
    axes[1, col].set_title(f'{label} — |psi_4|, mean={psi_mod[:,0].mean():.3f}')

plt.tight_layout()
plt.savefig(f'{outdir}/boops_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved boops_2d.png')

# --- 2. Spatial map of |psi_6| for noisy triangular ---
pts = triangular_lattice_2d(25, 29, noise=0.08)
boops = rust.compute_2d_boops(pts, np.array([6]), boxsize, True)
psi6 = np.sqrt(boops[:, 0, 0]**2 + boops[:, 0, 1]**2)

fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(pts[:, 0], pts[:, 1], c=psi6, cmap='inferno', s=15, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='|psi_6|')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
ax.set_title('Spatial |psi_6| map — noisy triangular lattice')
plt.tight_layout()
plt.savefig(f'{outdir}/boops_2d_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved boops_2d_spatial.png')

# --- Spherical patterns ---

def random_sphere(N, seed=42):
    np.random.seed(seed)
    return np.column_stack([np.arccos(2*np.random.rand(N)-1), 2*np.pi*np.random.rand(N)])

def fibonacci_spiral(N):
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    theta = np.arccos(1 - 2 * (i + 0.5) / N)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return np.column_stack([theta, phi])

# --- 3. Spherical |psi_6|: random vs Fibonacci ---
N_sph = 1000
pts_rand = random_sphere(N_sph)
pts_fib = fibonacci_spiral(N_sph)

boops_rand = rust.compute_2sphere_boops(pts_rand, np.array([6]))
boops_fib = rust.compute_2sphere_boops(pts_fib, np.array([6]))
psi6_rand = np.sqrt(boops_rand[:, 0, 0]**2 + boops_rand[:, 0, 1]**2)
psi6_fib = np.sqrt(boops_fib[:, 0, 0]**2 + boops_fib[:, 0, 1]**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(psi6_rand, bins=40, range=(0, 1), density=True,
              color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('|psi_6|')
axes[0].set_title(f'Random (N={N_sph}) — mean={psi6_rand.mean():.3f}')

axes[1].hist(psi6_fib, bins=40, range=(0, 1), density=True,
              color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('|psi_6|')
axes[1].set_title(f'Fibonacci (N={N_sph}) — mean={psi6_fib.mean():.3f}')

plt.suptitle('|psi_6| on the sphere', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/boops_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved boops_sphere.png')

# --- 4. Mollweide maps of |psi_6| ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                          subplot_kw={'projection': 'mollweide'})

for ax, pts, psi6, label in [(axes[0], pts_rand, psi6_rand, 'Random'),
                               (axes[1], pts_fib, psi6_fib, 'Fibonacci')]:
    cart = np.column_stack([np.sin(pts[:,0])*np.cos(pts[:,1]),
                            np.sin(pts[:,0])*np.sin(pts[:,1]),
                            np.cos(pts[:,0])])
    lon = np.arctan2(cart[:,1], cart[:,0])
    lat = np.arcsin(np.clip(cart[:,2], -1, 1))
    sc = ax.scatter(lon, lat, c=psi6, cmap='inferno', s=5, vmin=0, vmax=1)
    ax.grid(True, alpha=0.2)
    ax.set_title(f'{label} (N={N_sph})')

plt.colorbar(sc, ax=axes, label='|psi_6|', shrink=0.6, pad=0.08)
plt.savefig(f'{outdir}/boops_sphere_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved boops_sphere_spatial.png')

print('\nAll BOOP examples done.')
