"""
Number variance (hyperuniformity diagnostics) in flat space and on the sphere.

The reduced number variance sigma_N^2 / <N>^2 scales as:
  - 1/<N> ~ R^{-d}           for Poisson
  - 1/<N>^{(d+1)/d} ~ R^{-(d+1)} for a hyperuniform lattice (surface-area scaling)

Demonstrates: point_variances, point_variances_2sphere

Test patterns: Poisson vs square/cubic lattice (flat), random vs Fibonacci spiral (sphere)
"""

import rust
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

outdir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(outdir, exist_ok=True)

boxsize = 1.0
n_samples = 1000000

# --- Helpers ---

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def square_lattice_2d(n):
    """Square lattice in [0,1)^2, n x n points. Tiles perfectly under PBC."""
    x = np.arange(n, dtype=float) / n
    xx, yy = np.meshgrid(x, x)
    return np.column_stack([xx.ravel(), yy.ravel()])

def poisson_3d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 3)

def cubic_lattice_3d(n):
    """Simple cubic lattice in [0,1)^3, n x n x n points. Tiles perfectly under PBC."""
    x = np.arange(n, dtype=float) / n
    xx, yy, zz = np.meshgrid(x, x, x)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

def random_sphere(N, seed=42):
    np.random.seed(seed)
    return np.column_stack([np.arccos(2*np.random.rand(N)-1), 2*np.pi*np.random.rand(N)])

def fibonacci_spiral(N):
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    theta = np.arccos(1 - 2 * (i + 0.5) / N)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return np.column_stack([theta, phi])

# --- 2D flat ---

n_sq = 150  # 150x150 = 22500
N_2d = n_sq * n_sq
pts_poisson_2d = poisson_2d(N_2d)
pts_sq = square_lattice_2d(n_sq)

radii_2d = np.linspace(0.008, 0.12, 25)

print(f'Computing 2D point variances (N={N_2d})...')
var_poisson_2d = rust.point_variances(pts_poisson_2d, radii_2d, boxsize, n_samples, True)
var_sq = rust.point_variances(pts_sq, radii_2d, boxsize, n_samples, True)

rho_2d = N_2d / boxsize**2
mean_N_2d = rho_2d * np.pi * radii_2d**2  # <N> in a disk of radius R

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(radii_2d, var_poisson_2d, 'o-', markersize=3, label='Poisson')
ax.loglog(radii_2d, var_sq, 's-', markersize=3, label='Square lattice')
# Poisson scaling: sigma^2/<N>^2 = 1/<N> ~ R^{-2}
ax.loglog(radii_2d, 1.0 / mean_N_2d, '--', color='gray', alpha=0.7,
          label=r'$\sim R^{-2}$ (Poisson)')
# HU scaling: sigma^2/<N>^2 ~ 1/<N>^{3/2} ~ R^{-3}
# Fit the prefactor from the data at the middle of the range
mid = len(radii_2d) // 2
hu_prefactor = var_sq[mid] * mean_N_2d[mid]**(3/2)
ax.loglog(radii_2d, hu_prefactor / mean_N_2d**(3/2), ':', color='red', alpha=0.7,
          label=r'$\sim R^{-3}$ (HU)')
ax.set_xlabel('Window radius R')
ax.set_ylabel(r'$\sigma_N^2 / \langle N \rangle^2$')
ax.set_title(f'Number variance — 2D (N={N_2d})')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/hyperuniformity_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hyperuniformity_2d.png')

# --- 3D flat ---

n_cub = 20  # 20^3 = 8000
N_3d = n_cub**3
pts_poisson_3d = poisson_3d(N_3d)
pts_cub = cubic_lattice_3d(n_cub)

radii_3d = np.linspace(0.015, 0.12, 25)

print(f'Computing 3D point variances (N={N_3d})...')
var_poisson_3d = rust.point_variances(pts_poisson_3d, radii_3d, boxsize, n_samples, True)
var_cub = rust.point_variances(pts_cub, radii_3d, boxsize, n_samples, True)

rho_3d = N_3d / boxsize**3
mean_N_3d = rho_3d * (4.0/3.0) * np.pi * radii_3d**3

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(radii_3d, var_poisson_3d, 'o-', markersize=3, label='Poisson')
ax.loglog(radii_3d, var_cub, 's-', markersize=3, label='Cubic lattice')
ax.loglog(radii_3d, 1.0 / mean_N_3d, '--', color='gray', alpha=0.7,
          label=r'$\sim R^{-3}$ (Poisson)')
mid3 = len(radii_3d) // 2
hu_pref_3d = var_cub[mid3] * mean_N_3d[mid3]**(4/3)
ax.loglog(radii_3d, hu_pref_3d / mean_N_3d**(4/3), ':', color='red', alpha=0.7,
          label=r'$\sim R^{-4}$ (HU)')
ax.set_xlabel('Window radius R')
ax.set_ylabel(r'$\sigma_N^2 / \langle N \rangle^2$')
ax.set_title(f'Number variance — 3D (N={N_3d})')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/hyperuniformity_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hyperuniformity_3d.png')

# --- 2-sphere ---

N_sph = 5000
pts_rand_sph = random_sphere(N_sph)
pts_fib = fibonacci_spiral(N_sph)

radii_sph = np.linspace(0.03, 0.6, 25)

print(f'Computing spherical point variances (N={N_sph})...')
var_rand_sph = rust.point_variances_2sphere(pts_rand_sph, radii_sph, n_samples)
var_fib = rust.point_variances_2sphere(pts_fib, radii_sph, n_samples)

# <N> in a spherical cap of geodesic radius r: N * (1 - cos r) / 2
mean_N_sph = N_sph * (1 - np.cos(radii_sph)) / 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(radii_sph, var_rand_sph, 'o-', markersize=3, label='Random')
ax.loglog(radii_sph, var_fib, 's-', markersize=3, label='Fibonacci spiral')
ax.loglog(radii_sph, 1.0 / mean_N_sph, '--', color='gray', alpha=0.7,
          label=r'$\sim 1/\langle N \rangle$ (Poisson)')
# HU scaling on the sphere: sigma^2 ~ boundary ~ R (cap perimeter), so
# sigma^2/<N>^2 ~ R / <N>^2 ~ 1/<N>^{3/2} (same as 2D since the sphere is 2-dimensional)
mid_s = len(radii_sph) // 2
hu_pref_sph = var_fib[mid_s] * mean_N_sph[mid_s]**(3/2)
ax.loglog(radii_sph, hu_pref_sph / mean_N_sph**(3/2), ':', color='red', alpha=0.7,
          label=r'$\sim 1/\langle N \rangle^{3/2}$ (HU)')
ax.set_xlabel('Cap radius (geodesic)')
ax.set_ylabel(r'$\sigma_N^2 / \langle N \rangle^2$')
ax.set_title(f'Number variance on the 2-sphere (N={N_sph})')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/hyperuniformity_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved hyperuniformity_sphere.png')

print('\nAll hyperuniformity examples done.')
