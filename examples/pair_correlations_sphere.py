"""
Pair correlation functions on the 2-sphere: radial g(r) and vector g(r).

Demonstrates: compute_radial_correlations_2sphere, compute_vector_rdf2sphere

Test patterns: uniform random on sphere, Fibonacci spiral
"""

import rust
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmasher as cmr
import os

outdir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(outdir, exist_ok=True)

# --- Generate test patterns ---

def random_sphere(N, seed=42):
    """Uniform random points on the unit sphere, in (theta, phi) format."""
    np.random.seed(seed)
    theta = np.arccos(2 * np.random.rand(N) - 1)
    phi = 2 * np.pi * np.random.rand(N)
    return np.column_stack([theta, phi])

def fibonacci_spiral(N):
    """Nearly-uniform Fibonacci spiral on the sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    theta = np.arccos(1 - 2 * (i + 0.5) / N)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return np.column_stack([theta, phi])

binsize = 0.05

pts_random = random_sphere(1000)
pts_fib = fibonacci_spiral(1000)

# --- 1. Radial g(r) on the sphere ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pts, label in [(axes[0], pts_random, 'Random (N=1000)'),
                        (axes[1], pts_fib, 'Fibonacci spiral (N=1000)')]:
    dummy_field = np.ones((pts.shape[0], 2))
    g_r, _ = rust.compute_radial_correlations_2sphere(pts, dummy_field, binsize, False)
    nbins = g_r.shape[0]
    bins = (np.arange(nbins) + 0.5) * binsize
    ax.plot(bins, g_r, c=cmr.ember(0.5), linewidth=0.75)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel('geodesic distance')
    ax.set_ylabel('g(r)')
    ax.set_title(label)

plt.suptitle('Radial g(r) on the 2-sphere', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/radial_gr_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved radial_gr_sphere.png')

# --- 2. Vector g(r) on the sphere (theta, phi map) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pts, label in [(axes[0], pts_random, 'Random'),
                        (axes[1], pts_fib, 'Fibonacci')]:
    pcf = rust.compute_vector_rdf2sphere(pts, binsize)
    N = pts.shape[0]
    ax.imshow(pcf, cmap=cmr.ember, aspect='auto',
              extent=[0, 2*np.pi, np.pi, 0])
    ax.set_xlabel('relative phi')
    ax.set_ylabel('relative theta')
    ax.set_title(f'Vector g(r) — {label}')

plt.suptitle('Spherical vector g(r) (relative angular coordinates)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/vector_gr_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved vector_gr_sphere.png')

# --- 3. Field correlations on the sphere ---
fig, ax = plt.subplots(figsize=(8, 4))

# Use f = 1 + cos(theta) so that <f> != 0, making connected and non-connected differ
z_field = (1 + np.cos(pts_random[:, 0]))[:, None]  # shape (N, 1)
mean_sq = np.mean(z_field**2)
variance = np.var(z_field)

g_r, c_r = rust.compute_radial_correlations_2sphere(pts_random, z_field, binsize, False)
_, c_r_conn = rust.compute_radial_correlations_2sphere(pts_random, z_field, binsize, True)
nbins = g_r.shape[0]
bins = (np.arange(nbins) + 0.5) * binsize

ax.plot(bins, c_r[:, 0] / mean_sq, label=r'$C(r) / \langle f^2 \rangle$', linewidth=0.8)
ax.plot(bins, c_r_conn[:, 0] / variance, label=r'$C_c(r) / \mathrm{Var}(f)$', linewidth=0.8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlim(0, np.pi)
ax.set_xlabel('geodesic distance')
ax.set_ylabel('Normalized C(r)')
ax.set_title(r'Field correlation on sphere — $f = 1 + \cos\theta$')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/field_correlations_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved field_correlations_sphere.png')

print('\nAll sphere pair correlation examples done.')
