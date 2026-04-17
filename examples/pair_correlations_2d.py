"""
2D pair correlation functions: radial g(r), vector g(r), field correlations, gyromorphic.

Demonstrates: compute_radial_correlations_2d, compute_vector_rdf2d,
  compute_bounded_vector_rdf2d, compute_nnbounded_vector_rdf2d,
  compute_pnn_vector_rdf2d, compute_pnn_rdf,
  compute_vector_gyromorphic_corr_2d, compute_radial_gyromorphic_corr_2d

Test patterns: Poisson, triangular lattice
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

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def triangular_lattice_2d(nx, ny):
    """Triangular lattice tiling [0,1)^2 with periodic boundary conditions."""
    pts = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5 * (j % 2)) / nx
            y = j / ny
            pts.append([x, y])
    return np.array(pts)

boxsize = 1.0
binsize = 0.005

pts_poisson = poisson_2d(1000)
# Use a slightly noisy triangular lattice to avoid bin-aliasing artifacts on the vector g(r)
pts_tri = triangular_lattice_2d(30, 35)
np.random.seed(99)
pts_tri += np.random.randn(*pts_tri.shape) * 0.002
pts_tri %= 1.0

# --- 1. Radial g(r) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pts, label in [(axes[0], pts_poisson, 'Poisson (N=1000)'),
                        (axes[1], pts_tri, 'Triangular lattice')]:
    dummy_field = np.ones((pts.shape[0], 2))
    g_r, _ = rust.compute_radial_correlations_2d(pts, dummy_field, boxsize, binsize, True, False)
    nbins = g_r.shape[0]
    bins = (np.arange(nbins) + 0.5) * binsize
    ax.plot(bins, g_r, c=cmr.ember(0.5), linewidth=0.75)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title(label)

plt.tight_layout()
plt.savefig(f'{outdir}/radial_gr_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved radial_gr_2d.png')

# --- 2. Vector g(r) (pair correlation function) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pts, label in [(axes[0], pts_poisson, 'Poisson'),
                        (axes[1], pts_tri, 'Triangular')]:
    pcf = rust.compute_vector_rdf2d(pts, boxsize, binsize, True)
    N = pts.shape[0]
    rho_n = N**2 / boxsize**2
    pcf_norm = pcf / (rho_n * binsize**2)
    center = pcf.shape[0] // 2
    width = int(0.25 / binsize)
    sl = pcf_norm[center-width:center+width+1, center-width:center+width+1]
    ax.imshow(sl, vmin=0, vmax=min(sl.max(), 5), cmap=cmr.ember,
              extent=[-0.25, 0.25, 0.25, -0.25])
    ax.set_title(f'Vector g(r) — {label}')
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(f'{outdir}/vector_gr_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved vector_gr_2d.png')

# --- 3. Bounded and NN-bounded vector g(r) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

pcf_bounded = rust.compute_bounded_vector_rdf2d(pts_tri, boxsize, binsize, 0.15, True)
pcf_nnbound = rust.compute_nnbounded_vector_rdf2d(pts_tri, boxsize, binsize, 6, True)
pcf_pnn = rust.compute_pnn_vector_rdf2d(pts_tri, 1, boxsize, binsize, True)

rho_n = pts_tri.shape[0]**2 / boxsize**2
for ax, data, title in [(axes[0], pcf_bounded, 'Bounded (r<0.15)'),
                         (axes[1], pcf_nnbound, 'NN-bounded (6 neighbors)'),
                         (axes[2], pcf_pnn, '1st NN only')]:
    data_norm = data / (rho_n * binsize**2)
    center = data.shape[0] // 2
    width = min(center, int(0.15 / binsize))
    sl = data_norm[center-width:center+width+1, center-width:center+width+1]
    ext = width * binsize
    ax.imshow(sl, vmin=0, vmax=max(sl.max() * 0.8, 1), cmap=cmr.ember,
              extent=[-ext, ext, ext, -ext])
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle('Triangular lattice — vector g(r) variants', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/vector_gr_2d_variants.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved vector_gr_2d_variants.png')

# --- 4. p-th nearest neighbor radial g(r) ---
fig, ax = plt.subplots(figsize=(8, 4))
for p in [1, 2, 3, 6]:
    g_pnn = rust.compute_pnn_rdf(pts_poisson, p, boxsize, binsize, True)
    nbins = g_pnn.shape[0]
    bins = (np.arange(nbins) + 0.5) * binsize
    ax.plot(bins, g_pnn, label=f'p={p}', linewidth=0.8)

ax.set_xlim(0, 0.3)
ax.set_xlabel('r')
ax.set_ylabel('g_p(r)')
ax.set_title('p-th nearest neighbor g(r) — Poisson')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/pnn_rdf_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved pnn_rdf_2d.png')

# --- 5. Field correlations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cosine field: f(x,y) = 1 + cos(2*pi*x/L)
# Non-connected C(0) = <f^2>, connected C(0) = <f^2> - <f>^2 = Var(f)
scalar_field = (1 + np.cos(2 * np.pi * pts_poisson[:, 0] / boxsize))[:, None]
mean_sq = np.mean(scalar_field**2)  # <f^2>
variance = np.var(scalar_field)     # Var(f)

g_r, c_r = rust.compute_radial_correlations_2d(pts_poisson, scalar_field, boxsize, binsize, True, False)
_, c_r_conn = rust.compute_radial_correlations_2d(pts_poisson, scalar_field, boxsize, binsize, True, True)
nbins = g_r.shape[0]
bins = (np.arange(nbins) + 0.5) * binsize

axes[0].plot(bins, g_r, linewidth=0.8)
axes[0].set_xlim(0, 0.4)
axes[0].set_xlabel('r'); axes[0].set_ylabel('g(r)')
axes[0].set_title('Radial g(r)')

axes[1].plot(bins, c_r[:, 0] / mean_sq, label=r'$C(r) / \langle f^2 \rangle$', linewidth=0.8)
axes[1].plot(bins, c_r_conn[:, 0] / variance, label=r'$C_c(r) / \mathrm{Var}(f)$', linewidth=0.8)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_xlim(0, 0.4)
axes[1].set_xlabel('r'); axes[1].set_ylabel('Normalized C(r)')
axes[1].set_title('Scalar field correlation')
axes[1].legend()

plt.suptitle(r'Field correlations — Poisson with $f = 1 + \cos(2\pi x / L)$', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/field_correlations_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved field_correlations_2d.png')

# --- 6. Gyromorphic correlations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rdf_gyro, corr_gyro = rust.compute_radial_gyromorphic_corr_2d(pts_tri, boxsize, binsize, True, 6)
nbins = len(rdf_gyro)
bins = (np.arange(nbins) + 0.5) * binsize
corr_mod = np.sqrt(corr_gyro[:, 0]**2 + corr_gyro[:, 1]**2)

axes[0].plot(bins, rdf_gyro, linewidth=0.8)
axes[0].set_xlim(0, 0.3)
axes[0].set_xlabel('r'); axes[0].set_ylabel('g(r)')
axes[0].set_title('Radial g(r)')

axes[1].plot(bins, corr_mod, linewidth=0.8)
axes[1].set_xlim(0, 0.3)
axes[1].set_xlabel('r'); axes[1].set_ylabel('|g_G(r)|')
axes[1].set_title('Gyromorphic correlation (order 6)')

plt.suptitle('Gyromorphic correlations — triangular lattice', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/gyromorphic_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved gyromorphic_2d.png')

print('\nAll 2D pair correlation examples done.')
