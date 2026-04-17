"""
3D pair correlation functions: radial g(r), vector g(r), field correlations.

Demonstrates: compute_radial_correlations_3d, compute_vector_rdf3d,
  compute_bounded_vector_rdf3d, compute_nnbounded_vector_rdf3d,
  compute_pnn_vector_rdf3d, compute_pnn_rdf (3D)

Test patterns: Poisson, FCC lattice
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

def poisson_3d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 3)

def fcc_lattice(n):
    """FCC lattice with n unit cells per side, in [0, 1]^3."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) / n
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for b in basis:
                    pts.append(b + np.array([i, j, k]) / n)
    return np.array(pts)

boxsize = 1.0
binsize = 0.01

pts_poisson = poisson_3d(2000)
pts_fcc = fcc_lattice(8)  # 8^3 * 4 = 2048 particles

print(f'Poisson: {pts_poisson.shape[0]} points')
print(f'FCC: {pts_fcc.shape[0]} points')

# --- 1. Radial g(r) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pts, label in [(axes[0], pts_poisson, f'Poisson (N={pts_poisson.shape[0]})'),
                        (axes[1], pts_fcc, f'FCC lattice (N={pts_fcc.shape[0]})')]:
    dummy_field = np.ones((pts.shape[0], 2))
    g_r, _ = rust.compute_radial_correlations_3d(pts, dummy_field, boxsize, binsize, True, False)
    nbins = g_r.shape[0]
    bins = (np.arange(nbins) + 0.5) * binsize
    # Restrict to r < L/2 to avoid the normalization artifact near the box diagonal
    mask = bins < 0.5 * boxsize
    ax.plot(bins[mask], g_r[mask], c=cmr.ember(0.5), linewidth=0.75)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title(label)

plt.tight_layout()
plt.savefig(f'{outdir}/radial_gr_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved radial_gr_3d.png')

# --- 2. Vector g(r) — xy slice through z=0 ---
# Use a coarser binsize for the vector RDF (finer bins just produce sharper deltas on a lattice)
binsize_vpcf = 0.02
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pts, label in [(axes[0], pts_poisson, 'Poisson'),
                        (axes[1], pts_fcc, 'FCC')]:
    pcf = rust.compute_vector_rdf3d(pts, boxsize, binsize_vpcf, True)
    N = pts.shape[0]
    rho_n = N**2 / boxsize**3
    pcf_norm = pcf / (rho_n * binsize_vpcf**3)
    center = pcf.shape[0] // 2
    width = int(0.25 / binsize_vpcf)
    sl = pcf_norm[center-width:center+width+1, center-width:center+width+1, center]
    ax.imshow(sl, vmin=0, vmax=min(sl.max(), 10), cmap=cmr.ember,
              extent=[-0.25, 0.25, 0.25, -0.25])
    ax.set_title(f'Vector g(r) xy slice — {label}')
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(f'{outdir}/vector_gr_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved vector_gr_3d.png')

# --- 3. Bounded vector g(r) variants for FCC ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

pcf_bounded = rust.compute_bounded_vector_rdf3d(pts_fcc, boxsize, binsize_vpcf, 0.2, True)
pcf_nnbound = rust.compute_nnbounded_vector_rdf3d(pts_fcc, boxsize, binsize_vpcf, 12, True)
pcf_pnn = rust.compute_pnn_vector_rdf3d(pts_fcc, 1, boxsize, binsize_vpcf, True)

rho_n = pts_fcc.shape[0]**2 / boxsize**3
for ax, data, title in [(axes[0], pcf_bounded, 'Bounded (r<0.2)'),
                         (axes[1], pcf_nnbound, 'NN-bounded (12)'),
                         (axes[2], pcf_pnn, '1st NN only')]:
    data_norm = data / (rho_n * binsize_vpcf**3)
    center = data.shape[0] // 2
    width = min(center, int(0.15 / binsize_vpcf))
    sl = data_norm[center-width:center+width+1, center-width:center+width+1, center]
    ext = width * binsize_vpcf
    ax.imshow(sl, vmin=0, vmax=max(sl.max() * 0.8, 1), cmap=cmr.ember,
              extent=[-ext, ext, ext, -ext])
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle('FCC lattice — vector g(r) variants (xy slice)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/vector_gr_3d_variants.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved vector_gr_3d_variants.png')

# --- 4. p-th nearest neighbor radial g(r) ---
fig, ax = plt.subplots(figsize=(8, 4))
for p in [1, 2, 3, 12]:
    g_pnn = rust.compute_pnn_rdf(pts_poisson, p, boxsize, binsize, True)
    nbins = g_pnn.shape[0]
    bins = (np.arange(nbins) + 0.5) * binsize
    ax.plot(bins, g_pnn, label=f'p={p}', linewidth=0.8)

ax.set_xlim(0, 0.4)
ax.set_xlabel('r')
ax.set_ylabel('g_p(r)')
ax.set_title('p-th nearest neighbor g(r) — 3D Poisson')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/pnn_rdf_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved pnn_rdf_3d.png')

# --- 5. Field correlations in 3D ---
fig, ax = plt.subplots(figsize=(8, 4))

scalar_field = (1 + np.cos(2 * np.pi * pts_poisson[:, 0] / boxsize))[:, None]
mean_sq = np.mean(scalar_field**2)
variance = np.var(scalar_field)

g_r, c_r = rust.compute_radial_correlations_3d(pts_poisson, scalar_field, boxsize, binsize, True, False)
_, c_r_conn = rust.compute_radial_correlations_3d(pts_poisson, scalar_field, boxsize, binsize, True, True)
nbins = g_r.shape[0]
bins = (np.arange(nbins) + 0.5) * binsize

ax.plot(bins, c_r[:, 0] / mean_sq, label=r'$C(r) / \langle f^2 \rangle$', linewidth=0.8)
ax.plot(bins, c_r_conn[:, 0] / variance, label=r'$C_c(r) / \mathrm{Var}(f)$', linewidth=0.8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlim(0, 0.4)
ax.set_xlabel('r'); ax.set_ylabel('Normalized C(r)')
ax.set_title(r'Field correlation — 3D Poisson, $f = 1 + \cos(2\pi x / L)$')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/field_correlations_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved field_correlations_3d.png')

print('\nAll 3D pair correlation examples done.')
