"""
SANN (Solid-Angle Nearest-Neighbor) BOOPs and neighborhood quantities.

Demonstrates: compute_sann_boops_2d, compute_sann_boops_3d,
  compute_sann_quantities_2d, compute_sann_quantities_3d,
  compute_metric_boops_2d, compute_metric_boops_3d

Test patterns: Poisson, triangular lattice (2D), FCC lattice (3D)
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

# --- Helpers ---

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def triangular_lattice_2d(nx, ny, noise=0.0, seed=0):
    np.random.seed(seed)
    pts = []
    for i in range(nx):
        for j in range(ny):
            pts.append([(i + 0.5 * (j % 2)) / nx, j / ny])
    pts = np.array(pts)
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise / nx
        pts %= 1.0
    return pts

def poisson_3d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 3)

def fcc_lattice(n):
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]) / n
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for b in basis:
                    pts.append(b + np.array([i,j,k])/n)
    return np.array(pts)

def psi_mod(boops):
    """Modulus of complex BOOPs: (N, n_orders, 2) -> (N, n_orders)."""
    return np.sqrt(boops[:, :, 0]**2 + boops[:, :, 1]**2)

# --- 1. 2D BOOPs: SANN vs Voronoi vs metric ---

print('2D BOOPs comparison...')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

pts_tri = triangular_lattice_2d(25, 29, noise=0.05)
pts_poi = poisson_2d(700)
spacing = 1.0 / 25
cutoff = 1.5 * spacing

for col, (pts, label) in enumerate([(pts_tri, 'Noisy triangular'),
                                     (pts_poi, 'Poisson (N=700)')]):
    boops_voro = rust.compute_2d_boops(pts, np.array([6]), boxsize, True)
    boops_sann = rust.compute_sann_boops_2d(pts, np.array([6]), boxsize, True)
    boops_metr = rust.compute_metric_boops_2d(pts, np.array([6]), boxsize, cutoff, True)

    for row, (b, method, color) in enumerate([
        (boops_voro, 'Voronoi', 'steelblue'),
        (boops_sann, 'SANN', 'coral'),
    ]):
        psi6 = psi_mod(b)[:, 0]
        axes[row, col].hist(psi6, bins=40, range=(0, 1), density=True,
                             color=color, edgecolor='white', alpha=0.8)
        axes[row, col].set_xlabel('|psi_6|')
        axes[row, col].set_title(f'{method} — {label}, mean={psi6.mean():.3f}')

    # Third column: overlay all three
    if col == 0:
        ax = axes[0, 2]
        for b, method, color, ls in [
            (boops_voro, 'Voronoi', 'steelblue', '-'),
            (boops_sann, 'SANN', 'coral', '--'),
            (boops_metr, f'Metric (r={cutoff:.3f})', 'seagreen', ':'),
        ]:
            psi6 = psi_mod(b)[:, 0]
            vals, edges = np.histogram(psi6, bins=40, range=(0, 1), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, vals, ls, color=color, label=method, linewidth=1.5)
        ax.set_xlabel('|psi_6|')
        ax.set_title('Noisy triangular — comparison')
        ax.legend(fontsize=9)

        ax = axes[1, 2]
        for b, method, color, ls in [
            (boops_voro, 'Voronoi', 'steelblue', '-'),
            (boops_sann, 'SANN', 'coral', '--'),
            (boops_metr, f'Metric (r={cutoff:.3f})', 'seagreen', ':'),
        ]:
            psi6 = psi_mod(b)[:, 0]
            vals, edges = np.histogram(psi6, bins=40, range=(0, 1), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, vals, ls, color=color, label=method, linewidth=1.5)
        ax.set_xlabel('|psi_6|')
        ax.set_title('Poisson — comparison')
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{outdir}/sann_boops_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved sann_boops_2d.png')

# --- 2. 2D spatial map: SANN vs Voronoi psi_6 ---

print('2D spatial maps...')
pts = triangular_lattice_2d(25, 29, noise=0.1)
boops_voro = rust.compute_2d_boops(pts, np.array([6]), boxsize, True)
boops_sann = rust.compute_sann_boops_2d(pts, np.array([6]), boxsize, True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, boops, label in [(axes[0], boops_voro, 'Voronoi'),
                          (axes[1], boops_sann, 'SANN')]:
    psi6 = psi_mod(boops)[:, 0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=psi6, cmap='inferno', s=12, vmin=0, vmax=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(f'{label} |psi_6| — mean={psi6.mean():.3f}')

plt.colorbar(sc, ax=axes, label='|psi_6|', shrink=0.8)
plt.savefig(f'{outdir}/sann_boops_2d_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved sann_boops_2d_spatial.png')

# --- 3. 2D SANN quantities vs Voronoi quantities ---

print('2D SANN quantities...')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, (pts, label) in enumerate([(pts_poi, 'Poisson'),
                                     (pts_tri, 'Noisy triangular')]):
    areas_v, nn_v, nnd_v = rust.compute_2d_all_voronoi_quantities(pts, boxsize, True)
    nn_s, nnd_s, shell_s = rust.compute_sann_quantities_2d(pts, boxsize, True)

    # Neighbor count comparison
    ax = axes[row, 0]
    bins_nn = np.arange(2.5, 14.5)
    ax.hist(nn_v, bins=bins_nn, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', label='Voronoi')
    ax.hist(nn_s.astype(int), bins=bins_nn, density=True, alpha=0.6, color='coral',
            edgecolor='white', label='SANN')
    ax.set_xlabel('Neighbors')
    ax.set_title(f'{label} — neighbor count')
    ax.legend(fontsize=9)

    # NN distance comparison
    ax = axes[row, 1]
    combined = np.concatenate([nnd_v, nnd_s])
    bins_d = np.linspace(combined.min(), combined.max(), 30)
    ax.hist(nnd_v, bins=bins_d, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', label='Voronoi')
    ax.hist(nnd_s, bins=bins_d, density=True, alpha=0.6, color='coral',
            edgecolor='white', label='SANN')
    ax.set_xlabel('NN distance')
    ax.set_title(f'{label} — NN distance')
    ax.legend(fontsize=9)

    # Voronoi area vs SANN shell radius
    ax = axes[row, 2]
    ax.hist(areas_v, bins=30, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', label='Voronoi area')
    ax2 = ax.twinx()
    ax2.hist(shell_s, bins=30, density=True, alpha=0.6, color='coral',
             edgecolor='white', label='SANN shell R')
    ax.set_xlabel('Value')
    ax.set_title(f'{label} — area vs shell radius')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

plt.tight_layout()
plt.savefig(f'{outdir}/sann_quantities_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved sann_quantities_2d.png')

# --- 4. 3D BOOPs: SANN on FCC and Poisson ---

print('3D SANN BOOPs...')
pts_fcc = fcc_lattice(6)
pts_3d_poi = poisson_3d(2000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# FCC: should show sharp peaks at known q_l values
ql_fcc = rust.compute_sann_boops_3d(pts_fcc, np.array([4, 6, 8, 10, 12]), boxsize, True)
ax = axes[0]
orders_3d = [4, 6, 8, 10, 12]
fcc_means = [ql_fcc[:, j].mean() for j in range(len(orders_3d))]
ax.bar(orders_3d, fcc_means, width=1.5, color='coral', edgecolor='white')
for j, l in enumerate(orders_3d):
    ax.text(l, fcc_means[j] + 0.01, f'{fcc_means[j]:.3f}', ha='center', fontsize=8)
ax.set_xlabel('Order l')
ax.set_ylabel('q_l')
ax.set_title(f'FCC (N={pts_fcc.shape[0]}) — SANN q_l (all particles identical)')
ax.set_ylim(0, 0.8)

# Poisson: broad distributions
ql_poi = rust.compute_sann_boops_3d(pts_3d_poi, np.array([4, 6, 8, 10, 12]), boxsize, True)
ax = axes[1]
for j, l in enumerate([4, 6, 8, 10, 12]):
    vals, edges = np.histogram(ql_poi[:, j], bins=40, range=(0, 0.8), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, vals, label=f'q{l}', linewidth=1.2)
ax.set_xlabel('q_l')
ax.set_title(f'Poisson (N={pts_3d_poi.shape[0]}) — SANN q_l')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{outdir}/sann_boops_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved sann_boops_3d.png')

# --- 5. 3D SANN quantities ---

print('3D SANN quantities...')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

np.random.seed(7)
pts_fcc_noisy = pts_fcc + np.random.randn(*pts_fcc.shape) * 0.003
pts_fcc_noisy %= 1.0

for row, (pts, label) in enumerate([(pts_fcc_noisy, f'Noisy FCC (N={pts_fcc.shape[0]})'),
                                     (pts_3d_poi, 'Poisson (N=2000)')]):
    nn_s, nnd_s, shell_s = rust.compute_sann_quantities_3d(pts, boxsize, True)

    ax = axes[row, 0]
    vals, counts = np.unique(nn_s, return_counts=True)
    ax.bar(vals.astype(int), counts / len(nn_s), color='coral', edgecolor='white')
    ax.set_xlabel('SANN neighbors')
    ax.set_title(f'{label} — neighbor count (mean={nn_s.mean():.1f})')

    ax = axes[row, 1]
    ax.hist(nnd_s, bins=30, density=True, color='coral', edgecolor='white')
    ax.set_xlabel('NN distance')
    ax.set_title(f'{label} — NN distance')

    ax = axes[row, 2]
    ax.hist(shell_s, bins=30, density=True, color='coral', edgecolor='white')
    ax.set_xlabel('Shell radius R(m)')
    ax.set_title(f'{label} — SANN shell radius')

plt.tight_layout()
plt.savefig(f'{outdir}/sann_quantities_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved sann_quantities_3d.png')

print('\nAll SANN examples done.')
