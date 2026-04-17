"""
2D Voronoi tessellation: quantities, tessellation visualization, furthest sites.

Demonstrates: compute_2d_all_voronoi_quantities, compute_2d_voronoi_areas,
  compute_2d_voronoi_neighbour_numbers, compute_2d_voronoi_nn_distances,
  voronoi_tessellation_2d, voronoi_furthest_sites

Test patterns: Poisson, perturbed triangular lattice
"""

import rust
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os

outdir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(outdir, exist_ok=True)

# --- Generate test patterns ---

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def noisy_triangular_lattice(nx, ny, noise=0.1, seed=7):
    np.random.seed(seed)
    pts = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5 * (j % 2)) / nx
            y = j / ny
            pts.append([x, y])
    pts = np.array(pts)
    spacing = 1.0 / nx
    pts += np.random.randn(*pts.shape) * noise * spacing
    pts %= 1.0  # periodic wrap
    return pts

boxsize = 1.0

pts_poisson = poisson_2d(300)
pts_noisy = noisy_triangular_lattice(20, 23, noise=0.15)

# --- 1. Voronoi quantities ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, (pts, label) in enumerate([(pts_poisson, 'Poisson'), (pts_noisy, 'Noisy triangular')]):
    areas, nn_counts, nn_dists = rust.compute_2d_all_voronoi_quantities(pts, boxsize, True)

    axes[col, 0].hist(areas, bins=30, density=True, color='steelblue', edgecolor='white')
    axes[col, 0].set_xlabel('Voronoi cell area')
    axes[col, 0].set_title(f'{label} — area distribution')

    axes[col, 1].hist(nn_counts, bins=np.arange(2.5, 12.5), density=True,
                       color='steelblue', edgecolor='white')
    axes[col, 1].set_xlabel('Number of neighbors')
    axes[col, 1].set_title(f'{label} — neighbor count')

    axes[col, 2].hist(nn_dists, bins=30, density=True, color='steelblue', edgecolor='white')
    axes[col, 2].set_xlabel('NN distance')
    axes[col, 2].set_title(f'{label} — NN distance')

plt.tight_layout()
plt.savefig(f'{outdir}/voronoi_quantities_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved voronoi_quantities_2d.png')

# --- 2. Voronoi tessellation colored by coordination ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pts, label in [(axes[0], pts_poisson, 'Poisson (N=300)'),
                        (axes[1], pts_noisy, 'Noisy triangular')]:
    verts, edges, ci, co = rust.voronoi_tessellation_2d(pts, boxsize, True)
    cell_sizes = np.diff(co)

    polygons = []
    for i in range(pts.shape[0]):
        idx = ci[co[i]:co[i+1]]
        if len(idx) >= 3:
            polygons.append(verts[idx])
        else:
            polygons.append(np.zeros((0, 2)))

    pc = PolyCollection(polygons, edgecolors='k', linewidths=0.3)
    pc.set_array(cell_sizes.astype(float))
    pc.set_cmap('viridis')
    pc.set_clim(4, 8)
    ax.add_collection(pc)
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c='white', zorder=5)
    ax.set_xlim(0, boxsize); ax.set_ylim(0, boxsize)
    ax.set_aspect('equal')
    ax.set_title(label)

plt.colorbar(pc, ax=axes, label='neighbours', shrink=0.8)
plt.savefig(f'{outdir}/voronoi_tessellation_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved voronoi_tessellation_2d.png')

# --- 3. Voronoi furthest sites ---
fig, ax = plt.subplots(figsize=(7, 7))

pts = pts_poisson
furthest = rust.voronoi_furthest_sites(pts, boxsize, True)
# Filter to sites inside the box
mask = (furthest[:, 0] >= 0) & (furthest[:, 0] < boxsize) & \
       (furthest[:, 1] >= 0) & (furthest[:, 1] < boxsize)
furthest = furthest[mask]
furthest = furthest[np.argsort(furthest[:, 2])]  # sort by circumradius

# Scatter: particles + top 20 furthest sites
ax.scatter(pts[:, 0], pts[:, 1], s=10, c='steelblue', label='Particles')
top = furthest[-20:]
ax.scatter(top[:, 0], top[:, 1], s=30, c='red', marker='x', label='Top 20 voids')
ax.set_xlim(0, boxsize); ax.set_ylim(0, boxsize)
ax.set_aspect('equal')
ax.legend()
ax.set_title('Voronoi furthest sites — largest voids')
plt.tight_layout()
plt.savefig(f'{outdir}/voronoi_furthest_sites_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved voronoi_furthest_sites_2d.png')

print('\nAll 2D Voronoi examples done.')
