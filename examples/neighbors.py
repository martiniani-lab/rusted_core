"""
Nearest-neighbor statistics, metric neighbor counting, and clustering.

Demonstrates: compute_pnn_distances, compute_pnn_mean_nnbound_distances,
  metric_neighbors, monodisperse_metric_neighbors,
  metric_neighbors_2sphere, monodisperse_metric_neighbors_2sphere,
  cluster_by_distance

Test patterns: Poisson, low-density RSA, bidisperse disks, random sphere
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

# --- Generate test patterns ---

def poisson_2d(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2)

def rsa_2d(radius, max_attempts=100000, seed=42):
    """Random sequential adsorption of non-overlapping disks in [0,1]^2 with PBC."""
    np.random.seed(seed)
    pts = []
    for _ in range(max_attempts):
        candidate = np.random.rand(2)
        overlap = False
        for p in pts:
            d = candidate - p
            d -= np.round(d)  # PBC
            if np.dot(d, d) < (2 * radius)**2:
                overlap = True
                break
        if not overlap:
            pts.append(candidate)
    return np.array(pts)

def bidisperse_disks(N, ratio=1.4, seed=42):
    """Random positions with two radii (large and small)."""
    np.random.seed(seed)
    pts = np.random.rand(N, 2)
    radii = np.where(np.arange(N) < N//2, 0.01, 0.01 * ratio)
    return pts, radii

# --- 1. p-th nearest neighbor distances ---
pts = poisson_2d(500)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distance distributions for different p
for p in [1, 2, 3, 5, 10]:
    dists = rust.compute_pnn_distances(pts, p, boxsize, True)
    axes[0].hist(dists, bins=40, density=True, alpha=0.5, label=f'p={p}')
axes[0].set_xlabel('Distance to p-th NN')
axes[0].set_ylabel('Density')
axes[0].set_title('p-th NN distance distributions — Poisson')
axes[0].legend()

# Mean distance vs p
mean_dists = rust.compute_pnn_mean_nnbound_distances(pts, 20, boxsize, True)
rho = pts.shape[0] / boxsize**2
# Expected for Poisson: d_p ~ sqrt(p / (rho * pi))
p_vals = np.arange(1, 21)
expected = np.sqrt(p_vals / (rho * np.pi))

axes[1].plot(p_vals, mean_dists, 'o-', label='Measured', markersize=4)
axes[1].plot(p_vals, expected, '--', label=r'$\sqrt{p/\rho\pi}$', linewidth=0.8)
axes[1].set_xlabel('p (neighbor order)')
axes[1].set_ylabel('Mean distance')
axes[1].set_title('Mean p-th NN distance vs p')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{outdir}/pnn_distances.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved pnn_distances.png')

# --- 2. Metric neighbors (monodisperse) ---
pts_rsa = rsa_2d(radius=0.02)
N_rsa = pts_rsa.shape[0]
print(f'RSA: {N_rsa} particles (radius=0.02)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (pts_in, label) in [(axes[0], (poisson_2d(500), 'Poisson')),
                              (axes[1], (pts_rsa, 'RSA'))]:
    nc = rust.monodisperse_metric_neighbors(pts_in, 0.01, 1.5, boxsize, True)
    sc = ax.scatter(pts_in[:, 0], pts_in[:, 1], c=nc, cmap='viridis', s=10, vmin=0)
    plt.colorbar(sc, ax=ax, label='metric neighbors')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(f'{label} — metric neighbor count (threshold=1.5d)')

plt.tight_layout()
plt.savefig(f'{outdir}/metric_neighbors.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved metric_neighbors.png')

# --- 3. Polydisperse metric neighbors ---
pts_bi, radii_bi = bidisperse_disks(300)
nc_poly = rust.metric_neighbors(pts_bi, radii_bi, 1.2, boxsize, True)
nc_mono = rust.monodisperse_metric_neighbors(pts_bi, 0.01, 1.2, boxsize, True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, nc, title in [(axes[0], nc_mono, 'Monodisperse counting'),
                        (axes[1], nc_poly, 'Polydisperse counting')]:
    # Size points by their radius
    sizes = radii_bi * 5000
    sc = ax.scatter(pts_bi[:, 0], pts_bi[:, 1], c=nc, s=sizes, cmap='viridis',
                     edgecolors='k', linewidths=0.3, vmin=0)
    plt.colorbar(sc, ax=ax, label='metric neighbors')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title)

plt.suptitle('Bidisperse disks — metric neighbors (threshold=1.2)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/metric_neighbors_polydisperse.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved metric_neighbors_polydisperse.png')

# --- 4. Distance-based clustering ---
np.random.seed(5)
# 6 Gaussian blobs, some pairs closer together so they merge at larger threshold
centers = np.array([
    [0.15, 0.15], [0.25, 0.22],   # close pair
    [0.65, 0.75], [0.78, 0.72],   # close pair
    [0.15, 0.75], [0.50, 0.40],   # isolated
])
pts_clustered = np.vstack([c + 0.03 * np.random.randn(30, 2) for c in centers]) % 1.0

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, threshold, title in [(axes[0], 0.04, 'threshold=0.04'),
                               (axes[1], 0.17, 'threshold=0.17')]:
    cid = rust.cluster_by_distance(pts_clustered, threshold, boxsize, True)
    ax.scatter(pts_clustered[:, 0], pts_clustered[:, 1], c=cid, cmap='tab20', s=20)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    n_clusters = len(np.unique(cid))
    ax.set_title(f'{title} ({n_clusters} clusters)')

plt.suptitle('Distance-based clustering — Gaussian blobs', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved clustering.png')

# --- 5. Metric neighbors on the sphere ---
def random_sphere(N, seed=42):
    np.random.seed(seed)
    return np.column_stack([np.arccos(2*np.random.rand(N)-1), 2*np.pi*np.random.rand(N)])

pts_sph = random_sphere(500)
nc_sph = rust.monodisperse_metric_neighbors_2sphere(pts_sph, 0.05, 1.5)

fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})
cart = np.column_stack([np.sin(pts_sph[:,0])*np.cos(pts_sph[:,1]),
                        np.sin(pts_sph[:,0])*np.sin(pts_sph[:,1]),
                        np.cos(pts_sph[:,0])])
lon = np.arctan2(cart[:,1], cart[:,0])
lat = np.arcsin(np.clip(cart[:,2], -1, 1))
sc = ax.scatter(lon, lat, c=nc_sph, cmap='viridis', s=8)
plt.colorbar(sc, ax=ax, label='metric neighbors', shrink=0.7)
ax.grid(True, alpha=0.2)
ax.set_title('Metric neighbors on sphere (monodisperse, threshold=1.5)')
plt.savefig(f'{outdir}/metric_neighbors_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved metric_neighbors_sphere.png')

print('\nAll neighbor examples done.')
