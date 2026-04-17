"""
Spherical Voronoi tessellation: quantities and visualization.

Demonstrates: compute_2sphere_all_voronoi_quantities, voronoi_tessellation_2sphere

Test patterns: icosahedron, Fibonacci spiral, uniform random
"""

import rust
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

outdir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(outdir, exist_ok=True)

# --- Helpers ---

def sphere_to_cart(pts):
    return np.column_stack([np.sin(pts[:,0])*np.cos(pts[:,1]),
                            np.sin(pts[:,0])*np.sin(pts[:,1]),
                            np.cos(pts[:,0])])

def cart_to_lonlat(cart):
    return np.arctan2(cart[:,1], cart[:,0]), np.arcsin(np.clip(cart[:,2], -1, 1))

def great_circle_arc(p1, p2, n_seg=10):
    dot = np.clip(np.dot(p1, p2), -1, 1)
    omega = np.arccos(dot)
    if omega < 1e-10: return np.array([p1])
    t = np.linspace(0, 1, n_seg)
    s = np.sin(omega)
    return np.outer(np.sin((1-t)*omega)/s, p1) + np.outer(np.sin(t*omega)/s, p2)

def view_direction(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])

# --- Generate test patterns ---

def icosahedron_points():
    golden = (1 + np.sqrt(5)) / 2
    v = np.array([[0,1,golden],[0,-1,golden],[0,1,-golden],[0,-1,-golden],
                  [1,golden,0],[-1,golden,0],[1,-golden,0],[-1,-golden,0],
                  [golden,0,1],[-golden,0,1],[golden,0,-1],[-golden,0,-1]], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.column_stack([np.arccos(np.clip(v[:,2],-1,1)), np.arctan2(v[:,1],v[:,0]) % (2*np.pi)])

def fibonacci_spiral(N):
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    theta = np.arccos(1 - 2 * (i + 0.5) / N)
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    return np.column_stack([theta, phi])

def random_sphere(N, seed=42):
    np.random.seed(seed)
    return np.column_stack([np.arccos(2*np.random.rand(N)-1), 2*np.pi*np.random.rand(N)])

def rsa_sphere(exclusion_angle, max_attempts=200000, seed=42):
    """Random sequential adsorption on the sphere with a geodesic exclusion zone."""
    np.random.seed(seed)
    pts = []
    for _ in range(max_attempts):
        theta = np.arccos(2*np.random.rand() - 1)
        phi = 2*np.pi*np.random.rand()
        # Check geodesic distance to all accepted points
        overlap = False
        cos_excl = np.cos(exclusion_angle)
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        for t2, p2 in pts:
            cos_dist = ct*np.cos(t2) + st*np.sin(t2)*np.cos(phi - p2)
            if cos_dist > cos_excl:
                overlap = True
                break
        if not overlap:
            pts.append([theta, phi])
    return np.array(pts)

# --- 1. Voronoi quantities comparison ---
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

pts_rsa = rsa_sphere(0.08)
print(f'RSA on sphere: {pts_rsa.shape[0]} points (exclusion angle=0.08)')

for col, (pts, label) in enumerate([(random_sphere(500), 'Random (500)'),
                                     (pts_rsa, f'RSA ({pts_rsa.shape[0]})'),
                                     (fibonacci_spiral(500), 'Fibonacci (500)')]):
    areas, nn, dists = rust.compute_2sphere_all_voronoi_quantities(pts)

    axes[0, col].hist(areas / (4*np.pi/pts.shape[0]), bins=30, density=True,
                       color='steelblue', edgecolor='white')
    axes[0, col].set_xlabel('area / mean'); axes[0, col].set_title(f'{label} — area')

    axes[1, col].hist(nn, bins=np.arange(2.5, 12.5), density=True,
                       color='steelblue', edgecolor='white')
    axes[1, col].set_xlabel('neighbours'); axes[1, col].set_title(f'{label} — coordination')

    axes[2, col].hist(dists, bins=30, density=True, color='steelblue', edgecolor='white')
    axes[2, col].set_xlabel('NN geodesic dist'); axes[2, col].set_title(f'{label} — NN distance')

plt.tight_layout()
plt.savefig(f'{outdir}/voronoi_quantities_sphere.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved voronoi_quantities_sphere.png')

# --- 2. 3D tessellation ---

def plot_voronoi_3d(pts, title, filename, color_by='neighbours', elev=20, azim=30):
    N = pts.shape[0]
    cart = sphere_to_cart(pts)
    verts, edges, ci, co = rust.voronoi_tessellation_2sphere(pts)
    areas, nn, dists = rust.compute_2sphere_all_voronoi_quantities(pts)
    cell_sizes = np.diff(co)

    if color_by == 'neighbours':
        colors = cell_sizes.astype(float); cmap='viridis'; vmin=4; vmax=8; clabel='neighbours'
    else:
        colors = areas/(4*np.pi/N); cmap='RdBu_r'; vmin=0.5; vmax=1.5; clabel='area/mean'

    cm = plt.get_cmap(cmap); norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cam = view_direction(elev, azim)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        if np.dot(cart[i], cam) < -0.05: continue
        idx = ci[co[i]:co[i+1]]
        if len(idx) < 3: continue
        cell_v = verts[idx]
        boundary = []
        for k in range(len(idx)):
            arc = great_circle_arc(cell_v[k], cell_v[(k+1)%len(idx)], 8)
            boundary.extend(arc[:-1].tolist())
        boundary = np.array(boundary) * 1.001
        poly = Poly3DCollection([boundary], alpha=0.85, facecolor=cm(norm(colors[i])),
                                edgecolor='k', linewidths=0.2)
        ax.add_collection3d(poly)

    front = np.dot(cart, cam) > -0.05
    depth = np.dot(cart[front], cam)
    ax.scatter(cart[front,0]*1.003, cart[front,1]*1.003, cart[front,2]*1.003,
               c='black', s=2+6*depth, zorder=10, depthshade=False)

    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_zlim(-1.05,1.05)
    ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.55, label=clabel, pad=0.02)
    plt.savefig(f'{outdir}/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')

print('\n3D tessellations:')
plot_voronoi_3d(icosahedron_points(), 'Icosahedron (N=12)',
                'voronoi_3d_ico.png', 'neighbours', elev=15, azim=45)
plot_voronoi_3d(random_sphere(200, seed=7), 'Random (N=200)',
                'voronoi_3d_random200.png', 'neighbours', elev=25, azim=40)
plot_voronoi_3d(fibonacci_spiral(500), 'Fibonacci spiral (N=500)',
                'voronoi_3d_fib500.png', 'neighbours', elev=20, azim=30)
plot_voronoi_3d(fibonacci_spiral(500), 'Fibonacci spiral (N=500) — area',
                'voronoi_3d_fib500_area.png', 'area', elev=20, azim=30)

print('\nAll spherical Voronoi examples done.')
