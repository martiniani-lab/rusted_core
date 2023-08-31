import rust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cmasher as cmr
import hickle as hkl
import sys

# points = np.random.rand(10000,2)
points = hkl.load("/home/mathias/quasiquasi7crystal2d_K100 (2).hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY2D_phi0.6_a0.0_N262144_K256_points_0.hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY2D_phi0.6_a0.0_N50000000_K5050.0_points_0.hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY3D_phi0.25_a0.0_N1000000_K64_points_0.hkl")[:,:-1]
# plt.scatter(points[:,0], points[:,1])
# plt.show()

ndim = points.shape[1]
npoints = points.shape[0]

points -= points.min()
points /= points.max()
boxsize = 1.0
radius = boxsize / (npoints)**(1.0/ndim)
binsize = radius / 20.0
periodic = False
logscaleplot = False

if ndim == 2:
    
    boop_orders = np.array([5,6,7])
    
    boops = rust.compute_2d_boops(points, boop_orders, boxsize, periodic)
    print(boops.shape)
    radial_rdf, g7 = rust.compute_radial_correlations_2d(points, boops[:,-1,:], boxsize, binsize, periodic)
    print(radial_rdf.shape)
    print(g7.shape)
elif ndim == 3:
    print("Not implemented in 3d for now")
else: 
    print("Wrong dimensionality!")
    
nbins = radial_rdf.shape[0]
bins = (np.arange(0, nbins) + 0.5)*binsize
print(bins.shape)

np.savetxt("radial_rdf_test.csv", np.vstack([bins,radial_rdf]))

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, radial_rdf)
plt.savefig("radial_rdf_test.png", dpi = 300)
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.scatter(boops[:,-1,0], boops[:,-1,1])
plt.savefig("psi7_test.png", dpi = 300)
plt.close()