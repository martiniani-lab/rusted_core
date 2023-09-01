import rust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cmasher as cmr
import hickle as hkl
import sys

points = np.random.rand(100000,2)
points = hkl.load("/home/mathias/quasiquasi7crystal2d_K100 (2).hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY2D_phi0.6_a0.0_N262144_K256_points_0.hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY2D_phi0.6_a0.0_N50000000_K5050.0_points_0.hkl")[:,:-1]
# points = hkl.load("/home/mathias/Documents/snek/HPY3D_phi0.25_a0.0_N1000000_K64_points_0.hkl")[:,:-1]
# plt.scatter(points[:,0], points[:,1])
# plt.show()

ndim = points.shape[1]
npoints = points.shape[0]

boxsize = 1.0
radius = boxsize / (npoints)**(1.0/ndim)
binsize = radius / 20.0
periodic = False
logscaleplot = False

if ndim == 2:
    vector_rdf = rust.compute_vector_rdf2d(points, boxsize, binsize, periodic)
elif ndim == 3:
    print("Not implemented in 3d for now")
else: 
    print("Wrong dimensionality!")
    
nbins = np.ceil(boxsize/binsize)
rho_n = npoints * npoints / ( boxsize * boxsize)
vector_rdf /= rho_n * binsize * binsize
    
np.savetxt("vector_rdf_test.csv", vector_rdf)

if periodic:
    center = int(vector_rdf.shape[0]/2)
    width = int(vector_rdf.shape[1]/2)
else:
    center = int(vector_rdf.shape[0]/4)
    width = int(vector_rdf.shape[1]/4)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
if logscaleplot:
    pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember)
else:
    vmax = np.min([vector_rdf.max(), 10])
    pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1], vmin = 0, vmax = vmax, cmap=cmr.ember)
fig.colorbar(pc)
plt.savefig("vector_rdf_test.png", dpi = 300)
plt.close()

width = int(vector_rdf.shape[1]/10)
fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(vector_rdf[center, center-width:center+width+1])
plt.savefig("vector_rdf_test_centerline.png", dpi = 300)
plt.close()