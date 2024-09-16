import rust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cmasher as cmr
import hickle as hkl
import sys
import os

def cut_circle(r, rad=0.5):
    idx = np.nonzero(np.linalg.norm(r,axis=-1)<=rad)
    r = np.squeeze(r[idx])
    return r

file_path = sys.argv[1]
[dname,file_name] = os.path.split(file_path)
dname = os.path.abspath(dname)

if '.hkl' in file_path:
    points = hkl.load(file_path)[:,:-1]
elif '.txt' in file_path:
    
    with open(file_path, 'r') as file:
        first_line = file.readline()
    # Determine the delimiter based on the first line
    if ',' in first_line:
        delimiter = ','
    elif ' ' in first_line:
        delimiter = None
    elif "\t" in first_line:
        delimiter = None
    else:
        raise NotImplementedError("Delimiter not identified")
    
    points = np.loadtxt(file_path, delimiter=delimiter)[:,0:2]
elif '.csv' in file_path:
    
    with open(file_path, 'r') as file:
        first_line = file.readline()
    # Determine the delimiter based on the first line
    if ',' in first_line:
        delimiter = ','
    elif ' ' in first_line:
        delimiter = ' '
    else:
        raise NotImplementedError("Delimiter not identified")
    
    points = np.loadtxt(file_path, delimiter=delimiter)[:,0:2]
else:
    print("Wrong file format")
    sys.exit()

# points = hkl.load("/home/mathias/Documents/snek/hyperalg-master/HPY2D/phi0.6/a-3.0/HPY2D_phi0.6_a-3.0_N50000000_K5050.0_points_0.hkl")[:,:-1]
points -= np.mean(points)
# points = cut_circle(points, rad=0.01)
points /= np.amax(points)
points *= 0.5
# print(np.amax(points))
# np.savetxt("excerpt_rose.csv", points)

ndim = points.shape[1]
npoints = points.shape[0]

order = 100

boxsize = 1.0
radius = boxsize / (npoints)**(1.0/ndim)
binsize = radius / 20.0
periodic = True
logscaleplot = False
vmaxmax = 2

if ndim == 2:
    # vector_rdf = rust.compute_vector_rdf2d(points, boxsize, binsize, periodic)
    vector_rdf, vector_orientation = rust.compute_vector_orientation_corr_2d(points, boxsize, binsize, periodic, order)
elif ndim == 3:
    vector_rdf = rust.compute_vector_rdf3d(points, boxsize, binsize, periodic)
    nbins = np.ceil(boxsize/binsize)
    rho_n = npoints * npoints / ( boxsize * boxsize * boxsize)
    vector_rdf /= rho_n * binsize * binsize * binsize
    
    hkl.dump(vector_rdf, file_name+"vector_rdf_test.hkl")
    
    if periodic:
        center = int(vector_rdf.shape[0]/2)
        width = int(vector_rdf.shape[1]/2)
    else:
        center = int(vector_rdf.shape[0]/4)
        width = int(vector_rdf.shape[1]/4)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    if logscaleplot:
        pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1, center],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember)
    else:
        vmax = np.min([vector_rdf.max(), vmaxmax])
        pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1, center], vmin = 0, vmax = vmax, cmap=cmr.ember)
    fig.colorbar(pc)
    plt.savefig(file_name+"_vector_rdf_test.png", dpi = 300)
    plt.close()
    
    sys.exit()
else: 
    print("Wrong dimensionality!")
    
nbins = np.ceil(boxsize/binsize)
rho_n = npoints * npoints / ( boxsize * boxsize)
vector_rdf /= rho_n * binsize * binsize
vector_orientation /= rho_n * binsize * binsize

vector_orientation = np.sum(vector_orientation**2,axis=-1)
    
np.savetxt(file_name+"vector_rdf_test.csv", vector_rdf)
np.savetxt(file_name+"vector_orientation_test.csv", vector_orientation)

if periodic:
    center = int(vector_rdf.shape[0]/2)
    width = int(vector_rdf.shape[1]/8)
else:
    center = int(vector_rdf.shape[0]/2)
    width = int(vector_rdf.shape[1]/16)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
if logscaleplot:
    pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember)
else:
    vmax = np.min([vector_rdf.max(), vmaxmax])
    pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1], vmin = 0, vmax = vmax, cmap=cmr.ember)
fig.colorbar(pc)
plt.savefig(file_name+"_vector_rdf_test.png", dpi = 300)
plt.close()

width = int(vector_rdf.shape[1]/10)
fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(vector_rdf[center, center-width:center+width+1])
plt.savefig(file_name +"_vector_rdf_test_centerline.png", dpi = 300)
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
if logscaleplot:
    pc = ax.imshow(vector_orientation[center-width:center+width+1, center-width:center+width+1],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember)
else:
    vmax = np.min([vector_orientation.max(), vmaxmax])
    pc = ax.imshow(vector_orientation[center-width:center+width+1, center-width:center+width+1], vmin = 0, vmax = None, cmap=cmr.ember)
fig.colorbar(pc)
plt.savefig(file_name+"_vector_orientation_test.png", dpi = 300)
plt.close()
