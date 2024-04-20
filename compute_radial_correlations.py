import rust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cmasher as cmr
import hickle as hkl
import sys
import os

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
        delimiter = ' '
    else:
        raise NotImplementedError("Delimiter not identified")
    
    points = np.loadtxt(file_path, delimiter=delimiter)[:,0:2]
else:
    print("Wrong file format")
    sys.exit()

ndim = points.shape[1]
npoints = points.shape[0]

points -= points.min()
points /= points.max()
boxsize = 1.0
radius = boxsize / (npoints)**(1.0/ndim)
binsize = radius / 20.0
periodic = False
connected = False
logscaleplot = False

print(f"Found {npoints} points in {ndim}d")

if ndim == 2:
    
    boop_orders = np.array([6])
    
    boops = rust.compute_2d_boops(points, boop_orders, boxsize, periodic)
    print(boops.shape)
    
    _, gboop = rust.compute_radial_correlations_2d(points, boops[:,-1,:], boxsize, binsize, periodic,connected)
    
    nK =100
    peak_angle = 0
    nK = 82.3286
    peak_angle = 2*np.pi/6.0 *  (2.27)/(2.0*np.pi)
    K = 2.0 * np.pi * nK / boxsize
    exponent = K * points[:,0] * np.cos(peak_angle) + K * points[:,1] * np.sin(peak_angle)
    
    translational_x = np.cos(exponent)
    translational_y = np.sin(exponent)
    translational = np.vstack([translational_x,translational_y]).T
    
    np.savetxt(file_name+"_translational_"+str(nK)+".csv", translational)
    
    radial_rdf, corr = rust.compute_radial_correlations_2d(points, translational, boxsize, binsize, periodic, connected)

elif ndim == 3:
    
    dummy = np.ones(points.shape[0]).reshape(-1,1)
    
    radial_rdf, _ = rust.compute_radial_correlations_3d(points, dummy, boxsize, binsize, periodic, connected)
    
    nbins = radial_rdf.shape[0]
    bins = (np.arange(0, nbins) + 0.5)*binsize
    print(bins.shape)

    if connected:
        suffix = "_connected"
    else:
        suffix = ""
    
    np.savetxt(file_name+"_radial_rdf.csv", np.vstack([bins,radial_rdf]).T)
    fig = plt.figure()#figsize=(10,10))
    ax = fig.gca()
    pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5))
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r"$r$",fontsize=18)
    ax.set_ylabel(r"$g(r)$",fontsize=18)
    plt.savefig(file_name+"_radial_rdf.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
    plt.close()
    
    sys.exit()
    
else: 
    print("Wrong dimensionality!")
    
nbins = radial_rdf.shape[0]
bins = (np.arange(0, nbins) + 0.5)*binsize
print(bins.shape)

if connected:
    suffix = "_connected"
else:
    suffix = ""

#np.savetxt("points.csv", points)
np.savetxt(file_name+"_radial_rdf.csv", np.vstack([bins,radial_rdf]).T)
np.savetxt(file_name+"_radial_corr"+suffix+".csv", np.vstack([bins,corr[:,0]+corr[:,1]]).T)
data = np.loadtxt(file_name+"_radial_rdf.csv")
bins = data[:,0]
radial_rdf = data[:,1]

fig = plt.figure()#figsize=(10,10))
ax = fig.gca()
ax.set_xlim(0,0.5)
pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5))
ax.tick_params(labelsize=18)
ax.set_xlabel(r"$r$",fontsize=18)
ax.set_ylabel(r"$g(r)$",fontsize=18)
plt.savefig(file_name+"_radial_rdf.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, corr[:,0]+corr[:,1])
ax.set_xlim(0,0.5)
plt.savefig(file_name+"_radial_corr_"+str(nK)+suffix+".png", dpi = 300)
plt.close()

np.savetxt(file_name+"_radial_gboop"+suffix+ ".csv", np.vstack([bins,gboop[:,0]+gboop[:,1]]).T)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, gboop[:,0]+gboop[:,1])
ax.set_xlim(0,0.5)
plt.savefig(file_name+"_radial_gboop"+suffix+".png", dpi = 300)
plt.close()

np.savetxt(file_name+"_boop"+str(boop_orders[-1])+"_test.csv", np.vstack([boops[:,-1,0],boops[:,-1,1]]).T)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.scatter(boops[:,-1,0], boops[:,-1,1])
plt.savefig(file_name+"psi"+str(boop_orders[-1])+"_test.png", dpi = 300)
plt.close()
