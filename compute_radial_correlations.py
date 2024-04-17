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

points = hkl.load(file_path)[:,:-1]

ndim = points.shape[1]
npoints = points.shape[0]

points -= points.min()
points /= points.max()
boxsize = 1.0
radius = boxsize / (npoints)**(1.0/ndim)
binsize = radius / 20.0
periodic = True
logscaleplot = False

if ndim == 2:
    
    boop_orders = np.array([6])
    
    boops = rust.compute_2d_boops(points, boop_orders, boxsize, periodic)
    print(boops.shape)
    
    _, gboop = rust.compute_radial_correlations_2d(points, boops[:,-1,:], boxsize, binsize, periodic)
    
    nK =100
    K = 2.0 * np.pi * nK / boxsize
    
    translational_x = np.cos(K*points[:,0])
    translational_y = np.sin(K*points[:,0])
    translational = np.vstack([translational_x,translational_y]).T
    
    np.savetxt(file_name+"_translational_"+str(nK)+".csv", translational)
    
    radial_rdf, corr = rust.compute_radial_correlations_2d(points, translational, boxsize, binsize, periodic)

    print(radial_rdf.shape)
    # print(g7.shape)
elif ndim == 3:
    print("Not implemented in 3d for now")
else: 
    print("Wrong dimensionality!")
    
nbins = radial_rdf.shape[0]
bins = (np.arange(0, nbins) + 0.5)*binsize
print(bins.shape)

#np.savetxt("points.csv", points)
np.savetxt(file_name+"_radial_rdf_test.csv", np.vstack([bins,radial_rdf]).T)
np.savetxt(file_name+"_radial_corr_test.csv", np.vstack([bins,corr[:,0]+corr[:,1]]).T)
data = np.loadtxt(file_name+"_radial_rdf_test.csv")
bins = data[:,0]
radial_rdf = data[:,1]

fig = plt.figure()#figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5), label = 'Minimized System')
#pc = ax.plot(r/Nk, g*(r/Nk>=2*rad),c='grey',linestyle='dashed', label='Percus-Yevick')
ax.set_xlim(0,0.5)
ax.legend(loc='upper right',fontsize=24)
pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5))
ax.tick_params(labelsize=18)
ax.set_xlabel(r"$r$",fontsize=18)
ax.set_ylabel(r"$g(r)$",fontsize=18)
plt.savefig(file_name+"_radial_rdf_test.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, corr[:,0]+corr[:,1])
ax.set_xlim(0,0.5)
plt.savefig(file_name+"_radial_corr_"+str(nK)+"_test.png", dpi = 300)
plt.close()

np.savetxt(file_name+"_radial_gboop_test.csv", np.vstack([bins,gboop[:,0]+gboop[:,1]]).T)

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
pc = ax.plot(bins, gboop[:,0]+gboop[:,1])
ax.set_xlim(0,0.1)
plt.savefig(file_name+"_radial_gboop_test.png", dpi = 300)
plt.close()

# fig = plt.figure(figsize=(10,10))
# ax = fig.gca()
# pc = ax.scatter(boops[:,-1,0], boops[:,-1,1])
# plt.savefig("psi6_"+str(phi)+"_test.png", dpi = 300)
# plt.close()
