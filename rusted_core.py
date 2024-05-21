import rust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cmasher as cmr
import hickle as hkl
import sys
import os
import argparse
from scipy.special import gamma

def main(input_file, columns = np.arange(2), bin_width = 1/20, output_path = "", phi = None, periodic = True, connected = False, rdf = False):
    '''
    Simple front-end for Rusted Core
    '''
    
    print("Rusted Core: Computing usual correlations in point patterns since 2023.\n© M. Casiulis, 2023.\n")
    
    input_file = os.path.abspath(input_file)
    [dname,file_name] = os.path.split(input_file)
    dname = os.path.abspath(dname)

    if '.hkl' in file_name:
        points = hkl.load(file_name)[:,columns]
    elif '.txt' in file_name:
        
        with open(input_file, 'r') as file:
            first_line = file.readline()
        # Determine the delimiter based on the first line
        if ',' in first_line:
            delimiter = ','
        elif ' ' in first_line:
            delimiter = ' '
        else:
            raise NotImplementedError("Delimiter not identified")
        
        points = np.loadtxt(input_file, delimiter=delimiter)[:,columns]
    else:
        print("Wrong file format")
        sys.exit()
        
    ndim = points.shape[1]
    npoints = points.shape[0]
    
    # Rescale the points
    points -= points.min()
    points /= points.max()
    boxsize = 1.0
    if phi is None:
        radius = 0.5 * boxsize / (npoints)**(1.0/ndim)
    else:
        # volume = surface_d * r^d, phi = N volume / boxsize^d, r = boxsize (phi/ N surface_d)^(1/d)
        radius = boxsize * (phi / (npoints * unitball_volume(ndim)) )**(1.0/ndim)
    binsize = bin_width * radius
    
    print(f"Found {npoints} points in {ndim}d\n")
    
    if ndim == 2:
        fig = plt.figure(figsize=[7,7])
        ax = fig.gca()
        ax.set_xlim((0,boxsize))
        ax.set_ylim((0,boxsize))
        r_ = ax.transData.transform([radius,0])[0] - ax.transData.transform([0,0])[0]
        marker_size = np.pi * r_**2
        pc = ax.scatter(points[:,0],points[:,1], s=marker_size, edgecolors='none')
        plt.savefig(os.path.join(output_path,file_name+'scatter.png'), bbox_inches = 'tight', pad_inches = 0,dpi=300)
        plt.close()
        
        (areas, neighbours, distances) = rust.compute_2d_all_voronoi_quantities(points, boxsize, periodic)
        
        local_phi = unitball_volume(ndim) * radius**ndim / areas
        
        if rdf:
            dummy = local_phi.reshape(npoints, 1)
            radial_rdf, area_corr = rust.compute_radial_correlations_2d(points, dummy, boxsize, binsize, periodic,connected)
            
            nbins = radial_rdf.shape[0]
            bins = (np.arange(0, nbins) + 0.5)*binsize
            fig = plt.figure()#figsize=(10,10))
            ax = fig.gca()
            pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5), linewidth=0.75)
            ax.set_xlim(0,0.5)
            ax.tick_params(labelsize=18)
            ax.set_xlabel(r"$r$",fontsize=18)
            ax.set_ylabel(r"$g(r)$",fontsize=18)
            plt.savefig(file_name+"_radial_rdf.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
            plt.close()
            
            nbins = radial_rdf.shape[0]
            bins = (np.arange(0, nbins) + 0.5)*binsize
            fig = plt.figure()#figsize=(10,10))
            ax = fig.gca()
            pc = ax.plot(bins, area_corr,c=cmr.ember(0.5), linewidth=0.75)
            ax.set_xlim(0,0.5)
            ax.tick_params(labelsize=18)
            ax.set_xlabel(r"$r$",fontsize=18)
            ax.set_ylabel(r"$C(r)$",fontsize=18)
            plt.savefig(file_name+"_corr.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
            plt.close()
    

def unitball_volume(ndim):
    return np.pi**(ndim/2.0) / gamma(ndim/2.0 + 1)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute user-defined correlations in a configuration provided via an input file.")
    # Required arguments
    parser.add_argument("input_file", type=str, help="File name")
    # Optional arguments
    ## Tasks
    parser.add_argument("-rdf", "--radial_distribution_function", action = 'store_true', help = "Compute the rdf\
        default = false", default = False)
    ## Parameters
    parser.add_argument("--n_cpus", type=int, help="Number of cpus to use for computation\
        default = os.cpu_count", default=os.cpu_count())
    parser.add_argument("--phi", type=float, help = "Packing fraction, used to determine radius\
        default = None", default = None)
    parser.add_argument("-bw", "--bin_width", type=float, help = "Width of the bins, in units of radii OR typical distances\
        default = 1/20", default = 1/20)
    parser.add_argument("-fbc", "--free_boundary_condition", action='store_true', help = "Switch to free boundary conditions instead of periodic ones\
        default = False", default = False)
    parser.add_argument("-c", "--connected", action='store_true', help = "Switch to connected correlation functions\
        default = False", default = False)
    
    args = parser.parse_args()
    
    rdf = args.radial_distribution_function
    
    input_file = args.input_file
    phi = args.phi
    bin_width = args.bin_width
    periodic = not args.free_boundary_condition
    connected = args.connected
    
    main(input_file, phi = phi, bin_width=bin_width, periodic = periodic, connected=connected, rdf = rdf)