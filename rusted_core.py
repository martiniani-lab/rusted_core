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

def main(input_file, 
         columns = np.arange(2), fields_columns = None, output_path = "", skip = 1,
         rdf = False, pcf = False, radial_bound = False, nn_bound = False, nn_order = -1, voronoi_quantities = False, compute_boops = False, gyromorphic_cf = False, compute_furthest_sites = False,
         metric_clusters = False,
         bin_width = 1/20, phi = None, starting_box_size = None, boop_orders = np.array([6]), orientation_order = 6, cluster_threshold = 1.0, radial_bound_value = 0.1, nn_bound_value = 6,
         periodic = True, connected = False,
         logscaleplot = False, vmaxmax = 1e1):
    '''
    Simple front-end for Rusted Core
    '''
    
    print("Rusted Core: Computing usual correlations in point patterns since 2023.\n© M. Casiulis, 2023.\n")
    
    input_file = os.path.abspath(input_file)
    [dname,file_name] = os.path.split(input_file)
    dname = os.path.abspath(dname)
    
    if connected:
        suffix = "_connected"
    else:
        suffix = ""

    if '.hkl' in file_name:
        points = hkl.load(input_file)[:,columns]
        if fields_columns is not None:
            fields =  hkl.load(input_file)[:,fields_columns]
    elif '.csv' in file_name:
        points = np.loadtxt(input_file, delimiter=',')
    elif '.txt' in file_name:
        
        with open(input_file, 'r') as file:
            for line in range(skip+1):
                first_line = file.readline()
        # Determine the delimiter based on the first line
        if ',' in first_line:
            delimiter = ','
        elif "\t" in first_line:
            delimiter = None
        elif ' ' in first_line:
            delimiter = None
        else:
            raise NotImplementedError("Delimiter not identified")
        
        points = np.loadtxt(input_file, delimiter=delimiter, skiprows=skip)[:,columns]
        if fields_columns is not None:
            fields =  np.loadtxt(input_file, delimiter=delimiter)[:,fields_columns]
    else:
        print("Wrong file format")
        sys.exit()
        
    ndim = points.shape[1]
    npoints = points.shape[0]
    
    # Rescale the points
    points -= points.min()
    if box_size is None:
        points /= points.max()
    else:
        points /= starting_box_size
        if points.max() > 1.0:
            print("Wrong box size!")
            sys.exit()
    boxsize = 1.0
    if phi is None:
        radius = 0.5 * boxsize / (npoints)**(1.0/ndim)
    else:
        # volume = unitvolume_d * r^d, phi = N volume / boxsize^d, r = boxsize (phi/ N unitvolume_d)^(1/d)
        radius = boxsize * (phi / (npoints * unitball_volume(ndim)) )**(1.0/ndim)
    binsize = bin_width * radius
    
    print(f"Found {npoints} points in {ndim}d\n")
    
    if ndim == 2:
        fig = plt.figure(figsize=[10,10])
        ax = fig.gca()
        ax.set_xlim((0,boxsize))
        ax.set_ylim((0,boxsize))
        r_ = ax.transData.transform([radius,0])[0] - ax.transData.transform([0,0])[0]
        marker_size = np.pi * r_**2 *0.75 # Somehow a prefactor is necessary? Check this eventually
        pc = ax.scatter(points[:,0],points[:,1], s=marker_size, edgecolors='none')
        plt.savefig(os.path.join(output_path,file_name+'_scatter.png'), bbox_inches = 'tight', pad_inches = 0,dpi=300)
        plt.close()
        
        if metric_clusters:
            threshold = cluster_threshold * 2.0 * radius
            cluster_id = rust.cluster_by_distance(points, threshold, boxsize, periodic)
            
            np.savetxt(os.path.join(output_path,file_name+'_cluster_id.csv'), cluster_id, fmt="%d")
            
            # Output number of particles and gyration radius of each cluster in a txt file
            distinct_cluster_ids, cluster_counts = np.unique(cluster_id, return_counts=True)
            cluster_radii = []
            for index in distinct_cluster_ids:
                # XXX Put that on the rust side with PBCs if needed
                cluster_points = points[np.where(cluster_id == index)]
                cluster_radius = np.sqrt(np.mean(np.linalg.norm(cluster_points - np.mean(cluster_points),axis=-1)**2)) / (2.0 * radius)
                cluster_radii.append(cluster_radius)
            cluster_radii = np.array(cluster_radii)
            np.savetxt(os.path.join(output_path,file_name+'_cluster_data.txt'), np.vstack([cluster_counts, cluster_radii]).T, fmt="%d %f")
        
        if voronoi_quantities:
        
            (areas, neighbours, distances) = rust.compute_2d_all_voronoi_quantities(points, boxsize, periodic)
            local_phis = unitball_volume(ndim) * radius**ndim / areas
            
            np.savetxt(os.path.join(output_path,file_name+'_voronoi_quantities.csv'), np.vstack([areas, local_phis, neighbours, distances]).T )
        
        if compute_boops:
            
            boops = rust.compute_2d_boops(points, boop_orders, boxsize, periodic)
            np.savetxt(os.path.join(output_path,file_name+'_boops.csv'), boops.reshape(npoints, 2*boop_orders.shape[0]) )
        
        if rdf:
            if fields_columns is None and not compute_boops:
                if nn_order > 0:
                    radial_rdf = rust.compute_pnn_rdf(points, nn_order, boxsize, binsize, periodic)
                else:
                    dummy = np.ones(points.shape)
                    radial_rdf, _ = rust.compute_radial_correlations_2d(points, dummy, boxsize, binsize, periodic,connected)
                
                
                nbins = radial_rdf.shape[0]
                bins = (np.arange(0, nbins) + 0.5)*binsize
                fig = plt.figure()#figsize=(10,10))
                ax = fig.gca()
                pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5), linewidth=0.75)
                ax.set_xlim(0,0.5)
                ax.tick_params(labelsize=18)
                ax.set_xlabel(r"$r$",fontsize=18)
                ax.set_ylabel(r"$g(r)$",fontsize=18)
                plt.savefig(file_name+"_rdf.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
                plt.close()
                
                np.savetxt(os.path.join(output_path,file_name+'_rdf.csv'), np.vstack([bins, radial_rdf]).T )
                
            elif compute_boops:
            
                for i in range(boop_orders.shape[0]):
                
                    order_string = str(boop_orders[i])
                
                    radial_rdf, gboop = rust.compute_radial_correlations_2d(points, boops[:,i,:], boxsize, binsize, periodic,connected)
                
                    nbins = radial_rdf.shape[0]
                    bins = (np.arange(0, nbins) + 0.5)*binsize
                    fig = plt.figure(figsize=(10,10))
                    ax = fig.gca()
                    pc = ax.plot(bins, gboop[:,0]+gboop[:,1])
                    ax.set_xlim(0,0.5)
                    plt.savefig(file_name+"_radial_g"+order_string+"boop"+suffix+".png", dpi = 300)
                    plt.close()
                    
                    fig = plt.figure()#figsize=(10,10))
                    ax = fig.gca()
                    pc = ax.plot(bins, radial_rdf,c=cmr.ember(0.5), linewidth=0.75)
                    ax.set_xlim(0,0.5)
                    ax.tick_params(labelsize=18)
                    ax.set_xlabel(r"$r$",fontsize=18)
                    ax.set_ylabel(r"$g(r)$",fontsize=18)
                    plt.savefig(file_name+"_rdf.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
                    plt.close()
                    
                    if i == 0:
                        np.savetxt(os.path.join(output_path,file_name+'_rdf.csv'), np.vstack([bins, radial_rdf]).T )
                    np.savetxt(file_name+"_radial_g"+order_string+"boop"+suffix+ ".csv", np.vstack([bins,gboop[:,0]+gboop[:,1]]).T)
            
            else: 
                radial_rdf, fields_corr = rust.compute_radial_correlations_2d(points, fields, boxsize, binsize, periodic,connected)
            
                nbins = radial_rdf.shape[0]
                bins = (np.arange(0, nbins) + 0.5)*binsize
                fig = plt.figure()#figsize=(10,10))
                ax = fig.gca()
                for k in range(fields_corr.shape[-1]):
                    pc = ax.plot(bins, fields_corr[:,k],c=cmr.ember(0.5), linewidth=0.75)
                ax.set_xlim(0,0.5)
                ax.tick_params(labelsize=18)
                ax.set_xlabel(r"$r$",fontsize=18)
                ax.set_ylabel(r"$C(r)$",fontsize=18)
                plt.savefig(file_name+"_radial_corr.png", bbox_inches = 'tight',pad_inches = 0, dpi = 300)
                plt.close()
                
                
                np.savetxt(os.path.join(output_path,file_name+'_rdf.csv'), np.vstack([bins, radial_rdf]).T )
                np.savetxt(os.path.join(output_path,file_name+'_radia_corr.csv'), np.vstack([bins, fields_corr]).T )
                
        if pcf:
            if gyromorphic_cf:
                vector_rdf, vector_orientation = rust.compute_vector_gyromorphic_corr_2d(points, boxsize, binsize, periodic, orientation_order)
                vector_orientation = np.sum(vector_orientation**2,axis=-1)
            else:
                if nn_order > 0: 
                    vector_rdf = rust.compute_pnn_vector_rdf2d(points, nn_order, boxsize, binsize, periodic)
                elif radial_bound:
                    vector_rdf = rust.compute_bounded_vector_rdf2d(points, boxsize, binsize, radial_bound_value, periodic)
                elif nn_bound:
                    vector_rdf = rust.compute_nnbounded_vector_rdf2d(points, boxsize, binsize, nn_bound_value, periodic)
                else:
                    vector_rdf = rust.compute_vector_rdf2d(points, boxsize, binsize, periodic)
                
            nbins = np.ceil(boxsize/binsize)
            rho_n = npoints * npoints / ( boxsize * boxsize)
            vector_rdf /= rho_n * binsize * binsize
            np.savetxt(output_path+file_name+"vector_rdf.csv", vector_rdf)
            
            if periodic:
                center = int(vector_rdf.shape[0]/2)
                width = int(vector_rdf.shape[1]/2)
            else:
                center = int(vector_rdf.shape[0]/2)
                width = int(vector_rdf.shape[1]/2)
                
            fig = plt.figure(figsize=(10,10))
            ax = fig.gca()
            if logscaleplot:
                pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember, extent=[-0.5,0.5,0.5,-0.5])
            else:
                vmax = np.min([vector_rdf.max(), vmaxmax])
                pc = ax.imshow(vector_rdf[center-width:center+width+1, center-width:center+width+1], vmin = 0, vmax = vmax, cmap=cmr.ember, extent=[-0.5,0.5,0.5,-0.5])
            fig.colorbar(pc)
            plt.savefig(output_path+file_name+"_vector_rdf.png", dpi = 300)
            plt.close()
            
            if gyromorphic_cf:
                vector_orientation /= rho_n * binsize * binsize
                vector_orientation /= npoints
                np.savetxt(output_path+file_name+"vector_orientation.csv", vector_orientation)
                
                fig = plt.figure(figsize=(10,10))
                ax = fig.gca()
                if logscaleplot:
                    pc = ax.imshow(vector_orientation[center-width:center+width+1, center-width:center+width+1],norm=clr.LogNorm(vmin=1e-3,vmax=1e1), cmap=cmr.ember)
                else:
                    vmax = np.min([vector_orientation.max(), vmaxmax])
                    pc = ax.imshow(vector_orientation[center-width:center+width+1, center-width:center+width+1], vmin = 0, vmax = None, cmap=cmr.ember)
                fig.colorbar(pc)
                plt.savefig(output_path+file_name+"_vector_orientation.png", dpi = 300)
                plt.close()
                
        if compute_furthest_sites:
            furthest_sites = rust.voronoi_furthest_sites(points, boxsize, periodic)
            furthest_sites = furthest_sites[np.argwhere(furthest_sites[:,0]>= 0)[:,0],:]
            furthest_sites = furthest_sites[np.argwhere(furthest_sites[:,1]>= 0)[:,0],:]
            furthest_sites = furthest_sites[np.argwhere(furthest_sites[:,0] < boxsize)[:,0],:]
            furthest_sites = furthest_sites[np.argwhere(furthest_sites[:,1] < boxsize)[:,0],:]
            furthest_sites = furthest_sites[np.argsort(furthest_sites[:,2])]
            
            fig = plt.figure(figsize=[10,10])
            ax = fig.gca()
            ax.set_xlim((0,boxsize))
            ax.set_ylim((0,boxsize))
            r_ = ax.transData.transform([radius,0])[0] - ax.transData.transform([0,0])[0]
            marker_size = np.pi * r_**2 *0.75 # Somehow a prefactor is necessary? Check this eventually
            pc = ax.scatter(points[:,0],points[:,1], s=marker_size, edgecolors='none')
            pc = ax.scatter(furthest_sites[-5:,0],furthest_sites[-5:,1], s=0.1*marker_size, edgecolors='none', c = 'r')
            plt.savefig(os.path.join(output_path,file_name+'_furthest_site.png'), bbox_inches = 'tight', pad_inches = 0,dpi=300)
            plt.close()
            
            print(furthest_sites.shape)
            print(furthest_sites)
    

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
    parser.add_argument("-pcf", "--pair_correlation_function", action = 'store_true', help = "Compute the pcf\
        default = false", default = False)
    parser.add_argument("-rb", "--radial_bound", action = 'store_true', help = "Compute pcf up to a finite metric bound\
        default = false", default = False)
    parser.add_argument("-nnb", "--nn_bound", action = 'store_true', help = "Compute pcf up to a finite neighbor index bound\
        default = false", default = False)
    parser.add_argument("-voro", "--voronoi_quantities", action="store_true", help = "Compute Voronoi cell areas, number of neighbours, and nn distances\
        default = False", default = False)
    parser.add_argument("-boops", "--compute_boops", action="store_true", help = "Compute Steinhardt's Bond-Orientational Order Parameters\
        default = False", default = False)
    parser.add_argument("--boop_orders", nargs = "+", type = int, help = "BOOP orders to compute\
        default=6", default = None)
    parser.add_argument("-gcf", "--gyromorphic_cf", action = 'store_true', help = "Compute the ocf\
        default = false", default = False)
    parser.add_argument("-oo", "--orientation_order", type = int, help = "OCF order to compute\
        default=6", default = 6)
    parser.add_argument("-fs", "--furthest_sites", action = 'store_true', help = "Compute the list of Voronoi sites with distance to closest point\
        default = false", default = False)
    parser.add_argument("-mc","--metric_clusters", action = 'store_true', help = "Cluster particles into groups using metric distance as a filter\
        default = false", default = False)
    parser.add_argument("-th", "--cluster_threshold", type = float, help = "Value of the threshold in distance used to cluster points, in units of the diameter defined through phi\
        default = 1.0", default = 1.0)
    ## Parameters
    parser.add_argument("-col", "--columns", nargs = "+", type = int, help = "indices of columns to use for point coordinates\
        default=first two", default = None)
    parser.add_argument("-fcol", "--fields_columns", nargs = "+", type = int, help = "indices of columns to use for fields to correlate\
        default=None", default = None)
    parser.add_argument("-s", "--skip", type = int, help = "Number of lines to skip in file\
        default = 0", default = 0)
    parser.add_argument("--n_cpus", type=int, help="Number of cpus to use for computation\
        default = os.cpu_count", default=os.cpu_count())
    parser.add_argument("--phi", type=float, help = "Packing fraction, used to determine radius\
        default = None", default = None)
    parser.add_argument("-rbv", "--radial_bound_value", type = float, help = "Value of finite metric bound, in units of L\
        default = 0.1", default = 0.1)
    parser.add_argument("-nnbv", "--nn_bound_value", type = int, help = "Value of finite neighbor index bound\
        default = 6", default = 6)
    parser.add_argument("-nno", "--nn_order", type = int, help = "Value of nn order to compute rdf at\
        default = inf", default = -1)
    parser.add_argument("-L", "--box_size", type=float, help = "Box size, if exact value known\
        default = None", default = None)
    parser.add_argument("-bw", "--bin_width", type=float, help = "Width of the bins, in units of radii OR typical distances\
        default = 1/20", default = 1/20)
    parser.add_argument("-fbc", "--free_boundary_condition", action='store_true', help = "Switch to free boundary conditions instead of periodic ones\
        default = False", default = False)
    parser.add_argument("-c", "--connected", action='store_true', help = "Switch to connected correlation functions\
        default = False", default = False)
    ## Plotting options
    parser.add_argument("--logscaleplot", action='store_true', help = "Use log scales on plots where it's an option\
        default = False", default = False)
    parser.add_argument("--vmaxmax", type=float, help = "Vmax at which maps are cropped\
        default = 1e9", default = 1e9)
    parser.add_argument("-o", "--output_path", type=str, help="Path to output directory\
        Default = here", default = "")
    
    args = parser.parse_args()
    
    rdf = args.radial_distribution_function
    pcf = args.pair_correlation_function
    radial_bound = args.radial_bound
    nn_bound = args.nn_bound
    nn_order = args.nn_order
    voronoi_quantities = args.voronoi_quantities
    compute_boops = args.compute_boops
    boop_orders_args = args.boop_orders
    if boop_orders_args     != None:
        boop_orders_args = tuple(boop_orders_args)
        boop_orders = np.array(boop_orders_args)
    else:
        boop_orders = np.array([6])
    gyromorphic_cf = args.gyromorphic_cf
    orientation_order = args.orientation_order
    compute_furthest_sites = args.furthest_sites
    metric_clusters = args.metric_clusters
    cluster_threshold = args.cluster_threshold
    
    input_file = args.input_file
    columns_args = args.columns
    if columns_args     != None:
        columns_args = tuple(columns_args)
        columns = np.array(columns_args)
    else:
        columns = np.arange(2)
    fields_columns_args = args.fields_columns
    if fields_columns_args     != None:
        fields_columns_args = tuple(fields_columns_args)
        fields_columns = np.array(fields_columns_args)
    else:
        fields_columns = None
    skip = args.skip
    phi = args.phi
    radial_bound_value = args.radial_bound_value
    nn_bound_value = args.nn_bound_value
    box_size = args.box_size
    bin_width = args.bin_width
    periodic = not args.free_boundary_condition
    connected = args.connected
    
    logscaleplot = args.logscaleplot
    vmaxmax = args.vmaxmax
    output_path = args.output_path
    
    main(input_file,
         columns = columns, fields_columns = fields_columns, skip = skip, phi = phi, starting_box_size=box_size,
         bin_width=bin_width, periodic = periodic, connected=connected,
         rdf = rdf, pcf = pcf, radial_bound=radial_bound, nn_bound=nn_bound, nn_order = nn_order, voronoi_quantities = voronoi_quantities, compute_boops=compute_boops, gyromorphic_cf = gyromorphic_cf, compute_furthest_sites = compute_furthest_sites,
         metric_clusters= metric_clusters,
         orientation_order = orientation_order, boop_orders= boop_orders, cluster_threshold = cluster_threshold, radial_bound_value=radial_bound_value, nn_bound_value=nn_bound_value,
         logscaleplot = logscaleplot, vmaxmax = vmaxmax, output_path = output_path)