use libm::hypot;
use ndarray::parallel::prelude::*;
use ndarray::Array;
use numpy::ndarray::{ArrayView1, ArrayViewD};
use rayon::iter::ParallelBridge;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

extern crate spade;
use spade::internals::FixedHandleImpl;
use spade::Triangulation;

use rand::RngExt;

use crate::geometry::{
    vec_no_clone,
    ensure_periodicity,
    euclidean_from_spherical,
    euclidean_from_spherical_single_point,
    great_circle_distance,
    ParticleClusterTags,
};
use crate::spatial::{
    compute_periodic_rstar_tree,
    compute_periodic_rstar_tree_3d,
    compute_decorated_rstar_tree,
    compute_decorated_rstar_tree_3d,
    create_delaunay_with_tags,
    count_points_in_disk,
    count_points_in_ball,
    count_points_in_disk_2sphere,
};

pub fn compute_pnn_distances(points: &ArrayViewD<'_, f64>, p: usize, box_lengths: &Vec<f64>, periodic: bool) -> Vec<f64> {

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    assert!(npoints > 1);
    assert!(ndim > 1);
    assert!(ndim < 4);
    assert!(box_lengths.len() == ndim);

    let mut pnn_distances = vec!(0.0; npoints);

    if ndim == 2 { // 2D case

        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

        pnn_distances.par_iter_mut().enumerate().for_each(|(i, pnn_distance_i)| {

            let current_point = [points[[i,0]], points[[i,1]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(p);
            let (_, dist2) = iter.next().unwrap();
            *pnn_distance_i = dist2.sqrt();

        });

    } else {
        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        pnn_distances.par_iter_mut().enumerate().for_each(|(i, pnn_distance_i)| {

            let current_point = [points[[i,0]], points[[i,1]], points[[i,2]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(p);
            let (_, dist2) = iter.next().unwrap();
            *pnn_distance_i = dist2.sqrt();

        });

    }

    return pnn_distances;

}

pub fn compute_pnn_mean_nnbound_distances(
    points: &ArrayViewD<'_, f64>,
    box_lengths: &Vec<f64>,
    nn_bound: usize,
    periodic: bool
) -> Vec<f64> {

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    assert!(npoints > 1);
    assert!(ndim > 1);
    assert!(ndim < 4);
    assert!(box_lengths.len() == ndim);

    // Store this in a vector that is addressable in a naïvely parallel loop, at the cost of memory
    let mut all_dists: Vec<Vec<f64>> = vec![vec![0.0; nn_bound]; npoints];

    if ndim == 2 { // 2D case

        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

        all_dists.par_iter_mut().enumerate().for_each(|(i, all_dists_i)| {
            let current_point = [points[[i,0]], points[[i,1]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(1);

            for nn_order in 0..nn_bound {
                let (_, dist2) = iter.next().unwrap();
                all_dists_i[nn_order] = dist2.sqrt();
            }

        });

    } else {
        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        all_dists.par_iter_mut().enumerate().for_each(|(i, all_dists_i)| {
            let current_point = [points[[i,0]], points[[i,1]], points[[i,2]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(1);

            for nn_order in 0..nn_bound {
                let (_, dist2) = iter.next().unwrap();
                all_dists_i[nn_order] += dist2.sqrt();
            }
        });
    }

    let mut mean_dists = vec![0.0; nn_bound];

    mean_dists.par_iter_mut().enumerate().for_each(|(nn_order, mean_dist_order)| {
        (0..npoints).into_iter().for_each(|i|{
            *mean_dist_order += all_dists[i][nn_order];
        });
        *mean_dist_order /= npoints as f64;
    });

    return mean_dists;
}

pub fn cluster_by_distance(points_array: &ArrayViewD<'_, f64>, threshold: f64, box_lengths: &Vec<f64>, periodic: bool) -> Vec<usize> {
    // tag particles by cluster they belong to, using a single threshold distance throughout

    let npoints = points_array.shape()[0];
    let ndim = points_array.shape()[1];

    assert!(npoints > 1);
    assert!(ndim == 2 || ndim == 3); // Need to implement/find delaunay in 3d

    let mut cluster_id: Vec<usize> = (0..npoints).collect();
    let mut done = false;

    // Make this into a LeftRight here and in point variance https://stackoverflow.com/questions/33390395/can-a-function-return-different-types-depending-on-conditional-statements-in-the
    if ndim == 2 {

        let mut delaunay = create_delaunay_with_tags(points_array, &cluster_id, periodic, &box_lengths);
        while !done {
            let change_list = Arc::new(RwLock::new(Vec::new()));

            cluster_id.par_iter_mut().enumerate().for_each( | (i, id)  |  {
                // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
                let fixed_handle = FixedHandleImpl::from_index(i);
                let dynamic_handle = delaunay.vertex(fixed_handle);
                let outgoing_edges = dynamic_handle.out_edges();

                let xi = points_array[[i, 0]];
                let yi = points_array[[i, 1]];

                for e in outgoing_edges {

                    let neigh_vertex = e.to().position();
                    let dx = neigh_vertex.x - xi;
                    let dy = neigh_vertex.y - yi;
                    let mut vector = vec![dx, dy];
                    if periodic {
                        ensure_periodicity(&mut vector, &box_lengths);
                    }

                    // Inherit lower of 2 tags if distance is smaller than threshold
                    let edge_length = hypot(vector[0], vector[1]);
                    let neigh_tag = e.to().data().tag;
                    if edge_length <= threshold && *id > neigh_tag {
                            *id = neigh_tag;
                            change_list.write().unwrap().push(i);
                    }

                }

            } );

            // Update tags in delaunay
            // TODO make this part better
            change_list.read().unwrap().clone().into_iter().for_each(| index | {
            let fixed_handle = FixedHandleImpl::from_index(index);
            (*delaunay.vertex_data_mut(fixed_handle)).tag = cluster_id[index];
            if periodic {
                for p in 1..8 {
                    let fixed_handle = FixedHandleImpl::from_index(index + npoints * p);
                    (*delaunay.vertex_data_mut(fixed_handle)).tag = cluster_id[index];
                }
            }

            });

            done = change_list.read().unwrap().len() == 0;
        }

    } else if ndim == 3 {

        // In that case, use an rtree.
        // It's a bit more annoying to navigate than a delaunay for propagation because order is not trivially preserved
        // Easy strategy is to have a vec<usize> with 2 elements as a tag, particle_id and cluster_id
        let id_pair_vec: Vec<ParticleClusterTags> = (0..npoints).zip(0..npoints).map(|(i,j)| { ParticleClusterTags{id:i, cluster_id:j} } ).collect();
        let id_pair_array = Array::from_vec(id_pair_vec);

        let mut rtree = compute_decorated_rstar_tree_3d(points_array, &id_pair_array.view(), box_lengths[0], box_lengths[1], box_lengths[2], periodic);
        while !done {
            // let change_list = Arc::new(RwLock::new(Vec::new()));
            let has_not_changed = Arc::new(RwLock::new(true));

            cluster_id.par_iter_mut().enumerate().for_each( | (i, id)  |  {
                // Here the fast option is to go through nearest neighbors and update based on that

                let mut neighbor_with_distance_2_iter = rtree.nearest_neighbor_iter_with_distance_2(&[points_array[[i,0]], points_array[[i,1]], points_array[[i,2]]]).skip(1);
                let mut still_neighbors = true;
                while still_neighbors {
                    let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
                    // let neigh_id = neighbor.data.id;
                    let neigh_cluster_id = neighbor.data.cluster_id;
                    if dist2 <= threshold.powi(2) && *id > neigh_cluster_id {
                        *id = neigh_cluster_id;
                        // change_list.write().unwrap().push(i); // Replaced by brute-force parallel loop for now since rtree is not ordered
                        if *has_not_changed.read().unwrap() {
                            *has_not_changed.write().unwrap() = false;
                        }
                    } else if dist2 >= threshold.powi(2) {
                        still_neighbors = false;
                    }
                }
            } );

            // Update tags in rtree
            // TODO make this part better // Right now: brute-forcing update
            done = *has_not_changed.read().unwrap();
            if !done {
                rtree.iter_mut().par_bridge().into_par_iter().for_each(|point_with_data| {
                    let particle_id = point_with_data.data.id;
                    point_with_data.data.cluster_id = cluster_id[particle_id];
                });
            }

        }


    } else {
        panic!("Not implemented yet!");
    }

    return cluster_id;

}

pub fn point_variances(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, box_lengths: &Vec<f64>, n_samples: usize, periodic: bool) -> Vec<f64> {

    let n_radii = radii.shape()[0];
    let means  = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];
    let means2 = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    assert!(npoints > 1);
    assert!(ndim < 4);
    assert!(box_lengths.len() == ndim);

    if ndim == 2 { // 2D case

        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

        // Throw points at random: this can be parallelized if necessary
        radii.to_vec().into_iter().enumerate().for_each(|(current_index,radius)| {

            (0..n_samples).into_par_iter().for_each( |_sample| {

                let mut rng = rand::rng();

                let mut random: f64 = rng.random();
                let x_center = random;
                random = rng.random();
                let y_center = random;

                let r_center = vec![x_center, y_center];
                let count = count_points_in_disk(&rtree_positions, r_center, radius);

                *means.get(current_index).unwrap().write().unwrap() += count as f64;
                *means2.get(current_index).unwrap().write().unwrap() += (count*count) as f64;

            });
        });

    } else { // 3D case

        let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        // Throw points at random: this can be parallelized if necessary
        radii.to_vec().into_iter().enumerate().for_each(|(current_index,radius)| {
            (0..n_samples).into_par_iter().for_each( |_sample| {

                let mut rng = rand::rng();

                let mut random: f64 = rng.random();
                let x_center = random;
                random = rng.random();
                let y_center = random;
                random = rng.random();
                let z_center = random;

                let r_center = vec![x_center, y_center, z_center];
                let count = count_points_in_ball(&rtree_positions, r_center, radius);

                *means.get(current_index).unwrap().write().unwrap() += count as f64;
                *means2.get(current_index).unwrap().write().unwrap() += (count*count) as f64;

            });
        });
    }

    let mut reduced_variances = Vec::new();
    for index in 0..n_radii {
        let current_mean = *means.get(index).unwrap().read().unwrap() / n_samples as f64;
        let current_mean2 = *means2.get(index).unwrap().read().unwrap() / n_samples as f64;

        let reduced_variance = current_mean2/ (current_mean*current_mean) - 1.0;
        reduced_variances.push(reduced_variance)
    }

    return reduced_variances;
}

pub fn count_metric_neighbors(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, threshold: f64, box_lengths: &Vec<f64>, periodic: bool) -> Vec<usize> {

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    let mut neighbor_counts = vec!(0; npoints);

    assert!(npoints > 1);
    assert!(ndim < 4);
    assert!(box_lengths.len() == ndim);

    let max_radius = *radii.into_par_iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let threshold2 = threshold * threshold;

    if ndim == 2 { // 2D case

        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_decorated_rstar_tree(&points, &radii, box_lengths[0], box_lengths[1], periodic);

        radii.to_vec().into_par_iter().zip(neighbor_counts.par_iter_mut()).enumerate().for_each(|(current_index,(radius, count))| {
            let mut neighbor_with_distance_2_iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]]]).skip(1);
            let mut still_neighbors = true;
            while still_neighbors {
                let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
                let neighbor_radius = neighbor.data;
                if dist2 <= threshold2 * (radius + neighbor_radius).powi(2) {
                    *count += 1;
                } else if dist2 >= threshold2 * (radius + max_radius).powi(2) {
                    still_neighbors = false;
                }
            }
        });

    } else { // 3D case

        let rtree_positions = compute_decorated_rstar_tree_3d(&points, &radii, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        radii.to_vec().into_par_iter().zip(neighbor_counts.par_iter_mut()).enumerate().for_each(|(current_index,(radius, count))| {
            let mut neighbor_with_distance_2_iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]], points[[current_index, 2]]]).skip(1);
            let mut still_neighbors = true;
            while still_neighbors {
                let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
                let neighbor_radius = neighbor.data;
                if dist2 <= threshold * (radius + neighbor_radius).powi(2) {
                    *count += 1;
                } else if dist2 >= threshold * (radius + max_radius).powi(2) {
                    still_neighbors = false;
                }
            }
        });

    }


    return neighbor_counts;

}

pub fn point_variances_2sphere(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, n_samples: usize) -> Vec<f64> {

    let n_radii = radii.shape()[0];
    let means  = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];
    let means2 = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    assert!(npoints > 1);
    assert!(ndim == 2, "Expected points in theta, phi representation"); // Expect points in theta, phi representation

    let points_euclidean = euclidean_from_spherical(&points);
    let indices = Array::from_iter(0..npoints);

    // Decorated rstar with index of particle to find spherical coordinates again
    let rtree_positions = compute_decorated_rstar_tree_3d(&points_euclidean.into_dyn().view(), &indices.view(), 2.0, 2.0, 2.0, false);

    // Throw points at random: this can be parallelized if necessary
    radii.to_vec().into_iter().enumerate().for_each(|(current_index,radius)| {
        (0..n_samples).into_par_iter().for_each( |_sample| {

            let mut rng = rand::rng();

            let mut random: f64 = rng.random();
            let phi_center = 2.0 * PI * random;
            random = rng.random();
            let theta_center = (2.0 * random - 1.0).acos();

            let r_center = [theta_center, phi_center];
            // The Euclidean distance is a LOWER BOUND of the geodetic distance on the sphere
            // Thus, all neighbors by the great-circle distance at some cutoff are a subset of the Euclidean neighbors at the same distance
            let count = count_points_in_disk_2sphere(&rtree_positions, &points, r_center, radius);

            *means.get(current_index).unwrap().write().unwrap() += count as f64;
            *means2.get(current_index).unwrap().write().unwrap() += (count*count) as f64;

        });
    });

    let mut reduced_variances = Vec::new();
    for index in 0..n_radii {
        let current_mean = *means.get(index).unwrap().read().unwrap() / n_samples as f64;
        let current_mean2 = *means2.get(index).unwrap().read().unwrap() / n_samples as f64;

        let reduced_variance = current_mean2/ (current_mean*current_mean) - 1.0;
        reduced_variances.push(reduced_variance)
    }

    return reduced_variances;
}

pub fn count_metric_neighbors_2sphere(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, threshold: f64) -> Vec<usize> {

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    let mut neighbor_counts = vec!(0; npoints);

    assert!(npoints > 1);
    assert!(ndim == 2); // Assumes theta,phi format

    let max_radius = *radii.into_par_iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let threshold2 = threshold * threshold;

    let points_euclidean = euclidean_from_spherical(&points);
    let indices = Array::from_iter(0..npoints);

    // Decorated rstar with index of particle to find spherical coordinates again
    let rtree_positions = compute_decorated_rstar_tree_3d(&points_euclidean.into_dyn().view(), &indices.view(), 2.0, 2.0, 2.0, false);

    radii.to_vec().into_par_iter().zip(neighbor_counts.par_iter_mut()).enumerate().for_each(|(current_index,(radius, count))| {
        // Query the R-tree using Euclidean coordinates (the tree was built from points_euclidean)
        let query_point = euclidean_from_spherical_single_point(points[[current_index,0]], points[[current_index,1]]);
        let mut neighbor_with_distance_2_iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&query_point).skip(1);
        let mut still_neighbors = true;
        while still_neighbors {
            let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
            let neighbor_index = neighbor.data;

            let neighbor_radius = radii[neighbor_index];

            // The euclidean distance is a LOWER bound of the geodesic distance
            // Thus, if the Euclidean distance is bigger than the threshold, the loop can be broken
            if dist2 >= threshold2 * (radius + max_radius).powi(2) {
                still_neighbors = false;
            } else {
                // Have to check the geodesic distance
                let current_point = [points[[current_index,0]], points[[current_index,1]]];
                let neighbor_point = [points[[neighbor_index,0]], points[[neighbor_index,1]]];
                let geodesic_distance = great_circle_distance(&current_point, &neighbor_point);
                let geodesic_dist2 = geodesic_distance * geodesic_distance;

                if geodesic_dist2 <= threshold2 * (radius + neighbor_radius).powi(2) {
                    *count += 1;
                } else if geodesic_dist2 >= threshold2 * (radius + max_radius).powi(2) {
                still_neighbors = false;
                }
            }
        }
    });

    return neighbor_counts;

}
