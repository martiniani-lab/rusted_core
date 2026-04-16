use ang::atan2;
use libm::hypot;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Axis, Dim, ShapeBuilder};
use numpy::ndarray::ArrayViewD;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

use crate::geometry::{
    vec_no_clone,
    ensure_periodicity,
    rdf_normalisation,
};
use crate::spatial::{
    compute_periodic_rstar_tree,
    compute_periodic_rstar_tree_3d,
};

pub fn compute_radial_correlations_2d(
    points: &ArrayViewD<'_, f64>,
    fields: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    periodic: bool,
    connected: bool,
) -> (Vec<f64>, Array<f64, Dim<[usize; 2]>>) {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let box_lengths = vec![box_size_x, box_size_y];

    // get the needed parameters from the input
    let n_particles = points.shape()[0];
    let field_dim = fields.shape()[1];
    let max_dist = if periodic {
        hypot(box_size_x / 2.0, box_size_y / 2.0)
    } else {
        hypot(box_size_x, box_size_y)
    };
    let nbins = (max_dist / binsize).ceil() as usize;
    let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    let field_corrs: Vec<Vec<Arc<RwLock<f64>>>> =
        vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); field_dim]; nbins];

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec_no_clone![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum();
    }

    for dim in 0..field_dim {
        mean_field[dim] /= n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    let counts: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let xi = points[[i, 0]];
            let xj = points[[j, 0]];
            let yi = points[[i, 1]];
            let yj = points[[j, 1]];

            let mut r_ij = vec![xj - xi, yj - yi];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(r_ij[0], r_ij[1]);
            assert!(
                dist_ij >= 0.0 && dist_ij < max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist_ij / binsize).floor() as usize;
            *(counts[index].write().unwrap()) += 1.0;

            // Also compute the field correlation
            for k in 0..field_dim {
                if connected {
                    *(field_corrs[index][k].write().unwrap()) +=
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]);
                } else {
                    *(field_corrs[index][k].write().unwrap()) +=
                        fields[[i, k]] * fields[[j, k]];
                }
            }
        }
    });

    (0..nbins).into_par_iter().for_each(|bin| {
        // normalise the values of the correlations by counts
        let current_count = *counts[bin].read().unwrap();
        if current_count != 0.0 {
            for k in 0..field_dim {
                *(field_corrs[bin][k].write().unwrap()) /= current_count;
            }

            // THEN only, compute the rdf from counts
            let bincenter = (bin as f64 + 0.5) * binsize;
            // Use the actual normalization in a square, not the lazy one, if periodic
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *(rdf[bin].write().unwrap()) =
                current_count / (n_particles as f64 * normalisation / 2.0); // the number of count is 1/ averaged over N 2/ normalised by the uniform case 3/ divided by two because pairs are counted once only
        }
    });

    let mut rdf_vector = vec![0.0; nbins];
    let mut field_corrs_vector = Array::<f64, _>::zeros((nbins, field_dim).f());


    field_corrs_vector
    .axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(rdf_vector.par_iter_mut())
    .enumerate()
    .for_each(|(bin, (mut field_bin, rdf_vector_bin))| {
        *rdf_vector_bin = *rdf.get(bin).unwrap().read().unwrap();
        for dim in 0..field_dim {
            field_bin[dim] = *field_corrs
            .get(bin)
            .unwrap()
            .get(dim)
            .unwrap()
            .read()
            .unwrap();
        }
    });

    return (rdf_vector, field_corrs_vector);
}


pub fn compute_pnn_rdf(
    points: &ArrayViewD<'_, f64>,
    box_lengths: &Vec<f64>,
    binsize: f64,
    nn_order: usize,
    periodic: bool
) -> Vec<f64> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let npoints = points.shape()[0];
    let ndim = points.shape()[1];

    assert!(npoints > 1);
    assert!(ndim > 1);
    assert!(ndim < 4);
    assert!(box_lengths.len() == ndim);

    let max_dist = if periodic {
        box_lengths.iter().map(|x| (x/2.0).powi(2)).sum::<f64>().sqrt()
    } else {
        box_lengths.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    };
    let nbins = (max_dist / binsize).ceil() as usize;
    let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    let counts: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];

    if ndim == 2 { // 2D case

        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

        (0..npoints).into_par_iter().for_each(|i| {
            let current_point = [points[[i,0]], points[[i,1]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(nn_order);
            let (_, dist2) = iter.next().unwrap();

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist2.sqrt() / binsize).floor() as usize;
            assert!(index < nbins);
            *(counts[index].write().unwrap()) += 1.0;
        });

    } else {
        // Construct the R*-tree, taking into account periodic boundary conditions
        // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
        let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        (0..npoints).into_par_iter().for_each(|i| {
            let current_point = [points[[i,0]], points[[i,1]], points[[i,2]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(nn_order);
            let (_, dist2) = iter.next().unwrap();

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist2.sqrt() / binsize).floor() as usize;
            assert!(index < nbins);
            *(counts[index].write().unwrap()) += 1.0;
        });
    }


    (0..nbins).into_par_iter().for_each(|bin| {
        // normalise the values of the correlations by counts
        let current_count = *counts[bin].read().unwrap();
        if current_count != 0.0 {

            // THEN only, compute the rdf from counts
            let bincenter = (bin as f64 + 0.5) * binsize;
            // Use the actual normalization in a square, not the lazy one, if periodic
            let normalisation = rdf_normalisation(&box_lengths, npoints, bincenter, binsize, periodic);
            *(rdf[bin].write().unwrap()) =
                current_count / (npoints as f64 * normalisation / 2.0); // the number of count is 1/ averaged over N 2/ normalised by the uniform case 3/ divided by two because pairs are counted once only
        }
    });

    let mut rdf_vector = vec![0.0; nbins];



    rdf_vector.par_iter_mut()
    .enumerate()
    .for_each(|(bin, rdf_vector_bin)| {
        *rdf_vector_bin = *rdf.get(bin).unwrap().read().unwrap();
    });

    return rdf_vector;
}


pub fn compute_radial_gyromorphic_corr_2d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    periodic: bool,
    order: u64,
) -> (Vec<f64>, Array<f64, Dim<[usize; 2]>>) {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let box_lengths = vec![box_size_x, box_size_y];

    // get the needed parameters from the input
    let n_particles = points.shape()[0];
    let max_dist = if periodic {
        hypot(box_size_x / 2.0, box_size_y / 2.0)
    } else {
        hypot(box_size_x, box_size_y)
    };
    let nbins = (max_dist / binsize).ceil() as usize;
    let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    let field_corrs: Vec<Vec<Arc<RwLock<f64>>>> =
        vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); 2]; nbins];

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    let counts: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let xi = points[[i, 0]];
            let xj = points[[j, 0]];
            let yi = points[[i, 1]];
            let yj = points[[j, 1]];

            let mut r_ij = vec![xj - xi, yj - yi];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(r_ij[0], r_ij[1]);
            assert!(
                dist_ij >= 0.0 && dist_ij < max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist_ij / binsize).floor() as usize;
            *(counts[index].write().unwrap()) += 1.0;

            // Also compute the field correlation
            let theta_ij = atan2(r_ij[1], r_ij[0]);
            let angle = order as f64 * theta_ij;
            *(field_corrs[index][0].write().unwrap()) += angle.cos();
            *(field_corrs[index][1].write().unwrap()) += angle.sin();
        }
    });

    (0..nbins).into_par_iter().for_each(|bin| {
        // normalise the values of the correlations by counts
        let current_count = *counts[bin].read().unwrap();
        if current_count != 0.0 {
            // THEN only, compute the rdf from counts
            let bincenter = (bin as f64 + 0.5) * binsize;
            // Use the actual normalization in a square, not the lazy one, if periodic
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *(rdf[bin].write().unwrap()) =
                current_count / (n_particles as f64 * normalisation / 2.0); // the number of count is 1/ averaged over N 2/ normalised by the uniform case 3/ divided by two because pairs are counted once only
                                                                            // The orientational order is just a Fourier mode of g, not an actual correlation function of a microscopic observable!
            *(field_corrs[bin][0].write().unwrap()) /=
                2.0 * PI * (n_particles as f64 * normalisation / 2.0);
            *(field_corrs[bin][1].write().unwrap()) /=
                2.0 * PI * (n_particles as f64 * normalisation / 2.0);
        }
    });

    let mut rdf_vector = vec![0.0; nbins];
    let mut field_corrs_vector = Array::<f64, _>::zeros((nbins, 2).f());

    field_corrs_vector
    .axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(rdf_vector.par_iter_mut())
    .enumerate()
    .for_each(|(bin, (mut field_bin, rdf_vector_bin))| {
        *rdf_vector_bin = *rdf.get(bin).unwrap().read().unwrap();
        for dim in 0..2 {
            field_bin[dim] = *field_corrs
            .get(bin)
            .unwrap()
            .get(dim)
            .unwrap()
            .read()
            .unwrap();
        }
    });

    return (rdf_vector, field_corrs_vector);
}

pub fn compute_radial_correlations_3d(
    points: &ArrayViewD<'_, f64>,
    fields: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    binsize: f64,
    periodic: bool,
    connected: bool,
) -> (Vec<f64>, Array<f64, Dim<[usize; 2]>>) {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let box_lengths = vec![box_size_x, box_size_y, box_size_z];

    // get the needed parameters from the input
    let n_particles = points.shape()[0];
    let field_dim = fields.shape()[1];
    let max_dist = if periodic {
        hypot(hypot(box_size_x / 2.0, box_size_y / 2.0), box_size_z / 2.0)
    } else {
        hypot(hypot(box_size_x, box_size_y), box_size_z)
    };
    let nbins = (max_dist / binsize).ceil() as usize;
    let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    let field_corrs: Vec<Vec<Arc<RwLock<f64>>>> =
        vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); field_dim]; nbins];

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec_no_clone![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum();
    }
    for dim in 0..field_dim {
        mean_field[dim] /= n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    let counts: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let xi = points[[i, 0]];
            let xj = points[[j, 0]];
            let yi = points[[i, 1]];
            let yj = points[[j, 1]];
            let zi = points[[i, 2]];
            let zj = points[[j, 2]];

            let mut r_ij = vec![xj - xi, yj - yi, zj - zi];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(hypot(r_ij[0], r_ij[1]), r_ij[2]);
            assert!(
                dist_ij >= 0.0 && dist_ij < max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist_ij / binsize).floor() as usize;
            *(counts[index].write().unwrap()) += 1.0;

            // Also compute the field correlation
            for k in 0..field_dim {
                if connected {
                    *(field_corrs[index][k].write().unwrap()) +=
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]);
                } else {
                    *(field_corrs[index][k].write().unwrap()) +=
                        fields[[i, k]] * fields[[j, k]];
                }
            }
        }
    });

    (0..nbins).into_par_iter().for_each(|bin| {
        // normalise the values of the correlations by counts
        let current_count = *counts[bin].read().unwrap();
        if current_count != 0.0 {
            for k in 0..field_dim {
                *(field_corrs[bin][k].write().unwrap()) /= current_count;
            }

            // THEN only, compute the rdf from counts
            let bincenter = (bin as f64 + 0.5) * binsize;
            // Use the actual normalization in a square, not the lazy one, if periodic
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *(rdf[bin].write().unwrap()) =
                current_count / (n_particles as f64 * normalisation / 2.0); // the number of count is 1/ averaged over N 2/ normalised by the uniform case 3/ divided by two because pairs are counted once only
        }
    });

    let mut rdf_vector = vec![0.0; nbins];
    let mut field_corrs_vector = Array::<f64, _>::zeros((nbins, field_dim).f());

    field_corrs_vector
    .axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(rdf_vector.par_iter_mut())
    .enumerate()
    .for_each(|(bin, (mut field_bin, rdf_vector_bin))| {
        *rdf_vector_bin = *rdf.get(bin).unwrap().read().unwrap();
        for dim in 0..field_dim {
            field_bin[dim] = *field_corrs
            .get(bin)
            .unwrap()
            .get(dim)
            .unwrap()
            .read()
            .unwrap();
        }
    });

    return (rdf_vector, field_corrs_vector);
}

pub fn compute_radial_correlations_2sphere(
    points: &ArrayViewD<'_, f64>,
    fields: &ArrayViewD<'_, f64>,
    binsize: f64,
    connected: bool,
) -> (Vec<f64>, Array<f64, Dim<[usize; 2]>>) {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    // get the needed parameters from the input
    assert!(points.shape()[1] == 2, "Points on the sphere should have only two coordinates, theta and phi.");
    let n_particles = points.shape()[0];
    let field_dim = fields.shape()[1];
    let max_dist = PI;

    let nbins = (max_dist / binsize).ceil() as usize;
    let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    let field_corrs: Vec<Vec<Arc<RwLock<f64>>>> =
        vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); field_dim]; nbins];

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec_no_clone![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum();
    }

    for dim in 0..field_dim {
        mean_field[dim] /= n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    let counts: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let thetai = points[[i, 0]];
            let thetaj = points[[j, 0]];
            let phii = points[[i, 1]];
            let phij = points[[j, 1]];

            let mut cos_dist_ij = thetai.cos() * thetaj.cos() + thetai.sin()* thetaj.sin() * (phii - phij).cos();
            // Deal with float precision issues at the extremes
            if cos_dist_ij < -1.0 {
                cos_dist_ij = -1.0;
            } else if cos_dist_ij > 1.0 {
                cos_dist_ij = 1.0;
            }

            let dist_ij = cos_dist_ij.acos();
            assert!(
                dist_ij >= 0.0 && dist_ij <= max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}, Cos distance {:?}",
                dist_ij, cos_dist_ij
            );

            // determine the relevant bin, and update the count at that bin for g(r)
            let index = (dist_ij / binsize).floor() as usize;
            *(counts[index].write().unwrap()) += 1.0;

            // Also compute the field correlation
            for k in 0..field_dim {
                if connected {
                    *(field_corrs[index][k].write().unwrap()) +=
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]);
                } else {
                    *(field_corrs[index][k].write().unwrap()) +=
                        fields[[i, k]] * fields[[j, k]];
                }
            }
        }
    });

    (0..nbins).into_par_iter().for_each(|bin| {
        // normalise the values of the correlations by counts
        let current_count = *counts[bin].read().unwrap();
        if current_count != 0.0 {
            for k in 0..field_dim {
                *(field_corrs[bin][k].write().unwrap()) /= current_count;
            }

            // THEN only, compute the rdf from counts
            let bincenter = (bin as f64 + 0.5) * binsize;
            // Use the actual normalization in a square, not the lazy one, if periodic
            let normalisation = (n_particles as f64).powi(2) * binsize * bincenter.sin() / 4.0;
            *(rdf[bin].write().unwrap()) =
                current_count / normalisation;
        }
    });

    let mut rdf_vector = vec![0.0; nbins];
    let mut field_corrs_vector = Array::<f64, _>::zeros((nbins, field_dim).f());


    field_corrs_vector
    .axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(rdf_vector.par_iter_mut())
    .enumerate()
    .for_each(|(bin, (mut field_bin, rdf_vector_bin))| {
        *rdf_vector_bin = *rdf.get(bin).unwrap().read().unwrap();
        for dim in 0..field_dim {
            field_bin[dim] = *field_corrs
            .get(bin)
            .unwrap()
            .get(dim)
            .unwrap()
            .read()
            .unwrap();
        }
    });

    return (rdf_vector, field_corrs_vector);
}
