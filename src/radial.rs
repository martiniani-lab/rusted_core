use ang::atan2;
use libm::hypot;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Axis, Dim, ShapeBuilder};
use numpy::ndarray::ArrayViewD;
use std::f64::consts::PI;
use crate::geometry::{
    atomic_add, atomic_read, atomic_vec, atomic_vec2d,
    clamped_bin,
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
    let counts = atomic_vec(nbins);
    let field_corrs = atomic_vec2d(nbins, field_dim);

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum::<f64>() / n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]], points[[j, 1]] - points[[i, 1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(r_ij[0], r_ij[1]);
            assert!(
                dist_ij >= 0.0 && dist_ij <= max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            let index = clamped_bin(dist_ij / binsize, nbins);
            atomic_add(&counts[index], 1.0);

            for k in 0..field_dim {
                if connected {
                    atomic_add(&field_corrs[index][k],
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]));
                } else {
                    atomic_add(&field_corrs[index][k],
                        fields[[i, k]] * fields[[j, k]]);
                }
            }
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
        let current_count = atomic_read(&counts[bin]);
        if current_count != 0.0 {
            for k in 0..field_dim {
                field_bin[k] = atomic_read(&field_corrs[bin][k]) / current_count;
            }
            let bincenter = (bin as f64 + 0.5) * binsize;
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *rdf_vector_bin = current_count / (n_particles as f64 * normalisation / 2.0);
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
    let counts = atomic_vec(nbins);

    if ndim == 2 {
        let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

        (0..npoints).into_par_iter().for_each(|i| {
            let current_point = [points[[i,0]], points[[i,1]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(nn_order);
            let (_, dist2) = iter.next().unwrap();
            let index = clamped_bin(dist2.sqrt() / binsize, nbins);
            atomic_add(&counts[index], 1.0);
        });

    } else {
        let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

        (0..npoints).into_par_iter().for_each(|i| {
            let current_point = [points[[i,0]], points[[i,1]], points[[i,2]]];
            let mut iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&current_point).skip(nn_order);
            let (_, dist2) = iter.next().unwrap();
            let index = clamped_bin(dist2.sqrt() / binsize, nbins);
            atomic_add(&counts[index], 1.0);
        });
    }

    let mut rdf_vector = vec![0.0; nbins];

    rdf_vector.par_iter_mut()
    .enumerate()
    .for_each(|(bin, rdf_vector_bin)| {
        let current_count = atomic_read(&counts[bin]);
        if current_count != 0.0 {
            let bincenter = (bin as f64 + 0.5) * binsize;
            let normalisation = rdf_normalisation(&box_lengths, npoints, bincenter, binsize, periodic);
            *rdf_vector_bin = current_count / (npoints as f64 * normalisation / 2.0);
        }
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
    let counts = atomic_vec(nbins);
    let field_corrs = atomic_vec2d(nbins, 2);

    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]], points[[j, 1]] - points[[i, 1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(r_ij[0], r_ij[1]);
            assert!(
                dist_ij >= 0.0 && dist_ij <= max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            let index = clamped_bin(dist_ij / binsize, nbins);
            atomic_add(&counts[index], 1.0);

            let theta_ij = atan2(r_ij[1], r_ij[0]);
            let angle = order as f64 * theta_ij;
            atomic_add(&field_corrs[index][0], angle.cos());
            atomic_add(&field_corrs[index][1], angle.sin());
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
        let current_count = atomic_read(&counts[bin]);
        if current_count != 0.0 {
            let bincenter = (bin as f64 + 0.5) * binsize;
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *rdf_vector_bin = current_count / (n_particles as f64 * normalisation / 2.0);
            let norm_factor = 2.0 * PI * (n_particles as f64 * normalisation / 2.0);
            field_bin[0] = atomic_read(&field_corrs[bin][0]) / norm_factor;
            field_bin[1] = atomic_read(&field_corrs[bin][1]) / norm_factor;
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
    let counts = atomic_vec(nbins);
    let field_corrs = atomic_vec2d(nbins, field_dim);

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum::<f64>() / n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]],
                            points[[j, 1]] - points[[i, 1]],
                            points[[j, 2]] - points[[i, 2]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let dist_ij = hypot(hypot(r_ij[0], r_ij[1]), r_ij[2]);
            assert!(
                dist_ij >= 0.0 && dist_ij <= max_dist,
                "Something is wrong with the distance between particles!\nDistance: {:?}",
                dist_ij
            );

            let index = clamped_bin(dist_ij / binsize, nbins);
            atomic_add(&counts[index], 1.0);

            for k in 0..field_dim {
                if connected {
                    atomic_add(&field_corrs[index][k],
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]));
                } else {
                    atomic_add(&field_corrs[index][k],
                        fields[[i, k]] * fields[[j, k]]);
                }
            }
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
        let current_count = atomic_read(&counts[bin]);
        if current_count != 0.0 {
            for k in 0..field_dim {
                field_bin[k] = atomic_read(&field_corrs[bin][k]) / current_count;
            }
            let bincenter = (bin as f64 + 0.5) * binsize;
            let normalisation = rdf_normalisation(&box_lengths, n_particles, bincenter, binsize, periodic);
            *rdf_vector_bin = current_count / (n_particles as f64 * normalisation / 2.0);
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
    let counts = atomic_vec(nbins);
    let field_corrs = atomic_vec2d(nbins, field_dim);

    // compute the mean values of the quantities used in correlations
    let mut mean_field: Vec<f64> = vec![0.0; field_dim];
    for dim in 0..field_dim {
        mean_field[dim] = fields.slice(s![.., dim]).into_par_iter().sum::<f64>() / n_particles as f64;
    }

    // go through all pairs just once for all correlations and compute both rdf and the correlation
    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let thetai = points[[i, 0]];
            let thetaj = points[[j, 0]];
            let phii = points[[i, 1]];
            let phij = points[[j, 1]];

            let mut cos_dist_ij = thetai.cos() * thetaj.cos() + thetai.sin()* thetaj.sin() * (phii - phij).cos();
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

            let index = clamped_bin(dist_ij / binsize, nbins);
            atomic_add(&counts[index], 1.0);

            for k in 0..field_dim {
                if connected {
                    atomic_add(&field_corrs[index][k],
                        (fields[[i, k]] - mean_field[k]) * (fields[[j, k]] - mean_field[k]));
                } else {
                    atomic_add(&field_corrs[index][k],
                        fields[[i, k]] * fields[[j, k]]);
                }
            }
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
        let current_count = atomic_read(&counts[bin]);
        if current_count != 0.0 {
            for k in 0..field_dim {
                field_bin[k] = atomic_read(&field_corrs[bin][k]) / current_count;
            }
            let bincenter = (bin as f64 + 0.5) * binsize;
            let normalisation = (n_particles as f64).powi(2) * binsize * bincenter.sin() / 4.0;
            *rdf_vector_bin = current_count / normalisation;
        }
    });

    return (rdf_vector, field_corrs_vector);
}
