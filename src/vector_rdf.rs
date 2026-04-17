use ang::atan2;
use libm::hypot;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Dim, ShapeBuilder, Zip};
use numpy::ndarray::ArrayViewD;
use std::f64::consts::PI;

use crate::geometry::{
    atomic_add, atomic_read, atomic_vec2d, atomic_vec3d,
    clamped_bin,
    ensure_periodicity,
    relative_distance_vec_spherical,
};
use crate::spatial::{
    compute_periodic_rstar_tree,
    compute_periodic_rstar_tree_3d,
};

pub fn compute_vector_rdf_2d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins = if periodic {
        (box_size_x / binsize).ceil() as usize
    } else {
        2 * (box_size_x / binsize).ceil() as usize
    };

    let bc_shift_factor = if periodic { 1.0 } else { 2.0 };

    let box_lengths = vec![box_size_x, box_size_y];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec2d(nbins, nbins);

    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]], points[[j, 1]] - points[[i, 1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }

            let index_x = clamped_bin((r_ij[0] + bc_shift_factor * 0.5 * box_size_x) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + bc_shift_factor * 0.5 * box_size_y) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y], 1.0);

            let index_x_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_x - r_ij[0]) / binsize, nbins);
            let index_y_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_y - r_ij[1]) / binsize, nbins);
            atomic_add(&rdf[index_x_symm][index_y_symm], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    return rdf_vector;
}


pub fn compute_bounded_vector_rdf_2d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    radial_bound: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec2d(nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_with_distance_2_iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]]]).skip(1);
        let bound2 = radial_bound * radial_bound;
        loop {
            let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
            if dist2 > bound2 { break; }
            let mut r_ij = [neighbor[0] - points[[current_index,0]], neighbor[1] - points[[current_index,1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    return rdf_vector;
}

pub fn compute_nnbounded_vector_rdf_2d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    nn_bound: usize,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let n_particles = points.shape()[0];
    // let inter_spacing = (box_size_x * box_size_y / n_particles as f64).sqrt();
    // let radial_bound = (nn_bound as f64).sqrt() * inter_spacing * 2.0;

    let radial_bound = if periodic {
        hypot(box_size_x / 2.0, box_size_y / 2.0)
    } else {
        hypot(box_size_x, box_size_y)
    };

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y];

    let rdf = atomic_vec2d(nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_iter = rtree_positions.nearest_neighbor_iter(&[points[[current_index,0]], points[[current_index,1]]]).skip(1);
        for _ in 0..nn_bound {
            let neighbor = neighbor_iter.next().unwrap();
            let mut r_ij = [neighbor[0] - points[[current_index,0]], neighbor[1] - points[[current_index,1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    return rdf_vector;
}

pub fn compute_pnn_vector_rdf_2d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    nn_order: usize,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let n_particles = points.shape()[0];
    // let inter_spacing = (box_size_x * box_size_y / n_particles as f64).sqrt();
    // let radial_bound = (nn_order as f64).sqrt() * inter_spacing * 2.0;

    let radial_bound = if periodic {
        hypot(box_size_x / 2.0, box_size_y / 2.0)
    } else {
        hypot(box_size_x, box_size_y)
    };

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y];

    let rdf = atomic_vec2d(nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_iter = rtree_positions.nearest_neighbor_iter(&[points[[current_index,0]], points[[current_index,1]]]).skip(nn_order);
        let neighbor = neighbor_iter.next().unwrap();
        let mut r_ij = [neighbor[0] - points[[current_index,0]], neighbor[1] - points[[current_index,1]]];
        if periodic {
            ensure_periodicity(&mut r_ij, &box_lengths);
        }
        let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
        let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
        atomic_add(&rdf[index_x][index_y], 1.0);
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    return rdf_vector;
}

pub fn compute_vector_gyromorphic_corr_2d(
    points: &ArrayViewD<'_, f64>,
    box_size: f64,
    binsize: f64,
    periodic: bool,
    order: u64,
) -> (Array<f64, Dim<[usize; 2]>>, Array<f64, Dim<[usize; 3]>>) {
    let npoints = points.shape()[0];
    let ndim = points.shape()[1];
    assert!(npoints > 1);
    assert!(ndim == 2);

    // Compute the correlations
    let (rdf, corr) = compute_particle_gyromorphic_correlations(
        &points, box_size, box_size, binsize, periodic, order,
    );

    return (rdf, corr);
}

pub fn compute_particle_gyromorphic_correlations(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    binsize: f64,
    periodic: bool,
    order: u64,
) -> (Array<f64, Dim<[usize; 2]>>, Array<f64, Dim<[usize; 3]>>) {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins = if periodic {
        (box_size_x / binsize).ceil() as usize
    } else {
        2 * (box_size_x / binsize).ceil() as usize
    };

    let bc_shift_factor = if periodic { 1.0 } else { 2.0 };

    let box_lengths = vec![box_size_x, box_size_y];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec2d(nbins, nbins);
    let corr = atomic_vec3d(nbins, nbins, 2);

    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]], points[[j, 1]] - points[[i, 1]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }

            let index_x = clamped_bin((r_ij[0] + bc_shift_factor * 0.5 * box_size_x) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + bc_shift_factor * 0.5 * box_size_y) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y], 1.0);

            let theta_ij = atan2(r_ij[1], r_ij[0]);
            let angle = order as f64 * theta_ij;
            let dpsinx = angle.cos();
            let dpsiny = angle.sin();

            atomic_add(&corr[index_x][index_y][0], dpsinx);
            atomic_add(&corr[index_x][index_y][1], dpsiny);

            let index_x_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_x - r_ij[0]) / binsize, nbins);
            let index_y_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_y - r_ij[1]) / binsize, nbins);
            atomic_add(&rdf[index_x_symm][index_y_symm], 1.0);

            atomic_add(&corr[index_x_symm][index_y_symm][0], dpsinx);
            atomic_add(&corr[index_x_symm][index_y_symm][1], dpsiny);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
    let mut corr_vector = Array::<f64, _>::zeros((nbins, nbins, 2).f());

    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    Zip::indexed(&mut corr_vector).par_for_each(|(i, j, dim), corr_val| {
        *corr_val = atomic_read(&corr[i][j][dim]);
    });

    return (rdf_vector, corr_vector);
}

pub fn compute_vector_rdf_3d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    binsize: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins = if periodic {
        (box_size_x / binsize).ceil() as usize
    } else {
        2 * (box_size_x / binsize).ceil() as usize
    };

    let bc_shift_factor = if periodic { 1.0 } else { 2.0 };

    let box_lengths = vec![box_size_x, box_size_y, box_size_z];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec3d(nbins, nbins, nbins);

    (0..n_particles).into_par_iter().for_each(|i| {
        for j in i + 1..n_particles {
            let mut r_ij = [points[[j, 0]] - points[[i, 0]],
                            points[[j, 1]] - points[[i, 1]],
                            points[[j, 2]] - points[[i, 2]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }

            let index_x = clamped_bin((r_ij[0] + bc_shift_factor * 0.5 * box_size_x) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + bc_shift_factor * 0.5 * box_size_y) / binsize, nbins);
            let index_z = clamped_bin((r_ij[2] + bc_shift_factor * 0.5 * box_size_z) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y][index_z], 1.0);

            let index_x_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_x - r_ij[0]) / binsize, nbins);
            let index_y_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_y - r_ij[1]) / binsize, nbins);
            let index_z_symm = clamped_bin((bc_shift_factor * 0.5 * box_size_z - r_ij[2]) / binsize, nbins);
            atomic_add(&rdf[index_x_symm][index_y_symm][index_z_symm], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j, k), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j][k]);
    });

    return rdf_vector;
}

pub fn compute_bounded_vector_rdf_3d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    binsize: f64,
    radial_bound: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y, box_size_z];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec3d(nbins, nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_with_distance_2_iter = rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]], points[[current_index, 2]]]).skip(1);
        let bound2 = radial_bound * radial_bound;
        loop {
            let (neighbor, dist2) = neighbor_with_distance_2_iter.next().unwrap();
            if dist2 > bound2 { break; }
            let mut r_ij = [neighbor[0] - points[[current_index,0]], neighbor[1] - points[[current_index,1]], neighbor[2] - points[[current_index, 2]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
            let index_z = clamped_bin((r_ij[2] + radial_bound) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y][index_z], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j, k), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j][k]);
    });

    return rdf_vector;
}

pub fn compute_nnbounded_vector_rdf_3d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    binsize: f64,
    nn_bound: usize,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    // let n_particles = points.shape()[0];
    // let inter_spacing = (box_size_x * box_size_y * box_size_z / n_particles as f64).cbrt();
    // let radial_bound = (nn_bound as f64).cbrt() * inter_spacing * 2.0;

    let radial_bound = if periodic {
        hypot(hypot(box_size_x / 2.0, box_size_y / 2.0), box_size_z / 2.0)
    } else {
        hypot(hypot(box_size_x, box_size_y),box_size_z)
    };

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y, box_size_z];

    let n_particles = points.shape()[0];
    let rdf = atomic_vec3d(nbins, nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_iter = rtree_positions.nearest_neighbor_iter(&[points[[current_index,0]], points[[current_index,1]], points[[current_index, 2]]]).skip(1);
        for _ in 0..nn_bound {
            let neighbor = neighbor_iter.next().unwrap();
            let mut r_ij = [neighbor[0] - points[[current_index,0]], neighbor[1] - points[[current_index,1]], neighbor[2] - points[[current_index, 2]]];
            if periodic {
                ensure_periodicity(&mut r_ij, &box_lengths);
            }
            let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
            let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
            let index_z = clamped_bin((r_ij[2] + radial_bound) / binsize, nbins);
            atomic_add(&rdf[index_x][index_y][index_z], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j, k), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j][k]);
    });

    return rdf_vector;
}

pub fn compute_pnn_vector_rdf_3d(
    points: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    binsize: f64,
    nn_order: usize,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let n_particles = points.shape()[0];

    let radial_bound = if periodic {
        hypot(hypot(box_size_x / 2.0, box_size_y / 2.0), box_size_z / 2.0)
    } else {
        hypot(hypot(box_size_x, box_size_y), box_size_z)
    };

    let nbins = (2.0 * radial_bound / binsize).ceil() as usize;
    let box_lengths = vec![box_size_x, box_size_y, box_size_z];

    let rdf = atomic_vec3d(nbins, nbins, nbins);

    let rtree_positions = compute_periodic_rstar_tree_3d(
        &points, box_lengths[0], box_lengths[1], box_lengths[2], periodic
    );

    (0..n_particles).into_par_iter().for_each(|current_index| {
        let mut neighbor_iter = rtree_positions.nearest_neighbor_iter(
            &[points[[current_index,0]], points[[current_index,1]], points[[current_index, 2]]]
        ).skip(nn_order);
        let neighbor = neighbor_iter.next().unwrap();
        let mut r_ij = [
            neighbor[0] - points[[current_index,0]],
            neighbor[1] - points[[current_index,1]],
            neighbor[2] - points[[current_index, 2]]
        ];
        if periodic {
            ensure_periodicity(&mut r_ij, &box_lengths);
        }
        let index_x = clamped_bin((r_ij[0] + radial_bound) / binsize, nbins);
        let index_y = clamped_bin((r_ij[1] + radial_bound) / binsize, nbins);
        let index_z = clamped_bin((r_ij[2] + radial_bound) / binsize, nbins);
        atomic_add(&rdf[index_x][index_y][index_z], 1.0);
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins, nbins).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j, k), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j][k]);
    });

    return rdf_vector;
}

pub fn compute_vector_rdf_2sphere(
    points: &ArrayViewD<'_, f64>,
    binsize: f64,
) -> Array<f64, Dim<[usize; 2]>> {
    // Check that the binsize is physical
    assert!(
        binsize > 0.0,
        "Something is wrong with the binsize used for the RDF"
    );

    let nbins_theta = (PI / binsize).ceil() as usize;
    let nbins_phi = (2.0 * PI / binsize).ceil() as usize;

    let n_particles = points.shape()[0];
    let rdf = atomic_vec2d(nbins_theta, nbins_phi);

    (0..n_particles).into_par_iter().for_each(|i| {
        for j in 0..n_particles {
            let (dist_theta, dist_phi) = relative_distance_vec_spherical(
                points[[i, 0]], points[[i, 1]], points[[j, 0]], points[[j, 1]]);

            let index_x = clamped_bin(dist_theta / binsize, nbins_theta);
            let index_y = clamped_bin(dist_phi / binsize, nbins_phi);
            atomic_add(&rdf[index_x][index_y], 1.0);
        }
    });

    let mut rdf_vector = Array::<f64, _>::zeros((nbins_theta, nbins_phi).f());
    Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
        *rdf_val = atomic_read(&rdf[i][j]);
    });

    return rdf_vector;
}
