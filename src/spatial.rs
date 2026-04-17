use ndarray::parallel::prelude::*;
use numpy::ndarray::{ArrayView1, ArrayViewD};
use std::fmt::Display;

extern crate spade;
use spade::{DelaunayTriangulation, Point2};

use rstar::{RTree, primitives::GeomWithData};

use ndarray::{Array, Dim};

use crate::geometry::{
    PointWithTag, ListPointWithTag,
    euclidean_from_spherical_single_point, great_circle_distance,
    stereographic_project,
};

pub fn compute_periodic_rstar_tree(
    positions: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool
) -> RTree<[f64;2]> {

    let n_copies: usize = if periodic { 9 } else { 1 };
    let n_particles = positions.shape()[0];

    let points_vector: Vec<[f64;2]> = (0..n_copies*n_particles).into_par_iter().filter_map(|i| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let nx = match quotient / 3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let ny = match quotient % 3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };

        let shifted_x = positions[[index, 0]] + nx * box_size_x;
        let shifted_y = positions[[index, 1]] + ny * box_size_y;

        let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
        let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;

        if is_in_x && is_in_y {
            Some([shifted_x, shifted_y])
        } else {
            None
        }
    }).collect();

    RTree::bulk_load(points_vector)
}


pub fn compute_periodic_rstar_tree_3d(
    positions: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    periodic: bool
) -> RTree<[f64;3]> {

    let n_copies: usize = if periodic { 27 } else { 1 };
    let n_particles = positions.shape()[0];

    let points_vector: Vec<[f64;3]> = (0..n_copies*n_particles).into_par_iter().filter_map(|i| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let triplet_1 = quotient % 3;
        let nonuplet = quotient / 3;
        let triplet_2 = nonuplet % 3;
        let triplet_3 = nonuplet / 3;

        let nx = match triplet_1 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let ny = match triplet_2 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let nz = match triplet_3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };

        let shifted_x = positions[[index, 0]] + nx * box_size_x;
        let shifted_y = positions[[index, 1]] + ny * box_size_y;
        let shifted_z = positions[[index, 2]] + nz * box_size_z;

        let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
        let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;
        let is_in_z = shifted_z >= -0.5 * box_size_z && shifted_z <= 1.5 * box_size_z;

        if is_in_x && is_in_y && is_in_z {
            Some([shifted_x, shifted_y, shifted_z])
        } else {
            None
        }
    }).collect();

    RTree::bulk_load(points_vector)
}

pub fn compute_decorated_rstar_tree<T: Display + Send + Sync + Copy>(
    positions: &ArrayViewD<'_, f64>,
    field: &ArrayView1<'_, T>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool
) -> RTree<ListPointWithTag<T, 2>> {

    let n_copies: usize = if periodic { 9 } else { 1 };
    let n_particles = positions.shape()[0];

    let tagged_points_vector: Vec<ListPointWithTag<T, 2>> = (0..n_copies*n_particles).into_par_iter().filter_map(|i| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let nx = match quotient / 3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let ny = match quotient % 3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };

        let shifted_x = positions[[index, 0]] + nx * box_size_x;
        let shifted_y = positions[[index, 1]] + ny * box_size_y;

        let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
        let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;

        if is_in_x && is_in_y {
            Some(ListPointWithTag::new([shifted_x, shifted_y], field[[index]]))
        } else {
            None
        }
    }).collect();

    RTree::bulk_load(tagged_points_vector)
}


pub fn compute_decorated_rstar_tree_3d<T: Display + Send + Sync + Copy>(
    positions: &ArrayViewD<'_, f64>,
    field: &ArrayView1<'_, T>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    periodic: bool
) -> RTree<ListPointWithTag<T, 3>> {

    let n_copies: usize = if periodic { 27 } else { 1 };
    let n_particles = positions.shape()[0];

    let tagged_points_vector: Vec<ListPointWithTag<T,3>> = (0..n_copies*n_particles).into_par_iter().filter_map(|i| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let triplet_1 = quotient % 3;
        let nonuplet = quotient / 3;
        let triplet_2 = nonuplet % 3;
        let triplet_3 = nonuplet / 3;

        let nx = match triplet_1 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let ny = match triplet_2 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };
        let nz = match triplet_3 { 0 => 0.0, 1 => -1.0, 2 => 1.0, _ => unreachable!() };

        let shifted_x = positions[[index, 0]] + nx * box_size_x;
        let shifted_y = positions[[index, 1]] + ny * box_size_y;
        let shifted_z = positions[[index, 2]] + nz * box_size_z;

        let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
        let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;
        let is_in_z = shifted_z >= -0.5 * box_size_z && shifted_z <= 1.5 * box_size_z;

        if is_in_x && is_in_y && is_in_z {
            Some(ListPointWithTag::new([shifted_x, shifted_y, shifted_z], field[[index]]))
        } else {
            None
        }
    }).collect();

    RTree::bulk_load(tagged_points_vector)
}

pub fn create_delaunay_with_tags<T: Display + Send + Sync + Copy>(points_array: &ArrayViewD<'_, f64>, tags: &Vec<T>, periodic: bool, box_lengths: &Vec<f64>)
-> DelaunayTriangulation<PointWithTag<T>> {
    // Delaunay triangulation where each point carries data with some arbitrary type

    // get the needed parameters from the input
    let n_particles = points_array.shape()[0];

    let n_loop = if periodic { 9 * n_particles } else { n_particles };

    let default_tag = tags[0];
    // Enforce periodicity by adding copies.
    let mut tagged_points: Vec<PointWithTag<T>> = vec![PointWithTag { position: Point2::new(0.0, 0.0), tag: default_tag}; n_loop];

    tagged_points.par_iter_mut().enumerate().for_each(|(i, tagged_pointi)| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let nx = match quotient / 3 {
            0 => 0.0,
            1 => -1.0,
            2 => 1.0,
            _ => 2.0,
        };
        let ny = match quotient % 3 {
            0 => 0.0,
            1 => -1.0,
            2 => 1.0,
            _ => 2.0,
        };

        assert!(nx < 2.0 && ny < 2.0, "Unexpected error when cloning boxes.");

        let xi = points_array[[index, 0]];
        let yi = points_array[[index, 1]];

        let shifted_x: f64 = xi + nx * box_lengths[0];
        let shifted_y: f64 = yi + ny * box_lengths[1];

        let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);

        tagged_pointi.position = newpoint;
        tagged_pointi.tag = tags[index];
    });

    // Bulk load the points
    // bulk_load_stable ensures that the ordering of points is conserved, assuming no precise overlap
    let tagged_delaunay = DelaunayTriangulation::<PointWithTag<T>>::bulk_load_stable(tagged_points).unwrap();

    return tagged_delaunay;
}

pub fn create_delaunay(points_array: &ArrayViewD<'_, f64>, periodic: bool, box_lengths: &Vec<f64>) -> DelaunayTriangulation<Point2<f64>> {

    // get the needed parameters from the input
    let n_particles = points_array.shape()[0];

    let n_loop = if periodic { 9 * n_particles } else { n_particles };

    // Enforce periodicity by adding copies.
    let mut points: Vec<Point2<f64>> = vec![Point2::new(0.0, 0.0); n_loop];

    points.par_iter_mut().enumerate().for_each(|(i, pointi)| {
        let index = i % n_particles;
        let quotient = i / n_particles;
        let nx = match quotient / 3 {
            0 => 0.0,
            1 => -1.0,
            2 => 1.0,
            _ => 2.0,
        };
        let ny = match quotient % 3 {
            0 => 0.0,
            1 => -1.0,
            2 => 1.0,
            _ => 2.0,
        };

        assert!(nx < 2.0 && ny < 2.0, "Unexpected error when cloning boxes.");

        let xi = points_array[[index, 0]];
        let yi = points_array[[index, 1]];

        let shifted_x: f64 = xi + nx * box_lengths[0];
        let shifted_y: f64 = yi + ny * box_lengths[1];

        let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);

        *pointi = newpoint;
    });

    // Bulk load the points
    // bulk_load_stable ensures that the ordering of points is conserved, assuming no precise overlap
    let delaunay = DelaunayTriangulation::<Point2<f64>>::bulk_load_stable(points).unwrap();

    return delaunay;
}

/// Build a Delaunay triangulation on the 2-sphere via stereographic projection.
/// Takes Cartesian coordinates (N×3, points on the unit sphere).
/// Each vertex carries a tag equal to its particle index so that the original
/// coordinates can be looked up from the triangulation.
/// Build a Delaunay triangulation on the 2-sphere via stereographic projection.
/// Takes Cartesian coordinates (N×3, points on the unit sphere).
///
/// One data point is chosen as the projection pole (the most peripheral one)
/// and excluded from the 2D Delaunay. The remaining N-1 points are projected
/// and triangulated. The pole's neighbors are the convex hull vertices of the
/// projected set — callers "stitch" the pole back in by treating outer faces
/// as triangles involving the pole.
///
/// Each Delaunay vertex carries a tag equal to its original particle index.
/// Returns (delaunay, pole_index).
pub fn create_delaunay_sphere(
    cartesian_points: &Array<f64, Dim<[usize; 2]>>
) -> (DelaunayTriangulation<PointWithTag<usize>>, usize) {
    let n_particles = cartesian_points.shape()[0];

    // Pick the most peripheral data point as the projection pole:
    // the one with the smallest dot product with the centroid.
    let mut cx = 0.0_f64; let mut cy = 0.0_f64; let mut cz = 0.0_f64;
    for i in 0..n_particles {
        cx += cartesian_points[[i, 0]];
        cy += cartesian_points[[i, 1]];
        cz += cartesian_points[[i, 2]];
    }
    let mut pole_idx = 0;
    let mut min_dot = f64::INFINITY;
    for i in 0..n_particles {
        let dot = cartesian_points[[i, 0]] * cx
                + cartesian_points[[i, 1]] * cy
                + cartesian_points[[i, 2]] * cz;
        if dot < min_dot {
            min_dot = dot;
            pole_idx = i;
        }
    }

    let pole = [
        cartesian_points[[pole_idx, 0]],
        cartesian_points[[pole_idx, 1]],
        cartesian_points[[pole_idx, 2]],
    ];

    // Project all points except the pole
    let projected = stereographic_project(cartesian_points, &pole);

    // Build tagged Point2 vector for the N-1 non-pole points
    let points: Vec<PointWithTag<usize>> = (0..n_particles)
        .filter(|&i| i != pole_idx)
        .map(|i| PointWithTag {
            position: Point2::new(projected[i][0], projected[i][1]),
            tag: i,
        })
        .collect();

    // bulk_load_stable preserves ordering
    (DelaunayTriangulation::<PointWithTag<usize>>::bulk_load_stable(points).unwrap(), pole_idx)
}

pub fn count_points_in_disk(rtree: &RTree<[f64;2]>, r_center: [f64; 2], radius: f64) -> usize {
    rtree.locate_within_distance(r_center, radius*radius).count()
}

pub fn count_points_in_ball(rtree: &RTree<[f64;3]>, r_center: [f64; 3], radius: f64) -> usize {
    rtree.locate_within_distance(r_center, radius*radius).count()
}

pub fn count_points_in_disk_2sphere(rtree: &RTree<GeomWithData<[f64;3],usize>>, spherical_points: &ArrayViewD<'_, f64>, centerpoint_spherical: [f64; 2], radius: f64) -> usize {

    // let centerpoint: Point3<f64> = Point3::new(r_center[0], r_center[1], r_center[2]);
    let centerpoint_euclidean = euclidean_from_spherical_single_point(centerpoint_spherical[0], centerpoint_spherical[1]);

    // let points = rtree.lookup_in_circle(&centerpoint, &(radius*radius));
    let points = rtree.locate_within_distance(centerpoint_euclidean, radius*radius).collect::<Vec<_>>();
    // Need to find, within these candidates, which points are actually within the disk in geodetic distance
    let count: usize = points.into_par_iter().map(|point_with_index| {
        let point_index = point_with_index.data;
        let point_spherical = [spherical_points[[point_index,0]], spherical_points[[point_index,1]]];
        let dist = great_circle_distance(&centerpoint_spherical, &point_spherical);
        let is_neighbor = if dist < radius { 1 } else { 0 } as usize;
        is_neighbor
    }).collect::<Vec<usize>>().into_par_iter().sum();

    return count
}
