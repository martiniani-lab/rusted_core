use ang::atan2;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis, Dim, ShapeBuilder};
use numpy::ndarray::ArrayViewD;
use std::collections::HashMap;
use std::f64::INFINITY;

extern crate spade;
use spade::internals::FixedHandleImpl;
use spade::{DelaunayTriangulation, Point2, Triangulation};

extern crate geo;
use geo::algorithm::area::Area;
use geo::{LineString, Polygon};

use std::f64::consts::PI;

use crate::geometry::{
    ensure_periodicity,
    euclidean_from_spherical,
    tangent_frame,
    spherical_triangle_area,
    spherical_circumcenter,
    spherical_harmonic,
    great_circle_distance,
};
use crate::spatial::{
    create_delaunay, create_delaunay_sphere,
    compute_periodic_rstar_tree, compute_periodic_rstar_tree_3d,
};

pub fn compute_steinhardt_boops_2d(
    points_array: &ArrayViewD<'_, f64>,
    boop_order_array: &ArrayViewD<'_, isize>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    // get the needed parameters from the input
    let n_particles = points_array.shape()[0];
    let boop_orders_number = boop_order_array.shape()[0];

    let box_lengths = vec![box_size_x, box_size_y];

    let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f());

    let delaunay = create_delaunay(points_array, periodic, &box_lengths);

    boop_vectors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut boops_i)| {
            // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);
            let outgoing_edges = dynamic_handle.out_edges();

            let mut neighbour_count = 0;

            let xi = points_array[[i, 0]];
            let yi = points_array[[i, 1]];

            for e in outgoing_edges {
                neighbour_count += 1;

                let neigh_vertex = e.to().position();
                let dx = neigh_vertex.x - xi;
                let dy = neigh_vertex.y - yi;
                let mut vector = [dx, dy];
                if periodic {
                    ensure_periodicity(&mut vector, &box_lengths);
                }

                let theta = atan2(vector[1], vector[0]);

                for n in 0..boop_orders_number {
                    let order = boop_order_array[[n]] as f64;
                    let angle = order * theta;

                    let dpsinx = angle.cos();
                    let dpsiny = angle.sin();

                    boops_i[[n, 0]] += dpsinx;
                    boops_i[[n, 1]] += dpsiny;
                }
            }

            for n in 0..boop_orders_number {
                boops_i[[n, 0]] /= neighbour_count as f64;
                boops_i[[n, 1]] /= neighbour_count as f64;
            }
        });

    return boop_vectors;
}

pub fn compute_steinhardt_boops_2sphere(
    points_array: &ArrayViewD<'_, f64>,   // (N, 2) in theta, phi
    boop_order_array: &ArrayViewD<'_, isize>,
) -> Array<f64, Dim<[usize; 3]>> {
    let n_particles = points_array.shape()[0];
    let boop_orders_number = boop_order_array.shape()[0];

    let cartesian = euclidean_from_spherical(points_array);
    let (delaunay, pole_idx) = create_delaunay_sphere(&cartesian);

    // Collect convex hull vertex tags (= pole's neighbors on the sphere)
    let hull_tags: Vec<usize> = delaunay.convex_hull()
        .map(|edge| edge.from().data().tag)
        .collect();

    let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f());

    boop_vectors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut boops_i)| {
            let ni = [cartesian[[i, 0]], cartesian[[i, 1]], cartesian[[i, 2]]];
            let (e1, e2) = tangent_frame(&ni);

            // Collect neighbor particle indices
            let mut neighbors: Vec<usize> = Vec::new();

            if i == pole_idx {
                // Pole particle: neighbors are the convex hull vertices
                neighbors.extend_from_slice(&hull_tags);
            } else {
                // Non-pole: use Delaunay edges
                let di = if i < pole_idx { i } else { i - 1 };
                let fixed_handle = FixedHandleImpl::from_index(di);
                let dynamic_handle = delaunay.vertex(fixed_handle);

                for e in dynamic_handle.out_edges() {
                    neighbors.push(e.to().data().tag);
                    // If this edge borders the outer face, the pole is also a neighbor
                    if e.face().as_inner().is_none() {
                        neighbors.push(pole_idx);
                    }
                }
            }

            let neighbour_count = neighbors.len();
            for &j in &neighbors {
                let xj = [cartesian[[j, 0]], cartesian[[j, 1]], cartesian[[j, 2]]];

                let dot = xj[0]*ni[0] + xj[1]*ni[1] + xj[2]*ni[2];
                let dx = xj[0] - dot * ni[0];
                let dy = xj[1] - dot * ni[1];
                let dz = xj[2] - dot * ni[2];
                let comp1 = dx*e1[0] + dy*e1[1] + dz*e1[2];
                let comp2 = dx*e2[0] + dy*e2[1] + dz*e2[2];
                let theta = atan2(comp2, comp1);

                for n in 0..boop_orders_number {
                    let order = boop_order_array[[n]] as f64;
                    let angle = order * theta;
                    boops_i[[n, 0]] += angle.cos();
                    boops_i[[n, 1]] += angle.sin();
                }
            }

            for n in 0..boop_orders_number {
                boops_i[[n, 0]] /= neighbour_count as f64;
                boops_i[[n, 1]] /= neighbour_count as f64;
            }
        });

    return boop_vectors;
}

pub fn compute_voronoi_quantities_2sphere(
    points_array: &ArrayViewD<'_, f64>,   // (N, 2) in theta, phi
    voronoi_areas: bool,
    voronoi_neighbour_count: bool,
    voronoi_nn_distance: bool,
) -> (Option<Vec<f64>>, Option<Vec<usize>>, Option<Vec<f64>>) {
    let n_particles = points_array.shape()[0];

    let cartesian = euclidean_from_spherical(points_array);
    let (delaunay, pole_idx) = create_delaunay_sphere(&cartesian);
    let pole_cart = [cartesian[[pole_idx, 0]], cartesian[[pole_idx, 1]], cartesian[[pole_idx, 2]]];

    // Convex hull tags = pole's neighbors (needed for the pole particle's cell)
    let hull_tags: Vec<usize> = delaunay.convex_hull()
        .map(|edge| edge.from().data().tag)
        .collect();

    let mut areas_vector = vec![0.0; n_particles];
    let mut neighbour_counts_vector: Vec<usize> = vec![0; n_particles];
    let mut nn_distances_vector = vec![0.0; n_particles];

    // Helper: get Cartesian coords for particle tag
    let cart = |tag: usize| -> [f64; 3] {
        [cartesian[[tag, 0]], cartesian[[tag, 1]], cartesian[[tag, 2]]]
    };

    if voronoi_areas {
        areas_vector.par_iter_mut().enumerate().for_each(|(i, area_i)| {
            let pi = cart(i);
            let mut circumcenters: Vec<[f64; 3]> = Vec::new();

            if i == pole_idx {
                // Pole cell: circumcenters of (pole, hull[k], hull[k+1])
                for k in 0..hull_tags.len() {
                    let a = pole_cart;
                    let b = cart(hull_tags[k]);
                    let c = cart(hull_tags[(k + 1) % hull_tags.len()]);
                    circumcenters.push(spherical_circumcenter(&a, &b, &c));
                }
            } else {
                let di = if i < pole_idx { i } else { i - 1 };
                let fixed_handle = FixedHandleImpl::from_index(di);
                let dynamic_handle = delaunay.vertex(fixed_handle);

                // Collect edges so we can look up the previous neighbor
                let edges_vec: Vec<_> = dynamic_handle.out_edges().collect();
                let n_edges = edges_vec.len();

                // Find this vertex's two hull neighbors (for stitching)
                let hull_set: std::collections::HashSet<usize> = hull_tags.iter().copied().collect();
                let my_hull_neighbors: Vec<usize> = edges_vec.iter()
                    .map(|e| e.to().data().tag)
                    .filter(|t| hull_set.contains(t))
                    .collect();

                for k in 0..n_edges {
                    let face = edges_vec[k].face();
                    if let Some(inner_face) = face.as_inner() {
                        let [v0, v1, v2] = inner_face.vertices();
                        circumcenters.push(spherical_circumcenter(
                            &cart(v0.data().tag), &cart(v1.data().tag), &cart(v2.data().tag)));
                    } else {
                        // Outer face replaces TWO stitched triangles.
                        // The inner circumcenters before and after this gap are already
                        // in the right cyclic order — we just need to order the two
                        // stitched circumcenters correctly within the gap.
                        assert!(my_hull_neighbors.len() == 2,
                            "Hull vertex should have exactly 2 hull neighbors");
                        let sa = spherical_circumcenter(&pi, &cart(my_hull_neighbors[0]), &pole_cart);
                        let sb = spherical_circumcenter(&pi, &cart(my_hull_neighbors[1]), &pole_cart);

                        // Pick the order that matches the winding: the first stitched
                        // circumcenter should be angularly closer to the last inner one.
                        let prev_cc = if circumcenters.is_empty() {
                            // Outer face is the first in the sequence — the "previous"
                            // circumcenter wraps around from the last inner face
                            let last_inner_k = (0..n_edges).rev()
                                .find(|&j| edges_vec[j].face().as_inner().is_some())
                                .unwrap();
                            let [v0, v1, v2] = edges_vec[last_inner_k].face().as_inner().unwrap().vertices();
                            spherical_circumcenter(&cart(v0.data().tag), &cart(v1.data().tag), &cart(v2.data().tag))
                        } else {
                            *circumcenters.last().unwrap()
                        };

                        // Check angular distance from prev_cc to sa vs sb
                        let dot_a = sa[0]*prev_cc[0] + sa[1]*prev_cc[1] + sa[2]*prev_cc[2];
                        let dot_b = sb[0]*prev_cc[0] + sb[1]*prev_cc[1] + sb[2]*prev_cc[2];
                        if dot_a >= dot_b {
                            // sa is closer to prev_cc → sa comes first
                            circumcenters.push(sa);
                            circumcenters.push(sb);
                        } else {
                            circumcenters.push(sb);
                            circumcenters.push(sa);
                        }
                    }
                }
            }

            let n_verts = circumcenters.len();
            let mut cell_area = 0.0;
            for k in 0..n_verts {
                cell_area += spherical_triangle_area(&pi, &circumcenters[k], &circumcenters[(k + 1) % n_verts]);
            }
            *area_i = cell_area;
        });
    }

    if voronoi_neighbour_count {
        neighbour_counts_vector.par_iter_mut().enumerate().for_each(|(i, count_i)| {
            if i == pole_idx {
                *count_i = hull_tags.len();
            } else {
                let di = if i < pole_idx { i } else { i - 1 };
                let fixed_handle = FixedHandleImpl::from_index(di);
                for e in delaunay.vertex(fixed_handle).out_edges() {
                    *count_i += 1;
                    if e.face().as_inner().is_none() { *count_i += 1; } // pole neighbor
                }
            }
        });
    }

    if voronoi_nn_distance {
        nn_distances_vector.par_iter_mut().enumerate().for_each(|(i, distance_i)| {
            let pi_sph = [points_array[[i, 0]], points_array[[i, 1]]];
            let mut nn_distance = INFINITY;

            let neighbors: Vec<usize> = if i == pole_idx {
                hull_tags.clone()
            } else {
                let di = if i < pole_idx { i } else { i - 1 };
                let fixed_handle = FixedHandleImpl::from_index(di);
                let mut nb: Vec<usize> = delaunay.vertex(fixed_handle).out_edges()
                    .map(|e| e.to().data().tag).collect();
                // Check if pole is a neighbor (any outer face)
                let has_outer = delaunay.vertex(fixed_handle).out_edges()
                    .any(|e| e.face().as_inner().is_none());
                if has_outer { nb.push(pole_idx); }
                nb
            };

            for j in neighbors {
                let pj_sph = [points_array[[j, 0]], points_array[[j, 1]]];
                let dist = great_circle_distance(&pi_sph, &pj_sph);
                if dist < nn_distance { nn_distance = dist; }
            }
            *distance_i = nn_distance;
        });
    }

    let output_areas = if voronoi_areas { Some(areas_vector) } else { None };
    let output_counts = if voronoi_neighbour_count { Some(neighbour_counts_vector) } else { None };
    let output_distances = if voronoi_nn_distance { Some(nn_distances_vector) } else { None };

    (output_areas, output_counts, output_distances)
}

pub fn compute_voronoi_quantities_2d(
    points_array: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool,
    voronoi_areas: bool,
    voronoi_neighbour_count: bool,
    voronoi_nn_distance: bool
) -> (Option<Vec<f64>>, Option<Vec<usize>>, Option<Vec<f64>>) {
    // get the needed parameters from the input
    let n_particles = points_array.shape()[0];

    let box_lengths = vec![box_size_x, box_size_y];

    let mut areas_vector = vec![0.0; n_particles];
    let mut neighbour_counts_vector: Vec<usize> = vec![0; n_particles];
    let mut nn_distances_vector = vec![0.0; n_particles];

    let delaunay = create_delaunay(points_array, periodic, &box_lengths);

    if voronoi_areas {
        areas_vector.par_iter_mut().enumerate().for_each(|(i, voronoi_area_i)| {
        // Find polygons, then compute their areas, https://docs.rs/geo/latest/geo/algorithm/area/trait.Area.html
        let voronoi_polygon: Polygon<f64> =
        get_centered_voronoi_cell(&delaunay, i);
        *voronoi_area_i = voronoi_polygon.unsigned_area();
        });
    };

    if voronoi_neighbour_count {
        neighbour_counts_vector.par_iter_mut().enumerate().for_each(|(i, count_i)| {
            // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);
            let outgoing_edges = dynamic_handle.out_edges();
            for _e in outgoing_edges {
                *count_i += 1;
            }
        });
    }

    if voronoi_nn_distance {
        nn_distances_vector.par_iter_mut().enumerate().for_each(|(i, distance_i)| {
            // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);
            let outgoing_edges = dynamic_handle.out_edges();

            let xi = points_array[[i, 0]];
            let yi = points_array[[i, 1]];

            let mut nn_distance = INFINITY;

            for e in outgoing_edges {
                let neigh_vertex = e.to().position();
                let dx = neigh_vertex.x - xi;
                let dy = neigh_vertex.y - yi;
                let mut vector = [dx, dy];
                if periodic {
                    ensure_periodicity(&mut vector, &box_lengths);
                }
                let edge_length = (vector[0]*vector[0] + vector[1]*vector[1]).sqrt();
                if edge_length < nn_distance {
                    nn_distance = edge_length;
                }
            }

            *distance_i = nn_distance;
        });
    }

    let output_areas = if voronoi_areas {
        Some(areas_vector)
    } else {
        None
    };

    let output_counts = if voronoi_neighbour_count {
        Some(neighbour_counts_vector)
    } else {
        None
    };

    let output_distances = if voronoi_nn_distance {
        Some(nn_distances_vector)
    } else {
        None
    };

    return (output_areas, output_counts, output_distances);

}


pub fn get_centered_voronoi_cell(
    delaunay: &DelaunayTriangulation<Point2<f64>>,
    i: usize,
) -> geo::Polygon<f64> {
    // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
    let current_handle = FixedHandleImpl::from_index(i);
    // Find Voronoi faces corresponding to both vertices of the Delaunay
    let voronoi_face = delaunay.vertex(current_handle).as_voronoi_face();

    // Convert the handles to Voronoi faces into polygons, in the form of vectors of 2-vectors.
    // To understand what's being done here: go through https://docs.rs/spade/latest/spade/index.html
    // The Delaunay, https://docs.rs/spade/latest/spade/struct.DelaunayTriangulation.html can be used to get Voronoi faces https://docs.rs/spade/latest/spade/handles/type.VoronoiFace.html#
    // Voronoi faces have an impl to get oriented edges, from which one can get Voronoi vertices https://docs.rs/spade/latest/spade/handles/enum.VoronoiVertex.html
    // Then, their positions can be found
    let mut voronoi_polygon: Vec<Vec<f64>> = Vec::new();
    let mut voronoi_center: Vec<f64> = vec![0.0; 2];
    let mut voronoi_counter: usize = 0;
    for voro_edge in voronoi_face.adjacent_edges() {
        let voro_vertex = voro_edge.from().position().unwrap();
        let vertex_x = voro_vertex.x;
        let vertex_y = voro_vertex.y;
        voronoi_polygon.push(vec![vertex_x, vertex_y]);
        voronoi_center[0] += vertex_x;
        voronoi_center[1] += vertex_y;
        voronoi_counter += 1;
    }
    voronoi_center[0] /= voronoi_counter as f64;
    voronoi_center[1] /= voronoi_counter as f64;
    let mut voronoi_polygon_tuple: Vec<(f64, f64)> = Vec::new();
    for k in 0..voronoi_counter {
        voronoi_polygon[k][0] -= voronoi_center[0];
        voronoi_polygon[k][1] -= voronoi_center[1];
        voronoi_polygon_tuple.push((voronoi_polygon[k][0], voronoi_polygon[k][1]));
    }

    // Return a polygon, define polygons through geo https://docs.rs/geo/latest/geo/struct.Polygon.html
    let voronoi_polygon_as_poly = Polygon::new(LineString::from(voronoi_polygon_tuple), vec![]);
    return voronoi_polygon_as_poly;
}

pub fn voronoi_furthest_site(
    points_array: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    // get the needed parameters from the input
    let box_lengths = vec![box_size_x, box_size_y];

    let delaunay = create_delaunay(points_array, periodic, &box_lengths);

    // Iterate over all inner faces (Delaunay triangles) and get circumcenters + circumradii
    let faces: Vec<_> = delaunay.inner_faces().collect();
    let n_faces = faces.len();

    let mut furthest_sites = Array::<f64, _>::zeros((n_faces, 3).f());

    furthest_sites
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut site_i)| {
            let face = &faces[i];
            let circumcenter = face.circumcenter();

            // Compute circumradius as the distance from circumcenter to any vertex
            let v0 = face.positions()[0];
            let dx = circumcenter.x - v0.x;
            let dy = circumcenter.y - v0.y;
            let circumradius = (dx*dx + dy*dy).sqrt();

            site_i[0] = circumcenter.x;
            site_i[1] = circumcenter.y;
            site_i[2] = circumradius;
        });

    return furthest_sites;
}

/// Return the Voronoi tessellation of a 2D periodic/free point pattern.
/// Returns (vertices, edges, cell_indices, cell_offsets) where:
///   vertices:     flat [x0,y0, x1,y1, ...] — all unique Voronoi vertex positions
///   edges:        flat [i0,j0, i1,j1, ...] — pairs of vertex indices forming edges
///   cell_indices: flat vertex indices for all cells (CSR values)
///   cell_offsets: length n_particles+1 (CSR offsets)
pub fn voronoi_tessellation_2d(
    points_array: &ArrayViewD<'_, f64>,
    box_size_x: f64,
    box_size_y: f64,
    periodic: bool,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n_particles = points_array.shape()[0];
    let box_lengths = vec![box_size_x, box_size_y];
    let delaunay = create_delaunay(points_array, periodic, &box_lengths);

    // Step 1: All Voronoi vertices = circumcenters of inner Delaunay faces
    let mut all_vertices: Vec<[f64; 2]> = Vec::new();
    let mut face_idx_to_voro_idx: HashMap<usize, usize> = HashMap::new();

    for face in delaunay.inner_faces() {
        let cc = face.circumcenter();
        let voro_idx = all_vertices.len();
        face_idx_to_voro_idx.insert(face.fix().index(), voro_idx);
        all_vertices.push([cc.x, cc.y]);
    }

    // Step 2: All Voronoi edges (process each undirected Delaunay edge once)
    let mut all_edges: Vec<[usize; 2]> = Vec::new();
    for edge in delaunay.directed_edges() {
        let rev = edge.rev();
        if edge.fix().index() > rev.fix().index() { continue; }
        let face_left = edge.face();
        let face_right = rev.face();
        if let (Some(li), Some(ri)) = (face_left.as_inner(), face_right.as_inner()) {
            if let (Some(&vi), Some(&vj)) = (
                face_idx_to_voro_idx.get(&li.fix().index()),
                face_idx_to_voro_idx.get(&ri.fix().index()),
            ) {
                all_edges.push([vi, vj]);
            }
        }
    }

    // Step 3: Per-cell vertex lists (only for the first n_particles vertices)
    let mut cell_indices_raw: Vec<usize> = Vec::new();
    let mut cell_offsets: Vec<usize> = vec![0];

    for i in 0..n_particles {
        let fixed_handle = FixedHandleImpl::from_index(i);
        let dynamic_handle = delaunay.vertex(fixed_handle);
        for edge in dynamic_handle.out_edges() {
            let face = edge.face();
            if let Some(inner_face) = face.as_inner() {
                if let Some(&voro_idx) = face_idx_to_voro_idx.get(&inner_face.fix().index()) {
                    cell_indices_raw.push(voro_idx);
                }
            }
        }
        cell_offsets.push(cell_indices_raw.len());
    }

    // Step 4: Compact — keep only vertices/edges referenced by the N cells
    let mut used: Vec<bool> = vec![false; all_vertices.len()];
    for &vi in &cell_indices_raw {
        used[vi] = true;
    }
    let mut old_to_new: Vec<usize> = vec![0; all_vertices.len()];
    let mut compact_vertices: Vec<[f64; 2]> = Vec::new();
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            old_to_new[old] = compact_vertices.len();
            compact_vertices.push(all_vertices[old]);
        }
    }
    let cell_indices: Vec<usize> = cell_indices_raw.iter().map(|&vi| old_to_new[vi]).collect();
    let compact_edges: Vec<[usize; 2]> = all_edges.iter()
        .filter(|&&[a, b]| used[a] && used[b])
        .map(|&[a, b]| [old_to_new[a], old_to_new[b]])
        .collect();

    // Flatten for return
    let vertices_flat: Vec<f64> = compact_vertices.iter().flat_map(|v| v.iter().copied()).collect();
    let edges_flat: Vec<usize> = compact_edges.iter().flat_map(|e| e.iter().copied()).collect();

    (vertices_flat, edges_flat, cell_indices, cell_offsets)
}

/// Return the Voronoi tessellation of a point pattern on the 2-sphere.
/// Returns (vertices, edges, cell_indices, cell_offsets) where:
///   vertices:     flat [x0,y0,z0, x1,y1,z1, ...] — Voronoi vertices on the unit sphere
///   edges:        flat [i0,j0, i1,j1, ...] — pairs of vertex indices forming edges
///   cell_indices: flat vertex indices for all cells (CSR values)
///   cell_offsets: length n_particles+1 (CSR offsets)
pub fn voronoi_tessellation_2sphere(
    points_array: &ArrayViewD<'_, f64>,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n_particles = points_array.shape()[0];
    let cartesian = euclidean_from_spherical(points_array);
    let (delaunay, pole_idx) = create_delaunay_sphere(&cartesian);
    let pole_cart = [cartesian[[pole_idx, 0]], cartesian[[pole_idx, 1]], cartesian[[pole_idx, 2]]];

    let cart = |tag: usize| -> [f64; 3] {
        [cartesian[[tag, 0]], cartesian[[tag, 1]], cartesian[[tag, 2]]]
    };

    let hull_tags: Vec<usize> = delaunay.convex_hull()
        .map(|edge| edge.from().data().tag)
        .collect();

    // Step 1: Voronoi vertices from inner Delaunay faces
    let mut all_vertices: Vec<[f64; 3]> = Vec::new();
    let mut face_idx_to_voro_idx: HashMap<usize, usize> = HashMap::new();

    for face in delaunay.inner_faces() {
        let [v0, v1, v2] = face.vertices();
        let voro_idx = all_vertices.len();
        face_idx_to_voro_idx.insert(face.fix().index(), voro_idx);
        all_vertices.push(spherical_circumcenter(
            &cart(v0.data().tag), &cart(v1.data().tag), &cart(v2.data().tag)));
    }

    // Step 1b: Voronoi vertices from stitched pole triangles (pole, hull[k], hull[k+1])
    let mut pole_face_voro_indices: Vec<usize> = Vec::new();
    for k in 0..hull_tags.len() {
        let voro_idx = all_vertices.len();
        pole_face_voro_indices.push(voro_idx);
        all_vertices.push(spherical_circumcenter(
            &pole_cart, &cart(hull_tags[k]), &cart(hull_tags[(k + 1) % hull_tags.len()])));
    }

    // Step 2: Voronoi edges — inner-inner edges + stitched edges
    let mut edges: Vec<[usize; 2]> = Vec::new();
    // Inner edges
    for edge in delaunay.directed_edges() {
        let rev = edge.rev();
        if edge.fix().index() > rev.fix().index() { continue; }
        let face_left = edge.face();
        let face_right = rev.face();
        if let (Some(li), Some(ri)) = (face_left.as_inner(), face_right.as_inner()) {
            if let (Some(&vi), Some(&vj)) = (
                face_idx_to_voro_idx.get(&li.fix().index()),
                face_idx_to_voro_idx.get(&ri.fix().index()),
            ) {
                edges.push([vi, vj]);
            }
        }
    }
    // Stitched edges: between consecutive pole-face circumcenters
    for k in 0..pole_face_voro_indices.len() {
        edges.push([
            pole_face_voro_indices[k],
            pole_face_voro_indices[(k + 1) % pole_face_voro_indices.len()],
        ]);
    }
    // Stitched edges: between pole-face circumcenters and adjacent inner-face circumcenters
    // Each hull edge (hull[k], hull[k+1]) has an inner face on one side; the pole face is on the other.
    // The Voronoi edge connects their circumcenters.
    for edge in delaunay.directed_edges() {
        if edge.face().as_inner().is_some() && edge.rev().face().as_inner().is_none() {
            // This directed edge has inner face on left, outer face on right (reversed)
            // The edge goes from vertex A to vertex B; the hull edge is (A, B)
            let a_tag = edge.from().data().tag;
            let b_tag = edge.to().data().tag;
            // Find which pole face this corresponds to
            for k in 0..hull_tags.len() {
                let next = (k + 1) % hull_tags.len();
                if (hull_tags[k] == a_tag && hull_tags[next] == b_tag) ||
                   (hull_tags[k] == b_tag && hull_tags[next] == a_tag) {
                    let inner_vi = face_idx_to_voro_idx[&edge.face().as_inner().unwrap().fix().index()];
                    edges.push([inner_vi, pole_face_voro_indices[k]]);
                    break;
                }
            }
        }
    }

    // Step 3: Per-cell vertex lists
    let mut cell_indices: Vec<usize> = Vec::new();
    let mut cell_offsets: Vec<usize> = vec![0];

    for i in 0..n_particles {
        if i == pole_idx {
            // Pole cell: circumcenters of stitched triangles
            for &vi in &pole_face_voro_indices {
                cell_indices.push(vi);
            }
        } else {
            let di = if i < pole_idx { i } else { i - 1 };
            let fixed_handle = FixedHandleImpl::from_index(di);
            let dynamic_handle = delaunay.vertex(fixed_handle);

            let edges_vec: Vec<_> = dynamic_handle.out_edges().collect();
            let n_edges = edges_vec.len();

            // Find this vertex's two hull neighbors (for stitching)
            let hull_set: std::collections::HashSet<usize> = hull_tags.iter().copied().collect();
            let my_hull_neighbors: Vec<usize> = edges_vec.iter()
                .map(|e| e.to().data().tag)
                .filter(|t| hull_set.contains(t))
                .collect();

            for k in 0..n_edges {
                let face = edges_vec[k].face();
                if let Some(inner_face) = face.as_inner() {
                    if let Some(&voro_idx) = face_idx_to_voro_idx.get(&inner_face.fix().index()) {
                        cell_indices.push(voro_idx);
                    }
                } else {
                    // Outer face: TWO stitched circumcenters, ordered to match winding
                    let sa = spherical_circumcenter(&cart(i), &cart(my_hull_neighbors[0]), &pole_cart);
                    let sb = spherical_circumcenter(&cart(i), &cart(my_hull_neighbors[1]), &pole_cart);

                    let prev_cc = if cell_indices.len() > *cell_offsets.last().unwrap() {
                        let last_vi = cell_indices[cell_indices.len() - 1];
                        all_vertices[last_vi]
                    } else {
                        // Wrap: find the last inner face's circumcenter
                        let last_inner_k = (0..n_edges).rev()
                            .find(|&j| edges_vec[j].face().as_inner().is_some())
                            .unwrap();
                        let inner_face = edges_vec[last_inner_k].face().as_inner().unwrap();
                        let [v0, v1, v2] = inner_face.vertices();
                        spherical_circumcenter(&cart(v0.data().tag), &cart(v1.data().tag), &cart(v2.data().tag))
                    };

                    let dot_a = sa[0]*prev_cc[0] + sa[1]*prev_cc[1] + sa[2]*prev_cc[2];
                    let dot_b = sb[0]*prev_cc[0] + sb[1]*prev_cc[1] + sb[2]*prev_cc[2];
                    let (first, second) = if dot_a >= dot_b { (sa, sb) } else { (sb, sa) };

                    let vi1 = all_vertices.len();
                    all_vertices.push(first);
                    cell_indices.push(vi1);
                    let vi2 = all_vertices.len();
                    all_vertices.push(second);
                    cell_indices.push(vi2);
                }
            }
        }
        cell_offsets.push(cell_indices.len());
    }

    // Flatten
    let vertices_flat: Vec<f64> = all_vertices.iter().flat_map(|v| v.iter().copied()).collect();
    let edges_flat: Vec<usize> = edges.iter().flat_map(|e| e.iter().copied()).collect();

    (vertices_flat, edges_flat, cell_indices, cell_offsets)
}

/// 2D Bond-Orientational Order Parameters using a metric (distance) cutoff
/// for neighbor determination via R*-tree, instead of Delaunay/Voronoi.
///
/// Output shape: (N, n_orders, 2) — complex ψ_n per particle, same as
/// `compute_steinhardt_boops_2d`.
pub fn compute_metric_boops_2d(
    points_array: &ArrayViewD<'_, f64>,
    boop_order_array: &ArrayViewD<'_, isize>,
    box_size_x: f64,
    box_size_y: f64,
    cutoff: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 3]>> {
    let n_particles = points_array.shape()[0];
    let boop_orders_number = boop_order_array.shape()[0];
    let box_lengths = [box_size_x, box_size_y];

    let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f());

    let rtree = compute_periodic_rstar_tree(points_array, box_size_x, box_size_y, periodic);
    let cutoff2 = cutoff * cutoff;

    boop_vectors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut boops_i)| {
            let xi = points_array[[i, 0]];
            let yi = points_array[[i, 1]];

            let mut neighbour_count = 0;
            let mut iter = rtree.nearest_neighbor_iter_with_distance_2(&[xi, yi]).skip(1);
            loop {
                let (neighbor, dist2) = iter.next().unwrap();
                if dist2 > cutoff2 { break; }
                neighbour_count += 1;

                let mut r_ij = [neighbor[0] - xi, neighbor[1] - yi];
                if periodic {
                    ensure_periodicity(&mut r_ij, &box_lengths);
                }

                let theta = atan2(r_ij[1], r_ij[0]);

                for n in 0..boop_orders_number {
                    let order = boop_order_array[[n]] as f64;
                    let angle = order * theta;
                    boops_i[[n, 0]] += angle.cos();
                    boops_i[[n, 1]] += angle.sin();
                }
            }

            if neighbour_count > 0 {
                for n in 0..boop_orders_number {
                    boops_i[[n, 0]] /= neighbour_count as f64;
                    boops_i[[n, 1]] /= neighbour_count as f64;
                }
            }
        });

    boop_vectors
}

/// 3D Steinhardt Bond-Orientational Order Parameters using a metric (distance)
/// cutoff for neighbor determination via R*-tree, instead of Delaunay.
///
/// Computes q_l(i) = sqrt(4π/(2l+1) Σ_{m=-l}^{l} |q_lm(i)|²)
/// where q_lm(i) = (1/N_nb) Σ_{j∈neighbors} Y_lm(θ_ij, φ_ij).
///
/// Output shape: (N, n_orders) — scalar q_l per particle.
pub fn compute_metric_boops_3d(
    points_array: &ArrayViewD<'_, f64>,
    boop_order_array: &ArrayViewD<'_, isize>,
    box_size_x: f64,
    box_size_y: f64,
    box_size_z: f64,
    cutoff: f64,
    periodic: bool,
) -> Array<f64, Dim<[usize; 2]>> {
    let n_particles = points_array.shape()[0];
    let n_orders = boop_order_array.shape()[0];
    let box_lengths = [box_size_x, box_size_y, box_size_z];

    let mut ql_array = Array::<f64, _>::zeros((n_particles, n_orders).f());

    let rtree = compute_periodic_rstar_tree_3d(
        points_array, box_size_x, box_size_y, box_size_z, periodic);
    let cutoff2 = cutoff * cutoff;

    ql_array
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut ql_i)| {
            let xi = points_array[[i, 0]];
            let yi = points_array[[i, 1]];
            let zi = points_array[[i, 2]];

            // Collect neighbors within cutoff
            let mut neighbors: Vec<[f64; 3]> = Vec::new();
            let mut iter = rtree.nearest_neighbor_iter_with_distance_2(
                &[xi, yi, zi]).skip(1);
            loop {
                let (neighbor, dist2) = iter.next().unwrap();
                if dist2 > cutoff2 { break; }
                let mut r_ij = [neighbor[0] - xi, neighbor[1] - yi, neighbor[2] - zi];
                if periodic {
                    ensure_periodicity(&mut r_ij, &box_lengths);
                }
                neighbors.push(r_ij);
            }

            let n_nb = neighbors.len();
            if n_nb == 0 { return; }

            for (n_ord, &order) in boop_order_array.iter().enumerate() {
                let l = order as usize;
                let mut ql_sum = 0.0;

                for m in -(l as isize)..=(l as isize) {
                    let mut re = 0.0;
                    let mut im = 0.0;

                    for r_ij in &neighbors {
                        let dist = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1]
                                  + r_ij[2]*r_ij[2]).sqrt();
                        let cos_theta = r_ij[2] / dist;
                        let phi = r_ij[1].atan2(r_ij[0]);

                        let (ylm_re, ylm_im) = spherical_harmonic(l, m, cos_theta, phi);
                        re += ylm_re;
                        im += ylm_im;
                    }

                    re /= n_nb as f64;
                    im /= n_nb as f64;
                    ql_sum += re * re + im * im;
                }

                ql_i[n_ord] = (4.0 * PI / (2 * l + 1) as f64 * ql_sum).sqrt();
            }
        });

    ql_array
}
