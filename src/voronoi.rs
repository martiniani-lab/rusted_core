use ang::atan2;
use libm::hypot;
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

use crate::geometry::{
    ensure_periodicity,
    euclidean_from_spherical,
    tangent_frame,
    spherical_triangle_area,
    spherical_circumcenter,
    great_circle_distance,
};
use crate::spatial::{create_delaunay, create_delaunay_sphere};

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
                let mut vector = vec![dx, dy];
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

    // Convert to Cartesian (on the unit sphere) for angle computation
    let cartesian = euclidean_from_spherical(points_array);

    // Build the spherical Delaunay via stereographic projection
    let (delaunay, _) = create_delaunay_sphere(&cartesian);

    let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f());

    boop_vectors
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut boops_i)| {
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);
            let outgoing_edges = dynamic_handle.out_edges();

            let mut neighbour_count = 0;

            // Current point in Cartesian (= outward unit normal)
            let ni = [cartesian[[i, 0]], cartesian[[i, 1]], cartesian[[i, 2]]];

            // Local tangent frame at point i
            let (e1, e2) = tangent_frame(&ni);

            for e in outgoing_edges {
                // Neighbor's particle index (stored as tag in the Delaunay vertex)
                let j = e.to().data().tag;

                // Skip the pole vertex (tag == n_particles)
                if j == n_particles { continue; }

                neighbour_count += 1;

                // Neighbor's Cartesian coordinates
                let xj = [cartesian[[j, 0]], cartesian[[j, 1]], cartesian[[j, 2]]];

                // Geodesic direction from i to j: project xj onto the tangent plane at i
                let dot = xj[0]*ni[0] + xj[1]*ni[1] + xj[2]*ni[2];
                let dx = xj[0] - dot * ni[0];
                let dy = xj[1] - dot * ni[1];
                let dz = xj[2] - dot * ni[2];

                // Bond angle in local frame
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

    // Convert to Cartesian and build Delaunay (pole added as vertex N)
    let cartesian = euclidean_from_spherical(points_array);
    let (delaunay, pole) = create_delaunay_sphere(&cartesian);

    // Extended Cartesian: index 0..N-1 are particles, index N is the pole
    let mut ce = Vec::with_capacity((n_particles + 1) * 3);
    for i in 0..n_particles {
        ce.push(cartesian[[i, 0]]); ce.push(cartesian[[i, 1]]); ce.push(cartesian[[i, 2]]);
    }
    ce.push(pole[0]); ce.push(pole[1]); ce.push(pole[2]);

    let mut areas_vector = vec![0.0; n_particles];
    let mut neighbour_counts_vector: Vec<usize> = vec![0; n_particles];
    let mut nn_distances_vector = vec![0.0; n_particles];

    if voronoi_areas {
        areas_vector.par_iter_mut().enumerate().for_each(|(i, area_i)| {
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);

            let pi = [ce[i*3], ce[i*3+1], ce[i*3+2]];
            let mut circumcenters: Vec<[f64; 3]> = Vec::new();

            for edge in dynamic_handle.out_edges() {
                let face = edge.face();
                if let Some(inner_face) = face.as_inner() {
                    let [v0, v1, v2] = inner_face.vertices();
                    let t0 = v0.data().tag; let t1 = v1.data().tag; let t2 = v2.data().tag;
                    let a = [ce[t0*3], ce[t0*3+1], ce[t0*3+2]];
                    let b = [ce[t1*3], ce[t1*3+1], ce[t1*3+2]];
                    let c = [ce[t2*3], ce[t2*3+1], ce[t2*3+2]];
                    circumcenters.push(spherical_circumcenter(&a, &b, &c));
                }
            }

            let n_verts = circumcenters.len();
            let mut cell_area = 0.0;
            for k in 0..n_verts {
                let v1 = &circumcenters[k];
                let v2 = &circumcenters[(k + 1) % n_verts];
                cell_area += spherical_triangle_area(&pi, v1, v2);
            }
            *area_i = cell_area;
        });
    }

    if voronoi_neighbour_count {
        neighbour_counts_vector.par_iter_mut().enumerate().for_each(|(i, count_i)| {
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);
            for e in dynamic_handle.out_edges() {
                if e.to().data().tag != n_particles { *count_i += 1; }
            }
        });
    }

    if voronoi_nn_distance {
        nn_distances_vector.par_iter_mut().enumerate().for_each(|(i, distance_i)| {
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);

            let theta_i = points_array[[i, 0]];
            let phi_i = points_array[[i, 1]];
            let pi_sph = [theta_i, phi_i];

            let mut nn_distance = INFINITY;

            for e in dynamic_handle.out_edges() {
                let j = e.to().data().tag;
                if j == n_particles { continue; }
                let pj_sph = [points_array[[j, 0]], points_array[[j, 1]]];
                let dist = great_circle_distance(&pi_sph, &pj_sph);
                if dist < nn_distance {
                    nn_distance = dist;
                }
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
                let mut vector = vec![dx, dy];
                if periodic {
                    ensure_periodicity(&mut vector, &box_lengths);
                }
                let edge_length = hypot(vector[0], vector[1]);
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
            let circumradius = hypot(dx, dy);

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
    let (delaunay, pole) = create_delaunay_sphere(&cartesian);

    // Extended Cartesian: index N is the pole
    let mut ce = Vec::with_capacity((n_particles + 1) * 3);
    for i in 0..n_particles {
        ce.push(cartesian[[i, 0]]); ce.push(cartesian[[i, 1]]); ce.push(cartesian[[i, 2]]);
    }
    ce.push(pole[0]); ce.push(pole[1]); ce.push(pole[2]);

    // Step 1: Voronoi vertices = spherical circumcenters of all inner Delaunay faces
    let mut all_vertices: Vec<[f64; 3]> = Vec::new();
    let mut face_idx_to_voro_idx: HashMap<usize, usize> = HashMap::new();

    for face in delaunay.inner_faces() {
        let [v0, v1, v2] = face.vertices();
        let t0 = v0.data().tag; let t1 = v1.data().tag; let t2 = v2.data().tag;
        let a = [ce[t0*3], ce[t0*3+1], ce[t0*3+2]];
        let b = [ce[t1*3], ce[t1*3+1], ce[t1*3+2]];
        let c = [ce[t2*3], ce[t2*3+1], ce[t2*3+2]];

        let voro_idx = all_vertices.len();
        face_idx_to_voro_idx.insert(face.fix().index(), voro_idx);
        all_vertices.push(spherical_circumcenter(&a, &b, &c));
    }

    // Step 2: Voronoi edges (skip edges involving the pole)
    let mut edges: Vec<[usize; 2]> = Vec::new();
    for edge in delaunay.directed_edges() {
        let rev = edge.rev();
        if edge.fix().index() > rev.fix().index() { continue; }
        if edge.from().data().tag == n_particles || edge.to().data().tag == n_particles { continue; }
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

    // Step 3: Per-cell vertex lists for the N real particles
    let mut cell_indices: Vec<usize> = Vec::new();
    let mut cell_offsets: Vec<usize> = vec![0];

    for i in 0..n_particles {
        let fixed_handle = FixedHandleImpl::from_index(i);
        let dynamic_handle = delaunay.vertex(fixed_handle);
        for edge in dynamic_handle.out_edges() {
            let face = edge.face();
            if let Some(inner_face) = face.as_inner() {
                if let Some(&voro_idx) = face_idx_to_voro_idx.get(&inner_face.fix().index()) {
                    cell_indices.push(voro_idx);
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
