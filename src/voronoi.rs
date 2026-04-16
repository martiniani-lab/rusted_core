use ang::atan2;
use libm::hypot;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis, Dim, ShapeBuilder};
use numpy::ndarray::ArrayViewD;
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
    find_stereographic_pole,
    stereographic_project,
    spherical_triangle_area,
    great_circle_distance,
    PointWithTag,
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
    let delaunay = create_delaunay_sphere(&cartesian);

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
                neighbour_count += 1;

                // Neighbor's particle index (stored as tag in the Delaunay vertex)
                let j = e.to().data().tag;

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

    // Convert to Cartesian
    let cartesian = euclidean_from_spherical(points_array);

    // Set up the stereographic projection and build Delaunay
    let pole = find_stereographic_pole(&cartesian);
    let projected = stereographic_project(&cartesian, &pole);

    let points: Vec<PointWithTag<usize>> = (0..n_particles)
        .map(|i| PointWithTag {
            position: Point2::new(projected[i][0], projected[i][1]),
            tag: i,
        })
        .collect();
    let delaunay = DelaunayTriangulation::<PointWithTag<usize>>::bulk_load_stable(points).unwrap();

    let mut areas_vector = vec![0.0; n_particles];
    let mut neighbour_counts_vector: Vec<usize> = vec![0; n_particles];
    let mut nn_distances_vector = vec![0.0; n_particles];

    if voronoi_areas {
        areas_vector.par_iter_mut().enumerate().for_each(|(i, area_i)| {
            let fixed_handle = FixedHandleImpl::from_index(i);
            let dynamic_handle = delaunay.vertex(fixed_handle);

            // Collect spherical circumcenters of adjacent Delaunay faces.
            // out_edges() returns edges in cyclic order, so the circumcenters
            // are already in the correct winding order around the Voronoi cell.
            let pi = [cartesian[[i, 0]], cartesian[[i, 1]], cartesian[[i, 2]]];
            let mut circumcenters: Vec<[f64; 3]> = Vec::new();

            for edge in dynamic_handle.out_edges() {
                let face = edge.face();
                if let Some(inner_face) = face.as_inner() {
                    let [v0, v1, v2] = inner_face.vertices();
                    let i0 = v0.data().tag;
                    let i1 = v1.data().tag;
                    let i2 = v2.data().tag;
                    let a = [cartesian[[i0, 0]], cartesian[[i0, 1]], cartesian[[i0, 2]]];
                    let b = [cartesian[[i1, 0]], cartesian[[i1, 1]], cartesian[[i1, 2]]];
                    let c = [cartesian[[i2, 0]], cartesian[[i2, 1]], cartesian[[i2, 2]]];

                    // Spherical circumcenter = normalize(A×B + B×C + C×A)
                    let ab = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
                    let bc = [b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]];
                    let ca = [c[1]*a[2]-c[2]*a[1], c[2]*a[0]-c[0]*a[2], c[0]*a[1]-c[1]*a[0]];
                    let raw = [ab[0]+bc[0]+ca[0], ab[1]+bc[1]+ca[1], ab[2]+bc[2]+ca[2]];
                    let norm = (raw[0]*raw[0] + raw[1]*raw[1] + raw[2]*raw[2]).sqrt();
                    let mut cc = [raw[0]/norm, raw[1]/norm, raw[2]/norm];

                    // Pick the circumcenter on the same side as the triangle (not the antipodal one)
                    let centroid = [a[0]+b[0]+c[0], a[1]+b[1]+c[1], a[2]+b[2]+c[2]];
                    if cc[0]*centroid[0] + cc[1]*centroid[1] + cc[2]*centroid[2] < 0.0 {
                        cc = [-cc[0], -cc[1], -cc[2]];
                    }

                    circumcenters.push(cc);
                } else {
                    // Outer face: the Voronoi vertex is at infinity in the plane,
                    // which maps to the projection pole on the sphere
                    circumcenters.push(pole);
                }
            }

            // Area = sum of spherical triangle areas in a fan from the particle
            // to consecutive circumcenters (already in cyclic order from out_edges)
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
            for _e in dynamic_handle.out_edges() {
                *count_i += 1;
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
