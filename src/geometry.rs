use ang::atan2;
use numpy::ndarray::ArrayViewD;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis, Dim};
use rand::RngExt;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter, Result};

extern crate spade;
use spade::{Point2, HasPosition};

extern crate geo;

use rstar::primitives::GeomWithData;

// Vectors of RwLocks cannot be initialized with a clone!
macro_rules! vec_no_clone {
    ( $val:expr; $n:expr ) => {{
        let result: Vec<_> = std::iter::repeat_with(|| $val).take($n).collect();
        result
    }};
}
pub(crate) use vec_no_clone;

pub fn ensure_periodicity(v: &mut Vec<f64>, box_lengths: &Vec<f64>) {

    v.iter_mut().zip(box_lengths).for_each(|(coord, box_length)| {

        if *coord > box_length * 0.5 {
            *coord -= box_length;
        } else if *coord <= -box_length * 0.5 {
            *coord += box_length;
        }

    });
}

pub fn rdf_normalisation(box_lengths: &Vec<f64>, npoints: usize, bincenter: f64, binsize: f64, periodic: bool) -> f64 {

    let ndim = box_lengths.len();
    let volume: f64 = box_lengths.iter().product();
    let rho = npoints as f64 / volume;

    assert!(ndim > 1);
    assert!(ndim < 4);

    if ndim == 2 {

        let normalisation = if periodic {
            if bincenter <= box_lengths[0] / 2.0 {
                2.0 * PI * bincenter * binsize * rho
            // (2 pi r  rho dr)
            } else {
                binsize * rho
                    * 2.0
                    * bincenter
                    * (PI - 4.0 * (0.5 * box_lengths[0] / bincenter).acos())
                // rho dr * 2 r *( pi - 4 acos(L/2r))
            }
        } else {
            2.0 * PI * bincenter * binsize * rho
            // (2 pi r  rho dr)
        };

        return normalisation;

    } else {

        let normalisation = if periodic {
            if bincenter <= box_lengths[0] / 2.0 {
                4.0 * PI * bincenter * bincenter * binsize * rho
                // (4 pi r^2  rho dr)
            } else if bincenter <= box_lengths[0] / (2.0_f64.sqrt()) {
                binsize * rho
                    * 2.0
                    * bincenter
                    * PI
                    * (3.0 * box_lengths[0] - 4.0 * bincenter)
                // rho dr * 2 r * pi * ( 3 L - 4 r)
            } else {
                binsize * rho
                    * 2.0
                    * bincenter
                    * (3.0 * PI * box_lengths[0] - 4.0 * PI * bincenter
                        + 12.0
                            * bincenter
                            * (1.0
                                / (4.0 * bincenter * bincenter
                                    / (box_lengths[0] * box_lengths[0])
                                    - 1.0))
                                .acos()
                        - 12.0
                            * box_lengths[0]
                            * (1.0
                                / (4.0 * bincenter * bincenter
                                    / (box_lengths[0] * box_lengths[0])
                                    - 1.0)
                                    .sqrt())
                            .acos())
                // rho dr * 2 r * ( 3 pi l - 4 pi r + 12 r acos(1/(1-4 r^2 / L^2)) - 12 L acos(1/sqrt(4r^2/L^2 -1)) )
            }
        } else {
            4.0 * PI * bincenter * bincenter * binsize * rho
            // (4 pi r^2  rho dr)
        };

        return normalisation;

    }

}

pub fn relative_distance_vec_spherical(thetaref: f64, phiref: f64, theta: f64, phi: f64) -> (f64, f64) {

    // I think it's necessary to go through Cartesian for this unfortunately
    let r = vec![theta.sin() * phi.cos(),theta.sin() * phi.sin(),theta.cos()];

    // Rotate by -phiref around z first
    let r_rotated_once = vec![r[0]*phiref.cos() + r[1] * phiref.sin(), - r[0] * phiref.sin() + r[1] * phiref.cos(), r[2] ];

    // Rotate by +thetaref around y then
    let r_rotated_twice = vec![r_rotated_once[0]*thetaref.cos() - r_rotated_once[2] * thetaref.sin(),r_rotated_once[1],r_rotated_once[0]*thetaref.sin() + r_rotated_once[2] * thetaref.cos()];

    // Measure the spherical angles of the newly obtained vector
    let theta_relative = r_rotated_twice[2].acos();
    let phi_relative = atan2(r_rotated_twice[1], r_rotated_twice[0]).in_radians() + PI;

    (theta_relative, phi_relative)

}

pub fn great_circle_distance(point1: &[f64;2], point2: &[f64;2]) -> f64 {

    (point1[0].cos() * point2[0].cos() + point1[0].sin() * point2[0].sin() * (point1[1] - point2[1]).cos()).acos()

}

pub fn euclidean_from_spherical(points: &ArrayViewD<'_, f64>) -> Array<f64, Dim<[usize; 2]>> {

    points.axis_iter(Axis(0)).into_par_iter().map(|point| {
        euclidean_from_spherical_single_point(point[[0]], point[[1]])
    }).collect::<Vec<[f64;3]>>().into()

}

pub fn euclidean_from_spherical_single_point(theta: f64, phi: f64) -> [f64; 3] {
    let sin_theta = theta.sin();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos() ]
}

// Stereographic projection utilities

/// Construct an orthonormal frame for the tangent plane at a point on the unit sphere.
/// Given a unit normal vector n, returns (e1, e2) such that (e1, e2, n) form a right-handed frame.
pub fn tangent_frame(n: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick a vector not parallel to n
    let v = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // e1 = normalize(v - (v·n)n)  (Gram-Schmidt)
    let dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
    let e1_raw = [v[0] - dot * n[0], v[1] - dot * n[1], v[2] - dot * n[2]];
    let e1_norm = (e1_raw[0]*e1_raw[0] + e1_raw[1]*e1_raw[1] + e1_raw[2]*e1_raw[2]).sqrt();
    let e1 = [e1_raw[0]/e1_norm, e1_raw[1]/e1_norm, e1_raw[2]/e1_norm];

    // e2 = n × e1
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];

    (e1, e2)
}

/// Find the best stereographic projection pole for points on the unit sphere.
/// Returns the pole (unit vector) that is farthest from all data points.
/// Strategy: use the antipode of the centroid when it is well-defined, otherwise
/// draw random candidate poles and keep the one whose closest data point is farthest.
pub fn find_stereographic_pole(cartesian_points: &Array<f64, Dim<[usize; 2]>>) -> [f64; 3] {
    let n = cartesian_points.shape()[0];

    // Compute centroid of Cartesian positions
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for i in 0..n {
        cx += cartesian_points[[i, 0]];
        cy += cartesian_points[[i, 1]];
        cz += cartesian_points[[i, 2]];
    }
    cx /= n as f64;
    cy /= n as f64;
    cz /= n as f64;

    let norm = (cx*cx + cy*cy + cz*cz).sqrt();

    if norm > 0.01 {
        // Centroid is well-defined — use its antipode
        let pole = [-cx/norm, -cy/norm, -cz/norm];
        // Check that no data point is too close (dot > 0.9999 means < 0.8°)
        let mut max_dot = -1.0_f64;
        for i in 0..n {
            let dot = cartesian_points[[i, 0]] * pole[0]
                    + cartesian_points[[i, 1]] * pole[1]
                    + cartesian_points[[i, 2]] * pole[2];
            if dot > max_dot { max_dot = dot; }
        }
        if max_dot < 0.9999 {
            return pole;
        }
    }

    // Centroid is near origin or its antipode is too close to a data point.
    // Draw random candidate poles and keep the best one.
    let mut rng = rand::rng();
    let n_candidates = 100;
    let mut best_pole = [0.0, 0.0, 1.0];
    let mut best_max_dot = 1.0_f64;

    for _ in 0..n_candidates {
        // Uniform random point on the sphere (Marsaglia's method)
        let mut u: f64;
        let mut v: f64;
        let mut s: f64;
        loop {
            u = 2.0 * rng.random::<f64>() - 1.0;
            v = 2.0 * rng.random::<f64>() - 1.0;
            s = u*u + v*v;
            if s < 1.0 { break; }
        }
        let factor = 2.0 * (1.0 - s).sqrt();
        let candidate = [u * factor, v * factor, 1.0 - 2.0 * s];

        let mut max_dot = -1.0_f64;
        for i in 0..n {
            let dot = cartesian_points[[i, 0]] * candidate[0]
                    + cartesian_points[[i, 1]] * candidate[1]
                    + cartesian_points[[i, 2]] * candidate[2];
            if dot > max_dot { max_dot = dot; }
        }
        if max_dot < best_max_dot {
            best_max_dot = max_dot;
            best_pole = candidate;
        }
    }

    best_pole
}

/// Stereographic projection from a given pole onto the plane through the origin
/// perpendicular to that pole. Maps 3D Cartesian points on the unit sphere to 2D.
pub fn stereographic_project(
    cartesian_points: &Array<f64, Dim<[usize; 2]>>,
    pole: &[f64; 3]
) -> Vec<[f64; 2]> {
    let n = cartesian_points.shape()[0];

    // Orthonormal basis for the projection plane
    let (e1, e2) = tangent_frame(pole);

    let mut projected = Vec::with_capacity(n);

    for i in 0..n {
        let x = cartesian_points[[i, 0]];
        let y = cartesian_points[[i, 1]];
        let z = cartesian_points[[i, 2]];

        let dot = x * pole[0] + y * pole[1] + z * pole[2]; // x · p
        let denom = 1.0 - dot; // 1 - x·p

        // Projected 3D vector in the plane: x' = (x - (x·p)p) / (1 - x·p)
        let xp = (x - dot * pole[0]) / denom;
        let yp = (y - dot * pole[1]) / denom;
        let zp = (z - dot * pole[2]) / denom;

        // Decompose in the plane basis
        let u = xp * e1[0] + yp * e1[1] + zp * e1[2];
        let v = xp * e2[0] + yp * e2[1] + zp * e2[2];

        projected.push([u, v]);
    }

    projected
}

/// Area of a spherical triangle with vertices a, b, c on the unit sphere.
/// Uses the Van Oosterom–Strackee formula for the solid angle.
/// See https://ieeexplore.ieee.org/document/4121581
pub fn spherical_triangle_area(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    // numerator = a · (b × c)
    let cross = [
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0],
    ];
    let numerator = a[0]*cross[0] + a[1]*cross[1] + a[2]*cross[2];
    let ab = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    let bc = b[0]*c[0] + b[1]*c[1] + b[2]*c[2];
    let ca = c[0]*a[0] + c[1]*a[1] + c[2]*a[2];
    let denominator = 1.0 + ab + bc + ca;

    2.0 * numerator.atan2(denominator).abs()
}

// Types

pub struct PointWithTag<T> {
    pub position: Point2<f64>,
    pub tag: T,
}

impl<T> Copy for PointWithTag<T> where T: Copy {}
impl<T> Clone for PointWithTag<T> where T: Copy {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Display> HasPosition for PointWithTag<T> {
    type Scalar = f64;

    fn position(&self) -> Point2<f64> {
        self.position
    }
}

// Needed for rtree tags for cluster
#[derive(Clone)]
#[derive(Copy)]
#[derive(Debug)]
pub struct ParticleClusterTags {
    pub id: usize,
    pub cluster_id: usize
}
impl Display for ParticleClusterTags {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "id: {:?}, cluster_id: {:?}\n", self.id, self.cluster_id)?;
        Ok(())
    }
}

// https://docs.rs/rstar/latest/rstar/primitives/struct.GeomWithData.html
pub type ListPointWithTag<T, const N:usize> = GeomWithData<[f64; N], T>;
