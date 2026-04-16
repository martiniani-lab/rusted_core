use ang::atan2;
use numpy::ndarray::ArrayViewD;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis, Dim};
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
