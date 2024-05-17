use ndarray::Dim;
use numpy::{PyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // This is a pure function (no mutations of incoming data).
    // You can see this as the python array in the function arguments is readonly.
    // The object we return will need ot have the same lifetime as the Python.
    // Python will handle the objects deallocation.
    // We are having the Python as input with a lifetime parameter.
    // Basically, none of the data that comes from Python can survive
    // longer than Python itself. Therefore, if Python is dropped, so must our Rust Python-dependent variables.

    #[pyfn(m)]
    fn compute_vector_rdf2d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> &'py PyArray2<f64> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let vector_rdf = rust_fn::compute_vector_rdf2d(&array, box_size, binsize, periodic);
        let array_rdf = PyArray2::from_vec2(py, &vector_rdf).unwrap();

        return array_rdf;
    }

    #[pyfn(m)]
    fn compute_vector_rdf3d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> &'py PyArray3<f64> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let vector_rdf = rust_fn::compute_vector_rdf3d(&array, box_size, binsize, periodic);
        let array_rdf = PyArray3::from_vec3(py, &vector_rdf).unwrap();

        return array_rdf;
    }

    #[pyfn(m)]
    fn compute_vector_orientation_corr_2d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (&'py PyArray2<f64>, &'py PyArray<f64, Dim<[usize; 3]>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (vector_rdf, vector_corr) =
            rust_fn::compute_vector_orientation_corr_2d(&array, box_size, binsize, periodic, order);
        let array_rdf = PyArray2::from_vec2(py, &vector_rdf).unwrap();
        let array_corr = PyArray::from_owned_array(py, vector_corr);

        return (array_rdf, array_corr);
    }

    #[pyfn(m)]
    fn compute_radial_correlations_2d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        field: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        connected: bool,
    ) -> (&'py PyArray1<f64>, &'py PyArray<f64, Dim<[usize; 2]>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let field_array = unsafe { field.as_array() }; // Same for field
        let field_shape = field_array.shape(); // Need to feed npoints x fielddim values

        assert!(
            field_shape[0] == array.shape()[0],
            "You must provide as many field lines as particles!"
        );

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (rdf, field_corrs) = rust_fn::compute_radial_correlations_2d(
            &array,
            &field_array,
            box_size,
            box_size,
            binsize,
            periodic,
            connected,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_owned_array(py, field_corrs);

        return (array_rdf, pyarray_field_corrs);
    }

    #[pyfn(m)]
    fn compute_radial_orientation_corr_2d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (&'py PyArray1<f64>, &'py PyArray<f64, Dim<[usize; 2]>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (rdf, radial_corr) = rust_fn::compute_radial_orientation_corr_2d(
            &array, box_size, box_size, binsize, periodic, order,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let array_corr = PyArray::from_owned_array(py, radial_corr);

        return (array_rdf, array_corr);
    }

    #[pyfn(m)]
    fn compute_radial_correlations_3d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        field: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        connected: bool,
    ) -> (&'py PyArray1<f64>, &'py PyArray<f64, Dim<[usize; 2]>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let field_array = unsafe { field.as_array() }; // Same for field
        let field_shape = field_array.shape(); // Need to feed npoints x fielddim values

        assert!(
            field_shape[0] == array.shape()[0],
            "You must provide as many field lines as particles!"
        );

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (rdf, field_corrs) = rust_fn::compute_radial_correlations_3d(
            &array,
            &field_array,
            box_size,
            box_size,
            box_size,
            binsize,
            periodic,
            connected,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_owned_array(py, field_corrs);

        return (array_rdf, pyarray_field_corrs);
    }

    #[pyfn(m)]
    fn compute_2d_boops<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        orders: &PyArrayDyn<isize>,
        box_size: f64,
        periodic: bool,
    ) -> &'py PyArray<f64, Dim<[usize; 3]>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let boop_order_array = unsafe { orders.as_array() }; // Same for field

        let boops = rust_fn::compute_steinhardt_boops_2d(
            &array,
            &boop_order_array,
            box_size,
            box_size,
            periodic,
        );
        let array_boops = PyArray::from_owned_array(py, boops);

        return array_boops;
    }

    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
mod rust_fn {
    use ang::atan2;
    use libm::hypot;
    use ndarray::parallel::prelude::*;
    use ndarray::{s, Array, Axis, Dim, ShapeBuilder, Zip};
    use numpy::ndarray::ArrayViewD;
    use std::f64::consts::PI;
    use std::sync::{Arc, RwLock};
    extern crate spade;
    use spade::internals::FixedHandleImpl;
    use spade::{DelaunayTriangulation, Point2, Triangulation};

    extern crate geo;
    use geo::algorithm::area::Area;
    use geo::{LineString, Polygon};

    // Vectors of RwLocks cannot be initialized with a clone!
    macro_rules! vec_no_clone {
        ( $val:expr; $n:expr ) => {{
            let result: Vec<_> = std::iter::repeat_with(|| $val).take($n).collect();
            result
        }};
    }

    pub fn compute_vector_rdf2d(
        points: &ArrayViewD<'_, f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> Vec<Vec<f64>> {
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        println!("Found {:?} points in d = {:?}", npoints, ndim);
        assert!(ndim == 2);

        // Compute the correlations
        let rdf = compute_particle_correlations(&points, box_size, box_size, binsize, periodic);

        return rdf;
    }

    pub fn compute_particle_correlations(
        points: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        binsize: f64,
        periodic: bool,
    ) -> Vec<Vec<f64>> {
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

        let n_particles = points.shape()[0];
        let rdf: Vec<Vec<Arc<RwLock<f64>>>> =
            vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); nbins]; nbins];

        (0..n_particles).into_par_iter().for_each(|i| {
            for j in i + 1..n_particles {
                let xi = points[[i, 0]];
                let xj = points[[j, 0]];
                let yi = points[[i, 1]];
                let yj = points[[j, 1]];

                let mut r_ij = vec![xj - xi, yj - yi];
                if periodic {
                    ensure_periodicity(&mut r_ij, box_size_x, box_size_y);
                }

                // determine the relevant bin, and update the count at that bin
                let index_x = ((r_ij[0] + 0.5 * box_size_x) / binsize).floor() as usize;
                let index_y = ((r_ij[1] + 0.5 * box_size_y) / binsize).floor() as usize;
                *(rdf[index_x][index_y].write().unwrap()) += 1.0;

                // Use symmetry
                let index_x_symm = ((0.5 * box_size_x - r_ij[0]) / binsize).floor() as usize;
                let index_y_symm = ((0.5 * box_size_y - r_ij[1]) / binsize).floor() as usize;
                *(rdf[index_x_symm][index_y_symm].write().unwrap()) += 1.0;
            }
        });

        let mut rdf_vector = vec![vec![0.0; nbins]; nbins];
        rdf_vector.par_iter_mut().enumerate().for_each(| (i, rdf_vector_i)| {
            for j in 0..nbins {
                rdf_vector_i[j] = *rdf.get(i).unwrap().get(j).unwrap().read().unwrap();
            }
        });

        return rdf_vector;
    }

    pub fn compute_vector_orientation_corr_2d(
        points: &ArrayViewD<'_, f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Vec<Vec<f64>>, Array<f64, Dim<[usize; 3]>>) {
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        println!("Found {:?} points in d = {:?}", npoints, ndim);
        assert!(ndim == 2);

        // Compute the correlations
        let (rdf, corr) = compute_particle_orientation_correlations(
            &points, box_size, box_size, binsize, periodic, order,
        );

        return (rdf, corr);
    }

    pub fn compute_particle_orientation_correlations(
        points: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Vec<Vec<f64>>, Array<f64, Dim<[usize; 3]>>) {
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

        let n_particles = points.shape()[0];
        let rdf: Vec<Vec<Arc<RwLock<f64>>>> =
            vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); nbins]; nbins];
        let corr: Vec<Vec<Vec<Arc<RwLock<f64>>>>> = vec_no_clone![vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); 2]; nbins]; nbins];

        (0..n_particles).into_par_iter().for_each(|i| {
            for j in i + 1..n_particles {
                let xi = points[[i, 0]];
                let xj = points[[j, 0]];
                let yi = points[[i, 1]];
                let yj = points[[j, 1]];

                let mut r_ij = vec![xj - xi, yj - yi];
                if periodic {
                    ensure_periodicity(&mut r_ij, box_size_x, box_size_y);
                }

                // determine the relevant bin, and update the count at that bin
                let index_x = ((r_ij[0] + 0.5 * box_size_x) / binsize).floor() as usize;
                let index_y = ((r_ij[1] + 0.5 * box_size_y) / binsize).floor() as usize;
                *(rdf[index_x][index_y].write().unwrap()) += 1.0;

                // Compute the orientational part
                let theta_ij = atan2(r_ij[1], r_ij[0]);
                let angle = order as f64 * theta_ij;
                let dpsinx = angle.cos();
                let dpsiny = angle.sin();

                *(corr[index_x][index_y][0].write().unwrap()) += dpsinx;
                *(corr[index_x][index_y][1].write().unwrap()) += dpsiny;

                // Use symmetry
                let index_x_symm = ((0.5 * box_size_x - r_ij[0]) / binsize).floor() as usize;
                let index_y_symm = ((0.5 * box_size_y - r_ij[1]) / binsize).floor() as usize;
                *(rdf[index_x_symm][index_y_symm].write().unwrap()) += 1.0;

                *(corr[index_x_symm][index_y_symm][0].write().unwrap()) += dpsinx;
                *(corr[index_x_symm][index_y_symm][1].write().unwrap()) += dpsiny;
            }
        });

        let mut rdf_vector = vec![vec![0.0; nbins]; nbins];
        let mut corr_vector = Array::<f64, _>::zeros((nbins, nbins, 2).f());
        
        rdf_vector.par_iter_mut().enumerate().for_each(|(i, rdf_i)| {
            for j in 0..nbins {
                rdf_i[j] = *rdf
                    .get(i)
                    .unwrap()
                    .get(j)
                    .unwrap()
                    .read()
                    .unwrap();
            }
        });
        
        // https://docs.rs/ndarray/latest/ndarray/struct.Zip.html
        // https://stackoverflow.com/questions/75824934/parallel-computation-of-values-for-ndarray-array2f64-in-rust
        Zip::indexed(&mut corr_vector).par_for_each(|(i, j, dim), corr_val| {
            *corr_val = *corr
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .get(dim)
            .unwrap()
            .read()
            .unwrap();
        });
        return (rdf_vector, corr_vector);
    }

    pub fn compute_vector_rdf3d(
        points: &ArrayViewD<'_, f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> Vec<Vec<Vec<f64>>> {
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        println!("Found {:?} points in d = {:?}", npoints, ndim);
        assert!(ndim == 3);

        // Compute the correlations
        let rdf = compute_particle_correlations_3d(
            &points, box_size, box_size, box_size, binsize, periodic,
        );

        return rdf;
    }

    pub fn compute_particle_correlations_3d(
        points: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        box_size_z: f64,
        binsize: f64,
        periodic: bool,
    ) -> Vec<Vec<Vec<f64>>> {
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

        let n_particles = points.shape()[0];
        let rdf: Vec<Vec<Vec<Arc<RwLock<f64>>>>> = vec_no_clone![vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); nbins]; nbins]; nbins];

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
                    ensure_periodicity_3d(&mut r_ij, box_size_x, box_size_y, box_size_z);
                }

                // determine the relevant bin, and update the count at that bin
                let index_x = ((r_ij[0] + 0.5 * box_size_x) / binsize).floor() as usize;
                let index_y = ((r_ij[1] + 0.5 * box_size_y) / binsize).floor() as usize;
                let index_z = ((r_ij[2] + 0.5 * box_size_z) / binsize).floor() as usize;
                *(rdf[index_x][index_y][index_z].write().unwrap()) += 1.0;

                // Use symmetry
                let index_x_symm = ((0.5 * box_size_x - r_ij[0]) / binsize).floor() as usize;
                let index_y_symm = ((0.5 * box_size_y - r_ij[1]) / binsize).floor() as usize;
                let index_z_symm = ((0.5 * box_size_z - r_ij[2]) / binsize).floor() as usize;
                *(rdf[index_x_symm][index_y_symm][index_z_symm]
                    .write()
                    .unwrap()) += 1.0;
            }
        });

        // Could make this an array and actually parallelize over all dimensions
        let mut rdf_vector = vec![vec![vec![0.0; nbins]; nbins]; nbins];
        rdf_vector.par_iter_mut().enumerate().for_each(|(i, rdf_i)| {
            for j in 0..nbins {
                for k in 0..nbins {
                    rdf_i[j][k] = *rdf
                        .get(i)
                        .unwrap()
                        .get(j)
                        .unwrap()
                        .get(k)
                        .unwrap()
                        .read()
                        .unwrap();
                }
            }
        });

        return rdf_vector;
    }

    pub fn ensure_periodicity(v: &mut Vec<f64>, box_size_x: f64, box_size_y: f64) {
        if v[0] > box_size_x * 0.5 {
            v[0] -= box_size_x;
        } else if v[0] <= -box_size_x * 0.5 {
            v[0] += box_size_x;
        }

        if v[1] > box_size_y * 0.5 {
            v[1] -= box_size_y;
        } else if v[1] <= -box_size_y * 0.5 {
            v[1] += box_size_y;
        }
    }

    pub fn ensure_periodicity_3d(
        v: &mut Vec<f64>,
        box_size_x: f64,
        box_size_y: f64,
        box_size_z: f64,
    ) {
        if v[0] > box_size_x * 0.5 {
            v[0] -= box_size_x;
        } else if v[0] <= -box_size_x * 0.5 {
            v[0] += box_size_x;
        }

        if v[1] > box_size_y * 0.5 {
            v[1] -= box_size_y;
        } else if v[1] <= -box_size_y * 0.5 {
            v[1] += box_size_y;
        }

        if v[2] > box_size_z * 0.5 {
            v[2] -= box_size_z;
        } else if v[2] <= -box_size_z * 0.5 {
            v[2] += box_size_z;
        }
    }

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
        // (0..n_particles).into_par_iter().for_each(|i| {
        //     for dim in 0..field_dim {
        //         mean_field[dim] += fields[[i,dim]];
        //     }
        // });
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
                    ensure_periodicity(&mut r_ij, box_size_x, box_size_y);
                }
                let dist_ij = hypot(r_ij[0], r_ij[1]);
                assert!(
                    dist_ij >= 0.0 && dist_ij < max_dist,
                    "Something is wrong with the distance between particles!"
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
                let normalisation = if periodic {
                    if bincenter <= box_size_x / 2.0 {
                        2.0 * PI * bincenter * binsize * n_particles as f64
                            / (box_size_x * box_size_y)
                    // (2 pi r  rho dr)
                    } else {
                        binsize * n_particles as f64 / (box_size_x * box_size_y)
                            * 2.0
                            * bincenter
                            * (PI - 4.0 * (0.5 * box_size_x / bincenter).acos())
                        // rho dr * 2 r *( pi - 4 acos(L/2r))
                    }
                } else {
                    2.0 * PI * bincenter * binsize * n_particles as f64 / (box_size_x * box_size_y)
                    // (2 pi r  rho dr)
                };
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

    pub fn compute_radial_orientation_corr_2d(
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
                    ensure_periodicity(&mut r_ij, box_size_x, box_size_y);
                }
                let dist_ij = hypot(r_ij[0], r_ij[1]);
                assert!(
                    dist_ij >= 0.0 && dist_ij < max_dist,
                    "Something is wrong with the distance between particles!"
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
                let normalisation = if periodic {
                    if bincenter <= box_size_x / 2.0 {
                        2.0 * PI * bincenter * binsize * n_particles as f64
                            / (box_size_x * box_size_y)
                    // (2 pi r  rho dr)
                    } else {
                        binsize * n_particles as f64 / (box_size_x * box_size_y)
                            * 2.0
                            * bincenter
                            * (PI - 4.0 * (0.5 * box_size_x / bincenter).acos())
                        // rho dr * 2 r *( pi - 4 acos(L/2r))
                    }
                } else {
                    2.0 * PI * bincenter * binsize * n_particles as f64 / (box_size_x * box_size_y)
                    // (2 pi r  rho dr)
                };
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
                    ensure_periodicity_3d(&mut r_ij, box_size_x, box_size_y, box_size_z);
                }
                let dist_ij = hypot(hypot(r_ij[0], r_ij[1]), r_ij[2]);
                assert!(
                    dist_ij >= 0.0 && dist_ij < max_dist,
                    "Something is wrong with the distance between particles!"
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
                let normalisation = if periodic {
                    if bincenter <= box_size_x / 2.0 {
                        4.0 * PI * bincenter * bincenter * binsize * n_particles as f64
                            / (box_size_x * box_size_y * box_size_z)
                        // (4 pi r^2  rho dr)
                    } else if bincenter <= box_size_x / (2.0_f64.sqrt()) {
                        binsize * n_particles as f64 / (box_size_x * box_size_y * box_size_z)
                            * 2.0
                            * bincenter
                            * PI
                            * (3.0 * box_size_x - 4.0 * bincenter)
                        // rho dr * 2 r * pi * ( 3 L - 4 r)
                    } else {
                        binsize * n_particles as f64 / (box_size_x * box_size_y * box_size_z)
                            * 2.0
                            * bincenter
                            * (3.0 * PI * box_size_x - 4.0 * PI * bincenter
                                + 12.0
                                    * bincenter
                                    * (1.0
                                        / (4.0 * bincenter * bincenter
                                            / (box_size_x * box_size_x)
                                            - 1.0))
                                        .acos()
                                - 12.0
                                    * box_size_x
                                    * (1.0
                                        / (4.0 * bincenter * bincenter
                                            / (box_size_x * box_size_x)
                                            - 1.0)
                                            .sqrt())
                                    .acos())
                        // rho dr * 2 r * ( 3 pi l - 4 pi r + 12 r acos(1/(1-4 r^2 / L^2)) - 12 L acos(1/sqrt(4r^2/L^2 -1)) )
                    }
                } else {
                    4.0 * PI * bincenter * bincenter * binsize * n_particles as f64
                        / (box_size_x * box_size_y * box_size_z)
                    // (4 pi r^2  rho dr)
                };
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

        let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f());

        // Enforce periodicity by adding copies.
        let mut points: Vec<Point2<f64>> = vec![Point2::new(0.0, 0.0); 9 * n_particles];

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

            let shifted_x: f64 = xi + nx * box_size_x;
            let shifted_y: f64 = yi + ny * box_size_y;

            let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);

            *pointi = newpoint;
        });

        // Bulk load the points
        // bulk_load_stable ensures that the ordering of points is conserved, assuming no precise overlap
        let delaunay = DelaunayTriangulation::<Point2<f64>>::bulk_load_stable(points).unwrap();

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
                        ensure_periodicity(&mut vector, box_size_x, box_size_y);
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
}
