use ndarray::Dim;
use numpy::{PyArray, PyArrayMethods, PyArray1, PyArray2, PyArray3, PyArrayDyn, IntoPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python, Bound};

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray2<f64>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let vector_rdf = rust_fn::compute_vector_rdf2d(&array, box_size, binsize, periodic);
        let array_rdf = PyArray2::from_array_bound(py, &vector_rdf);

        return array_rdf;
    }

    #[pyfn(m)]
    fn compute_vector_rdf3d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray3<f64>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let vector_rdf = rust_fn::compute_vector_rdf3d(&array, box_size, binsize, periodic);
        let array_rdf = PyArray3::from_array_bound(py, &vector_rdf);

        return array_rdf;
    }

    #[pyfn(m)]
    fn compute_vector_orientation_corr_2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Bound<'py,PyArray2<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 3]>>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (vector_rdf, vector_corr) =
            rust_fn::compute_vector_orientation_corr_2d(&array, box_size, binsize, periodic, order);
        let array_rdf = PyArray2::from_array_bound(py, &vector_rdf);
        let array_corr = PyArray::from_array_bound(py, &vector_corr);

        return (array_rdf, array_corr);
    }

    #[pyfn(m)]
    fn compute_radial_correlations_2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        field: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        connected: bool,
    ) -> (Bound<'py,PyArray1<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 2]>>>) {
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
        let array_rdf = PyArray::from_vec_bound(py, rdf);
        let pyarray_field_corrs = PyArray::from_array_bound(py, &field_corrs);

        return (array_rdf, pyarray_field_corrs);
    }

    #[pyfn(m)]
    fn compute_radial_orientation_corr_2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Bound<'py,PyArray1<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 2]>>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (rdf, radial_corr) = rust_fn::compute_radial_orientation_corr_2d(
            &array, box_size, box_size, binsize, periodic, order,
        );
        let array_rdf = PyArray::from_vec_bound(py, rdf);
        let array_corr = PyArray::from_array_bound(py, &radial_corr);

        return (array_rdf, array_corr);
    }

    #[pyfn(m)]
    fn compute_radial_correlations_3d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        field: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        connected: bool,
    ) -> (Bound<'py,PyArray1<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 2]>>>) {
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
        let array_rdf = PyArray::from_vec_bound(py, rdf);
        let pyarray_field_corrs = PyArray::from_owned_array_bound(py, field_corrs);

        return (array_rdf, pyarray_field_corrs);
    }

    #[pyfn(m)]
    fn compute_2d_boops<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        orders: &Bound<'py, PyArrayDyn<isize>>,
        box_size: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray<f64, Dim<[usize; 3]>>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let boop_order_array = unsafe { orders.as_array() }; // Same for field

        let boops = rust_fn::compute_steinhardt_boops_2d(
            &array,
            &boop_order_array,
            box_size,
            box_size,
            periodic
        );
        let array_boops = PyArray::from_array_bound(py, &boops);

        return array_boops;
    }
    
    #[pyfn(m)]
    fn compute_2d_all_voronoi_quantities<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>> ) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let (areas, neighbour_counts, nn_distances) = rust_fn::compute_voronoi_quantities_2d(
            &array,
            box_size,
            box_size,
            periodic,
            true,
            true,
            true
        );
        let array_areas = PyArray1::from_vec_bound(py, areas.unwrap());
        let array_neighbour_counts = PyArray1::from_vec_bound(py, neighbour_counts.unwrap());
        let array_nn_distances = PyArray1::from_vec_bound(py, nn_distances.unwrap());
        
        return (array_areas, array_neighbour_counts, array_nn_distances);
    }
    
    
    #[pyfn(m)]
    fn compute_2d_voronoi_areas<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let (areas,_,_) = rust_fn::compute_voronoi_quantities_2d(
            &array,
            box_size,
            box_size,
            periodic,
            true,
            false,
            false
        );
        let array_areas = PyArray1::from_vec_bound(py, areas.unwrap());
        
        return array_areas;
    }
    
    #[pyfn(m)]
    fn compute_2d_voronoi_neighbour_numbers<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let (_,neighbour_counts,_) = rust_fn::compute_voronoi_quantities_2d(
            &array,
            box_size,
            box_size,
            periodic,
            false,
            true,
            false
        );
        let array_neighbour_counts = PyArray1::from_vec_bound(py, neighbour_counts.unwrap());
        
        return array_neighbour_counts;
    }
    
    #[pyfn(m)]
    fn compute_2d_voronoi_nn_distances<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let (_,_,nn_distances) = rust_fn::compute_voronoi_quantities_2d(
            &array,
            box_size,
            box_size,
            periodic,
            false,
            false,
            true
        );
        let array_nn_distances = PyArray1::from_vec_bound(py, nn_distances.unwrap());
        
        return array_nn_distances;
    }
    
    #[pyfn(m)]
    fn voronoi_furthest_sites<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray<f64, Dim<[usize; 2]>>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let furthest_sites = rust_fn::voronoi_furthest_site(
            &array,
            box_size,
            box_size,
            periodic
        );
        let array_furthest_sites = PyArray::from_array_bound(py, &furthest_sites);
        
        return array_furthest_sites;
    }
    
        
    #[pyfn(m)]
    fn cluster_by_distance<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        threshold: f64,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        let cluster_id = rust_fn::cluster_by_distance(
            &array,
            threshold,
            &vec![box_size; 2],
            periodic
        );
        let cluster_id = PyArray1::from_vec_bound(py, cluster_id);
        
        return cluster_id;
    }
    
    #[pyfn(m)]
    fn point_variances<'py>(
        py: Python<'py>,
        x: Bound<'py, PyArrayDyn<f64>>,
        radii: Bound<'py, PyArray1<f64>>,
        box_size: f64,
        n_samples: usize,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { x.as_array() }; // Convert to ndarray type
        let radii = unsafe { radii.as_array() }; // Convert to ndarray type

        let ndim = array.shape()[1];
        
        // Mutate the data
        // No need to return any value as the input data is mutated
        let reduced_variances = rust_fn::point_variances(&array, &radii, &vec![box_size; ndim], n_samples, periodic);
        reduced_variances.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    fn metric_neighbors<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        radii: &Bound<'py, PyArray1<f64>>,
        threshold: f64,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let radii_array = unsafe { radii.as_array() }; // Same for radii
        
        let neighbor_counts = rust_fn::count_metric_neighbors(
            &array,
            &radii_array,
            threshold,
            &vec![box_size; 2],
            periodic
        );
        let neighbor_counts = PyArray1::from_vec_bound(py, neighbor_counts);
        
        return neighbor_counts;
    }
    
    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
// XXX This part can/should be broken down into separate mods in rs files eventually
mod rust_fn {
    use ang::atan2;
    use libm::hypot;
    use ndarray::parallel::prelude::*;
    use ndarray::{s, Array, Axis, Dim, ShapeBuilder, Zip};
    use numpy::ndarray::{ArrayView1,ArrayViewD};
    use rayon::iter::ParallelBridge;
    use std::f64::consts::PI;
    use std::f64::INFINITY;
    use std::fmt::Display;
    use std::sync::{Arc, RwLock};
    extern crate spade;
    use spade::internals::FixedHandleImpl;
    use spade::{DelaunayTriangulation, Point2, Triangulation, HasPosition};

    extern crate geo;
    use geo::algorithm::area::Area;
    use geo::{LineString, Polygon};
    
    use rand::Rng;
    use rstar::{RTree, RTreeObject, AABB, Point, primitives::GeomWithData};

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
    ) -> Array<f64, Dim<[usize; 2]>> {
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        assert!(npoints > 1);
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
        
        let box_lengths = vec![box_size_x, box_size_y];

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
                    ensure_periodicity(&mut r_ij, &box_lengths);
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
        
        let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
        Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
            *rdf_val = *rdf
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .read()
            .unwrap();
        });

        return rdf_vector;
    }

    pub fn compute_vector_orientation_corr_2d(
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
        
        let box_lengths = vec![box_size_x, box_size_y];

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
                    ensure_periodicity(&mut r_ij, &box_lengths);
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

        let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins).f());
        let mut corr_vector = Array::<f64, _>::zeros((nbins, nbins, 2).f());
        
        Zip::indexed(&mut rdf_vector).par_for_each(|(i, j), rdf_val| {
            *rdf_val = *rdf
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .read()
            .unwrap();
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
    ) -> Array<f64, Dim<[usize; 3]>> {
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        assert!(npoints > 1);
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
        
        let box_lengths = vec![box_size_x, box_size_y, box_size_z];

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
                    ensure_periodicity(&mut r_ij, &box_lengths);
                }

                // XXX Probably something off for aperiodic in these numbers here? TODO check.
                
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
        let mut rdf_vector = Array::<f64, _>::zeros((nbins, nbins, nbins).f());
        Zip::indexed(&mut rdf_vector).par_for_each(|(i, j, k), rdf_val| {
            *rdf_val = *rdf
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .get(k)
            .unwrap()
            .read()
            .unwrap();
        });

        return rdf_vector;
    }

    pub fn ensure_periodicity(v: &mut Vec<f64>, box_lengths: &Vec<f64>) {
        
        v.iter_mut().zip(box_lengths).for_each(|(coord, box_length)| { 
            
            if *coord > box_length * 0.5 {
                *coord -= box_length;
            } else if *coord <= -box_length * 0.5 {
                *coord += box_length;
            }
            
        });
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
        periodic: bool
    ) -> Array<f64, Dim<[usize; 2]>> {
        // get the needed parameters from the input
        let box_lengths = vec![box_size_x, box_size_y];
        
        // Create delaunay triangulation, and count total number of faces inside to create the output array
        let delaunay = create_delaunay(points_array, periodic, &box_lengths);
        let n_faces = delaunay.num_inner_faces();
        let mut furthest_sites =  Array::<f64, _>::zeros((n_faces, 3).f());
        
        // Go through delaunay
        delaunay.inner_faces().into_iter()
        .zip(furthest_sites.axis_iter_mut(Axis(0)))
        .par_bridge()
        .into_par_iter()
        .for_each(|(face_i, mut site_i)| {
            let circumcenter = face_i.circumcenter();
            site_i[[0]] = circumcenter.x;
            site_i[[1]] = circumcenter.y;
            
            site_i[[2]] = INFINITY;
            
            for vertex in face_i.vertices() {
                let vertex_coords = vertex.position();
                let distance = hypot(vertex_coords.x - circumcenter.x, vertex_coords.y - circumcenter.y);
                if distance <= site_i[[2]] {
                    site_i[[2]] = distance;
                }
            }
            
         });
        
        return furthest_sites;
        
    }
    
    pub fn cluster_by_distance(points_array: &ArrayViewD<'_, f64>, threshold: f64, box_lengths: &Vec<f64>, periodic: bool) -> Vec<usize> {
        // tag particles by cluster they belong to, using a single threshold distance throughout
        
        let npoints = points_array.shape()[0];
        let ndim = points_array.shape()[1];

        assert!(npoints > 1);
        assert!(ndim == 2); // Need to implement/find delaunay in 3d
        
        let mut cluster_id: Vec<usize> = (0..npoints).collect();
        let mut done = false;
        
        // Make this into a LeftRight here and in point variance https://stackoverflow.com/questions/33390395/can-a-function-return-different-types-depending-on-conditional-statements-in-the
        if ndim == 2 {
            
            let mut delaunay = create_delaunay_with_tags(points_array, &cluster_id, periodic, &box_lengths);
            while !done {
                let change_list = Arc::new(RwLock::new(Vec::new()));

                cluster_id.par_iter_mut().enumerate().for_each( | (i, id)  |  {
                    // In this scenario, the from_index is safe because only the n_particles first values are explored out of 9 times that
                    let fixed_handle = FixedHandleImpl::from_index(i);
                    let dynamic_handle = delaunay.vertex(fixed_handle);
                    let outgoing_edges = dynamic_handle.out_edges();
                    
                    let xi = points_array[[i, 0]];
                    let yi = points_array[[i, 1]];
    
                    for e in outgoing_edges {
    
                        let neigh_vertex = e.to().position();
                        let dx = neigh_vertex.x - xi;
                        let dy = neigh_vertex.y - yi;
                        let mut vector = vec![dx, dy];
                        if periodic {
                            ensure_periodicity(&mut vector, &box_lengths);
                        }
                        
                        // Inherit lower of 2 tags if distance is smaller than threshold
                        let edge_length = hypot(vector[0], vector[1]);
                        let neigh_tag = e.to().data().tag;
                        if edge_length <= threshold && *id > neigh_tag {
                                *id = neigh_tag;
                                change_list.write().unwrap().push(i);
                        }
                        
                    }
                    
                } );
                
                // Update tags in delaunay
                // TODO make this part better
                change_list.read().unwrap().clone().into_iter().for_each(| index | {
                let fixed_handle = FixedHandleImpl::from_index(index);
                (*delaunay.vertex_data_mut(fixed_handle)).tag = cluster_id[index];
                if periodic {
                    for p in 1..8 {
                        let fixed_handle = FixedHandleImpl::from_index(index + npoints * p);
                        (*delaunay.vertex_data_mut(fixed_handle)).tag = cluster_id[index];
                    }
                }
                    
                });

                done = change_list.read().unwrap().len() == 0;
            }
            
        } else {
            panic!("Not implemented yet!");
        }
        
        return cluster_id;
        
    }
    
    pub struct PointWithTag<T> {
        position: Point2<f64>,
        tag: T,
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
    
    pub fn point_variances(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, box_lengths: &Vec<f64>, n_samples: usize, periodic: bool) -> Vec<f64> {

        let n_radii = radii.shape()[0];
        let means  = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];
        let means2 = vec_no_clone![Arc::new(RwLock::new(0.0_f64)); n_radii];

        let npoints = points.shape()[0];
        let ndim = points.shape()[1];

        assert!(npoints > 1);
        assert!(ndim < 4);
        assert!(box_lengths.len() == ndim);

        if ndim == 2 { // 2D case

            // Construct the R*-tree, taking into account periodic boundary conditions
            // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
            let rtree_positions = compute_periodic_rstar_tree(&points, box_lengths[0], box_lengths[1], periodic);

            // Throw points at random: this can be parallelized if necessary
            radii.to_vec().into_iter().enumerate().for_each(|(current_index,radius)| {

                (0..n_samples).into_par_iter().for_each( |_sample| {

                    let mut rng = rand::thread_rng();

                    let mut random: f64 = rng.gen();
                    let x_center = random; 
                    random = rng.gen();
                    let y_center = random;

                    let r_center = vec![x_center, y_center];
                    let count = count_points_in_disk(&rtree_positions, r_center, radius);

                    *means.get(current_index).unwrap().write().unwrap() += count as f64;
                    *means2.get(current_index).unwrap().write().unwrap() += (count*count) as f64;
                    
                });
            });

        } else { // 3D case

            let rtree_positions = compute_periodic_rstar_tree_3d(&points, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

            // Throw points at random: this can be parallelized if necessary
            radii.to_vec().into_iter().enumerate().for_each(|(current_index,radius)| {
                (0..n_samples).into_par_iter().for_each( |_sample| {

                    let mut rng = rand::thread_rng();

                    let mut random: f64 = rng.gen();
                    let x_center = random; 
                    random = rng.gen();
                    let y_center = random;
                    random = rng.gen();
                    let z_center = random;

                    let r_center = vec![x_center, y_center, z_center];
                    let count = count_points_in_ball(&rtree_positions, r_center, radius);

                    *means.get(current_index).unwrap().write().unwrap() += count as f64;
                    *means2.get(current_index).unwrap().write().unwrap() += (count*count) as f64;
                    
                });
            });
        }

        let mut reduced_variances = Vec::new();
        for index in 0..n_radii {
            let current_mean = *means.get(index).unwrap().read().unwrap() / n_samples as f64;
            let current_mean2 = *means2.get(index).unwrap().read().unwrap() / n_samples as f64;

            let reduced_variance = current_mean2/ (current_mean*current_mean) - 1.0;
            reduced_variances.push(reduced_variance)
        }

        return reduced_variances;
    }

    pub fn count_metric_neighbors(points: &ArrayViewD<'_, f64>, radii: &ArrayView1<'_, f64>, threshold: f64, box_lengths: &Vec<f64>, periodic: bool) -> Vec<usize> {
        
        let npoints = points.shape()[0];
        let ndim = points.shape()[1];
        
        let mut neighbor_counts = vec!(0; npoints);
        
        assert!(npoints > 1);
        assert!(ndim < 4);
        assert!(box_lengths.len() == ndim);
        
        let max_radius = *radii.into_par_iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        if ndim == 2 { // 2D case

            // Construct the R*-tree, taking into account periodic boundary conditions
            // See https://en.wikipedia.org/wiki/R-tree and https://en.wikipedia.org/wiki/R*-tree
            let rtree_positions = compute_decorated_rstar_tree(&points, &radii, box_lengths[0], box_lengths[1], periodic);

            radii.to_vec().into_par_iter().zip(neighbor_counts.par_iter_mut()).enumerate().for_each(|(current_index,(radius, count))| {
                rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]]])
                .skip(1)
                .for_each(| (neighbor, dist2) | {
                    let neighbor_radius = neighbor.data;
                    if dist2 < (radius + neighbor_radius).powi(2) {
                        *count += 1;
                    }
                });
            });

        } else { // 3D case

            let rtree_positions = compute_decorated_rstar_tree_3d(&points, &radii, box_lengths[0], box_lengths[1], box_lengths[2], periodic);

            radii.to_vec().into_par_iter().zip(neighbor_counts.par_iter_mut()).enumerate().for_each(|(current_index,(radius, count))| {
                rtree_positions.nearest_neighbor_iter_with_distance_2(&[points[[current_index,0]], points[[current_index,1]], points[[current_index,2]]])
                .skip(1)
                .for_each(| (neighbor, dist2) | {
                    let neighbor_radius = neighbor.data;
                    if dist2 < (radius + neighbor_radius).powi(2) {
                        *count += 1;
                    }
                });
            });
            
        }
        
        
        return neighbor_counts;
        
    }
    
    pub fn count_points_in_disk(rtree: &RTree<[f64;2]>, r_center: Vec<f64>, radius: f64) -> usize {

        let centerpoint = [r_center[0], r_center[1]];

        // let points = rtree.lookup_in_circle(&centerpoint, &(radius*radius));
        let points = rtree.locate_within_distance(centerpoint, radius*radius).collect::<Vec<_>>();
        // let count = points.len();
        let count = points.len();

        return count
    }

    pub fn count_points_in_ball(rtree: &RTree<[f64;3]>, r_center: Vec<f64>, radius: f64) -> usize {

        // let centerpoint: Point3<f64> = Point3::new(r_center[0], r_center[1], r_center[2]);
        let centerpoint = [r_center[0], r_center[1], r_center[2]];

        // let points = rtree.lookup_in_circle(&centerpoint, &(radius*radius));
        let points = rtree.locate_within_distance(centerpoint, radius*radius).collect::<Vec<_>>();
        // let count = points.len();
        let count = points.len();

        return count
    }

    pub fn compute_periodic_rstar_tree(
        positions: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        periodic: bool
    ) -> RTree<[f64;2]> {


        // Create an empty for the vector of points to load into the tree
        // let mut points_vector: Vec<Point2<f64>> = Vec::new();
        // let mut points_vector: Arc<RwLock<Vec<Point2<f64>>>> = Arc::new(RwLock::new(Vec::new()));
        let points_vector: Arc<RwLock<Vec<[f64;2]>>> = Arc::new(RwLock::new(Vec::new()));

        let n_copies: usize = if periodic { 9 } else { 1 };
        // Get the number of particles from positions, and consider 9 copies of the system to take into account the PBCs
        let n_particles = positions.shape()[0];
        (0..n_copies*n_particles).into_par_iter().for_each(|i| {
        // for i in 0..9 * n_particles {
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

            let shifted_x: f64 = positions[[index, 0]] + nx * box_size_x;
            let shifted_y: f64 = positions[[index, 1]] + ny * box_size_y;

            let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
            let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;

            if is_in_x && is_in_y {

                // let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);
                let newpoint = [shifted_x, shifted_y];

                // points_vector.push(newpoint);
                points_vector.write().unwrap().push(newpoint);
            }
        // }
        });

        //Bulk-load the vector into a new rtree, then return it
        // It's apparently better? http://ftp.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-74/files/FORUM_18.pdf
        let rtree = RTree::bulk_load(points_vector.read().unwrap().to_vec());

        rtree
    }


    pub fn compute_periodic_rstar_tree_3d(
        positions: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        box_size_z: f64,
        periodic: bool
    ) -> RTree<[f64;3]> {


        // Create an empty for the vector of points to load into the tree
        // let mut points_vector: Vec<Point2<f64>> = Vec::new();
        // let mut points_vector: Arc<RwLock<Vec<Point3<f64>>>> = Arc::new(RwLock::new(Vec::new()));
        let points_vector: Arc<RwLock<Vec<[f64;3]>>> = Arc::new(RwLock::new(Vec::new()));

        // Get the number of particles from positions, and consider 27 copies of the system to take into account the PBCs
        let n_copies: usize = if periodic { 27 } else { 1 };
        let n_particles = positions.shape()[0];
        (0..n_copies*n_particles).into_par_iter().for_each(|i| {

            let index = i % n_particles;
            let quotient = i / n_particles;
            let triplet_1 = quotient % 3;
            let nonuplet = quotient / 3;
            let triplet_2 = nonuplet % 3;
            let triplet_3 = nonuplet / 3;

            let nx = match triplet_1 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };
            let ny = match triplet_2 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };
            let nz = match triplet_3 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };

            assert!(nx < 2.0 && ny < 2.0 && nz < 2.0, "Unexpected error when cloning boxes.");

            let shifted_x: f64 = positions[[index, 0]] + nx * box_size_x;
            let shifted_y: f64 = positions[[index, 1]] + ny * box_size_y;
            let shifted_z: f64 = positions[[index, 2]] + nz * box_size_z;

            let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
            let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;
            let is_in_z = shifted_z >= -0.5 * box_size_z && shifted_z <= 1.5 * box_size_z;

            if is_in_x && is_in_y && is_in_z {

                // let newpoint: Point3<f64> = Point3::new(shifted_x, shifted_y, shifted_z);
                let newpoint = [shifted_x, shifted_y, shifted_z];

                // points_vector.push(newpoint);
                points_vector.write().unwrap().push(newpoint);
            }
        // }
        });

        //Bulk-load the vector into a new rtree, then return it
        // It's apparently better? http://ftp.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-74/files/FORUM_18.pdf
        let rtree = RTree::bulk_load(points_vector.read().unwrap().to_vec());

        rtree
    }
    
    // https://docs.rs/rstar/latest/rstar/primitives/struct.GeomWithData.html
    type ListPointWithTag<T, const N:usize> = GeomWithData<[f64; N], T>;

    pub fn compute_decorated_rstar_tree(
        positions: &ArrayViewD<'_, f64>,
        field: &ArrayView1<'_, f64>, // Assumed scalar for now
        box_size_x: f64,
        box_size_y: f64,
        periodic: bool
    ) -> RTree<ListPointWithTag<f64, 2>> {


        // Create an empty for the vector of points to load into the tree
        // let mut points_vector: Vec<Point2<f64>> = Vec::new();
        // let mut points_vector: Arc<RwLock<Vec<Point2<f64>>>> = Arc::new(RwLock::new(Vec::new()));
        let tagged_points_vector: Arc<RwLock<Vec<ListPointWithTag<f64, 2>>>> = Arc::new(RwLock::new(Vec::new()));

        let n_copies: usize = if periodic { 9 } else { 1 };
        // Get the number of particles from positions, and consider 9 copies of the system to take into account the PBCs
        let n_particles = positions.shape()[0];
        (0..n_copies*n_particles).into_par_iter().for_each(|i| {
        // for i in 0..9 * n_particles {
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

            let shifted_x: f64 = positions[[index, 0]] + nx * box_size_x;
            let shifted_y: f64 = positions[[index, 1]] + ny * box_size_y;

            let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
            let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;

            if is_in_x && is_in_y {

                // let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);
                let newpoint = [shifted_x, shifted_y];
                let new_tagged_point = ListPointWithTag::new(newpoint, field[[index]]);

                // points_vector.push(newpoint);
                tagged_points_vector.write().unwrap().push(new_tagged_point);
            }
        // }
        });

        //Bulk-load the vector into a new rtree, then return it
        // It's apparently better? http://ftp.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-74/files/FORUM_18.pdf
        let rtree = RTree::bulk_load(tagged_points_vector.read().unwrap().to_vec());

        rtree
    }
    
    
    pub fn compute_decorated_rstar_tree_3d(
        positions: &ArrayViewD<'_, f64>,
        field: &ArrayView1<'_, f64>, // Assumed scalar for now
        box_size_x: f64,
        box_size_y: f64,
        box_size_z: f64,
        periodic: bool
    ) -> RTree<ListPointWithTag<f64, 3>> {


        // Create an empty for the vector of points to load into the tree
        // let mut points_vector: Vec<Point2<f64>> = Vec::new();
        // let mut points_vector: Arc<RwLock<Vec<Point3<f64>>>> = Arc::new(RwLock::new(Vec::new()));
        let tagged_points_vector: Arc<RwLock<Vec<ListPointWithTag<f64,3>>>> = Arc::new(RwLock::new(Vec::new()));

        // Get the number of particles from positions, and consider 27 copies of the system to take into account the PBCs
        let n_copies: usize = if periodic { 27 } else { 1 };
        let n_particles = positions.shape()[0];
        (0..n_copies*n_particles).into_par_iter().for_each(|i| {

            let index = i % n_particles;
            let quotient = i / n_particles;
            let triplet_1 = quotient % 3;
            let nonuplet = quotient / 3;
            let triplet_2 = nonuplet % 3;
            let triplet_3 = nonuplet / 3;

            let nx = match triplet_1 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };
            let ny = match triplet_2 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };
            let nz = match triplet_3 {
                0 => 0.0,
                1 => -1.0,
                2 => 1.0,
                _ => 2.0,
            };

            assert!(nx < 2.0 && ny < 2.0 && nz < 2.0, "Unexpected error when cloning boxes.");

            let shifted_x: f64 = positions[[index, 0]] + nx * box_size_x;
            let shifted_y: f64 = positions[[index, 1]] + ny * box_size_y;
            let shifted_z: f64 = positions[[index, 2]] + nz * box_size_z;

            let is_in_x = shifted_x >= -0.5 * box_size_x && shifted_x <= 1.5 * box_size_x;
            let is_in_y = shifted_y >= -0.5 * box_size_y && shifted_y <= 1.5 * box_size_y;
            let is_in_z = shifted_z >= -0.5 * box_size_z && shifted_z <= 1.5 * box_size_z;

            if is_in_x && is_in_y && is_in_z {

                // let newpoint: Point3<f64> = Point3::new(shifted_x, shifted_y, shifted_z);
                let newpoint = [shifted_x, shifted_y, shifted_z];
                let new_tagged_point = ListPointWithTag::new(newpoint, field[[index]]);

                // points_vector.push(newpoint);
                tagged_points_vector.write().unwrap().push(new_tagged_point);
            }
        // }
        });

        //Bulk-load the vector into a new rtree, then return it
        // It's apparently better? http://ftp.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-74/files/FORUM_18.pdf
        let rtree = RTree::bulk_load(tagged_points_vector.read().unwrap().to_vec());

        rtree
    }
    
}
