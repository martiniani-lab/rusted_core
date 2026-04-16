use ndarray::{Array,Dim};
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python, Bound};

// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

mod geometry;
mod spatial;
mod vector_rdf;
mod radial;
mod voronoi;
mod neighbors;

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
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_vector_rdf_2d(&array, box_size, box_size, binsize, periodic);
        PyArray2::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_vector_rdf3d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray3<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_vector_rdf_3d(&array, box_size, box_size, box_size, binsize, periodic);
        PyArray3::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_vector_rdf2sphere<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        binsize: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_vector_rdf_2sphere(&array, binsize);
        PyArray2::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_bounded_vector_rdf2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        radial_bound: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray2<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_bounded_vector_rdf_2d(&array, box_size, box_size, binsize, radial_bound, periodic);
        PyArray2::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_nnbounded_vector_rdf2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        nn_bound: usize,
        periodic: bool,
    ) -> Bound<'py, PyArray2<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_nnbounded_vector_rdf_2d(&array, box_size, box_size, binsize, nn_bound, periodic);
        PyArray2::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_pnn_vector_rdf2d<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        p: usize,
        box_size: f64,
        binsize: f64,
        periodic: bool
    ) -> Bound<'py, PyArray2<f64>> {
        let array = unsafe { points.as_array() };
        assert!(p>0, "Order for pth-nn distribution needs to be natural integer");
        let result = vector_rdf::compute_pnn_vector_rdf_2d(&array, box_size, box_size, binsize, p, periodic);
        PyArray2::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_pnn_rdf<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        nn_order: usize,
        box_size: f64,
        binsize: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { points.as_array() };
        assert!(nn_order>0, "Order for pth-nn distribution needs to be natural integer");
        let ndim = array.shape()[1];
        let box_lengths = vec![box_size; ndim];
        let result = radial::compute_pnn_rdf(&array, &box_lengths, binsize, nn_order, periodic);
        PyArray1::from_vec(py, result)
    }

    #[pyfn(m)]
    fn compute_bounded_vector_rdf3d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        radial_bound: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray3<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_bounded_vector_rdf_3d(&array, box_size, box_size, box_size, binsize, radial_bound, periodic);
        PyArray3::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_nnbounded_vector_rdf3d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        nn_bound: usize,
        periodic: bool,
    ) -> Bound<'py, PyArray3<f64>> {
        let array = unsafe { points.as_array() };
        let result = vector_rdf::compute_nnbounded_vector_rdf_3d(&array, box_size, box_size, box_size, binsize, nn_bound, periodic);
        PyArray3::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_pnn_vector_rdf3d<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        p: usize,
        box_size: f64,
        binsize: f64,
        periodic: bool
    ) -> Bound<'py, PyArray3<f64>> {
        let array = unsafe { points.as_array() };
        assert!(p>0, "Order for pth-nn distribution needs to be natural integer");
        let result = vector_rdf::compute_pnn_vector_rdf_3d(&array, box_size, box_size, box_size, binsize, p, periodic);
        PyArray3::from_array(py, &result)
    }

    #[pyfn(m)]
    fn compute_vector_gyromorphic_corr_2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Bound<'py,PyArray2<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 3]>>>) {
        let array = unsafe { points.as_array() };
        let (rdf, corr) = vector_rdf::compute_vector_gyromorphic_corr_2d(&array, box_size, binsize, periodic, order);
        let array_rdf = PyArray2::from_array(py, &rdf);
        let array_corr = PyArray::from_array(py, &corr);
        (array_rdf, array_corr)
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
        let array = unsafe { points.as_array() };
        let field_array = unsafe { field.as_array() };
        let field_shape = field_array.shape();

        assert!(
            field_shape[0] == array.shape()[0],
            "You must provide as many field lines as particles!"
        );

        let (rdf, field_corrs) = radial::compute_radial_correlations_2d(
            &array, &field_array, box_size, box_size, binsize, periodic, connected,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_array(py, &field_corrs);
        (array_rdf, pyarray_field_corrs)
    }

    #[pyfn(m)]
    fn compute_radial_gyromorphic_corr_2d<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        binsize: f64,
        periodic: bool,
        order: u64,
    ) -> (Bound<'py,PyArray1<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 2]>>>) {
        let array = unsafe { points.as_array() };
        let (rdf, radial_corr) = radial::compute_radial_gyromorphic_corr_2d(
            &array, box_size, box_size, binsize, periodic, order,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let array_corr = PyArray::from_array(py, &radial_corr);
        (array_rdf, array_corr)
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
        let array = unsafe { points.as_array() };
        let field_array = unsafe { field.as_array() };
        let field_shape = field_array.shape();

        assert!(
            field_shape[0] == array.shape()[0],
            "You must provide as many field lines as particles!"
        );

        let (rdf, field_corrs) = radial::compute_radial_correlations_3d(
            &array, &field_array, box_size, box_size, box_size, binsize, periodic, connected,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_owned_array(py, field_corrs);
        (array_rdf, pyarray_field_corrs)
    }

    #[pyfn(m)]
    fn compute_radial_correlations_2sphere<'py>(
        py: Python<'py>,
        points: Bound<'py, PyArrayDyn<f64>>,
        field: Bound<'py, PyArrayDyn<f64>>,
        binsize: f64,
        connected: bool,
    ) -> (Bound<'py,PyArray1<f64>>, Bound<'py,PyArray<f64, Dim<[usize; 2]>>>) {
        let array = unsafe { points.as_array() };
        let field_array = unsafe { field.as_array() };
        let field_shape = field_array.shape();

        assert!(
            field_shape[0] == array.shape()[0],
            "You must provide as many field lines as particles!"
        );

        let (rdf, field_corrs) = radial::compute_radial_correlations_2sphere(
            &array, &field_array, binsize, connected,
        );
        let array_rdf = PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_array(py, &field_corrs);
        (array_rdf, pyarray_field_corrs)
    }

    #[pyfn(m)]
    fn compute_2d_boops<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        orders: &Bound<'py, PyArrayDyn<isize>>,
        box_size: f64,
        periodic: bool,
    ) -> Bound<'py, PyArray<f64, Dim<[usize; 3]>>> {
        let array = unsafe { points.as_array() };
        let boop_order_array = unsafe { orders.as_array() };
        let boops = voronoi::compute_steinhardt_boops_2d(&array, &boop_order_array, box_size, box_size, periodic);
        PyArray::from_array(py, &boops)
    }

    #[pyfn(m)]
    fn compute_2sphere_boops<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        orders: &Bound<'py, PyArrayDyn<isize>>,
    ) -> Bound<'py, PyArray<f64, Dim<[usize; 3]>>> {
        let array = unsafe { points.as_array() };
        let boop_order_array = unsafe { orders.as_array() };
        let boops = voronoi::compute_steinhardt_boops_2sphere(&array, &boop_order_array);
        PyArray::from_array(py, &boops)
    }

    #[pyfn(m)]
    fn compute_2sphere_all_voronoi_quantities<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>> ) {
        let array = unsafe { points.as_array() };
        let (areas, neighbour_counts, nn_distances) = voronoi::compute_voronoi_quantities_2sphere(
            &array, true, true, true
        );
        let array_areas = PyArray1::from_vec(py, areas.unwrap());
        let array_neighbour_counts = PyArray1::from_vec(py, neighbour_counts.unwrap());
        let array_nn_distances = PyArray1::from_vec(py, nn_distances.unwrap());
        (array_areas, array_neighbour_counts, array_nn_distances)
    }

    #[pyfn(m)]
    fn compute_2d_all_voronoi_quantities<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>> ) {
        let array = unsafe { points.as_array() };
        let (areas, neighbour_counts, nn_distances) = voronoi::compute_voronoi_quantities_2d(
            &array, box_size, box_size, periodic, true, true, true
        );
        let array_areas = PyArray1::from_vec(py, areas.unwrap());
        let array_neighbour_counts = PyArray1::from_vec(py, neighbour_counts.unwrap());
        let array_nn_distances = PyArray1::from_vec(py, nn_distances.unwrap());
        (array_areas, array_neighbour_counts, array_nn_distances)
    }

    #[pyfn(m)]
    fn compute_2d_voronoi_areas<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { points.as_array() };
        let (areas,_,_) = voronoi::compute_voronoi_quantities_2d(
            &array, box_size, box_size, periodic, true, false, false
        );
        PyArray1::from_vec(py, areas.unwrap())
    }

    #[pyfn(m)]
    fn compute_2d_voronoi_neighbour_numbers<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        let array = unsafe { points.as_array() };
        let (_,neighbour_counts,_) = voronoi::compute_voronoi_quantities_2d(
            &array, box_size, box_size, periodic, false, true, false
        );
        PyArray1::from_vec(py, neighbour_counts.unwrap())
    }

    #[pyfn(m)]
    fn compute_2d_voronoi_nn_distances<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { points.as_array() };
        let (_,_,nn_distances) = voronoi::compute_voronoi_quantities_2d(
            &array, box_size, box_size, periodic, false, false, true
        );
        PyArray1::from_vec(py, nn_distances.unwrap())
    }

    #[pyfn(m)]
    fn voronoi_furthest_sites<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray<f64, Dim<[usize; 2]>>> {
        let array = unsafe { points.as_array() };
        let furthest_sites = voronoi::voronoi_furthest_site(&array, box_size, box_size, periodic);
        PyArray::from_array(py, &furthest_sites)
    }

    #[pyfn(m)]
    fn compute_pnn_distances<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        p: usize,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { points.as_array() };
        assert!(p>0, "Order for pth-nn distribution needs to be natural integer");
        let ndim = array.shape()[1];
        let box_lengths = vec!(box_size; ndim);
        let pnn_distances = neighbors::compute_pnn_distances(&array, p, &box_lengths, periodic);
        PyArray1::from_vec(py, pnn_distances)
    }

    #[pyfn(m)]
    fn compute_pnn_mean_nnbound_distances<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        nn_bound: usize,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { points.as_array() };
        assert!(nn_bound>0, "Order for pth-nn distribution needs to be natural integer");
        let ndim = array.shape()[1];
        let box_lengths = vec!(box_size; ndim);
        let pnn_distances = neighbors::compute_pnn_mean_nnbound_distances(&array, &box_lengths, nn_bound, periodic);
        PyArray1::from_vec(py, pnn_distances)
    }

    #[pyfn(m)]
    fn cluster_by_distance<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        threshold: f64,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        let array = unsafe { points.as_array() };
        let ndim = array.shape()[1];
        let box_lengths = vec![box_size; ndim];
        let cluster_id = neighbors::cluster_by_distance(&array, threshold, &box_lengths, periodic);
        PyArray1::from_vec(py, cluster_id)
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
        let array = unsafe { x.as_array() };
        let radii = unsafe { radii.as_array() };
        let ndim = array.shape()[1];
        let reduced_variances = neighbors::point_variances(&array, &radii, &vec![box_size; ndim], n_samples, periodic);
        reduced_variances.into_pyarray(py)
    }

    #[pyfn(m)]
    fn point_variances_2sphere<'py>(
        py: Python<'py>,
        x: Bound<'py, PyArrayDyn<f64>>,
        radii: Bound<'py, PyArray1<f64>>,
        n_samples: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        let array = unsafe { x.as_array() };
        let radii = unsafe { radii.as_array() };
        let reduced_variances = neighbors::point_variances_2sphere(&array, &radii, n_samples);
        reduced_variances.into_pyarray(py)
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
        let array = unsafe { points.as_array() };
        let radii_array = unsafe { radii.as_array() };
        let neighbor_counts = neighbors::count_metric_neighbors(
            &array, &radii_array, threshold, &vec![box_size; 2], periodic
        );
        PyArray1::from_vec(py, neighbor_counts)
    }

    #[pyfn(m)]
    fn monodisperse_metric_neighbors<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        radius: f64,
        threshold: f64,
        box_size: f64,
        periodic: bool
    ) -> Bound<'py, PyArray1<usize>> {
        let array = unsafe { points.as_array() };
        let n_particles = array.shape()[0];
        let radii_array_owned = Array::from_elem(n_particles, radius);
        let radii_array = radii_array_owned.view();
        let neighbor_counts = neighbors::count_metric_neighbors(
            &array, &radii_array, threshold, &vec![box_size; 2], periodic
        );
        PyArray1::from_vec(py, neighbor_counts)
    }

    #[pyfn(m)]
    fn metric_neighbors_2sphere<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        radii: &Bound<'py, PyArray1<f64>>,
        threshold: f64
    ) -> Bound<'py, PyArray1<usize>> {
        let array = unsafe { points.as_array() };
        let radii_array = unsafe { radii.as_array() };
        let neighbor_counts = neighbors::count_metric_neighbors_2sphere(&array, &radii_array, threshold);
        PyArray1::from_vec(py, neighbor_counts)
    }

    #[pyfn(m)]
    fn monodisperse_metric_neighbors_2sphere<'py>(
        py: Python<'py>,
        points: &Bound<'py, PyArrayDyn<f64>>,
        radius: f64,
        threshold: f64
    ) -> Bound<'py, PyArray1<usize>> {
        let array = unsafe { points.as_array() };
        let n_particles = array.shape()[0];
        let radii_array_owned = Array::from_elem(n_particles, radius);
        let radii_array = radii_array_owned.view();
        let neighbor_counts = neighbors::count_metric_neighbors_2sphere(&array, &radii_array, threshold);
        PyArray1::from_vec(py, neighbor_counts)
    }

    Ok(())
}
