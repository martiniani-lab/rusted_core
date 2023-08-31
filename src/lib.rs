use numpy::{PyArray, PyArray1, PyArray2, PyArrayDyn};
use ndarray::Dim;
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
        periodic: bool
    ) -> & 'py PyArray2<f64> {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type

        // Mutate the data
        // No need to return any value as the input data is mutated
        let vector_rdf = rust_fn::compute_vector_rdf2d(&array, box_size, binsize, periodic);
        let array_rdf =  PyArray2::from_vec2(py, &vector_rdf).unwrap();
        
        return array_rdf
    }

    #[pyfn(m)]
    fn compute_radial_correlations_2d<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        field: &PyArrayDyn<f64>,
        box_size: f64,
        binsize: f64,
        periodic: bool
    ) -> (& 'py PyArray1<f64>, & 'py PyArray<f64, Dim<[usize;2]>>) {
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let field_array = unsafe { field.as_array()}; // Same for field
        let field_shape = field_array.shape(); // Need to feed npoints x fielddim values
        
        assert!(field_shape[0] == array.shape()[0], "You must provide as many field lines as particles!");

        // Mutate the data
        // No need to return any value as the input data is mutated
        let (rdf, field_corrs) = rust_fn::compute_radial_correlations_2d(&array, &field_array, box_size, box_size, binsize, periodic);
        let array_rdf =  PyArray::from_vec(py, rdf);
        let pyarray_field_corrs = PyArray::from_owned_array(py, field_corrs);
        
        return(array_rdf, pyarray_field_corrs)
    }
    
    #[pyfn(m)]
    fn compute_2d_boops<'py>(
        py: Python<'py>,
        points: &PyArrayDyn<f64>,
        orders: &PyArrayDyn<usize>,
        box_size: f64,
        periodic: bool
    )  -> & 'py PyArray<f64, Dim<[usize; 3]>> {
        
        // First we convert the Python numpy array into Rust ndarray
        // Here, you can specify different array sizes and types.
        let array = unsafe { points.as_array() }; // Convert to ndarray type
        let boop_order_array = unsafe { orders.as_array()}; // Same for field
        
        let boops = rust_fn::compute_steinhardt_boops_2d(&array, &boop_order_array, box_size, box_size, periodic);
        let array_boops = PyArray::from_owned_array(py, boops);
        
        return array_boops
    }
    
    Ok(())

}

// The rust side functions
// Put it in mod to separate it from the python bindings
mod rust_fn {
    use ndarray::parallel::prelude::*;
    use ndarray::{Array,Dim,ShapeBuilder,s};
    use numpy::ndarray::ArrayViewD;
    use std::sync::{Arc, RwLock};
    use ang::atan2;
    use libm::hypot;
    use std::f64::consts::PI;
    extern crate spade;
    use nalgebra::Point2;
    use spade::delaunay::FloatDelaunayTriangulation;

    // Vectors of RwLocks cannot be initialized with a clone!
    macro_rules! vec_no_clone {
        ( $val:expr; $n:expr ) => {{
            let result: Vec<_> = std::iter::repeat_with(|| $val).take($n).collect();
            result
        }};
    }

    pub fn compute_vector_rdf2d(points: &ArrayViewD<'_, f64>, box_size: f64, binsize: f64, periodic: bool) -> Vec<Vec<f64>> {

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
        periodic: bool
    ) -> Vec<Vec<f64>> {
        // Check that the binsize is physical
        assert!(
            binsize > 0.0,
            "Something is wrong with the binsize used for the RDF"
        );

        let nbins = if periodic { (box_size_x / binsize).ceil() as usize} else { 2 * (box_size_x / binsize).ceil() as usize };
        
        let n_particles = points.shape()[0];
        let rdf: Vec<Vec<Arc<RwLock<f64>>>> = vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); nbins]; nbins];

        
        (0..n_particles).into_par_iter().for_each(|i| {
            for j in i+1..n_particles {
                
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
        for i in 0..nbins {
            for j in 0..nbins {
                rdf_vector[i][j] = *rdf.get(i).unwrap().get(j).unwrap().read().unwrap();
            }
        }
        
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
    
    
    pub fn compute_radial_correlations_2d(
        points: &ArrayViewD<'_, f64>,
        fields: &ArrayViewD<'_, f64>,
        box_size_x: f64,
        box_size_y: f64,
        binsize: f64,
        periodic: bool
    ) -> (Vec<f64>, Array<f64, Dim<[usize; 2]>>) {
        // Check that the binsize is physical
        assert!(
            binsize > 0.0,
            "Something is wrong with the binsize used for the RDF"
        );

        // get the needed parameters from the input
        let n_particles = points.shape()[0];
        let field_dim = fields.shape()[1];
        let nbins = if periodic { (box_size_x / binsize).ceil() as usize} else { 2 * (box_size_x / binsize).ceil() as usize };
        let max_dist = hypot(box_size_x / 2.0, box_size_y / 2.0);
        let rdf: Vec<Arc<RwLock<f64>>> = vec_no_clone![Arc::new(RwLock::new(0.0)); nbins];
        let field_corrs: Vec<Vec<Arc<RwLock<f64>>>> = vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); nbins]; field_dim];

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
                    *(field_corrs[index][k].write().unwrap()) += (fields[[i,k]] - mean_field[k]) * (fields[[j,k]] - mean_field[k]);
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
                // Use the actual normalization in a square, not the lazy one
                let normalisation = if bincenter <= box_size_x / 2.0 {
                    2.0 * PI * bincenter * binsize * n_particles as f64 / (box_size_x * box_size_y)
                // (2 pi r  rho dr)
                } else {
                    binsize * n_particles as f64 / (box_size_x * box_size_y)
                        * 2.0
                        * bincenter
                        * (PI - 4.0 * (0.5 * box_size_x / bincenter).acos()) // rho dr * 2 r *( pi - 4 acos(L/2r))
                };
                *(rdf[bin].write().unwrap()) = current_count / (n_particles as f64 * normalisation / 2.0); // the number of count is 1/ averaged over N 2/ normalised by the uniform case 3/ divided by two because pairs are counted once only
            }
        });
        
        let mut rdf_vector = vec![0.0; nbins];
        let mut field_corrs_vector = Array::<f64, _>::zeros((nbins, field_dim).f());
        for bin in 0..nbins {
            rdf_vector[bin] =  *rdf.get(bin).unwrap().read().unwrap();
            for dim in 0..field_dim {
                field_corrs_vector[[bin,dim]] = *field_corrs.get(bin).unwrap().get(dim).unwrap().read().unwrap();
            }
        }
        
        return (rdf_vector, field_corrs_vector)
        
    }
    
    
    
    pub fn compute_steinhardt_boops_2d(
        points: &ArrayViewD<'_, f64>,
        boop_order_array: &ArrayViewD<'_, usize>,
        box_size_x: f64,
        box_size_y: f64,
        periodic: bool
    ) -> Array<f64, Dim<[usize; 3]>> {
        
        // get the needed parameters from the input
        let n_particles = points.shape()[0];
        let boop_orders_number = boop_order_array.shape()[1];
        
        let boops: Vec<Vec<Vec<Arc<RwLock<f64>>>>> = vec_no_clone![vec_no_clone![vec_no_clone![Arc::new(RwLock::new(0.0)); n_particles]; boop_orders_number]; 2];
        let mut handles = Vec::new();

        let mut delaunay = FloatDelaunayTriangulation::with_walk_locate();

        let max_index_periodic: usize = if periodic {
            9 * n_particles
        } else {
            n_particles
        };
        
        // Enforce periodicity by adding copies.
        for i in 0..max_index_periodic {
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
            
            let xi = points[[index, 0]];
            let yi = points[[index, 1]];

            let shifted_x: f64 = xi + nx * box_size_x;
            let shifted_y: f64 = yi + ny * box_size_y;

            let newpoint: Point2<f64> = Point2::new(shifted_x, shifted_y);

            let handle = delaunay.insert(newpoint);
            handles.push(handle);
        }

        (0..n_particles).into_par_iter().for_each(|i| {
            // let delaunay_handle = FloatDelaunayTriangulation::locate_vertex(Point2::new(x[i], y[i]));
            let dynamic_handle = delaunay.vertex(handles[i]);
            let outgoing_edges = dynamic_handle.ccw_out_edges();

            let mut neighbour_count = 0;
            
            let xi = points[[i, 0]];
            let yi = points[[i, 1]];

            for e in outgoing_edges {
                neighbour_count += 1;

                let neigh_vertex = *e.to();
                let dx = neigh_vertex[0] - xi;
                let dy = neigh_vertex[1] - yi;
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
                    
                    *(boops[i][n][0].write().unwrap()) += dpsinx;
                    *(boops[i][n][1].write().unwrap()) += dpsiny;
                    
                }
            }

            
            for n in 0..boop_orders_number {
                *(boops[i][n][0].write().unwrap()) /= neighbour_count as f64;
                *(boops[i][n][1].write().unwrap()) /= neighbour_count as f64;
            }
        });
        
        let mut boop_vectors = Array::<f64, _>::zeros((n_particles, boop_orders_number, 2).f()); 
        for i in 0..n_particles{
            for n in 0..boop_orders_number {
                boop_vectors[[i,n,0]] = *(boops[i][n][0].read().unwrap());
                boop_vectors[[i,n,1]] = *(boops[i][n][1].read().unwrap());
            }
        }
        
        return boop_vectors;
    }
    
}
