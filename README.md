# **Rusted Core**
## A simple code to compute usual correlations and observables in point patterns

### Requires:
- rust https://www.rust-lang.org/tools/install
- maturin https://www.maturin.rs/installation
- numpy
- matplotlib
- cmasher
- hickle

### Installation:
1. Install rust, python, then pip install the python packages.
2. From the main directory, run `maturin develop --release`
3. Use rusted_core.py as a simple script to perform calculations and/or use the examples as guides to code your own helper scripts
4. Voilà!

### Current functionalities
This is very much a WIP but the code already supports
- Radial g(r) and radial field correlations (for arbitrary scalar or vector fields) in 2d, 3d, or on the sphere, either connected or non-connected, for either square periodic or free boundary conditions.
- Vector g(r) in 2d, 3d, and on the sphere with options to compute only up to a radial bound or to the p-th nearest metric neighbor
- Radial or vector statistics of the p-th nearest metric neighbor distances, relying on R-Trees for speed
- Steinhardt's BOOPs in 2d and on spheres
- Gyromorphic correlation in 2d
- Voronoi quantities (nearest neighbor distance, Voronoi cell area, Voronoi number of neighbors) and option to compute quantities averaged over Voronoi neighbors in 2d and on spheres
- Cluster tagging according to metric distance between particles
- Neighbor counts using metric cut-off, including for polydisperse systems
- Metric-distance-cutoff and SANN versions of Steinhardt's BOOPs in 2d and 3d that bypass the Delaunay construction

### TODO
- Wait for a good implementation of 3d Delaunay in pure rust to implement clean Steinhardt's 3d BOOPs
- Add simple K-function and/or Fry plots functions
- Add an option to normalize g via the summands, g(r) = sum (bin / norm(bin)) instead of g(r) = sum(bin) / norm(r).
- Try to implement a kernel-based version of g to reduce binning issues
- Make a few functions there a bit more type-agnostic (par_iter permitting).
