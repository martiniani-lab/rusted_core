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
3. Look at the example .py files, tweak the (for now) hard-coded values to take care of the relevant computation, then run it like a usual python script (This does not apply trivially to Apple Silicon for now -- maturin compiles but paths are messed up)
4. Voilà!

### Current functionalities
This is very much a WIP but the code already supports
- Radial g(r) and radial field correlations (for arbitrary scalar or vector fields) in 2d and 3d, either connected or non-connected, for either square periodic or free boundary conditions.
- Vector g(r) in 2d and 3d, with options to compute only up to a radial bound or to the p-th nearest metric neighbor
- Radial or vector statistics of the p-th nearest metric neighbor distances, relying on R-Trees for speed
- Steinhardt's BOOPs in 2d
- Gyromorphic correlation in 2d
- Voronoi quantities (nearest neighbor distance, Voronoi cell area, Voronoi number of neighbors) and option to compute quantities averaged over Voronoi neighbors in 2d
- Cluster tagging according to metric distance between particles
- Neighbor counts using metric cut-off, including for polydisperse systems

### TODO
- Carry over Steinhardt's 3d BOOPs from hyperalg
- Compute a few quantities from R-Tree
- Add simple K-function and/or Fry plots functions?
- Add an option to normalize g via the summands, g(r) = sum (bin / norm(bin)) instead of g(r) = sum(bin) / norm(r).
- Try to implement a kernel-based version of g to reduce binning issues?
- Think of other useful functions?
- Clean up front-end
- Refactor lib.rs with separate files and classes; possibly make a few functions there a bit more type-agnostic (par_iter permitting).
