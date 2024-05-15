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
3. Look at the example .py files, tweak the (for now) hard-coded values to take care of the relevant computation, then run it like a usual python script
4. Voilà!

### Current functionalities
This is very much a WIP but the code already supports
- Radial g(r) and radial field correlations (for arbitrary scalar or vector fields) in 2d and 3d, either connected or non-connected, for either square periodic or free boundary conditions.
- Vector g(r) in 2d and 3d
- Steinhardt's BOOPs in 2d
- Our long-range orientational correlation in 2d

### TODO
- Carry over smarter multithreading for Delaunay quantities and Voronoi cell area statistics from Active Brawnians
- Carry over variance measurements from dedicated library
- Carry over Steinhardt's 3d BOOPs from hyperalg
- Add simple functions for nearest-neighbor statistics
- Add simple functions for k-nearest-neighbor statistics like in my wls files
- Carry over R-Tree construction from my pattern-matching code for metric neighbor statistics
- Add simple K-function and/or Fry plots functions?
- Add an option to normalize g via the summands.
- Try to implement a kernel-based version of g to reduce binning issues?
- Think of other useful functions?
