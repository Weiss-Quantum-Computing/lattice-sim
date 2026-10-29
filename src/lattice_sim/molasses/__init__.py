"""Polarization-gradient cooling (optical molasses) of single Cs atoms in the
3D lattice: semiclassical Monte Carlo wave-function model.

Modules:
    levels   - Cs D2 F = 4 -> F' = 3, 4, 5 (and a J = 1/2 -> 3/2 test system):
               dipole matrices, branching, and the exact decomposition of the
               adiabatically eliminated light-shift operator into
               |E|^2, (E*.F)(E.F) and (E.F)(E*.F).
    beams    - molasses beam geometries (lab Brewster-angle lin-perp-lin pairs,
               orthogonal 3D lin-perp-lin, 1D lin-perp-lin).
    kernel   - numba kernels: free/trapped atom simulation, dragged-atom
               force statistics.
    model    - configuration, run, and analysis (temperature, scattering rate,
               energy relaxation, hopping estimate, field width).
"""
