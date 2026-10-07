# lattice-sim

A first-principles simulation of the Weiss-group caesium 3D optical lattice
(843.222 nm, blue detuned): light field and polarization, magnetic field and
quantization axis, state-dependent trapping, spin-dependent transport,
collisional gates, and the noise sources acting on each. The aim is a
quantitative error budget for the controlled-collision gate and its echo
sequences, with figures suitable for publication.

## Design

Each layer is a separate module that can be tested on its own, and every
layer must reproduce a known result before the next one builds on it.

| module | scope |
|---|---|
| `atom` | 6S1/2 hyperfine-Zeeman structure (full 16 x 16 Hamiltonian and Breit-Rabi); scalar, vector and tensor polarizabilities at the lattice wavelength |
| `fields` | Lattice beams (geometry, waists, polarizations, EOM rotation including extinction ratio and thermal drift); total magnetic field = bias + fictitious field from the vector light shift; quantization axis |
| `motion` | State-resolved potentials and wave-packet dynamics: 1D along the transport axis first, then a full 3D grid with cross-axis coupling |
| `spin` | MW/RF pulses (storage-transport maps, pi/2, pi trains), pulse errors, Raman scattering as Lindblad terms |
| `gates` | Two-atom collisional phase from 3D overlap, echo sequences, entangling phase and fidelity |
| `noise` | Monte Carlo over B(t), I(t), theta(t) drawn from measured spectra; filter-function analytics as a cross-check |
| `viz` | Publication-style figures and an interactive explorer |

Stack: NumPy/SciPy core, so the group can read and maintain every solver.
The wave-function propagator is written so its arrays can move to JAX on a
GPU for the 3D grid. QuTiP is optional for the spin layer. Per-gate error
rates feed the Stim stabilizer simulation in `Weiss-Quantum-Computing/Cluster`.
Transport ramps come from `Weiss-Quantum-Computing/Transport-Optimizer`.

## Phases and acceptance tests

| phase | content | accepted when it reproduces |
|---|---|---|
| P0 | ground state, constants, parameters | magic field, RF splittings and field sensitivities (done, `tests/test_ground_state.py`) |
| P1 | light field and static potentials | the ER316 calibration files (trap position and ~10 kHz curvature vs rotation); measured full-depth trap frequencies 19 / 16 / 12.5 kHz. In progress: x pair done in 1D, ER = 316 confirmed from the theta = 0 split; mid-rotation residuals open (`figures/er316_comparison.png`) |
| P2 | 1D transport | Transport-Optimizer band P0 and its validator |
| P3 | single-atom coherence | measured T2*, echo T2, leg-asymmetry phase |
| P4 | two-atom gate | overlap-vs-voltage table (0.1 overlap at 4340 V, 0.5 at 4785 V); then predicts chi and its budget |
| P5 | full sequences | interleaved x/y/z transport with extended hold; error budget by noise source |

A standing question for P1-P3: transport pair |3,-1>/|3,+1> or |3,-3>/|3,+3>.
The comparison is a scan over the noise model, not a fixed assumption.

## Parameters

`params/apparatus.yaml` holds the apparatus parameters with a source and a
status (`current`, `thesis`, `computed`, `open`) for each entry. Open entries
are scanned rather than assumed.

## Run the tests

```
pip install -e ".[dev]"
pytest
```
