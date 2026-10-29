# lattice-sim — notes for Claude Code

A first-principles simulation of the Weiss-group Cs 3D optical lattice
(843.222 nm, blue detuned) for the controlled-collision cluster-state
experiment. It covers light field, magnetic field, state-dependent
potentials, transport, spin echoes with noise, and collisional gates. Read `README.md` for the phase plan, `docs/findings.md` for what has
been established, and `docs/data_requests.md` for open inputs.

## Commands

```
python -m pip install -e ".[dev,viz]"   # once, in a venv (Windows, macOS, Linux)
python -m pytest -q                     # ~3.5 min (200-220 s on the lab PC, Oct 2026); must pass before every commit
python scripts/<name>.py [args]         # scripts chdir to the repo root; paths are repo-relative
```

Slow scripts (minutes): `build_model_tables.py`, `build_profile_tables.py`,
`build_harmonic_scenarios.py`, `design_robust_ramp.py`. Their outputs
(`data/*.npz`) are git-ignored caches; regenerate in this order when needed:
`build_model_tables.py` → `build_harmonic_scenarios.py pm3|pm1` →
`build_profile_tables.py` → analysis and figure scripts. Phase 4: `p4_collision.py` (~7 min) → `p4_overlap_check.py`, `p4_summary.py` → `figure_p4.py`.
`data/pair_physics_m{1,3}.npz` is rebuilt automatically on first use (~30 s), as are the
`data/chain_tracks_*.npz` and `data/suite_*.npz` caches; each records the inputs it was built from
(`lattice_sim.cache`) and is rebuilt when they change.

## Layout

- `src/lattice_sim/atom` — ground-state Breit–Rabi / 16×16 Hamiltonian; D1/D2 light-shift coefficients.
- `src/lattice_sim/fields` — one crossed lattice pair with a rotating arm (`lattice_pair.py`), measured extinction-ratio profiles (`extinction.py`), state potentials by exact diagonalization (`potential.py`).
- `src/lattice_sim/motion` — split-step 1D transport (`transport1d.py`), exact Gaussian reference (`gaussian_exact.py`), vectorized harmonic scenarios (`fast_gaussian.py`).
- `src/lattice_sim/control` — robust max-min ramp optimizer.
- `src/lattice_sim/spin` — pair physics vs θ, echo sequences with filter-function noise and leg transients (`echo.py`), MW pulse trains (`pulses.py`).
- `src/lattice_sim/gates` — two-atom contact interaction: Gaussian overlap, M = 0 spin space with F-channel projectors, DIS scattering lengths, phase and spin-exchange leakage along a ramp (`collision.py`).
- `src/lattice_sim/spectra` — microwave carrier and Δn = ±1 sidebands of |4,−4⟩ → |3,−3⟩ versus lattice alignment (beam centering, crossing angle, rotation, ellipticity, bias field), 1D per pair (`sidebands.py`); beam offsets and tilts of each arm and the resulting gradients across the lattice (`beams.py`, `scripts/figure_tilt_gradients.py`); `scripts/sideband_explorer.py` is the interactive window (Tk; model functions are tested in `tests/test_sideband_explorer.py`), `scripts/figure_sidebands.py` the sensitivity figure. `scripts/labplot.py` is a vendored copy of the lab's shared plot panel (Weiss-Quantum-Computing/lab-plot): never edit it here, refresh it with that repo's `tools/vendor.py`.
- `src/lattice_sim/spectra/microwave.py` — the DDS's square / Blackman / AFP pulses (per the dds-control firmware) and the frequency-scan signal they give; `spectra/grid.py` — the spectrum over a 9×9×9 block of sites (amplitude-node interpolation). Explorer extensions beside the explorer (listed in its `EXTENSIONS`): `scripts/explorer_microwave.py`, `explorer_angle.py` (energies vs angle, Wells energy reference), `explorer_grid.py` (site maps and the lab's alignment slices), `explorer_chain.py` (opens the chain window).
- `src/lattice_sim/gates/chain.py` — a chain of 2–7 atoms through the echoed collisional gate (the Phase 4 model per neighbour pair); `scripts/chain_explorer.py` is its window (standalone, or from the explorer); tests in `tests/test_chain.py`. `data/chain_tracks_*.npz` are git-ignored caches (~30 s each, rebuilt on first use).
- `src/lattice_sim/molasses` — the imaging / PGC molasses in the lattice: semiclassical Monte Carlo wave-function model (Cs F = 4 with F′ = 5, 4, 3 adiabatically eliminated, six Brewster beams lin⊥lin, depumping and repumping; numba kernels in `kernel.py`), validated against Dalibard & Cohen-Tannoudji (1989) and Salomon et al. (1990); `tables.py` interpolates the scans of `scripts/molasses_scan.py` (`data/molasses/*.jsonl`, committed, ~2 h to regenerate). `scripts/molasses_explorer.py` is its window (standalone, or from the explorer via `explorer_molasses.py`), `scripts/figure_molasses.py` the figures; tests in `tests/test_molasses.py`.
- `scripts/score_suite.py` — scores a folder of ILC target pairs (the ramp A/B suite format, X1/X2 volts on a 2 µs grid) with the calibration, the first-principles model and, with `--split`, the full tables; `scripts/figure_model_vs_calibration.py` shows where model and calibration disagree.
- `scripts/design_suite_shape_robust.py` — suite-format rise robust to calibration and model shapes (exact Gaussian minimax, DE); `export_suite_target.py` writes it as an ILC target pair; `validate_suite_rs.py` checks it split-step on the full tables; `ramps/suite/` holds the exported targets (RS25).
- `src/lattice_sim/cluster` — cluster-state model: fillings and bonds (`lattice.py`), the experiment sequence as a Clifford circuit with Pauli noise presets (`sequence.py`), stabilizers, measurement settings, entropy and sampled ⟨S_i⟩ (`stabilizers.py`); Stim is the reference (`pip install stim`, in the dev extras). `tools/cluster_engine.js` is the browser engine (CHP tableau + Pauli frames), checked against Stim under node in `tests/test_cluster.py`; `tools/cluster_explorer_template.html` + `scripts/cluster_build_explorer.py` → `tools/cluster_explorer.html` (committed; rebuild after editing the template, engine or presets). `scripts/figure_cluster.py` → `figures/cluster_stabilizers.png`.
- `src/lattice_sim/viz/style.py` — the figure style; use `apply()` in new figure scripts.
- `scripts/` — analyses and figures; `figures/` — committed outputs; `ramps/` — ramps in Transport-Optimizer CSV format for the experiment.
- `params/apparatus.yaml` — apparatus parameters, each with `source` and `status` (current / thesis / computed / open). It is the reference, but no code loads it: the values are literals in the modules (spacing 4.8374 µm in `fields/lattice_pair.py`, `gates/collision.py`, `gates/chain.py`, `spin/pair_physics.py`, `spectra/sidebands.py`, `spectra/grid.py`, `spectra/beams.py` and some scripts; waist 60 µm; full-depth trap frequencies 19 / 16 / 12.5 kHz in `sidebands.nominal_site`). Change a value in the yaml and in the code together; `tests/test_apparatus_constants.py` checks the main ones agree.
- Related repository: `Weiss-Quantum-Computing/supertime-analysis` (Supertime sequence files `sequences/2024-2026/*.dt4`, a standard-library reader `supertime_editor.model.Sequence`, run-log and VI notes). Use it for sequence timing and channel values (e.g. the PGC hold: `PGC_vapor_loading.dt4`, molasses 0.6 vs 5 for imaging, 1000 ms, full lattice).
- `data/reference/` — Transport-Optimizer calibration CSVs; `data/measured/` — lab measurements (e.g. ramp-polarimeter exports).

## Conventions

- Units: energies as frequencies E/h in Hz; fields in gauss (sensitivities in Hz/mG); positions in m; sequence times in µs; θ in rad (θ = π ↔ V_π).
- Sign convention (x pair): the EO (rotating) arm side is +x; |4,−4⟩ moves toward it, |3,−3⟩ away (measured). Model: `rotation_sense=+1`, B_q at +45° in x–y. The ellipticity handedness `h` is **unknown**; results that depend on it are given for both signs.
- Magic field |3,−1⟩↔|4,+1⟩: 1.39304 G (computed; theses quote 1.393 G).
- The lattice scale used so far is the 10 kHz x trap (quarter depth). y and z are currently approximations (x scaled by depth).
- Cross-platform: `pathlib` for paths, `encoding="utf-8"` (and `newline=""` for CSV) on every `open`, no shell-specific code in Python. CI runs the tests on Linux, macOS and Windows.

## Working rules

- Every quantitative claim must be backed by a test or a script in the repo. A phase is accepted only when it reproduces a known result (README table); report a failed acceptance test as failed rather than tuning parameters until it passes.
- Prefer two independent methods for anything new (e.g. split-step vs exact Gaussian, filter function vs Monte Carlo) and add the agreement as a test.
- Assumptions live as named module-level constants; each unknown is listed in `docs/data_requests.md` with the constant it replaces.
- Record results in `docs/findings.md` (dated sections, scientific style: concise, quantitative, with the file that backs each statement). Figures follow `viz/style.py` and journal conventions: labelled axes with units, panel letters, no dual y-axes for different quantities, colour-blind-safe palette.
- Commit messages: imperative subject, body with the key numbers.
