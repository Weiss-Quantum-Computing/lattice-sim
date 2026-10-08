# Data and questions for the model

Ordered by how much each item changes the predictions. "Enters as" names the
constant or file in the code that the item replaces. Defaults are thesis
values or guesses and are flagged in `scripts/p3_echo_analysis.py`.

## A. Dominant or decisive

1. **Map-pulse fidelity.** For each transport pair: how the storage states |3,−1⟩ / |4,+1⟩ are mapped onto the transport states (which transitions, MW or RF, single pulse, AFP or multi-step), and the measured transfer efficiency of each step. *Enters as* `MAP_FIDELITY`. The default is 0.998 per step, with three steps for |m_F| = 3. This is currently the largest term in the budget.
2. **Scattering during the storage hold.** The lattice depths (all three axes) during the hold, the Raman/Rayleigh split from your Kramers–Heisenberg analysis, and any measured storage T₁/T₂ at hold depth. *Enters as* `SCATTER_PER_S` (default 0.1 s⁻¹, all assumed to destroy coherence). This is the second-largest term and grows linearly with hold time.
3. **EOM leg transient.** Rotation angle and ellipticity of the rotating arm for the second transport relative to the first, at several separations (e.g. 16.7, 33, 67, 133, 267 ms). The polarimeter crossing times per leg give the rotation; a quarter-wave plate before the analyzer (Stokes S3) gives the signed ellipticity at the 10⁻⁵ level and the handedness. *Enters as* `TRANSIENT` (defaults 0.5°, δη = 10⁻⁴, τ = 50 ms). This sets how long the hold must be.
4. **Your echo data.** Contrast and fringe phase against leg separation (e.g. the 16 ms and 133 ms spin-echo scans), with the state pair, ramp file, hold voltage and hold time used. This is the acceptance test for Phase 3 as a whole.

## B. Noise spectra (scope or spectrum-analyzer data)

5. **Magnetic field**, lock on: MAG-13 traces or spectrum, DC–2 kHz, in absolute units (µG/√Hz), ideally taken during echo runs; plus the line-trigger jitter and line-frequency wander. *Enters as* `FIELD_SHAPE`. At present the shape is a guess, scaled to Du's 0.054π.
6. **Lattice intensity**, X (and Y, Z): in-loop and out-of-loop relative intensity noise, DC–100 kHz, in dBc/Hz. Also a photodiode trace through a full sequence including the EOM ramps, to show intensity changes from one leg to the next (polarization-to-intensity conversion after the EOM). *Enters as* `RIN` and `LEG_INTENSITY_SIGMA` (default 10⁻⁴).
7. **Trek output**: ripple, and shot-to-shot reproducibility of V(t) near π/2 (rotation noise).

## C. Calibration and geometry

8. **Trap frequency at θ = π/2** for |4,−4⟩ and |3,−3⟩ (parametric modulation or sidebands). This gives the ellipticity handedness and the curvature the ramps depend on most.
9. **Operating depths** during transport and during the hold for each axis (is the x trap at 10 kHz right?); the ramps actually used on x, y and z (CSV); and the hold voltage and duration for the collision.
10. **Occupied region**: which sites and planes hold atoms, and any measured beam waists. *Enters as* `SITE_A` (default ±10 µm, 60 µm waist).
11. **Microwave and RF**: Rabi frequencies, pulse lengths, shot-to-shot amplitude stability, detuning errors, the minimum π-pulse spacing, and whether the pulse phase can be switched (needed for XY-8). *Enters as* `PULSE_EPS`, `PULSE_DET`.
12. **Du's |3,±1⟩ intensity-noise experiment**: the actual sequence (ramp time, park, leg spacing, depth) and the extinction ratio of the old EOMs against angle (Du Fig. 5.23). Needed to close the failed acceptance test (see findings).
13. **The y and z pairs**: beam geometry and the extinction-ratio profiles of their rotating arms. They are currently modelled by scaling x with depth.

## D. Collisional gate (Phase 4)

14. **Channel scattering lengths** a_F (F = 0, 2, 4, 6) for f = 3 + f = 3 at the bias field (1.39 G), from a coupled-channel calculation on the Cs potentials (e.g. the Hutson or Julienne groups), or any measured collisional phase for a known overlap. *Enters as* `dis_channel_lengths()` in `gates/collision.py` (the DIS estimate gives a_eff = 1600 a₀ for |3,±3⟩, against the 2800 a₀ used so far). This sets the stop voltage to within 60–150 V.
15. **Transverse trap frequencies** (y, z) at the x collision. *Enters as* `TRANSVERSE_HZ` in `scripts/p4_collision.py` (default 8.0 / 6.25 kHz). U scales as √(f_y f_z).
16. **Trek stop-voltage reproducibility** shot to shot, and settling at the stop point. The phase slope is 7–15 mrad/V.
17. **Collision sequence**: the ramp and stop (or hold) voltage and duration actually used, and the target phase (π/2 for the √ZZ cluster-state gate, or π).

## E. Alignment and sidebands

18. **Measured |4,−4⟩ → |3,−3⟩ spectra** (carrier and sidebands) for a few deliberate alignment changes: a known beam-centering offset, a crossing-angle change, the EOM rotation used for the sideband scan, and the lattice depth; with the measured temperature or n̄. *Enters as* the defaults of `scripts/sideband_explorer.py` (`nominal_site`) and validates the 1D-per-pair sideband model (`figures/sidebands_alignment.png`).
19. **x₀(θ) or f(θ) of |3,−3⟩ at θ ≈ 0.4π and 0.6π.** Decides between the calibration and the first-principles shape of the transport path, which changes the predicted excitation of ramps faster than 3 ms by 10–100× (`figures/model_vs_calibration.png`, `scripts/score_suite.py`).
