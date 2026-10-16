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
20. **Carrier and sideband-frequency gradients across the lattice** (per plane along z, per column along x) at the current alignment, and after a known change of one arm's pointing. Decides whether an observed carrier gradient is a field gradient or a differential arm offset × ellipticity (`figures/tilt_gradients.png`); the sideband-frequency gradient gives the common beam offset directly.

## F. Projection sideband cooling

21. **AFP Rabi frequency at the cooling powers** (updated 2026-10-15). The model now derives it from the lab: 0.1425 W (−2) and 0.075 W (−1) from the Coolscan manifest, and the DDS calibration "0.306 corresponds to 0.12 ms Blackman pulse for f4new transition" (`Powerscan.py:317`), giving 6.77 / 4.91 kHz peak with the Blackman amplitude area 0.42 T and Rabi ∝ √power. Still to confirm: (a) the conditions of the π-power calibration — lattice depth and rotation (θ = 0?), temperature, and whether it was set on the trapped carrier (assumed coupling 1, `CAL_IS_BARE_COUPLING`) and is still current (the same cooling powers appear in `DDScool.py`, last updated 2015, before the April 2022 power-supply change moved the π power from 0.318 to 0.306 W); (b) a direct measurement at the cooling power: a carrier Rabi flop or square/Blackman π time at 0.075 W and 0.1425 W, or a transfer-versus-power curve of the −1 and −2 AFPs on cold atoms. *Enters as* `CAL_POWER_W`, `CAL_BLACKMAN_PI_S`, `LAB_POWER_W` in `cooling/afp.py`. The lab angles are sensitive to it (lab sequence 0.945 / 0.962 / 0.971 at 0.5 / 1 / 2× the power), the optimized angles are not (0.984 / 0.987 / 0.987) (findings 2026-10-15).
22. **Initial temperature / n̄ per axis before cooling**: a sideband spectrum (Δn = ±1 ratio) right after PGC and OP, at full depth. *Enters as* `INITIAL_TEMPERATURE_K` (default 5 µK). This is the largest single lever on the predicted final fraction (lab powers: 96.2 % at 5 µK, 93.0 % at 6 µK).
23. **Ground fraction against cycle number** (e.g. after 10, 20, 30, 40, 50 cycles) and the post-cooling sidebands per axis for the lab sequence. This is the acceptance test of the cooling model; also one run with loop 1 at θ₋₂ = 6°.
24. **OP heating and darkness**: ground-state fraction after N extra OP pulses on cold atoms (dark-state scatter), and after one deliberate |4,−4⟩ → |3,−3⟩ → OP round trip on cold atoms (recoil heating per OP event). *Enters as* `DARK_EXCURSION_PROB` (default 0) in `cooling/optical_pumping.py`; the model gives 16 / 18 / 10 % per axis for one OP from n = 0.
25. **Reference frequency definition**: is "Reference fc" the carrier measured at θ = 0 (assumed), and were the −1 / −2 offsets set on cold or hot atoms? The carrier moves by −2.1 / −1.1 / −0.6 kHz (x / y / z) at 12°.
26. **OP beam direction** relative to the lattice axes (assumed (x̂ + ŷ)/√2), and the ER(θ) of the y and z rotating arms near 0–15°. *Enters as* `OP_BEAM_DIRECTION_LAB`, `EXTINCTION_RATIO` (2×10³ on all axes).
