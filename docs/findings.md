# Lattice Sim — findings log

Code: `Weiss-Quantum-Computing/lattice-sim`. Every entry below is backed by a test or a script in that repo. This file (`docs/findings.md`) is the source of the copy in the Lattice Sim project.

## 2026-10-07 — Phase 0–1: ground state and static potentials

**Magic field.** |3,−1⟩↔|4,+1⟩ is first-order field-insensitive at 1.39304 G. Breit–Rabi and full 16×16 diagonalization agree, as does the lowest-order closed form. The parameter reference's 1.3924 G is 0.6 mG low; the theses' 1.393 G is correct. The qubit frequency differs by < 1 mHz between the two values. (`tests/test_ground_state.py`)

**ER316 = rotating-arm extinction ratio 316:1 (25 dB).** The Transport-Optimizer calibration files `calibration_{F3,F4}_10kHz_ER316.csv` were generated with a 10 kHz trap (θ = 0, pure lattice) and a rotating arm of extinction ratio 316. Evidence: the first-principles x-pair model gives a θ = 0 trap-frequency split f(4,−3) − f(3,−3) of 101.2 Hz at ER = 316, against 101.1 Hz in the files. ER = 250 and 400 give 113.8 and 90.0 Hz. The split is zero for pure linear polarization. (`tests/test_lattice_potential.py`)

**Model vs calibration across the rotation.** The trap positions agree to within a few nm for θ < 0.35π and θ > 0.65π; the displacement of a/2 = 2.419 µm at θ = π is reproduced exactly. Near θ = π/2 the curves differ by up to 0.13 µm in position and 0.75 kHz in trap frequency. At θ = π/2 the model gives 3.1 kHz for |3,−3⟩ where the files give 3.8 kHz. Candidate explanations: a different treatment of the fictitious-field projection, or of the ellipticity axis, in whatever generated the files. Du's first-order projection (Eq. 3.8) fits worse than the exact diagonalization (44 nm vs 27 nm rms). The generating script would settle this, as would a measured trap frequency near π/2 (sideband spectroscopy). (`figures/er316_comparison.png`, `scripts/fit_er316_calibration.py`)

**Best-fit geometry for the files.** Quantization field at 45° in the x–y plane (rms error rises steeply away from 45–50°; 0° and 90° are excluded). One ellipticity handedness fits markedly better than the other (27 nm vs 91 nm rms).

**Differential light shift.** The F=4/F=3 scalar differential is −0.226 % of the depth at 843.222 nm, i.e. 10.8 Hz per vibrational level at 9.5 kHz and 21.5 Hz at 19 kHz. The same code gives 29.9 Hz per level at 847 nm and 15 kHz, reproducing Wu (2020, p. 25: 28–30 Hz).

**Vector-to-scalar ratio** at 843.222 nm: 0.374 (from D1/D2 line strengths with counter-rotating terms).

**Rotating arms (current setup, beams drawn as /\\):** x right, y left, z top.

## 2026-10-07 — Phase 2: 1D transport (x pair, 10 kHz, ER 316)

**Engines.** A split-step Fourier propagator on tabulated potentials, and an exact Gaussian (Heller) solution for harmonic traps. On the Transport-Optimizer seed ramp with the calibration potentials they agree to 1×10⁻⁶ in P0; refining the time step (0.25 → 0.1 µs) and the grid (1024 → 2048 points) changes nothing at that level. (`tests/test_transport1d.py`)

**The optimizer's propagator is not converged at its default grid.** Exact band-minimum P0 on the seed ramp: F3 0.99950, F4 0.99884. The optimizer's node method gives F3 0.99871 / 0.99929 / 0.99942 at N_X = 5001 / 10001 / 20001, converging towards the exact value. At the default N_X = 10001 the error, 2×10⁻⁴, exceeds the 8×10⁻⁵ improvement reported for K06 over the seed. The exact engine confirms that K06 does improve the band P0, by 0.8×10⁻⁴ for F3 and 1.2×10⁻⁴ for F4.

**On the model potentials, K06 leaves |3,−3⟩ 1 % excited.** Band-minimum P0 is 0.9877 (final 0.9911), against 0.99958 on the calibration. |3,+3⟩ gives 0.99880 and |4,−3⟩ 0.99880. Replacing the model by its own harmonic approximation gives 0.9893. The loss therefore comes from the harmonic parameters, not from anharmonicity: the model's trap frequency at π/2 is 3.06 kHz, against 3.81 kHz in the calibration. The ramp relies on cancelling mid-path excitation in its second half, which makes it sensitive to the π/2 trap frequency. A measured trap frequency near π/2 is the most valuable single calibration input. (`figures/transport_p2.png` b)

**Extinction ratio and the |3,±1⟩ pair.** Ellipticity in the rotating arm produces a scalar lattice in quadrature with the original one. Near θ = π/2 its wells coincide with the transport sites, so it deepens one partner's trap and shallows the other's. At θ = π/2 the trap depths are:

- ER = ∞: 279 kHz for |3,±3⟩ and 93 kHz for |3,±1⟩.
- ER = 316: 364 / 188 kHz for |3,±3⟩ and 180 / 5.9 kHz for |3,±1⟩.

At ER = 316 the shallow |3,±1⟩ partner is essentially untrapped along x. Propagated through K06 it ends with P0 < 10⁻⁴, spread over both neighbouring sites. The shallow ±1 partner recovers half its ideal depth at ER ≈ 1.3×10³ and 80 % at ER ≈ 10⁴. Which partner is shallow depends on the ellipticity handedness. This may contribute to Du's observed loss of |3,±1⟩ contrast near and beyond π/2 (Du p. 102–103), which was attributed to intensity noise. (`scripts/depth_vs_extinction.py`, `figures/transport_p2.png` d)

**|3,±1⟩ needs its own ramp.** Its displacement is concentrated in a narrower window around π/2 than that of |3,±3⟩ (`figures/transport_p2.png` c). Even at ER = ∞, K06 leaves the shallow ±1 partner with final P0 = 0.43; at ER = 10⁴, 3×10³ and 10³ the values are 0.12, 0.015 and 2×10⁻⁵. K06 was designed on the |m_F| = 3 calibration and is not suited to |m_F| = 1.

## 2026-10-07 — Sign convention and robust ramps

**Transport direction (measured).** On x, |4,−4⟩ moves toward the EO arm (the arm closest to the optical-pumping beam) and |3,−3⟩ moves away from it. This matches the model with the EO-arm side as +x, and matches the sign convention of the ER316 calibration files. The observation fixes the product of rotation sense and quantization-field sign, which is all the motion depends on, so the unknown field polarity does not matter for transport.

**Remaining unknown: the handedness of the ellipticity.** It decides which partner of each pair is deepened at π/2. The trap-frequency ratio f(|4,−4⟩)/f(|3,−3⟩) with the lattice parked at θ = π/2 discriminates it: 1.86 (h = −1) or 1.01 (h = +1) at ER = 316, against 1.33 for pure linear polarization. The ratio is independent of the absolute depth calibration. (`figures/robust_ramps.png` d)

**Robust ramp R3 for |3,±3⟩ (5 ms).** It maximizes the worst stop-anywhere P0 over 14 harmonic scenarios: both states, ER ∈ {316, 10³, ∞}, both handedness values, and ±5 % trap-frequency error. The worst case is 1 − P0 = 1.6×10⁻³. K06 over the same scenarios gives 1.4×10⁻²; it is better (4–9×10⁻⁴) everywhere except |3,−3⟩ at ER 316, h = −1. Validation on the full anharmonic potentials at ER 316, h = −1: |3,−3⟩ 0.99898, |3,+3⟩ 0.99854. On the original calibration curves R3 gives 0.99901. (`ramps/R3_5ms_robust.csv`, Transport-Optimizer CSV format)

**Robust ramp R1 for |3,±1⟩ (ER ≥ 10⁴).** The worst case over 14 scenarios is 1.9×10⁻³ at 5 ms and 6.6×10⁻⁴ at 8 ms. On the full potentials at ER 10⁴, h = −1: 0.99898 / 0.99889 at 5 ms and 0.99969 / 0.99939 at 8 ms. R1 dwells near θ ≈ 0.5π, where the |m_F| = 1 displacement is concentrated (`figures/robust_ramps.png` a).

**Engine.** A vectorized exact-Gaussian propagator with piecewise-exact steps agrees with the adaptive ODE solver to 5×10⁻⁷ and scores a 5 ms ramp in 0.3 s for any number of scenarios. (`lattice_sim/motion/fast_gaussian.py`, `lattice_sim/control/ramp_optimizer.py`)

## 2026-10-07 — Measured extinction ratio (scan 133-ms-SE-offset-neg2V-fine-2, x)

**Profile.** From the ramp polarimeter's dip fits (`data/measured/…_extinction.csv`): ER > 3×10⁴ within ±15° of θ = 90°, where the values are noise-limited lower bounds; 1.0–1.4×10³ at 135–150°, about 25 % lower on the ramp back than on the ramp out; 1.5–3.5×10³ at rest and at the hold. Legs 1 and 2, which are 133 ms apart, agree within the error bars at most angles. The measurement gives |ellipticity| only, not its handedness. (`figures/extinction_echo.png` a)

**Transport with the measured profile** (full anharmonic potentials, x, 10 kHz scale):

- |3,±3⟩, h = −1: K06 gives 0.99926 / 0.99864 and R3 gives 0.99889 / 0.99848. With the new EOMs, K06 is adequate.
- |3,±1⟩ is no longer excluded, since its shallow partner keeps most of its depth at π/2. R1 at 8 ms gives 0.99968 / 0.99909 (h = −1) and 0.99886 / 0.99953 (h = +1). R1 at 5 ms is worse, with a worst case of 0.99680, because its design assumed ER ≥ 10⁴ at every angle and the measured ~10³ at 135–150° lies outside its scenarios.

**Echo phase from the light shift.** The transport partners |3,±m⟩ accumulate a differential light-shift phase during each transport, peaked within ±20° of θ = π/2 (`figures/extinction_echo.png` b–d). Along the measured 9 ms out-hold-back trajectory at the 10 kHz scale:

| Pair | Ellipticity-independent ("drag") | Ellipticity contribution (upper bound) |
|---|---|---|
| |3,±3⟩ | 206 rad | ≤ 147 rad |
| |3,±1⟩ | 35 rad | ≤ 82 rad |

The upper bounds follow from the ER lower bounds near π/2. The echo cancels these phases only to the extent that the two legs are identical. Keeping the leg-to-leg residual below 0.1 rad requires:

- the differential shift (intensity, or rotation timing near π/2) to match between legs to about 3×10⁻⁴ for |3,±3⟩, or 9×10⁻⁴ for |3,±1⟩;
- the ellipticity near π/2 to match to |δη| < 5.5×10⁻⁶ for |3,±3⟩, or 8.4×10⁻⁶ for |3,±1⟩.

The differential shift per unit ellipticity peaks at π/2 at 1.27 MHz (|3,±3⟩) and 1.45 MHz (|3,±1⟩), so the two pairs are equally exposed to ellipticity changes; |3,±1⟩ has a six times smaller drag term. (`scripts/echo_ellipticity_sensitivity.py`)

**Leg-to-leg ellipticity in this scan.** At the angles where both legs have resolved ER values (|θ − 90°| > 37.5°), the measured differences imply an echo phase of −0.7 ± 0.6 rad (|3,±3⟩) and −0.25 ± 0.22 rad (|3,±1⟩). Both are consistent with zero, but not with 0.1 rad. The crossed-analyzer method measures η², so it cannot reach δη ~ 10⁻⁵ near π/2, where 80–90 % of the sensitivity lies. A signed, linear ellipticity measurement would: a quarter-wave plate plus balanced detection, i.e. Stokes S3. It would also determine the handedness.

## 2026-10-07 — Phase 3: echo sequences and noise (single atom)

**Engine.** A sequence is a list of π pulses, storage holds and transports. Each noise channel enters as a coefficient times the toggling sign (field, intensity, ellipticity, rotation). Random noise is handled with filter functions; the Monte Carlo agrees to 1.5 %. Leg-to-leg changes are deterministic transients tracked per EOM, and the beam envelope makes them site-dependent. Pulse trains are full SU(2) with amplitude and detuning errors. Line-synchronous placement is built in: 60 Hz harmonics cancel by 10⁵ when the transports start whole line periods apart. (`lattice_sim/spin/echo.py`, `pulses.py`, `pair_physics.py`; `tests/test_echo.py`)

**Field-noise calibration.** An assumed lock-on spectrum shape is scaled to reproduce Du's stretched-state echo, 0.054π total of which 0.046π is field noise after removing the atom-number floor. The transport pairs' differential Zeeman sensitivities are 2.11 kHz/mG (|3,±3⟩) and 0.70 kHz/mG (|3,±1⟩).

**Du's |3,±1⟩ contrast loss — acceptance test not passed.**

- White intensity noise at Du's measured −120 dBc/Hz gives contrast ≥ 0.999 in every case, so it cannot explain the loss.
- Fitting a leg-to-leg change to the complete loss after 2.5 ms parked at π/2 requires either intensity σ = 1.1×10⁻³ or ellipticity σ_η = 5.4×10⁻⁵ (0.14 % of η at ER 625).
- With the assumed timing (1.5 ms ramps, 1/60 s legs, ER 625), neither mechanism reproduces the angle scan: the model gives 0.99 at 0.45π and 0.95 at 0.65π, against a measured 0.65 and < 0.1.

Du's actual sequence and old-EOM ER(θ) are needed (data request 12).

**Predictions** (x pair, measured ER profile, ramps R3 for |3,±3⟩ and R1 at 8 ms for |3,±1⟩, a single central π pulse below 50 ms separation and an XY-8 train of 9 pulses at 50 ms and above, all defaults in `docs/data_requests.md`). Contrast at 150 ms separation:

| Pair | x only | x-y-z interleaved |
|---|---|---|
| |3,±3⟩ | 0.95 | 0.89 |
| |3,±1⟩ | 0.97 | 0.95 |

The budget is dominated by the assumed map-pulse fidelity and hold scattering. Field noise comes next (2 % for |3,±3⟩ x-y-z). Leg intensity drift at 10⁻⁴ costs 0.3 % (|3,±3⟩) and 0.1 % (|3,±1⟩). White intensity noise, the π train (XY-8) and motional excitation are each ≤ 0.1 %. (`figures/p3_echo.png` b)

**EOM transient and hold length.** The residual echo phase from the transient decays as e^(−separation/τ). For 0.5° rotation and δη = 10⁻⁴ at τ = 50 ms, |φ| < 0.1 rad requires about 90 ms start-to-start for |3,±3⟩ and about 175 ms for |3,±1⟩. The |3,±1⟩ pair is more sensitive because its differential shift changes more steeply with θ. The phase is deterministic, so it can be calibrated out if it is stable; its dephasing through the beam envelope is ≤ 0.3 %. (`figures/p3_echo.png` a)

**π trains.** With 1 % amplitude and detuning errors, XY-8 stays at 1−C ≈ 1×10⁻³ for any input state up to 25 pulses. CP and CPMG reach 4 % at 9 pulses for the worst input state. (`figures/p3_echo.png` c)

**Timing constraint.** Interleaved x-y-z transports do not fit in one line period. The minimum start-to-start separation is 33.3 ms for |3,±3⟩ (block 31.8 ms) and 50 ms for |3,±1⟩ (block 49.8 ms).

**Limits of this phase.** The y and z pairs are x scaled by depth. Map pulses are instantaneous. Motional dephasing is approximated by the product of P0. No collision is modelled (Phase 4).

## 2026-10-07 — Phase 4: controlled collisions on the x axis

**Model.** The left atom in |3,+m⟩ and the right atom in |3,−m⟩ approach as the rotating arm turns; their separation is d(θ) = a − |x₀(+m)| − |x₀(−m)| from the first-principles potentials with the measured extinction profile. The contact energy is U/h = 2ħ a O / m_Cs, with O the Gaussian density overlap (x from the two tracked trap frequencies; y, z at the assumed `TRANSVERSE_HZ` = 8.0 / 6.25 kHz). The spin space is the seven states |L:μ, R:−μ⟩ (total M = 0), coupled by the scattering-length operator Σ_F a_F P_F; each state carries its Zeeman energy at 1.393 G and the lattice energy of both atoms at the current packet positions (sudden approximation). The phase and the leakage out of |L:+m, R:−m⟩ follow from integrating this 7×7 Hamiltonian along the ramp. (`lattice_sim/gates/collision.py`, `tests/test_collision.py`)

**Acceptance test — passed.** Fed with Transport-Optimizer's ER316 calibration, the overlap engine reproduces `tools/overlap_vs_voltage.py`: relative overlap 0.1 / 0.5 / 0.9 at 4339 / 4785 / 5155 V, against 4340 / 4785 / 5155 V; and U/h = 9.478 kHz at full overlap for 2800 a₀ and 7 kHz transverse, against 9.48 kHz. The Gaussian overlap agrees with quadrature to 10⁻⁶, and with the numerically diagonalized site eigenstates to within 0.9 % at every stop voltage used below. (`scripts/p4_collision.py`, `scripts/p4_overlap_check.py`; `figures/p4_collision.png` f)

**Scattering lengths.** The degenerate-internal-states estimate, using the Cs singlet and triplet lengths a_S = 280.37 a₀ and a_T = 2440 a₀ (Chin et al. 2004), gives a_F = 1495 / 1596 / 1833 / 2204 a₀ for F = 0 / 2 / 4 / 6 in f = 3 + f = 3. The diagonal (effective) length is 1596 a₀ for |3,±3⟩ and 1866 a₀ for |3,±1⟩, 57 % and 67 % of the 2800 a₀ used so far. The operator is not diagonal in m_F, so it couples |L:+m, R:−m⟩ to the other M = 0 pairs. DIS neglects the Feshbach structure of Cs near 1.4 G and is an estimate only; both it and a uniform 2800 a₀ are carried through.

**Energies at the hold voltages** (h = −1; h = +1 differs by < 1 %):

| Pair | V (V) | d (nm) | U/h, 2800 a₀ / DIS (kHz) | smallest gap (kHz) |
|---|---|---|---|---|
| |3,±3⟩ | 4340 | 195 | 0.94 / 0.54 | 6.6 |
| |3,±3⟩ | 4785 | 103 | 4.78 / 2.72 | 1.9 |
| |3,±3⟩ | 5155 | 40 | 8.63 / 4.92 | 0.08 |
| |3,±1⟩ | 4340 | 65 | 7.07 / 4.71 | 2.1 |
| |3,±1⟩ | 4785 | 34 | 8.75 / 5.83 | 0.6 |
| |3,±1⟩ | 5155 | 13 | 9.45 / 6.30 | 0.04 |

The |3,±1⟩ partners come together about 1000 V earlier in the rotation than the |3,±3⟩ partners (`figures/p4_collision.png` a). The spin-exchange channels are detuned by the differential lattice energy of the exchanged pair, which vanishes as both atoms approach the same site; for θ → π the gaps fall below U, and spin exchange is then limited only by the true a_F differences (panels b, c).

**Gate design: ramp out and back, no hold.** With the robust ramps (R3 for |3,±3⟩, R1 at 8 ms for |3,±1⟩) run out to a stop voltage and straight back, the ramps alone supply the collisional phase:

| Pair | φ | Stop voltage, 2800 a₀ / DIS (V) | dφ/dV (mrad/V) | Leakage, 2800 a₀ / DIS | Round-trip P0 (+m, −m), DIS |
|---|---|---|---|---|---|
| |3,±3⟩ | π/2 | 4301 / 4429 | 7.8 / 6.5 | 1.5×10⁻⁶ / 2.8×10⁻⁵ | 0.9947, 0.9995 |
| |3,±3⟩ | π | 4463 / 4612 | 12.6 / 11.6 | 3.3×10⁻⁵ / 3.7×10⁻⁴ | 0.9940, 0.9998 |
| |3,±1⟩ | π/2 | 3405 / 3467 | 11.6 / 9.3 | 3.7×10⁻⁵ / 5.1×10⁻⁵ | 0.9994, 0.9911 |
| |3,±1⟩ | π | 3525 / 3611 | 14.9 / 13.3 | 1.4×10⁻⁴ / 1.7×10⁻⁴ | 0.9984, 0.9983 |

The handedness changes the stop voltages by ≤ 4 V. (`scripts/p4_collision.py`, `scripts/p4_summary.py`; `figures/p4_collision.png` d, e)

Consequences:

- **Phase stability.** A 0.1 rad phase error corresponds to 7–15 V of stop voltage, so the Trek stop point must reproduce to ≲ 1 V for 10 mrad.
- **Scattering length.** The uncertainty in a_eff (1600–2800 a₀) moves the stop point by 130–150 V for |3,±3⟩ and 60–90 V for |3,±1⟩. The stop point must therefore be calibrated on the atoms; the model fixes only the range.
- **Leakage** stays ≤ 4×10⁻⁴ at these stop points, because the atoms do not reach the region where the gaps close. Ramps pushed to full overlap would leak about 10⁻² (|3,±3⟩) and 4×10⁻² (|3,±1⟩) under DIS.
- **Motion.** The |3,±1⟩ stop points (3400–3600 V) lie below the 3800 V stop-anywhere band, and truncating a ramp designed for full transport stops the packets abruptly. The round-trip P0 is 0.991–0.9998 across both pairs and both targets. A dedicated collision ramp (out to the stop voltage with zero end velocity, then back) is needed for P0 > 0.999.

**Limits of this phase.** Only the x pair. The transverse frequencies are assumed. DIS replaces the coupled-channel a_F. The sudden L/R basis fails near full overlap, which the stop points above avoid. Motion and spin are computed separately: P0 from single-atom transport, the phase from the Gaussian overlap.

## 2026-10-08 — Ramp suite 2026-09-28 scored; model vs calibration shape

**Scoring the suite.** `scripts/score_suite.py` reads the suite's ILC target pairs (θ = 90° (V₁/V90₁ + V₂/V90₂), K0 from the header) and scores the |4,−3⟩ / |3,−3⟩ pair over the suite's trap-frequency scales ×0.80–1.15. On the ER316 calibration (the suite's own model) the exact-Gaussian engine reproduces the suite's predictions: S25 1.1×10⁻⁴, N15 3×10⁻⁵ (suite 1.4×10⁻⁵, classical phonons), G180 3.5×10⁻⁵, HSER 1.3×10⁻⁶, C25 0.10, D8N25 4×10⁻⁵. (`<suite>/lattice_sim_scores.json`)

**The first-principles model excites |3,−3⟩ 10–100× more on the fast ramps.** With the measured extinction profile (h = −1), 1 − P0 at the end of the rise is 2.6×10⁻² for S25, 2.4×10⁻² for N15, 1.3×10⁻² for N175, 1.5×10⁻² for G180 and 5×10⁻⁴ for the 4.7 ms S47/HSER family; the SF/SW stop sets sit at 4×10⁻³ (h = −1) and 1.5×10⁻³ (h = +1). |4,−3⟩ agrees with the calibration in every case. The same holds for pure linear polarization (S25: 5.8×10⁻³ for |3,−3⟩, 5×10⁻⁵ for |4,−3⟩), so it is not the piecewise-linear profile. Split-step on the full measured-profile tables gives the same picture (S25 end: 2.2×10⁻² / 5.8×10⁻⁴).

**Where model and calibration differ** (`figures/model_vs_calibration.png`, `scripts/figure_model_vs_calibration.py`). Pure-linear and measured-profile models match the π/2 trap frequency of the calibration (3.83–3.96 kHz against 3.81 for |3,−3⟩; 4.43–4.54 against 4.57 for |4,−3⟩) but differ by a smooth dipole of ±50–80 nm in x₀ and ±300–500 Hz in f on either side of π/2, with opposite signs for the two states. The ER 316 model matches away from π/2 and misses the π/2 frequency by 0.75 kHz (the 2026-10-07 finding). The calibration thus behaves like ER 316 away from π/2 and like pure linear light at π/2, which a generator that applied the ellipticity to the vector shift but not to the quadrature scalar lattice would produce. The suite's robustness band is a uniform frequency scale and does not cover a shape change; ramps faster than ~3 ms are sensitive to it. The suite's A/B tests (fast N15/N175/S25 against slow S47/HSER, |3,−3⟩ against |4,−3⟩) discriminate the two shapes; a measured x₀(θ) or f(θ) at θ ≈ 0.4π and 0.6π for |3,−3⟩ would settle it directly. Until then, ramps should be designed robust over both shapes (both scenario sets in `control/ramp_optimizer.py`).

**Light-shift phase per record** of the |4,−3⟩ / |3,−3⟩ pair (drag plus measured ellipticity, K0 = 0.94): −150 rad (N15) to −370 rad (S47), proportional to the time spent near π/2; the suite's relative numbers scale the same way.

## 2026-10-08 — Microwave carrier and sidebands versus alignment

**Model** (`lattice_sim/spectra/sidebands.py`, `tests/test_sidebands.py`). The |4,−4⟩ → |3,−3⟩ spectrum has motional sidebands only because the two states see different potentials. Each pair is a 1D `LatticePair` with its own depth, spacing, rotation, extinction ratio and per-arm amplitude (beam centering: exp(−r²/w²), w = 60 µm), the bias field projected into the pair's frame; the vibrational ladders of both states come from the exact 16×16 potentials, lines are weighted by thermal population × |⟨n′|n⟩|², and the carrier is the differential n = 0 energy (Zeeman + scalar + vector light shift + zero point) summed over the three pairs. Tests: harmonic ladder, displaced-oscillator Franck–Condon factor, zero-point hyperfine shift (−0.226 % of Σ hf/4), centering lowers f by exp(−r²/w²)^{1/2}.

**Nominal configuration** (full depth 19/16/12.5 kHz, 3 % rotation on every pair, ER 3×10⁴, 1.393 G at 45°, 1 µK): carrier −0.9 kHz from the free atom, wells displaced by 30–37 nm per axis, Δn = +1 sidebands at −f(|3,−3⟩) and Δn = −1 at +f(|4,−4⟩) (the two trap frequencies differ by 10–40 Hz). Sideband-to-carrier ratio 0.19 (x, Δn = +1) at 1 µK. Sensitivities (`figures/sidebands_alignment.png`): a 20 µm centering offset of both x arms lowers f_x by 10 % (the rotating arm alone: 5 %) and the x sideband ratio by only 3 %; 0.1° of crossing half-angle moves f_x by 2 % at fixed intensity (see the correction below). The carrier sits at −500 Hz with no rotation, −900 Hz at 3 % and −5 kHz at 10 %; at ER 10² it moves to −1.8 kHz (h = −1) or 0 (h = +1), and it moves by 550 Hz between a 0° and a 90° bias-field angle; the Zeeman term adds −2.45 kHz per mG. The sideband ratio is set by the rotation (0 → 0.19 → ~1 from 0 to 3 to 10 %), the temperature, and the bias-field angle: the differential displacement is first order in the fictitious field only through its projection on the bias field, and the x pair's fictitious field lies along its beams (lab y), so the x sideband ratio goes from 0 at a 0° field angle to 0.44 at 90° (0.19 at the nominal 45°). The relative sideband strengths of the three pairs therefore measure the field angle directly. Not yet compared with measured spectra (data request 18).

## 2026-10-08 — Beam offsets, tilts and carrier gradients across the lattice

**Correction.** The crossing-angle sensitivity quoted above (1 Hz per 0.1°) was computed at fixed trap frequency, the depth-calibration convention of `StatePotential`. At fixed intensity the trap frequency scales as 1/spacing: 0.1° of half-angle (1.7 mrad) changes f_x by 2 %, 1 mrad of differential in-plane tilt by 1.1 % (216 Hz at 19 kHz), uniformly across the lattice and without moving the carrier. `with_alignment` and the explorer now do this.

**Geometry** (`lattice_sim/spectra/beams.py`, `scripts/figure_tilt_gradients.py`, `figures/tilt_gradients.png`). Each arm has an offset at the atoms, an in-plane and a vertical tilt, a waist and a pivot distance (a tilt about a pivot L upstream also moves the beam by L × tilt). The spectrum is evaluated at sites along x (fringes) and z (imaging axis) over ±20 µm, y and z pairs nominal, and the gradients are in Hz per µm:

| x-pair change | carrier at centre (Hz) | f_x (kHz) | dcarrier/dz | df_x/dz | dcarrier/dx | df_x/dx |
|---|---|---|---|---|---|---|
| nominal | −903 | 18.95 | 0 | 0 | 0 | 0 |
| both arms +10 µm in z | −882 | 18.43 | −3.4 | +94 | 0 | 0 |
| both arms +10 µm in x | −882 | 18.44 | 0 | 0 | −3.4 | +93 |
| rotating arm alone +10 µm in z | −1077 | 18.69 | +28 | +47 | 0 | 0 |
| arms ±10 µm in z (differential) | −882 | 18.43 | −60 | 0 | 0 | 0 |
| differential vertical tilt 1 mrad, pivot at the atoms | −903 | 18.95 | 0 | 0 | 0 | 0 |
| common vertical tilt 1 mrad, pivot at the atoms | −903 | 18.95 | 0 | 0 | 0 | 0 |
| differential in-plane tilt 1 mrad (crossing), pivot at the atoms | −902 | 19.17 | 0 | 0 | 0 | 0 |
| bias-field gradient 0.1 G/cm (reference) | | | −24.5 | 0 | | |

- **Pure wave-vector tilts do not show in the spectrum.** A differential vertical tilt of 1 mrad shears the lattice (fringe normal tilted by δ/sin α = 0.66°) and changes nothing else; a common tilt changes nothing; a differential in-plane tilt only rescales f_x. Tilts matter through where they put each beam at the atoms (L × tilt, 300 µm per mrad for a 30 cm pivot), so tilt and position are the same observable in the spectrum and can only be separated geometrically (walking the beam about a pivot at the atoms).
- **A common offset gives a depth gradient**: df_x/f_x = −2 r₀/w² per unit length (0.56 % per µm at r₀ = 10 µm, w = 60 µm; the depth changes by 4 r₀/w² — corrected 2026-10-15, this read −1 %) along the offset direction, with a carrier gradient of −3.4 Hz/µm, i.e. −0.036 Hz of carrier per Hz of sideband. The sideband-frequency gradient measures the offset: r₀ = −(w²/2) d ln f/dz.
- **A differential offset of the two arms gives a carrier gradient with no sideband gradient**, the same signature as a magnetic-field gradient. The mechanism is the ellipticity: the fictitious field at the trap minimum is ∝ η B (A − B) for arm amplitudes A, B, lies along the beams and is first order in the imbalance, so it moves the |4,−4⟩ → |3,−3⟩ carrier through its projection on the bias field. Per µm of relative arm offset: 3.4 Hz/µm at ER 3×10⁴, 18 Hz/µm at ER 10³, independent of the rotation (0, 3, 6 % give 34, 32, 26 Hz/µm for a 10 µm single-arm offset), odd in which arm is offset, along the direction of the relative offset. One µm of relative misalignment at ER 3×10⁴ mimics a 14 mG/cm field gradient. The same differential offset also costs contrast at the lattice edge (residual intensity (A − B)²).
- **Decoupling recipe.** Measure the carrier and sideband-frequency gradients along x and z. A sideband gradient with a −0.036× carrier gradient is a common offset (fix by centering; the sideband gradient gives the offset). A carrier gradient without a sideband gradient is either a field gradient or a differential arm offset; halving the lattice intensity halves the light-induced part and leaves the field part, changing the extinction ratio changes only the light-induced part, and the light-induced part lies along the relative beam offset while the field gradient has a fixed direction. The explorer reports both gradients for the chosen arm offsets.

## 2026-10-13 — Projection sideband cooling versus EO rotation

*Correction 2026-10-15: this section assumed one Rabi frequency Ω₀/2π = 20 kHz for both AFP pulses. The lab drives them at 0.1425 W (−2) and 0.075 W (−1), i.e. 6.77 / 4.91 kHz; the numbers at the lab powers are in the 2026-10-15 section below. The values here remain correct for Ω₀ = 20 kHz (the rebuilt tables reproduce them to the quoted digits) and are kept as the sensitivity scan.*

**Sequence modelled (lab, 2026-10-13).** Per axis and step: −2 AFP at θ₋₂, centred 2f above the θ = 0 carrier (sweep ±6 kHz); −1 AFP at θ₋₁, centred f above it (±4 kHz in loop 1, ±6 kHz in loop 2); optical pumping (OP) at θ = 0. Loop 1: 30 cycles at (θ₋₂, θ₋₁) = (12°, 6°); loop 2: 10 cycles at (13°, 12°). AFP 3 ms. Programmed −1 offsets 18.1 / 15.1 / 12.5 kHz (x / y / z) set the full-depth trap frequencies. Initial state thermal at 5 µK (Wu p.26), n̄ = 5.3 / 6.4 / 7.7.

**Model** (`lattice_sim/cooling/`, `tests/test_cooling.py`). Per axis, the |4,−4⟩ and |3,−3⟩ ladders (50 levels) and their Franck–Condon (FC) overlaps come from the exact potentials (ER 2×10³, h = −1, 1.393 G at 45°). Each AFP (Wang Eq. 2.2, 8 µs DDS steps) is propagated on the full two-ladder Hamiltonian, so off-resonant carrier and sideband couplings, anharmonicity and the carrier shift with θ are included. The two pulses combine incoherently; ramps are adiabatic. OP: σ⁻ Markov chain with exact 6P₃/₂ F′ = 4 branching (15/7 = 2.14 photons from |3,−3⟩); each photon is absorbed along the beam ((x̂ + ŷ)/√2, assumed) and emitted with the π/σ dipole pattern; recoil matrices on the lattice eigenstates. The 3D population is propagated exactly (`simulate_3d`) and in a factorized form (`simulate_factorized`); they agree to 5×10⁻⁵ on the lab sequence (0.97903 vs 0.97908). Tests: piecewise AFP vs adaptive ODE (≤ 2×10⁻⁴), branching ratios (7/12, 7/15), mean photon number, kick matrices vs the analytic displaced-oscillator formula, Lamb–Dicke limit, 3D vs factorized.

**FC factors are far from uniform in n.** |⟨n−k|n⟩|² of the displaced wells (34 nm at 6°, 69 nm at 12° on x) has Laguerre nodes. On x, Δn = −2 at 12° vanishes near n = 11–13; Δn = −1 at 6° near n ≈ 24; at 12° Δn = −1 vanishes near n ≈ 6. At 6° the low-n Δn = −2 factors are small (0.009 for n = 2).

**The AFP saturates almost everywhere.** A two-level AFP (3 ms, ±4 kHz) reaches 0.98 transfer at |A| = 0.07 for Ω₀/2π = 20 kHz and at |A| = 0.1 for 14 kHz, the latter reproducing Wang's criterion. Consequently, at the lab angles all levels transfer with P > 0.99 except within ±1–2 levels of an FC node (`figures/psc_cooling.png` c). The transitions are therefore driven equally well at a given rotation except near the nodes, and the relevant question is where the nodes sit relative to the population. A level where the −2 pulse fails is rescued by the −1 pulse acting as Δn = −1, at the cost of one OP event. The per-level step outcome is in panel d. Above Ω₀ ≈ 30 kHz the carrier coupling (Ω₀²|A_nn|²/4Δ ≈ 7 kHz at 40 kHz) shifts the lines out of the sweep, and cooling becomes worse (panel i).

**Optical pumping sets the cost.** One OP event from n = 0 leaves the ground state with probability 0.16 / 0.18 / 0.10 (x / y / z) and 0.39 in 3D. The n = 1 → 0 step always requires OP. This is larger than the ~7 % per motion-step OP heating quoted by Wu (p.47) and Kumar (p.67). The definition of that number (axes, population) is not stated, so this comparison is **open**, not passed.

**Predictions, 5 µK, 40 cycles** (factorized model; `scripts/psc_analysis.py`, `data/psc_results.json`):

| Ω₀/2π (kHz) | lab (12/6°, 13/12°) | loop 1 only ×40 | loop 2 only ×40 | best, same angle on all axes | best, per axis |
|---|---|---|---|---|---|
| 10 | 0.9728 | 0.9711 | 0.9570 | 0.9868 at (6/6°, 11/11°) | 0.9890 |
| 14 | 0.9767 | 0.9754 | 0.9651 | 0.9873 at (6/6°, 11/11°) | 0.9894 |
| 20 | 0.9791 | 0.9781 | 0.9695 | 0.9879 at (6/6°, 9/18°) | 0.9894 |
| 40 | 0.9623 | 0.9614 | 0.9380 | 0.9835 at (4/15°, 8/16°) | 0.9853 |

- **Loop 1 matters most.** Lowering the loop-1 −2 angle from 12° to 6° raises the 3D ground fraction by 0.9–1.4 points for Ω₀ = 10–20 kHz. The 12° setting puts a Δn = −2 node at n ≈ 9–13, where much of the initial population passes. Within 0.2 points of the optimum, θ₋₂ = 4–6° and θ₋₁ = 5–18° (panel g). The 6/6°, 9/18° setting found at 20 kHz gives 0.9866 / 0.9873 / 0.9879 at 10 / 14 / 20 kHz, so it is robust to the unknown Rabi frequency.
- **Loop 2 matters little once loop 1 is good.** With loop 1 at 6/6°, keeping loop 2 at the lab 13/12° gives 0.9875 (20 kHz) and 0.9865 (10 kHz), within 0.04 points of the loop-2 optimum. With the lab loop 1, loop 2 spans 1.5–2.3 points over 9–20°. The 30 + 10 split beats 40 cycles of loop 1 by 0.1 point and 40 cycles of loop 2 by 1 point.
- **Per-axis angles add only 0.15–0.2 points.**
- **Centring the pulses on the actual lines** (the carrier shifts by −2.1 / −1.1 / −0.6 kHz at 12°; the −2 lines of hot y and z atoms (n = 20–30) sit at −3…−6 kHz through the differential trap frequency) raises the lab sequence from 0.9791 to 0.9830 at 20 kHz.
- **The sequence is not converged at 40 cycles.** It removes at most one quantum per axis per cycle without OP. The lab setting gives 0.9479 / 0.9791 / 0.9857 / 0.9878 after 30 / 40 / 50 / 60 cycles.
- **The initial temperature dominates the outcome.** The lab sequence gives 0.9984 / 0.9925 / 0.9791 / 0.9569 / 0.9265 at 3 / 4 / 5 / 6 / 7 µK; the optimum gives 0.9993 / 0.9962 / 0.9879 / 0.9726 / 0.9497. The gain from the angles grows with temperature.

**Acceptance against Du's 96 % (3D): not reproduced at the nominal inputs.** The model gives 97.9 % for the lab sequence. It reaches 96 % at T ≈ 5.8 µK, or at 5 µK with a dark-state scatter of ≈ 0.8 % per OP pulse (0.9674 at 0.5 %, 0.9559 at 1 %). Lattice Raman scattering, site inhomogeneity and microwave drifts are not modelled. Validation spectra are needed (data requests 26–29).

**Explorer.** `tools/psc_explorer.html` (built by `scripts/psc_build_explorer.py`) runs the factorized model in the browser on the exported tables (`scripts/psc_export_explorer.py`). It reproduces the Python results to ≤ 2×10⁻⁴ for Ω₀ ≤ 20 kHz and 2×10⁻³ at 40 kHz (band truncation); the check is `node tools/check_engine.js`. Pipeline: `psc_build_ladders.py` (5 min) → `psc_pulse_tables.py` (4 min per Ω₀) → `psc_analysis.py` (5 min) → `figure_psc.py`, `psc_export_explorer.py` → `psc_build_explorer.py`.

## 2026-10-13 — Comparison with the measured post-cooling excitation

*Correction 2026-10-15: computed at Ω₀ = 20 kHz for both pulses; see the 2026-10-15 section for the lab powers.*

**Measurement (lab, current sequence).** Non-ground fraction 3.1 / 3.8 / 1.2 % on x / y / z, background 0.8 %, mostly spontaneous emission. Background-subtracted: 2.3 / 3.0 / 0.4 %, i.e. 3D ground fraction 0.944 (0.921 without subtraction). The model at nominal inputs (5 µK, no dark-state scatter) gives 0.35 / 1.19 / 0.57 % (Ω₀ = 20 kHz). **Acceptance test: failed at nominal inputs.** (`scripts/psc_measured_comparison.py`)

**The axis pattern points to optical pumping, not the initial temperature.** A hotter start raises all three axes, with z between x and y, because z has the lowest trap frequency: at 8 µK the model gives 2.3 / 5.5 / 3.7 %. The measurement has z lowest. Dark-state scatter per OP pulse heats mainly x and y, because the OP beam lies in the x–y plane and the absorption kick has no z component. At 2 %/OP (20 kHz) the model gives 2.5 / 3.1 / 1.1 %, and at 1.5 %/OP 2.0 / 2.7 / 1.0 %. That matches x and y; z is still predicted 0.6–0.7 points too high. The same pattern would arise if the recoil per OP event is larger than modelled, for example from extra photons scattered before the atom reaches the dark state. An extra-OP-pulse test on cold atoms separates the two (data request 29).

**Protocol under 1.5–2 % dark scatter per OP** (5 µK):

| Ω₀ (kHz) | dark | lab 30+10 | 6/6° + 13/12°, 30+10 | lab 40+10 | 6/6° + 13/12°, 40+10 | lab 60+10 |
|---|---|---|---|---|---|---|
| 20 | 2 % | 0.934 | 0.942 | 0.941 | 0.946 | 0.942 |
| 20 | 1.5 % | 0.945 | 0.953 | 0.952 | 0.957 | — |
| 10 | 2 % | 0.927 | 0.940 | 0.937 | 0.946 | 0.942 |

The loop-1 angle change still gains 0.8–1.3 points, mainly on y (3.1 → 2.5 %). The loop-1 optimum stays at θ₋₂ = 5–6°. Extra cycles now saturate: the lab sequence gives 0.940 / 0.943 / 0.944 after 50 / 60 / 70 cycles at 2 %/OP. Each 0.5 %/OP of dark-state scatter costs about 1.1 points of 3D ground fraction, so the OP darkness is the main lever if this hypothesis holds.

## 2026-10-13 — Shape-robust suite ramp RS25

**Design** (`scripts/design_suite_shape_robust.py`). The suite's own parametrization (`globalopt.theta_of`: angle warp exp(Σ c_j cos jπs), K = 12, composed with a 7th-order smoothstep in time), searched by vectorized differential evolution (144 members, 300 generations, seeded with the suite design where one exists) and Powell polish. Cost: worst 1 − P0 at the end of the rise (exact Gaussian, which includes the breathing excitation the suite's classical sloshing cost omits) over three shapes (ER316 calibration; first-principles model with the measured extinction profile, h = −1 and h = +1) × |3,−3⟩, |4,−3⟩ × K0 = 0.94 × {0.80 (weight 0.3), 0.85, 0.92, 1.00, 1.10, 1.15}; per-EOM PAR slew penalized above 9 V/µs. Acceptance: `theta_of` reproduces the N175 target to 2×10⁻⁸ deg, and the scorer reproduces `score_suite.py` for N175 (model 1.3×10⁻² / 9.1×10⁻³).

**Achievable robustness versus duration** (worst 1 − P0 at end of rise, ×0.85–1.15, design grid):

| Rise | Calibration | Model h = −1 | Model h = +1 | Slew (V/µs) |
|---|---|---|---|---|
| 1.75 ms (RS175) | 2.8×10⁻³ | 2.8×10⁻³ | 2.8×10⁻³ | 8.2 |
| 2.0 ms | 2.6×10⁻³ | 2.6×10⁻³ | 2.6×10⁻³ | 7.4 |
| 2.5 ms (RS25) | 4.1×10⁻⁴ | 5.5×10⁻⁴ | 5.4×10⁻⁴ | 5.65 |

The minimax binds across shapes: at 1.75 ms no design in this family satisfies calibration and model together below ~3×10⁻³ (K = 16 gave no improvement in 220 generations). The 2.0 ms run was unseeded and may not be converged. RS25 dwells ~0.9 ms between 75° and 110°, where the shapes differ, and moves faster elsewhere (`figures/suite_shape_robust.png` a).

**RS25 against the suite** (`scripts/figure_suite_shape_robust.py`, `data/suite_shape_robust_scan.json`; 0.01 steps in scale; "record" is one 200 µs | rise | hold | fall | 200 µs record, worst over holds 0–1 ms and over ×0.85–1.15):

| Ramp | Rise, cal / h−1 / h+1 | Record, cal / h−1 / h+1 |
|---|---|---|
| S25 | 1.5×10⁻⁴ / 2.7×10⁻² / 7.6×10⁻³ | 5×10⁻⁴ / 0.10 / 3.0×10⁻² |
| RS25 | 4.1×10⁻⁴ / 5.8×10⁻⁴ / 7.1×10⁻⁴ | 1.6×10⁻³ / 2.3×10⁻³ / 2.7×10⁻³ |
| N175 | 3×10⁻⁶ / 1.3×10⁻² / 9.1×10⁻³ | 1×10⁻⁵ / 5.1×10⁻² / 3.6×10⁻² |
| RS175 | 2.8×10⁻³ / 3.1×10⁻³ / 3.2×10⁻³ | 1.1×10⁻² / 1.2×10⁻² / 1.3×10⁻² |

For reference, S47 under the model gives 5×10⁻⁴ at the end of the rise and 0.75–1.7×10⁻³ after a record with 1 ms hold (`score_suite.py`). RS25 therefore heats like S47 under either shape at 1.9× its speed. Outside the robust band RS25 reaches 1.8×10⁻³ at ×0.80 and 4.3×10⁻³ at ×0.75 (calibration).

**Validation on the full anharmonic potentials** (`scripts/validate_suite_rs.py`, `data/validate_suite_rs.json`, `tests/test_suite_rs.py`). Split-step on the measured-profile tables (h = ±1, both states, ×0.85 / 1 / 1.15) agrees with the exact-Gaussian engine to 10–25 % in every scenario above 10⁻⁴: RS25 ≤ 5.9×10⁻⁴, S25 up to 2.3×10⁻².

**Hardware.** `run_ilc.py init` limit check (Trek-EOM-ILC) passes on both channels: peak 5128 / 5137 V, slew 5.64 / 5.65 V/µs, 1.13 mA into the assumed 200 pF (S25 1.61 mA, N15 1.86 mA). The record is 3201 points (6.402 ms), the same burst period as S25 and C25. Transport phase 0.63 × run 024 (suite model), predicted winding 0.157π per plane against 0.147π for S25, so the S25 / RS25 pair differs in shape only.

**A/B prediction (S25 against RS25, same duration).** If the calibration describes the lattice, both are null at the x-temperature resolution (≤ 1.6×10⁻³ per record). If the first-principles model does, S25 loses 3–10 % per record and RS25 ≤ 0.3 %. RS175 is not recommended: after a record it is 1.1–1.3×10⁻² under every shape, worse than N175 if the calibration holds and only 3–4× better if the model holds. Target files: `ramps/suite/target_RS25X{1,2}.csv`, `ramps/suite/RS25_prediction.json`.

**Limits.** 1D along x; transverse motion and y/z coupling not modelled. The ×0.80 planes are weighted at 30 %, as in the suite. The heating in an echo (two records) depends on the record-to-record timing, which is not modelled; the per-record values bound each record.

## 2026-10-15 — Projection cooling at the lab microwave powers

**Inputs.** The cooling block compiled by the Freq-Scan-GUI (`output/Coolscan-manifest.json`, written 2026-10-13: "cooling block, 240 pulses (30+10 cycles) … 0.075/0.1425 W; HSR -4000/-6000 Hz (-6000 2nd loop); AFP 3 ms") runs every −2 pulse at 0.1425 W (HSR −6 kHz) and every −1 pulse at 0.075 W (HSR −4 kHz in loop 1, −6 kHz in loop 2); the powers are the same in both loops (`p_m2`, `p_m1`; only `hsr_m1b` changes). The older `dds-control/scripts/DDScool.py:152-160, 271-285` (AFP block last updated 2015) uses the same two powers for the same two pulses (`AFPPower1` = 0.1425 W for −2, `AFPPower0` = 0.075 W for −1), but HSR −8 kHz on both pulses in loop 2; its `AFPPower2` = 0.48 W appears only in commented-out variants. The DDS calibration "0.306 corresponds to 0.12 ms Blackman pulse for f4new transition" (`Powerscan.py:317`, `Freqscan.py:337`; after the April 2022 power-supply change, previously 0.318 W) with the firmware's amplitude-shaped Blackman (power = powmax·w², `pulse.h:945`, area 0.42 T Ω) gives Ω/2π = 9.92 kHz at 0.306 W and, with Ω ∝ √P, **6.77 kHz for the −2 pulse and 4.91 kHz for the −1 pulse** (`cooling/afp.py`: `CAL_POWER_W`, `CAL_BLACKMAN_PI_S`, `BLACKMAN_AREA_FACTOR`, `LAB_POWER_W`, `rabi_from_power`). Assumption: the calibration, taken on trapped atoms, is the bare (coupling-1) value; at θ = 0 the ground-state carrier overlap is ≈ 1 (`CAL_IS_BARE_COUPLING`, data request 26). Tests: 0.306 W → 9.9206 kHz, a firmware-discretized Blackman (60 × 2 µs) at that Rabi frequency inverts a two-level atom (independent propagation), √P scaling, the manifest values, 8 µs AFP vs adaptive ODE at 4.91 kHz (`tests/test_cooling.py`). Not modelled: the firmware sweep covers κ = 8 µs (N − 1)/L = 0.9973 of the range and is centred −(1 − κ)·HSR = +16 Hz off the set frequency (−6 kHz sweep).

The model now takes a Rabi frequency per pulse kind (`psc.StepLibrary`, `TableLibrary`; default: the lab values), and the tables are keyed by each pulse's own Rabi frequency. Sets (`psc.rabi_sets()`): the lab powers scaled by 0.25–4 (Rabi 0.5–2×, ratio Ω₋₂/Ω₋₁ = 1.38 fixed) and the previous uniform scan (both pulses at 10 / 14 / 20 / 40 kHz). The rebuilt uniform sets reproduce every number of the 2026-10-13 table (e.g. 0.9728 / 0.9767 / 0.9791 / 0.9623 for the lab sequence).

**The AFP no longer saturates.** An isolated resonant line reaches 0.98 transfer for |A| ≥ 0.27 (−2 pulse, ±6 kHz) and |A| ≥ 0.30 (−1 pulse, ±4 kHz) at the lab powers, against 0.092 / 0.075 at 20 kHz. On x at 12° the −2 pulse transfers > 0.99 on 6 of the levels n = 2–29 (25 at 20 kHz); at 6° it transfers 0.33 / 0.66 / 0.86 / 0.95 for n = 2–5 (0.67 / 0.95 / 1.00 / 1.00 at 20 kHz) (`figures/psc_cooling.png` c). The transfer therefore follows the FC factor over most of the ladder; the 2026-10-13 statement that the AFP saturates almost everywhere holds only for Ω of about 14 kHz and above.

**Predictions at the lab powers, 5 µK, 40 cycles** (`scripts/psc_analysis.py`, `data/psc_results.json`):

| quantity | lab powers (6.77 / 4.91 kHz) | 20 kHz, both pulses (2026-10-13) |
|---|---|---|
| lab sequence (12/6°, 13/12°) | 0.9623 | 0.9791 |
| non-ground x / y / z (%) | 0.65 / 2.12 / 1.03 | 0.35 / 1.19 / 0.57 |
| loop 1 only ×40 / loop 2 only ×40 | 0.9600 / 0.9401 | 0.9781 / 0.9695 |
| best, same angle on all axes | 0.9866 at (5/6°, 10/18°) | 0.9879 at (6/6°, 9/18°) |
| best, per axis | 0.9883 | 0.9894 |
| loop 1 at its optimum, loop 2 at 13/12° | 0.9859 (5/6°) | 0.9875 (6/6°) |
| pulses centred on the actual lines | 0.9671 | 0.9830 |
| after 30 / 50 / 60 cycles (lab angles) | 0.9195 / 0.9747 / 0.9792 | 0.9479 / 0.9857 / 0.9878 |
| 3 / 4 / 6 / 7 µK, lab angles | 0.9960 / 0.9846 / 0.9296 / 0.8888 | 0.9984 / 0.9925 / 0.9569 / 0.9265 |
| 3 / 4 / 6 / 7 µK, optimum | 0.9991 / 0.9955 / 0.9706 / 0.9474 | 0.9993 / 0.9962 / 0.9726 / 0.9497 |

- **The loop-1 angle change is worth 2.4 points instead of 0.8.** Moving the loop-1 −2 angle from 12° to 5–6° (6/6° gives 0.9856) raises the 3D ground fraction from 0.9623 to 0.9859. Within 0.2 points of the loop-1 optimum, θ₋₂ = 4–6° and θ₋₁ = 6–8° or 11–20° (panel g). With the lab loop 1, loop 2 spans 0.929–0.964 over 9–20°. The best 40-cycle split is now 10 + 30 (0.9641; 30 + 10: 0.9623, 40 + 0: 0.9619, 0 + 40: 0.9571).
- **The optimum is insensitive to the power, the lab angles are not** (panel i). At 0.25 / 0.5 / 1 / 2 / 4× the lab powers the lab sequence gives 0.9081 / 0.9447 / 0.9623 / 0.9714 / 0.9758, the optimum at each power 0.9781 / 0.9843 / 0.9866 / 0.9873 / 0.9874. The lab-power optimum (5/6°, 10/18°) gives 0.9840 / 0.9873 / 0.9876 at 0.5× / 2× / 20 kHz; the 20 kHz optimum (6/6°, 9/18°) gives 0.9859 at the lab powers.
- The factorized and the exact 3D propagation now differ by 1.1×10⁻³ on the lab sequence (0.96234 vs 0.96343; 5×10⁻⁵ at 20 kHz). OP heating per event is unchanged (0.16 / 0.18 / 0.10).

**Du's 96 % at the nominal inputs.** At the lab powers the lab sequence gives 0.9623 at 5 µK without dark-state scatter; 0.96 is reached at T = 5.08 µK, or at 5 µK with 0.10 %/OP dark-state scatter (20 kHz: 5.88 µK or 0.82 %/OP). The 3D number therefore agrees with Du at the nominal inputs once the lab powers are used; the per-axis pattern does not (below).

**Measured per-axis excitation** (`scripts/psc_measured_comparison.py`, `data/psc_measured_comparison.json`; measured, background-subtracted 2.3 / 3.0 / 0.4 % on x / y / z, 3D 0.944). At 5 µK without dark scatter the model gives 0.65 / 2.12 / 1.03 %; at 6 µK 1.28 / 3.83 / 2.08 %. With dark-state scatter per OP, the best single value (rms over the three axes, 0.25 %/OP grid) is 1.0 %/OP: 1.76 / 3.10 / 1.33 %, rms deviation 0.62 points, with z predicted 0.9 points too high and x 0.5 points too low (20 kHz: 1.75 %/OP, 2.25 / 2.90 / 1.07 %, rms 0.39). The dark-scatter hypothesis therefore fits the axis pattern less well at the lab powers than at 20 kHz, and z remains the axis the model does not reproduce. **Acceptance against the per-axis measurement: failed**, at the lab powers as at 20 kHz.

**Protocol under dark-state scatter, lab powers** (5 µK):

| dark per OP | lab 30+10 | 6/6° + 13/12°, 30+10 | lab 40+10 | 6/6° + 13/12°, 40+10 | lab 60+10 |
|---|---|---|---|---|---|
| 1 % | 0.939 | 0.962 | 0.952 | 0.966 | 0.959 |
| 1.5 % | 0.928 | 0.951 | 0.941 | 0.955 | 0.948 |
| 2 % | 0.917 | 0.940 | 0.930 | 0.944 | 0.937 |

The loop-1 change gains 2.2–2.3 points at every dark-scatter level (y: 3.1 → 1.7 % at 1 %/OP); the loop-1 optimum stays at θ₋₂ = 5° (θ₋₁ = 6° at 1 %/OP, 17° at 1.5–2 %/OP). Lowering θ₋₂ in loop 1 gains more than 20 extra cycles at the lab angles.

**Not changed in this update.** The manifest's −1 offsets are X 18.5 (the centre of that file's 14.5–22.5 kHz X-offset scan) / Y 15.0 / Z 12.0 kHz and its reference fc = f4new = 56 593 570.0 Hz; the model keeps the 18.1 / 15.1 / 12.5 kHz offsets and 56 592 433.1 Hz of 2026-10-13 (`OFFSET_M1_HZ`, `params/apparatus.yaml`). The rotation angles are not in the manifest.

**Explorer.** The selector lists the Rabi-frequency sets (lab powers first). A Python port of `tools/psc_engine.js` run on the exported tables reproduces `psc_analysis.py` to ≤ 1×10⁻⁴ for the lab-power sets, 1.7×10⁻⁴ at 20 kHz and 1.8×10⁻³ at 40 kHz (band truncation). `node tools/check_engine.js` (updated for named sets) was not run: there is no Node.js on the build machine. Runtimes on 2026-10-15 (4 cores, shared with two other jobs): `psc_build_ladders.py` 22 min, `psc_pulse_tables.py --workers 4` 26 min for the 9 sets (3240 tables), `psc_analysis.py` 25 min, `psc_measured_comparison.py` 4 min, `figure_psc.py` 9 s, export and build 6 s.

## 2026-10-15 — Sideband solver speed and the vibrational-level cut

**Solver.** `vibrational_levels` now uses a sparse shift-invert Lanczos solve (ARPACK, fixed start vector) of the same fourth-order finite-difference Hamiltonian; the LAPACK banded solver stays as `vibrational_levels_banded`. On the nominal x wells the 13 lowest levels agree to < 10⁻⁶ Hz and the Franck–Condon matrices to < 10⁻⁹ (`tests/test_sidebands.py`). `StatePotential.potentials` diagonalizes once for both states. One nominal `spectrum` call: 2.97 s → 0.34 s; with the per-pair level cache (`spectrum(..., cache=)`) a temperature change recomputes nothing and a one-pair change 0.11 s. The test suite went from 31 s to 17 s.

**Level cut.** `SiteConfig.n_levels = 6` normalizes the thermal weights over six levels. At 1 µK this biases the x Δn = +1 sideband-to-carrier ratio low by 2 % (0.189 vs 0.193 converged with 13 levels; Δn = −1: 0.076 vs 0.078) and leaves the carrier unchanged; at 5 µK the bias is 45 % (0.32 vs 0.59 converged with 30 levels; carrier weight 0.18 vs 0.047). `figures/sidebands_alignment.png` is at 1 µK (its docstring said 5 µK). The explorer now keeps enough levels that ≤ 10⁻³ of the population lies above the last one (13 at 1 µK, 59 at 5 µK; `scripts/sideband_explorer.py`, `tests/test_sideband_explorer.py`).

## 2026-10-15 — Sideband model over θ ∈ [−π, π]

**Symmetries** (`lattice_sim/spectra/sidebands.py`, `tests/test_large_angle.py`). For any extinction ratio, handedness, arm amplitudes and field, V(x; −θ) = V(−x; θ) and V(x; θ + π) = V(x + a/2; θ) (to 10⁻⁴ Hz). Each well therefore moves by exactly a/2 = 2.4187 µm between θ = 0 and π, the two states in opposite directions; at ±π the |3,−3⟩ well of the neighbouring site lies on the |4,−4⟩ well, and the spectrum at π − θ equals that at θ with the site offset k ↔ 1 − k. Each state's well is tracked from θ = 0 in steps ≤ 7.5° (agrees with `StatePotential.track` to 2 nm), and overlaps with the other state's wells on sites k = −1, 0, +1 are computed (`site_lines`; `lines` and `carrier_weight` are summed over k). At small θ nothing changes (nominal carrier −903.06997 Hz, line weights to 10⁻⁹).

**Mid angles.** At 90° (x pair, nominal) the wells are a/2 apart, the carrier weight is 0, and the line strength lies −879 kHz from the carrier (rms 25 kHz, from exact sum rules). From 0 to 90° V_min rises from 47 Hz to 2.003 MHz (|4,−4⟩) and 2.500 MHz (|3,−3⟩), and the zero-point energy E₀ − V_min falls from 9.498 kHz to 5.360 and 1.743 kHz; the x pair's part of the carrier reaches −494 kHz (zero-point part +3.6 kHz). Checks: at θ = 0, ER = ∞ the lattice is a pure sinusoid and E₀ − V_min = f/2 − E_q/4 (E_q/h = 16.04 Hz) to 2 mHz; a sinc-DVR solver agrees with E₀ to 0.4 mHz at every angle; second-order perturbation theory agrees to 0.03 Hz except for the flat-bottomed |3,−3⟩ well at 90° (−6.6 Hz).

**State labels.** The old overlap check (`StatePotential.potential`, "strongly mixed") raised for |4,−4⟩ at 1.5× depth near θ = 100–130° (1.1–1.3 MHz up the wall), at ER 10² and at 1.2 G, and for |3,±1⟩, |4,0⟩, |4,+1⟩, |3,0⟩ at nominal settings over |θ| ≈ 35–145° at the trap minimum. `spectrum()` now labels states adiabatically (energy rank at the bias field), identical where the old labels exist. At 90° the stretched states at the trap minimum are 0.838 (|4,−4⟩) and 0.615 (|3,−3⟩) of their bias-field states; the microwave matrix element of these tilted states is not modelled.

## 2026-10-15 — Microwave pulses of the DDS and the scan they measure

**Model** (`lattice_sim/spectra/microwave.py`, `tests/test_microwave.py`, `scripts/explorer_microwave.py`). The three pulse shapes are modelled as the dds-control microwave compiler (`NewDDS2-Cooling2.0-Cal-V2`, identical to `-Cal` for these pulses) writes them. Blackman: field amplitude stepped every 2 µs (power = powmax·w², `pulse.h:945`), area 0.42 T × peak Rabi frequency. AFP: sin² amplitude (power ∝ sin⁴) and a √(1 − sin⁴) sweep from f − HSR to f + HSR in 8 µs tics; the firmware's sweep covers κ = 8 µs (N − 1)/L of the nominal range (centre +10.7 Hz at 3 ms, HSR −4 kHz). Square: 1 µs steps. Rabi frequency ∝ √power, calibrated by the lab's 0.12 ms Blackman π pulse at 0.306 W: Ω/2π = 9.92 kHz √(P/0.306 W). Each line |up, n⟩ → |down, n′⟩ is a two-level system with coupling s = |⟨n′|n⟩| (3D product); lines from one initial state combine as p(n)[1 − Π(1 − P)]. Checks: propagator vs `solve_ivp` 10⁻⁸ (Blackman), 2·10⁻⁵ (AFP), Rabi formula (square); response table vs direct propagation < 0.01 max, < 10⁻³ rms. Independent lines vs a 1D multi-level model: within 0.015 (0.6 ms Blackman π) and 0.01 (0.6 ms square π), but 0.064 off for the 0.12 ms Blackman π (peak Rabi 9.9 kHz vs f = 19 kHz) and up to 0.04 for the lab AFP near a sideband (carrier light shift ~0.3 kHz).

**Nominal lattice, 1 µK.** The ground-state carrier couples with s = 0.840, so a 0.12 ms Blackman π on the lattice carrier needs 0.433 W rather than 0.306 W (if the calibration was taken on the free atom; data request 21). The lab AFP (3 ms, 0.08 W, −4 kHz) transfers ≥ 0.99 over ±2.5 kHz for any line with s ≳ 0.3; its single-line FWHM is 7.9 kHz (carrier) and 7.6 kHz (y Δn = +1), so at this power the sidebands saturate like the carrier. A 0.6 ms Blackman π pulse resolves them: FWHM 2.6 kHz, ground-state x Δn = +1 peak 0.19. The firmware's commented-out π-pulse rule (÷0.30265, "integrating power") assumes Rabi ∝ power and is not compiled.

## 2026-10-15 — The spectrum across a 9 × 9 × 9 block of sites

**Model** (`lattice_sim/spectra/grid.py`, `tests/test_grid.py`, `scripts/explorer_grid.py`). Sites at (i a_x, j a_y, k a_z), i, j, k = −4 … 4; each pair's arm amplitudes at a site follow from `beams.PairGeometry` in its own frame. A pair's configuration varies across the block only through its two arm amplitudes, so its levels are computed exactly at 5 × 5 Chebyshev nodes and interpolated (vs direct at 40 µm offsets: f within 0.8 Hz, carrier 1.3·10⁻⁴ Hz, sideband ratio 4·10⁻⁶); 601 vs 1201 position points change f by 0.02 Hz. The Rayleigh range (13 mm) makes the waist constant to 10⁻⁶ over the block. The displays follow the lab's analysis (`Data-Analysis/readstats9p0 2026PC.py`): stacked 1D slices per column, row and plane (`plotslices`, l. 4695) with a line through the carrier centres and a parabola through the sideband spacing (`linethroughcenters` l. 4353, `fitcenterprofile` l. 4415), and per-plane site maps "Z = k" (`plottemp` l. 1724) with a 3D plane fit (`planefit3d` l. 3892).

**Nominal block, all arms centred.** The sites are not identical: the 60 µm waist lowers f_x to 0.81 of the centre value at x = z = 19.35 µm, and the carrier spans 891 Hz (weighted rms 170 Hz) as a saddle — the two arms of a pair cross at ±5°, so off-axis sites see them unbalanced, and the ellipticity turns the imbalance into a carrier shift (at x = 19 µm: −1110 / −797 / −370 Hz for y = +19 / 0 / −19 µm; with ER = ∞, −740 / −800 / −740 Hz). The block is inversion-symmetric, so neither the plane fit nor the slice lines show a gradient.

**Offsets (x arms, along z).** EOM arm +10 µm: dcarrier/dz +29.2 Hz/µm and df_x/dz +48.2 Hz/µm on the central line (the 2026-10-08 table: +28, +47), 26.8 Hz/µm from the whole-block plane fit (edge sites are shallower). Differential ±10 µm: df_x/dz = 0.00, dcarrier/dz −62.0 Hz/µm (table: −60). Common +10 µm: d ln f/dz = 0.556 %/µm = 2 r₀/w². Grid gradients agree with `beams.gradients` at the same span to 6·10⁻⁶; the slice-centre line equals the plane fit; the spacing parabola's vertex is at +1.02 slices for one arm 10 µm up and +2.14 for both.

## 2026-10-15 — A chain of atoms through the echoed collisional gate

**Model** (`lattice_sim/gates/chain.py`, `scripts/chain_explorer.py`, `tests/test_chain.py`). The Phase 4 pair model repeated along x: in each leg the |3,+m⟩ packet of atom j meets the |3,−m⟩ packet of atom j + 1 (no other packets come within a spacing), the colliding configuration gains φ = 2π∫U/h dt per leg, and the π pulse swaps the labels, so each neighbour pair gets diag(1, e^{−iφ₁}, e^{−iφ₂}, 1) (√ZZ for φ₁ = φ₂ = π/2). An atom with n interacting neighbours has the isolated fringe times Π cos(Φ/2), Φ = φ₁ + φ₂, shifted by ±(φ₁ − φ₂)/2 per neighbour: cos^n(Φ/2) for identical legs — confirmed against explicit state-vector simulations of chains of 2–7 atoms with unequal leg phases (10⁻¹²).

**Checks.** The calibration model puts relative overlaps 10⁻³ / 0.1 / 0.5 / 0.9 at 3796 / 4339 / 4785 / 5155 V (lab table 3800 / 4340 / 4785 / 5155), full-overlap U/h = 9.478 kHz, π/2 in 26.4 µs; the hold overlaps match `data/p4_results.json` (U to 10⁻⁴, separation to 0.01 nm); ∫U dt matches `CollisionModel.evolve` to 2·10⁻⁴ when the gaps are large and within 2 % with the real gaps (7-state model). Defaults (2800 a₀): |3,±3⟩ at 4150 V on the R3 ramp needs a 448 µs hold (separation 244 nm, relative overlap 0.031, U/h = 294 Hz; the ramps give 0.237π of the π/2; 7-state −0.4 %, leakage 3·10⁻⁹); |3,±1⟩ at 3300 V on R1, 225 µs (7-state −1.8 %); with the DIS length for |3,±3⟩, 1.09 ms. Assumptions (named in `chain.py`): adiabatic ground-state Gaussians in 1D, ideal instantaneous pulses, no noise, identical legs, spin exchange neglected in the phase, and the qubit map `PLUS_MOVER_QUBIT`.
