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

**Nominal configuration** (full depth 19/16/12.5 kHz, 3 % rotation on every pair, ER 3×10⁴, 1.393 G at 45°, 1 µK): carrier −0.9 kHz from the free atom, wells displaced by 30–37 nm per axis, Δn = +1 sidebands at −f(|3,−3⟩) and Δn = −1 at +f(|4,−4⟩) (the two trap frequencies differ by 10–40 Hz). Sideband-to-carrier ratio 0.19 (x, Δn = +1) at 1 µK. Sensitivities (`figures/sidebands_alignment.png`): a 20 µm centering offset of both x arms lowers f_x by 10 % (the rotating arm alone: 5 %) and the x sideband ratio by only 3 %; 0.1° of crossing half-angle moves f_x by 1 Hz (negligible). The carrier sits at −500 Hz with no rotation, −900 Hz at 3 % and −5 kHz at 10 %; at ER 10² it moves to −1.8 kHz (h = −1) or 0 (h = +1), and it moves by 550 Hz between a 0° and a 90° bias-field angle; the Zeeman term adds −2.45 kHz per mG. The sideband ratio is set by the rotation (0 → 0.19 → ~1 from 0 to 3 to 10 %), the temperature, and the bias-field angle: the differential displacement is first order in the fictitious field only through its projection on the bias field, and the x pair's fictitious field lies along its beams (lab y), so the x sideband ratio goes from 0 at a 0° field angle to 0.44 at 90° (0.19 at the nominal 45°). The relative sideband strengths of the three pairs therefore measure the field angle directly. Not yet compared with measured spectra (data request 18).
