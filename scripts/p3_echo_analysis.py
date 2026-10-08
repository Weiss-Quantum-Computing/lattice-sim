"""Phase 3: echo contrast and phase for transport sequences, with error budget.

1. Calibrate the field-noise spectrum to Du (2025): stretched-state echo with
   two 1/60 s legs, sigma_phi(field) = sqrt(0.054^2 - 0.028^2) pi.
2. Du's |3,+/-1> intensity experiment: which leg-to-leg mechanism reproduces
   the loss of contrast after 2.5 ms parked at pi/2, and what each then
   predicts for the rotation-angle scan (0.65 at 0.45 pi, < 0.1 at 0.65 pi).
3. Predictions for the current apparatus: |3,+/-3> (R3 ramp) and |3,+/-1>
   (R1, 8 ms) with the measured extinction profile, echo legs separated by
   an extended storage hold with an XY-8 pi train, x only and x-y-z
   interleaved; contrast budget by source versus leg separation.

All assumptions are module-level constants, each listed in the data-request
section of docs/findings.md. Usage: python scripts/p3_echo_analysis.py ER_CSV
"""

import json
import sys

import numpy as np

from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.motion.transport1d import load_waveform
from lattice_sim.spin.echo import (EchoModel, LegTransient, NoiseSpectrum, Sequence, Transport,
                                   coefficients, filter_variance, site_intensities)
from lattice_sim.spin.pair_physics import PairPhysics
from lattice_sim.spin.pulses import worst_input_contrast

# --- assumptions (defaults; replace with measured values) -------------------
KAPPA_STRETCHED = 2452.0              # Hz/mG, |4,-4> -> |3,-3>
DU_SIGMA_FIELD = np.sqrt(0.054**2 - 0.028**2) * np.pi
FIELD_SHAPE = NoiseSpectrum(          # lock-on shape guess (mG^2/Hz, mG rms)
    white=(1e-3) ** 2, flicker=(3e-3) ** 2, f_min=0.05,
    lines=[(30.0, 5e-3)] + [(60.0 * n, 10e-3 / n) for n in range(1, 11)])
RIN = NoiseSpectrum(white=1e-12, flicker=1e-11, f_min=0.05, lines=[(50.0, 1e-5)])  # Du Fig 5.21
LEG_INTENSITY_SIGMA = 1e-4            # quasi-static dI/I between transports
TRANSIENT = LegTransient(d_theta_rad=np.deg2rad(0.5), d_eta=1e-4, tau_ms=50.0)
SITE_A = site_intensities(half_extent_um=10.0, waist_um=60.0)
SCATTER_PER_S = 0.1                   # coherence-destroying scattering in the hold
PULSE_EPS, PULSE_DET = 0.01, 0.01     # MW amplitude error (sigma), detuning / Rabi
MAP_FIDELITY = {1: 0.998, 3: 0.998**3}  # per storage<->transport map (3 steps for |m|=3)
MOTION_P0 = {3: 0.99983 * 0.99988, 1: 0.99993 * 0.99972}  # final P0 product per transport
RAMPS = {3: "data/R3_5ms_robust.csv", 1: "data/R1_8ms_robust.csv"}
HOLD_AT_PI_US = 600.0


def calibrate_field():
    seq = Sequence([("wait", 1e6 / 60), ("pi",), ("wait", 1e6 / 60)], dt_us=10)
    g = seq.build()
    c = np.full(g.t_us.size, KAPPA_STRETCHED)
    v = filter_variance(g, c, FIELD_SHAPE)
    k = DU_SIGMA_FIELD / np.sqrt(v)
    return FIELD_SHAPE.scaled(k), k


def ramp(m, theta_max=np.pi):
    t, th, _ = load_waveform(RAMPS[m])
    if theta_max < np.pi:
        stop = np.searchsorted(th, theta_max)
        t, th = t[: stop + 1], np.minimum(th[: stop + 1], theta_max)
    return t, th


def echo_sequence(m, separation_ms, n_pi=1, axes=("x",), lead_ms=1.0, sync_hz=60.0):
    """pi/2 - transports - storage hold with n_pi pulses - transports - pi/2.

    The hold is lengthened so that the second block of transports starts an
    integer number of line periods after the first (line-synchronous)."""
    t, th = ramp(m)
    trs = [Transport(t, th, HOLD_AT_PI_US, ax) for ax in axes]
    block_us = sum(tr.duration_us for tr in trs)
    period_us = 1e6 / sync_hz
    sep_us = max(separation_ms * 1e3, block_us + 50)
    sep_us = np.ceil(sep_us / period_us) * period_us        # start-to-start, whole periods
    gap_us = sep_us - block_us
    segs = [("wait", lead_ms * 1e3)] + [("transport", tr) for tr in trs]
    spacing = gap_us / n_pi
    segs.append(("wait", spacing / 2))
    for k in range(n_pi):
        segs.append(("pi",))
        segs.append(("wait", spacing if k < n_pi - 1 else spacing / 2))
    segs += [("transport", tr) for tr in trs]
    segs.append(("wait", lead_ms * 1e3))
    return Sequence(segs, dt_us=5.0, label=f"m{m} sep {sep_us/1e3:.1f} ms N{n_pi} {''.join(axes)}")


def du_intensity_test(pair1, field):
    """Du's |3,+/-1> experiment (old EOMs, ER ~ 625 assumed, 1.5 ms ramps)."""
    eta = lambda th: np.full_like(np.asarray(th, float), 1 / np.sqrt(625.0))
    def seq(theta_max, park_us):
        tt = np.linspace(0, 1500 * theta_max / (np.pi / 2), 400)
        thr = theta_max * np.sin(np.pi / 2 * tt / tt[-1]) ** 2
        tr = Transport(tt, thr, park_us + 1.0, "x")
        half = 1e6 / 60
        pad = half - tr.duration_us
        return Sequence([("wait", pad / 2), ("transport", tr), ("wait", pad / 2), ("pi",),
                         ("wait", pad / 2), ("transport", tr), ("wait", pad / 2)], dt_us=5.0)
    rows = {}
    for label, kw in (("white RIN only", dict()),
                      ("leg intensity drift", dict(leg=True)),
                      ("leg ellipticity drift", dict(ell=True))):
        rows[label] = {}
        # fit the leg-to-leg amplitude so that contrast = 0.1 after 2.5 ms parked at pi/2
        def contrast(theta_max, park_us, amp):
            s = seq(theta_max, park_us)
            model = EchoModel(pair1, eta, field, RIN,
                              transient=LegTransient(0.0, amp if kw.get("ell") else 0.0, 1e9),
                              leg_intensity_sigma=amp if kw.get("leg") else 0.0)
            r = model.analyse(s)
            if kw.get("ell"):  # treat d_eta as random leg-to-leg, rms amp: Gaussian dephasing
                g = s.build()
                c = coefficients(g, pair1, eta)["ellipticity"]
                legs = [k for k in np.unique(g.leg) if k >= 0]
                v = sum((2 * np.pi * np.sum((g.sign * c)[g.leg == k]) * g.dt_us * 1e-6) ** 2 for k in legs)
                return r["contrast"] / r["c_inhomogeneous"] * np.exp(-v * amp**2 / 2)
            return r["contrast"] / r["c_inhomogeneous"]
        amp = 0.0
        if kw:
            lo, hi = 1e-7, 1e-1
            for _ in range(60):
                mid = np.sqrt(lo * hi)
                (lo, hi) = (mid, hi) if contrast(np.pi / 2, 2500, mid) > 0.1 else (lo, mid)
            amp = np.sqrt(lo * hi)
        rows[label]["fitted_amplitude"] = amp
        for name, th, park in (("park 0 ms", np.pi / 2, 0), ("park 1.5 ms", np.pi / 2, 1500),
                               ("park 2.5 ms", np.pi / 2, 2500), ("0.45 pi", 0.45 * np.pi, 0),
                               ("0.56 pi", 0.56 * np.pi, 0), ("0.65 pi", 0.65 * np.pi, 0)):
            rows[label][name] = contrast(th, park, amp)
    return rows


def predictions(field, profile, separations=(16.7, 33.3, 50, 100, 150, 200), taus=(20, 50, 100)):
    out = []
    eta = profile.eta
    for m in (3, 1):
        pair = PairPhysics.load_or_compute(m)
        for axes in (("x",), ("x", "y", "z")):
            for sep in separations:
                for tau in taus:
                    n_pi = 1 if sep < 40 else 9
                    seq = echo_sequence(m, sep, n_pi, axes)
                    model = EchoModel(pair, eta, field, RIN,
                                      transient=LegTransient(TRANSIENT.d_theta_rad, TRANSIENT.d_eta, tau),
                                      leg_intensity_sigma=LEG_INTENSITY_SIGMA, site_a=SITE_A,
                                      scatter_rate_per_s=SCATTER_PER_S,
                                      motional_contrast=MOTION_P0[m] ** len(axes))
                    r = model.analyse(seq)
                    r["c_pulses"] = worst_input_contrast("XY-8", n_pi, eps_sigma=PULSE_EPS,
                                                         detuning_over_rabi=PULSE_DET, n_shots=150)
                    r["c_maps"] = MAP_FIDELITY[m] ** (4 * len(axes))
                    r["contrast"] *= r["c_pulses"] * r["c_maps"]
                    r.update(m=m, axes="".join(axes), separation_ms=sep, tau_ms=tau, n_pi=n_pi,
                             actual_separation_ms=float(seq.label.split("sep ")[1].split(" ms")[0]))
                    out.append(r)
                    print(f"m{m} {''.join(axes):3s} sep {r['actual_separation_ms']:5.1f} tau {tau:3d}: C {r['contrast']:.4f} "
                          f"phase {r['phase_rad']:+.3f}  [field {r['c_field']:.4f} int {r['c_intensity']:.4f} "
                          f"leg {r['c_leg_drift']:.4f} inhom {r['c_inhomogeneous']:.4f} scat {r['c_scatter']:.4f} "
                          f"mot {r['c_motion']:.4f} pulses {r['c_pulses']:.4f} maps {r['c_maps']:.4f}]", flush=True)
    return out


def main(er_csv):
    field, k = calibrate_field()
    print(f"field spectrum scaled by {k:.3f} to reproduce Du's {DU_SIGMA_FIELD/np.pi:.4f} pi")
    pair1 = PairPhysics.load_or_compute(1)
    du = du_intensity_test(pair1, field)
    for label, row in du.items():
        print(f"Du |3,+/-1> test, {label}: amplitude {row['fitted_amplitude']:.2e}; " +
              "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "fitted_amplitude"))
    profile = ExtinctionProfile.from_rows(read_rampol_er(er_csv), 1, "away")
    pred = predictions(field, profile)
    with open("data/p3_results.json", "w", encoding="utf-8") as fh:
        json.dump({"field_scale": k, "du": du, "predictions": pred}, fh, indent=1, default=float)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])  # paths below are relative to the repo root
    main(sys.argv[1])
