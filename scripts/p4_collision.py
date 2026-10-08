"""Phase 4: controlled collisions of |3,+m>/|3,-m> pairs on the x axis.

1. Acceptance: the overlap engine fed with Transport-Optimizer's calibration
   reproduces tools/overlap_vs_voltage.py (overlap 0.1 / 0.5 / 0.9 at 4340 /
   4785 / 5155 V; U/h = 9.48 kHz at full overlap, 7 kHz transverse, 2800 a0).
2. First-principles pairs (measured extinction profile, both handedness
   signs): separation, overlap and collision energy vs theta; lattice-energy
   gaps to the spin-exchange channels.
3. Collision ramps (R3 for |3,+/-3>, R1 8 ms for |3,+/-1>) run out to a stop
   angle and straight back (no hold), under two scattering models: uniform
   2800 a0 (Du) and the DIS channel lengths. Scans the stop voltage over
   2900-5400 V and reports where the collisional phase reaches pi/2 and pi,
   with the leakage out of the |L:+m, R:-m> state.
Usage: python scripts/p4_collision.py ER_CSV
"""

import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from fit_er316_calibration import load
from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.gates.collision import (MU, build_collision_model, collision_ramp,
                                         dis_channel_lengths, interaction_hz, overlap_density,
                                         scattering_operator, uniform_lengths)
from lattice_sim.motion.transport1d import load_waveform
from lattice_sim.constants import A0

V_PI = 5400.0
TRANSVERSE_HZ = (8.0e3, 6.25e3)        # y, z trap frequencies at quarter depth (assumed)
HOLD_VOLTAGES = (4340.0, 4785.0, 5155.0)
RAMPS = {3: "data/R3_5ms_robust.csv", 1: "data/R1_8ms_robust.csv"}


def acceptance_calibration():
    cal = {n: load("data/reference", n) for n in ("F3", "F4")}
    sp = {n: (CubicSpline(np.pi * fr, np.abs(x) * 1e-6), CubicSpline(np.pi * fr, f * 1e3))
          for n, (fr, x, f) in cal.items()}
    alat = float(sp["F3"][0](np.pi) + sp["F4"][0](np.pi))
    v = np.linspace(0, V_PI, 54001)
    th = np.pi * v / V_PI
    d = alat - sp["F3"][0](th) - sp["F4"][0](th)
    o = overlap_density(d, sp["F3"][1](th), sp["F4"][1](th), 7e3, 7e3)
    rel = o / o[-1]
    out = {f"V_overlap_{lev}": float(v[np.argmax(rel >= lev)]) for lev in (0.1, 0.5, 0.9)}
    out["U_full_7kHz_2800a0_kHz"] = float(interaction_hz(2800 * A0, o[-1]) / 1e3)
    return out


def main(er_csv):
    print("acceptance (calibration files):", acceptance_calibration())
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    profile = ExtinctionProfile.from_rows(read_rampol_er(er_csv), 1, "away")
    a_dis = dis_channel_lengths()
    ops = {"uniform 2800 a0": scattering_operator(uniform_lengths(2800.0)),
           "DIS": scattering_operator(a_dis)}
    print("DIS channel lengths (a0):", {k: round(v, 1) for k, v in a_dis.items()})
    results = {"dis_lengths_a0": a_dis, "pairs": []}
    for m in (3, 1):
        for h in (-1, 1):
            t0 = time.time()
            cm = build_collision_model(m, LatticePair(extinction_ratio=profile, handedness=h), b,
                                       transverse_hz=TRANSVERSE_HZ)
            i0 = cm.index
            t, th = load_waveform(RAMPS[m])[:2]
            for vh in HOLD_VOLTAGES:
                theta_h = np.pi * vh / V_PI
                o = cm.overlap(theta_h)
                gaps = {int(mu): float(cm.zeeman_hz[j] + cm._lat(theta_h)[j])
                        for j, mu in enumerate(MU) if mu != m}
                row = dict(m=m, h=h, hold_V=vh, theta_hold_pi=theta_h / np.pi,
                           separation_nm=float(cm.separation(theta_h) * 1e9), gaps_hz=gaps)
                for name, op in ops.items():
                    row[name] = dict(U_hold_hz=float(interaction_hz(1.0, o) * np.real(op[cm.index, cm.index])))
                results["pairs"].append(row)
                print(f"m{m} h{h:+d} {vh:.0f} V (theta {theta_h/np.pi:.3f} pi, d {row['separation_nm']:.0f} nm): "
                      + ", ".join(f"U[{n}] {row[n]['U_hold_hz']/1e3:.2f} kHz" for n in ops)
                      + "; gaps " + ", ".join(f"{k:+d}: {v/1e3:+.1f}k" for k, v in gaps.items()))
            # gate design: stop the ramp at theta_hold with no hold; scan theta_hold
            th_scan = np.pi * np.linspace(2900, 5400, 51) / V_PI
            for name, op in ops.items():
                curve = cm.ramp_phase_curve(t, th, th_scan, op)
                results[f"scan_m{m}_h{h:+d}_{name}"] = {"theta": th_scan.tolist(), "phi": curve[:, 0].tolist(),
                                                       "leak": curve[:, 1].tolist()}
                for target in (np.pi / 2, np.pi):
                    k = int(np.argmax(curve[:, 0] >= target))
                    if k == 0:
                        print(f"  m{m} h{h:+d} {name}: phase already {curve[0,0]/np.pi:.2f} pi at {th_scan[0]/np.pi*V_PI:.0f} V")
                        continue
                    if curve[k, 0] < target:
                        print(f"  m{m} h{h:+d} {name}: phase never reaches {target/np.pi:.1f} pi (max {curve[:,0].max():.2f})")
                        continue
                    # linear interpolation for theta and leakage at the target phase
                    f = (target - curve[k - 1, 0]) / (curve[k, 0] - curve[k - 1, 0])
                    th_t = th_scan[k - 1] + f * (th_scan[k] - th_scan[k - 1])
                    lk = curve[k - 1, 1] + f * (curve[k, 1] - curve[k - 1, 1])
                    print(f"  m{m} h{h:+d} {name:16s}: phase {target/np.pi:.1f} pi at stop voltage "
                          f"{th_t/np.pi*V_PI:.0f} V (no hold), leakage {lk:.1e}", flush=True)
            curves = {"theta": cm.thetas.tolist(), "separation_nm": (cm.separation(cm.thetas) * 1e9).tolist(),
                      "overlap": cm.overlap(cm.thetas).tolist(), "lattice_hz": cm.lattice_hz.tolist(),
                      "zeeman_hz": cm.zeeman_hz.tolist()}
            results[f"curves_m{m}_h{h:+d}"] = curves
            print(f"  (m{m} h{h:+d}: {time.time()-t0:.0f}s)")
    with open("data/p4_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main(sys.argv[1] if len(sys.argv) > 1 else "data/measured/133-ms-SE-offset-neg2V-fine-2_extinction.csv")
