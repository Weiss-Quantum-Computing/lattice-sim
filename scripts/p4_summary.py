"""Phase 4 summary: gate stop points, phase slope, and single-atom round-trip P0.

Reads data/p4_results.json (from p4_collision.py). For each pair (h = -1),
scattering model and target phase (pi/2, pi) it reports the stop voltage,
the phase slope dphi/dV there (sets the Trek reproducibility needed), the
spin-exchange leakage, and the motional ground-state population P0 of each
partner after the out-and-back collision ramp (split-step on the
measured-extinction potential tables, data/x_ERmeas133_h-1_10kHz_F3m*.npz,
built by build_profile_tables.py).
Writes data/p4_summary.json.
Usage: python scripts/p4_summary.py
"""

import json

import numpy as np

from lattice_sim.gates.collision import collision_ramp
from lattice_sim.motion.transport1d import PotentialTable, Transport1D, load_waveform

V_PI = 5400.0
RAMPS = {3: "data/R3_5ms_robust.csv", 1: "data/R1_8ms_robust.csv"}
MODELS = ("uniform 2800 a0", "DIS")
TARGETS = (0.5, 1.0)                     # collisional phase in units of pi


def stop_point(scan, target_pi):
    v = np.array(scan["theta"]) / np.pi * V_PI
    phi = np.array(scan["phi"]) / np.pi
    leak = np.array(scan["leak"])
    reached = phi >= target_pi
    if not reached.any() or reached[0]:
        raise ValueError(f"phase {target_pi:g} pi not crossed inside the scan "
                         f"(range {phi.min():.2f}-{phi.max():.2f} pi)")
    k = int(np.argmax(reached))
    f = (target_pi - phi[k - 1]) / (phi[k] - phi[k - 1])
    vs = v[k - 1] + f * (v[k] - v[k - 1])
    slope = np.pi * (phi[k] - phi[k - 1]) / (v[k] - v[k - 1])       # rad / V
    return vs, slope, leak[k - 1] + f * (leak[k] - leak[k - 1])


def round_trip_p0(table, t, theta, v_stop):
    tc, thc = collision_ramp(t, theta, np.pi * v_stop / V_PI, 0.0)
    tr = Transport1D(table)
    _, p0, _ = tr.propagate(tc, thc, dt_us=0.25, sample_every_us=tc[-1])
    return float(p0[-1])


def main():
    r = json.load(open("data/p4_results.json", encoding="utf-8"))
    out = []
    for m in (3, 1):
        t, theta = load_waveform(RAMPS[m])[:2]
        tables = {s: PotentialTable.load(f"data/x_ERmeas133_h-1_10kHz_F3m{s * m:+d}.npz") for s in (1, -1)}
        for name in MODELS:
            for target in TARGETS:
                vs, slope, leak = stop_point(r[f"scan_m{m}_h-1_{name}"], target)
                p0 = {f"{s * m:+d}": round_trip_p0(tables[s], t, theta, vs) for s in (1, -1)}
                row = dict(m=m, model=name, target_pi=target, stop_V=vs, slope_mrad_per_V=1e3 * slope,
                           leak=leak, p0=p0)
                out.append(row)
                print(f"|3,+/-{m}> {name:16s} {target:.1f} pi: stop {vs:6.0f} V, "
                      f"dphi/dV {1e3 * slope:5.1f} mrad/V, leak {leak:.1e}, P0 "
                      + ", ".join(f"m{k} {v:.5f}" for k, v in p0.items()), flush=True)
    with open("data/p4_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])
    main()
