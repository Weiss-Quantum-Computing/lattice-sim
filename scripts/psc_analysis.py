"""Projection sideband cooling: the lab sequence, rotation-angle scans and
sensitivities, from the precomputed ladders and pulse tables.

Needs data/psc_ladders.npz and data/psc_pulses.npz.
Writes data/psc_results.json.
Usage: python scripts/psc_analysis.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

OMEGAS = (10.0, 14.0, 20.0, 40.0)
GRID = np.arange(1.0, 20.01, 1.0)
LAB = dict(t2a=12.0, t1a=6.0, t2b=13.0, t1b=12.0)


def sequence(t2a, t1a, t2b, t1b, cycles=(30, 10), hsr1=(-4e3, -6e3)):
    """Same angles on all axes unless dicts are given."""
    def d(v):
        return v if isinstance(v, dict) else dict.fromkeys(psc.AXES, float(v))
    return [psc.Loop(cycles[0], d(t2a), d(t1a), -6e3, hsr1[0]),
            psc.Loop(cycles[1], d(t2b), d(t1b), -6e3, hsr1[1])]


def run(lib, op, **kw):
    cyc = kw.pop("cycles", (30, 10))
    hist = psc.simulate_factorized(sequence(**kw, cycles=cyc), lib, op)
    return psc.ground_3d_factorized(hist), hist


def summary(hist):
    g = [float(np.prod([h[a][0] for a in psc.AXES])) for h in hist]
    return dict(ground_3d=g, ground={a: [float(h[a][0]) for h in hist] for a in psc.AXES},
                nbar={a: [float(np.arange(h[a].size) @ h[a]) for h in hist] for a in psc.AXES})


def scan_loop(lib, op, which, base):
    out = np.zeros((GRID.size, GRID.size))
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            kw = dict(base)
            kw.update({"t2a": a, "t1a": b} if which == 1 else {"t2b": a, "t1b": b})
            out[i, j] = run(lib, op, **kw)[0]
    return out


def optimize_same_angle(lib, op, start=LAB, sweeps=3):
    """Coordinate search over the four angles (same on all axes), 1 deg grid."""
    best = dict(start)
    g = run(lib, op, **best)[0]
    for _ in range(sweeps):
        for which in (1, 2):
            m = scan_loop(lib, op, which, best)
            i, j = np.unravel_index(np.argmax(m), m.shape)
            if m[i, j] > g:
                g = m[i, j]
                best.update({"t2a": GRID[i], "t1a": GRID[j]} if which == 1 else {"t2b": GRID[i], "t1b": GRID[j]})
    return {k: float(v) for k, v in best.items()}, float(g)


def optimize_per_axis(lib, op, start, sweeps=2):
    """Coordinate search with independent angles per axis."""
    best = {k: dict.fromkeys(psc.AXES, float(v)) for k, v in start.items()}
    g = run(lib, op, **best)[0]
    for _ in range(sweeps):
        for axis in psc.AXES:
            for keys in (("t2a", "t1a"), ("t2b", "t1b")):
                for a in GRID:
                    for b in GRID:
                        trial = {k: dict(v) for k, v in best.items()}
                        trial[keys[0]][axis], trial[keys[1]][axis] = float(a), float(b)
                        gt = run(lib, op, **trial)[0]
                        if gt > g + 1e-6:
                            g, best = gt, trial
    return best, float(g)


if __name__ == "__main__":
    t0 = time.time()
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    op = psc.optical_pumping_model(ladders)
    res = dict(omegas_khz=list(OMEGAS), grid_deg=GRID.tolist(), lab=LAB, per_omega={})
    for om in OMEGAS:
        lib = psc.TableLibrary(ladders, tables, om)
        g_lab, h_lab = run(lib, op, **LAB)
        r = dict(lab=g_lab, lab_history=summary(h_lab))
        r["loop1_only_40"] = run(lib, op, t2a=12, t1a=6, t2b=12, t1b=6)[0]
        r["loop2_only_40"] = run(lib, op, t2a=13, t1a=12, t2b=13, t1b=12)[0]
        r["scan_loop1"] = scan_loop(lib, op, 1, LAB).tolist()
        r["scan_loop2"] = scan_loop(lib, op, 2, LAB).tolist()
        best, g_best = optimize_same_angle(lib, op)
        r["best_same"], r["best_same_ground"] = best, g_best
        r["best_same_history"] = summary(run(lib, op, **best)[1])
        pa, g_pa = optimize_per_axis(lib, op, best)
        r["best_per_axis"], r["best_per_axis_ground"] = pa, g_pa
        # robustness: the lab and best-same settings evaluated at the other Rabi frequencies
        r["split"] = {f"{c1}+{40 - c1}": run(lib, op, **LAB, cycles=(c1, 40 - c1))[0] for c1 in (0, 10, 20, 30, 40)}
        r["cycles_total"] = {str(n): run(lib, op, **LAB, cycles=(int(round(0.75 * n)), n - int(round(0.75 * n))))[0]
                             for n in (10, 20, 30, 40, 50, 60)}
        res["per_omega"][f"{om:g}"] = r
        print(f"Omega0 {om:g} kHz: lab {g_lab:.4f}; best same-angle {g_best:.4f} at {best}; "
              f"per-axis {g_pa:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    # cross-evaluation: each Omega's best settings at every Omega
    cross = {}
    for om_set in OMEGAS:
        best = res["per_omega"][f"{om_set:g}"]["best_same"]
        cross[f"{om_set:g}"] = {f"{om:g}": run(psc.TableLibrary(ladders, tables, om), op, **best)[0] for om in OMEGAS}
    res["cross"] = cross
    # dark-state leakage at the lab setting, Omega0 = 20 kHz
    lib = psc.TableLibrary(ladders, tables, 20.0)
    res["dark_leak"] = {}
    for pdark in (0.0, 0.002, 0.005, 0.01, 0.02):
        op_d = psc.optical_pumping_model(ladders, dark_prob=pdark)
        res["dark_leak"][str(pdark)] = dict(lab=run(lib, op_d, **LAB)[0],
                                            best20=run(lib, op_d, **res["per_omega"]["20"]["best_same"])[0])
    # 3D (exact within model) check of the factorized result at 20 kHz
    hist3, _ = psc.simulate_3d(sequence(**LAB), lib, op)
    res["check_3d_lab_20"] = dict(ground_3d=hist3[-1]["ground_3d"], total=hist3[-1]["total"],
                                  factorized=res["per_omega"]["20"]["lab"])
    # single-OP heating (acceptance comparison with Wu p.47 / Kumar p.67, ~7 %)
    heat = {a: op.mixed(a, op.p_k) for a in psc.AXES}
    res["op_heating"] = dict(mean_photons=float(np.arange(op.p_k.size) @ op.p_k),
                             p_leave_ground={a: float(1 - heat[a][0, 0]) for a in psc.AXES},
                             p_leave_ground_3d=float(1 - np.prod([heat[a][0, 0] for a in psc.AXES])))
    Path("data/psc_results.json").write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("check_3d_lab_20", "op_heating", "dark_leak", "cross")}, indent=1))
    print(f"done in {time.time() - t0:.0f} s")
