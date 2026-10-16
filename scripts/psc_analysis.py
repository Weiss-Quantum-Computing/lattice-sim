"""Projection sideband cooling: the lab sequence, rotation-angle scans and
sensitivities, from the precomputed ladders and pulse tables.

Rabi-frequency sets (``psc.rabi_sets()``): 'lab' is the headline (the
manifest powers 0.1425 / 0.075 W of the -2 / -1 pulse -> 6.77 / 4.91 kHz);
'lab_P0.25' .. 'lab_P4' scale both powers; '10' .. '40' put both pulses at
the same Omega0 (the sensitivity scan used before 2026-10-15).

Needs data/psc_ladders.npz and data/psc_pulses.npz.
Writes data/psc_results.json.
Usage: python scripts/psc_analysis.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import afp, psc  # noqa: E402

SETS = psc.rabi_sets()
HEADLINE = psc.LAB_SET
GRID = np.arange(1.0, 20.01, 1.0)
LAB = dict(t2a=12.0, t1a=6.0, t2b=13.0, t1b=12.0)
TEMPS_UK = (3.0, 4.0, 5.0, 6.0, 7.0)
DARK = (0.0, 0.002, 0.005, 0.01, 0.02)
MEASURED_3D = 0.96            # Du p.48


def sequence(t2a, t1a, t2b, t1b, cycles=(30, 10), hsr1=(-4e3, -6e3)):
    """Same angles on all axes unless dicts are given."""
    def d(v):
        return v if isinstance(v, dict) else dict.fromkeys(psc.AXES, float(v))
    return [psc.Loop(cycles[0], d(t2a), d(t1a), -6e3, hsr1[0]),
            psc.Loop(cycles[1], d(t2b), d(t1b), -6e3, hsr1[1])]


def run(lib, op, temperature_k=psc.INITIAL_TEMPERATURE_K, **kw):
    cyc = kw.pop("cycles", (30, 10))
    hist = psc.simulate_factorized(sequence(**kw, cycles=cyc), lib, op,
                                   p0=psc.initial_state(temperature_k=temperature_k))
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


def afp_threshold(omega_hz, hsr_hz, target=0.98):
    """Smallest coupling |A| for which an isolated resonant line reaches the
    target transfer (3 ms AFP, 8 us steps)."""
    f = lambda a: afp.two_level_transfer(a, 0.0, hsr_hz, omega_hz) - target  # noqa: E731
    return float(brentq(f, 1e-3, 1.0, xtol=1e-4))


def match(fun, lo, hi, target=MEASURED_3D):
    """Root of fun(x) = target on [lo, hi], or None if not bracketed."""
    a, b = fun(lo) - target, fun(hi) - target
    return float(brentq(lambda x: fun(x) - target, lo, hi, xtol=1e-3 * (hi - lo))) if a * b < 0 else None


if __name__ == "__main__":
    t0 = time.time()
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    op = psc.optical_pumping_model(ladders)
    res = dict(sets={k: {str(kind): v for kind, v in om.items()} for k, om in SETS.items()},
               headline=HEADLINE, lab_power_w={str(k): v for k, v in afp.LAB_POWER_W.items()},
               power_scales=list(psc.LAB_POWER_SCALES), uniform_khz=list(psc.UNIFORM_OMEGA_KHZ),
               grid_deg=GRID.tolist(), lab=LAB, per_set={})
    res["afp_threshold_098"] = {
        name: {"m2": afp_threshold(om[2] * 1e3, -6e3), "m1": afp_threshold(om[1] * 1e3, -4e3)}
        for name, om in SETS.items()}
    for name, om in SETS.items():
        lib = psc.TableLibrary(ladders, tables, name)
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
        r["split"] = {f"{c1}+{40 - c1}": run(lib, op, **LAB, cycles=(c1, 40 - c1))[0] for c1 in (0, 10, 20, 30, 40)}
        r["cycles_total"] = {str(n): run(lib, op, **LAB, cycles=(int(round(0.75 * n)), n - int(round(0.75 * n))))[0]
                             for n in (10, 20, 30, 40, 50, 60)}
        # loop 1 optimized with loop 2 kept at the lab 13/12 deg
        s1 = np.array(r["scan_loop1"])
        i, j = np.unravel_index(np.argmax(s1), s1.shape)
        r["best_loop1_lab_loop2"] = dict(t2a=float(GRID[i]), t1a=float(GRID[j]), ground=float(s1[i, j]))
        res["per_set"][name] = r
        print(f"set {name} (-2 {om[2]:.2f}, -1 {om[1]:.2f} kHz): lab {g_lab:.4f}; best same-angle {g_best:.4f} "
              f"at {best}; per-axis {g_pa:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    # cross-evaluation: each set's best settings at every set
    res["cross"] = {s_set: {s: run(psc.TableLibrary(ladders, tables, s), op,
                                   **res["per_set"][s_set]["best_same"])[0] for s in SETS}
                    for s_set in SETS}
    lib = psc.TableLibrary(ladders, tables, HEADLINE)
    best_h = res["per_set"][HEADLINE]["best_same"]
    # initial temperature, headline set
    res["temperature"] = dict(T_uK=list(TEMPS_UK),
                              lab=[run(lib, op, t * 1e-6, **LAB)[0] for t in TEMPS_UK],
                              best=[run(lib, op, t * 1e-6, **best_h)[0] for t in TEMPS_UK], best_angles=best_h)
    # dark-state leakage, headline set
    res["dark_leak"] = {}
    for pdark in DARK:
        op_d = psc.optical_pumping_model(ladders, dark_prob=pdark)
        res["dark_leak"][str(pdark)] = dict(lab=run(lib, op_d, **LAB)[0], best=run(lib, op_d, **best_h)[0])
    # inputs at which the lab sequence gives Du's 0.96 (headline set and 20 kHz)
    res["du_match"] = {}
    for name in (HEADLINE, "20"):
        lb = psc.TableLibrary(ladders, tables, name)
        res["du_match"][name] = dict(
            T_uK=match(lambda t: run(lb, op, t * 1e-6, **LAB)[0], 3.0, 8.0),
            dark=match(lambda d: run(lb, psc.optical_pumping_model(ladders, dark_prob=d), **LAB)[0], 0.0, 0.03))
    # pulses centred on the actual n = k -> 0 lines instead of the programmed offsets
    res["line_centred_lab"] = {}
    for name in (HEADLINE, "20"):
        om = {k: v * 1e3 for k, v in SETS[name].items()}
        lc = psc.StepLibrary(ladders, om, center_mode="line")
        res["line_centred_lab"][name] = dict(line=run(lc, op, **LAB)[0], programmed=res["per_set"][name]["lab"])
    # 3D (exact within model) check of the factorized result, headline set
    hist3, _ = psc.simulate_3d(sequence(**LAB), lib, op)
    res["check_3d_lab"] = dict(set=HEADLINE, ground_3d=hist3[-1]["ground_3d"], total=hist3[-1]["total"],
                               factorized=res["per_set"][HEADLINE]["lab"])
    # single-OP heating (acceptance comparison with Wu p.47 / Kumar p.67, ~7 %)
    heat = {a: op.mixed(a, op.p_k) for a in psc.AXES}
    res["op_heating"] = dict(mean_photons=float(np.arange(op.p_k.size) @ op.p_k),
                             p_leave_ground={a: float(1 - heat[a][0, 0]) for a in psc.AXES},
                             p_leave_ground_3d=float(1 - np.prod([heat[a][0, 0] for a in psc.AXES])))
    with open(Path("data/psc_results.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh)
    print(json.dumps({k: res[k] for k in ("afp_threshold_098", "check_3d_lab", "op_heating", "dark_leak",
                                          "temperature", "du_match", "line_centred_lab")}, indent=1))
    print(json.dumps({s: {k: res["per_set"][s][k] for k in ("lab", "loop1_only_40", "loop2_only_40", "best_same",
                                                            "best_same_ground", "best_per_axis_ground",
                                                            "best_loop1_lab_loop2", "split", "cycles_total")}
                      for s in SETS}, indent=1))
    print(json.dumps(res["cross"], indent=1))
    print(f"done in {time.time() - t0:.0f} s")
