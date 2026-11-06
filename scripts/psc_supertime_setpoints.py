"""Cooling-block Trek setpoints: the current Supertime sequence in EO volts,
its rotation angles, the cooling model at those angles, and recommended
setpoints in EO volts.

Source sequence (supertime-analysis, 2026-10-28):
  sequences/2026/Cooling(30+10)_3ms_X2detect_2EOx_checkcooling_9planes_changed imaging_newramps.dt4
Per axis step: "Begin/End rotate lattice polarization" (0.1 ms linear, 20 us
steps) to the -2 setpoint; -2 AFP (DDS trigger, 3 ms); "EO ramp" (0.1 ms) to
the -1 setpoint; -1 AFP (3 ms); "Begin/End Restore lattice polarization"
(0.1 ms) to rest; OP 0.1 ms. Loop 1: 30 cycles, loop 2: 10 cycles, order
x, y, z; full lattice depth (4.2 V setpoints). EO volts (user units of the
Trek conversion VIs):

            rest     loop 1 (-2 / -1)   loop 2 (-2 / -1)
  X#1, X#2  0.012    0.512 / 0.312      0.650 / 0.512   (both channels equal)
  Y#1      -0.051    0.449 / 0.249      0.600 / 0.449
  Z#1       0.056    0.556 / 0.356      0.670 / 0.556

Angle calibration. The polarization rotation is linear in the EO voltage
(Pockels cell + quarter-wave plate). X has two EOs in series:
theta_x = 90 deg (dV1 / V90_X1 + dV2 / V90_X2), with V90 = 11.412 and 9.99 EO V
(SEQUENCE_VALUES_BRIEF, the Trek conversion VIs), dV from rest. The y and z
EOs (one each) have no documented 90-degree point. Two hypotheses:
  'single' : each y/z EO has the mean x-EO calibration (V90 = 10.70 EO V), so
             a y/z step rotates half as far as the same step on x (two EOs);
  'same'   : the y/z steps rotate as far as the same step on x.
The measured per-axis excitation is used to choose between them.

Needs data/psc_ladders.npz and the 'lab' set in data/psc_pulses.npz.
Writes data/psc_supertime_setpoints.json.
Usage: python scripts/psc_supertime_setpoints.py
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

V90_X1, V90_X2 = 11.412, 9.99
V90_YZ_SINGLE = 0.5 * (V90_X1 + V90_X2)      # ASSUMPTION ('single'); data request 32
REST = {"x": 0.012, "y": -0.051, "z": 0.056}
CURRENT = {   # EO volts: (loop-1 -2, loop-1 -1, loop-2 -2, loop-2 -1)
    "x": (0.512, 0.312, 0.650, 0.512),
    "y": (0.449, 0.249, 0.600, 0.449),
    "z": (0.556, 0.356, 0.670, 0.556),
}
MEASURED_PCT = np.array([3.1, 3.8, 1.2]) - 0.8   # x, y, z non-ground, background 0.8 % removed
DARK = (0.0, 0.01, 0.015)
GRID = np.arange(2.0, 14.01, 0.5)               # degrees searched per setpoint
CYCLES = (30, 10)


def deg_per_volt(axis, hyp):
    x = 90.0 * (1.0 / V90_X1 + 1.0 / V90_X2)
    if axis == "x" or hyp == "same":
        return x
    return 90.0 / V90_YZ_SINGLE


def to_deg(axis, v, hyp):
    return (v - REST[axis]) * deg_per_volt(axis, hyp)


def to_volt(axis, deg, hyp):
    return REST[axis] + deg / deg_per_volt(axis, hyp)


def snap(deg):
    return float(np.clip(np.round(deg * 2) / 2, 0.5, 20.0))


def sequence(angles, cycles=CYCLES):
    """angles[axis] = (t2a, t1a, t2b, t1b) in degrees."""
    return [psc.Loop(cycles[0], {a: angles[a][0] for a in psc.AXES}, {a: angles[a][1] for a in psc.AXES}, -6e3, -4e3),
            psc.Loop(cycles[1], {a: angles[a][2] for a in psc.AXES}, {a: angles[a][3] for a in psc.AXES}, -6e3, -6e3)]


def evaluate(lib, op, angles, cycles=CYCLES, temperature_k=5e-6):
    h = psc.simulate_factorized(sequence(angles, cycles), lib, op, p0=psc.initial_state(temperature_k=temperature_k))
    ex = np.array([1 - h[-1][a][0] for a in psc.AXES]) * 100
    return float(np.prod(1 - ex / 100)), ex


def optimize(lib, op, start, cycles=CYCLES, sweeps=2):
    """Coordinate search, per axis, over (loop-1 -2, loop-1 -1) then
    (loop-2 -2, loop-2 -1) on GRID. Axes couple only through OP recoil, so
    per-axis passes converge in two sweeps."""
    best = {a: list(v) for a, v in start.items()}
    g, _ = evaluate(lib, op, best, cycles)
    for _ in range(sweeps):
        for a in psc.AXES:
            for i, j in ((0, 1), (2, 3)):
                for u, w in itertools.product(GRID, GRID):
                    trial = {k: list(v) for k, v in best.items()}
                    trial[a][i], trial[a][j] = float(u), float(w)
                    gt, _ = evaluate(lib, op, trial, cycles)
                    if gt > g + 1e-6:
                        g, best = gt, trial
    return best, g


if __name__ == "__main__":
    ladders = psc.load_ladders()
    tables = dict(np.load(Path("data") / "psc_pulses.npz"))
    lib = psc.TableLibrary(ladders, tables, "lab")
    out = dict(current_volts=CURRENT, rest=REST, measured_pct=MEASURED_PCT.tolist(), hypotheses={})
    check, _ = evaluate(lib, psc.optical_pumping_model(ladders),
                        {a: (12.0, 6.0, 13.0, 12.0) for a in psc.AXES})
    print(f"check: lab sequence at 12/6, 13/12 deg = {check:.4f} (findings 2026-10-15: 0.9623)")
    for hyp in ("single", "same"):
        cur = {a: [snap(to_deg(a, v, hyp)) for v in CURRENT[a]] for a in psc.AXES}
        print(f"\n=== hypothesis '{hyp}': current angles (deg, 0.5 grid) {cur}")
        r = dict(current_deg=cur, per_dark={})
        for d in DARK:
            op = psc.optical_pumping_model(ladders, dark_prob=d)
            g_cur, ex_cur = evaluate(lib, op, cur)
            best, g_best = optimize(lib, op, cur)
            g_best_50, ex_best_50 = evaluate(lib, op, best, (40, 10))
            _, ex_best = evaluate(lib, op, best)
            volts = {a: [round(to_volt(a, t, hyp), 3) for t in best[a]] for a in psc.AXES}
            rms = float(np.sqrt(np.mean((ex_cur - MEASURED_PCT) ** 2)))
            r["per_dark"][str(d)] = dict(current_ground=g_cur, current_excitation_pct=ex_cur.tolist(), rms_vs_measured=rms,
                                         best_deg=best, best_volts=volts, best_ground=g_best,
                                         best_excitation_pct=ex_best.tolist(),
                                         best_40_10_ground=g_best_50, best_40_10_excitation_pct=ex_best_50.tolist())
            print(f"dark {100*d:.1f} %/OP: current 3D {g_cur:.4f}, excitation {np.round(ex_cur, 2)} "
                  f"(measured {MEASURED_PCT}, rms {rms:.2f})")
            print(f"   optimum 3D {g_best:.4f} {np.round(ex_best, 2)}; 40+10 cycles {g_best_50:.4f}")
            print(f"   optimum deg {best}")
            print(f"   optimum EO V {volts}")
        out["hypotheses"][hyp] = r

    # Candidate setpoint sets (EO volts) evaluated under both calibrations:
    # the 1 %/OP optimum of each hypothesis, and partial changes.
    opt = {h: out["hypotheses"][h]["per_dark"]["0.01"]["best_volts"] for h in ("single", "same")}
    cur_v = {a: list(v) for a, v in CURRENT.items()}
    cands = {
        "current": cur_v,
        "x only (1 %/OP optimum)": dict(cur_v, x=opt["single"]["x"]),
        "x + y ('single' optimum), z current": dict(cur_v, x=opt["single"]["x"], y=opt["single"]["y"]),
        "'single' optimum (x, y, z)": opt["single"],
        "'same' optimum (x, y, z)": opt["same"],
    }
    out["candidates"] = {}
    print("\n=== candidate EO-volt sets under both calibrations (3D ground fraction, 30+10 cycles)")
    for name, volts in cands.items():
        row = {}
        for hyp in ("single", "same"):
            ang = {a: [snap(to_deg(a, v, hyp)) for v in volts[a]] for a in psc.AXES}
            for d in (0.0, 0.01):
                g, ex = evaluate(lib, psc.optical_pumping_model(ladders, dark_prob=d), ang)
                row[f"{hyp}_{d}"] = dict(ground=g, excitation_pct=ex.tolist())
        g50 = {f"{hyp}_{d}": evaluate(lib, psc.optical_pumping_model(ladders, dark_prob=d),
                                      {a: [snap(to_deg(a, v, hyp)) for v in volts[a]] for a in psc.AXES}, (40, 10))[0]
               for hyp in ("single", "same") for d in (0.0, 0.01)}
        out["candidates"][name] = dict(volts=volts, results=row, results_40_10=g50)
        print(f"{name:40s} " + "  ".join(f"{k}: {v['ground']:.4f}" for k, v in row.items())
              + "   | 40+10: " + "  ".join(f"{v:.4f}" for v in g50.values()))
    Path("data/psc_supertime_setpoints.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
