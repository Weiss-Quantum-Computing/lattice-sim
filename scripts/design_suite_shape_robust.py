"""Design a suite transport ramp that is robust to the shape of x0(theta), f(theta).

The 2026-09-28 ramp suite (Ramps/suite-20260928) was designed on the
Transport-Optimizer ER316 calibration only, robust over a uniform trap-frequency
scale. The first-principles model with the measured extinction profile differs
from the calibration by a smooth dipole either side of pi/2 (findings,
2026-10-08), and the suite's fast ramps excite |3,-3> by 1-3 % under it.

This script searches the suite's own parametrization (``theta_of``: an angle
warp exp(sum_j c_j cos(j pi s)) composed with a 7th-order smoothstep in time,
see suite designs/source/globalopt.py) for a rise that keeps the worst
1 - P0 at the end of the rise small over

    shapes  {ER316 calibration, model h = -1, model h = +1}
  x states  {|3,-3> (F3), |4,-3> (F4)}
  x scales  K0 x {0.80 (weight 0.3), 0.85, 0.92, 1.00, 1.10, 1.15},

scored with the exact-Gaussian engine (``motion.fast_gaussian``), which
includes the breathing (squeezing) excitation that the suite's classical
sloshing cost does not. Per-EOM slew (PAR split) is penalized above SLEW_CAP.

Usage: python scripts/design_suite_shape_robust.py T_US [K] [SEED] [MAXITER]
Writes data/suite_rs_K0.94_<T>_K<K>_s<seed>.json (coefficients and scores).
Needs data/suite_harm_ERmeas133_h{-1,+1}.npz (built by score_suite.py).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_er316_calibration import load as load_calibration  # noqa: E402
from lattice_sim.motion.fast_gaussian import propagate_scenarios  # noqa: E402

K0 = 0.94                      # quarter intensity / calibration trap frequency (suite README)
THETA_W_DEG = 180.0
SCALES = np.array([0.80, 0.85, 0.92, 1.00, 1.10, 1.15])
WEIGHTS = np.array([0.3, 1.0, 1.0, 1.0, 1.0, 1.0])   # x0.80 (planes 7-8) counted at 30 %, as in the suite
SLEW_CAP = 9.0                 # V/us per EOM; 10 V/us at the assumed 200 pF, N15 runs at 9.3
V90 = {"X1": 5128.3, "X2": 5137.4}
DT_US = 0.5
STATES = ("F3", "F4")


# ------------------------------------------------------------- suite shape
def smooth7(u):
    return u ** 4 * (35 - 84 * u + 70 * u ** 2 - 20 * u ** 3)


def theta_of(c, thw=THETA_W_DEG, n=500):
    """The suite's rise theta(t) in degrees on n points (globalopt.theta_of)."""
    s = np.linspace(0, 1, n)
    j = np.arange(1, len(c) + 1)
    w = np.exp(np.clip(np.cos(np.pi * np.outer(s, j)) @ c, -30, 30))
    F = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
    F /= F[-1]
    return thw * np.interp(smooth7(np.linspace(0, 1, n)), s, F)


def slew_par(th_deg, t_us):
    """Peak per-EOM slew (V/us) for the PAR split (each EOM takes theta/2)."""
    return np.abs(np.diff(th_deg)).max() / (t_us / (th_deg.size - 1)) * max(V90.values()) / 180.0


# ------------------------------------------------------------- scenarios
def shape_splines():
    """{shape: {state: (x0 spline, f spline)}} with theta in rad."""
    out = {"cal": {}}
    for name in STATES:
        frac, x, f = load_calibration("data/reference", name)
        out["cal"][name] = (CubicSpline(np.pi * frac, np.abs(x) * 1e-6), CubicSpline(np.pi * frac, f * 1e3))
    for h in (-1, 1):
        d = np.load(f"data/suite_harm_ERmeas133_h{h:+d}.npz")
        out[f"model_h{h:+d}"] = {n: (CubicSpline(d["theta"], d[f"x_{n}"]), CubicSpline(d["theta"], d[f"f_{n}"]))
                                 for n in STATES}
    return out


class Scorer:
    def __init__(self, t_us, shapes=None, scales=SCALES, weights=WEIGHTS, k0=K0):
        self.t_us = t_us
        self.n = int(round(t_us / DT_US)) + 1
        self.t = np.arange(self.n) * DT_US
        self.shapes = shape_splines() if shapes is None else shapes
        self.labels, self.w = [], []
        for shape, sp in self.shapes.items():
            for state in STATES:
                for sc, wt in zip(scales, weights):
                    self.labels.append((shape, state, float(sc)))
                    self.w.append(wt)
        self.w = np.array(self.w)
        self.scales, self.k0 = np.asarray(scales), k0

    def infidelity(self, thetas_deg):
        """1 - P0 at the end of the rise, shape (n_candidates, n_scenarios)."""
        thetas = np.deg2rad(np.atleast_2d(thetas_deg))
        x0, om = [], []
        for th in thetas:
            for shape, sp in self.shapes.items():
                for state in STATES:
                    cx, cf = sp[state]
                    xx, ff = cx(th), cf(th)
                    for sc in self.scales:
                        x0.append(xx)
                        om.append(2 * np.pi * ff * self.k0 * sc)
        p0 = propagate_scenarios(self.t, np.array(x0), np.array(om), sample_every=self.n - 1)[:, -1]
        return (1 - p0).reshape(thetas.shape[0], -1)

    def cost(self, C):
        """Vectorized cost; C has shape (K, n_candidates) as scipy passes it."""
        cands = np.atleast_2d(C).T
        ths = np.array([theta_of(c, n=self.n) for c in cands])
        inf = self.infidelity(ths)
        worst = (inf * self.w).max(axis=1)
        slew = np.array([slew_par(th, self.t_us) for th in ths])
        return np.log10(worst + 1e-10) + 0.3 * np.maximum(0.0, slew - SLEW_CAP) ** 2

    def report(self, c):
        th = theta_of(c, n=self.n)
        inf = self.infidelity(th)[0]
        rows = {f"{s}|{st}|x{sc:.2f}": float(v) for (s, st, sc), v in zip(self.labels, inf)}
        robust = {s: float(max(v for (ss, st, sc), v in zip(self.labels, inf) if ss == s and sc >= 0.85))
                  for s in self.shapes}
        x080 = {s: float(max(v for (ss, st, sc), v in zip(self.labels, inf) if ss == s and sc < 0.85))
                for s in self.shapes}
        return dict(per_scenario=rows, worst_robust=robust, worst_x080=x080, slew_V_per_us=float(slew_par(th, self.t_us)))


def main(t_us, K=12, seed=0, maxiter=300):
    os.chdir(Path(__file__).resolve().parents[1])
    sc = Scorer(t_us)
    bounds = [(-3.0 / j, 3.0 / j) for j in range(1, K + 1)]
    rng = np.random.default_rng(seed)
    pop = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(12 * K, K))
    seeds = []
    seed_files = sorted(Path("data").glob(f"glob2_K{K0:.2f}_{int(t_us)}_180_K{K}_s*.json"))  # suite design
    seed_files += [Path(f) for f in os.environ.get("SEEDS", "").split(",") if f]   # e.g. a shorter robust design
    for fn in seed_files:
        c = np.array(json.load(open(fn, encoding="utf-8"))["c"])[:K]
        seeds.append(np.r_[c, np.zeros(K - c.size)])      # pad a lower-order design with zeros
    for i, c in enumerate(seeds):
        pop[i] = c
    if seeds:
        r0 = sc.report(seeds[0])
        print(f"suite seed: worst {r0['worst_robust']}  slew {r0['slew_V_per_us']:.2f}", flush=True)
    started = time.time()
    gen = [0]

    def cb(xk, convergence=None):
        gen[0] += 1
        if gen[0] % 20 == 0:
            print(f"  gen {gen[0]:4d}  best cost {sc.cost(xk[:, None])[0]:.3f}  ({time.time()-started:.0f} s)", flush=True)

    r = differential_evolution(lambda C: sc.cost(C), bounds, init=pop, maxiter=maxiter, tol=1e-7,
                               mutation=(0.5, 1.0), recombination=0.7, seed=seed, polish=False,
                               vectorized=True, updating="deferred", callback=cb)
    p = minimize(lambda c: sc.cost(c[:, None])[0], r.x, method="Powell",
                 options=dict(maxiter=6000, xtol=1e-5, ftol=1e-7))
    c = p.x if p.fun < r.fun else r.x
    rep = sc.report(c)
    print(f"T {t_us:.0f} us K {K} seed {seed}: DE {r.fun:.3f} -> polish {p.fun:.3f} in {time.time()-started:.0f} s; "
          f"worst robust {rep['worst_robust']} x0.80 {rep['worst_x080']} slew {rep['slew_V_per_us']:.2f} V/us", flush=True)
    out = Path("data") / f"suite_rs_K{K0:.2f}_{int(t_us)}_K{K}_s{seed}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(K0=K0, T_us=t_us, theta_w=THETA_W_DEG, K=K, slew_cap=SLEW_CAP, c=list(map(float, c)),
                       cost=float(min(p.fun, r.fun)), **rep), fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), int(a[1]) if len(a) > 1 else 12, int(a[2]) if len(a) > 2 else 0,
         int(a[3]) if len(a) > 3 else 300)
