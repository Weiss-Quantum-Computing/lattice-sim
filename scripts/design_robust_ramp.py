"""Design a transport ramp that is robust across model scenarios.

Usage: python scripts/design_robust_ramp.py PAIR DURATION_US TERMS OUT_STEM
  PAIR = pm3 | pm1. Scenarios: every data/harm_F3m{+/-}{3|1}_*.npz file,
  plus +/-5 % trap-frequency scale on the extreme-ER files.
"""

import glob
import sys
import time

import numpy as np

from lattice_sim.control.ramp_optimizer import RampProblem, save_ramp
from lattice_sim.motion.fast_gaussian import HarmonicScenario
from lattice_sim.motion.transport1d import load_waveform

M = {"pm3": 3, "pm1": 1}


def scenarios(pair):
    m = M[pair]
    files = sorted(glob.glob(f"data/harm_F3m-{m}_*.npz") + glob.glob(f"data/harm_F3m+{m}_*.npz"))
    out = [HarmonicScenario.load(f) for f in files]
    for f in files:
        if "ERinf" in f:
            out += [HarmonicScenario.load(f, 0.95), HarmonicScenario.load(f, 1.05)]
    return out


def main(pair, duration, terms, stem, rounds=4):
    sc = scenarios(pair)
    print(f"{len(sc)} scenarios")
    prob = RampProblem(sc, duration_us=duration, terms=terms)
    t, th, volts = load_waveform("data/K06_5ms_capon_polyband.csv")
    if duration == t[-1]:
        bm, fin = prob.score_waveform(t, th, volts)
        print(f"K06 as written: worst band P0 {bm.min():.6f}")
        for s_, b_, f_ in zip(sc, bm, fin):
            print(f"    {s_.name:36s} band {b_:.6f}  final {f_:.6f}")
    try:
        seed = np.load(f"data/{stem}_coeffs.npy")
        if seed.size != terms:
            raise FileNotFoundError
        print("restarting from the previous result")
    except FileNotFoundError:
        seed = prob.fit(t * duration / t[-1], th)
    started = time.time()
    a, hist = seed, []
    for round_ in range(rounds):
        a, score, result, h = prob.optimize(a, max_iter=60, verbose=False)
        hist += h
        print(f"  round {round_+1}: worst band P0 {score:.6f} ({result.message})", flush=True)
    print(f"\n({len(hist)} iterations, {time.time()-started:.0f}s)")
    print(prob.report(a, "optimized"))
    np.save(f"data/{stem}_coeffs.npy", a)
    save_ramp(f"data/{stem}.csv", prob, a)
    print(f"wrote data/{stem}.csv")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
