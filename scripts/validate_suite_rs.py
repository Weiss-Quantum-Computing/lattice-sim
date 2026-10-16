"""Validate suite rises on the full anharmonic potentials (split-step) against the
exact-Gaussian harmonic engine, for the measured extinction profile, both
handedness values, |3,-3> and |4,-3>, and trap-frequency scales K0 x {0.85, 1, 1.15}.

Depth scaling of the full tables: V -> V k^2 (trap frequency scales by k).

Usage: python scripts/validate_suite_rs.py SUITE_DIR STEM [STEM ...]
Needs data/x_ERmeas133_h{-1,+1}_10kHz_F{3m-3,4m-3}.npz (build_profile_tables.py).
Writes data/validate_suite_rs.json (merged over runs).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_suite_shape_robust import Scorer, shape_splines  # noqa: E402
from score_suite import read_pair, rise_end_index  # noqa: E402
from lattice_sim.motion.transport1d import PotentialTable, Transport1D  # noqa: E402

SCALES = (0.85, 1.0, 1.15)
TABLE = "data/x_ERmeas133_h{h:+d}_10kHz_F{f}m-3.npz"


def main(folder, stems):
    out_path = Path("data/validate_suite_rs.json")
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    shapes = shape_splines()
    for stem in stems:
        t_us, theta, k0, _ = read_pair(folder, stem)
        i_top = rise_end_index(theta)
        t_r, th_r = t_us[: i_top + 1], theta[: i_top + 1]
        t_r = t_r - t_r[0]
        T = float(t_r[-1])
        res = {}
        for h in (-1, 1):
            sub = {f"model_h{h:+d}": shapes[f"model_h{h:+d}"]}
            harm = Scorer(T, shapes=sub, scales=np.array(SCALES), weights=np.ones(len(SCALES)), k0=k0)
            gauss = harm.infidelity(np.rad2deg(np.interp(harm.t, t_r, th_r)))[0]
            for (shape, state, sc), g in zip(harm.labels, gauss):
                table = PotentialTable.load(TABLE.format(h=h, f=state[1]))
                scaled = PotentialTable(table.thetas, table.x, table.values_hz * (k0 * sc) ** 2)
                ts, p0, _ = Transport1D(scaled).propagate(t_r, th_r, dt_us=0.25, sample_every_us=T)
                key = f"{shape}|{state}|x{sc:.2f}"
                res[key] = dict(split=float(1 - p0[-1]), gauss=float(g))
                print(f"{stem:8s} {key:24s} split-step {1-p0[-1]:.2e}  exact Gaussian {g:.2e}", flush=True)
        results[stem] = res
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    import os
    folder = str(Path(sys.argv[1]).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    main(folder, sys.argv[2:])
