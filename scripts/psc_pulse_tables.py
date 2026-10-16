"""Pulse transfer matrices |U|^2 of the -2 and -1 AFP pulses on every axis
and every rotation 0.5-20 deg, for named sets of per-pulse Rabi frequencies.

Sets (``psc.rabi_sets()``): 'lab' (0.1425 / 0.075 W -> 6.77 / 4.91 kHz for the
-2 / -1 pulse, Coolscan manifest and the DDS pi-power calibration), the lab
powers scaled by 0.25-4 ('lab_P0.25' .. 'lab_P4'), and the uniform
sensitivity scan '10', '14', '20', '40' (both pulses at that Omega0, kHz).
Variants: -2 pulse at HSR -6 kHz; -1 pulse at HSR -4 kHz and -6 kHz; all
centred at the programmed offsets (k x f_a above the theta = 0 carrier).
Tables are keyed by each pulse's own Rabi frequency (``psc.table_key``), so
sets that share a pulse share its table.
Needs data/psc_ladders.npz (scripts/psc_build_ladders.py).
Writes data/psc_pulses.npz (git-ignored, ~4 min per set).
Usage: python scripts/psc_pulse_tables.py [--workers N] [set ...]   (default: all sets)
--workers N runs N processes, one (set, axis) at a time.
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

VARIANTS = ((2, -6e3), (1, -4e3), (1, -6e3))
OUT = Path("data") / "psc_pulses.npz"


def thetas(ladders):
    return sorted(k[1] for k in ladders if k[0] == "x" and isinstance(k[1], float) and k[1] > 0)


def compute(task):
    """Tables of one set on one axis that are not in `have`."""
    name, axis, have = task
    om = psc.rabi_sets()[name]
    ladders = psc.load_ladders()
    out = {}
    for th in thetas(ladders):
        lad = psc.ladder_at(ladders, axis, th)
        for kind, hsr in VARIANTS:
            key = psc.table_key(om[kind], axis, th, kind, hsr)
            if key not in have and key not in out:
                out[key] = psc.pulse_matrix(lad, kind, hsr, om[kind] * 1e3).astype(np.float32)
    return name, axis, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    sets = psc.rabi_sets()
    names = args.sets or list(sets)
    arrays = dict(np.load(OUT)) if OUT.exists() else {}
    have = frozenset(arrays)
    tasks = [(name, axis, have) for name in names for axis in psc.AXES]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for name, axis, out in pool.imap(compute, tasks):
            arrays.update(out)
            om = sets[name]
            print(f"set {name} (-2 {om[2]:.3f} kHz, -1 {om[1]:.3f} kHz), {axis}: {len(out)} new, "
                  f"{time.time() - t0:.0f} s", flush=True)
            if axis == psc.AXES[-1]:
                np.savez_compressed(OUT, **arrays)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({len(arrays)} tables) in {time.time() - t0:.0f} s")
