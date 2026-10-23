"""Export the projection-cooling model for the interactive explorer
(tools/psc_explorer.html): Franck-Condon factors and line positions on a
0.5 deg grid, the AFP transfer matrices on a 1 deg grid (banded, uint16),
optical-pumping heating matrices, and the reference results of
psc_analysis.py, for every Rabi-frequency set of ``psc.rabi_sets()`` (the lab
per-pulse values first). Pulse keys are '{set}_{axis}_{theta:.1f}_{kind}_{hsr}'.

Needs data/psc_ladders.npz, data/psc_pulses.npz, data/psc_results.json.
Writes data/psc_explorer.json (git-ignored).
Usage: python scripts/psc_export_explorer.py
"""

import base64
import json
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cooling import psc  # noqa: E402

N = 50
BAND = 3
VARIANTS = ((2, -6e3), (1, -4e3), (1, -6e3))
SETS = psc.rabi_sets()
PULSE_THETAS = np.arange(1.0, 20.01, 1.0)


def set_label(name, om):
    if name == psc.LAB_SET:
        return f"lab powers: {om[2]:.2f} / {om[1]:.2f} kHz"
    if name.startswith("lab_P"):
        return f"lab powers ×{name[5:]}: {om[2]:.2f} / {om[1]:.2f} kHz"
    return f"both {om[2]:g} kHz"


def b64_u16(a):
    q = np.clip(np.rint(np.asarray(a) * 65535), 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def banded(t, n_full):
    """Four blocks (uu, ul, lu, ll) of |U|^2 [out, in], diagonals -BAND..BAND,
    first N levels: array (4, 2*BAND+1, N) with [blk, d, i] = block[i + d, i]."""
    out = np.zeros((4, 2 * BAND + 1, N))
    blocks = [t[:N, :N], t[:N, n_full:n_full + N], t[n_full:n_full + N, :N], t[n_full:n_full + N, n_full:n_full + N]]
    for b, blk in enumerate(blocks):
        for k, d in enumerate(range(-BAND, BAND + 1)):
            for i in range(N):
                j = i + d
                if 0 <= j < N:
                    out[b, k, i] = blk[j, i]
    return out


if __name__ == "__main__":
    ladders = psc.load_ladders()
    tables = np.load(Path("data") / "psc_pulses.npz")
    results = json.loads(Path("data/psc_results.json").read_text(encoding="utf-8"))
    op = psc.optical_pumping_model(ladders)
    fc_thetas = sorted(k[1] for k in ladders if k[0] == "x" and isinstance(k[1], float))
    exp = dict(N=N, band=BAND, axes=list(psc.AXES), offsets=psc.OFFSET_M1_HZ, fc_thetas=fc_thetas,
               pulse_thetas=PULSE_THETAS.tolist(), omegas=list(SETS),
               sets={name: dict(m2=om[2], m1=om[1], label=set_label(name, om)) for name, om in SETS.items()},
               variants=[f"{k}_{h:g}" for k, h in VARIANTS], fc={}, lines={}, pulses={}, heat={},
               initial_temperature_uK=psc.INITIAL_TEMPERATURE_K * 1e6)
    for axis in psc.AXES:
        rows, lrows = [], []
        for th in fc_thetas:
            lad = psc.ladder_at(ladders, axis, th)
            a2 = lad.overlap ** 2
            m1 = [float(a2[n - 1, n]) for n in range(1, N)]
            m2 = [float(a2[n - 2, n]) for n in range(2, N)]
            off = psc.OFFSET_M1_HZ[axis]
            l1 = [float(lad.e_u[n] - lad.e_l[n - 1] + lad.carrier_shift_hz - off) for n in range(1, N)]
            l2 = [float(lad.e_u[n] - lad.e_l[n - 2] + lad.carrier_shift_hz - 2 * off) for n in range(2, N)]
            rows.append(dict(m1=np.round(m1, 5).tolist(), m2=np.round(m2, 5).tolist(),
                             d_nm=round(lad.displacement_m * 1e9, 2), cs=round(lad.carrier_shift_hz, 1)))
            lrows.append(dict(l1=np.round(l1, 1).tolist(), l2=np.round(l2, 1).tolist()))
        exp["fc"][axis], exp["lines"][axis] = rows, lrows
        heat = op.mixed(axis, op.p_k)[:N, :N]
        exp["heat"][axis] = np.round(heat, 6).tolist()
    for name, om in SETS.items():
        for axis in psc.AXES:
            for th in PULSE_THETAS:
                for kind, hsr in VARIANTS:
                    t = np.asarray(tables[psc.table_key(om[kind], axis, th, kind, hsr)], float)
                    exp["pulses"][f"{name}_{axis}_{th:.1f}_{kind}_{hsr:g}"] = b64_u16(banded(t, t.shape[0] // 2))
    exp["op"] = dict(mean_photons=float(np.arange(op.p_k.size) @ op.p_k),
                     p_k=np.round(op.p_k[:12], 6).tolist())
    exp["reference"] = {om: dict(lab=r["lab"], lab_g3=r["lab_history"]["ground_3d"],
                                 best=r["best_same"], best_ground=r["best_same_ground"],
                                 best_g3=r["best_same_history"]["ground_3d"],
                                 per_axis=r["best_per_axis"], per_axis_ground=r["best_per_axis_ground"],
                                 scan1=np.round(r["scan_loop1"], 5).tolist(), scan2=np.round(r["scan_loop2"], 5).tolist())
                        for om, r in results["per_set"].items()}
    exp["grid_deg"] = results["grid_deg"]
    exp["op_heating"] = results["op_heating"]
    exp["du_match"] = results["du_match"]
    out = Path("data") / "psc_explorer.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(exp, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
