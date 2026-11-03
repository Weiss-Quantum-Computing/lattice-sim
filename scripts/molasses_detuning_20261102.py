"""Molasses detuning scan during a 500 ms hold in a shallow X lattice (2026-11-02, runs 000-004).

Hopping sequence, both image sets in X (1.75 uW x 150 ms), 500 ms hold between them with the X lattice
at 2.7 V (about quarter depth; the run log says full depth, corrected by Maarten 2026-11-02), molasses
power scanned 0-5 uW (51 points x 2 sweeps) at four detunings set by the AOM (lock at -125.5 MHz from
F'=5): 99, 95, 90, 85 MHz -> -26.5, -30.5, -35.5, -40.5 MHz. The same AOM sets the imaging detuning,
so image brightness at fixed dose is measured at each detuning as well. Run 000: -26.5 MHz, 0 uW,
20 shots (imaging-only reference). Run 004's log line says "aom 99MHz" with "detuning 40MHz"; the
85 MHz AOM setting is assumed (the brightness ordering agrees).

Per run: occupancy re-thresholded as scripts/camera_runs.py; brightness, separation and miscalls of
image 1 (image 0 in this script's numbering is the first image set). Per scan point: atoms of image 1,
atoms of image 2, losses (filled -> empty), hops (empty -> filled), nearest-neighbour hop-loss pairs
by direction. Run 002 has a camera frame slip at shots 23-25 (shot 23 empty in both images, 24-25
misregistered), after which camera shot i belongs to scan point (i + 1) mod 51: the 5.0 uW label of
shot 50 holds a surviving 0 uW shot and the sweep-2 survival curve leads sweep 1 by one point (sum of
squared differences 0.076 at lag -1 against 0.237 at lag 0; the other runs are best at lag 0). Shots
23-25 and the aborted end of the acquisition (99-101) are dropped.

Writes data/measured/molasses_detuning_20261102_{points,runs}.csv and prints a summary.
Usage: python scripts/molasses_detuning_20261102.py ["~/Desktop/Data/Camera Control/20261102"]
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import camera_runs as cr  # noqa: E402

DATA = "~/Desktop/Data/Camera Control/20261102"
OUT = REPO / "data" / "measured"
NPTS, P0, P1 = 51, 0.0, 5.0
GAMMA_MHZ = 5.234
UW_TO_MW_CM2_PER_BEAM = 0.71                 # Li's 140 um waist, fiber total -> per beam
ISAT = 1.1023                                # mW/cm^2, cycling transition
RUNS = {  # run: (AOM MHz, detuning MHz, scanned)
    "000": (99, -26.5, False),
    "001": (99, -26.5, True),
    "002": (95, -30.5, True),
    "003": (90, -35.5, True),
    "004": (85, -40.5, True),
}
BAD_SHOTS = {"002": {23, 24, 25, 99, 100, 101}}
SHIFT_FROM = {"002": (26, 1)}                # (first shot, point offset) after the frame slip
DIRS = [n[3] for n in cr.NEIGH]


def scattering_rate(p_uw, det_mhz):
    """Two-level rate for six beams (s^-1), nominal detuning: (Gamma/2) s / (1 + s + 4 d^2 / Gamma^2)."""
    s = 6 * UW_TO_MW_CM2_PER_BEAM * p_uw / ISAT
    return np.pi * GAMMA_MHZ * 1e6 * s / (1 + s + 4 * (det_mhz / GAMMA_MHZ) ** 2)


def run_arrays(folder, run):
    h, occ_cam, keep = cr.load(folder, run)
    occ, thr, cat = cr.rethreshold(h[keep], occ_cam[keep], cr.cam_thresholds(folder, run))
    idx = np.nonzero(keep)[0]
    o1, o2 = occ[:, 0, cr.PLANES], occ[:, 1, cr.PLANES]
    hop, loss = (o1 == 0) & (o2 == 1), (o1 == 1) & (o2 == 0)
    first, off = SHIFT_FROM.get(run, (len(h) + 1, 0))
    point = np.where(idx >= first, idx + off, idx) % NPTS
    use = ~np.isin(idx, list(BAD_SHOTS.get(run, ())))
    return dict(h=h[keep], cat=cat, thr=thr, o1=o1, o2=o2, hop=hop, loss=loss, point=point, use=use)


def crossing(x, y, level, rising):
    """first x where y crosses level (linear interpolation), searching from the peak outward."""
    k = int(np.argmax(y))
    rng = range(k, 0, -1) if rising else range(k, len(y) - 1)
    for i in rng:
        j = i - 1 if rising else i + 1
        if (y[j] - level) * (y[i] - level) <= 0 and y[j] != y[i]:
            return float(x[j] + (level - y[j]) * (x[i] - x[j]) / (y[i] - y[j]))
    return np.nan


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DATA
    os.chdir(REPO)
    grid = np.linspace(P0, P1, NPTS)
    prow, rrow = [], []
    for run, (aom, det, scanned) in RUNS.items():
        a = run_arrays(folder, run)
        n = len(a["o1"])
        per = lambda m: m.reshape(n, -1).sum(1)  # noqa: E731
        n1, n2, nh, nl = per(a["o1"]), per(a["o2"]), per(a["hop"]), per(a["loss"])
        ro = cr.readout(a["h"][a["use"]], a["cat"][a["use"]], a["thr"])
        pts = range(NPTS) if scanned else [0]
        stay = []
        for k in pts:
            m = a["use"] & ((a["point"] == k) if scanned else True)
            real, chance = cr.moves(a["hop"], a["loss"], list(np.nonzero(m)[0]))
            p = grid[k] if scanned else 0.0
            row = dict(run=run, aom_mhz=aom, detuning_mhz=det, power_uw=round(p, 2),
                       scatter_per_s=round(float(scattering_rate(p, det))), shots=int(m.sum()),
                       atoms1=int(n1[m].sum()), atoms2=int(n2[m].sum()), losses=int(nl[m].sum()),
                       hops=int(nh[m].sum()))
            row.update({f"pairs_{d}": real[d] for d in DIRS})
            row.update({f"chance_{d}": round(chance[d], 2) for d in DIRS})
            prow.append(row)
            stay.append(1 - row["losses"] / max(row["atoms1"], 1))
        r = dict(run=run, aom_mhz=aom, detuning_mhz=det, shots_used=int(a["use"].sum()),
                 bright=round(ro["bright"]), sep=round(ro["sep"], 2), miscall=f"{ro['miscall']:.1e}")
        rows = [q for q in prow if q["run"] == run]
        if scanned:
            s = np.convolve(stay, np.ones(3) / 3, mode="same")
            s[0], s[-1] = stay[0], stay[-1]
            k = int(np.argmax(s[1:])) + 1
            win = [q for q in rows if 0.3 <= q["power_uw"] <= 3.2]
            r.update(peak_power_uw=round(grid[k], 2), peak_stay_3pt=round(float(s[k]), 3),
                     low_edge_uw=round(crossing(grid[1:], s[1:], 0.5, True), 2),
                     high_edge_uw=round(crossing(grid[1:], s[1:], 0.5, False), 2),
                     plateau90_lo_uw=round(crossing(grid[1:], s[1:], 0.9, True), 2),
                     plateau90_hi_uw=round(crossing(grid[1:], s[1:], 0.9, False), 2),
                     zero_uw_stay=round(stay[0], 3), zero_uw_atoms=rows[0]["atoms1"],
                     zero_uw_hops_per_atom=round(rows[0]["hops"] / max(rows[0]["atoms1"], 1), 4))
            near = [q for q in rows if abs(q["power_uw"] - grid[k]) < 0.15]
            r["peak_hops_per_atom"] = round(sum(q["hops"] for q in near) / sum(q["atoms1"] for q in near), 4)
            for d in DIRS:
                r[f"win_pairs_{d}"] = sum(q[f"pairs_{d}"] for q in win)
                r[f"win_chance_{d}"] = round(sum(q[f"chance_{d}"] for q in win), 1)
        else:
            r.update(zero_uw_stay=round(stay[0], 3), zero_uw_atoms=rows[0]["atoms1"],
                     zero_uw_hops_per_atom=round(rows[0]["hops"] / max(rows[0]["atoms1"], 1), 4))
        rrow.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("points", prow), ("runs", rrow)):
        keys = list(dict.fromkeys(k for q in rows for k in q))
        with open(OUT / f"molasses_detuning_20261102_{name}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# written by scripts/molasses_detuning_20261102.py from {folder}\n")
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)
    for r in rrow:
        print({k: v for k, v in r.items() if not k.startswith("win_chance")})


if __name__ == "__main__":
    main()
