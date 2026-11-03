"""Imaging readout and exposure budget from the 2026-11-02 detuning runs (000-004).

1. Readout: per run, image and neighbour category, the empty and atom peak centres and robust widths
   of the re-thresholded fit heights (scripts/camera_runs.py; the lab tunes the camera cutoffs per
   run, so the valley thresholds are the operating point). Model: atom width w1 = m B, empty width
   w0 = sqrt(f^2 + (k B)^2), with f the readout floor (image 2, category 0, few atoms) and k, m fitted
   on image 1, category 0 of runs 001-004; separation S(B) = B / sqrt(w0^2 + w1^2), limit 1/sqrt(k^2+m^2).
2. Exposure at fixed counts per plane at -40.5 MHz: counts per uW per 100 ms from run 004 (1.75 uW x
   150 ms), with optical saturation P / (1 + P / PSAT); extra thermal hops per image set from the
   2026-10-30 full-depth X fit (0.12 % per 5 s at 2 uW, ln h linear in 1/P with slope 23.3 uW,
   -25.5 MHz), with the -40.5 MHz power converted to -25.5 MHz-equivalent scattering by the ratio of
   counts per saturated uW (run 004 against 7187 counts per uW per 100 ms on 2026-10-29, fitted over
   2.1-3.0 uW and corrected to the saturated axis at 2.5 uW) and the molasses on
   for 9 planes x exposure per set.

Writes data/measured/molasses_imaging_budget_20261102_{widths,exposure}.csv and prints the summary.
Usage: python scripts/molasses_imaging_budget_20261102.py ["~/Desktop/Data/Camera Control/20261102"]
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
RUNS = {"000": -26.5, "001": -26.5, "002": -30.5, "003": -35.5, "004": -40.5}
FIT_RUNS = ("001", "002", "003", "004")       # 000: same settings as 001, 5 h earlier, broader atom peak
DOSE_UW, DOSE_MS = 1.75, 150.0               # imaging in all runs
PSAT_UW = 22.0                               # 2026-10-29 pooled saturation fit
REF_COUNTS_PER_UW_100MS = 7187.0             # -25.5 MHz, 2026-10-29, linear fit B = a P over 2.1-3.0 uW
REF_MID_UW = 2.5                             # middle of that fit range (saturation correction of a)
HOP_REF, HOP_P_REF, HOP_SLOPE_UW = 0.0012, 2.0, 23.3   # per 5 s, full depth X, 2026-10-30
PLANES_PER_SET = 9
TARGET_COUNTS = 15000.0


def peaks(h, cat, thr, img, c):
    v = h[:, img, cr.PLANES][cat[:, img, cr.PLANES] == c]
    t = thr[img, c]
    lo, hi = v[v < t], v[v >= t]
    if len(lo) < 30 or len(hi) < 30:
        return None
    c0, c1 = np.median(lo), np.median(hi)
    w0, w1 = 1.4826 * np.median(np.abs(lo - c0)), 1.4826 * np.median(np.abs(hi - c1))
    return dict(atoms=int(len(hi)), c0=float(c0), w0=float(w0), c1=float(c1), w1=float(w1), bright=float(c1 - c0),
                sep=float((c1 - c0) / np.hypot(w0, w1)), threshold=float(t))


def thermal_hops_per_set(p_eq, t_s):
    return HOP_REF / 5.0 * np.exp(HOP_SLOPE_UW / HOP_P_REF - HOP_SLOPE_UW / p_eq) * t_s


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DATA
    os.chdir(REPO)
    rows = []
    for run, det in RUNS.items():
        h, occ_cam, keep = cr.load(folder, run)
        h = h[keep]
        _, thr, cat = cr.rethreshold(h, occ_cam[keep], cr.cam_thresholds(folder, run))
        for img in (0, 1):
            for c in range(4):
                p = peaks(h, cat, thr, img, c)
                if p:
                    rows.append(dict(run=run, detuning_mhz=det, image=img + 1, category=c,
                                     **{k: (round(v, 3) if k == "sep" else round(v)) for k, v in p.items()}))
    fit = [r for r in rows if r["run"] in FIT_RUNS and r["category"] == 0]
    f = float(np.median([r["w0"] for r in fit if r["image"] == 2]))
    im1 = [r for r in fit if r["image"] == 1]
    k = float(np.mean([np.sqrt(max(r["w0"] ** 2 - f ** 2, 0)) / r["bright"] for r in im1]))
    m = float(np.mean([r["w1"] / r["bright"] for r in im1]))
    sep = lambda b: b / np.sqrt(f ** 2 + (k * b) ** 2 + (m * b) ** 2)  # noqa: E731
    print(f"readout model: floor f = {f:.0f}, k = {k:.3f}, m = {m:.3f}; separation limit {1 / np.hypot(k, m):.2f} sigma")
    for r in rows:
        if r["category"] in (0, 3):
            print(f"  {r['run']} img {r['image']} cat {r['category']}: B {r['bright']} w0 {r['w0']} w1 {r['w1']} "
                  f"(w1/B {r['w1'] / r['bright']:.3f}) sep {r['sep']:.2f} (model {sep(r['bright']):.2f})")
    for b in (5000, 7500, 10000, 15000, 20000, 25000):
        print(f"  model separation at {b} counts: {sep(b):.2f} sigma")

    r004 = next(r for r in rows if r["run"] == "004" and r["image"] == 1 and r["category"] == 0)
    sat = lambda p: p / (1 + p / PSAT_UW)  # noqa: E731
    per_uw_100ms = r004["bright"] / (sat(DOSE_UW) * DOSE_MS / 100)
    ratio = per_uw_100ms / (REF_COUNTS_PER_UW_100MS * (1 + REF_MID_UW / PSAT_UW))
    print(f"-40.5 MHz: {per_uw_100ms:.0f} counts per saturated-uW per 100 ms; scattering per uW {ratio:.2f} of -25.5 MHz (10-29)")
    erow = []
    for p in (1.5, 1.75, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5):
        t_ms = TARGET_COUNTS / (per_uw_100ms * sat(p)) * 100
        hops = thermal_hops_per_set(ratio * p, PLANES_PER_SET * t_ms / 1000)
        erow.append(dict(detuning_mhz=-40.5, power_uw=p, counts=TARGET_COUNTS, exposure_ms=round(t_ms, 1),
                         p_equiv_uw=round(ratio * p, 3), counts_per_100ms=round(TARGET_COUNTS / t_ms * 100),
                         thermal_hops_per_set=f"{hops:.2e}"))
        print(f"  {p:.2f} uW: {t_ms:5.1f} ms per plane for {TARGET_COUNTS:.0f} counts, "
              f"P_eq {ratio * p:.2f} uW, extra thermal hops per set {hops:.1e}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (("widths", rows), ("exposure", erow)):
        with open(OUT / f"molasses_imaging_budget_20261102_{name}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# written by scripts/molasses_imaging_budget_20261102.py from {folder}\n")
            w = csv.DictWriter(fh, list(rr[0]))
            w.writeheader()
            w.writerows(rr)


if __name__ == "__main__":
    main()
