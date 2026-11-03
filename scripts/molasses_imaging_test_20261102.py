"""X imaging at fixed dose and the dark lifetime at full depth, -40.5 MHz (2026-11-02/03, runs 005-011).

Imaging test (the settings proposed by scripts/molasses_imaging_budget_20261102.py): both image sets in X
at full depth, 50 ms dark between them, AOM 85 MHz (-40.5 MHz), 50 shots each:
005 1.75 uW x 150 ms (reference), 008 2.0 x 130, 009 2.2 x 120, 010 2.5 x 110, 011 3.0 x 95.
Per run (re-thresholded as scripts/camera_runs.py, which drops shots with > 50 'hops'): brightness,
separation, miscalls, peak widths (image 1, category 0), hops and losses per atom of image 1 with
Poisson / binomial errors, nearest-neighbour hop-loss pairs, and the thermal hops per image set the
budget model predicts (same constants).

Lifetime: X at full depth, dark hold 0.05-58.05 s in 2 s steps, one shot per point; 006 repump on
during the hold (re-thresholded here), 007 repump off. 007's .atom/.hist were overwritten by a
one-shot acquisition after the camera analysed the 30-shot run, so 007 uses the camera's per-point
table (007processed.txt, cutoffs 6500/8000/8000/9500). Fit: kept(t) = A exp(-t / tau), weighted.

Writes data/measured/molasses_imaging_test_20261102_{imaging,lifetime}.csv and prints the summary.
Usage: python scripts/molasses_imaging_test_20261102.py ["~/Desktop/Data/Camera Control/20261102"]
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import camera_runs as cr  # noqa: E402
import molasses_imaging_budget_20261102 as bud  # noqa: E402

DATA = "~/Desktop/Data/Camera Control/20261102"
OUT = REPO / "data" / "measured"
IMAGING = {"005": (1.75, 150.0), "008": (2.0, 130.0), "009": (2.2, 120.0), "010": (2.5, 110.0), "011": (3.0, 95.0)}
SCATTER_RATIO = 0.774                       # -40.5 vs -25.5 MHz per saturated uW (budget script)
LIFETIME = {"006": "repump on", "007": "repump off"}
T0, T1, NT = 0.05, 58.05, 30


def expdecay(t, a, tau):
    return a * np.exp(-t / tau)


def heating(t, a, tau_bg, t_h):
    """background loss plus a loss that grows with time (heating towards boil-off), Gaussian in t."""
    return a * np.exp(-t / tau_bg - (t / t_h) ** 2)


def lifetime_points(folder, run):
    if run == "007":
        with open(Path(folder).expanduser() / "007processed.txt", encoding="utf-8") as fh:
            rows = [line.split("\t") for line in fh.read().splitlines()[1:] if line.strip()]
        return [(float(r[0]), int(r[1]), int(r[2]), None) for r in rows]
    res = cr.analyse(folder, run, scan=(T0, T1, NT))
    return [(p["x"], p["atoms1"], p["atoms1"] - p["losses"], p["hops"]) for p in res["points"]]


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DATA
    os.chdir(REPO)
    irow = []
    for run, (p, t) in IMAGING.items():
        r = cr.analyse(folder, run)
        h, occ_cam, keep = cr.load(folder, run)
        _, thr, cat = cr.rethreshold(h[keep], occ_cam[keep], cr.cam_thresholds(folder, run))
        pk = bud.peaks(h[keep], cat, thr, 0, 0)
        n = r["atoms1"]
        pred = bud.thermal_hops_per_set(SCATTER_RATIO * p, bud.PLANES_PER_SET * t / 1000)
        irow.append(dict(run=run, power_uw=p, exposure_ms=t, dose_uw_s=round(p * t / 1000, 4),
                         shots_used=r["shots_used"], bright=round(r["bright"]),
                         bright_per_sat_uw_s=round(r["bright"] / (p / (1 + p / bud.PSAT_UW) * t / 1000)),
                         sep=round(r["sep"], 2), miscall=f"{r['miscall']:.1e}", w0=round(pk["w0"]), w1=round(pk["w1"]),
                         atoms1=n, hops=r["hops"], hops_pct=round(100 * r["hops"] / n, 3),
                         hops_err_pct=round(100 * np.sqrt(max(r["hops"], 1)) / n, 3), losses=r["losses"],
                         losses_pct=round(100 * r["losses"] / n, 2),
                         losses_err_pct=round(100 * np.sqrt(r["losses"] * (1 - r["losses"] / n)) / n, 2),
                         nn_pairs=sum(r["pairs"].values()), nn_chance=round(sum(r["pairs_chance"].values()), 1),
                         pairs_z=r["pairs"]["+z"] + r["pairs"]["-z"], pairs_minus_z=r["pairs"]["-z"],
                         model_thermal_hops_pct=round(100 * pred, 4)))
    for q in irow:
        print(f"{q['run']} {q['power_uw']} uW x {q['exposure_ms']:.0f} ms: B {q['bright']} "
              f"({q['bright_per_sat_uw_s']} per sat-uW s) sep {q['sep']} w1/B {q['w1'] / q['bright']:.3f} miscall {q['miscall']} | "
              f"hops {q['hops']}/{q['atoms1']} = {q['hops_pct']:.2f} +- {q['hops_err_pct']:.2f} % (model thermal "
              f"{q['model_thermal_hops_pct']:.3f} %), NN pairs {q['nn_pairs']} (chance {q['nn_chance']}, -z {q['pairs_minus_z']}) | "
              f"losses {q['losses_pct']:.2f} +- {q['losses_err_pct']:.2f} %")
    ref = [q for q in irow if q["run"] != "011"]
    hp = sum(q["hops"] for q in ref) / sum(q["atoms1"] for q in ref)
    print(f"pooled 005-010 hops {100 * hp:.3f} %; 011 excess {irow[-1]['hops_pct'] - 100 * hp:.2f} %")
    on = np.array([9 * q["exposure_ms"] / 1000 for q in irow])
    ls, le = np.array([q["losses_pct"] for q in irow]), np.array([q["losses_err_pct"] for q in irow])
    (sl, ic), lc = np.polyfit(on, ls, 1, w=1 / le, cov="unscaled")
    print(f"losses against molasses-on time per set: {sl:.2f} +- {np.sqrt(lc[0, 0]):.2f} %/s, "
          f"intercept {ic:.2f} +- {np.sqrt(lc[1, 1]):.2f} %")

    lrow = []
    for run, lab in LIFETIME.items():
        pts = lifetime_points(folder, run)
        t = np.array([q[0] for q in pts])
        n1 = np.array([q[1] for q in pts], float)
        k = np.array([q[2] for q in pts], float)
        f = k / n1
        err = np.sqrt(np.clip(f * (1 - f), 1 / n1, None) / n1)
        (a, tau), cov = curve_fit(expdecay, t, f, p0=[1, 60], sigma=err, absolute_sigma=True)
        chi2 = float(np.sum(((f - expdecay(t, a, tau)) / err) ** 2) / (len(t) - 2))
        print(f"{run} ({lab}): A = {a:.3f} +- {np.sqrt(cov[0, 0]):.3f}, tau = {tau:.1f} +- {np.sqrt(cov[1, 1]):.1f} s, "
              f"chi2/dof {chi2:.2f}, {int(n1.sum())} atoms; loss rate {100 / tau:.2f} %/s")
        (a2, tb, th), cov2 = curve_fit(heating, t, f, p0=[1, 60, 50], sigma=err, absolute_sigma=True,
                                       bounds=([0.5, 1, 1], [1.5, 1e4, 1e4]))
        chi2b = float(np.sum(((f - heating(t, a2, tb, th)) / err) ** 2) / (len(t) - 3))
        e2 = np.sqrt(np.diag(cov2))
        print(f"   with heating term A exp(-t/tau_bg - (t/t_h)^2): tau_bg = {tb:.0f} +- {e2[1]:.0f} s, "
              f"t_h = {th:.0f} +- {e2[2]:.0f} s, chi2/dof {chi2b:.2f}")
        early = t <= 12.1
        r_early = -np.polyfit(t[early], np.log(f[early]), 1, w=f[early] / err[early])[0]
        print(f"   initial loss rate (0-12 s, log-linear) {100 * r_early:.2f} %/s")
        for q, fi, ei in zip(pts, f, err):
            lrow.append(dict(run=run, condition=lab, hold_s=q[0], atoms1=q[1], kept=q[2], kept_frac=round(fi, 4),
                             kept_err=round(ei, 4), fit_A=round(a, 4), fit_tau_s=round(tau, 2),
                             fit_tau_err_s=round(float(np.sqrt(cov[1, 1])), 2), fit_chi2=round(chi2, 2),
                             heat_A=round(a2, 4), heat_tau_bg_s=round(tb, 1), heat_t_h_s=round(th, 1),
                             heat_chi2=round(chi2b, 2), early_rate_per_s=round(r_early, 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (("imaging", irow), ("lifetime", lrow)):
        with open(OUT / f"molasses_imaging_test_20261102_{name}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# written by scripts/molasses_imaging_test_20261102.py from {folder}\n")
            w = csv.DictWriter(fh, list(rr[0]))
            w.writeheader()
            w.writerows(rr)


if __name__ == "__main__":
    main()
