"""X and X2 imaging at -40.5 MHz, full depth, no state preparation (2026-11-03, runs 012-017).

All runs: AOM 85 MHz (-40.5 MHz), 50 ms dark between the image sets, 50 shots.
012 X / X, 2.5 uW x 120 ms; 013 X / X, 2.2 uW x 150 ms; 014 X2 / X2, 2.2 uW x 150 ms;
015 X / X2, 2.2 uW x 150 ms; 016 X / X2, 2.2 / 1.8 uW x 150 ms; 017 X / X2, 2.1 / 1.7 uW x 150 ms
(power of image 1 / image 2; the camera's image-2 cutoffs were lowered to 5000-7000 for 016-017).

1. Readout per image (re-thresholded as scripts/camera_runs.py): brightness, atom-peak width w1 / B,
   separation, valley thresholds; hops and losses per atom of image 1 (Jeffreys 68 % intervals),
   nearest-neighbour hop-loss pairs; the fraction of hop sites whose image-1 height lies within
   NEAR_THR counts below the threshold (dim atoms missed in image 1), against the fraction of all empty
   sites in that window.
2. Floating-centre displacement ('Displace 1' of the .state file, state image = image 2). The camera
   measures it from the image-1 floating-centre position ('optional before float center fit = 1'), so
   in 014 (X2 / X2) it is the X2 -> X2 change between the image sets and in 015 (X / X2) the X -> X2
   transfer. Per run: median, robust sigma, per-shot median offset, tails, and the correlation of the
   per-shot state-I fraction (statebound = 0) with the per-shot median.
3. X2 site quantisation in 015: per group of atoms the unbiased |<exp(2 pi i (d / p - f x - f_y y))>|^2
   (x, y = column, row), grouped by (shot, plane), scanned over the period p (px) and the phase step per
   column f and per row f_y; null from shuffling atoms between shots. The fractional part of
   a_X / a_X2 equals f (mod 1, up to sign).
4. Image-2 height against |displacement| in 015, and the amplitude a fixed-centre fit on the X grid
   would lose for the measured 015 displacements, exp(-d^2 / 2 s^2), s = SPOT_SIGMA_PX.

Writes data/measured/molasses_x2_imaging_20261102_{readout,summary,periodicity,hist,shots}.csv; prints the summary.
Usage: python scripts/molasses_x2_imaging_20261102.py ["~/Desktop/Data/Camera Control/20261102"]
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import camera_runs as cr  # noqa: E402
import molasses_imaging_budget_20261102 as bud  # noqa: E402

DATA = "~/Desktop/Data/Camera Control/20261102"
OUT = REPO / "data" / "measured"
RUNS = {"012": ("X", "X", 2.5, 2.5, 120.0), "013": ("X", "X", 2.2, 2.2, 150.0),
        "014": ("X2", "X2", 2.2, 2.2, 150.0), "015": ("X", "X2", 2.2, 2.2, 150.0),
        "016": ("X", "X2", 2.2, 1.8, 150.0), "017": ("X", "X2", 2.1, 1.7, 150.0)}
STATE_RUNS = ("014", "015", "016", "017")
NEAR_THR = 3000.0                 # counts below the threshold that count as 'just missed'
DMAX_PX = 2.5                     # displacements beyond this are excluded from the periodicity test
P_SCAN = np.round(np.arange(0.50, 1.61, 0.02), 2)
F_SCAN = np.round(np.arange(0.0, 1.0, 0.01), 2)
FY_SCAN = np.round(np.arange(-0.20, 0.21, 0.01), 2)
SPOT_SIGMA_PX = (1.0, 1.27)       # spotSize 1.41 / sqrt 2 (.atom) and spotSize2 1.27 (.state), meaning not documented
X_SPACING_PX = (4.8374 / 0.52, 4.8374 / 0.511)
H_BINS = np.arange(-6000, 40001, 500)
D_BINS = np.round(np.arange(-4.0, 4.01, 0.1), 2)


def jeffreys(k, n):
    lo, hi = beta.ppf([0.16, 0.84], k + 0.5, n - k + 0.5)
    return 100 * k / n, 100 * lo, 100 * hi


def coherence(d, grp, x, y, p, f, fy):
    zc = np.exp(2j * np.pi * (d / p - f * x - fy * y))
    s = np.bincount(grp, zc.real) + 1j * np.bincount(grp, zc.imag)
    n = np.bincount(grp)
    return float(np.sum(np.abs(s) ** 2 - n) / np.sum(n * (n - 1)))


def periodicity(folder, run, rng):
    a, d = cr.state(folder, run)
    sel = (a > 0) & (np.abs(d) < DMAX_PX)
    sel[:, :1] = sel[:, 8:] = False
    s, z, y, x = np.nonzero(sel)
    v = d[sel]
    grp = s * 10 + z
    best = max((coherence(v, grp, x, y, p, f, 0), p, f) for p in P_SCAN for f in F_SCAN)
    _, p0, f0 = best
    fy0 = max(FY_SCAN, key=lambda fy: coherence(v, grp, x, y, p0, f0, fy))
    for _ in range(2):            # refine p, f with the row term
        p0 = max(P_SCAN, key=lambda p: coherence(v, grp, x, y, p, f0, fy0))
        f0 = max(F_SCAN, key=lambda f: coherence(v, grp, x, y, p0, f, fy0))
        fy0 = max(FY_SCAN, key=lambda fy: coherence(v, grp, x, y, p0, f0, fy))
    rows = [dict(run=run, scan="p", value=p, coherence=round(coherence(v, grp, x, y, p, f0, fy0), 4)) for p in P_SCAN]
    rows += [dict(run=run, scan="f", value=f, coherence=round(coherence(v, grp, x, y, p0, f, fy0), 4)) for f in F_SCAN]
    rows += [dict(run=run, scan="p_no_column_step", value=p, coherence=round(coherence(v, grp, x, y, p, 0, 0), 4))
             for p in P_SCAN]
    null = []
    for _ in range(20):
        perm = rng.permutation(len(v))
        null.append(coherence(v[perm], grp, x[perm], y[perm], p0, f0, fy0))
    return rows, dict(p_px=float(p0), f=float(f0), fy=float(fy0), coherence=coherence(v, grp, x, y, p0, f0, fy0),
                      null_mean=float(np.mean(null)), null_max=float(np.max(null)))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DATA
    os.chdir(REPO)
    rng = np.random.default_rng(20261103)
    rrow, srow, prow, hrow, shrow = [], [], [], [], []
    for run, (l1, l2, pw, pw2, t) in RUNS.items():
        h, oc, keep = cr.load(folder, run)
        occ, thr, cat = cr.rethreshold(h[keep], oc[keep], cr.cam_thresholds(folder, run))
        r = cr.analyse(folder, run)
        hk = h[keep]
        for img, lat in ((0, l1), (1, l2)):
            for c in range(4):
                pk = bud.peaks(hk, cat, thr, img, c)
                if pk:
                    rrow.append(dict(run=run, image=img + 1, lattice=lat, category=c, atoms=pk["atoms"],
                                     bright=round(pk["bright"]), w0=round(pk["w0"]), w1=round(pk["w1"]),
                                     w1_over_b=round(pk["w1"] / pk["bright"], 3), sep=round(pk["sep"], 2),
                                     threshold=round(pk["threshold"])))
        # hop sites near threshold (all re-thresholded shots, planes 1-7)
        o = occ[:, :, cr.PLANES]
        hop = (o[:, 0] == 0) & (o[:, 1] == 1)
        z1 = (hk[:, 0, cr.PLANES] - thr[0][cat[:, 0, cr.PLANES]])
        near_hop = float(np.mean(z1[hop] > -NEAR_THR)) if hop.any() else np.nan
        near_empty = float(np.mean((z1 > -NEAR_THR) & (z1 < 0) & (o[:, 0] == 0)) / np.mean(o[:, 0] == 0))
        n = r["atoms1"]
        hp, hlo, hhi = jeffreys(r["hops"], n)
        lp, llo, lhi = jeffreys(r["losses"], n)
        im1 = next(q for q in rrow if q["run"] == run and q["image"] == 1 and q["category"] == 0)
        im2 = next(q for q in rrow if q["run"] == run and q["image"] == 2 and q["category"] == 0)
        row = dict(run=run, lattice1=l1, lattice2=l2, power_uw=pw, power2_uw=pw2, exposure_ms=t, shots_used=r["shots_used"],
                   bright1=im1["bright"], bright2=im2["bright"], w1b_1=im1["w1_over_b"], w1b_2=im2["w1_over_b"],
                   sep1=im1["sep"], sep2=im2["sep"], atoms1=n, hops=r["hops"], hops_pct=round(hp, 3),
                   hops_lo=round(hlo, 3), hops_hi=round(hhi, 3), losses=r["losses"], losses_pct=round(lp, 2),
                   losses_lo=round(llo, 2), losses_hi=round(lhi, 2), nn_pairs=sum(r["pairs"].values()),
                   nn_chance=round(sum(r["pairs_chance"].values()), 1), hop_sites_near_thr=round(near_hop, 3),
                   empty_sites_near_thr=round(near_empty, 4))
        for img in (0, 1):
            cnt, edges = np.histogram(hk[:, img, cr.PLANES][cat[:, img, cr.PLANES] == 0], H_BINS)
            hrow += [dict(run=run, kind=f"height_image{img + 1}", bin_lo=e, count=int(c_)) for e, c_ in zip(edges, cnt)]
        # image-2 false positives on sites empty in image 1 (e0) and image-1 atoms below the image-2 cutoff,
        # at the camera cutoff and at the valley
        cam = cr.cam_thresholds(folder, run)
        c2 = cat[:, 1, cr.PLANES]
        h2 = hk[:, 1, cr.PLANES]
        for lab, t2 in (("cam", cam[1]), ("valley", thr[1])):
            row[f"e0_{lab}"] = round(float(np.mean(h2[o[:, 0] == 0] > t2[c2[o[:, 0] == 0]])), 4)
            row[f"below_t_{lab}"] = round(float(np.mean(h2[o[:, 0] == 1] < t2[c2[o[:, 0] == 1]])), 4)
        row["cam_thr2"] = " ".join(str(int(t_)) for t_ in cam[1])
        both = (o[:, 0] == 1) & (o[:, 1] == 1)
        row["sd_log_h1_over_h2"] = round(float(np.log(hk[:, 0, cr.PLANES][both] / hk[:, 1, cr.PLANES][both]).std()), 3)
        if run in STATE_RUNS:
            a, d = cr.state(folder, run)
            a, d = a[keep], d[keep]
            sel = a > 0
            sel[:, :1] = sel[:, 8:] = False
            v = d[sel]
            med = np.median(v)
            shots = [i for i in range(len(a)) if sel[i].sum() > 30]
            off = np.array([np.median(d[i][sel[i]]) for i in shots])
            frac1 = np.array([np.mean(a[i][sel[i]] == 1) for i in shots])
            res = np.concatenate([d[i][sel[i]] - np.median(d[i][sel[i]]) for i in shots])
            cnt, edges = np.histogram(v, D_BINS)
            hrow += [dict(run=run, kind="displacement", bin_lo=round(e, 2), count=int(c_)) for e, c_ in zip(edges, cnt)]
            shrow += [dict(run=run, shot=int(i), atoms=int(sel[i].sum()), offset_px=round(float(o_), 4),
                           state1_frac=round(float(f_), 4)) for i, o_, f_ in zip(shots, off, frac1)]
            row.update(state_atoms=int(len(v)), d_median=round(float(med), 3),
                       d_sigma=round(float(1.4826 * np.median(np.abs(v - med))), 3),
                       d_shot_offset_mean=round(float(off.mean()), 3), d_shot_offset_sd=round(float(off.std()), 3),
                       d_sigma_offset_removed=round(float(1.4826 * np.median(np.abs(res))), 3),
                       d_abs_gt_1=round(float(np.mean(np.abs(v) > 1)), 4),
                       r_abs_gt_1p5=int(np.sum(np.abs(res) > 1.5)), r_abs_gt_2=int(np.sum(np.abs(res) > 2)),
                       d_gt_state_II_n=int(np.sum(v > -cr.STATE_II_PX)),
                       d_beyond_state_II=round(float(np.mean(v > -cr.STATE_II_PX)), 4),
                       d_beyond_state_I=round(float(np.mean(v < -cr.STATE_I_PX)), 4),
                       state1_frac_min=round(float(frac1.min()), 2), state1_frac_max=round(float(frac1.max()), 2),
                       corr_state1_frac_vs_offset=round(float(np.corrcoef(frac1, off)[0, 1]), 3))
            if run == "015":
                rows, best = periodicity(folder, run, rng)
                prow += rows
                row.update(x2_period_px=best["p_px"], x2_column_step=best["f"], x2_row_step=best["fy"],
                           x2_coherence=round(best["coherence"], 3), x2_null_mean=round(best["null_mean"], 4),
                           x2_null_max=round(best["null_max"], 4))
                m = sel & (occ[:, 0] == 1) & (occ[:, 1] == 1)
                ratio, dv = hk[:, 1][m] / hk[:, 0][m], d[m]
                for lo, hi in ((0, 0.5), (0.5, 1), (1, 3)):
                    k = (np.abs(dv) >= lo) & (np.abs(dv) < hi)
                    row[f"h2_over_h1_d{lo}_{hi}"] = round(float(np.median(ratio[k])), 3)
                dd = dv[np.abs(dv) < 3]
                for s in SPOT_SIGMA_PX:
                    g = np.exp(-dd ** 2 / (2 * s ** 2))
                    row[f"fixed_centre_loss_median_s{s}"] = round(float(1 - np.median(g)), 3)
                    row[f"fixed_centre_p5_s{s}"] = round(float(np.percentile(g, 5)), 3)
        srow.append(row)

    for q in srow:
        print(f"{q['run']} {q['lattice1']}/{q['lattice2']} {q['power_uw']}/{q['power2_uw']} uW x {q['exposure_ms']:.0f} ms: "
              f"B {q['bright1']}/{q['bright2']}, w1/B {q['w1b_1']}/{q['w1b_2']}, sep {q['sep1']}/{q['sep2']} | "
              f"hops {q['hops']}/{q['atoms1']} = {q['hops_pct']:.2f} ({q['hops_lo']:.2f}-{q['hops_hi']:.2f}) %, "
              f"NN {q['nn_pairs']} (chance {q['nn_chance']}), hop sites within {NEAR_THR:.0f} below thr "
              f"{q['hop_sites_near_thr']} (empties {q['empty_sites_near_thr']}) | losses {q['losses_pct']:.2f} "
              f"({q['losses_lo']:.2f}-{q['losses_hi']:.2f}) % | sd log(h1/h2) {q['sd_log_h1_over_h2']}")
        if "d_sigma" in q:
            print(f"    displacement: median {q['d_median']}, sigma {q['d_sigma']} ({q['d_sigma_offset_removed']} per-shot "
                  f"offset removed), per-shot offset {q['d_shot_offset_mean']} +- {q['d_shot_offset_sd']} px; |d| > 1 "
                  f"{q['d_abs_gt_1']:.4f}, beyond state II {q['d_beyond_state_II']:.4f}, beyond state I "
                  f"{q['d_beyond_state_I']:.4f}; state-I fraction {q['state1_frac_min']}-{q['state1_frac_max']}, "
                  f"corr with offset {q['corr_state1_frac_vs_offset']}")
            n_ = q["state_atoms"]
            print("    tails (offset removed): " + ", ".join(
                f"|r| > {lab}: {k} = {jeffreys(k, n_)[0]:.2f} ({jeffreys(k, n_)[1]:.2f}-{jeffreys(k, n_)[2]:.2f}) %"
                for lab, k in (("1.5", q["r_abs_gt_1p5"]), ("2", q["r_abs_gt_2"]))))
        print(f"    image-2 cutoffs {q['cam_thr2']}: e0 {q['e0_cam']:.4f} (valley {q['e0_valley']:.4f}); image-1 atoms below "
              f"cutoff {q['below_t_cam']:.4f} (valley {q['below_t_valley']:.4f})")
        if "x2_period_px" in q:
            print(f"    X2 quantisation: p {q['x2_period_px']} px, column step {q['x2_column_step']}, row step "
                  f"{q['x2_row_step']}, coherence {q['x2_coherence']} (shuffled null {q['x2_null_mean']}, max "
                  f"{q['x2_null_max']})")
            for A in X_SPACING_PX:
                print(f"    a_X = {A:.2f} px: a_X / p = {A / q['x2_period_px']:.2f}; 11.47 gives p = {A / 11.47:.3f} px")
            print(f"    h2/h1 vs |d| (0-0.5, 0.5-1, 1-3 px): {q['h2_over_h1_d0_0.5']}, {q['h2_over_h1_d0.5_1']}, "
                  f"{q['h2_over_h1_d1_3']}; fixed-centre amplitude loss for these d: "
                  + ", ".join(f"s = {s}: median {q[f'fixed_centre_loss_median_s{s}']}, 5th pct {q[f'fixed_centre_p5_s{s}']}"
                              for s in SPOT_SIGMA_PX))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (("readout", rrow), ("summary", srow), ("periodicity", prow),
                     ("hist", hrow), ("shots", shrow)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(OUT / f"molasses_x2_imaging_20261102_{name}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# written by scripts/molasses_x2_imaging_20261102.py from {folder}\n")
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rr)


if __name__ == "__main__":
    main()
