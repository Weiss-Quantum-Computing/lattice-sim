"""Lattice jitter relative to the imaging system: run 018 of 2026-11-02 (taken 2026-11-03, 'jitter_2EOx_2x20 pics_plane 4').

30 shots, each two image sets of 20 pictures of the same plane (objective at plane 4, X lattice at full depth,
2.2 uW x 150 ms, 20 ms between pictures, 50 ms + two junk frames between the sets). Every atom the camera calls
occupied in a picture is fitted with the 2D Gaussian of readstats_focus.fit_atom (CameraControl) in a 9 x 9 px
patch around its grid point; the grid origin of each shot comes from camStack on the sum of its 40 pictures and the
pitch from a fit of the atom positions (readstats_focus.pitchfit). The lattice position in a picture is the median
over its atoms of (fitted position - grid point), in camera pixels, rows and columns separately; its error is
1.2533 x the atoms' robust sigma / sqrt(n).

Statistics: the picture-to-picture scatter within a shot (the per-shot mean removed), the set-to-set step across
the 50 ms gap, the shot-to-shot scatter of the per-shot means, the Allan deviation against the number of pictures
averaged (one picture = 170 ms), a linear drift per shot over the 6.8 s, the spot widths and amplitudes against
the picture index, and the noise floor from the atoms' own scatter. Camera scale 0.52 um/px (Wu).

Writes data/measured/jitter_20261102_{atoms,pictures,shots,allan,summary}.csv and prints the summary.
Usage: python scripts/jitter_20261102.py ["~/Desktop/Data/Camera Control/20261102"] [path to CameraControl]
"""

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAMERACONTROL = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else REPO.parent
sys.path.insert(0, str(CAMERACONTROL))
import readstats_focus as rf  # noqa: E402

DATA = "~/Desktop/Data/Camera Control/20261102"
RUN = "018"
OUT = REPO / "data" / "measured"
NSETS, NPIC, Y, X = 2, 20, 9, 9
SPACING0 = (9.37, 8.89)                 # readstats_config FocusSpacing (rows, columns), the start of the pitch fit
PIC_PERIOD_S = 0.170                    # 150 ms exposure + 20 ms
SET_GAP_S = 0.120                       # last picture of set 1 to the first of set 2: 50 + 15 + 20 + 15 + 20 ms
UM_PER_PX = 0.52
ALLAN_N = (1, 2, 3, 5, 10, 20)


def timestamps(fn):
    pat = re.compile(rb"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}")
    out = []
    with open(fn, "rb") as fh:
        for line in fh:
            if pat.match(line):
                out.append(line.strip().decode())
    return out


def seconds(ts):
    h, m, s = ts[-8:].split(":")
    return 3600 * int(h) + 60 * int(m) + int(s)


def fit_shot(imgs, occ, origin, spacing):
    """every occupied site of every picture: (set, pic, i, j, row, col, amp, wr, wc); positions in camera px"""
    rows = []
    for s in range(NSETS):
        for k in range(NPIC):
            img = imgs[s * NPIC + k].astype(float)
            for i in range(Y):
                for j in range(X):
                    if not occ[s][k, i, j]:
                        continue
                    rr = origin[0] + i * spacing[0]
                    cc = origin[1] + j * spacing[1]
                    r0, c0 = int(round(rr)), int(round(cc))
                    if r0 - rf.HALF < 0 or r0 + rf.HALF >= rf.ROW_MX or c0 - rf.HALF < 0 or c0 + rf.HALF >= rf.COL_MX:
                        continue
                    patch = img[r0 - rf.HALF:r0 + rf.HALF + 1, c0 - rf.HALF:c0 + rf.HALF + 1]
                    fit = rf.fit_atom(patch, rf.HALF + (rr - r0), rf.HALF + (cc - c0))
                    if fit is None or fit[1] < rf.MINAMP:
                        continue
                    bg, amp, cr, cf, wr, wc = fit
                    rows.append((s, k, i, j, r0 - rf.HALF + cr, c0 - rf.HALF + cf, amp, wr, wc))
    return rows


def robust(v):
    v = np.asarray(v, float)
    m = np.median(v)
    return m, 1.4826 * np.median(np.abs(v - m))


def allan(series, n):
    """non-overlapping Allan deviation of a 1-D series at averaging length n (samples), nan if < 2 blocks"""
    s = np.asarray(series, float)
    s = s[np.isfinite(s)]
    m = len(s) // n
    if m < 2:
        return np.nan, 0
    blocks = s[:m * n].reshape(m, n).mean(1)
    return float(np.sqrt(0.5 * np.mean(np.diff(blocks) ** 2))), m - 1


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else DATA).expanduser()
    os.chdir(REPO)
    rawfn, atomfn = folder / f"{RUN}.txt", folder / f"{RUN}.txt.atom"
    occs = rf.atom_occupancy(str(atomfn), NSETS, (NPIC, Y, X))
    ts = timestamps(rawfn)
    print(f"{len(occs)} shots in the .atom file, {len(ts)} time stamps")
    atoms = []
    origins = []
    spacing = SPACING0
    for r, imgs in rf.raw_runs(str(rawfn), NSETS * NPIC):
        if r >= len(occs):
            break
        occ = [(occs[r][s] != 0) for s in range(NSETS)]
        summed = imgs.sum(0).astype(float)
        origin = rf.cam_stack(summed, 5.0, 5.0, spacing[0], spacing[1], Y)
        origins.append(origin)
        rows = fit_shot(imgs, occ, origin, spacing)
        atoms += [(r,) + row for row in rows]
        if r == 0:                      # the pitch from the first shot's atoms (row against i, column against j), then used for every shot
            q = np.array(rows, float)
            pr = np.polyfit(q[:, 2], q[:, 4], 1)[0]
            pc = np.polyfit(q[:, 3], q[:, 5], 1)[0]
            if 8.0 < pr < 10.5 and 8.0 < pc < 10.5:
                spacing = (float(pr), float(pc))
                print(f"grid pitch fitted on shot 1: rows {spacing[0]:.3f} px, columns {spacing[1]:.3f} px (started from {SPACING0})")
                origin = rf.cam_stack(summed, 5.0, 5.0, spacing[0], spacing[1], Y)
                origins[-1] = origin
                rows = fit_shot(imgs, occ, origin, spacing)
                atoms = [(r,) + row for row in rows]
            else:
                print(f"pitch fit gave {pr:.3f} x {pc:.3f} px; keeping {SPACING0}")
    A = np.array(atoms, float)          # r, s, k, i, j, row, col, amp, wr, wc
    origins = np.array(origins)
    print(f"{len(A)} atom fits over {len(origins)} shots; grid origin per shot: rows {origins[:, 0].mean():.2f} +- {origins[:, 0].std():.2f}, "
          f"columns {origins[:, 1].mean():.2f} +- {origins[:, 1].std():.2f} px (camStack on the shot's summed image)")
    # one fixed reference grid for the whole run: the mean origin; the offset of an atom is its position minus its grid point
    ref = origins.mean(0)
    A_off_r = A[:, 5] - (ref[0] + A[:, 3] * spacing[0])
    A_off_c = A[:, 6] - (ref[1] + A[:, 4] * spacing[1])

    # per picture
    nshots = len(origins)
    P = np.full((nshots, NSETS, NPIC, 9), np.nan)     # n, row_off, row_err, col_off, col_err, wr, wc, amp, sigma_atoms_row
    for r in range(nshots):
        for s in range(NSETS):
            for k in range(NPIC):
                m = (A[:, 0] == r) & (A[:, 1] == s) & (A[:, 2] == k)
                n = int(m.sum())
                if n < 3:
                    continue
                mr, sr = robust(A_off_r[m])
                mc, sc = robust(A_off_c[m])
                P[r, s, k] = (n, mr, 1.2533 * sr / np.sqrt(n), mc, 1.2533 * sc / np.sqrt(n), np.median(A[m, 8]), np.median(A[m, 9]),
                              np.median(A[m, 7]), sr)
    pic_index = np.arange(NSETS * NPIC)
    t_pic = np.where(pic_index < NPIC, pic_index * PIC_PERIOD_S, NPIC * PIC_PERIOD_S + SET_GAP_S + (pic_index - NPIC) * PIC_PERIOD_S)
    Pf = P.reshape(nshots, NSETS * NPIC, 9)
    off = Pf[:, :, [1, 3]]                             # (shot, picture, [row, col])
    err = Pf[:, :, [2, 4]]
    shotmean = np.nanmean(off, 1)                      # (shot, 2)
    within = off - shotmean[:, None, :]
    res = {}
    for a, name in ((0, "row"), (1, "col")):
        w = within[:, :, a]
        e = err[:, :, a]
        sd_pic = float(np.nanstd(w))
        floor = float(np.sqrt(np.nanmean(e ** 2)))
        jitter = float(np.sqrt(max(sd_pic ** 2 - floor ** 2, 0)))
        set1 = np.nanmean(off[:, :NPIC, a], 1)
        set2 = np.nanmean(off[:, NPIC:, a], 1)
        step = set2 - set1
        slopes = []
        for r in range(nshots):
            ok = np.isfinite(off[r, :, a])
            if ok.sum() > 10:
                slopes.append(np.polyfit(t_pic[ok], off[r, ok, a], 1)[0])
        sm = shotmean[:, a]
        res[name] = dict(sd_picture=sd_pic, floor=floor, jitter_picture=jitter,
                         set_step_mean=float(np.nanmean(step)), set_step_sd=float(np.nanstd(step)),
                         sd_shot=float(np.nanstd(sm)), shot_range=[float(np.nanmin(sm)), float(np.nanmax(sm))],
                         drift_per_s_median=float(np.median(slopes)), drift_per_s_sd=float(np.std(slopes)),
                         allan=[(n, ) + allan_pooled(within[:, :, a], n) for n in ALLAN_N])
    # correlation of row and column within-shot residuals, and per-atom residual check
    ok = np.isfinite(within[:, :, 0]) & np.isfinite(within[:, :, 1])
    corr_rc = float(np.corrcoef(within[:, :, 0][ok], within[:, :, 1][ok])[0, 1])
    # the per-site systematic (a residual grid error, the fixed-pattern offset of each site): the median offset of each site
    # over every shot and picture, minus the run's overall median. A different mix of occupied sites from shot to shot would
    # turn it into an apparent shot-to-shot motion, so the shot statistics are repeated with it taken out of every atom.
    sitemap = np.full((2, Y, X), np.nan)
    for i in range(Y):
        for j in range(X):
            m = (A[:, 3] == i) & (A[:, 4] == j)
            if m.sum() >= 10:
                sitemap[0, i, j] = np.median(A_off_r[m]) - np.median(A_off_r)
                sitemap[1, i, j] = np.median(A_off_c[m]) - np.median(A_off_c)
    # the pattern as an affine distortion of the grid: offset = t + M (site position in px); M's antisymmetric part is a
    # rotation of the lattice axes against the camera's, its diagonal a pitch error, the symmetric off-diagonal a shear
    gi, gj = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
    okm = np.isfinite(sitemap[0])
    G = np.column_stack([np.ones(okm.sum()), (gi[okm] - 4) * spacing[0], (gj[okm] - 4) * spacing[1]])
    aff_r = np.linalg.lstsq(G, sitemap[0][okm], rcond=None)[0]      # drow = a0 + a1 x_row + a2 x_col
    aff_c = np.linalg.lstsq(G, sitemap[1][okm], rcond=None)[0]      # dcol = b0 + b1 x_row + b2 x_col
    affine = dict(rotation_mrad=1e3 * 0.5 * (aff_r[2] - aff_c[1]), shear_mrad=1e3 * 0.5 * (aff_r[2] + aff_c[1]),
                  pitch_err_rows=aff_r[1], pitch_err_cols=aff_c[2],
                  resid_rows=float(np.std(sitemap[0][okm] - G @ aff_r)), resid_cols=float(np.std(sitemap[1][okm] - G @ aff_c)))
    print(f"\nper-site offset pattern: sd {np.nanstd(sitemap[0]):.3f} / {np.nanstd(sitemap[1]):.3f} px (rows / columns) over {okm.sum()} sites; "
          f"affine fit: rotation of the lattice against the camera axes {affine['rotation_mrad']:+.2f} mrad ({np.degrees(affine['rotation_mrad'] / 1e3):+.3f} deg), "
          f"shear {affine['shear_mrad']:+.2f} mrad, pitch error rows {1e3 * affine['pitch_err_rows']:+.1f}, columns {1e3 * affine['pitch_err_cols']:+.1f} px per 1000 px; "
          f"residual {affine['resid_rows']:.3f} / {affine['resid_cols']:.3f} px")
    corr_r = A_off_r - np.nan_to_num(sitemap[0][A[:, 3].astype(int), A[:, 4].astype(int)])
    corr_c = A_off_c - np.nan_to_num(sitemap[1][A[:, 3].astype(int), A[:, 4].astype(int)])
    shotmean_sc = np.full((nshots, 2), np.nan)
    for r in range(nshots):
        m = (A[:, 0] == r)
        shotmean_sc[r] = (np.median(corr_r[m]), np.median(corr_c[m]))
    tsec0 = np.array([seconds(t) for t in ts[:nshots]], float) if len(ts) >= nshots else np.arange(nshots, dtype=float)
    tmin = (tsec0 - tsec0[0]) / 60.0
    for a, name in ((0, "row"), (1, "col")):
        res[name]["sitemap_sd"] = float(np.nanstd(sitemap[a]))
        res[name]["sd_shot_sitecorr"] = float(np.nanstd(shotmean_sc[:, a]))
        sl, ic = np.polyfit(tmin, shotmean[:, a], 1)
        res[name]["shot_drift_px_per_min"] = float(sl)
        res[name]["sd_shot_detrended"] = float(np.std(shotmean[:, a] - (sl * tmin + ic)))
    # atoms' own residual from the picture's common mode (how much an atom moves on its own)
    atom_res_r = np.full(len(A), np.nan)
    atom_res_c = np.full(len(A), np.nan)
    for r in range(nshots):
        for s in range(NSETS):
            for k in range(NPIC):
                m = (A[:, 0] == r) & (A[:, 1] == s) & (A[:, 2] == k)
                if np.isfinite(P[r, s, k, 1]):
                    atom_res_r[m] = A_off_r[m] - P[r, s, k, 1]
                    atom_res_c[m] = A_off_c[m] - P[r, s, k, 3]
    # widths and amplitude against the picture index (median over shots)
    wr_k = np.nanmedian(Pf[:, :, 5], 0); wc_k = np.nanmedian(Pf[:, :, 6], 0); amp_k = np.nanmedian(Pf[:, :, 7], 0); n_k = np.nanmean(Pf[:, :, 0], 0)
    tsec = np.array([seconds(t) for t in ts[:nshots]]) if len(ts) >= nshots else np.arange(nshots) * np.nan
    cycle = float(np.median(np.diff(tsec))) if nshots > 1 and np.isfinite(tsec).all() else np.nan

    print(f"\natoms per picture {np.nanmean(Pf[:, :, 0]):.1f} (first picture {n_k[0]:.1f}, last {n_k[-1]:.1f}); "
          f"per-picture precision (floor) rows {res['row']['floor']:.3f} px, columns {res['col']['floor']:.3f} px")
    print(f"atoms' own scatter about the common mode: rows {robust(atom_res_r[np.isfinite(atom_res_r)])[1]:.3f} px, "
          f"columns {robust(atom_res_c[np.isfinite(atom_res_c)])[1]:.3f} px (robust sigma per atom)")
    for name in ("row", "col"):
        q = res[name]
        print(f"\n{name}s (camera {'rows' if name == 'row' else 'columns'}):")
        print(f"  picture to picture within a shot: sd {q['sd_picture']:.3f} px, floor {q['floor']:.3f} -> jitter {q['jitter_picture']:.3f} px = {UM_PER_PX * 1e3 * q['jitter_picture']:.0f} nm")
        print(f"  set 2 - set 1 (across the 50 ms gap): mean {q['set_step_mean']:+.3f} px, sd {q['set_step_sd']:.3f} px")
        print(f"  drift within a shot (6.8 s): median {1e3 * UM_PER_PX * q['drift_per_s_median']:+.0f} nm/s, sd over shots {1e3 * UM_PER_PX * q['drift_per_s_sd']:.0f} nm/s")
        print(f"  shot to shot (cycle {cycle:.0f} s): sd {q['sd_shot']:.3f} px = {UM_PER_PX * 1e3 * q['sd_shot']:.0f} nm, range {q['shot_range'][0]:+.2f} to {q['shot_range'][1]:+.2f} px; "
              f"with the per-site pattern (sd {q['sitemap_sd']:.3f} px over the sites) taken out {q['sd_shot_sitecorr']:.3f} px; "
              f"trend {1e3 * UM_PER_PX * q['shot_drift_px_per_min']:+.0f} nm/min, sd about the trend {q['sd_shot_detrended']:.3f} px")
        print("  Allan deviation within shots: " + ", ".join(f"{n} pic ({n * PIC_PERIOD_S:.2f} s): {ad:.3f} px" for (n, ad, nb) in q['allan']))
    print(f"\nrow-column correlation of the within-shot residuals r = {corr_rc:.2f}")
    print(f"spot widths (1/e, px): rows {np.nanmedian(wr_k):.2f}, columns {np.nanmedian(wc_k):.2f}; first picture {wr_k[0]:.2f}/{wc_k[0]:.2f}, "
          f"last {wr_k[-1]:.2f}/{wc_k[-1]:.2f}; amplitude first {amp_k[0]:.0f}, last {amp_k[-1]:.0f}")
    print("amplitude by picture (median over shots): set 1 " + " ".join(f"{v:.0f}" for v in amp_k[:NPIC]) + " | set 2 " + " ".join(f"{v:.0f}" for v in amp_k[NPIC:]))
    print("width rows by picture: set 1 " + " ".join(f"{v:.2f}" for v in wr_k[:NPIC]) + " | set 2 " + " ".join(f"{v:.2f}" for v in wr_k[NPIC:]))
    print("width cols by picture: set 1 " + " ".join(f"{v:.2f}" for v in wc_k[:NPIC]) + " | set 2 " + " ".join(f"{v:.2f}" for v in wc_k[NPIC:]))

    OUT.mkdir(parents=True, exist_ok=True)
    hdr = f"# written by scripts/jitter_20261102.py from {folder}/{RUN}; pitch {spacing[0]:.3f} x {spacing[1]:.3f} px, reference grid origin {ref[0]:.3f}, {ref[1]:.3f} px\n"
    with open(OUT / "jitter_20261102_atoms.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        w.writerow(["shot", "set", "picture", "i", "j", "row_px", "col_px", "row_off_px", "col_off_px", "amp", "w_rows_px", "w_cols_px", "res_row_px", "res_col_px"])
        for q, orr, occ_, rr_, rc_ in zip(A, A_off_r, A_off_c, atom_res_r, atom_res_c):
            w.writerow([int(q[0]), int(q[1]), int(q[2]), int(q[3]), int(q[4]), round(q[5], 4), round(q[6], 4), round(orr, 4), round(occ_, 4),
                        round(q[7]), round(q[8], 3), round(q[9], 3), round(rr_, 4), round(rc_, 4)])
    with open(OUT / "jitter_20261102_pictures.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        w.writerow(["shot", "set", "picture", "t_in_shot_s", "atoms", "row_off_px", "row_err_px", "col_off_px", "col_err_px", "w_rows_px", "w_cols_px", "amp", "timestamp"])
        for r in range(nshots):
            for s in range(NSETS):
                for k in range(NPIC):
                    q = P[r, s, k]
                    w.writerow([r + 1, s + 1, k + 1, round(t_pic[s * NPIC + k], 3), int(q[0]) if np.isfinite(q[0]) else 0] +
                               [round(v, 4) if np.isfinite(v) else "" for v in q[1:8]] + [ts[r] if r < len(ts) else ""])
    with open(OUT / "jitter_20261102_shots.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        w.writerow(["shot", "timestamp", "row_mean_px", "col_mean_px", "row_set_step_px", "col_set_step_px", "origin_row_px", "origin_col_px"])
        for r in range(nshots):
            w.writerow([r + 1, ts[r] if r < len(ts) else "", round(shotmean[r, 0], 4), round(shotmean[r, 1], 4),
                        round(float(np.nanmean(off[r, NPIC:, 0]) - np.nanmean(off[r, :NPIC, 0])), 4),
                        round(float(np.nanmean(off[r, NPIC:, 1]) - np.nanmean(off[r, :NPIC, 1])), 4), round(origins[r, 0], 3), round(origins[r, 1], 3)])
    with open(OUT / "jitter_20261102_allan.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        w.writerow(["axis", "pictures", "tau_s", "allan_px", "pairs"])
        for name in ("row", "col"):
            for (n, ad, nb) in res[name]["allan"]:
                w.writerow([name, n, round(n * PIC_PERIOD_S, 3), round(ad, 4), nb])
    with open(OUT / "jitter_20261102_summary.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        w.writerow(["axis", "sd_picture_px", "floor_px", "jitter_picture_px", "set_step_mean_px", "set_step_sd_px", "sd_shot_px", "sd_shot_sitecorr_px", "sitemap_sd_px",
                    "shot_drift_px_per_min", "sd_shot_detrended_px", "drift_nm_per_s_median", "drift_nm_per_s_sd", "corr_row_col", "cycle_s", "atoms_per_picture", "atom_sigma_px"])
        for a, name in ((0, "row"), (1, "col")):
            q = res[name]
            ar = (atom_res_r if a == 0 else atom_res_c)
            w.writerow([name, round(q["sd_picture"], 4), round(q["floor"], 4), round(q["jitter_picture"], 4), round(q["set_step_mean"], 4), round(q["set_step_sd"], 4),
                        round(q["sd_shot"], 4), round(q["sd_shot_sitecorr"], 4), round(q["sitemap_sd"], 4), round(q["shot_drift_px_per_min"], 4), round(q["sd_shot_detrended"], 4),
                        round(1e3 * UM_PER_PX * q["drift_per_s_median"], 1), round(1e3 * UM_PER_PX * q["drift_per_s_sd"], 1), round(corr_rc, 3),
                        round(cycle, 1), round(float(np.nanmean(Pf[:, :, 0])), 1), round(robust(ar[np.isfinite(ar)])[1], 4)])
    with open(OUT / "jitter_20261102_sitemap.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(hdr)
        w = csv.writer(fh)
        fh.write("# affine fit of the pattern: " + ", ".join(f"{k} {v:.4f}" for k, v in affine.items()) + "\n")
        w.writerow(["i", "j", "row_offset_px", "col_offset_px", "row_affine_px", "col_affine_px"])
        for i in range(Y):
            for j in range(X):
                g = np.array([1.0, (i - 4) * spacing[0], (j - 4) * spacing[1]])
                w.writerow([i, j] + [round(v, 4) if np.isfinite(v) else "" for v in (sitemap[0, i, j], sitemap[1, i, j])] + [round(float(g @ aff_r), 4), round(float(g @ aff_c), 4)])


def allan_pooled(within, n):
    """Allan deviation at n pictures pooled over the shots (the pairs of consecutive block means of every shot)"""
    d2 = []
    for row in within:
        s = row[np.isfinite(row)]
        m = len(s) // n
        if m >= 2:
            b = s[:m * n].reshape(m, n).mean(1)
            d2 += list(np.diff(b) ** 2)
    return (float(np.sqrt(0.5 * np.mean(d2))) if d2 else np.nan, len(d2))


if __name__ == "__main__":
    main()
