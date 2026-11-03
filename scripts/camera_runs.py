"""Re-analyse CameraGui runs (.hist / .atom / .stat / .state) for hops, losses, readout and displacement.

The analysis used for the 2026-10 molasses work (docs/findings.md, 2026-10-29/30 sections), packaged
so a new data set goes through the same steps:

* per-run re-thresholding: the camera's fixed cutoffs (per neighbour category 0_0/1_0/0_1/1_1, the
  category being occupancy of plane z-1 plus 2x plane z+1) are replaced by the valley between the
  empty and atom peaks of that run's histogram, per image and category, iterated with the occupancy
  to a fixed point (as Camera Control/20261026/molasses_reanalysis/reanalyze.py);
* brightness (median atom minus median empty peak, image 1, category 0_0), separation
  (centre difference / quadrature width) and the expected miscalls per site at the cutoff;
* hops (empty -> filled) and losses (filled -> empty) between image 1 and image 2 per atom of
  image 1, planes 1-7; nearest-neighbour hop-loss pairs by direction against a chance level from
  pairing each shot's hops with other shots' losses;
* per scan point: shots are ordered sweep after sweep (point = shot mod n_points);
* a shot with more than --bad-hops 'hops' (misregistered, or a new loading paired with image 1 at the
  end of an aborted acquisition) is dropped, with the shot before it if that one has an empty image 2;
* loss and hops against plane, row and column;
* the floating-centre displacement from the CameraGui state analysis (.state 'Displace 1', pixels,
  atoms classified in the state image): median, robust sigma, tail fractions.

Usage:
  python scripts/camera_runs.py DATA_DIR RUN [RUN ...] [--scan START STOP NPTS] [--out FILE.json]
e.g. python scripts/camera_runs.py "~/Desktop/Data/Camera Control/20261029" 023 024 --state
     python scripts/camera_runs.py DATA 013 --scan 0 5 51
Needs numpy and scipy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, norm

SETS, Z, Y, X = 2, 9, 9, 9
PLANES = slice(1, 8)                       # planes 1-7, the subset the lab's analysis uses
CAM_THR = np.array([9000.0, 11000.0, 12000.0, 14000.0])
NEIGH = [(1, 0, 0, "+z"), (-1, 0, 0, "-z"), (0, 1, 0, "+y"), (0, -1, 0, "-y"), (0, 0, 1, "+x"), (0, 0, -1, "-x")]
STATE_II_PX, STATE_I_PX = -2.28, 1.41      # calibrated state displacements in the 2026-10-30 .state headers


def _lines(path, prefixes):
    out = {p: [] for p in prefixes}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            key, sep, val = line.partition("=")
            if sep and key.strip() in out:
                out[key.strip()].append([float(v) for v in val.split()])
    return out


def load(folder, run):
    """(heights (shots, 2, 9, 9, 9), camera occupancy, kept-shot mask); aborted all-zero shots dropped."""
    folder = Path(folder).expanduser()
    h = np.loadtxt(folder / f"{run}.txt.hist").reshape(-1, SETS, Z, Y, X)
    a = _lines(folder / f"{run}.txt.atom", ("Before", "After", "After 1"))
    after = a["After"] or a["After 1"]
    occ = np.array([[b, f] for b, f in zip(a["Before"], after)], float).reshape(-1, SETS, Z, Y, X)
    occ = (np.abs(occ) > 0).astype(int)
    n = min(len(h), len(occ))
    h, occ = h[:n], occ[:n]
    keep = np.abs(h).reshape(n, -1).max(1) > 0
    return h, occ, keep


def cam_thresholds(folder, run):
    with open(Path(folder).expanduser() / f"{run}.txt.atom", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "thresholds =" in line:
                v = [float(x) for x in line.split("thresholds =")[1].split(";")[0].split()]
                return np.array([v[0:4], v[4:8]])
    return np.tile(CAM_THR, (SETS, 1))


def category(occ):
    pad = np.zeros((occ.shape[0], SETS, Z + 2, Y, X), int)
    pad[:, :, 1:-1] = occ
    return pad[:, :, :-2] + 2 * pad[:, :, 2:]


def valley(vals, guess):
    lo, hi = vals[vals < guess], vals[vals >= guess]
    if len(lo) < 30 or len(hi) < 30:
        return None
    grid = np.linspace(np.median(lo), np.median(hi), 400)
    return float(grid[np.argmin(gaussian_kde(vals, bw_method=0.08)(grid))])


def rethreshold(h, occ_cam, cam):
    occ, thr = occ_cam.copy(), np.array(cam, float)
    guess = thr.copy()
    for _ in range(20):
        cat = category(occ)
        new_thr = thr.copy()
        for s in range(SETS):
            for c in range(4):
                t = valley(h[:, s, PLANES][cat[:, s, PLANES] == c], guess[s, c])
                if t is not None:
                    new_thr[s, c] = t
        new = (h > new_thr[np.arange(SETS)[None, :, None, None, None], cat]).astype(int)
        done = (new == occ).all()
        occ, thr = new, new_thr
        if done:
            break
    return occ, thr, category(occ)


def readout(h, cat, thr):
    """brightness, separation (sigma), expected miscalls per site: image 1, category 0_0."""
    v = h[:, 0, PLANES][cat[:, 0, PLANES] == 0]
    t = thr[0, 0]
    lo, hi = v[v < t], v[v >= t]
    if len(lo) < 30 or len(hi) < 30:
        return dict(bright=np.nan, sep=np.nan, miscall=np.nan)
    c0, w0 = np.median(lo), 1.4826 * np.median(np.abs(lo - np.median(lo)))
    c1, w1 = np.median(hi), 1.4826 * np.median(np.abs(hi - np.median(hi)))
    p1 = len(hi) / len(v)
    eps = (1 - p1) * norm.sf(t, c0, w0) + p1 * norm.cdf(t, c1, w1)
    return dict(bright=float(c1 - c0), sep=float((c1 - c0) / np.hypot(w0, w1)), miscall=float(eps))


def _pairs(hp, ls):
    d = {n[3]: 0 for n in NEIGH}
    used = np.zeros_like(ls, bool)
    for z, y, x in zip(*np.nonzero(hp)):
        for dz, dy, dx, name in NEIGH:
            zz, yy, xx = z - dz, y - dy, x - dx
            if 0 <= zz < hp.shape[0] and 0 <= yy < Y and 0 <= xx < X and ls[zz, yy, xx] and not used[zz, yy, xx]:
                used[zz, yy, xx] = True
                d[name] += 1
                break
    return d


def moves(hop, loss, shots):
    """nearest-neighbour hop-loss pairs by direction (loss site -> hop site) and the chance level."""
    real = {n[3]: 0 for n in NEIGH}
    chance = {n[3]: 0.0 for n in NEIGH}
    for k, i in enumerate(shots):
        for a, b in _pairs(hop[i], loss[i]).items():
            real[a] += b
        for j in range(1, 6):
            jj = shots[(k + j) % len(shots)]
            if jj != i:
                for a, b in _pairs(hop[i], loss[jj]).items():
                    chance[a] += b / 5
    return real, chance


def state(folder, run):
    s = _lines(Path(folder).expanduser() / f"{run}.txt.state", ("After 1", "Displace 1"))
    return np.array(s["After 1"]).reshape(-1, Z, Y, X), np.array(s["Displace 1"]).reshape(-1, Z, Y, X)


def displacement_stats(a, d):
    v = d[:, PLANES][a[:, PLANES] > 0]
    med = np.median(v)
    f = lambda c: float(np.mean(c))  # noqa: E731
    return dict(n=int(len(v)), median=float(med), sigma=float(1.4826 * np.median(np.abs(v - med))),
                abs_gt_0p5=f(np.abs(v - med) > 0.5), abs_gt_1=f(np.abs(v - med) > 1), gt_1=f(v - med > 1),
                lt_m1=f(v - med < -1), beyond_state_II=f(v > -STATE_II_PX), beyond_state_I=f(v < -STATE_I_PX),
                by_plane=[f(np.abs(d[:, z][a[:, z] > 0]) > 1) for z in range(1, 8)])


def analyse(folder, run, scan=None, bad_hops=50, with_state=False):
    h, occ_cam, keep = load(folder, run)
    occ, thr, cat = rethreshold(h[keep], occ_cam[keep], cam_thresholds(folder, run))
    idx = np.nonzero(keep)[0]
    o1, o2 = occ[:, 0, PLANES], occ[:, 1, PLANES]
    hop, loss = (o1 == 0) & (o2 == 1), (o1 == 1) & (o2 == 0)
    ns = len(o1)
    per = lambda a: a.reshape(ns, -1).sum(1)  # noqa: E731
    n1, n2, nh, nl = per(o1), per(o2), per(hop), per(loss)
    use = np.ones(ns, bool)
    for i in np.nonzero(nh > bad_hops)[0]:         # misregistered shot or aborted acquisition
        use[i] = False
        if i > 0 and n2[i - 1] == 0:               # the shot with an empty image 2 that precedes it
            use[i - 1] = False
    shots = np.nonzero(use)[0]
    out = dict(run=run, shots=int(ns), shots_used=int(use.sum()), thr=np.round(thr).tolist(), **readout(h[keep], cat, thr))
    out.update(atoms1=int(n1[use].sum()), atoms2=int(n2[use].sum()), hops=int(nh[use].sum()), losses=int(nl[use].sum()))
    real, chance = moves(hop, loss, list(shots))
    out.update(pairs=real, pairs_chance={k: round(v, 1) for k, v in chance.items()})
    prof = {}
    for lab, ax in (("z", (1, 2)), ("y", (0, 2)), ("x", (0, 1))):
        a = o1[use].sum(0).sum(ax)
        prof[lab] = dict(atoms=a.tolist(), loss=(loss[use].sum(0).sum(ax) / np.maximum(a, 1)).round(4).tolist())
    out["profiles"] = prof
    if scan is not None:
        grid = np.linspace(*scan)
        x = grid[idx % len(grid)]
        out["points"] = [dict(x=float(g), shots=int((use & (x == g)).sum()), atoms1=int(n1[use & (x == g)].sum()),
                              atoms2=int(n2[use & (x == g)].sum()), hops=int(nh[use & (x == g)].sum()),
                              losses=int(nl[use & (x == g)].sum())) for g in grid]
    if with_state and (Path(folder).expanduser() / f"{run}.txt.state").exists():
        a, d = state(folder, run)
        out["displacement"] = displacement_stats(a[keep][use], d[keep][use])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("folder")
    p.add_argument("runs", nargs="+")
    p.add_argument("--scan", nargs=3, type=float, metavar=("START", "STOP", "NPTS"))
    p.add_argument("--bad-hops", type=int, default=50)
    p.add_argument("--state", action="store_true", help="read the .state displacement as well")
    p.add_argument("--out")
    a = p.parse_args()
    scan = None if a.scan is None else (a.scan[0], a.scan[1], int(a.scan[2]))
    res = []
    for run in a.runs:
        r = analyse(a.folder, run, scan, a.bad_hops, a.state)
        res.append(r)
        n = max(r["atoms1"], 1)
        print(f"{run}: shots {r['shots_used']}/{r['shots']} bright {r['bright']:.0f} sep {r['sep']:.2f} "
              f"miscall {r['miscall']:.1e} | atoms {r['atoms1']} hops {100 * r['hops'] / n:.2f}% "
              f"losses {100 * r['losses'] / n:.2f}% | pairs {sum(r['pairs'].values())} "
              f"(chance {sum(r['pairs_chance'].values()):.1f})")
        if "displacement" in r:
            d = r["displacement"]
            print(f"   displacement: sigma {d['sigma']:.3f} px, |d|>1 {100 * d['abs_gt_1']:.2f}%, "
                  f"> +2.28 {100 * d['beyond_state_II']:.2f}%, < -1.41 {100 * d['beyond_state_I']:.2f}%")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
