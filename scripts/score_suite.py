"""Score a folder of ILC target pairs (the ramp A/B suite format) with lattice-sim.

Each target is a pair target_<stem>X1.csv / target_<stem>X2.csv on a 2 us
grid, one record 'lead | rise | hold | mirrored fall | lead', each channel in
volts of its own EOM. The total rotation is

    theta(t) = (pi/2) (V1 / V90_1 + V2 / V90_2),

with V90 read from the '# scaled to this channel's V90 ... V' header line.
The transported pair is |4,-3> / |3,-3> (the Transport-Optimizer F4 / F3
calibration states).

Engines, each on the suite's trap-frequency scales K0 x {0.80, 0.85, 1, 1.15}
(K0 = 0.94 for quarter intensity, from the target header):

  cal    harmonic parameters of the ER316 calibration files (the suite's own
         model), exact Gaussian propagation;
  model  harmonic parameters of the first-principles potentials with the
         measured extinction profile (handedness -1 and +1);
  split  split-step on the full anharmonic model tables when
         data/x_ERmeas133_h-1_10kHz_F{3m-3,4m-3}.npz exist (slow, --split).

Reported per target and engine: 1 - P0 at the end of the rise, the worst
1 - P0 inside the stop-anywhere band (theta >= 126.7 deg, i.e. V >= 3800 of
5400 in Transport-Optimizer units) during the rise, and 1 - P0 at the end
of the record (after the return). Also the differential light-shift phase
(drag plus measured ellipticity) of the pair over one record.

Usage: python scripts/score_suite.py SUITE_DIR [--split] [--stems S25,N15,...]
(--stems=S25,N15 also works). SUITE_DIR is resolved against the current
directory before the script moves to the repo root. Writes
SUITE_DIR/lattice_sim_scores.json; stops with an error when the folder has
no target pairs (or none of the given stems).

Caches (data/suite_*.npz, git-ignored) record the inputs they were built
from (the ER CSV's digest, the bias field, the angle grid; ``lattice_sim.cache``)
and are rebuilt when those change.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from fit_er316_calibration import load as load_calibration
from lattice_sim import cache
from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.fast_gaussian import propagate_scenarios
from lattice_sim.motion.transport1d import PotentialTable, Transport1D

ER_CSV = "data/measured/133-ms-SE-offset-neg2V-fine-2_extinction.csv"
STATES = {"F3": (3, -3), "F4": (4, -3)}
V90_DEFAULT = {"X1": 5128.3, "X2": 5137.4}
SCALES = (0.80, 0.85, 1.0, 1.15)
BAND_DEG = 3800.0 / 5400.0 * 180.0
DT_US = 0.5
B_Q = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
DEPTH_K0 = {"1/4": 0.94, "1/2": 1.33, "1/8": 0.67}     # trap-frequency scale per target-header depth


def read_target(path):
    """(t_us, V, header dict) of one channel file."""
    head, rows = {}, []
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.search(r"V90 ([0-9.]+) V", line)
                if m:
                    head["v90"] = float(m.group(1))
                m = re.search(r"designed for (\S+) lattice intensity", line)
                if m:
                    head["depth"] = m.group(1)
                head.setdefault("title", line[2:].strip())
            elif not line.startswith("time_us"):
                rows.append([float(v) for v in line.split(",")])
    data = np.array(rows)
    return data[:, 0], data[:, 1], head


def read_pair(folder, stem):
    t1, v1, h1 = read_target(Path(folder) / f"target_{stem}X1.csv")
    t2, v2, h2 = read_target(Path(folder) / f"target_{stem}X2.csv")
    assert np.allclose(t1, t2)
    # V90 of each EOM (Trek-EOM-ILC config) when the header does not state it
    v90_1, v90_2 = h1.get("v90", V90_DEFAULT["X1"]), h2.get("v90", V90_DEFAULT["X2"])
    theta = np.pi / 2 * (v1 / v90_1 + v2 / v90_2)
    depth = h1.get("depth", "1/4")
    if depth not in DEPTH_K0:
        raise ValueError(f"{Path(folder) / f'target_{stem}X1.csv'}: unknown lattice intensity {depth!r} in the "
                         f"header (known: {', '.join(DEPTH_K0)})")
    return t1, theta, DEPTH_K0[depth], h1["title"]


def rise_end_index(theta):
    """Index where the rise reaches its top (start of the hold)."""
    top = theta.max()
    return int(np.argmax(theta >= top - 1e-9))


# --------------------------------------------------------------- scenarios
def calibration_scenarios():
    out = {}
    for name, state in STATES.items():
        frac, x, f = load_calibration("data/reference", name)
        out[name] = (CubicSpline(np.pi * frac, np.abs(x) * 1e-6), CubicSpline(np.pi * frac, f * 1e3))
    return out


def _cache_inputs(**extra):
    return dict(er_csv=Path(ER_CSV).name, er_csv_digest=cache.file_digest(ER_CSV), b_q=B_Q,
                states={k: list(v) for k, v in STATES.items()}, **extra)


def model_scenarios(handed, n=361, cache_dir="data"):
    path = Path(cache_dir) / f"suite_harm_ERmeas133_h{handed:+d}.npz"
    inputs = _cache_inputs(kind="model_scenarios", handedness=handed, n=n)
    d = cache.load(path, inputs)
    if d is not None:
        thetas = d["theta"]
        return {name: (CubicSpline(thetas, d[f"x_{name}"]), CubicSpline(thetas, d[f"f_{name}"])) for name in STATES}
    profile = ExtinctionProfile.from_rows(read_rampol_er(ER_CSV), 1, "away")
    model = StatePotential(LatticePair(extinction_ratio=profile, handedness=handed), B_Q)
    thetas = np.unique(np.concatenate([np.linspace(0, np.pi, n), np.linspace(0.4 * np.pi, 0.6 * np.pi, n // 2)]))
    out, save = {}, {"theta": thetas}
    for name, state in STATES.items():
        started = time.time()
        x0, f = model.track(state, thetas)
        save[f"x_{name}"], save[f"f_{name}"] = np.abs(x0), f
        out[name] = (CubicSpline(thetas, np.abs(x0)), CubicSpline(thetas, f))
        print(f"  model h{handed:+d} {state}: tracked in {time.time()-started:.0f} s "
              f"(f at pi/2 {f[np.argmin(np.abs(thetas-np.pi/2))]/1e3:.2f} kHz)", flush=True)
    cache.save(path, inputs, **save)
    return out


def score_harmonic(t_us, theta, scen, k0):
    """1-P0 metrics for both states and every scale, exact-Gaussian engine."""
    tg = np.arange(t_us[0], t_us[-1] + 1e-9, DT_US)
    th = np.interp(tg, t_us, theta)
    names, x0, om = [], [], []
    for name, (cx, cf) in scen.items():
        for sc in SCALES:
            names.append((name, sc))
            x0.append(cx(th))
            om.append(2 * np.pi * cf(th) * k0 * sc)
    p0 = propagate_scenarios(tg, np.array(x0), np.array(om))
    i_top = rise_end_index(th)
    band = (th >= np.deg2rad(BAND_DEG)) & (np.arange(th.size) <= i_top)
    out = {}
    for k, (name, sc) in enumerate(names):
        out[f"{name}_x{sc:.2f}"] = dict(rise=float(1 - p0[k, i_top]), band=float(1 - p0[k, band].min()),
                                        end=float(1 - p0[k, -1]))
    worst = {m: max(v[m] for key, v in out.items() if not key.endswith("x0.80")) for m in ("rise", "band", "end")}
    worst80 = {m: max(v[m] for key, v in out.items() if key.endswith("x0.80")) for m in ("rise", "band", "end")}
    return dict(per_scenario=out, worst_robust=worst, worst_x080=worst80)


def score_split(t_us, theta, tables, k0):
    """Split-step on the full tables; the depth scale enters as V * k0^2."""
    out = {}
    for name, table in tables.items():
        scaled = PotentialTable(table.thetas, table.x, table.values_hz * k0**2)
        ts, p0, _ = Transport1D(scaled).propagate(t_us, theta, dt_us=0.25, sample_every_us=10.0)
        th_s = np.interp(ts, t_us, theta)
        i_top = rise_end_index(th_s)
        band = (th_s >= np.deg2rad(BAND_DEG)) & (np.arange(th_s.size) <= i_top)
        out[name] = dict(rise=float(1 - p0[i_top]), band=float(1 - p0[band].min()), end=float(1 - p0[-1]))
    return out


# ------------------------------------------------------------ light-shift phase
def pair_energy_curves(n_deg=2.0, cache_dir="data"):
    """E(4,-3) - E(3,-3) at the trap bottoms (minimum + zero point) vs theta,
    pure linear and with the measured ellipticity (h = -1), Hz."""
    path = Path(cache_dir) / "suite_pair_energy_F4m-3_F3m-3.npz"
    inputs = _cache_inputs(kind="pair_energy", n_deg=n_deg)
    d = cache.load(path, inputs)
    if d is not None:
        return d["theta"], d["lin"], d["meas"]
    thetas = np.deg2rad(np.arange(0, 180 + 1e-9, n_deg))
    profile = ExtinctionProfile.from_rows(read_rampol_er(ER_CSV), 1, "away")
    models = {"lin": StatePotential(LatticePair(), B_Q),
              "meas": StatePotential(LatticePair(extinction_ratio=profile, handedness=-1), B_Q)}
    curves = {}
    for key, model in models.items():
        vals = []
        for th in thetas:
            e = {}
            for state in STATES.values():
                sign = 1 if state[0] == 4 else -1
                guess = sign * model.pair.spacing_m / 2 * np.sin(th / 2) ** 2
                x0, f = model.minimum(state, th, guess=guess)
                e[state] = model.potential(state, np.array([x0]), th)[0] + f / 2
            vals.append(e[(4, -3)] - e[(3, -3)])
        curves[key] = np.array(vals)
    cache.save(path, inputs, theta=thetas, lin=curves["lin"], meas=curves["meas"])
    return thetas, curves["lin"], curves["meas"]


def find_stems(folder, stems=None):
    """Stems of the target pairs in a folder (only those in ``stems`` if given);
    ValueError when there are none."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"suite folder {folder} does not exist")
    files = sorted(glob.glob(str(folder / "target_*X1.csv")))
    all_stems = [re.search(r"target_(.+)X1\.csv", Path(f).name).group(1) for f in files]
    chosen = all_stems if stems is None else [s for s in all_stems if s in stems]
    if not chosen:
        raise ValueError(f"no target pairs target_<stem>X1.csv / X2.csv in {folder}"
                         + (f" for the stems {', '.join(stems)}" if stems else ""))
    missing = [] if stems is None else [s for s in stems if s not in all_stems]
    if missing:
        print("not in the folder:", ", ".join(missing))
    return chosen


def split_tables():
    """The --split tables, checked against ER_CSV when they record their inputs
    (scripts/build_profile_tables.py); None if they are missing or were built
    from another extinction CSV."""
    try:
        tables = {name: PotentialTable.load(f"data/x_ERmeas133_h-1_10kHz_F{s[0]}m{s[1]:+d}.npz")
                  for name, s in STATES.items()}
    except FileNotFoundError as err:
        print("no split-step tables:", err)
        return None
    digest = cache.file_digest(ER_CSV)
    for name, table in tables.items():
        made = (table.inputs or {}).get("er_csv_digest")
        if made is None:
            print(f"split-step table {name}: no record of its extinction CSV (built before 2026-10-23); "
                  f"assuming {ER_CSV}")
        elif made != digest:
            print(f"split-step table {name} was built from another extinction CSV "
                  f"({table.inputs.get('er_csv')}); rebuild it with build_profile_tables.py - split skipped")
            return None
    return tables


def light_shift_phases(t_us, theta, k0, thetas_e, e_lin, e_meas):
    """Differential light-shift phase (rad) of the pair over one record: pure linear,
    and with the measured ellipticity, at the depth of trap-frequency scale k0.

    k0 is a trap-FREQUENCY scale, so the depth goes as k0^2. The ellipticity
    (vector) shift is linear in depth x eta (``pair_physics``: S ~ I), so it
    scales as k0^2 like the pure-linear term (it was scaled by k0: +6 % / -25 % /
    +49 % at 1/4, 1/2, 1/8 intensity). The pure-linear k0^2 is itself the
    scalar-dominated approximation of a mix of ~I (light shift), ~I^2
    (fictitious-field drag) and ~sqrt(I) (zero point) contributions."""
    dt = np.gradient(t_us) * 1e-6
    e_lin_t = np.interp(theta, thetas_e, e_lin)
    e_ell_t = np.interp(theta, thetas_e, e_meas) - e_lin_t
    return dict(phase_rad_linear=float(2 * np.pi * np.sum(e_lin_t * k0**2 * dt)),
                phase_rad_measured_er=float(2 * np.pi * np.sum((e_lin_t + e_ell_t) * k0**2 * dt)))


def main(folder, stems=None, do_split=False):
    stems = find_stems(folder, stems)
    cal = calibration_scenarios()
    models = {h: model_scenarios(h) for h in (-1, 1)}
    thetas_e, e_lin, e_meas = pair_energy_curves()
    tables = split_tables() if do_split else None
    results = {}
    for stem in stems:
        t_us, theta, k0, title = read_pair(folder, stem)
        started = time.time()
        r = dict(title=title, k0=k0, period_ms=float(t_us[-1] * 1e-3), top_deg=float(np.rad2deg(theta.max())),
                 cal=score_harmonic(t_us, theta, cal, k0),
                 model_hm1=score_harmonic(t_us, theta, models[-1], k0),
                 model_hp1=score_harmonic(t_us, theta, models[1], k0))
        r.update(light_shift_phases(t_us, theta, k0, thetas_e, e_lin, e_meas))
        if tables is not None:
            r["split_hm1"] = score_split(t_us, theta, tables, k0)
        results[stem] = r
        print(f"{stem:8s} k0 {k0:.2f} top {r['top_deg']:5.1f} deg  "
              f"cal worst rise/band/end {r['cal']['worst_robust']['rise']:.1e}/{r['cal']['worst_robust']['band']:.1e}/{r['cal']['worst_robust']['end']:.1e}  "
              f"model h-1 {r['model_hm1']['worst_robust']['rise']:.1e}/{r['model_hm1']['worst_robust']['band']:.1e}/{r['model_hm1']['worst_robust']['end']:.1e}  "
              f"h+1 {r['model_hp1']['worst_robust']['rise']:.1e}/{r['model_hp1']['worst_robust']['band']:.1e}/{r['model_hp1']['worst_robust']['end']:.1e}  "
              f"phase {r['phase_rad_measured_er']:+.0f} rad"
              + (f"  split F3/F4 end {r['split_hm1']['F3']['end']:.1e}/{r['split_hm1']['F4']['end']:.1e}" if tables else "")
              + f"  ({time.time()-started:.0f}s)", flush=True)
    out = Path(folder) / "lattice_sim_scores.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Score a folder of ILC target pairs with lattice-sim.")
    parser.add_argument("suite_dir", help="folder with target_<stem>X1.csv / target_<stem>X2.csv")
    parser.add_argument("--split", action="store_true", help="also split-step on the full model tables")
    parser.add_argument("--stems", default=None, help="comma-separated stems to score, e.g. S25,N15")
    args = parser.parse_args(argv)
    stems = None if args.stems is None else [s for s in args.stems.split(",") if s]
    # the suite path is relative to where the script is started, not to the repo root
    return Path(args.suite_dir).resolve(), stems, args.split


if __name__ == "__main__":
    folder, stems, split = parse_args(sys.argv[1:])
    os.chdir(Path(__file__).resolve().parents[1])
    try:
        main(folder, stems, split)
    except ValueError as err:
        sys.exit(f"score_suite: {err}")
