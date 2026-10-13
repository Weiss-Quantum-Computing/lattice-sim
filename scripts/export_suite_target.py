"""Write a designed rise as an ILC target pair in the 2026-09-28 ramp-suite format.

Record on the ILC's 2 us grid: 200 us at 0 | rise | 1000 us hold at the top |
mirrored fall | 200 us at 0. PAR split (each EOM takes theta/2), each channel
scaled to its own V90 (5128.3 V X1, 5137.4 V X2), EO zero not applied, as in
Ramps/suite-20260928/designs/source/build_suite.py.

The predicted transport phase uses the suite's model (build_suite.E_MODEL:
E(F4) - E(F3) at the trap minimum from the ER316 calibration, sigma+/sigma-
standing-wave depth), and is reported relative to run 024 through the suite's
own S47 entry (phase_rad -1148.03 rad = 1.1068 x run 024).

Usage: python scripts/export_suite_target.py DESIGN_JSON STEM OUT_DIR "description"
Writes OUT_DIR/target_<STEM>X1.csv, target_<STEM>X2.csv and <STEM>_prediction.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_suite_shape_robust import K0, Scorer, theta_of  # noqa: E402
from fit_er316_calibration import load as load_calibration  # noqa: E402

DT = 2.0
LEAD, HOLD = 200, 1000
V90 = {"X1": 5128.3, "X2": 5137.4}
CEIL = {"X1": 5594.0, "X2": 5924.0}
S47_PHASE_RAD, S47_PHASE_REL = -1148.0339023756335, 1.1068138429910808   # suite_predictions.json
REF_PHASE = S47_PHASE_RAD / S47_PHASE_REL
HB, H, M, D = 1.054571817e-34, 6.62607015e-34, 133 * 1.66053906660e-27, 4.8374e-6


def suite_dE():
    """build_suite / softsplit dE_of_theta: E(F4) - E(F3) at the trap minimum (Hz) vs degrees."""
    out = 0.0
    for name, sgn in (("F4", 1), ("F3", -1)):
        frac, x, f_khz = load_calibration("data/reference", name)
        f = f_khz * 1e3
        u0 = M * (2 * np.pi * f[0]) ** 2 * D ** 2 / (2 * np.pi ** 2) / H
        out = out + sgn * (u0 / 2 * (1 - (f / f[0]) ** 2) + f / 2)
        deg = frac * 180
    return deg, out


def suite_phase(theta_deg_record, k0=K0):
    deg, de = suite_dE()
    return 2 * np.pi * np.sum(np.interp(theta_deg_record, deg, de) * k0 ** 2) * DT * 1e-6


def record(rise):
    z = np.zeros(int(LEAD / DT))
    top = np.full(int(HOLD / DT) - 1, rise[-1])
    th = np.r_[z, rise, top, rise[::-1], z]
    return np.arange(len(th)) * DT, th


def main(design, stem, out_dir, what):
    d = json.load(open(design, encoding="utf-8"))
    T = float(d["T_us"])
    rise = theta_of(np.array(d["c"]), d["theta_w"], int(round(T / DT)) + 1)
    t, th = record(rise)
    v = {"X1": th / 2 / 90 * V90["X1"], "X2": th / 2 / 90 * V90["X2"]}
    phase = suite_phase(th)
    rep = Scorer(T).report(np.array(d["c"]))
    pred = dict(stem=stem, rise_us=T, depth="1/4", scheme="PAR", what=what, period_ms=len(t) * DT / 1e3,
                peakA=float(v["X1"].max()), peakB=float(v["X2"].max()),
                slewA=float(np.abs(np.diff(v["X1"])).max() / DT), slewB=float(np.abs(np.diff(v["X2"])).max() / DT),
                phase_rad=float(phase), phase_rel=float(phase / REF_PHASE), winding_pred=float(0.25 * phase / REF_PHASE),
                lattice_sim_rise_1mP0_worst_x085_115=rep["worst_robust"],
                lattice_sim_rise_1mP0_x080=rep["worst_x080"], design_file=Path(design).name,
                design_coefficients=d["c"], K0=K0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wr = rep["worst_robust"]
    for ch in ("X1", "X2"):
        vv = v[ch]
        with open(out / f"target_{stem}{ch}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {stem} {ch}: {what}; scheme PAR; designed for 1/4 lattice intensity; lattice-sim, added to the 2026-09-28 suite\n")
            f.write(f"# record: {LEAD} us 0 | rise {T:.0f} us | {HOLD} us hold at {th.max():.1f} deg total | mirrored fall | {LEAD} us 0; "
                    f"{len(t)} pts at {DT:g} us = {pred['period_ms']:.4f} ms period\n")
            f.write(f"# scaled to this channel's V90 {V90[ch]:.1f} V (EO zero not applied); peak {vv.max():.1f} V of {CEIL[ch]:.0f} V ceiling, "
                    f"peak slew {np.abs(np.diff(vv)).max() / DT:.2f} V/us\n")
            f.write(f"# predicted (lattice-sim exact Gaussian, 1 - P0 at end of rise, worst over |3,-3>,|4,-3> and trap freq x0.85-1.15): "
                    f"calibration {wr['cal']:.1e}, model h-1 {wr['model_h-1']:.1e}, model h+1 {wr['model_h+1']:.1e}; "
                    f"transport phase {pred['phase_rel']:.2f} x run 024's\n")
            f.write("time_us,voltage_V\n")
            np.savetxt(f, np.column_stack([t, vv]), delimiter=",", fmt="%.6f")
    with open(out / f"{stem}_prediction.json", "w", encoding="utf-8") as f:
        json.dump(pred, f, indent=1)
    print(json.dumps({k: pred[k] for k in ("stem", "rise_us", "period_ms", "peakA", "peakB", "slewA", "slewB",
                                           "phase_rel", "winding_pred")}, indent=None))
    print("rise 1-P0 worst x0.85-1.15:", wr, " x0.80:", rep["worst_x080"])


if __name__ == "__main__":
    import os
    design = str(Path(sys.argv[1]).resolve())
    out_dir = str(Path(sys.argv[3]).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    main(design, sys.argv[2], out_dir, sys.argv[4])
