"""Figure: the lab's PGC-field scans (2023-2026) and the 2026-10-29 imaging power scan.

a  survival over the PGC hold against the field offset from each fitted centre,
   for the clean 1000 ms scans, with the Gaussian fits (per axis, colour by day);
b  fitted FWHM against the fitted peak survival (both measured; the estimated
   hold power is 0.42-0.47 uW for every scan, since AOMrescale was used to keep it
   constant, so it does not separate them);
c  hops and losses per atom per image cycle against the imaging power at
   -25.5 MHz: 2026-10-26 (0.23 s dark hold between the image sets, power from
   the AOM curve) and 2026-10-29 (new calibration, 2 s dark hold).

Hold power estimate (printed per scan): Supertime's request (0.6, or 0.5 for 2023-11-07 run 042)
through the rescaled 2014 file gives the control voltage; the delivered power is
the rescale value (taken as that day's full-power reading, AOMrescale's purpose)
times the earlier measured AOM curve at that voltage, normalized to its maximum.
Uncertain by about 2x. Writes figures/molasses_lab_analysis.png.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lattice_sim.molasses import aom  # noqa: E402
from lattice_sim.viz.style import DOUBLE_COLUMN_IN, apply  # noqa: E402

MEAS = REPO / "data" / "measured"
CLEAN_DAYS = ("20231107", "20240530", "20250721", "20250820")     # 2026-05-14: old coil driver box, coarse scans
DAY_COL = {"20231107": "#CC79A7", "20240530": "#E69F00", "20250721": "#009E73", "20250820": "#0072B2"}
AX_MARK = {"x": "o", "y": "s", "z": "^"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def gauss(b, base, amp, b0, sig):
    return base + amp * np.exp(-0.5 * ((b - b0) / sig) ** 2)


def fit_scans(rows):
    scans = {}
    for r in rows:
        scans.setdefault((r["day"], r["run"]), []).append(r)
    out = []
    for (day, run), rr in sorted(scans.items()):
        b = np.array([float(r["field_mG"]) for r in rr])
        s = np.array([float(r["survival"]) for r in rr])
        e = np.maximum(np.array([float(r["survival_err"]) for r in rr]), 0.01)
        j = int(np.argmax(s))
        span = b.max() - b.min()
        try:
            p, c = curve_fit(gauss, b, s, p0=[min(s), s[j] - min(s), b[j], span / 5], sigma=e, absolute_sigma=True,
                             bounds=([0, 0, b.min() - span, 5], [1, 1, b.max() + span, 5 * span]), maxfev=20000)
            pe = np.sqrt(np.diag(c))
        except RuntimeError:
            continue
        out.append(dict(day=day, run=run, axis=rr[0]["axis"], hold=float(rr[0]["hold_ms"]),
                        rescale=float(rr[0]["rescale"]), request=float(rr[0]["request"]), b=b, s=s, e=e,
                        p=p, pe=pe, fwhm=2.3548 * p[3], efwhm=2.3548 * pe[3]))
    return out


def hold_power_uw(request, rescale, curves):
    v = float(aom.request_to_volts(request, rescale, curves["supertime"]))
    vo, po = curves["old"]
    return rescale * float(np.interp(v, vo, po)) / po.max()


def good(f):
    return f["hold"] == 1000 and f["day"] in CLEAN_DAYS and f["p"][0] + f["p"][1] > 0.3 and f["efwhm"] < 0.3 * f["fwhm"]


def main():
    apply()
    curves = aom.load_curves()
    fits = fit_scans(read_csv(MEAS / "pgc_field_scans.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), constrained_layout=True)
    seen = set()
    for f in fits:
        if not good(f):
            continue
        c = DAY_COL[f["day"]]
        lab = f"{f['day'][:4]}-{f['day'][4:6]}" if f["day"] not in seen else None
        seen.add(f["day"])
        ax[0].errorbar(f["b"] - f["p"][2], f["s"], f["e"], color=c, marker=AX_MARK[f["axis"]], ms=2.5, ls="none",
                       lw=0.6, capsize=0, label=lab)
        bb = np.linspace(-150, 150, 200)
        ax[0].plot(bb, gauss(bb + f["p"][2], *f["p"]), color=c, lw=0.6, alpha=0.7)
        ax[1].errorbar(f["p"][0] + f["p"][1], f["fwhm"], f["efwhm"], color=c, marker=AX_MARK[f["axis"]], ms=3.5,
                       ls="none", capsize=0)
    ax[0].set_xlabel("Field offset from the fitted centre (mG)")
    ax[0].set_ylabel("Survival over the 1 s PGC hold")
    ax[0].set_xlim(-150, 150)
    ax[0].legend(fontsize=5, loc="lower center", ncol=2, handletextpad=0.2, columnspacing=0.6)
    ax[1].set_xlabel("Peak survival (fit)")
    ax[1].set_ylabel("FWHM of survival against field (mG)")
    ax[1].text(0.03, 0.97, "markers: o x, □ y, △ z", transform=ax[1].transAxes, va="top", fontsize=5.5,
               color="#555555")

    p24 = float(aom.delivered(aom.IMAGING_REQUEST, aom.RESCALE_NOW, curves)[1])
    r26 = [r for r in read_csv(MEAS / "molasses_imaging_20261026_reanalysis.csv") if float(r["detuning_MHz"]) == 25.5
           and r["run"] in ("000", "001", "002", "003", "004", "005", "020", "025", "027", "028")]
    p26 = np.array([p24 * float(aom.relative_power(float(r["rescale"]), curves=curves)) for r in r26])
    h26 = np.array([float(r["hops_per_atom_%"]) for r in r26])
    l26 = np.array([float(r["losses_per_atom_%"]) for r in r26])
    r29 = read_csv(MEAS / "molasses_imaging_20261029_reanalysis.csv")
    p29 = np.array([float(r["power_uw"]) for r in r29])
    n29 = np.array([float(r["atoms_img1"]) for r in r29])
    h29 = 100 * np.array([float(r["hops"]) for r in r29]) / n29
    l29 = 100 * np.array([float(r["losses"]) for r in r29]) / n29
    o = np.argsort(p26)
    ax[2].plot(p26[o], h26[o], color="#0072B2", marker="o", ms=3, lw=0.8, label="hops, 10-26 (0.23 s dark)")
    ax[2].plot(p26[o], l26[o], color="#0072B2", marker="o", ms=3, lw=0.8, ls="--", mfc="none",
               label="losses, 10-26")
    ax[2].plot(p29, h29, color="#D55E00", marker="s", ms=3, lw=0.8, label="hops, 10-29 (2 s dark)")
    ax[2].plot(p29, l29, color="#D55E00", marker="s", ms=3, lw=0.8, ls="--", mfc="none", label="losses, 10-29")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Imaging power, fiber total (µW)")
    ax[2].set_ylabel("Per atom per image cycle (%)")
    ax[2].legend(fontsize=5)
    for a, s in zip(ax, "abc"):
        a.text(-0.2, 1.03, s, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_lab_analysis.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))
    for f in fits:
        print(f["day"], f["run"], f["axis"], f"hold {f['hold']:.0f}", f"P {hold_power_uw(f['request'], f['rescale'], curves):.2f} uW",
              f"peak {f['p'][0] + f['p'][1]:.2f} centre {f['p'][2]:.1f} FWHM {f['fwhm']:.0f}+-{f['efwhm']:.0f}",
              "used" if good(f) else "-")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
