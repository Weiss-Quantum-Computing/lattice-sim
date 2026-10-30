"""Figure: linearity of the atom brightness with the imaging power (2026-10-26 and 2026-10-29).

a  brightness per 100 ms exposure against the delivered fiber power, with the
   proportional fit B = a P per series. 10-26: power from the measured +100 MHz
   AOM curve (lattice_sim.molasses.aom) at each rescale value, also used for
   the +95 and +105 MHz runs; 10-29: the request in the new calibration.
b  brightness per unit power, normalized to each series' mean, against power,
   for -25.5 MHz on the two power axes of 10-26: the measured AOM curve
   (filled) and the 24/r the Supertime file promises (open), with 10-29.

Prints the proportional and affine fits, the power-law exponent, a
saturation-form fit B = A P / (1 + P/Ps) to the pooled -25.5 MHz data, and the
Lorentzian scattering ratios between the detunings.
Writes figures/molasses_linearity.png.
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
GAMMA_MHZ = 5.234              # Cs D2 natural linewidth / 2 pi
COL = {20.5: "#D55E00", 25.5: "#0072B2", 30.5: "#009E73"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def series():
    """{label: (detuning, P_aom, P_file, B)} with P_file = None for 10-29."""
    curves = aom.load_curves()
    p24 = float(aom.delivered(aom.IMAGING_REQUEST, aom.RESCALE_NOW, curves)[1])
    out = {}
    rows = read_csv(MEAS / "molasses_imaging_20261026_reanalysis.csv")
    for d in COL:
        rr = [r for r in rows if float(r["detuning_MHz"]) == d and int(r["run"]) < 20]   # 020+: repeats at r 24
        r = np.array([float(x["rescale"]) for x in rr])
        out[f"10-26, −{d:g} MHz"] = (d, p24 * aom.relative_power(r, curves=curves), p24 * aom.RESCALE_NOW / r,
                                    np.array([float(x["brightness"]) for x in rr]))
    r29 = read_csv(MEAS / "molasses_imaging_20261029_reanalysis.csv")
    out["10-29, −25.5 MHz"] = (25.5, np.array([float(x["power_uw"]) for x in r29]), None,
                               np.array([float(x["brightness"]) for x in r29]))
    return out


def fit_summary(p, b):
    a0 = (p @ b) / (p @ p)
    (a1, c1), *_ = np.linalg.lstsq(np.vstack([p, np.ones_like(p)]).T, b, rcond=None)
    k = np.polyfit(np.log(p), np.log(b), 1)[0]
    return dict(a=a0, rms=np.std(b - a0 * p) / b.mean(), slope=a1, offset=c1, exponent=k)


def lorentz(d):
    return 1 / (1 + 4 * (d / GAMMA_MHZ) ** 2)


def main():
    apply()
    s = series()
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN * 0.7, 2.5), constrained_layout=True)
    pp = np.linspace(0, 4.5, 50)
    for lab, (d, p, pf, b) in s.items():
        f = fit_summary(p, b)
        mk = "s" if lab.startswith("10-29") else "o"
        ax[0].plot(p, b * 1e-3, mk, color=COL[d], ms=3, mfc="none" if mk == "s" else COL[d], label=lab)
        ax[0].plot(pp, f["a"] * pp * 1e-3, color=COL[d], lw=0.6, ls="--" if mk == "s" else "-")
        print(f"{lab}: a = {f['a']:.0f} counts/uW, rms {100 * f['rms']:.1f} %, affine {f['slope']:.0f} uW^-1 "
              f"+ {f['offset']:.0f}, exponent {f['exponent']:.2f}")
        if pf is not None:
            g = fit_summary(pf, b)
            print(f"   on the 24/r axis: a = {g['a']:.0f}, rms {100 * g['rms']:.1f} %, exponent {g['exponent']:.2f}")
    ax[0].set_xlim(0, 4.5)
    ax[0].set_ylim(0, None)
    ax[0].set_xlabel("Imaging power, fiber total (µW)")
    ax[0].set_ylabel("Atom brightness (10$^3$ counts / 100 ms)")
    ax[0].legend(fontsize=5.5, loc="upper left")

    d, p, pf, b = s["10-26, −25.5 MHz"]
    _, p9, _, b9 = s["10-29, −25.5 MHz"]
    for x, mfc, lab in ((p, COL[25.5], "10-26, measured AOM curve"), (pf, "none", "10-26, 24/r (Supertime file)")):
        y = b / x
        ax[1].plot(x, y / y.mean(), "o", color=COL[25.5], mfc=mfc, ms=3, label=lab)
    y9 = b9 / p9
    ax[1].plot(p9, y9 / y9.mean(), "s", color="#555555", mfc="none", ms=3, label="10-29, new calibration")
    ax[1].axhline(1, color="#000000", lw=0.6)
    ax[1].set_ylim(0.85, 1.15)
    ax[1].set_xlabel("Imaging power, fiber total (µW)")
    ax[1].set_ylabel("Brightness / power (norm.)")
    ax[1].legend(fontsize=5.5, loc="lower left")
    for a, t in zip(ax, "ab"):
        a.text(-0.2, 1.03, t, transform=a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    out = REPO / "figures" / "molasses_linearity.png"
    fig.savefig(out)
    print("wrote", out.relative_to(REPO))

    pa, ba = np.r_[p, p9], np.r_[b, b9]
    (amp, ps), cov = curve_fit(lambda x, A, Ps: A * x / (1 + x / Ps), pa, ba, p0=[7500, 30])
    print(f"-25.5 MHz pooled: B = A P/(1+P/Ps), A = {amp:.0f}, Ps = {ps:.0f} +- {np.sqrt(cov[1, 1]):.0f} uW")
    a = {d: fit_summary(*s[f"10-26, −{d:g} MHz"][1:4:2])["a"] for d in COL}
    for d in (20.5, 30.5):
        print(f"slope ratio -{d:g}/-25.5: measured {a[d] / a[25.5]:.2f}, Lorentzian {lorentz(d) / lorentz(25.5):.2f}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
