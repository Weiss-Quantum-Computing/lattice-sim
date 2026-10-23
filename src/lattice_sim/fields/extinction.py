"""Measured extinction ratio of the rotating arm versus rotation angle.

Reads the extinction-ratio export of the ramp polarimeter
(Weiss-Quantum-Computing/ramp-polarimeter, ``<scan>_extinction.csv``; '#'
header, one row per measurement) and builds an interpolated profile
ER(theta) for one leg and direction, usable directly as the
``extinction_ratio`` of ``fields.lattice_pair.LatticePair``.

Method choice. The 'dip' rows (Imin fitted within +-8 deg of crossed) have
the smallest uncertainty and cover the ramp every 15 deg; 'crossing' and
'er_fit' rows are noise- or drift-limited at the 1e3 level on this scan and
are not used. Rows flagged lower_bound = 1 are taken at their bound, so the
profile is a lower bound on ER (an upper bound on ellipticity) there.

The polarimeter measures |ellipticity| only: ER = 1 / eta^2 with eta the
minor/major field-amplitude ratio. The handedness is not measured.

Rows with a blank or non-finite leg, angle or ER are dropped; a profile
needs at least two points. Outside the measured angles the profile holds
the end values, with a warning (RANGE_TOLERANCE_RAD beyond the ends).
"""

import csv
import warnings
from dataclasses import dataclass

import numpy as np

RANGE_TOLERANCE_RAD = np.deg2rad(1.0)   # queries this far outside the measured angles warn


def read_rampol_er(path):
    """Rows of a rampol extinction CSV as dicts with numeric fields parsed."""
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [ln for ln in handle if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    numeric = ("leg", "t_ms", "rotation_deg", "er", "er_sigma_lo", "er_sigma_hi", "lower_bound")
    for r in rows:
        for k in numeric:
            r[k] = float(r[k]) if r.get(k, "") not in ("", None) else np.nan
    return rows


@dataclass
class ExtinctionProfile:
    """ER(theta) by log-linear interpolation in |rotation|; theta in rad."""
    theta: np.ndarray
    er: np.ndarray
    lower: np.ndarray
    label: str = ""

    def __call__(self, theta):
        theta = np.abs(np.asarray(theta, float))
        if theta.size and (np.min(theta) < self.theta[0] - RANGE_TOLERANCE_RAD
                           or np.max(theta) > self.theta[-1] + RANGE_TOLERANCE_RAD):
            warnings.warn(f"extinction profile {self.label}: asked at |rotation| "
                          f"{np.rad2deg(np.min(theta)):.1f}-{np.rad2deg(np.max(theta)):.1f} deg, measured "
                          f"{np.rad2deg(self.theta[0]):.1f}-{np.rad2deg(self.theta[-1]):.1f} deg; the end values "
                          "are held outside", stacklevel=2)
        return np.exp(np.interp(theta, self.theta, np.log(self.er)))

    def eta(self, theta):
        return 1.0 / np.sqrt(self(theta))

    @classmethod
    def from_rows(cls, rows, leg, direction, method="dip", include_static=True):
        """Profile for one leg ('away' or 'back'), anchored by the static
        points of that leg: the rest before 'away' ('rest' for leg 1, 'after
        <leg - 1>' for a later leg, which the scan files under the previous
        leg), the hold at pi, and 'after <leg>' for 'back'."""
        rows = [r for r in rows if np.isfinite(r["leg"]) and np.isfinite(r["rotation_deg"])
                and np.isfinite(r["er"]) and r["er"] > 0]
        pick = [r for r in rows if r["method"] == method and int(r["leg"]) == leg
                and r["direction"] == direction]
        rest = "rest" if leg == 1 else f"after {leg - 1}"
        if include_static:
            pick += [r for r in rows if r["method"] == method and r["direction"] == "static"
                     and (abs(r["rotation_deg"]) > 170 or abs(r["rotation_deg"]) < 1)
                     and ((int(r["leg"]) == leg and r["segment"].startswith("hold")) or
                          (direction == "away" and r["segment"] == rest) or
                          (direction == "back" and int(r["leg"]) == leg and r["segment"].startswith("after")))]
        if len(pick) < 2:
            raise ValueError(f"extinction profile leg {leg} {direction} ({method}): {len(pick)} usable "
                             "point(s), at least 2 needed")
        th = np.deg2rad(np.abs([r["rotation_deg"] for r in pick]))
        er = np.array([r["er"] for r in pick])
        lo = np.array([r["lower_bound"] == 1 for r in pick])
        order = np.argsort(th)
        th, er, lo = th[order], er[order], lo[order]
        # several points at the same angle (e.g. two hold points): keep the lowest ER
        keep_t, keep_e, keep_l = [], [], []
        for t, e, l in zip(th, er, lo):
            if keep_t and abs(t - keep_t[-1]) < np.deg2rad(1.0):
                if e < keep_e[-1]:
                    keep_e[-1], keep_l[-1] = e, l
                continue
            keep_t.append(t), keep_e.append(e), keep_l.append(l)
        return cls(np.array(keep_t), np.array(keep_e), np.array(keep_l),
                   f"leg {leg} {direction} ({method})")
