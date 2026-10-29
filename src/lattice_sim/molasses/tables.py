"""Tabulated molasses Monte Carlo results and their interpolation.

The tables are the JSON-lines files written by ``scripts/molasses_scan.py``
(one ``model.summary`` dict per line):

* ``intensity.jsonl`` - zero field, full lattice depth, a rectangular grid of
  detuning x intensity;
* ``field.jsonl``     - the same at two intensities with a residual field
  along one lattice axis at a time.

Interpolation is bilinear in (detuning, log intensity) on the logarithm of
the positive quantities (temperature, scattering rate, tau_E), clamped at the
grid edges (no extrapolation; ``inside`` reports it). Hopping rates are
computed from the interpolated T and tau_E with the Arrhenius form of
``model.arrhenius_hop_rate``, so they stay smooth over many decades.

Survival after a molasses pulse of duration t assumes every hop loses the
atom from its site (``HOP_IS_LOSS``): P = exp(-sum_i Gamma_h,i t).
"""

import json
from pathlib import Path

import numpy as np

from .model import arrhenius_hop_rate

HOP_IS_LOSS = True
LI_ATTEMPT_RATE_PER_S = 265.0      # Li (2009) Sec. 3.5.3: Arrhenius fit to the measured 1D hopping (with T = 10 uK)
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "molasses"


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bracket(grid, x):
    """Index i and weight w with x = (1-w) grid[i] + w grid[i+1], clamped."""
    g = np.asarray(grid, float)
    if len(g) == 1:
        return 0, 0.0
    x = float(np.clip(x, g[0], g[-1]))
    i = int(np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2))
    return i, (x - g[i]) / (g[i + 1] - g[i])


def crossing_field_mg(b_mg, t_uk, factor=2.0):
    """|B| at which T first reaches factor x T(B = 0), by linear interpolation
    between scanned points (inf if it never does). No functional form assumed."""
    b = np.abs(np.asarray(b_mg, float))
    t = np.asarray(t_uk, float)
    o = np.argsort(b)
    b, t = b[o], t[o]
    target = factor * t[0]
    above = np.nonzero(t >= target)[0]
    if len(above) == 0:
        return np.inf
    j = above[0]
    if j == 0:
        return 0.0
    return float(b[j - 1] + (target - t[j - 1]) * (b[j] - b[j - 1]) / (t[j] - t[j - 1]))


def survival(hop_rate_per_s, duration_ms):
    """Probability to stay on the site for a molasses pulse of duration_ms (all axes summed)."""
    total = np.sum(np.atleast_2d(hop_rate_per_s), axis=-1)
    return np.exp(-total * duration_ms * 1e-3)


class Tables:
    """Zero-field grid (detuning x intensity) and field scans from the scan files."""

    def __init__(self, intensity_rows, field_rows=()):
        rows = [r for r in intensity_rows if not any(r["b_mg"]) and r.get("depth_scale", 1.0) == 1.0]
        self.detunings = np.array(sorted({r["detuning"] for r in rows}))
        self.intensities = np.array(sorted({r["intensity"] for r in rows}))
        nd, ni = len(self.detunings), len(self.intensities)
        self.t = np.full((nd, ni, 3), np.nan)
        self.t_err = np.full((nd, ni, 3), np.nan)
        self.tau = np.full((nd, ni, 3), np.nan)
        self.scatter = np.full((nd, ni), np.nan)
        self.depump = np.full((nd, ni), np.nan)
        self.light_shift = np.full((nd, ni), np.nan)
        self.barrier = None
        for r in rows:
            a = int(np.nonzero(self.detunings == r["detuning"])[0][0])
            b = int(np.nonzero(self.intensities == r["intensity"])[0][0])
            self.t[a, b] = r["t_uk"]
            self.t_err[a, b] = r["t_uk_err"]
            self.tau[a, b] = r["tau_e_ms"]
            self.scatter[a, b] = r["scatter_rate"]
            self.depump[a, b] = r["depump_frac"]
            self.light_shift[a, b] = r["light_shift_uk"]
            self.barrier = np.asarray(r["barrier_uk"], float)
        self.complete = bool(rows) and not np.isnan(self.scatter).any()
        self.field_rows = [r for r in field_rows]

    @classmethod
    def from_dir(cls, folder=DATA_DIR):
        folder = Path(folder)
        return cls(load_jsonl(folder / "intensity.jsonl"), load_jsonl(folder / "field.jsonl"))

    # .. zero-field grid ----------------------------------------------------
    def inside(self, detuning, intensity):
        return bool(self.detunings[0] <= detuning <= self.detunings[-1]
                    and self.intensities[0] <= intensity <= self.intensities[-1])

    def _interp_log(self, arr, detuning, intensity):
        i, wi = _bracket(self.detunings, detuning)
        j, wj = _bracket(np.log(self.intensities), np.log(intensity))
        la = np.log(arr)
        i1 = min(i + 1, len(self.detunings) - 1)
        j1 = min(j + 1, len(self.intensities) - 1)
        v = ((1 - wi) * (1 - wj) * la[i, j] + wi * (1 - wj) * la[i1, j]
             + (1 - wi) * wj * la[i, j1] + wi * wj * la[i1, j1])
        return np.exp(v)

    def temperature(self, detuning, intensity):
        """Per-axis kinetic temperature (uK)."""
        return self._interp_log(self.t, detuning, intensity)

    def scattering(self, detuning, intensity):
        return float(self._interp_log(self.scatter, detuning, intensity))

    def tau_e(self, detuning, intensity):
        return self._interp_log(self.tau, detuning, intensity)

    def hop_rate(self, detuning, intensity, attempt_per_s=None):
        """Arrhenius hopping rate per axis (s^-1) at full depth; attempt rate 1/tau_E unless given."""
        tau = self.tau_e(detuning, intensity) if attempt_per_s is None else 1e3 / attempt_per_s
        return arrhenius_hop_rate(self.temperature(detuning, intensity), tau, self.barrier)

    def node_hop_rates(self, attempt_per_s=None):
        """(nd, ni, 3) hopping rates at the grid nodes."""
        tau = self.tau if attempt_per_s is None else 1e3 / attempt_per_s
        return arrhenius_hop_rate(self.t, tau, self.barrier[None, None, :])

    # .. field scans --------------------------------------------------------
    def field_pairs(self):
        return sorted({(r["detuning"], r["intensity"]) for r in self.field_rows})

    def field_curve(self, detuning, intensity, axis):
        """(B_mg, T_uk (n,3), tau_ms (n,3), barrier) along one axis for a scanned (detuning, intensity),
        the zero-field point included."""
        pts = []
        for r in self.field_rows:
            if r["detuning"] != detuning or r["intensity"] != intensity:
                continue
            b = np.asarray(r["b_mg"], float)
            others = np.delete(b, axis)
            if np.any(others != 0):
                continue
            pts.append((b[axis], r["t_uk"], r["tau_e_ms"], r["barrier_uk"]))
        pts.sort(key=lambda p: p[0])
        if not pts:
            return None
        b = np.array([p[0] for p in pts])
        t = np.array([p[1] for p in pts])
        tau = np.array([p[2] for p in pts])
        return b, t, tau, np.asarray(pts[0][3], float)

    def doubling_field(self, detuning, intensity, axis):
        c = self.field_curve(detuning, intensity, axis)
        if c is None:
            return np.nan
        return crossing_field_mg(c[0], c[1].mean(axis=1))

    def nearest_field_pair(self, detuning, intensity):
        pairs = self.field_pairs()
        if not pairs:
            return None
        return min(pairs, key=lambda p: (abs(p[0] - detuning), abs(np.log(p[1] / intensity))))
