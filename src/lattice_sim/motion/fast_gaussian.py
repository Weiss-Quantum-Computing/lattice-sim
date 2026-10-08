"""Fast, vectorized exact-Gaussian transport for many harmonic scenarios.

Same physics as ``gaussian_exact`` (Heller Gaussian in a harmonic trap with
moving centre x0(t) and frequency omega(t)) but integrated with exact
piecewise propagators instead of an adaptive ODE solver, and for S scenarios
at once:

* omega is held at its midpoint value over each step of length dt, and x0 is
  linear in time within the step. Then the displacement y = q - x0 and the
  Riccati variable are propagated exactly by the constant-omega rotation.
* The width is carried by z(t) with z'' = -omega^2 z, alpha = (m/2) z'/z,
  z(0) = 1, z'(0) = i omega(0) (the initial ground state).

The error is O(dt^2) in the trap's acceleration and frequency change per
step; at dt = 0.5 us it agrees with the adaptive solver to ~1e-6 in P0 on
the Transport-Optimizer ramps. One evaluation of a 5 ms ramp takes ~0.1 s
regardless of the number of scenarios.
"""

import numpy as np

from ..constants import MASS_CS
from .gaussian_exact import _overlap


def _rotate(y, ydot, w, dt):
    c, s = np.cos(w * dt), np.sin(w * dt)
    return c * y + s / w * ydot, -w * s * y + c * ydot


def propagate_scenarios(t_us, x0, omega, sample_every=None):
    """Propagate S scenarios along a common time grid.

    t_us: (N,) uniform time grid. x0, omega: (S, N) trap centre (m) and
    angular frequency (rad/s) at those times. Returns P0 with shape (S, N)
    (or (S, N_samples) with ``sample_every`` steps between samples).
    """
    t = (np.asarray(t_us, float) - t_us[0]) * 1e-6
    dt = t[1] - t[0]
    x0 = np.atleast_2d(np.asarray(x0, float))
    omega = np.atleast_2d(np.asarray(omega, float))
    n = t.size
    stride = 1 if sample_every is None else int(sample_every)
    # state
    q = x0[:, 0].copy()
    p = np.zeros_like(q)
    z = np.ones_like(q, dtype=complex)
    zdot = 1j * omega[:, 0].astype(complex)
    out = [np.ones(q.shape)]
    for i in range(n - 1):
        w = 0.5 * (omega[:, i] + omega[:, i + 1])
        v = (x0[:, i + 1] - x0[:, i]) / dt
        y, ydot = _rotate(q - x0[:, i], p / MASS_CS - v, w, dt)
        q = y + x0[:, i + 1]
        p = MASS_CS * (ydot + v)
        z, zdot = _rotate(z, zdot, w, dt)
        if (i + 1) % stride == 0:
            alpha = 0.5 * MASS_CS * zdot / z
            out.append(_overlap(alpha, q, p, 0.5j * MASS_CS * omega[:, i + 1], x0[:, i + 1]))
    return np.array(out).T


class HarmonicScenario:
    """x0(theta) and f(theta) for one state and model variant, splined."""

    def __init__(self, theta, x0_m, f_hz, name="", f_scale=1.0):
        from scipy.interpolate import CubicSpline
        self.name = name
        self._x0 = CubicSpline(theta, np.abs(x0_m))
        self._f = CubicSpline(theta, np.asarray(f_hz) * f_scale)

    @classmethod
    def load(cls, path, f_scale=1.0):
        d = np.load(path)
        from pathlib import Path
        name = Path(path).stem
        if f_scale != 1.0:
            name += f"_f{f_scale:g}"
        return cls(d["theta"], d["x0_m"], d["f_hz"], name, f_scale)

    def x0(self, theta):
        return self._x0(theta)

    def omega(self, theta):
        return 2 * np.pi * self._f(theta)
