"""Robust design of the EOM rotation ramp theta(t) for state-dependent transport.

Parametrization (as in Transport-Optimizer, but directly in theta):

    theta(t) / pi = s(t) = t/T + sum_k a_k sin(k pi t / T),  k = 1..K

with zero velocity and zero jerk at both ends (four linear equalities).

Objective: maximize the worst lab-frame ground-state population over a set
of scenarios and over every time in the stop-anywhere band (theta above the
band edge, or below the low edge). Lab-frame P0 at time t equals the P0 that
remains if the ramp is halted at t, so this is the stop-anywhere guarantee.
Scenarios are harmonic models x0(theta), f(theta) of each transported state
under different extinction ratios, ellipticity handedness and depth errors;
scoring uses ``motion.fast_gaussian``. The result should be validated with
the split-step engine on the full anharmonic potentials.

Constraints: monotonic theta, EOM slew <= SLEW_V_PER_US for a full-scale
voltage V_PI per pi of rotation.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..motion.fast_gaussian import propagate_scenarios

V_PI = 5400.0


@dataclass
class RampProblem:
    scenarios: list
    duration_us: float = 5000.0
    terms: int = 10
    dt_us: float = 0.5
    sample_us: float = 2.5
    band_high_v: float = 3800.0
    band_low_v: float = 100.0
    slew_v_per_us: float = 20.0
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.t = np.arange(0.0, self.duration_us + 1e-9, self.dt_us)
        k = np.arange(1, self.terms + 1)
        arg = np.pi * np.outer(self.t / self.duration_us, k)
        self.sines = np.sin(arg)
        self.cosines = np.pi * k / self.duration_us * np.cos(arg)
        self.stride = int(round(self.sample_us / self.dt_us))
        self.t_samples = self.t[:: self.stride]

    # --- shape -----------------------------------------------------------
    def s(self, a):
        return self.t / self.duration_us + self.sines @ a

    def sdot(self, a):
        return 1.0 / self.duration_us + self.cosines @ a

    def theta(self, a):
        return np.pi * np.clip(self.s(a), 0.0, 1.0)

    def voltage(self, a):
        return V_PI * np.clip(self.s(a), 0.0, 1.0)

    def endpoint_matrix(self):
        k = np.arange(1, self.terms + 1, dtype=float)
        sign = (-1.0) ** k
        A = np.vstack([np.pi * k, np.pi * k * sign, (np.pi * k) ** 3, (np.pi * k) ** 3 * sign])
        b = -np.array([1.0, 1.0, 0.0, 0.0])
        return A, b

    def project(self, a):
        A, b = self.endpoint_matrix()
        return a - A.T @ np.linalg.solve(A @ A.T, A @ a - b)

    def fit(self, t_us, theta):
        target = np.interp(self.t, t_us, theta) / np.pi - self.t / self.duration_us
        a, *_ = np.linalg.lstsq(self.sines, target, rcond=None)
        return self.project(a)

    # --- scoring ---------------------------------------------------------
    def p0(self, a):
        key = np.asarray(a).tobytes()
        if key not in self._cache:
            th = self.theta(a)
            x0 = np.array([sc.x0(th) for sc in self.scenarios])
            om = np.array([sc.omega(th) for sc in self.scenarios])
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = propagate_scenarios(self.t, x0, om, sample_every=self.stride)
        return self._cache[key]

    def band(self, a):
        v = self.voltage(a)[:: self.stride]
        return (v >= self.band_high_v - 1e-9) | (v <= self.band_low_v + 1e-9)

    def band_min(self, a):
        """Worst band P0 per scenario, shape (S,)."""
        return self.p0(a)[:, self.band(a)].min(axis=1)

    def score_waveform(self, t_us, theta, volts):
        """Band-min P0 per scenario for an arbitrary sampled ramp (no refit)."""
        th = np.interp(self.t, t_us, theta)
        x0 = np.array([sc.x0(th) for sc in self.scenarios])
        om = np.array([sc.omega(th) for sc in self.scenarios])
        p = propagate_scenarios(self.t, x0, om, sample_every=self.stride)
        v = np.interp(self.t, t_us, volts)[:: self.stride]
        band = (v >= self.band_high_v - 1e-9) | (v <= self.band_low_v + 1e-9)
        return p[:, band].min(axis=1), p[:, -1]

    def report(self, a, label=""):
        bm = self.band_min(a)
        p = self.p0(a)
        lines = [f"{label} worst band P0 {bm.min():.6f}  (1-P0 = {1-bm.min():.2e})"]
        for sc, b, fin in zip(self.scenarios, bm, p[:, -1]):
            lines.append(f"    {sc.name:36s} band {b:.6f}  final {fin:.6f}")
        return "\n".join(lines)

    # --- optimization ----------------------------------------------------
    def constraints(self):
        A, b = self.endpoint_matrix()
        interior = slice(1, -1)
        return [
            dict(type="eq", fun=lambda c: A @ c[:-1] - b),
            dict(type="ineq", fun=lambda c: self.sdot(c[:-1])[interior]),
            dict(type="ineq", fun=lambda c: self.slew_v_per_us - V_PI * self.sdot(c[:-1])[interior]),
            # c[-1] = u = -log10(1 - z): worst-case "nines"
            dict(type="ineq", fun=lambda c: -np.log10(np.clip(1 - self.band_min(c[:-1]), 1e-12, None)) - c[-1]),
        ]

    def optimize(self, a0, max_iter=60, verbose=True):
        """Maximize the worst band P0 (as minimize the worst log-infidelity)."""
        a0 = self.project(np.asarray(a0, float))
        u0 = -np.log10(max(1 - self.band_min(a0).min(), 1e-12)) - 1e-3
        best = dict(a=a0.copy(), score=self.band_min(a0).min())
        history = []

        def callback(c):
            a = c[:-1]
            if np.abs(self.endpoint_matrix()[0] @ a - self.endpoint_matrix()[1]).max() > 1e-6:
                return
            if self.sdot(a)[1:-1].min() < -1e-9 or V_PI * self.sdot(a).max() > self.slew_v_per_us + 1e-6:
                return
            score = self.band_min(a).min()
            history.append(score)
            if score > best["score"]:
                best.update(a=a.copy(), score=score)
            if verbose and len(history) % 5 == 0:
                print(f"    iter {len(history):3d}  worst band P0 {score:.6f}  best {best['score']:.6f}", flush=True)

        result = minimize(
            lambda c: -c[-1],
            np.concatenate([a0, [u0]]),
            method="SLSQP",
            constraints=self.constraints(),
            callback=callback,
            options=dict(maxiter=max_iter, ftol=1e-9, eps=1e-6),
        )
        return best["a"], best["score"], result, history


def save_ramp(path, problem, a, grid_us=1.25):
    """Write a ramp in Transport-Optimizer's CSV format."""
    t = np.arange(0.0, problem.duration_us + 1e-9, grid_us)
    v = V_PI * np.clip(np.interp(t, problem.t, problem.s(a)), 0.0, 1.0)
    v = np.maximum.accumulate(v)
    slew = np.diff(v) / np.diff(t)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("time_us,voltage_V,slew_to_next_V_per_us\n")
        for i in range(t.size):
            nxt = f"{slew[i]:.6f}" if i < slew.size else ""
            handle.write(f"{t[i]:.6f},{v[i]:.6f},{nxt}\n")
