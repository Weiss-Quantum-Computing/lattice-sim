"""Exact transport in a harmonic trap with moving centre and changing frequency.

For H = p^2/2m + (1/2) m omega(t)^2 (x - x0(t))^2 a Gaussian stays Gaussian
(Heller 1975). Writing psi ~ exp(i [alpha (x-q)^2 + p (x-q) + gamma] / hbar):

    dq/dt = p / m
    dp/dt = -m omega^2 (q - x0)
    dalpha/dt = -2 alpha^2 / m - m omega^2 / 2

with alpha(0) = i m omega(0) / 2 for the initial ground state. The lab-frame
ground-state population is the overlap of this Gaussian with the
instantaneous ground state (centre x0, alpha_g = i m omega / 2), evaluated in
closed form. No grid and no basis truncation: this is the reference against
which the numerical propagators are checked on harmonic potentials.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import HBAR, MASS_CS


def _overlap(alpha1, q1, p1, alpha2, q2):
    """|<g|psi>|^2 for normalized Gaussians; g has zero momentum."""
    a1 = -2j * alpha1 / HBAR
    a2 = -2j * alpha2 / HBAR
    n1 = (a1.real / np.pi) ** 0.25
    n2 = (a2.real / np.pi) ** 0.25
    s = a1 + np.conj(a2)
    b = a1 * q1 + np.conj(a2) * q2 + 1j * p1 / HBAR
    c = -a1 * q1**2 / 2 - np.conj(a2) * q2**2 / 2 - 1j * p1 * q1 / HBAR
    amp = n1 * n2 * np.sqrt(2 * np.pi / s) * np.exp(b**2 / (2 * s) + c)
    return np.abs(amp) ** 2


def propagate_gaussian(t_us, theta, center_of_theta, frequency_hz_of_theta,
                       sample_times_us=None, rtol=1e-10, atol=1e-12):
    """Lab-frame P0(t) and centre lag for a harmonic trap following theta(t)."""
    t_us = np.asarray(t_us, float)
    theta = np.asarray(theta, float)
    t_s = (t_us - t_us[0]) * 1e-6

    def th(time_s):
        return np.interp(time_s, t_s, theta)

    def omega(time_s):
        return 2 * np.pi * float(frequency_hz_of_theta(th(time_s)))

    def x0(time_s):
        return float(center_of_theta(th(time_s)))

    # Scaled variables: q in um, p in m*um/s ... keep SI but split alpha re/im
    def rhs(time_s, y):
        q, p, ar, ai = y
        w = omega(time_s)
        alpha = ar + 1j * ai
        dalpha = -2 * alpha**2 / MASS_CS - MASS_CS * w**2 / 2
        return [p / MASS_CS, -MASS_CS * w**2 * (q - x0(time_s)), dalpha.real, dalpha.imag]

    w0 = omega(0.0)
    y0 = [x0(0.0), 0.0, 0.0, MASS_CS * w0 / 2]
    if sample_times_us is None:
        sample_times_us = t_us
    eval_s = (np.asarray(sample_times_us, float) - t_us[0]) * 1e-6
    # Scale state so the tolerances are meaningful for every component
    scale = np.array([1e-6, MASS_CS * 1e-3, MASS_CS * w0, MASS_CS * w0])
    sol = solve_ivp(lambda t, z: np.asarray(rhs(t, z * scale)) / scale, (0.0, t_s[-1]),
                    np.asarray(y0) / scale, t_eval=eval_s, rtol=rtol, atol=atol,
                    method="DOP853", max_step=1e-6)
    q, p, ar, ai = sol.y * scale[:, None]
    alpha = ar + 1j * ai
    p0 = np.array([
        _overlap(alpha[i], q[i], p[i], 1j * MASS_CS * omega(s) / 2, x0(s))
        for i, s in enumerate(sol.t)
    ])
    lag = q - np.array([x0(s) for s in sol.t])
    return np.asarray(sample_times_us, float), p0, lag
