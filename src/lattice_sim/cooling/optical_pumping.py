"""Optical pumping to |4,-4> and the motional heating it causes.

Pumping. sigma- light on 6S1/2 F=4 -> 6P3/2 F'=4 (OP) and F=3 -> F'=4
(repump), both through the same fibre (Li p.94), quantization axis along
the beam. Every ground state except |4,-4> has exactly one sigma- excitation,
to |4', m-1>, so the number of scattered photons until the atom is in the
dark state is a Markov chain over the ground sublevels with the 6P3/2 F'=4
branching ratios (Wigner 3j/6j, exact). Rates do not enter the photon count.

Recoil. Each photon is absorbed along the OP beam and emitted with the
dipole pattern of its polarization (pi: sin^2, sigma: (1 + cos^2)/2 about the
quantization axis). For lattice axis a, a momentum kick hbar k u along the
fringe direction (u = k_hat . a_hat) acts on the vibrational state as
exp(i k u x); with real eigenfunctions |<m|e^{ikux}|n>|^2 depends on |u|
only. Kicks along different lattice axes from the same photon are treated
as independent (the three axes are orthogonal; the correlation is a
second-order effect), but the photon NUMBER is shared by the three axes.
"""

from functools import lru_cache
from math import factorial, sqrt

import numpy as np

from ..constants import HBAR, MASS_CS

D2_WAVELENGTH_M = 852.347e-9
I_NUC2 = 7                       # 2 I
J_G2, J_E2 = 1, 3                # 2 J for 6S1/2 and 6P3/2
F_EXCITED = 4
# ASSUMPTION: the OP beam (and quantization field) lies at 45 deg in the
# lab x-y plane (Kumar p.13: bias along x+y; Du p.77: field set by the OP
# beam). Lattice fringe directions are lab x, y, z (spectra.sidebands).
OP_BEAM_DIRECTION_LAB = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
AXIS_DIRECTIONS = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
# ASSUMPTION: probability per OP pulse that an atom already in |4,-4> scatters
# from the polarization impurity of the OP beam (one pi excitation to
# |4',-4> followed by re-pumping). Default 0; the measured value is open
# (data request 29).
DARK_EXCURSION_PROB = 0.0


# ---------------------------------------------------------------- angular algebra
def _f(n):
    return factorial(int(round(n)))


def wigner3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol (Racah formula); arguments may be half-integers."""
    if abs(m1 + m2 + m3) > 1e-9 or min(j1 + j2 - j3, j1 - j2 + j3, -j1 + j2 + j3) < 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    tri = _f(j1 + j2 - j3) * _f(j1 - j2 + j3) * _f(-j1 + j2 + j3) / _f(j1 + j2 + j3 + 1)
    pre = sqrt(tri * _f(j1 + m1) * _f(j1 - m1) * _f(j2 + m2) * _f(j2 - m2) * _f(j3 + m3) * _f(j3 - m3))
    kmin = int(round(max(0, j2 - j3 - m1, j1 - j3 + m2)))
    kmax = int(round(min(j1 + j2 - j3, j1 - m1, j2 + m2)))
    s = 0.0
    for k in range(kmin, kmax + 1):
        s += (-1) ** k / (_f(k) * _f(j1 + j2 - j3 - k) * _f(j1 - m1 - k) * _f(j2 + m2 - k)
                          * _f(j3 - j2 + m1 + k) * _f(j3 - j1 - m2 + k))
    return (-1) ** int(round(j1 - j2 - m3)) * pre * s


def wigner6j(j1, j2, j3, j4, j5, j6):
    """Wigner 6j symbol {j1 j2 j3; j4 j5 j6} (Racah formula)."""
    def delta(a, b, c):
        if min(a + b - c, a - b + c, -a + b + c) < 0:
            return None
        return _f(a + b - c) * _f(a - b + c) * _f(-a + b + c) / _f(a + b + c + 1)
    ds = [delta(j1, j2, j3), delta(j1, j5, j6), delta(j4, j2, j6), delta(j4, j5, j3)]
    if any(d is None for d in ds):
        return 0.0
    pre = sqrt(np.prod(ds))
    a = [j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3]
    b = [j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4]
    s = 0.0
    for t in range(int(round(max(a))), int(round(min(b))) + 1):
        s += (-1) ** t * _f(t + 1) / (np.prod([_f(t - x) for x in a]) * np.prod([_f(x - t) for x in b]))
    return pre * s


@lru_cache(maxsize=None)
def decay_branching(m_excited):
    """{(F, m_F, q): probability} for 6P3/2 |F'=4, m'> -> 6S1/2 |F, m_F> + photon
    of polarization q = m' - m_F (sums to 1)."""
    out = {}
    j, jp, i = J_G2 / 2, J_E2 / 2, I_NUC2 / 2
    for f in (3, 4):
        six = wigner6j(j, jp, 1, F_EXCITED, f, i)
        for m in range(-f, f + 1):
            q = m_excited - m
            if abs(q) > 1:
                continue
            three = wigner3j(f, 1, F_EXCITED, m, q, -m_excited)
            out[(f, m, q)] = (2 * F_EXCITED + 1) * (2 * f + 1) * (J_E2 + 1) * six**2 * three**2
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def photon_number_distribution(start=(3, -3), k_max=40, dark_excursion=False):
    """P(k photons scattered before reaching |4,-4>) and the fraction of
    emitted photons that are pi-polarized, for an atom starting in ``start``.
    With dark_excursion=True the atom starts in |4,-4> and is promoted once
    to |4',-4> by a pi photon (the polarization-impurity path)."""
    states = [(f, m) for f in (3, 4) for m in range(-f, f + 1)]
    index = {s: i for i, s in enumerate(states)}
    dark = index[(4, -4)]
    n = len(states)
    step = np.zeros((n, n))          # one absorption + emission from each bright state
    pi_weight = np.zeros(n)          # probability that this emission is pi
    for (f, m), i in index.items():
        if i == dark:
            continue
        me = m - 1
        for (f2, m2, q), p in decay_branching(me).items():
            step[index[(f2, m2)], i] += p
            if q == 0:
                pi_weight[i] += p
    p = np.zeros(n)
    dist = np.zeros(k_max + 1)
    n_pi, n_tot = 0.0, 0.0
    if dark_excursion:
        for (f2, m2, q), w in decay_branching(-4).items():
            p[index[(f2, m2)]] += w
            if q == 0:
                n_pi += w
        n_tot += 1.0
        dist_offset = 1
    else:
        p[index[start]] = 1.0
        dist_offset = 0
    dist[dist_offset] += p[dark]
    p[dark] = 0.0
    for k in range(dist_offset + 1, k_max + 1):
        n_pi += float(pi_weight @ p)
        n_tot += float(p.sum())
        p = step @ p
        dist[k] = p[dark]
        p[dark] = 0.0
    return dist, (n_pi / n_tot if n_tot else 0.0)


# ---------------------------------------------------------------- recoil matrices
def lamb_dicke(f_hz, wavelength_m=D2_WAVELENGTH_M):
    """eta = k x_zp with x_zp = sqrt(hbar / 2 m omega)."""
    return 2 * np.pi / wavelength_m * np.sqrt(HBAR / (2 * MASS_CS * 2 * np.pi * f_hz))


def kick_matrices(x, psi, kappas):
    """|<m| exp(i kappa x) |n>|^2 for each kappa (rad/m): shape (len(kappas), N, N)."""
    dx = 1.0  # psi is normalized on the grid (sum |psi|^2 = 1)
    out = np.empty((len(kappas), psi.shape[0], psi.shape[0]))
    for j, kap in enumerate(kappas):
        ph = psi * np.exp(1j * kap * x)[None, :]
        out[j] = np.abs(ph @ psi.T * dx).T ** 2
    return out


def harmonic_kick_probability(m, n, eta):
    """Analytic |<m| D(i eta) |n>|^2 for a harmonic oscillator."""
    from scipy.special import eval_genlaguerre, gammaln
    lo, hi = min(m, n), max(m, n)
    d = hi - lo
    log = -eta**2 + 2 * d * np.log(eta) + gammaln(lo + 1) - gammaln(hi + 1)
    return float(np.exp(log) * eval_genlaguerre(lo, d, eta**2) ** 2)


def emission_projection_weights(axis_dir, quant_dir, pi_fraction, n_nodes=41, n_theta=64, n_phi=128):
    """Weights on |u| nodes in [0, 1] of the projection u = k_hat . axis for an
    emitted photon (mixture of pi and sigma dipole patterns about quant_dir)."""
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    st = np.sqrt(1 - ct**2)
    q = quant_dir / np.linalg.norm(quant_dir)
    e1 = np.cross(q, [0.0, 0.0, 1.0]) if abs(q[2]) < 0.9 else np.cross(q, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(q, e1)
    pattern = pi_fraction * 0.75 * st**2 + (1 - pi_fraction) * 0.375 * (1 + ct**2)   # per unit d(cos) per 2pi
    nodes = np.linspace(0, 1, n_nodes)
    w = np.zeros(n_nodes)
    for c, s, wc, pat in zip(ct, st, wt, pattern):
        k = (c * q[None, :] + s * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        u = np.abs(k @ axis_dir)
        idx = np.clip(np.rint(u * (n_nodes - 1)).astype(int), 0, n_nodes - 1)
        np.add.at(w, idx, wc * pat / n_phi)
    return nodes, w / w.sum()


def photon_cycle_matrix(x, psi, axis, pi_fraction, beam=OP_BEAM_DIRECTION_LAB,
                        wavelength_m=D2_WAVELENGTH_M):
    """Vibrational transition matrix (axis a, [m, n]) for one absorbed and
    one emitted photon."""
    k = 2 * np.pi / wavelength_m
    a = AXIS_DIRECTIONS[axis]
    u_abs = abs(float(beam @ a))
    absorb = kick_matrices(x, psi, [k * u_abs])[0]
    nodes, w = emission_projection_weights(a, beam, pi_fraction)
    keep = w > 0
    emit = np.tensordot(w[keep], kick_matrices(x, psi, k * nodes[keep]), axes=1)
    return emit @ absorb
