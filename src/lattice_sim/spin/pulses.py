"""Microwave pulse trains with amplitude and detuning errors (Bloch picture).

Pulses are SU(2) rotations exp(-i (theta/2) n.sigma) with
n = (Omega cos phi, Omega sin phi, delta) / |.|, the rotation angle scaled by
(1 + eps) for an amplitude error eps and lengthened by the detuning.
Approximation: (1 + eps) multiplies the whole generalized Rabi rotation,
detuning part included; physically eps scales Omega only, which also tilts
the axis, n ~ (Omega (1 + eps) cos phi, Omega (1 + eps) sin phi, delta). The
difference is second order in eps x delta / Omega. Free
evolution between pulses is ideal here (its noise is handled by
``spin.echo``), so this module isolates the pulse-error contribution.

Contrast is the length of the shot-averaged Bloch vector's transverse
component before the final pi/2, i.e. the fringe amplitude when the final
pulse phase is scanned.
"""

import numpy as np

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)


def rotation(angle, phase, eps=0.0, detuning_over_rabi=0.0):
    n = np.array([np.cos(phase), np.sin(phase), detuning_over_rabi])
    norm = np.linalg.norm(n)
    n = n / norm
    a = angle * (1 + eps) * norm
    gen = n[0] * SX + n[1] * SY + n[2] * SZ
    return np.cos(a / 2) * np.eye(2) - 1j * np.sin(a / 2) * gen


TRAINS = {
    "CP (pi_x)": lambda n: [0.0] * n,
    "CPMG (pi_y)": lambda n: [np.pi / 2] * n,
    "XY-8": lambda n: ([0, np.pi / 2, 0, np.pi / 2, np.pi / 2, 0, np.pi / 2, 0] * (n // 8 + 1))[:n],
    "XY-4": lambda n: ([0, np.pi / 2, 0, np.pi / 2] * (n // 4 + 1))[:n],
}


def train_contrast(kind, n_pi, eps_sigma=0.01, detuning_over_rabi=0.0, n_shots=400, seed=0,
                   input_phase=0.0):
    """Shot-averaged fringe contrast for pi/2 - n x pi - (pi/2 scanned).

    eps is a quasi-static amplitude error drawn per shot (Gaussian, sigma
    eps_sigma) and common to all pulses of that shot. input_phase rotates the
    first pi/2 (0: about x, so CPMG pi pulses are parallel to the spin).
    """
    rng = np.random.default_rng(seed)
    phases = TRAINS[kind](n_pi)
    bloch = np.zeros(3)
    psi0 = np.array([1, 0], complex)
    for _ in range(n_shots):
        eps = rng.normal(0, eps_sigma)
        u = rotation(np.pi / 2, input_phase, eps, detuning_over_rabi)
        for ph in phases:
            u = rotation(np.pi, ph, eps, detuning_over_rabi) @ u
        psi = u @ psi0
        rho = np.outer(psi, psi.conj())
        bloch += np.real([np.trace(rho @ SX), np.trace(rho @ SY), np.trace(rho @ SZ)])
    bloch /= n_shots
    return float(np.hypot(bloch[0], bloch[1]))


def worst_input_contrast(kind, n_pi, **kw):
    """Minimum contrast over input phases (an unknown qubit state in a gate)."""
    return min(train_contrast(kind, n_pi, input_phase=p, **kw)
               for p in np.linspace(0, np.pi, 7))
