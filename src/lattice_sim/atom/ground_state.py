"""Caesium 6S_1/2 hyperfine-Zeeman structure.

Two independent routes to the same energies, so each checks the other:

* ``breit_rabi``: the closed-form Breit-Rabi expression for a static field of
  magnitude B (Steck Eq. 24 sign conventions).
* ``ground_hamiltonian``: the full 16 x 16 Hamiltonian
  H = A I.J + mu_B (g_J J + g_I I).B in the uncoupled |m_J, m_I> basis for an
  arbitrary field vector. This is the form later modules extend with the
  vector and tensor light shifts, which do not reduce to Breit-Rabi.

All energies are returned as frequencies in Hz; fields are in gauss.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from ..constants import A_HFS_HZ, G_I, G_J, I_NUC, MU_B_HZ_PER_G, NU_HFS_HZ

F_UPPER = I_NUC + 0.5
F_LOWER = I_NUC - 0.5


def spin_matrices(j):
    """Return (Jx, Jy, Jz) for spin j in the |j, m> basis, m descending."""
    m = np.arange(j, -j - 1, -1)
    jp = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), k=1)
    jx = 0.5 * (jp + jp.T)
    jy = -0.5j * (jp - jp.T)
    jz = np.diag(m)
    return jx, jy, jz


@lru_cache(maxsize=None)
def _operators():
    j_ops = spin_matrices(0.5)
    i_ops = spin_matrices(I_NUC)
    eye_j, eye_i = np.eye(2), np.eye(int(2 * I_NUC + 1))
    J = [np.kron(op, eye_i) for op in j_ops]
    I = [np.kron(eye_j, op) for op in i_ops]
    F = [a + b for a, b in zip(J, I)]
    return J, I, F


def ground_hamiltonian(b_vector_gauss):
    """Hyperfine plus Zeeman Hamiltonian / h (Hz) for a field vector in gauss."""
    J, I, _ = _operators()
    b = np.asarray(b_vector_gauss, dtype=float)
    hyperfine = A_HFS_HZ * sum(i @ j for i, j in zip(I, J))
    zeeman = MU_B_HZ_PER_G * sum(
        component * (G_J * j + G_I * i) for component, j, i in zip(b, J, I)
    )
    return hyperfine + zeeman


def eigenlevels(b_vector_gauss):
    """Diagonalize the ground manifold; label states by (F, m_F) along B.

    m_F is the expectation of F projected on the field direction; F is
    assigned by energy ordering (upper nine states are F = 4). That is right
    at any field: the stretched |4,-4> stays above every F = 3 level (by
    >= 1.2 GHz from 100 G to 100 kG in Breit-Rabi; the 2026-10-15 review
    suspected a crossing near 3 kG, which does not occur;
    ``tests/test_ground_state.py``). Returns
    {(F, m_F): (energy_Hz, eigenvector)}.
    """
    b = np.asarray(b_vector_gauss, dtype=float)
    norm = np.linalg.norm(b)
    if norm == 0:
        raise ValueError("labels need a nonzero field to define the axis")
    axis = b / norm
    _, _, F = _operators()
    f_along = sum(a * f for a, f in zip(axis, F))
    energies, vectors = np.linalg.eigh(ground_hamiltonian(b))
    out = {}
    n_lower = int(2 * F_LOWER + 1)
    for index in range(energies.size):
        vec = vectors[:, index]
        m = float(np.real(vec.conj() @ f_along @ vec))
        f = F_LOWER if index < n_lower else F_UPPER
        out[(f, int(round(m)))] = (float(energies[index]), vec)
    if len(out) != energies.size:
        raise RuntimeError("m_F labels collided; field too small or too large")
    return out


def breit_rabi(f, m, b_gauss):
    """Breit-Rabi energy / h (Hz) of |F, m_F> in a field of magnitude b_gauss.

    Energies are measured from the hyperfine centroid, the same zero as
    ``ground_hamiltonian``.
    """
    two_i_plus_1 = 2 * I_NUC + 1
    x = (G_J - G_I) * MU_B_HZ_PER_G * b_gauss / NU_HFS_HZ
    sign = 1.0 if f == F_UPPER else -1.0
    root_argument = 1 + 4 * m * x / two_i_plus_1 + x**2
    if abs(m) == F_UPPER:
        # Stretched states: the square root is exactly |1 +/- x|, keep its sign
        root = 1 + np.sign(m) * x
    else:
        root = np.sqrt(root_argument)
    return (
        -NU_HFS_HZ / (2 * two_i_plus_1)
        + G_I * MU_B_HZ_PER_G * m * b_gauss
        + sign * 0.5 * NU_HFS_HZ * root
    )


def transition_frequency(upper, lower, b_gauss):
    """f(upper) - f(lower) in Hz for (F, m_F) tuples at field magnitude b_gauss."""
    return breit_rabi(*upper, b_gauss) - breit_rabi(*lower, b_gauss)


def field_sensitivity(upper, lower, b_gauss, step=1e-4):
    """d f / d B in Hz per gauss (central difference)."""
    return (
        transition_frequency(upper, lower, b_gauss + step)
        - transition_frequency(upper, lower, b_gauss - step)
    ) / (2 * step)


def magic_field(upper=(4, 1), lower=(3, -1), bracket=(0.5, 3.0)):
    """Field (G) where the transition is first-order field insensitive."""
    return brentq(lambda b: field_sensitivity(upper, lower, b), *bracket, xtol=1e-10)
