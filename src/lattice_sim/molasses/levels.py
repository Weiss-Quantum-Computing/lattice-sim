"""Atomic level structure for the molasses model.

The excited states are adiabatically eliminated (|Delta| >> Gamma, Omega).
For one excited hyperfine level F' the ground-state operator

    M_F'(u) = W^dag W,   W = sum_q u_q D_q,   u_q = e_q^* . u

(u the complex field vector in units of one beam's amplitude, D_q the dipole
matrix F -> F' for photon polarization q, normalised so that the stretched
transition |F=F_max, m=F> -> |F', F'> has element 1) is a Hermitian operator
bilinear in u*, u. Any such operator on a single F manifold is a combination
of rank 0, 1 and 2 tensors, which the three operators

    B1 = |u|^2,  B2 = (u*.F)(u.F),  B3 = (u.F)(u*.F)

span exactly ([B2 - B3] = i (u* x u).F carries the vector part). The
coefficients (a, b, c) of M_F' = a B1 + b B2 + c B3 are found by a least-squares
fit over random fields; ``Transition.check`` verifies the fit is exact to
machine precision. This keeps the kernel independent of the number of
excited levels and avoids building W^dag W at every time step.

Normalisation: a single beam of intensity I drives the stretched transition
with Rabi frequency Omega_0 = Gamma sqrt(I / (2 I_sat)), I_sat the stretched
(cycling) saturation intensity. Then the effective ground Hamiltonian is

    H/hbar = sum_F' (Omega_0^2/4) (Delta_F' - i Gamma/2)/(Delta_F'^2 + Gamma^2/4) M_F'(u)

(Delta_F' = omega_L - omega_F'), whose anti-Hermitian part is the decay
(-i/2) Gamma'_F' M_F'. Data: D. A. Steck, Cesium D Line Data rev. 2.1.6.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sympy import Rational, S
from sympy.physics.wigner import wigner_3j, wigner_6j

# Cs D2 (Steck Tables 3, 7)
GAMMA_D2 = 2 * np.pi * 5.234e6          # rad/s, natural linewidth
LAMBDA_D2 = 852.347_27e-9               # m
ISAT_CYCLING_MW_CM2 = 1.1023            # mW/cm^2, |4,+-4> -> |5',+-5'> sigma+-
# excited hyperfine offsets omega_F' - omega_5' (Hz)
D2_OFFSETS_HZ = {5: 0.0, 4: -251.0916e6, 3: -(251.0916e6 + 201.2871e6)}
G_F4 = 0.250_390                        # g_F of 6S_1/2 F = 4 (Steck Table 7, low field)
G_F3 = -0.251_17

# spherical basis vectors e_q, q = -1, 0, +1 (index q + 1)
E_SPH = np.array([[1, -1j, 0] / np.sqrt(2),
                  [0, 0, 1],
                  [-1, -1j, 0] / np.sqrt(2)], dtype=complex)


def spin_ops(f):
    """(Fx, Fy, Fz) for spin f in the |f, m> basis, m ascending (-f ... f)."""
    m = np.arange(-f, f + 1)
    fp = np.zeros((len(m), len(m)))
    for i in range(len(m) - 1):            # F+ |m> = sqrt(f(f+1) - m(m+1)) |m+1>
        fp[i + 1, i] = np.sqrt(f * (f + 1) - m[i] * (m[i] + 1))
    fx = 0.5 * (fp + fp.T)
    fy = -0.5j * (fp - fp.T)
    fz = np.diag(m).astype(float)
    return fx.astype(complex), fy, fz.astype(complex)


def _half(x):
    return Rational(int(round(2 * x)), 2)


@lru_cache(maxsize=None)
def hyperfine_dipole(f, fp, j=0.5, jp=1.5, i_nuc=3.5):
    """D_q (q = -1, 0, 1), shape (3, 2F'+1, 2F+1): <F' m+q| d_q |F m> up to a
    common factor <J'||d||J>, Wigner-Eckart (Steck Eqs. 33-36). If i_nuc is None the
    levels are fine-structure levels (no hyperfine coupling), f = j, fp = jp."""
    d = np.zeros((3, int(round(2 * fp + 1)), int(round(2 * f + 1))))
    F, Fp = _half(f), _half(fp)
    if i_nuc is None:
        red = S(1)
    else:
        J, Jp, I = _half(j), _half(jp), _half(i_nuc)
        red = (-1) ** int(round(fp + j + 1 + i_nuc)) * ((2 * F + 1) * (2 * Fp + 1)) ** S.Half \
            * wigner_6j(J, Jp, 1, Fp, F, I)
    for qi, q in enumerate((-1, 0, 1)):
        for mi in range(int(round(2 * f + 1))):
            m = -f + mi
            mp = m + q
            if abs(mp) > fp + 1e-9:
                continue
            Mp, M = _half(mp), _half(m)
            # <F' m'| d_q |F m> = (-1)^(F'-m') (F' 1 F; -m' q m) <F'||d||F>
            w = wigner_3j(Fp, 1, F, -Mp, q, M)
            d[qi, int(round(mp + fp)), mi] = float((-1) ** int(round(fp - mp)) * w * red)
    return d


@dataclass
class ExcitedLevel:
    label: str
    offset_rad_s: float          # omega_F' - omega_ref (rad/s); Delta_F' = Delta_ref - offset
    d: np.ndarray                # (3, dim_e, dim_g), stretched-normalised
    coeffs: np.ndarray = field(default=None)   # (a, b, c) of M = a B1 + b B2 + c B3
    branch_to_g: float = 1.0     # probability that a decay returns to the modelled ground manifold


@dataclass
class Transition:
    name: str
    f_ground: float
    gamma: float                 # rad/s
    wavelength: float            # m
    isat_mw_cm2: float
    g_f: float                   # ground Lande factor (Zeeman omega = g_f mu_B B F)
    levels: list
    mass: float

    @property
    def dim(self):
        return int(round(2 * self.f_ground + 1))

    @property
    def k(self):
        return 2 * np.pi / self.wavelength

    def spin(self):
        return spin_ops(self.f_ground)

    def w_matrix(self, level, u):
        """W = sum_q u_q D_q for a Cartesian field vector u."""
        uq = E_SPH.conj() @ np.asarray(u, dtype=complex)   # u_q = e_q^* . u
        return np.tensordot(uq, level.d, axes=(0, 0))

    def m_exact(self, level, u):
        w = self.w_matrix(level, u)
        return w.conj().T @ w

    def basis(self, u):
        fx, fy, fz = self.spin()
        u = np.asarray(u, dtype=complex)
        uf = u[0] * fx + u[1] * fy + u[2] * fz
        ucf = u.conj()[0] * fx + u.conj()[1] * fy + u.conj()[2] * fz
        return (np.vdot(u, u).real * np.eye(self.dim), ucf @ uf, uf @ ucf)

    def fit_coeffs(self, n=24, seed=1):
        rng = np.random.default_rng(seed)
        for lev in self.levels:
            rows, rhs = [], []
            for _ in range(n):
                u = rng.normal(size=3) + 1j * rng.normal(size=3)
                b = self.basis(u)
                rows.append(np.stack([x.ravel() for x in b], axis=1))
                rhs.append(self.m_exact(lev, u).ravel())
            a = np.concatenate(rows)
            y = np.concatenate(rhs)
            sol, *_ = np.linalg.lstsq(a, y, rcond=None)
            lev.coeffs = sol.real.copy()
        return self

    def check(self, n=10, seed=7):
        """Largest relative deviation of the decomposition from W^dag W."""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for lev in self.levels:
            for _ in range(n):
                u = rng.normal(size=3) + 1j * rng.normal(size=3)
                exact = self.m_exact(lev, u)
                approx = sum(c * b for c, b in zip(lev.coeffs, self.basis(u)))
                worst = max(worst, np.abs(exact - approx).max() / np.abs(exact).max())
        return worst

    def decay_to_ground(self, level):
        """sum_{q, m} |D_q[m', m]|^2 for each excited m' (branching into this ground manifold)."""
        return np.einsum("qem,qem->e", level.d, level.d)


def cs_d2_f4(include=(5, 4, 3)):
    """Cs D2 from 6S_1/2 F = 4 to 6P_3/2 F' in ``include``; detuning reference F' = 5."""
    d5 = hyperfine_dipole(4, 5)
    norm = d5[2, 10, 8]                     # |4,4> -> |5',5'>, q = +1
    levels = []
    for fp in include:
        d = hyperfine_dipole(4, fp) / norm
        lev = ExcitedLevel(f"F'={fp}", 2 * np.pi * D2_OFFSETS_HZ[fp], d)
        levels.append(lev)
    from ..constants import MASS_CS
    tr = Transition("Cs D2, F=4", 4.0, GAMMA_D2, LAMBDA_D2, ISAT_CYCLING_MW_CM2, G_F4, levels, MASS_CS)
    for lev in tr.levels:
        b = tr.decay_to_ground(lev)
        lev.branch_to_g = float(b.mean())
    return tr.fit_coeffs()


def j_half_to_three_half(mass=None, gamma=GAMMA_D2, wavelength=LAMBDA_D2):
    """J = 1/2 -> J' = 3/2 without hyperfine structure (Dalibard & Cohen-Tannoudji 1989)."""
    d = hyperfine_dipole(0.5, 1.5, i_nuc=None)
    d = d / d[2, 3, 1]                      # |1/2,1/2> -> |3/2,3/2>, q = +1
    from ..constants import MASS_CS
    lev = ExcitedLevel("J'=3/2", 0.0, d)
    tr = Transition("J=1/2 -> 3/2", 0.5, gamma, wavelength, ISAT_CYCLING_MW_CM2, 0.0, [lev],
                    MASS_CS if mass is None else mass)
    return tr.fit_coeffs()


def rabi0_sq(transition, intensity_mw_cm2):
    """Omega_0^2 (rad^2/s^2) of one beam on the stretched transition."""
    return transition.gamma ** 2 * intensity_mw_cm2 / (2 * transition.isat_mw_cm2)
