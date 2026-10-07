"""Ground-state light shift of caesium far from resonance.

Model. The lattice (843.222 nm) is detuned by 3.8 THz from D2 and 20 THz
from D1, far more than any excited-state hyperfine splitting (< 0.6 GHz).
In that regime the light shift of 6S1/2 is, per field-mode amplitude E,

    H_LS = K * sum_l w_l [ c^s_l g^s_l(F) |E|^2 + c^v_l g^v_l Im(E* x E) . sigma ]

with sigma = 2J the electron-spin Pauli vector, line weights w_l = Gamma_l /
omega_l^3, angular coefficients c^s = (2, 1) and c^v = (1, -1) for (D2, D1)
(Grimm, Weidemueller and Ovchinnikov, Adv. At. Mol. Opt. Phys. 42, 95 (2000),
Eq. 20, written as an operator with g_F m_F -> <sigma_z>), and

    g^s_l(F) = 1/(omega - omega_l(F)) - 1/(omega + omega_l(F))
    g^v_l    = 1/(omega - omega_l)    + 1/(omega + omega_l)

including the counter-rotating terms. The scalar part uses the detuning from
each ground hyperfine level F, which gives the F=3/F=4 differential light
shift; the vector part uses the fine-structure detuning (its F dependence is
a 0.1 % correction). The tensor shift of a J = 1/2 ground state vanishes at
this order.

The overall constant K is never needed: the simulation fixes the lattice
depth from a measured trap frequency, and only the ratios below enter.
Sign convention: blue detuning gives a positive (repulsive) scalar shift.
"""

import numpy as np

from ..constants import A_HFS_HZ, I_NUC, NU_HFS_HZ

C_LIGHT = 299_792_458.0

# Steck, Cesium D Line Data rev. 2.1.6, Tables 5 and 6
LINES = {
    "D2": {"nu_Hz": 351.725_718_50e12, "gamma_Hz": 5.234e6, "c_s": 2.0, "c_v": 1.0},
    "D1": {"nu_Hz": 335.116_048_807e12, "gamma_Hz": 4.575e6, "c_s": 1.0, "c_v": -1.0},
}


def hyperfine_offset_hz(f):
    """Energy of ground level F relative to the hyperfine centroid, in Hz."""
    return A_HFS_HZ / 2 * (f * (f + 1) - I_NUC * (I_NUC + 1) - 0.75)


def _weights(wavelength_m, f=None):
    nu = C_LIGHT / wavelength_m
    scalar, vector = 0.0, 0.0
    for line in LINES.values():
        nu_l = line["nu_Hz"]
        # Transition frequency from level F: excited centroid minus ground level
        nu_f = nu_l - (hyperfine_offset_hz(f) if f is not None else 0.0)
        w = line["gamma_Hz"] / nu_l**3
        scalar += w * line["c_s"] * (1 / (nu - nu_f) - 1 / (nu + nu_f))
        vector += w * line["c_v"] * (1 / (nu - nu_l) + 1 / (nu + nu_l))
    return scalar, vector


def scalar_coefficient(wavelength_m, f=None):
    """Scalar light-shift coefficient for ground level F (arbitrary units)."""
    return _weights(wavelength_m, f)[0]


def vector_coefficient(wavelength_m):
    """Vector light-shift coefficient (same units as the scalar one)."""
    return _weights(wavelength_m)[1]


def vector_to_scalar_ratio(wavelength_m):
    """Vector / scalar coefficient, using the hyperfine-centroid detuning."""
    return vector_coefficient(wavelength_m) / scalar_coefficient(wavelength_m)


def differential_scalar_ratio(wavelength_m):
    """(U_F=4 - U_F=3) / U for the scalar shift: the hyperfine differential
    light shift per unit depth. Negative means F=4 is shifted less."""
    s4 = scalar_coefficient(wavelength_m, 4)
    s3 = scalar_coefficient(wavelength_m, 3)
    return (s4 - s3) / scalar_coefficient(wavelength_m)


def effective_hyperfine_estimate(wavelength_m):
    """Closed-form check of the differential ratio: -nu_hfs / Delta_eff."""
    nu = C_LIGHT / wavelength_m
    d2 = nu - LINES["D2"]["nu_Hz"]
    d1 = nu - LINES["D1"]["nu_Hz"]
    # Delta_eff from (2/d2^2 + 1/d1^2) / (2/d2 + 1/d1), ignoring line-weight differences
    return -NU_HFS_HZ * (2 / d2**2 + 1 / d1**2) / (2 / d2 + 1 / d1)
