"""Differential physics of a transport pair |3,+m> / |3,-m> along one lattice axis.

Everything the echo engine needs about the light and the field, as functions
of the rotation angle theta, for one axis at the 10 kHz lattice scale:

* dE0(theta): differential trap-bottom energy E(+m) - E(-m) (Hz) for pure
  linear polarization (the fictitious-field "drag"), scaling as I^2;
* S(theta): d(Delta E)/d(eta) (Hz per unit ellipticity), scaling as I;
* the differential Zeeman sensitivity kappa_B = d f(+m, -m)/dB (Hz/mG);
* the storage-qubit (|4,+1>, |3,-1>) differential light shift at the trap
  bottom (Hz), from the hyperfine-resolved scalar polarizability.

Other axes are approximated by scaling the x-pair curves by their depth
relative to x (depth ~ f_trap^2) until the y and z geometries are modelled.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from ..atom.ground_state import field_sensitivity, magic_field
from ..atom.polarizability import differential_scalar_ratio
from ..constants import LATTICE_WAVELENGTH_M
from ..fields.lattice_pair import LatticePair
from ..fields.potential import StatePotential

A_LAT = 4.8374e-6
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _bottom(model, state, theta):
    sign = 1 if state[1] > 0 else -1
    guess = sign * A_LAT / 2 * np.sin(theta / 2) ** 2
    x0, f = model.minimum(state, theta, guess=guess)
    return model.potential(state, np.array([x0]), theta)[0] + f / 2


def compute_curves(m, thetas, eta=0.005, trap_hz=10e3):
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    lin = StatePotential(LatticePair(), b, trap_frequency_hz=trap_hz)
    ell = StatePotential(LatticePair(extinction_ratio=1 / eta**2, handedness=-1), b,
                         trap_frequency_hz=trap_hz)
    e0, s = [], []
    for th in thetas:
        d0 = _bottom(lin, (3, m), th) - _bottom(lin, (3, -m), th)
        d1 = _bottom(ell, (3, m), th) - _bottom(ell, (3, -m), th)
        e0.append(d0)
        s.append((d1 - d0) / eta)
    return np.array(e0), np.array(s)


@dataclass
class PairPhysics:
    m: int
    thetas: np.ndarray
    de0_hz: np.ndarray
    s_hz: np.ndarray
    trap_hz: float = 10e3

    def __post_init__(self):
        self._e0 = CubicSpline(self.thetas, self.de0_hz)
        self._s = CubicSpline(self.thetas, self.s_hz)
        b0 = magic_field()
        self.kappa_b_hz_per_mg = field_sensitivity((3, self.m), (3, -self.m), b0) / 1e3
        # storage qubit differential light shift in the vibrational ground
        # state: the light shift follows the mean potential energy <V>, which
        # is h f / 4 per axis for a harmonic ground state (half the zero-point
        # energy; per vibrational level it rises by h f / 2, as in the Wu
        # test), times the hyperfine differential ratio of the scalar
        # polarizability. The three axes are x and the y, z approximations
        # of ``echo.AXIS_SCALE`` (depth ~ f^2).
        f_axes = self.trap_hz * np.array([1.0, 8.0 / 9.5, 6.25 / 9.5])
        self.storage_ls_hz = abs(differential_scalar_ratio(LATTICE_WAVELENGTH_M)) * np.sum(f_axes) / 4

    def de0(self, theta):
        return self._e0(np.clip(theta, 0, np.pi))

    def s(self, theta):
        return self._s(np.clip(theta, 0, np.pi))

    def ddtheta(self, theta, eta):
        th = np.clip(theta, 0, np.pi)
        return self._e0(th, 1) + eta * self._s(th, 1)

    @classmethod
    def load_or_compute(cls, m, cache_dir=DATA_DIR, n=91):
        path = Path(cache_dir) / f"pair_physics_m{m}.npz"
        if path.exists():
            d = np.load(path)
            return cls(m, d["thetas"], d["de0"], d["s"])
        thetas = np.linspace(0, np.pi, n)
        e0, s = compute_curves(m, thetas)
        np.savez(path, thetas=thetas, de0=e0, s=s)
        return cls(m, thetas, e0, s)
