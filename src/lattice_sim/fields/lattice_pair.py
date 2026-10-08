"""Field of one crossed lattice pair with a polarization-rotating arm.

Geometry (Du 2025, Sec. 3.3.2, Eq. 3.7, written here for the x pair):

* both arms propagate nearly along +y and cross at a full angle 2 alpha, so
  k_static = k (sin a, cos a, 0) and k_rot = k (-sin a, cos a, 0);
  the fringes run along x with spacing a_lat = lambda / (2 sin alpha);
* the static arm is linearly polarized along z (perpendicular to the plane
  of the pair);
* the rotating arm's linear polarization is turned by theta about its own
  k vector: e_r = cos(theta) z + s sin(theta) u, with u = k_rot_hat x z in
  the x-y plane and s = +/-1 the sense of rotation;
* the relative phase between the arms is pi, so at theta = 0 the intensity
  minimum (a blue-detuned trap) is at x = 0.

Imperfect polarization of the rotating arm is an extinction ratio ER: a
component in quadrature, orthogonal to the major axis, with amplitude
eta = 1/sqrt(ER) and handedness h = +/-1. ER = inf is pure linear. ER may
be a number or a callable of theta (e.g. a measured
``fields.extinction.ExtinctionProfile``).

Positions are in metres along the fringe direction; fields are dimensionless
with unit amplitude per arm.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticePair:
    wavelength_m: float = 843.222e-9
    spacing_m: float = 4.8374e-6
    rotation_sense: int = 1
    extinction_ratio: object = np.inf  # number, or callable ER(theta)
    handedness: int = 1
    relative_phase: float = np.pi
    # field amplitude of each arm at the atoms relative to nominal (sqrt of
    # the local intensity): beam centering and arm imbalance enter here. The
    # depth calibration of ``fields.potential`` refers to unit amplitudes.
    static_amplitude: float = 1.0
    rotating_amplitude: float = 1.0

    @property
    def half_angle(self):
        return np.arcsin(self.wavelength_m / (2 * self.spacing_m))

    @property
    def k(self):
        return 2 * np.pi / self.wavelength_m

    def polarizations(self, theta):
        """Unit polarization vectors (static, rotating) at rotation theta."""
        a = self.half_angle
        z = np.array([0.0, 0.0, 1.0])
        u = np.array([np.cos(a), np.sin(a), 0.0])  # k_rot_hat x z
        major = np.cos(theta) * z + self.rotation_sense * np.sin(theta) * u
        minor = -np.sin(theta) * z + self.rotation_sense * np.cos(theta) * u
        er = self.extinction_ratio(theta) if callable(self.extinction_ratio) else self.extinction_ratio
        eta = 0.0 if np.isinf(er) else 1 / np.sqrt(float(er))
        rot = (major + 1j * self.handedness * eta * minor) / np.sqrt(1 + eta**2)
        return z.astype(complex), rot

    def field(self, x, theta):
        """Complex field vectors, shape (len(x), 3), along the fringe axis (y = z = 0)."""
        x = np.atleast_1d(np.asarray(x, float))
        e_s, e_r = self.polarizations(theta)
        q = self.k * np.sin(self.half_angle)
        phase_s = self.static_amplitude * np.exp(1j * q * x)[:, None]
        phase_r = self.rotating_amplitude * np.exp(-1j * q * x + 1j * self.relative_phase)[:, None]
        return phase_s * e_s + phase_r * e_r

    def intensity_and_spin(self, x, theta):
        """|E|^2 and Im(E* x E) along x; shapes (N,) and (N, 3)."""
        e = self.field(x, theta)
        intensity = np.sum(np.abs(e) ** 2, axis=1)
        spin = np.imag(np.cross(np.conj(e), e))
        return intensity, spin
