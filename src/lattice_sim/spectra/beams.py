"""Beam geometry of one lattice pair: offsets and tilts of each arm, and the
resulting per-site lattice parameters across the occupied region.

Lab frame for the x pair: fringes along x, beams propagating near +y, static
polarization along z (the same frame as ``fields.lattice_pair``). Each arm
has a wave vector

    k = k (sin phi cos delta_v, cos phi cos delta_v, sin delta_v),

phi the in-plane angle from +y toward +x (+alpha for the static arm,
-alpha for the rotating arm, plus the arm's in-plane tilt) and delta_v its
vertical tilt. A tilt about a pivot a distance L upstream also moves the
beam at the atoms by L x tilt, which is added to the arm's transverse
offset there. Each arm is a Gaussian beam of waist w about its axis.

From the two wave vectors:

* the fringe wave vector dk = k_s - k_r sets the spacing a = 2 pi / |dk| and
  the fringe normal n; a differential vertical tilt rotates n out of the
  x axis by ~ delta_v / sin(alpha) (11 x the tilt for alpha = 5 deg), a
  differential in-plane tilt changes alpha and hence the spacing;
* the static polarization is z made perpendicular to k_s; the pair frame is
  (n, e_pol x n, e_pol), and the bias field is projected into it, so a
  tilt of the fringes changes the projection of the fictitious field (along
  the beams) on the bias field;
* at a site r the field amplitude of each arm is exp(-d^2 / w^2) with d its
  distance from the arm's axis, so an offset of the beams gives a depth
  gradient across the lattice (4 r0 / w^2 per unit length for an offset r0
  of both arms) and a differential offset gives an imbalance gradient.

The depth at a site is the product of the two amplitudes (times the
nominal depth) and the trap frequency scales as sqrt(depth) / a. The 1D
spectrum model of ``sidebands`` is then evaluated per site; the shear of the
lattice (sites at different heights sit at different x) does not enter a
separable 1D-per-pair model.

Other pairs can be described the same way in their own frames (``PAIR_FRAMES``).
"""

from dataclasses import dataclass, replace

import numpy as np

from ..constants import LATTICE_WAVELENGTH_M
from .sidebands import PAIR_FRAMES, AxisConfig, SiteConfig, spectrum

NOMINAL_HALF_ANGLE = np.arcsin(LATTICE_WAVELENGTH_M / (2 * 4.8374e-6))


@dataclass
class ArmGeometry:
    """One arm: transverse offset of its axis at the atoms (along the fringe
    direction and along the static polarization), tilts, waist, pivot."""
    offset_fringe_m: float = 0.0       # along the fringe direction (lab x for the x pair)
    offset_pol_m: float = 0.0          # along the static polarization (lab z for the x pair)
    tilt_inplane_rad: float = 0.0      # rotates k about the polarization axis, toward +fringe
    tilt_vertical_rad: float = 0.0     # rotates k toward +polarization axis
    waist_m: float = 60e-6
    pivot_m: float = 0.0               # distance of the tilt pivot upstream of the atoms

    def axis_point(self):
        """A point on the beam axis in the pair's local lab frame (at y = 0)."""
        return np.array([self.offset_fringe_m + self.pivot_m * self.tilt_inplane_rad, 0.0,
                         self.offset_pol_m + self.pivot_m * self.tilt_vertical_rad])


def wave_vector(phi, delta_v, k=2 * np.pi / LATTICE_WAVELENGTH_M):
    return k * np.array([np.sin(phi) * np.cos(delta_v), np.cos(phi) * np.cos(delta_v), np.sin(delta_v)])


@dataclass
class PairGeometry:
    """Both arms of one pair plus the nominal lattice parameters."""
    static: ArmGeometry
    rotating: ArmGeometry
    half_angle_rad: float = NOMINAL_HALF_ANGLE
    wavelength_m: float = LATTICE_WAVELENGTH_M

    def wave_vectors(self):
        k_s = wave_vector(self.half_angle_rad + self.static.tilt_inplane_rad, self.static.tilt_vertical_rad)
        k_r = wave_vector(-self.half_angle_rad + self.rotating.tilt_inplane_rad, self.rotating.tilt_vertical_rad)
        return k_s, k_r

    def lattice(self):
        """spacing (m), fringe normal, pair frame (3x3, rows = local x, y, z in the pair's lab frame)."""
        k_s, k_r = self.wave_vectors()
        dk = k_s - k_r
        spacing = 2 * np.pi / np.linalg.norm(dk)
        n = dk / np.linalg.norm(dk)
        e_pol = np.array([0.0, 0.0, 1.0])
        k_hat = k_s / np.linalg.norm(k_s)
        e_pol = e_pol - (e_pol @ k_hat) * k_hat
        e_pol = e_pol - (e_pol @ n) * n
        e_pol /= np.linalg.norm(e_pol)
        return spacing, n, np.array([n, np.cross(e_pol, n), e_pol])

    def amplitudes(self, r):
        """Field amplitude of each arm at point r (pair's lab frame), relative to its axis."""
        out = []
        for arm, k in zip((self.static, self.rotating), self.wave_vectors()):
            k_hat = k / np.linalg.norm(k)
            rel = np.asarray(r, float) - arm.axis_point()
            d2 = rel @ rel - (rel @ k_hat) ** 2
            out.append(float(np.exp(-d2 / arm.waist_m**2)))
        return out


def site_axis_config(base: AxisConfig, geometry: PairGeometry, r_local, spacing_nominal_m=4.8374e-6,
                     to_lab=None):
    """AxisConfig of one pair at a site, from the beam geometry.

    ``base`` carries the nominal depth (as ``trap_frequency_hz`` at
    ``spacing_nominal_m`` with unit amplitudes), rotation, extinction ratio
    and handedness. ``r_local`` is the site position in the pair's lab frame.
    The returned frame maps the global lab frame to the pair frame:
    ``to_lab`` (3x3) is the pair-local-lab -> global-lab rotation
    (``PAIR_FRAMES[axis].T``) and defaults to the pair's standard frame.
    """
    spacing, _, frame_local = geometry.lattice()
    a_s, a_r = geometry.amplitudes(r_local)
    to_lab = PAIR_FRAMES[base.axis] if to_lab is None else np.asarray(to_lab, float)
    frame = frame_local @ to_lab
    return replace(base, spacing_m=spacing, trap_frequency_hz=base.trap_frequency_hz * spacing_nominal_m / spacing,
                   static_amplitude=a_s, rotating_amplitude=a_r, frame=frame)


def spectra_along(site: SiteConfig, axis, geometry: PairGeometry, points_local, max_dn=1):
    """Spectrum of the site configuration at several positions (pair-local
    lab frame), with pair ``axis`` built from ``geometry`` at each one."""
    out = []
    for r in points_local:
        axes = [site_axis_config(cfg, geometry, r) if cfg.axis == axis else cfg for cfg in site.axes]
        out.append(spectrum(replace(site, axes=axes), max_dn=max_dn))
    return out


def gradients(site: SiteConfig, axis, geometry: PairGeometry, half_span_m=10e-6):
    """Carrier and trap-frequency gradients (Hz/m) along the fringe direction
    and along the polarization direction, from sites +/- half_span_m from
    the centre, plus the centre spectrum."""
    pts = {"centre": np.zeros(3),
           "fringe+": np.array([half_span_m, 0, 0]), "fringe-": np.array([-half_span_m, 0, 0]),
           "pol+": np.array([0, 0, half_span_m]), "pol-": np.array([0, 0, -half_span_m])}
    sp = dict(zip(pts, spectra_along(site, axis, geometry, pts.values())))
    out = {"centre": sp["centre"]}
    for name in ("fringe", "pol"):
        plus, minus = sp[name + "+"], sp[name + "-"]
        out[f"carrier_hz_per_m_{name}"] = (plus["carrier_shift_hz"] - minus["carrier_shift_hz"]) / (2 * half_span_m)
        out[f"f_lower_hz_per_m_{name}"] = (plus["axes"][axis]["f_lower_hz"] - minus["axes"][axis]["f_lower_hz"]) / (2 * half_span_m)
    return out
