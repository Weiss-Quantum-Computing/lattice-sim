"""Microwave carrier and motional sidebands of a trapped atom versus lattice alignment.

The microwave photon carries no useful momentum, so a |4,-4> -> |3,-3>
spectrum shows vibrational sidebands only because the two states see
different potentials: with the rotating arm turned by a small theta (and
with any ellipticity, arm imbalance or bias-field tilt) the |4,-4> and
|3,-3> wells are displaced and have different curvatures, and the
Franck-Condon overlaps <n'|n> between the two vibrational ladders are no
longer diagonal. The sideband frequencies are the trap frequencies of the
final state, the carrier shift is the differential energy of the two
vibrational ground states (Zeeman + scalar + vector light shift + zero
point), and the sideband strengths measure the differential displacement.

Model. Each lattice pair is treated as a 1D ``fields.lattice_pair.LatticePair``
with its own depth, spacing, extinction ratio, rotation and arm amplitudes,
and the bias field projected into the pair's local frame (fringe,
propagation, static polarization). The vibrational levels of each state are
the eigenstates of the 1D potential from ``fields.potential.StatePotential``
(exact diagonalization of the 16 x 16 Hamiltonian at every point), found
with a banded fourth-order finite-difference Hamiltonian. Lines are
weighted by a thermal distribution over the initial vibrational level and
by |<n'|n>|^2; the carrier includes the three axes' zero-point and
light-shift contributions added as independent 1D problems (the vector
shifts of different pairs add vectorially in reality; this is a first
approximation).

Beam centering enters as the local field amplitude of each arm at the
atoms, exp(-r^2 / w^2) for a transverse offset r and waist w; the crossing
angle enters through the spacing lambda / (2 sin alpha).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import eig_banded

from ..atom.ground_state import eigenlevels
from ..constants import H, HBAR, KB, LATTICE_WAVELENGTH_M, MASS_CS
from ..fields.lattice_pair import LatticePair
from ..fields.potential import StatePotential

UPPER, LOWER = (4, -4), (3, -3)      # the stretched microwave transition

# Lab layout (params/apparatus.yaml): x and z pairs enter from the same
# side, y from 90 deg counter-clockwise, z vertical. Local frame of each
# pair = (fringe direction, propagation direction, static polarization),
# right-handed, so that LatticePair's x, y, z are these three directions.
PAIR_FRAMES = {
    "x": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    "y": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float),
    "z": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float),
}


def b_in_pair_frame(axis, b_lab_gauss):
    """Bias-field vector in the local frame of pair ``axis``."""
    return PAIR_FRAMES[axis] @ np.asarray(b_lab_gauss, float)


def vibrational_levels(x, v_hz, n_levels=4, half_window_m=1.2e-6):
    """Lowest vibrational levels of a 1D potential V(x)/h (Hz) on a uniform grid.

    Returns (energies_hz, wavefunctions) with wavefunctions normalized on
    the grid (sum |psi|^2 = 1), solved on +/- half_window_m around the
    minimum with a banded 4th-order finite-difference kinetic operator.
    """
    x = np.asarray(x, float)
    v = np.asarray(v_hz, float)
    dx = x[1] - x[0]
    centre = int(np.argmin(v))
    half = int(np.ceil(half_window_m / dx))
    lo, hi = max(0, centre - half), min(x.size, centre + half + 1)
    n = hi - lo
    c = HBAR / (2 * MASS_CS * dx**2) / (2 * np.pi)
    band = np.zeros((3, n))
    band[0] = c * 2.5 + v[lo:hi]
    band[1, :-1] = c * (-4.0 / 3.0)
    band[2, :-2] = c * (1.0 / 12.0)
    energies, vectors = eig_banded(band, lower=True, select="i", select_range=(0, n_levels - 1))
    psi = np.zeros((n_levels, x.size))
    psi[:, lo:hi] = vectors.T
    psi /= np.sqrt(np.sum(psi**2, axis=1))[:, None]
    # sign convention: positive at the centre of mass of |psi|^2 for n = 0
    return energies, psi


def franck_condon(psi_a, psi_b):
    """|<b n'|a n>|^2 matrix, shape (n_b, n_a), for wavefunctions on one grid."""
    return np.abs(psi_b @ psi_a.T) ** 2


def thermal_weights(f_hz, temperature_k, n_levels):
    """Boltzmann weights of the lowest n_levels of an oscillator at f_hz."""
    if temperature_k <= 0:
        w = np.zeros(n_levels)
        w[0] = 1.0
        return w
    r = np.exp(-H * f_hz / (KB * temperature_k))
    w = r ** np.arange(n_levels)
    return w / w.sum()


@dataclass
class AxisConfig:
    """One lattice pair as seen by an atom at one site."""
    axis: str                              # 'x', 'y' or 'z' (sets the local frame)
    trap_frequency_hz: float               # theta = 0, pure linear, unit amplitudes
    spacing_m: float = 4.8374e-6
    theta: float = 0.0                     # rotation of this pair's EOM, rad
    extinction_ratio: object = np.inf
    handedness: int = -1
    rotation_sense: int = 1
    static_amplitude: float = 1.0          # local field amplitude of each arm
    rotating_amplitude: float = 1.0

    def pair(self):
        return LatticePair(spacing_m=self.spacing_m, rotation_sense=self.rotation_sense,
                           extinction_ratio=self.extinction_ratio, handedness=self.handedness,
                           static_amplitude=self.static_amplitude,
                           rotating_amplitude=self.rotating_amplitude)


@dataclass
class SiteConfig:
    """The three pairs, the bias field and the temperature at one site."""
    axes: list
    b_lab_gauss: np.ndarray
    temperature_k: float = 1e-6            # after Raman sideband cooling; n_bar ~ 0.4 at 19 kHz
    n_levels: int = 6
    grid_points: int = 1201
    grid_half_width_m: float = 1.5e-6


def axis_levels(cfg: AxisConfig, b_lab_gauss, n_levels=4, grid_points=1201, half_width_m=1.5e-6,
                states=(UPPER, LOWER)):
    """Vibrational energies (Hz, including the state's light shift) and
    wavefunctions of each state along one axis, on a common grid."""
    model = StatePotential(cfg.pair(), b_in_pair_frame(cfg.axis, b_lab_gauss),
                           trap_frequency_hz=cfg.trap_frequency_hz)
    x = np.linspace(-half_width_m, half_width_m, grid_points)
    out = {}
    for state in states:
        v = model.potential(state, x, cfg.theta)
        e, psi = vibrational_levels(x, v, n_levels)
        out[state] = dict(x=x, v_hz=v, energies_hz=e, psi=psi,
                          x0_m=float(np.sum(x * psi[0] ** 2)))
    return out


def spectrum(site: SiteConfig, states=(UPPER, LOWER), max_dn=2):
    """Lines of the |upper> -> |lower> spectrum for the site configuration.

    Frequencies are microwave frequencies E(upper, n) - E(lower, n') of the
    |upper, n> -> |lower, n'> lines. Returns a dict with 'carrier_hz' (the
    n = 0 -> n' = 0 line, absolute), 'zeeman_hz' (the free-atom transition
    at the bias field), 'carrier_shift_hz' (carrier minus free atom: the
    differential light shift plus zero point), 'axes' (per axis: trap
    frequencies of both states, displacement, Franck-Condon matrix) and
    'lines': list of (axis, dn = n' - n, offset from the carrier in Hz,
    weight), with weight = thermal x |<n'|n>|^2. A line with dn = +1 (one
    quantum added) lies at -f_lower; dn = -1 lies at +f_upper. The carrier's
    own weight is the product over axes of the thermal diagonal terms, and
    each sideband carries the diagonal terms of the other two axes.
    """
    upper, lower = states
    b = np.asarray(site.b_lab_gauss, float)
    levels = eigenlevels(b)
    zeeman = levels[upper][0] - levels[lower][0]
    per_axis, carrier_shift, carrier_weight, lines = {}, 0.0, 1.0, []
    for cfg in site.axes:
        lv = axis_levels(cfg, b, site.n_levels, site.grid_points, site.grid_half_width_m, states)
        e_u, e_l = lv[upper]["energies_hz"], lv[lower]["energies_hz"]
        fc = franck_condon(lv[upper]["psi"], lv[lower]["psi"])        # (n_lower, n_upper)
        f_u, f_l = float(e_u[1] - e_u[0]), float(e_l[1] - e_l[0])
        w = thermal_weights(f_u, site.temperature_k, site.n_levels)
        carrier_shift += float(e_u[0] - e_l[0])
        diag = float(np.sum(w * np.diag(fc)))
        carrier_weight *= diag
        per_axis[cfg.axis] = dict(f_upper_hz=f_u, f_lower_hz=f_l,
                                  x0_upper_m=lv[upper]["x0_m"], x0_lower_m=lv[lower]["x0_m"],
                                  displacement_m=lv[lower]["x0_m"] - lv[upper]["x0_m"],
                                  franck_condon=fc, thermal=w, diagonal=diag, levels=lv)
        for n in range(site.n_levels):
            for n2 in range(site.n_levels):
                dn = n2 - n
                if dn == 0 or abs(dn) > max_dn:
                    continue
                offset = float((e_u[n] - e_u[0]) - (e_l[n2] - e_l[0]))
                lines.append((cfg.axis, dn, offset, float(w[n] * fc[n2, n])))
    # a sideband on one axis leaves the other axes in their diagonal (carrier)
    # terms, so every line carries the product of the other axes' diagonals
    lines = [(ax, dn, off, wt * carrier_weight / per_axis[ax]["diagonal"]) for ax, dn, off, wt in lines]
    return dict(zeeman_hz=float(zeeman), carrier_hz=float(zeeman + carrier_shift),
                carrier_shift_hz=float(carrier_shift), carrier_weight=carrier_weight,
                axes=per_axis, lines=lines)


def collapse_lines(lines, axis=None):
    """Sum the weights of lines by (axis, dn): {(axis, dn): (mean offset Hz, weight)}."""
    out = {}
    for ax, dn, off, wt in lines:
        if axis is not None and ax != axis:
            continue
        key = (ax, dn)
        s_off, s_w = out.get(key, (0.0, 0.0))
        out[key] = (s_off + off * wt, s_w + wt)
    return {k: (v[0] / v[1] if v[1] > 0 else 0.0, v[1]) for k, v in out.items()}


def line_shape(freq_hz, lines, carrier_weight, width_hz):
    """Lorentzian stick spectrum on a frequency axis relative to the carrier."""
    f = np.asarray(freq_hz, float)
    y = carrier_weight * width_hz**2 / (f**2 + width_hz**2)
    for _, _, off, wt in lines:
        y = y + wt * width_hz**2 / ((f - off) ** 2 + width_hz**2)
    return y


# ------------------------------------------------------------ alignment helpers
def amplitude_from_offset(offset_m, waist_m=60e-6):
    """Field amplitude of a Gaussian beam at a transverse offset, relative to the axis."""
    return float(np.exp(-np.asarray(offset_m, float) ** 2 / waist_m**2))


def spacing_from_crossing(half_angle_rad, wavelength_m=LATTICE_WAVELENGTH_M):
    return wavelength_m / (2 * np.sin(half_angle_rad))


def nominal_site(trap_hz=(19e3, 16e3, 12.5e3), rotation_fraction=0.03, extinction_ratio=3e4,
                 handedness=-1, b_gauss=1.39304, b_angle_deg=45.0, temperature_k=1e-6):
    """The lab's sideband configuration: full depth, a small rotation of
    every pair (rotation_fraction of 180 deg) so the sidebands show, 1 uK."""
    axes = [AxisConfig(ax, f, theta=rotation_fraction * np.pi, extinction_ratio=extinction_ratio,
                       handedness=handedness)
            for ax, f in zip("xyz", trap_hz)]
    phi = np.deg2rad(b_angle_deg)
    return SiteConfig(axes, b_gauss * np.array([np.cos(phi), np.sin(phi), 0.0]), temperature_k)


def with_alignment(site: SiteConfig, axis, static_offset_m=0.0, rotating_offset_m=0.0,
                   crossing_half_angle_deg=None, waist_m=60e-6, **axis_fields):
    """Copy of ``site`` with one pair's centering, crossing angle and any
    AxisConfig field changed."""
    axes = []
    for cfg in site.axes:
        if cfg.axis == axis:
            changes = dict(static_amplitude=amplitude_from_offset(static_offset_m, waist_m),
                           rotating_amplitude=amplitude_from_offset(rotating_offset_m, waist_m))
            if crossing_half_angle_deg is not None:
                changes["spacing_m"] = spacing_from_crossing(np.deg2rad(crossing_half_angle_deg))
            changes.update(axis_fields)
            cfg = replace(cfg, **changes)
        axes.append(cfg)
    return replace(site, axes=axes)
