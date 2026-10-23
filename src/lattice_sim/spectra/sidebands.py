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

Large rotation angles (any theta in [-pi, pi]). As theta grows the two
states' wells move apart in opposite directions (spin-dependent transport).
Two exact symmetries of the field (constant extinction ratio, any
handedness, arm amplitudes and bias field) fix the geometry:

* V(x; -theta) = V(-x; theta): the field at -theta is the mirror image of
  the field at theta, so the spectrum is even in theta (site offsets k -> -k);
* V(x; theta + pi) = V(x + a/2; theta): turning the arm by pi reverses its
  field, which moves the fringes by half a spacing. Each state's well
  therefore moves by exactly a/2 between theta = 0 and pi, the two states
  in opposite directions, so at theta = +/-pi the final state's well of the
  neighbouring site (k = +/-1) sits on the initial state's well (the
  collisional-gate geometry), and the spectrum at pi - theta equals the
  spectrum at theta with the site offsets k and 1 - k (or -1 - k) exchanged.

The atom starts in the upper state's well of site 0, followed continuously
from theta = 0 (``track_wells``). The lower state's wells of site 0 and of
the neighbouring sites (``SITE_OFFSETS``) are solved on one grid placed
over all wells involved, and every line is the sum over the final site:
with a periodic lattice the lines of different k lie at the same
frequency. ``lines`` keeps its format (summed over k); ``site_lines``
resolves the site. At small theta only k = 0 has a nonzero overlap and the
results equal the earlier same-site model.

Potentials are the adiabatic branches: the eigenvalue of the 16 x 16
Hamiltonian with the energy rank that |F, m_F> has at the bias field alone
(``state_potentials``). This equals the maximal-overlap labelling of
``StatePotential.potentials`` wherever that labelling is defined (overlap
> 0.5) and stays defined where the fictitious field turns the quantization
axis away from the bias field; ``label_purity`` reports how much of the
bias-field |F, m_F> the state at the trap minimum still is.
"""

import functools
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import CubicSpline
from scipy.linalg import eig_banded
from scipy.sparse.linalg import eigsh

from ..atom.ground_state import eigenlevels
from ..constants import H, HBAR, KB, LATTICE_WAVELENGTH_M, MASS_CS
from ..fields.lattice_pair import LatticePair
from ..fields.potential import StatePotential

UPPER, LOWER = (4, -4), (3, -3)      # the stretched microwave transition

# Large rotation angles (module docstring)
SITE_OFFSETS = (-1, 0, 1)            # final-state sites, in lattice spacings from the atom's own site
LEVEL_WINDOW_M = 1.2e-6              # half-width of the eigenvalue window around each well (starting value)
WINDOW_EDGE_LIMIT = 1e-8             # |psi|^2 of the highest level at the window edge / its peak: above, the
                                     # window grows (z pair, 0.3 depth, 60 levels: 4.5e-3 at 1.2 um moves
                                     # E_59 by 4.8 kHz, 1.8e-6 moves E_45 by 0.5 Hz, 1e-11 moves nothing)
WINDOW_GROWTH = 1.25                 # factor per enlargement
WINDOW_MAX_SPACINGS = 0.45           # window half-width cap in lattice spacings (the barrier is at a/2)

# Vibrational levels kept (``n_levels_for``): enough that at most DROPPED_POPULATION of the
# thermal population lies above the last one, between MIN_LEVELS and MAX_LEVELS
MIN_LEVELS = 6
MAX_LEVELS = 60                      # cap; near it the levels approach the lattice barrier (0.3 depth,
                                     # 5 uK, z pair: ~3 % above level 60, reported as dropped_population)
DROPPED_POPULATION = 1e-3
TRACK_MAX_STEP_RAD = np.pi / 24      # largest theta step of the well tracker; a well must move < a/2 per
                                     # step (fastest seen: x-pair |3,-3> next to 90 deg, ~0.9 um per step)
TRACK_POINTS_PER_PERIOD = 64         # potential samples per lattice period in the tracker
LABEL_PURITY_LIMIT = 0.5             # below: the |F, m_F> label refers to the local field, not the bias field
FC_NOISE_FLOOR = 1e-24               # |<n'|n>|^2 below this is eigenvector round-off (tails ~1e-16) -> 0

# Energies versus angle (``energies_vs_angle``)
ENERGY_WINDOW_M = 0.8e-6             # half-width of the window for the two lowest levels (> 6 x_zp at 3 kHz)
ENERGY_COARSE_STEP = 4               # potential diagonalized every 4th grid point, cubic spline in between
HARMONIC_FIT_STEP_M = 25e-9          # spacing of the 5 points of the quartic fit at the trap minimum

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


def _hamiltonian_window(x, v_hz, half_window_m, centre=None):
    """Window (lo, hi), kinetic coefficient and potential of the
    finite-difference Hamiltonian on +/- half_window_m around grid index
    ``centre`` (default: the global minimum)."""
    x = np.asarray(x, float)
    v = np.asarray(v_hz, float)
    dx = x[1] - x[0]
    centre = int(np.argmin(v)) if centre is None else int(centre)
    half = int(np.ceil(half_window_m / dx))
    lo, hi = max(0, centre - half), min(x.size, centre + half + 1)
    c = HBAR / (2 * MASS_CS * dx**2) / (2 * np.pi)
    return lo, hi, c, v[lo:hi]


def _embed(x, lo, hi, vectors, n_levels):
    psi = np.zeros((n_levels, np.asarray(x).size))
    psi[:, lo:hi] = vectors.T
    return psi / np.sqrt(np.sum(psi**2, axis=1))[:, None]


def vibrational_levels(x, v_hz, n_levels=4, half_window_m=1.2e-6, centre=None):
    """Lowest vibrational levels of a 1D potential V(x)/h (Hz) on a uniform grid.

    Returns (energies_hz, wavefunctions) with wavefunctions normalized on
    the grid (sum |psi|^2 = 1), solved on +/- half_window_m around the
    minimum (or around grid index ``centre`` when the grid holds several
    wells) with a fourth-order finite-difference kinetic operator. The
    levels come from a sparse shift-invert Lanczos solve about a shift just
    below the potential minimum (the kinetic stencil is positive
    semidefinite, so no level lies below it);
    ``vibrational_levels_banded`` is the LAPACK banded reference it is
    tested against, about 50 x slower.
    """
    lo, hi, c, v = _hamiltonian_window(x, v_hz, half_window_m, centre)
    n = hi - lo
    off1, off2 = np.full(n - 1, c * (-4.0 / 3.0)), np.full(n - 2, c * (1.0 / 12.0))
    h = sparse.diags([off2, off1, c * 2.5 + v, off1, off2], [-2, -1, 0, 1, 2], format="csc")
    # a fixed start vector makes ARPACK, and so every figure, repeat bit for bit
    energies, vectors = eigsh(h, k=n_levels, sigma=v.min() - 1.0, which="LM", v0=np.ones(n))
    order = np.argsort(energies)
    return energies[order], _embed(x, lo, hi, vectors[:, order], n_levels)


def vibrational_levels_banded(x, v_hz, n_levels=4, half_window_m=1.2e-6, centre=None):
    """Reference for ``vibrational_levels``: the LAPACK banded eigensolver."""
    lo, hi, c, v = _hamiltonian_window(x, v_hz, half_window_m, centre)
    n = hi - lo
    band = np.zeros((3, n))
    band[0] = c * 2.5 + v
    band[1, :-1] = c * (-4.0 / 3.0)
    band[2, :-2] = c * (1.0 / 12.0)
    energies, vectors = eig_banded(band, lower=True, select="i", select_range=(0, n_levels - 1))
    return energies, _embed(x, lo, hi, vectors, n_levels)


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


def thermal_weights_from_energies(energies_hz, temperature_k):
    """Boltzmann weights of levels at the given energies (Hz), normalized over them:
    exp(-h (E_n - E_0) / kT), the anharmonic level spacing included."""
    e = np.asarray(energies_hz, float)
    if temperature_k <= 0:
        w = np.zeros(e.size)
        w[0] = 1.0
        return w
    w = np.exp(-H * (e - e[0]) / (KB * temperature_k))
    return w / w.sum()


def dropped_population(energies_hz, temperature_k):
    """Thermal population above the last of the given levels, as a fraction of
    the whole: the ladder continued geometrically with its top spacing (an
    estimate; z pair at 0.3 depth, 5 uK, 60 levels: 3.03 % against 2.97 % from
    a Boltzmann sum over 110 computed levels, tests/test_sideband_explorer.py)."""
    e = np.asarray(energies_hz, float)
    if temperature_k <= 0 or e.size < 2:
        return 0.0
    beta = H / (KB * temperature_k)
    kept = float(np.sum(np.exp(-beta * (e - e[0]))))
    r = float(np.exp(-beta * max(float(e[-1] - e[-2]), 0.0)))
    if r >= 1.0:
        return 1.0
    tail = float(np.exp(-beta * (e[-1] - e[0]))) * r / (1.0 - r)
    return tail / (kept + tail)


def n_levels_for(temperature_k, trap_hz):
    """Vibrational levels per pair so that at most DROPPED_POPULATION of the
    thermal population of a harmonic ladder at the lowest of the trap
    frequencies lies above the last kept level; at least MIN_LEVELS, at most
    MAX_LEVELS (at the cap more is dropped: ``spectrum`` reports it)."""
    if temperature_k <= 0:
        return MIN_LEVELS
    r = float(np.exp(-H * float(np.min(trap_hz)) / (KB * temperature_k)))
    if r <= 0:
        return MIN_LEVELS
    if r >= 1:
        return MAX_LEVELS
    need = int(np.ceil(np.log(DROPPED_POPULATION) / np.log(r))) + 1
    return int(min(MAX_LEVELS, max(MIN_LEVELS, need)))


# ------------------------------------------------------- adiabatic potentials
def _ranks(model):
    """Energy rank (0 = lowest of 16) of every |F, m_F> at the bias field alone."""
    states = list(model._bias)
    energies = np.array([model._bias[s][0] for s in states])
    order = np.argsort(energies)
    if np.min(np.diff(energies[order])) < 1.0:
        raise ValueError("bias-field levels (nearly) degenerate: energy rank does not label the states")
    return {states[i]: rank for rank, i in enumerate(order)}


def state_potentials(model: StatePotential, states, x, theta):
    """{state: potential / h (Hz)} of the adiabatic branch of each |F, m_F>.

    The branch is the eigenvalue of the full Hamiltonian with the energy
    rank that |F, m_F> has at the bias field alone, minus its bias-field
    energy. Within each hyperfine manifold the levels are ordered by m_F
    along the total (bias + fictitious) field and do not cross unless that
    field vanishes, so this is the potential an atom follows adiabatically;
    it equals ``StatePotential.potentials`` wherever that is defined and
    never raises its "strongly mixed" error.
    """
    energies = np.linalg.eigvalsh(model.hamiltonians(np.atleast_1d(np.asarray(x, float)), theta))
    ranks = _ranks(model)
    return {s: energies[:, ranks[s]] - model._bias[s][0] for s in states}


def label_purity(model: StatePotential, state, x, theta):
    """|<F, m_F at the bias field | adiabatic state>|^2 at positions x (1 = pure)."""
    _, vectors = np.linalg.eigh(model.hamiltonians(np.atleast_1d(np.asarray(x, float)), theta))
    rank = _ranks(model)[state]
    return np.abs(vectors[:, :, rank] @ model._bias[state][1].conj()) ** 2


def _period_minima(x, v):
    """Local minima of a potential sampled over one period (periodic
    neighbours), refined by a parabola through three samples."""
    prev, nxt = np.roll(v, 1), np.roll(v, -1)
    idx = np.flatnonzero((v <= prev) & (v < nxt))
    dx = x[1] - x[0]
    out = []
    for i in idx:
        curv = prev[i] - 2 * v[i] + nxt[i]
        out.append(x[i] + (0.5 * (prev[i] - nxt[i]) / curv * dx if curv > 0 else 0.0))
    return np.array(out)


def _track_step(model, states, positions, theta):
    """New well positions at theta: of each state's minima (all copies, period a)
    the one nearest to its previous position."""
    a = model.pair.spacing_m
    x = (np.arange(TRACK_POINTS_PER_PERIOD) / TRACK_POINTS_PER_PERIOD - 0.5) * a
    pots = state_potentials(model, states, x, theta)
    out = {}
    for s in states:
        minima = _period_minima(x, pots[s])
        p = positions[s]
        if minima.size == 0:                      # flat potential: nothing to follow
            out[s] = p
            continue
        copies = minima + a * np.round((p - minima) / a)
        out[s] = float(copies[np.argmin(np.abs(copies - p))])
    return out


def track_wells(model: StatePotential, states, thetas, start=0.0, max_step=TRACK_MAX_STEP_RAD):
    """Well position (m) of each state at each theta, followed continuously
    from theta = 0, where every state sits at ``start`` (the atom's site).

    Positive and negative angles are followed outwards from 0 in steps of at
    most ``max_step``, so a list of angles costs one sweep. The positions
    are accurate to ~1 nm (a parabola through 64 samples per period); the
    spectrum refines them on its own grid. Returns {state: array}.
    """
    thetas = np.atleast_1d(np.asarray(thetas, float))
    out = {s: np.empty(thetas.size) for s in states}
    for side in (1, -1):
        sel = np.flatnonzero(thetas >= 0) if side > 0 else np.flatnonzero(thetas < 0)
        if sel.size == 0:
            continue
        order = sel[np.argsort(side * thetas[sel], kind="stable")]
        pos = _track_step(model, states, {s: float(start) for s in states}, 0.0)
        prev = 0.0
        for i in order:
            th = thetas[i]
            n = int(np.ceil(abs(th - prev) / max_step - 1e-9))
            for t in np.linspace(prev, th, n + 1)[1:]:
                pos = _track_step(model, states, pos, t)
            prev = th
            for s in states:
                out[s][i] = pos[s]
    return out


def _parabola_min(x, v, i):
    """(x_min, v_min) of the parabola through samples i - 1, i, i + 1."""
    if i <= 0 or i >= v.size - 1:
        return float(x[i]), float(v[i])
    y0, y1, y2 = v[i - 1], v[i], v[i + 1]
    curv = y0 - 2 * y1 + y2
    if curv <= 0:
        return float(x[i]), float(y1)
    s = 0.5 * (y0 - y2) / curv
    return float(x[i] + s * (x[1] - x[0])), float(y1 - 0.25 * (y0 - y2) * s)


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
    frame: object = None                   # 3x3 lab -> pair frame; None = PAIR_FRAMES[axis]

    def b_local(self, b_lab_gauss):
        frame = PAIR_FRAMES[self.axis] if self.frame is None else np.asarray(self.frame, float)
        return frame @ np.asarray(b_lab_gauss, float)

    def pair(self):
        return LatticePair(spacing_m=self.spacing_m, rotation_sense=self.rotation_sense,
                           extinction_ratio=self.extinction_ratio, handedness=self.handedness,
                           static_amplitude=self.static_amplitude,
                           rotating_amplitude=self.rotating_amplitude)

    def model(self, b_lab_gauss):
        """The pair's StatePotential at the bias field."""
        return StatePotential(self.pair(), self.b_local(b_lab_gauss), trap_frequency_hz=self.trap_frequency_hz)


@dataclass
class SiteConfig:
    """The three pairs, the bias field and the temperature at one site.

    ``n_levels`` None (the default): chosen from the temperature and the
    lowest trap frequency (``n_levels_for``); an int fixes it."""
    axes: list
    b_lab_gauss: np.ndarray
    temperature_k: float = 1e-6            # after Raman sideband cooling; n_bar ~ 0.4 at 19 kHz
    n_levels: object = None
    grid_points: int = 1201
    grid_half_width_m: float = 1.5e-6


def site_n_levels(site: SiteConfig):
    """The level count a spectrum of the site uses (``SiteConfig.n_levels``)."""
    if site.n_levels is not None:
        return int(site.n_levels)
    return n_levels_for(site.temperature_k, [cfg.trap_frequency_hz for cfg in site.axes])


def _edge_fraction(psi_row):
    """|psi|^2 at the outermost points of its support / its peak."""
    nz = np.flatnonzero(psi_row)
    if nz.size == 0:
        return 0.0
    p = psi_row ** 2
    return float(max(p[nz[0]], p[nz[-1]]) / p.max())


def axis_levels(cfg: AxisConfig, b_lab_gauss, n_levels=4, grid_points=1201, half_width_m=1.5e-6,
                states=(UPPER, LOWER), site_offsets=None, window_m=None):
    """Vibrational energies (Hz, including the state's light shift) and
    wavefunctions of each state along one axis, on a common grid.

    ``window_m`` is the half-width of the eigenvalue window around each
    well. None (the default): LEVEL_WINDOW_M, enlarged by WINDOW_GROWTH
    until the highest level's |psi|^2 at the window edge is below
    WINDOW_EDGE_LIMIT of its peak (high levels of a low trap frequency
    otherwise feel the window as a hard wall), at most WINDOW_MAX_SPACINGS
    lattice spacings. Every state dict carries 'window_m' and
    'edge_fraction' (the largest such ratio over its wells).

    Each state's well is followed from theta = 0 (``track_wells``). The grid
    has the spacing of ``linspace(-half_width_m, half_width_m, grid_points)``
    and is that grid whenever every well's window fits in it (small theta);
    otherwise it is extended, on the same lattice of points, over all wells.
    ``site_offsets`` ({state: offsets k}; default: the last state on
    ``SITE_OFFSETS``) adds that state's wells k spacings from its own one,
    where their window overlaps another state's well window (elsewhere
    every overlap with them is exactly zero on the grid).

    Returns {state: dict(x, v_hz, energies_hz, psi, x0_m, x_min_m, v_min_hz,
    purity, sites)}; ``sites`` maps k to dict(psi, energies_hz, x0_m,
    x_min_m) of that well (k = 0 is the state's own well, the same arrays).
    """
    if window_m is not None:
        return _axis_levels(cfg, b_lab_gauss, n_levels, grid_points, half_width_m, states, site_offsets,
                            window_m)
    cap = WINDOW_MAX_SPACINGS * cfg.spacing_m
    window = min(LEVEL_WINDOW_M, cap)
    while True:
        out = _axis_levels(cfg, b_lab_gauss, n_levels, grid_points, half_width_m, states, site_offsets, window)
        if max(d["edge_fraction"] for d in out.values()) <= WINDOW_EDGE_LIMIT or window >= cap:
            return out
        window = min(cap, window * WINDOW_GROWTH)


def _axis_levels(cfg, b_lab_gauss, n_levels, grid_points, half_width_m, states, site_offsets, window_m):
    model = cfg.model(b_lab_gauss)
    a = cfg.spacing_m
    if site_offsets is None:
        site_offsets = {states[-1]: SITE_OFFSETS} if len(states) > 1 else {}
    pos = {s: float(p[0]) for s, p in track_wells(model, states, [cfg.theta]).items()}
    wells = [(s, 0, pos[s]) for s in states]
    for s, offsets in site_offsets.items():
        others = [pos[o] for o in states if o != s]
        for k in offsets:
            if k != 0 and any(abs(pos[s] + k * a - o) < 2 * window_m for o in others):
                wells.append((s, k, pos[s] + k * a))
    dx = 2 * half_width_m / (grid_points - 1)
    lo_m = min(c for *_, c in wells) - window_m - 2 * dx
    hi_m = max(c for *_, c in wells) + window_m + 2 * dx
    i_lo = min(0, int(np.floor((lo_m + half_width_m) / dx)))
    i_hi = max(grid_points - 1, int(np.ceil((hi_m + half_width_m) / dx)))
    if i_lo == 0 and i_hi == grid_points - 1:
        x = np.linspace(-half_width_m, half_width_m, grid_points)
    else:
        x = np.arange(i_lo, i_hi + 1) * dx - half_width_m
    potentials = state_potentials(model, states, x, cfg.theta)
    out = {}
    for s, k, c in wells:
        v = potentials[s]
        near = np.flatnonzero(np.abs(x - c) < a / 4)
        centre = int(near[np.argmin(v[near])])
        e, psi = vibrational_levels(x, v, n_levels, window_m, centre=centre)
        x_min, v_min = _parabola_min(x, v, centre)
        well = dict(psi=psi, energies_hz=e, x0_m=float(np.sum(x * psi[0] ** 2)), x_min_m=x_min)
        edge = _edge_fraction(psi[-1])
        if k == 0:
            out[s] = dict(x=x, v_hz=v, energies_hz=e, psi=psi, x0_m=well["x0_m"], x_min_m=x_min,
                          v_min_hz=v_min, purity=float(label_purity(model, s, [x_min], cfg.theta)[0]),
                          sites={0: well}, window_m=float(window_m), edge_fraction=edge)
        else:
            out[s]["sites"][k] = well
            out[s]["edge_fraction"] = max(out[s]["edge_fraction"], edge)
    return out


@functools.lru_cache(maxsize=64)
def _bias_levels(b_tuple):
    """eigenlevels at a bias field, cached (a grid evaluates one field hundreds of times).
    Callers only read the result."""
    return eigenlevels(np.array(b_tuple))


def _extinction_key(er):
    """Hashable identity of an extinction ratio: the number, or for a measured
    profile (``fields.extinction.ExtinctionProfile``) its angle and ER arrays;
    None for any other callable (not cached)."""
    if not callable(er):
        return er
    theta, values = getattr(er, "theta", None), getattr(er, "er", None)
    if isinstance(theta, np.ndarray) and isinstance(values, np.ndarray):
        return (type(er).__name__, theta.tobytes(), values.tobytes())
    return None


def _levels_key(cfg, b, site, states):
    """Hashable identity of one axis' level calculation (None: do not cache)."""
    er = _extinction_key(cfg.extinction_ratio)
    if er is None:
        return None
    frame = None if cfg.frame is None else tuple(np.asarray(cfg.frame, float).ravel())
    fields = tuple((k, er if k == "extinction_ratio" else v) for k, v in vars(cfg).items() if k != "frame")
    return (fields, frame, tuple(b), site_n_levels(site), site.grid_points, site.grid_half_width_m,
            tuple(states))


def _sum_rule(lv_upper, lv_lower, w, carrier_hz):
    """Centroid and rms width (Hz, from the carrier) of the whole 1D
    line-strength distribution, every final level and site included, from the
    sum rules sum_n' |<n'|n>|^2 (E_n - E_n')^j = <n|(V_upper - V_lower)^j|n>
    (the kinetic energies cancel); the thermal spread over n is included."""
    d = lv_upper["v_hz"] - lv_lower["v_hz"] - carrier_hz
    rho = lv_upper["psi"] ** 2
    first, second = rho @ d, rho @ d**2
    centroid = float(w @ first)
    return centroid, float(np.sqrt(max(float(w @ second) - centroid**2, 0.0)))


def spectrum(site: SiteConfig, states=(UPPER, LOWER), max_dn=2, cache=None):
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

    Neighbouring sites (any theta; module docstring). The final state may
    be in the lower state's well of the atom's own site (k = 0) or of a
    neighbouring one (k = +/-1, along the pair's fringe direction); those
    lines lie at the same frequencies. 'lines', 'carrier_weight' and each
    axis' 'diagonal' are summed over k; 'site_lines' lists (axis, k, dn,
    offset, weight) per site and 'carrier_weight_same_site' is the k = 0
    carrier. Per axis, 'franck_condon' is the k = 0 matrix,
    'franck_condon_sites' and 'diagonal_sites' map k to the matrix and the
    diagonal term, 'displacement_m' is x(lower, k = 0) - x(upper) and
    'nearest_displacement_m' that to the nearest lower-state well (site
    'nearest_site'); 'centroid_offset_hz' and 'rms_width_hz' locate the
    whole 1D line strength (all n', all sites; sum rules), and
    'strength_in_lines' is the fraction of it in the carrier and the listed
    lines (|dn| <= max_dn, n' < n_levels); 'dn_weights' maps dn to the
    axis' own 1D weight of all its dn lines (dn = 0: 'diagonal'), so
    dn_weights[1] / dn_weights[0] is that pair's sideband-to-carrier ratio
    even when another pair's carrier vanishes. When the wells are far apart
    (mid-range theta) the strength moves to large dn and the carrier weight
    vanishes; nothing here divides by it. 'purity_upper' / 'purity_lower'
    are ``label_purity`` at each trap minimum.

    Levels and temperature. ``site.n_levels`` None picks the level count
    from the temperature (``n_levels_for``; 'n_levels' in the result). The
    thermal weights use the computed (anharmonic) energies of the initial
    state, normalized over the kept levels; per axis 'dropped_population'
    is the estimated thermal population above the last kept level
    (``dropped_population``; above DROPPED_POPULATION only at the
    MAX_LEVELS cap or with a fixed n_levels), and the result's
    'dropped_population' is that of the 3D initial state, 1 - prod(1 - d).
    'window_m' / 'window_edge_fraction' give the eigen-window used and the
    highest level's |psi|^2 at its edge (``axis_levels``).

    ``cache`` (a dict, optional) keeps each axis' vibrational levels between
    calls, so a change of one pair recomputes only that pair (the
    interactive explorer passes one). A change of temperature recomputes
    nothing when the level count stays the same; with n_levels None it
    usually changes the count, and then every pair is recomputed.
    """
    upper, lower = states
    n_lv = site_n_levels(site)
    b = np.asarray(site.b_lab_gauss, float)
    levels = _bias_levels(tuple(float(v) for v in b))
    zeeman = levels[upper][0] - levels[lower][0]
    per_axis, carrier_shift, raw = {}, 0.0, []
    for cfg in site.axes:
        key = None if cache is None else _levels_key(cfg, b, site, states)
        lv = cache.get(key) if key is not None else None
        if lv is None:
            lv = axis_levels(cfg, b, n_lv, site.grid_points, site.grid_half_width_m, states)
            if key is not None:
                if len(cache) > 256:
                    cache.clear()
                cache[key] = lv
        e_u, e_l = lv[upper]["energies_hz"], lv[lower]["energies_hz"]
        fcs = {}
        for k, well in sorted(lv[lower]["sites"].items()):
            f = franck_condon(lv[upper]["psi"], well["psi"])                # (n_lower, n_upper)
            f[f < FC_NOISE_FLOOR] = 0.0
            fcs[k] = f
        fc_all = sum(fcs.values())
        f_u, f_l = float(e_u[1] - e_u[0]), float(e_l[1] - e_l[0])
        w = thermal_weights_from_energies(e_u, site.temperature_k)
        carrier = float(e_u[0] - e_l[0])
        carrier_shift += carrier
        diag = float(np.sum(w * np.diag(fc_all)))
        n_idx = np.arange(n_lv)
        listed = np.abs(n_idx[:, None] - n_idx[None, :]) <= max_dn       # [n', n]
        centroid, rms = _sum_rule(lv[upper], lv[lower], w, carrier)
        x_u, x_l = lv[upper]["x0_m"], lv[lower]["x0_m"]
        k_near = int(np.round((x_u - x_l) / cfg.spacing_m))
        per_axis[cfg.axis] = dict(
            f_upper_hz=f_u, f_lower_hz=f_l, x0_upper_m=x_u, x0_lower_m=x_l, displacement_m=x_l - x_u,
            franck_condon=fcs[0], thermal=w, diagonal=diag, levels=lv,
            franck_condon_sites=fcs, diagonal_sites={k: float(np.sum(w * np.diag(f))) for k, f in fcs.items()},
            x0_lower_sites_m={k: well["x0_m"] for k, well in lv[lower]["sites"].items()},
            nearest_site=k_near, nearest_displacement_m=x_l + k_near * cfg.spacing_m - x_u,
            centroid_offset_hz=centroid, rms_width_hz=rms,
            strength_in_lines=float(np.sum(w * np.sum(np.where(listed, fc_all, 0.0), axis=0))),
            dn_weights={dn: float(np.sum(w[max(0, -dn):n_lv - max(0, dn)]
                                         * np.diagonal(fc_all, offset=-dn)))
                        for dn in range(-max_dn, max_dn + 1)},
            purity_upper=lv[upper]["purity"], purity_lower=lv[lower]["purity"],
            dropped_population=dropped_population(e_u, site.temperature_k),
            window_m=max(lv[s].get("window_m", LEVEL_WINDOW_M) for s in states),
            window_edge_fraction=max(lv[s].get("edge_fraction", 0.0) for s in states))
        for n in range(n_lv):
            for n2 in range(n_lv):
                dn = n2 - n
                if dn == 0 or abs(dn) > max_dn:
                    continue
                offset = float((e_u[n] - e_u[0]) - (e_l[n2] - e_l[0]))
                raw.append((cfg.axis, dn, offset, float(w[n] * fc_all[n2, n]),
                            {k: float(w[n] * f[n2, n]) for k, f in fcs.items()}))
    # a sideband on one axis leaves the other axes in their diagonal (carrier)
    # terms, so every line carries the product of the other axes' diagonals
    diagonals = {ax: d["diagonal"] for ax, d in per_axis.items()}
    carrier_weight = float(np.prod(list(diagonals.values())))
    others = {ax: float(np.prod([d for a2, d in diagonals.items() if a2 != ax])) for ax in diagonals}
    lines = [(ax, dn, off, wt * others[ax]) for ax, dn, off, wt, _ in raw]
    site_lines = [(ax, k, dn, off, wk * others[ax]) for ax, dn, off, _, by_k in raw for k, wk in by_k.items()]
    return dict(zeeman_hz=float(zeeman), carrier_hz=float(zeeman + carrier_shift),
                carrier_shift_hz=float(carrier_shift), carrier_weight=carrier_weight,
                carrier_weight_same_site=float(np.prod([d["diagonal_sites"][0] for d in per_axis.values()])),
                strength_in_lines=carrier_weight + sum(l[3] for l in lines),
                axes=per_axis, lines=lines, site_lines=site_lines, n_levels=n_lv,
                dropped_population=float(1.0 - np.prod([1.0 - d["dropped_population"]
                                                         for d in per_axis.values()])))


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


# ------------------------------------------------------- energies versus angle
def zero_point_estimate(harmonic_hz, c3_hz_per_m3, c4_hz_per_m4):
    """Zero-point energy (Hz) of V = V_min + (1/2) kappa x^2 + c3 x^3 + c4 x^4
    to second order: f/2 + 3 c4 x_zp^4 - 11 c3^2 x_zp^6 / f, x_zp^2 = hbar / (2 m omega).
    For V0 sin^2(q x) this is f/2 - E_q/4 with E_q = hbar^2 q^2 / (2 m)."""
    x_zp2 = HBAR / (2 * MASS_CS * 2 * np.pi * harmonic_hz)
    return harmonic_hz / 2 + 3 * c4_hz_per_m4 * x_zp2**2 - 11 * c3_hz_per_m3**2 * x_zp2**3 / harmonic_hz


def _local_quartic(model, state, theta, x_guess, step=HARMONIC_FIT_STEP_M):
    """Minimum of the exact potential near x_guess and its Taylor coefficients
    there: (x_min, v_min, kappa, c3, c4), from a quartic through 5 points."""
    u = np.arange(-2.0, 3.0)                        # positions in units of step
    v = state_potentials(model, (state,), x_guess + u * step, theta)[state]
    p = np.polynomial.Polynomial(np.polynomial.polynomial.polyfit(u, v, 4))
    dp, ddp = p.deriv(), p.deriv(2)
    s = 0.0
    for _ in range(6):                              # Newton on V'(u) = 0
        if ddp(s) <= 0:
            break
        s -= dp(s) / ddp(s)
    q = p(np.polynomial.Polynomial([s, 1.0]))       # re-expanded about the minimum
    c = np.concatenate([q.coef, np.zeros(5)])[:5] / step ** np.arange(5)
    return float(x_guess + s * step), float(c[0]), float(2 * c[2]), float(c[3]), float(c[4])


def energies_vs_angle(cfg: AxisConfig, b_lab_gauss, thetas, states=(UPPER, LOWER), dx_m=2.5e-9,
                      half_window_m=ENERGY_WINDOW_M, coarse_step=ENERGY_COARSE_STEP, progress=None,
                      cancelled=None):
    """Light-shift energies of each state's well in one pair versus its
    polarization angle, the well followed continuously from theta = 0.

    All energies are E/h in Hz relative to the state's bias-field energy (the
    free atom at the bias field), so V_min is the light shift at the trap
    minimum. Per state: 'x_m' (trap minimum), 'v_min_hz', 'e0_hz' (ground
    level), 'e1_hz', 'zpe_hz' = e0 - v_min (zero-point energy), 'trap_hz'
    (e1 - e0), 'harmonic_hz' (curvature at the minimum), 'zpe_estimate_hz'
    (``zero_point_estimate``: harmonic + quartic + cubic, an independent
    check of 'zpe_hz') and 'purity' (``label_purity`` at the minimum).
    'carrier_hz' = e0(upper) - e0(lower), this pair's light-shift and
    zero-point part of the carrier, split into 'v_min_diff_hz' and
    'zpe_diff_hz'. The potential is diagonalized every ``coarse_step``-th
    point of a dx_m grid on +/- half_window_m around the well and splined
    in between; V_min and the curvature come from the exact potential.
    ``progress(i, n)`` and ``cancelled()`` serve a GUI worker (cancelled:
    returns None).
    """
    thetas = np.atleast_1d(np.asarray(thetas, float))
    model = cfg.model(b_lab_gauss)
    positions = track_wells(model, states, thetas)
    half = int(np.ceil(half_window_m / dx_m / coarse_step)) * coarse_step
    t_fine = np.arange(-half, half + 1) * dx_m
    t_coarse = t_fine[::coarse_step]
    keys = ("x_m", "v_min_hz", "e0_hz", "e1_hz", "zpe_hz", "trap_hz", "harmonic_hz", "zpe_estimate_hz",
            "purity")
    out = {s: {k: np.empty(thetas.size) for k in keys} for s in states}
    for i, th in enumerate(thetas):
        if cancelled is not None and cancelled():
            return None
        for s in states:
            xc = np.round(positions[s][i] / dx_m) * dx_m
            v_coarse = state_potentials(model, (s,), xc + t_coarse, th)[s]
            v = CubicSpline(t_coarse, v_coarse)(t_fine)
            e, _ = vibrational_levels(xc + t_fine, v, 2, half_window_m + dx_m)
            j = int(np.argmin(v))
            x_min, v_min, kappa, c3, c4 = _local_quartic(model, s, th, _parabola_min(xc + t_fine, v, j)[0])
            f_h = np.sqrt(max(kappa, 0.0) * H / MASS_CS) / (2 * np.pi)
            row = dict(x_m=x_min, v_min_hz=v_min, e0_hz=e[0], e1_hz=e[1], zpe_hz=e[0] - v_min,
                       trap_hz=e[1] - e[0], harmonic_hz=f_h,
                       zpe_estimate_hz=zero_point_estimate(f_h, c3, c4) if f_h > 0 else np.nan,
                       purity=label_purity(model, s, [x_min], th)[0])
            for k in keys:
                out[s][k][i] = row[k]
        if progress is not None:
            progress(i + 1, thetas.size)
    result = dict(theta=thetas, states=out, axis=cfg.axis)
    if len(states) == 2:
        up, lo = states
        result.update(carrier_hz=out[up]["e0_hz"] - out[lo]["e0_hz"],
                      v_min_diff_hz=out[up]["v_min_hz"] - out[lo]["v_min_hz"],
                      zpe_diff_hz=out[up]["zpe_hz"] - out[lo]["zpe_hz"])
    return result


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


def with_alignment(site: SiteConfig, axis, static_offset_m=None, rotating_offset_m=None,
                   crossing_half_angle_deg=None, waist_m=60e-6, **axis_fields):
    """Copy of ``site`` with one pair's centering, crossing angle and any
    AxisConfig field changed. The crossing angle changes the spacing at
    fixed intensity, so the trap frequency scales as 1 / spacing
    (``trap_frequency_hz`` refers to the depth at the nominal spacing).
    An arm offset left at None keeps that arm's amplitude, so calls can be
    chained (0 centres the arm)."""
    axes = []
    for cfg in site.axes:
        if cfg.axis == axis:
            changes = {}
            if static_offset_m is not None:
                changes["static_amplitude"] = amplitude_from_offset(static_offset_m, waist_m)
            if rotating_offset_m is not None:
                changes["rotating_amplitude"] = amplitude_from_offset(rotating_offset_m, waist_m)
            if crossing_half_angle_deg is not None:
                spacing = spacing_from_crossing(np.deg2rad(crossing_half_angle_deg))
                changes["spacing_m"] = spacing
                changes["trap_frequency_hz"] = cfg.trap_frequency_hz * cfg.spacing_m / spacing
            changes.update(axis_fields)
            cfg = replace(cfg, **changes)
        axes.append(cfg)
    return replace(site, axes=axes)
