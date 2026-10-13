"""Projection sideband cooling (PSC) of the 3D lattice.

One cooling step on axis a (lab sequence, 2026-10-13):

1. rotate the axis-a EO arm to theta_2 and drive the "-2" AFP, centred
   2 f_a above the reference carrier: |4,-4,n> -> |3,-3,n-2>;
2. ramp to theta_1 and drive the "-1" AFP, centred f_a above the carrier.
   It returns |3,-3,n-2> -> |4,-4,n-1> and, for atoms still in |4,-4,n>,
   drives |4,-4,n> -> |3,-3,n-1>;
3. ramp back to theta = 0 and optically pump: atoms in F=3 are returned to
   |4,-4> by ~2 photons, which heat all three axes.

A cycle is x, y, z; the lab runs 30 cycles with (theta_2, theta_1) ~ (12, 6)
deg and the "-1" sweep +-4 kHz, then 10 cycles with ~ (13, 12) deg and both
sweeps +-6 kHz. AFP pulses are 3 ms.

Model. Each axis is a 1D problem on the exact state-dependent potentials
(``spectra.sidebands.axis_levels``): the two vibrational ladders, their
Franck-Condon overlaps and the carrier shift with rotation. Each AFP is
propagated on the full ladder (``afp.afp_unitary``), so the off-resonant
carrier and other sidebands, the anharmonicity and the Franck-Condon nodes
are all included. Vibrational coherences are dropped between pulses (the
ramps between pulses take ms, and site-to-site trap-frequency spread
dephases them), so a step is a product of |U|^2 matrices; the ramps are
taken as adiabatic (labels preserved). Spectator axes are untouched by a
pulse (their differential trap frequency shifts the line by
(f_u - f_l) n ~ 0.1 kHz, ignored). Optical pumping happens at theta = 0 and
uses the recoil matrices of ``optical_pumping``.

The 3D population P[nx, ny, nz] is propagated exactly within this model
(``simulate_3d``); ``simulate_factorized`` treats the axes as independent
(mean-field spectator heating) and is what the interactive explorer runs.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import H, KB
from ..spectra.sidebands import AxisConfig, UPPER, LOWER, axis_levels
from . import afp
from .optical_pumping import (DARK_EXCURSION_PROB, photon_cycle_matrix,
                              photon_number_distribution)

AXES = ("x", "y", "z")
# Programmed "-1" offsets (lab, 2026-10-13); the "-2" offsets are 2x these.
# Used as the theta = 0 |4,-4> trap frequency (n = 0 -> 1) of each axis.
OFFSET_M1_HZ = {"x": 18.1e3, "y": 15.1e3, "z": 12.5e3}
B_GAUSS = 1.39304
B_ANGLE_DEG = 45.0
# Extinction ratio of the rotating arm at small angles: measured 1.5-3.5e3
# at rest on x (findings 2026-10-07); y and z assumed the same.
EXTINCTION_RATIO = 2e3
HANDEDNESS = -1
# Initial state after PGC: thermal at 5 uK (Wu p.26: ~5 uK, <n> ~ 6 at 15 kHz).
INITIAL_TEMPERATURE_K = 5e-6
N_LEVELS = 50
GRID_POINTS = 2401
GRID_HALF_WIDTH_M = 1.6e-6
CACHE = Path("data") / "psc_ladders.npz"


@dataclass
class Loop:
    cycles: int
    theta2_deg: dict        # per axis, the "-2" pulse angle
    theta1_deg: dict        # per axis, the "-1" pulse angle
    hsr2_hz: float = -6e3
    hsr1_hz: float = -4e3


def lab_sequence(theta2=(12.0, 13.0), theta1=(6.0, 12.0)):
    """The current lab sequence: 30 + 10 cycles (same angle on every axis)."""
    return [Loop(30, dict.fromkeys(AXES, theta2[0]), dict.fromkeys(AXES, theta1[0]), -6e3, -4e3),
            Loop(10, dict.fromkeys(AXES, theta2[1]), dict.fromkeys(AXES, theta1[1]), -6e3, -6e3)]


def b_lab():
    phi = np.deg2rad(B_ANGLE_DEG)
    return B_GAUSS * np.array([np.cos(phi), np.sin(phi), 0.0])


def axis_config(axis, theta_deg, trap_hz, er=EXTINCTION_RATIO, handedness=HANDEDNESS):
    return AxisConfig(axis, trap_hz, theta=np.deg2rad(theta_deg), extinction_ratio=er,
                      handedness=handedness)


def calibrated_trap_hz(axis, er=EXTINCTION_RATIO, handedness=HANDEDNESS):
    """Depth parameter for which the theta = 0 |4,-4> spacing (0 -> 1) equals
    the programmed -1 offset (one fixed-point iteration is enough: the ratio
    is depth-independent to < 1e-4)."""
    target = OFFSET_M1_HZ[axis]
    lv = axis_levels(axis_config(axis, 0.0, target, er, handedness), b_lab(), 2,
                     GRID_POINTS, GRID_HALF_WIDTH_M, states=(UPPER,))
    f = lv[UPPER]["energies_hz"][1] - lv[UPPER]["energies_hz"][0]
    return target * target / f


@dataclass
class Ladder:
    """Both vibrational ladders of one axis at one rotation."""
    axis: str
    theta_deg: float
    e_u: np.ndarray          # |4,-4,n> energies from its own n = 0 (Hz)
    e_l: np.ndarray          # |3,-3,m> energies from its own m = 0 (Hz)
    overlap: np.ndarray      # [m, n] = <3,-3,m | 4,-4,n>
    carrier_shift_hz: float  # carrier at theta minus carrier at theta = 0, this axis
    displacement_m: float


def compute_ladders(thetas_deg, n_levels=N_LEVELS, er=EXTINCTION_RATIO, handedness=HANDEDNESS):
    """Ladders on a grid of angles for the three axes, plus the theta = 0
    |4,-4> eigenfunctions used for the recoil matrices."""
    out = {}
    for axis in AXES:
        trap = calibrated_trap_hz(axis, er, handedness)
        ref = None
        for th in thetas_deg:
            lv = axis_levels(axis_config(axis, th, trap, er, handedness), b_lab(), n_levels,
                             GRID_POINTS, GRID_HALF_WIDTH_M)
            u, l = lv[UPPER], lv[LOWER]
            carrier = u["energies_hz"][0] - l["energies_hz"][0]
            if ref is None:
                if abs(th) > 1e-12:
                    raise ValueError("thetas_deg must start at 0")
                ref = carrier
                out[(axis, "x_grid")] = u["x"]
                out[(axis, "psi0")] = u["psi"]
            out[(axis, th)] = Ladder(axis, th, u["energies_hz"] - u["energies_hz"][0],
                                     l["energies_hz"] - l["energies_hz"][0],
                                     l["psi"] @ u["psi"].T, carrier - ref,
                                     l["x0_m"] - u["x0_m"])
    return out


def save_ladders(ladders, path=CACHE):
    arrays = {}
    for key, val in ladders.items():
        axis, tag = key
        if isinstance(val, Ladder):
            p = f"{axis}_{tag:.3f}"
            arrays[p + "_eu"], arrays[p + "_el"], arrays[p + "_ov"] = val.e_u, val.e_l, val.overlap
            arrays[p + "_meta"] = np.array([val.carrier_shift_hz, val.displacement_m])
        else:
            arrays[f"{axis}_{tag}"] = val
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_ladders(path=CACHE):
    z = np.load(path)
    out = {}
    for key in z.files:
        axis, rest = key.split("_", 1)
        if rest in ("x_grid", "psi0"):
            out[(axis, rest)] = z[key]
        elif rest.endswith("_eu"):
            th = float(rest[:-3])
            p = key[:-3]
            meta = z[p + "_meta"]
            out[(axis, th)] = Ladder(axis, th, z[p + "_eu"], z[p + "_el"], z[p + "_ov"],
                                     float(meta[0]), float(meta[1]))
    return out


def ladder_at(ladders, axis, theta_deg):
    key = (axis, float(round(theta_deg, 3)))
    if key not in ladders:
        raise KeyError(f"no ladder for {axis} at {theta_deg} deg; available: "
                       f"{sorted(k[1] for k in ladders if k[0] == axis and isinstance(k[1], float))}")
    return ladders[key]


# ---------------------------------------------------------------- one pulse, one step
def pulse_matrix(ladder: Ladder, kind, hsr_hz, omega0_hz=afp.OMEGA0_HZ, center_hz=None,
                 site_detuning_hz=0.0):
    """|U|^2 of one AFP on this ladder; kind is 1 or 2 (the "-1" or "-2"
    pulse), centred at kind x the programmed offset unless center_hz is given.
    site_detuning_hz shifts this site's lines (e.g. a field or light-shift
    offset)."""
    center = kind * OFFSET_M1_HZ[ladder.axis] if center_hz is None else center_hz
    u = afp.afp_unitary(ladder.e_u, ladder.e_l, ladder.overlap, center, hsr_hz, omega0_hz,
                        ladder.carrier_shift_hz + site_detuning_hz)
    return afp.transfer_probability(u)


def step_matrix(t2, t1):
    """Combine the two pulses (incoherently): returns (T_uu, T_lu), [m, n] =
    probability that |4,-4,n> ends in |4,-4,m> or |3,-3,m> before OP."""
    n = t2.shape[0] // 2
    p = t1 @ t2[:, :n]
    return p[:n], p[n:]


@dataclass
class OpticalPumping:
    """Photon-number distribution and per-photon recoil matrices per axis."""
    p_k: np.ndarray                  # P(k photons) from |3,-3>
    p_k_dark: np.ndarray             # photon distribution of a dark-state excursion
    cycle: dict                      # axis -> one-photon vibrational matrix [m, n]
    dark_prob: float = DARK_EXCURSION_PROB
    powers: dict = field(default_factory=dict)

    def matrix_power(self, axis, k):
        key = (axis, k)
        if key not in self.powers:
            self.powers[key] = np.linalg.matrix_power(self.cycle[axis], k)
        return self.powers[key]

    def mixed(self, axis, dist):
        """sum_k P(k) M_a^k."""
        return sum(p * self.matrix_power(axis, k) for k, p in enumerate(dist) if p > 1e-9)


def optical_pumping_model(ladders, dark_prob=DARK_EXCURSION_PROB):
    p_k, pi_frac = photon_number_distribution((3, -3))
    p_dark, _ = photon_number_distribution(dark_excursion=True)
    cycle = {}
    for axis in AXES:
        cycle[axis] = photon_cycle_matrix(ladders[(axis, "x_grid")], ladders[(axis, "psi0")], axis, pi_frac)
    return OpticalPumping(p_k, p_dark, cycle, dark_prob)


# ---------------------------------------------------------------- sequences
def thermal(n_levels, f_hz, temperature_k=INITIAL_TEMPERATURE_K):
    """Thermal populations of the lowest n_levels, normalized over the full
    (infinite) ladder: the tail above the truncation is counted as lost
    (never cooled), not redistributed."""
    r = np.exp(-H * f_hz / (KB * temperature_k))
    return (1 - r) * r ** np.arange(n_levels)


def initial_state(n_levels=N_LEVELS, temperature_k=INITIAL_TEMPERATURE_K):
    return {a: thermal(n_levels, OFFSET_M1_HZ[a], temperature_k) for a in AXES}


class StepLibrary:
    """Caches the pulse matrices of a sequence."""

    def __init__(self, ladders, omega0_hz=afp.OMEGA0_HZ, center_mode="programmed"):
        self.ladders, self.omega0, self.center_mode = ladders, omega0_hz, center_mode
        self.cache = {}

    def pulse(self, axis, kind, theta, hsr):
        key = (axis, kind, round(theta, 3), hsr)
        if key not in self.cache:
            lad = ladder_at(self.ladders, axis, theta)
            center = None
            if self.center_mode == "line":   # centre on the actual n=1 (or 2) -> 0 line
                center = (lad.e_u[kind] - lad.e_l[0]) + lad.carrier_shift_hz
            self.cache[key] = pulse_matrix(lad, kind, hsr, self.omega0, center)
        return self.cache[key]

    def step(self, axis, loop: Loop):
        t2 = self.pulse(axis, 2, loop.theta2_deg[axis], loop.hsr2_hz)
        t1 = self.pulse(axis, 1, loop.theta1_deg[axis], loop.hsr1_hz)
        return step_matrix(t2, t1)


def _apply(mat, arr, axis_index):
    return np.moveaxis(np.tensordot(mat, arr, axes=([1], [axis_index])), 0, axis_index)


def simulate_3d(sequence, lib: StepLibrary, op: OpticalPumping, p0=None, record_every_step=False):
    """Exact (within the model) 3D population evolution. Returns the history
    of the ground-state fractions after every cycle and the final P."""
    if p0 is None:
        p0 = initial_state(lib.ladders[("x", 0.0)].e_u.size)
        p = np.einsum("i,j,k->ijk", p0["x"], p0["y"], p0["z"])
    else:
        p = p0
    hist = [_summary(p)]
    kmax_l = np.nonzero(op.p_k > 1e-7)[0]
    for loop in sequence:
        for _ in range(loop.cycles):
            for ai, axis in enumerate(AXES):
                t_uu, t_lu = lib.step(axis, loop)
                stay = _apply(t_uu, p, ai)
                pumped = _apply(t_lu, p, ai)
                if op.dark_prob > 0:
                    leak = op.dark_prob * stay
                    stay = stay - leak
                else:
                    leak = None
                new = stay
                for dist, src in ((op.p_k, pumped), (op.p_k_dark, leak)):
                    if src is None:
                        continue
                    for k in np.nonzero(dist > 1e-7)[0]:
                        q = src
                        for bi, b in enumerate(AXES):
                            q = _apply(op.matrix_power(b, int(k)), q, bi)
                        new = new + dist[k] * q
                p = new
            hist.append(_summary(p))
    return hist, p


def _summary(p):
    px = p.sum(axis=(1, 2))
    py = p.sum(axis=(0, 2))
    pz = p.sum(axis=(0, 1))
    n = np.arange(p.shape[0])
    return dict(total=float(p.sum()), ground_3d=float(p[0, 0, 0]),
                ground=dict(x=float(px[0]), y=float(py[0]), z=float(pz[0])),
                nbar=dict(x=float(n @ px), y=float(n @ py), z=float(n @ pz)))


def simulate_factorized(sequence, lib: StepLibrary, op: OpticalPumping, p0=None, keep=False):
    """Axes treated as independent: an OP event on axis a heats the
    spectators b with the shared photon-number distribution, weighted by the
    fraction of atoms pumped. Returns history of per-axis distributions."""
    p = dict(initial_state(lib.ladders[("x", 0.0)].e_u.size) if p0 is None else p0)
    heat = {a: op.mixed(a, op.p_k) for a in AXES}
    heat_dark = {a: op.mixed(a, op.p_k_dark) for a in AXES}
    hist = [{a: p[a].copy() for a in AXES}]
    for loop in sequence:
        for _ in range(loop.cycles):
            for axis in AXES:
                t_uu, t_lu = lib.step(axis, loop)
                stay, pumped = t_uu @ p[axis], t_lu @ p[axis]
                frac = pumped.sum()
                leak = op.dark_prob * stay.sum()
                new = (1 - op.dark_prob) * stay + heat[axis] @ pumped + op.dark_prob * heat_dark[axis] @ stay
                for b in AXES:
                    if b == axis:
                        continue
                    p[b] = (1 - frac - leak) * p[b] + frac * heat[b] @ p[b] + leak * heat_dark[b] @ p[b]
                p[axis] = new
            hist.append({a: p[a].copy() for a in AXES})
    return hist


class TableLibrary(StepLibrary):
    """Step library backed by precomputed pulse matrices
    (scripts/psc_pulse_tables.py), keyed '{omega_kHz}_{axis}_{theta:.1f}_{kind}_{hsr}'."""

    def __init__(self, ladders, tables, omega0_khz=afp.OMEGA0_HZ / 1e3):
        super().__init__(ladders, omega0_khz * 1e3)
        self.tables, self.om = tables, omega0_khz

    def pulse(self, axis, kind, theta, hsr):
        key = (axis, kind, round(theta, 3), hsr)
        if key not in self.cache:
            name = f"{self.om:g}_{axis}_{theta:.1f}_{kind}_{hsr:g}"
            self.cache[key] = np.asarray(self.tables[name], float)
        return self.cache[key]


def ground_3d_factorized(hist):
    return float(np.prod([hist[-1][a][0] for a in AXES]))


def step_outcomes(lib, axis, loop: Loop):
    """Per initial level n of |4,-4>: probability of ending one level lower
    in |4,-4> without OP ('clean'), of being optically pumped ('pumped'), of
    being left in |4,-4,n> ('stuck'), and the mean change of n of the
    unpumped part."""
    t_uu, t_lu = lib.step(axis, loop)
    n = np.arange(t_uu.shape[0])
    clean = np.array([t_uu[k - 1, k] if k > 0 else 0.0 for k in n])
    stuck = np.diag(t_uu).copy()
    pumped = t_lu.sum(axis=0)
    return dict(clean=clean, stuck=stuck, pumped=pumped,
                pumped_from=t_lu, t_uu=t_uu)
