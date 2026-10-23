"""A line of neutral atoms on neighbouring x sites through the spin-echoed
two-leg collisional gate (Du 2025, Sec. 3.4 and 4.2.2.3).

Sequence. pi/2 on every atom; leg 1: map the qubit states onto |3,+m> and
|3,-m>, rotate the EOM arm out to the hold angle, hold, rotate back, map
back; a global pi pulse; leg 2 = leg 1; a final pi/2 whose phase is scanned.
During a leg the |3,+m> component of every atom moves +x and the |3,-m>
component -x (the repo's sign convention), so the |3,+m> packet of atom j
meets the |3,-m> packet of atom j+1 between the two sites. Their density
overlap and contact energy are those of the Phase 4 pair model
(``collision.CollisionModel``): the chain is that pair repeated, and every
neighbour pair is identical in this homogeneous model. The |3,-m> packet of
atom j and the |3,+m> packet of atom j+1 move apart, and packets two sites
apart stay at least one spacing apart, so only that one configuration of
each neighbour pair collides.

Phase. The configuration (atom j in |3,+m>, atom j+1 in |3,-m>) picks up
phi = 2 pi integral U(t)/h dt per leg (U from ``collision.interaction_hz``
and the pair overlap along theta(t)), spin exchange neglected; the
7-state ``CollisionModel.evolve`` gives the same phase when the exchange
channels are far detuned (tested) and, for the first-principles models,
the leakage. Because the pi pulse swaps the qubit states between the legs,
leg 1 puts phi_1 on |q_j, q_j+1> = |A, B> and leg 2 puts phi_2 on |B, A>
(A the qubit state mapped onto |3,+m>): the pair gate is
diag(1, e^-i phi_1, e^-i phi_2, 1), i.e. sqrt(ZZ) for phi_1 = phi_2 = pi/2.

Ramsey signal. For an atom with n interacting neighbours the fringe of the
final pi/2 phase is the isolated atom's fringe with contrast
prod cos(Phi/2) and phase shift sum +-(phi_1 - phi_2)/2, Phi = phi_1 + phi_2
the total collisional phase of the pair; with identical legs that is
cos^n(Phi/2) (end atoms n = 1, interior atoms n = 2). ``chain_fringes``
computes the same signal from an explicit state-vector simulation of the
pulses and the leg phases (the second method; tested against
``cos_rule_fringes``).

Assumptions (named constants below): ideal instantaneous pulses and map
pulses; no field, intensity or pointing noise; identical legs, so the
light-shift phases cancel; ground-state Gaussian packets that follow the
trap minima adiabatically (no motional excitation); the transverse overlap
of two identical Gaussians at TRANSVERSE_HZ; a uniform scattering length
(Du's 2800 a0, uncertain) or the diagonal DIS estimate.
"""

import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from ..atom.ground_state import magic_field
from ..spin.pulses import rotation
from .collision import (MU, CollisionModel, collision_ramp, dis_channel_lengths, interaction_hz,
                        oscillator_width, scattering_operator, uniform_lengths)

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MEASURED_ER_CSV = DATA_DIR / "measured" / "133-ms-SE-offset-neg2V-fine-2_extinction.csv"
ROBUST_RAMPS = {3: DATA_DIR / "R3_5ms_robust.csv", 1: DATA_DIR / "R1_8ms_robust.csv"}  # Phase 4 choice

V_PI_V = 5400.0                       # EOM voltage at theta = pi (repo convention)
LATTICE_SPACING_M = 4.8374e-6
TRANSVERSE_HZ = (8.0e3, 6.25e3)       # y, z trap frequencies at quarter depth (assumed, as Phase 4)
PHASE4_TRAP_HZ = 10e3                 # x trap scale of Phase 4 (quarter depth)
DEFAULT_SCATTERING_A0 = 2800.0        # Du's assumption for F = 3 + F = 3; uncertain
TARGET_LEG_PHASE = np.pi / 2          # collisional phase per leg for sqrt(ZZ)
PHASE_DT_US = 0.5                     # integration step of the phase
MAX_HOLD_US = 50e3                    # cap of the solved hold time
QUBIT_STATES = ("|3,−1⟩", "|4,+1⟩")   # qubit basis at the magic field; index 0, 1
PLUS_MOVER_QUBIT = 1                  # assumed: |4,+1> is mapped onto |3,+m> (moves +x), |3,-1> onto |3,-m>
PHASE4_THETAS = np.unique(np.concatenate([np.linspace(0, 0.6 * np.pi, 40),
                                          np.linspace(0.6 * np.pi, np.pi, 121)]))


def voltage_to_theta(v):
    return np.pi * np.asarray(v, float) / V_PI_V


def theta_to_voltage(theta):
    return V_PI_V * np.asarray(theta, float) / np.pi


# ------------------------------------------------------------ pair models
def calibration_model(transverse_hz=TRANSVERSE_HZ, ref_dir=REFERENCE_DIR):
    """Pair model from Transport-Optimizer's ER316 calibration files.

    The |3,-3> track (calibration_F3) moves -x; the |4,-3> track
    (calibration_F4, the optimizer's stand-in partner) plays the +x mover,
    and the spacing is the sum of the two end displacements, exactly as in
    tools/overlap_vs_voltage.py. No lattice energies (the 7-state evolve is
    not meaningful for this model).
    """
    tracks = {}
    for name in ("F3", "F4"):
        with open(Path(ref_dir) / f"calibration_{name}_10kHz_ER316.csv", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        fr = np.array([float(r["polarization_rotation_over_pi"]) for r in rows])
        x = np.abs([float(r["minimum_position_continuous_um"]) for r in rows]) * 1e-6
        f = np.array([float(r["trap_frequency_curvature_kHz"]) for r in rows]) * 1e3
        tracks[name] = (np.pi * fr, x, f)
    th, x_minus, f_minus = tracks["F3"]
    th4, x_plus, f_plus = tracks["F4"]
    if not np.allclose(th, th4):
        raise ValueError("calibration files on different angle grids")
    spacing = float(x_minus[-1] + x_plus[-1])
    return CollisionModel(3, th, x_plus, x_minus, f_plus, f_minus, np.zeros((th.size, 7)), spacing,
                          tuple(transverse_hz), magic_field())


def _model_key(m, handedness, er, trap_hz, spacing_m, b_vector):
    """File key of a track cache. er None (the measured profile) also hashes the
    bytes of MEASURED_ER_CSV, so a replaced or edited CSV builds new tracks."""
    tag = "ERmeas" if er is None else f"ER{float(er):.4g}"
    base = f"m{m}_h{int(handedness):+d}_{tag}_{trap_hz:.0f}Hz"
    extra = f"{spacing_m:.6e}_{np.round(b_vector, 7).tolist()}"
    if er is None:
        extra += "_" + hashlib.sha1(Path(MEASURED_ER_CSV).read_bytes()).hexdigest()
    return base + "_" + hashlib.sha1(extra.encode()).hexdigest()[:8]


def phase4_b_vector():
    """1.39304 G at +45 deg in x-y, the Phase 4 field."""
    return magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])


def first_principles_model(m, handedness=-1, er=None, trap_hz=PHASE4_TRAP_HZ, spacing_m=LATTICE_SPACING_M,
                           b_vector=None, thetas=None, transverse_hz=TRANSVERSE_HZ, cache_dir=DATA_DIR):
    """``collision.build_collision_model`` for the |3,+m>/|3,-m> pair, cached.

    er None = the measured extinction profile used in Phase 4 (leg 1, away);
    a number = a constant ER. With the defaults this is exactly the Phase 4
    model (``scripts/p4_collision.py``). The packet tracks take ~30 s on the
    Phase 4 angle grid; they are stored in ``data/chain_tracks_<key>.npz``
    (git-ignored) when ``cache_dir`` is given and ``thetas`` is the default.
    """
    from ..fields.lattice_pair import LatticePair
    from .collision import build_collision_model
    b_vector = phase4_b_vector() if b_vector is None else np.asarray(b_vector, float)
    path = None
    if cache_dir is not None and thetas is None:
        path = Path(cache_dir) / f"chain_tracks_{_model_key(m, handedness, er, trap_hz, spacing_m, b_vector)}.npz"
        if path.exists():
            d = np.load(path)
            return CollisionModel(m, d["thetas"], d["x0_plus"], d["x0_minus"], d["f_plus"], d["f_minus"],
                                  d["lattice_hz"], float(d["spacing_m"]), tuple(transverse_hz),
                                  float(np.linalg.norm(b_vector)))
    if er is None:
        from ..fields.extinction import ExtinctionProfile, read_rampol_er
        er_value = ExtinctionProfile.from_rows(read_rampol_er(MEASURED_ER_CSV), 1, "away")
    else:
        er_value = float(er)
    pair = LatticePair(extinction_ratio=er_value, handedness=int(handedness), spacing_m=float(spacing_m))
    cm = build_collision_model(m, pair, b_vector, thetas=PHASE4_THETAS if thetas is None else thetas,
                               trap_hz=trap_hz, transverse_hz=tuple(transverse_hz))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, thetas=cm.thetas, x0_plus=cm.x0_plus, x0_minus=cm.x0_minus, f_plus=cm.f_plus,
                 f_minus=cm.f_minus, lattice_hz=cm.lattice_hz, spacing_m=cm.spacing_m)
    return cm


def has_lattice_energies(cm):
    return bool(np.any(cm.lattice_hz != 0))


def with_transverse(cm, transverse_hz):
    return replace(cm, transverse_hz=tuple(float(f) for f in transverse_hz))


def full_overlap(cm):
    """Density overlap (m^-3) at theta = pi, the normalization of the lab's table."""
    return float(cm.overlap(np.pi))


def effective_scattering_length_m(m, kind="uniform", a_a0=DEFAULT_SCATTERING_A0):
    """Diagonal scattering length of |L:+m, R:-m> (m): uniform a, or the DIS estimate."""
    lengths = uniform_lengths(a_a0) if kind == "uniform" else dis_channel_lengths()
    op = scattering_operator(lengths)
    i = int(np.flatnonzero(MU == m)[0])
    return float(np.real(op[i, i]))


def scattering_op(kind="uniform", a_a0=DEFAULT_SCATTERING_A0):
    return scattering_operator(uniform_lengths(a_a0) if kind == "uniform" else dis_channel_lengths())


# ------------------------------------------------------------ packets
class Tracks:
    """Centre displacement (m) and width (m) of the two packets vs theta."""

    def __init__(self, cm):
        self.cm = cm
        self._xp = CubicSpline(cm.thetas, cm.x0_plus)
        self._xm = CubicSpline(cm.thetas, cm.x0_minus)
        self._fp = CubicSpline(cm.thetas, cm.f_plus)
        self._fm = CubicSpline(cm.thetas, cm.f_minus)

    def __call__(self, theta):
        th = np.clip(np.asarray(theta, float), self.cm.thetas[0], self.cm.thetas[-1])
        return (self._xp(th), self._xm(th), oscillator_width(self._fp(th)), oscillator_width(self._fm(th)))


def site_positions(n_atoms, spacing_m):
    """Site centres (m), the chain centred on x = 0."""
    return (np.arange(n_atoms) - (n_atoms - 1) / 2) * spacing_m


def gaussian_density(x, centre, width):
    """Ground-state density |psi|^2 (1/m) of width s = sqrt(hbar / m omega)."""
    return np.exp(-(np.asarray(x) - centre) ** 2 / width**2) / np.sqrt(np.pi * width**2)


def neighbour_counts(n_atoms):
    return np.array([(j > 0) + (j < n_atoms - 1) for j in range(n_atoms)], int)


# ------------------------------------------------------------ timing
def robust_ramp(m):
    """The Phase 4 ramp of the pair (R3 5 ms for |3,+-3>, R1 8 ms for |3,+-1>)."""
    from ..motion.transport1d import load_waveform
    t, th = load_waveform(ROBUST_RAMPS[m])[:2]
    return t - t[0], th


def smooth_ramp(theta_hold, duration_us, n=401):
    """theta(t) = theta_hold sin^2(pi t / 2T): zero slope at both ends."""
    t = np.linspace(0.0, float(duration_us), n)
    return t, theta_hold * np.sin(np.pi * t / (2 * duration_us)) ** 2


def leg_trajectory(ramp_t, ramp_theta, theta_hold, hold_us):
    """One leg: the ramp up to theta_hold, the hold, the reversed ramp (``collision_ramp``)."""
    theta_hold = min(float(theta_hold), float(ramp_theta[-1]))
    return collision_ramp(ramp_t, ramp_theta, theta_hold, float(hold_us))


def interaction_along(cm, theta, a_m):
    """U / h (Hz) of the colliding configuration along theta."""
    return interaction_hz(a_m, cm.overlap(np.asarray(theta, float)))


def integrated_phase(cm, t_us, theta_t, a_m, dt_us=PHASE_DT_US):
    """(t grid in us, cumulative phase 2 pi int U/h dt in rad) along theta(t)."""
    span = float(t_us[-1] - t_us[0])
    n = max(2, int(np.ceil(span / dt_us - 1e-9)) + 1)
    t = np.linspace(t_us[0], t_us[-1], n)
    u = interaction_along(cm, np.interp(t, t_us, theta_t), a_m)
    return t, 2 * np.pi * cumulative_trapezoid(u, t * 1e-6, initial=0.0)


def leg_phase(cm, ramp_t, ramp_theta, theta_hold, hold_us, a_m, dt_us=PHASE_DT_US):
    t, th = leg_trajectory(ramp_t, ramp_theta, theta_hold, hold_us)
    return float(integrated_phase(cm, t, th, a_m, dt_us)[1][-1])


def hold_for_phase(cm, ramp_t, ramp_theta, theta_hold, a_m, target=TARGET_LEG_PHASE, max_us=MAX_HOLD_US):
    """Hold time (us) that brings the leg phase to target; (hold, ramp-only phase, capped)."""
    phi_ramp = leg_phase(cm, ramp_t, ramp_theta, theta_hold, 0.0, a_m)
    u = float(interaction_along(cm, min(theta_hold, ramp_theta[-1]), a_m))
    if phi_ramp >= target:
        return 0.0, phi_ramp, False
    if u <= 0:
        return max_us, phi_ramp, True
    hold = (target - phi_ramp) / (2 * np.pi * u) * 1e6
    return (min(hold, max_us), phi_ramp, hold > max_us)


# ------------------------------------------------------------ Ramsey signal
def _apply_all(psi, n, u):
    """Apply the 2x2 unitary u to every qubit of the n-qubit state psi."""
    psi = psi.reshape((2,) * n)
    for k in range(n):
        psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [k])), 0, k)
    return psi.reshape(-1)


def _bits(n):
    idx = np.arange(2**n)
    return (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1      # qubit 0 = most significant


def leg_phase_pattern(n, pair_phases):
    """Phase (rad) of every basis state in one leg: pair (j, j+1) collides
    when atom j is in the |3,+m> qubit state and atom j+1 in the other."""
    b = _bits(n)
    phases = np.zeros(2**n)
    for j, phi in enumerate(np.broadcast_to(pair_phases, (n - 1,)) if n > 1 else []):
        hit = (b[:, j] == PLUS_MOVER_QUBIT) & (b[:, j + 1] != PLUS_MOVER_QUBIT)
        phases[hit] += phi
    return phases


def chain_fringes(n_atoms, phi1, phi2, scan_phases):
    """State-vector simulation: P(|4,+1>) of every atom vs the final pi/2 phase.

    All atoms start in |3,-1> (index 0). Pulses: pi/2 about x, leg 1
    (amplitude e^-i phi on colliding configurations), pi about x, leg 2,
    pi/2 about the scanned phase. phi1, phi2: per pair (scalar or n-1).
    Returns an (n_atoms, n_scan) array.
    """
    n = int(n_atoms)
    psi = np.zeros(2**n, complex)
    psi[0] = 1.0
    psi = _apply_all(psi, n, rotation(np.pi / 2, 0.0))
    psi = psi * np.exp(-1j * leg_phase_pattern(n, phi1))
    psi = _apply_all(psi, n, rotation(np.pi, 0.0))
    psi = psi * np.exp(-1j * leg_phase_pattern(n, phi2))
    b = _bits(n)
    out = np.empty((n, len(scan_phases)))
    for i, ph in enumerate(scan_phases):
        p = np.abs(_apply_all(psi, n, rotation(np.pi / 2, ph))) ** 2
        out[:, i] = p @ (b == 1)
    return out


def coherence_factors(n_atoms, phi1, phi2):
    """Complex factor F_k = C_k e^{i delta_k} of every atom's coherence:
    a right neighbour contributes e^{-i(phi1-phi2)/2} cos((phi1+phi2)/2),
    a left neighbour e^{+i(phi1-phi2)/2} cos((phi1+phi2)/2)."""
    n = int(n_atoms)
    p1 = np.broadcast_to(np.asarray(phi1, float), (max(n - 1, 0),))
    p2 = np.broadcast_to(np.asarray(phi2, float), (max(n - 1, 0),))
    f = np.ones(n, complex)
    for j in range(n - 1):
        c = np.cos((p1[j] + p2[j]) / 2)
        d = (p1[j] - p2[j]) / 2
        f[j] *= c * np.exp(-1j * d)
        f[j + 1] *= c * np.exp(1j * d)
    return f


def cos_rule_fringes(n_atoms, phi1, phi2, scan_phases):
    """The analytic fringe P(|4,+1>) = (1 - Re[F_k e^{i phase}]) / 2, which for
    identical legs is (1 - cos^n(Phi/2) cos(phase)) / 2 (isolated atom: n = 0)."""
    f = coherence_factors(n_atoms, phi1, phi2)
    ph = np.asarray(scan_phases, float)
    return (1 - np.real(f[:, None] * np.exp(1j * ph[None, :]))) / 2


def neighbour_contrast(n_neighbours, total_phase):
    """cos^n(Phi/2): signed fringe contrast relative to an isolated atom."""
    return np.cos(np.asarray(total_phase, float) / 2) ** n_neighbours


# ------------------------------------------------------------ the whole sequence
@dataclass
class GateSettings:
    m: int = 3
    n_atoms: int = 5
    hold_v: float = 4150.0
    hold_us: float = 500.0
    lock_hold: bool = True            # solve hold_us for TARGET_LEG_PHASE
    ramp: str = "robust"              # "robust" (Phase 4 ramp, truncated) or "smooth"
    smooth_ramp_us: float = 1000.0
    gap_us: float = 200.0             # between the legs (pi pulse in the middle), and at both ends / 2
    scattering: str = "uniform"       # "uniform" or "DIS"
    a_a0: float = DEFAULT_SCATTERING_A0
    transverse_hz: tuple = TRANSVERSE_HZ
    n_frames: int = 900
    scan_points: int = 73


def scan_range_v(m):
    return (3600.0, V_PI_V) if m == 3 else (2900.0, V_PI_V)


def compute_sequence(cm, s, hold_scan=True):
    """Everything the chain window draws, for pair model cm and settings s."""
    cm = with_transverse(cm, s.transverse_hz)
    tracks = Tracks(cm)
    theta_hold = float(voltage_to_theta(s.hold_v))
    if s.ramp == "robust":
        ramp_t, ramp_th = robust_ramp(s.m)
    else:
        ramp_t, ramp_th = smooth_ramp(theta_hold, s.smooth_ramp_us)
    theta_hold = min(theta_hold, float(ramp_th[-1]))
    a_m = effective_scattering_length_m(s.m, s.scattering, s.a_a0)
    phi_ramp = leg_phase(cm, ramp_t, ramp_th, theta_hold, 0.0, a_m)
    hold_us = float(s.hold_us)
    capped = False
    if s.lock_hold:
        hold_us, phi_ramp, capped = hold_for_phase(cm, ramp_t, ramp_th, theta_hold, a_m)
    leg_t, leg_th = leg_trajectory(ramp_t, ramp_th, theta_hold, hold_us)
    ramp_us = float(leg_t[-1] - hold_us) / 2
    leg_us = float(leg_t[-1])
    g = float(s.gap_us)
    start1, start2 = g / 2, g / 2 + leg_us + g
    t_end = start2 + leg_us + g / 2
    pulses = [("π/2", 0.0), ("π", start1 + leg_us + g / 2), ("π/2", t_end)]
    maps = [start1, start1 + leg_us, start2, start2 + leg_us]
    segments = []
    for k, t0 in enumerate((start1, start2), 1):
        segments += [(f"leg {k}: ramp out", t0, t0 + ramp_us), (f"leg {k}: hold", t0 + ramp_us, t0 + ramp_us + hold_us),
                     (f"leg {k}: ramp back", t0 + ramp_us + hold_us, t0 + leg_us)]

    def theta_of(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t)
        for t0 in (start1, start2):
            inside = (t >= t0) & (t <= t0 + leg_us)
            out[inside] = np.interp(t[inside] - t0, leg_t, leg_th)
        return out

    n = max(2, int(np.ceil(t_end / PHASE_DT_US)) + 1)
    t = np.linspace(0.0, t_end, n)
    # leg breakpoints on the grid so the plateau ends are not smeared
    t = np.unique(np.concatenate([t, start1 + leg_t, start2 + leg_t]))
    theta = theta_of(t)
    u = interaction_along(cm, theta, a_m)
    phase = 2 * np.pi * cumulative_trapezoid(u, t * 1e-6, initial=0.0)
    phi1 = float(np.interp(start1 + leg_us, t, phase))
    phi2 = float(phase[-1] - phi1)
    o_full = full_overlap(cm)

    tf = np.linspace(0.0, t_end, int(s.n_frames))
    thf = theta_of(tf)
    xp, xm, sp, sm = tracks(thf)
    in_leg = np.zeros(tf.size, bool)
    for t0 in (start1, start2):
        in_leg |= (tf >= t0) & (tf <= t0 + leg_us)
    frames = dict(t_us=tf, theta=thf, x_plus=xp, x_minus=xm, s_plus=sp, s_minus=sm, in_leg=in_leg,
                  rel_overlap=cm.overlap(thf) / o_full, u_hz=interaction_along(cm, thf, a_m),
                  phase=np.interp(tf, t, phase), separation=cm.separation(thf))

    scan = np.linspace(0, 2 * np.pi, int(s.scan_points))
    fr_sim = chain_fringes(s.n_atoms, phi1, phi2, scan)
    fr_rule = cos_rule_fringes(s.n_atoms, phi1, phi2, scan)
    ref = {n_nb: (1 - neighbour_contrast(n_nb, phi1 + phi2) * np.cos(scan)) / 2 for n_nb in (0, 1, 2)}

    out = dict(settings=s, a_m=a_m, theta_hold=theta_hold, hold_us=hold_us, hold_capped=capped,
               ramp_us=ramp_us, leg_us=leg_us, t_end_us=t_end, pulses=pulses, maps=maps,
               segments=segments, t_us=t, theta=theta, u_hz=u, phase=phase, phi1=phi1, phi2=phi2,
               rel_overlap=cm.overlap(theta) / o_full,
               phi_ramp=phi_ramp, frames=frames, spacing_m=cm.spacing_m, o_full=o_full,
               # leg 2 is not simulated separately: the 7-state check (seven_state_check) runs on leg 1,
               # leg 2 is assumed identical (same ramp, hold and model); PLUS_MOVER_QUBIT only signs the shifts
               notes=("seven-state check on leg 1 only; leg 2 assumed identical",
                      f"PLUS_MOVER_QUBIT = {PLUS_MOVER_QUBIT} assumed: sets the sign of the fringe phase shifts only"),
               hold=dict(rel_overlap=float(cm.overlap(theta_hold) / o_full),
                         overlap_m3=float(cm.overlap(theta_hold)),
                         u_hz=float(interaction_along(cm, theta_hold, a_m)),
                         separation_m=float(cm.separation(theta_hold)),
                         widths_m=tuple(float(w) for w in tracks(theta_hold)[2:])),
               scan_phases=scan, fringes_sim=fr_sim, fringes_rule=fr_rule, fringes_ref=ref,
               neighbours=neighbour_counts(s.n_atoms), factors=coherence_factors(s.n_atoms, phi1, phi2),
               ramp_t=ramp_t, ramp_theta=ramp_th, leg_t=leg_t, leg_theta=leg_th)
    if hold_scan:
        out["hold_scan"] = hold_voltage_scan(cm, s, hold_us, a_m)
    return out


def hold_voltage_scan(cm, s, hold_us, a_m, n=73, dt_us=1.0):
    """Leg phase vs hold voltage at a fixed hold time (the gate's tuning knob).
    cm must already carry the transverse frequencies of s (``with_transverse``)."""
    lo, hi = scan_range_v(s.m)
    volts = np.linspace(lo, hi, n)
    if s.ramp == "robust":
        rt, rth = robust_ramp(s.m)
    phis = []
    for v in volts:
        th = float(voltage_to_theta(v))
        if s.ramp != "robust":
            rt, rth = smooth_ramp(th, s.smooth_ramp_us)
        phis.append(leg_phase(cm, rt, rth, th, hold_us, a_m, dt_us))
    phis = np.array(phis)
    return dict(volts=volts, phi_leg=phis, total=2 * phis, hold_us=hold_us)


def seven_state_check(cm, s, result):
    """Leg-1 phase and leakage from the 7-state ``CollisionModel.evolve``."""
    cm = with_transverse(cm, s.transverse_hz)
    op = scattering_op(s.scattering, s.a_a0)
    phi, pops = cm.evolve(result["leg_t"], result["leg_theta"], op, dt_us=1.0)
    return dict(phi=float(phi), leakage=float(1 - pops[cm.index]))
