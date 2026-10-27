"""Experiment sequences for the cluster state as Clifford circuits.

A sequence is a JSON-serialisable dict, shared with the browser explorer
(``tools/cluster_engine.js``)::

    {"shape": [nx, ny, nz], "occupancy": [0/1, ...], "steps": [step, ...]}

Steps ("sites" null = every atom present at that point):

* ``{"op": "pulse", "q": k, "phase_q": p, "sites": ...}`` - a resonant
  qubit pulse R_phi(k pi/2) about the equatorial axis phi = p pi/2 (global
  two-photon pulse, or addressed when sites are given). k, p integers.
* ``{"op": "rz", "q": k, "sites": [...]}`` - addressed phase shift Z(k pi/2)
  (the AC-Zeeman targeted phase of Wang et al., Science 2016).
* ``{"op": "collide", "axes": "xyz", "mode": "echo" | "single"}`` - the
  collisional gate along the listed axes. "echo": leg 1 on every listed
  axis, a global pi pulse about x, leg 2 (Du 2025, Sec. 4.2.2.3; one axis per
  step = Du's sequential plan, several axes in one step = interleaved legs
  sharing one pi pulse). Each leg puts the phase LEG_PHASE_ECHO on every
  bond. "single": one leg per axis with phase pi, no echo.
* ``{"op": "measure", "basis": "X"|"Y"|"Z" or "bases": {site: basis}, "sites": ...}``
  - state detection after a basis rotation (lossless detection, Wu 2019).
* ``{"op": "clear", "sites": [...]}`` - push out these atoms (they leave the
  array; their correlations are traced out).

Physics of one leg (``gates/chain.py``): the |3,+m> component of the atom at
r moves +e and meets the |3,-m> component of the atom at s = r + e, so the
configuration |q_r = PLUS_MOVER_QUBIT, q_s != PLUS_MOVER_QUBIT> gains
exp(-i phi). With the echo, the pi pulse swaps the configurations and the
bond gate is diag(1, e^-i phi, e^-i phi, 1) = exp(+i phi/2 Z Z) up to a
global phase, i.e. Stim's SQRT_ZZ_DAG for phi = pi/2, followed by X on every
atom (the pi pulse). A single leg with phi = pi is X_s CZ X_s (for
PLUS_MOVER_QUBIT = 1). Any other leg phase is not Clifford and is rejected;
phase errors enter through the noise model as twirled Pauli channels.

Qubit basis: |0> = |3,-1>, |1> = |4,+1> (``chain.QUBIT_STATES``); every atom
starts in |0> after optical pumping and the map to the qubit basis.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .lattice import AXES, Region, bonds

PLUS_MOVER_QUBIT = 1              # as gates/chain.py: |4,+1> -> |3,+m> moves +e (unconfirmed, data request 25)
LEG_PHASE_ECHO = np.pi / 2        # per leg, echoed gate (Du: total pi/2 per collision stage)
LEG_PHASE_SINGLE = np.pi          # single leg, Jaksch-type gate
ECHO_PI_PHASE_Q = 0               # the echo pi pulse is about x

_PULSE = {  # (phase_q, q) -> Stim gate, R_phi(q pi/2) up to a global phase
    (0, 1): "SQRT_X", (0, 2): "X", (0, 3): "SQRT_X_DAG",
    (1, 1): "SQRT_Y", (1, 2): "Y", (1, 3): "SQRT_Y_DAG",
    (2, 1): "SQRT_X_DAG", (2, 2): "X", (2, 3): "SQRT_X",
    (3, 1): "SQRT_Y_DAG", (3, 2): "Y", (3, 3): "SQRT_Y",
}
_RZ = {1: "S", 2: "Z", 3: "S_DAG"}
_BASIS = {"X": "MX", "Y": "MY", "Z": "M"}


# ------------------------------------------------------------ noise
@dataclass
class NoiseModel:
    """Pauli error rates per operation. Defaults: ideal. See ``noise_preset``."""
    prep: float = 0.0             # X error per atom after state preparation
    pulse: float = 0.0            # depolarizing per atom per global pulse
    addressed: float = 0.0        # depolarizing per target per addressed operation
    spectator: float = 0.0        # Z error per non-target atom per addressed operation
    echo_contrast: float = 1.0    # single-atom echo contrast C per collision step: Z error (1 - C) / 2
    phase_offset: float = 0.0     # rad, systematic error of the total pair phase Phi (ideal pi)
    phase_sigma: float = 0.0      # rad, shot-to-shot rms of Phi
    leak: float = 0.0             # two-qubit depolarizing per bond (spin-exchange leakage stand-in)
    readout: float = 0.0          # flip probability per detected atom

    def zz_probability(self):
        """Twirled ZZ error per bond for a pair-phase error d ~ N(offset, sigma):
        the bond gate gains exp(i d/4 ZZ), so p = <sin^2(d/4)>."""
        return 0.5 * (1.0 - np.exp(-self.phase_sigma**2 / 8.0) * np.cos(self.phase_offset / 2.0))

    def to_dict(self):
        return asdict(self)


# Values from docs/findings.md and the theses; each preset names its source. The
# echo contrast C is applied per collision step, so the "sequential" presets go
# with one axis per step (Du's plan) and the "interleaved" ones with one x-y-z step.
_COMMON = dict(prep=1e-3, pulse=1.7e-3, addressed=3.8e-3, spectator=2.1e-3, phase_sigma=0.1, readout=6e-4)
_SOURCES = ("Φ rms 0.1 rad ≈ 7–15 V of stop voltage (findings, Phase 4); single-qubit 1.7e-3 "
            "(Wu p. 14); addressed fidelity 0.9962, crosstalk 0.9979 (Kumar Ch. 3); detection "
            "0.9994 (Wu, Kumar); preparation 1e-3 assumed")
NOISE_PRESETS = {
    "ideal": (NoiseModel(), "no errors"),
    "|3,±3⟩ sequential": (NoiseModel(echo_contrast=0.95, leak=4e-4, **_COMMON),
                          "C = 0.95 per single-axis echo (findings 2026-10-07, x only, 150 ms); leakage "
                          "≤ 4e-4 (Phase 4 stop points, DIS); " + _SOURCES),
    "|3,±3⟩ interleaved x-y-z": (NoiseModel(echo_contrast=0.89, leak=4e-4, **_COMMON),
                                 "C = 0.89 for the interleaved x-y-z echo (findings 2026-10-07); "
                                 "leakage ≤ 4e-4; " + _SOURCES),
    "|3,±1⟩ sequential": (NoiseModel(echo_contrast=0.97, leak=1.7e-4, **_COMMON),
                          "C = 0.97 per single-axis echo (findings 2026-10-07); leakage 1.7e-4 "
                          "(Phase 4, DIS, φ = π); " + _SOURCES),
    "|3,±1⟩ interleaved x-y-z": (NoiseModel(echo_contrast=0.95, leak=1.7e-4, **_COMMON),
                                 "C = 0.95 for the interleaved x-y-z echo (findings 2026-10-07); "
                                 "leakage 1.7e-4; " + _SOURCES),
}


def noise_preset(name):
    return copy.deepcopy(NOISE_PRESETS[name][0])


# ------------------------------------------------------------ helpers
def region_of(spec):
    return Region(*[int(v) for v in spec["shape"]])


def _sites(step, occ):
    s = step.get("sites")
    present = [int(i) for i in np.flatnonzero(occ)]
    if s is None:
        return present
    return [int(i) for i in s if occ[int(i)]]


def is_unitary_step(step):
    return step["op"] in ("pulse", "rz", "collide")


def unitary_prefix(spec):
    """The sequence up to (not including) the first measure or clear step."""
    out = dict(spec)
    steps = []
    for st in spec["steps"]:
        if not is_unitary_step(st):
            break
        steps.append(st)
    out["steps"] = steps
    return out


def occupancy_after(spec, k=None):
    """Occupancy after the first k steps (all steps when k is None)."""
    occ = np.array(spec["occupancy"], bool)
    for st in spec["steps"][:k]:
        if st["op"] == "clear":
            occ[_sites(st, occ)] = False
    return occ


def collision_bonds(spec, step, occ):
    reg = region_of(spec)
    return {a: bonds(reg, occ, a) for a in step["axes"]}


# ------------------------------------------------------------ compile
def to_stim(spec, noise=None, final_mpp=None, measure_tail=True):
    """Stim circuit of the sequence. ``noise`` (NoiseModel) adds Pauli channels.
    ``final_mpp``: list of stim.PauliString measured noiselessly at the end (MPP).
    ``measure_tail=False`` stops before the first measure / clear step."""
    import stim
    n = region_of(spec).n
    occ = np.array(spec["occupancy"], bool)
    c = stim.Circuit()
    nz = noise or NoiseModel()
    present = list(np.flatnonzero(occ))
    c.append("R", range(n))
    if nz.prep and present:
        c.append("X_ERROR", present, nz.prep)
    for st in spec["steps"]:
        op = st["op"]
        if not measure_tail and not is_unitary_step(st):
            break
        if op == "pulse":
            q, p = int(st["q"]) % 4, int(st.get("phase_q", 0)) % 4
            tg = _sites(st, occ)
            if q and tg:
                c.append(_PULSE[(p, q)], tg)
                _single_noise(c, nz, st, tg, occ)
        elif op == "rz":
            q = int(st["q"]) % 4
            tg = _sites(st, occ)
            if q and tg:
                c.append(_RZ[q], tg)
                _single_noise(c, nz, st, tg, occ, force_addressed=True)
        elif op == "collide":
            _collide(c, spec, st, occ, nz)
        elif op == "measure":
            tg = _sites(st, occ)
            bases = st.get("bases")
            for i in tg:
                b = bases.get(str(i), bases.get(i, "Z")) if bases else st.get("basis", "Z")
                c.append(_BASIS[b], [i], nz.readout) if nz.readout else c.append(_BASIS[b], [i])
        elif op == "clear":
            tg = _sites(st, occ)
            if tg:
                c.append("R", tg)          # traced out: reset discards the correlations
                occ[tg] = False
        else:
            raise ValueError(f"unknown step {op!r}")
    if final_mpp:
        for ps in final_mpp:
            c.append("MPP", stim.target_combined_paulis(ps))
    return c


def _single_noise(c, nz, st, tg, occ, force_addressed=False):
    addressed = force_addressed or st.get("sites") is not None
    if addressed:
        if nz.addressed:
            c.append("DEPOLARIZE1", tg, nz.addressed)
        others = sorted(set(np.flatnonzero(occ)) - set(tg))
        if nz.spectator and others:
            c.append("Z_ERROR", others, nz.spectator)
    elif nz.pulse:
        c.append("DEPOLARIZE1", tg, nz.pulse)


def _collide(c, spec, st, occ, nz):
    mode = st.get("mode", "echo")
    by_axis = collision_bonds(spec, st, occ)
    all_bonds = [b for a in st["axes"] for b in by_axis[a]]
    present = list(np.flatnonzero(occ))
    if mode == "echo":
        flat = [q for b in all_bonds for q in b]
        if flat:
            c.append("SQRT_ZZ_DAG", flat)          # exp(+i pi/4 ZZ): legs 1 and 2, pi pulse between
        if present:
            c.append(_PULSE[(ECHO_PI_PHASE_Q, 2)], present)
            if nz.pulse:
                c.append("DEPOLARIZE1", present, nz.pulse)
        p_zz = nz.zz_probability()
        if p_zz and flat:
            for r, s in all_bonds:
                c.append("CORRELATED_ERROR", [stim_target_z(r), stim_target_z(s)], p_zz)
    elif mode == "single":
        for r, s in all_bonds:
            flip = s if PLUS_MOVER_QUBIT == 1 else r
            c.append("X", [flip])
            c.append("CZ", [r, s])
            c.append("X", [flip])
        p = nz.zz_probability()       # leg-phase error enters as Z_r, Z_s and Z_r Z_s rotations by d/4
        if p and all_bonds:
            for r, s in all_bonds:
                c.append("Z_ERROR", [r, s], p)
                c.append("CORRELATED_ERROR", [stim_target_z(r), stim_target_z(s)], p)
    else:
        raise ValueError(f"unknown collision mode {mode!r}")
    if nz.leak and all_bonds:
        c.append("DEPOLARIZE2", [q for b in all_bonds for q in b], nz.leak)
    if nz.echo_contrast < 1.0 and present:
        c.append("Z_ERROR", present, (1.0 - nz.echo_contrast) / 2.0)


def stim_target_z(q):
    import stim
    return stim.target_z(int(q))


# ------------------------------------------------------------ standard sequences
def cluster_sequence(occupancy, shape, axes="xyz", interleaved=False, mode="echo"):
    """Du's plan: global pi/2 about x (the sqrt(X) that makes -y), then the gate
    along each axis in turn (or all axes interleaved in one echo)."""
    steps = [{"op": "pulse", "q": 1, "phase_q": 0, "sites": None}]
    groups = [axes] if interleaved else list(axes)
    steps += [{"op": "collide", "axes": g, "mode": mode} for g in groups]
    return {"shape": list(shape), "occupancy": [int(v) for v in occupancy], "steps": steps}


def check_axes(axes):
    if not axes or any(a not in AXES for a in axes) or len(set(axes)) != len(axes):
        raise ValueError(f"axes must be a non-empty subset of 'xyz', got {axes!r}")
