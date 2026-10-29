"""Configuration, execution and analysis of the molasses Monte Carlo.

Units at this interface: intensity per beam in mW/cm^2 (at the atoms),
detuning from F=4 -> F'=5 in MHz (negative = red), field in mG, lattice
trap frequencies in kHz, times in ms, temperatures in uK.

Temperature is the kinetic temperature per axis, T_i = 2 <p_i^2/2m> / k_B,
over the recorded window. The energy relaxation time tau_E is the 1/e time
of the autocorrelation of the per-axis lattice energy E_i = p_i^2/2m + V_i.

Hopping (``hop_rate``): crossings of a barrier are counted directly when
they occur. For the cold, deep lattice they are far too rare to observe, and
the rate is extrapolated with the 1D Arrhenius form Li (2009, Sec. 3.5.1)
fitted to the lab's hopping data, Gamma_h = Gamma_a erfc(sqrt(U/k_B T)),
with the attempt rate taken as the energy relaxation rate,
Gamma_a = 1/tau_E (ATTEMPT_RATE_MODEL). Li measured Gamma_a = 265 s^-1 at
T = 10 uK; the comparison is in docs/findings.md (2026-10-29).
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import erfc

from ..constants import HBAR, KB, MASS_CS
from . import beams as bm
from . import kernel as kn
from . import levels as lv

MU_B_HZ_PER_G = 1.399_624_493_61e6
LATTICE_SPACING_M = 4.8374e-6                      # as fields/lattice_pair.py
TRAP_FULL_KHZ = (19.0, 16.0, 12.5)                 # x, y, z at full depth (current)
# Excited-state (6P3/2) to ground (6S1/2) scalar polarizability ratio at 843.222 nm, estimated
# from Safronova-type reduced matrix elements (6S, 7S, 6D3/2, 6D5/2, 8S, 7D); the tensor part of
# 6P3/2 is neglected. Status: computed estimate (data request 42: the measured trap shift of the
# molasses resonance from the imaging scan plan, step 1).
EXCITED_TO_GROUND_LATTICE = -0.32
REPUMP_DIR = (INV := 1 / np.sqrt(2), -INV, 0.0)    # along the OP/repump beam (-45 deg in x-y)
ATTEMPT_RATE_MODEL = "1/tau_E"


@dataclass
class Config:
    intensity: float = 0.7              # mW/cm^2 per beam at the atoms
    detuning: float = -51.0             # MHz from F'=5
    b_mg: tuple = (0.0, 0.0, 0.0)       # residual field (lattice x, y, z), mG
    depth_scale: float = 1.0            # lattice depth / full depth
    trap_khz: tuple = TRAP_FULL_KHZ     # full-depth trap frequencies
    lattice: bool = True
    geometry: str = "lab"               # "lab", "orthogonal", "1d"
    azimuth_offset_deg: float = 0.0
    imbalance: tuple = None             # relative beam intensities (6)
    levels: tuple = (5, 4, 3)           # F' included
    transition: str = "cs"              # "cs" or "jhalf"
    beta: float = EXCITED_TO_GROUND_LATTICE
    doppler: bool = True
    n_atoms: int = 256
    t_total_ms: float = 4.0
    t_burn_ms: float = 1.5
    t_init_uk: float = 8.0
    dt_max_us: float = 0.2
    sample_us: float = 5.0
    seed: int = 1
    motion: tuple = (True, True, True)
    drag: bool = False                  # dragged-atom mode: r(t) = r0 + drag_v t, record momentum transfer
    drag_v: tuple = (0.0, 0.0, 0.0)     # m/s
    dt_phase: float = 0.5                  # step control: max |H| dt
    dt_jump: float = kn.DT_JUMP         # step control: max scattering probability per step

    def lattice_u(self):
        """Barrier heights U_i (J) and spacings a_i (m)."""
        f = np.asarray(self.trap_khz) * 1e3
        a = np.full(3, LATTICE_SPACING_M)
        u = 2 * MASS_CS * a ** 2 * f ** 2 * self.depth_scale
        if not self.lattice:
            u[:] = 0.0
            a[:] = 0.0
        return u, a

    def trap_hz(self):
        return np.asarray(self.trap_khz) * 1e3 * np.sqrt(self.depth_scale)


_TRANSITIONS = {}


def transition(cfg):
    key = (cfg.transition, tuple(cfg.levels))
    if key not in _TRANSITIONS:
        _TRANSITIONS[key] = lv.cs_d2_f4(cfg.levels) if cfg.transition == "cs" else lv.j_half_to_three_half()
    return _TRANSITIONS[key]


def geometry(cfg):
    if cfg.geometry == "lab":
        return bm.lab_brewster(cfg.azimuth_offset_deg, imbalance=cfg.imbalance)
    if cfg.geometry == "orthogonal":
        return bm.orthogonal_linperp(cfg.imbalance)
    if cfg.geometry == "1d":
        return bm.one_d_linperp()
    raise ValueError(cfg.geometry)


@dataclass
class Result:
    cfg: Config
    t_uk: np.ndarray                 # per-axis kinetic temperature (N atoms averaged), uK
    t_uk_err: np.ndarray
    scatter_rate: float              # photons / s per atom
    depump_frac: float               # fraction of scattering events that ended in F = 3
    hops_per_s: np.ndarray           # directly observed barrier crossings per atom per s, per axis
    tau_e_ms: np.ndarray             # energy relaxation time per axis
    light_shift_uk: float            # mean molasses light-shift energy <H_herm>, uK
    e_samples: np.ndarray = field(repr=False, default=None)  # (N, ns, 3) J
    p_transfer: np.ndarray = field(repr=False, default=None) # (N, 3) momentum transferred (drag mode)
    t_rec_s: np.ndarray = field(repr=False, default=None)
    per_atom_t_uk: np.ndarray = field(repr=False, default=None)

    @property
    def t_mean(self):
        return float(np.mean(self.t_uk))

    def barrier_uk(self):
        u, _ = self.cfg.lattice_u()
        return u / KB * 1e6

    def hop_rate(self):
        """Extrapolated (Arrhenius) hopping rate per axis, s^-1; see module docstring."""
        ub = self.barrier_uk()
        ga = 1.0 / (self.tau_e_ms * 1e-3)
        return ga * erfc(np.sqrt(ub / np.maximum(self.t_uk, 1e-9)))


def _initial(cfg, tr, n, rng):
    u, a = cfg.lattice_u()
    kt = KB * cfg.t_init_uk * 1e-6
    p0 = rng.normal(size=(n, 3)) * np.sqrt(MASS_CS * kt)
    r0 = np.zeros((n, 3))
    f = cfg.trap_hz()
    for i in range(3):
        if a[i] > 0:
            w = 2 * np.pi * f[i]
            r0[:, i] = np.clip(rng.normal(size=n) * np.sqrt(kt / (MASS_CS * w ** 2)), -0.4 * a[i], 0.4 * a[i])
        else:
            r0[:, i] = rng.uniform(0, 1e-6, size=n)
    if cfg.drag:
        p0[:] = MASS_CS * np.asarray(cfg.drag_v)
        r0[:] = rng.uniform(0, 1e-6, size=(n, 3))
    g = tr.dim
    psi = np.zeros((n, kn.GMAX), dtype=complex)
    mi = rng.integers(0, g, size=n)
    psi[np.arange(n), mi] = 1.0
    for i in range(3):
        if not cfg.motion[i] and not cfg.drag:
            p0[:, i] = 0.0
    return r0, p0, psi


def run(cfg):
    tr = transition(cfg)
    geo = geometry(cfg)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_atoms
    nb = len(geo["k_hat"])
    kmag = tr.k
    kv = geo["k_hat"] * kmag
    eps = geo["eps"]
    amp = np.sqrt(geo["rel_intensity"])
    phases = rng.uniform(0, 2 * np.pi, size=(n, nb))
    seeds = rng.integers(1, 2 ** 62, size=n, dtype=np.int64).astype(np.uint64)
    r0, p0, psi0 = _initial(cfg, tr, n, rng)
    fx, fy, fz = tr.spin()
    g = tr.dim
    mvals = np.zeros(kn.GMAX)
    mvals[:g] = np.arange(-tr.f_ground, tr.f_ground + 1)
    cp = np.zeros(kn.GMAX)
    cp[:g - 1] = [np.sqrt(tr.f_ground * (tr.f_ground + 1) - m * (m + 1)) for m in mvals[:g - 1]]
    nl = len(tr.levels)
    dmat = np.zeros((nl, 3, kn.EMAX, kn.GMAX))
    dims_e = np.zeros(nl, dtype=np.int64)
    for L, lev in enumerate(tr.levels):
        de = lev.d.shape[1]
        dmat[L, :, :de, :g] = lev.d
        dims_e[L] = de
    offs = np.array([lev.offset_rad_s for lev in tr.levels])
    ca = np.array([lev.coeffs[0] for lev in tr.levels])
    cb = np.array([lev.coeffs[1] for lev in tr.levels])
    cc = np.array([lev.coeffs[2] for lev in tr.levels])
    branch = np.array([lev.branch_to_g for lev in tr.levels])
    om2q = lv.rabi0_sq(tr, cfg.intensity) / 4.0
    delta = 2 * np.pi * cfg.detuning * 1e6
    wz = 2 * np.pi * tr.g_f * MU_B_HZ_PER_G * np.asarray(cfg.b_mg, dtype=float) * 1e-3
    uvec, avec = cfg.lattice_u()
    ns = int((cfg.t_total_ms - cfg.t_burn_ms) * 1e3 / cfg.sample_us) + 1
    out = kn.simulate(r0, p0, psi0, phases, seeds, kv, eps, amp, kmag, np.asarray(REPUMP_DIR),
                      mvals, cp, g, dmat, dims_e, offs, ca, cb, cc, branch,
                      om2q, delta, tr.gamma, HBAR, tr.mass, wz, uvec, avec, cfg.beta,
                      bool(cfg.doppler), np.asarray(cfg.motion, dtype=np.bool_),
                      cfg.t_total_ms * 1e-3, cfg.t_burn_ms * 1e-3, cfg.dt_max_us * 1e-6,
                      cfg.sample_us * 1e-6, ns, np.asarray(cfg.drag_v, dtype=float),
                      bool(cfg.drag), cfg.dt_phase, cfg.dt_jump)
    ke, trec, nsc, ndep, hops, esamp, lsh, ptr, nsteps, ptr2 = out
    tpa = 2 * ke / trec[:, None] / KB * 1e6                      # per-atom, per-axis uK
    tm = tpa.mean(axis=0)
    terr = tpa.std(axis=0) / np.sqrt(n)
    rate = nsc.sum() / trec.sum()
    dep = ndep.sum() / max(nsc.sum(), 1)
    hps = hops.sum(axis=0) / trec.sum()
    tau = np.array([energy_relaxation_ms(esamp[:, :, i], cfg.sample_us) for i in range(3)])
    res = Result(cfg, tm, terr, float(rate), float(dep), hps, tau,
                 float(lsh.sum() / max(ns * n, 1) / KB * 1e6), esamp, ptr, trec, tpa)
    res.steps_per_ms = float(nsteps.mean() / (cfg.t_total_ms))
    res.p_transfer_2 = ptr2
    res.scatter_rate_err = float(np.std(nsc / trec) / np.sqrt(n))   # atom-to-atom spread (local polarization)
    return res


def energy_relaxation_ms(e, sample_us):
    """1/e time of the normalised autocorrelation of E(t) (atoms x samples)."""
    e = e - e.mean(axis=1, keepdims=True)
    var = (e * e).mean()
    if var <= 0:
        return np.nan
    ns = e.shape[1]
    for lag in range(1, ns // 2):
        c = (e[:, :-lag] * e[:, lag:]).mean() / var
        if c < np.exp(-1):
            c_prev = (e[:, :-(lag - 1)] * e[:, lag - 1:]).mean() / var if lag > 1 else 1.0
            frac = (c_prev - np.exp(-1)) / (c_prev - c)
            return (lag - 1 + frac) * sample_us * 1e-3
    return ns // 2 * sample_us * 1e-3


# ------------------------------------------------------------------ analytic references
def free_space_t_salomon_uk(intensity, detuning_mhz, c=0.35):
    """Salomon et al. 1990: k_B T = C hbar Omega^2/|delta| (Omega per beam, stretched transition)."""
    om2 = lv.GAMMA_D2 ** 2 * intensity / (2 * lv.ISAT_CYCLING_MW_CM2)
    return c * HBAR * om2 / (2 * np.pi * abs(detuning_mhz) * 1e6) / KB * 1e6


def dct_linperp(intensity, detuning_mhz, gamma=lv.GAMMA_D2, wavelength=lv.LAMBDA_D2, isat=lv.ISAT_CYCLING_MW_CM2):
    """Dalibard & Cohen-Tannoudji (1989) 1D lin-perp-lin, J=1/2 -> 3/2, |delta| >> Gamma:
    friction alpha = 3 hbar k^2 |delta|/Gamma, dipole-force diffusion D = (3/8) V0^2 k^2/Gamma'
    (V0 = (2/3) hbar |delta| s0, Gamma' = (2/9) Gamma s0, s0 = Omega0^2/(2 delta^2), spatial average),
    k_B T = D/alpha = (3/8) V0."""
    k = 2 * np.pi / wavelength
    d = 2 * np.pi * detuning_mhz * 1e6
    om2 = gamma ** 2 * intensity / (2 * isat)
    s0 = om2 / (2 * d * d)
    v0 = (2 / 3) * HBAR * abs(d) * s0
    gp = (2 / 9) * gamma * s0
    return dict(alpha=3 * HBAR * k * k * abs(d) / gamma, diffusion=0.375 * v0 ** 2 * k * k / gp,
                kt=0.375 * v0, v0=v0, gamma_p=gp, s0=s0, scatter=(5 / 6) * gamma * s0)


MAX_SATURATION = 0.1     # validity of adiabatic elimination of the excited state (low saturation)


def total_saturation(intensity, detuning_mhz, n_beams=6):
    """Saturation parameter of all beams together on the cycling transition,
    s = n (I/I_sat) / (1 + 4 Delta^2/Gamma^2). The model eliminates the excited
    state adiabatically and is used only for s <= MAX_SATURATION."""
    g = lv.GAMMA_D2 / (2 * np.pi * 1e6)
    return n_beams * intensity / lv.ISAT_CYCLING_MW_CM2 / (1 + 4 * (detuning_mhz / g) ** 2)


def optical_pumping_rate(intensity, detuning_mhz):
    """Characteristic F=4 pumping rate of six beams, Gamma' ~ (2/9) Gamma s0 x 6 (per s)."""
    d = 2 * np.pi * detuning_mhz * 1e6
    om2 = lv.GAMMA_D2 ** 2 * intensity / (2 * lv.ISAT_CYCLING_MW_CM2)
    return (2 / 9) * lv.GAMMA_D2 * om2 / (2 * d * d) * 6


def tail_temperature_uk(e_samples_j, t_kin_uk, start=3.0):
    """Per-axis temperature of the energy tail: for a 1D thermal oscillator P(E) ~ exp(-E/kT), so
    kT = <E - E_c | E > E_c> for any threshold E_c; E_c = start x the kinetic temperature.
    Returns (T_tail per axis in uK, number of samples above threshold per axis)."""
    out, nabove = np.zeros(3), np.zeros(3, dtype=int)
    for i in range(3):
        e = e_samples_j[:, :, i].ravel() / KB * 1e6
        ec = start * t_kin_uk[i]
        sel = e[e > ec]
        nabove[i] = sel.size
        out[i] = (sel - ec).mean() if sel.size > 20 else np.nan
    return out, nabove


def summary(res):
    """JSON-able summary of a Result."""
    tt, na = tail_temperature_uk(res.e_samples, res.t_uk)
    d = dict(intensity=res.cfg.intensity, detuning=res.cfg.detuning, b_mg=list(res.cfg.b_mg),
             depth_scale=res.cfg.depth_scale, trap_khz=list(res.cfg.trap_khz), geometry=res.cfg.geometry,
             n_atoms=res.cfg.n_atoms, t_total_ms=res.cfg.t_total_ms, t_burn_ms=res.cfg.t_burn_ms,
             t_uk=res.t_uk.tolist(), t_uk_err=res.t_uk_err.tolist(), t_tail_uk=tt.tolist(),
             n_tail=na.tolist(), scatter_rate=res.scatter_rate, depump_frac=res.depump_frac,
             hops_obs_per_s=res.hops_per_s.tolist(), tau_e_ms=res.tau_e_ms.tolist(),
             light_shift_uk=res.light_shift_uk, barrier_uk=res.barrier_uk().tolist(),
             per_atom_t_sd_uk=res.per_atom_t_uk.std(axis=0).tolist(),
             saturation=float(total_saturation(res.cfg.intensity, res.cfg.detuning)))
    return d


def arrhenius_hop_rate(t_uk, tau_e_ms, barrier_uk):
    """Gamma_h = Gamma_a erfc(sqrt(U/kT)), Gamma_a = 1/tau_E (s^-1), elementwise."""
    t = np.maximum(np.asarray(t_uk, float), 1e-9)
    return 1.0 / (np.asarray(tau_e_ms, float) * 1e-3) * erfc(np.sqrt(np.asarray(barrier_uk, float) / t))


def doubling_field_mg(b_values_mg, t_values_uk):
    """Field at which T doubles, from a fit T = T0 (1 + (B/B2)^2) to the points with T < 4 T0."""
    b = np.abs(np.asarray(b_values_mg, float))
    t = np.asarray(t_values_uk, float)
    t0 = t[np.argmin(b)]
    sel = t < 4 * t0
    if sel.sum() < 3:
        sel = np.argsort(b)[:3]
    A = np.vstack([np.ones(np.count_nonzero(sel) if sel.dtype == bool else len(sel)), b[sel] ** 2]).T
    (c0, c2), *_ = np.linalg.lstsq(A, t[sel], rcond=None)
    if c2 <= 0:
        return np.inf, c0
    return float(np.sqrt(c0 / c2)), float(c0)
