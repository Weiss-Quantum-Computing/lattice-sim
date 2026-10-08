"""Controlled collisions of |3,+m> and |3,-m> atoms on neighbouring sites.

Geometry. The left atom (site 0) in |3,+m> moves +x, the right atom (site a)
in |3,-m> moves -x; their separation is d(theta) = a - |x0(+m)| - |x0(-m)|.
With harmonic ground states (widths s = sqrt(hbar / m omega)) the density
overlap is

    O = O_x O_y O_z,   O_x = exp(-d^2 / (s1^2 + s2^2)) / sqrt(pi (s1^2 + s2^2)),
    O_y = 1 / (sqrt(2 pi) s_y)   (same transverse width for both atoms),

and a contact interaction of scattering length a gives U/h = 2 hbar a O / m.

Spin structure. Two f = 3 atoms in distinct orbitals (L, R) with total
M = 0 span |L: mu, R: -mu>, mu = -3..3. A contact interaction acts on the
symmetrized spin state s_mu = (|mu,-mu> + |-mu,mu>)/sqrt(2), so

    <L mu, R -mu| V |L mu', R -mu'> = (2 hbar / m) O <s_mu| A |s_mu'>,

with A the scattering-length operator. A is either Sum_F a_F P_F over the
total-F channels (F = 0, 2, 4, 6), or the degenerate-internal-states (DIS)
estimate a_S P_singlet + a_T P_triplet restricted to f = 3 + f = 3, with
the Cs singlet and triplet scattering lengths of Chin et al., PRA 70, 032701
(2004). DIS ignores Feshbach resonances and is an order-of-magnitude
estimate only; the real channel scattering lengths near 1.39 G are an open
input (docs/data_requests.md).

Pair energies. Each basis state carries the Zeeman energy of its two atoms
(Breit-Rabi at the bias field) and the lattice energy of the two atoms at the
current wave-packet positions (sudden approximation: an atom converted to
another m_F stays where its wave packet is on the collision time scale).
The model is valid while the two orbitals are distinct, i.e. below full
overlap; near theta = pi the L/R basis becomes overcomplete.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

from ..atom.ground_state import _operators, breit_rabi, magic_field
from ..constants import A0, H, HBAR, MASS_CS

A_SINGLET_A0 = 280.37   # Chin et al. 2004, Table IV
A_TRIPLET_A0 = 2440.0
MU = np.arange(-3, 4)   # basis order |L: mu, R: -mu>


# ------------------------------------------------------------ overlap
def oscillator_width(f_hz):
    return np.sqrt(HBAR / (MASS_CS * 2 * np.pi * np.asarray(f_hz, float)))


def overlap_density(d_m, f1_hz, f2_hz, fy_hz, fz_hz):
    """Two-atom density overlap integral (m^-3) for harmonic ground states."""
    s2 = oscillator_width(f1_hz) ** 2 + oscillator_width(f2_hz) ** 2
    ox = np.exp(-np.asarray(d_m) ** 2 / s2) / np.sqrt(np.pi * s2)
    oy = 1 / (np.sqrt(2 * np.pi) * oscillator_width(fy_hz))
    oz = 1 / (np.sqrt(2 * np.pi) * oscillator_width(fz_hz))
    return ox * oy * oz


def interaction_hz(a_m, overlap_m3):
    """U / h for scattering length a (m) and density overlap (m^-3)."""
    return 2 * HBAR * a_m * overlap_m3 / MASS_CS


# ------------------------------------------------------------ spin algebra
@lru_cache(maxsize=None)
def _f3_states():
    """|3, m> (Condon-Shortley phases) in the 16-dim single-atom space."""
    J, I, F = _operators()
    f2 = sum(op @ op for op in F)
    fz = F[2]
    w, v = np.linalg.eigh(f2)
    sub = v[:, np.isclose(w, 12.0)]
    fz_sub = sub.conj().T @ fz @ sub
    wz, vz = np.linalg.eigh(fz_sub)
    top = sub @ vz[:, np.argmax(wz)]
    f_minus = F[0] - 1j * F[1]
    states = {3: top / np.linalg.norm(top)}
    for m in range(3, -3, -1):
        nxt = f_minus @ states[m]
        states[m - 1] = nxt / np.linalg.norm(nxt)
    return states


@lru_cache(maxsize=None)
def _symmetric_states():
    """s_mu = (|mu,-mu> + |-mu,mu>)/sqrt(2) in the 256-dim two-atom space."""
    st = _f3_states()
    return np.array([(np.kron(st[m], st[-m]) + np.kron(st[-m], st[m])) / np.sqrt(2) for m in MU])


@lru_cache(maxsize=None)
def _two_atom_operators():
    J, _, F = _operators()
    eye = np.eye(16)
    s_tot = [np.kron(j, eye) + np.kron(eye, j) for j in J]
    f_tot = [np.kron(f, eye) + np.kron(eye, f) for f in F]
    s2 = sum(op @ op for op in s_tot)
    f2 = sum(op @ op for op in f_tot)
    return s2, f2


def channel_projectors():
    """{F: matrix <s_mu|P_F|s_mu'>} on the 7-state basis, F = 0, 2, 4, 6."""
    s = _symmetric_states()
    _, f2 = _two_atom_operators()
    # orthonormal basis of the symmetric M = 0 f3+f3 subspace (4 states)
    u = np.array([s[MU == m][0] for m in (0, 1, 2, 3)])
    u[0] = u[0] / np.linalg.norm(u[0])
    gram = u.conj() @ f2 @ u.T
    w, v = np.linalg.eigh(gram)
    proj = {}
    for fval in (0, 2, 4, 6):
        cols = v[:, np.isclose(w, fval * (fval + 1))]
        vec = cols.T @ u                      # |F, 0> in the 256-dim space
        proj[fval] = np.real_if_close(np.array(
            [[(s[i].conj() @ vec.conj().T) @ (vec @ s[j]) for j in range(7)] for i in range(7)]))
    return proj


def dis_channel_lengths(a_s=A_SINGLET_A0, a_t=A_TRIPLET_A0):
    """a_F (in a0) for f = 3 + f = 3 from the DIS approximation."""
    s2, f2 = _two_atom_operators()
    s = _symmetric_states()
    u = np.array([s[MU == m][0] for m in (0, 1, 2, 3)])
    u[0] = u[0] / np.linalg.norm(u[0])
    w, v = np.linalg.eigh(u.conj() @ f2 @ u.T)
    p_triplet = s2 / 2.0
    out = {}
    for fval in (0, 2, 4, 6):
        vec = (v[:, np.isclose(w, fval * (fval + 1))].T @ u)[0]
        pt = float(np.real(vec.conj() @ p_triplet @ vec))
        out[fval] = (1 - pt) * a_s + pt * a_t
    return out


def scattering_operator(a_f_a0):
    """7x7 matrix <s_mu|A|s_mu'> in metres for channel lengths {F: a_F (a0)}."""
    proj = channel_projectors()
    return sum(a_f_a0[f] * proj[f] for f in (0, 2, 4, 6)) * A0


def uniform_lengths(a_a0=2800.0):
    return {f: a_a0 for f in (0, 2, 4, 6)}


# ------------------------------------------------------------ collision model
@dataclass
class CollisionModel:
    """Two-atom collision along one axis, as a function of theta.

    thetas: grid; x0_plus/x0_minus: |displacement| (m) of the +m and -m
    packets; f_plus/f_minus: their trap frequencies (Hz); lattice_hz:
    (n_theta, 7) lattice energy of each basis state at the packet positions
    (relative to the initial state); spacing_m, transverse (f_y, f_z) in Hz.
    """
    m: int
    thetas: np.ndarray
    x0_plus: np.ndarray
    x0_minus: np.ndarray
    f_plus: np.ndarray
    f_minus: np.ndarray
    lattice_hz: np.ndarray
    spacing_m: float = 4.8374e-6
    transverse_hz: tuple = (8.0e3, 6.25e3)
    b_gauss: float = field(default_factory=magic_field)

    def __post_init__(self):
        self._lat = CubicSpline(self.thetas, self.lattice_hz, axis=0)
        self._sep = CubicSpline(self.thetas, self.spacing_m - self.x0_plus - self.x0_minus)
        self._f1 = CubicSpline(self.thetas, self.f_plus)
        self._f2 = CubicSpline(self.thetas, self.f_minus)
        zee = np.array([breit_rabi(3, mu, self.b_gauss) + breit_rabi(3, -mu, self.b_gauss) for mu in MU])
        self.zeeman_hz = zee - zee[MU == self.m][0]
        self.index = int(np.flatnonzero(MU == self.m)[0])

    def separation(self, theta):
        return self._sep(theta)

    def overlap(self, theta):
        return overlap_density(self.separation(theta), self._f1(theta), self._f2(theta),
                               *self.transverse_hz)

    def hamiltonian_hz(self, theta, a_op_m):
        diag = self.zeeman_hz + self._lat(theta)
        return np.diag(diag) + 2 * HBAR / MASS_CS * self.overlap(theta) * a_op_m

    def evolve(self, t_us, theta_t, a_op_m, dt_us=0.5):
        """Propagate the initial |L:+m, R:-m> through theta(t).

        Returns (collisional phase in rad, final populations of the 7 states).
        The phase is that of the initial-state amplitude relative to the
        evolution without interaction, unwrapped step by step (it can exceed pi).
        """
        # Uniform steps covering the whole span: the nominal dt_us is shrunk so
        # that no fractional step at the end is dropped.
        span = float(t_us[-1] - t_us[0])
        n_steps = max(1, int(np.ceil(span / dt_us - 1e-9)))
        dt_us = span / n_steps
        t = t_us[0] + dt_us * (np.arange(n_steps) + 0.5)
        th = np.interp(t, t_us, theta_t)
        psi = np.zeros(7, complex)
        psi[self.index] = 1.0
        phase0, phases = 0.0, []
        for theta in th:
            hmat = self.hamiltonian_hz(theta, a_op_m)
            psi = expm(-2j * np.pi * hmat * dt_us * 1e-6) @ psi
            phase0 += 2 * np.pi * (self.zeeman_hz[self.index] + self._lat(theta)[self.index]) * dt_us * 1e-6
            phases.append(-np.angle(psi[self.index] * np.exp(1j * phase0)))
        phi = float(np.unwrap(np.array(phases))[-1]) if phases else 0.0
        return phi, np.abs(psi) ** 2

    def ramp_phase_curve(self, t_us, theta, theta_holds, a_op_m, dt_us=1.0):
        """Collisional phase and leakage for ramps stopped at each theta_hold (no hold)."""
        out = []
        for th_h in theta_holds:
            tr, thr = collision_ramp(t_us, theta, th_h, 0.0)
            phi, pops = self.evolve(tr, thr, a_op_m, dt_us)
            out.append((phi, 1 - pops[self.index]))
        return np.array(out)


def collision_ramp(t_us, theta, theta_hold, hold_us):
    """theta(t): the ramp up to theta_hold, a hold, and the time-reversed ramp."""
    stop = int(np.searchsorted(theta, theta_hold))
    tu, thu = t_us[: stop + 1] - t_us[0], np.minimum(theta[: stop + 1], theta_hold)
    tr = tu[-1]
    t = np.concatenate([tu, tr + hold_us + (tr - tu[::-1])[1:]])
    th = np.concatenate([thu, thu[::-1][1:]])
    if hold_us > 0:
        # end of the hold; with no hold this point would duplicate t = tr
        t = np.concatenate([t[: stop + 1], [tr + hold_us], t[stop + 1:]])
        th = np.concatenate([th[: stop + 1], [theta_hold], th[stop + 1:]])
    return t, th


def build_collision_model(m, lattice_pair, b_vector_gauss, thetas=None, trap_hz=10e3,
                          transverse_hz=(8.0e3, 6.25e3)):
    """CollisionModel for the |3,+m>/|3,-m> pair from the first-principles potentials."""
    from ..fields.potential import StatePotential
    if thetas is None:
        thetas = np.unique(np.concatenate([np.linspace(0, 0.6 * np.pi, 40),
                                           np.linspace(0.6 * np.pi, np.pi, 121)]))
    model = StatePotential(lattice_pair, b_vector_gauss, trap_frequency_hz=trap_hz)
    a = lattice_pair.spacing_m
    xp, fp = model.track((3, m), thetas)
    xm, fm = model.track((3, -m), thetas)
    x_left, x_right = np.abs(xp), a - np.abs(xm)
    lat = np.zeros((thetas.size, 7))
    for k, th in enumerate(thetas):
        for j, mu in enumerate(MU):
            lat[k, j] = (model.potential((3, mu), np.array([x_left[k]]), th)[0]
                         + model.potential((3, -mu), np.array([x_right[k]]), th)[0])
    lat -= lat[:, list(MU).index(m)][:, None]
    return CollisionModel(m, thetas, np.abs(xp), np.abs(xm), fp, fm, lat, a, transverse_hz,
                          float(np.linalg.norm(b_vector_gauss)))
