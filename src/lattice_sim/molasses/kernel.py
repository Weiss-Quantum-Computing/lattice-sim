"""numba kernels for the semiclassical Monte Carlo wave-function molasses model.

One trajectory = one atom: classical position r and momentum p, internal
state psi on the ground manifold (dimension G = 2F+1), evolved under the
non-Hermitian effective Hamiltonian (see ``levels``)

    H psi = C1 |u|^2 psi + C2 (u*.F)(u.F) psi + C3 (u.F)(u*.F) psi + (w_Z . F) psi,
    C_n   = sum_L kappa_L coeff_L[n],
    kappa_L = (Omega_0^2/4) (Delta_L - i Gamma/2) / (Delta_L^2 + Gamma^2/4),

with u(r) = sum_b amp_b eps_b exp(i (k_b . r + phi_b)) and w_Z = g_F mu_B B/hbar.
The norm of psi decays at the photon scattering rate; a quantum jump is made
when |psi|^2 falls below a uniform random threshold (waiting-time method,
checked once per step). At a jump: excited level L is drawn with weight
Gamma'_L <M_L>, the absorbed photon's beam b with weight |W_L^(b) psi|^2
times the Doppler factor of that beam (this is where Doppler cooling enters),
the emitted polarization q with weight |D_q^dag e|^2 and its direction from
the dipole pattern of e_q; the atom receives hbar k_b and -hbar k n_hat. A
decay to F = 3 (probability 1 - branch_L) is followed by repumping cycles
(F = 3 -> F' = 4, return to F = 4 with probability 7/12 per cycle), each
giving one repump-beam kick and one isotropic emission kick; the internal
state is then reset to a random |m>.

The mechanical force between jumps is the reactive force
F = -hbar grad <H_herm>; the dissipative (radiation-pressure) force is the
mean of the absorption kicks. The motion uses a kick-drift-kick leapfrog;
psi uses RK4 at the mid-step position. The step adapts to the local light
shift (|H| dt <= DT_PHASE) and scattering rate (gamma dt <= DT_JUMP).

Lattice: V = sum_i U_i sin^2(pi x_i / a_i) about the site centre (blue
lattice, atoms at the intensity minimum). Crossing |x_i| = a_i/2 counts as a
hop and maps x_i to the neighbouring site. The excited state sees beta times
the ground lattice potential, shifting every Delta_L by (1 - beta) V/hbar.
"""

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                  # pragma: no cover - fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    prange = range

DT_PHASE = 0.25          # max |H| dt per step (rad)
DT_JUMP = 0.05           # max scattering probability per step
GMAX = 9
EMAX = 11
INV_SQRT2 = 0.7071067811865476


# ----------------------------------------------------------------- random numbers
@njit(cache=True)
def _rng_next(state):
    # xorshift64*; state is a length-1 uint64 array
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    return (x * np.uint64(2685821657736338717)) >> np.uint64(11)


@njit(cache=True)
def _uniform(state):
    return (_rng_next(state) + 0.5) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _gauss(state):
    u1 = _uniform(state)
    u2 = _uniform(state)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


@njit(cache=True)
def _iso_dir(state, out):
    z = 2.0 * _uniform(state) - 1.0
    ph = 2.0 * np.pi * _uniform(state)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    out[0] = s * np.cos(ph)
    out[1] = s * np.sin(ph)
    out[2] = z


@njit(cache=True)
def _dipole_dir(state, q, out):
    # emission direction for a dipole e_q (q = -1, 0, 1) about z: rejection sampling
    while True:
        _iso_dir(state, out)
        if q == 0:
            acc = 1.0 - out[2] * out[2]
        else:
            acc = 1.0 - 0.5 * (out[0] * out[0] + out[1] * out[1])
        if _uniform(state) < acc:
            return


# ----------------------------------------------------------------- spin algebra
@njit(cache=True)
def _apply_dotf(w0, w1, w2, psi, out, mvals, cp, g):
    # out = (w . F) psi for a complex vector w; F+ |m_i> = cp[i] |m_{i+1}>
    wp = 0.5 * (w0 - 1j * w1)        # coefficient of F+
    wm = 0.5 * (w0 + 1j * w1)        # coefficient of F-
    for i in range(g):
        out[i] = w2 * mvals[i] * psi[i]
    for i in range(g - 1):
        out[i + 1] += wp * cp[i] * psi[i]
        out[i] += wm * cp[i] * psi[i + 1]


@njit(cache=True)
def _hpsi(psi, out, u, c1, c2, c3, wz, mvals, cp, g, t1, t2):
    # out = H psi
    u2 = (u[0] * np.conj(u[0]) + u[1] * np.conj(u[1]) + u[2] * np.conj(u[2])).real
    # (u*.F)(u.F) psi
    _apply_dotf(u[0], u[1], u[2], psi, t1, mvals, cp, g)
    _apply_dotf(np.conj(u[0]), np.conj(u[1]), np.conj(u[2]), t1, t2, mvals, cp, g)
    for i in range(g):
        out[i] = c1 * u2 * psi[i] + c2 * t2[i]
    # (u.F)(u*.F) psi
    _apply_dotf(np.conj(u[0]), np.conj(u[1]), np.conj(u[2]), psi, t1, mvals, cp, g)
    _apply_dotf(u[0], u[1], u[2], t1, t2, mvals, cp, g)
    for i in range(g):
        out[i] += c3 * t2[i]
    # Zeeman
    _apply_dotf(wz[0] + 0j, wz[1] + 0j, wz[2] + 0j, psi, t1, mvals, cp, g)
    for i in range(g):
        out[i] += t1[i]


@njit(cache=True)
def _field(r, kv, eps, amp, phase, nb, u, du, want_grad):
    for j in range(3):
        u[j] = 0.0
        for i in range(3):
            du[i, j] = 0.0
    for b in range(nb):
        th = kv[b, 0] * r[0] + kv[b, 1] * r[1] + kv[b, 2] * r[2] + phase[b]
        e = amp[b] * (np.cos(th) + 1j * np.sin(th))
        for j in range(3):
            v = e * eps[b, j]
            u[j] += v
            if want_grad:
                for i in range(3):
                    du[i, j] += 1j * kv[b, i] * v


@njit(cache=True)
def _norm2(psi, g):
    s = 0.0
    for i in range(g):
        s += (psi[i] * np.conj(psi[i])).real
    return s


@njit(cache=True)
def _lattice_v(r, uvec, avec):
    v = 0.0
    for i in range(3):
        if avec[i] > 0.0:
            s = np.sin(np.pi * r[i] / avec[i])
            v += uvec[i] * s * s
    return v


@njit(cache=True)
def _add_lattice_force(r, uvec, avec, f):
    for i in range(3):
        if avec[i] > 0.0:
            f[i] -= uvec[i] * np.pi / avec[i] * np.sin(2.0 * np.pi * r[i] / avec[i])


@njit(cache=True)
def _coeffs(delta_ref, vlat, beta, hbar, om2q, gamma, offs, ca, cb, cc, nl, kap, gpr):
    # kappa_L (complex, rad/s) and Gamma'_L (rad/s); returns C1, C2, C3
    c1 = 0.0 + 0.0j
    c2 = 0.0 + 0.0j
    c3 = 0.0 + 0.0j
    shift = (1.0 - beta) * vlat / hbar
    for L in range(nl):
        d = delta_ref - offs[L] + shift
        den = d * d + 0.25 * gamma * gamma
        k = om2q * (d - 0.5j * gamma) / den
        kap[L] = k
        gpr[L] = om2q * gamma / den
        c1 += k * ca[L]
        c2 += k * cb[L]
        c3 += k * cc[L]
    return c1, c2, c3


@njit(cache=True)
def _force(psi, u, du, c1r, c2r, c3r, mvals, cp, g, hbar, fout, a1, b1, a2, b2):
    # reactive force -hbar grad <H_herm>, psi normalised by the caller's norm
    nrm = _norm2(psi, g)
    _apply_dotf(u[0], u[1], u[2], psi, a1, mvals, cp, g)                         # (u.F) psi
    _apply_dotf(np.conj(u[0]), np.conj(u[1]), np.conj(u[2]), psi, b1, mvals, cp, g)  # (u*.F) psi
    for i in range(3):
        _apply_dotf(du[i, 0], du[i, 1], du[i, 2], psi, a2, mvals, cp, g)
        _apply_dotf(np.conj(du[i, 0]), np.conj(du[i, 1]), np.conj(du[i, 2]), psi, b2, mvals, cp, g)
        s2 = 0.0
        s3 = 0.0
        for m in range(g):
            s2 += (np.conj(a1[m]) * a2[m]).real
            s3 += (np.conj(b1[m]) * b2[m]).real
        s1 = (np.conj(u[0]) * du[i, 0] + np.conj(u[1]) * du[i, 1] + np.conj(u[2]) * du[i, 2]).real
        fout[i] = -hbar * 2.0 * (c1r * s1 + (c2r * s2 + c3r * s3) / nrm)


@njit(cache=True)
def _expect_h(psi, u, c1r, c2r, c3r, mvals, cp, g, a1, b1):
    nrm = _norm2(psi, g)
    _apply_dotf(u[0], u[1], u[2], psi, a1, mvals, cp, g)
    _apply_dotf(np.conj(u[0]), np.conj(u[1]), np.conj(u[2]), psi, b1, mvals, cp, g)
    u2 = (u[0] * np.conj(u[0]) + u[1] * np.conj(u[1]) + u[2] * np.conj(u[2])).real
    return c1r * u2 + (c2r * _norm2(a1, g) + c3r * _norm2(b1, g)) / nrm


@njit(cache=True)
def _w_apply(dm, de, uq0, uqm, uqp, psi, out, g):
    # out = (sum_q u_q D_q) psi, dm: (3, EMAX, GMAX) for one level; uqm = u_{-1}, uq0, uqp = u_{+1}
    for e in range(de):
        out[e] = 0.0
    for e in range(de):
        acc = 0.0 + 0.0j
        for m in range(g):
            acc += (uqm * dm[0, e, m] + uq0 * dm[1, e, m] + uqp * dm[2, e, m]) * psi[m]
        out[e] = acc


@njit(cache=True)
def _sph(u):
    # u_q = e_q^* . u, e_{+1} = -(x+iy)/sqrt2, e_0 = z, e_{-1} = (x-iy)/sqrt2
    um = INV_SQRT2 * (u[0] + 1j * u[1])
    up = -INV_SQRT2 * (u[0] - 1j * u[1])
    return um, u[2], up


@njit(cache=True)
def _rk4(psi, u, c1, c2, c3, wz, mvals, cp, g, dt, k1, k2, k3, k4, tmp, t1, t2):
    _hpsi(psi, k1, u, c1, c2, c3, wz, mvals, cp, g, t1, t2)
    for i in range(g):
        tmp[i] = psi[i] - 0.5j * dt * k1[i]
    _hpsi(tmp, k2, u, c1, c2, c3, wz, mvals, cp, g, t1, t2)
    for i in range(g):
        tmp[i] = psi[i] - 0.5j * dt * k2[i]
    _hpsi(tmp, k3, u, c1, c2, c3, wz, mvals, cp, g, t1, t2)
    for i in range(g):
        tmp[i] = psi[i] - 1j * dt * k3[i]
    _hpsi(tmp, k4, u, c1, c2, c3, wz, mvals, cp, g, t1, t2)
    for i in range(g):
        psi[i] = psi[i] - 1j * dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True)
def _jump(st, psi, u, r, p, mass, hbar, kmag, kv, eps, amp, phase, nb, dmat, dims_e, branch,
          gpr, kap, ca, cb, cc, nl, mvals, cp, g, gamma, delta_ref, vlat, beta, doppler,
          krep, ework, a1, b1, wb, ndir, motion):
    """One quantum jump; modifies psi (normalised) and p. Returns 1 if a depump/repump occurred."""
    nrm = _norm2(psi, g)
    for i in range(g):
        psi[i] /= np.sqrt(nrm)
    _apply_dotf(u[0], u[1], u[2], psi, a1, mvals, cp, g)
    _apply_dotf(np.conj(u[0]), np.conj(u[1]), np.conj(u[2]), psi, b1, mvals, cp, g)
    u2 = (u[0] * np.conj(u[0]) + u[1] * np.conj(u[1]) + u[2] * np.conj(u[2])).real
    na = _norm2(a1, g)
    nbb = _norm2(b1, g)
    tot = 0.0
    wl = np.zeros(nl)
    for L in range(nl):
        wl[L] = max(0.0, gpr[L] * (ca[L] * u2 + cb[L] * na + cc[L] * nbb))
        tot += wl[L]
    x = _uniform(st) * tot
    lev = nl - 1
    for L in range(nl):
        x -= wl[L]
        if x <= 0.0:
            lev = L
            break
    de = dims_e[lev]
    # absorption: beam b with weight |W^(b) psi|^2 x Doppler factor
    # local detuning of the chosen level, from kappa_L / Gamma'_L = (Delta_L - i Gamma/2) / Gamma
    dl = kap[lev].real / gpr[lev] * gamma
    wbt = 0.0
    for b in range(nb):
        th = kv[b, 0] * r[0] + kv[b, 1] * r[1] + kv[b, 2] * r[2] + phase[b]
        ee = amp[b] * (np.cos(th) + 1j * np.sin(th))
        v0 = ee * eps[b, 0]
        v1 = ee * eps[b, 1]
        v2 = ee * eps[b, 2]
        um = INV_SQRT2 * (v0 + 1j * v1)
        up = -INV_SQRT2 * (v0 - 1j * v1)
        _w_apply(dmat[lev], de, v2, um, up, psi, ework, g)
        w = 0.0
        for e in range(de):
            w += (ework[e] * np.conj(ework[e])).real
        if doppler:
            kvdot = (kv[b, 0] * p[0] + kv[b, 1] * p[1] + kv[b, 2] * p[2]) / mass
            w *= (dl * dl + 0.25 * gamma * gamma) / ((dl - kvdot) ** 2 + 0.25 * gamma * gamma)
        wb[b] = w
        wbt += w
    x = _uniform(st) * wbt
    bsel = nb - 1
    for b in range(nb):
        x -= wb[b]
        if x <= 0.0:
            bsel = b
            break
    for i in range(3):
        if motion[i]:
            p[i] += hbar * kv[bsel, i]
    # excited amplitude for level lev (full field)
    um, u0, up = _sph(u)
    _w_apply(dmat[lev], de, u0, um, up, psi, ework, g)
    ne = 0.0
    for e in range(de):
        ne += (ework[e] * np.conj(ework[e])).real
    for e in range(de):
        ework[e] /= np.sqrt(ne)
    if _uniform(st) > branch[lev]:
        # decay to F = 3, then repump cycles F=3 -> F'=4 (return to F=4 with p = 7/12)
        _iso_dir(st, ndir)
        for i in range(3):
            if motion[i]:
                p[i] -= hbar * kmag * ndir[i]
        while True:
            for i in range(3):
                if motion[i]:
                    p[i] += hbar * kmag * krep[i]
            _iso_dir(st, ndir)
            for i in range(3):
                if motion[i]:
                    p[i] -= hbar * kmag * ndir[i]
            if _uniform(st) < 7.0 / 12.0:
                break
        mi = int(_uniform(st) * g)
        if mi >= g:
            mi = g - 1
        for i in range(g):
            psi[i] = 0.0
        psi[mi] = 1.0
        return 1
    # emission polarization q: weight |D_q^dag e|^2
    wq = np.zeros(3)
    tq = 0.0
    for qi in range(3):
        s = 0.0
        for m in range(g):
            acc = 0.0 + 0.0j
            for e in range(de):
                acc += dmat[lev, qi, e, m] * ework[e]
            s += (acc * np.conj(acc)).real
        wq[qi] = s
        tq += s
    x = _uniform(st) * tq
    qsel = 2
    for qi in range(3):
        x -= wq[qi]
        if x <= 0.0:
            qsel = qi
            break
    for m in range(g):
        acc = 0.0 + 0.0j
        for e in range(de):
            acc += dmat[lev, qsel, e, m] * ework[e]
        psi[m] = acc
    nn = _norm2(psi, g)
    for m in range(g):
        psi[m] /= np.sqrt(nn)
    _dipole_dir(st, qsel - 1, ndir)
    for i in range(3):
        if motion[i]:
            p[i] -= hbar * kmag * ndir[i]
    return 0


@njit(parallel=True, cache=True, fastmath=True)
def simulate(r0, p0, psi0, phases, seeds,
             kv, eps, amp, kmag, krep,
             mvals, cp, g, dmat, dims_e, offs, ca, cb, cc, branch,
             om2q, delta_ref, gamma, hbar, mass, wz,
             uvec, avec, beta, doppler, motion,
             t_total, t_burn, dt_max, t_sample, n_samples,
             drag_v, drag, dt_phase, dt_jump):
    """Run N independent trajectories. With drag True (and motion all False)
    the position is prescribed, r(t) = r0 + drag_v t, and the momentum the
    light transfers after t_burn is recorded instead (p_tr; p_tr2 is the part
    after the midpoint of the recorded window).

    Returns (ke_sum (N,3), t_rec (N), n_scat (N), n_dep (N), hops (N,3),
    e_samples (N, n_samples, 3), lsh_sum (N), p_transfer (N,3), steps (N), p_transfer_2nd_half (N,3))."""
    n = r0.shape[0]
    nb = kv.shape[0]
    nl = offs.shape[0]
    ke_sum = np.zeros((n, 3))
    t_rec = np.zeros(n)
    n_scat = np.zeros(n)
    n_dep = np.zeros(n)
    hops = np.zeros((n, 3))
    e_samp = np.zeros((n, n_samples, 3))
    lsh_sum = np.zeros(n)
    p_tr = np.zeros((n, 3))
    nsteps = np.zeros(n)
    dragging = drag
    t_mid = 0.5 * (t_burn + t_total)
    p_tr2 = np.zeros((n, 3))
    for a in prange(n):
        st = np.zeros(1, dtype=np.uint64)
        st[0] = seeds[a]
        r = r0[a].copy()
        p = p0[a].copy()
        psi = psi0[a].copy()
        ph = phases[a]
        u = np.zeros(3, dtype=np.complex128)
        du = np.zeros((3, 3), dtype=np.complex128)
        k1 = np.zeros(GMAX, dtype=np.complex128)
        k2 = np.zeros(GMAX, dtype=np.complex128)
        k3 = np.zeros(GMAX, dtype=np.complex128)
        k4 = np.zeros(GMAX, dtype=np.complex128)
        tmp = np.zeros(GMAX, dtype=np.complex128)
        t1 = np.zeros(GMAX, dtype=np.complex128)
        t2 = np.zeros(GMAX, dtype=np.complex128)
        a1 = np.zeros(GMAX, dtype=np.complex128)
        b1 = np.zeros(GMAX, dtype=np.complex128)
        a2 = np.zeros(GMAX, dtype=np.complex128)
        b2 = np.zeros(GMAX, dtype=np.complex128)
        ework = np.zeros(EMAX, dtype=np.complex128)
        wb = np.zeros(nb)
        ndir = np.zeros(3)
        kap = np.zeros(nl, dtype=np.complex128)
        gpr = np.zeros(nl)
        f = np.zeros(3)
        rmid = np.zeros(3)
        thresh = _uniform(st)
        # initial force
        _field(r, kv, eps, amp, ph, nb, u, du, True)
        vlat = _lattice_v(r, uvec, avec)
        c1, c2, c3 = _coeffs(delta_ref, vlat, beta, hbar, om2q, gamma, offs, ca, cb, cc, nl, kap, gpr)
        _force(psi, u, du, c1.real, c2.real, c3.real, mvals, cp, g, hbar, f, a1, b1, a2, b2)
        _add_lattice_force(r, uvec, avec, f)
        t = 0.0
        next_s = t_burn
        si = 0
        fmax = 0.5 * (g - 1)
        ff = fmax * (fmax + 1.0)
        wzn = np.sqrt(wz[0] ** 2 + wz[1] ** 2 + wz[2] ** 2) * fmax
        while t < t_total:
            # step size from the local light shift and scattering rate
            u2 = (u[0] * np.conj(u[0]) + u[1] * np.conj(u[1]) + u[2] * np.conj(u[2])).real
            hsc = (abs(c1) + (abs(c2) + abs(c3)) * ff) * max(u2, 1e-3) + wzn
            gsc = 0.0
            for L in range(nl):
                gsc += gpr[L] * (abs(ca[L]) + (abs(cb[L]) + abs(cc[L])) * ff) * max(u2, 1e-3)
            dt = dt_max
            if dt_phase / hsc < dt:
                dt = dt_phase / hsc
            if gsc > 0.0 and dt_jump / gsc < dt:
                dt = dt_jump / gsc
            # kick - drift - (internal) - drift - kick
            rec = 1.0 if t >= t_burn else 0.0
            rec2 = 1.0 if t >= t_mid else 0.0
            for i in range(3):
                if motion[i]:
                    p[i] += 0.5 * dt * f[i]
                if dragging:
                    p_tr[a, i] += rec * 0.5 * dt * f[i]
                    p_tr2[a, i] += rec2 * 0.5 * dt * f[i]
                    rmid[i] = r[i] + 0.5 * dt * drag_v[i]
                elif motion[i]:
                    rmid[i] = r[i] + 0.5 * dt * p[i] / mass
                else:
                    rmid[i] = r[i]
            _field(rmid, kv, eps, amp, ph, nb, u, du, False)
            vlat = _lattice_v(rmid, uvec, avec)
            c1, c2, c3 = _coeffs(delta_ref, vlat, beta, hbar, om2q, gamma, offs, ca, cb, cc, nl, kap, gpr)
            _rk4(psi, u, c1, c2, c3, wz, mvals, cp, g, dt, k1, k2, k3, k4, tmp, t1, t2)
            for i in range(3):
                if dragging:
                    r[i] = rmid[i] + 0.5 * dt * drag_v[i]
                elif motion[i]:
                    r[i] = rmid[i] + 0.5 * dt * p[i] / mass
            # lattice hop: map to the neighbouring site
            for i in range(3):
                if avec[i] > 0.0 and motion[i]:
                    if r[i] > 0.5 * avec[i]:
                        r[i] -= avec[i]
                        if t >= t_burn:
                            hops[a, i] += 1.0
                    elif r[i] < -0.5 * avec[i]:
                        r[i] += avec[i]
                        if t >= t_burn:
                            hops[a, i] += 1.0
            _field(r, kv, eps, amp, ph, nb, u, du, True)
            vlat = _lattice_v(r, uvec, avec)
            c1, c2, c3 = _coeffs(delta_ref, vlat, beta, hbar, om2q, gamma, offs, ca, cb, cc, nl, kap, gpr)
            # quantum jump?
            if _norm2(psi, g) < thresh:
                pb0 = p[0]
                pb1 = p[1]
                pb2 = p[2]
                motion_kick = motion.copy()
                if dragging:
                    for i in range(3):
                        motion_kick[i] = True
                dep = _jump(st, psi, u, r, p, mass, hbar, kmag, kv, eps, amp, ph, nb, dmat, dims_e,
                            branch, gpr, kap, ca, cb, cc, nl, mvals, cp, g, gamma, delta_ref, vlat, beta,
                            doppler, krep, ework, a1, b1, wb, ndir, motion_kick)
                if dragging:
                    p_tr[a, 0] += rec * (p[0] - pb0)
                    p_tr[a, 1] += rec * (p[1] - pb1)
                    p_tr[a, 2] += rec * (p[2] - pb2)
                    p_tr2[a, 0] += rec2 * (p[0] - pb0)
                    p_tr2[a, 1] += rec2 * (p[1] - pb1)
                    p_tr2[a, 2] += rec2 * (p[2] - pb2)
                    p[0] = pb0
                    p[1] = pb1
                    p[2] = pb2
                if t >= t_burn:
                    n_scat[a] += 1.0
                    n_dep[a] += dep
                thresh = _uniform(st)
            _force(psi, u, du, c1.real, c2.real, c3.real, mvals, cp, g, hbar, f, a1, b1, a2, b2)
            _add_lattice_force(r, uvec, avec, f)
            for i in range(3):
                if motion[i]:
                    p[i] += 0.5 * dt * f[i]
                elif dragging:
                    p_tr[a, i] += rec * 0.5 * dt * f[i]
                    p_tr2[a, i] += rec2 * 0.5 * dt * f[i]
            t += dt
            nsteps[a] += 1.0
            if t >= t_burn:
                t_rec[a] += dt
                for i in range(3):
                    ke_sum[a, i] += dt * p[i] * p[i] / (2.0 * mass)
                if si < n_samples and t >= next_s:
                    lsh_sum[a] += hbar * _expect_h(psi, u, c1.real, c2.real, c3.real, mvals, cp, g, a1, b1)
                    for i in range(3):
                        ev = p[i] * p[i] / (2.0 * mass)
                        if avec[i] > 0.0:
                            s = np.sin(np.pi * r[i] / avec[i])
                            ev += uvec[i] * s * s
                        e_samp[a, si, i] = ev
                    si += 1
                    next_s += t_sample
    return ke_sum, t_rec, n_scat, n_dep, hops, e_samp, lsh_sum, p_tr, nsteps, p_tr2
