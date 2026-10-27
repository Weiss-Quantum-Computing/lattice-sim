"""Stabilizers, measurement configurations, entanglement and sampled
stabilizer values of the cluster-state sequences (Stim reference).

Site stabilizer. Every atom starts in |0>, stabilized by Z_i; the image of
Z_i through the unitary part of the sequence is the stabilizer S_i of that
site. For the echoed gate after the initial sqrt(X) (R_x(pi/2)),

    S_i = +-P_i prod_{j in nb(i)} Z_j,   P_i = X for odd degree, Y for even,

where the degree counts the occupied neighbours along the collided axes
(``predicted_type``; tested against Stim and an explicit state vector). Du
2025 (p. 57) lists the opposite assignment (corner, 3 neighbours: Y; edge, 4:
X). That assignment is what a |+> start (R_y(pi/2) first pulse) gives, or
the X-outputs of the Stim tableau, not the state prepared from |0> by
sqrt(X); the Z(pi/2) correction that reduces the measurement to two
configurations therefore belongs on the even-degree vertices for the sqrt(X)
start (``tests/test_cluster.py``).

Fidelity. For Pauli noise the state is a mixture of E|psi><psi|E; the
fidelity is the probability that the accumulated error commutes with every
S_i, i.e. that every noiseless stabilizer readout is +1 (``sample_exact``).
The bound from measured stabilizers is F >= 1 - sum_i (1 - <S_i>) / 2
(commuting projectors; Toth and Guhne, PRA 72, 022340 (2005)). The mean
(1/n) sum <S_i> quoted as a bound in Du Eq. 2.4 is not one in general.
"""

import numpy as np

from .lattice import neighbours
from .sequence import region_of, to_stim, unitary_prefix

CONFIGS = ("XZ-A", "XZ-B", "YZ-A", "YZ-B", "centre-A", "centre-B")


def _sim(spec, measure_tail=True):
    import stim
    sim = stim.TableauSimulator()
    c = to_stim(spec, measure_tail=measure_tail)
    sim.set_num_qubits(region_of(spec).n)
    sim.do(c)
    return sim


def site_stabilizers(spec):
    """{site: stim.PauliString} for the unitary prefix of the sequence (image of Z_i)."""
    import stim
    pre = unitary_prefix(spec)
    n = region_of(spec).n
    t = stim.Tableau.from_circuit(to_stim(pre).without_noise(), ignore_reset=True) if pre["steps"] else stim.Tableau(n)
    if len(t) < n:
        t = t + stim.Tableau(n - len(t))
    occ = np.array(spec["occupancy"], bool)
    return {int(i): t.z_output(int(i)) for i in np.flatnonzero(occ)}


def collided_axes(spec):
    axes = ""
    for st in unitary_prefix(spec)["steps"]:
        if st["op"] == "collide":
            axes += "".join(a for a in st["axes"] if a not in axes)
    return axes


def predicted_type(spec):
    """{site: ('X' | 'Y', sorted neighbour list)} from the degree-parity rule."""
    reg = region_of(spec)
    occ = np.array(spec["occupancy"], bool)
    nb = neighbours(reg, occ, collided_axes(spec))
    return {int(i): ("X" if len(nb[i]) % 2 else "Y", sorted(nb[i])) for i in np.flatnonzero(occ)}


def pauli_letter(ps, q):
    return "_XYZ"[ps[q]]


def describe(ps, n=None):
    """(sign, {qubit: letter}) of a Pauli string."""
    n = len(ps) if n is None else n
    sign = int(np.real(ps.sign))
    return sign, {q: pauli_letter(ps, q) for q in range(n) if ps[q]}


def expectation(spec, pauli):
    """<P> in {-1, 0, +1} after the whole sequence (random outcomes for measurements)."""
    import stim
    return int(_sim(spec).peek_observable_expectation(stim.PauliString(pauli)))


def entropy_bits(spec, sites, measure_tail=True):
    """Von Neumann entropy (bits) of the reduced state on ``sites``:
    S_A = rank(G restricted to A) - |A| for the n stabilizer generators G."""
    sim = _sim(spec, measure_tail)
    gens = sim.canonical_stabilizers()
    sites = [int(s) for s in sites]
    if not sites:
        return 0
    m = np.zeros((len(gens), 2 * len(sites)), np.uint8)
    for r, g in enumerate(gens):
        for k, q in enumerate(sites):
            p = g[q]
            m[r, k] = p in (1, 2)        # X component
            m[r, len(sites) + k] = p in (2, 3)
    return _gf2_rank(m) - len(sites)


def _gf2_rank(m):
    m = m.copy()
    rank = 0
    rows, cols = m.shape
    for col in range(cols):
        piv = next((r for r in range(rank, rows) if m[r, col]), None)
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


# ------------------------------------------------------------ measurement configurations
def config_bases(spec, config):
    """{site: basis} of a configuration. "XZ-A": X on parity-0 (A) sites, Z on B;
    "YZ-B": Y on B, Z on A; "centre-A": each A site in its own stabilizer letter
    (needs addressed rotations where the letters differ), Z on B."""
    reg = region_of(spec)
    par = reg.parity()
    occ = np.array(spec["occupancy"], bool)
    kind, sub = config.split("-")
    centre_parity = 0 if sub == "A" else 1
    stabs = site_stabilizers(spec) if kind == "centre" else None
    out = {}
    for i in np.flatnonzero(occ):
        if par[i] != centre_parity:
            out[int(i)] = "Z"
        elif kind == "centre":
            out[int(i)] = pauli_letter(stabs[int(i)], int(i)) if stabs[int(i)][int(i)] else "Z"
        else:
            out[int(i)] = kind[0]
    return out


def measurable(stabs, bases):
    """Sites whose stabilizer is a product of the measured single-site Paulis."""
    out = []
    for i, ps in stabs.items():
        if all(bases.get(q) == pauli_letter(ps, q) for q in range(len(ps)) if ps[q]):
            out.append(i)
    return sorted(out)


def sample_config(spec, config, noise=None, shots=2000, seed=None):
    """Sampled <S_i> (relative to the ideal sign) for the stabilizers a
    configuration measures, from shot records of the noisy sequence followed
    by detection in the configuration's bases. Returns {site: (mean, stderr)}."""
    pre = unitary_prefix(spec)
    stabs = site_stabilizers(spec)
    bases = config_bases(spec, config)
    ok = measurable(stabs, bases)
    order = sorted(bases)
    seq = dict(pre)
    seq["steps"] = pre["steps"] + [{"op": "measure", "bases": {str(i): b for i, b in bases.items()},
                                    "sites": order}]
    c = to_stim(seq, noise)
    rec = c.compile_sampler(seed=seed).sample(shots).astype(np.int8)
    col = {q: k for k, q in enumerate(order)}
    out = {}
    for i in ok:
        ps = stabs[i]
        sign = int(np.real(ps.sign))
        par = np.zeros(shots, np.int8)
        for q in range(len(ps)):
            if ps[q]:
                par ^= rec[:, col[q]]
        v = sign * (1 - 2 * par.astype(float))
        out[i] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(shots)))
    return out


def sample_exact(spec, noise=None, shots=2000, seed=None):
    """Noiseless readout of every S_i after the noisy unitary prefix:
    ({site: <S_i>}, fidelity, union bound)."""
    stabs = site_stabilizers(spec)
    sites = sorted(stabs)
    c = to_stim(unitary_prefix(spec), noise, final_mpp=[stabs[i] for i in sites])
    rec = c.compile_sampler(seed=seed).sample(shots)
    val = 1 - 2 * rec.astype(float)          # MPP of the signed generator: +1 ideally
    means = {i: float(val[:, k].mean()) for k, i in enumerate(sites)}
    fid = float(np.mean(~rec.any(axis=1)))
    bound = 1 - 0.5 * sum(1 - m for m in means.values())
    return means, fid, bound
