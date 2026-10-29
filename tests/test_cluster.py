"""Tests for the cluster-state model (lattice_sim.cluster) and its browser
engine (tools/cluster_engine.js, run under node when available)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from lattice_sim.cluster import lattice as L  # noqa: E402
from lattice_sim.cluster import sequence as S  # noqa: E402
from lattice_sim.cluster import stabilizers as T  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


# ------------------------------------------------------------ helpers
def _spec(region, occ, steps):
    return {"shape": list(region.shape), "occupancy": [int(v) for v in occ], "steps": steps}


def statevector(spec):
    """Explicit state vector of a unitary sequence with the leg physics written
    out (phase exp(-i phi) on |q_r = 1, q_s = 0> per leg, pi pulse about x
    between legs), independent of the Stim gate decomposition."""
    reg = S.region_of(spec)
    n = reg.n
    occ = np.array(spec["occupancy"], bool)
    bits = (np.arange(2**n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    psi = np.zeros(2**n, complex)
    psi[0] = 1
    px, py = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])

    def rot(theta, phi):
        return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * (np.cos(phi) * px + np.sin(phi) * py)

    def apply(psi, u, qubits):
        p = psi.reshape((2,) * n)
        for k in qubits:
            p = np.moveaxis(np.tensordot(u, p, axes=([1], [k])), 0, k)
        return p.reshape(-1)

    present = list(np.flatnonzero(occ))
    for st in spec["steps"]:
        tg = present if st.get("sites") is None else [int(i) for i in st["sites"]]
        if st["op"] == "pulse":
            psi = apply(psi, rot(st["q"] * np.pi / 2, st.get("phase_q", 0) * np.pi / 2), tg)
        elif st["op"] == "rz":
            a = st["q"] * np.pi / 2
            psi = apply(psi, np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)]), tg)
        elif st["op"] == "collide":
            bl = [b for a in st["axes"] for b in L.bonds(reg, occ, a)]
            leg_phase = S.LEG_PHASE_ECHO if st.get("mode", "echo") == "echo" else S.LEG_PHASE_SINGLE

            def leg(psi):
                ph = np.zeros(2**n)
                for r, s in bl:
                    ph += leg_phase * ((bits[:, r] == S.PLUS_MOVER_QUBIT) & (bits[:, s] != S.PLUS_MOVER_QUBIT))
                return psi * np.exp(-1j * ph)
            psi = leg(psi)
            if st.get("mode", "echo") == "echo":
                psi = apply(psi, rot(np.pi, 0), present)
                psi = leg(psi)
    return psi


def sv_expectation(psi, n, ps):
    mats = {1: np.array([[0, 1], [1, 0]]), 2: np.array([[0, -1j], [1j, 0]]), 3: np.diag([1, -1])}
    p = psi.reshape((2,) * n)
    for q in range(n):
        if ps[q]:
            p = np.moveaxis(np.tensordot(mats[ps[q]], p, axes=([1], [q])), 0, q)
    return float(np.real(ps.sign * np.vdot(psi, p.reshape(-1))))


SMALL = [
    ("2x2x2 block", L.Region(2, 2, 2), None),
    ("3x2x1 plane", L.Region(3, 2, 1), None),
    ("2x2x3 with vacancy", L.Region(2, 2, 3), [5]),
]


# ------------------------------------------------------------ physics
@pytest.mark.parametrize("name,region,holes", SMALL)
def test_stim_matches_statevector(name, region, holes):
    """Every Stim site stabilizer is +1 on the explicit state vector (sequential
    echo, interleaved echo, single-leg pi gate, addressed phase shifts)."""
    occ = L.block(region, region.shape)
    if holes:
        occ[holes] = False
    variants = [S.cluster_sequence(occ, region.shape),
                S.cluster_sequence(occ, region.shape, interleaved=True),
                S.cluster_sequence(occ, region.shape, mode="single")]
    v = S.cluster_sequence(occ, region.shape)
    v["steps"] += [{"op": "rz", "q": 1, "sites": [int(np.flatnonzero(occ)[0])]},
                   {"op": "pulse", "q": 1, "phase_q": 1, "sites": None}]
    variants.append(v)
    for spec in variants:
        psi = statevector(spec)
        for i, ps in T.site_stabilizers(spec).items():
            assert sv_expectation(psi, region.n, ps) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("shape,size", [((4, 4, 3), (4, 4, 3)), ((7, 7, 4), (5, 5, 2)), ((6, 1, 1), (6, 1, 1))])
def test_degree_parity_rule(shape, size):
    """S_i = +-P_i prod Z_nb with P = X (odd degree) or Y (even degree) for the
    sqrt(X) start, for a sorted block, a block with vacancies and a random filling."""
    reg = L.Region(*shape)
    fills = [L.block(reg, size), L.remove_random(L.block(reg, size), k=3, rng=4),
             L.random_filling(reg, 0.54, rng=5), L.peripheral(reg, L.block(reg, size), 0.3, rng=6)]
    for occ in fills:
        spec = S.cluster_sequence(occ, shape)
        pred = T.predicted_type(spec)
        for i, ps in T.site_stabilizers(spec).items():
            sign, d = T.describe(ps)
            letter, nb = pred[i]
            if not nb:
                assert d == {i: "Y"}                       # isolated atom after sqrt(X): -Y
                continue
            assert d.pop(i) == letter
            assert sorted(d) == nb and set(d.values()) == {"Z"}


def test_du_categories_correspond_to_plus_start():
    """Du's listed 4x4x3 categories (3: Y, 4: X, 5: Y, 6: X) are those of a
    |+> start (R_y(pi/2) first pulse); with Du's sqrt(X) start the letters are
    the opposite (3: X, 4: Y, 5: X, 6: Y)."""
    reg = L.Region(4, 4, 3)
    occ = L.block(reg, (4, 4, 3))
    deg = L.degrees(reg, occ)
    for phase_q, expect in ((0, {3: "X", 4: "Y", 5: "X", 6: "Y"}), (1, {3: "Y", 4: "X", 5: "Y", 6: "X"})):
        spec = S.cluster_sequence(occ, reg.shape)
        spec["steps"][0]["phase_q"] = phase_q
        got = {}
        for i, ps in T.site_stabilizers(spec).items():
            got.setdefault(int(deg[i]), set()).add(T.pauli_letter(ps, i))
        assert got == {k: {v} for k, v in expect.items()}


def test_z_quarter_turn_on_even_degree_gives_two_configurations():
    """Z(pi/2) on the even-degree vertices makes every centre letter X, so XZ-A
    and XZ-B together measure all 48 stabilizers of the 4x4x3 block; on the
    odd-degree vertices (Du's prescription) they measure 0."""
    reg = L.Region(4, 4, 3)
    occ = L.block(reg, (4, 4, 3))
    deg = L.degrees(reg, occ)
    for parity, expect in ((0, 48), (1, 0)):
        spec = S.cluster_sequence(occ, reg.shape)
        spec["steps"].append({"op": "rz", "q": 1, "sites": [int(i) for i in np.flatnonzero(occ & (deg % 2 == parity))]})
        stabs = T.site_stabilizers(spec)
        n_meas = sum(len(T.measurable(stabs, T.config_bases(spec, c))) for c in ("XZ-A", "XZ-B"))
        assert n_meas == expect


def test_bell_pair_along_a_path():
    """Measuring every off-path atom in Z and every interior path atom in its
    own centre letter leaves the two path ends in a Bell pair (2 bits of mutual
    information, Du Sec. 2.2.4 / Cluster Route1.py) - here across the 4x4x3 block."""
    reg = L.Region(4, 4, 3)
    occ = L.block(reg, (4, 4, 3))
    spec = S.cluster_sequence(occ, reg.shape)
    path = [reg.index(0, 0, 0), reg.index(1, 0, 0), reg.index(1, 1, 0), reg.index(1, 1, 1), reg.index(2, 1, 1),
            reg.index(3, 1, 1), reg.index(3, 2, 1), reg.index(3, 3, 1), reg.index(3, 3, 2)]
    a, b = path[0], path[-1]
    stabs = T.site_stabilizers(spec)
    bases = {i: "Z" for i in np.flatnonzero(occ) if i not in path}
    bases.update({i: T.pauli_letter(stabs[i], i) for i in path[1:-1]})
    spec["steps"].append({"op": "measure", "bases": {str(k): v for k, v in bases.items()}, "sites": sorted(int(k) for k in bases)})
    sa, sb, sab = T.entropy_bits(spec, [a]), T.entropy_bits(spec, [b]), T.entropy_bits(spec, [a, b])
    assert (sa, sb, sab) == (1, 1, 0)


def test_entropy_of_a_half_block():
    """The x cut through a 4x4x3 cluster carries one bit per cut bond pair: 12 bits."""
    reg = L.Region(4, 4, 3)
    occ = L.block(reg, (4, 4, 3))
    spec = S.cluster_sequence(occ, reg.shape)
    half = [i for i in range(reg.n) if reg.coords(i)[0] < 2]
    assert T.entropy_bits(spec, half) == 12
    assert T.entropy_bits(S.cluster_sequence(occ, reg.shape, axes="yz"), half) == 0


def test_twirled_zz_probability():
    nm = S.NoiseModel(phase_offset=0.2)
    assert nm.zz_probability() == pytest.approx(np.sin(0.05) ** 2, rel=1e-12)
    nm = S.NoiseModel(phase_sigma=0.3)
    rng = np.random.default_rng(1)
    assert nm.zz_probability() == pytest.approx(np.mean(np.sin(rng.normal(0, 0.3, 400000) / 4) ** 2), rel=2e-2)


def test_noisy_values_and_fidelity():
    """Exact <S_i> from noiseless readout agree with configuration sampling, and
    the fidelity equals the product rule for independent dephasing (p per atom)."""
    reg = L.Region(3, 3, 1)
    occ = L.block(reg, (3, 3, 1))
    spec = S.cluster_sequence(occ, reg.shape, axes="xy")
    nm = S.NoiseModel(echo_contrast=0.96)
    means, fid, bound = T.sample_exact(spec, nm, 40000, seed=3)
    p = (1 - 0.96) / 2
    # a Z error on atom j commutes with the diagonal gates and flips only S_j; two
    # collision steps flip S_j when an odd number of them hit j
    flip = 2 * p * (1 - p)
    for i, m in means.items():
        assert m == pytest.approx(1 - 2 * flip, abs=0.01)
    assert fid == pytest.approx((1 - flip) ** occ.sum(), abs=0.01)
    assert bound <= fid + 0.01
    vals = T.sample_config(spec, "centre-A", nm, 40000, seed=4)
    for i, (m, se) in vals.items():
        assert m == pytest.approx(means[i], abs=5 * se + 0.01)
    assert L.degrees(reg, occ, "xy").max() == 4


# ------------------------------------------------------------ browser engine
def _engine_cases():
    reg = L.Region(4, 4, 3)
    occ = L.block(reg, (4, 4, 3))
    rng = np.random.default_rng(9)
    cases = []
    base = S.cluster_sequence(occ, reg.shape)
    holes = S.cluster_sequence(L.remove_random(occ, k=4, rng=2), reg.shape, interleaved=True)
    mixed = S.cluster_sequence(L.random_filling(L.Region(5, 4, 3), 0.6, rng=3), (5, 4, 3), axes="xz", mode="single")
    mixed["steps"] += [{"op": "rz", "q": 3, "sites": list(range(0, 60, 7))},
                       {"op": "pulse", "q": 1, "phase_q": 3, "sites": list(range(1, 60, 5))},
                       {"op": "pulse", "q": 2, "phase_q": 1, "sites": None},
                       {"op": "collide", "axes": "y", "mode": "echo"}]
    planes = S.cluster_sequence(L.plane(L.Region(5, 5, 3), "z", 1), (5, 5, 3), axes="xy")
    for name, spec in (("block", base), ("holes", holes), ("mixed", mixed), ("plane", planes)):
        n = int(np.prod(spec["shape"]))
        paulis = []
        stabs = T.site_stabilizers(spec)
        keys = sorted(stabs)
        for k in range(12):                       # products of stabilizers and random strings
            if k % 2 == 0:
                pick = rng.choice(keys, size=3, replace=False)
                ps = stabs[int(pick[0])] * stabs[int(pick[1])] * stabs[int(pick[2])]
            else:
                ps = stim.PauliString.random(n)
            sign, d = T.describe(ps, n)
            paulis.append({"sign": sign, "ops": {str(q): v for q, v in d.items()}})
        regions = [[int(i) for i in rng.choice(n, size=m, replace=False)] for m in (1, 5, n // 2)]
        cases.append(dict(name=name, spec=spec, paulis=paulis, regions=regions))
    noisy = dict(name="noisy", spec=base, noise=S.noise_preset("|3,±3⟩ sequential").to_dict(), shots=20000,
                 configs=["XZ-A", "YZ-B", "centre-A"])
    cases.append(noisy)
    meas = dict(base)
    meas["steps"] = base["steps"] + [{"op": "measure", "basis": "Z", "sites": list(range(0, 48, 3))},
                                     {"op": "clear", "sites": [1, 2]},
                                     {"op": "collide", "axes": "x", "mode": "echo"}]
    cases.append(dict(name="measured", spec=meas, paulis=[], regions=[[5, 6, 7], [10, 11, 25, 26], list(range(24))]))
    return cases


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if NODE is None:
        pytest.skip("node not available")
    cases = _engine_cases()
    path = tmp_path_factory.mktemp("cluster") / "cases.json"
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    out = subprocess.run([NODE, str(REPO / "tools" / "check_cluster_engine.js"), str(path)],
                         capture_output=True, text=True, check=True, cwd=REPO)
    return cases, json.loads(out.stdout)


def test_engine_gate_tables(engine):
    _, res = engine
    assert res["rowmul_mismatches"] == 0
    for name, table in res["gates"].items():
        t = stim.Tableau.from_named_gate(name) if name != "I" else stim.Tableau(1)
        imgs = [t.x_output(0), t.x_output(0) * t.z_output(0) * 1j, t.z_output(0)]   # Y = i X Z
        for (sign, letter), img in zip(table, imgs):
            assert int(np.real(img.sign)) == sign and img[0] == letter


def test_engine_matches_stim(engine):
    cases, res = engine
    for c, r in zip(cases, res["cases"]):
        spec = c["spec"]
        if c["name"] != "measured":
            for i, ps in T.site_stabilizers(spec).items():
                sign, d = T.describe(ps)
                assert r["stabilizers"][str(i)] == {"sign": sign, "ops": {str(q): v for q, v in d.items()}}, (c["name"], i)
            for P, v in zip(c.get("paulis", []), r["expectations"]):
                ps = stim.PauliString("".join(P["ops"].get(str(q), "_") for q in range(int(np.prod(spec["shape"])))))
                assert v == T.expectation(spec, ps * P["sign"]), c["name"]
        for sites, v in zip(c.get("regions", []), r["entropies"]):
            assert v == T.entropy_bits(spec, sites), (c["name"], sites)


def test_engine_noise_sampling(engine):
    cases, res = engine
    c = next(c for c in cases if c["name"] == "noisy")
    r = next(r for r in res["cases"] if r["name"] == "noisy")
    nm = S.NoiseModel(**c["noise"])
    means, fid, bound = T.sample_exact(c["spec"], nm, c["shots"], seed=5)
    sd = 1 / np.sqrt(c["shots"])
    for i, m in means.items():
        assert r["exact"]["means"][str(i)] == pytest.approx(m, abs=6 * sd)
    assert r["exact"]["fidelity"] == pytest.approx(fid, abs=6 * np.sqrt(fid * (1 - fid) / c["shots"]) + 1e-3)
    for cfg in c["configs"]:
        py = T.sample_config(c["spec"], cfg, nm, c["shots"], seed=6)
        js = r["configs"][cfg]
        assert sorted(int(k) for k in js) == sorted(py)
        for i, (m, se) in py.items():
            assert js[str(i)][0] == pytest.approx(m, abs=8 * se + 1e-3)


def test_explorer_is_built():
    """tools/cluster_explorer.html is the current build of the template, engine and presets."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cluster_build_explorer", REPO / "scripts" / "cluster_build_explorer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = (REPO / "tools" / "cluster_explorer.html").read_text(encoding="utf-8")
    assert built == mod.build(), "run python scripts/cluster_build_explorer.py"
