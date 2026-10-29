"""Cluster-state stabilizers for the 4 x 4 x 3 sorted block (Du's plan).

(a) Stabilizer letter of every site after sqrt(X) and the echoed gates along
    x, y, z: X for an odd number of neighbours, Y for an even number.
(b) <S_i> by neighbour count under the noise presets (exact for the Pauli model).
(c) Stabilizers recorded by the two sublattice configurations XZ-A + XZ-B
    after Z(pi/2) on the even-degree sites (chosen from the actual filling, or
    planned for the perfect block), and by the four configurations XZ-A/B,
    YZ-A/B without the correction, for a perfect block, a block with 1-3
    vacancies and a block with uncleared peripheral atoms (20 fillings each).
    Z(pi/2) on the odd-degree sites (Du's prescription) records none.
(d) State fidelity and the stabilizer bound 1 - sum(1 - <S_i>)/2 against the
    echo contrast per collision step (sequential x, y, z), 48 atoms.

Writes figures/cluster_stabilizers.png and data/cluster_results.json.
Usage: python scripts/figure_cluster.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cluster import lattice as L  # noqa: E402
from lattice_sim.cluster import sequence as S  # noqa: E402
from lattice_sim.cluster import stabilizers as T  # noqa: E402
from lattice_sim.viz.style import COLORS, DOUBLE_COLUMN_IN, apply  # noqa: E402

SHOTS = 20000
REG = L.Region(6, 6, 5)
SIZE = (4, 4, 3)


def recorded(spec, configs):
    stabs = T.site_stabilizers(spec)
    got = set()
    for c in configs:
        got |= set(T.measurable(stabs, T.config_bases(spec, c)))
    return len(got), len(stabs)


def with_rz(spec, parity, planned=None):
    """Z(pi/2) on the sites of the given degree parity; degrees from the actual
    filling, or from ``planned`` (the intended occupancy) when given."""
    occ = np.array(spec["occupancy"], bool)
    ref = occ if planned is None else np.array(planned, bool)
    deg = L.degrees(S.region_of(spec), ref, T.collided_axes(spec))
    out = json.loads(json.dumps(spec))
    out["steps"].append({"op": "rz", "q": 1, "sites": [int(i) for i in np.flatnonzero(occ & ref & (deg % 2 == parity))]})
    return out


def main():
    apply()
    res = {}
    block = L.block(REG, SIZE)
    spec = S.cluster_sequence(block, REG.shape)
    deg = L.degrees(REG, block)
    stabs = T.site_stabilizers(spec)

    # (b) <S_i> by degree for the presets
    presets = [("|3,±3⟩ sequential", False), ("|3,±3⟩ interleaved x-y-z", True),
               ("|3,±1⟩ sequential", False), ("|3,±1⟩ interleaved x-y-z", True)]
    res["presets"] = {}
    for name, inter in presets:
        sp = S.cluster_sequence(block, REG.shape, interleaved=inter)
        means, fid, bound = T.sample_exact(sp, S.noise_preset(name), SHOTS, seed=1)
        by = {int(d): float(np.mean([means[i] for i in means if deg[i] == d])) for d in (3, 4, 5, 6)}
        res["presets"][name] = dict(by_degree=by, mean=float(np.mean(list(means.values()))), fidelity=fid, bound=bound)

    # (c) recorded stabilizers by configuration scheme and filling
    rng = np.random.default_rng(3)
    fills = {"perfect": [block]}
    for k in (1, 2, 3):
        fills[f"{k} vacanc{'y' if k == 1 else 'ies'}"] = [L.remove_random(block, k=k, rng=rng) for _ in range(20)]
    fills["peripheral 0.15"] = [L.peripheral(REG, block, 0.15, rng=rng) for _ in range(20)]
    schemes = {"Z(π/2) from actual filling, 2": lambda sp: recorded(with_rz(sp, 0), ("XZ-A", "XZ-B")),
               "Z(π/2) from planned block, 2": lambda sp: recorded(with_rz(sp, 0, block), ("XZ-A", "XZ-B")),
               "no Z(π/2), XZ and YZ, 4": lambda sp: recorded(sp, ("XZ-A", "XZ-B", "YZ-A", "YZ-B"))}
    res["recorded_odd_rule_perfect"] = recorded(with_rz(spec, 1), ("XZ-A", "XZ-B"))[0]
    res["recorded"] = {}
    for fname, occs in fills.items():
        res["recorded"][fname] = {}
        for sname, f in schemes.items():
            fr = [r / n for r, n in (f(S.cluster_sequence(o, REG.shape)) for o in occs)]
            res["recorded"][fname][sname] = float(np.mean(fr))

    # (d) fidelity and bound vs echo contrast
    cs = np.linspace(0.90, 1.0, 11)
    fid, bnd, mean = [], [], []
    for c in cs:
        m, f, b = T.sample_exact(spec, S.NoiseModel(echo_contrast=float(c)), SHOTS, seed=2)
        fid.append(f); bnd.append(b); mean.append(float(np.mean(list(m.values()))))
    res["contrast_scan"] = dict(C=cs.tolist(), fidelity=fid, bound=bnd, mean=mean)
    res["entropy_half_x_bits"] = T.entropy_bits(spec, [i for i in range(REG.n) if REG.coords(i)[0] < 3])

    fig, ax = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.2), gridspec_kw=dict(width_ratios=[1.6, 1, 1.1, 1]))
    # (a) letters in the three planes, side by side
    a = ax[0]
    G = 1.2                                           # site pitch in the drawing
    o = [(REG.shape[k] - SIZE[k]) // 2 for k in range(3)]
    for z in range(SIZE[2]):
        for y in range(SIZE[1]):
            for x in range(SIZE[0]):
                i = REG.index(o[0] + x, o[1] + y, o[2] + z)
                letter = T.pauli_letter(stabs[i], i)
                sign = "−" if np.real(stabs[i].sign) < 0 else ""
                px, py = G * (x + z * (SIZE[0] + 0.7)), G * (SIZE[1] - 1 - y)
                a.add_patch(plt.Circle((px, py), 0.5, color=COLORS["red"] if letter == "X" else COLORS["teal"], lw=0))
                a.text(px, py + 0.02, sign + letter, ha="center", va="center", color="white", fontsize=4.4)
        a.text(G * (z * (SIZE[0] + 0.7) + (SIZE[0] - 1) / 2), G * SIZE[1] - 0.4, f"z = {z}", ha="center", fontsize=5.5)
    a.set_xlim(-0.7, G * (SIZE[2] * (SIZE[0] + 0.7) - 0.7) - 0.6)
    a.set_ylim(-0.7, G * SIZE[1] + 0.1)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("a  letter of $S_i$: X odd, Y even degree", loc="left", fontsize=6.5)

    b = ax[1]
    cols = [COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["ochre"]]
    marks = ["o", "s", "^", "D"]
    for (name, _), c, mk in zip(presets, cols, marks):
        by = res["presets"][name]["by_degree"]
        b.plot(list(by), list(by.values()), marker=mk, color=c, ms=3, label=name)
    b.set_xlabel("collided neighbours")
    b.set_ylabel("⟨S$_i$⟩")
    b.set_xticks([3, 4, 5, 6])
    b.set_ylim(0.78, 1.0)
    b.legend(fontsize=4.3, loc="upper center", ncols=2, handlelength=1.5, columnspacing=0.8)
    b.set_title("b", loc="left", fontsize=7)

    c = ax[2]
    names = list(fills)
    w = 0.26
    for k, (sname, col) in enumerate(zip(schemes, [COLORS["blue"], COLORS["grey"], COLORS["ochre"]])):
        c.bar(np.arange(len(names)) + (k - 1) * w, [res["recorded"][f][sname] for f in names], w, color=col, label=sname)
    c.set_xticks(np.arange(len(names)), ["perfect", "1 vac.", "2 vac.", "3 vac.", "periph."], rotation=30, fontsize=5.5)
    c.set_ylabel("fraction of $S_i$ recorded")
    c.set_ylim(0, 1.5)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.legend(fontsize=4.3, loc="upper center", handlelength=1.2, title="settings", title_fontsize=4.3)
    c.set_title("c", loc="left", fontsize=7)

    d = ax[3]
    d.plot(cs, fid, color=COLORS["blue"], marker="o", ms=2.5, label="fidelity")
    d.plot(cs, np.clip(bnd, 0, None), color=COLORS["red"], ls="--", label="stabilizer bound (≥ 0)")
    d.plot(cs, mean, color=COLORS["teal"], marker="s", ms=2.5, label="mean ⟨S$_i$⟩")
    d.set_xlabel("echo contrast C per step")
    d.set_ylabel("value")
    d.set_ylim(0, 1.02)
    d.legend(fontsize=4.3, loc="center left")
    d.set_title("d", loc="left", fontsize=7)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/cluster_stabilizers.png", bbox_inches="tight", pad_inches=0.03)
    Path("data/cluster_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
