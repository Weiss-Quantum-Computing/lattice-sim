"""Molasses Monte Carlo scans (lattice_sim.molasses).

    python scripts/molasses_scan.py intensity   # T, scattering, tau_E vs intensity, several detunings, full depth
    python scripts/molasses_scan.py field       # T vs residual field along x, y, z
    python scripts/molasses_scan.py li          # Li (2009) conditions: 15 kHz, -45.5 MHz
    python scripts/molasses_scan.py salomon     # free space, orthogonal lin-perp-lin (Salomon 1990)

Each point is appended to data/molasses/<name>.jsonl as soon as it finishes
(restartable: points already present are skipped). Runtime: ~0.5-4 min per
point on 2 cores, proportional to intensity/|detuning| and to the atom count.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lattice_sim.molasses import levels as L  # noqa: E402
from lattice_sim.molasses import model as M  # noqa: E402

OUT = REPO / "data" / "molasses"
DETUNINGS = (-51.0, -40.0, -30.0, -20.0, -15.0, -10.0)
INTENSITIES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
FIELD_MG = (0.0, 50.0, 100.0, 200.0, 400.0, 800.0)
FIELD_INTENSITIES = (0.2, 0.7, 2.0)
FIELD_DETUNINGS = (-51.0, -30.0, -20.0, -10.0)
N_ATOMS = 192


def burn(intensity):
    """Burn-in (ms): longer at low intensity, where optical pumping is slow compared with the trap period."""
    return min(3.0, 0.8 + 0.1 / intensity)


def key(d):
    return (round(d["intensity"], 4), round(d["detuning"], 3), tuple(round(x, 3) for x in d["b_mg"]),
            round(d["depth_scale"], 4), tuple(round(x, 3) for x in d["trap_khz"]), d["geometry"])


def run_points(name, cfgs):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(key(json.loads(line)))
    for cfg in cfgs:
        k = key(dict(intensity=cfg.intensity, detuning=cfg.detuning, b_mg=cfg.b_mg, depth_scale=cfg.depth_scale,
                     trap_khz=cfg.trap_khz, geometry=cfg.geometry))
        if k in done:
            continue
        t0 = time.time()
        d = M.summary(M.run(cfg))
        d["runtime_s"] = time.time() - t0
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d) + "\n")
        print(name, k[:3], "T", np.round(d["t_uk"], 2), "R", round(d["scatter_rate"]), f"{d['runtime_s']:.0f} s",
              flush=True)


def load(name):
    path = OUT / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv):
    what = argv[1] if len(argv) > 1 else "intensity"
    if what == "intensity":
        cfgs = [M.Config(intensity=i, detuning=d, n_atoms=N_ATOMS, t_total_ms=burn(i) + 1.7, t_burn_ms=burn(i),
                         seed=11) for d in DETUNINGS for i in INTENSITIES]
        cfgs = [c for c in cfgs if M.total_saturation(c.intensity, c.detuning) <= M.MAX_SATURATION]
        # cheapest first
        cfgs.sort(key=lambda c: c.intensity / abs(c.detuning))
        run_points("intensity", cfgs)
    elif what == "field":
        pts = [(d, i) for d in FIELD_DETUNINGS for i in FIELD_INTENSITIES
               if M.total_saturation(i, d) <= M.MAX_SATURATION]
        cfgs = []
        for d, i in pts:
            for ax in range(3):
                for b in FIELD_MG:
                    if b == 0 and ax > 0:
                        continue
                    bv = [0.0, 0.0, 0.0]
                    bv[ax] = b
                    cfgs.append(M.Config(intensity=i, detuning=d, b_mg=tuple(bv), n_atoms=64,
                                         t_total_ms=burn(i) + 1.2, t_burn_ms=burn(i), seed=13))
        cfgs.sort(key=lambda c: c.intensity / abs(c.detuning))
        run_points("field", cfgs)
    elif what == "li":
        cfgs = [M.Config(intensity=i, detuning=-45.5, trap_khz=(15.0, 15.0, 15.0), n_atoms=N_ATOMS,
                         t_total_ms=2.5, t_burn_ms=0.8, seed=17) for i in (0.1, 0.2, 0.35, 0.7, 1.4, 2.8)]
        run_points("li", cfgs)
    elif what == "salomon":
        cfgs = []
        ng = 10                                   # delta = -10 Gamma, light-shift parameter x = Omega^2/(|delta| Gamma)
        d = -ng * L.GAMMA_D2 / 2 / np.pi / 1e6
        for x in (0.05, 0.1, 0.2):
            cfgs.append(M.Config(intensity=2 * L.ISAT_CYCLING_MW_CM2 * ng * x, detuning=d, geometry="orthogonal",
                                 lattice=False, n_atoms=384, t_total_ms=2.5, t_burn_ms=1.0, t_init_uk=10, seed=5))
        run_points("salomon", cfgs)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
