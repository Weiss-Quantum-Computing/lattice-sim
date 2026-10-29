"""Molasses Monte Carlo scans (lattice_sim.molasses).

    python scripts/molasses_scan.py intensity   # T, scattering, tau_E vs intensity, several detunings, full depth
    python scripts/molasses_scan.py lab1026     # the lab's 2026-10-26 power scan at -25.5 MHz (rescale 14-24)
    python scripts/molasses_scan.py depth       # T and hopping at 0.1, 0.25, 0.5 of full depth
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
# Lab: locked to the F=4 -> F'=4/5 crossover (-125.5 MHz from F'=5), single-pass AOM +75 (old), +90, +95,
# +100 (current), +105 MHz -> -50.5, -35.5, -30.5, -25.5, -20.5 MHz (LAB_DETUNINGS).
LAB_DETUNINGS = (-50.5, -35.5, -30.5, -25.5, -20.5)
DETUNINGS = (-51.0, -40.0, -35.5, -30.0, -25.5, -20.0, -15.0, -10.0)
INTENSITIES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
FIELD_MG = (0.0, 50.0, 100.0, 200.0, 400.0, 800.0)
FIELD_INTENSITIES = (0.2, 0.7, 1.4)    # 1.4 ~ the present 0.4-0.5 uW per beam in a 140 um waist
FIELD_DETUNINGS = (-51.0, -25.5)
LAB_1026_I24 = 1.46                     # mW/cm^2 per beam at rescale 24, request 5 (2.2 uW out of the fiber)
LAB_1026_RESCALE = (14, 16, 18, 20, 22, 24)
DEPTH_DETUNINGS = (-51.0, -25.5)
DEPTH_INTENSITIES = (0.35, 0.7, 1.4)
DEPTH_SCALES = (0.1, 0.25, 0.5)
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
    elif what == "depth":
        cfgs = [M.Config(intensity=i, detuning=d, depth_scale=f, n_atoms=N_ATOMS, t_total_ms=burn(i) + 1.7,
                         t_burn_ms=burn(i), seed=19)
                for d in DEPTH_DETUNINGS for i in DEPTH_INTENSITIES for f in DEPTH_SCALES
                if M.total_saturation(i, d) <= M.MAX_SATURATION]
        cfgs.sort(key=lambda c: c.intensity / abs(c.detuning))
        run_points("depth", cfgs)
    elif what == "lab1026":
        # The lab's 2026-10-26 hopping-sequence power scan at +100 MHz (-25.5 MHz): rescale r = 14 ... 24 with
        # request 5; at r = 24 the request gave 2.2 uW out of the fiber, ~0.45 uW per beam, 1.46 mW/cm^2 per
        # beam (140 um waist). Power taken as ~ 1/r. Total saturation up to 0.14: beyond MAX_SATURATION,
        # reported as an extrapolation of the model (summary carries the saturation).
        cfgs = [M.Config(intensity=LAB_1026_I24 * 24.0 / r, detuning=-25.5, n_atoms=128,
                         t_total_ms=burn(1.0) + 1.7, t_burn_ms=burn(1.0), seed=23) for r in LAB_1026_RESCALE]
        cfgs.sort(key=lambda c: c.intensity)
        run_points("lab1026", cfgs)
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
