"""Model potential tables using a measured ER(theta) profile (leg 1, away).

Usage: python scripts/build_profile_tables.py ER_CSV HANDEDNESS STATE [STATE ...]
  STATE as F,M e.g. 3,-1
"""

import sys
from pathlib import Path

import numpy as np

from lattice_sim import cache
from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential
from lattice_sim.motion.transport1d import PotentialTable, grid_for_direction


def main(path, handed, states):
    profile = ExtinctionProfile.from_rows(read_rampol_er(path), 1, "away")
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    model = StatePotential(LatticePair(extinction_ratio=profile, handedness=handed), b)
    for state in states:
        sign = 1 if state[1] > 0 or state[0] == 4 else -1
        table = PotentialTable.from_model(model, state, np.linspace(0, np.pi, 2881), grid_for_direction(sign))
        tag = f"x_ERmeas133_h{handed:+d}_10kHz_F{state[0]}m{state[1]:+d}"
        # the name says ERmeas133 whatever CSV was given: record which one (score_suite checks it)
        table.save(f"data/{tag}.npz", inputs=dict(er_csv=Path(path).name, er_csv_digest=cache.file_digest(path),
                                                  handedness=handed, b=b, state=list(state)))
        print("built", tag, flush=True)


if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parents[1])  # paths below are relative to the repo root
    states = [tuple(int(v) for v in s.split(",")) for s in sys.argv[3:]]
    main(sys.argv[1], int(sys.argv[2]), states)
