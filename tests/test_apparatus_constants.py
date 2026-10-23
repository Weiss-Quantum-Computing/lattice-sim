"""params/apparatus.yaml is the reference; the modules carry the values as literals.
This keeps the main ones in sync (CLAUDE.md, Layout)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.constants import LATTICE_WAVELENGTH_M
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.gates import chain, collision
from lattice_sim.spectra import beams, grid, sidebands
from lattice_sim.spin import pair_physics

PARAMS = yaml.safe_load((Path(__file__).resolve().parents[1] / "params" / "apparatus.yaml").read_text(encoding="utf-8"))


def test_lattice_constants_match_the_parameter_file():
    lat = PARAMS["lattice"]
    assert LATTICE_WAVELENGTH_M * 1e9 == pytest.approx(lat["wavelength_nm"]["value"], abs=1e-6)
    spacing_um = lat["spacing_um"]["value"]["x"]                    # 4.837, rounded in the file
    for value in (LatticePair().spacing_m, collision.CollisionModel.__dataclass_fields__["spacing_m"].default,
                  chain.LATTICE_SPACING_M, pair_physics.A_LAT, sidebands.AxisConfig("x", 1.0).spacing_m,
                  grid.NOMINAL_SPACING_M):
        assert value * 1e6 == pytest.approx(spacing_um, abs=5e-4)
    waist_um = lat["waist_um"]["value"]
    assert grid.WAIST_M * 1e6 == waist_um and beams.ArmGeometry().waist_m * 1e6 == waist_um
    f = lat["trap_frequency_full_depth_kHz"]["value"]
    nominal = {c.axis: c.trap_frequency_hz for c in sidebands.nominal_site().axes}
    assert [nominal[a] / 1e3 for a in "xyz"] == pytest.approx([f["x"], f["y"], f["z"]])


def test_magic_field_matches_the_parameter_file():
    b = next(v for section in PARAMS.values() if isinstance(section, dict)
             for k, v in section.items() if k == "quantization_field_G")["value"]
    assert magic_field() == pytest.approx(b, abs=1e-5)
    assert np.linalg.norm(sidebands.nominal_site().b_lab_gauss) == pytest.approx(b, abs=1e-5)
