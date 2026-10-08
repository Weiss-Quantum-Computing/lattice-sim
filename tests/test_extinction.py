"""Tests for measured extinction-ratio profiles."""

import numpy as np
import pytest

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

CSV = """# extinction ratio along the ramp - scan test
method,leg,direction,segment,t_ms,rotation_deg,analyzer_deg,rate_deg_per_ms,er,er_sigma_lo,er_sigma_hi,lower_bound,imin_mV,imin_sigma_mV,imax_V,n,note
dip,1,static,rest,-0.3,0.04,0,0,3000,10,10,0,0,0,5,1,
dip,1,away,up 1,10.8,-45,0,0,4000,500,700,0,0,0,5,1,
dip,1,away,up 1,11.8,-90,0,0,30000,,,1,0,0,5,1,
dip,1,away,up 1,13.2,-135,0,0,1300,25,27,0,0,0,5,1,
dip,1,static,hold 1,14.8,-180,0,0,1500,20,20,0,0,0,5,1,
"""


def test_profile_reads_and_interpolates(tmp_path):
    path = tmp_path / "scan_extinction.csv"
    path.write_text(CSV)
    p = ExtinctionProfile.from_rows(read_rampol_er(path), 1, "away")
    assert np.rad2deg(p.theta) == pytest.approx([0.04, 45, 90, 135, 180])
    assert p(np.pi / 2) == pytest.approx(30000)
    assert p.lower.tolist() == [False, False, True, False, False]
    # log-linear between 45 and 90 deg
    assert p(np.deg2rad(67.5)) == pytest.approx(np.sqrt(4000 * 30000), rel=1e-6)


def test_callable_extinction_equals_constant():
    b = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    x = np.linspace(-2e-6, 0.5e-6, 101)
    v_num = StatePotential(LatticePair(extinction_ratio=500.0), b).potential((3, -3), x, 1.2)
    v_fun = StatePotential(LatticePair(extinction_ratio=lambda th: 500.0), b).potential((3, -3), x, 1.2)
    assert np.allclose(v_num, v_fun)
