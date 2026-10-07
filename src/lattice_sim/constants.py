"""Physical constants and caesium ground-state data.

Atomic data: D. A. Steck, "Cesium D Line Data", rev. 2.1.6.
Fundamental constants: CODATA 2018 (exact SI values where defined).
"""

import numpy as np

# Fundamental constants (SI)
H = 6.626_070_15e-34          # J s (exact)
HBAR = H / (2 * np.pi)
KB = 1.380_649e-23            # J/K (exact)
AMU = 1.660_539_066_60e-27    # kg
A0 = 5.291_772_109_03e-11     # m, Bohr radius
MU_B_HZ_PER_G = 1.399_624_493_61e6   # mu_B / h in Hz per gauss

# Caesium-133
MASS_CS = 2.206_946_50e-25    # kg (Steck Table 1)
I_NUC = 3.5                   # nuclear spin
G_J = 2.002_540_32            # 6S_1/2 fine-structure Lande factor (Steck Table 7)
G_I = -0.000_398_853_95       # nuclear g-factor, Steck sign convention (Table 7)
A_HFS_HZ = 2.298_157_942_5e9  # 6S_1/2 magnetic-dipole constant A/h (Steck Table 5)
NU_HFS_HZ = 9.192_631_770e9   # ground hyperfine splitting (SI second definition)

# Lattice
LATTICE_WAVELENGTH_M = 843.222e-9
