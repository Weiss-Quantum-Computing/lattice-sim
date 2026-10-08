"""Shared figure style: journal-like (Nature / Physical Review) single-column
and double-column figures with inward ticks, thin axes and sans-serif type.

Usage:
    from lattice_sim.viz.style import apply, COLORS
    apply()
"""

import matplotlib as mpl

COLORS = {"blue": "#1f5fa8", "red": "#c2452d", "teal": "#2a9d8f", "ochre": "#b5872a",
          "grey": "#6b6b6b", "purple": "#7a5195", "orange": "#e07b39"}

SINGLE_COLUMN_IN = 3.4
DOUBLE_COLUMN_IN = 7.0


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.0,
    })
