"""Assemble the cluster-state explorer: inline the engine (tools/cluster_engine.js)
and the noise presets (lattice_sim.cluster.sequence.NOISE_PRESETS) into
tools/cluster_explorer_template.html.

Writes tools/cluster_explorer.html (self-contained, committed; tests/test_cluster.py
checks that it is up to date). Usage: python scripts/cluster_build_explorer.py
"""

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

from lattice_sim.cluster.sequence import NOISE_PRESETS  # noqa: E402

OUT = Path("tools/cluster_explorer.html")


def build():
    tpl = Path("tools/cluster_explorer_template.html").read_text(encoding="utf-8")
    engine = Path("tools/cluster_engine.js").read_text(encoding="utf-8")
    presets = {k: {"noise": m.to_dict(), "source": src} for k, (m, src) in NOISE_PRESETS.items()}
    data = json.dumps(presets, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace("/*__ENGINE__*/", engine).replace("/*__PRESETS__*/", data)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
