"""Assemble the interactive projection-cooling explorer: inline the exported
model data (data/psc_explorer.json) and the engine (tools/psc_engine.js) into
tools/psc_explorer_template.html.

Writes tools/psc_explorer.html (git-ignored, ~3 MB, self-contained).
Usage: python scripts/psc_build_explorer.py
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

if __name__ == "__main__":
    tpl = Path("tools/psc_explorer_template.html").read_text(encoding="utf-8")
    data = Path("data/psc_explorer.json").read_text(encoding="utf-8").replace("</", "<\\/")
    engine = Path("tools/psc_engine.js").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/", data).replace("/*__ENGINE__*/", engine)
    out = Path("tools/psc_explorer.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
