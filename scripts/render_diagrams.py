"""Render the Mermaid blocks in docs/architecture.md and docs/agent-workflow.md to SVG + PNG.

Mermaid runs inside headless Chrome (fetched from jsDelivr), and the PNG is taken by the same
browser, so text is measured and drawn with the same fonts (no clipped labels).

  python scripts/render_diagrams.py [--chrome /opt/google/chrome/chrome]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"


def chrome_bin(explicit: str | None) -> str:
    for c in (explicit, "google-chrome", "/opt/google/chrome/chrome", "chromium", "chromium-browser"):
        if c and (shutil.which(c) or Path(c).exists()):
            return shutil.which(c) or c
    raise SystemExit("Chrome/Chromium not found; pass --chrome")


def render(name: str, chrome: str, tmp: Path) -> None:
    md = (ROOT / "docs" / f"{name}.md").read_text()
    src = re.search(r"```mermaid\n(.*?)```", md, re.S).group(1)
    out = ROOT / "docs" / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.mmd").write_text(src)
    page = tmp / f"render_{name}.html"
    page.write_text(f"""<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:10px;background:#fff"><div id="out"></div><script type="module">
import mermaid from "{MERMAID}";
mermaid.initialize({{startOnLoad:false, theme:"default", fontFamily:"Arial, Helvetica, sans-serif",
  flowchart:{{htmlLabels:false, curve:"basis", padding:16, wrappingWidth:260}}}});
const {{svg}} = await mermaid.render("d", {json.dumps(src)});
document.getElementById("out").innerHTML = svg;
</script></body></html>""")
    prof = f"--user-data-dir={tmp / 'chrome-profile'}"  # never attach to the user's running Chrome
    dom = subprocess.run([chrome, "--headless=new", prof, "--disable-gpu", "--no-sandbox", "--virtual-time-budget=15000",
                          "--dump-dom", page.as_uri()], capture_output=True, text=True, timeout=120).stdout
    svg = re.sub(r"<br\s*>", "<br/>", re.search(r"(<svg[\s\S]*?</svg>)\s*</div>", dom).group(1))
    x, y, w, h = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    x, y, w, h = x - 20, y - 20, w + 40, h + 100   # mermaid's bbox is a little tight
    svg = re.sub(r'viewBox="[^"]+"', f'viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}"', svg, count=1)
    svg = re.sub(r"max-width:\s*[\d.]+px;?", f"max-width: {w:.1f}px;", svg, count=1)
    xml.dom.minidom.parseString(svg)
    (out / f"{name}.svg").write_text(svg)
    shot = tmp / f"{name}.html"
    shot.write_text('<html><body style="margin:0;background:#fff">' + svg.replace(f"max-width: {w:.1f}px;", "") + "</body></html>")
    subprocess.run([chrome, "--headless=new", prof, "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--force-device-scale-factor=2", f"--window-size={int(w)},{int(h)}",
                    f"--screenshot={out / (name + '.png')}", shot.as_uri()], capture_output=True, timeout=120)
    print(f"wrote docs/diagrams/{name}.svg and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome")
    a = ap.parse_args()
    chrome = chrome_bin(a.chrome)
    with tempfile.TemporaryDirectory() as t:
        for name in ("architecture", "agent-workflow"):
            render(name, chrome, Path(t))


if __name__ == "__main__":
    main()
