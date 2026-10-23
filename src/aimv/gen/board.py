"""Choice boards: generate a labelled set of frames and lay them out for the director to pick from.

projects/<slug>/boards/<name>/board.json
  {"title": "...", "question": "...", "defaults": {"provider": "openai", "size": "2048x1152", "quality": "high"},
   "groups": [{"id": "A", "title": "Red Light", "note": "...",
               "frames": [{"id": "A1", "caption": "0:00 verse 1", "prompt": "..."}]}]}

`aimv board <project> <name>` generates every frame that has no image yet (never pays twice), then writes
index.html (the board) and contact.jpg (one image with every label, for chat).
"""

from __future__ import annotations

import html
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..project import Project
from . import hf, images, keys, ledger


def _frames(spec: dict):
    for g in spec["groups"]:
        for f in g["frames"]:
            yield g, f


def estimate(spec: dict, todo: list, provider: str = "openai") -> float:
    d = spec.get("defaults", {})
    if provider == "higgsfield":  # Higgsfield renders "2k" 16:9 at 2688x1520, priced by hf.image_usd
        return sum(hf.image_usd({"quality": f.get("quality", d.get("quality", "high")), "resolution": "2k",
                                 "aspect_ratio": f.get("aspect_ratio", d.get("aspect_ratio", "16:9")),
                                 "image_urls": f.get("refs", [])}) for _, f in todo)
    return sum(images.estimate_openai(f.get("size", d.get("size", "2048x1152")), f.get("quality", d.get("quality", "high")),
                                      1, len(f.get("refs", []))) for _, f in todo)


def generate(p: Project, name: str, workers: int = 4, dry_run: bool = False, log=print) -> Path:
    bdir = p.root / "boards" / name
    spec = json.loads((bdir / "board.json").read_text())
    d = spec.get("defaults", {})
    todo = [(g, f) for g, f in _frames(spec) if not (bdir / f"{f['id']}.png").exists()]
    provider = d.get("provider", "auto")
    if provider == "auto":
        provider = "openai" if keys.get("OPENAI_API_KEY") else "higgsfield"
    est = estimate(spec, todo, provider)
    t = ledger.totals(p)
    cap, _ = ledger.caps(p)
    log(f"{len(todo)} frames to generate, about ${est:.2f} (spent so far ${t['total_usd']:.2f} of ${cap:.2f} cap)")
    if est + t["total_usd"] > cap:
        raise ledger.BudgetExceeded(f"This board (${est:.2f}) would pass the project cap ${cap:.2f}; raise budget.cap_usd")
    if dry_run or not todo:
        return render(p, name)

    log(f"provider: {provider}")

    def one(g, f):
        refs = [p.root / r for r in f.get("refs", [])]
        if provider == "higgsfield":
            return _hf_frame(p, spec, name, f, refs, bdir)
        return images.openai_image(
            p, f["prompt"], bdir / f"{f['id']}.png", phase=spec.get("phase", "look"),
            model=f.get("model", d.get("model", "gpt-image-2.5-sunburst")),
            size=f.get("size", d.get("size", "2048x1152")), quality=f.get("quality", d.get("quality", "high")),
            refs=refs, tag=f"{name}/{f['id']}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, g, f): f["id"] for g, f in todo}
        for fu in as_completed(futs):
            try:
                fu.result()
                log(f"  {futs[fu]} ok")
            except Exception as e:  # keep going; a failed frame can be retried by re-running
                log(f"  {futs[fu]} FAILED: {e}")
    return render(p, name)


HF_IMAGE = {"gpt-image-2.5-sunburst": "marketing-studio/image/sunburst",
            "gpt-image-2.5-flare": "marketing-studio/image/flare"}
SIZE_TO_RES = {"1024": "1k", "1536": "1k", "2048": "2k", "2560": "2k", "3840": "4k"}


def _hf_frame(p: Project, spec: dict, name: str, f: dict, refs: list[Path], bdir: Path) -> Path:
    """One frame through Higgsfield (same GPT Image 2.5 models, billed to the Higgsfield wallet)."""
    from PIL import Image

    d = spec.get("defaults", {})
    model = f.get("model", d.get("model", "gpt-image-2.5-sunburst"))
    size = f.get("size", d.get("size", "2048x1152"))
    args = {"prompt": f["prompt"], "aspect_ratio": f.get("aspect_ratio", d.get("aspect_ratio", "16:9")),
            "resolution": SIZE_TO_RES.get(size.split("x")[0], "2k"), "quality": f.get("quality", d.get("quality", "high"))}
    if refs:
        args["image_urls"] = [hf.upload(r) for r in refs]
    files = hf.run(p, HF_IMAGE[model], args, bdir / f"{f['id']}.raw", phase=spec.get("phase", "look"),
                   tag=f"{name}/{f['id']}", log=lambda m: None)
    out = bdir / f"{f['id']}.png"
    Image.open(files[0]).save(out)
    files[0].unlink()
    return out


def render(p: Project, name: str) -> Path:
    bdir = p.root / "boards" / name
    spec = json.loads((bdir / "board.json").read_text())
    groups_html = []
    for g in spec["groups"]:
        cells = []
        for f in g["frames"]:
            img = bdir / f"{f['id']}.png"
            pic = (f'<img src="{f["id"]}.png" loading="lazy" alt="{html.escape(f["id"])}">' if img.exists()
                   else '<div class="missing">not generated yet</div>')
            cells.append(f'<figure><div class="lab">{html.escape(f["id"])}</div>{pic}'
                         f'<figcaption>{html.escape(f.get("caption", ""))}</figcaption></figure>')
        note = f'<p class="note">{html.escape(g["note"])}</p>' if g.get("note") else ""
        groups_html.append(f'<section><h2><span>{html.escape(g["id"])}</span> {html.escape(g["title"])}</h2>{note}'
                           f'<div class="row">{"".join(cells)}</div></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(spec['title'])}</title>
<style>
:root {{ --bg:#f4f3ef; --ink:#151517; --muted:#6c6b70; --card:#fff; --line:#dddbd5; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f0f12; --ink:#ececf0; --muted:#9a99a3; --card:#18181d; --line:#2a2a31; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1600px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 6px; }}
.q {{ color:var(--muted); margin:0 0 22px; max-width:900px; }}
section {{ margin:0 0 30px; }}
h2 {{ font-size:17px; margin:0 0 4px; display:flex; gap:10px; align-items:center; }}
h2 span {{ background:var(--ink); color:var(--bg); border-radius:6px; padding:1px 9px; font-size:14px; }}
.note {{ color:var(--muted); margin:0 0 10px; max-width:900px; }}
.row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; position:relative; }}
figure img {{ display:block; width:100%; aspect-ratio:16/9; object-fit:cover; cursor:zoom-in; }}
.lab {{ position:absolute; top:8px; left:8px; background:rgba(0,0,0,.72); color:#fff; font:600 13px ui-monospace,Menlo,monospace; padding:2px 8px; border-radius:5px; }}
figcaption {{ padding:8px 10px; color:var(--muted); font-size:13px; }}
.missing {{ aspect-ratio:16/9; display:grid; place-items:center; color:var(--muted); }}
.safe {{ pointer-events:none; position:absolute; top:0; bottom:38px; left:34.18%; right:34.18%; border-left:1px dashed rgba(255,255,255,.35); border-right:1px dashed rgba(255,255,255,.35); display:none; }}
body.crop .safe {{ display:block; }}
#zoom {{ position:fixed; inset:0; background:rgba(0,0,0,.92); display:none; place-items:center; z-index:9; cursor:zoom-out; }}
#zoom img {{ max-width:96vw; max-height:94vh; }}
label {{ color:var(--muted); font-size:13px; }}
</style></head><body><main>
<h1>{html.escape(spec['title'])}</h1><p class="q">{html.escape(spec.get('question', ''))}</p>
<label><input type="checkbox" onchange="document.body.classList.toggle('crop',this.checked)"> show the 9:16 crop band</label>
{"".join(groups_html)}
</main><div id="zoom" onclick="this.style.display='none'"><img alt=""></div>
<script>
document.querySelectorAll('figure').forEach(f => {{ const s=document.createElement('div'); s.className='safe'; f.appendChild(s); }});
document.querySelectorAll('figure img').forEach(i => i.onclick = () => {{ const z=document.getElementById('zoom'); z.querySelector('img').src=i.src; z.style.display='grid'; }});
</script></body></html>"""
    (bdir / "index.html").write_text(page)
    _contact(bdir, spec)
    return bdir / "index.html"


def _contact(bdir: Path, spec: dict) -> None:
    """One labelled jpg of the whole board (rows = groups, a long group wrapped every 12 frames, so the sheet stays
    inside JPEG's 65,535 px limit)."""
    from PIL import Image, ImageDraw, ImageFont

    wrap = 12
    rows = [g["frames"][i:i + wrap] for g in spec["groups"] for i in range(0, max(len(g["frames"]), 1), wrap)]
    cols = max(len(r) for r in rows)
    tw, th = 640, 360
    sheet = Image.new("RGB", (cols * tw, len(rows) * th), (20, 20, 22))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for ri, row in enumerate(rows):
        for ci, f in enumerate(row):
            img = bdir / f"{f['id']}.png"
            if img.exists():
                im = Image.open(img).convert("RGB")
                im.thumbnail((tw, th))
                sheet.paste(im, (ci * tw, ri * th))
            draw.rectangle([ci * tw + 8, ri * th + 8, ci * tw + 78, ri * th + 46], fill=(0, 0, 0))
            draw.text((ci * tw + 16, ri * th + 11), f["id"], fill=(255, 255, 255), font=font)
    sheet.save(bdir / "contact.jpg", quality=88)
