"""Image generation: OpenAI gpt-image-2.5 (Sunburst/Flare) directly. (Through Higgsfield: gen/hf.py.)

Prices (USD per image, OpenAI direct, read 2026-10-05):
              low     medium  high    xhigh   max
  1536x864    0.004   0.008   0.032   0.058   0.129
  2048x1152   0.005   0.011   0.042   0.075   0.169
Other sizes scale by pixel area from the 2048x1152 row. Reference images add input tokens (a few cents at most),
covered by a 25% margin on the estimate.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

from ..project import Project
from . import keys, ledger

OPENAI_URL = "https://api.openai.com/v1/images"
PRICE_2048 = {"low": 0.005, "medium": 0.011, "high": 0.042, "xhigh": 0.075, "max": 0.169}
MARGIN = 1.25


def estimate_openai(size: str, quality: str, n: int = 1, refs: int = 0) -> float:
    w, h = (int(v) for v in size.lower().split("x"))
    base = PRICE_2048[quality] * (w * h) / (2048 * 1152)
    return round(n * base * MARGIN + 0.003 * refs, 4)


def openai_image(p: Project, prompt: str, out: Path, *, phase: str, model: str = "gpt-image-2.5-sunburst",
                 size: str = "2048x1152", quality: str = "high", refs: list[Path] | None = None,
                 tag: str = "") -> Path:
    """One image. With `refs`, uses the edits endpoint (up to 16 reference images)."""
    key = keys.require("OPENAI_API_KEY", "OpenAI images")
    refs = refs or []
    est = estimate_openai(size, quality, 1, len(refs))
    req = {"model": model, "size": size, "quality": quality, "refs": [str(r) for r in refs], "tag": tag, "prompt": prompt}
    with ledger.spend(p, phase=phase, provider="openai", model=model, usd_estimate=est, request=req) as rec:
        hdr = {"Authorization": f"Bearer {key}"}
        t = time.time()
        if refs:
            files = [("image[]", (r.name, open(r, "rb"), "image/png" if r.suffix == ".png" else "image/jpeg")) for r in refs]
            data = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": "1", "output_format": "png"}
            r = requests.post(f"{OPENAI_URL}/edits", headers=hdr, data=data, files=files, timeout=600)
        else:
            body = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": 1, "output_format": "png"}
            r = requests.post(f"{OPENAI_URL}/generations", headers={**hdr, "Content-Type": "application/json"},
                              data=json.dumps(body), timeout=600)
        if r.status_code != 200:
            rec["status"] = "not_charged" if r.status_code in (400, 401, 403, 429) else "error"
            raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:400]}")
        res = r.json()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
        rec["files"] = [str(out.relative_to(p.root))]
        rec["usage"] = res.get("usage")
        rec["seconds"] = round(time.time() - t, 1)
    return out

