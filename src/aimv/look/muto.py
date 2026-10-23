"""The "stitched photographs" look (after Blu's MUTO): frames held on the song's grid, nothing interpolated.

Every step on the grid is one "photograph": it gets its own small camera nudge, exposure and white-balance
flicker and fresh grain. During a `stop` span the image content can freeze (hold) while the photos keep coming,
so the world still ticks while the figure stands still.

Input is either a video (sampled at each step's song time, mapped into the clip) or a folder of numbered frames
(one per step). Output is a 24 fps H.264 file with the matching slice of the song.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from ..project import Project

OUT_W, OUT_H, FPS = 1920, 1080, 24


def grid(p: Project, t0: float, t1: float, division: int = 4) -> list[float]:
    """Step times in [t0, t1): `division` steps per beat (4 = 16th notes, 2 = 8ths), linear between beats."""
    beats = [b["t"] for b in json.loads((p.bd / "beats.json").read_text())["beats"]]
    out = []
    for a, b in zip(beats, beats[1:]):
        for k in range(division):
            t = a + (b - a) * k / division
            if t0 <= t < t1:
                out.append(t)
    return out


def stops(p: Project) -> list[tuple[float, float]]:
    ev = json.loads((p.bd / "hits.json").read_text())["events"]
    return [(s["t0"], s["t1"]) for s in ev.get("stop", [])]


def _duration(video: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def _video_frame(video: Path, t: float, tmp: Path) -> Image.Image:
    t = min(max(t, 0.0), _duration(video) - 0.06)  # before the clip: first frame; after it: last frame
    f = tmp / f"v{t:.4f}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.4f}", "-i", str(video), "-frames:v", "1", str(f)],
                   check=True)
    return Image.open(f).convert("RGB")


def _photo(img: Image.Image, rng: np.random.Generator, cam: tuple[float, float, float], strength: float,
           size: tuple[int, int] | None = None) -> Image.Image:
    """One 'photograph': camera nudge (pan x, pan y, zoom) + jitter, exposure/white-balance flicker, grain.
    size: output (w, h), default 1920x1080 (aimv paint passes 1080x1920 for its 9:16 cut)."""
    w, h = img.size
    zoom = 1.06 + cam[2] + rng.normal(0, 0.004 * strength)
    cw, ch = w / zoom, h / zoom
    cx = w / 2 + cam[0] * w + rng.normal(0, 0.004 * w * strength)
    cy = h / 2 + cam[1] * h + rng.normal(0, 0.003 * h * strength)
    cx = min(max(cx, cw / 2), w - cw / 2)
    cy = min(max(cy, ch / 2), h - ch / 2)
    box = (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))
    im = img.crop(box).resize(size or (OUT_W, OUT_H), Image.LANCZOS)
    im = ImageEnhance.Brightness(im).enhance(1 + rng.normal(0, 0.035 * strength))
    a = np.asarray(im).astype(np.float32)
    a *= np.array([1 + rng.normal(0, 0.012 * strength), 1.0, 1 + rng.normal(0, 0.012 * strength)])  # white balance drift
    a += rng.normal(0, 6.0 * strength, a.shape[:2])[..., None]  # grain
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def render(p: Project, out: Path, t0: float, t1: float, *, video: Path | None = None, video_t0: float = 0.0,
           frames: list[Path] | None = None, division: int = 4, hold_stops: bool = True,
           pan: tuple[float, float] = (0.0, 0.0), zoom: float = 0.0, strength: float = 1.0, seed: int = 7) -> Path:
    """Render the look.

    video:   source clip; song time t is read from clip time (t - t0 + video_t0).
    frames:  one image per step instead of a video (extra steps reuse the last frame).
    pan:     total camera travel over the window as a fraction of the frame (x, y), applied in steps.
    hold_stops: freeze the content during `stop` spans (photos keep coming)."""
    steps = grid(p, t0, t1, division)
    st = stops(p) if hold_stops else []
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="aimv-muto-") as tmp:
        tmp = Path(tmp)
        photos = []
        held_t = None
        for i, t in enumerate(steps):
            in_stop = next(((a, b) for a, b in st if a <= t < b), None)
            src_t = (in_stop[0] if in_stop else t)
            if frames is not None:
                k = min(i if not in_stop else steps.index(next(s for s in steps if s >= in_stop[0])), len(frames) - 1)
                img = Image.open(frames[k]).convert("RGB")
            else:
                if in_stop and held_t is None:
                    held_t = src_t
                if not in_stop:
                    held_t = None
                img = _video_frame(video, (held_t if in_stop else t) - t0 + video_t0, tmp)
            u = i / max(len(steps) - 1, 1)
            cam = (pan[0] * (u - 0.5), pan[1] * (u - 0.5), zoom * u)
            photos.append(_photo(img, rng, cam, strength))
        # 24 fps timeline: each output frame shows the latest photo at its song time.
        seq = tmp / "seq"
        seq.mkdir()
        n = int(round((t1 - t0) * FPS))
        for j in range(n):
            t = t0 + j / FPS
            k = max(0, np.searchsorted(steps, t, side="right") - 1)
            photos[k].save(seq / f"{j:05d}.jpg", quality=92)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(seq / "%05d.jpg"),
                        "-ss", f"{t0:.4f}", "-t", f"{t1 - t0:.4f}", "-i", str(p.bd / "audio/mix.wav"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)
    return out
