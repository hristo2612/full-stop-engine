"""Views for checking a choreography: the street map (prototype 6's streetmap.py) and labelled contact sheets."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..look import muto
from .choreo import Choreo

KICK, SNARE, OTHER = (60, 60, 255), (255, 230, 60), (235, 235, 235)  # BGR


def _font(size: int):
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hit_colour(ch: Choreo, t: float):
    kicks = ch.grid.hit_times("kick")
    snares = ch.grid.hit_times("snare")
    if any(abs(t - k) < 0.03 for k in kicks):
        return KICK
    if any(abs(t - s) < 0.03 for s in snares):
        return SNARE
    return OTHER


def street_map(ch: Choreo, out: Path, scale: int = 4, row_px: int = 8000) -> Path:
    """The whole street with the obstacles (yellow boxes, green contacts) and every painted pose as a dot at its
    anchor, coloured by the hit it lands on (red kick, cyan snare, white other photos), with the moves labelled.
    With a world: the street and the world in two dimensions (y marks every 500 px, the levels' ground lines)."""
    sc = ch.scene
    if sc.world is None:
        img = (sc.street.overview(scale, left=True)[..., ::-1] * 0.75).astype(np.uint8).copy()
        ox, oy = sc.street.overview_x0(scale), 0  # street x of the image's left edge (negative with prepended plates)
    else:  # the street and the world round it (2D): invalid areas dark, the ground lines of the levels cyan
        rgb, cov, ox, oy = sc.street.overview2d(scale)
        img = (rgb[..., ::-1] * 0.75).astype(np.uint8)
        img[cov < 0.5] = (22, 20, 18)
        img = img.copy()
    Hs, Ws = img.shape[:2]
    f = 1 / scale
    X = lambda v: int((v - ox) * f)
    Y = lambda v: int((v - oy) * f)
    lw = max(1, int(round(6 * f * 2)))
    for xm in range(int(np.ceil(ox / 1000)) * 1000, sc.bounds[2], 1000):  # a street x scale every 1000 px
        cv2.line(img, (X(xm), 0), (X(xm), 30 if xm % 5000 else 70), (255, 255, 255), 2)
        if xm % 5000 == 0:
            cv2.putText(img, f"x {xm}", (X(xm) + 6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if sc.world is not None:
        for ym in range(int(np.ceil(oy / 500)) * 500, sc.bounds[3], 500):
            cv2.line(img, (0, Y(ym)), (20, Y(ym)), (255, 255, 255), 2)
            cv2.putText(img, f"y {ym}", (24, Y(ym) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for lv in sc.levels:
            xs = np.arange(lv["x0"], lv["x1"], 32)
            pts = np.int32([[X(x), Y(np.interp(x, lv["px"], lv["py"]))] for x in xs])
            cv2.polylines(img, [pts], False, (255, 255, 0), lw)
            cv2.putText(img, lv["id"], (pts[0][0] + 4, pts[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    for o in sc.obstacles.values():
        x0, y0, x1, y1 = X(o.box[0]), Y(o.box[1]), X(o.box[2]), Y(o.box[3])
        cv2.rectangle(img, (x0, max(y0, 2)), (x1, min(y1, Hs - 2)), (0, 200, 255), lw)
        cv2.putText(img, o.id, (x0 + 4, max(y0, 2) + 22 if y0 < 40 else y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 255), 2)
        for c in o.contacts.values():
            if c.horizontal:
                cv2.line(img, (X(c.x0), Y(c.y)), (X(c.x1), Y(c.y)), (0, 255, 0), lw)
            elif c.vertical:
                cv2.line(img, (X(c.x), Y(c.y0)), (X(c.x), Y(c.y1)), (0, 255, 0), lw)
    for e in ch.events:
        if e.layer not in ("wall", "air"):
            continue
        t = ch.grid.t(e.q)
        cv2.circle(img, (X(e.x), Y(e.y)), max(4, int(20 * f)), hit_colour(ch, t), -1)
    for q, x, y, text in ch.labels:
        t = ch.grid.t(q)
        cv2.putText(img, f"{text} {ch.grid.addr(q)}", (X(x) - 20, Y(y) + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, hit_colour(ch, t), 2)
    n = max(1, math.ceil(Ws / max(1, row_px // scale)))
    rw = math.ceil(Ws / n)  # rows of equal length
    rows = [img[:, i:i + rw] for i in range(0, Ws, rw)]
    rows = [np.pad(r, ((0, 8), (0, rw - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def video_frame(video: Path, t_song: float, t0: float, tmp: Path, name: str) -> Image.Image:
    """The exact frame of a video (that starts at song time t0) shown at song time t_song."""
    j = max(math.ceil((t_song - t0) * muto.FPS - 1e-6), 0)
    f = tmp / f"{name}.png"
    # seek to the middle of frame j (accurate seek: decodes from the keyframe before it)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{(j + 0.5) / muto.FPS:.4f}", "-i", str(video),
                    "-frames:v", "1", str(f)], check=True)
    return Image.open(f).convert("RGB")


def contact_sheet(shots: list[tuple[float, str]], video: Path, t0: float, out: Path, against: Path | None = None,
                  against_t0: float | None = None, labels: tuple[str, str] = ("new", "before"),
                  cols: int | None = None, tile_w: int = 640) -> Path:
    """Exact frames at song times, labelled (time and move). With `against`, each shot is a pair: this video on the
    left, the other one on the right (for comparing versions frame by frame)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout
    vw, vh = [int(v) for v in probe.strip().split(",")[:2]]
    tw = tile_w if vw >= vh else int(tile_w * 0.5)
    th = int(round(tw * vh / vw))
    pair = against is not None
    cell_w = tw * (2 if pair else 1) + (6 if pair else 0)
    cols = cols or (2 if pair else (4 if vw >= vh else 8))
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols - 1) * 6, rows * (th + 6)), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    font = _font(max(14, tw // 30))
    small = _font(max(12, tw // 40))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, (t, label) in enumerate(shots):
            x, y = (k % cols) * (cell_w + 6), (k // cols) * (th + 6)
            im = video_frame(video, t, t0, tmp, f"a{k}").resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (x, y))
            if pair:
                im2 = video_frame(against, t, against_t0 if against_t0 is not None else t0, tmp, f"b{k}")
                sheet.paste(im2.resize((tw, th), Image.LANCZOS), (x + tw + 6, y))
                for xo, lab in ((x, labels[0]), (x + tw + 6, labels[1])):
                    d.rectangle([xo + 6, y + th - 30, xo + 14 + d.textlength(lab, font=small), y + th - 8], fill=(0, 0, 0))
                    d.text((xo + 10, y + th - 28), lab, fill=(255, 210, 90), font=small)
            j = math.ceil((t - t0) * muto.FPS - 1e-6)
            txt = f"{t0 + j / muto.FPS:.2f}s  {label}"
            d.rectangle([x + 6, y + 6, x + 14 + d.textlength(txt, font=font), y + 12 + font.size], fill=(0, 0, 0))
            d.text((x + 10, y + 8), txt, fill=(255, 255, 255), font=font)
    sheet.save(out, quality=90)
    return out


def default_shots(ch: Choreo, t0: float, t1: float, n: int = 16) -> list[tuple[float, str]]:
    """Shots for a contact sheet: the moves' key moments in the range (each just after its photo), topped up with
    evenly spaced times."""
    shots = []
    for q, _, _, text in sorted(ch.labels):
        t = ch.grid.t(q)
        if t0 <= t < t1:
            shots.append((t + 0.005, f"{ch.grid.addr(q)} {text}"))
    if len(shots) > n:
        idx = np.linspace(0, len(shots) - 1, n).round().astype(int)
        shots = [shots[i] for i in idx]
    gap = (t1 - t0) / (2 * n)
    for t in np.linspace(t0 + 0.05, t1 - 0.05, 4 * n):
        if len(shots) >= n:
            break
        if all(abs(t - s) > gap for s, _ in shots):
            shots.append((float(t), ch.grid.addr(ch.grid.q(float(t))).split("+")[0]))
    return sorted(shots)[:n]


# ---------------------------------------------------------------------------------------------- method comparisons

def grid_video(videos: list[Path], labels: list[tuple[str, str]], out: Path, cols: int = 3,
               cell: tuple[int, int] = (640, 360), note: list[str] | None = None, audio: Path | None = None,
               loop: bool = False) -> Path:
    """Renders of the same choreography side by side, frame for frame (all start at the same song time), each cell
    labelled (title, subtitle); a free cell shows `note`. The song comes from `audio` (default: the first video).
    loop: shorter videos start again until the longest one ends."""
    caps = [cv2.VideoCapture(str(v)) for v in videos]
    n = len(videos) + (1 if note else 0)
    rows = math.ceil(n / cols)
    W, H = cell
    font, small = _font(max(16, H // 16)), _font(max(13, H // 22))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        k = 0
        while True:
            frames, ended = [], []
            for c in caps:
                ok, f = c.read()
                frames.append(f if ok else None)
                ended.append(not ok)
            if all(ended):
                break
            if loop:
                for i, c in enumerate(caps):
                    if ended[i]:
                        c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, f = c.read()
                        frames[i] = f if ok else None
            sheet = Image.new("RGB", (cols * W, rows * H), (12, 12, 14))
            d = ImageDraw.Draw(sheet)
            for i, f in enumerate(frames):
                x, y = (i % cols) * W, (i // cols) * H
                if f is not None:
                    sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA),
                                                             cv2.COLOR_BGR2RGB)), (x, y))
                t1, t2 = labels[i]
                tw = max(d.textlength(t1, font=font), d.textlength(t2, font=small)) + 16
                d.rectangle([x + 6, y + 6, x + 6 + tw, y + 12 + font.size + small.size + 6], fill=(0, 0, 0))
                d.text((x + 12, y + 9), t1, fill=(255, 220, 90), font=font)
                d.text((x + 12, y + 14 + font.size), t2, fill=(235, 235, 235), font=small)
            if note:
                i = len(videos)
                x, y = (i % cols) * W, (i // cols) * H
                for j, line in enumerate(note):
                    d.text((x + 18, y + 18 + j * (small.size + 8)), line, fill=(220, 220, 220), font=small)
            sheet.save(tmp / f"{k:05d}.jpg", quality=92)
            k += 1
        for c in caps:
            c.release()
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(muto.FPS), "-i", str(tmp / "%05d.jpg"),
                        "-i", str(audio or videos[0]), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    return out


def photo_sheet(rows: list[tuple[str, list[tuple[Path, str]]]], out: Path, tile: tuple[int, int] = (320, 180),
                crop: float = 0.62, align: list[int] | None = None) -> Path:
    """Photos in rows (one row per method: a label and its photos with captions), each photo cropped round its
    centre (the camera keeps him there) to `crop` of the frame; align: the column each row's key photo goes in
    (the rows then line up on their key photos)."""
    tw, th = tile
    lab_w = 360
    offs = [0] * len(rows) if align is None else [max(align) - a for a in align]
    ncol = max(o + len(r[1]) for o, r in zip(offs, rows))
    sheet = Image.new("RGB", (lab_w + ncol * (tw + 4), len(rows) * (th + 26)), (14, 14, 16))
    d = ImageDraw.Draw(sheet)
    font, small = _font(16), _font(13)
    for ri, ((label, photos), o) in enumerate(zip(rows, offs)):
        y = ri * (th + 26)
        for li, line in enumerate(label.split("\n")):
            d.text((10, y + 10 + li * 20), line, fill=(255, 220, 90) if li == 0 else (230, 230, 230),
                   font=font if li == 0 else small)
        for ci, (f, cap) in enumerate(photos):
            im = Image.open(f)
            w, h = im.size
            cw, chh = int(w * crop), int(h * crop)
            im = im.crop(((w - cw) // 2, (h - chh) // 2, (w + cw) // 2, (h + chh) // 2)).resize((tw, th), Image.LANCZOS)
            x = lab_w + (o + ci) * (tw + 4)
            sheet.paste(im, (x, y))
            d.text((x + 4, y + th + 4), cap, fill=(255, 120, 90) if "plant" in cap or "key" in cap else (200, 200, 200),
                   font=small)
    sheet.save(out, quality=90)
    return out
