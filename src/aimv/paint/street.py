"""The street: plates stitched into one long strip, game-level edits, stored as tiles (prototype 6's make_canvas.py,
stitch.py and edit_strip.py, made streaming and extendable).

Spec: projects/<p>/paint/streets/<name>/street.json

    {"version": 1,
     "plates": ["boards/07-street/P1.png", ...],   project-relative, in street order (left to right)
     "left": ["boards/09-opening/L1.png", ...],    plates prepended on the left, nearest first (optional)
     "keep": 0.4,                                  fraction of each plate taken from the previous one (outpaint canvas)
     "seam_feather": 24, "gain_fade": 1400,        stitching (see stitch below)
     "edits": [{"cut": [5203, 6100]},              remove strip columns x0..x1 (a cross-faded join)
               {"insert": 11100, "src": [10400, 11800], "mirror": true, "stretch": 1.5},
               {"insert": -9000, "src": [-10400, -9200], "mirror": true, "stretch": 1.5}],   left of x 0
     "edit_feather": 96,                           cross-fade width of every edit join
     "edge_cover": 0.98,                           the far left end: least share of rows a plate must cover
     "tile": 2048}                                 tile width in px

Edit coordinates are in strip px (the stitched strip before edits); obstacles are in street px (after edits).

Patches: {"patch": "paint/streets/main/patches/bench.png", "window": [-9500, -7300]} pastes an edited copy of a
street window back in place (street px, after the cuts and inserts, so they never move anything). `aimv paint window`
exports the window (full resolution, full height) and its placement; an image edit adds an obstacle to it; the
patch is aligned to the street by SIFT (a similarity, so a resized or slightly shifted edit still fits), its tone
matched to the street with a low-frequency gain, and pasted only where it differs from the street (mask "diff",
the default: the changed pixels, closed, grown by `grow` px and feathered; "full": the whole window, feathered at
its border). region: [x0, y0, x1, y1] (street px) limits what it may change. build.json records each patch's fit
and the box it changed.

Build (`aimv paint street`): every plate after the first was outpainted from a canvas holding the right `keep` of
the previous plate (make_canvas), so the overlaps line up within a pixel or two but the texture is new. Per join:
SIFT + RANSAC similarity, chained into strip coordinates; a low-frequency gain match along the seam faded out over
`gain_fade` px of the new plate; a minimum-difference vertical seam inside the middle of the overlap, feathered.
The strip is written as tiles (strip/) while it is built, never held whole; edits then render the street tiles
(tiles/) piece by piece. Adding plates at the end only rewrites the tiles right of the new plate's overlap, so the
earlier tiles keep their content hash (and their scene caches).

Prepending (`left`): each left plate was outpainted from a canvas holding the left `keep` of the plate to its right
on its right (make_canvas left=True). Street x 0 stays where it is and the new street gets negative x, stored as
tiles s-0001.png, t-0001.png ... (tile k < 0 covers [k * tile, (k + 1) * tile), the leftmost one may be narrower).
Nothing right of x 0 is rewritten: the first left plate meets the first plate exactly at x 0 (gain-matched and
blended into the first plate's edge columns on the left side only), and edits left of x 0 lay their pieces out
leftward from x 0. Stitching runs in a mirrored frame (x' = -1 - x), where prepending is appending.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..util import read_json, sha256_file, write_json

TILE = 2048
OY = 4  # canvas margin above the first plate so a negative dy fits


# ------------------------------------------------------------------------------------------------ spec

def load_spec(d: Path) -> dict:
    f = d / "street.json"
    if not f.exists():
        raise SystemExit(f"No street spec: {f}")
    s = read_json(f)
    s.setdefault("keep", 0.4)
    s.setdefault("seam_feather", 24)
    s.setdefault("gain_fade", 1400)
    s.setdefault("edits", [])
    s.setdefault("edit_feather", 96)
    s.setdefault("tile", TILE)
    s.setdefault("left", [])
    if not s.get("plates"):
        raise SystemExit(f"{f}: no plates")
    for e in s["edits"]:
        _edit_side(e)  # an edit may not straddle x 0
    if s.get("world"):
        from .world import load_spec as world_spec
        world_spec(s)  # checked early (cells, ids, at, from)
    return s


def make_canvas(prev: Path, out: Path, keep: float = 0.4, left: bool = False) -> Path:
    """Outpaint canvas for the next plate: the right `keep` of the previous plate on the left, flat grey elsewhere.
    left=True (a plate prepended on the left): the left `keep` of the previous plate on the right."""
    img = cv2.imread(str(prev))
    if img is None:
        raise SystemExit(f"cannot read {prev}")
    h, w = img.shape[:2]
    k = int(round(w * keep))
    canvas = np.full_like(img, 128)
    if left:
        canvas[:, w - k:] = img[:, :k]
    else:
        canvas[:, :k] = img[:, w - k:]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def tile_name(prefix: str, k: int) -> str:
    """File name of tile k: s0003.png right of x 0, s-0001.png, s-0002.png ... left of it (prepended plates)."""
    return f"{prefix}{k:04d}.png" if k >= 0 else f"{prefix}-{-k:04d}.png"


def tile_files(d: Path, prefix: str) -> dict[int, Path]:
    """The stored tiles of one tiled image: {k: path}."""
    out = {}
    for f in d.glob(f"{prefix}*.png"):
        try:
            out[int(f.stem[len(prefix):])] = f
        except ValueError:
            continue
    return out


def _edit_side(e: dict) -> int:
    """1 for an edit right of x 0, -1 for one left of it (strip px); 0 for a patch (street px, either side)."""
    if "patch" in e:
        if not isinstance(e.get("window"), list) or len(e["window"]) != 2 or e["window"][1] <= e["window"][0]:
            raise SystemExit(f"street edits: a patch needs its window [x0, x1] (street px): {e}")
        return 0
    xs = list(e["cut"]) if "cut" in e else [e["insert"], *e["src"]]
    if all(x >= 0 for x in xs):
        return 1
    if all(x <= 0 for x in xs):
        return -1
    raise SystemExit(f"street edits: {e} straddles x 0 (edit left and right of the first plate separately)")


# ------------------------------------------------------------------------------------------------ stitching

def similarity(A: np.ndarray, B: np.ndarray, K: int) -> tuple[np.ndarray, int, float]:
    """Similarity mapping B's left overlap onto A's right overlap (in A's overlap coordinates)."""
    W = A.shape[1]
    ga = cv2.cvtColor((A[:, W - K:] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor((B[:, :K + 200] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(10000)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    if len(good) < 12:
        raise SystemExit(f"stitch: only {len(good)} matches in the overlap (was the plate outpainted from the "
                         f"previous one with keep={K / W:.2f}?)")
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    cv2.setRNGSeed(12345)
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=5000)
    res = np.linalg.norm(pb @ M[:, :2].T + M[:, 2] - pa, axis=1)[inl.ravel() > 0]
    return M, int(inl.sum()), float(np.median(res))


def _read_plate(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"cannot read plate {p}")
    return img.astype(np.float32) / 255


def _warp_local(P: np.ndarray, M: np.ndarray, x0: int, w: int, SH: int, flags=cv2.INTER_LANCZOS4) -> np.ndarray:
    """Warp plate P with canvas transform M into canvas columns [x0, x0 + w)."""
    Ml = M.astype(np.float64).copy()
    Ml[0, 2] -= x0
    return cv2.warpAffine(P, Ml, (w, SH), flags=flags, borderMode=cv2.BORDER_CONSTANT)


def _valid_local(H: int, W: int, M: np.ndarray, x0: int, w: int, SH: int) -> np.ndarray:
    v = _warp_local(np.ones((H, W), np.float32), M, x0, w, SH, flags=cv2.INTER_LINEAR)
    return cv2.erode((v > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)


def _region(M: np.ndarray, W: int, SW: int) -> tuple[int, int]:
    """Canvas column range a plate warped by M can touch (a little margin for the interpolation kernel)."""
    xs = [M[0, 0] * x + M[0, 1] * y + M[0, 2] for x in (0, W) for y in (0, 2000)]
    return max(int(np.floor(min(xs))) - 8, 0), min(int(np.ceil(max(xs))) + 8, SW)


def _seam(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Minimum-difference vertical seam (dynamic programming on colour + gradient difference): x per row."""
    d = np.abs(A - B).sum(2)
    gx = lambda im: np.abs(cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3))
    gy = lambda im: np.abs(cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3))
    cost = cv2.GaussianBlur(d, (0, 0), 2) + 0.5 * np.abs(gx(A) - gx(B)) + 0.5 * np.abs(gy(A) - gy(B))
    hh, ww = cost.shape
    acc = cost.copy()
    back = np.zeros((hh, ww), np.int8)
    for y in range(1, hh):
        prev = acc[y - 1]
        left = np.r_[np.inf, prev[:-1]]
        right = np.r_[prev[1:], np.inf]
        stack = np.vstack([left, prev, right])
        k = stack.argmin(0)
        acc[y] += stack[k, np.arange(ww)]
        back[y] = k - 1
    seam = np.zeros(hh, np.int64)
    seam[-1] = int(acc[-1].argmin())
    for y in range(hh - 1, 0, -1):
        seam[y - 1] = np.clip(seam[y] + back[y, seam[y]], 0, ww - 1)
    return seam


def _join(buf: np.ndarray, hv: np.ndarray, warped: np.ndarray, valid: np.ndarray, gain_fade: float,
          feather: int) -> tuple:
    """Blend a warped plate into the working buffer (both over the same canvas columns): a low-frequency gain match
    along the seam band faded out to the right over gain_fade, and a minimum-difference seam in the middle of the
    overlap, feathered. Returns (buf, hv, seam_x per row, overlap c0, c1, rows r0, r1)."""
    SH, wl = valid.shape
    ov = (valid > 0) & (hv > 0)
    cols = np.nonzero(ov.any(0))[0]
    c0, c1 = int(cols.min()), int(cols.max())
    rows = np.nonzero(ov[:, (c0 + c1) // 2])[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    # gain match along the seam band (low frequency, per channel, a function of y), faded out to the right
    band = slice(c0 + (c1 - c0) // 4, c1 - (c1 - c0) // 4)
    A_low = cv2.GaussianBlur(buf[r0:r1, band], (0, 0), 40)
    B_low = cv2.GaussianBlur(warped[r0:r1, band], (0, 0), 40)
    ratio = np.median((A_low + 0.01) / (B_low + 0.01), axis=1)
    ratio = cv2.GaussianBlur(np.clip(ratio, 0.6, 1.6).astype(np.float32).reshape(-1, 1, 3), (1, 0), sigmaX=1,
                             sigmaY=25).reshape(-1, 3)
    gain = np.ones((SH, 3), np.float32)
    gain[r0:r1] = ratio
    gain[:r0] = ratio[0]
    gain[r1:] = ratio[-1]
    xx = np.arange(wl, dtype=np.float32)
    fade = np.clip(1 - (xx - float(c0 + c1) / 2) / gain_fade, 0, 1).astype(np.float32)[None, :, None]
    warped = (warped * (1 + (gain[:, None, :] - 1) * fade)).astype(np.float32)
    # minimum-difference seam inside the middle of the overlap
    s0, s1 = c0 + int((c1 - c0) * 0.3), c0 + int((c1 - c0) * 0.7)
    seam = _seam(buf[r0:r1, s0:s1], warped[r0:r1, s0:s1])
    seam_x = np.full(SH, s0 + seam[0], np.float32)
    seam_x[r0:r1] = s0 + seam
    seam_x[r1:] = s0 + seam[-1]
    seam_x = cv2.GaussianBlur(seam_x.reshape(-1, 1), (1, 0), sigmaX=1, sigmaY=3).ravel()
    wB = np.clip((xx[None, :] - seam_x[:, None]) / feather + 0.5, 0, 1) * valid
    wB = np.where(hv > 0, wB, valid)
    buf = buf * (1 - wB[..., None]) + warped * wB[..., None]
    hv = np.maximum(hv, valid)
    return buf, hv, seam_x, c0, c1, r0, r1


class TileWriter:
    """Writes a long image as fixed-width PNG tiles from columns pushed left to right (in strip coordinates)."""

    def __init__(self, d: Path, tile: int, height: int, prefix: str, keep_from: int = 0, old: dict | None = None):
        self.d, self.tile, self.h, self.prefix = d, tile, height, prefix
        d.mkdir(parents=True, exist_ok=True)
        self.buf: np.ndarray | None = None
        self.x = keep_from  # next column to receive
        self.hashes: dict[int, str] = dict(old or {})

    def push(self, cols: np.ndarray) -> None:
        """Append columns (H x n x 3 uint8) at self.x."""
        if cols.shape[1] == 0:
            return
        self.buf = cols if self.buf is None else np.concatenate([self.buf, cols], axis=1)
        self.x += cols.shape[1]
        self._flush(final=False)

    def _flush(self, final: bool) -> None:
        start = self.x - (0 if self.buf is None else self.buf.shape[1])
        while self.buf is not None and self.buf.shape[1] > 0:
            k = start // self.tile
            off = start - k * self.tile
            need = self.tile - off
            if self.buf.shape[1] < need and not final:
                return
            part = self.buf[:, :need]
            self._write(k, off, part)
            self.buf = self.buf[:, part.shape[1]:]
            start += part.shape[1]

    def _write(self, k: int, off: int, part: np.ndarray) -> None:
        f = self.d / f"{self.prefix}{k:04d}.png"
        if off > 0:  # completing a tile that was partly written by an earlier build
            old = cv2.imread(str(f))
            part = np.concatenate([old[:, :off], part], axis=1)
        h = hashlib.sha256(part.tobytes()).hexdigest()[:16]
        if self.hashes.get(k) != h or not f.exists():
            cv2.imwrite(str(f), part)
        self.hashes[k] = h

    def close(self) -> dict[int, str]:
        self._flush(final=True)
        # drop tiles past the end (the street got shorter)
        n = (self.x + self.tile - 1) // self.tile
        for f in self.d.glob(f"{self.prefix}*.png"):
            if int(f.stem[len(self.prefix):]) >= n:
                f.unlink()
        return {k: v for k, v in self.hashes.items() if k < n}


class MirrorTileWriter(TileWriter):
    """A TileWriter in the mirrored frame (x' = -1 - x) for the street left of x 0: mirrored tile k' is stored the
    right way round as tile -(k' + 1), right-aligned at its right end (the leftmost tile may be narrower)."""

    def _write(self, k: int, off: int, part: np.ndarray) -> None:
        if off:
            raise ValueError("a mirrored tile writer always starts at x' 0")
        kk = -(k + 1)
        f = self.d / tile_name(self.prefix, kk)
        part = np.ascontiguousarray(part[:, ::-1])
        h = hashlib.sha256(part.tobytes()).hexdigest()[:16]
        if self.hashes.get(kk) != h or not f.exists():
            cv2.imwrite(str(f), part)
        self.hashes[kk] = h

    def close(self) -> dict[int, str]:
        self._flush(final=True)
        n = (self.x + self.tile - 1) // self.tile
        for k, f in tile_files(self.d, self.prefix).items():
            if k < -n:
                f.unlink()
        return {k: v for k, v in self.hashes.items() if -n <= k < 0}


def read_tiles(d: Path, prefix: str, tile: int, x0: int, x1: int, height: int) -> np.ndarray:
    """Columns [x0, x1) of a tiled image (BGR uint8), edge-replicated outside the stored range. Tiles k >= 0 start at
    k * tile; tiles k < 0 (left of x 0) end at (k + 1) * tile."""
    files = tile_files(d, prefix)
    out = np.zeros((height, x1 - x0, 3), np.uint8)
    if not files:
        return out
    k_lo, k_hi = min(files), max(files)

    def start(k: int, w: int) -> int:
        return k * tile if k >= 0 else (k + 1) * tile - w

    x_lo = start(k_lo, Image.open(files[k_lo]).size[0]) if k_lo < 0 else k_lo * tile
    x_hi = k_hi * tile + Image.open(files[k_hi]).size[0] if k_hi >= 0 else (k_hi + 1) * tile
    a0, b0 = max(x0, x_lo), min(x1, x_hi)
    if b0 > a0:
        for k in range(a0 // tile, (b0 - 1) // tile + 1):
            if k not in files:
                continue
            img = cv2.imread(str(files[k]))
            s = start(k, img.shape[1])
            a, b = max(a0, s), min(b0, s + img.shape[1])
            if b > a:
                out[:, a - x0:b - x0] = img[:, a - s:b - s]
    if x0 < x_lo:
        c = min(x_lo - x0, out.shape[1])
        out[:, :c] = out[:, min(c, out.shape[1] - 1)][:, None]
    if x1 > x_hi:
        e = max(x_hi - x0, 1)
        out[:, e:] = out[:, e - 1][:, None]
    return out


def read_strip(d: Path, tile: int, x0: int, x1: int, height: int, patched: bool = True) -> np.ndarray:
    """Strip columns [x0, x1) (BGR uint8); patched: with the join patch of the first left plate (join-0.png) over
    the first columns right of x 0."""
    out = read_tiles(d / "strip", "s", tile, x0, x1, height)
    pf = d / "strip" / "join-0.png"
    if patched and x1 > 0 and pf.exists():
        p = cv2.imread(str(pf))
        a, b = max(x0, 0), min(x1, p.shape[1])
        if b > a:
            out[:, a - x0:b - x0] = p[:, a:b]
    return out


def stitch(proj_root: Path, d: Path, spec: dict, prev: dict | None = None, log=print) -> dict:
    """Stitch the plates into strip tiles (d/strip). Incremental: plates already stitched in the last build with the
    same content and settings are not redone; only tiles right of the first changed plate's overlap are rewritten."""
    plates = [proj_root / p for p in spec["plates"]]
    shas = [sha256_file(p) for p in plates]
    keep, feather, gain_fade = float(spec["keep"]), int(spec["seam_feather"]), float(spec["gain_fade"])
    tile = int(spec["tile"])
    params = {"keep": keep, "feather": feather, "gain_fade": gain_fade, "tile": tile}
    prev = prev or {}
    first = cv2.imread(str(plates[0]))
    H, W = first.shape[:2]
    K = int(round(W * keep))
    # which plates can be reused
    m = 0
    if prev.get("params") == params and prev.get("plate_size") == [W, H] and (d / "strip").exists():
        for i, (pp, s) in enumerate(zip(prev.get("plates", []), shas)):
            if pp["sha"] != s:
                break
            m = i + 1
        m = min(m, len(prev.get("plates", [])))
        if m == len(plates) and len(prev["plates"]) == len(plates):
            log(f"strip: up to date ({len(plates)} plates)")
            return prev
    if m <= 1:
        m = 0  # nothing (or only the first plate) to reuse: build from scratch
    # 1. transforms: T[n] maps plate n to canvas coordinates (before OY)
    T = [np.float64([[1, 0, 0], [0, 1, 0]])]
    joins = []
    if m:
        T = [np.float64(pp["T"]) for pp in prev["plates"][:m]]
        joins = list(prev["joins"][:m - 1])
    for n in range(max(len(T), 1), len(plates)):
        A, B = _read_plate(plates[n - 1]), _read_plate(plates[n])
        if B.shape[:2] != (H, W):
            raise SystemExit(f"plate {plates[n].name} is {B.shape[1]}x{B.shape[0]}, expected {W}x{H}")
        M, ninl, med = similarity(A, B, K)
        step = np.vstack([M, [0, 0, 1]])
        step[0, 2] += W - K
        T.append((np.vstack([T[-1], [0, 0, 1]]) @ step)[:2])
        joins.append({"join": f"{plates[n - 1].stem}-{plates[n].stem}", "inliers": ninl, "median_px": round(med, 2),
                      "scale": round(float(np.hypot(M[0, 0], M[0, 1])), 5), "dx": round(float(M[0, 2]), 2),
                      "dy": round(float(M[1, 2]), 2)})
        log(f"strip: join {joins[-1]['join']}: {ninl} inliers, median {med:.2f} px, scale {joins[-1]['scale']}")
    SW = int(np.ceil(max(t[0, 2] for t in T) + W)) + 4
    SH = H + 8
    Mc = []
    for t in T:
        M = t.copy()
        M[1, 2] += OY
        Mc.append(M)
    # 2. crop: rows and columns every plate covers (frozen once the street exists, so old tiles stay valid)
    if m:
        y0, y1, x0 = prev["crop"][1], prev["crop"][3], prev["crop"][0]
    else:
        have = np.zeros((SH, SW), bool)
        for n, M in enumerate(Mc):
            a, b = _region(M, W, SW)
            have[:, a:b] |= _valid_local(H, W, M, a, b - a, SH) > 0
        full_rows = np.nonzero(have[:, 50:-50].all(1))[0]
        y0, y1 = int(full_rows.min()), int(full_rows.max()) + 1
        xcols = np.nonzero(have[y0:y1].all(0))[0]
        x0 = int(xcols.min())
    # 3. streaming blend. The working buffer holds canvas columns [bx, bx + width)
    reg = [_region(M, W, SW) for M in Mc]
    if m:
        start_n = m
        bx = reg[m][0]
        keep_from = bx - x0
        old = read_tiles(d / "strip", "s", tile, bx - x0, reg[m - 1][1] - x0, y1 - y0)
        buf = np.zeros((SH, reg[m - 1][1] - bx, 3), np.float32)
        buf[y0:y1] = old.astype(np.float32) / 255
        hv = _valid_local(H, W, Mc[m - 1], bx, reg[m - 1][1] - bx, SH)
        seams = list(prev["seams"][:m - 1])
        tw = TileWriter(d / "strip", tile, y1 - y0, "s", keep_from=keep_from,
                        old={int(k): v for k, v in prev["tiles"].items() if int(k) < keep_from // tile})
    else:
        start_n = 1
        bx = reg[0][0]
        buf = _warp_local(_read_plate(plates[0]), Mc[0], bx, reg[0][1] - bx, SH)
        hv = _valid_local(H, W, Mc[0], bx, reg[0][1] - bx, SH)
        buf *= hv[..., None]
        seams = []
        tw = TileWriter(d / "strip", tile, y1 - y0, "s", keep_from=0)
        if bx > x0:
            raise SystemExit("stitch: first plate does not start at the strip's left edge")
        bx_out = x0
        buf = buf[:, bx_out - bx:]
        hv = hv[:, bx_out - bx:]
        bx = bx_out

    def out_cols(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        o = np.clip(a[y0:y1], 0, 1)
        ok = v[y0:y1] > 0
        if not ok.all():  # rows the plates drifted off (the far right end): from the nearest covered row
            o = _fill_rows(o, ok)
        return (o * 255 + 0.5).astype(np.uint8)

    for n in range(start_n, len(plates)):
        P = _read_plate(plates[n])
        a, b = reg[n]
        # flush what plate n cannot touch
        if a > bx:
            tw.push(out_cols(buf[:, :a - bx], hv[:, :a - bx]))
            buf, hv = buf[:, a - bx:], hv[:, a - bx:]
            bx = a
        # grow the buffer to cover plate n
        if bx + buf.shape[1] < b:
            extra = b - (bx + buf.shape[1])
            buf = np.concatenate([buf, np.zeros((SH, extra, 3), np.float32)], axis=1)
            hv = np.concatenate([hv, np.zeros((SH, extra), np.float32)], axis=1)
        wl = buf.shape[1]
        warped = _warp_local(P, Mc[n], bx, wl, SH)
        valid = _valid_local(H, W, Mc[n], bx, wl, SH)
        buf, hv, seam_x, c0, c1, r0, r1 = _join(buf, hv, warped, valid, gain_fade, feather)
        seams.append({"join": joins[n - 1]["join"], "x_mean": round(float(seam_x[r0:r1].mean()) + bx - x0, 1),
                      "overlap": [int(c0 + bx - x0), int(c1 + bx - x0)]})
        log(f"strip: seam {seams[-1]['join']} at x {seams[-1]['x_mean']:.0f}")
    # the strip ends where the last plate's coverage of the cropped rows ends; when the last plates drifted off the
    # crop's rows (no column of the last plate covers them all), where at least edge_cover of the rows are covered
    cov = (hv[y0:y1] > 0).mean(0)
    xcols = np.nonzero(cov >= 1.0)[0]
    if not len(xcols) or xcols.max() + 1 < buf.shape[1] - W // 2:
        xcols = np.nonzero(cov > float(spec.get("edge_cover", 0.98)))[0]
    x1_local = int(xcols.max()) + 1
    tw.push(out_cols(buf[:, :x1_local], hv[:, :x1_local]))
    tiles = tw.close()
    x1 = bx + x1_local
    info = {"params": params, "plate_size": [W, H], "crop": [x0, y0, x1, y1], "size": [x1 - x0, y1 - y0],
            "plates": [{"path": str(spec["plates"][i]), "sha": shas[i], "T": T[i].tolist(),
                        "offset": [round(T[i][0, 2] - x0, 2), round(T[i][1, 2] + OY - y0, 2)]}
                       for i in range(len(plates))],
            "joins": joins, "seams": seams, "tiles": {str(k): v for k, v in sorted(tiles.items())}}
    log(f"strip: {x1 - x0} x {y1 - y0} px in {len(tiles)} tiles")
    return info


def _fill_rows(img: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Fill the pixels a column does not cover (ok False) from the nearest covered row of that column."""
    h = img.shape[0]
    idx = np.where(ok, np.arange(h)[:, None], -1)
    fwd = np.maximum.accumulate(idx, axis=0)  # last covered row at or above
    bwd = np.where(ok, np.arange(h)[:, None], h)
    bwd = np.minimum.accumulate(bwd[::-1], axis=0)[::-1]  # first covered row at or below
    src = np.where(fwd >= 0, fwd, bwd)
    src = np.clip(src, 0, h - 1)
    return np.take_along_axis(img, src[..., None].repeat(img.shape[2], 2), axis=0)


def _mirror_T(T: np.ndarray, W: int, cx0: float) -> np.ndarray:
    """A plate transform in the mirrored frame: mirrored plate px (W - 1 - u, v) to x' = -1 - (X - cx0), Y."""
    a, b, c = T[0]
    d_, e, f = T[1]
    return np.float64([[a, -b, cx0 - 1 - a * (W - 1) - c], [-d_, e, d_ * (W - 1) + f]])


def stitch_left(proj_root: Path, d: Path, spec: dict, right: dict, prev: dict | None = None,
                log=print) -> dict | None:
    """Stitch the left plates (spec "left", nearest first) into strip tiles left of x 0 (s-0001.png ...), in the
    mirrored frame x' = -1 - x. Every join is the usual SIFT similarity, gain match and minimum-difference seam in
    the middle of the overlap; for the first left plate that seam lies right of x 0 (away from the first plate's own
    left border), so the columns between x 0 and the seam are stored as a join patch (strip/join-0.png) that the
    street right of x 0 is rendered with. The strip tiles right of x 0 are never rewritten. Rebuilt whenever a left
    plate changes; tile hashes keep unchanged tiles (and their caches) as they were."""
    sdir = d / "strip"
    lefts = list(spec.get("left") or [])
    if not lefts:
        for k, f in tile_files(sdir, "s").items():
            if k < 0:
                f.unlink()
        if (sdir / "join-0.png").exists():
            (sdir / "join-0.png").unlink()
        return None
    paths = [proj_root / spec["plates"][0]] + [proj_root / p for p in lefts]
    shas = [sha256_file(p) for p in paths]
    keep, feather, gain_fade = float(spec["keep"]), int(spec["seam_feather"]), float(spec["gain_fade"])
    tile = int(spec["tile"])
    W, H = right["plate_size"]
    cx0, y0, _, y1 = right["crop"]
    params = {"keep": keep, "feather": feather, "gain_fade": gain_fade, "tile": tile, "crop": [cx0, y0, y1],
              "first": shas[0], "join": "overlap"}
    cover = float(spec.get("edge_cover", 0.98))
    if "edge_cover" in spec:
        params["edge_cover"] = cover
    have = {k for k in tile_files(sdir, "s") if k < 0}
    pf = sdir / "join-0.png"
    if prev and prev.get("params") == params and [pp["sha"] for pp in prev["plates"]] == shas[1:] \
            and have == {int(k) for k in prev["tiles"]} and (prev.get("patch") is None) != pf.exists():
        log(f"strip: left side up to date ({len(lefts)} plates)")
        return prev
    K = int(round(W * keep))

    def plate(n: int) -> np.ndarray:
        P = _read_plate(paths[n])
        if P.shape[:2] != (H, W):
            raise SystemExit(f"plate {paths[n].name} is {P.shape[1]}x{P.shape[0]}, expected {W}x{H}")
        return np.ascontiguousarray(P[:, ::-1])

    # 1. transforms in the mirrored frame: the first plate sits at x' < 0, each left plate is "appended" to its right
    T = [_mirror_T(np.float64(right["plates"][0]["T"]), W, cx0)]
    joins = []
    A = plate(0)
    for n in range(1, len(paths)):
        B = plate(n)
        M, ninl, med = similarity(A, B, K)
        step = np.vstack([M, [0, 0, 1]])
        step[0, 2] += W - K
        T.append((np.vstack([T[-1], [0, 0, 1]]) @ step)[:2])
        joins.append({"join": f"{paths[n - 1].stem}-{paths[n].stem}", "inliers": ninl, "median_px": round(med, 2),
                      "scale": round(float(np.hypot(M[0, 0], M[0, 1])), 5), "dx": round(float(M[0, 2]), 2),
                      "dy": round(float(M[1, 2]), 2)})
        log(f"strip: join {joins[-1]['join']} (left): {ninl} inliers, median {med:.2f} px, scale {joins[-1]['scale']}")
        A = B
    SW = int(np.ceil(max(t[0, 2] for t in T) + W)) + 4
    SH = H + 8
    Mc = []
    for t in T:
        M = t.copy()
        M[1, 2] += OY
        Mc.append(M)
    reg = [_region(M, W, SW) for M in Mc]
    # 2. the first left plate joins the first plate inside their overlap (an ordinary seam, away from the first
    # plate's own left border). The buffer starts at the overlap's far end, x' -K (x K - 1); the columns left of x' 0
    # are the first plate's, rewritten only up to the seam: they become the join patch right of x 0
    P1, L1 = plate(0), plate(1)
    bx = -K - 8
    wl = reg[1][1] - bx
    buf = _warp_local(P1, Mc[0], bx, wl, SH)
    hv = _valid_local(H, W, Mc[0], bx, wl, SH)
    buf *= hv[..., None]
    warped = _warp_local(L1, Mc[1], bx, wl, SH)
    valid = _valid_local(H, W, Mc[1], bx, wl, SH)
    buf, hv, seam_x, c0, c1, sr0, sr1 = _join(buf, hv, warped, valid, gain_fade, feather)
    xm = float(seam_x[sr0:sr1].mean()) + bx
    seams = [{"join": joins[0]["join"], "x_mean": round(-1 - xm, 1),
              "overlap": [int(-1 - (c1 + bx)), int(-1 - (c0 + bx))]}]
    log(f"strip: seam {joins[0]['join']} at x {seams[0]['x_mean']:.0f} (inside the overlap, right of x 0)")
    tw = MirrorTileWriter(sdir, tile, y1 - y0, "s")

    def out_cols(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        o = np.clip(a[y0:y1], 0, 1)
        ok = v[y0:y1] > 0
        if not ok.all():
            o = _fill_rows(o, ok)
        return (o * 255 + 0.5).astype(np.uint8)

    # the join patch: strip columns right of x 0 that the seam changed (the rest of the first plate is untouched)
    patch = np.ascontiguousarray(out_cols(buf[:, :-bx], hv[:, :-bx])[:, ::-1])  # x 0 .. -bx - 1
    pure = read_tiles(sdir, "s", tile, 0, patch.shape[1], y1 - y0)
    differ = np.nonzero((patch != pure).any(axis=(0, 2)))[0]
    xp = int(differ.max()) + 1 if len(differ) else 0
    pf = sdir / "join-0.png"
    patch_info = None
    if xp:
        cv2.imwrite(str(pf), patch[:, :xp])
        patch_info = {"x": [0, xp], "sha": hashlib.sha256(patch[:, :xp].tobytes()).hexdigest()[:16]}
        log(f"strip: the join rewrites strip x 0 to {xp} (the first plate's overlap with {paths[1].stem})")
    elif pf.exists():
        pf.unlink()
    buf, hv, bx = buf[:, -bx:], hv[:, -bx:], 0

    # 3. the plates further left: streaming blend as on the right side
    for n in range(2, len(paths)):
        P = plate(n)
        a, b = reg[n]
        if a > bx:
            tw.push(out_cols(buf[:, :a - bx], hv[:, :a - bx]))
            buf, hv = buf[:, a - bx:], hv[:, a - bx:]
            bx = a
        if bx + buf.shape[1] < b:
            extra = b - (bx + buf.shape[1])
            buf = np.concatenate([buf, np.zeros((SH, extra, 3), np.float32)], axis=1)
            hv = np.concatenate([hv, np.zeros((SH, extra), np.float32)], axis=1)
        wl = buf.shape[1]
        warped = _warp_local(P, Mc[n], bx, wl, SH)
        valid = _valid_local(H, W, Mc[n], bx, wl, SH)
        buf, hv, seam_x, c0, c1, sr0, sr1 = _join(buf, hv, warped, valid, gain_fade, feather)
        xm = float(seam_x[sr0:sr1].mean()) + bx
        seams.append({"join": joins[n - 1]["join"], "x_mean": round(-1 - xm, 1),
                      "overlap": [int(-1 - (c1 + bx)), int(-1 - (c0 + bx))]})
        log(f"strip: seam {seams[-1]['join']} at x {seams[-1]['x_mean']:.0f}")
    # the far left end: every row covered (or nearly; edge_cover lowers it for a far plate the joins shrank, its
    # missing top and bottom rows filled from the nearest covered row)
    xcols = np.nonzero((hv[y0:y1] > 0).mean(0) > cover)[0]
    x1_local = int(xcols.max()) + 1
    tw.push(out_cols(buf[:, :x1_local], hv[:, :x1_local]))
    tiles = tw.close()
    width = bx + x1_local
    info = {"params": params, "size": [width, y1 - y0], "x_min": -width, "patch": patch_info,
            "plates": [{"path": str(lefts[i - 1]), "sha": shas[i], "T": T[i].tolist(),
                        "x_range": [round(-T[i][0, 2] - W, 1), round(-T[i][0, 2], 1)]} for i in range(1, len(paths))],
            "joins": joins, "seams": seams, "tiles": {str(k): v for k, v in sorted(tiles.items())}}
    log(f"strip: left side {width} x {y1 - y0} px in {len(tiles)} tiles (x {-width} to 0)")
    return info


# ------------------------------------------------------------------------------------------------ edits

@dataclass
class Piece:
    src0: int  # strip columns (before the join extensions)
    src1: int
    mirror: bool = False
    stretch: float = 1.0
    ext_l: int = 0  # source columns added for the cross-fade
    ext_r: int = 0

    @property
    def plain(self) -> bool:
        return not self.mirror and self.stretch == 1.0

    def width(self) -> int:
        w = self.src1 - self.src0 + self.ext_l + self.ext_r
        return w if self.stretch == 1.0 else int(round(w * self.stretch))


def pieces(strip_w: int, edits: list[dict], feather: int) -> list[Piece]:
    """The edited street as strip pieces in order; consecutive pieces are cross-faded over `feather` columns."""
    cuts = sorted([tuple(e["cut"]) for e in edits if "cut" in e])
    base: list[Piece] = []
    x = 0
    for a, b in cuts:
        if a < x or b <= a:
            raise SystemExit(f"street edits: bad or overlapping cut {a}..{b}")
        base.append(Piece(x, a))
        x = b
    base.append(Piece(x, strip_w))
    out = list(base)
    for e in sorted([e for e in edits if "insert" in e], key=lambda e: e["insert"]):
        at = int(e["insert"])
        k = next((i for i, p in enumerate(out) if p.plain and p.src0 < at < p.src1), None)
        if k is None:
            raise SystemExit(f"street edits: insert at {at} is not inside the kept strip")
        p = out[k]
        s0, s1 = e["src"]
        new = Piece(int(s0), int(s1), bool(e.get("mirror", False)), float(e.get("stretch", 1.0)))
        out[k:k + 1] = [Piece(p.src0, at), new, Piece(at, p.src1)]
    h = feather // 2
    for i in range(len(out) - 1):
        out[i].ext_r = h
        out[i + 1].ext_l = h
    return out


def xmap(strip_x: float, ps: list[Piece], feather: int) -> float | None:
    """Street x of a strip column (None when the column was cut). The first plain piece holding it wins."""
    for p, X in zip(ps, _piece_offsets(ps, feather)):
        if p.plain and p.src0 <= strip_x < p.src1:
            return X + p.ext_l + (strip_x - p.src0)
    return None


def _piece_offsets(ps: list[Piece], feather: int) -> list[int]:
    out, X = [], 0
    for p in ps:
        out.append(X)
        X += p.width() - feather
    return out


class _Composer:
    """Street columns from strip pieces: plain pieces are read from the strip tiles, mirrored or stretched ones are
    rendered once and kept; consecutive pieces cross-fade over F columns. Piece coordinates start at 0; `base` is the
    strip x of piece coordinate 0 (0 right of x 0, the left strip's left end for the street left of x 0)."""

    def __init__(self, d: Path, tile: int, sh: int, ps: list[Piece], F: int, base: int = 0, patched: bool = True):
        self.d, self.tile, self.sh, self.ps, self.F, self.base = d, tile, sh, ps, F, base
        self.patched = patched
        self.offs = _piece_offsets(ps, F)
        self.total = self.offs[-1] + ps[-1].width()
        self.special: dict[int, np.ndarray] = {}
        self.w_ramp = np.linspace(0, 1, F, dtype=np.float32)

    def _piece_cols(self, i: int, a: int, b: int) -> np.ndarray:
        """Columns [a, b) of piece i's rendered image (float32 BGR 0..255)."""
        p, tile, sh, o = self.ps[i], self.tile, self.sh, self.base
        if p.plain:
            return read_strip(self.d, tile, o + p.src0 - p.ext_l + a, o + p.src0 - p.ext_l + b, sh,
                              self.patched).astype(np.float32)
        if i not in self.special:
            src = read_strip(self.d, tile, o + p.src0 - p.ext_l, o + p.src1 + p.ext_r, sh,
                             self.patched).astype(np.float32)
            if p.mirror:
                src = src[:, ::-1]
            if p.stretch != 1.0:
                src = cv2.resize(src, (int(round(src.shape[1] * p.stretch)), sh), interpolation=cv2.INTER_CUBIC)
            self.special[i] = src
        return self.special[i][:, a:b]

    def cols(self, X0: int, X1: int) -> np.ndarray:
        """Edited street columns [X0, X1) in piece coordinates (BGR uint8)."""
        ps, offs, F, w_ramp = self.ps, self.offs, self.F, self.w_ramp
        acc = np.zeros((self.sh, X1 - X0, 3), np.float32)
        for i, p in enumerate(ps):
            P0, P1 = offs[i], offs[i] + p.width()
            a, b = max(X0, P0), min(X1, P1)
            if b <= a:
                continue
            wcol = np.ones(b - a, np.float32)
            xs = np.arange(a, b)
            if i > 0:  # left overlap: ramps in
                m = xs < P0 + F
                wcol[m] = w_ramp[xs[m] - P0]
            if i < len(ps) - 1:  # right overlap: ramps out
                m = xs >= P1 - F
                wcol[m] *= 1 - w_ramp[xs[m] - (P1 - F)]
            acc[:, a - X0:b - X0] += self._piece_cols(i, a - P0, b - P0) * wcol[None, :, None]
        return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


def _side_edits(spec: dict, side: int, shift: int = 0) -> list[dict]:
    """The edits on one side of x 0, shifted by `shift` strip px."""
    out = []
    for e in spec["edits"]:
        if _edit_side(e) != side:
            continue
        e = dict(e)
        if "cut" in e:
            e["cut"] = [v + shift for v in e["cut"]]
        else:
            e["insert"] = e["insert"] + shift
            e["src"] = [v + shift for v in e["src"]]
        out.append(e)
    return out


def render_street(d: Path, spec: dict, strip_info: dict, prev: dict | None = None, log=print,
                  patches: "Patches | None" = None) -> dict:
    """Render the edited street tiles right of x 0 (d/tiles) from the strip tiles, patches pasted in; unchanged
    tiles are not rewritten."""
    tile = int(spec["tile"])
    F = int(spec["edit_feather"])
    sw, sh = strip_info["size"]
    ps = pieces(sw, _side_edits(spec, 1), F)
    comp = _Composer(d, tile, sh, ps, F)
    total, offs = comp.total, comp.offs
    prev = prev or {}
    paste = patches.apply if patches else (lambda img, x0: img)
    tw = TileWriter(d / "tiles", tile, sh, "t", old={int(k): v for k, v in prev.get("tiles", {}).items()})
    for X0 in range(0, total, tile):
        tw.push(paste(comp.cols(X0, min(X0 + tile, total)), X0))
    tiles = tw.close()
    # tiles the join patch changed are also kept as they are without it (tiles/pure/): the light maps right of x 0
    # are computed from those, so prepending plates never changes how the old street is lit
    light = {}
    pdir = d / "tiles" / "pure"
    pf = d / "strip" / "join-0.png"
    if pf.exists():
        pure = _Composer(d, tile, sh, ps, F, patched=False)
        xp = Image.open(pf).size[0]
        pdir.mkdir(parents=True, exist_ok=True)
        for k in range(0, (xp - 1) // tile + 1):
            img = paste(pure.cols(k * tile, min((k + 1) * tile, total)), k * tile)
            light[k] = hashlib.sha256(img.tobytes()).hexdigest()[:16]
            f = pdir / tile_name("t", k)
            if not f.exists() or light[k] != hashlib.sha256(cv2.imread(str(f)).tobytes()).hexdigest()[:16]:
                cv2.imwrite(str(f), img)
    for k, f in tile_files(pdir, "t").items() if pdir.exists() else []:
        if k not in light:
            f.unlink()
    info = {"size": [total, sh], "tile": tile,
            "pieces": [{"src": [p.src0, p.src1], "mirror": p.mirror, "stretch": p.stretch, "x": offs[i],
                        "w": p.width()} for i, p in enumerate(ps)],
            "tiles": {str(k): v for k, v in sorted(tiles.items())}}
    if light:
        info["light_tiles"] = {str(k): v for k, v in sorted(light.items())}
    log(f"street: {total} x {sh} px in {len(tiles)} tiles ({len(ps)} pieces)")
    return info


def render_street_left(d: Path, spec: dict, left: dict | None, prev: dict | None = None, log=print,
                       patches: "Patches | None" = None) -> dict | None:
    """Render the street left of x 0 (tiles t-0001.png ...) from the left strip and the edits left of x 0, patches
    pasted in. The pieces are laid out leftward from x 0, so an edit there moves everything further left and nothing
    right of x 0."""
    tdir = d / "tiles"
    if not left:
        for k, f in tile_files(tdir, "t").items():
            if k < 0:
                f.unlink()
        return None
    tile = int(spec["tile"])
    F = int(spec["edit_feather"])
    lw, sh = left["size"]
    ps = pieces(lw, _side_edits(spec, -1, lw), F)
    comp = _Composer(d, tile, sh, ps, F, base=-lw)
    total = comp.total
    old = {int(k): v for k, v in ((prev or {}).get("tiles") or {}).items()}
    tiles = {}
    tdir.mkdir(parents=True, exist_ok=True)
    k = -1
    while (k + 1) * tile > -total:
        X0, X1 = max(k * tile, -total), (k + 1) * tile
        img = comp.cols(X0 + total, X1 + total)
        if patches:
            img = patches.apply(img, X0)
        h = hashlib.sha256(img.tobytes()).hexdigest()[:16]
        f = tdir / tile_name("t", k)
        if old.get(k) != h or not f.exists():
            cv2.imwrite(str(f), img)
        tiles[k] = h
        k -= 1
    for kk, f in tile_files(tdir, "t").items():
        if kk < 0 and kk not in tiles:
            f.unlink()
    info = {"size": [total, sh], "x_min": -total,
            "pieces": [{"src": [p.src0 - lw, p.src1 - lw], "mirror": p.mirror, "stretch": p.stretch,
                        "x": comp.offs[i] - total, "w": p.width()} for i, p in enumerate(ps)],
            "tiles": {str(kk): v for kk, v in sorted(tiles.items())}}
    log(f"street: left side {total} x {sh} px in {len(tiles)} tiles ({len(ps)} pieces), x {-total} to 0")
    return info


def build(proj_root: Path, d: Path, log=print) -> dict:
    """`aimv paint street`: stitch (incrementally) and render the edited street tiles. Writes d/build.json."""
    spec = load_spec(d)
    old = read_json(d / "build.json") if (d / "build.json").exists() else {}
    s = dict(stitch(proj_root, d, spec, old.get("strip"), log))
    s.pop("left", None)
    left = stitch_left(proj_root, d, spec, s, (old.get("strip") or {}).get("left"), log)
    patches = Patches.prepare(proj_root, d, spec, s, left, log)
    st = dict(render_street(d, spec, s, old.get("street"), log, patches))
    st.pop("left", None)
    st.pop("patches", None)
    st_left = render_street_left(d, spec, left, (old.get("street") or {}).get("left"), log, patches)
    if left:
        s["left"], st["left"] = left, st_left
    if patches:
        st["patches"] = patches.info
    write_json(d / "build.json", {"version": 1, "strip": s, "street": st}, nd=4)
    # the world round the street (world.py): placed on the street just written, kept in world/
    from . import world as WD

    old_w = None
    if old.get("world") and (d / "world" / "world.json").exists():
        old_w = read_json(d / "world" / "world.json")
    w = WD.build(proj_root, d, spec, s, st, old_w, log)
    if w is None:
        return {"strip": s, "street": st}
    write_json(d / "build.json", {"version": 1, "strip": s, "street": st,
                                  "world": {"info": "world/world.json", "cells": len(w["cells"]),
                                            "tiles": WD.info_hash(w)}}, nd=4)
    return {"strip": s, "street": st, "world": w}


def street_x(d: Path, strip_x: float) -> float | None:
    """Street x of a strip column on either side of x 0 (None when an edit cut it)."""
    spec = load_spec(d)
    b = read_json(d / "build.json")
    F = int(spec["edit_feather"])
    if strip_x >= 0:
        return xmap(strip_x, pieces(b["strip"]["size"][0], _side_edits(spec, 1), F), F)
    lw = b["strip"]["left"]["size"][0]
    ps = pieces(lw, _side_edits(spec, -1, lw), F)
    x = xmap(strip_x + lw, ps, F)
    return None if x is None else x - b["street"]["left"]["size"][0]


# ------------------------------------------------------------------------------------------------ patches

def _base_composers(d: Path, spec: dict, strip_info: dict, left: dict | None):
    """The unpatched street as composers: (right, left or None, left width in street px)."""
    tile, F = int(spec["tile"]), int(spec["edit_feather"])
    sw, sh = strip_info["size"]
    right = _Composer(d, tile, sh, pieces(sw, _side_edits(spec, 1), F), F)
    if not left:
        return right, None, 0
    lw = left["size"][0]
    comp = _Composer(d, tile, sh, pieces(lw, _side_edits(spec, -1, lw), F), F, base=-lw)
    return right, comp, comp.total


def _base_cols(comps, X0: int, X1: int) -> np.ndarray:
    """Unpatched street columns [X0, X1) (BGR uint8), either side of x 0."""
    right, left, lt = comps
    parts = []
    if X0 < 0:
        if left is None:
            raise SystemExit(f"street x {X0} is left of the street")
        a = max(X0, -lt)
        if a > X0:
            raise SystemExit(f"street x {X0} is left of the street (it starts at {-lt})")
        parts.append(left.cols(a + lt, min(X1, 0) + lt))
    if X1 > 0:
        if X1 > right.total:
            raise SystemExit(f"street x {X1} is right of the street (it ends at {right.total})")
        parts.append(right.cols(max(X0, 0), X1))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def align_patch(base: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Similarity mapping an edited copy of a window (any resolution) onto the window itself (BGR uint8 both):
    (M, inliers, median residual px)."""
    h, w = base.shape[:2]
    k = h / patch.shape[0]  # a uniform first guess (an edit of a crop, upscaled, keeps its aspect)
    S = np.float64([[k, 0, 0], [0, k, 0], [0, 0, 1]])
    p0 = cv2.resize(patch, (max(1, int(round(patch.shape[1] * k))), h),
                    interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    sift = cv2.SIFT_create(12000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        raise SystemExit("patch: no features to align it by")
    m = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 12:
        raise SystemExit(f"patch: only {len(good)} matches with its window (is it an edit of that window?)")
    pb = np.float32([kb[g.queryIdx].pt for g in good])
    pa = np.float32([ka[g.trainIdx].pt for g in good])
    cv2.setRNGSeed(12345)
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=5000)
    if M is None:
        raise SystemExit("patch: no consistent alignment with its window")
    res = np.linalg.norm(pb @ M[:, :2].T + M[:, 2] - pa, axis=1)[inl.ravel() > 0]
    return (np.vstack([M, [0, 0, 1]]) @ S)[:2], int(inl.sum()), float(np.median(res))


def _blur_masked(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian blur of img (H x W x C) where w > 0."""
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def patch_layer(base: np.ndarray, patch: np.ndarray, mode: str = "diff", feather: int = 48, grow: int = 24,
                thresh: float = 28.0, min_area: int = 1500, keep: np.ndarray | None = None,
                tone_from: np.ndarray | None = None, M: np.ndarray | None = None) -> dict:
    """The aligned, tone-matched patch and its alpha over the window (base: the street window, BGR uint8).
    keep: an optional 0/1 mask (window px) of where the patch may change the street. tone_from: an optional 0/1 mask
    of the base pixels the tone may come from (default all; a world insert leaves out its placeholder)."""
    h, w = base.shape[:2]
    if M is None:
        M, ninl, med = align_patch(base, patch)
    else:  # placed as given (an exact edit of the built street: no features to find against the unpatched one)
        M, ninl, med = np.float64(M), 0, 0.0
    warped = cv2.warpAffine(patch.astype(np.float32), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones(patch.shape[:2], np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.erode((valid > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    B = base.astype(np.float32)

    def changed(P: np.ndarray) -> np.ndarray:
        dd = np.abs(cv2.GaussianBlur(B, (0, 0), 2) - cv2.GaussianBlur(P, (0, 0), 2)).max(2)
        return ((dd > thresh) & (valid > 0)).astype(np.uint8)

    # tone: a low-frequency gain from the pixels the edit left alone (the edit model shifts colours a little)
    ch0 = cv2.dilate(changed(warped), np.ones((31, 31), np.uint8))
    wt = valid * (1 - ch0)
    if tone_from is not None:
        wt = wt * tone_from.astype(np.float32)
    gain = _blur_masked(B + 4, wt, 60) / np.maximum(_blur_masked(warped + 4, wt, 60), 1e-3)
    # a large changed area (a whole shop front) has no unchanged pixel within reach of the 60 px blur: there the gain
    # comes from further out (400 px), else the window's median, never the clip of an empty ratio
    near = np.clip(cv2.GaussianBlur(wt, (0, 0), 60) / 0.02, 0, 1)[..., None]
    if (near < 1).any():
        far = _blur_masked(B + 4, wt, 400) / np.maximum(_blur_masked(warped + 4, wt, 400), 1e-3)
        reach = np.clip(cv2.GaussianBlur(wt, (0, 0), 400) / 0.02, 0, 1)[..., None]
        sel = wt > 0.5
        flat = (np.median((B + 4)[sel], axis=0) / np.maximum(np.median((warped + 4)[sel], axis=0), 1e-3)) \
            if sel.any() else np.ones(3, np.float32)
        gain = gain * near + (far * reach + flat * (1 - reach)) * (1 - near)
    gain = np.clip(gain, 0.6, 1.6)
    warped = np.clip(warped * gain, 0, 255)
    # where it may paste: inside the window, ramped down over `feather` px at its border
    ramp = cv2.erode(valid.astype(np.uint8), np.ones((feather + 1, feather + 1), np.uint8)).astype(np.float32)
    ramp = cv2.GaussianBlur(ramp, (0, 0), feather / 4)
    if mode == "full":
        a = ramp
    elif mode == "diff":
        c = changed(warped)
        c = cv2.morphologyEx(c, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(c)
        big = np.zeros(n, np.uint8)
        big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        c = big[lab]
        if grow:
            c = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        a = cv2.GaussianBlur(c.astype(np.float32), (0, 0), max(feather / 4, 1)) * ramp
    else:
        raise SystemExit(f"patch: mask is diff or full, not {mode!r}")
    if keep is not None:
        a = a * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), max(feather / 4, 1))
    a = np.clip(a, 0, 1)
    ys, xs = np.nonzero(a > 0.5)
    return {"layer": warped, "alpha": a, "valid": valid, "inliers": ninl, "median_px": round(med, 2),
            "scale": round(float(np.hypot(M[0, 0], M[0, 1])), 4), "dx": round(float(M[0, 2]), 1),
            "dy": round(float(M[1, 2]), 1),
            "changed": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None}


class Patches:
    """The street's patch edits, aligned and ready to paste into street columns (street px)."""

    def __init__(self, items: list[tuple[int, int, np.ndarray, np.ndarray]], info: list[dict]):
        self.items = items  # (x0, y0, layer BGR float32, alpha) cropped to where alpha > 0
        self.info = info

    def __bool__(self) -> bool:
        return bool(self.items)

    @classmethod
    def prepare(cls, proj_root: Path, d: Path, spec: dict, strip_info: dict, left: dict | None,
                log=print) -> "Patches | None":
        es = [e for e in spec["edits"] if "patch" in e]
        if not es:
            return None
        comps = _base_composers(d, spec, strip_info, left)
        F = int(spec["edit_feather"])
        items, info = [], []
        for e in es:
            f = proj_root / e["patch"]
            img = cv2.imread(str(f))
            if img is None:
                raise SystemExit(f"street patch: cannot read {f}")
            x0, x1 = (int(v) for v in e["window"])
            base = _base_cols(comps, x0, x1)
            keep = None
            if e.get("region"):
                rx0, ry0, rx1, ry1 = (int(v) for v in e["region"])
                keep = np.zeros(base.shape[:2], np.uint8)
                keep[max(ry0, 0):max(ry1, 0), max(rx0 - x0, 0):max(rx1 - x0, 0)] = 1
            M = None
            if e.get("align") == "none":  # an exact edit of the built street, placed as it is
                if img.shape[:2] != base.shape[:2]:
                    raise SystemExit(f"street patch {f.name}: align none needs it at its window's size "
                                     f"({base.shape[1]} x {base.shape[0]}), not {img.shape[1]} x {img.shape[0]}")
                M = np.float64([[1, 0, 0], [0, 1, 0]])
            elif e.get("align") not in (None, "sift"):
                raise SystemExit(f"street patch {f.name}: align is sift or none, not {e.get('align')!r}")
            r = patch_layer(base, img, str(e.get("mask", "diff")), int(e.get("feather", F // 2)),
                            int(e.get("grow", 24)), float(e.get("thresh", 28.0)), int(e.get("min_area", 1500)), keep,
                            M=M)
            a = r["alpha"]
            ys, xs = np.nonzero(a > 0.002)
            row = {"patch": str(e["patch"]), "sha": sha256_file(f)[:16], "window": [x0, x1],
                   "inliers": r["inliers"], "median_px": r["median_px"], "scale": r["scale"],
                   "changed": None}
            if len(xs):
                a0, a1, b0, b1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                items.append((x0 + int(b0), int(a0), r["layer"][a0:a1, b0:b1].copy(), a[a0:a1, b0:b1].copy()))
            if r["changed"]:
                cx0, cy0, cx1, cy1 = r["changed"]
                row["changed"] = [x0 + cx0, cy0, x0 + cx1, cy1]
            info.append(row)
            log(f"street: patch {Path(e['patch']).name} on x {x0} to {x1}: {r['inliers']} inliers, median "
                f"{r['median_px']} px, scale {r['scale']}, changed {row['changed']}")
        return cls(items, info)

    def apply(self, img: np.ndarray, X0: int) -> np.ndarray:
        """Paste the patches into street columns img (BGR uint8) starting at street x X0."""
        X1 = X0 + img.shape[1]
        out = None
        for px0, py0, layer, a in self.items:
            lh, lw = a.shape
            c0, c1 = max(X0, px0), min(X1, px0 + lw)
            if c1 <= c0:
                continue
            if out is None:
                out = img.astype(np.float32)
            r0, r1 = py0, min(py0 + lh, img.shape[0])
            al = a[:r1 - r0, c0 - px0:c1 - px0, None]
            reg = out[r0:r1, c0 - X0:c1 - X0]
            out[r0:r1, c0 - X0:c1 - X0] = reg * (1 - al) + layer[:r1 - r0, c0 - px0:c1 - px0] * al
        return img if out is None else np.clip(out + 0.5, 0, 255).astype(np.uint8)


def export_window(d: Path, x0: int, x1: int, out: Path, proj_root: Path | None = None) -> dict:
    """`aimv paint window`: street columns [x0, x1) at full resolution and full height (PNG) and its placement
    (the same name, .json) with the patch edit to add to street.json once the edited copy exists."""
    s = Street(d)
    if x0 < s.X0 or x1 > s.W or x1 <= x0:
        raise SystemExit(f"window {x0} to {x1}: the street runs from {s.X0} to {s.W}")
    img = s.window(x0, x1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.ascontiguousarray(img[..., ::-1]))
    b = read_json(d / "build.json")["street"]
    over = [p["patch"] for p in b.get("patches", []) if p["window"][0] < x1 and p["window"][1] > x0]
    rel = lambda f: str(f.relative_to(proj_root)) if proj_root and f.is_relative_to(proj_root) else str(f)
    edited = out.with_name(out.stem + "-edit" + out.suffix)
    info = {"street": d.name, "window": [x0, x1], "size": [x1 - x0, s.H], "image": rel(out),
            "tiles": {str(k): s.tile_hash(k) for k in range(x0 // s.tile, (x1 - 1) // s.tile + 1)},
            "patched": over,
            "edit": {"patch": rel(edited), "window": [x0, x1]}}
    write_json(out.with_suffix(".json"), info)
    return info


# ------------------------------------------------------------------------------------------------ reading

class Street:
    """The built street, read by windows. Only the tiles a window needs are loaded (small LRU). Street x runs from X0
    (0, or negative with prepended plates) to W, rows from 0 to H. With a world (world.py) the street px extend in two
    dimensions: rows above the street are negative, and `bounds` (x0, y0, x1, y1) holds the street and the world;
    a window then has valid and invalid pixels (world=False reads the street alone)."""

    def __init__(self, d: Path, cache_tiles: int = 6, world: bool = True):
        self.d = Path(d)
        b = self.d / "build.json"
        if not b.exists():
            raise SystemExit(f"street not built: run `aimv paint street` ({b} missing)")
        build_info = read_json(b)
        info = build_info["street"]
        self.W, self.H = info["size"]
        self.tile = int(info["tile"])
        self.hashes = {int(k): v for k, v in info["tiles"].items()}  # right of x 0
        self.n = len(self.hashes)
        # the tiles right of x 0 as the right side's plates alone made them (the join patch left out): the light
        # maps of the old street are computed from these
        self.light_tiles = {int(k): v for k, v in (info.get("light_tiles") or {}).items()}
        self.light_hashes = {**self.hashes, **self.light_tiles}
        left = info.get("left") or {}
        self.X0 = -int(left["size"][0]) if left else 0
        self.left_hashes = {int(k): v for k, v in (left.get("tiles") or {}).items()}
        self.k_min = min(self.left_hashes, default=0)
        self._lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cap = cache_tiles
        # the world round the street (world.py): an overlay of RGBA tiles on the street's own tile grid
        self.world = None
        if world and build_info.get("world") and (self.d / "world" / "world.json").exists():
            from .world import WorldTiles
            w = WorldTiles(self.d)
            self.world = w if w.overlay else None
        self.bands: list[tuple[int, int, int, int]] = []  # world cells above the street's left part (v8, P2b)
        if self.world is not None:
            self.bounds = tuple(int(v) for v in self.world.bounds)
            # the world's first column: where it starts to replace the street (cells over the street's left part,
            # above its rows, do not move it: world.json "xw" when the world has them, else the overlay's start)
            xw = self.world.info.get("xw")
            self.xw = min(int(xw if xw is not None else self.world.overlay[0]), self.W)
            self.bands = [tuple(int(v) for v in b) for b in self.world.info.get("bands") or []]
        else:
            self.bounds = (self.X0, 0, self.W, self.H)
            self.xw = None
        self.Y0, self.Y1 = self.bounds[1], self.bounds[3]

    def tile_span(self, k: int) -> tuple[int, int]:
        """Street x range [x0, x1) of tile k."""
        if k >= 0:
            return k * self.tile, min((k + 1) * self.tile, self.W)
        return max(k * self.tile, self.X0), (k + 1) * self.tile

    def tile_hash(self, k: int) -> str | None:
        return self.hashes.get(k) if k >= 0 else self.left_hashes.get(k)

    def tile_rgb(self, k: int) -> np.ndarray:
        if k in self._lru:
            self._lru.move_to_end(k)
            return self._lru[k]
        img = cv2.imread(str(self.d / "tiles" / tile_name("t", k)))
        if img is None:
            raise SystemExit(f"missing street tile {k} in {self.d / 'tiles'}")
        img = np.ascontiguousarray(img[..., ::-1])
        self._lru[k] = img
        while len(self._lru) > self._cap:
            self._lru.popitem(last=False)
        return img

    def light_rgb(self, k: int) -> np.ndarray:
        """Tile k (k >= 0) as the right side's plates alone made it (for the light maps of the street right of x 0)."""
        if k not in self.light_tiles:
            return self.tile_rgb(k)
        img = cv2.imread(str(self.d / "tiles" / "pure" / tile_name("t", k)))
        if img is None:
            raise SystemExit(f"missing pure street tile {k} in {self.d / 'tiles' / 'pure'}")
        return np.ascontiguousarray(img[..., ::-1])

    def street_only(self, x0: int, x1: int, y0: int = 0, y1: int | None = None) -> bool:
        """True when the box [x0, x1) x [y0, y1) is the plain street: rows inside [0, H) and left of the world (the
        1D street's own code paths apply, unchanged). World cells over the street's left part (`bands`) write
        nothing in the street's rows, so a box inside them is the plain street there too."""
        y1 = self.H if y1 is None else y1
        return 0 <= y0 and y1 <= self.H and (self.world is None or x1 <= self.xw)

    def window(self, x0: int, x1: int, y0: int | None = None, y1: int | None = None, valid: bool = False,
               fill: bool = True):
        """RGB uint8 columns [x0, x1) and rows [y0, y1) (default 0 to H). On the plain street the columns are
        edge-replicated outside it (as always); where the box reaches the world, the street and the world overlay,
        and pixels neither holds are invalid (filled from the nearest valid pixel unless fill=False). valid=True also
        returns the validity mask (bool)."""
        r0, r1 = (0 if y0 is None else int(y0)), (self.H if y1 is None else int(y1))
        if self.street_only(x0, x1, r0, r1):
            out = self._window1d(x0, x1)
            if (r0, r1) != (0, self.H):
                out = out[r0:r1]
            if not valid:
                return out
            v = np.zeros(out.shape[:2], bool)
            v[:, max(self.X0 - x0, 0):max(min(self.W, x1) - x0, 0)] = True
            return out, v
        h, w = r1 - r0, x1 - x0
        out = np.zeros((h, w, 3), np.uint8)
        v = np.zeros((h, w), bool)
        sa, sb = max(x0, self.X0), min(x1, self.W)
        ra, rb = max(r0, 0), min(r1, self.H)
        if sb > sa and rb > ra:
            out[ra - r0:rb - r0, sa - x0:sb - x0] = self._window1d(sa, sb)[ra:rb]
            v[ra - r0:rb - r0, sa - x0:sb - x0] = True
        if self.world is not None:
            rgb, al = self.world.read(x0, x1, r0, r1)
            out[al] = rgb[al]
            v |= al
        if fill and not v.all():
            from .world import fill_nearest
            out = fill_nearest(out, v)
        return (out, v) if valid else out

    def valid_box(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True when every pixel of the box is valid (street or world; the world is checked at 1/8, conservatively)."""
        if self.X0 <= x0 and x1 <= self.W and 0 <= y0 and y1 <= self.H:
            return True
        if self.world is None:
            return False
        v8, gx0, gy0 = self.world.valid8()
        i0, i1 = int(np.floor((y0 - gy0) / 8)), int(np.ceil((y1 - gy0) / 8))
        j0, j1 = int(np.floor((x0 - gx0) / 8)), int(np.ceil((x1 - gx0) / 8))
        if i0 < 0 or j0 < 0 or i1 > v8.shape[0] or j1 > v8.shape[1]:
            return False
        # a block passes when the part of the box inside it lies in the street (exact), or when valid8 has it whole
        bx = gx0 + 8 * np.arange(j0, j1)
        by = gy0 + 8 * np.arange(i0, i1)
        colok = (np.maximum(x0, bx) >= self.X0) & (np.minimum(x1, bx + 8) <= self.W)
        rowok = (np.maximum(y0, by) >= 0) & (np.minimum(y1, by + 8) <= self.H)
        return bool((v8[i0:i1, j0:j1] | (rowok[:, None] & colok[None, :])).all())

    def _window1d(self, x0: int, x1: int) -> np.ndarray:
        """RGB uint8 columns [x0, x1), full height; edge-replicated outside the street."""
        cols = np.clip(np.arange(x0, x1), self.X0, self.W - 1)
        a0, b0 = int(cols[0]), int(cols[-1]) + 1
        out = np.empty((self.H, b0 - a0, 3), np.uint8)
        for k in range(a0 // self.tile, (b0 - 1) // self.tile + 1):
            t = self.tile_rgb(k)
            s = self.tile_span(k)[0]
            a, b = max(a0, s), min(b0, s + t.shape[1])
            out[:, a - a0:b - a0] = t[:, a - s:b - s]
        if a0 == x0 and b0 == x1:
            return out
        return out[:, cols - a0]

    def overview(self, scale: int = 8, left: bool = False) -> np.ndarray:
        """The street at 1/scale (RGB uint8), tile by tile (cheap enough for maps and low-frequency light): right of
        x 0, or (left=True) all of it from X0; overview_x0 gives its left end in street px."""
        parts = []
        for k in range(self.k_min if left else 0, self.n):
            t = self.tile_rgb(k)
            parts.append(cv2.resize(t, (t.shape[1] // scale, self.H // scale), interpolation=cv2.INTER_AREA))
        return np.concatenate(parts, axis=1)

    def overview_x0(self, scale: int = 8) -> int:
        """Street x of the left end of overview(scale, left=True)."""
        return -sum((self.tile_span(k)[1] - self.tile_span(k)[0]) // scale for k in range(self.k_min, 0)) * scale

    def overview2d(self, scale: int = 8, box: tuple | None = None, linear: bool = False):
        """The street and the world at 1/scale over `box` (default: all of it) on a grid aligned to multiples of
        scale: (image, cover, ox, oy). image is the mean of each block's valid pixels (RGB uint8, or linear light
        float32 with linear=True), cover the share of its pixels that are valid, (ox, oy) the street px of the
        grid's origin."""
        from .scene import srgb2lin

        bx0, by0, bx1, by1 = box or self.bounds
        ox, oy = int(np.floor(bx0 / scale)) * scale, int(np.floor(by0 / scale)) * scale
        w, h = -(-(int(bx1) - ox) // scale), -(-(int(by1) - oy) // scale)
        img = np.zeros((h, w, 3), np.float32)
        cov = np.zeros((h, w), np.float32)
        step = max(scale, (2048 // scale) * scale)
        for c0 in range(ox, ox + w * scale, step):
            c1 = min(c0 + step, ox + w * scale)
            rgb, v = self.window(c0, c1, oy, oy + h * scale, valid=True, fill=False)
            px = srgb2lin(rgb) if linear else rgb.astype(np.float32)
            vf = v.astype(np.float32)
            nb = (c1 - c0) // scale
            s = (px * vf[..., None]).reshape(h, scale, nb, scale, 3).sum(axis=(1, 3))
            n = vf.reshape(h, scale, nb, scale).sum(axis=(1, 3))
            j = (c0 - ox) // scale
            img[:, j:j + nb] = s / np.maximum(n, 1)[..., None]
            cov[:, j:j + nb] = n / (scale * scale)
        if not linear:
            img = np.clip(img + 0.5, 0, 255).astype(np.uint8)
        return img, cov, ox, oy


def info_text(d: Path) -> str:
    b = read_json(d / "build.json")
    s, st = b["strip"], b["street"]
    lines = [f"plates: {len(s['plates'])} ({', '.join(Path(p['path']).stem for p in s['plates'])})",
             f"strip:  {s['size'][0]} x {s['size'][1]} px, {len(s['tiles'])} tiles",
             f"street: {st['size'][0]} x {st['size'][1]} px, {len(st['tiles'])} tiles, {len(st['pieces'])} pieces"]
    for j, sm in zip(s["joins"], s["seams"]):
        lines.append(f"  {j['join']}: {j['inliers']} inliers, median {j['median_px']} px, scale {j['scale']}, "
                     f"seam at strip x {sm['x_mean']:.0f}")
    if s.get("left"):
        sl, stl = s["left"], st["left"]
        lines.append(f"left:   {len(sl['plates'])} plates ({', '.join(Path(p['path']).stem for p in sl['plates'])}), "
                     f"strip {sl['size'][0]} px, street {stl['size'][0]} px (x {stl['x_min']} to 0), "
                     f"{len(stl['tiles'])} tiles, {len(stl['pieces'])} pieces")
        for j, sm in zip(sl["joins"], sl["seams"]):
            lines.append(f"  {j['join']}: {j['inliers']} inliers, median {j['median_px']} px, scale {j['scale']}, "
                         f"seam at strip x {sm['x_mean']:.0f}")
        lines.append(f"whole street: x {stl['x_min']} to {st['size'][0]} ({st['size'][0] - stl['x_min']} px)")
    return "\n".join(lines)


def seams_preview(d: Path, out: Path) -> Path:
    """Full-resolution crops around every plate join (strip coordinates), for checking (left joins first)."""
    b = read_json(d / "build.json")["strip"]
    tile = b["params"]["tile"]
    h = b["size"][1]
    crops = []
    seams = list(reversed((b.get("left") or {}).get("seams", []))) + list(b["seams"])
    for sm in seams:
        cx = int(round(sm["x_mean"]))
        c = read_strip(d, tile, cx - 500, cx + 500, h).copy()
        cv2.line(c, (500, 0), (500, 30), (0, 0, 255), 3)
        crops.append(cv2.resize(c, (500, h // 2)))
    if crops:
        cv2.imwrite(str(out), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
