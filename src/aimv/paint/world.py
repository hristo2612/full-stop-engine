"""The world: street plates placed in two dimensions round the street (cells above, below and right of an anchor
plate), for a film that leaves the street level.

Spec: the `world` section of street.json

    "world": {"anchor": "boards/11-verse2/V4.png",          a plate of `plates` (unedited where the cells meet it)
              "cells": [{"id": "T1", "path": "boards/18-world/T1.png", "at": [2304, 0]},
                        {"id": "T2u", "path": "boards/18-world/T2u.png", "at": [4608, -1501], "from": ["T2", "T1"]},
                        ...],
              "seam_feather": 24, "gain_fade": 1400}              optional (the street's own by default)

`at` is a cell's nominal top-left in anchor px (the anchor plate's own pixels: its top-left is (0, 0), x right, y down),
the grid the cells were outpainted on. `from` names the cells (or the anchor, by its file stem) it was outpainted from
(default: every placed cell and the anchor its grid box overlaps). `place: "nominal"` skips the feature match (the
prediction below is used as it is).

Build (`aimv paint street`, after the street): the anchor's strip transform maps anchor px to street px, so a cell's
grid placement is that map after its `at`. Each cell, in order, is predicted from the placed neighbour its grid box
overlaps most (its grid offset from that neighbour, applied to the neighbour's placement: the drift of the cells
before it carried along), then placed by a SIFT + RANSAC similarity against the world composed so far (the street and
the cells before it) in its overlap with its `from` neighbours, matches gated to GATE px of the prediction. It is
gain-matched (a low-frequency gain per side, faded out over `gain_fade` px into the cell) and joined on one
minimum-difference seam per side: a vertical seam in the overlap with a neighbour on its left or right, a horizontal
one with a neighbour above or below (an L of neighbours gets both), each in the middle of its overlap and feathered
over `seam_feather` px.

Storage: the world is an overlay on the street, never a rewrite of it. strip/ and tiles/ stay as the street build made
them; world/w{ix}_{iy}.png are RGBA tiles of side `tile` on the street's own tile grid (tile (ix, iy) covers street x
[ix * tile, (ix + 1) * tile) and y [iy * tile, (iy + 1) * tile)), alpha 255 where the world defines the pixel: the
cells, and the street pixels a seam changed (the anchor's overlap with the first cells). world/world.json holds the
placements, seams and tile hashes; world/valid8.png the world and street together at 1/8 (a block is valid when all
its pixels are). Adding cells at the end only rewrites the tiles they reach, so the other tiles keep their hashes (and
their scene caches).
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from ..util import read_json, sha256_file, write_json

GATE = 80.0  # px: a feature match must agree with the predicted placement within this
MARGIN = 96  # px of drift room round a cell's nominal box
FLOOR_SKIP = 64  # a floor cell is placed on the street's rows from this far under its floor (a street patch feathers
#                  its seam into the rows above it; the placement must not move when that patch is redone)
VERSION = "world-1"
SKY_ALPHA = 254  # the sky step's pixels: told from the cells' own (255), so the step can be redone
SKY_VERSION = 2  # the sky step is redone when its code changes
PROP_ALPHA = 253  # the prop step's pixels (a prop, its contact shadow): restored from world/props/under.npz
PROPS_VERSION = 1


def wtile_name(ix: int, iy: int) -> str:
    return f"w{ix}_{iy}.png"


def _wkey(k: str) -> tuple[int, int]:
    a, b = k.split("_")
    return int(a), int(b)


# ------------------------------------------------------------------------------------------------ spec

def load_spec(spec: dict) -> dict | None:
    """The street spec's world section, checked and with its defaults (None without one)."""
    w = spec.get("world")
    if not w:
        return None
    if not w.get("anchor"):
        raise SystemExit("street.json world: give its anchor (one of the plates)")
    anchor_id = Path(w["anchor"]).stem
    cells, seen = [], {anchor_id}
    for c in w.get("cells") or []:
        if "path" not in c:
            raise SystemExit(f"street.json world: a cell needs its path: {c}")
        cid = str(c.get("id") or Path(c["path"]).stem)
        at = c.get("at")
        if not (isinstance(at, list) and len(at) == 2):
            raise SystemExit(f"street.json world: cell {cid} needs at: [x, y] (anchor px, its top-left)")
        if cid in seen:
            raise SystemExit(f"street.json world: cell id {cid} used twice (or the anchor's)")
        frm = c.get("from")
        if frm is not None:
            frm = [str(f) for f in frm]
            bad = [f for f in frm if f not in seen]
            if bad:
                raise SystemExit(f"street.json world: cell {cid} comes from {', '.join(bad)}, which are not placed "
                                 f"before it")
        place = str(c.get("place", "sift"))
        if place not in ("sift", "nominal"):
            raise SystemExit(f"street.json world: cell {cid}: place is sift or nominal, not {place!r}")
        floor = c.get("floor")
        if floor is not None and not isinstance(floor, (int, float)):
            raise SystemExit(f"street.json world: cell {cid}: floor is a street row (y), not {floor!r}")
        cells.append({"id": cid, "path": str(c["path"]), "at": [float(at[0]), float(at[1])], "from": frm,
                      "place": place, **({"floor": int(floor)} if floor is not None else {})})
        seen.add(cid)
    sky = []
    for k, r in enumerate(w.get("sky") or []):
        x = r.get("x")
        if not (isinstance(x, list) and len(x) == 2 and x[1] > x[0]):
            raise SystemExit(f"street.json world: sky {k} needs x: [x0, x1]")
        sky.append({"x": [int(x[0]), int(x[1])], "up": int(r.get("up", 320)), "src": int(r.get("src", 150)),
                    "taper": [int(v) for v in r.get("taper", [400, 400])], "seam": int(r.get("seam", 40)),
                    "darken": float(r.get("darken", 0.18)), "borrow": [[int(a), int(b)] for a, b in r.get("borrow", [])],
                    "paint_out": [[int(a), int(b), int(c)] for a, b, c in r.get("paint_out", [])]})
    props, pids = [], set()
    for p in w.get("props") or []:
        miss = [k for k in ("id", "kit", "x", "base") if k not in p]
        if miss:
            raise SystemExit(f"street.json world: a prop needs {', '.join(miss)}: {p}")
        if str(p["id"]) in pids:
            raise SystemExit(f"street.json world: prop id {p['id']} used twice")
        pids.add(str(p["id"]))
        props.append({"id": str(p["id"]), "kit": str(p["kit"]), "x": int(p["x"]), "base": int(p["base"]),
                      "flip": bool(p.get("flip", False)), "shadow": float(p.get("shadow", 0.65)),
                      "front_rails": bool(p.get("front_rails", False)),
                      "front_cols": [[int(a), int(b)] for a, b in p.get("front_cols", [])],
                      "highlights": float(p.get("highlights", 1.0)), "warm": float(p.get("warm", 0.0)),
                      **({"floor": int(p["floor"])} if p.get("floor") is not None else {})})
    inserts = []
    for k, e in enumerate(w.get("inserts") or []):
        if not isinstance(e.get("x"), (int, float)) or not isinstance(e.get("length"), (int, float)) or e["length"] == 0:
            raise SystemExit(f"street.json world: insert {k} needs x and a length (> 0 opens the world, < 0 cuts it): "
                             f"{e}")
        pats = []
        for q in e.get("patches") or []:
            if "path" not in q or not (isinstance(q.get("window"), list) and len(q["window"]) == 4):
                raise SystemExit(f"street.json world: insert {k}: a patch needs path and window [x0, y0, x1, y1]")
            if q.get("align") not in (None, "sift", "none"):
                raise SystemExit(f"street.json world: insert {k}: a patch's align is sift or none, not {q['align']!r}")
            pats.append({"path": str(q["path"]), "window": [int(v) for v in q["window"]],
                         "region": [int(v) for v in q.get("region", q["window"])],
                         **({"align": "none"} if q.get("align") == "none" else {}),
                         **({"tone": False} if q.get("tone") is False else {})})
        inserts.append({"x": int(e["x"]), "length": int(e["length"]), "patches": pats,
                        **({"clean": True} if e.get("clean") else {})})
    if len({e["x"] for e in inserts}) < len(inserts):
        raise SystemExit("street.json world: two inserts at one x")
    for e in inserts:  # a cut takes [x, x - length) out: nothing else may start inside it
        if e["length"] < 0 and any(e["x"] < f["x"] < e["x"] - e["length"] for f in inserts):
            raise SystemExit(f"street.json world: an insert lies inside the cut at {e['x']}")
    grade = []
    for k, g in enumerate(w.get("grade") or []):
        gx, gy = g.get("x"), g.get("y")
        if not (isinstance(gx, list) and len(gx) == 2 and gx[1] > gx[0] and isinstance(gy, list) and len(gy) == 2
                and gy[1] > gy[0]):
            raise SystemExit(f"street.json world: grade {k} needs x: [x0, x1] and y: [y0, y1]")
        grade.append({"x": [int(gx[0]), int(gx[1])], "y": [int(gy[0]), int(gy[1])],
                      "falloff": int(g.get("falloff", 600)), "below": int(g.get("below", g.get("falloff", 600))),
                      "luma": round(float(g.get("luma", 30)), 2),
                      "color": [int(v) for v in g.get("color", [255, 125, 55])],
                      "texture": round(float(g.get("texture", 1.0)), 3)})  # (rounded: world.json keeps 9 digits)
    return {"anchor": str(w["anchor"]), "anchor_id": anchor_id, "cells": cells, "sky": sky, "props": props,
            "inserts": sorted(inserts, key=lambda e: e["x"]), "grade": grade,
            "props_kit": str(w.get("props_kit", "boards/26-props/cut")),
            "seam_feather": int(w.get("seam_feather", spec.get("seam_feather", 24))),
            "gain_fade": float(w.get("gain_fade", spec.get("gain_fade", 1400)))}


def anchor_transform(spec: dict, strip: dict) -> tuple[np.ndarray, list[int]]:
    """Similarity (2 x 3) from anchor plate px to street px, and the plate's size: its strip transform, the strip's
    crop, and the street piece holding its left half (it must be plain: not cut, mirrored or stretched there)."""
    from .street import OY, _piece_offsets, _side_edits, pieces

    a = spec["world"]["anchor"]
    plates = [str(p) for p in spec["plates"]]
    if a not in plates:
        raise SystemExit(f"street.json world: the anchor {a} is not one of the plates (plates, right of x 0)")
    i = plates.index(a)
    T = np.round(np.float64(strip["plates"][i]["T"]), 4)  # as build.json keeps it (the same after a reload)
    cx0, cy0 = strip["crop"][0], strip["crop"][1]
    W, H = strip["plate_size"]
    F = int(spec["edit_feather"])
    ps = pieces(strip["size"][0], _side_edits(spec, 1), F)
    xs = [T[0, 0] * u + T[0, 1] * v + T[0, 2] - cx0 for u in (0, W / 2) for v in (0, H)]
    a0, a1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    off = None
    for p, X in zip(ps, _piece_offsets(ps, F)):
        if p.plain and p.src0 <= max(a0, 0) and a1 <= p.src1:
            off = X + p.ext_l - p.src0
    if off is None:
        raise SystemExit(f"street.json world: the anchor {Path(a).stem} is cut or edited by the street edits; the "
                         f"world needs its left half whole")
    A = T.copy()
    A[0, 2] += off - cx0
    A[1, 2] += OY - cy0
    return A, [int(W), int(H)]


def _compose(M: np.ndarray, at: tuple[float, float]) -> np.ndarray:
    """M after a translation by `at` (cell px to anchor px), as 2 x 3."""
    return (np.vstack([M, [0, 0, 1]]) @ np.float64([[1, 0, at[0]], [0, 1, at[1]], [0, 0, 1]]))[:2]


def _bbox(M: np.ndarray, w: int, h: int, m: int = 0) -> tuple[int, int, int, int]:
    pts = np.float64([[0, 0], [w, 0], [0, h], [w, h]]) @ M[:, :2].T + M[:, 2]
    return (int(np.floor(pts[:, 0].min())) - m, int(np.floor(pts[:, 1].min())) - m,
            int(np.ceil(pts[:, 0].max())) + m, int(np.ceil(pts[:, 1].max())) + m)


def _local(M: np.ndarray, bx: int, by: int) -> np.ndarray:
    Ml = M.astype(np.float64).copy()
    Ml[0, 2] -= bx
    Ml[1, 2] -= by
    return Ml


def _coverage(M: np.ndarray, w: int, h: int, bx: int, by: int, bw: int, bh: int, erode: int = 5) -> np.ndarray:
    """Where a w x h image placed by M (street px) covers the box [bx, bx + bw) x [by, by + bh) (bool)."""
    v = cv2.warpAffine(np.ones((h, w), np.float32), _local(M, bx, by), (bw, bh), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    v = (v > 0.999).astype(np.uint8)
    if erode:
        v = cv2.erode(v, np.ones((erode, erode), np.uint8))
    return v > 0


# ------------------------------------------------------------------------------------------------ the overlay

class Overlay:
    """The world overlay while it is built: tiles of BGR + alpha (uint8, side T), in memory up to `cap` tiles and
    spilled to world/.spill beyond that; `start` holds the tiles of the last build to continue from."""

    def __init__(self, wd: Path, T: int, start: dict[tuple[int, int], str] | None = None, cap: int = 20):
        self.wd, self.T, self.cap = wd, T, cap
        self.start = dict(start or {})  # (ix, iy) -> hash of a tile on disk to continue from
        self.mem: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.spill = wd / ".spill"
        self.spilled: set[tuple[int, int]] = set()

    def keys(self) -> set[tuple[int, int]]:
        return set(self.mem) | self.spilled | set(self.start)

    def get(self, k: tuple[int, int], create: bool = False) -> np.ndarray | None:
        if k in self.mem:
            self.mem.move_to_end(k)
            return self.mem[k]
        t = None
        if k in self.spilled:
            t = np.load(self.spill / f"{k[0]}_{k[1]}.npy")
        elif k in self.start:
            t = cv2.imread(str(self.wd / wtile_name(*k)), cv2.IMREAD_UNCHANGED)
            if t is None or t.shape != (self.T, self.T, 4):
                raise SystemExit(f"world: tile {wtile_name(*k)} is missing or damaged; rebuild the world")
        elif create:
            t = np.zeros((self.T, self.T, 4), np.uint8)
        if t is None:
            return None
        self.mem[k] = t
        while len(self.mem) > self.cap:
            kk, tt = self.mem.popitem(last=False)
            self.spill.mkdir(parents=True, exist_ok=True)
            np.save(self.spill / f"{kk[0]}_{kk[1]}.npy", tt)
            self.spilled.add(kk)
        return t

    def read(self, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """BGR uint8 and alpha (bool) over [x0, x1) x [y0, y1)."""
        T = self.T
        out = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        al = np.zeros((y1 - y0, x1 - x0), bool)
        for iy in range(y0 // T, (y1 - 1) // T + 1):
            for ix in range(x0 // T, (x1 - 1) // T + 1):
                t = self.get((ix, iy))
                if t is None:
                    continue
                a0, b0 = max(x0, ix * T), max(y0, iy * T)
                a1, b1 = min(x1, (ix + 1) * T), min(y1, (iy + 1) * T)
                sub = t[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
                out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = sub[..., :3]
                al[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = sub[..., 3] > 0
        return out, al

    def read_alpha(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """The raw alpha (uint8: 255 cells, SKY_ALPHA, PROP_ALPHA, 0 none) over [x0, x1) x [y0, y1)."""
        T = self.T
        out = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for iy in range(y0 // T, (y1 - 1) // T + 1):
            for ix in range(x0 // T, (x1 - 1) // T + 1):
                t = self.get((ix, iy))
                if t is None:
                    continue
                a0, b0 = max(x0, ix * T), max(y0, iy * T)
                a1, b1 = min(x1, (ix + 1) * T), min(y1, (iy + 1) * T)
                out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = t[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T, 3]
        return out

    def write(self, x0: int, y0: int, bgr: np.ndarray, alpha: np.ndarray, value: int = 255) -> None:
        """Set the pixels where alpha is True (BGR uint8 over a box at (x0, y0)) and mark them defined (alpha
        `value`: 255 for the cells, SKY_ALPHA for the sky step)."""
        T = self.T
        h, w = alpha.shape
        x1, y1 = x0 + w, y0 + h
        for iy in range(y0 // T, (y1 - 1) // T + 1):
            for ix in range(x0 // T, (x1 - 1) // T + 1):
                a0, b0 = max(x0, ix * T), max(y0, iy * T)
                a1, b1 = min(x1, (ix + 1) * T), min(y1, (iy + 1) * T)
                m = alpha[b0 - y0:b1 - y0, a0 - x0:a1 - x0]
                if not m.any():
                    continue
                t = self.get((ix, iy), create=True)
                sub = t[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
                sub[m, :3] = bgr[b0 - y0:b1 - y0, a0 - x0:a1 - x0][m]
                sub[m, 3] = value

    def finish(self, old: dict[tuple[int, int], str]) -> dict[tuple[int, int], str]:
        """Write every tile that changed (PNG), drop the stale ones, return the hashes."""
        hashes = {}
        for k in sorted(self.keys()):
            f = self.wd / wtile_name(*k)
            if k in self.start and k not in self.mem and k not in self.spilled:
                hashes[k] = self.start[k]  # never touched by this build
                continue
            t = self.get(k)
            if t is None or not t[..., 3].any():
                continue
            h = hashlib.sha256(t.tobytes()).hexdigest()[:16]
            if old.get(k) != h or not f.exists():
                cv2.imwrite(str(f), t)
            hashes[k] = h
        for f in self.wd.glob("w*.png"):
            try:
                k = _wkey(f.stem[1:])
            except ValueError:
                continue
            if k not in hashes:
                f.unlink()
        if self.spill.exists():
            for f in self.spill.glob("*.npy"):
                f.unlink()
            self.spill.rmdir()
        return hashes


# ------------------------------------------------------------------------------------------------ sky

def sky_band(img: np.ndarray, top: np.ndarray, r: dict, x0: int = 0, y0: int = 0,
             defined: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The open sky continued above the world's top edge (street.json world `sky`): img (H x W x 3, float) over
    street x x0.. and y y0.., top the first row each column defines (-1: none). Each column repeats its own band of
    `src` rows under the top edge upward (a shift, so the rain keeps its slant), `up` rows at most, tapered to nothing
    over `taper` [left, right] px at the region's ends; the first `seam` rows cross-fade from a reflection of the
    column's top rows; `darken` going up. paint_out [[x0, x1, y_low], ...]: structures reaching into the band (an
    antenna's tip) painted out of the source first; borrow [[from_x, dx], ...]: from street x from_x on, the band
    comes from dx px away (far buildings reaching into it there). A column whose cells already reach well above the
    region's edge (a sky row's cell) takes its nearest neighbour's edge, and `defined` pixels are never painted over
    (so only its holes are filled). Returns the image and the pixels added."""
    H, W = img.shape[:2]
    up, src, seam, dk = r["up"], r["src"], max(r["seam"], 1), r["darken"]
    tl, tr = max(r["taper"][0], 1), max(r["taper"][1], 1)
    rx0, rx1 = r["x"]
    srcimg = img.copy()
    for j in range(W):  # above the top edge: the column's top pixel, so painting out sees sky, not black
        if top[j] >= 0:
            srcimg[:top[j], j] = img[top[j], j]
    for px0, px1, ylow in r["paint_out"]:
        j0, j1 = max(px0 - x0, 0), min(px1 - x0, W)
        i1 = min(max(ylow - y0, 0), H)
        if j1 <= j0 or i1 == 0:
            continue
        w0, w1 = max(j0 - 200, 0), min(j1 + 200, W)
        m = np.zeros((H, w1 - w0), np.uint8)
        m[:i1, j0 - w0:j1 - w0] = 255
        sub = np.clip(srcimg[:, w0:w1], 0, 255).astype(np.uint8)
        srcimg[:, w0:w1] = cv2.inpaint(sub, m, 9, cv2.INPAINT_TELEA).astype(np.float32)
    out = img.copy()
    added = np.zeros(img.shape[:2], bool)
    top = top.copy()
    has = top >= 0
    if has.any():
        hi = has & (top < float(np.median(top[has])) - up / 2)
        lo = np.nonzero(has & ~hi)[0]
        for j in np.nonzero(hi)[0] if len(lo) else []:
            top[j] = top[lo[np.argmin(np.abs(lo - j))]]
    for j in range(W):
        t = int(top[j])
        if t < 0:
            continue
        x = x0 + j
        js = j
        for fx, dx in r["borrow"]:
            if x >= fx and 0 <= j + dx < W and top[j + dx] >= 0:
                js = j + dx
        ts = int(top[js])
        S = srcimg[ts:ts + src, js]
        if len(S) < src:
            continue
        u = int(up * min(1.0, max(0.0, (rx1 - x) / tr), max(0.0, (x - rx0) / tl)))
        for d in range(1, min(u, t) + 1):
            if defined is not None and defined[t - d, j]:
                continue
            c = S[(src - d % src) % src]
            if d <= seam:
                w = d / seam
                c = (1 - w) * img[min(t + d - 1, H - 1), j] + w * c
            out[t - d, j] = c * (1 - dk * d / up)
            added[t - d, j] = True
    # the added rows' fine detail kept, their tone smoothed across the columns (the borrowed band's joins)
    sm = cv2.GaussianBlur(out, (0, 0), 6)
    hi = out - sm
    out = np.where(added[..., None], sm + hi, out)
    return out, added


def clear_sky(ov: "Overlay", regions: list[dict]) -> None:
    """Undo the sky step over these regions' tile columns (its pixels have alpha SKY_ALPHA)."""
    T = ov.T
    cols = {ix for r in regions for ix in range(r["x"][0] // T, (r["x"][1] - 1) // T + 1)}
    for k in sorted(ov.keys()):
        if k[0] in cols:
            t = ov.get(k)
            if t is not None and (t[..., 3] == SKY_ALPHA).any():
                m = t[..., 3] == SKY_ALPHA
                t[m] = 0


def apply_sky(ov: "Overlay", r: dict) -> int:
    """The sky step over one region (sky_band on the overlay's top edge there); returns the pixels added."""
    T = ov.T
    x0, x1 = r["x"]
    ixs = list(range(x0 // T, (x1 - 1) // T + 1))
    iys = sorted({k[1] for k in ov.keys() if k[0] in ixs})
    if not iys:
        return 0
    W = x1 - x0
    top = np.zeros(W, np.int64)
    ok = np.zeros(W, bool)
    for iy in iys:  # from the top tile row down: each column's first row the cells define
        for ix in ixs:
            t = ov.get((ix, iy))
            if t is None:
                continue
            a0, a1 = max(x0, ix * T), min(x1, (ix + 1) * T)
            sub = t[:, a0 - ix * T:a1 - ix * T, 3] == 255
            has, first = sub.any(0), sub.argmax(0)
            cols = np.arange(a0 - x0, a1 - x0)
            sel = ~ok[cols] & has
            top[cols[sel]] = iy * T + first[sel]
            ok[cols[sel]] = True
    if not ok.any():
        return 0
    edge = ok & (top >= np.median(top[ok]) - r["up"] / 2)  # not the sky rows' cells reaching far higher
    ylo, yhi = int(top[edge].min()) - r["up"] - 1, int(top[edge].max()) + r["src"] + 1
    bgr, al = ov.read(x0, ylo, x1, yhi)
    out, added = sky_band(bgr.astype(np.float32), np.where(ok, np.maximum(top - ylo, 0), -1), r, x0, ylo, al)
    ov.write(x0, ylo, np.clip(out + 0.5, 0, 255).astype(np.uint8), added, value=SKY_ALPHA)
    return int(added.sum())


# ------------------------------------------------------------------------------------------------ inserts

def _shift_left(ov: "Overlay", X: int, L: int) -> None:
    """A cut: every overlay pixel in [X, X + L) gone and every one at x >= X + L moved L px left (tile by tile from
    the left, so nothing is overwritten before it is moved)."""
    T = ov.T
    keys = ov.keys()
    if not keys:
        return
    ix1 = max(k[0] for k in keys)
    for iy in sorted({k[1] for k in keys}):
        y0, y1 = iy * T, (iy + 1) * T
        raw = ov.read_alpha(X, y0, X + L, y1)
        if raw.any():
            bgr, _ = ov.read(X, y0, X + L, y1)
            ov.write(X, y0, bgr, raw > 0, value=0)
        for ix in range((X + L) // T, ix1 + 1):
            a0, a1 = max(X + L, ix * T), (ix + 1) * T
            if a1 <= a0 or ov.get((ix, iy)) is None:
                continue
            raw = ov.read_alpha(a0, y0, a1, y1)
            if not raw.any():
                continue
            bgr, _ = ov.read(a0, y0, a1, y1)
            ov.write(a0, y0, bgr, raw > 0, value=0)
            for v in np.unique(raw[raw > 0]):
                ov.write(a0 - L, y0, bgr, raw == v, value=int(v))


def _shift_right(ov: "Overlay", X: int, L: int) -> None:
    """Every overlay pixel at x >= X moved L px right (tile by tile from the right end, so nothing is overwritten
    before it is moved); [X, X + L) is left empty."""
    T = ov.T
    keys = ov.keys()
    if not keys:
        return
    ix1 = max(k[0] for k in keys)
    for iy in sorted({k[1] for k in keys}):
        y0, y1 = iy * T, (iy + 1) * T
        for ix in range(ix1, X // T - 1, -1):
            a0, a1 = max(X, ix * T), (ix + 1) * T
            if a1 <= a0 or ov.get((ix, iy)) is None:
                continue
            raw = ov.read_alpha(a0, y0, a1, y1)
            if not raw.any():
                continue
            bgr, _ = ov.read(a0, y0, a1, y1)
            ov.write(a0, y0, bgr, raw > 0, value=0)
            for v in np.unique(raw[raw > 0]):
                ov.write(a0 + L, y0, bgr, raw == v, value=int(v))


def _luma8(bgr: np.ndarray) -> np.ndarray:
    """8-bit luma (Rec. 601) of float BGR 0..1."""
    return 255 * (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2])


def sky_glow(bgr: np.ndarray, X0: int, Y0: int, r: dict, valid: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, float]:
    """A warm glow lifted out of the sky's own clouds (street.json world `grade`): bgr (float BGR 0..1) is a window
    whose top-left is street (X0, Y0). Inside the core box r["x"] x r["y"] the weight is 1, falling off smoothly to
    0 over r["falloff"] px outside it (r["below"] px under it); it is strongest at the core's foot (the underside of a cloud bank lit by the
    city below, half as strong at its top) and follows the clouds' texture (`texture`: 1, the cloud detail lighter
    than its surroundings takes more of it, the dark gaps less); what is already brighter than `luma` takes less of it,
    nothing past 1.4 times it (the horizon band). The window is screened toward `color` (RGB) with the
    strength whose mean 8-bit luma over the core (its valid pixels) is `luma`. Returns the graded window, the weight
    map and the strength."""
    H, W = bgr.shape[:2]
    (x0, x1), (y0, y1) = r["x"], r["y"]
    F = max(float(r.get("falloff", 600)), 1.0)
    cx, cy = (x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0
    ax, ay = (x1 - x0) / 2, (y1 - y0) / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Fb = max(float(r.get("below", F)), 1.0)  # (a shorter falloff under the core: the horizon band stays)
    dx, dy = np.maximum(np.abs(xx - cx) - ax, 0), np.maximum(np.abs(yy - cy) - ay, 0)
    dy = np.where(yy > cy, dy * F / Fb, dy)
    u = np.clip(np.sqrt(dx * dx + dy * dy) / F, 0, 1)
    shape = (1 - u) ** 2 * (1 + 2 * u)
    L = _luma8(bgr)
    det = cv2.GaussianBlur(L, (0, 0), 6) - cv2.GaussianBlur(L, (0, 0), 40)
    sd = float(np.std(det[shape > 0.5])) + 1e-3 if (shape > 0.5).any() else 1.0
    tex = np.clip(0.6 + 0.4 * det / (1.5 * sd), 0.3, 1.0)
    k = float(r.get("texture", 1.0))
    foot = np.clip(0.5 + 0.5 * (yy - (cy - ay)) / max(2 * ay, 1), 0.5, 1.0)
    target = float(r.get("luma", 30))
    # it lifts the dark sky toward the glow and leaves what is already as bright alone (the horizon band below)
    room = np.clip((1.4 * target - L) / max(0.4 * target, 1.0), 0, 1)
    wmap = shape * foot * ((1 - k) + k * tex) * room
    if valid is not None:
        wmap = wmap * valid
    col = np.float32(list(r.get("color", [255, 125, 55]))[::-1]) / 255
    core = shape > 0.999
    if valid is not None:
        core &= valid > 0.5
    if not core.any():
        return bgr, wmap, 0.0

    def screen(st: float) -> np.ndarray:
        return 1 - (1 - bgr) * (1 - np.clip(st * wmap, 0, 1)[..., None] * col)
    if _luma8(bgr)[core].mean() >= target:
        return bgr, wmap, 0.0
    lo, hi = 0.0, 4.0
    for _ in range(36):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _luma8(screen(mid))[core].mean() < target else (lo, mid)
    st = (lo + hi) / 2
    return screen(st), wmap, st


def apply_grade(ov: "Overlay", grades: list[dict], log=print) -> None:
    """The grade step (street.json world `grade`), after the cells and the inserts: each region's warm glow
    (sky_glow) written into the cells' own pixels."""
    for r in grades:
        (x0, x1), (y0, y1), F = r["x"], r["y"], max(r["falloff"], r["below"])
        X0, Y0, X1, Y1 = x0 - F, y0 - F, x1 + F, y1 + F
        bgr8, al = ov.read(X0, Y0, X1, Y1)
        if not al.any():
            log(f"world: grade over x {x0} to {x1}, y {y0} to {y1}: nothing placed there")
            continue
        b = bgr8.astype(np.float32) / 255
        out, wmap, st = sky_glow(b, X0, Y0, r, al.astype(np.float32))
        new8 = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
        changed = (new8 != bgr8).any(2) & al
        ov.write(X0, Y0, new8, changed, 255)
        core = np.zeros(al.shape, bool)
        core[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
        core &= al
        log(f"world: grade over x {x0} to {x1}, y {y0} to {y1}: core luma {_luma8(b)[core].mean():.1f} to "
            f"{_luma8(out)[core].mean():.1f} (strength {st:.3f})")


def apply_inserts(ov: "Overlay", proj_root: Path, inserts: list[dict], log=print) -> None:
    """The world's inserts (street.json world `inserts`), after the cells: at each x (where the cells put it) the
    overlay opens `length` px, everything right of it moving right, in every row. The gap gets a mirrored copy of
    the world left of it (a placeholder), then each patch: an image of the world round the gap with its middle new
    (made from a canvas of the world after the insert, the middle grey), aligned by SIFT on its known parts and
    pasted in `region` like the street's patches (mask full). A negative length is a cut: [x, x - length) leaves the
    world and everything right of it moves left (its patches then make the seam). The rightmost insert goes first,
    so each x stays where the cells put it. A `clean` insert's patches take nothing from the placeholder: their tone comes only from the
    real world (and the patches before them) in their window, and over the placeholder in their region they paste
    whole, feathered only into real world (or an earlier patch)."""
    from .street import patch_layer
    shifts = {e["x"]: sum(f["length"] for f in inserts if f["x"] < e["x"]) for e in inserts}
    for e in sorted(inserts, key=lambda e: -e["x"]):
        X, L = e["x"], e["length"]
        if L < 0:  # a cut: [X, X - L) out, everything right of it L px left
            _shift_left(ov, X, -L)
            log(f"world: cut at x {X} ({X + shifts[X]} after the inserts left of it), {-L} px out")
            continue
        _shift_right(ov, X, L)
        T = ov.T
        iys = sorted({k[1] for k in ov.keys()})
        if iys:
            y0, y1 = iys[0] * T, (iys[-1] + 1) * T
            for a in range(X, X + L, T):  # the placeholder: the world left of the gap, mirrored
                w = min(T, X + L - a)
                src0 = X - (a - X) - w
                bgr, _ = ov.read(src0, y0, src0 + w, y1)
                raw = ov.read_alpha(src0, y0, src0 + w, y1)
                ov.write(a, y0, np.ascontiguousarray(bgr[:, ::-1]), raw[:, ::-1] > 0, value=255)
        log(f"world: insert at x {X} ({X + shifts[X]} after the inserts left of it), {L} px")
    for e in inserts:
        G0 = e["x"] + shifts[e["x"]]
        G1 = G0 + e["length"]  # the gap, after the inserts left of it (the patches' coordinates)
        done = None
        if e.get("clean") and e["length"] > 0:
            iys = sorted({k[1] for k in ov.keys()})
            Y0, Y1 = iys[0] * ov.T, (iys[-1] + 1) * ov.T
            done = np.zeros((Y1 - Y0, G1 - G0), bool)  # the gap's pixels a patch covered (the rest: placeholder)
        for q in e["patches"]:
            f = proj_root / q["path"]
            if not f.exists():
                log(f"world: insert at {e['x']}: no patch {q['path']} yet (the placeholder stays)")
                continue
            wx0, wy0, wx1, wy1 = q["window"]
            base, al = ov.read(wx0, wy0, wx1, wy1)
            img = cv2.imread(str(f))
            keep = np.zeros(base.shape[:2], np.uint8)
            rx0, ry0, rx1, ry1 = q["region"]
            keep[max(ry0 - wy0, 0):max(ry1 - wy0, 0), max(rx0 - wx0, 0):max(rx1 - wx0, 0)] = 1
            ph = None
            if done is not None:  # the placeholder in the window
                ph = np.zeros(keep.shape, bool)
                a0, a1, b0, b1 = max(G0, wx0), min(G1, wx1), max(Y0, wy0), min(Y1, wy1)
                if a0 < a1 and b0 < b1:
                    ph[b0 - wy0:b1 - wy0, a0 - wx0:a1 - wx0] = ~done[b0 - Y0:b1 - Y0, a0 - G0:a1 - G0]
            M = None
            if q.get("align") == "none":  # placed as it is: a plate made on a canvas of the world after the insert,
                # whose window holds nothing of the world under it to find (a cut's patches over a whole new stretch)
                if img.shape[:2] != base.shape[:2]:
                    raise SystemExit(f"world: insert patch {f.name}: align none needs it at its window's size "
                                     f"({base.shape[1]} x {base.shape[0]}), not {img.shape[1]} x {img.shape[0]}")
                M = np.float64([[1, 0, 0], [0, 1, 0]])
            elif q.get("align") not in (None, "sift"):
                raise SystemExit(f"world: insert patch {f.name}: align is sift or none, not {q.get('align')!r}")
            tone_from = None if ph is None else ~ph
            if q.get("tone") is False:  # the plate's own tone (no gain from the world under its window)
                tone_from = np.zeros(keep.shape, bool)
            r = patch_layer(base, img, mode="full", keep=keep, tone_from=tone_from, M=M)
            a = r["alpha"]
            if ph is not None:
                over = ph & (keep > 0) & (r["valid"] > 0)
                a = np.where(over, np.float32(1), a)  # the plate whole over the placeholder, no feather into it
                if a0 < a1 and b0 < b1:
                    done[b0 - Y0:b1 - Y0, a0 - G0:a1 - G0] |= over[b0 - wy0:b1 - wy0, a0 - wx0:a1 - wx0]
            new = np.clip(base.astype(np.float32) * (1 - a[..., None]) + r["layer"] * a[..., None] + 0.5, 0, 255)
            ov.write(wx0, wy0, new.astype(np.uint8), (a > 0.002) & (al | (keep > 0)), value=255)
            log(f"world: insert patch {Path(q['path']).name}: "
                + ("placed as it is" if M is not None else f"{r['inliers']} inliers, median {r['median_px']} px, "
                                                            f"scale {r['scale']}")
                + (", its own tone" if q.get("tone") is False else "") + (" (clean)" if ph is not None else ""))


# ------------------------------------------------------------------------------------------------ props

def _prop_rgba(kit: Path, kid: str, flip: bool) -> tuple[np.ndarray, dict]:
    """A kit cut-out at its real size: RGBA float32 0..1 (BGR order), its base row on its bottom edge; and its
    top surface (row, col0, col1) at that size. kit/props.json has each one's real height, base row and top."""
    meta = read_json(kit / "props.json")[kid]
    im = cv2.imread(str(kit / f"{kid}.png"), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        raise SystemExit(f"world: prop cut-out {kit / (kid + '.png')} must be RGBA")
    im = im[:int(meta["base_row"]) + 1]
    s = float(meta["real_h"]) / im.shape[0]
    im = cv2.resize(im, (max(1, round(im.shape[1] * s)), int(meta["real_h"])), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255
    w = im.shape[1]
    top = {"row": meta["top"]["row"] * s, "col0": meta["top"]["col0"] * s, "col1": meta["top"]["col1"] * s}
    if flip:
        im = im[:, ::-1].copy()
        top["col0"], top["col1"] = w - 1 - top["col1"], w - 1 - top["col0"]
    return im, top


def _prop_grade(bgr: np.ndarray, a: np.ndarray, plate: np.ndarray, near: np.ndarray,
                highlights: float = 1.0, warm: float = 0.0) -> np.ndarray:
    """A cut-out's colours matched to the plate behind it (float BGR 0..1): in Lab, black level and median to the
    plate's (the spread x1.25, so it reads in front), the plate's colour cast and at most 0.8 of its own chroma, a 5%
    veil of the plate's mean colour. highlights: its lightness times this at its own 95th percentile and above, eased
    from 1 at its median (0.9: the highlights 10% down, the darks and mid-tones as they were). warm: a cast toward
    the city's sodium light on top of the plate's (at 1: a* +3, b* +8 at its median lightness and above, eased in from
    its black level, so the blacks stay neutral)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    P, Q = cv2.cvtColor(plate, cv2.COLOR_BGR2Lab)[near], lab[a > 0.6]
    if len(P) < 50 or len(Q) < 50:
        return bgr
    pl, ql = np.percentile(P[:, 0], [5, 50]), np.percentile(Q[:, 0], [5, 50])
    s = float(np.clip((pl[1] - pl[0]) / max(ql[1] - ql[0], 1e-3) * 1.25, 0.3, 1.0))
    lab[..., 0] = np.clip(pl[0] + (lab[..., 0] - ql[0]) * s, 0, 100)
    if highlights != 1.0:
        m50, m95 = np.percentile(lab[..., 0][a > 0.6], [50, 95])
        k = np.clip((lab[..., 0] - m50) / max(m95 - m50, 1e-3), 0, 1)
        lab[..., 0] *= 1 - (1 - highlights) * k
    c = float(np.clip(np.hypot(P[:, 1], P[:, 2]).mean() / max(np.hypot(Q[:, 1], Q[:, 2]).mean(), 1e-3), 0.3, 0.8))
    for i in (1, 2):
        lab[..., i] = (lab[..., i] - Q[:, i].mean()) * c + P[:, i].mean() * 0.7
    if warm:
        l0, l50 = np.percentile(lab[..., 0][a > 0.6], [5, 50])
        k = np.clip((lab[..., 0] - l0) / max(l50 - l0, 1e-3), 0, 1) * warm
        lab[..., 1] += 3.0 * k
        lab[..., 2] += 8.0 * k
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(out * 0.95 + plate[near].mean(0) * 0.05, 0, 1)


def _rain_streaks(h: int, w: int, rng: np.random.Generator, density: float = 1 / 2600, lean: float = 15.0) -> np.ndarray:
    """Rain over a prop like the plates' own: thin bright streaks 25 to 90 px long, leaning `lean` degrees from the
    vertical, tops to the right (measured on Full Stop's sky)."""
    r = np.zeros((h, w), np.float32)
    t = np.radians(lean)
    for _ in range(rng.poisson(h * w * density)):
        x, y, L = rng.uniform(-20, w + 20), rng.uniform(-40, h), rng.uniform(25, 90)
        cv2.line(r, (int(x), int(y)), (int(x - L * np.sin(t)), int(y + L * np.cos(t))), float(rng.uniform(0.025, 0.08)),
                 1, cv2.LINE_AA)
    return cv2.GaussianBlur(r, (0, 0), 0.6)


def prop_composite(plate: np.ndarray, prop: np.ndarray, ox: int, oy: int, rng: np.random.Generator,
                   shadow: float = 0.65, front_rails: bool = False, front_cols: list | None = None,
                   highlights: float = 1.0, warm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A prop drawn into a plate window (float BGR 0..1), its top-left at (ox, oy): graded to the plate behind it,
    the plate's grain and rain over it, a wet glint on its top silhouette, ambient occlusion at its foot, a soft
    dark contact line along its base on the surface it stands on; front_rails: a guard rail in front of it (rows of
    the window whose median is a thin bright line) stays in front, and so do front_cols (window columns: tubes);
    highlights, warm: see _prop_grade. Returns the new window and the prop's drawn alpha."""
    H, W = plate.shape[:2]
    h, w = prop.shape[:2]
    base = oy + h
    a = prop[..., 3].copy()
    near = np.zeros((H, W), bool)
    near[max(base - int(1.3 * h), 0):base, max(ox - w, 0):min(ox + 2 * w, W)] = True  # what stands behind it
    near[oy:base, ox:ox + w] &= a < 0.05
    g = _prop_grade(prop[..., :3], a, plate, near, highlights, warm)
    lum = plate.mean(2)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.2)
    mad = float(np.median(np.abs(hp[near]))) * 1.4826 if near.any() else 0.005
    g = g + (_rain_streaks(h, w, rng) + rng.normal(0, mad, (h, w)).astype(np.float32))[..., None]
    solid = (a > 0.5).astype(np.uint8)
    edge = cv2.GaussianBlur((solid - cv2.erode(solid, np.ones((3, 1), np.uint8), anchor=(0, 0))).astype(np.float32),
                            (0, 0), 0.8)
    g = g + 0.07 * edge[..., None]
    yy = np.arange(h, dtype=np.float32)[:, None]
    g = np.clip(g * (1 - 0.18 * np.clip((yy - 0.88 * h) / (0.12 * h), 0, 1))[..., None], 0, 1)
    out = plate.copy()
    sh = np.zeros((H, W), np.float32)
    cols = np.nonzero(solid[-max(3, h // 20):].any(0))[0]
    fx0, fx1 = (ox + int(cols.min()), ox + int(cols.max())) if len(cols) else (ox, ox + w)
    cx, rx = (fx0 + fx1) // 2, max((fx1 - fx0) // 2, 4)
    cv2.ellipse(sh, (cx, base), (int(rx * 1.12), max(int(0.06 * h), 6)), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 0.03 * w + 3)
    core = np.zeros((H, W), np.float32)
    cv2.line(core, (fx0, base + 1), (fx1, base + 1), 1.0, 3)
    sh = np.maximum(sh, cv2.GaussianBlur(core, (0, 0), 1.5))
    out *= (1 - shadow * sh)[..., None]
    if front_rails:
        prof = np.median(lum, axis=1)
        rail = (prof - cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), sigmaX=1, sigmaY=12).ravel()) > 0.025
        rail = cv2.dilate(rail.astype(np.uint8).reshape(-1, 1), np.ones((5, 1), np.uint8)).ravel() > 0
        a[np.nonzero(rail[oy:base])[0]] = 0
    for c0, c1 in front_cols or []:
        c0, c1 = max(c0 - ox, 0), min(c1 - ox, w)
        if c1 > c0:
            a[:, c0:c1] = 0
    out[oy:base, ox:ox + w] = out[oy:base, ox:ox + w] * (1 - a[..., None]) + g * a[..., None]
    return out, a


def clear_props(ov: "Overlay", wd: Path) -> None:
    """Undo the last build's props: their pixels (alpha PROP_ALPHA) back to what was under them."""
    f = wd / "props" / "under.npz"
    if not f.exists():
        return
    z = np.load(f)
    for pid in reversed([str(v) for v in z["order"]]):
        x0, y0, x1, y1 = [int(v) for v in z[f"{pid}:box"]]
        m = z[f"{pid}:mask"]
        bgr, al = ov.read(x0, y0, x1, y1)
        raw = ov.read_alpha(x0, y0, x1, y1)
        here = m & (raw == PROP_ALPHA)
        old_bgr = np.zeros_like(bgr)
        old_bgr[m] = z[f"{pid}:bgr"]
        old_a = np.zeros(m.shape, np.uint8)
        old_a[m] = z[f"{pid}:alpha"]
        for v in np.unique(old_a[here]):
            ov.write(x0, y0, old_bgr, here & (old_a == v), value=int(v))
    f.unlink()


def prop_plate(ov: "Overlay", street, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """What a prop is drawn over (BGR uint8) and where it is defined: the overlay, and with a street (a floor prop)
    the street's own pixels where the overlay has none."""
    bgr, al = ov.read(x0, y0, x1, y1)
    if street is None:
        return bgr, al
    rgb, sv = street.window(x0, x1, y0, y1, valid=True, fill=False)
    use = sv & ~al
    bgr[use] = rgb[..., ::-1][use]
    return bgr, al | sv


def prop_foot(d: Path, proj_root: Path, pid: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """A floor prop's foot as the build drew it: the same cut-out, graded the same way over the same plate (the
    world with the props' pixels restored from world/props/under.npz, and the street), for a street patch of its
    rows at and below its floor. Returns its window [x0, y0, x1, y1] (street px), the composite (BGR uint8) and the
    foot's mask (bool: the prop's pixels at and below the floor)."""
    from .street import Street
    spec = read_json(d / "street.json")
    ws = load_spec(spec)
    p = next(q for q in ws["props"] if q["id"] == pid)
    if p.get("floor") is None:
        raise SystemExit(f"prop {pid} has no floor")
    kit = proj_root / ws["props_kit"]
    img, _ = _prop_rgba(kit, p["kit"], p["flip"])
    h, w = img.shape[:2]
    x0, y0 = p["x"] - w, p["base"] - 2 * h
    x1, y1 = p["x"] + 2 * w, p["base"] + h // 2 + 8
    wt = WorldTiles(d)
    rgb, al = wt.read(x0, x1, y0, y1)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    und = np.load(d / "world" / "props" / "under.npz")
    order = [str(v) for v in und["order"]]
    for q in reversed(order[order.index(pid):] if pid in order else []):  # this prop and the ones drawn after it:
        bx0, by0, bx1, by1 = [int(v) for v in und[f"{q}:box"]]             # back to what was under them
        m = und[f"{q}:mask"]
        ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        fb = np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8)
        fb[m] = und[f"{q}:bgr"]
        fa = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        fa[m] = und[f"{q}:alpha"]
        sub = m[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        bgr[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0][sub] = fb[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0][sub]
        al[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0][sub] = fa[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0][sub] > 0
    st = Street(d, world=False)
    srgb, sv = st.window(x0, x1, y0, y1, valid=True, fill=False)
    use = sv & ~al
    bgr[use] = srgb[..., ::-1][use]
    ox, oy = p["x"] - x0, p["base"] - h - y0
    rng = np.random.default_rng(int(hashlib.sha256(p["id"].encode()).hexdigest()[:8], 16))
    new, a = prop_composite(bgr.astype(np.float32) / 255, img, ox, oy, rng, p["shadow"], p["front_rails"],
                            [[c0 - x0, c1 - x0] for c0, c1 in p["front_cols"]], p["highlights"], p["warm"])
    new8 = np.clip(new * 255 + 0.5, 0, 255).astype(np.uint8)
    foot = (new8 != bgr).any(2)
    foot[:max(p["floor"] - y0, 0)] = False
    return [x0, y0, x1, y1], new8, foot


def apply_props(ov: "Overlay", wd: Path, kit: Path, props: list[dict], log=print, street=None) -> list[dict]:
    """The prop step: each kit cut-out drawn into the overlay where the spec puts it (prop_composite), its pixels
    PROP_ALPHA; what was under them saved to world/props/under.npz, each prop's matte to world/props/<id>-mask.png.
    A prop with a `floor` (over the street's left part, standing on the street's rows) is graded to the overlay and
    the street behind it and writes nothing at or below its floor (its foot there is a street patch: prop_foot).
    Returns each prop's placement: its silhouette box, its top surface and its matte (street px)."""
    pd = wd / "props"
    pd.mkdir(exist_ok=True)
    for old in pd.glob("*-mask.png"):
        old.unlink()
    store, order, out = {}, [], []
    for p in props:
        img, top = _prop_rgba(kit, p["kit"], p["flip"])
        h, w = img.shape[:2]
        x0, y0 = p["x"] - w, p["base"] - 2 * h
        x1, y1 = p["x"] + 2 * w, p["base"] + h // 2 + 8
        bgr, al = prop_plate(ov, street if p.get("floor") is not None else None, x0, y0, x1, y1)
        ox, oy = p["x"] - x0, p["base"] - h - y0
        if not al[oy:oy + h, ox:ox + w][img[..., 3] > 0.5].all():
            log(f"world: prop {p['id']} ({p['kit']}) at {p['x']}, {p['base']} is not wholly on the world: left out")
            continue
        rng = np.random.default_rng(int(hashlib.sha256(p["id"].encode()).hexdigest()[:8], 16))
        plate = bgr.astype(np.float32) / 255
        new, a = prop_composite(plate, img, ox, oy, rng, p["shadow"], p["front_rails"],
                                [[c0 - x0, c1 - x0] for c0, c1 in p["front_cols"]], p["highlights"],
                                p["warm"])
        new8 = np.clip(new * 255 + 0.5, 0, 255).astype(np.uint8)
        pa = np.zeros(al.shape, bool)
        pa[oy:oy + h, ox:ox + w] = a > 0
        changed = ((new8 != bgr).any(2) | pa) & al  # the prop's own pixels count even where they match the plate
        if p.get("floor") is not None:
            changed[max(p["floor"] - y0, 0):] = False  # the street's rows keep the street (the foot: prop_foot)
        raw = ov.read_alpha(x0, y0, x1, y1)
        store[f"{p['id']}:box"] = np.int64([x0, y0, x1, y1])
        store[f"{p['id']}:mask"] = changed
        store[f"{p['id']}:bgr"] = bgr[changed]
        store[f"{p['id']}:alpha"] = raw[changed]
        order.append(p["id"])
        ov.write(x0, y0, new8, changed, value=PROP_ALPHA)
        ys, xs = np.nonzero(a > 0.5)
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        cv2.imwrite(str(pd / f"{p['id']}-mask.png"), np.clip(a[by0:by1, bx0:bx1] * 255 + 0.5, 0, 255).astype(np.uint8))
        sx, sy = x0 + ox, y0 + oy
        out.append({"id": p["id"], "kit": p["kit"], "box": [sx + bx0, sy + by0, sx + bx1, sy + by1],
                    "top": {"y": round(sy + top["row"]), "x0": round(sx + top["col0"]), "x1": round(sx + top["col1"])},
                    "mask": f"world/props/{p['id']}-mask.png"})
        log(f"world: prop {p['id']} ({p['kit']}) at x {sx + bx0} to {sx + bx1}, top {round(sy + top['row'])}")
    if order:
        np.savez(pd / "under.npz", order=np.array(order), **store)
    return out


# ------------------------------------------------------------------------------------------------ seams

def _seam_cost(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Colour and gradient difference of two BGR float images (the strip's seam cost)."""
    d = np.abs(A - B).sum(2)
    ga, gb = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    gx = lambda g: np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = lambda g: np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.GaussianBlur(d, (0, 0), 2) + 0.5 * np.abs(gx(ga) - gx(gb)) + 0.5 * np.abs(gy(ga) - gy(gb))


def _dp_seam(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost vertical path through cost (rows x cols), moving at most one column per row: x per row."""
    hh, ww = cost.shape
    acc = cost.astype(np.float64).copy()
    back = np.zeros((hh, ww), np.int8)
    for y in range(1, hh):
        prev = acc[y - 1]
        stack = np.vstack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])
        k = stack.argmin(0)
        acc[y] += stack[k, np.arange(ww)]
        back[y] = k - 1
    seam = np.zeros(hh, np.int64)
    seam[-1] = int(acc[-1].argmin())
    for y in range(hh - 1, 0, -1):
        seam[y - 1] = np.clip(seam[y] + back[y, seam[y]], 0, ww - 1)
    return seam


def _blur_masked(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def _side_of(ov: np.ndarray, cell: tuple[int, int, int, int]) -> str | None:
    """The side of the cell an overlap lies on (left, right, top, bottom): the edge it touches and spans most of."""
    ys, xs = np.nonzero(ov)
    if len(xs) < 500:
        return None
    cx0, cy0, cx1, cy1 = cell
    ox0, oy0, ox1, oy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    wf, hf = (ox1 - ox0) / max(cx1 - cx0, 1), (oy1 - oy0) / max(cy1 - cy0, 1)
    e = 12
    score = {"left": hf if ox0 <= cx0 + e else 0.0, "right": hf if ox1 >= cx1 - e else 0.0,
             "top": wf if oy0 <= cy0 + e else 0.0, "bottom": wf if oy1 >= cy1 - e else 0.0}
    side = max(score, key=score.get)
    return side if score[side] > 0 else None


def _side_seam(buf: np.ndarray, warped: np.ndarray, band: np.ndarray, side: str, F: int,
               gain_fade: float) -> dict:
    """One side's seam and gain: the seam (x per row for left/right, y per column for top/bottom) in the middle of
    the band, its feathered weight for the new cell, and the low-frequency gain along it with its fade."""
    vertical = side in ("left", "right")
    if not vertical:  # work transposed: a top/bottom band is a left/right band of the transposed images
        buf, warped, band = buf.transpose(1, 0, 2), warped.transpose(1, 0, 2), band.T
    n_rows, n_cols = band.shape
    rows = np.nonzero(band.any(1))[0]
    cols = np.nonzero(band.any(0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    # gain along the seam (low frequency, per channel, a function of the position along it)
    g0, g1 = c0 + (c1 - c0) // 4, c1 - (c1 - c0) // 4
    if g1 <= g0:
        g0, g1 = c0, c1
    bm = band[r0:r1, g0:g1].astype(np.float32)
    A_low = _blur_masked(buf[r0:r1, g0:g1], bm, 40)
    B_low = _blur_masked(warped[r0:r1, g0:g1], bm, 40)
    ratio = (A_low + 0.01) / (B_low + 0.01)
    ratio[bm < 0.5] = np.nan
    have = ~np.isnan(ratio[..., 0]).all(1)
    gain = np.ones((n_rows, 3), np.float32)
    if have.any():
        med = np.ones((r1 - r0, 3), np.float32)
        med[have] = np.nanmedian(ratio[have], axis=1)
        idx = np.arange(r1 - r0)
        for ch in range(3):
            med[:, ch] = np.interp(idx, idx[have], med[have, ch])
        med = cv2.GaussianBlur(np.clip(med, 0.6, 1.6).astype(np.float32).reshape(-1, 1, 3), (1, 0), sigmaX=1,
                               sigmaY=25).reshape(-1, 3)
        gain[r0:r1] = med
        gain[:r0] = med[0]
        gain[r1:] = med[-1]
    mid = (c0 + c1) / 2
    xx = np.arange(n_cols, dtype=np.float32)
    toward = 1.0 if side in ("left", "top") else -1.0  # the cell lies right of a left band, below a top band ...
    fade = np.clip(1 - toward * (xx - mid) / gain_fade, 0, 1).astype(np.float32)
    # the seam: minimum difference in the middle of the band (after the gain), kept inside the band
    s0, s1 = c0 + int((c1 - c0) * 0.3), c0 + int((c1 - c0) * 0.7)
    s1 = max(s1, s0 + 1)
    gw = warped[r0:r1, s0:s1] * gain[r0:r1, None, :]
    cost = _seam_cost(buf[r0:r1, s0:s1], gw)
    cost[~band[r0:r1, s0:s1]] += 1e3
    seam = _dp_seam(cost)
    sx = np.full(n_rows, s0 + seam[0], np.float32)
    sx[r0:r1] = s0 + seam
    sx[r1:] = s0 + seam[-1]
    sx = cv2.GaussianBlur(sx.reshape(-1, 1), (1, 0), sigmaX=1, sigmaY=3).ravel()
    w = np.clip(toward * (xx[None, :] - sx[:, None]) / F + 0.5, 0, 1).astype(np.float32)
    out = {"w": w, "gain": gain, "fade": fade, "seam_mean": float(sx[r0:r1].mean()), "band": [c0, r0, c1, r1]}
    if not vertical:
        out["w"] = w.T
        out["band"] = [r0, c0, r1, c1]
    out["vertical"] = vertical
    return out


def blend_cell(buf: np.ndarray, hv: np.ndarray, warped: np.ndarray, valid: np.ndarray,
               bands: dict[str, np.ndarray], F: int, gain_fade: float, floor_row: int | None = None
               ) -> tuple[np.ndarray, np.ndarray, list]:
    """Blend a warped cell (BGR float, valid bool) into the world box (buf BGR float, hv bool): per side a gain match
    and a seam; the cell wins past every seam, and wherever the world had nothing. With `floor_row` (a cell above the
    street) its side below keeps only its gain: the seam is the floor itself, and the cell gives nothing at or below
    it. Returns (buf, wB, seams)."""
    bh, bw = valid.shape
    sides = [(s, _side_seam(buf, warped, b, s, F, gain_fade)) for s, b in bands.items() if b.any()]
    if floor_row is not None:
        for s, r in sides:
            if s == "bottom":
                r["w"] = np.ones_like(r["w"])
                r["seam_mean"] = float(floor_row)
    if sides:
        num = np.zeros((bh, bw, 3), np.float32)
        den = np.zeros((bh, bw), np.float32)
        amax = np.zeros((bh, bw), np.float32)
        for s, r in sides:
            if r["vertical"]:
                f = np.broadcast_to(r["fade"][None, :], (bh, bw))
                g = r["gain"][:, None, :]
            else:
                f = np.broadcast_to(r["fade"][:, None], (bh, bw))
                g = r["gain"][None, :, :]
            num += f[..., None] * g
            den += f
            np.maximum(amax, f, out=amax)
        G = np.where(den[..., None] > 0, num / np.maximum(den, 1e-6)[..., None], 1.0)
        warped = (warped * (1 + (G - 1) * amax[..., None])).astype(np.float32)
    wB = valid.astype(np.float32)
    for _, r in sides:
        wB *= r["w"]
    wB = np.where(hv, wB, valid.astype(np.float32))
    if floor_row is not None:
        wB[floor_row:] = 0
    buf = buf * (1 - wB[..., None]) + warped * wB[..., None]
    seams = [{"side": s, "at": round(r["seam_mean"], 1), "band": r["band"]} for s, r in sides]
    return buf.astype(np.float32), wB, seams


# ------------------------------------------------------------------------------------------------ placement

def match_cell(img8: np.ndarray, N: np.ndarray, base8: np.ndarray, region: np.ndarray, bx: int, by: int,
               cid: str) -> tuple[np.ndarray, int, float]:
    """Similarity cell px -> street px by SIFT + RANSAC in the overlap `region` (the world box's pixels the cell's
    neighbours cover, bool over the box at (bx, by)), matches gated to GATE px of the predicted map N."""
    h, w = img8.shape[:2]
    reg = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8))
    if reg.sum() < 4000:
        raise SystemExit(f"world: cell {cid} hardly overlaps the world where its neighbours are (check its at and "
                         f"from)")
    Nl = _local(N, bx, by)
    cm = cv2.warpAffine(reg, cv2.invertAffineTransform(Nl), (w, h), flags=cv2.INTER_NEAREST)
    g = int(GATE)
    cm = cv2.dilate(cm, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    sift = cv2.SIFT_create(10000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(base8, cv2.COLOR_BGR2GRAY), reg * 255)
    kc, dc = sift.detectAndCompute(cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY), cm * 255)
    if da is None or dc is None or len(ka) < 12 or len(kc) < 12:
        raise SystemExit(f"world: cell {cid}: no features in its overlap")
    m = cv2.BFMatcher().knnMatch(dc, da, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    pc = np.float32([kc[gq.queryIdx].pt for gq in good]).reshape(-1, 2)
    pb = np.float32([ka[gq.trainIdx].pt for gq in good]).reshape(-1, 2)
    near = np.linalg.norm(pc @ Nl[:, :2].T.astype(np.float32) + Nl[:, 2].astype(np.float32) - pb, axis=1) < GATE
    pc, pb = pc[near], pb[near]
    if len(pc) < 12:
        raise SystemExit(f"world: cell {cid}: only {len(pc)} matches near its predicted place (within {GATE:.0f} px); "
                         f"was it outpainted from a canvas of its neighbours at its at?")
    cv2.setRNGSeed(12345)
    M, inl = cv2.estimateAffinePartial2D(pc, pb, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=5000)
    if M is None or inl.sum() < 12:
        raise SystemExit(f"world: cell {cid}: no consistent placement in its overlap")
    res = np.linalg.norm(pc @ M[:, :2].T + M[:, 2] - pb, axis=1)[inl.ravel() > 0]
    M = M.astype(np.float64)
    M[0, 2] += bx
    M[1, 2] += by
    s, sN = np.hypot(M[0, 0], M[1, 0]), np.hypot(N[0, 0], N[1, 0])
    da_ = np.degrees(np.arctan2(M[1, 0], M[0, 0]) - np.arctan2(N[1, 0], N[0, 0]))
    c = np.float64([w / 2, h / 2])
    dc_ = np.linalg.norm((M[:, :2] @ c + M[:, 2]) - (N[:, :2] @ c + N[:, 2]))
    if abs(s / sN - 1) > 0.02 or abs(da_) > 0.75 or dc_ > GATE:
        raise SystemExit(f"world: cell {cid}: the match puts it {dc_:.0f} px from its predicted place, scaled "
                         f"{s / sN:.4f}, turned {da_:.2f} deg: check its at, or give it place: nominal")
    return M, int(inl.sum()), float(np.median(res))


def _predict(c: dict, N0: np.ndarray, w: int, h: int, placed: dict, ats: dict) -> np.ndarray:
    """Where a cell should be: its grid offset from the placed neighbour its grid box overlaps most (its `from`
    ones when given), applied to that neighbour's placement, so the drift of the cells before it is carried along and
    only the model's own drift is left for the match to find."""
    gx0, gy0, gx1, gy1 = _bbox(N0, w, h)
    best, area = None, 0.0
    for nid, p in placed.items():
        if c["from"] is not None and nid not in c["from"]:
            continue
        pw, ph = p["size"]
        ax, ay = ats[nid]
        dx, dy = c["at"][0] - ax, c["at"][1] - ay  # grid offset of the cell from the neighbour (anchor px)
        ox = max(0.0, min(w, dx + pw) - max(0.0, dx))
        oy = max(0.0, min(h, dy + ph) - max(0.0, dy))
        if ox * oy > area:
            best, area = nid, ox * oy
    if best is None:
        return N0
    ax, ay = ats[best]
    return _compose(placed[best]["M"], (c["at"][0] - ax, c["at"][1] - ay))


def _read_cell(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"world: cannot read cell {p}")
    return img


# ------------------------------------------------------------------------------------------------ build

def build(proj_root: Path, d: Path, spec: dict, strip: dict, street_info: dict, prev: dict | None = None,
          log=print) -> dict | None:
    """Place the world's cells (street.json world) round the built street: world/ tiles and world/world.json.
    Returns the world info (None without a world section). Incremental: cells added at the end of an unchanged
    list are placed on top of the last build."""
    from .street import Street

    ws = load_spec(spec)
    wd = d / "world"
    if ws is None:
        return None
    wd.mkdir(parents=True, exist_ok=True)
    T = int(spec["tile"])
    F, gain_fade = ws["seam_feather"], ws["gain_fade"]
    A, (AW, AH) = anchor_transform(spec, strip)
    anchor_sha = sha256_file(proj_root / ws["anchor"])
    street = Street(d, world=False)
    ax0 = _bbox(A, AW, AH)[0]
    params = {"version": VERSION, "anchor": ws["anchor"], "anchor_sha": anchor_sha,
              "A": [[round(float(v), 9) for v in r] for r in A], "tile": T, "feather": F, "gain_fade": gain_fade,
              "street": {str(k): street.tile_hash(k) for k in range(ax0 // T - 1, street.n)}}
    if ws["inserts"]:  # an insert moves every cell right of it: any change places the world again
        params["inserts"] = [dict(e, patches=[dict(q, sha=sha256_file(proj_root / q["path"])) if
                                              (proj_root / q["path"]).exists() else dict(q, sha=None)
                                              for q in e["patches"]]) for e in ws["inserts"]]
    if ws["grade"]:  # a grade changes the cells' own pixels: any change places the world again
        params["grade"] = ws["grade"]
    cells = []
    for c in ws["cells"]:
        f = proj_root / c["path"]
        if not f.exists():
            raise SystemExit(f"world: cell {c['id']}: no file {f}")
        cells.append(dict(c, sha=sha256_file(f)))
    key = lambda c: (c["id"], c["path"], c["sha"], list(c["at"]), c["from"], c["place"], c.get("floor"))
    prev = prev or {}
    old_tiles = {_wkey(k): v for k, v in (prev.get("tiles") or {}).items()}
    m = 0
    if prev.get("params") == params:
        pc = prev.get("cells") or []
        if len(pc) <= len(cells) and all(key(a) == key(b) for a, b in zip(pc, cells)) and \
                all((wd / wtile_name(*k)).exists() for k in old_tiles):
            m = len(pc)
    sky, old_sky = ws["sky"], (prev.get("sky") or []) if m else []
    kit = proj_root / ws["props_kit"]
    props = []
    for p in ws["props"]:
        f = kit / f"{p['kit']}.png"
        if not f.exists():
            raise SystemExit(f"world: prop {p['id']}: no cut-out {f}")
        props.append(dict(p, sha=sha256_file(f)))
    if m == len(cells) and m == len(prev.get("cells") or []) and prev and prev.get("sky", []) == sky and \
            (not sky or prev.get("sky_version") == SKY_VERSION) and prev.get("props", []) == props and \
            (not props or prev.get("props_version") == PROPS_VERSION):
        log(f"world: up to date ({len(cells)} cells)")
        return prev
    if (ws["inserts"] or ws["grade"]) and m and m < len(cells):
        m = 0  # (the overlay is already opened by the inserts: new cells are placed on a fresh one)
    ov = Overlay(wd, T, start=old_tiles if m else None)
    if m:
        clear_props(ov, wd)  # the last build's props, then its sky, go first: the cells see the cells only
    else:
        (wd / "props" / "under.npz").unlink(missing_ok=True)
    clear_sky(ov, old_sky + sky)
    placed = {ws["anchor_id"]: {"M": A, "size": [AW, AH], "anchor": True}}
    ats = {ws["anchor_id"]: [0.0, 0.0], **{c["id"]: c["at"] for c in cells}}
    records = list(prev["cells"][:m]) if m else []
    for r in records:
        placed[r["id"]] = {"M": np.float64(r["T"]), "size": r["size"]}
    if m:
        log(f"world: {m} cells kept from the last build")
    for c in cells[m:]:
        img8 = _read_cell(proj_root / c["path"])
        h, w = img8.shape[:2]
        N0 = _compose(A, c["at"])  # on the grid
        N = _predict(c, N0, w, h, placed, ats)  # from the neighbour it overlaps most (their drift carried along)
        bx0, by0, bx1, by1 = _bbox(N, w, h, MARGIN)
        bw, bh = bx1 - bx0, by1 - by0
        # the world so far in the cell's box: the street, then the overlay
        rgb, sv = street.window(bx0, bx1, by0, by1, valid=True, fill=False)
        base = np.ascontiguousarray(rgb[..., ::-1])
        obgr, oal = ov.read(bx0, by0, bx1, by1)
        base[oal] = obgr[oal]
        hv = sv | oal
        # the neighbours it came from, and where each covers the box
        frm = c["from"]
        if frm is None:
            nb = {}
            for nid, p in placed.items():
                pw, ph = p["size"]
                qx0, qy0, qx1, qy1 = _bbox(p["M"], pw, ph)
                if qx0 < bx1 and qx1 > bx0 and qy0 < by1 and qy1 > by0:
                    nb[nid] = p
        else:
            nb = {nid: placed[nid] for nid in frm}
        covs = {}
        for nid, p in nb.items():
            cv_ = _coverage(p["M"], p["size"][0], p["size"][1], bx0, by0, bw, bh)
            if p.get("anchor"):
                cv_ &= sv  # the anchor as the street has it
            covs[nid] = cv_ & hv
        floor = c.get("floor")
        if floor is not None:  # a cell above the street (its left part): the street is its neighbour below, its
            covs["street"] = sv.copy()  # rows from FLOOR_SKIP under the floor (the seam's own patch may change those)
            covs["street"][:max(floor + FLOOR_SKIP - by0, 0)] = False
        region = np.zeros((bh, bw), bool)
        for cv_ in covs.values():
            region |= cv_
        if c["place"] == "nominal":
            M, ninl, med = N, 0, 0.0
        else:
            M, ninl, med = match_cell(img8, N, base, region, bx0, by0, c["id"])
        # the cell in the box, and the seams with its neighbours
        Ml = _local(M, bx0, by0)
        warped = cv2.warpAffine(img8.astype(np.float32) / 255, Ml, (bw, bh), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT)
        valid = _coverage(M, w, h, bx0, by0, bw, bh)
        ys, xs = np.nonzero(valid)
        cbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        bands: dict[str, np.ndarray] = {}
        froms: dict[str, list] = {}
        for nid, cv_ in covs.items():
            o = cv_ & valid
            s = _side_of(o, cbox)
            if s is None:
                continue
            bands[s] = bands.get(s, np.zeros((bh, bw), bool)) | o
            froms.setdefault(s, []).append(nid)
        buf = base.astype(np.float32) / 255
        fr = None if floor is None else int(np.clip(floor - by0, 0, bh))
        buf, wB, seams = blend_cell(buf, hv, warped, valid, bands, F, gain_fade, floor_row=fr)
        newal = (wB > 0) | (valid & ~sv)
        if fr is not None:
            newal[fr:] = False  # nothing at or below its floor: the street keeps its rows
        out8 = (np.clip(buf, 0, 1) * 255 + 0.5).astype(np.uint8)
        ov.write(bx0, by0, out8, newal)
        for sm in seams:
            sm["from"] = froms[sm["side"]]
            b = sm["band"]
            sm["band"] = [b[0] + bx0, b[1] + by0, b[2] + bx0, b[3] + by0]
            sm["at"] = round(sm["at"] + (bx0 if sm["side"] in ("left", "right") else by0), 1)
        cc = [w / 2, h / 2]
        cN, cM, cP = N0[:, :2] @ cc + N0[:, 2], M[:, :2] @ cc + M[:, 2], N[:, :2] @ cc + N[:, 2]
        rec = {"id": c["id"], "path": c["path"], "sha": c["sha"], "at": c["at"], "from": c["from"],
               "place": c["place"], **({"floor": floor} if floor is not None else {}),
               "size": [w, h], "T": M.tolist(), "box": list(_bbox(M, w, h)),
               "nominal_dxy": [round(float(cM[0] - cN[0]), 2), round(float(cM[1] - cN[1]), 2)],
               "drift_dxy": [round(float(cM[0] - cP[0]), 2), round(float(cM[1] - cP[1]), 2)],
               "scale": round(float(np.hypot(M[0, 0], M[1, 0])), 5),
               "rot_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 4),
               "inliers": ninl, "median_px": round(med, 2), "seams": seams}
        records.append(rec)
        placed[c["id"]] = {"M": M, "size": [w, h]}
        sides = ", ".join(f"{s['side']} seam at {s['at']:.0f} ({'+'.join(s['from'])})" for s in seams) or "no seams"
        log(f"world: {c['id']} at {rec['box'][0]}, {rec['box'][1]} (off the grid {rec['nominal_dxy'][0]:+.1f}, "
            f"{rec['nominal_dxy'][1]:+.1f} px, off its neighbour {rec['drift_dxy'][0]:+.1f}, {rec['drift_dxy'][1]:+.1f} px, "
            f"scale {rec['scale']}, {ninl} inliers, median {med:.2f} px); {sides}")
    if ws["inserts"] and m < len(cells):  # only when the cells were placed now (else the old tiles have them)
        apply_inserts(ov, proj_root, params["inserts"], log)
    if ws["grade"] and m < len(cells):  # (likewise)
        apply_grade(ov, ws["grade"], log)
    for r in sky:
        n = apply_sky(ov, r)
        log(f"world: sky over x {r['x'][0]} to {r['x'][1]}, {r['up']} px up ({n} px)")
    placed_props = apply_props(ov, wd, kit, props, log, street=street)
    tiles = ov.finish(old_tiles)
    info = {"version": 1, "params": params, "inserts": params.get("inserts", []), "sky": sky,
            "sky_version": SKY_VERSION, "props": props,
            "props_version": PROPS_VERSION, "placed_props": placed_props, "anchor": {"path": ws["anchor"], "id": ws["anchor_id"], "sha": anchor_sha,
                                                       "A": A.tolist(), "size": [AW, AH]},
            "cells": records, "tile": T, "tiles": {f"{k[0]}_{k[1]}": v for k, v in sorted(tiles.items())}}
    info.update(_extent(street, wd, T, tiles, _floor_bands(records, T)))
    write_json(wd / "world.json", info, nd=9)
    log(f"world: {len(records)} cells in {len(tiles)} tiles, overlay x {info['overlay'][0]} to {info['overlay'][2]}, "
        f"y {info['overlay'][1]} to {info['overlay'][3]}")
    return info


def _floor_bands(records: list[dict], gap: int) -> list[list[int]]:
    """The boxes of the cells with a floor (above the street's left part), grouped where they overlap or come within
    `gap` px of each other, each down to its floor: [x0, y0, x1, floor]."""
    boxes = sorted([int(r["box"][0]), int(r["box"][1]), int(r["box"][2]), int(r["floor"])]
                   for r in records if r.get("floor") is not None)
    bands: list[list[int]] = []
    for b in boxes:
        if bands and b[0] <= bands[-1][2] + gap:
            g = bands[-1]
            g[1], g[2], g[3] = min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])
        else:
            bands.append(list(b))
    return bands


def _extent(street, wd: Path, T: int, tiles: dict, bands: list[list[int]] | None = None) -> dict:
    """The overlay's box and the union of street and world at 1/8 (world/valid8.png: 255 where a block's pixels are
    all valid). With floor cells, also `bands` (their boxes) and `xw`: the world's first column outside them, where
    the world takes over from the street (the cells above the street's left part do not move it)."""
    if not tiles:
        return {"overlay": None, "bounds": [street.X0, 0, street.W, street.H], "valid8": None}
    ox0 = oy0 = 1 << 30
    ox1 = oy1 = -(1 << 30)
    xw = 1 << 30
    alphas = {}
    for k in tiles:
        a = cv2.imread(str(wd / wtile_name(*k)), cv2.IMREAD_UNCHANGED)[..., 3] > 0
        alphas[k] = a
        ys, xs = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
        if len(xs):
            ox0, ox1 = min(ox0, k[0] * T + int(xs.min())), max(ox1, k[0] * T + int(xs.max()) + 1)
            oy0, oy1 = min(oy0, k[1] * T + int(ys.min())), max(oy1, k[1] * T + int(ys.max()) + 1)
        if bands:
            out = a.copy()
            for bx0, by0, bx1, by1 in bands:  # the floor cells' pixels left out
                tx0, ty0 = k[0] * T, k[1] * T
                c0, c1 = max(bx0 - tx0, 0), min(bx1 - tx0, T)
                r0, r1 = max(by0 - ty0, 0), min(by1 - ty0, T)
                if c1 > c0 and r1 > r0:
                    out[r0:r1, c0:c1] = False
            xs2 = np.nonzero(out.any(0))[0]
            if len(xs2):
                xw = min(xw, k[0] * T + int(xs2.min()))
    X0, Y0 = min(street.X0, ox0), min(0, oy0)
    X1, Y1 = max(street.W, ox1), max(street.H, oy1)
    gx0, gy0 = X0 // 8 * 8, Y0 // 8 * 8
    w8, h8 = -(-(X1 - gx0) // 8), -(-(Y1 - gy0) // 8)
    v8 = np.zeros((h8, w8), np.uint8)
    # the street's rectangle: blocks entirely inside it
    bx = gx0 + 8 * np.arange(w8)
    by = gy0 + 8 * np.arange(h8)
    inx = (bx >= street.X0) & (bx + 8 <= street.W)
    iny = (by >= 0) & (by + 8 <= street.H)
    v8[np.ix_(iny, inx)] = 255
    # the overlay tiles: the union of street and overlay pixels, block by block
    for (ix, iy), a in alphas.items():
        u = a.copy()
        tx0, ty0 = ix * T, iy * T
        sx0, sx1 = max(street.X0, tx0) - tx0, min(street.W, tx0 + T) - tx0
        sy0, sy1 = max(0, ty0) - ty0, min(street.H, ty0 + T) - ty0
        if sx1 > sx0 and sy1 > sy0:
            u[sy0:sy1, sx0:sx1] = True
        blk = u.reshape(T // 8, 8, T // 8, 8).all(axis=(1, 3))
        j0, i0 = (tx0 - gx0) // 8, (ty0 - gy0) // 8
        sub = v8[max(i0, 0):i0 + T // 8, max(j0, 0):j0 + T // 8]
        bb = blk[max(-i0, 0):max(-i0, 0) + sub.shape[0], max(-j0, 0):max(-j0, 0) + sub.shape[1]]
        sub[...] = np.where(bb, 255, sub)
    cv2.imwrite(str(wd / "valid8.png"), v8)
    out = {"overlay": [ox0, oy0, ox1, oy1], "bounds": [X0, Y0, X1, Y1],
           "valid8": {"file": "valid8.png", "origin": [gx0, gy0], "size": [w8, h8]}}
    if bands:
        out["xw"] = int(xw) if xw < (1 << 30) else int(ox1)
        out["bands"] = bands
    return out


# ------------------------------------------------------------------------------------------------ reading

class WorldTiles:
    """The built world overlay, read by tiles (a small LRU): RGB and alpha over any box."""

    def __init__(self, d: Path, cache_tiles: int = 8):
        self.wd = Path(d) / "world"
        self.info = read_json(self.wd / "world.json")
        self.tile = int(self.info["tile"])
        self.hashes = {_wkey(k): v for k, v in self.info["tiles"].items()}
        self.overlay = self.info.get("overlay")  # [x0, y0, x1, y1] of the defined pixels, or None
        self.bounds = self.info["bounds"]
        self.A = np.float64(self.info["anchor"]["A"])
        self._lru: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cap = cache_tiles
        self._v8 = None

    def touches(self, x0: int, x1: int, y0: int, y1: int) -> bool:
        o = self.overlay
        return bool(o) and x0 < o[2] and x1 > o[0] and y0 < o[3] and y1 > o[1]

    def tile_rgba(self, ix: int, iy: int) -> np.ndarray | None:
        k = (ix, iy)
        if k not in self.hashes:
            return None
        if k in self._lru:
            self._lru.move_to_end(k)
            return self._lru[k]
        t = cv2.imread(str(self.wd / wtile_name(ix, iy)), cv2.IMREAD_UNCHANGED)
        if t is None:
            raise SystemExit(f"missing world tile {wtile_name(ix, iy)} in {self.wd}")
        t = np.ascontiguousarray(t[..., [2, 1, 0, 3]])  # RGBA
        self._lru[k] = t
        while len(self._lru) > self._cap:
            self._lru.popitem(last=False)
        return t

    def read(self, x0: int, x1: int, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 and alpha (bool) of the overlay over [x0, x1) x [y0, y1)."""
        T = self.tile
        out = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        al = np.zeros((y1 - y0, x1 - x0), bool)
        if not self.touches(x0, x1, y0, y1):
            return out, al
        for iy in range(y0 // T, (y1 - 1) // T + 1):
            for ix in range(x0 // T, (x1 - 1) // T + 1):
                t = self.tile_rgba(ix, iy)
                if t is None:
                    continue
                a0, b0 = max(x0, ix * T), max(y0, iy * T)
                a1, b1 = min(x1, (ix + 1) * T), min(y1, (iy + 1) * T)
                sub = t[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
                out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = sub[..., :3]
                al[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = sub[..., 3] > 0
        return out, al

    def valid8(self) -> tuple[np.ndarray, int, int] | None:
        """The union of street and world at 1/8 (bool, a block valid when all its pixels are) and its origin."""
        v = self.info.get("valid8")
        if not v:
            return None
        if self._v8 is None:
            self._v8 = cv2.imread(str(self.wd / v["file"]), cv2.IMREAD_GRAYSCALE) > 0
        return self._v8, int(v["origin"][0]), int(v["origin"][1])

    def content_id(self, ix: int, iy: int) -> str | None:
        return self.hashes.get((ix, iy))


def fill_nearest(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Every invalid pixel takes the value of the nearest valid one (for blurs over a window with holes)."""
    if valid.all() or not valid.any():
        return img
    _, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(valid)
    idx = lab - 1
    return img[ys[idx], xs[idx]]


def info_hash(info: dict) -> str:
    return hashlib.sha256(json.dumps(info.get("tiles", {}), sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ sheets

def contact(d: Path, out: Path, scale: int = 8) -> Path:
    """The street round the world and the world at 1/scale: invalid areas dark, every cell's box (cyan) with its id,
    every seam (red: vertical ones as a line across their band, horizontal ones the same), x and y every 1000 px."""
    from .street import Street

    s = Street(d)
    if s.world is None:
        raise SystemExit("no world built (street.json world, then aimv paint street)")
    info = s.world.info
    o = info["overlay"]
    box = (max(s.X0, o[0] - 6000), s.Y0, s.bounds[2], s.Y1)
    img, cov, ox, oy = s.overview2d(scale, box)
    img = img[..., ::-1].copy()
    img[cov < 0.5] = (24, 22, 20)
    X = lambda v: int(round((v - ox) / scale))
    Y = lambda v: int(round((v - oy) / scale))
    for xm in range(int(np.ceil(box[0] / 1000)) * 1000, int(box[2]), 1000):
        cv2.line(img, (X(xm), 0), (X(xm), 8 if xm % 5000 else 20), (255, 255, 255), 1)
        if xm % 5000 == 0:
            cv2.putText(img, f"x {xm}", (X(xm) + 3, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for ym in range(int(np.ceil(box[1] / 1000)) * 1000, int(box[3]), 1000):
        cv2.line(img, (0, Y(ym)), (10, Y(ym)), (255, 255, 255), 1)
        cv2.putText(img, f"y {ym}", (12, Y(ym) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for c in info["cells"]:
        M = np.float64(c["T"])
        w, h = c["size"]
        pts = (np.float64([[0, 0], [w, 0], [w, h], [0, h]]) @ M[:, :2].T + M[:, 2])
        pts[:, 0] += insert_shift(info, float(pts[:, 0].mean()))  # drawn where the inserts moved it
        cv2.polylines(img, [np.int32([[X(px), Y(py)] for px, py in pts])], True, (255, 220, 0), 1)
        cv2.putText(img, c["id"], (X(pts[0][0]) + 4, Y(pts[0][1]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 220, 0), 1)
        for sm in c["seams"]:
            b = sm["band"]
            if sm["side"] in ("left", "right"):
                cv2.line(img, (X(sm["at"]), Y(b[1])), (X(sm["at"]), Y(b[3])), (40, 40, 255), 1)
            else:
                cv2.line(img, (X(b[0]), Y(sm["at"])), (X(b[2]), Y(sm["at"])), (40, 40, 255), 1)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def seams_sheet(d: Path, out: Path, half: int = 300, per_row: int = 6) -> Path:
    """Full-resolution crops (2 * half square) centred on the middle of every seam, labelled with the cell, the
    side and the neighbours (the seam's mean position marked with short red ticks at the crop's edges)."""
    from .street import Street

    s = Street(d)
    if s.world is None:
        raise SystemExit("no world built")
    crops = []
    for c in s.world.info["cells"]:
        for sm in c["seams"]:
            b = sm["band"]
            if sm["side"] in ("left", "right"):
                cx, cy = sm["at"], (b[1] + b[3]) / 2
            else:
                cx, cy = (b[0] + b[2]) / 2, sm["at"]
            x0, y0 = int(cx) - half, int(cy) - half
            im = s.window(x0, x0 + 2 * half, y0, y0 + 2 * half)[..., ::-1].copy()
            t = 12
            if sm["side"] in ("left", "right"):
                cv2.line(im, (half, 0), (half, t), (0, 0, 255), 2)
                cv2.line(im, (half, 2 * half - t), (half, 2 * half), (0, 0, 255), 2)
            else:
                cv2.line(im, (0, half), (t, half), (0, 0, 255), 2)
                cv2.line(im, (2 * half - t, half), (2 * half, half), (0, 0, 255), 2)
            lab = f"{c['id']} {sm['side']} ({'+'.join(sm['from'])})"
            cv2.rectangle(im, (0, 2 * half - 24), (12 + 9 * len(lab), 2 * half), (0, 0, 0), -1)
            cv2.putText(im, lab, (6, 2 * half - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            crops.append(im)
    if not crops:
        raise SystemExit("no seams")
    rows = []
    for i in range(0, len(crops), per_row):
        r = crops[i:i + per_row]
        r += [np.zeros_like(crops[0])] * (per_row - len(r))
        rows.append(np.hstack(r))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# ------------------------------------------------------------------------------------------------ canvas

def canvas(proj_root: Path, d: Path, name: str, rect: tuple[float, float, float, float], out: Path | None = None,
           frame: str = "anchor") -> dict:
    """`aimv paint street <p> --canvas NAME --rect x0 y0 x1 y1`: the outpaint canvas for a new cell, exact: every
    known pixel of the street and the world inside the rect, resampled into the cell's own frame, flat grey (128)
    elsewhere. The rect is in anchor px (the grid of the cells' `at`: x0, y0 is the new cell's at, the rect's size
    its size; frame="street": street px). The cell's frame is where the build will predict it: its grid offset from
    the placed cell its rect overlaps most, applied to that cell's placement (so the drift of the cells so far is in
    the canvas, and the new cell lands where the build looks for it). Written next to the last cell (or `out`)."""
    from .street import Street

    s = Street(d)
    if s.world is None:
        info = read_json(d / "build.json")
        if not info.get("world"):
            raise SystemExit("the street has no world (add a world section to street.json, run aimv paint street)")
    x0, y0, x1, y1 = (float(v) for v in rect)
    W, H = int(round(x1 - x0)), int(round(y1 - y0))
    if W <= 0 or H <= 0:
        raise SystemExit(f"canvas: an empty rect {rect}")
    winfo = read_json(d / "world" / "world.json")
    A = np.float64(winfo["anchor"]["A"])
    if frame == "anchor":
        aid = winfo["anchor"]["id"]
        placed = {aid: {"M": A, "size": winfo["anchor"]["size"]}}
        ats = {aid: [0.0, 0.0]}
        for c in winfo.get("cells", []):
            placed[c["id"]] = {"M": np.float64(c["T"]), "size": c["size"]}
            ats[c["id"]] = c["at"]
        M = _predict({"at": [x0, y0], "from": None}, _compose(A, (x0, y0)), W, H, placed, ats)  # canvas -> street
        M[0, 2] += insert_shift(winfo, float(M[0, 0] * W / 2 + M[0, 2]))  # where the inserts moved it
    else:
        M = np.float64([[1, 0, x0], [0, 1, y0]])
    bx0, by0, bx1, by1 = _bbox(M, W, H, 8)
    rgb, v = s.window(bx0, bx1, by0, by1, valid=True)
    Ml = _local(M, bx0, by0)
    img = cv2.warpAffine(np.ascontiguousarray(rgb[..., ::-1]), Ml, (W, H),
                         flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    vv = cv2.warpAffine(v.astype(np.float32), Ml, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT)
    known = cv2.erode((vv > 0.999).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img[~known] = 128
    if out is None:
        last = winfo["cells"][-1]["path"] if winfo.get("cells") else winfo["anchor"]["path"]
        out = (proj_root / last).with_name(f"canvas-{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    sides = [n for n, edge in (("left", known[:, 0]), ("right", known[:, -1]), ("top", known[0]),
                               ("bottom", known[-1])) if edge.mean() > 0.5]
    return {"canvas": str(out), "size": [W, H], "known": round(float(known.mean()), 4), "sides": sides,
            "street_box": [bx0 + 8, by0 + 8, bx1 - 8, by1 - 8]}


# ------------------------------------------------------------------------------------------------ the grid

def grid_to_street(winfo: dict, u: float, v: float) -> tuple[float, float]:
    """A point on the cells' grid (anchor px, as `at` and boards/18-world/make.py) to street px, through the cell
    that holds it (the last placed one, as on top in the world), else the anchor: the drift of each cell included."""
    for c in reversed(winfo.get("cells", [])):
        cx, cy = u - c["at"][0], v - c["at"][1]
        w, h = c["size"]
        if 0 <= cx <= w and 0 <= cy <= h:
            T = np.float64(c["T"])
            x, y = float(T[0, 0] * cx + T[0, 1] * cy + T[0, 2]), float(T[1, 0] * cx + T[1, 1] * cy + T[1, 2])
            return x + insert_shift(winfo, x), y
    A = np.float64(winfo["anchor"]["A"])
    x, y = float(A[0, 0] * u + A[0, 1] * v + A[0, 2]), float(A[1, 0] * u + A[1, 1] * v + A[1, 2])
    return x + insert_shift(winfo, x), y


def insert_shift(winfo: dict, x: float) -> float:
    """How far the world's inserts moved a point that the cells placed at x (before the inserts): the lengths of
    the inserts at or left of it."""
    return float(sum(e["length"] for e in winfo.get("inserts") or [] if x >= e["x"]))


def street_before_inserts(winfo: dict, x: float) -> float:
    """A street x after the inserts back to where the cells placed it (a point in an insert's gap: its x)."""
    shift = 0.0
    for e in winfo.get("inserts") or []:
        if x - shift < e["x"]:
            break
        if x - shift < e["x"] + e["length"]:
            return float(e["x"])
        shift += e["length"]
    return x - shift


def street_to_grid(winfo: dict, x: float, y: float) -> tuple[float, float]:
    """The inverse of grid_to_street (through the last placed cell holding the street point, else the anchor)."""
    x = street_before_inserts(winfo, x)
    for c in reversed(winfo.get("cells", [])):
        Ti = cv2.invertAffineTransform(np.float64(c["T"]))
        cx, cy = Ti[0, 0] * x + Ti[0, 1] * y + Ti[0, 2], Ti[1, 0] * x + Ti[1, 1] * y + Ti[1, 2]
        w, h = c["size"]
        if 0 <= cx <= w and 0 <= cy <= h:
            return float(cx + c["at"][0]), float(cy + c["at"][1])
    Ai = cv2.invertAffineTransform(np.float64(winfo["anchor"]["A"]))
    return float(Ai[0, 0] * x + Ai[0, 1] * y + Ai[0, 2]), float(Ai[1, 0] * x + Ai[1, 1] * y + Ai[1, 2])
