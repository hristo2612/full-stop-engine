"""The scene on the street: obstacles and their contacts, occluders, no-paint zones, parallax layers, lights, and the
light maps the paint model needs (prototype 6's scene.py, by windows instead of one 1.8 GB array).

Registry: projects/<p>/paint/streets/<name>/obstacles.json (street px; the facade is a flat elevation, so wall
space = image space):

    {"version": 1,
     "ground": {"wall_base": 1212, "feet": 1196, "kerb_top": 1370, "kerb_bot": 1420,
                "dy": [[-5000, -20], [-1000, -10], [0, 0]]},   optional: the ground line's offset along x (piecewise
                                                                linear, constant beyond the ends; plates are not level)
     "obstacles": {
       "ac": {"kind": "ac", "box": [4246, 765, 4499, 947],
              "contacts": {"top": {"y": 765}},               a horizontal line (x0, x1 default to the box)
              "occlude": [{"box": [4240, 758, 4506, 953], "method": "rect"},
                          {"box": [4262, 945, 4300, 992], "method": "dark"}],
              "note": "AC unit, top 1.75 m"},
       "pipeB": {"kind": "pipe", "box": [...], "contacts": {"pole": {"x": 5602}}, "occlude": "dark"}, ...},
     "no_paint": [{"id": "alley", "box": [1962, 0, 2698, 1643]}, ...],
     "layers": [{"id": "redlight", "kind": "pole", "parallax": 0.12, "x": 2002,
                 "parts": [{"box": [...], "method": "dark", "vmax": 0.4}, {"box": [...], "method": "rect"}]}],
     "lights": [{"id": "red", "x": 2080, "y": 535, "sigma": 420, "strength": 0.6, "color": [1.0, 0.07, 0.05]}]}

Contacts are lines (`{"y": ...}` horizontal, `{"x": ...}` vertical, with optional x0/x1 or y0/y1) or points
(`{"pt": [x, y]}`): the surface he stands on, the bar he grabs, the top he plants his hands on. Occluder methods:
`rect` (the whole box), `dark` (black iron inside the box: dark and not warm; wet iron is neutral or bluish, the
darkest brick is still warm), `mask` (a hand-painted PNG the size of the box, white = in front of the wall).
A gap obstacle (kind `gap`) is never paintable and can give the J2 air layer its glow colour (`glow`).

Light maps (as in prototype 3 and 6): relief (local luminance over its sigma-10 blur), wet speculars (linear light
over its sigma-4 blur), and the illumination on the wall: local light (sigma 60) over the street's large-scale
level (sigma 900 along x), so pale render and dark brick take primer alike while lamp pools survive, and the local
light colour (neutral on a wide local average, a little warm). The low-frequency parts come from a 1/8 overview of
the whole street (cached by content hash, small); the rest is computed per window from the street tiles. Occluder
and paintable masks are cached per tile by content hash (8-bit PNG) in projects/<p>/paint/.cache/.

Prepended street (x < 0): the low-frequency maps right of x 0 are computed from the tiles right of x 0 only, exactly
as before the plates were prepended (same cache, same values), and the street left of x 0 gets its own maps (with a
little of the right side as context) that blend into the right side's edge values over the 2048 px left of x 0. So
prepending never changes how anything right of x 0 is lit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..util import read_json
from .street import Street

VERSION = "scene-1"  # bump when a map's formula changes (part of every cache key)
KINDS = ("edge", "gap", "ledge", "pipe", "beam", "ac", "step", "ladder", "door", "glass", "wall", "pole", "box",
         "sign", "shutter", "window", "other", "cabinet", "bin", "bench", "railing", "wire", "bollard")
# solid obstacles: nothing of the figure may be hidden behind them (a move can say `behind: <obstacle>`), and
# within their occluders he must be above their top (hands and feet may touch it); paint may pass behind thin ones
SOLID_KINDS = ("box", "cabinet", "ac", "bin", "bench", "step", "ledge", "door", "sign", "bollard")
SRGB_LUT = np.where(np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92,
                    ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb2lin(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return SRGB_LUT[x]
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin2srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin: np.ndarray) -> np.ndarray:
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ registry

@dataclass
class Contact:
    name: str
    x: float | None = None  # vertical line at x (or a point with y)
    y: float | None = None  # horizontal line at y
    x0: float | None = None
    x1: float | None = None
    y0: float | None = None
    y1: float | None = None

    @property
    def horizontal(self) -> bool:
        return self.y is not None and self.x is None

    @property
    def vertical(self) -> bool:
        return self.x is not None and self.y is None

    def dist(self, px: float, py: float) -> float:
        """Distance of a point from the contact (line segment or point)."""
        if self.horizontal:
            dx = max(self.x0 - px, 0.0, px - self.x1)
            return float(np.hypot(dx, py - self.y))
        if self.vertical:
            dy = max(self.y0 - py, 0.0, py - self.y1)
            return float(np.hypot(px - self.x, dy))
        return float(np.hypot(px - self.x, py - self.y))


@dataclass
class Obstacle:
    id: str
    kind: str
    box: list[float]
    contacts: dict[str, Contact] = field(default_factory=dict)
    occlude: list[dict] = field(default_factory=list)
    glow: list[float] | None = None
    note: str = ""
    solid: bool = False  # see SOLID_KINDS; a part can say "solid": false (a bracket, a lamp, a conduit)
    wall: bool = False  # a gap with the wall behind it (between awnings, a scaffold's end and a ledge): paint may go
    #                     there and it is no void (only the route treats it as a gap: no ground)

    @property
    def x0(self) -> float:
        return float(self.box[0])

    @property
    def x1(self) -> float:
        return float(self.box[2])

    @property
    def y0(self) -> float:
        return float(self.box[1])

    @property
    def y1(self) -> float:
        return float(self.box[3])

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    def contact(self, *names: str) -> Contact:
        for n in names:
            if n in self.contacts:
                return self.contacts[n]
        raise KeyError(f"obstacle {self.id!r} ({self.kind}) has no contact {' or '.join(names)!r} "
                       f"(has {', '.join(self.contacts) or 'none'})")


def _contact(name: str, c: dict, box: list[float]) -> Contact:
    if "pt" in c:
        return Contact(name, x=float(c["pt"][0]), y=float(c["pt"][1]))
    k = Contact(name, x=c.get("x"), y=c.get("y"))
    if k.y is not None and k.x is None:
        k.x0, k.x1 = float(c.get("x0", box[0])), float(c.get("x1", box[2]))
    elif k.x is not None and k.y is None:
        k.y0, k.y1 = float(c.get("y0", box[1])), float(c.get("y1", box[3]))
    elif k.x is None and k.y is None:
        raise ValueError(f"contact {name}: give x, y or pt")
    return k


def load_registry(d: Path) -> dict:
    f = d / "obstacles.json"
    if not f.exists():
        raise SystemExit(f"No obstacle registry: {f}")
    spec = read_json(f)
    g = spec.get("ground") or {}
    for k in ("wall_base", "feet"):
        if k not in g:
            raise SystemExit(f"{f}: ground.{k} missing")
    g.setdefault("kerb_top", g["wall_base"] + 150)
    g.setdefault("kerb_bot", g["kerb_top"] + 50)
    obs = {}
    for oid, o in (spec.get("obstacles") or {}).items():
        if "box" not in o or len(o["box"]) != 4:
            raise SystemExit(f"{f}: obstacle {oid} needs a box [x0, y0, x1, y1]")
        kind = o.get("kind", "other")
        if kind not in KINDS:
            raise SystemExit(f"{f}: obstacle {oid}: unknown kind {kind!r} (one of {', '.join(KINDS)})")
        occ = o.get("occlude", [])
        if isinstance(occ, str):
            occ = [{"box": o["box"], "method": occ}]
        try:
            contacts = {n: _contact(n, c, o["box"]) for n, c in (o.get("contacts") or {}).items()}
        except ValueError as e:
            raise SystemExit(f"{f}: obstacle {oid}: {e}") from e
        obs[oid] = Obstacle(oid, kind, [float(v) for v in o["box"]], contacts, occ, o.get("glow"), o.get("note", ""),
                            bool(o.get("solid", kind in SOLID_KINDS)), bool(o.get("wall", False)))
    spec["ground"] = g
    spec["obstacles_parsed"] = obs
    return spec


def _savez(f: Path, **arrays) -> None:
    """np.savez, atomic: render workers share the cache, and one may read a file another is still writing."""
    tmp = f.parent / f".tmp-{os.getpid()}-{f.name}"
    np.savez(tmp, **arrays)
    os.replace(tmp, f)


def _imwrite(f: Path, img: np.ndarray) -> None:
    """cv2.imwrite, atomic (as _savez)."""
    tmp = f.parent / f".tmp-{os.getpid()}-{f.name}"
    cv2.imwrite(str(tmp), img)
    os.replace(tmp, f)


def _levels(spec: list) -> list[dict]:
    """ground.levels: raised grounds (a walkway, a ledge, the roofs) over a street x range, each with its feet line
    (a number, or a polyline [[x, y], ...] in street px, linear between its points and constant beyond them).
    "street": true makes a level the street's own ground there (the pavement follows it: ground_dy)."""
    out = []
    for lv in spec:
        if "x" not in lv or "feet" not in lv:
            raise SystemExit(f"ground.levels: a level needs x: [x0, x1] and feet (a y or [[x, y], ...]): {lv}")
        x0, x1 = (float(v) for v in lv["x"])
        f = lv["feet"]
        if isinstance(f, int | float):
            px, py = [x0, x1], [float(f), float(f)]
        else:
            pts = sorted((float(p[0]), float(p[1])) for p in f)
            px, py = [p[0] for p in pts], [p[1] for p in pts]
        out.append({"id": str(lv.get("id", f"level{len(out) + 1}")), "x0": x0, "x1": x1, "px": px, "py": py,
                    "street": bool(lv.get("street", False))})
    return out


def _nblur(a: np.ndarray, w: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Gaussian blur of a (2D or H x W x C) normalized over the weights w (where w is 0 the result is what the
    weighted neighbours give)."""
    if a.ndim == 3:
        num = cv2.GaussianBlur(a * w[..., None], (0, 0), sigmaX=sx, sigmaY=sy)
        den = cv2.GaussianBlur(w, (0, 0), sigmaX=sx, sigmaY=sy)[..., None]
    else:
        num = cv2.GaussianBlur(a * w, (0, 0), sigmaX=sx, sigmaY=sy)
        den = cv2.GaussianBlur(w, (0, 0), sigmaX=sx, sigmaY=sy)
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


def layer_rows(L: dict, key: str, y0: int, y1: int) -> np.ndarray:
    """Rows [y0, y1) (street y) of a parallax layer's array (alpha, fg, bg, or a lamp mask), zero outside its rows:
    a layer holds the street's rows 0 to H, or (in the world) the rows round its parts from L["y0"]."""
    return rows_of(L[key], int(L.get("y0", 0)), y0, y1)


def rows_of(a: np.ndarray, o: int, y0: int, y1: int) -> np.ndarray:
    """Rows [y0, y1) (street y) of an array whose row 0 is street y o, zero outside its rows (a view when inside)."""
    if y0 >= o and y1 <= o + a.shape[0]:
        return a[y0 - o:y1 - o]
    out = np.zeros((y1 - y0,) + a.shape[1:], a.dtype)
    b0, b1 = max(y0, o), min(y1, o + a.shape[0])
    if b1 > b0:
        out[b0 - y0:b1 - y0] = a[b0 - o:b1 - o]
    return out


# ------------------------------------------------------------------------------------------------ masks

def dark_mask(rgb: np.ndarray, vmax: float = 0.30) -> np.ndarray:
    """Black iron (pipes, railings, ladder, pole) in an RGB float crop: dark and not warm."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    m = ((b - r >= -0.012) & (v < vmax)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _part_mask(street: Street, part: dict, base: Path) -> tuple[np.ndarray, list[int]]:
    """Mask of one occluder or layer part, over its (clipped) box."""
    x0, y0, x1, y1 = [int(round(v)) for v in part["box"]]
    y0, y1 = max(y0, street.Y0), min(y1, street.Y1)  # the street's rows (0 to H), or the world's
    method = part.get("method", "rect")
    if method == "rect":
        return np.ones((y1 - y0, x1 - x0), np.float32), [x0, y0, x1, y1]
    if method == "dark":
        rgb = street.window(x0, x1, y0, y1).astype(np.float32) / 255
        return dark_mask(rgb, float(part.get("vmax", 0.30))), [x0, y0, x1, y1]
    if method == "mask":
        m = cv2.imread(str(base / part["mask"]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise SystemExit(f"occluder mask not found: {base / part['mask']}")
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return m, [x0, y0, x1, y1]
    raise SystemExit(f"unknown occluder method {method!r} (rect, dark, mask)")


class Scene:
    """Obstacles plus windowed light maps and masks for one built street."""

    def __init__(self, street_dir: Path, cache_dir: Path):
        self.d = Path(street_dir)
        self.street = Street(self.d)
        self.spec = load_registry(self.d)
        self.ground = self.spec["ground"]
        self.wall_base = int(self.ground["wall_base"])
        self.feet = float(self.ground["feet"])
        self.kerb_top, self.kerb_bot = int(self.ground["kerb_top"]), int(self.ground["kerb_bot"])
        prof = self.ground.get("dy") or []
        self._gx = np.float64([p[0] for p in prof]) if prof else None
        self._gdy = np.float64([p[1] for p in prof]) if prof else None
        self.obstacles: dict[str, Obstacle] = self.spec["obstacles_parsed"]
        self.no_paint = [z for z in self.spec.get("no_paint", [])]
        for o in self.obstacles.values():
            if o.kind == "gap" and not o.wall:  # the gap's rows and the street's (a gap between roofs goes down
                y0_ = 0 if o.y0 >= 0 else o.y0  # to the street)
                self.no_paint.append({"id": o.id, "box": [o.x0 + 2, y0_, o.x1 - 2,
                                                          self.street.H if o.y1 <= self.street.H else o.y1]})
        self.layers = self.spec.get("layers", [])
        if self.street.world is not None:  # a layer whose parts lie beyond the street and the world (a retired
            # cell's lamp) has no pixels to stand in front of: left out (its light still glows where it is in view)
            bx0, _, bx1, _ = self.street.bounds
            self.layers = [L for L in self.layers if any(p["box"][2] > bx0 and p["box"][0] < bx1 for p in L["parts"])]
        self.lights = self.spec.get("lights", [])
        # optional colour cast on the light colour map (prototype 6 had about [0.97, 1.10, 1.0] by accident: a float32
        # mean over the whole street; the port is neutral)
        self.tint_gain = np.float32((self.spec.get("look") or {}).get("tint_gain", [1.0, 1.0, 1.0]))
        self.H, self.W = self.street.H, self.street.W
        self.X0 = self.street.X0  # the street's left end (negative with prepended plates)
        # the world round the street (world.py): rows Y0 to Y1 (0 and H without one), bounds (x0, y0, x1, y1), and
        # its first column xw (None without one); levels (raised grounds) and voids (sky, gaps: never paintable)
        self.world = self.street.world
        self.Y0, self.Y1, self.bounds, self.xw = self.street.Y0, self.street.Y1, self.street.bounds, self.street.xw
        self.bands = list(getattr(self.street, "bands", []) or [])  # world cells above the street's left part
        self.levels = _levels(self.ground.get("levels") or [])
        self._street_levels = [lv for lv in self.levels if lv["street"]]
        self.voids = list(self.spec.get("voids") or [])
        self._lf_world = None
        self._lf_bands: dict[tuple, dict] = {}
        self._tile_masks2: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._void_cache: dict[str, np.ndarray] = {}
        self.cache = Path(cache_dir) / "scene" / self.d.name
        self.cache.mkdir(parents=True, exist_ok=True)
        self._lf = None
        self._lf_left = None
        self._layer_cache: dict[str, dict] = {}
        self._tile_masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def ground_dy(self, x):
        """The ground line's offset at street x (0 without a ground.dy profile); x may be an array. A street level
        (ground.levels with "street": true: the world's own street level, drifted with its cells) sets it over its
        x range: the feet line and the pavement move together there."""
        if self._street_levels:
            xa = np.asarray(x, dtype=np.float64)
            v = np.interp(xa, self._gx, self._gdy) if self._gx is not None else np.zeros(xa.shape)
            v = np.array(v, dtype=np.float64)
            for lv in self._street_levels:
                m = (xa >= lv["x0"]) & (xa < lv["x1"])
                if np.any(m):
                    v = np.where(m, np.interp(xa, lv["px"], lv["py"]) - self.feet, v)
            return float(v) if np.isscalar(x) else v
        if self._gx is None:
            return 0.0 if np.isscalar(x) else np.zeros(np.shape(x))
        v = np.interp(x, self._gx, self._gdy)
        return float(v) if np.isscalar(x) else v

    def feet_at(self, x: float) -> float:
        """The ground line feet stand on at street x: the level holding x (ground.levels; the last one listed wins
        where two overlap), else the street's (feet plus the dy profile). Steps at a level's ends."""
        for lv in reversed(self.levels):
            if lv["x0"] <= x < lv["x1"]:
                return float(np.interp(x, lv["px"], lv["py"]))
        return self.feet + self.ground_dy(x)

    def level_at(self, x: float) -> str:
        """The id of the level whose ground holds street x ("street" outside every level)."""
        for lv in reversed(self.levels):
            if lv["x0"] <= x < lv["x1"]:
                return lv["id"]
        return "street"

    def wall_base_at(self, x: float) -> float:
        return self.wall_base + self.ground_dy(x)

    def obstacle(self, oid: str) -> Obstacle:
        if oid not in self.obstacles:
            raise KeyError(f"unknown obstacle {oid!r} (known: {', '.join(self.obstacles)})")
        return self.obstacles[oid]

    # -------------------------------------------------------------- low-frequency light (whole street, 1/8)
    def _lf_key(self) -> str:
        h = hashlib.sha256(VERSION.encode())
        h.update(json.dumps(self.street.light_hashes, sort_keys=True).encode())
        h.update(str(self.wall_base).encode())
        return h.hexdigest()[:16]

    def lf(self) -> dict:
        if self._lf is not None:
            return self._lf
        f = self.cache / f"lf-{self._lf_key()}.npz"
        if f.exists():
            try:
                self._lf = dict(np.load(f))
                return self._lf
            except (FileNotFoundError, OSError, ValueError):  # replaced by another process as we read: rebuild
                pass
        for old in self.cache.glob("lf-*.npz"):
            old.unlink(missing_ok=True)
        parts_l, parts_c = [], []
        H8 = self.H // 8
        for k in range(self.street.n):
            t = srgb2lin(self.street.light_rgb(k))  # the right side's own plates (no join patch)
            w8 = t.shape[1] // 8
            parts_l.append(cv2.resize(luminance(t), (w8, H8), interpolation=cv2.INTER_AREA))
            parts_c.append(cv2.resize(t, (w8, H8), interpolation=cv2.INTER_AREA))
        small = np.concatenate(parts_l, axis=1)
        small_c = np.concatenate(parts_c, axis=1)
        loc = cv2.GaussianBlur(small, (0, 0), 60 / 8)
        wall_rows = small[: self.wall_base // 8]
        level = cv2.GaussianBlur(np.median(wall_rows, axis=0).reshape(1, -1).astype(np.float32), (0, 0),
                                 sigmaX=900 / 8, sigmaY=0.1)
        col = cv2.GaussianBlur(small_c, (0, 0), 60 / 8)
        tint_s = col / (loc[..., None] + 1e-4)
        # neutral on a wide local average (prototype: the whole street's mean), so a longer street never shifts it
        norm = cv2.GaussianBlur(tint_s.mean(0, keepdims=True), (0, 0), sigmaX=3000 / 8, sigmaY=0.1)
        self._lf = {"loc": loc.astype(np.float32), "level": level.astype(np.float32),
                    "col": col.astype(np.float32), "norm": norm.astype(np.float32),
                    "sy": np.float32(H8 / self.H)}
        _savez(f, **self._lf)
        return self._lf

    def lf_left(self) -> dict | None:
        """Low-frequency maps of the street left of x 0 (prepended plates), from X0 to a little right of x 0 for
        context; None without prepended plates. Same formulas as lf(); "x0" is the street x of column 0."""
        if self.X0 >= 0:
            return None
        if self._lf_left is not None:
            return self._lf_left
        t = self.street.tile
        xa = int(np.floor(self.X0 / 8.0)) * 8
        xb = min(3 * t, self.W // 8 * 8)
        h = hashlib.sha256((VERSION + "left").encode())
        h.update(json.dumps(self.street.left_hashes, sort_keys=True).encode())
        h.update(json.dumps([self.street.hashes.get(k) for k in range(3)]).encode())
        h.update(f"{self.wall_base} {xa} {xb}".encode())
        f = self.cache / f"lfl-{h.hexdigest()[:16]}.npz"
        if f.exists():
            try:
                self._lf_left = dict(np.load(f))
                return self._lf_left
            except (FileNotFoundError, OSError, ValueError):  # replaced by another process as we read: rebuild
                pass
        for old in self.cache.glob("lfl-*.npz"):
            old.unlink(missing_ok=True)
        parts_l, parts_c = [], []
        H8 = self.H // 8
        for c0 in range(xa, xb, t):
            c1 = min(c0 + t, xb)
            win = srgb2lin(self.street.window(c0, c1))
            parts_l.append(cv2.resize(luminance(win), ((c1 - c0) // 8, H8), interpolation=cv2.INTER_AREA))
            parts_c.append(cv2.resize(win, ((c1 - c0) // 8, H8), interpolation=cv2.INTER_AREA))
        small = np.concatenate(parts_l, axis=1)
        small_c = np.concatenate(parts_c, axis=1)
        loc = cv2.GaussianBlur(small, (0, 0), 60 / 8)
        level = cv2.GaussianBlur(np.median(small[: self.wall_base // 8], axis=0).reshape(1, -1).astype(np.float32),
                                 (0, 0), sigmaX=900 / 8, sigmaY=0.1)
        col = cv2.GaussianBlur(small_c, (0, 0), 60 / 8)
        tint_s = col / (loc[..., None] + 1e-4)
        norm = cv2.GaussianBlur(tint_s.mean(0, keepdims=True), (0, 0), sigmaX=3000 / 8, sigmaY=0.1)
        self._lf_left = {"loc": loc.astype(np.float32), "level": level.astype(np.float32),
                         "col": col.astype(np.float32), "norm": norm.astype(np.float32),
                         "sy": np.float32(H8 / self.H), "x0": np.int32(xa)}
        _savez(f, **self._lf_left)
        return self._lf_left

    def _up(self, a: np.ndarray, x0: int, w: int, sy: float, rows: int, origin: int = 0) -> np.ndarray:
        """Upsample a 1/8 map to street columns [x0, x0 + w) (as cv2.resize INTER_LINEAR on the whole street); origin:
        the street x of the map's column 0."""
        M = np.float32([[1 / 8, 0, (x0 - origin + 0.5) / 8 - 0.5], [0, sy, 0.5 * sy - 0.5]])
        return cv2.warpAffine(a, M, (w, rows), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    # -------------------------------------------------------------- masks per tile (cached by content hash)
    def _tile_key(self, k: int, parts: list[dict]) -> str:
        h = hashlib.sha256(VERSION.encode())
        for j in (k - 1, k, k + 1):
            # tile 0's key ignores the street left of x 0 unless a part reaches into it (prepending keeps the caches)
            if k == 0 and j == -1 and not any(p["box"][0] < 0 for p in parts):
                j = None
            h.update(str(self.street.tile_hash(j) if j is not None else None).encode())
        h.update(json.dumps(parts, sort_keys=True).encode())
        # the ground lines (the dy profile and notes enter only through the tiles they change, as parts)
        g = {k: v for k, v in self.ground.items() if k != "dy" and not k.startswith("note")}
        h.update(json.dumps(g, sort_keys=True).encode())
        for p in parts:
            if p.get("method") == "mask":
                mf = self.d / p["mask"]
                h.update(mf.read_bytes() if mf.exists() else b"missing")
        return h.hexdigest()[:16]

    def occluder_parts(self) -> list[dict]:
        return [dict(part, obstacle=o.id) for o in self.obstacles.values() for part in o.occlude]

    def part_solid(self, part: dict) -> bool:
        o = self.obstacles[part["obstacle"]]
        return bool(part.get("solid", o.solid))

    def occluders(self, x0: int, x1: int, y0: int, y1: int, solid: bool | None = None,
                  skip: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
        """Occluder mask over street [x0, x1) x [y0, y1) (float 0..1): solid parts (solid=True), thin ones (False) or
        all (None), leaving out the obstacles in `skip`; and the touch band (the top 14 px of each solid part, where
        hands and feet may rest on it)."""
        occ = np.zeros((y1 - y0, x1 - x0), np.float32)
        touch = np.zeros_like(occ)
        for p in self.occluder_parts():
            if p["obstacle"] in skip or (solid is not None and self.part_solid(p) != solid):
                continue
            bx0, by0, bx1, by1 = p["box"]
            if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
                continue
            pm, (px0, py0, px1, py1) = _part_mask(self.street, p, self.d)
            a0, b0 = max(px0, x0), max(py0, y0)
            a1, b1 = min(px1, x1), min(py1, y1)
            if a1 > a0 and b1 > b0:
                sub = pm[b0 - py0:b1 - py0, a0 - px0:a1 - px0]
                occ[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = np.maximum(occ[b0 - y0:b1 - y0, a0 - x0:a1 - x0], sub)
                if self.part_solid(p):
                    t1 = min(b1, py0 + 14)
                    if t1 > b0:
                        touch[b0 - y0:t1 - y0, a0 - x0:a1 - x0] = 1.0
        return occ, touch

    def tile_masks(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(occ, paintable) float32 for street tile k, from the cache when its inputs are unchanged."""
        if k in self._tile_masks:
            return self._tile_masks[k]
        X0, X1 = self.street.tile_span(k)
        m = 16
        parts = [p for p in self.occluder_parts() if p["box"][2] > X0 - m and p["box"][0] < X1 + m]
        zones = [z for z in self.no_paint if z["box"][2] > X0 - m and z["box"][0] < X1 + m]
        gdy = np.round(self.ground_dy(np.arange(X0 - m, X1 + m, dtype=np.float64))).astype(int)
        extra = [{"ground_dy": [int(gdy.min()), int(gdy.max()), int(gdy.sum())]}] if gdy.any() else []
        key = self._tile_key(k, parts + zones + extra)
        fo, fp = self.cache / f"occ-{k:04d}-{key}.png", self.cache / f"paint-{k:04d}-{key}.png"
        occ = pnt = None
        if fo.exists() and fp.exists():  # (another process may replace them as we read: then rebuilt here)
            o_, p_ = cv2.imread(str(fo), cv2.IMREAD_UNCHANGED), cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if o_ is not None and p_ is not None:
                occ, pnt = o_.astype(np.float32) / 65535, p_.astype(np.float32) / 65535
        if occ is None:
            for old in list(self.cache.glob(f"occ-{k:04d}-*.png")) + list(self.cache.glob(f"paint-{k:04d}-*.png")):
                old.unlink(missing_ok=True)
            a, b = X0 - m, X1 + m
            occ = np.zeros((self.H, b - a), np.float32)
            for p in parts:
                pm, (x0, y0, x1, y1) = _part_mask(self.street, p, self.d)
                c0, c1 = max(x0, a), min(x1, b)
                r0, r1 = max(y0, 0), min(y1, self.H)  # (a part reaching above the street: its street rows here)
                if c1 > c0 and r1 > r0:
                    occ[r0:r1, c0 - a:c1 - a] = np.maximum(occ[r0:r1, c0 - a:c1 - a],
                                                           pm[r0 - y0:r1 - y0, c0 - x0:c1 - x0])
            occ = cv2.GaussianBlur(occ, (0, 0), 0.8)[:, m:m + X1 - X0]
            pnt = np.zeros((self.H, b - a), np.float32)
            if gdy.any():  # the paintable wall ends at the ground line, column by column
                pnt[np.arange(self.H)[:, None] < (self.wall_base - 6 + gdy)[None, :]] = 1
            else:
                pnt[:self.wall_base - 6] = 1
            for z in zones:
                x0, y0, x1, y1 = [int(round(v)) for v in z["box"]]
                pnt[max(y0, 0):max(y1, 0), max(x0 - a, 0):max(x1 - a, 0)] = 0
            pnt = cv2.GaussianBlur(pnt, (0, 0), 1.5)[:, m:m + X1 - X0]
            _imwrite(fo, (np.clip(occ, 0, 1) * 65535 + 0.5).astype(np.uint16))
            _imwrite(fp, (np.clip(pnt, 0, 1) * 65535 + 0.5).astype(np.uint16))
        if len(self._tile_masks) > 6:
            self._tile_masks.pop(next(iter(self._tile_masks)))
        self._tile_masks[k] = (occ, pnt)
        return occ, pnt

    def masks(self, x0: int, x1: int, y0: int | None = None, y1: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(occ, paintable) float32 over street [x0, x1) x [y0, y1) (rows default 0 to H). On the plain street the
        street tiles' own masks (cached per tile); in the world the world's mask tiles (2D, cached per tile); above
        the street's left part, inside a band of floor cells (v8), the world's mask tiles too (nothing elsewhere)."""
        r0, r1 = (0 if y0 is None else int(y0)), (self.H if y1 is None else int(y1))
        xb = min(max(self._x1d(), x0), x1)  # columns left of xb: the street tiles' own masks
        if xb == x1 and (r0, r1) == (0, self.H):
            return self._masks1d(x0, x1)
        occ = np.zeros((r1 - r0, x1 - x0), np.float32)
        pnt = np.zeros((r1 - r0, x1 - x0), np.float32)
        T = self.street.tile
        if xb > x0:
            a, b = max(r0, 0), min(r1, self.H)
            if b > a:
                o, p = self._masks1d(x0, xb)
                occ[a - r0:b - r0, :xb - x0] = o[a:b]
                pnt[a - r0:b - r0, :xb - x0] = p[a:b]
            for bx0, by0, bx1, by1 in self.bands:  # above the street's left part, a band of floor cells: its rows
                c0, c1 = max(x0, int(bx0)), min(xb, int(bx1))  # from the world's own mask tiles (paintable where
                ra, rb = max(r0, int(by0)), min(r1, int(by1), 0)  # the cells are; their occluders, voids, no-paint)
                if c1 <= c0 or rb <= ra:
                    continue
                for iy in range(ra // T, (rb - 1) // T + 1):
                    for ix in range(c0 // T, (c1 - 1) // T + 1):
                        o, p = self.tile_masks2(ix, iy)
                        a0, b0 = max(c0, ix * T), max(ra, iy * T)
                        a1, b1 = min(c1, (ix + 1) * T), min(rb, (iy + 1) * T)
                        occ[b0 - r0:b1 - r0, a0 - x0:a1 - x0] = o[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
                        pnt[b0 - r0:b1 - r0, a0 - x0:a1 - x0] = p[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
        if x1 > xb:
            for iy in range(r0 // T, (r1 - 1) // T + 1):
                for ix in range(xb // T, (x1 - 1) // T + 1):
                    o, p = self.tile_masks2(ix, iy)
                    a0, b0 = max(xb, ix * T), max(r0, iy * T)
                    a1, b1 = min(x1, (ix + 1) * T), min(r1, (iy + 1) * T)
                    occ[b0 - r0:b1 - r0, a0 - x0:a1 - x0] = o[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
                    pnt[b0 - r0:b1 - r0, a0 - x0:a1 - x0] = p[b0 - iy * T:b1 - iy * T, a0 - ix * T:a1 - ix * T]
        return occ, pnt

    def _masks1d(self, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray]:
        cols = np.clip(np.arange(x0, x1), self.X0, self.W - 1)
        occ = np.zeros((self.H, x1 - x0), np.float32)
        pnt = np.zeros((self.H, x1 - x0), np.float32)
        t = self.street.tile
        for k in range(int(cols[0]) // t, int(cols[-1]) // t + 1):
            o, p = self.tile_masks(k)
            s = self.street.tile_span(k)[0]
            sel = (cols >= s) & (cols < s + o.shape[1])
            occ[:, sel] = o[:, cols[sel] - s]
            pnt[:, sel] = p[:, cols[sel] - s]
        return occ, pnt

    # -------------------------------------------------------------- parallax layers
    def layer(self, L: dict) -> dict:
        """A parallax layer (its alpha and colour) and the street behind it rebuilt by row interpolation."""
        lid = L["id"]
        if lid in self._layer_cache:
            return self._layer_cache[lid]
        parts = L["parts"]
        bx0 = int(min(p["box"][0] for p in parts))
        bx1 = int(max(p["box"][2] for p in parts))
        top = int(min(p["box"][1] for p in parts)) - 4
        bot = int(L.get("bottom", max(p["box"][3] for p in parts) + 2))
        reach = int(L.get("reach", 140))  # max parallax shift the rebuilt background must cover
        # rows: the street's (0 to H), or the world's round the parts when they reach above or below the street
        two_d = (any(p["box"][1] < 0 or p["box"][3] > self.H for p in parts) or bx1 > self.W or bx0 < self.X0) \
            and self.world is not None  # rows outside the street's, or columns in the world beyond it
        ly0 = max(top - 64, self.Y0) if two_d else 0
        ly1 = min(bot + 64, self.Y1) if two_d else self.H
        if two_d:
            a0, a1 = max(bx0 - reach - 40, self.bounds[0]), min(bx1 + reach + 40, self.bounds[2])
        else:
            a0, a1 = max(bx0 - reach - 40, self.X0), min(bx1 + reach + 40, self.W)
        win = (lambda: self.street.window(a0, a1, ly0, ly1)) if two_d else (lambda: self.street.window(a0, a1))
        h = hashlib.sha256((VERSION + "fill-mirror" + json.dumps(L, sort_keys=True)).encode())
        h.update(win().tobytes())
        f = self.cache / f"layer-{lid}-{h.hexdigest()[:16]}.npz"
        d = None
        if f.exists():
            try:
                d = dict(np.load(f))
            except (FileNotFoundError, OSError, ValueError):  # another process replaced it as we read: rebuild
                d = None
        if d is None:
            for old in self.cache.glob(f"layer-{lid}-*.npz"):
                if old != f:
                    old.unlink(missing_ok=True)
            street = win().astype(np.float32) / 255
            H, w = street.shape[:2]
            alpha = np.zeros((H, w), np.float32)
            for p in parts:
                pm, (x0, y0, x1, y1) = _part_mask(self.street, p, self.d)
                y0, y1 = y0 - ly0, y1 - ly0
                alpha[y0:y1, x0 - a0:x1 - a0] = np.maximum(alpha[y0:y1, x0 - a0:x1 - a0], pm)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
            bg = street.copy()
            hole = cv2.dilate((alpha > 0.05).astype(np.uint8), np.ones((5, 31), np.uint8)) > 0
            mirror = lambda i: (w - 1) - np.abs((w - 1) - np.abs(i))  # an index reflected into the window
            for y in range(max(top - ly0, 0), min(bot - ly0, H)):
                xs = np.nonzero(hole[y])[0]
                if len(xs) == 0:
                    continue
                # the street behind it, row by row: the wall's own texture either side mirrored into the gap and
                # cross-faded (seamless at both edges; a flat blend between the edge colours read as a streaky panel)
                l, r = max(xs.min() - 2, 1), min(xs.max() + 2, w - 2)
                i = np.arange(r - l + 1)
                tt = np.linspace(0, 1, r - l + 1, dtype=np.float32)[:, None]
                bg[y, l:r + 1] = street[y, mirror(l - 1 - i)] * (1 - tt) + street[y, mirror(r + 1 + i[::-1])] * tt
            d = {"a0": np.int32(a0), "alpha": alpha.astype(np.float32), "bg": bg.astype(np.float32),
                 "fg": street.astype(np.float32)}
            if two_d:
                d["y0"] = np.int32(ly0)
            _savez(f, **d)
        d["a0"] = int(d["a0"])
        d["y0"] = int(d.get("y0", 0))  # the street y of its row 0 (layer_rows reads rows by street y)
        d["parallax"] = float(L.get("parallax", 0.1))
        d["x"] = float(L.get("x", (bx0 + bx1) / 2))
        d["id"] = lid
        if d["parallax"] == 0:  # it never moves against the plate: the street behind it is the plate itself; a
            # light's lamp (a light's `layer`: a lamp on the roofs, there to blink) gets each rect part (its head) as a
            # soft round falloff from the centre (full in the middle 40% of the radius), never a square
            d["bg"] = d["fg"]
            for p in parts if lid in {lt.get("layer") for lt in self.lights} else []:
                if p.get("method", "rect") == "rect":
                    x0, y0, x1, y1 = (float(v) for v in p["box"])
                    c0, r0 = max(int(x0) - d["a0"] - 4, 0), max(int(y0) - d["y0"] - 4, 0)
                    H_, W_ = d["alpha"].shape
                    yy, xx = np.ogrid[r0:min(int(y1) - d["y0"] + 4, H_), c0:min(int(x1) - d["a0"] + 4, W_)]
                    r = np.hypot(xx - ((x0 + x1) / 2 - d["a0"]), yy - ((y0 + y1) / 2 - d["y0"]))
                    r = r / (min(x1 - x0, y1 - y0) / 2)
                    k = np.clip((1 - r) / 0.6, 0, 1).astype(np.float32)
                    d["alpha"][r0:r0 + k.shape[0], c0:c0 + k.shape[1]] = k * k * (3 - 2 * k)
        self._layer_cache[lid] = d
        return d

    # -------------------------------------------------------------- a window of maps
    def span(self, x0: int, x1: int, y0: int | None = None, y1: int | None = None) -> "Span":
        """All maps for street columns [x0, x1), full height (computed here, never stored whole); rows [y0, y1)
        where the world reaches above or below the street."""
        if (y0 is not None or y1 is not None) or x1 + 48 > min(self._x1d(), (self.xw or np.inf) - 2048):
            return self._span2d(x0, x1, y0, y1)
        m = 48  # the widest high-frequency blur (sigma 10) needs about 3 sigma of context
        rgb8 = self.street.window(x0 - m, x1 + m)
        lin_full = srgb2lin(rgb8)
        lum = luminance(lin_full)
        lum_s = cv2.GaussianBlur(lum, (0, 0), 1.2)
        relief = np.clip(lum_s / (cv2.GaussianBlur(lum, (0, 0), 10) + 1e-4), 0.55, 1.45) ** 0.8
        spec = np.clip(lin_full - cv2.GaussianBlur(lin_full, (0, 0), 4) * 1.08, 0, None)
        sl = slice(m, m + x1 - x0)
        lf = self.lf()
        w = x1 - x0
        sy = float(lf["sy"])
        loc = self._up(lf["loc"], x0, w, sy, self.H)
        level = self._up(lf["level"], x0, w, 1.0, 1)
        rel = np.clip(loc / (level + 1e-4), 0.55, 2.2) ** 0.6
        e_wall = (0.30 * rel).astype(np.float32)
        col = self._up(lf["col"], x0, w, sy, self.H)
        norm = self._up(lf["norm"], x0, w, 1.0, 1)
        ll = self.lf_left() if x0 < 0 else None
        if ll is not None:  # left of x 0: the left side's maps, blended into the right side's edge values near x 0
            o = int(ll["x0"])
            r = np.clip((np.arange(x0, x1, dtype=np.float32) + 2048) / 2048, 0, 1)
            r = np.where(np.arange(x0, x1) < 0, r, 1.0).astype(np.float32)
            mix = lambda L, R: (L * (1 - r) + R * r).astype(np.float32) if L.ndim == 2 else \
                (L * (1 - r)[None, :, None] + R * r[None, :, None]).astype(np.float32)
            loc = mix(self._up(ll["loc"], x0, w, sy, self.H, o), loc)
            level = mix(self._up(ll["level"], x0, w, 1.0, 1, o), level)
            rel = np.clip(loc / (level + 1e-4), 0.55, 2.2) ** 0.6
            e_wall = (0.30 * rel).astype(np.float32)
            col = mix(self._up(ll["col"], x0, w, sy, self.H, o), col)
            norm = mix(self._up(ll["norm"], x0, w, 1.0, 1, o), norm)
        tint = (1 + 0.3 * (col / (loc[..., None] + 1e-4) / norm - 1)) * np.float32([1.03, 1.0, 0.95]) * self.tint_gain
        occ, pnt = self.masks(x0, x1)
        lin = lin_full[:, sl].copy()
        pole_layers = []
        for L in self.layers:
            d = self.layer(L)
            a0, lw = d["a0"], d["bg"].shape[1]
            c0, c1 = max(a0, x0), min(a0 + lw, x1)
            if c1 > c0:  # the street behind the layer replaces the street in the window
                lin[:, c0 - x0:c1 - x0] = srgb2lin(d["bg"][:, c0 - a0:c1 - a0])
            pole_layers.append(d)
        return Span(x0, x1, self.H, lin, lum[:, sl].copy(), relief[:, sl].astype(np.float32),
                    spec[:, sl].astype(np.float32), e_wall, tint.astype(np.float32), occ, pnt, pole_layers)

    # -------------------------------------------------------------- the world (2D): masks, maps, crops
    def _x1d(self) -> float:
        """Columns left of this are the plain street (its own per-tile masks and light maps): every street tile that
        ends, with its 16 px mask margin, left of the world's first column (inf without a world)."""
        if self.xw is None:
            return np.inf
        if not hasattr(self, "_x1d_v"):
            ends = [self.street.tile_span(k)[1] for k in range(self.street.k_min, self.street.n)]
            ok = [e for e in ends if e + 16 <= self.xw]
            self._x1d_v = max(ok) if ok else self.X0
        return self._x1d_v

    def _content_id(self, ix: int, iy: int) -> str:
        """What a world grid tile holds: the street tiles under it and its overlay tile (their hashes)."""
        T = self.street.tile
        ids = []
        if iy * T < self.H and (iy + 1) * T > 0:
            ids += [str(self.street.tile_hash(k)) for k in range(ix * T // T, ((ix + 1) * T - 1) // T + 1)]
        ids.append(str(self.world.content_id(ix, iy) if self.world is not None else None))
        return ",".join(ids)

    def void_parts(self) -> list[dict]:
        """The voids (obstacles.json voids: sky, the open space at a gap) and the gaps: where only sky or a void is
        behind him (he is the J2 skin there, never paint); invalid pixels are voids too."""
        out = [dict(v, kind="void") for v in self.voids]
        for o in self.obstacles.values():
            if o.kind == "gap" and not o.wall:  # (a gap with the wall behind it is no void)
                out.append({"id": o.id, "box": [o.x0 + 2, min(0.0, o.y0), o.x1 - 2, max(float(self.H), o.y1)],
                            "method": "rect", "kind": "gap"})
        return out

    def _void_part(self, v: dict) -> tuple[np.ndarray, list[int]]:
        """One void's mask (float 0..1) over its box: rect, poly (points), mask (a PNG the size of the box) or sky
        (smooth pixels joined to the box's top edge: the sky over the roofs, low-textured far buildings too), less its
        `except` boxes (the structures standing in it: a stair house, a tank, a billboard back)."""
        m, b = self._void_part0(v)
        ex = v.get("except") or []
        if ex:
            m = m.copy()
            for e in ex:
                ex0, ey0, ex1, ey1 = [int(round(c)) for c in e]
                m[max(ey0 - b[1], 0):max(ey1 - b[1], 0), max(ex0 - b[0], 0):max(ex1 - b[0], 0)] = 0
        return m, b

    def _void_part0(self, v: dict) -> tuple[np.ndarray, list[int]]:
        x0, y0, x1, y1 = [int(round(c)) for c in v["box"]]
        method = v.get("method", "rect")
        if method == "rect":
            return np.ones((y1 - y0, x1 - x0), np.float32), [x0, y0, x1, y1]
        if method == "poly":
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.int32([[px - x0, py - y0] for px, py in v["poly"]])], 1)
            return m.astype(np.float32), [x0, y0, x1, y1]
        if method == "mask":
            m = cv2.imread(str(self.d / v["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise SystemExit(f"void mask not found: {self.d / v['mask']}")
            return cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, \
                [x0, y0, x1, y1]
        if method == "sky":
            h = hashlib.sha256((VERSION + "sky" + json.dumps(v, sort_keys=True)).encode())
            rgb = self.street.window(x0, x1, y0, y1)
            h.update(rgb.tobytes())
            key = h.hexdigest()[:16]
            if key in self._void_cache:
                return self._void_cache[key], [x0, y0, x1, y1]
            f = self.cache / f"void-{v.get('id', 'sky')}-{key}.png"
            m0 = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
            if m0 is not None:
                m = m0.astype(np.float32) / 255
            else:
                g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (0, 0), 2.0)
                mag = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                                                cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)), (0, 0), 3.0)
                smooth = ((mag < float(v.get("smooth", 0.02))) & (g < float(v.get("vmax", 0.5)))).astype(np.uint8)
                smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                n, lab = cv2.connectedComponents(smooth)
                top = np.unique(lab[0][smooth[0] > 0])
                m = np.isin(lab, top[top > 0]).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(np.float32)
                _imwrite(f, (m * 255).astype(np.uint8))
            self._void_cache[key] = m
            return m, [x0, y0, x1, y1]
        raise SystemExit(f"unknown void method {method!r} (rect, poly, mask, sky)")

    def void(self, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        """Where only sky or a void is behind him over street [x0, x1) x [y0, y1) (float 0..1): the voids, the gaps,
        and every pixel neither the street nor the world holds."""
        out = np.zeros((y1 - y0, x1 - x0), np.float32)
        for v in self.void_parts():
            bx0, by0, bx1, by1 = v["box"]
            if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
                continue
            pm, (px0, py0, px1, py1) = self._void_part(v)
            a0, b0, a1, b1 = max(px0, x0), max(py0, y0), min(px1, x1), min(py1, y1)
            if a1 > a0 and b1 > b0:
                np.maximum(out[b0 - y0:b1 - y0, a0 - x0:a1 - x0], pm[b0 - py0:b1 - py0, a0 - px0:a1 - px0],
                           out=out[b0 - y0:b1 - y0, a0 - x0:a1 - x0])
        if not self.street.valid_box(x0, y0, x1, y1):
            _, valid = self.street.window(x0, x1, y0, y1, valid=True, fill=False)
            out = np.maximum(out, (~valid).astype(np.float32))
        return out

    def tile_masks2(self, ix: int, iy: int) -> tuple[np.ndarray, np.ndarray]:
        """(occ, paintable) float32 for world grid tile (ix, iy) (street x [ix * T, (ix + 1) * T), y the same), from
        the street and the world as they are there: paintable is valid, above the pavement line, outside the no-paint
        zones and the voids. Cached by content hash like the street tiles' masks."""
        k = (ix, iy)
        if k in self._tile_masks2:
            return self._tile_masks2[k]
        T, m = self.street.tile, 16
        X0, X1, Y0, Y1 = ix * T, (ix + 1) * T, iy * T, (iy + 1) * T
        a, b, ya, yb = X0 - m, X1 + m, Y0 - m, Y1 + m
        inside = lambda bx: bx[2] > a and bx[0] < b and bx[3] > ya and bx[1] < yb
        parts = [p for p in self.occluder_parts() if inside(p["box"])]
        zones = [z for z in self.no_paint if inside(z["box"])]
        voids = [v for v in self.void_parts() if inside(v["box"])]
        gdy = np.round(self.ground_dy(np.arange(a, b, dtype=np.float64))).astype(int)
        h = hashlib.sha256((VERSION + "2d").encode())
        for jy in (iy - 1, iy, iy + 1):
            for jx in (ix - 1, ix, ix + 1):
                h.update(self._content_id(jx, jy).encode())
        h.update(json.dumps(parts + zones + voids, sort_keys=True).encode())
        g = {kk: v for kk, v in self.ground.items() if kk not in ("dy", "levels") and not kk.startswith("note")}
        h.update(json.dumps([g, int(gdy.min()), int(gdy.max()), int(gdy.sum())], sort_keys=True).encode())
        for p in parts + voids:
            if p.get("method") == "mask":
                mf = self.d / p["mask"]
                h.update(mf.read_bytes() if mf.exists() else b"missing")
        key = h.hexdigest()[:16]
        name = f"{ix}_{iy}"
        fo, fp = self.cache / f"occ2-{name}-{key}.png", self.cache / f"paint2-{name}-{key}.png"
        occ = pnt = None
        if fo.exists() and fp.exists():  # (another process may replace them as we read: then rebuilt here)
            o_, p_ = cv2.imread(str(fo), cv2.IMREAD_UNCHANGED), cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if o_ is not None and p_ is not None:
                occ, pnt = o_.astype(np.float32) / 65535, p_.astype(np.float32) / 65535
        if occ is None:
            for old in list(self.cache.glob(f"occ2-{name}-*.png")) + list(self.cache.glob(f"paint2-{name}-*.png")):
                old.unlink(missing_ok=True)
            occ = np.zeros((yb - ya, b - a), np.float32)
            for p in parts:
                pm, (x0, y0, x1, y1) = _part_mask(self.street, p, self.d)
                c0, c1, r0, r1 = max(x0, a), min(x1, b), max(y0, ya), min(y1, yb)
                if c1 > c0 and r1 > r0:
                    sub = occ[r0 - ya:r1 - ya, c0 - a:c1 - a]
                    np.maximum(sub, pm[r0 - y0:r1 - y0, c0 - x0:c1 - x0], out=sub)
            occ = cv2.GaussianBlur(occ, (0, 0), 0.8)[m:m + T, m:m + T]
            _, valid = self.street.window(a, b, ya, yb, valid=True, fill=False)
            rows = np.arange(ya, yb)[:, None]
            pnt = (valid & (rows < (self.wall_base - 6 + gdy)[None, :])).astype(np.float32)
            for z in zones:
                x0, y0, x1, y1 = [int(round(v)) for v in z["box"]]
                pnt[max(y0 - ya, 0):max(y1 - ya, 0), max(x0 - a, 0):max(x1 - a, 0)] = 0
            if voids:
                pnt *= 1 - self.void(a, b, ya, yb)
            pnt = cv2.GaussianBlur(pnt, (0, 0), 1.5)[m:m + T, m:m + T]
            _imwrite(fo, (np.clip(occ, 0, 1) * 65535 + 0.5).astype(np.uint16))
            _imwrite(fp, (np.clip(pnt, 0, 1) * 65535 + 0.5).astype(np.uint16))
        if len(self._tile_masks2) > 6:
            self._tile_masks2.pop(next(iter(self._tile_masks2)))
        self._tile_masks2[k] = (occ, pnt)
        return occ, pnt

    def lf_world(self) -> dict | None:
        """Low-frequency maps of the world at 1/8 (2D, from a little left of the world's first column to its right
        end, every row of the world): loc and col as on the street (normalized over the valid pixels), level a
        geometric mean of the wall's luminance (sigma 900 px along x, 600 along y), norm the tint's wide average.
        "ox", "oy": the street px of the grid's origin. None without a world."""
        if self.world is None:
            return None
        if self._lf_world is not None:
            return self._lf_world
        xa = int(np.floor((self.xw - 3 * 2048) / 8)) * 8
        self._lf_world = self._lf2d((xa, self.Y0, self.bounds[2], self.Y1), "lfw", "world")
        return self._lf_world

    def lf_band(self, b: tuple) -> dict:
        """The same maps over a band of world cells above the street's left part (x a little wider on each side,
        rows from the world's top to the street's foot), its own cache (v8, P2b)."""
        if b not in self._lf_bands:
            xa = int(np.floor((b[0] - 2048) / 8)) * 8
            xb = int(np.ceil((b[2] + 2048) / 8)) * 8
            self._lf_bands[b] = self._lf2d((xa, self.Y0, xb, self.Y1), f"lfb{xa}", f"band {list(b)}")
        return self._lf_bands[b]

    def _lf2d(self, box: tuple, prefix: str, tag: str) -> dict:
        T = self.street.tile
        xa = box[0]
        h = hashlib.sha256((VERSION + tag).encode())
        h.update(json.dumps(self.world.info["tiles"], sort_keys=True).encode())
        h.update(json.dumps([self.street.tile_hash(k) for k in range(max(xa // T - 1, self.street.k_min),
                                                                     self.street.n)]).encode())
        h.update(json.dumps([self.wall_base, list(box), self.ground.get("dy")]).encode())
        f = self.cache / f"{prefix}-{h.hexdigest()[:16]}.npz"
        if f.exists():
            try:
                return dict(np.load(f))
            except (FileNotFoundError, OSError, ValueError):  # replaced by another process as we read: rebuild
                pass
        for old in self.cache.glob(f"{prefix}-*.npz"):
            old.unlink(missing_ok=True)
        img, cov, ox, oy = self.street.overview2d(8, box, linear=True)
        vs = (cov > 0.5).astype(np.float32)
        lum = luminance(img)
        nb = lambda a, w, sx, sy: _nblur(a, w, sx, sy)
        loc = nb(lum, vs, 60 / 8, 60 / 8)
        col = nb(img, vs, 60 / 8, 60 / 8)
        xx = ox + 8 * np.arange(img.shape[1]) + 4
        yy = oy + 8 * np.arange(img.shape[0]) + 4
        wall = vs * (yy[:, None] < (self.wall_base + self.ground_dy(xx.astype(np.float64)))[None, :])
        level = np.exp(nb(np.log(lum + 1e-3), wall, 900 / 8, 600 / 8)).astype(np.float32)
        tint_s = col / (loc[..., None] + 1e-4)
        norm = nb(tint_s, vs, 3000 / 8, 1500 / 8)
        out = {"loc": loc.astype(np.float32), "level": level, "col": col.astype(np.float32),
               "norm": norm.astype(np.float32), "ox": np.int32(ox), "oy": np.int32(oy)}
        _savez(f, **out)
        return out

    @staticmethod
    def _upw(a: np.ndarray, ox: int, oy: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """Upsample a 1/8 world map (origin ox, oy) to street [x0, x0 + w) x [y0, y0 + h)."""
        M = np.float32([[1 / 8, 0, (x0 - ox + 0.5) / 8 - 0.5], [0, 1 / 8, (y0 - oy + 0.5) / 8 - 0.5]])
        return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def _lf1d(self, x0: int, x1: int) -> tuple:
        """The plain street's low-frequency maps over columns [x0, x1), rows 0 to H (as span() computes them)."""
        lf = self.lf()
        w = x1 - x0
        sy = float(lf["sy"])
        loc = self._up(lf["loc"], x0, w, sy, self.H)
        level = self._up(lf["level"], x0, w, 1.0, 1)
        col = self._up(lf["col"], x0, w, sy, self.H)
        norm = self._up(lf["norm"], x0, w, 1.0, 1)
        ll = self.lf_left() if x0 < 0 else None
        if ll is not None:
            o = int(ll["x0"])
            r = np.clip((np.arange(x0, x1, dtype=np.float32) + 2048) / 2048, 0, 1)
            r = np.where(np.arange(x0, x1) < 0, r, 1.0).astype(np.float32)
            mix = lambda L, R: (L * (1 - r) + R * r).astype(np.float32) if L.ndim == 2 else \
                (L * (1 - r)[None, :, None] + R * r[None, :, None]).astype(np.float32)
            loc = mix(self._up(ll["loc"], x0, w, sy, self.H, o), loc)
            level = mix(self._up(ll["level"], x0, w, 1.0, 1, o), level)
            col = mix(self._up(ll["col"], x0, w, sy, self.H, o), col)
            norm = mix(self._up(ll["norm"], x0, w, 1.0, 1, o), norm)
        return loc, level, col, norm

    def _span2d(self, x0: int, x1: int, y0: int | None, y1: int | None) -> "Span":
        """span() where the window reaches the world (or rows are given): the same maps over [x0, x1) x [y0, y1).
        Without rows (0 to H) the high-frequency maps are made as on the street (no rows of context above or below)."""
        m = 48
        default = y0 is None and y1 is None
        r0, r1 = (0 if y0 is None else int(y0)), (self.H if y1 is None else int(y1))
        mv = 0 if default else m
        rgb8 = self.street.window(x0 - m, x1 + m, r0 - mv, r1 + mv)
        lin_full = srgb2lin(rgb8)
        lum = luminance(lin_full)
        lum_s = cv2.GaussianBlur(lum, (0, 0), 1.2)
        relief = np.clip(lum_s / (cv2.GaussianBlur(lum, (0, 0), 10) + 1e-4), 0.55, 1.45) ** 0.8
        spec = np.clip(lin_full - cv2.GaussianBlur(lin_full, (0, 0), 4) * 1.08, 0, None)
        sl = (slice(mv, mv + r1 - r0), slice(m, m + x1 - x0))
        w, h = x1 - x0, r1 - r0
        # the low-frequency maps: the street's (rows 0 to H) left of the world, the world's in it, blended over the
        # 2048 px left of its first column; rows above and below the street are the world's
        xs = np.arange(x0, x1, dtype=np.float32)
        rx = np.zeros(w, np.float32) if self.xw is None else np.clip((xs - (self.xw - 2048)) / 2048, 0, 1)
        r = np.broadcast_to(rx[None, :], (h, w)).copy()
        rows = np.arange(r0, r1)
        if self.world is not None:
            r[(rows < 0) | (rows >= self.H)] = 1.0
        loc = np.zeros((h, w), np.float32)
        level = np.zeros((h, w), np.float32)
        col = np.zeros((h, w, 3), np.float32)
        norm = np.zeros((h, w, 3), np.float32)
        if (r < 1).any():
            l1, v1, c1, n1 = self._lf1d(x0, x1)
            ri = np.clip(rows, 0, self.H - 1)
            k = 1 - r
            loc += l1[ri] * k
            level += v1 * k
            col += c1[ri] * k[..., None]
            norm += n1 * k[..., None]
        if (r > 0).any():
            lw = self.lf_world()
            ox, oy = int(lw["ox"]), int(lw["oy"])
            wl = self._upw(lw["loc"], ox, oy, x0, r0, w, h)
            wv = self._upw(lw["level"], ox, oy, x0, r0, w, h)
            wc = self._upw(lw["col"], ox, oy, x0, r0, w, h)
            wn = self._upw(lw["norm"], ox, oy, x0, r0, w, h)
            for b in self.bands:  # left of the world's maps: a band of cells above the street's left part, its own
                cb = (xs < ox) & (xs >= b[0] - 2048) & (xs < b[2] + 2048)
                if cb.any():
                    lb = self.lf_band(b)
                    bx, by = int(lb["ox"]), int(lb["oy"])
                    wl[:, cb] = self._upw(lb["loc"], bx, by, x0, r0, w, h)[:, cb]
                    wv[:, cb] = self._upw(lb["level"], bx, by, x0, r0, w, h)[:, cb]
                    wc[:, cb] = self._upw(lb["col"], bx, by, x0, r0, w, h)[:, cb]
                    wn[:, cb] = self._upw(lb["norm"], bx, by, x0, r0, w, h)[:, cb]
            loc += wl * r
            level += wv * r
            col += wc * r[..., None]
            norm += wn * r[..., None]
        rel = np.clip(loc / (level + 1e-4), 0.55, 2.2) ** 0.6
        e_wall = (0.30 * rel).astype(np.float32)
        tint = (1 + 0.3 * (col / (loc[..., None] + 1e-4) / norm - 1)) * np.float32([1.03, 1.0, 0.95]) * self.tint_gain
        occ, pnt = self.masks(x0, x1) if default else self.masks(x0, x1, r0, r1)
        lin = lin_full[sl].copy()
        pole_layers = []
        for L in self.layers:
            d = self.layer(L)
            a0, lw_ = d["a0"], d["bg"].shape[1]
            ly0 = d["y0"]
            c0, c1 = max(a0, x0), min(a0 + lw_, x1)
            b0, b1 = max(ly0, r0), min(ly0 + d["bg"].shape[0], r1)
            if c1 > c0 and b1 > b0:
                lin[b0 - r0:b1 - r0, c0 - x0:c1 - x0] = srgb2lin(d["bg"][b0 - ly0:b1 - ly0, c0 - a0:c1 - a0])
            pole_layers.append(d)
        return Span(x0, x1, self.H, lin, lum[sl].copy(), relief[sl].astype(np.float32),
                    spec[sl].astype(np.float32), e_wall, tint.astype(np.float32), occ, pnt, pole_layers, r0, r1)

    def span_rows(self, y0: int, y1: int) -> tuple[int | None, int | None]:
        """The rows a span for a crop over rows [y0, y1) should hold: None, None (0 to H, the street's own) while
        the crop stays within the street's rows, else the crop's rows with room to move (multiples of 256)."""
        if self.world is None or (y0 >= 0 and y1 <= self.H):
            return None, None
        a = max(int(np.floor((y0 - 384) / 256)) * 256, int(np.floor(self.Y0 / 8)) * 8)
        b = min(int(np.ceil((y1 + 384) / 256)) * 256, self.Y1)
        return a, b

    def clamp_crop(self, cx: float, cy: float, cw: float, ch: float, margin: float = 30) -> tuple[float, float]:
        """A crop's centre moved as little as possible so the crop (with `margin` to spare) shows only valid
        pixels. On the plain street the street's own rule (the crop inside the street, each axis on its own); in
        the world the nearest centre whose crop is all street or world."""
        ox = float(np.clip(cx, self.X0 + cw / 2 + margin, self.W - cw / 2 - margin))
        oy = float(np.clip(cy, ch / 2 + margin, self.H - ch / 2 - margin)) if ch < self.H - 2 * margin else self.H / 2
        if self.world is None:
            return ox, oy
        hw, hh = cw / 2 + margin, ch / 2 + margin
        ok = lambda x, y: self.street.valid_box(x - hw, y - hh, x + hw, y + hh)
        if ok(cx, cy):
            return float(cx), float(cy)
        reach = 4000.0
        best = (ox, oy) if ok(ox, oy) and np.hypot(ox - cx, oy - cy) <= reach else None
        cand = self._nearest_valid(cx, cy, hw, hh, int(reach))
        if cand is not None and (best is None or np.hypot(cand[0] - cx, cand[1] - cy) <
                                 np.hypot(best[0] - cx, best[1] - cy) - 1e-6):
            best = cand
        if best is None:  # no crop that size is all valid near here (a frame wider than the world there): it stays
            # where it was asked, inside the world's bounds; its invalid pixels show the nearest valid ones
            x0, y0, x1, y1 = self.bounds
            bx = float(np.clip(cx, x0 + hw, x1 - hw)) if x1 - x0 > 2 * hw else (x0 + x1) / 2
            by = float(np.clip(cy, y0 + hh, y1 - hh)) if y1 - y0 > 2 * hh else (y0 + y1) / 2
            best = (bx, by)
        return best

    def _nearest_valid(self, cx: float, cy: float, hw: float, hh: float, reach: int = 4000):
        """The nearest centre (within `reach` px) whose box of half sizes hw, hh is all valid (1/8 blocks of the
        world's valid8, then moved back toward (cx, cy) as far as it stays valid)."""
        v8, gx0, gy0 = self.world.valid8()
        bad = (~v8).astype(np.int32)
        I = np.pad(bad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        hx, hy = int(np.ceil(hw / 8)) + 1, int(np.ceil(hh / 8)) + 1
        R = reach // 8
        ci, cj = int(round((cy - gy0) / 8)), int(round((cx - gx0) / 8))
        i0, i1 = max(ci - R, hy), min(ci + R, v8.shape[0] - hy)
        j0, j1 = max(cj - R, hx), min(cj + R, v8.shape[1] - hx)
        if i1 <= i0 or j1 <= j0:
            return None
        ii, jj = np.arange(i0, i1)[:, None], np.arange(j0, j1)[None, :]
        n = I[ii + hy, jj + hx] - I[ii - hy, jj + hx] - I[ii + hy, jj - hx] + I[ii - hy, jj - hx]
        cand = n == 0
        if not cand.any():
            return None
        d2 = ((gy0 + 8 * ii + 4) - cy) ** 2 + ((gx0 + 8 * jj + 4) - cx) ** 2
        d2 = np.where(cand, d2, np.inf)
        k = np.unravel_index(int(np.argmin(d2)), d2.shape)
        px, py = float(gx0 + 8 * (j0 + k[1]) + 4), float(gy0 + 8 * (i0 + k[0]) + 4)
        ok = lambda x, y: self.street.valid_box(x - hw, y - hh, x + hw, y + hh)
        if not ok(px, py):
            return None
        # back toward (cx, cy) as far as it stays valid: along the line, then each axis on its own
        def back(x, y, tx, ty):
            lo, hi = 0.0, 1.0
            if ok(tx, ty):
                return tx, ty
            for _ in range(14):
                mid = (lo + hi) / 2
                if ok(x + (tx - x) * mid, y + (ty - y) * mid):
                    lo = mid
                else:
                    hi = mid
            return x + (tx - x) * lo, y + (ty - y) * lo

        px, py = back(px, py, cx, cy)
        for _ in range(2):
            px, py = back(px, py, cx, py)
            px, py = back(px, py, px, cy)
        return px, py

    def inside(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True when the box lies on valid pixels (the street, or the world)."""
        return self.street.valid_box(x0, y0, x1, y1)

    # -------------------------------------------------------------- levels
    def measure_line(self, x0: int, x1: int, y_at, search: int = 60, step: int = 64) -> list[list[float]]:
        """The strongest horizontal edge (a parapet's top, a ledge, a walkway's boards) near a guessed line, every
        `step` px from x0 to x1: [[x, y], ...]. y_at(x) gives the guess; the edge is searched within +-search px."""
        out = []
        for xa in range(int(x0), int(x1), step):
            xb = min(xa + step, int(x1))
            yg = float(y_at((xa + xb) / 2))
            ya, yb = int(yg - search - 4), int(yg + search + 4)
            rgb, v = self.street.window(xa, xb, ya, yb, valid=True)
            if v.mean() < 0.9:
                continue
            g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
            gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).mean(1)
            gy[:4], gy[-4:] = 0, 0
            k = int(np.argmax(gy))
            if gy[k] < 4.0:
                continue
            out.append([float((xa + xb) / 2), float(ya + k)])
        return out

    def levels_sheet(self, out: Path, scale: int = 2, half: int = 220, row_px: int = 6000) -> tuple[Path, dict]:
        """For every level: the world along its ground line (+-half px, at 1/scale), its feet polyline (green) and
        the measured edge near it (red dots, measure_line), in rows. Returns the sheet and the measured lines
        ({level id: [[x, y], ...]}) to fit the levels' feet to."""
        strips, measured = [], {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        for lv in self.levels:
            y_at = lambda x, lv=lv: float(np.interp(x, lv["px"], lv["py"]))
            pts = self.measure_line(lv["x0"], lv["x1"], y_at)
            measured[lv["id"]] = [[round(x), round(y)] for x, y in pts]
            ymin = int(min(lv["py"])) - half
            ymax = int(max(lv["py"])) + half
            x0, x1 = int(lv["x0"]), int(lv["x1"])
            for c0 in range(x0, x1, row_px):
                c1 = min(c0 + row_px, x1)
                img = self.street.window(c0, c1, ymin, ymax)[..., ::-1]
                img = cv2.resize(img, ((c1 - c0) // scale, (ymax - ymin) // scale), interpolation=cv2.INTER_AREA)
                P = lambda x, y: (int((x - c0) / scale), int((y - ymin) / scale))
                xs = np.arange(c0, c1, 16)
                line = np.int32([P(x, y_at(x)) for x in xs])
                cv2.polylines(img, [line], False, (0, 255, 0), 1)
                for x, y in pts:
                    if c0 <= x < c1:
                        cv2.circle(img, P(x, y), 3, (0, 0, 255), -1)
                cv2.rectangle(img, (0, 0), (420, 22), (0, 0, 0), -1)
                cv2.putText(img, f"{lv['id']}: x {c0} to {c1}, y {ymin} to {ymax}", (6, 16), font, 0.5,
                            (255, 255, 255), 1)
                strips.append(img)
        if not strips:
            return out, measured
        wmax = max(s_.shape[1] for s_ in strips)
        sheet = np.vstack([np.pad(s_, ((0, 6), (0, wmax - s_.shape[1]), (0, 0))) for s_ in strips])
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out, measured

    # -------------------------------------------------------------- preview
    def preview(self, out: Path, scale: int = 4) -> Path:
        """Occluders red, parallax layers blue, no-paint zones grey, obstacle boxes and contacts yellow (1/scale).
        With a world: the street and the world in two dimensions at 1/8 (voids purple, invalid areas dark, the
        levels' ground lines cyan)."""
        if self.world is not None:
            return self._preview2d(out)
        rows = []
        ox = self.street.overview_x0(scale)  # street x of the image's left edge
        for k in range(self.street.k_min, self.street.n):
            X0, X1 = self.street.tile_span(k)
            vis = self.street.tile_rgb(k).astype(np.float32) / 255
            occ, pnt = self.tile_masks(k)
            vis[..., 0] = np.maximum(vis[..., 0], occ)
            for L in self.layers:
                d = self.layer(L)
                c0, c1 = max(d["a0"], X0), min(d["a0"] + d["alpha"].shape[1], X1)
                if c1 > c0:
                    vis[:, c0 - X0:c1 - X0, 2] = np.maximum(vis[:, c0 - X0:c1 - X0, 2],
                                                            d["alpha"][:, c0 - d["a0"]:c1 - d["a0"]])
            nop = 1 - pnt
            vis = vis * (1 - 0.5 * nop[..., None]) + 0.25 * nop[..., None]
            rows.append(cv2.resize((np.clip(vis, 0, 1) * 255).astype(np.uint8), ((X1 - X0) // scale, self.H // scale),
                                   interpolation=cv2.INTER_AREA))
        img = np.ascontiguousarray(np.concatenate(rows, axis=1)[..., ::-1])
        sx = lambda v: int((v - ox) / scale)
        for o in self.obstacles.values():
            x0, x1 = sx(o.box[0]), sx(o.box[2])
            y0, y1 = int(o.box[1] / scale), int(o.box[3] / scale)
            cv2.rectangle(img, (x0, max(y0, 1)), (x1, min(y1, img.shape[0] - 2)), (0, 200, 255), 2)
            cv2.putText(img, o.id, (x0 + 3, max(y0, 1) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            for c in o.contacts.values():
                if c.horizontal:
                    cv2.line(img, (sx(c.x0), int(c.y / scale)), (sx(c.x1), int(c.y / scale)), (0, 255, 0), 2)
                elif c.vertical:
                    cv2.line(img, (sx(c.x), int(c.y0 / scale)), (sx(c.x), int(c.y1 / scale)), (0, 255, 0), 2)
                else:
                    cv2.circle(img, (sx(c.x), int(c.y / scale)), 4, (0, 255, 0), -1)
        if ox < 0:  # x 0 and a scale every 1000 px
            for xm in range(int(np.ceil(ox / 1000)) * 1000, self.W, 1000):
                cv2.line(img, (sx(xm), 0), (sx(xm), 10 if xm else 40), (255, 255, 255), 1)
                cv2.putText(img, str(xm), (sx(xm) + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        gy = int(self.feet / scale)
        cv2.line(img, (0, gy), (img.shape[1], gy), (255, 255, 0), 1)
        # two rows when the street is long
        if img.shape[1] > 6000:
            n = int(np.ceil(img.shape[1] / 4000))
            w = int(np.ceil(img.shape[1] / n))
            img = np.vstack([np.pad(img[:, i * w:(i + 1) * w], ((0, 6), (0, w - img[:, i * w:(i + 1) * w].shape[1]),
                                                                 (0, 0))) for i in range(n)])
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out


    def _preview2d(self, out: Path, scale: int = 8, row_w: int = 5000) -> Path:
        """preview() with a world: everything at 1/scale in two dimensions, wrapped into rows row_w px wide."""
        rgb, cov, ox, oy = self.street.overview2d(scale)
        h8, w8 = cov.shape
        vis = rgb.astype(np.float32) / 255
        occ8 = np.zeros((h8, w8), np.float32)
        pnt8 = np.zeros((h8, w8), np.float32)
        void8 = np.zeros((h8, w8), np.float32)
        step = (2048 // scale) * scale
        y0, y1 = oy, oy + h8 * scale
        for c0 in range(ox, ox + w8 * scale, step):
            c1 = min(c0 + step, ox + w8 * scale)
            o, p = self.masks(c0, c1, y0, y1)
            v = self.void(c0, c1, y0, y1)
            j, nb = (c0 - ox) // scale, (c1 - c0) // scale
            red = lambda a: a.reshape(h8, scale, nb, scale).mean(axis=(1, 3))
            occ8[:, j:j + nb], pnt8[:, j:j + nb], void8[:, j:j + nb] = red(o), red(p), red(v)
        vis[..., 0] = np.maximum(vis[..., 0], occ8)
        nop = (1 - pnt8) * (cov > 0.5)
        vis = vis * (1 - 0.5 * nop[..., None]) + 0.25 * nop[..., None]
        vis = vis * (1 - 0.5 * void8[..., None]) + np.float32([0.45, 0.1, 0.5]) * 0.5 * void8[..., None]
        vis[cov < 0.5] = (0.08, 0.08, 0.09)
        img = np.ascontiguousarray((np.clip(vis, 0, 1) * 255).astype(np.uint8)[..., ::-1])
        X = lambda v: int((v - ox) / scale)
        Y = lambda v: int((v - oy) / scale)
        for L in self.layers:
            d = self.layer(L)
            a = cv2.resize(d["alpha"], (max(d["alpha"].shape[1] // scale, 1), max(d["alpha"].shape[0] // scale, 1)),
                           interpolation=cv2.INTER_AREA)
            ax, ay = X(d["a0"]), Y(d["y0"])
            sub = img[ay:ay + a.shape[0], ax:ax + a.shape[1]]
            sub[..., 0] = np.maximum(sub[..., 0], (a[:sub.shape[0], :sub.shape[1]] * 255).astype(np.uint8))
        for o in self.obstacles.values():
            cv2.rectangle(img, (X(o.box[0]), Y(o.box[1])), (X(o.box[2]), Y(o.box[3])), (0, 200, 255), 1)
            cv2.putText(img, o.id, (X(o.box[0]) + 3, Y(o.box[1]) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            for c in o.contacts.values():
                if c.horizontal:
                    cv2.line(img, (X(c.x0), Y(c.y)), (X(c.x1), Y(c.y)), (0, 255, 0), 1)
                elif c.vertical:
                    cv2.line(img, (X(c.x), Y(c.y0)), (X(c.x), Y(c.y1)), (0, 255, 0), 1)
                else:
                    cv2.circle(img, (X(c.x), Y(c.y)), 3, (0, 255, 0), -1)
        xs = np.arange(ox, ox + w8 * scale, 32, dtype=np.float64)
        street = np.int32([[X(x), Y(self.feet + self.ground_dy(x))] for x in xs if self.X0 <= x < self.W])
        if len(street):
            cv2.polylines(img, [street], False, (255, 255, 0), 1)
        for lv in self.levels:
            pts = np.int32([[X(x), Y(np.interp(x, lv["px"], lv["py"]))] for x in np.arange(lv["x0"], lv["x1"], 32)])
            cv2.polylines(img, [pts], False, (255, 255, 0), 1)
            cv2.putText(img, lv["id"], (pts[0][0] + 3, pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        for xm in range(int(np.ceil(ox / 1000)) * 1000, ox + w8 * scale, 1000):
            cv2.line(img, (X(xm), 0), (X(xm), 8 if xm % 5000 else 20), (255, 255, 255), 1)
            if xm % 5000 == 0:
                cv2.putText(img, str(xm), (X(xm) + 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for ym in range(int(np.ceil(oy / 500)) * 500, oy + h8 * scale, 500):
            cv2.line(img, (0, Y(ym)), (8, Y(ym)), (255, 255, 255), 1)
            cv2.putText(img, f"y {ym}", (10, Y(ym) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        rw = row_w // scale
        if img.shape[1] > rw * 1.2:  # rows of row_w px of street, each cut to the rows it has (and the scale)
            bands = []
            for i in range(int(np.ceil(img.shape[1] / rw))):
                rows = np.nonzero((cov[:, i * rw:(i + 1) * rw] > 0.5).any(1))[0]
                a, b = (max(int(rows.min()) - 4, 0), int(rows.max()) + 5) if len(rows) else (0, 40)
                band = np.vstack([img[:40, i * rw:(i + 1) * rw], img[max(a, 40):b, i * rw:(i + 1) * rw]])
                bands.append(np.pad(band, ((0, 6), (0, rw - band.shape[1]), (0, 0))))
            img = np.vstack(bands)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out


@dataclass
class Span:
    """Maps for street columns [x0, x1): lin (street in linear light, layers removed), lum, relief, spec, e_wall,
    tint, occ (occluders), paint (paintable), and the parallax layers that may enter it."""

    x0: int
    x1: int
    H: int
    lin: np.ndarray
    lum: np.ndarray
    relief: np.ndarray
    spec: np.ndarray
    e_wall: np.ndarray
    tint: np.ndarray
    occ: np.ndarray
    paint: np.ndarray
    layers: list[dict]
    y0: int = 0  # its rows: [y0, y1) (0 to H on the plain street)
    y1: int | None = None

    def __post_init__(self):
        if self.y1 is None:
            self.y1 = self.y0 + self.lin.shape[0]

    def covers(self, x0: int, x1: int, y0: int | None = None, y1: int | None = None) -> bool:
        return self.x0 <= x0 and x1 <= self.x1 and (y0 is None or self.y0 <= y0) and (y1 is None or y1 <= self.y1)
