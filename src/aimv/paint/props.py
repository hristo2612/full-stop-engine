"""Props: painted set pieces that are not the runner. The box from board B1 (06-v2): a rectangle of thick white
primer brush lines with drips on the wall (an old one dry, faded and rain-washed, with someone else curled in it:
a drawing painted grey, still), keyframed in time (it is repainted at a new size, thickness or bulge on
its keys; the old outline washes out as a ghost), hairline cracks that grow from where he hits it, the shatter (the
outline and its cracks burst into flying white primer and black paint shards that hang in the air through a song
stop, then fall and settle at the foot of the wall as spatter; the broken remains of the outline stay and wash), and
the rules stencilled around it in black spray paint (hard edges with stencil bridges cut through the letters, a soft
overspray halo, a few runs) that crack and peel after the shatter.

A prop template (moves.py: box) registers a BoxProp on the choreography; other templates add cracks (inside, pound)
and the shatter (burst). The renderer (render.py History) turns props into paint marks on the wall and shards in the
air. Geometry is street px, times are song 16ths, and everything comes from seeds, so a prop paints the same in any
render range. Text is ours (a bold condensed sans cut into a stencil), never generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paintfx import add_drips, noise

STENCIL_FONTS = ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Impact.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf")


# ---------------------------------------------------------------------------------------------- data

@dataclass
class BoxKey:
    """One painting of the outline: from song 16th q on, until the next key (or the shatter)."""

    q: float
    cx: float  # centre x
    bottom: float  # the outer bottom edge
    w: float  # outer size
    h: float
    thick: float  # line width
    bulge: float = 0.0  # the sides pushed out locally (where his hands press), as a fraction of the width
    seed: int = 0
    bulge_at: float = 0.5  # the height of the push, as a fraction of the box height from its top
    bulge_h: float = 0.12  # its vertical spread (a Gaussian's sigma, as a fraction of the height)
    old: float = 0.0  # 0: painted on its key (wet, then drying); up to 1: painted long ago (dry, faded, rain-washed)
    paint_in: float = 0.0  # seconds it takes to be painted in (a brush sweep), from its key
    drop: bool = False  # one photo of the box falling onto its next key (v8: it slams down): gone at once after it
    bright: bool = False  # bright fresh primer from its first photo (the slammed box: it reads at once)
    sides: tuple = ()  # v8b: ((side, seconds after the key), ...): each side painted at its own time (only those)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.cx - self.w / 2, self.bottom - self.h, self.cx + self.w / 2, self.bottom

    @property
    def inner(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.box
        t = self.thick
        return x0 + t, y0 + t, x1 - t, y1 - t

    def bow(self, y):
        """How far the sides are pushed out at height y (y may be an array): a local bump round bulge_at, so the box
        stays a rectangle with the paint bent out only where he presses."""
        s = (np.asarray(y, dtype=np.float64) - (self.bottom - self.h)) / max(self.h, 1)
        v = self.bulge * self.w * np.exp(-0.5 * ((s - self.bulge_at) / max(self.bulge_h, 1e-3)) ** 2)
        return float(v) if np.ndim(v) == 0 else v

    def width_at(self, y: float) -> float:
        """Inner width at height y (the bulge widens the middle)."""
        x0, y0, x1, y1 = self.inner
        return (x1 - x0) + 2 * self.bow(y)

    def side_x(self, y: float, side: str) -> float:
        """The inner edge of the left or right side at height y (bulge included)."""
        x0, _, x1, _ = self.inner
        return x1 + self.bow(y) if side == "right" else x0 - self.bow(y)


@dataclass
class Crack:
    q: float
    x: float
    y: float
    size: float  # how far this growth reaches (px)
    seed: int


@dataclass
class Stencil:
    q: float
    lines: list[str]
    x: float  # the block's left, centre or right edge (align)
    y: float  # its top
    h: float  # cap height in px
    align: str = "left"
    rot: float = 0.0
    seed: int = 0
    # a body running through it (the stencil template's smear): (song 16th, length px, 16ths it takes, rows (y0, y1)
    # in street px or None for all of it, +1 dragged right / -1 left): its letters dragged along, staying smeared
    smear: tuple | None = None
    albedo: float | None = None  # its paint (None: black spray; 0.7: white primer letters, on a dark fascia)
    fade: tuple | None = None  # (song 16th it starts to fade, 16ths it takes): gone after (a lyric line's end)
    runs: bool = True  # (v8b: false, no paint runs: a sign's STOP, where a run down the S read as a dollar sign)


@dataclass
class Shatter:
    q: float
    x: float  # the burst point (where he breaks through)
    y: float
    dirx: float = 1.0  # the main direction of the burst
    seed: int = 0
    fly: float = 0.65  # how much of the outline flies (the far side stays as the broken remains)
    power: float = 1.0  # the explosion: more and faster black flecks, splatter thrown across the wall
    splats: int = 0  # paint splatter that lands on the wall round the burst point, spreading over several photos
    stencils: float = 0.0  # share of the stencils' letters that break off and fly (the nearest first)


@dataclass
class Burst:
    """A paint explosion without a box (the leap's take-off, a light shattered): black flecks thrown from a point
    (or pale glass: albedo) and splatter landing on the wall round it over a few photos."""

    q: float  # song 16th
    x: float
    y: float
    dirx: float = 1.0  # the main direction of the throw
    seed: int = 0
    power: float = 1.0
    flecks: int = 40
    splats: int = 40
    spread: float = 60.0  # px: how far from the point the flecks start
    albedo: float = 0.02  # black paint; about 0.6: glass


@dataclass
class Figure:
    """Someone else in a box: a drawing from the pose library, painted grey and old, still the whole film."""

    pose: str
    x: float  # its bottom-centre anchor (bc) in street px
    y: float
    scale: float  # on top of the pose's own
    mirror: bool = False
    old: float = 0.8
    seed: int = 0
    q: float | None = None  # when it is painted (song 16th; None: with the box's first key)
    paint_in: float = 0.0  # seconds it takes to be painted in


@dataclass
class Bump:
    """A hit on the box's side from inside: the side gives (a bump that springs back) and primer flakes fall."""

    q: float
    side: str  # right or left
    amp: float  # px the side gives
    y: float  # where, as a share of the box height from its top
    flakes: int
    seed: int = 0


@dataclass
class BoxProp:
    id: str
    keys: list[BoxKey]
    stencils: list[Stencil] = field(default_factory=list)
    cracks: list[Crack] = field(default_factory=list)
    shatter: Shatter | None = None
    seed: int = 0
    figure: Figure | None = None
    bumps: list[Bump] = field(default_factory=list)
    peel_from: float | None = None  # song 16th: the stencils start to crack and peel (slowly) before the shatter
    _ink: dict = field(default_factory=dict, repr=False)

    def stencil_box(self, i: int) -> tuple[float, float, float, float]:
        """Street box (x0, y0, x1, y1) of stencil i's letters, with a little of the overspray round them."""
        if i not in self._ink:
            st = self.stencils[i]
            _, hard, x0, y0 = stencil_mask(st)
            ys, xs = np.nonzero(hard > 0.3)
            pad = 0.12 * st.h
            self._ink[i] = (x0 + xs.min() - pad, y0 + ys.min() - pad, x0 + xs.max() + 1 + pad, y0 + ys.max() + 1 + pad)
        return self._ink[i]

    def outer(self, key: "BoxKey") -> tuple[float, float, float, float]:
        """The street box a key's outline covers, its local push included."""
        x0, y0, x1, y1 = key.box
        b = key.bulge * key.w
        return x0 - b, y0, x1 + b, y1

    def regions(self, q: float, every: bool = False) -> list[tuple[float, float, float, float]]:
        """What a box camera must hold at song 16th q: the outline (as it is then) and every stencil already there
        (every=True: all of them, painted or not, for a steady frame)."""
        out = [self.outer(self.geometry(q))]
        out += [self.stencil_box(i) for i, st in enumerate(self.stencils) if every or st.q <= q + 1e-6]
        return out

    def key_at(self, q: float) -> BoxKey | None:
        """The outline on the wall at song 16th q (None before the first key or after the shatter)."""
        if self.shatter is not None and q >= self.shatter.q - 1e-6:
            return None
        k = None
        for key in self.keys:
            if key.q <= q + 1e-6:
                k = key
        return k

    def geometry(self, q: float) -> BoxKey:
        """The outline's geometry at q (the first key before it is painted, the last one after the shatter)."""
        k = None
        for key in self.keys:
            if key.q <= q + 1e-6:
                k = key
        return k or self.keys[0]

    def coord(self, fld: str | None, axis: str) -> float:
        """A coordinate of the first key: x0, x1, cx, y0 (top), y1 (bottom), ix0, ix1, iy0, iy1 (inner)."""
        k = self.keys[0]
        x0, y0, x1, y1 = k.box
        ix0, iy0, ix1, iy1 = k.inner
        v = {"x0": x0, "x1": x1, "cx": k.cx, "y0": y0, "y1": y1, "ix0": ix0, "ix1": ix1, "iy0": iy0, "iy1": iy1,
             "bottom": y1, "top": y0}
        if fld is None:
            return k.cx if axis == "x" else (y0 + y1) / 2
        if fld not in v:
            raise KeyError(f"box {self.id} has no {fld} (x0, x1, cx, y0, y1, ix0, ix1, iy0, iy1)")
        return v[fld]


# ---------------------------------------------------------------------------------------------- the outline

def brush_line(length: float, thick: float, rng: np.random.Generator) -> np.ndarray:
    """A horizontal stroke of thick primer (H x W float32, the stroke along x with a margin): a wobbling centre line
    and width, ragged edges, bristle streaks along it, a ragged start, and paint running dry toward one end."""
    pad = int(thick * 0.5) + 6
    L = int(round(length))
    H, W = int(round(thick)) + 2 * pad, L + 2 * pad
    xx = np.arange(W, dtype=np.float32)[None, :]
    yy = np.arange(H, dtype=np.float32)[:, None]
    yc = H / 2 + noise(1, W, 70, rng)[0] * thick * 0.07
    half = thick / 2 * (1 + 0.10 * noise(1, W, 30, rng)[0])
    rag_t = noise(1, W, 6, rng)[0] * 1.6 + noise(1, W, 2, rng)[0] * 0.5
    rag_b = noise(1, W, 6, rng)[0] * 1.6 + noise(1, W, 2, rng)[0] * 0.5
    d = yy - yc[None, :]
    prof = np.where(d < 0, np.clip((half + rag_t - (-d)) / 1.6, 0, 1), np.clip((half + rag_b - d) / 1.6, 0, 1))
    xs = pad + rng.uniform(-3, 5) + noise(H, 1, 4, rng)[:, 0:1] * 3
    xe = pad + L - rng.uniform(0, 8) + noise(H, 1, 3, rng)[:, 0:1] * 6
    inside = np.clip((xx - xs) / 2.5, 0, 1) * np.clip((xe - xx) / 2.5, 0, 1)
    bristle = np.clip(0.93 + 0.07 * noise(H, W, 1.6, rng, aniso=(1, 22)), 0, 1)
    u = (xx - pad) / max(L, 1)
    if rng.random() < 0.5:
        u = 1 - u
    dry = noise(H, W, 2.5, rng, aniso=(1, 14))
    gaps = np.where(dry > 2.5 - 1.1 * np.clip(u - 0.6, 0, 1) / 0.4, 0.35, 1.0)
    return (prof * inside * bristle * gaps).astype(np.float32)


def outline(key: BoxKey, parts: bool = False):
    """The box outline for one key: (mask, x0, y0) in street px, drips not included (they grow while wet). parts:
    ({top, bottom, left, right: its own mask}, x0, y0) on the same canvas, the same strokes (v8b: sides painted one
    by one)."""
    rng = np.random.default_rng(key.seed)
    x0, y0, x1, y1 = key.box
    t = key.thick
    bul = key.bulge * key.w
    m = int(t + bul + 20)
    X0, Y0 = int(math.floor(x0)) - m, int(math.floor(y0)) - m
    Wc, Hc = int(math.ceil(x1)) + m - X0, int(math.ceil(y1)) + m - Y0
    canvas = np.zeros((Hc, Wc), np.float32)
    part = {n: np.zeros((Hc, Wc), np.float32) for n in ("top", "bottom", "left", "right")} if parts else {}

    def put(mask: np.ndarray, ox: float, oy: float, name: str = "") -> None:
        h, w = mask.shape
        ax, ay = int(round(ox)) - X0, int(round(oy)) - Y0
        a0, b0 = max(ay, 0), max(ax, 0)
        a1, b1 = min(ay + h, Hc), min(ax + w, Wc)
        if a1 > a0 and b1 > b0:
            sub = mask[a0 - ay:a1 - ay, b0 - ax:b1 - ax]
            for cv_ in [canvas] + ([part[name]] if name in part else []):
                cv_[a0:a1, b0:b1] = 1 - (1 - cv_[a0:a1, b0:b1]) * (1 - sub)

    # top and bottom: straight strokes overshooting the corners a little
    for yc, name in ((y0 + t / 2, "top"), (y1 - t / 2, "bottom")):
        o0, o1 = rng.uniform(0.15, 0.8) * t, rng.uniform(0.15, 0.8) * t
        s = brush_line(key.w + o0 + o1, t * rng.uniform(0.92, 1.06), rng)
        pad = (s.shape[1] - int(round(key.w + o0 + o1))) / 2
        put(s, x0 - o0 - pad, yc - s.shape[0] / 2, name)
    # the sides: vertical strokes, bowed out by the bulge
    for xc, side in ((x0 + t / 2, -1), (x1 - t / 2, 1)):
        name = "left" if side < 0 else "right"
        o0, o1 = rng.uniform(0.1, 0.6) * t, rng.uniform(0.1, 0.6) * t
        s = np.ascontiguousarray(brush_line(key.h + o0 + o1, t * rng.uniform(0.92, 1.06), rng).T)
        hh, ww = s.shape
        pad = (hh - int(round(key.h + o0 + o1))) / 2
        top = y0 - o0 - pad
        if bul > 0:
            shift = side * key.bow(top + np.arange(hh))
            ext = int(math.ceil(bul)) + 2
            s = np.pad(s, ((0, 0), (ext, ext)))
            mx = np.arange(s.shape[1], dtype=np.float32)[None, :] - shift[:, None].astype(np.float32)
            my = np.repeat(np.arange(hh, dtype=np.float32)[:, None], s.shape[1], 1)
            s = cv2.remap(s, mx, my, cv2.INTER_LINEAR, borderValue=0)
            put(s, xc - ww / 2 - ext, top, name)
        else:
            put(s, xc - ww / 2, top, name)
    if parts:
        return {n: np.clip(v, 0, 1) for n, v in part.items()}, X0, Y0
    return np.clip(canvas, 0, 1), X0, Y0


# ---------------------------------------------------------------------------------------------- cracks

def crack_paths(prop: BoxProp) -> list[list[np.ndarray]]:
    """Polylines of every crack growth, in order: each growth extends the tips of the earlier growths from about
    the same point and starts new branches there; lines wander and fork like hairline cracks in old paint."""
    tips: dict[tuple[int, int], list[list[float]]] = {}
    out = []
    for c in prop.cracks:
        rng = np.random.default_rng(c.seed)
        key = (int(round(c.x / 60)), int(round(c.y / 60)))
        old = tips.setdefault(key, [])
        lines = []
        # a growth extends a few of the earlier tips from this point a little and starts new branches here
        ext = [old[i] for i in rng.choice(len(old), size=min(3, len(old)), replace=False)] if old else []
        starts = [list(tp) + [0.35] for tp in ext] + [[c.x, c.y, rng.uniform(-math.pi, math.pi), 1.0] for _ in
                                                      range(int(rng.integers(1, 3)) if old else 3)]
        new_tips = [tp for tp in old if not any(tp is e for e in ext)]
        for sx, sy, ang, share in starts:
            length = c.size * share * rng.uniform(0.55, 1.0)
            pts, x, y, a, run = [(sx, sy)], sx, sy, ang, 0.0
            while run < length:
                step = rng.uniform(3, 7)
                a += rng.normal(0, 0.42) + (rng.choice([-1, 1]) * rng.uniform(0.5, 0.9) if rng.random() < 0.08 else 0)
                a += 0.12 * math.sin(ang - a)  # keeps its general direction
                x, y = x + step * math.cos(a), y + step * math.sin(a)
                run += step
                pts.append((x, y))
                if rng.random() < 0.06 and run < length * 0.8:  # a short side branch
                    b = a + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
                    bx, by, bp = x, y, [(x, y)]
                    for _ in range(int(rng.integers(2, 7))):
                        b += rng.normal(0, 0.3)
                        bx, by = bx + 6 * math.cos(b), by + 6 * math.sin(b)
                        bp.append((bx, by))
                    lines.append(np.float32(bp))
            lines.append(np.float32(pts))
            new_tips.append([x, y, a])
        tips[key] = new_tips
        out.append(lines)
    return out


def draw_lines(lines: list[np.ndarray], width: float, offset: tuple[float, float] = (0, 0),
               pad: int = 6) -> tuple[np.ndarray, int, int]:
    """Anti-aliased polylines into a mask over their bounding box: (mask, x0, y0)."""
    if not lines:
        return np.zeros((1, 1), np.float32), 0, 0
    allp = np.concatenate(lines)
    X0 = int(math.floor(allp[:, 0].min() + offset[0])) - pad
    Y0 = int(math.floor(allp[:, 1].min() + offset[1])) - pad
    X1 = int(math.ceil(allp[:, 0].max() + offset[0])) + pad
    Y1 = int(math.ceil(allp[:, 1].max() + offset[1])) + pad
    S = 4  # drawn at 4x, then reduced (sub-pixel hairlines)
    img = np.zeros(((Y1 - Y0) * S, (X1 - X0) * S), np.uint8)
    for ln in lines:
        p = ((ln + np.float32(offset) - np.float32([X0, Y0])) * S).round().astype(np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, 255, max(1, int(round(width * S))), cv2.LINE_AA)
    m = cv2.resize(img.astype(np.float32) / 255, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
    return np.clip(m * 1.15, 0, 1), X0, Y0


# ---------------------------------------------------------------------------------------------- stencils

def _font(size: int) -> ImageFont.FreeTypeFont:
    for f in STENCIL_FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stencil_glyph(ch: str, size: int, bridge: float) -> np.ndarray:
    """One letter cut as a stencil: every enclosed counter (O, D, A, R ...) is held by bridges cut through the
    letter above and below it, as a real stencil must be (uint8 mask, white = paint)."""
    f = _font(size)
    bb = f.getbbox(ch)
    w, h = max(bb[2], 1) + 4, int(size * 1.3)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((2, 0), ch, font=f, fill=255)
    a = np.array(im)
    if not a.any():
        return a
    m = (a > 127).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return a
    bw = max(2, int(round(bridge)))
    for c, hh in zip(cnts, hier[0]):
        if hh[3] < 0:  # an outer contour: no counter
            continue
        x, y, cw, chh = cv2.boundingRect(c)
        if cw * chh < 6:
            continue
        mx = x + cw // 2
        cols = slice(max(mx - bw // 2, 0), mx - bw // 2 + bw)
        a[:y + 1, cols] = 0  # through the letter above the counter
        a[y + chh - 1:, cols] = 0  # and below it
    return a


def stencil_smear(paint: np.ndarray, length: float, frac: float, rows: tuple[int, int] | None = None,
                  seed: int = 0, right: bool = True) -> np.ndarray:
    """A stencil's wet paint dragged sideways by a body running through it: each row's ink pulled up to
    frac * length px along the run in streaks (a per-row strength, broken along the drag) that thin out, the letters
    themselves thinned where it was pulled from. paint must already have `length` px of room on that side; rows: the
    band the body crossed (array rows), the rest untouched."""
    if frac <= 0:
        return paint
    a = paint if right else paint[:, ::-1]
    h, w = a.shape
    L = max(1.0, length * min(frac, 1.0))
    rng = np.random.default_rng(seed)
    streak = cv2.GaussianBlur(rng.uniform(0.25, 1.0, (h, 1)).astype(np.float32), (1, 0), sigmaX=0.1, sigmaY=1.2)
    broken = np.clip(0.55 + 0.45 * noise(h, w, 6, rng, aniso=(1, 40)), 0.15, 1.0)  # long streaks along the drag
    d = (a * streak).astype(np.float32)
    k = float(np.exp(-3.0 / L))  # falls to 5% over the drag
    for x in range(1, w):
        np.maximum(d[:, x], d[:, x - 1] * k, out=d[:, x])
    out = np.maximum(a * (1 - 0.45 * min(frac, 1.0) * streak), d * broken * 0.9)
    if rows is not None:
        keep = np.ones(h, bool)
        keep[max(rows[0], 0):max(rows[1], 0)] = False
        out[keep] = a[keep]
    out = np.clip(out, 0, 1).astype(np.float32)
    return out if right else np.ascontiguousarray(out[:, ::-1])


def stencil_mask(st: Stencil) -> tuple[np.ndarray, np.ndarray, int, int]:
    """A stencilled rule in black spray paint: (paint, runs-free hard letters, x0, y0) in street px. The paint is the
    hard letters (sharp edges, a little mottled), plus a speckled overspray halo; runs are added by the renderer."""
    S = 3  # rendered at 3x, reduced with area averaging
    rng = np.random.default_rng(st.seed)
    f = _font(100)
    cap = (f.getbbox("H")[3] - f.getbbox("H")[1]) / 100.0
    size = int(round(st.h * S / cap))
    gap = int(round(st.h * S * 0.30))
    track = st.h * S * 0.05
    rows = []
    for line in st.lines:
        glyphs, xs, x = [], [], 0.0
        fl = _font(size)
        for chr_ in line:
            if chr_ == " ":
                x += fl.getlength(" ") * 0.9
                continue
            g = stencil_glyph(chr_, size, st.h * S * 0.085)
            glyphs.append(g)
            xs.append(int(round(x)))
            x += fl.getlength(chr_) + track
        width = int(math.ceil(x))
        hh = max((g.shape[0] for g in glyphs), default=size)
        row = np.zeros((hh, width + 8), np.uint8)
        for g, gx in zip(glyphs, xs):
            gw = min(g.shape[1], row.shape[1] - gx)
            row[:g.shape[0], gx:gx + gw] = np.maximum(row[:g.shape[0], gx:gx + gw], g[:, :gw])
        ys, xs = np.nonzero(row.any(1))[0], np.nonzero(row.any(0))[0]
        rows.append(row[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else row)  # the ink only
    W = max(r.shape[1] for r in rows)
    H = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    pad = int(st.h * S * 0.5)
    big = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
    y = pad
    for r in rows:
        if st.align == "right":
            x = pad + W - r.shape[1]
        elif st.align == "center":
            x = pad + (W - r.shape[1]) // 2
        else:
            x = pad
        big[y:y + r.shape[0], x:x + r.shape[1]] = r
        y += r.shape[0] + gap
    hard = big.astype(np.float32) / 255
    k90 = int(round(st.rot / 90.0))
    if k90:  # quarter turns: the block stands up (words down a narrow pillar, verse 2), its padding all round
        hard = np.ascontiguousarray(np.rot90(hard, k90))
    if st.rot - 90 * k90:
        M = cv2.getRotationMatrix2D((hard.shape[1] / 2, hard.shape[0] / 2), st.rot - 90 * k90, 1.0)
        hard = cv2.warpAffine(hard, M, (hard.shape[1], hard.shape[0]), flags=cv2.INTER_LINEAR)
    hard = cv2.resize(hard, (hard.shape[1] // S, hard.shape[0] // S), interpolation=cv2.INTER_AREA)
    h, w = hard.shape
    mottle = np.clip(0.95 + 0.05 * noise(h, w, 3, rng) + 0.03 * noise(h, w, 1, rng), 0, 1)
    halo = cv2.GaussianBlur(hard, (0, 0), max(1.0, st.h * 0.07)) * 0.30 + \
        cv2.GaussianBlur(hard, (0, 0), max(2.0, st.h * 0.22)) * 0.09
    speck = (rng.random((h, w)) < 0.45).astype(np.float32) * 0.8 + 0.2
    paint = np.clip(np.maximum(np.clip(hard * 1.08, 0, 1) * mottle, halo * speck), 0, 1)
    pw, ph = w * 1.0, h * 1.0
    if st.align == "right":
        x0 = st.x - pw + pad / S
    elif st.align == "center":
        x0 = st.x - pw / 2
    else:
        x0 = st.x - pad / S
    return paint.astype(np.float32), hard.astype(np.float32), int(round(x0)), int(round(st.y - pad / S))


def peel(hard: np.ndarray, paint: np.ndarray, k: int, seed: int, rate: float = 0.02,
         most: float = 0.45) -> np.ndarray:
    """The stencil after k beats of peeling: flakes come off (angular patches, a little more each beat) and cracks
    run through it (the wall shows through)."""
    if k <= 0:
        return paint
    rng = np.random.default_rng(seed)
    h, w = paint.shape
    p = min(most, rate * k)
    n = noise(h, w, 7, rng) * 0.7 + noise(h, w, 2.5, rng) * 0.3
    thr = np.quantile(n, 1 - p)
    flakes = cv2.GaussianBlur((n > thr).astype(np.float32), (0, 0), 0.6)
    lines = []
    for j in range(min(1 + k // 3, 6)):
        r2 = np.random.default_rng(seed + 101 * (j + 1))
        x, y, a = r2.uniform(0, w), r2.uniform(0, h), r2.uniform(-math.pi, math.pi)
        pts = [(x, y)]
        for _ in range(int(r2.integers(10, 30))):
            a += r2.normal(0, 0.35)
            x, y = x + 6 * math.cos(a), y + 6 * math.sin(a)
            pts.append((x, y))
        lines.append(np.float32(pts))
    cr, cx0, cy0 = draw_lines(lines, 1.6, pad=0)
    crm = np.zeros((h, w), np.float32)
    a0, b0 = max(cy0, 0), max(cx0, 0)
    a1, b1 = min(cy0 + cr.shape[0], h), min(cx0 + cr.shape[1], w)
    if a1 > a0 and b1 > b0:
        crm[a0:a1, b0:b1] = cr[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
    return (paint * (1 - flakes) * (1 - crm)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- the shatter

@dataclass
class Shard:
    mask: np.ndarray  # alpha, small
    cx: float  # its centre at the shatter
    cy: float
    vx: float
    vy: float
    spin: float  # degrees per second
    albedo: float  # white primer or black paint
    land: float = math.inf  # seconds of paint clock after the shatter when it reaches the pavement
    land_x: float = 0.0


SHARD_DRAG = 1.6  # 1/s: they slow down in the air
SHARD_G = 760.0  # px/s2 (for a 450 px figure): light flakes of dry paint fall slower than drops


def shard_pos(s: Shard, tau: float, u: float = 1.0) -> tuple[float, float, float]:
    """Centre and rotation of a shard tau seconds (paint clock) after the shatter."""
    k = SHARD_DRAG
    f = (1 - math.exp(-k * tau)) / k
    return s.cx + s.vx * f, s.cy + s.vy * f + 0.5 * SHARD_G * u * tau * tau, s.spin * tau


def shatter(prop: BoxProp, crack_masks: list[tuple[np.ndarray, int, int]], u: float = 1.0,
            ground=None) -> tuple[list[Shard], np.ndarray, int, int, np.ndarray]:
    """Break the last outline into shards: Voronoi cells over the outline (smaller near the burst point); the cells
    on the burst side fly (white primer), with black flecks from the cracks and the figure's paint; the rest stays on
    the wall as the broken remains. ground(x): the pavement line. Returns (shards, remains mask, x0, y0, fly mask)."""
    sh = prop.shatter
    key = prop.geometry(sh.q - 1e-3)
    out, X0, Y0 = outline(key)
    rng = np.random.default_rng(sh.seed)
    H, W = out.shape
    ys, xs = np.nonzero(out > 0.35)
    if len(xs) == 0:
        return [], out, X0, Y0, np.zeros_like(out)
    bx, by = sh.x - X0, sh.y - Y0
    d = np.hypot(xs - bx, ys - by)
    # seeds: denser near the burst point
    wgt = 1.0 / (40 + d)
    n_seed = int(np.clip(len(xs) / 650, 30, 140))
    pick = rng.choice(len(xs), size=n_seed, replace=False, p=wgt / wgt.sum())
    seeds = np.full((H, W), 255, np.uint8)
    seeds[ys[pick], xs[pick]] = 0
    _, lab = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lab = lab * (out > 0.02)
    span = max(key.w, key.h)
    shards, fly = [], np.zeros_like(out)
    for li in np.unique(lab[lab > 0]):
        cell = lab == li
        cy_, cx_ = np.nonzero(cell)
        cxm, cym = float(cx_.mean()), float(cy_.mean())
        dd = math.hypot(cxm - bx, cym - by)
        side = (cxm - bx) * sh.dirx
        p_fly = np.clip(1.25 - dd / (sh.fly * 2.2 * span) + (0.25 if side > -0.25 * key.w else -0.35), 0.03, 0.97)
        if rng.random() > p_fly:
            continue
        a0, a1, b0, b1 = cy_.min(), cy_.max() + 1, cx_.min(), cx_.max() + 1
        m = (out * cell)[a0:a1, b0:b1]
        if m.sum() < 4:
            continue
        fly[a0:a1, b0:b1] = np.maximum(fly[a0:a1, b0:b1], m)
        ang = math.atan2(cym - by, cxm - bx)
        sp = rng.uniform(220, 760) * u
        vx = sp * math.cos(ang) * 0.8 + sh.dirx * rng.uniform(120, 520) * u
        vy = sp * math.sin(ang) * 0.6 - rng.uniform(120, 480) * u
        shards.append(Shard(np.pad(m, 2), X0 + (b0 + b1) / 2, Y0 + (a0 + a1) / 2, vx, vy,
                            float(rng.uniform(-420, 420)), 0.74))
    # black flecks: the cracks' paint and the figure's, torn off at the burst point
    for _ in range(int(rng.integers(40, 70) * sh.power)):
        r = rng.uniform(2.0, 7.5) * u
        n = int(rng.integers(5, 9))
        angs = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = r * rng.uniform(0.5, 1.3, n)
        sz = int(math.ceil(r * 1.4)) + 3
        poly = np.int32(np.stack([sz + rad * np.cos(angs), sz + rad * np.sin(angs)], 1) * 4)
        m = np.zeros((2 * sz * 4, 2 * sz * 4), np.uint8)
        cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
        m = cv2.resize(m.astype(np.float32) / 255, (2 * sz, 2 * sz), interpolation=cv2.INTER_AREA)
        ox, oy = sh.x + rng.normal(0, key.w * 0.18), sh.y + rng.normal(0, key.h * 0.25)
        ang = math.atan2(oy - sh.y, ox - sh.x + 1e-3)
        sp = rng.uniform(300, 1000) * u * sh.power ** 0.5
        shards.append(Shard(m, ox, oy, sp * math.cos(ang) * 0.6 + sh.dirx * rng.uniform(200, 700) * u,
                            sp * math.sin(ang) * 0.6 - rng.uniform(150, 600) * u, float(rng.uniform(-600, 600)),
                            0.02))
    # when each one reaches the pavement (a little in front of the wall's foot)
    for s in shards:
        gl = (ground(s.cx) if ground else 1200.0) + rng.uniform(0, 30) * u
        tau = 0.0
        while tau < 6.0:
            x, y, _ = shard_pos(s, tau, u)
            if y + s.mask.shape[0] / 2 >= gl:
                break
            tau += 1 / 96
        s.land, s.land_x = tau, shard_pos(s, tau, u)[0]
    remains = out * (1 - np.clip(fly * 3, 0, 1))
    return shards, remains.astype(np.float32), X0, Y0, fly


def _land(s: Shard, ground, u: float, rng: np.random.Generator) -> None:
    gl = (ground(s.cx) if ground else 1200.0) + rng.uniform(0, 30) * u
    tau = 0.0
    while tau < 6.0:
        x, y, _ = shard_pos(s, tau, u)
        if y + s.mask.shape[0] / 2 >= gl:
            break
        tau += 1 / 96
    s.land, s.land_x = tau, shard_pos(s, tau, u)[0]


def _chip(rng: np.random.Generator, r: float) -> np.ndarray:
    """A small angular flake (alpha)."""
    n = int(rng.integers(4, 8))
    angs = np.sort(rng.uniform(0, 2 * math.pi, n))
    rad = r * rng.uniform(0.5, 1.3, n)
    sz = int(math.ceil(r * 1.4)) + 3
    poly = np.int32(np.stack([sz + rad * np.cos(angs), sz + rad * np.sin(angs)], 1) * 4)
    m = np.zeros((2 * sz * 4, 2 * sz * 4), np.uint8)
    cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255, (2 * sz, 2 * sz), interpolation=cv2.INTER_AREA)


def flakes(prop: BoxProp, b: Bump, u: float = 1.0, ground=None) -> list[Shard]:
    """Primer flakes knocked off the box's side by a hit from inside: small white chips that drop and flutter down
    the wall from where the side gave."""
    key = prop.geometry(b.q)
    x0, y0, x1, y1 = key.box
    rng = np.random.default_rng(b.seed)
    sx = x1 - key.thick / 2 if b.side == "right" else x0 + key.thick / 2
    sy = y0 + b.y * (y1 - y0)
    out = []
    for _ in range(b.flakes):
        m = _chip(rng, rng.uniform(1.5, 4.5) * u)
        d = 1 if b.side == "right" else -1
        s = Shard(m, sx + rng.normal(0, 4 * u), sy + rng.normal(0, 0.15 * key.h), d * rng.uniform(20, 160) * u,
                  -rng.uniform(0, 120) * u, float(rng.uniform(-300, 300)), 0.74)
        _land(s, ground, u, rng)
        out.append(s)
    return out


def splats(prop: BoxProp, u: float = 1.0) -> list[tuple[float, float, float, np.ndarray]]:
    """Paint splatter thrown across the wall by the explosion: (seconds after the shatter it lands, x, y, mask),
    the far ones later, so it spreads out over several photos."""
    sh = prop.shatter
    return _splats(sh.x, sh.y, sh.dirx, sh.seed, sh.splats, sh.power, u)


def burst_splats(b: Burst, u: float = 1.0) -> list[tuple[float, float, float, np.ndarray]]:
    """A burst's splatter on the wall, as the explosion's (splats)."""
    return _splats(b.x, b.y, b.dirx, b.seed, b.splats, b.power, u)


def burst_shards(b: Burst, u: float = 1.0, ground=None) -> list[Shard]:
    """A burst's flecks: small angular pieces thrown from round its point, mostly its way and up, falling to the
    pavement."""
    rng = np.random.default_rng([b.seed, 29])
    out = []
    for _ in range(int(b.flecks)):
        r = rng.uniform(1.5, 6.0) * u * (0.7 if b.albedo > 0.3 else 1.0)
        m = _chip(rng, r)
        ox, oy = b.x + rng.normal(0, b.spread * u), b.y + rng.normal(0, b.spread * 0.6 * u)
        ang = math.atan2(oy - b.y, ox - b.x + 1e-3)
        sp = rng.uniform(250, 900) * u * b.power ** 0.5
        s = Shard(m, ox, oy, sp * math.cos(ang) * 0.6 + b.dirx * rng.uniform(150, 600) * u,
                  sp * math.sin(ang) * 0.6 - rng.uniform(150, 550) * u, float(rng.uniform(-600, 600)), b.albedo)
        _land(s, ground, u, rng)
        out.append(s)
    return out


def _splats(x0: float, y0: float, dirx: float, seed: int, n: int, power: float,
            u: float) -> list[tuple[float, float, float, np.ndarray]]:
    rng = np.random.default_rng([seed, 23])
    out = []
    for _ in range(n):
        ang = rng.uniform(-math.pi * 0.9, math.pi * 0.6) if dirx > 0 else rng.uniform(math.pi * 0.4, math.pi * 1.9)
        dist = rng.gamma(2.0, 110) * u * power
        x, y = x0 + dist * math.cos(ang) * (1.3 if math.cos(ang) * dirx > 0 else 0.8), y0 + dist * math.sin(ang)
        r = rng.uniform(2.5, 9.0) * u * (1.4 if dist < 120 * u else 1.0)
        m = spatter(rng, r, False)
        if rng.random() < 0.35:  # a streak: the drop skidded along the wall
            m = cv2.resize(m, (int(m.shape[1] * rng.uniform(1.6, 2.6)), m.shape[0]), interpolation=cv2.INTER_LINEAR)
        out.append((dist / (1400 * u) + rng.uniform(0, 0.05), x, y, m))
    return out


def stencil_shatter(prop: BoxProp, i: int, hard: np.ndarray, x0: int, y0: int, u: float = 1.0,
                    ground=None) -> tuple[list[Shard], np.ndarray]:
    """Stencil i at the explosion: its letters cut into pieces; the pieces nearest the burst point break off and fly
    (black paint). Returns (shards, the stencil's fly mask in its own px)."""
    sh = prop.shatter
    rng = np.random.default_rng([sh.seed, 31, i])
    H, W = hard.shape
    ys, xs = np.nonzero(hard > 0.4)
    fly = np.zeros_like(hard)
    if len(xs) < 20 or sh.stencils <= 0:
        return [], fly
    n_seed = int(np.clip(len(xs) / 90, 8, 80))
    pick = rng.choice(len(xs), size=n_seed, replace=False)
    seeds = np.full((H, W), 255, np.uint8)
    seeds[ys[pick], xs[pick]] = 0
    _, lab = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lab = lab * (hard > 0.05)
    cells = []
    for li in np.unique(lab[lab > 0]):
        cy_, cx_ = np.nonzero(lab == li)
        d = math.hypot(x0 + cx_.mean() - sh.x, y0 + cy_.mean() - sh.y)
        cells.append((d, li, cy_, cx_))
    cells.sort(key=lambda c: c[0])
    shards = []
    for d, li, cy_, cx_ in cells[:int(round(sh.stencils * len(cells)))]:
        if rng.random() < 0.15:  # a few stay
            continue
        a0, a1, b0, b1 = cy_.min(), cy_.max() + 1, cx_.min(), cx_.max() + 1
        m = (hard * (lab == li))[a0:a1, b0:b1]
        fly[a0:a1, b0:b1] = np.maximum(fly[a0:a1, b0:b1], m)
        cx, cy = x0 + (b0 + b1) / 2, y0 + (a0 + a1) / 2
        ang = math.atan2(cy - sh.y, cx - sh.x)
        sp = rng.uniform(250, 700) * u * sh.power / (1 + d / (500 * u))
        s = Shard(np.pad(m, 2), cx, cy, sp * math.cos(ang), sp * math.sin(ang) - rng.uniform(80, 360) * u,
                  float(rng.uniform(-500, 500)), 0.02)
        _land(s, ground, u, rng)
        shards.append(s)
    return shards, fly


def spatter(rng: np.random.Generator, r: float, white: bool) -> np.ndarray:
    """A small splat of paint at the wall's foot where a shard landed (blob plus a few droplets)."""
    sz = int(r * 3) + 4
    m = np.zeros((2 * sz, 2 * sz), np.float32)
    cv2.ellipse(m, (sz, sz), (max(1, int(r)), max(1, int(r * rng.uniform(0.5, 0.9)))), rng.uniform(0, 180), 0, 360,
                1.0, -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(3, 9))):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 2.6) * r
        cv2.circle(m, (int(sz + d * math.cos(a)), int(sz + d * math.sin(a) * 0.7)),
                   max(1, int(rng.uniform(0.15, 0.4) * r)), 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.6) * (0.9 if white else 0.95)


# ---------------------------------------------------------------------------------------------- brush-stroke reveal

def stroke_masks(fig: np.ndarray, boxes: list, angles: list, over, seed: int, splats=40,
                 bands: list | None = None, sweeps: list | None = None) -> tuple[list, list, int]:
    """A drawing revealed by brush strokes (the cold open: giant wet strokes slam onto the wall and turn out to be
    him). fig: the drawing's matte (0 to 1, his size); boxes: one [x0, y0, x1, y1] per stroke, shares of the matte's
    painted box; angles: the strokes' slant (degrees); over, splats: px the brush runs past his silhouette and the
    splatter thrown off its end (a number, or one per stroke); sweeps: per stroke 0 to 1, how far the brush's path
    runs on past him along the stroke (a long sweep: the black band carrying on beyond his outline). Returns (figure
    parts, overshoots, pad): each stroke's share of the figure (the matte inside its ragged rotated box; the last
    stroke takes whatever is left, so he is whole) and its overshoot (the brush past his silhouette, the sweep and the
    splatter), all on the matte's frame padded by `pad` px a side. bands (a list): gets each stroke's primer swath,
    a broad roller pass of white laid first, wider and longer than the stroke, so the black sits on clean white."""
    n = len(boxes)
    overs_px = [float(o) for o in (over if isinstance(over, list | tuple) else [over] * n)]
    splat_n = [int(v) for v in (splats if isinstance(splats, list | tuple) else [splats] * n)]
    sweeps = [float(v) for v in (sweeps or [0.0] * n)]
    rng = np.random.default_rng(seed)
    F0 = fig.astype(np.float32)
    ys, xs = np.nonzero(F0 > 0.3)
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    pad = int(math.ceil(max(overs_px) * 3 + 30 + 0.45 * span * max(sweeps + [0.0])))
    F = np.pad(F0, pad)
    H, W = F.shape
    ys, xs = np.nonzero(F > 0.3)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    solid = (F > 0.15).astype(np.uint8)
    left = np.ones((H, W), np.float32)  # the share of the figure no stroke has taken yet
    parts, overs = [], []
    for i, (bb, ang) in enumerate(zip(boxes, angles)):
        ov_px, sw = overs_px[i], sweeps[i]
        x0 = bx0 + bb[0] * (bx1 - bx0)
        x1 = bx0 + bb[2] * (bx1 - bx0)
        y0 = by0 + bb[1] * (by1 - by0)
        y1 = by0 + bb[3] * (by1 - by0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a = math.radians(float(ang))
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)  # along the stroke
        v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)  # across it
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        nz = noise(H, W, 7, rng) * 0.18 + noise(H, W, 2, rng) * 0.06  # a ragged, bristly edge
        inside = np.clip(1 - np.maximum(np.abs(u) / max(hw, 1), np.abs(v) / max(hh, 1)) + nz, 0, 0.12) / 0.12
        if i == n - 1:  # the last stroke: whatever is left of him, so the figure is whole
            inside = np.maximum(inside, (F > 0.02).astype(np.float32))
        take = inside * left
        parts.append(F * take)
        left = left * (1 - inside)
        # the brush's bristles: streaks along the stroke (a random profile across it), denser in the middle
        prof = cv2.GaussianBlur(rng.random((1, 4096)).astype(np.float32), (0, 0), 1.2)[0]
        prof = np.clip((prof - prof.mean()) / max(prof.std(), 1e-6) * 0.35 + 0.75, 0, 1)
        bristle = prof[np.clip((v + 2048).astype(np.int32), 0, 4095)]
        ring = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(ov_px) + 1,) * 2)) - solid
        reach = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), max(ov_px / 6, 1)) * ring
        ov = reach * np.clip(inside * 1.4 - 0.2, 0, 1) * bristle
        L = hw * (1 + 0.9 * sw)  # the brush's path: on past him along the stroke (a long sweep)
        if sw > 0:
            band = np.clip(1 - np.abs(v) / max(hh * 0.42, 1) + nz * 0.8, 0, 0.15) / 0.15
            taper = np.clip((L - np.abs(u)) / max(0.25 * L, 1), 0, 1)  # its ends running dry
            ov = np.maximum(ov, band * taper * bristle * np.clip(0.55 + 0.45 * noise(H, W, 3, rng), 0, 1))
        sp = np.zeros((H, W), np.float32)  # splatter thrown off the stroke's far end
        ex, ey = cx + math.cos(a) * L, cy + math.sin(a) * L
        for _ in range(splat_n[i]):
            d = abs(rng.normal(0, ov_px * 1.6)) + ov_px * 0.4
            th = a + rng.normal(0, 0.9)
            px, py = ex + math.cos(th) * d, ey + math.sin(th) * d
            r = max(1.0, min(rng.gamma(1.6, min(ov_px, 26.0) * 0.06), 7.0))  # drops, not blobs
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(sp, (int(px), int(py)), int(r), float(rng.uniform(0.6, 1.0)), -1, cv2.LINE_AA)
        overs.append(np.clip(np.maximum(ov, sp) * (1 - (F > 0.15)), 0, 1))
        if bands is not None:  # the primer swath: a roller pass of white, wider and longer than the stroke
            hw_s, hh_s = L * 1.08 + ov_px, hh * 1.3 + ov_px
            rz = noise(H, W, 11, rng) * 0.05 + noise(H, W, 3, rng) * 0.02  # a roller's cleaner, stippled edge
            sw_m = np.clip(1 - np.maximum(np.abs(u) / max(hw_s, 1), np.abs(v) / max(hh_s, 1)) + rz, 0, 0.05) / 0.05
            stip = np.clip(0.9 + 0.1 * noise(H, W, 2, rng), 0, 1)
            bands.append(np.clip(sw_m * stip, 0, 1).astype(np.float32))
    return parts, overs, pad
