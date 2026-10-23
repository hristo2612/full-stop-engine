"""Render the choreography: the paint model and compositor (prototype 6's side.py with prototype 3's paint), then
the MUTO look (aimv.look.muto._photo, each photo held from its own song time) and the song slice.

Paint model, per photo: primer patches under the poses, the poses painted in order on top (the newest wet, deepest
black and glossier, with drips that grow while it is wet), older poses fading per painted frame (keepers slowly,
in-between poses fast), rain-washed, drying; now and then the animator rolls primer over an old stretch of the trail.
Paint is lit by the street: albedo x illumination x relief x light colour, plus the plate's wet speculars, the red
light's glow, and the paint reflected in the wet pavement. Paint lives behind the occluders and never in no-paint
zones. The J2 air layer is a flat paint skin with a rim of the gap's glow and spray and drips falling in later
photos; J1 is a take-off burst, a whip photo and a landing splash. Weather per photo: puddle ripples, rain streaks,
a passing car's light.

Primer modes (choreography `primer`, fade rules in `primer_fade`):
    all       a patch under every pose, fading to half once its figure is a ghost (prototype 6: a long pale band)
    keepers   patches only under keepers (beats, landings, grabs); in-between poses go straight on the wall
    none      no patches at all: the paint sits on the bare wall
    ghosts    a patch under the newest poses that fades away, and every pose painted out with primer (its silhouette,
              a little larger) two poses later, so the trail is pale figures straight on the wall (the C4 board)

Props (props.py, the box): the outline is white primer, wet and glossy when (re)painted, drying paler, with drips
that grow while it is wet; an outline replaced by a new key washes out to a ghost. An old box (painted long ago) is
dry from the first photo, its drips full length, faded and streaked by the rain; someone else curled in it is a
drawing in grey paint, as old, still. Cracks are black hairlines with a
pale chipped edge. Stencils are black spray paint, landing with a spray burst (a cloud of mist that settles); after
the shatter (or from peel_from) they crack and peel a little each beat and wash like the ghosts, and so do the
broken remains of the outline. Shards fly in front of the wall, lit by the wall's light, hang still while the song
stands still (the paint clock), then fall and settle as spatter at the wall's foot; the explosion throws splats
across the wall and pieces of the stencils' letters. Hits from inside bend a side out and knock primer flakes off.
A light hint dims a street light and its lamp for a photo (a flicker). A J2 flight from a clip leaves a trail of pale
ghost skins in the air.

Memory stays bounded: the street is read by spans (scene.Span), masks are generated on demand from per-mark seeds
and kept in a small cache, and photos are written to disk as they are made.
"""

from __future__ import annotations

import bisect
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..look import muto
from ..util import write_json
from .camera import Cam, framing_stats, plan
from .choreo import Choreo, Ev
from . import props as PR
from . import truelook as TL
from . import skinlook as SL
from . import windows as WIN
from . import reveal as RV
from .paintfx import add_drips, car_light, noise, puddle_rings, rain, ripple_warp, rough_patch
from .scene import lin2srgb, rows_of

SIZES = {"16:9": (2048, 1152), "9:16": (1152, 2048)}
LOOK = {"16:9": (1920, 1080), "9:16": (1080, 1920)}
FX_KINDS = ("box", "remains", "chip", "crack", "stencil", "spatter", "figure", "mist")  # prop marks (props.py)
SKIN_VOID = 0.25  # a wall pose with more of its paint than this over a void (the world) is the J2 skin instead
SKIN_GHOSTS = 3  # its trail: pale skins fading over this many 16ths
SKIN_SHADOW = (22.0, 30.0, 9.0, 0.38)  # the skin's shadow on the wall: offset x, y, blur (px at 450 px), strength
SKIN_HOLD = 3  # skin and paint swap only when the new background holds this many photos (no flicker along a roof)
SKIN_FADE = 2  # photos the primer patch cross-fades over where paint meets skin
SKIN_RIM = (6.0, 1.2, 0.95, (0.24, 0.215, 0.185))  # a skin's pale rim (the city's glow on its edge): width, blur (px at
#                                                 450 px), strength, linear colour
GLASS_GLOW = (5.0, 0.55, 9.0)  # paint on glass: the screens' glow round his edges: spread, strength, light blur (px)
GLASS_SOLID = 3.0  # paint on lit glass: its coverage times this (the dry-brush gaps closed: a solid black cut-out)
PRIMER_FADE = {  # floor: what a patch fades to; tau: seconds of painted frames to get there
    "all": {"floor": 0.5, "tau": 0.45},
    "keepers": {"floor": 0.3, "tau": 0.6},
    "none": {"floor": 0.0, "tau": 0.3},
    "ghosts": {"floor": 0.0, "tau": 0.2, "out": 0.72, "out_keeper": 0.5, "out_other_start": 0.35, "out_other": 0.0,
               "out_tau": 0.3, "out_after": 2},
}


def skin_hysteresis(qs: list, raw: list, hold: int = SKIN_HOLD, fade: int = SKIN_FADE,
                    gap: float = 8.0) -> tuple[list[bool], list[float]]:
    """Skin or paint per photo (raw: sky behind most of him), in time order (qs: 16ths): the state swaps only where
    the new value holds `hold` photos (a gap of more than `gap` 16ths starts afresh from the next photos' majority),
    and the paint photos next to a skin stretch get their primer patch cross-faded over `fade` photos."""
    n = len(raw)
    states = []
    st = None
    for j in range(n):
        if st is None or (j and qs[j] - qs[j - 1] > gap):
            ahead = raw[j:j + hold]
            st = sum(ahead) * 2 > len(ahead) if ahead else raw[j]
        elif raw[j] != st and j + hold <= n and all(r == raw[j] for r in raw[j:j + hold]):
            st = raw[j]
        states.append(bool(st))
    fades = [1.0] * n
    for j in range(n):
        if states[j]:
            continue
        near = [k for k in range(1, fade + 1) if (j - k >= 0 and states[j - k] and qs[j] - qs[j - k] <= gap)
                or (j + k < n and states[j + k] and qs[j + k] - qs[j] <= gap)]
        if near:
            fades[j] = min(near) / (fade + 1)
    return states, fades


# ---------------------------------------------------------------------------------------------- history

@dataclass
class Mark:
    id: int
    kind: str  # primer, cover, pose, splash, paintout, phantom
    t: float
    x0: float
    y0: float
    w: int
    h: int
    albedo: float = 0.035
    keeper: bool = False
    q: float = 0.0
    drips: tuple | None = None
    halo: bool = False
    gen: tuple = ()  # how to make the mask: ("ev", ev_index, cut) | ("patch", seed, w, h, vertical, gain, nd, L)
    of: int = -1  # paintout: the pose mark it paints out
    end: float = math.inf  # props: when it is replaced (a box key by the next one, cracks by the shatter)
    light: bool = False  # a pose with a light trail (trail=light): its ghost washes away completely
    nowash: bool = False  # (v8c) a move's `wash: false`: the rain does not wash its paint (a pose held as paint)
    ink: float = 1.0  # its own dark paint's albedo factor (a move's `ink`: the coil before the drop, darker)
    trail_w: float = 1.0  # a move's trail_in: how much of its own trail it leaves (0: the light trail's, washing away)
    clip: tuple | None = None  # (v8c) primer kept inside this street box (x0, y0, x1, y1): a move's `primer_clip`


@dataclass
class AirItem:
    kind: str  # sil | smear
    q: float
    x0: float = 0.0
    y0: float = 0.0
    gen: tuple = ()
    glow: tuple = (1.0, 0.55, 0.22)
    flags: dict = field(default_factory=dict)



def _clip_ramp(shape: tuple, box, x0: float, y0: float, seed: int, u: float = 1.0) -> np.ndarray:
    """(v8c) A mask's weight inside a street box (x0, y0, x1, y1), its own crop at (x0, y0): 1 inside, 0 outside, the
    left and right sides a dry brush edge (a few px of rough wander and a short ramp, row by row), top and bottom
    straight."""
    hh, ww = shape
    r = np.random.default_rng(seed)
    wob = cv2.GaussianBlur(r.normal(0, 1, (hh, 1)).astype(np.float32), (0, 0), sigmaX=0.1, sigmaY=9.0)
    wob = (wob / max(float(wob.std()), 1e-6)).ravel() * 4.0 * u
    xs = np.arange(ww, dtype=np.float32)[None, :]
    ramp = 6.0 * u
    a = np.clip((xs - (box[0] - x0 + wob[:, None])) / ramp + 0.5, 0, 1)
    b = np.clip(((box[2] - x0 + wob[:, None]) - xs) / ramp + 0.5, 0, 1)
    ys = np.arange(hh, dtype=np.float32)[:, None]
    v = ((ys >= box[1] - y0) & (ys < box[3] - y0)).astype(np.float32)
    return (a * b * v).astype(np.float32)

class History:
    """Every mark the choreography paints, in paint order, with lazily generated masks."""

    def __init__(self, ch: Choreo):
        self.ch = ch
        self.lib = ch.lib
        sc = ch.scene
        self.wall_base, self.feet = sc.wall_base, sc.feet
        self.events: list[Ev] = ch.events
        self.mode = ch.primer
        fade = dict(PRIMER_FADE.get(self.mode, PRIMER_FADE["all"]))
        fade.update(ch.primer_fade)
        self.fade = fade
        beats = ch.grid.beats
        self.frame_dt = float(np.median(np.diff(beats))) / 4  # a 16th: ghosts fade per painted frame
        self.u = ch.u  # street-px constants below are for a 450 px figure (prototype 6)
        self.marks: list[Mark] = []
        self.air: dict[float, list[AirItem]] = {}
        self.drops: list[tuple] = []  # (t_spawn, x, y, vx, vy, r)
        self.reveal: dict[int, dict] = {}  # pose mark -> its brush-stroke reveal (a pose's `strokes`)
        self.fig_box: dict[float, tuple] = {}
        self.fig_ev: dict[float, int] = {}  # photo q -> the event of his figure on it (screenlevels.py's smears)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._placed: dict[int, tuple] = {}
        self._build()

    # -------------------------------------------------------------- placing poses
    def _sprite(self, e: Ev) -> np.ndarray:
        d = self.lib.sprite(e.pose)
        if e.flags.get("boost"):  # a move's paint made denser (gain, gamma): the dry-brush streaks closed up
            gn, gm = e.flags["boost"]
            d = np.clip(d * float(gn), 0, 1) ** float(gm)
        if e.flags.get("fill_holes"):  # interior holes of the matte up to N px closed (a hood's opening read as an eye)
            d = fill_holes(d, int(e.flags["fill_holes"]))
        return d

    def place(self, i: int) -> tuple[np.ndarray, float, float]:
        """Density matte of event i at its scale and rotation, and its top-left in street px."""
        e = self.events[i]
        s, jx, jy = self._placed[i]
        d0 = self._sprite(e)
        d = cv2.resize(d0, (max(1, int(round(d0.shape[1] * s))), max(1, int(round(d0.shape[0] * s)))),
                       interpolation=cv2.INTER_AREA)
        ax, ay = np.float32(self.lib.anchor(e.pose, "center" if e.layer == "stamp" else e.anchor)) * s
        if e.mirror:
            d = d[:, ::-1]
            ax = d.shape[1] - ax
        if e.flipv:
            d = d[::-1]
            ay = d.shape[0] - ay
        if e.rot:
            pad = int(max(d.shape) * 0.35)
            d = np.pad(d, pad)
            ax, ay = ax + pad, ay + pad
            M = cv2.getRotationMatrix2D((float(ax), float(ay)), e.rot, 1.0)
            d = cv2.warpAffine(d, M, (d.shape[1], d.shape[0]), flags=cv2.INTER_LINEAR)
            ys, xs = np.nonzero(d > 0.01)
            cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            d = d[cy0:cy1, cx0:cx1]
            ax, ay = ax - cx0, ay - cy0
        return np.ascontiguousarray(d), e.x - ax + jx, e.y - ay + jy

    def _droplets(self, d, X0, Y0, rng, n, t, vx=(200, 900), vy=(-500, 100), xmin=None, rmax=7):
        u = self.u
        vx, vy, rmax = (vx[0] * u, vx[1] * u), (vy[0] * u, vy[1] * u), rmax * u
        ys, xs = np.nonzero(d > 0.4)
        if xmin is not None:
            keep = xs + X0 > xmin
            ys, xs = ys[keep], xs[keep]
        if len(xs) == 0:
            return
        pick = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        for k in pick:
            self.drops.append((t, X0 + xs[k] + rng.normal(0, 20 * u), Y0 + ys[k] + rng.normal(0, 20 * u),
                               rng.uniform(*vx), rng.uniform(*vy), rng.uniform(1.2 * u, rmax)))

    def _mark(self, **kw) -> Mark:
        m = Mark(id=len(self.marks), **kw)
        self.marks.append(m)
        return m

    def _patch(self, rng, kind, t, x0, y0, w, h, albedo, vertical, gain, nd, L, q):
        return self._mark(kind=kind, t=t, x0=x0, y0=y0, w=w, h=h, albedo=albedo, q=q,
                          gen=("patch", int(rng.integers(0, 2 ** 31)), w, h, vertical, gain, nd, L))

    def _build(self) -> None:
        ch = self.ch
        sc = ch.scene
        # one random stream per move, seeded by its source file (include) and its place there: a segment file paints
        # the same in the film as alone, and a change to one move leaves the paint of the moves after it alone
        rngs: dict[tuple, np.random.Generator] = {}
        self.fx: dict[int, dict] = {}  # prop marks: their masks and parameters
        self._skin = self._skin_states()  # event -> (a skin?, its primer patch's strength): with hysteresis
        g = ch.grid
        u = self.u
        freezes = [(a, b) for a, b in ch.freezes]
        for i, e in enumerate(self.events):
            t = g.t(e.q)
            mv = ch.moves[e.move] if 0 <= e.move < len(ch.moves) else None
            src = mv.src if mv is not None else 0
            key = (src, mv.local_i if mv is not None else -1)
            if key not in rngs:
                rngs[key] = np.random.default_rng([ch.sources[src]["seed"] if src < len(ch.sources) else ch.seed,
                                                   key[1] + 1])
            rng = rngs[key]
            if TL.is_true(ch, e):  # out of the paint (truelook.py): a solid figure, no primer, marks, skin or droplets
                TL.add(self, i, e, self.lib.m(e.pose))
                continue
            if e.layer == "smear":
                self.air.setdefault(e.q, []).append(AirItem("smear", e.q, flags=dict(e.flags)))
                continue
            m = self.lib.m(e.pose)
            if m["kind"] == "stamp" or e.layer == "stamp":
                s = (e.w or 400.0) / self._sprite(e).shape[1]
            else:
                s = m["scale"] * e.scale * (1 + rng.normal(0, 0.012))
            j = rng.normal(0, 2.5, 2) if e.jitter else (0.0, 0.0)
            self._placed[i] = (s, float(j[0]), float(j[1]))
            d, x0, y0 = self.place(i)
            h, w = d.shape
            sk = self._skin.get(i)
            if sk is not None and sk[0]:
                # only sky or a void behind him (the world: rooftops between structures, a gap): the J2 skin, never
                # paint with primer, and a soft shadow on whatever wall is nearest (PAINT.md, The world)
                e.layer = "air"
                e.flags = dict(e.flags, skin=True, skin_auto=True, ghosts=e.flags.get("ghosts", SKIN_GHOSTS))
            if e.layer == "air":
                glow = tuple(e.flags.get("glow", (1.0, 0.55, 0.22)))
                self.air.setdefault(e.q, []).append(AirItem("sil", e.q, x0, y0, ("ev", i, None), glow,
                                                            {"ghosts": e.flags.get("ghosts", 0.0),
                                                             "hang": e.flags.get("hang"),
                                                             "skin": bool(e.flags.get("skin"))}))
                self.fig_box[e.q] = (x0, y0, x0 + w, y0 + h)
                self.fig_ev[e.q] = i
                if not e.flags.get("skin"):
                    self._droplets(d, x0, y0, rng, 70, t, vx=(-150, 250), vy=(-50, 250), rmax=5)  # spray and drips
                self._mark(kind="phantom", t=t, x0=x0, y0=y0, w=0, h=0, q=e.q)
                continue
            if e.layer == "stamp":
                if e.flags.get("swath"):  # the stamp's shape painted in primer (bright wet white), staying on the
                    self._fx_mark("remains", t - 0.0015, d, x0, y0, swath=True)  # wall: a band, a block (v8)
                    continue
                if e.flags.get("primer") and self.mode != "none":
                    pw, ph = int(w * 1.05), int(min(h * 1.1, self.wall_base + sc.ground_dy(e.x) - y0 + 10))
                    pm_ = self._patch(rng, "primer", t - 0.0015, int(x0 - 0.02 * w), int(y0 - 0.05 * h), pw, ph,
                                      0.74, False, 0.93, 8, 50 * u, e.q)
                    pm_.keeper = e.flags.get("primer") == "keep"  # (keep: it stays on the wall like the mark)
                    pm_.clip = tuple(e.flags["primer_clip"]) if e.flags.get("primer_clip") is not None else None
                drips = (int(rng.integers(6, 14)), float(rng.uniform(40, 110)) * u, int(rng.integers(0, 10 ** 6)))
                if e.flags.get("drips"):  # (a stage's own drips: [n, length px], the cold open's splat, v8b)
                    drips = (int(e.flags["drips"][0]), float(e.flags["drips"][1]) * u, drips[2])
                self._mark(kind="splash", t=t - 0.0005, x0=x0, y0=y0, w=w, h=h, albedo=0.03, keeper=True, q=e.q,
                           drips=drips, gen=("ev", i, None, 0.97))
                if e.flags.get("splash"):  # droplets flying off it (drops, 40; lead: s already in the air on its photo)
                    self._droplets(d, x0, y0, rng, int(e.flags.get("drops", 40)), t - float(e.flags.get("lead", 0.0)),
                                   vx=(-350, 350), vy=(-600, -100), rmax=6)
                continue
            # a pose on the wall
            self.fig_box[e.q] = (x0, y0, x0 + w, y0 + h)
            self.fig_ev[e.q] = i
            # flags trail=light (a figure repainted in one place, as in the box): no keepers (not even a move's key
            # drawings), so its ghosts and paint-outs wash away instead of piling up; clip: the box its primer patch
            # stays inside;
            # patch: the patch's strength (a thin one in the box, so the box stays an outline on the bare wall)
            light = e.flags.get("trail") == "light"
            keeper = (e.keeper and not light) or (not light and ((abs(e.q - round(e.q)) < 1e-6 and
                                                                  int(round(e.q)) % 4 == 0) or rng.random() < 0.06))
            patch = self.mode == "all" or self.mode == "ghosts" or (self.mode == "keepers" and keeper)
            pw, ph = int(w * rng.uniform(1.05, 1.3)), int(h * rng.uniform(0.95, 1.2))
            if e.flags.get("patch_grow"):  # back on the wall after a flight: the primer arrives gradually, small
                j, n_g = e.flags["patch_grow"]  # round him, growing
                fg = 0.45 + 0.55 * min(1.0, j / max(n_g, 1))
                pw, ph = max(int(pw * fg), 8), max(int(ph * fg), 8)
            px0 = int(x0 + w / 2 - pw / 2 + 0.05 * w + rng.normal(0, 8 * u))
            py0 = int(y0 + h / 2 - ph / 2 + rng.normal(-10 * u, 25 * u))
            wb = self.wall_base + sc.ground_dy(e.x)
            if py0 + ph > wb:
                py0 = min(py0, int(wb - ph + rng.uniform(-20, 30) * u))
                ph = min(ph, int(wb - py0 + 4))
            vert = rng.random() < 0.3
            gain = float(rng.uniform(0.82, 0.97))
            nd, L = int(rng.integers(2, 7)), float(rng.uniform(15, 55)) * u
            alb = float(rng.uniform(0.58, 0.76))
            for ck_ in ("clip", "primer_clip"):  # (a box's; v8c: a move's primer_clip)
                if e.flags.get(ck_) is None:
                    continue
                c0, c1, c2, c3 = e.flags[ck_]
                nx0, ny0 = max(px0, int(c0)), max(py0, int(c1))
                nx1, ny1 = min(px0 + pw, int(c2)), min(py0 + ph, int(c3))
                px0, py0, pw, ph = nx0, ny0, max(nx1 - nx0, 0), max(ny1 - ny0, 0)
                patch = patch and pw > 8 and ph > 8
            if patch and self.mode != "none":
                self._patch(rng, "primer", t - 0.001, px0, py0, pw, ph, alb, vert,
                            gain * float(e.flags.get("patch", 1.0)) * self._skin.get(i, (False, 1.0))[1], nd, L, e.q)
            if e.flags.get("takeoff"):
                # the part of the figure past the edge leaves the wall: in J1 it bursts into spatter, in J2 it is the
                # first photo of the flying paint skin
                edge = float(e.flags.get("edge", x0 + w))
                c = int(round(edge - x0))
                if 0 < c < w:
                    burst = e.flags.get("burst")
                    seed = int(rng.integers(0, 2 ** 31))
                    if burst == "J1":
                        self._droplets(d, x0, y0, rng, 650, t, xmin=edge - 60, vx=(250, 1600), vy=(-650, 250), rmax=11)
                        self._droplets(d, x0, y0, rng, 40, t, xmin=edge - 10, vx=(150, 700), vy=(-300, 100), rmax=19)
                    else:
                        self._droplets(d, x0, y0, rng, 60, t, xmin=edge - 10, vx=(100, 500), vy=(-200, 100), rmax=4)
                    glow = tuple(e.flags.get("glow", (1.0, 0.55, 0.22)))
                    self.air.setdefault(e.q, []).append(AirItem("sil", e.q, x0, y0, ("front", i, c, burst, seed), glow))
            tw = float(e.flags.get("trail_w", 1.0))  # (a move's trail_in: its trail fading in from the light trail's)
            mk = self._mark(kind="pose", t=t, x0=x0, y0=y0, w=w, h=h, keeper=keeper, q=e.q,
                            drips=(int(rng.integers(4, 11)), float(rng.uniform(20, 70)) * u,
                                   int(rng.integers(0, 10 ** 6))),
                            halo=bool(e.flags.get("halo")), gen=("ev", i, None), light=light,
                            ink=float(e.flags.get("ink", 1.0)), trail_w=tw, nowash=bool(e.flags.get("nowash")))
            if e.flags.get("strokes"):  # painted by brush strokes slamming onto the wall (the cold open)
                self._reveal_build(mk, d, x0, y0, e.flags, rng)
            elif e.flags.get("drip_in"):  # formed by paint running down the wall (cold open B, v8b)
                self._drip_build(mk, d, e.flags["drip_in"])
            if self.mode == "ghosts":
                # painted out with primer when the next pose goes up (its time is set below)
                self._mark(kind="paintout", t=math.inf, x0=x0 - 8, y0=y0 - 8, w=w + 16, h=h + 16, keeper=keeper,
                           q=e.q, albedo=alb, gen=("out", mk.id, int(rng.integers(0, 2 ** 31))), of=mk.id,
                           trail_w=tw, clip=tuple(e.flags["primer_clip"]) if e.flags.get("primer_clip") is not None
                           else None)
            # now and then the animator rolls primer over an old stretch of the trail
            frozen = any(a <= e.q < b for a, b in freezes)
            if self.mode in ("all", "keepers") and rng.random() < 0.14 and not frozen and e.anchor == "foot":
                cw, chh = int(rng.uniform(120, 280) * u), int(rng.uniform(180, 440) * u)
                cx0 = int(e.x - rng.uniform(300, 900) * u - cw / 2)
                fe = sc.feet_at(e.x)
                cy0 = int(rng.uniform(fe - 470 * u, fe - chh * 0.6))
                self._patch(rng, "cover", t + 0.002, cx0, cy0, cw, chh, float(rng.uniform(0.62, 0.82)),
                            rng.random() < 0.5, float(rng.uniform(0.45, 0.8)), int(rng.integers(1, 5)), 60.0 * u, e.q)
        self.pose_times = sorted(m.t for m in self.marks if m.kind in ("pose", "phantom"))
        # a pose is painted out with primer when the `out_after`-th pose after it goes up
        after = int(self.fade.get("out_after", 2))
        for m in self.marks:
            if m.kind == "paintout":
                k = bisect.bisect_right(self.pose_times, self.marks[m.of].t + 1e-6) + after - 1
                m.t = self.pose_times[k] + 0.0015 if k < len(self.pose_times) else math.inf
        # primer (and paint-outs) under every pose, then poses, covers and splashes in time order
        # (self.fx, the prop marks' masks and parameters, starts before the events: a stamp's swath is one)
        self.shards: list[tuple] = []  # (song t of the shatter, its paint clock, [Shard])
        self._peel: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._build_props()
        for rv in self.reveal.values():  # the strokes' overshoot and splatter: they stay on the wall
            for tj, band, ox, oy in rv.get("bands", []):  # under each, the pale primer the brush lays (he reads)
                self._fx_mark("remains", tj - 0.001, band, ox, oy, swath=True)
            for tj, ov, ox, oy, seed in rv.get("overs", []):
                self._fx_mark("spatter", tj, ov, ox, oy, albedo=0.03,
                              drips=(int(4 + seed % 5), 45.0 * self.u, seed))
        prim = sorted([m for m in self.marks if m.kind in ("primer", "paintout")], key=lambda m: m.t)
        fixt = sorted([m for m in self.marks if m.kind in FX_KINDS], key=lambda m: m.t)
        rest = sorted([m for m in self.marks if m.kind in ("pose", "cover", "splash")], key=lambda m: m.t)
        self.order = prim + fixt + rest
        self._air_q = sorted(self.air)
        self._ox0 = np.array([m.x0 for m in self.order], np.float64)
        self._ox1 = np.array([m.x0 + m.w for m in self.order], np.float64)
        self._ot = np.array([m.t for m in self.order], np.float64)
        dr = np.array(self.drops, np.float64).reshape(-1, 6)
        self._drops = dr[np.argsort(dr[:, 0])] if len(dr) else dr
        wash_rng = np.random.default_rng(5)
        self.wash_tex = np.clip(noise(1500, 1500, 3, wash_rng, aniso=(25, 1)) * 0.5 + 0.2, 0, 1)

    # -------------------------------------------------------------- props
    def _fx_mark(self, kind: str, t: float, mask: np.ndarray, x0: int, y0: int, end: float = math.inf,
                 **data) -> Mark:
        h, w = mask.shape
        mk = self._mark(kind=kind, t=t, x0=x0, y0=y0, w=w, h=h, keeper=True, end=end)
        self.fx[mk.id] = {"mask": mask.astype(np.float32), **data}
        return mk

    def _reveal_build(self, mk: Mark, d: np.ndarray, x0: float, y0: float, fl: dict, rng) -> None:
        """A pose's brush-stroke reveal (flags strokes: [[t, box, angle, over, splats, drops, sweep], ...] with the
        move's over, splats, drops as defaults, seed): the drawing cut into the strokes' shares (props.stroke_masks),
        each stroke's primer swath and its overshoot and splatter kept for the wall (made in _build after the
        props), and the wet drops it throws off along its direction."""
        st = fl["strokes"]
        opt = lambda s_, k, key, dflt: float(s_[k]) if len(s_) > k and s_[k] is not None else float(fl.get(key, dflt))
        bands: list = []
        parts, overs, pad = PR.stroke_masks(d, [s_[1] for s_ in st], [s_[2] for s_ in st],
                                            [opt(s_, 3, "over", 20.0) * self.u for s_ in st], int(fl.get("seed", 1)),
                                            [int(opt(s_, 4, "splats", 40)) for s_ in st], bands,
                                            [opt(s_, 6, "sweep", 0.0) for s_ in st])
        h, w = d.shape
        crop = lambda a: a[pad:pad + h, pad:pad + w]
        self.reveal[mk.id] = {"t": [float(s_[0]) for s_ in st], "parts": [crop(p_) for p_ in parts], "cache": {},
                              "overs": [(float(s_[0]), ov, int(round(x0)) - pad, int(round(y0)) - pad,
                                         int(rng.integers(0, 10 ** 6))) for s_, ov in zip(st, overs)],
                              "bands": [(float(s_[0]), b_, int(round(x0)) - pad, int(round(y0)) - pad)
                                        for s_, b_ in zip(st, bands)] if fl.get("primer", True) else []}
        for s_, part in zip(st, parts):
            tj, a = float(s_[0]), math.radians(float(s_[2]))
            sx = 1.0 if math.cos(a) >= 0 else -1.0
            # thrown off along the stroke: already in the air on its photo (the slam), the first one is the thumbnail
            self._droplets(crop(part), x0, y0, rng, int(opt(s_, 5, "drops", 90)), tj - 0.06,
                           vx=(sx * 250 if sx > 0 else -1300, 1300 if sx > 0 else -250), vy=(-450, 150), rmax=8)

    def _drip_build(self, mk: Mark, d: np.ndarray, di: dict) -> None:
        """A pose formed by paint running down the wall (flags drip_in: {t0, t1, streaks, lead, amp, seed}): from t0
        to t1 a ragged front runs down his drawing from its top to its feet, everything above it painted, and
        `streaks` thin runs of paint lead it (each up to `lead` of his height ahead, never past his own outline's
        foot in its column). The front's raggedness (`amp`, a share of his height) settles as it reaches the feet,
        so he is whole at t1."""
        rng = np.random.default_rng(int(di.get("seed", 1)))
        F = (d > 0.02)
        ys, xs = np.nonzero(F)
        h, w = d.shape
        top, bot = float(ys.min()), float(ys.max() + 1)
        colbot = np.full(w, -1.0, np.float32)
        for x_ in np.unique(xs):
            colbot[x_] = ys[xs == x_].max()
        nz = cv2.GaussianBlur(rng.normal(0, 1, (1, w)).astype(np.float32), (0, 0), max(w / 40, 1.0))[0]
        nz = nz / max(float(np.abs(nz).max()), 1e-6)
        cols = xs[rng.choice(len(xs), size=min(int(di.get("streaks", 24)), len(xs)), replace=False)]
        streaks = [(int(c_), float(rng.uniform(0.4, 1.0)), float(rng.uniform(0.9, 2.6)), float(rng.uniform(0, 0.25)))
                   for c_ in cols]
        self.reveal[mk.id] = {"kind": "drip", "t0": float(di["t0"]), "t1": float(di["t1"]), "d": d, "top": top,
                              "bot": bot, "colbot": colbot, "nz": nz, "streaks": streaks,
                              "lead": float(di.get("lead", 0.3)) * (bot - top), "amp": float(di.get("amp", 0.07)) *
                              (bot - top)}

    def _dripped(self, rv: dict, t: float) -> np.ndarray | None:
        """A drip_in pose's painted share at song time t (before its own drips): None before t0."""
        if t < rv["t0"] - 1e-6:
            return None
        u = min(1.0, max(0.0, (t - rv["t0"]) / max(rv["t1"] - rv["t0"], 1e-6)))
        if u >= 1.0:
            return rv["d"]
        d = rv["d"]
        h, w = d.shape
        e = 1 - (1 - u) ** 1.4  # quick off the splat, slowing as it reaches his feet
        span = rv["bot"] - rv["top"]
        front = rv["top"] + (span + rv["amp"]) * e + rv["amp"] * (1 - u) * rv["nz"]  # per column
        yy = np.arange(h, dtype=np.float32)[:, None]
        fill = np.clip((front[None, :] - yy) / 6.0, 0, 1)
        out = d * fill
        for x_, sp, wd, delay in rv["streaks"]:  # the runs leading the front, inside his column's reach
            if u <= delay:
                continue
            ya = float(front[x_])
            r_ = max(1, int(round(wd * 1.2)))  # (its bead never past his own outline's foot in that column)
            yb = min(ya + rv["lead"] * sp * min(1.0, (u - delay) * 3.0), float(rv["colbot"][x_]) - r_)
            if yb <= ya + 1:
                continue
            cv2.line(out, (x_, int(ya)), (x_, int(yb)), 0.92, max(1, int(round(wd))), cv2.LINE_AA)
            cv2.circle(out, (x_, int(yb)), r_, 0.95, -1, cv2.LINE_AA)
        return np.clip(out, 0, 1)

    def _revealed(self, rv: dict, t: float, pct: float, n: int, L: float, seed: int):
        """(the revealed mask at song time t with each stroke's drips grown from its own time, the paint age of the
        newest stroke) of a brush-stroke reveal, or (None, 0) before its first stroke. A drip_in reveal (cold open
        B): its painted share with the pose's drips grown from t0, the paint's age from t0."""
        if rv.get("kind") == "drip":
            m_ = self._dripped(rv, t)
            if m_ is None:
                return None, 0.0
            pg = pct - self.ch.paint_clock(rv["t0"])
            return add_drips(m_.copy(), np.random.default_rng(seed), n, 0.9, L, min(1.0, 0.25 + pg / 0.5)), pg
        k = sum(1 for tj in rv["t"] if tj <= t + 1e-6)
        if k == 0:
            return None, 0.0
        pc = self.ch.paint_clock
        out = None
        for j in range(k):
            pg = pct - pc(rv["t"][j])
            mj = add_drips(rv["parts"][j].copy(), np.random.default_rng([seed, j]), max(2, n // 2), 0.9, L,
                           min(1.0, 0.25 + pg / 0.5))
            out = mj if out is None else np.maximum(out, mj)
        return out, pct - pc(rv["t"][k - 1])

    def _build_props(self) -> None:
        """Marks for every prop (props.py): box keys, cracks, stencils, the remains and the spatter, and the shards."""
        ch, g, u = self.ch, self.ch.grid, self.u
        sc = ch.scene
        beat = float(np.median(np.diff(g.beats)))
        for b in ch.bursts:  # paint explosions without a box: flecks in the air, splatter on the wall
            tb = g.t(b.q)
            fl = PR.burst_shards(b, u, ground=sc.wall_base_at)
            self.shards.append((tb, ch.paint_clock(tb), fl))
            self._settle(tb, fl, np.random.default_rng([b.seed, 19]))
            rng = np.random.default_rng([b.seed, 13])
            for dt, x, y, sm in PR.burst_splats(b, u):
                self._fx_mark("spatter", ch.paint_time(tb, dt), sm, int(x - sm.shape[1] / 2), int(y - sm.shape[0] / 2),
                              albedo=b.albedo if b.albedo > 0.3 else 0.03,
                              drips=(int(rng.integers(0, 3)), float(rng.uniform(10, 40)) * u,
                                     int(rng.integers(0, 10 ** 6))))
        for pid, pr in ch.props.items():
            t_sh = g.t(pr.shatter.q) if pr.shatter else math.inf
            keys = [k for k in sorted(pr.keys, key=lambda k: k.q) if not pr.shatter or k.q < pr.shatter.q]
            for i, k in enumerate(keys):
                m, X0, Y0 = PR.outline(k)
                nxt = g.t(keys[i + 1].q) if i + 1 < len(keys) else t_sh
                rng = np.random.default_rng([k.seed, 7])
                q_end = keys[i + 1].q if i + 1 < len(keys) else (pr.shatter.q if pr.shatter else math.inf)
                bumps = [(g.t(b.q), b.amp, k.box[1] + b.y * k.h - Y0, b.side, k.cx - X0, 0.16 * k.h)
                         for b in pr.bumps if k.q - 1e-6 <= b.q < q_end]
                by = "drop" if k.drop else ("key" if i + 1 < len(keys) else "shatter")
                if k.sides:  # v8b: each side its own mark, painted (wet, its drips running) from its own time
                    parts, X0, Y0 = PR.outline(k, parts=True)
                    for j, (name, dt) in enumerate(k.sides):
                        rj = np.random.default_rng([k.seed, 7, j + 1])
                        self._fx_mark("box", g.t(k.q) + dt, parts[name], X0, Y0, end=nxt, by=by,
                                      drips=(0 if k.drop else int(rj.integers(3, 6)), float(rj.uniform(0.18, 0.3)) * k.h,
                                             int(rj.integers(0, 10 ** 6))), old=k.old,
                                      wash=int(rj.integers(0, 10 ** 6)), paint_in=k.paint_in, bumps=bumps,
                                      bright=k.bright)
                    continue
                self._fx_mark("box", g.t(k.q), m, X0, Y0, end=nxt, by=by,
                              drips=(0 if k.drop else int(rng.integers(12, 20)), float(rng.uniform(0.18, 0.3)) * k.h,
                                     int(rng.integers(0, 10 ** 6))), old=k.old, wash=int(rng.integers(0, 10 ** 6)),
                              paint_in=k.paint_in, bumps=bumps, bright=k.bright)
            for b in pr.bumps:  # primer flakes knocked off by each hit, falling to the wall's foot
                fl = PR.flakes(pr, b, u, ground=sc.wall_base_at)
                tb = g.t(b.q)
                self.shards.append((tb, ch.paint_clock(tb), fl))
                self._settle(tb, fl, np.random.default_rng([b.seed, 5]))
            if pr.figure is not None:  # someone else in the box: grey, old, still
                fg = pr.figure
                d0 = self.lib.sprite(fg.pose)
                s_ = self.lib.m(fg.pose)["scale"] * fg.scale
                d = cv2.resize(d0, (max(1, int(round(d0.shape[1] * s_))), max(1, int(round(d0.shape[0] * s_)))),
                               interpolation=cv2.INTER_AREA)
                ax, ay = np.float32(self.lib.anchor(fg.pose, "bc")) * s_
                if fg.mirror:
                    d, ax = np.ascontiguousarray(d[:, ::-1]), d.shape[1] - ax
                tf = g.t(fg.q) if fg.q is not None else (g.t(keys[0].q) if keys else 0.0)
                self._fx_mark("figure", tf, d, int(round(fg.x - ax)), int(round(fg.y - ay)), old=fg.old,
                              wash=fg.seed % 10 ** 6, paint_in=fg.paint_in)
            paths = PR.crack_paths(pr)
            fly = None
            if pr.shatter:
                cms = []
                shards, remains, RX0, RY0, fly = PR.shatter(pr, cms, u, ground=sc.wall_base_at)
                fly = (cv2.dilate((fly > 0.1).astype(np.uint8), np.ones((15, 15), np.uint8)), RX0, RY0)
                self._fx_mark("remains", t_sh, remains, RX0, RY0)
                pc0 = ch.paint_clock(t_sh)
                self.shards.append((t_sh, pc0, shards))
                self._settle(t_sh, shards, np.random.default_rng([pr.shatter.seed, 11]))
                rng = np.random.default_rng([pr.shatter.seed, 13])
                for dt, x, y, sm in PR.splats(pr, u):  # splatter thrown across the wall, spreading out
                    self._fx_mark("spatter", ch.paint_time(t_sh, dt), sm, int(x - sm.shape[1] / 2),
                                  int(y - sm.shape[0] / 2), albedo=0.03,
                                  drips=(int(rng.integers(0, 3)), float(rng.uniform(10, 40)) * u,
                                         int(rng.integers(0, 10 ** 6))))
            for c, lines in zip(pr.cracks, paths):
                t = g.t(c.q)
                for kind, width, off in (("chip", 1.5, (1.0, 1.0)), ("crack", 1.1, (0.0, 0.0))):
                    cm, cx0, cy0 = PR.draw_lines(lines, width * u ** 0.5, offset=off)
                    if fly is None:
                        self._fx_mark(kind, t, cm, cx0, cy0)
                        continue
                    self._fx_mark(kind, t, cm, cx0, cy0, end=t_sh)
                    fm, fx0, fy0 = fly  # after the shatter: only what was not on the flying outline stays
                    keep = cm.copy()
                    a0, b0 = max(fy0 - cy0, 0), max(fx0 - cx0, 0)
                    a1, b1 = min(fy0 + fm.shape[0] - cy0, cm.shape[0]), min(fx0 + fm.shape[1] - cx0, cm.shape[1])
                    if a1 > a0 and b1 > b0:
                        keep[a0:a1, b0:b1] *= 1 - fm[a0 + cy0 - fy0:a1 + cy0 - fy0, b0 + cx0 - fx0:b1 + cx0 - fx0]
                    self._fx_mark(kind, t_sh, keep, cx0, cy0, after=True)
            for i, st in enumerate(pr.stencils):
                paint, hard, sx0, sy0 = PR.stencil_mask(st)
                rng = np.random.default_rng([st.seed, 3])
                sfly = None
                if pr.shatter and pr.shatter.stencils > 0:  # its letters nearest the explosion break off and fly
                    ssh, sfly = PR.stencil_shatter(pr, i, hard, sx0, sy0, u, ground=sc.wall_base_at)
                    self.shards.append((t_sh, ch.paint_clock(t_sh), ssh))
                    self._settle(t_sh, ssh, np.random.default_rng([st.seed, 17]))
                    sfly = cv2.dilate((sfly > 0.1).astype(np.float32), np.ones((3, 3), np.uint8))
                # the spray burst as it goes on: a speckled cloud of black mist round the letters, settling in half
                # a second (its own mark, wider than the letters' overspray)
                pad = int(round(1.8 * st.h))
                hp = np.pad(hard, pad)
                cl = cv2.GaussianBlur(hp, (0, 0), max(4.0, 0.6 * st.h)) * 2.6 + \
                    cv2.GaussianBlur(hp, (0, 0), max(6.0, 1.3 * st.h)) * 1.8
                rs = np.random.default_rng([st.seed, 41])
                mist = np.clip(cl, 0, 1) ** 1.5 * (0.25 + 0.75 * (rs.random(hp.shape) < 0.4)) * 0.9
                self._fx_mark("mist", g.t(st.q), cv2.GaussianBlur(mist, (0, 0), 0.7), sx0 - pad, sy0 - pad,
                              albedo=st.albedo)
                smear = None
                if st.smear is not None:  # a body runs through it: room on the smear's side, smeared from its time
                    qs, ln, over, rows, side = st.smear
                    pw = int(math.ceil(ln))
                    padw = ((0, 0), (0, pw)) if side > 0 else ((0, 0), (pw, 0))
                    paint, hard = np.pad(paint, padw), np.pad(hard, padw)
                    if side < 0:
                        sx0 -= pw
                    if sfly is not None:
                        sfly = np.pad(sfly, padw)
                    rr = None if rows is None else (int(rows[0]) - sy0, int(rows[1]) - sy0)
                    smear = (g.t(qs), max(g.t(qs + over) - g.t(qs), 1e-3), ln, rr, side > 0)
                self._fx_mark("stencil", g.t(st.q), paint, sx0, sy0, hard=hard, seed=st.seed, t_sh=t_sh, beat=beat,
                              drips=(int(rng.integers(2, 5)) * (1 if st.runs else 0), float(rng.uniform(0.5, 1.2)) * st.h,
                                     int(rng.integers(0, 10 ** 6))), h=st.h, fly=sfly,
                              t_peel=g.t(pr.peel_from) if pr.peel_from is not None else None, smear=smear,
                              albedo=st.albedo,
                              fade=(g.t(st.fade[0]), max(g.t(st.fade[0] + st.fade[1]) - g.t(st.fade[0]), 1e-3))
                              if st.fade is not None else None)

    def _settle(self, t0: float, shards: list, rng: np.random.Generator) -> None:
        """Each shard settles at the foot of the wall where it lands, as spatter."""
        ch, sc, u = self.ch, self.ch.scene, self.u
        for sh in shards:
            if not math.isfinite(sh.land):
                continue
            white = sh.albedo > 0.3
            r = max(2.0, math.sqrt(float(sh.mask.sum())) * (0.55 if white else 0.9))
            sp = PR.spatter(rng, r, white)
            wb = sc.wall_base_at(sh.land_x)
            yy = wb - rng.uniform(8, 70) * u - sp.shape[0] / 2
            self._fx_mark("spatter", ch.paint_time(t0, sh.land), sp, int(sh.land_x - sp.shape[1] / 2),
                          int(yy), albedo=0.7 if white else 0.03,
                          drips=(int(rng.integers(1, 4)), float(rng.uniform(15, 50)) * u,
                                 int(rng.integers(0, 10 ** 6))))

    def _paint_in(self, mk: np.ndarray, p: float, seed: int) -> np.ndarray:
        """A mark painted in up to share p (0 to 1): a brush sweep from its top left to its bottom right, the wet edge
        ragged."""
        if p >= 1:
            return mk
        h, w = mk.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nz = noise(h, w, 9, np.random.default_rng(seed)) * 0.08
        f = 0.55 * yy / max(h, 1) + 0.45 * xx / max(w, 1) + nz
        return mk * np.clip((p * 1.2 - f) / 0.06, 0, 1)

    def _bumped(self, mk: np.ndarray, bumps: list, t: float) -> np.ndarray:
        """The outline where hits from inside push a side out: a bump that springs back over a few photos."""
        for tb, amp, yc, side, xm, sig in bumps:
            dt = t - tb
            if not 0 <= dt < 0.45:
                continue
            a = amp * math.exp(-dt / 0.1)
            if a < 0.5:
                continue
            h, w = mk.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            shift = a * np.exp(-0.5 * ((yy - yc) / max(sig, 1)) ** 2)
            on = (xx > xm) if side == "right" else (xx < xm)
            mx = xx - np.where(on, shift if side == "right" else -shift, 0)
            mk = cv2.remap(mk, mx, yy, cv2.INTER_LINEAR, borderValue=0)
        return mk

    def _fx_state(self, m: Mark, t: float, pct: float):
        """(mask, albedo, wet) of a prop mark at time t, or None when it is gone."""
        f = self.fx[m.id]
        pc = self.ch.paint_clock
        page = pct - pc(m.t)
        mask = f["mask"]
        wash = lambda mk, k: mk * (1 - k * self._wash(*mk.shape))
        if f.get("paint_in") and page < f["paint_in"]:  # being painted in: a brush sweep, a third done on its key
            mask = self._paint_in(mask, 0.35 + 0.65 * max(page, 0.0) / f["paint_in"], m.id)
        if m.kind == "box" and f.get("bumps"):
            mask = self._bumped(mask, f["bumps"], t)
        if m.kind == "figure":  # someone else's old grey painting: faded and streaked by the rain, never wet
            o = f["old"]
            return self._washed(mask, 0.12 + 0.25 * o, f["wash"]) * (1 - 0.12 * o), 0.035 * self.ch.ink + 0.1 * o, 0.0
        if m.kind == "box":
            s = 1.0
            if t >= m.end - 1e-6:
                if f["by"] in ("shatter", "drop"):  # (a falling outline: gone as the next is painted)
                    return None
                s = 0.04 + 0.96 * math.exp(-(pct - pc(m.end)) / 0.35)  # painted over: a ghost that washes out
            n, L, seed = f["drips"]
            o = f.get("old", 0.0)
            if o > 0:  # painted long ago: its drips run their full length, dry, faded, rain-washed
                mk = add_drips(mask * s, np.random.default_rng(seed), n, 0.9 * s, L, 1.0, width=(1.4, 3.6))
                return self._washed(mk, 0.25 + 0.45 * o, f["wash"]) * (1 - 0.3 * o), 0.72 - 0.2 * o, 0.0
            mk = add_drips(mask * s, np.random.default_rng(seed), n, 0.9 * s, L, min(1.0, 0.15 + page / 1.2),
                           width=(1.4, 3.6))
            if page > 2:
                mk = wash(mk, min(1.0, page / 40) * 0.25)
            if f.get("bright"):  # (the slammed box: bright primer at once, a little wet sheen)
                return mk, 0.8 - 0.06 * min(1.0, page / 30), max(0.0, 1 - page / 0.8) * 0.35
            return mk, 0.50 + 0.28 * min(1.0, page / 1.5) - 0.06 * min(1.0, page / 30), max(0.0, 1 - page / 1.4) * 0.8
        if m.kind == "remains" and f.get("swath"):  # a stroke's primer swath: bright wet white, staying
            return wash(mask * (0.9 + 0.1 * math.exp(-page / 12)), min(1.0, page / 25) * 0.2), 0.8, \
                max(0.0, 1 - page / 0.8) * 0.6
        if m.kind == "remains":
            return wash(mask * (0.5 + 0.5 * math.exp(-page / 10)), min(1.0, page / 20) * 0.35), 0.72, 0.0
        if m.kind in ("chip", "crack"):
            if t >= m.end - 1e-6:
                return None
            s = (0.55 if m.kind == "chip" else 0.95) * ((0.6 + 0.4 * math.exp(-page / 8)) if f.get("after") else 1)
            return mask * s, (0.62 if m.kind == "chip" else 0.015 * self.ch.ink), 0.0
        if m.kind == "stencil":
            fd = f.get("fade")
            if fd is not None and t >= fd[0] + fd[1] - 1e-6:  # faded out: gone
                return None
            base = mask
            if f.get("smear") is not None and t >= f["smear"][0]:  # dragged along by him running through it
                ts, dur, ln, rr, right = f["smear"]
                lv = min(8, int(math.ceil(8 * (t - ts) / dur)))  # grows over `over` 16ths, in 8 steps, then stays
                key = ("smear", m.id, lv)
                if key not in self._peel:
                    self._peel[key] = PR.stencil_smear(mask, ln, lv / 8, rr, f["seed"], right)
                    while len(self._peel) > 24:
                        self._peel.popitem(last=False)
                base = mask = self._peel[key]
            t_p = f.get("t_peel") if f.get("t_peel") is not None else f["t_sh"]
            if t >= t_p:  # cracking and peeling a little each beat (half as fast before the shatter, from the roll)
                if t < f["t_sh"]:
                    k = int((pct - pc(t_p)) / f["beat"] * 0.5)
                elif t_p < f["t_sh"]:
                    k = int((pc(f["t_sh"]) - pc(t_p)) / f["beat"] * 0.5 + (pct - pc(f["t_sh"])) / f["beat"])
                else:
                    k = int((pct - pc(f["t_sh"])) / f["beat"])
                key = (m.id, min(k, 40))
                if key not in self._peel:
                    self._peel[key] = PR.peel(f["hard"], mask, min(k, 40), f["seed"])
                    while len(self._peel) > 24:
                        self._peel.popitem(last=False)
                base = self._peel[key]
                if t >= f["t_sh"]:  # washing like the ghosts after the shatter
                    base = base * (0.55 + 0.45 * math.exp(-(pct - pc(f["t_sh"])) / 8))
            if t >= f["t_sh"] and f.get("fly") is not None:  # the letters that flew are gone
                base = base * (1 - f["fly"])
            n, L, seed = f["drips"]
            mk = add_drips(base, np.random.default_rng(seed), n, 0.85, L, min(1.0, 0.2 + page / 0.8), width=(0.6, 1.5))
            if fd is not None and t > fd[0]:  # fading out
                mk = mk * max(0.0, 1 - (t - fd[0]) / fd[1])
            if f.get("albedo") is not None:  # white letters: primer, drying a little paler
                return mk, float(f["albedo"]) * (1.0 - 0.06 * min(1.0, page / 2)), max(0.0, 1 - page / 0.5) * 0.5
            return mk, (0.014 + 0.012 * min(1.0, page / 2)) * self.ch.ink, max(0.0, 1 - page / 0.5) * 0.5
        if m.kind == "mist":  # a stencil's spray burst, settling
            if page >= 0.6:
                return None
            return mask * math.exp(-page / 0.2), (0.02 * self.ch.ink if f.get("albedo") is None else
                                                  float(f["albedo"])), 0.0
        if m.kind == "spatter":
            n, L, seed = f["drips"]
            mk = add_drips(mask, np.random.default_rng(seed), n, 0.85, L, min(1.0, 0.2 + page / 1.0), width=(1.0, 2.6))
            if page > 1:
                mk = wash(mk, min(1.0, page / 20) * 0.3)
            alb = f["albedo"] if f["albedo"] > 0.3 else f["albedo"] * self.ch.ink  # black spatter only
            return mk, alb, max(0.0, 1 - page / 0.8) * 0.6
        raise ValueError(m.kind)

    def _washed(self, mk: np.ndarray, k: float, seed: int) -> np.ndarray:
        """A mask streaked by the rain: the wash texture (vertical streaks) at an offset of its own."""
        h, w = mk.shape
        th, tw = self.wash_tex.shape
        if h > th or w > tw:
            return mk * (1 - k * 0.5)
        r = np.random.default_rng(seed)
        oy, ox = int(r.integers(0, th - h + 1)), int(r.integers(0, tw - w + 1))
        return mk * (1 - k * self.wash_tex[oy:oy + h, ox:ox + w])

    def _wash(self, h: int, w: int) -> np.ndarray:
        """The rain's wash texture over an h x w mask (tiled for a mark bigger than the texture: a long smear, a
        painted word)."""
        t = self.wash_tex
        if h > t.shape[0] or w > t.shape[1]:
            t = np.tile(t, (-(-h // t.shape[0]), -(-w // t.shape[1])))
        return t[:h, :w]

    def shard_layer(self, t: float, x0: int, y0: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and albedo of the shards in the air at song time t (they move on the paint clock)."""
        A = np.zeros((h, w), np.float32)
        Al = np.zeros((h, w), np.float32)
        pct = self.ch.paint_clock(t)
        for t0, pc0, shards in self.shards:
            if t < t0 - 1e-6:
                continue
            tau = pct - pc0 + 0.05
            for s in shards:
                if tau >= s.land:
                    continue
                cx, cy, rot = PR.shard_pos(s, tau, self.u)
                sh, sw = s.mask.shape
                if cx + sw < x0 or cx - sw > x0 + w or cy + sh < y0 or cy - sh > y0 + h:
                    continue
                M = cv2.getRotationMatrix2D((sw / 2, sh / 2), rot, 1.0)
                size = int(math.ceil(math.hypot(sw, sh))) + 2
                M[0, 2] += (size - sw) / 2
                M[1, 2] += (size - sh) / 2
                d = cv2.warpAffine(s.mask, M, (size, size), flags=cv2.INTER_LINEAR)
                ax, ay = int(round(cx - size / 2)) - x0, int(round(cy - size / 2)) - y0
                sx0, sy0, dx0, dy0 = max(0, -ax), max(0, -ay), max(0, ax), max(0, ay)
                cw, chh = min(size - sx0, w - dx0), min(size - sy0, h - dy0)
                if cw <= 0 or chh <= 0:
                    continue
                a = np.clip(d[sy0:sy0 + chh, sx0:sx0 + cw] * 1.1, 0, 1)
                A[dy0:dy0 + chh, dx0:dx0 + cw] = 1 - (1 - A[dy0:dy0 + chh, dx0:dx0 + cw]) * (1 - a)
                Al[dy0:dy0 + chh, dx0:dx0 + cw] = Al[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - a) + s.albedo * a
        return A, Al

    # -------------------------------------------------------------- masks on demand
    def mask(self, m: Mark) -> np.ndarray:
        if m.id in self._cache:
            self._cache.move_to_end(m.id)
            return self._cache[m.id]
        g = m.gen
        if g[0] == "patch":
            _, seed, w, h, vertical, gain, nd, L = g
            r = np.random.default_rng(seed)
            pm = rough_patch(w, h, r, vertical=vertical) * gain
            out = add_drips(pm, r, nd, 0.8 if m.kind == "primer" else 0.7, L, 1.0)
        elif g[0] == "ev":
            d, _, _ = self.place(g[1])
            out = d * (g[3] if len(g) > 3 else 1.0)
        elif g[0] == "out":
            # the figure painted over with primer: its silhouette grown a little, rough brush texture
            src = self.marks[g[1]]
            d = self.mask(src)
            r = np.random.default_rng(g[2])
            sil = cv2.dilate((d > 0.12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            sil = np.pad(sil.astype(np.float32), 8)
            sil = cv2.GaussianBlur(sil, (0, 0), 2.5)
            hh, ww = sil.shape
            tex = np.clip(0.78 + 0.22 * noise(hh, ww, 3, r, aniso=(1, 9)) + 0.1 * noise(hh, ww, 14, r), 0, 1)
            out = np.clip(sil * 1.3, 0, 1) * tex
        else:
            raise ValueError(g)
        if m.clip is not None and m.kind in ("primer", "paintout"):  # (v8c) a move's primer_clip: none outside it
            out = out * _clip_ramp(out.shape, m.clip, m.x0, m.y0, m.id, self.u)
        self._cache[m.id] = out.astype(np.float32)
        while len(self._cache) > 400:  # a photo sees about 100 marks; this keeps a process under about 1 GB
            self._cache.popitem(last=False)
        return self._cache[m.id]

    def _skin_states(self) -> dict[int, tuple[bool, float]]:
        """Which wall poses in the world are skins (sky or a void behind most of him), photo by photo in time order,
        with hysteresis: they swap between paint and skin only where the new background holds SKIN_HOLD photos, and
        the primer patches of the SKIN_FADE paint photos next to a skin stretch are cross-faded (a third, two thirds).
        {event index: (skin, patch strength)}; empty without a world."""
        sc = self.ch.scene
        if sc.world is None:
            return {}
        cand = []
        for i, e in enumerate(self.events):
            if not (e.layer == "wall" or e.flags.get("skin_auto")) or e.pose is None:
                continue
            if e.x < sc.xw and not (sc.bands and e.y < 1000 * self.u):
                continue  # (the plain street left of the world is never a skin; above it, by its cells, it may be)
            m = self.lib.m(e.pose)
            if m["kind"] == "stamp" or TL.is_true(self.ch, e):  # (a true figure is never a J2 skin)
                continue
            self._placed[i] = (m["scale"] * e.scale, 0.0, 0.0)  # its place without the jitter (no random draws)
            d, x0, y0 = self.place(i)
            del self._placed[i]
            if e.x < sc.xw and y0 >= 0:
                continue  # (on the street's own rows: the plain street)
            cand.append((e.q, i, self._over_void(e, d, x0, y0)))
        cand.sort()
        states, fades = skin_hysteresis([q for q, _, _ in cand], [r for _, _, r in cand])
        out = {}
        for (q, i, _), st, fd in zip(cand, states, fades):
            out[i] = (st, fd)
            if not st and self.events[i].flags.get("skin_auto"):  # a skin from an earlier pass: on the wall again
                e = self.events[i]
                e.layer = "wall"
                e.flags = {k: v for k, v in e.flags.items() if k not in ("skin", "skin_auto")}
        return out

    def _over_void(self, e: Ev, d: np.ndarray, x0: float, y0: float) -> bool:
        """A wall pose with sky or a void behind most of him (in the world only; a take-off photo leaves the wall its
        own way): more than SKIN_VOID of his paint over the scene's voids (voids, gaps, invalid pixels)."""
        sc = self.ch.scene
        if sc.world is None or e.flags.get("takeoff") or (e.x < sc.xw and y0 >= 0):
            return False  # (left of the world, only a pose reaching above the street, by the cells there)
        X0, Y0 = int(math.floor(x0)), int(math.floor(y0))
        h, w = d.shape
        m = d > 0.2
        if not m.any():
            return False
        v = sc.void(X0, X0 + w, Y0, Y0 + h)
        return float((v[m] > 0.5).mean()) > SKIN_VOID

    def held_skins(self, q: float) -> list:
        """The skins still showing on photo q: the latest figure before it was a skin and nothing new is painted on q
        (a held pose, a slow-down's repeated frame); a wall pose or a flight after it ends the hold."""
        if not hasattr(self, "_fig_q"):
            self._fig_q = sorted(self.fig_box)
        k = bisect.bisect_right(self._fig_q, q + 1e-6) - 1
        if k < 0 or abs(self._fig_q[k] - q) < 1e-6:
            return []
        return [it for it in self.air.get(self._fig_q[k], []) if it.kind == "sil" and it.flags.get("skin")]

    def skin_shadow(self, out: np.ndarray, items: list, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """The skins' soft shadow on the wall behind them (down and right of him, only on wall: never on sky)."""
        u = self.u
        S = np.zeros((h, w), np.float32)
        dx, dy = int(round(SKIN_SHADOW[0] * u)), int(round(SKIN_SHADOW[1] * u))
        for it in items:
            a = np.clip(self.air_mask(it) * 2.0, 0, 1)
            ax0, ay0 = int(round(it.x0)) - x0 + dx, int(round(it.y0)) - y0 + dy
            sx0, sy0, ddx, ddy = max(0, -ax0), max(0, -ay0), max(0, ax0), max(0, ay0)
            cw, chh = min(a.shape[1] - sx0, w - ddx), min(a.shape[0] - sy0, h - ddy)
            if cw > 0 and chh > 0:
                np.maximum(S[ddy:ddy + chh, ddx:ddx + cw], a[sy0:sy0 + chh, sx0:sx0 + cw],
                           out=S[ddy:ddy + chh, ddx:ddx + cw])
        if not S.any():
            return out
        S = cv2.GaussianBlur(S, (0, 0), SKIN_SHADOW[2] * u)
        wall = 1 - self.ch.scene.void(x0, x0 + w, y0, y0 + h)
        return out * (1 - SKIN_SHADOW[3] * S * wall)[..., None]

    def air_mask(self, it: AirItem) -> np.ndarray:
        g = it.gen
        if g[0] == "ev":
            return self.place(g[1])[0]
        _, i, c, burst, seed = g
        d = self.place(i)[0]
        front = d.copy()
        front[:, :c] = 0
        if burst == "J1":  # dissolve: solid at the edge, breaking up toward the front
            r = np.random.default_rng(seed)
            h, w = d.shape
            nz = noise(h, w, 6, r) * 0.35 + noise(h, w, 2, r) * 0.25
            ramp = np.clip((np.arange(w)[None, :] - c) / 120.0, 0, 1)
            front = front * (nz + 0.55 - 0.9 * ramp > 0)
        return front

    # -------------------------------------------------------------- the paint state
    def frames_since(self, m: Mark, t: float) -> int:
        return bisect.bisect_right(self.pose_times, t + 1e-6) - bisect.bisect_right(self.pose_times, m.t + 1e-6)

    def ghost_strength(self, m: Mark, t: float) -> float:
        n = self.frames_since(m, t)
        if m.kind == "splash":
            age = self.ch.paint_clock(t) - self.ch.paint_clock(m.t)
            return 0.62 + 0.38 * math.exp(-age / 1.5)
        if n == 0:
            return 1.0
        age = n * self.frame_dt
        light = 0.33 * math.exp(-age / 0.12)  # the light trail: repainted in one place, nothing piles up
        if m.light:
            return light
        own = 0.22 + 0.38 * math.exp(-age / 2.5) if m.keeper else 0.05 + 0.28 * math.exp(-age / 0.12)
        return own if m.trail_w >= 1.0 else m.trail_w * own + (1 - m.trail_w) * light  # (trail_in: fading in)

    def layers(self, t: float, x0: int, y0: int, w: int, h: int, glass=None):
        """Paint state at time t in the region: premultiplied albedo, coverage, wetness (prototype 3). glass: shop
        window glass (verse 2's screens), an h x w mask (0 to 1) or a list of (mask, song t its screens die): no
        primer on it (no patch under a pose, no paint-out, no cover), and a pose leaves no ghost there (only the pose
        of its own photo shows: a cut-out moving across); a mark painted after a window's screens died treats that
        glass as wall (primer, so the black figure reads on the dead glass)."""
        if glass is not None and not isinstance(glass, list):
            glass = [(glass, math.inf)]
        Pm = np.zeros((h, w), np.float32)
        Cv = np.zeros((h, w), np.float32)
        Wt = np.zeros((h, w), np.float32)
        sel = np.nonzero((self._ot <= t + 1e-6) & (self._ox1 >= x0) & (self._ox0 <= x0 + w))[0]
        pc = self.ch.paint_clock
        pct = pc(t)
        fade = self.fade
        for oi in sel:
            m = self.order[oi]
            if m.y0 > y0 + h or m.y0 + m.h + 120 < y0:
                continue
            touch = None
            if m.kind in FX_KINDS:
                r = self._fx_state(m, t, pct)
                if r is None:
                    continue
                mask, albedo, wet = r
            else:
                mask = self.mask(m)
                page = pct - pc(m.t)
            if m.kind in FX_KINDS:
                pass
            elif m.kind in ("pose", "splash"):
                s = self.ghost_strength(m, t)
                n, L, seed = m.drips
                rv = self.reveal.get(m.id)
                if rv is not None:  # painted stroke by stroke: what is on the wall so far, each stroke wet from its own
                    mask, page = self._revealed(rv, t, pct, n, L, seed)  # time, its drips running from it
                    if mask is None:
                        continue
                    n = 0
                mask = mask * s
                grow = min(1.0, 0.25 + page / 0.5) if m.kind == "pose" else min(1.0, 0.15 + page / 1.2)
                mask = add_drips(mask, np.random.default_rng(seed), n, 0.9 * s, L, grow)
                if page > 0.3 and not m.nowash:
                    wash = self._wash(*mask.shape)
                    mask = mask * (1 - min(1.0, page / 3.0) * 0.45 * wash)
                albedo = (0.018 if page < 0.12 else (0.026 if page < 0.4 else 0.04)) * self.ch.ink * m.ink
                wet = max(0.0, 1 - page / 0.6)
                if m.kind == "pose" and self.frames_since(m, t) == 0 and rv is None:
                    if m.halo:  # punched into the splash: a crisp primed outline
                        halo = cv2.GaussianBlur(cv2.dilate((mask > 0.25).astype(np.float32), np.ones((11, 11), np.uint8)),
                                                (0, 0), 1.8) * 0.85
                    else:
                        halo = cv2.GaussianBlur(cv2.dilate((mask > 0.2).astype(np.float32), np.ones((9, 9), np.uint8)),
                                                (0, 0), 4) * 0.35
                    touch = (halo, 0.7)
            elif m.kind == "paintout":
                n = self.frames_since(m, t)  # painted frames since the paint-out (it went up with the next pose)
                fl = fade["out_other"]  # in-between poses: a light paint-out that washes away
                s = fade["out_other_start"] * (fl + (1 - fl) * math.exp(-n * self.frame_dt / fade["out_tau"]))
                if m.keeper:  # beats, landings, grabs: a pale figure that stays
                    sk = fade["out"] * (fade["out_keeper"] + (1 - fade["out_keeper"]) *
                                        math.exp(-n * self.frame_dt / fade["out_tau"]))
                    s = sk if m.trail_w >= 1.0 else m.trail_w * sk + (1 - m.trail_w) * s  # (trail_in: fading in)
                mask = mask * s
                albedo = m.albedo - 0.12 * min(1.0, page / 4.0)
                wet = max(0.0, 1 - page / 0.8) * 0.6
                if page > 0.5:
                    wash = self._wash(*mask.shape)
                    mask = mask * (1 - min(1.0, page / 4.0) * 0.3 * wash)
            else:
                albedo = m.albedo - 0.12 * min(1.0, page / 4.0)
                wet = max(0.0, 1 - page / 0.8) * 0.6
                if m.kind == "primer" and not m.keeper:
                    n = self.frames_since(m, t) - 1  # 16ths since its own figure was painted
                    if n > 0:
                        mask = mask * (fade["floor"] + (1 - fade["floor"]) * math.exp(-n * self.frame_dt / fade["tau"]))
                if page > 0.5:
                    wash = self._wash(*mask.shape)
                    mask = mask * (1 - min(1.0, page / 4.0) * 0.3 * wash)
            mh, mw = mask.shape
            ax0, ay0 = int(round(m.x0)) - x0, int(round(m.y0)) - y0
            sx0, sy0 = max(0, -ax0), max(0, -ay0)
            dx0, dy0 = max(0, ax0), max(0, ay0)
            cw, chh = min(mw - sx0, w - dx0), min(mh - sy0, h - dy0)
            if cw <= 0 or chh <= 0:
                continue
            gs = None
            if glass:
                parts = [G[dy0:dy0 + chh, dx0:dx0 + cw] for G, t_off in glass if m.t < t_off]
                gs = np.minimum(sum(parts), 1.0) if parts else None
                if gs is not None and not gs.any():
                    gs = None
            if touch is not None:
                th = touch[0][sy0:sy0 + chh, sx0:sx0 + cw]
                if gs is not None:
                    th = th * (1 - gs)
                Pm[dy0:dy0 + chh, dx0:dx0 + cw] = Pm[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - th) + touch[1] * th
                Cv[dy0:dy0 + chh, dx0:dx0 + cw] = Cv[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - th) + th
            a = mask[sy0:sy0 + chh, sx0:sx0 + cw]
            if gs is not None and (m.kind in ("primer", "paintout", "cover") or
                                   (m.kind == "pose" and self.frames_since(m, t) > 0)):
                a = a * (1 - gs)
            Pm[dy0:dy0 + chh, dx0:dx0 + cw] = Pm[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - a) + albedo * a
            Cv[dy0:dy0 + chh, dx0:dx0 + cw] = Cv[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - a) + a
            Wt[dy0:dy0 + chh, dx0:dx0 + cw] = Wt[dy0:dy0 + chh, dx0:dx0 + cw] * (1 - a) + wet * a
        return Pm, Cv, Wt

    def air_layer(self, t: float, q: float, x0: int, y0: int, w: int, h: int):
        """Alpha and linear colour of airborne paint: flying silhouettes, spatter, the J1 smear."""
        A = np.zeros((h, w), np.float32)
        rim = np.zeros((h, w), np.float32)
        fold = np.zeros((h, w), np.float32)
        pale = np.zeros((h, w), np.float32)  # skins' pale rim (the city's glow catching the paint's edge)
        items = self.air.get(q, [])
        glow = np.float32(items[0].glow if items and items[0].kind == "sil" else (1.0, 0.55, 0.22))
        # a ghost trail in the air (a flight's `ghosts`: the skins of the photos before it as pale primer
        # silhouettes, like the trail on the wall, fading over `ghosts` 16ths and gone after twice that)
        G = np.zeros((h, w), np.float32)
        shown = [(it, 1.0) for it in items if it.kind == "sil"]
        if not shown:  # a skin held: no new figure on this photo, the last one shown was a skin (it stays, as paint does)
            shown = [(it, 1.0) for it in self.held_skins(q)]
        for qq in self._air_q[max(0, bisect.bisect_left(self._air_q, q - 40) - 1):bisect.bisect_left(self._air_q, q)]:
            for it in self.air[qq]:
                n = float(it.flags.get("ghosts", 0.0))
                if it.kind != "sil" or n <= 0 or q <= qq:
                    continue
                if SL.trail_owns(self, q, it):  # skinlook draws this flight's echoes (4 soft smears, 35% to 8%)
                    continue
                hang = it.flags.get("hang")  # (landing 16th, 16ths): the whole arc hangs there after the landing
                if hang and q > hang[0] + 1e-6:
                    if q <= hang[0] + hang[1] + 1e-6:
                        shown.insert(0, (it, 0.3))
                    elif q <= hang[0] + hang[1] + 2 * n + 1e-6:
                        shown.insert(0, (it, 0.3 * math.exp(-(q - hang[0] - hang[1]) / n)))
                elif q - qq <= 2 * n + 1e-6:
                    shown.insert(0, (it, 0.42 * math.exp(-(q - qq - 1) / n)))
        for it, ga in shown:
            raw = self.air_mask(it)
            if ga < 1.0:
                d = cv2.GaussianBlur(np.clip(raw * 2.0, 0, 1), (0, 0), 1.5) * ga
                ax0, ay0 = int(round(it.x0)) - x0, int(round(it.y0)) - y0
                sx0, sy0, dx0, dy0 = max(0, -ax0), max(0, -ay0), max(0, ax0), max(0, ay0)
                cw, chh = min(d.shape[1] - sx0, w - dx0), min(d.shape[0] - sy0, h - dy0)
                if cw > 0 and chh > 0:
                    G[dy0:dy0 + chh, dx0:dx0 + cw] = 1 - (1 - G[dy0:dy0 + chh, dx0:dx0 + cw]) * (
                        1 - d[sy0:sy0 + chh, sx0:sx0 + cw])
                continue
            d = np.clip(raw * 2.4, 0, 1)
            d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            d = cv2.filter2D(d, -1, np.ones((1, 9), np.float32) / 9)
            e = np.clip(d - cv2.erode(d, np.ones((5, 5), np.uint8)), 0, 1) * (d < 0.97)
            ax0, ay0 = int(round(it.x0)) - x0, int(round(it.y0)) - y0
            mh, mw = d.shape
            sx0, sy0 = max(0, -ax0), max(0, -ay0)
            dx0, dy0 = max(0, ax0), max(0, ay0)
            cw, chh = min(mw - sx0, w - dx0), min(mh - sy0, h - dy0)
            if cw <= 0 or chh <= 0:
                continue
            a = d[sy0:sy0 + chh, sx0:sx0 + cw]
            A[dy0:dy0 + chh, dx0:dx0 + cw] = 1 - (1 - A[dy0:dy0 + chh, dx0:dx0 + cw]) * (1 - a)
            fl = np.clip(1 - raw[sy0:sy0 + chh, sx0:sx0 + cw] * 1.25, 0, 1) * a
            fold[dy0:dy0 + chh, dx0:dx0 + cw] = np.maximum(fold[dy0:dy0 + chh, dx0:dx0 + cw], fl)
            rim[dy0:dy0 + chh, dx0:dx0 + cw] = np.maximum(rim[dy0:dy0 + chh, dx0:dx0 + cw], e[sy0:sy0 + chh, sx0:sx0 + cw])
            if it.flags.get("skin"):  # a thin soft pale rim inside its edge, so he reads against a dark skyline
                k = max(3, int(round(SKIN_RIM[0] * self.u))) | 1
                cnts, _ = cv2.findContours((d > 0.3).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                solid = np.zeros(d.shape, np.uint8)
                cv2.drawContours(solid, cnts, -1, 1, -1)  # his outline only (no rims round the pale gaps inside)
                solid = solid.astype(np.float32)
                ring = np.clip(solid - cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), 0, 1)
                ring = cv2.GaussianBlur(ring, (0, 0), SKIN_RIM[1] * self.u) * d * SKIN_RIM[2]
                pale[dy0:dy0 + chh, dx0:dx0 + cw] = np.maximum(pale[dy0:dy0 + chh, dx0:dx0 + cw],
                                                               ring[sy0:sy0 + chh, sx0:sx0 + cw])
        # droplets in flight (spawned by take-offs, splashes and flying silhouettes), falling under gravity
        gr = 2500.0 * self.u  # 9.81 m/s2 at about 255 px per metre
        lay = np.zeros((h, w), np.float32)
        dr = self._drops
        if len(dr):
            i0 = int(np.searchsorted(dr[:, 0], self.ch.paint_clock_back(t, 0.6) - 1e-6))
            i1 = int(np.searchsorted(dr[:, 0], t + 1e-6, side="right"))
            frz = [(self.ch.grid.t(a), self.ch.grid.t(b) - 0.004) for a, b in self.ch.freezes]
            for ts, x, y, vx, vy, r in dr[i0:i1]:
                dt = t - ts
                for fa, fb in frz:  # droplets hang in the air while the song stands still
                    if fa < t and fb > ts:
                        dt -= max(0.0, min(t, fb) - max(ts, fa))
                px, py = x + vx * dt, y + vy * dt + 0.5 * gr * dt * dt
                if py > self.wall_base + self.ch.scene.ground_dy(px) + 78 * self.u:
                    continue
                ex, ey = vx * 0.012, (vy + gr * dt) * 0.012  # 1/80 s exposure streak
                p0 = (int(px - x0 - ex), int(py - y0 - ey))
                p1 = (int(px - x0), int(py - y0))
                if -50 < p1[0] < w + 50 and -50 < p1[1] < h + 50:
                    cv2.line(lay, p0, p1, 0.95, max(1, int(r * 0.9)), cv2.LINE_AA)
                    cv2.circle(lay, p1, max(1, int(r * 0.6)), 0.98, -1, cv2.LINE_AA)
        A = 1 - (1 - A) * (1 - np.clip(lay, 0, 1))
        for it in items:
            if it.kind == "smear":
                A = self._smear(A, it.flags, x0, y0, w, h)
        col = np.full((h, w, 3), 0.004, np.float32) + rim[..., None] * glow * 0.045 \
            + fold[..., None] * (0.016 + 0.01 * glow)
        if pale.any():
            col = col * (1 - pale[..., None]) + np.float32(SKIN_RIM[3]) * pale[..., None]
        if G.max() > 0:  # the pale ghosts behind the newest skin
            At = A + (1 - A) * G
            gcol = 0.05 + 0.035 * glow
            col = (col * A[..., None] + gcol * ((1 - A) * G)[..., None]) / np.maximum(At, 1e-4)[..., None]
            A = At
        A, col = SL.apply(self, q, A, col, x0, y0, w, h)  # inside a skin_look stretch: the skins opaque (skinlook.py)
        return A, col

    def _smear(self, A, fl, x0, y0, w, h):
        """J1's whip photo: a spray smear over the gap and the body streaked along the jump."""
        u = float(fl.get("u", 1.0))
        sm = self.lib.sprite(fl["stamp"])
        ys, xs = np.nonzero(sm > 0.35)
        sm = sm[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]
        sw, sh = int(1150 * u), int(300 * u)
        sm = cv2.resize(sm, (sw, sh), interpolation=cv2.INTER_AREA)
        sm = cv2.filter2D(sm, -1, np.ones((1, 31), np.float32) / 31)
        near, far = fl["gap"]
        X0, Y0 = int(near - 60 * u), int(fl["y"] - 496 * u)
        body = self.lib.sprite(fl["body"])
        bs = self.lib.m(fl["body"])["scale"]
        body = cv2.resize(body, (int(body.shape[1] * bs * 1.25), int(body.shape[0] * bs)), interpolation=cv2.INTER_AREA)
        body = np.pad(body, ((0, 0), (int(160 * u), int(160 * u))))
        k = int(241 * u) | 1
        body = cv2.filter2D(body, -1, np.ones((1, k), np.float32) / k) * 2.2
        bx0, by0 = int(near + 370 * u - body.shape[1] / 2) - x0, int(fl["y"] - 556 * u) - y0
        for arr, ox, oy, cap, gain in ((body, bx0, by0, 0.9, 1.0), (sm, X0 - x0, Y0 - y0, 0.97, 1.3)):
            bh, bw = arr.shape
            sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
            cw, chh = min(bw - sx0, w - dx0), min(bh - sy0, h - dy0)
            if cw > 0 and chh > 0:
                a = np.clip(arr[sy0:sy0 + chh, sx0:sx0 + cw] * gain, 0, cap)
                A[dy0:dy0 + chh, dx0:dx0 + cw] = 1 - (1 - A[dy0:dy0 + chh, dx0:dx0 + cw]) * (1 - a)
        return A

    def figure_boxes(self) -> dict:
        return dict(self.fig_box)


# ---------------------------------------------------------------------------------------------- one photograph

class Renderer:
    """Renders photos of one choreography; keeps one street span (maps) and moves it as the camera travels."""

    def __init__(self, ch: Choreo, hist: History | None = None):
        self.ch = ch
        self.sc = ch.scene
        self.hist = hist or History(ch)
        self.span = None
        from .screens import Screens
        self.screens = Screens(self.sc.d, self.sc.spec.get("screens"))  # verse 2's windows switched off
        from .windows import Windows
        self.windows = Windows(self.sc.d, ch)  # the final chorus: the world's windows lit along his path (windows.py)
        from .doors import Doors
        self.doors = Doors(self.sc.d, ch)  # v8b, the outro: the stair house's door opening and shutting (doors.py)
        from .screenlevels import ScreenLights
        self.slights = ScreenLights(ch, self.hist)  # v8 verse 2: each screen at its level on cues (screenlevels.py)

    def _glass(self, x0: int, y0: int, w: int, h: int, per: bool = False):
        """Shop window glass (obstacles.json `glass`) over the street box, softened a little; None without any.
        per=True: a list of (mask, song t its screens die: the choreography's screens_off for its id, else never)."""
        hits = [g for g in self.sc.spec.get("glass") or []
                if g["box"][2] > x0 and g["box"][0] < x0 + w and g["box"][3] > y0 and g["box"][1] < y0 + h]
        if not hits:
            return None
        out = []
        for gl in hits:
            a0, b0, a1, b1 = (int(round(v)) for v in gl["box"])
            G = np.zeros((h, w), np.float32)
            G[max(b0 - y0, 0):max(b1 - y0, 0), max(a0 - x0, 0):max(a1 - x0, 0)] = 1
            off = self.ch.screens_off.get(str(gl.get("id")))
            out.append((cv2.GaussianBlur(G, (0, 0), 1.5), self.ch.grid.t(off[0]) if off else math.inf))
        if per:
            return out
        return np.minimum(sum(G for G, _ in out), 1.0)

    def _span_for(self, x0: int, x1: int, y0: int = 0, y1: int | None = None):
        y1 = self.sc.H if y1 is None else y1
        if self.span is None or not self.span.covers(x0, x1, y0, y1):
            ya, yb = self.sc.span_rows(y0, y1)  # None, None: the street's rows 0 to H
            a = max(x0 - 256, self.sc.bounds[0] - 64)
            b = max(x1 + 256, a + 4608)
            self.span = self.sc.span(a, b, ya, yb)
        return self.span

    def photo(self, k: int, t: float, q: float, cam: Cam, size: tuple[int, int]) -> np.ndarray:
        """One photograph (sRGB float 0..1, size (w, h))."""
        ins = TL.insert_photo(self, k, t, q, cam, size)
        if ins is not None:  # he steps out of the paint (truelook.py): the full-frame insert, the photo's rain over it
            return ins
        ch, sc, hist = self.ch, self.sc, self.hist
        OUT_W, OUT_H = size
        cx, cy, cw, chh, rot = cam.cx, cam.cy, cam.cw, cam.ch, cam.rot
        ca, sa = abs(math.cos(math.radians(rot))), abs(math.sin(math.radians(rot)))
        hx, hy = (cw * ca + chh * sa) / 2 + 6, (cw * sa + chh * ca) / 2 + 6
        x0, y0 = int(cx - hx), max(int(cy - hy), sc.Y0)  # the street's rows 0 to H, or the world's
        x1, y1 = int(cx + hx) + 1, min(int(cy + hy) + 1, sc.Y1)
        w, h = x1 - x0, y1 - y0
        sp = self._span_for(x0, x1, y0, y1)
        sx = slice(x0 - sp.x0, x1 - sp.x0)
        sy = slice(y0 - sp.y0, y1 - sp.y0)  # the span's rows start at sp.y0 (0 on the street)
        seed = ch.seed
        rng = np.random.default_rng(1000 * seed + k)
        rng_w = np.random.default_rng(77 * seed + k)
        gdy = int(round(sc.ground_dy(cx)))  # the ground line here (plates are not level)
        wall_base, kerb_top, kerb_bot = sc.wall_base + gdy, sc.kerb_top + gdy, sc.kerb_bot + gdy
        base, E = sp.lin[sy, sx].copy(), sp.e_wall[sy, sx]
        dark = None
        if ch.screens_off:  # verse 2: the windows whose screens are off (their bands from the off plates)
            base, E, dark = self.screens.blend(base, E.copy(), x0, y0, ch.screens_off, q)
        if self.windows.active:  # the final chorus: windows lit along his path, a beat at a time (windows.py)
            base, E = self.windows.blend(base, E.copy(), x0, y0, q)
        base = WIN.flicker(ch, base, x0, y0, q, k)  # the city's lights flickering faintly in a `flicker` stretch
        if self.doors.active:  # v8b: a door open (the open twin through the doorway, the leaf at its angle, the spill)
            base, E = self.doors.blend(base, E.copy(), x0, y0, t)
        if self.slights.active:  # v8: the screens at their levels; the paint round them lit as the wall now is
            b0 = base
            base = self.slights.apply(base, x0, y0, t)
            if base is not b0:
                from .scene import luminance
                r_ = cv2.GaussianBlur((luminance(base) + 0.004) / (luminance(b0) + 0.004), (0, 0), 6)
                E = E * np.clip(r_, 0.15, 1.6)
        lin = ripple_warp(base, wall_base + 4 - y0, rng_w, amp=2.0)
        # wall paint (behind the real objects, never in the gap; on shop window glass a flat cut-out)
        Gw = self._glass(x0, y0, w, h, per=True)
        Pm, Cv, Wt = hist.layers(t, x0, y0, w, h, glass=Gw)
        G = None  # lit glass (the screens on behind it): a flat solid cut-out; dead glass paints as wall
        if Gw:
            G = np.minimum(sum(M for M, _ in Gw), 1.0) * (1.0 if dark is None else (1 - dark))
            if not G.any():
                G = None
        m = sp.paint[sy, sx] * (1 - sp.occ[sy, sx])
        Cv, Pm = Cv * m, Pm * m
        alb = Pm / np.maximum(Cv, 1e-4)
        tint = sp.tint[sy, sx]
        for L in sc.lights:  # a light's colour taken out of the paint round it (`neutral`): white primer stays white
            k_n = light_opt(L, "neutral")
            if k_n <= 0 or abs(L["x"] - (x0 + x1) / 2) > w / 2 + 4 * L.get("sigma", 400):
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            gn = k_n * np.exp(-((xx - L["x"]) ** 2 + (yy - L["y"]) ** 2) / (2 * (1.6 * L.get("sigma", 420.0)) ** 2))
            grey = tint.mean(axis=2, keepdims=True)
            tint = tint + (grey - tint) * np.clip(gn, 0, 1)[..., None]
        relief, sheen_ = sp.relief[sy, sx], sp.spec[sy, sx]
        if dark is not None and dark.any():  # where the screens died the street's texture maps (from the plate with
            # them on: the faces) do not apply: paint there flat, in a neutral light
            relief, sheen_ = relief * (1 - dark) + dark, sheen_ * (1 - dark[..., None])
            tint = tint + (tint.mean(axis=2, keepdims=True) - tint) * dark[..., None]
        if G is not None:  # on the glass: flat black, no wall texture, no sheen
            relief, sheen_ = relief * (1 - G) + G, sheen_ * (1 - G[..., None])
        paint = (alb * E * relief)[..., None] * tint + sheen_ * (0.1 + 0.4 * Wt[..., None])
        dims = dict(ch.light_hints.get(round(q, 4), {}))  # a flicker: lights dimmed on this photo
        for lid, f_ in self.doors.lights(t).items():  # (a door's light: as far as it is open)
            dims[lid] = dims.get(lid, 1.0) * f_
        for lid, q_out in ch.lights_out.items():  # knocked out: dark from then on
            if q >= q_out - 1e-6:
                dims[lid] = 0.03
        for L in sc.lights:
            if abs(L["x"] - (x0 + x1) / 2) > w / 2 + 4 * L.get("sigma", 400):
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            g = L.get("strength", 0.6) * light_opt(L, "paint") * dims.get(L.get("id"), 1.0) * np.exp(
                -((xx - L["x"]) ** 2 + (yy - L["y"]) ** 2) / (2 * L.get("sigma", 420.0) ** 2))
            if G is not None:
                g = g * (1 - G)
            paint += (alb * g)[..., None] * np.float32(L.get("color", [1.0, 0.07, 0.05]))
        soft = cv2.GaussianBlur(Cv, (0, 0), 0.6)
        if G is not None:  # on lit glass the figure is a solid cut-out: the brush's dry gaps closed up
            soft = soft * (1 - G) + G * np.clip(soft * GLASS_SOLID, 0, 1)
        out = lin * (1 - soft[..., None]) + paint * soft[..., None]
        if G is not None:  # backlit by the screens behind the glass: a thin soft glow of their light round his edges
            sg = soft * G
            if sg.max() > 0.05:
                halo = np.clip(cv2.GaussianBlur(sg, (0, 0), GLASS_GLOW[0] * self.ch.u) * 1.6 - sg, 0, 1) * G * (1 - sg)
                out += (GLASS_GLOW[1] * halo)[..., None] * cv2.GaussianBlur(lin, (0, 0), GLASS_GLOW[2] * self.ch.u)
        # the wall paint in the wet pavement
        delta = out - lin
        pav0, pav1 = wall_base - y0, kerb_top - y0
        if 0 < pav0 < h:
            n = min(pav1, h) - pav0
            src_rows = np.clip(2 * pav0 - np.arange(pav0, pav0 + n) - 2, 0, h - 1)
            refl = cv2.GaussianBlur(delta[src_rows], (0, 0), sigmaX=2.0, sigmaY=5.0)
            att = (0.38 * np.exp(-np.arange(n) / 55.0))[:, None, None]
            refl = ripple_warp(refl, 0, rng, amp=3.0)
            sheen = np.clip(cv2.GaussianBlur(sp.lum[y0 - sp.y0 + pav0:y0 - sp.y0 + pav0 + n, sx], (0, 0), 6) * 9, 0.25,
                            1.0)[..., None]
            out[pav0:pav0 + n] += refl * att * sheen
        # shards of a shattered prop in the air, lit by the wall's light
        if hist.shards:
            As, Al = hist.shard_layer(t, x0, y0, w, h)
            if As.max() > 0:
                col = (Al * sp.e_wall[sy, sx] * 1.15)[..., None] * sp.tint[sy, sx] + 0.003
                out = out * (1 - As[..., None]) + col * As[..., None]
        rk = RV.layers(ch, t, x0, y0, w, h)  # the paint-in (reveal.py): what of the world is painted in yet
        # parallax layers (the red light pole on the kerb)
        for L in sp.layers:
            dx = -L["parallax"] * (cx - L["x"])
            a0, lw = L["a0"], L["alpha"].shape[1]
            if a0 + lw + abs(dx) < x0 or a0 - abs(dx) > x1:
                continue
            M = np.float32([[1, 0, dx + a0 - x0], [0, 1, 0]])
            lr = lambda a: rows_of(a, L.get("y0", 0), y0, y1)  # the layer's rows (0 to H on the street)
            pm = cv2.warpAffine(lr(L["alpha"]), M, (w, h), borderValue=0)
            pcol = cv2.warpAffine(lr(_fg_lin(L)), M, (w, h), borderValue=0)
            mine = [lt for lt in sc.lights if lt.get("layer") == L["id"]]
            f = min([dims.get(lt.get("id"), 1.0) for lt in mine], default=1.0)
            if any(q >= ch.lights_out.get(lt.get("id"), math.inf) - 1e-6 for lt in mine):
                # knocked out: every lit pixel of the lamp dark (its glowing figure too), a dead lamp
                spec_ = next((l_ for l_ in sc.layers if l_.get("id") == L["id"]), {})
                dead = cv2.warpAffine(lr(_lamp(L, every=True, parts=spec_.get("parts"))), M, (w, h),
                                      borderValue=0)
                pcol = pcol * (1 - dead)[..., None] + np.float32([0.006, 0.0045, 0.004]) * dead[..., None]  # a dark lens
                near = cv2.GaussianBlur(cv2.dilate(dead, np.ones((41, 41), np.uint8)), (0, 0), 8)
                rgb = out
                redness = np.clip((rgb[..., 0] - 1.3 * np.maximum(rgb[..., 1], rgb[..., 2])) / 0.02, 0, 1)
                out = out * (1 - 0.85 * near * redness)[..., None]  # its red bloom on the wall round it goes too
            elif f != 1.0:  # the lamp itself flickers: its lit (red) pixels dimmed
                red = cv2.warpAffine(lr(_lamp(L)), M, (w, h), borderValue=0)
                pcol = pcol * (1 - (1 - f) * red)[..., None]
            if rk is not None:  # (a lamp in the world not yet painted in is not there yet either)
                pm = pm * rk[0]
            out = out * (1 - pm[..., None]) + pcol * pm[..., None]
        if rk is not None:  # the world not yet painted in: primer and pencil; the strokes' wet paint
            out = RV.composite(ch, out, sp, rk[0], rk[1], t, x0, y0, sx, sy)
        if getattr(hist, "true_q", None):  # he is true (truelook.py): the solid figure over the plate and everything
            # on it (lamps and their blinks, lit windows, signs, screens); only rain, haze and grain go over him
            out = TL.composite(hist, out, lin, t, q, x0, y0, w, h)
        items = hist.air.get(q, [])
        skins = [it for it in items if it.kind == "sil" and it.flags.get("skin")] or \
            (hist.held_skins(q) if not any(it.kind == "sil" for it in items) else [])
        if skins:  # a skin over a void casts a soft shadow on the nearest wall
            out = hist.skin_shadow(out, skins, x0, y0, w, h)
        if any(it.kind == "smear" for it in items):  # the whip photo: the camera swung during the exposure
            out = cv2.filter2D(out, -1, np.ones((1, 71), np.float32) / 71, borderType=cv2.BORDER_REFLECT)
        A, col = hist.air_layer(t, q, x0, y0, w, h)
        A = self.doors.occlude(A, x0, y0, t)  # (inside a door: seen only through its doorway)
        if A.max() > 0:
            out = out * (1 - A[..., None]) + col * A[..., None]
        out = puddle_rings(out, wall_base + 10 - y0, rng, scale=cw / 1560 if OUT_W > OUT_H else chh / 1560)
        out = car_light(out, x0, y0, rng, t, KERB_BOT=kerb_bot)
        img = _to_out(out, cx - x0, cy - y0, cw, rot, OUT_W, OUT_H)
        img = rain(img, rng, strength=max(OUT_W, OUT_H) / 2048)
        return np.clip(img, 0, 1)


def fill_holes(d: np.ndarray, n: int) -> np.ndarray:
    """A matte with its interior holes (gaps the wall showed through, closed round by paint) up to n px filled with
    the density round them."""
    from scipy import ndimage as ndi

    m = d > 0.15
    holes = ndi.binary_fill_holes(m) & ~m
    lab, k = ndi.label(holes)
    if not k:
        return d
    small = np.isin(lab, 1 + np.nonzero(ndi.sum(holes, lab, range(1, k + 1)) <= n)[0])
    return np.where(small, cv2.dilate(d, np.ones((5, 5), np.uint8), iterations=4), d).astype(d.dtype)


def light_opt(L: dict, key: str) -> float:
    """A light's `paint` (its glow on his paint, 1) or `neutral` (its colour taken out of the paint round it, 0). A
    red light (red over three times its green and blue; a light with no colour is red) defaults to paint 0 and
    neutral 1 (the director's pink note on v5: white primer stays white by every red light in the film; their glow stays
    on the plates and lamps)."""
    if key in L:
        return float(L[key])
    r, g, b = (float(v) for v in (L.get("color") or [1.0, 0.07, 0.05])[:3])
    red = r > 3 * max(g, b)
    return (0.0 if red else 1.0) if key == "paint" else (1.0 if red else 0.0)


def _lamp(L: dict, every: bool = False, parts: list | None = None) -> np.ndarray:
    """Where a parallax layer's lamp is lit: its bright red and pink pixels (the red pedestrian light, its glowing
    figure), soft-edged; every: all its bright pixels, the white core of the glowing figure too, and inside the
    layer's `rect` parts (the lamp heads) everything brighter than the dark casing (a dead lamp: no figure left)."""
    k = "lamp_all" if every else "lamp"
    if L.get(k) is None:
        fg = L["fg"]
        r, gb = fg[..., 0], np.maximum(fg[..., 1], fg[..., 2])
        if every:  # any red-dominant pixel, however dim (the lamp's whole glow), and every bright one
            red = np.clip((r - 1.2 * gb - 0.02) / 0.05, 0, 1) * np.clip((r - 0.08) / 0.08, 0, 1)
            lit = np.maximum(red, np.clip((fg.max(2) - 0.45) / 0.2, 0, 1))
            heads = np.zeros_like(r)
            for p_ in parts or []:
                if p_.get("method") == "rect":
                    x0, y0, x1, y1 = (int(v) for v in p_["box"])
                    o = L.get("y0", 0)  # the layer's first row (0 on the street)
                    heads[max(y0 - o, 0):max(y1 - o, 0), max(x0 - L["a0"], 0):max(x1 - L["a0"], 0)] = 1.0
            lit = np.maximum(lit, heads * np.clip((fg.max(2) - 0.1) / 0.08, 0, 1)) * L["alpha"]
            lit = cv2.dilate(lit.astype(np.float32), np.ones((7, 7), np.uint8))
        else:
            lit = np.clip((r - 0.25) / 0.25, 0, 1) * np.clip((r - 1.15 * gb) / 0.1, 0, 1) * L["alpha"]
        L[k] = cv2.GaussianBlur(lit.astype(np.float32), (0, 0), 2.0)
    return L[k]


def _fg_lin(L: dict) -> np.ndarray:
    if "fg_lin" not in L or L["fg_lin"] is None:
        from .scene import srgb2lin
        L["fg_lin"] = srgb2lin(L["fg"])
    return L["fg_lin"]


def _to_out(lin_region, cxr, cyr, cw, rot, OUT_W, OUT_H):
    srgb = lin2srgb(lin_region)
    s = OUT_W / cw
    M = cv2.getRotationMatrix2D((float(cxr), float(cyr)), rot, s)
    M[0, 2] += OUT_W / 2 - cxr
    M[1, 2] += OUT_H / 2 - cyr
    return cv2.warpAffine(srgb, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


def look(img: np.ndarray, k: int, aspect: str, seed: int = 7, preview: bool = False) -> Image.Image:
    """The MUTO look for one photo (jitter, exposure and white-balance flicker, grain), seeded per photo so a photo
    looks the same whatever range it is rendered in."""
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    size = LOOK[aspect] if not preview else tuple(v // 2 for v in LOOK[aspect])
    return muto._photo(im, np.random.default_rng([seed, k]), (0.0, 0.0, 0.0), 1.0, size=size)


# ---------------------------------------------------------------------------------------------- jobs

@dataclass
class Job:
    k: int
    q: float
    t: float
    cams: dict  # aspect -> Cam


def plan_jobs(ch: Choreo, hist: History, aspects: list[str]) -> tuple[list, dict]:
    photos = ch.photos()
    boxes = hist.figure_boxes()
    cams = {a: plan(ch, photos, boxes, a) for a in aspects}
    jobs = [Job(p.k, p.q, p.t, {a: cams[a][n] for a in aspects}) for n, p in enumerate(photos)]
    return jobs, cams


_W: dict = {}


def _worker_init(proj_root: str, choreo_path: str, overrides: dict, debug: bool = False) -> None:
    cv2.setNumThreads(2)
    from ..project import Project
    from .choreo import open_choreo
    ch = open_choreo(Project.open(proj_root), choreo_path, overrides)
    _W["r"] = Renderer(ch)
    _W["debug"] = debug


def _render_chunk(args) -> list[tuple[int, float]]:
    jobs, aspects, out_dir, preview, keep_png = args
    r: Renderer = _W["r"]
    times = []
    for j in jobs:
        t0 = time.time()
        for a in aspects:
            size = SIZES[a] if not preview else tuple(v // 2 for v in SIZES[a])
            img = r.photo(j.k, j.t, j.q, j.cams[a], size)
            if _W.get("debug"):
                img = debug_overlay(r, img, j, a, size)
            d = Path(out_dir) / a.replace(":", "x")
            if keep_png:
                cv2.imwrite(str(d / f"raw-{j.k:05d}.png"), cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            look(img, j.k, a, preview=preview).save(d / f"{j.k:05d}.jpg", quality=93)
        times.append((j.k, time.time() - t0))
    return times


def render_photos(p, ch: Choreo, jobs: list[Job], aspects: list[str], out_dir: Path, preview: bool = False,
                  workers: int | None = None, keep_png: bool = False, overrides: dict | None = None,
                  log=print, debug: bool = False) -> dict:
    """Render the photos of `jobs` (in parallel worker processes, contiguous chunks) into out_dir/<aspect>/."""
    for a in aspects:
        (out_dir / a.replace(":", "x")).mkdir(parents=True, exist_ok=True)
    n = len(jobs)
    workers = workers or max(1, min(8, (os.cpu_count() or 4) // 2, n // 6 or 1))
    t0 = time.time()
    if workers == 1:
        _worker_init(str(p.root), str(ch.path), overrides or {}, debug)
        times = _render_chunk((jobs, aspects, str(out_dir), preview, keep_png))
    else:
        import multiprocessing as mp
        chunks = max(workers * 2, 1)
        size = math.ceil(n / chunks)
        parts = [jobs[i:i + size] for i in range(0, n, size)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(str(p.root), str(ch.path), overrides or {},
                                                                  debug)) as pool:
            times = []
            done = 0
            for res in pool.imap(_render_chunk, [(pt, aspects, str(out_dir), preview, keep_png) for pt in parts]):
                times.extend(res)
                done += len(res)
                log(f"  {done}/{n} photos, {time.time() - t0:.0f}s")
    wall = time.time() - t0
    per = [s for _, s in times]
    return {"photos": n, "wall_s": round(wall, 1), "per_photo_s": round(float(np.mean(per)) if per else 0, 3),
            "per_photo_wall_s": round(wall / max(n, 1), 3), "workers": workers}


def encode(p, ch: Choreo, jobs: list[Job], src: Path, out: Path, t0: float, t1: float, aspect: str,
           crf: int = 17) -> Path:
    """24 fps video from the photos (each held from its own song time) with the song slice."""
    fps = muto.FPS
    ts = np.array([j.t for j in jobs])
    ks = [j.k for j in jobs]
    with tempfile.TemporaryDirectory(prefix="seq-", dir=src.parent) as tmp:  # same volume: hard links
        tmp = Path(tmp)
        n = int(round((t1 - t0) * fps))
        for i in range(n):
            t = t0 + i / fps
            j = max(0, int(np.searchsorted(ts, t + 1e-6, side="right")) - 1)
            f = src / f"{ks[j]:05d}.jpg"
            os.link(f, tmp / f"{i:05d}.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.jpg"),
                        "-ss", f"{t0:.4f}", "-t", f"{t1 - t0:.4f}", "-i", str(p.bd / "audio/mix.wav"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                        "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)], check=True)
    return out


def peek(ch: Choreo, jobs: list[Job], work: Path, aspects: list[str], out: Path) -> Path:
    """A labelled grid of rendered photos (photo index, grid address, song time): the --only iteration loop. One row
    per photo when both formats are rendered."""
    from PIL import ImageDraw

    cols = len(aspects) if len(aspects) > 1 else 3
    tiles = []
    for j in jobs:
        for a in aspects:
            im = Image.open(work / a.replace(":", "x") / f"{j.k:05d}.jpg")
            im.thumbnail((800, 800))
            ImageDraw.Draw(im).rectangle([0, 0, 260, 22], fill=(0, 0, 0))
            ImageDraw.Draw(im).text((6, 5), f"k{j.k}  {ch.grid.addr(j.q)}  {j.t:.3f}s  {a}", fill=(255, 255, 255))
            tiles.append(im)
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (16, 16, 18))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=88)
    return out


def select(ch: Choreo, jobs: list[Job], t0: float | None, t1: float | None, only: list[int] | None) -> list[Job]:
    if only is not None:
        return [j for j in jobs if j.k in set(only)]
    t0 = ch.window[0] if t0 is None and ch.window else (t0 if t0 is not None else jobs[0].t)
    t1 = ch.window[1] if t1 is None and ch.window else (t1 if t1 is not None else jobs[-1].t + 1)
    first = max([j.k for j in jobs if j.t <= t0 + 1e-6], default=jobs[0].k)
    return [j for j in jobs if j.k >= first and j.t < t1 - 1e-6]


def render(p, ch: Choreo, out_dir: Path, t0: float | None = None, t1: float | None = None,
           aspects: tuple[str, ...] = ("16:9", "9:16"), preview: bool = False, only: list[int] | None = None,
           workers: int | None = None, video: bool = True, keep: bool = False, overrides: dict | None = None,
           name: str | None = None, review: bool = False, force: bool = False, log=print, debug: bool = False) -> dict:
    """`aimv paint render`: photos, look, videos, a steps.json log of every photo's time and framing, the QA
    (qa.py: <name>-qa.json, -qa.jpg, -moves.jpg) and, with review, small review copies (not when the QA has errors,
    unless force)."""
    hist = History(ch)
    jobs_all, cams = plan_jobs(ch, hist, list(aspects))
    jobs = select(ch, jobs_all, t0, t1, only)
    if not jobs:
        raise SystemExit("no photos in that range")
    t0 = t0 if t0 is not None else (ch.window[0] if ch.window else jobs[0].t)
    t1 = t1 if t1 is not None else (ch.window[1] if ch.window else jobs[-1].t + 0.1)
    name = name or f"{ch.name}-{t0:.2f}-{t1:.2f}" + ("-preview" if preview else "")
    if debug and not name.endswith("-debug"):
        name += "-debug"  # never in place of a clean render
    work = out_dir / f".work-{name}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log(f"render {ch.name}: {len(jobs)} photos ({jobs[0].t:.3f} to {jobs[-1].t:.3f} s), {', '.join(aspects)}"
        + (" preview" if preview else ""))
    stats = render_photos(p, ch, jobs, list(aspects), work, preview, workers, keep_png=keep, overrides=overrides,
                          log=log, debug=debug)
    outs = {}
    boxes = hist.figure_boxes()
    keys = sorted(boxes)
    fb = [boxes[keys[max(int(np.searchsorted(keys, j.q + 1e-6, side="right")) - 1, 0)]] for j in jobs]
    for a in aspects:
        tag = a.replace(":", "x")
        stats[f"framing_{tag}"] = framing_stats([j.cams[a] for j in jobs], fb, a == "9:16")
        if video and only is None:
            outs[a] = str(encode(p, ch, jobs, work / tag, out_dir / f"{name}-{tag}.mp4", t0, t1, a))
            log(f"  {outs[a]}")
    log_rows = [{"k": j.k, "t": round(j.t, 4), "q": j.q, "addr": ch.grid.addr(j.q),
                 **{f"cam_{a.replace(':', 'x')}": j.cams[a].as_list() for a in aspects}} for j in jobs]
    write_json(out_dir / f"{name}-steps.json", {"choreo": ch.name, "t0": t0, "t1": t1, "stats": stats,
                                                "photos": log_rows}, nd=4)
    if only is not None:
        stats["peek"] = str(peek(ch, jobs, work, list(aspects), out_dir / f"{name}-peek.jpg"))
    # QA of the rendered photos
    from . import qa as QA
    from .choreo import Photo
    sel = {j.k for j in jobs}
    photos = [ph for ph in ch.photos() if ph.k in sel]
    res = QA.run(ch, hist, photos, {a: [j.cams[a] for j in jobs] for a in aspects})
    res["flags"] = [f for f in res["flags"] if f["k"] in sel]
    res["errors"] = sum(f["level"] == "error" for f in res["flags"])
    res["warnings"] = len(res["flags"]) - res["errors"]
    sizes = {a: (SIZES[a] if not preview else tuple(v // 2 for v in SIZES[a])) for a in aspects}
    looks = {a: (LOOK[a] if not preview else tuple(v // 2 for v in LOOK[a])) for a in aspects}
    stats.update(QA.sheets(ch, res, jobs, work, name, out_dir, sizes, looks, list(aspects)))
    write_json(out_dir / f"{name}-qa.json", {"choreo": ch.name, "t0": t0, "t1": t1, **QA.public(res),
                                            "solver": solver_notes(ch, t0, t1)}, nd=3)
    stats["qa"] = {"errors": res["errors"], "warnings": res["warnings"], "json": str(out_dir / f"{name}-qa.json")}
    for line in QA.lines(res)[1:]:
        log(line)
    log(QA.banner(res, stats.get("qa_sheet", "")))
    if review and video and only is None:
        if res["errors"] and not force:
            log(f"no review copies: the QA has {res['errors']} errors (--force makes them anyway)")
        else:
            rev = {"16:9": ("1600:900", 27), "9:16": ("720:1280", 26)}
            for a, f in outs.items():
                sz, crf = rev[a]
                dst = out_dir / f"{name}-review-{a.replace(':', 'x')}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-vf", f"scale={sz}", "-c:v", "libx264", "-crf",
                                str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(dst)], check=True)
                stats.setdefault("review", []).append(str(dst))
    if keep or only is not None:
        stats["photos_dir"] = str(work)
    else:
        shutil.rmtree(work)
    stats["videos"] = outs
    return stats


# ---------------------------------------------------------------------------------------------- the physics shown

def solver_notes(ch: Choreo, t0: float, t1: float) -> dict:
    """What the parkour solver measured for its moves that overlap [t0, t1] (ch.notes: spans with their speed
    factors and slack, flights, landings): the QA json's `solver`."""
    out = {}
    for mv in ch.moves:
        if mv.kind != "parkour" or mv.where not in ch.notes or mv.state.q0 is None:
            continue
        a, b = ch.grid.t(mv.state.q0), ch.grid.t(mv.state.q1)
        if b >= t0 and a <= t1:
            out[mv.where] = ch.notes[mv.where]
    return out


def debug_overlay(r: "Renderer", img: np.ndarray, j, aspect: str, size: tuple[int, int]) -> np.ndarray:
    """`aimv paint render --debug`: the physics drawn on one photo (sRGB float, before the look): the feet lines
    (green) and the props' tops (cyan), the obstacle boxes (solid orange, others grey) and the gaps' edges
    (magenta), the solved arc of the flight he is in (white), his lowest paint against the surface under it (a dot,
    green within the feet rule's tolerance, red outside it), and a label: speed, gait, span factor and slack."""
    from . import qa as QA

    ch, sc, hist = r.ch, r.sc, r.hist
    cam = j.cams[aspect]
    W, H = size
    s = W / cam.cw
    ang = math.radians(cam.rot)
    ca, sa = math.cos(ang), math.sin(ang)
    out = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))  # (cv2 draws text on 8 bits)
    u = ch.u
    lw = max(1, int(round(2 * W / 2048)))

    def P(x, y):
        dx, dy = (x - cam.cx) * s, (y - cam.cy) * s
        return int(round(dx * ca + dy * sa + W / 2)), int(round(-dx * sa + dy * ca + H / 2))

    x0, x1 = cam.cx - cam.cw * 0.75, cam.cx + cam.cw * 0.75
    y0, y1 = cam.cy - cam.ch * 0.75, cam.cy + cam.ch * 0.75
    xs = np.linspace(x0, x1, 160)
    gaps = [o for o in sc.obstacles.values() if o.kind == "gap"]
    pts = [P(x, sc.feet_at(x)) for x in xs if not any(o.x0 < x < o.x1 for o in gaps)]
    for a_, b_ in zip(pts, pts[1:]):
        if abs(a_[0] - b_[0]) < W * 0.05:
            cv2.line(out, a_, b_, (51, 255, 76), lw, cv2.LINE_AA)
    for o in sc.obstacles.values():
        if o.x1 < x0 or o.x0 > x1 or o.y1 < y0 or o.y0 > y1:
            continue
        if o.kind == "gap":
            for ex in (o.x0, o.x1):
                cv2.line(out, P(ex, y0), P(ex, y1), (255, 51, 255), lw, cv2.LINE_AA)
            continue
        col = (255, 140, 26) if o.solid else (153, 153, 153)
        cv2.rectangle(out, P(o.x0, o.y0), P(o.x1, o.y1), col, lw, cv2.LINE_AA)
        cv2.putText(out, o.id, P(o.x0, o.y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * W / 2048 * 2, col, lw)
        k = o.contacts.get("top")
        if k is not None and k.y is not None and k.x0 is not None:
            cv2.line(out, P(k.x0, k.y), P(k.x1, k.y), (51, 230, 255), lw + 1, cv2.LINE_AA)
    info = (getattr(ch, "pk", None) or {}).get(round(j.q, 4), {})
    arc = info.get("arc")
    if arc:
        ap = [P(a_, b_) for a_, b_ in arc]
        for a_, b_ in zip(ap, ap[1:]):
            cv2.line(out, a_, b_, (255, 255, 255), lw, cv2.LINE_AA)
    # his lowest paint against the surface (the feet rule)
    shown = next((i for i in range(len(ch.events) - 1, -1, -1) if abs(ch.events[i].q - j.q) < 1e-6 and
                  ch.events[i].layer in ("wall", "air") and ch.events[i].pose is not None and
                  ch.lib.m(ch.events[i].pose)["kind"] == "pose"), None)
    label = []
    if shown is not None:
        e = ch.events[shown]
        d, fx0, fy0 = hist.place(shown)
        ys_, xs_ = np.nonzero(d > 0.4)
        if len(ys_):
            low = fy0 + float(np.percentile(ys_, 99.5))
            lowx = fx0 + float(np.median(xs_[ys_ >= np.percentile(ys_, 99.5) - 2]))
            surf = e.flags.get("surface")
            surf = float(surf) if surf is not None else sc.feet_at(lowx)
            if e.flags.get("ground") or info.get("foot"):
                ok = abs(low - surf) <= QA.FEET_TOL * u + (6 * u if e.jitter else 0)
                col = (51, 255, 76) if ok else (255, 38, 38)
                cv2.circle(out, P(lowx, low), max(4, 5 * lw), col, -1, cv2.LINE_AA)
                cv2.line(out, P(lowx - 30 * u, surf), P(lowx + 30 * u, surf), col, lw + 1, cv2.LINE_AA)
                label.append(f"feet {low - surf:+.0f}px")
        label.insert(0, f"{e.tag or e.pose}")
    if info:
        label.insert(0, f"{info.get('v_mps', 0):.1f} m/s {info.get('gait', '')}")
        if info.get("factor") is not None:
            label.append(f"span x{info['factor']:.3f}" + ("" if info.get("ok", True) else " OUT"))
        if info.get("rise_m") is not None:  # a flight: its rise, and how far his feet clear what he jumps
            label.append(f"rise {info['rise_m']:.2f} m" + (f", feet clear {info['clear_m']:+.2f} m"
                                                            if info.get("clear_m") is not None else ""))
    label.insert(0, f"{j.t:.3f}s {ch.grid.addr(j.q)}")
    fs = 0.6 * W / 1024 if W < H else 0.6 * W / 2048 * 1.6
    y = int(28 * fs / 0.6)
    for k_, line in enumerate(label):
        org = (12, y + k_ * int(26 * fs / 0.6))
        cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw + 3, cv2.LINE_AA)
        cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 102), lw, cv2.LINE_AA)
    return out.astype(np.float32) / 255
