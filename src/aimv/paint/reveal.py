"""Paint-in on the beat (v8, the treatment's option A): the world ahead of him does not exist yet. It
is pale primer with faint graphite pencil lines that trace the plate's main edges (so you can read what is coming), and
giant wet brush strokes sweep it into existence on the kicks and snares, a chunk ahead of him each time; what is
painted stays painted.

    paint_in: {from: 101.5s, to: 115s, front: 34700,          the world right of x 34,700 is primer from `from` on
               on: [snare, kick, beats], min: 0.6,            the hits that paint (beats, 8ths: the grid's), min strength
               ahead: [0.33, 0.65],                           the front after a stroke: that share of a screen (the
                                                              16:9 crop at zoom 1) ahead of where he will be when the
                                                              next stroke comes, at most the second share ahead of him
               dur: 2.5,                                      a stroke's sweep, in photos (16ths)
               reveals: [{at: 123.2s, x: [64050, 65300], dur: 3, dir: down}],   explicit strokes (overrides)
               grade: [[101.5s, 0.7, 0.0], [115s, 0.85, 0.1]],  the revealed world's saturation and warmth over time
               trail: {from: 112.69s, sat: 1.15, warm: 0.5, ramp: 700},   more colour and warmth where he has passed
                                                              (from `from` on)
               frame_v: {zoom: 1.3, at: 0.3}}                 the 9:16 spring camera meanwhile: that much wider, his
                                                              torso at `at` of its width (the front in its right part);
                                                              false: as the camera is
Nothing is painted in the song's stops: the hit that ends a stop paints far enough ahead to carry him to the next.

A stroke is a few broad swaths of wet black paint (each a brush's width, about half his height), sweeping along the
band in 2 or 3 photos one after another, a little apart: behind each swath's leading edge the plate shows through,
its edge a band of wet black with the brush's hair lines, bristle streaks running ahead onto the primer, splatter
flying ahead of it. Where a swath stops its rounded end stays as the border's black rim, with the streaks, the
splatter and drips running down the primer, until the next stroke paints over them.

The renderer (render.py) lays it over the photo before the figure in the air: unrevealed world (the plate, its lamps
and parallax layers, the wall paint) is primer and pencil; the figure, the rain and the grain go over everything.
QA: he never stands on or lands on unrevealed world (the `reveal` rule, an error)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .choreo import ChoreoError, Ctx
from .moves import template

SWATH = 320.0  # px (450 px figure): a swath's spacing across the sweep (a broad brush, about 3/4 of his height)
PRIMER_SRGB = (0.80, 0.79, 0.755)  # the bare primer (A1's pale grey-white)
GRAPHITE_SRGB = (0.24, 0.24, 0.25)
PENCIL_ALPHA = 0.5
WET_LIN = (0.010, 0.0095, 0.009)  # the wet black paint


@dataclass
class Stroke:
    a: float  # along the sweep: from a to b (street x for dir right, y for dir down)
    b: float
    t0: float  # song s
    d: float  # s
    seed: int
    weight: float = 1.0
    dir: str = "right"
    lo: float = -1e9  # across the sweep: the band's limits (rows for right, columns for down)
    hi: float = 1e9
    vref: float = 0.0  # across the sweep: his level (the swaths sweep in one after another from above it, a diagonal)


@dataclass
class PaintIn:
    t0: float
    t1: float
    front: float
    on: list
    min: float
    ahead: tuple
    dur: float  # photos
    overrides: list = field(default_factory=list)
    grade: list = field(default_factory=list)  # [(t, sat, warm)]
    trail: dict = field(default_factory=dict)
    seed: int = 0
    frame_v: dict | None = None  # the 9:16 spring camera during the paint-in: {zoom, at} (camera.py)


@template("paint_in", prop=True)
def paint_in(c: Ctx) -> None:
    """The world painted in on the beat ahead of him (prop; see reveal.py)."""
    q0 = c.time("from")
    if q0 is None:
        raise ChoreoError("paint_in: give from")
    q1 = c.time("to")
    g = c.ch.grid
    t0 = g.t(c.song(q0))
    t1 = g.t(c.song(q1)) if q1 is not None else 1e9
    on = c.arg("on", ["kick", "snare"])
    on = [on] if isinstance(on, str) else list(on)
    for k in on:
        if k not in ("kick", "snare", "hat", "impact", "beats", "8ths"):
            raise ChoreoError(f"paint_in: on: {k!r} (kick, snare, hat, impact, beats, 8ths)")
    ahead = c.arg("ahead", [0.33, 0.65])
    if not isinstance(ahead, list | tuple) or len(ahead) != 2 or not 0 < float(ahead[0]) <= float(ahead[1]):
        raise ChoreoError("paint_in: ahead is [near, far], shares of a screen, near at most far")
    ovs = []
    for i, r in enumerate(c.arg("reveals") or []):
        if not isinstance(r, dict) or "at" not in r or "x" not in r:
            raise ChoreoError(f"paint_in: reveals[{i + 1}] is {{at, x: [x0, x1], dur, dir, y: [y0, y1]}}")
        xs = [c.coord(v, "x") for v in r["x"]]
        ys = [c.coord(v, "y") for v in r["y"]] if r.get("y") else None
        dr = str(r.get("dir", "right"))
        if dr not in ("right", "down"):
            raise ChoreoError(f"paint_in: reveals[{i + 1}] dir: right or down")
        ovs.append({"t": g.t(c.song(c.at(r["at"], key=f"reveals[{i + 1}].at"))), "x": xs, "y": ys,
                    "dur": float(r.get("dur", c.arg("dur", 2.5))), "dir": dr, "weight": float(r.get("weight", 1.2))})
    grade = []
    for i, gk in enumerate(c.arg("grade") or []):
        if not isinstance(gk, list | tuple) or len(gk) != 3:
            raise ChoreoError(f"paint_in: grade[{i + 1}] is [time, saturation, warmth]")
        grade.append((g.t(c.song(c.at(gk[0], key=f"grade[{i + 1}]"))), float(gk[1]), float(gk[2])))
    trail = dict(c.arg("trail") or {})
    if trail.get("from") is not None:
        trail["t"] = g.t(c.song(c.at(trail.pop("from"), key="trail.from")))
    fv = c.arg("frame_v", {"zoom": 1.3, "at": 0.3})
    if fv is not False and fv is not None and (not isinstance(fv, dict) or not 0.1 <= float(fv.get("at", 0.3)) <= 0.5
                                               or not 0.5 <= float(fv.get("zoom", 1.3)) <= 2.5):
        raise ChoreoError("paint_in: frame_v is {zoom (0.5 to 2.5), at (0.1 to 0.5)} or false")
    if c.dry:
        return
    c.ch.paint_in = PaintIn(t0=t0, t1=t1, front=c.coord(c.require("front"), "x"), on=on, min=float(c.arg("min", 0.5)),
                            ahead=(float(ahead[0]), float(ahead[1])), dur=float(c.arg("dur", 2.5)), overrides=ovs,
                            grade=sorted(grade), trail=trail,
                            seed=int(np.random.default_rng([c.mv.seed, c.mv.local_i, 97]).integers(0, 2 ** 31)),
                            frame_v=dict(fv) if fv else None)
    c.ch._reveal_plan = None


# ------------------------------------------------------------------------------------------------ the schedule

def plan(ch) -> dict | None:
    """The strokes and his track (cached on the choreography): {"spec", "strokes", "tt", "tx"}."""
    pi = getattr(ch, "paint_in", None)
    if pi is None:
        return None
    pl = getattr(ch, "_reveal_plan", None)
    if pl is not None:
        return pl
    from .camera import CW0
    from .check import figure_track
    tr = figure_track(ch)
    tt = np.array([r["t"] for r in tr]) if tr else np.array([pi.t0, pi.t0 + 1])
    tx = np.array([r["x"] for r in tr]) if tr else np.array([pi.front, pi.front])
    X = lambda t: float(np.interp(t, tt, tx))
    g = ch.grid
    screen = CW0 * ch.u  # the 16:9 crop at zoom 1
    t_end = min(pi.t1, max((p_.t for p_ in ch.photos()), default=pi.t0) + 1.0)  # (no `to`: the last photo)
    step16 = g.t(g.resolve("1.1.2")[0]) - g.t(g.resolve("1.1.1")[0])
    d = pi.dur * step16
    ev = []
    for k in pi.on:
        if k in ("beats", "8ths"):
            st = 4 if k == "beats" else 2
            q = math.ceil(g.q(pi.t0) / st - 1e-9) * st
            while g.t(q) <= t_end:
                ev.append((g.t(q), 0.6))
                q += st
        else:
            ev.extend((t, s) for t, s in g.hits.get(k, []) if s >= pi.min and pi.t0 <= t <= t_end)
    # nothing is painted in the song's stops (the silence): the hit that ends one paints far enough to carry him on
    stops = [(float(sp_["t0"]), float(sp_["t1"])) for sp_ in g.stops]
    ev = [(t, s) for t, s in ev if not any(a - 1e-3 <= t < b - 0.02 for a, b in stops)]
    for a, b in stops:
        if pi.t0 <= b <= t_end:
            ev.append((b, 1.0))
    ev.sort()
    merged = []
    for t, s in ev:  # (a kick on a beat: one stroke, the stronger)
        if merged and t - merged[-1][0] < 0.06:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s))
        else:
            merged.append((t, s))
    rng = np.random.default_rng(pi.seed)
    strokes = [Stroke(a=-1e9, b=pi.front, t0=pi.t0 - 10.0, d=0.2, seed=int(rng.integers(0, 2 ** 31)),
                      vref=ch.scene.feet_at(pi.front) - 225 * ch.u)]
    front = pi.front
    lo_s, hi_s = pi.ahead[0] * screen, pi.ahead[1] * screen
    for i, (te, s) in enumerate(merged):
        tn = merged[i + 1][0] if i + 1 < len(merged) else te + 4 * step16
        target = min(X(tn) + lo_s, X(te) + max(hi_s, X(tn) - X(te) + lo_s))  # (ahead[0] when the next one comes)
        if target - front < 0.06 * screen:  # far enough ahead already
            continue
        strokes.append(Stroke(a=front - 0.1 * screen, b=target, t0=te, d=d, seed=int(rng.integers(0, 2 ** 31)),
                              weight=float(np.clip(s, 0.5, 1.3)), vref=ch.scene.feet_at(target) - 225 * ch.u))
        front = target
    for o in pi.overrides:
        if o["dir"] == "right":
            a, b = o["x"]
            lo, hi = o["y"] if o["y"] else (-1e9, 1e9)
        else:
            a, b = o["y"] if o["y"] else (ch.scene.Y0, ch.scene.Y1)
            lo, hi = o["x"]
        strokes.append(Stroke(a=a, b=b, t0=o["t"], d=o["dur"] * step16, seed=int(rng.integers(0, 2 ** 31)),
                              weight=o["weight"], dir=o["dir"], lo=lo, hi=hi,
                              vref=ch.scene.feet_at(b) - 225 * ch.u if o["dir"] == "right" else (lo + hi) / 2))
    strokes.sort(key=lambda s_: s_.t0)
    pl = {"spec": pi, "strokes": strokes, "tt": tt, "tx": tx}
    ch._reveal_plan = pl
    return pl


# ------------------------------------------------------------------------------------------------ the strokes

def _h(k, seed) -> np.ndarray:
    """A hash of integers to [0, 1)."""
    k = np.asarray(k, dtype=np.float64)
    return np.modf(np.abs(np.sin(k * 12.9898 + (seed % 9973) * 78.233 + 0.5) * 43758.5453))[0]


def _n1(v, scale: float, seed: int) -> np.ndarray:
    """Smooth 1D value noise in [-1, 1] at absolute coordinate v (the same on every photo)."""
    z = np.asarray(v, dtype=np.float64) / scale
    k = np.floor(z)
    f = z - k
    f = f * f * (3 - 2 * f)
    return 2 * (_h(k, seed) * (1 - f) + _h(k + 1, seed) * f) - 1


def stroke_layers(st: Stroke, t: float, U: np.ndarray, V: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
    """One stroke at song time t over a grid of positions along the sweep (U, 1D) and across it (V, 1D): its reveal
    coverage R and its wet black paint K, each (len(V), len(U))."""
    nV, nU = len(V), len(U)
    R = np.zeros((nV, nU), np.float32)
    K = np.zeros((nV, nU), np.float32)
    if t < st.t0:
        return R, K
    S = SWATH * u
    Va = np.clip(V, st.lo, st.hi)
    inside_band = (V >= st.lo - 1) & (V <= st.hi + 1)
    if not inside_band.any():
        return R, K
    i0, i1 = int(math.floor((Va.min() - S) / S)), int(math.ceil((Va.max() + S) / S))
    front = np.full(nV, -1e12)
    left = np.full(nV, 1e12)
    settle = np.zeros(nV)  # 0 while its swath sweeps, to 1 as it soaks in
    moving = np.zeros(nV)
    for i in range(i0, i1 + 1):
        r_ = _h(np.array([i * 7 + 1, i * 7 + 2, i * 7 + 3, i * 7 + 4, i * 7 + 5, i * 7 + 6]), st.seed)
        yc = (i + 0.3 * (r_[0] - 0.5)) * S
        half = S * (0.62 + 0.14 * r_[1])
        # one swath after the other from the top down (the brush's diagonal), a little jitter
        delay = st.d * (0.42 * float(np.clip((yc - (st.vref - 800 * u)) / (1500 * u), 0, 1)) + 0.08 * r_[2])
        dd = max(st.d - delay, 1e-3)
        p = float(np.clip((t - st.t0 - delay) / dd, 0, 1))
        if p <= 0:
            continue
        p = 1 - (1 - p) ** 2.2  # a brush slams on fast and slows to its end
        over = (r_[3] - 0.35) * 110 * u * st.weight
        cap = S * (0.35 + 0.35 * r_[4])
        uu = (V - yc) / half
        inn = np.abs(uu) < 1
        if not inn.any():
            continue
        capd = cap * (1 - np.sqrt(np.clip(1 - uu * uu, 0, 1)))
        ea = 22 * u * _n1(V, 26 * u, st.seed + i) + 9 * u * _n1(V, 7 * u, st.seed + 31 * i)
        a_ = st.a - 40 * u
        b_ = st.b + over
        f = a_ + (b_ - a_) * p + ea - capd
        lft = a_ + capd + 14 * u * _n1(V, 19 * u, st.seed + 77 * i)
        sel = inn & (f > lft) & inside_band
        better = sel & (f > front)
        front = np.where(better, f, front)
        st_end = st.t0 + delay + dd
        settle = np.where(better, np.clip((t - st_end) / 0.3, 0, 1), settle)
        moving = np.where(better, 1.0 - min(1.0, p), moving)
        left = np.where(sel, np.minimum(left, lft), left)
    has = front > -1e11
    if not has.any():
        return R, K
    Ub = U[None, :]
    fr, lf = front[:, None], left[:, None]
    # the wet stroke: the brush lays black paint on the primer at its front, wide while it sweeps (most of the chunk:
    # the stroke itself), and the world develops behind its leading edge (the plate showing through the paint),
    # soaking in to the border's rim when it stops; the brush's hair lines through it
    chunk = float(np.clip(0.6 * (st.b - st.a), 180 * u, 560 * u)) if st.a > -1e8 else 40 * u
    wet = (1 - settle)[:, None]
    wk = 36 * u + (chunk - 36 * u) * wet
    lag = 0.42 * (wk - 36 * u)  # (the plate behind the paint's solid leading part)
    soft = np.maximum(1.5, 0.2 * (wk - 36 * u))
    R = (np.clip((fr - lag - Ub) / soft + 0.5, 0, 1) * np.clip((Ub - lf) / 1.5 + 0.5, 0, 1)).astype(np.float32)
    R[~has] = 0
    dist = (fr - Ub) / wk
    # a solid wet core at the leading edge (the hairs a faint ripple in it), then the dry-brush tail: the brush's hair
    # lines breaking up, more of them the further back, the world showing between them
    fine = (0.93 + 0.07 * _n1(V, 2.2 * u, st.seed + 5))[:, None]
    hs = (0.5 + 0.5 * _n1(V, 3.1 * u, st.seed + 6))[:, None] * (0.75 + 0.25 * _n1(V, 13 * u, st.seed + 8))[:, None]
    thr = np.clip((dist - 0.45) / 0.55, 0, 1) * 1.05
    tail = np.clip((hs - thr) / 0.12 + 0.5, 0, 1)
    band = np.where(dist <= 0.45, 0.97 * fine, 0.95 * tail) * (Ub <= fr) * (dist <= 1)
    # bristle streaks running ahead of the front onto the primer
    sl = np.clip((_n1(V, 3.4 * u, st.seed + 9) - 0.25) / 0.75, 0, 1) * (40 + 70 * st.weight) * u
    ahead = (Ub > fr) & (Ub - fr < sl[:, None])
    streak = np.where(ahead, np.clip(1 - (Ub - fr) / np.maximum(sl[:, None], 1e-3), 0, 1) ** 1.2 * 0.9, 0)
    K = np.maximum(band, streak).astype(np.float32)
    K[~has] = 0
    return R, K


def _splatter_drips(st: Stroke, t: float, u: float):
    """The stroke's splatter dots and drips at time t: [(x, y, r, j)] and [(x, y0, y1, w, j)], y relative to the
    stroke's middle height (the caller's), only for strokes that sweep right (the border runs down the primer)."""
    rng = np.random.default_rng(st.seed + 1)
    dots, drips = [], []
    if t < st.t0 or st.a < -1e8:
        return dots, drips
    n = int(30 * st.weight + 10)
    S = SWATH * u
    for j in range(n):
        x = st.b + rng.uniform(-0.15, 1.0) ** 2 * 260 * u * (1 if rng.random() < 0.85 else -1)
        y = rng.uniform(-3.5, 3.5) * S  # relative: placed round the band's visible middle by the caller
        r = (1.5 + 7 * rng.random() ** 3) * u
        ta = st.t0 + st.d * (0.45 + 0.5 * rng.random())
        if t >= ta:
            dots.append((x, y, r, j))
    for j in range(int(6 + 6 * st.weight)):
        x = st.b - rng.uniform(0, 30) * u
        y = rng.uniform(-3.5, 3.5) * S
        L = (40 + 200 * rng.random() ** 2) * u
        te = st.t0 + st.d
        if t > te:
            ln = L * (1 - math.exp(-(t - te) / 0.9))
            drips.append((x, y, y + ln, (2.5 + 4 * rng.random()) * u, j))
    return dots, drips


def layers(ch, t: float, x0: int, y0: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The reveal coverage R (1: the world shows) and the wet black paint K over the crop at song time t, or None
    when the paint-in is not on (everything shows)."""
    pl = plan(ch)
    if pl is None or t < pl["spec"].t0:
        return None
    u = ch.u
    R = np.zeros((h, w), np.float32)
    xs = np.arange(x0, x0 + w, dtype=np.float64)
    ys = np.arange(y0, y0 + h, dtype=np.float64)
    pad = 900 * u
    strokes = [s for s in pl["strokes"] if s.t0 <= t]
    sc = ch.scene
    done = []
    for si, st in enumerate(strokes):
        if st.dir == "right":
            if st.b + pad < x0 or st.a - pad > x0 + w or st.hi < y0 or st.lo > y0 + h:
                continue
            r_, k_ = stroke_layers(st, t, xs, ys, u)
        else:
            if st.b + pad < y0 or st.a - pad > y0 + h or st.hi < x0 or st.lo > x0 + w:
                continue
            r_, k_ = stroke_layers(st, t, ys, xs, u)
            r_, k_ = np.ascontiguousarray(r_.T), k_.T
        k_ = np.ascontiguousarray(k_)
        if st.dir == "right":  # splatter and drips on the primer round its end
            dots, drips = _splatter_drips(st, t, u)
            if dots or drips:
                canvas = np.zeros((h, w), np.uint8)
                ym = sc.feet_at(st.b) - 225 * u  # (round his height where it ends: the same on every photo)
                for x, y, r, j in dots:
                    cx_, cy_ = int(round(x - x0)), int(round(ym + y + 40 * u * _h(j, st.seed) - y0))
                    if -40 < cx_ < w + 40 and -40 < cy_ < h + 40:
                        rr = max(1, int(round(r)))
                        cv2.circle(canvas, (cx_, cy_), rr, 255, -1, cv2.LINE_AA)
                        if _h(j + 3, st.seed) < 0.35:  # flung: a tail back along the sweep, thinning
                            tl = int(round(r * (3 + 6 * _h(j + 4, st.seed))))
                            cv2.line(canvas, (cx_, cy_), (cx_ - tl, cy_ + int(round(tl * 0.12))), 255,
                                     max(1, rr // 2), cv2.LINE_AA)
                        for m_ in range(int(3 * _h(j + 5, st.seed))):  # satellites
                            ang = 6.283 * _h(j * 11 + m_, st.seed)
                            dd_ = r * (1.6 + 2.5 * _h(j * 13 + m_, st.seed))
                            cv2.circle(canvas, (int(round(cx_ + dd_ * math.cos(ang))), int(round(cy_ + dd_ * math.sin(ang)))),
                                       max(1, rr // 3), 255, -1, cv2.LINE_AA)
                for x, ya, yb, wd, j in drips:
                    cx_ = int(round(x - x0))
                    a_, b_ = int(round(ym + ya - y0)), int(round(ym + yb - y0))
                    if -20 < cx_ < w + 20 and b_ > -20 and a_ < h + 20:
                        cv2.line(canvas, (cx_, a_), (cx_, b_), 255, max(1, int(round(wd))), cv2.LINE_AA)
                        cv2.circle(canvas, (cx_, b_), max(1, int(round(wd * 0.9))), 255, -1, cv2.LINE_AA)
                k_ = np.maximum(k_, canvas.astype(np.float32) / 255 * 0.95)
        R = np.maximum(R, r_)
        done.append((r_, k_))
    K = np.zeros((h, w), np.float32)
    # a stroke's wet paint is painted over by any later stroke's reveal (the next stroke covers the old rim)
    later = np.zeros((h, w), np.float32)
    for r_, k_ in reversed(done):
        K = np.maximum(K, k_ * (1 - later))
        later = np.maximum(later, r_)
    return R, K


def revealed(ch, t: float, x: float, y: float) -> float:
    """How revealed the world is at one street point at song time t (1 when the paint-in is not on)."""
    lr = layers(ch, t, int(math.floor(x)), int(math.floor(y)), 1, 1)
    return 1.0 if lr is None else float(lr[0][0, 0])


# ------------------------------------------------------------------------------------------------ the look

_TEX: dict = {}


def _tex(seed: int = 3) -> np.ndarray:
    """A 512 px tile of primer texture (brushed plaster), repeating over the street, around 1."""
    if seed not in _TEX:
        rng = np.random.default_rng(seed)
        a = cv2.resize(rng.normal(0, 1, (16, 16)).astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
        b = cv2.GaussianBlur(rng.normal(0, 1, (512, 512)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=1.5)
        c = rng.normal(0, 1, (512, 512)).astype(np.float32)
        _TEX[seed] = 1 + 0.035 * a + 0.025 * b / (b.std() + 1e-6) + 0.012 * c
    return _TEX[seed]


def pencil(sp) -> np.ndarray:
    """The plate's main edges as graphite lines (0 to 1), for a whole span (cached on it)."""
    pe = getattr(sp, "_pencil", None)
    if pe is not None:
        return pe
    lum = np.clip(sp.lin.mean(axis=2), 0, 1) ** (1 / 2.2)
    small = cv2.resize(lum, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.6)
    g8 = np.clip(small * 255, 0, 255).astype(np.uint8)
    gx, gy = cv2.Sobel(small, cv2.CV_32F, 1, 0), cv2.Sobel(small, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag, 93)) * 255 * 0.25
    ed = cv2.Canny(g8, max(6.0, 0.45 * hi), max(12.0, hi), L2gradient=True)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ed > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = np.hypot(stats[1:, 2], stats[1:, 3]) >= 14
    ed = keep[lab].astype(np.float32)
    big = cv2.resize(ed, (sp.lin.shape[1], sp.lin.shape[0]), interpolation=cv2.INTER_LINEAR)
    big = np.clip(cv2.GaussianBlur(big, (0, 0), 0.7) * 1.6, 0, 1)
    try:
        sp._pencil = big
    except Exception:
        pass
    return big


def _srgb2lin(c) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def primer_look(sp, x0: int, y0: int, w: int, h: int, sx: slice, sy: slice) -> np.ndarray:
    """The unrevealed world over the crop (linear light): pale primer with the plate's edges in pencil."""
    tex = _tex()
    ty = (np.arange(y0, y0 + h) % 512)[:, None]
    tx_ = (np.arange(x0, x0 + w) % 512)[None, :]
    T = tex[ty, tx_]
    base = _srgb2lin(PRIMER_SRGB)[None, None, :] * T[..., None]
    pe = pencil(sp)[sy, sx]
    var = 0.75 + 0.25 * _n1(np.arange(x0, x0 + w), 90.0, 11)[None, :]  # the pencil's pressure along its lines
    a = (PENCIL_ALPHA * pe * var)[..., None]
    return base * (1 - a) + _srgb2lin(GRAPHITE_SRGB)[None, None, :] * a


def grade(ch, out: np.ndarray, t: float, x0: int) -> np.ndarray:
    """The revealed world's colour at time t: saturation and warmth over the film, more of both where he has passed."""
    pl = plan(ch)
    pi = pl["spec"]
    if not pi.grade and not pi.trail:
        return out
    if pi.grade:
        ts = [k[0] for k in pi.grade]
        sat = float(np.interp(t, ts, [k[1] for k in pi.grade]))
        warm = float(np.interp(t, ts, [k[2] for k in pi.grade]))
    else:
        sat, warm = 1.0, 0.0
    w = out.shape[1]
    sat_x = np.full(w, sat, np.float32)
    warm_x = np.full(w, warm, np.float32)
    if pi.trail:
        xh = float(np.interp(t, pl["tt"], pl["tx"]))
        ramp = float(pi.trail.get("ramp", 700)) * ch.u
        on_ = float(np.clip((t - pi.trail.get("t", -1e9)) / 0.35, 0, 1))  # (from its `from`, in over a beat)
        T = on_ * np.clip((xh - np.arange(x0, x0 + w)) / ramp, 0, 1).astype(np.float32)
        sat_x = sat_x + (float(pi.trail.get("sat", sat)) - sat) * T
        warm_x = warm_x + float(pi.trail.get("warm", 0.0)) * T
    lum = (out @ np.float32([0.2126, 0.7152, 0.0722]))[..., None]
    o = lum + sat_x[None, :, None] * (out - lum)
    wm = warm_x[None, :, None]
    return o * (1 + wm * np.float32([0.22, 0.06, -0.18]))


def composite(ch, out: np.ndarray, sp, R: np.ndarray, K: np.ndarray, t: float, x0: int, y0: int, sx: slice,
              sy: slice) -> np.ndarray:
    """The photo's world (linear, before the figure in the air) with the paint-in: primer and pencil where it is not
    revealed, graded where it is, the wet black paint of the strokes over both."""
    h, w = R.shape
    pr = primer_look(sp, x0, y0, w, h, sx, sy)
    g_ = grade(ch, out, t, x0)
    o = pr * (1 - R[..., None]) + g_ * R[..., None]
    if K.max() > 0:
        sheen = np.clip(K - np.roll(K, 2, axis=0), 0, 1) * 0.12  # a wet glint on the paint's upper edges
        o = o * (1 - K[..., None]) + np.float32(WET_LIN) * K[..., None] + sheen[..., None]
    return o
