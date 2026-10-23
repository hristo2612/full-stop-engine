"""The solid skin after the wall (Full Stop v7, review note; the lead's pick "A"): once he has left the wall he is
the paint figure of the J2 alley leap, but opaque, so he reads over dark brick and night sky as well as over the lit
alley: flat black emulsion with the drawing's own dry-brush strokes in dark grey, the white soles and socks left as
primer, and a thin rough off-white rim along his outline (rim_px, 3 px for a 450 px figure), broken like a dry brush
edge. The ghost trail and the drip flecks stay as render.py draws them.

The choreography's `skin_look` prop sets the stretch: `- skin_look: {from: 101.49s}` (to: optional; rim_px,
strokes, primer: overrides of SKIN_LOOK). render.History.air_layer ends with `A, col = SL.apply(self, q, A, col, x0,
y0, w, h)`: the skins shown on that photo inside a stretch are painted over its result, opaque. Outside every stretch
nothing changes (the alley leap's skin stays as it is). Where stretches overlap, the one that starts last wins (a
short stretch inside the long one: the big leap's trail).

The rim follows the camera (the lead's note on film-v7's big leap: in the wide frame he vanished): it is never
thinner on screen than `rim_min` px of the full-size frame, and `rim_wide` px once the frame is zoomed out (its scale
below `wide_scale` screen px per street px). The screen scale is read from the crop air_layer is asked for (the
camera's box plus its 6 px margin: the 16:9 one wider than tall, out to 1920 px; the 9:16 one out to 1080).

The flights' echo trail (a leap, a hop, a drop, a flip: the J2 silhouettes in the air, not the run's skins) is
skinlook's inside a stretch: the last `trail_n` photos of the flight as soft paint smears in `trail_col`, the
newest at `trail` opacity falling off geometrically to `trail_end` (the lead's note on the big leap: 4 echoes, 35%
to 8%, the J2 ghosts' softness, so his own drawing outweighs them); render.py skips its own ghosts of those items
there (`trail_owns`). `trail: 0` leaves the flights' ghosts to render.py. The run's skins keep render.py's ghosts.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .choreo import ChoreoError, Ctx
from .moves import template

SKIN_LOOK = {
    "ink": 0.0035,                       # flat black emulsion (linear)
    "strokes": 0.030,                    # the drawing's light dry-brush strokes, as dark grey (linear, at most)
    "primer": [0.30, 0.285, 0.26],       # off-white primer, lit by the night (linear)
    "rim_px": 3.0, "rim_gain": 1.15,     # the rim: px (a 450 px figure), its brightness against the primer
    "rim_break": -1.1,                   # where the dry edge breaks (noise threshold; lower: fewer breaks)
    "rim_min": 3.0, "rim_wide": 4.0,     # the rim at least this many px of the full-size frame (wide: zoomed out)
    "wide_scale": 1.05,                  # zoomed out: below this many screen px per street px (16:9 past 1.2x)
    "trail": 0.35,                       # a flight's newest echo's opacity (0: render.py's own ghosts instead)
    "trail_end": 0.08,                   # its oldest echo's
    "trail_n": 4,                        # echoes (photos back)
    "trail_col": [0.2, 0.19, 0.17],      # their paint (linear: a pale grey smear, not primer white)
    "trail_blur": 1.5,                   # px (a 450 px figure: the J2 ghosts' softness)
}


@template("skin_look", prop=True)
def skin_look(c: Ctx) -> None:
    """The solid skin from `from` (to: optional, the song's end by default); look overrides: rim_px, strokes, ink,
    primer, rim_gain, rim_break (skinlook.py)."""
    q0 = c.time("from")
    if q0 is None:
        raise ChoreoError("skin_look: give from")
    q1 = c.time("to") if c.has("to") else None
    if q1 is not None and q1 <= q0:
        raise ChoreoError("skin_look: to must be after from")
    look = {k: c.arg(k) for k in SKIN_LOOK if c.has(k)}
    if not 0 <= float(look.get("trail", 0.0)) <= 1 or not 0 <= float(look.get("trail_end", 0.0)) <= 1:
        raise ChoreoError("skin_look: trail and trail_end are opacities, 0 to 1")
    if int(look.get("trail_n", 4)) < 1:
        raise ChoreoError("skin_look: trail_n is 1 or more")
    spans = list(getattr(c.ch, "skin_looks", None) or [])
    spans.append({"q0": float(c.song(q0)), "q1": float(c.song(q1)) if q1 is not None else math.inf, "look": look})
    c.ch.skin_looks = spans


def span_at(ch, q: float) -> dict | None:
    """The stretch photo q is in (overlapping: the one that starts last)."""
    best = None
    for s in getattr(ch, "skin_looks", None) or []:
        if s["q0"] - 1e-6 <= q < s["q1"] - 1e-6 and (best is None or s["q0"] >= best["q0"]):
            best = s
    return best


def screen_scale(w: int, h: int) -> float:
    """Screen px of the full-size frame per street px, from the crop air_layer renders (the camera's box plus 6 px
    each side): 1920 px across a 16:9 crop, 1080 across a 9:16 one."""
    return (1920.0 if w >= h else 1080.0) / max(w - 13.0, 1.0)


def rim_width(L: dict, u: float, scr: float | None) -> float:
    """The rim's width in street px: rim_px's own (about 2 px for 3), widened so that it is at least rim_min px on
    screen (rim_wide when the frame is zoomed out)."""
    w0 = (max(3, int(round(5 * L["rim_px"] / 3.0 * u))) - 1) / 2.0
    if not scr:
        return w0
    want = float(L["rim_wide"] if scr < float(L["wide_scale"]) else L["rim_min"])
    return max(w0, want / scr)


def _noise(h: int, w: int, rng, sx: float, sy: float, ang: float = 0.0) -> np.ndarray:
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
    if ang:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        n = cv2.warpAffine(n, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return n / max(float(n.std()), 1e-6)


def figure(hist, it, L: dict, u: float, scr: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                          np.ndarray]:
    """(solid 0..1, whites 0..1, strokes 0..1, rim 0..1) of a skin item at its own crop (air_mask's); scr: the
    screen's px per street px (the rim kept wide enough to read), None: rim_px as drawn."""
    raw = hist.air_mask(it)
    d = np.clip(raw * 2.4, 0, 1)
    m = (cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0.3).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros(m.shape, np.uint8)
    cv2.drawContours(solid, cnts, -1, 1, -1)  # his outline, filled (no see-through gaps)
    holes = (solid > 0) & (raw < 0.25)  # the drawing's bare-wall holes
    ys = np.nonzero(solid.any(1))[0]
    low = np.zeros_like(holes)
    if len(ys):
        low[int(ys.max() - 0.2 * (ys.max() - ys.min())):] = True  # the white soles and socks: holes in his lowest fifth
    whites = cv2.morphologyEx((holes & low).astype(np.float32), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    s = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.7)
    strokes = np.clip(1 - raw * 1.25, 0, 1) * s * (1 - whites)
    rng = np.random.default_rng(int(abs(it.x0) * 7 + abs(it.y0)) % (2 ** 31))
    rw = rim_width(L, u, scr)
    if scr is None:
        k = max(3, int(round(5 * L["rim_px"] / 3.0 * u))) | 1
        ring = np.clip(s - cv2.erode(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), 0, 1)
    else:  # a band rw px deep inside his outline (sub-pixel: the distance to the outside, eased over a pixel)
        dist = cv2.distanceTransform((s > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        ring = np.clip(rw + 0.5 - dist, 0, 1) * (s > 0.02)
    n = _noise(*s.shape, rng, 3.0, 0.8, 30)
    ring = ring * np.clip(0.75 + 0.35 * n, 0, 1) * (n > float(L["rim_break"]))
    return s, cv2.GaussianBlur(whites, (0, 0), 0.6), strokes, ring.astype(np.float32)


def _paste(dst, src, ax0, ay0):
    h, w = dst.shape[:2]
    sx0, sy0, dx0, dy0 = max(0, -ax0), max(0, -ay0), max(0, ax0), max(0, ay0)
    cw, chh = min(src.shape[1] - sx0, w - dx0), min(src.shape[0] - sy0, h - dy0)
    if cw > 0 and chh > 0:
        dst[dy0:dy0 + chh, dx0:dx0 + cw] = np.maximum(dst[dy0:dy0 + chh, dx0:dx0 + cw], src[sy0:sy0 + chh, sx0:sx0 + cw])


def apply(hist, q: float, A: np.ndarray, col: np.ndarray, x0: int, y0: int, w: int, h: int):
    """render.History.air_layer's last step: inside a `skin_look` stretch, the skins shown on photo q painted over
    (A, col), opaque. Returns (A, col) unchanged outside every stretch or when no skin is shown."""
    sp = span_at(hist.ch, q)
    if sp is None:
        return A, col
    items = hist.air.get(q, [])
    # every figure in the air on this photo: the run's skins (skin: true) and the flights' J2 silhouettes alike (a
    # leap, a hop, the flips: the lead's pick A is the look of every skin after the wall)
    skins = [it for it in items if it.kind == "sil"]
    if not skins:
        skins = list(hist.held_skins(q))
    if not skins:
        return A, col
    L = dict(SKIN_LOOK, **sp["look"])
    u = hist.u
    scr = screen_scale(w, h)
    if float(L["trail"]) > 0:
        A, col = _trail(hist, q, L, A, col, x0, y0, w, h)
    S = np.zeros((h, w), np.float32)
    Wt = np.zeros((h, w), np.float32)
    St = np.zeros((h, w), np.float32)
    Rm = np.zeros((h, w), np.float32)
    for it in skins:
        s, wh, stk, ring = figure(hist, it, L, u, scr)
        ax0, ay0 = int(round(it.x0)) - x0, int(round(it.y0)) - y0
        _paste(S, s, ax0, ay0)
        _paste(Wt, wh, ax0, ay0)
        _paste(St, stk, ax0, ay0)
        _paste(Rm, ring, ax0, ay0)
    primer = np.float32(L["primer"])
    fig = np.full((h, w, 3), float(L["ink"]), np.float32) + (St * float(L["strokes"]))[..., None] * \
        np.float32([1.0, 0.97, 0.92])
    fig = fig * (1 - Wt[..., None]) + primer * Wt[..., None]
    fig = fig * (1 - Rm[..., None]) + primer * float(L["rim_gain"]) * Rm[..., None]
    A2 = 1 - (1 - A) * (1 - S)
    col2 = col * (1 - S[..., None]) + fig * S[..., None]
    return A2.astype(np.float32), col2.astype(np.float32)


def trail_owns(hist, q: float, it) -> bool:
    """Whether skinlook draws the echoes of air item `it` on photo q (a flight's silhouette inside a stretch with a
    trail): render.History.air_layer then skips its own ghost of it."""
    if it.kind != "sil" or it.flags.get("skin"):
        return False
    sp = span_at(hist.ch, q)
    return sp is not None and float(dict(SKIN_LOOK, **sp["look"])["trail"]) > 0


def echo_alpha(L: dict, dq: float) -> float:
    """The opacity of the echo dq photos back: trail at 1, trail_end at trail_n, geometric between; 0 outside."""
    n = max(1, int(L["trail_n"]))
    if dq < 1 - 1e-6 or dq > n + 1e-6:
        return 0.0
    a0, a1 = float(L["trail"]), float(L["trail_end"])
    if n == 1 or a0 <= 0:
        return a0
    return a0 * (max(a1, 1e-4) / a0) ** ((dq - 1) / (n - 1))


def _trail(hist, q: float, L: dict, A: np.ndarray, col: np.ndarray, x0: int, y0: int, w: int, h: int):
    """The flights' silhouettes of the last trail_n photos before q as soft paint smears over (A, col) (the J2
    ghosts' softness: the drawing's own paint doubled, blurred trail_blur px), echo_alpha strong."""
    import bisect
    G = np.zeros((h, w), np.float32)
    aq = getattr(hist, "_air_q", None) or sorted(hist.air)
    sig = max(0.5, float(L["trail_blur"]) * hist.u)
    for qq in aq[max(0, bisect.bisect_left(aq, q - 40) - 1):bisect.bisect_left(aq, q)]:
        for it in hist.air[qq]:
            if it.kind != "sil" or it.flags.get("skin") or q <= qq:
                continue
            a = echo_alpha(L, round(q - qq, 6))
            if a <= 0:
                continue
            d = cv2.GaussianBlur(np.clip(hist.air_mask(it) * 2.0, 0, 1), (0, 0), sig) * a
            ax0, ay0 = int(round(it.x0)) - x0, int(round(it.y0)) - y0
            sx0, sy0, dx0, dy0 = max(0, -ax0), max(0, -ay0), max(0, ax0), max(0, ay0)
            cw, chh = min(d.shape[1] - sx0, w - dx0), min(d.shape[0] - sy0, h - dy0)
            if cw > 0 and chh > 0:
                G[dy0:dy0 + chh, dx0:dx0 + cw] = 1 - (1 - G[dy0:dy0 + chh, dx0:dx0 + cw]) * (
                    1 - d[sy0:sy0 + chh, sx0:sx0 + cw])
    if not G.any():
        return A, col
    A2 = 1 - (1 - A) * (1 - G)
    tc = np.float32(L["trail_col"])
    col2 = (col * (A * (1 - G))[..., None] + tc * G[..., None]) / np.maximum(A2, 1e-4)[..., None]
    return A2.astype(np.float32), col2.astype(np.float32)
