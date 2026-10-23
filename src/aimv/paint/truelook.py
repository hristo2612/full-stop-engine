"""He becomes true (Full Stop, from "Full stop" in chorus 2): out of the paint, a solid living shadow on the roofs
(board 22-true T4). Three-dimensional, deep matte black with a little of its own shading for volume, only the canon
shoes' thick white midsoles and short white socks kept, dim (the uppers, laces and the flash go dark with him: no
brand's mark; the paint figure's signature, white soles), a thin warm rim of the city's light on the edges facing the glow, a soft contact shadow when he is
grounded, a faint dark after-image of the previous photos; no primer, no paint marks, no skins. The photo look stays:
stop-motion photos on the grid, the plate's rain and grain over him too.

Sources: photoreal poses on flat chroma green (boards/23-true-sheet TS1, TS2; the true Wan clips on green with pale
blue guides). greenkey.screen_matte keys a frame: alpha from the colour distance to the screen's own green (its
border median, in CrCb) with a guard where green clearly dominates, and the RGB despilled. A sheet or video entry in
poses.spec.json with "screen": "green" is matted that way by `aimv paint poses`, which also writes each drawing's RGB
next to its matte (<pose>.rgb.png, the same crop), so true drawings are ordinary library drawings (anchors, scale,
sequences, QA).

The choreography's `true` prop marks a stretch of the film true: `- true: {from: 103.53s}` (to: optional, the
end of the song by default; look: {...} overrides TRUE_LOOK). Every pose from then on is true: render.py's History
lets `add` place it (no primer, no marks, no skin, no droplets) and Renderer.photo calls `composite` after the
parallax layers: every plate layer (lamps and their blinks, lit windows, signs, screens) is under him, and only the
rain, haze and grain go over him. Wall marks painted before the stretch stay as they are (the empty shape he stepped
out of keeps its paint: true photos are not paint photos, so they do not fade or paint out the trail).

The moment he steps out of the paint is a full-frame clip (Wan, on the plate): `true_insert` puts it in place of the
photos of a stretch, one insert per aspect: `- true_insert: {clip: clips/finish/clips/tf-insert-16x9.mp4, from: 102.24s,
to: 103.53s, aspect: "16:9", camera: [cx, cy, cw, ch, rot]}` (camera: the one locked camera its start and end frames
were rendered at; frames: [first, last] to use part of it). Renderer.photo asks `insert_photo` first: each photo in the
stretch is the insert's frame at its time, cropped to the photo's own camera (so the camera's moves in the stretch
still read), with the photo's rain over it; the MUTO look follows as for every photo. `insert` (v8) is a hard cut to a
full-frame clip of another view (the straight-down drop): the clip fills the photo, at its own speed, synced by one
frame to a time.
"""

from __future__ import annotations

import bisect
import math

import cv2
import numpy as np

from .choreo import ChoreoError, Ctx
from .greenkey import cut_sheet, screen_matte  # noqa: F401  (the green key; re-exported)
from .moves import template
from .scene import lin2srgb, luminance, srgb2lin

# the look (linear light; px are street px for a 450 px figure, times ch.u)
TRUE_LOOK = {
    # body: floor + amp * min(lum / ref, cap) ** gamma (lum: the photo's own luminance), neutral; sRGB about 8 to 18
    "floor": 0.0017, "amp": 0.0022, "ref": 0.012, "cap": 3.0, "gamma": 0.8, "tint": [1.0, 1.0, 1.0],
    # the canon shoes' and socks' neutral whites (not skin), dimmed; only the midsoles and socks (shoe_whites: px for a
    # 686 px standing figure, the true sheets' size): socks are thick bright blobs reaching his outline, midsoles the
    # bright band hugging it (no deeper than sole_near), each with real white in it (sole_hi); thin strokes (the
    # flash, laces) and anything inside the dark upper stay dark
    "white": 0.30, "white_lo": 0.20, "white_hi": 0.70, "soles_only": True,
    "sole_lo": 0.18, "sole_hi": 0.45, "sock_r": 6, "sole_edge": 24, "sole_near": 9.0, "sole_thin": 3,
    # the rim: a band inside his outline, `rim_px` wide against light, `rim_dark_px` against a dark background;
    # warm, its strength from the frame's glow (rim_glow) and the plate just behind the edge, times rim_gain,
    # never below rim_min (his outline always reads) nor above rim_max; strongest on the edges facing up and back
    "rim_px": 3.0, "rim_dark_px": 5.0, "rim_pow": 1.3, "rim_warm": [1.0, 0.47, 0.18], "rim_gain": 13.5,
    "rim_min": 0.06, "rim_max": 0.9, "rim_glow": 0.6, "rim_local": 0.25, "rim_up": 0.9, "rim_back": 0.5,
    "rim_ambient": 0.05, "dark_lum": 0.03, "bloom": 2.2, "bloom_k": 0.35,
    # contact shadow on the surface under his feet when grounded (within shadow_near px of the feet line)
    "shadow": 0.5, "shadow_h": 7.0, "shadow_w": 1.2, "shadow_near": 14.0,
    # integration: edge softness (px), the plate's grain (sRGB std, measured from the plate when null), haze
    # (a share of the plate's soft colour), rain streaks over him (count per 1000 px of figure height)
    # (the edge: the alpha scaled smoothly, then refined at the output's scale: a blur of about `edge_px` output px and
    # a gentle contrast curve `edge_gain` about one half: crisp but anti-aliased at any camera; the rim, the contact
    # shadow and the after-images all follow that smooth line; haze_blur wide enough that it is an even veil, never
    # the shapes of lit windows behind him)
    "soft": 0.7, "edge_px": 1.0, "edge_gain": 1.25, "grain": None, "haze": 0.05, "haze_blur": 120.0, "rain": 26,
    "rain_alpha": [0.04, 0.10],
    # after-images: the previous photos as flat dark ghosts (newest first), softened
    "after": [0.30, 0.15], "after_blur": 1.2,
}


# ------------------------------------------------------------------------------------------------ the look

def _disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def shoe_whites(rgb: np.ndarray, a: np.ndarray, L: dict, u: float = 1.0) -> np.ndarray:
    """Where his whites stay white (bool): the socks and the midsoles only. rgb: the photo's linear RGB, a: alpha,
    u: px per px of a 686 px standing figure. Socks: thick bright blobs (opened by sock_r) that reach his outline;
    midsoles: bright bands within sole_edge of his outline, not thin (opened by sole_thin), mostly shallow (median depth
    under sole_near), kept only to sole_near deep; each with real white in it (sole_hi). The flash and the laces
    (thin), a grey mesh (dim) and a blob inside the dark upper (not at the outline) stay dark."""
    mn, mx = rgb.min(2), rgb.max(2)
    neutral = (mx - mn) / (mx + 1e-4) < 0.35
    inside = a > 0.5
    low = ((mn > L["sole_lo"]) & neutral & inside).astype(np.uint8)
    keep = np.zeros(a.shape, bool)
    if not low.any():
        return keep
    high = (mn > L["sole_hi"]) & neutral & inside
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    big = 120 * u * u

    def bright(m, area):
        return (high & m).sum() > max(40 * u * u, 0.03 * area)

    closed = cv2.morphologyEx(low, cv2.MORPH_CLOSE, _disk(2 * u))
    sock = cv2.morphologyEx(closed, cv2.MORPH_OPEN, _disk(L["sock_r"] * u))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sock, 8)
    for i in range(1, n):
        m = lab == i
        if st[i, 4] > big and dist[m].min() <= 5 * u and bright(m, st[i, 4]):
            keep |= m
    near = L["sole_near"] * u
    band = cv2.morphologyEx(low * (dist <= L["sole_edge"] * u).astype(np.uint8), cv2.MORPH_OPEN,
                            _disk(L["sole_thin"] * u))
    n, lab, st, _ = cv2.connectedComponentsWithStats(band, 8)
    for i in range(1, n):
        m = lab == i
        if st[i, 4] > big and np.median(dist[m]) <= near and bright(m, st[i, 4]):
            keep |= m & (dist <= near)
    return keep


def body_colour(rgb: np.ndarray, L: dict, keep: np.ndarray | None = None) -> np.ndarray:
    """Linear RGB of the solid figure from the photo's own linear RGB: near-black with its shading, whites dimmed
    (only where `keep`, when given: shoe_whites)."""
    lum = luminance(rgb)
    v = L["floor"] + L["amp"] * np.minimum(lum / L["ref"], L["cap"]) ** L["gamma"]
    mn, mx = rgb.min(2), rgb.max(2)
    w = np.clip((mn - L["white_lo"]) / (L["white_hi"] - L["white_lo"]), 0, 1)
    w = w * w * (3 - 2 * w) * np.clip((0.45 - (mx - mn) / (mx + 1e-4)) / 0.2, 0, 1)  # neutral whites, not skin
    if keep is not None:
        w = w * keep
    v = v + L["white"] * w
    return (v[..., None] * np.float32(L["tint"])[None, None, :]).astype(np.float32)


def _paste(H: int, W: int, arr: np.ndarray, ox: int, oy: int) -> tuple[np.ndarray, tuple] | tuple[None, None]:
    """arr (h, w[, c]) put at (ox, oy) on an H x W canvas of zeros."""
    h, w = arr.shape[:2]
    sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
    cw, chh = min(w - sx0, W - dx0), min(h - sy0, H - dy0)
    if cw <= 0 or chh <= 0:
        return None, None
    out = np.zeros((H, W) + arr.shape[2:], np.float32)
    out[dy0:dy0 + chh, dx0:dx0 + cw] = arr[sy0:sy0 + chh, sx0:sx0 + cw]
    return out, (dy0, dy0 + chh, dx0, dx0 + cw)


def rim_light(plate: np.ndarray, A: np.ndarray, facing: float, L: dict, u: float = 1.0,
              glow: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(rim weight, rim colour) of a figure with alpha A over a linear plate: a band inside his outline, wider where
    the plate behind the edge is dark, strongest on the edges facing up and back (away from `facing`, +1 right)."""
    outside = 1 - A
    k = 3 * L["rim_dark_px"] * u + 2
    light = cv2.GaussianBlur(plate * outside[..., None], (0, 0), k) / \
        (cv2.GaussianBlur(outside, (0, 0), k)[..., None] + 1e-4)
    ll = luminance(light)
    dark = 1 - np.clip(ll / L["dark_lum"], 0, 1)  # 1 against a dark background
    # the band inside his edge from the smooth alpha itself (no binary mask, no pixel steps): A blurred by a sigma of
    # width / 1.65 is about one half on his edge and about 0.95 `width` px inside it; the band falls from 1 to 0 over
    # that, against light rim_px wide and against a dark background rim_dark_px
    def depth_band(w):
        D = cv2.GaussianBlur(A, (0, 0), max(w * u / 1.65, 0.3))
        return np.clip((1 - D) / 0.5, 0, 1)
    band = (depth_band(L["rim_px"]) * (1 - dark) + depth_band(L["rim_dark_px"]) * dark) ** L["rim_pow"] * A
    gy, gx = np.gradient(cv2.GaussianBlur(A, (0, 0), 2.0))
    n = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / n, -gy / n  # his outline's outward normal
    lx, ly = -facing * L["rim_back"], -L["rim_up"]
    ln = math.hypot(lx, ly)
    dirw = np.clip(L["rim_ambient"] + (1 - L["rim_ambient"]) * (nx * lx + ny * ly) / ln, 0, 1)
    if glow is None:
        glow = float(np.percentile(luminance(plate), 85))
    s = np.clip(L["rim_gain"] * ((1 - L["rim_glow"]) * ll + L["rim_glow"] * glow), L["rim_min"], L["rim_max"])
    col = (np.float32(L["rim_warm"])[None, None, :] * s[..., None] * (1 - L["rim_local"])
           + light * L["rim_gain"] * 0.5 * L["rim_local"])
    return (band * dirw).astype(np.float32), col.astype(np.float32)


def contact_shadow(out: np.ndarray, A: np.ndarray, feet_y: float, L: dict, u: float = 1.0) -> np.ndarray:
    """A soft dark ellipse on the surface under his feet when the lowest of him is within shadow_near of feet_y."""
    ys, xs = np.nonzero(A > 0.5)
    if not len(ys) or feet_y - ys.max() > L["shadow_near"] * u or ys.max() - feet_y > L["shadow_near"] * u:
        return out
    low = ys.max()
    fx = xs[ys > low - 0.10 * (low - ys.min())]
    cx, half = (fx.min() + fx.max()) / 2, max((fx.max() - fx.min()) / 2, 28 * u) * L["shadow_w"]
    H, W = A.shape
    y0, y1 = max(0, int(feet_y - 3 * L["shadow_h"] * u)), min(H, int(feet_y + 4 * L["shadow_h"] * u) + 1)
    x0, x1 = max(0, int(cx - 2 * half)), min(W, int(cx + 2 * half) + 1)
    if y1 <= y0 or x1 <= x0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    e = np.exp(-(((xx - cx) / half) ** 2 + ((yy - feet_y) / (L["shadow_h"] * u)) ** 2) * 2.2) * (yy >= feet_y - 2)
    out = out.copy()
    out[y0:y1, x0:x1] *= (1 - L["shadow"] * e)[..., None]
    return out


def plate_grain(plate: np.ndarray) -> float:
    """The plate's grain: the sRGB std of its fine detail (a 1 px high-pass), robust (median absolute deviation)."""
    s = lin2srgb(np.clip(plate, 0, 1)).mean(2)
    hp = s - cv2.GaussianBlur(s, (0, 0), 1.0)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def _rain_over(H: int, W: int, A: np.ndarray, rng: np.random.Generator, n: int, alpha, u: float) -> np.ndarray:
    """Thin rain streaks (sRGB add) over his box only, as the plate's (about 9 degrees off vertical)."""
    ys, xs = np.nonzero(A > 0.05)
    lay = np.zeros((H, W), np.float32)
    if not len(ys):
        return lay
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ang = math.radians(rng.normal(9, 1.5))
    for _ in range(int(n * (y1 - y0 + 1) / 1000 * max(1.0, (x1 - x0 + 1) / max(y1 - y0 + 1, 1)))):
        x, y = rng.uniform(x0 - 20, x1 + 20), rng.uniform(y0 - 40, y1)
        ln = rng.uniform(14, 40) * u
        cv2.line(lay, (int(x), int(y)), (int(x + ln * math.sin(ang)), int(y + ln * math.cos(ang))),
                 float(rng.uniform(*alpha)), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(lay, (0, 0), 0.7) * A


def refine_alpha(a: np.ndarray, L: dict, scale: float = 1.0) -> np.ndarray:
    """His edge at the output's scale (scale: output px per image px): a blur of about edge_px output px (at least a
    third of an image px) and a gentle contrast curve about one half (edge_gain), so the line is crisp, anti-aliased
    and smooth whatever the camera's scale."""
    sig = max(float(L["edge_px"]) / max(scale, 1e-3), 0.35)
    b = cv2.GaussianBlur(a, (0, 0), sig)
    return np.clip((b - 0.5) * float(L["edge_gain"]) + 0.5, 0, 1).astype(np.float32)


def _veil(plate: np.ndarray, sigma: float) -> np.ndarray:
    """The plate blurred by a wide sigma (the haze), done small: an even veil of its colour, no shapes."""
    H, W = plate.shape[:2]
    k = max(1, int(sigma // 8))
    if k == 1:
        return cv2.GaussianBlur(plate, (0, 0), sigma)
    small = cv2.resize(plate, (max(1, W // k), max(1, H // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def compose(out: np.ndarray, plate: np.ndarray, figs: list, L: dict | None = None, u: float = 1.0,
            feet_y: float | None = None, rng: np.random.Generator | None = None, scale: float = 1.0) -> np.ndarray:
    """Put true figures over a linear image. figs: [(rgb, alpha, ox, oy, facing)] in the image's px, oldest first;
    the last is the current photo, the ones before it after-images (L["after"], newest first). plate: the plate
    without him (its light makes the rim, its grain and haze go on him). feet_y: the ground line under him in the
    image's rows (None: no contact shadow). scale: output px per image px (the camera's), for the edge."""
    L = dict(TRUE_LOOK, **(L or {}))
    rng = rng or np.random.default_rng(0)
    H, W = out.shape[:2]
    out = out.copy()
    n = len(figs)
    for k, (rgb, a, ox, oy, facing) in enumerate(figs):
        age = n - 1 - k
        if age > 0:  # an after-image: a flat dark ghost
            if age > len(L["after"]):
                continue
            Ab, _ = _paste(H, W, cv2.GaussianBlur(refine_alpha(a, L, scale), (0, 0), L["after_blur"] * u)
                           * L["after"][age - 1], ox, oy)
            if Ab is not None:
                out = out * (1 - Ab[..., None]) + L["floor"] * Ab[..., None]
            continue
        A, _ = _paste(H, W, refine_alpha(a, L, scale), ox, oy)  # the smooth edge: everything below follows it
        if A is None:
            continue
        C, _ = _paste(H, W, rgb, ox, oy)
        K = None
        if L["soles_only"]:  # (the sprite is at street scale: a standing figure is 450 * u px, the sheets' 686)
            kk = cv2.GaussianBlur(shoe_whites(rgb, a, L, 450.0 * u / 686.0).astype(np.float32), (0, 0), 0.8 * u)
            K, _ = _paste(H, W, kk, ox, oy)
        if feet_y is not None:
            out = contact_shadow(out, A, feet_y, L, u)
        fig = body_colour(C, L, K)
        rim, rcol = rim_light(plate, A, facing, L, u)
        fig = fig * (1 - rim[..., None]) + rcol * rim[..., None]
        if L["haze"] > 0:  # a whisper of the air between him and the camera: the plate's soft colour
            fig = fig * (1 - L["haze"]) + _veil(plate, L["haze_blur"] * u) * L["haze"]
        # grain: the plate's own, in sRGB
        g = L["grain"] if L["grain"] is not None else plate_grain(plate)
        if g > 0:
            s = lin2srgb(np.clip(fig, 0, 1)) + rng.normal(0, g, (H, W)).astype(np.float32)[..., None]
            fig = srgb2lin(np.clip(s, 0, 1))
        if L["rain"] > 0:
            r = _rain_over(H, W, A, rng, int(L["rain"]), L["rain_alpha"], u)
            fig = srgb2lin(np.clip(lin2srgb(np.clip(fig, 0, 1)) + r[..., None] * np.float32([0.85, 0.9, 1.0]), 0, 1))
        As = A  # (refined: the plate's sharpness at the output's scale)
        if L["soft"] > 0:  # his own detail as soft as the plate's
            fig = cv2.GaussianBlur(fig, (0, 0), 0.6 * L["soft"] * u)
        out = out * (1 - As[..., None]) + fig * As[..., None]
        if L["bloom_k"] > 0:  # the rim's light bleeding a pixel or two past his edge
            gl = cv2.GaussianBlur(rcol * rim[..., None], (0, 0), L["bloom"] * u)
            out = out + L["bloom_k"] * gl * (1 - As[..., None])
    return out


# ------------------------------------------------------------------------------------------------ the prop

@template("true", prop=True)
def true_(c: Ctx) -> None:
    """From this time on he is true (prop; truelook.py): from (time), to (optional: the song's end), look (a dict of
    TRUE_LOOK overrides). Every pose in the stretch is a solid figure: no primer, no marks, no skins."""
    q0 = c.time("from")
    if q0 is None:
        raise ChoreoError("true: give from")
    q1 = c.time("to")
    if q1 is not None and q1 <= q0:
        raise ChoreoError("true: to must be after from")
    look = c.arg("look") or {}
    if not isinstance(look, dict):
        raise ChoreoError("true: look is a dict of TRUE_LOOK overrides")
    bad = [k for k in look if k not in TRUE_LOOK]
    if bad:
        raise ChoreoError(f"true: unknown look keys {', '.join(bad)} (truelook.TRUE_LOOK)")
    spans = list(getattr(c.ch, "true", None) or [])
    spans.append({"q0": float(c.song(q0)), "q1": float(c.song(q1)) if q1 is not None else math.inf,
                  "look": dict(look)})
    c.ch.true = spans


def span_of(ch, e) -> dict | None:
    """The true stretch holding event e, or None: a pose on the wall or in the air, or a paint stamp (a template's
    take-off splash, its drips: in the true stretch nothing paints, so `add` drops them)."""
    spans = getattr(ch, "true", None)
    if not spans or e.pose is None or e.layer not in ("wall", "air", "stamp"):
        return None
    for s in spans:
        if s["q0"] - 1e-6 <= e.q < s["q1"] - 1e-6:
            return s
    return None


def is_true(ch, e) -> bool:
    return span_of(ch, e) is not None


# ------------------------------------------------------------------------------------------------ render hooks

def add(hist, i: int, e, m: dict) -> None:
    """History._build's branch for a true pose: placed at the library's scale (no size jitter, no offset), shown
    from its photo; no primer, no marks, no droplets (true photos are not paint photos: the wall's trail keeps its
    paint). A paint stamp in the stretch is dropped (true photos paint nothing)."""
    if e.layer not in ("wall", "air"):
        return
    s = float(m["scale"]) * float(e.scale)
    hist._placed[i] = (s, 0.0, 0.0)
    d, x0, y0 = hist.place(i)
    h, w = d.shape
    tq = hist.__dict__.setdefault("true_q", {})
    tq.setdefault(e.q, []).append(i)
    hist.fig_box[e.q] = (x0, y0, x0 + w, y0 + h)


def place_rgba(hist, i: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(linear RGB, alpha, x0, y0) of true event i: its photo RGB (the drawing's <pose>.rgb.png) transformed exactly
    as History.place transforms its matte (scale, anchor, mirror, flip, rotation). A drawing without RGB gets a flat
    mid grey (a solid black figure with its rim)."""
    e = hist.events[i]
    lib = hist.lib
    s, jx, jy = hist._placed[i]
    d0 = hist._sprite(e)
    rf = lib.d / f"{e.pose}.rgb.png"
    bgr = cv2.imread(str(rf)) if rf.exists() else None
    if bgr is None or bgr.shape[:2] != d0.shape[:2]:
        rgb0 = np.full(d0.shape + (3,), 0.012, np.float32)
    else:
        rgb0 = srgb2lin(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    st = np.dstack([rgb0, d0]).astype(np.float32)
    sz = (max(1, int(round(d0.shape[1] * s))), max(1, int(round(d0.shape[0] * s))))
    # smooth scaling: area averaging down, bicubic up (never nearest: an enlarged drawing keeps a smooth edge)
    st = cv2.resize(st, sz, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    st[..., 3] = np.clip(st[..., 3], 0, 1)
    st[..., :3] = np.maximum(st[..., :3], 0)
    ax, ay = np.float32(lib.anchor(e.pose, "center" if e.layer == "stamp" else e.anchor)) * s
    if e.mirror:
        st = st[:, ::-1]
        ax = st.shape[1] - ax
    if e.flipv:
        st = st[::-1]
        ay = st.shape[0] - ay
    if e.rot:
        pad = int(max(st.shape[:2]) * 0.35)
        st = np.pad(st, ((pad, pad), (pad, pad), (0, 0)))
        ax, ay = ax + pad, ay + pad
        M = cv2.getRotationMatrix2D((float(ax), float(ay)), e.rot, 1.0)
        st = cv2.warpAffine(st, M, (st.shape[1], st.shape[0]), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(st[..., 3] > 0.01)
        cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        st = st[cy0:cy1, cx0:cx1]
        ax, ay = ax - cx0, ay - cy0
    st = np.ascontiguousarray(st)
    return st[..., :3], np.clip(st[..., 3], 0, 1), float(e.x - ax + jx), float(e.y - ay + jy)


def shown(hist, q: float) -> list[int]:
    """The true events on photo q, oldest first: the current photo's (the latest true photo at or before q) and,
    on the photo it first shows, the after-images of the photos before it."""
    tq = getattr(hist, "true_q", None)
    if not tq:
        return []
    keys = hist.__dict__.get("_true_keys")
    if keys is None or len(keys) != len(tq):
        keys = sorted(tq)
        hist._true_keys = keys
    j = bisect.bisect_right(keys, q + 1e-6) - 1
    if j < 0:
        return []
    cur = tq[keys[j]][-1:]
    if abs(keys[j] - q) > 1e-6:
        return cur  # held: no new photo, no after-images
    span = span_of(hist.ch, hist.events[cur[0]]) or {}
    n_after = len((dict(TRUE_LOOK, **span.get("look", {})))["after"])
    prev = [tq[keys[jj]][-1] for jj in range(max(0, j - n_after), j)]
    return prev + cur


def composite(hist, out: np.ndarray, plate: np.ndarray, t: float, q: float, x0: int, y0: int, w: int,
              h: int) -> np.ndarray:
    """Renderer.photo's hook: the true figure of photo q (with its after-images) over `out` (linear, the street box
    x0, y0, w, h); plate: the plate there without paint."""
    ids = shown(hist, q)
    if not ids:
        return out
    ch = hist.ch
    span = span_of(ch, hist.events[ids[-1]]) or {}
    L = dict(TRUE_LOOK, **span.get("look", {}))
    figs = []
    for i in ids:
        rgb, a, fx0, fy0 = place_rgba(hist, i)
        figs.append((rgb, a, int(round(fx0)) - x0, int(round(fy0)) - y0, -1.0 if hist.events[i].mirror else 1.0))
    e = hist.events[ids[-1]]
    rgb, a, ox, oy, _ = figs[-1]
    ys, xs = np.nonzero(a > 0.5)
    feet = None
    if len(ys):
        fx = x0 + ox + float(xs[ys >= ys.max() - 6].mean())
        feet = ch.scene.feet_at(fx) - y0
    rng = np.random.default_rng([int(ch.seed), int(round(q * 1000)), 41])
    # the output's scale: Renderer.photo's box is the crop plus 6 px a side, and the photo is 2048 px on its long side
    # (16:9 wide, 9:16 tall)
    scale = (2048.0 / max(w - 12, 1)) if w >= h else (1152.0 / max(w - 12, 1))
    return compose(out, plate, figs, L, ch.u, feet, rng, scale)


# ------------------------------------------------------------------------------------------------ the insert

@template("true_insert", prop=True)
def true_insert(c: Ctx) -> None:
    """A full-frame clip in place of the photos from `from` to `to` (prop; truelook.py): clip (project path), aspect
    ("16:9" or "9:16": the film it is for), camera ([cx, cy, cw, ch, rot]: the locked camera its frames were made at),
    frames ([first, last], default the whole clip, spread evenly over the stretch)."""
    q0, q1 = c.time("from"), c.time("to")
    if q0 is None or q1 is None or q1 <= q0:
        raise ChoreoError("true_insert: give from and to (to after from)")
    clip = c.require("clip")
    aspect = str(c.arg("aspect", "16:9"))
    if aspect not in ("16:9", "9:16"):
        raise ChoreoError("true_insert: aspect is 16:9 or 9:16")
    cam = c.require("camera")
    if not isinstance(cam, list | tuple) or len(cam) != 5:
        raise ChoreoError("true_insert: camera is [cx, cy, cw, ch, rot]")
    root = c.ch.lib.d.parents[1]
    f = root / str(clip)
    if not f.exists():
        raise ChoreoError(f"true_insert: no clip {clip}")
    cap = cv2.VideoCapture(str(f))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    fr = c.arg("frames") or [0, max(n - 1, 0)]
    if not (isinstance(fr, list | tuple) and len(fr) == 2 and 0 <= int(fr[0]) < int(fr[1]) < max(n, 1)):
        raise ChoreoError(f"true_insert: frames is [first, last] within the clip's {n}")
    g = c.ch.grid
    ins = list(getattr(c.ch, "true_inserts", None) or [])
    ins.append({"clip": str(f), "aspect": aspect, "camera": [float(v) for v in cam],
                "t0": float(g.t(c.song(q0))), "t1": float(g.t(c.song(q1))), "frames": [int(fr[0]), int(fr[1])]})
    c.ch.true_inserts = ins


@template("insert", prop=True)
def insert(c: Ctx) -> None:
    """A hard cut to a full-frame clip in place of the photos from `from` to `to` (prop; truelook.py; v8, the
    treatment's D: the straight-down drop): clip (project path), aspect ("16:9" or "9:16": the film it is for). The
    clip plays at its own speed, a frame per photo at its time: `sync: [frame, time]` puts that frame on that time
    (default: its first frame on `from`). It fills the photo (scaled to cover it, centred on `focus`: [x, y] shares of
    the clip frame, default [0.5, 0.5]), with the photo's rain over it and the MUTO look after it."""
    q0, q1 = c.time("from"), c.time("to")
    if q0 is None or q1 is None or q1 <= q0:
        raise ChoreoError("insert: give from and to (to after from)")
    clip = c.require("clip")
    aspect = str(c.arg("aspect", "16:9"))
    if aspect not in ("16:9", "9:16"):
        raise ChoreoError("insert: aspect is 16:9 or 9:16")
    f = c.ch.lib.d.parents[1] / str(clip)
    if not f.exists():
        raise ChoreoError(f"insert: no clip {clip}")
    cap = cv2.VideoCapture(str(f))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    g = c.ch.grid
    t0, t1 = float(g.t(c.song(q0))), float(g.t(c.song(q1)))
    sy = c.arg("sync")
    if sy is None:
        fs, ts = 0, t0
    else:
        if not (isinstance(sy, list | tuple) and len(sy) == 2):
            raise ChoreoError("insert: sync is [frame, time]")
        fs, ts = int(sy[0]), float(g.t(c.song(c.at(sy[1], key="sync"))))
    if not 0 <= fs < n:
        raise ChoreoError(f"insert: sync frame {fs} is not in the clip's {n}")
    first, last = fs + (t0 - ts) * fps, fs + (t1 - ts) * fps
    if first < -0.5 or last > n - 0.5:
        raise ChoreoError(f"insert: from {t0:.2f} to {t1:.2f} s needs frames {first:.0f} to {last:.0f}, the clip has "
                          f"0 to {n - 1}")
    fo = c.arg("focus") or [0.5, 0.5]
    ins = list(getattr(c.ch, "true_inserts", None) or [])
    ins.append({"clip": str(f), "aspect": aspect, "t0": t0, "t1": t1, "fill": True, "sync": [fs, ts], "fps": fps,
                "n": n, "focus": [float(fo[0]), float(fo[1])]})
    c.ch.true_inserts = ins


def _cover(img: np.ndarray, size: tuple[int, int], focus) -> np.ndarray:
    """img scaled to cover size (W, H), cropped round focus (shares of img)."""
    h, w = img.shape[:2]
    W, H = size
    s = max(W / w, H / h)
    nw, nh = max(W, int(round(w * s))), max(H, int(round(h * s)))
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0 = int(np.clip(focus[0] * nw - W / 2, 0, nw - W))
    y0 = int(np.clip(focus[1] * nh - H / 2, 0, nh - H))
    return big[y0:y0 + H, x0:x0 + W]


_FRAMES: dict = {}


def _clip_frame(path: str, idx: int) -> np.ndarray:
    """Frame idx of a clip (BGR), cached a few at a time."""
    key = (path, idx)
    if key not in _FRAMES:
        if len(_FRAMES) > 8:
            _FRAMES.pop(next(iter(_FRAMES)))
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"true_insert: cannot read frame {idx} of {path}")
        _FRAMES[key] = img
    return _FRAMES[key]


def _street_to_out(cam, size) -> np.ndarray:
    """3x3: street px to a photo's output px (render._to_out: rotation by cam.rot about its centre, scale out/cw)."""
    cx, cy, cw, _ch, rot = cam
    W, H = size
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot), W / float(cw))
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    return np.vstack([M, [0.0, 0.0, 1.0]])


def insert_frame_index(ins: dict, t: float) -> int:
    if "sync" in ins:  # (an `insert`: the clip at its own speed)
        fs, ts = ins["sync"]
        return int(np.clip(round(fs + (t - ts) * ins["fps"]), 0, ins["n"] - 1))
    f0, f1 = ins["frames"]
    u = (t - ins["t0"]) / max(ins["t1"] - ins["t0"], 1e-6)
    return int(round(f0 + np.clip(u, 0.0, 1.0) * (f1 - f0)))


def insert_photo(r, k: int, t: float, q: float, cam, size: tuple[int, int]) -> np.ndarray | None:
    """Renderer.photo's first step: within a `true_insert` stretch for this photo's aspect, the photo (sRGB float,
    size) is the insert's frame at its time, cropped to the photo's own camera, with the photo's rain over it. Where
    the photo's crop reaches past the insert's frame (the camera moves through the stretch), the engine's own photo
    fills it (the same plate and paint; the figure is inside the insert), feathered into the insert."""
    from .paintfx import rain

    if getattr(r, "_no_insert", False):
        return None
    ch = r.ch
    aspect = "16:9" if size[0] >= size[1] else "9:16"
    ins = next((s for s in (getattr(ch, "true_inserts", None) or [])
                if s["aspect"] == aspect and s["t0"] - 1e-6 <= t < s["t1"] - 1e-6), None)
    if ins is None:
        return None
    img = _clip_frame(ins["clip"], insert_frame_index(ins, t))
    h, w = img.shape[:2]
    W, H = int(size[0]), int(size[1])
    if ins.get("fill"):  # an `insert`: a cut to the clip, filling the photo
        rgb = cv2.cvtColor(_cover(img, (W, H), ins["focus"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rng = np.random.default_rng(int(1000 * int(ch.seed) + int(k)))
        return np.clip(rain(rgb, rng, strength=max(size) / 2048), 0, 1)
    m_in = _street_to_out(ins["camera"], (w, h))
    m_out = _street_to_out([cam.cx, cam.cy, cam.cw, cam.ch, cam.rot], size)
    M = (m_in @ np.linalg.inv(m_out))[:2]  # output px to insert px
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    out = cv2.warpAffine(rgb, M, (W, H), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    rng = np.random.default_rng(int(1000 * int(ch.seed) + int(k)))  # the photo's own rain (as Renderer.photo seeds it)
    out = np.clip(rain(out, rng, strength=max(size) / 2048), 0, 1)
    cov = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if cov.min() > 0.4:  # the insert covers the whole photo (its edge pixels half-covered by the resampling)
        return out
    f = max(3, int(0.02 * max(W, H)))  # the feather: 2% of the frame
    cov = cv2.GaussianBlur(cv2.erode(cov, np.ones((2 * f + 1, 2 * f + 1), np.uint8)), (0, 0), f / 2)[..., None]
    r._no_insert = True
    try:
        base = r.photo(k, t, q, cam, size)
    finally:
        r._no_insert = False
    return np.clip(base * (1 - cov) + out * cov, 0, 1)
