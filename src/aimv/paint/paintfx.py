"""Paint and weather helpers from prototype 3 (tests/proto-3-hybrid/hybrid.py, via prototype 6's paint3.py),
unchanged except that car_light takes the kerb line as a parameter.

noise, rough_patch (roller/brush primer patch), add_drips; ripple_warp, rain, puddle_rings, car_light.
"""

import math

import cv2
import numpy as np


def noise(h, w, cell, rng, aniso=(1, 1)):
    gh, gw = max(2, int(h / (cell * aniso[0])) + 2), max(2, int(w / (cell * aniso[1])) + 2)
    n = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def rough_patch(w, h, rng, vertical=False):
    """Opacity mask of a rough primer patch, built from overlapping wide brush/roller strokes: ragged stroke ends,
    bristle streaks, paint running out toward the end of each stroke, soft gaps between strokes."""
    if vertical:
        return np.ascontiguousarray(rough_patch(h, w, rng, vertical=False).T)
    out = np.zeros((h, w), np.float32)
    sh = rng.uniform(40, 80)
    n = max(1, int(math.ceil(h / (sh * 0.72))))
    xx = np.arange(w, dtype=np.float32)[None, :]
    for k in range(n):
        yc = (k + 0.5) * h / n + rng.normal(0, sh * 0.12)
        hh = sh * rng.uniform(0.85, 1.25)
        ya, yb = int(max(0, yc - hh / 2 - 6)), int(min(h, yc + hh / 2 + 6))
        if yb <= ya:
            continue
        rows = np.arange(ya, yb, dtype=np.float32)[:, None]
        # soft top/bottom of the stroke with a wobbling edge
        wob = noise(yb - ya, w, 40, rng, aniso=(1, 3)) * 5
        prof = np.clip((hh / 2 - np.abs(rows - yc) + wob) / 5.0, 0, 1)
        # ragged ends: start and end move per row (bristles), paint thins out toward the end of the stroke
        xs = w * rng.uniform(-0.06, 0.28) + noise(yb - ya, 1, 6, rng) * 12
        xe = w * (1 - rng.uniform(-0.06, 0.28)) + noise(yb - ya, 1, 6, rng) * 14
        if rng.random() < 0.5:  # stroke direction: which end runs dry
            fade = np.clip((xe - xx) / rng.uniform(60, 160), 0, 1) ** 0.6
        else:
            fade = np.clip((xx - xs) / rng.uniform(60, 160), 0, 1) ** 0.6
        inside = np.clip((xx - xs) / 4, 0, 1) * np.clip((xe - xx) / 4, 0, 1)
        bristle = np.clip(0.9 + 0.15 * noise(yb - ya, w, 2.2, rng, aniso=(1, 18)), 0, 1)
        dry = noise(yb - ya, w, 3, rng, aniso=(1, 12))
        stroke = prof * inside * bristle * np.where(dry > 1.6 - 0.9 * (1 - fade), 0.2, 1.0) * (0.75 + 0.25 * fade)
        stroke *= rng.uniform(0.8, 1.0)
        out[ya:yb] = 1 - (1 - out[ya:yb]) * (1 - stroke)
    return out.astype(np.float32)


def add_drips(m, rng, n, color_level, max_len, grow, width=(0.8, 2.4)):
    """Paint drips running down from lower edges of a mask (in place); grow in [0,1] scales their length; width:
    the range of drip half-widths (px)."""
    h, w = m.shape
    if grow <= 0 or n <= 0:
        return m
    solid = m > 0.5
    lower = solid & ~np.vstack([solid[1:], np.zeros((1, w), bool)])  # bottom edge pixels
    ys, xs = np.nonzero(lower)
    if len(xs) == 0:
        return m
    pick = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    out = np.pad(m, ((0, int(max_len) + 4), (0, 0)))
    for k in pick:
        x, y = xs[k], ys[k]
        L = int(rng.uniform(0.25, 1.0) * max_len * grow)
        wdt = rng.uniform(*width)
        for dy in range(L):
            f = 1 - dy / max(L, 1)
            r = max(0.6, wdt * (0.55 + 0.45 * f))
            x0, x1 = int(x - r), int(x + r) + 1
            out[y + dy, max(x0, 0):min(x1, w)] = np.maximum(out[y + dy, max(x0, 0):min(x1, w)], color_level * (0.7 + 0.3 * f))
        cy = y + L
        cv2.circle(out, (int(x), int(cy)), int(round(wdt * 1.1)), float(color_level), -1)  # bead at the end
    return out

def ripple_warp(img, y0_rows, rng, amp=2.5):
    """Displace the wet ground rows a little, differently in every photo (rain on puddles)."""
    h, w = img.shape[:2]
    if y0_rows >= h:
        return img
    rows = slice(max(y0_rows, 0), h)
    hh = h - max(y0_rows, 0)
    dx = noise(hh, w, 18, rng, aniso=(0.5, 1.6)) * amp
    dy = noise(hh, w, 14, rng, aniso=(0.5, 1.6)) * amp * 0.6
    yy, xx = np.mgrid[0:hh, 0:w].astype(np.float32)
    ramp = np.clip((yy) / 30.0, 0, 1)
    out = img.copy()
    out[rows] = cv2.remap(img[rows], xx + dx * ramp, yy + dy * ramp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def rain(frame, rng, strength=1.0):
    """Rain streaks for one photograph (1/50 s exposure look), in output space, sRGB float."""
    h, w = frame.shape[:2]
    layer = np.zeros((h, w), np.float32)
    ang = math.radians(rng.normal(9, 1.5))
    for n, (lmin, lmax), (amin, amax), thick in [(900, (14, 34), (0.05, 0.12), 1), (160, (40, 90), (0.06, 0.14), 2),
                                                  (18, (90, 190), (0.05, 0.1), 5)]:
        for _ in range(int(n * strength)):
            x, y = rng.uniform(-50, w + 50), rng.uniform(-50, h)
            L = rng.uniform(lmin, lmax)
            x2, y2 = x + L * math.sin(ang), y + L * math.cos(ang)
            cv2.line(layer, (int(x), int(y)), (int(x2), int(y2)), float(rng.uniform(amin, amax)), thick, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    big = cv2.GaussianBlur(layer, (0, 0), 3) * 0.6
    lum = frame.mean(-1)
    vis = 0.35 + 0.65 * (1 - lum)  # streaks read against the dark, less against lit paint
    return np.clip(frame + ((layer + big) * vis)[..., None] * np.float32([0.85, 0.9, 1.0]), 0, 1)


def puddle_rings(lin_crop, y_from, rng, scale):
    h, w = lin_crop.shape[:2]
    if y_from >= h - 4:
        return lin_crop
    layer = np.zeros((h, w), np.float32)
    for _ in range(int(70 * (h - y_from) / h + 10)):
        y = int(rng.uniform(max(y_from, 0), h))
        x = int(rng.uniform(0, w))
        r = rng.uniform(3, 14) * scale * (0.6 + 0.8 * (y - y_from) / max(h - y_from, 1))
        cv2.ellipse(layer, (x, y), (int(r), max(1, int(r * 0.28))), 0, 0, 360, float(rng.uniform(0.02, 0.06)), 1,
                    cv2.LINE_AA)
    return lin_crop + cv2.GaussianBlur(layer, (0, 0), 0.7)[..., None] * (0.4 + lin_crop * 2)


def car_light(lin_crop, x0, y0, rng, t, KERB_BOT=1420):
    """Now and then a car passes during the exposure: a light trail on the road and its light on the wall."""
    h, w = lin_crop.shape[:2]
    if rng.random() > 0.16:
        return lin_crop
    warm = rng.random() < 0.5
    col = np.float32([1.0, 0.75, 0.45]) if warm else np.float32([1.0, 0.08, 0.05])
    # wash on the wall (head lights) or a faint red glow (tail lights)
    cx = rng.uniform(0, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - cx) / (w * 0.35)) ** 2) - (((yy - h * 0.75) / (h * 0.6)) ** 2))
    lin_crop = lin_crop * (1 + (0.35 if warm else 0.12) * blob[..., None] * col)
    # trail on the road below the kerb
    road_y = KERB_BOT + 25 - y0
    if 0 < road_y < h:
        trail = np.zeros((h, w), np.float32)
        xa, xb = sorted(rng.uniform(-0.2 * w, 1.2 * w, 2))
        for k in range(2):
            yk = int(road_y + 12 * k + rng.normal(0, 3))
            cv2.line(trail, (int(xa), yk), (int(xb), yk), 1.0, 3, cv2.LINE_AA)
        trail = cv2.GaussianBlur(trail, (0, 0), 2.0) + cv2.GaussianBlur(trail, (0, 0), 9) * 0.6
        lin_crop = lin_crop + trail[..., None] * col * 0.9
    return lin_crop
