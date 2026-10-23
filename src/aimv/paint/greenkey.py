"""The green key: drawings and clips shot on flat chroma green (Full Stop's true runner: boards/23-true-sheet, the
true Wan clips). No engine imports, so the pose library (poses.py) and truelook.py can both use it.

`screen_matte` keys a frame: alpha from the CrCb distance to the screen's own green (its border median), with a guard
where green clearly beats red and blue (the screen's shadows), and the RGB despilled. `cut_sheet` cuts a sheet's
figures in reading order (for scripts: clips/finish/true_frames.py; the library splits sheets its own way).
"""

from __future__ import annotations

import cv2
import numpy as np


def screen_colour(bgr: np.ndarray, border: int = 16, exclude: np.ndarray | None = None) -> np.ndarray:
    """The screen's BGR: the median of a frame's border, without the `exclude` mask's pixels (the guide shapes: a
    roof block along the bottom and up a side can be most of the border)."""
    b = border
    m = np.zeros(bgr.shape[:2], bool)
    m[:b], m[-b:], m[:, :b], m[:, -b:] = True, True, True, True
    if exclude is not None and (m & ~exclude).sum() > 100:
        m &= ~exclude
    return np.median(bgr[m], 0).astype(np.uint8)


def screen_matte(bgr: np.ndarray, key=None, t0: float = 6.0, t1: float = 60.0, guard: tuple = (95.0, 125.0),
                 core: tuple = (14.0, 42.0, 60.0, 2)) -> tuple[np.ndarray, np.ndarray]:
    """(alpha float32 0..1, despilled BGR uint8) of a frame shot on flat chroma green. key: the screen's YCrCb
    (default: the median of a 16 px border). The edge is soft, as the camera saw it (no hard threshold): alpha ramps
    with the CrCb distance from the screen's colour from t0 (its noise) to t1 (about what a dark figure's pixels
    reach), so an edge pixel half figure, half screen is about half; where green beats red and blue by more than
    guard[0] (the screen's shadows) alpha fades to 0 by guard[1]. The figure's inside is solid: the hard key (core:
    the old ramp t0, t1 and green guard) eroded by core[3] px is 1, so dark folds and the white soles never turn see-through.
    Despill: green no higher than the mean of red and blue plus 6 (a black hoodie stays neutral)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    if key is None:
        b = 16
        border = np.concatenate([ycc[:b].reshape(-1, 3), ycc[-b:].reshape(-1, 3), ycc[:, :b].reshape(-1, 3),
                                 ycc[:, -b:].reshape(-1, 3)])
        key = np.median(border, 0)
    d = np.hypot(ycc[..., 1] - key[1], ycc[..., 2] - key[2])
    f = bgr.astype(np.float32)
    gd = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    g0, g1 = guard
    a = np.clip((d - t0) / (t1 - t0), 0, 1) * np.clip((g1 - gd) / max(g1 - g0, 1e-3), 0, 1)
    c0, c1, cg, r = core
    hard = ((np.clip((d - c0) / (c1 - c0), 0, 1) > 0.5) & (gd <= cg)).astype(np.uint8)
    if r > 0:
        hard = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1)))
    a = np.maximum(a, hard.astype(np.float32)).astype(np.float32)
    lim = (f[..., 0] + f[..., 2]) / 2 + 6
    f[..., 1] = np.minimum(f[..., 1], lim)
    return a, np.clip(f, 0, 255).astype(np.uint8)


def cut_sheet(bgr: np.ndarray, n: int, pad: int = 6) -> list[tuple[np.ndarray, np.ndarray, tuple]]:
    """A green sheet's n figures in reading order: [(alpha, BGR, (x0, y0, w, h))], each the largest components
    (closed 15 px) with what lies within 9 px of them, cropped with `pad`."""
    a, clean = screen_matte(bgr)
    m = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    k, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    main = sorted(range(1, k), key=lambda i: -st[i, 4])[:n]
    if len(main) < n:
        raise ValueError(f"cut_sheet: found {len(main)} figures of {n}")
    H = bgr.shape[0]
    rows = max(1, round(n / 3))
    main.sort(key=lambda i: (int(cen[i][1] * rows / H), cen[i][0]))
    out = []
    for i in main:
        own = cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        aa = a * own
        ys, xs = np.nonzero(aa > 0.02)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, bgr.shape[1])
        out.append((aa[y0:y1, x0:x1].astype(np.float32), clean[y0:y1, x0:x1],
                    (int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
    return out
