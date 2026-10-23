"""The pose library: sprite sheets split into paint-density mattes with anchors, tags and roles (prototype 6's
sprites6.py).

Spec: projects/<p>/paint/poses/poses.spec.json

    {"version": 1,
     "ref_height": 450,                 body height in street px that a scale of 1 refers to
     "reference": "R1",                 sheets without a "height" match this sheet's median painted area
     "sheets": {"R1": {"path": "boards/05-sprites/R1.png", "n": 8, "cols": 4, "kind": "pose", "height": 670},
                "PK1": {"path": "boards/08-parkour/PK1.png", "n": 6, "cols": 3, "kind": "pose",
                        "boost": [1.12, 0.72]},                      density gain and gamma (dry-brushed sheets)
                "SP1": {"path": "boards/08-parkour/SP1.png", "n": 6, "cols": 3, "kind": "stamp"}},
     "cycles": {"run": [["R1-1", "R1-3", "R1-7", "R1-4"], ...], "walk": [["PK4-5"], ["PK4-6"]]},
     "phases": {"run": ["contact", "passing", "toeoff", "flight"]},
     "roles": {"crouch": "K1-1", "flight": ["PK1-2", "PK1-3", "PK1-4"], ...},
     "tags": {"PK1-1": {"move": "leap", "phase": "takeoff"}}}

Pose names are <sheet>-<k>, k counted along rows from the top left. The density matte is 1 minus pose/primer
reflectance in linear light, so brush texture and lighter folds stay partial (the multiply look). Scale: a sheet with
a "height" (standing figure height in sheet px) gets ref_height / height; others are put on the reference sheet's
scale by matching the median painted area of their poses (body area barely changes with pose), so every pose lands at
the same body size. Anchors (sprite px): cx (torso x: mean x of the upper 55%), top, bottom (lowest foot), left,
right, bottomright (hands planted), bbox. Templates use roles (crouch, takeoff, flight ...) so a project can map
them to its own drawings; tags carry the move, phase and cycle index of each pose.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..util import read_json, write_json

ANCHORS = ("foot", "bottom", "top", "right", "left", "br", "center", "cxc", "bc")


def density(img: np.ndarray) -> np.ndarray:
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bg = cv2.GaussianBlur(cv2.dilate(lum, np.ones((41, 41), np.uint8)), (0, 0), 25)
    lin = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    d = 1 - lin(lum) / np.maximum(lin(bg), 0.2)
    return np.clip((d - 0.12) / 0.88, 0, 1).astype(np.float32)


def split(img: np.ndarray, n: int, cols: int, kind: str, name: str = "", screen: str | None = None,
          boxes: list | None = None) -> list[np.ndarray]:
    """Cut one sheet into its n poses (largest painted components; drips and loose hands join their figure) or, for
    stamps, n fixed grid cells (splashes are clouds of droplets, not single components). screen: "green" for a sheet
    shot on chroma green (greenkey.screen_matte instead of the paint density: the true figure). boxes: a list that
    gets each pose's crop (y0, y1, x0, x1) in sheet px."""
    H, W = img.shape[:2]
    if screen == "green":
        from .greenkey import screen_matte
        d = screen_matte(img)[0]
    elif screen:
        raise SystemExit(f"sheet {name}: screen is green or nothing, not {screen!r}")
    else:
        d = density(img)
    if kind == "stamp":
        rows = n // cols
        cells = []
        for r in range(rows):
            for c in range(cols):
                x0, x1 = int(c * W / cols), int((c + 1) * W / cols)
                y0, y1 = int(r * H / rows), int((r + 1) * H / rows)
                dd = d[y0:y1, x0:x1].copy()
                dd[dd < 0.06] = 0
                ys, xs = np.nonzero(dd > 0.1)
                pad = 12
                a0, a1 = max(ys.min() - pad, 0), min(ys.max() + pad, dd.shape[0])
                b0, b1 = max(xs.min() - pad, 0), min(xs.max() + pad, dd.shape[1])
                cells.append(dd[a0:a1, b0:b1])
        return cells
    m = (d > 0.25).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    k, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    order = sorted(range(1, k), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    main = order[:n]
    if len(main) < n or stats[main[-1], cv2.CC_STAT_AREA] <= 15000:
        raise SystemExit(f"sheet {name}: found {len(main)} figures of {n} (areas "
                         f"{[int(stats[i, 4]) for i in order[:n + 2]]}); is n right, are the figures separated?")
    owner = {i: i for i in main}
    for i in order[n:]:
        if stats[i, cv2.CC_STAT_AREA] < 150:
            continue
        j = min(main, key=lambda j: np.hypot(*(cent[i] - cent[j])))
        bx, by, bw, bh = stats[j, :4]
        gx = max(bx - cent[i][0], 0, cent[i][0] - (bx + bw))
        gy = max(by - cent[i][1], 0, cent[i][1] - (by + bh))
        if np.hypot(gx, gy) < 60:  # only bits right next to the figure (drips, a loose hand)
            owner[i] = j
    nrows = max(1, n // max(cols, 1))
    rows_of = sorted(main, key=lambda i: (int(cent[i][1] * nrows / H), cent[i][0]))  # reading order
    out = []
    for i in rows_of:
        own = np.isin(lab, [j for j, o in owner.items() if o == i]).astype(np.uint8)
        own = cv2.dilate(own, np.ones((25, 25), np.uint8))
        own = own * (cv2.GaussianBlur((d > 0.3).astype(np.float32), (0, 0), 3) > 0.02)  # no faint primer specks
        ys, xs = np.nonzero(own)
        pad = 8
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
        out.append(d[y0:y1, x0:x1] * own[y0:y1, x0:x1])
        if boxes is not None:
            boxes.append((int(y0), int(y1), int(x0), int(x1)))
    return out


# ---------------------------------------------------------------------------------------------- video sprites

def key_color(img: np.ndarray, rgb, tol: float) -> np.ndarray:
    """Pixels of a guide block or pole (a flat pale colour, RGB) in a BGR frame, a little grown."""
    d = np.sqrt(((img.astype(np.float32) - np.float32(rgb[::-1])) ** 2).sum(2))
    m = (d < tol).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0


def video_frames(proj_root: Path, v: dict) -> list[tuple[int, float, np.ndarray]]:
    """(frame number, time, BGR frame) of a clip: v["path"], optional "crop" [x0, y0, x1, y1], "until" seconds,
    "step" (every n-th frame)."""
    cap = cv2.VideoCapture(str(proj_root / v["path"]))
    if not cap.isOpened():
        raise SystemExit(f"video {v['path']}: cannot open")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, f = [], 0
    step = int(v.get("step", 1))
    until = float(v.get("until", 1e9))
    while True:
        ok, img = cap.read()
        if not ok or f / fps > until + 1e-6:
            break
        if f % step == 0:
            if v.get("crop"):
                x0, y0, x1, y1 = v["crop"]
                img = img[y0:y1, x0:x1]
            out.append((f, f / fps, img))
        f += 1
    cap.release()
    return out


def _same(prev: tuple, mat: np.ndarray, x0: int, y0: int, iou_same: float = 0.85, iou_still: float = 0.7) -> bool:
    """Is a matte the same drawing as the previous frame's (a doubled frame with compression noise)? iou_same,
    iou_still: the overlaps above which it is (still: with the centroid within 2 px); a clip with subtle motion (a
    hood turning, a head lifting in a crouch) needs them higher, or its motion is dropped as doubles."""
    _, _, pm, px, py = prev
    if abs(px - x0) > 2 or abs(py - y0) > 2 or abs(pm.shape[0] - mat.shape[0]) > 3 or abs(pm.shape[1] - mat.shape[1]) > 3:
        return False
    h, w = min(pm.shape[0], mat.shape[0]), min(pm.shape[1], mat.shape[1])
    a, b = pm[:h, :w] > 0.3, mat[:h, :w] > 0.3
    # a new pose moves limbs by tens of px (overlap well under 0.6, the centroid moves); a doubled frame differs
    # only by noise (overlap about 0.9, the centroid within a pixel or two)
    iou = (a & b).sum() / max((a | b).sum(), 1)
    ya, xa = np.nonzero(pm > 0.3)
    yb, xb = np.nonzero(mat > 0.3)
    moved = np.hypot(xa.mean() + px - xb.mean() - x0, ya.mean() + py - yb.mean() - y0) if len(xa) and len(xb) else 99
    return iou > iou_same or (iou > iou_still and moved < 2.0)


def ground_feet(dd: np.ndarray, ox: int, oy: int, ground: float, near: float = 14.0) -> list[list[float]]:
    """The parts of a video frame's figure on the ground line (x0, x1 in clip px): a planted foot, the toe of one
    landing or leaving."""
    ys, xs = np.nonzero(dd > 0.4)
    sel = ys + oy >= ground - near
    if not sel.any():
        return []
    fx = np.sort(xs[sel] + ox)
    groups = np.split(fx, np.nonzero(np.diff(fx) > 25)[0] + 1)
    return [[float(g.min()), float(g.max())] for g in groups if len(g) > 15]


def stride(track: list[dict], f0: int, f1: int, m0: int, m1: int, steps: int) -> dict:
    """Stride-true travel of a run cycle from a clip (frames f0 to f1: from one touch-down to just before the same
    foot's next): the foot plants (a foot on the ground staying within 70 px over consecutive frames) measured over
    frames m0 to m1 give the step (the median distance between consecutive plants); each cycle frame's place in the
    cycle (clip px from the cycle's first plant) is the plant's place (k steps) plus the torso's offset from the
    planted foot on stance frames, and moves on evenly in time on flight frames (to the next cycle's start)."""
    def plants(rows):
        out = []
        for r in rows:
            if not r.get("feet"):
                continue
            c = max(r["feet"], key=lambda ab: ab[1] - ab[0])
            cx = (c[0] + c[1]) / 2
            if out and abs(cx - out[-1]["xs"][0]) < 70 and r["f"] - out[-1]["fs"][-1] <= 4:
                out[-1]["xs"].append(cx)
                out[-1]["fs"].append(r["f"])
            else:
                out.append({"xs": [cx], "fs": [r["f"]]})
        for g in out:
            g["x"] = float(np.median(g["xs"]))
        return out
    meas = plants([r for r in track if m0 <= r["f"] <= m1])
    xs = [g["x"] for g in meas]
    step = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
    cyc = [r for r in track if f0 <= r["f"] <= f1]
    pl = plants(cyc)
    # the cycle's own steps (to the same foot's next plant, when it is measured): its drawings' strides
    first = next((i for i, g in enumerate(meas) if pl and set(g["fs"]) & set(pl[0]["fs"])), None)
    f_next = f1 + 1  # the frame that starts the next cycle (the same foot's next touch-down)
    if first is not None and first + steps < len(meas):
        step = (meas[first + steps]["x"] - meas[first]["x"]) / steps
        f_next = meas[first + steps]["fs"][0]
    T: dict[int, float] = {}
    for k, g in enumerate(pl[:steps]):
        for f in g["fs"]:
            r = next(r_ for r_ in cyc if r_["f"] == f)
            T[f] = k * step + (r["x"] - g["x"]) - (cyc[0]["x"] - pl[0]["x"] if pl else 0.0)
    known = sorted(T)
    period = steps * step
    out = []
    for r in cyc:
        if r["f"] in T:
            out.append(T[r["f"]])
            continue
        a = max([f for f in known if f < r["f"]], default=None)
        b = min([f for f in known if f > r["f"]], default=None)
        ta = T[a] if a is not None else 0.0
        fa = a if a is not None else f0
        tb, fb = (T[b], b) if b is not None else (T[known[0]] + period if known else period,
                                                  f_next + known[0] - f0 if known else f_next)
        out.append(ta + (tb - ta) * (r["f"] - fa) / max(fb - fa, 1))
    out = list(np.maximum.accumulate(np.array(out))) if out else out
    return {"step": round(step, 2), "period": round(period, 2), "travel": [round(float(x), 2) for x in out],
            "plants": [round(x, 1) for x in xs], "plant_frames": [g["fs"] for g in meas],
            "steps_measured": [round(float(x), 1) for x in np.diff(xs)], "period_f": int(f_next - f0)}


def time_travel(nums: list[int], f1: int | None, step: float, steps: int) -> dict:
    """A run cycle's travel by time (`"travel": "time"`: a clip shot with a tracking camera, where neither the feet
    nor the torso travel): `steps` steps of `step` clip px over the cycle, each kept frame (nums, frame numbers) at
    its share of the cycle's frames, from its first to the frame after f1 (its range's end; default the last kept
    frame), so the frames play in order and the stride stays even."""
    if step <= 0 or not nums:
        raise SystemExit("travel: time needs step above 0 and the cycle's frames")
    period = float(step) * int(steps)
    pf = (int(f1) if f1 is not None else nums[-1]) + 1 - nums[0]
    return {"step": float(step), "period": round(period, 2), "period_f": int(pf),
            "travel": [round(period * (fn - nums[0]) / pf, 2) for fn in nums]}


def _fill_pole(dd: np.ndarray, pole: list) -> np.ndarray:
    """His paint across a pole that stood in front of him (a vertical guide shape): on every row with paint just
    left and just right of it, the gap is filled from one side to the other, so the figure is whole and a street pipe
    narrower than the clip's pole hides only what it covers. Bars (horizontal) are left alone."""
    x0, y0, x1, y1 = pole
    if y1 - y0 < 2 * (x1 - x0):
        return dd
    a, b = max(x0 - 1, 0), min(x1 + 1, dd.shape[1] - 1)
    lv = dd[:, max(a - 5, 0):a + 1].max(1)
    rv = dd[:, b:b + 6].max(1)
    rows = np.nonzero((lv > 0.3) & (rv > 0.3))[0]
    if not len(rows):
        return dd
    out = dd.copy()
    w = np.linspace(0, 1, b - a + 1, dtype=np.float32)[None, :]
    out[rows, a:b + 1] = lv[rows, None] * (1 - w) + rv[rows, None] * w
    return out


def _bridged(m: np.ndarray, km: np.ndarray, lab: np.ndarray):
    """A pole or bar in front of him splits the figure: its paint components that the guide shape joins (pieces on
    either side of it, within its width) become one. Returns connectedComponentsWithStats' (k, labels, stats,
    centroids) for the merged components (paint pixels only)."""
    ys_, xs_ = np.nonzero(km)
    thin = min(xs_.max() - xs_.min(), ys_.max() - ys_.min()) + 1  # the pole's width or the bar's thickness
    k_ = int(2 * thin + 31)
    bridge = km & (cv2.dilate(m, np.ones((k_, k_), np.uint8)) > 0)  # the stretch of the shape next to his paint
    _, lab_b = cv2.connectedComponents(np.maximum(m, bridge.astype(np.uint8)), connectivity=8)
    ys, xs = np.nonzero(lab > 0)
    L = lab[ys, xs]
    grp = np.zeros(int(lab.max()) + 1, np.int64)
    grp[L] = lab_b[ys, xs]  # every piece's bridged component
    uniq = np.unique(grp[1:])
    remap = np.zeros_like(grp)
    remap[1:] = np.searchsorted(uniq, grp[1:]) + 1
    lab = remap[lab].astype(np.int32)
    L = lab[ys, xs]
    k = len(uniq) + 1
    area = np.bincount(L, minlength=k)
    x0, y0 = np.full(k, 10 ** 9), np.full(k, 10 ** 9)
    x1, y1 = np.full(k, -1), np.full(k, -1)
    np.minimum.at(x0, L, xs)
    np.minimum.at(y0, L, ys)
    np.maximum.at(x1, L, xs)
    np.maximum.at(y1, L, ys)
    stats = np.stack([x0, y0, x1 - x0 + 1, y1 - y0 + 1, area], 1).astype(np.int64)
    stats[0] = 0
    cent = np.zeros((k, 2))
    nz = area > 0
    cent[nz, 0] = np.bincount(L, weights=xs, minlength=k)[nz] / area[nz]
    cent[nz, 1] = np.bincount(L, weights=ys, minlength=k)[nz] / area[nz]
    return k, lab, stats, cent


def video_split(frames: list, v: dict, name: str, rgbs: dict | None = None) -> tuple[list, dict]:
    """Matte every frame of a clip and keep the figure: the guide block keyed out (painted over with the wall), the
    paint density as for sheets, then only the figure's own blob (the largest dark component near where it was in
    the frame before, plus small bits touching it: loose hands, a drip), so the model's scribbles, footprints and
    smears elsewhere are dropped. Returns ([(frame, t, matte, ox, oy)], info) with the block's box and the ground
    line (the block's bottom) in frame px. A clip with `"screen": "green"` (shot on chroma green: the true figure) is
    keyed by greenkey.screen_matte instead of the paint density, its guide shapes painted with the screen's colour,
    and rgbs (a dict) gets each kept frame's despilled colour crop by frame number."""
    green = v.get("screen") == "green"
    if v.get("screen") and not green:
        raise SystemExit(f"video {name}: screen is green or nothing, not {v['screen']!r}")
    if green:
        from .greenkey import screen_colour, screen_matte
    block = None
    shapes = []  # the guide shapes one by one (a pole and the beam it leads to, a row of posts)
    out = []
    prev_c = None
    gap = float(v.get("join", 18))
    for f, t, img in frames:
        img = img.copy()
        km = None
        if v.get("key"):
            km = key_color(img, v["key"], float(v.get("tol", 42)))
            if km.any():
                ys, xs = np.nonzero(km)
                if block is None and km.sum() > 2000:
                    block = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    nk, _, kst, _ = cv2.connectedComponentsWithStats(km.astype(np.uint8), 8)
                    shapes = [[int(kst[i, 0]), int(kst[i, 1]), int(kst[i, 0] + kst[i, 2]), int(kst[i, 1] + kst[i, 3])]
                              for i in range(1, nk) if kst[i, cv2.CC_STAT_AREA] > 500]
                if green:  # the guide painted with the screen's own green, so the key takes it out with the rest
                    wall = screen_colour(img, exclude=km)  # (the guides out of the border median)
                else:
                    wall = np.median(img[(~km) & (img.mean(2) > 150)], axis=0) if ((~km) & (img.mean(2) > 150)).any() \
                        else np.float32([235, 235, 235])
                img[km] = wall.astype(np.uint8)
        rgb_full = None
        if green:
            d, rgb_full = screen_matte(img)
        else:
            d = density(img)
        if v.get("below"):  # nothing below this line (footprints on the ground)
            d[int(v["below"]):] = 0
        m = (d > 0.25).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        k, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        if k <= 1:
            continue
        if v.get("bridge") and km is not None and km.any() and k > 2:
            k, lab, stats, cent = _bridged(m, km, lab)
        cand = [i for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] > 400]
        if not cand:
            continue
        def score(i):
            a = stats[i, cv2.CC_STAT_AREA]
            if prev_c is None:
                return a
            return a / (1 + (np.hypot(*(cent[i] - prev_c)) / 150.0) ** 2)
        main = max(cand, key=score)
        bx, by, bw, bh = stats[main, :4]
        keep = [main]
        for i in range(1, k):
            if i == main or stats[i, cv2.CC_STAT_AREA] < 30:
                continue
            x, y, w, h = stats[i, :4]
            gx = max(bx - (x + w), 0, x - (bx + bw))
            gy = max(by - (y + h), 0, y - (by + bh))
            flat = v.get("flat") and h < 16 and y > by + bh - 30  # a footprint at the feet
            faint = float(d[lab == i].mean()) < 0.5  # a grey scribble, not the figure's paint
            if np.hypot(gx, gy) < gap and stats[i, cv2.CC_STAT_AREA] < 0.25 * stats[main, cv2.CC_STAT_AREA] \
                    and not flat and not faint:
                keep.append(i)
        own = np.isin(lab, keep).astype(np.uint8)
        own = cv2.dilate(own, np.ones((7, 7), np.uint8))
        dd = d * own
        if v.get("fill_pole") and km is not None and shapes:
            dd = _fill_pole(dd, max(shapes, key=lambda b_: (b_[3] - b_[1]) / max(b_[2] - b_[0], 1)))
            own = np.maximum(own, (dd > 0.02).astype(own.dtype))
        ys, xs = np.nonzero(own)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, d.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, d.shape[1])
        mat = dd[y0:y1, x0:x1]
        prev_c = cent[main]
        dd_ = v.get("dedupe") or {}
        if out and _same(out[-1], mat, int(x0), int(y0), float(dd_.get("iou", 0.85)), float(dd_.get("still", 0.7))):
            continue  # a repeated frame (the model doubled it, give or take a little noise): keep the first
        out.append((f, t, mat, int(x0), int(y0)))
        if rgbs is not None and rgb_full is not None:
            rgbs[f] = rgb_full[y0:y1, x0:x1].copy()
    gr = v.get("ground")
    if v.get("block"):  # set by hand (a clip whose roof guide spans the frame: the box is the block)
        block = [int(b) for b in v["block"]]
    if v.get("shapes"):
        shapes = [[int(b) for b in s_] for s_ in v["shapes"]]
    info = {"block": block, "ground": float(gr if isinstance(gr, int | float) else
                                            block[3] if block else frames[0][2].shape[0] - 1),
            "size": [int(frames[0][2].shape[1]), int(frames[0][2].shape[0])], "shapes": shapes}
    return out, info


def anchors(dd: np.ndarray) -> dict:
    ys, xs = np.nonzero(dd > 0.5)
    top, bot = int(np.percentile(ys, 0.15)), int(np.percentile(ys, 99.6))  # robust to specks and drips
    up = ys < top + 0.55 * (bot - top)
    sel = lambda m: [float(xs[m].mean()), float(ys[m].mean())]
    low = ys >= top + 0.7 * (bot - top)
    return {
        "w": int(dd.shape[1]), "h": int(dd.shape[0]),
        "area": float((dd > 0.3).sum()),
        "cx": float(xs[up].mean()), "cy": float(ys.mean()),
        "top": sel(ys <= top + 6), "bottom": sel((ys >= bot - 8) & (ys <= bot)),
        "bottomright": sel(low & (xs >= np.percentile(xs[low], 97))),
        "left": sel(xs <= xs.min() + 6), "right": sel(xs >= xs.max() - 6),
        "bbox": [int(xs.min()), int(top), int(xs.max()), int(bot)],
    }


def load_spec(d: Path) -> dict:
    f = d / "poses.spec.json"
    if not f.exists():
        raise SystemExit(f"No pose spec: {f}")
    s = read_json(f)
    s.setdefault("ref_height", 450.0)
    s.setdefault("cycles", {})
    s.setdefault("phases", {})
    s.setdefault("roles", {})
    s.setdefault("tags", {})
    s.setdefault("sequences", {})
    s.setdefault("motion", {})
    s.setdefault("stride_phase", {})
    return s


def build(proj_root: Path, d: Path, log=print) -> dict:
    """`aimv paint poses`: split every sheet, store mattes and poses.json, write the contact sheet."""
    spec = load_spec(d)
    meta, areas = {}, {}
    for name, sh in spec["sheets"].items():
        img = cv2.imread(str(proj_root / sh["path"]))
        if img is None:
            raise SystemExit(f"sheet {name}: cannot read {sh['path']}")
        boxes: list = []
        parts = split(img, int(sh["n"]), int(sh["cols"]), sh.get("kind", "pose"), name, sh.get("screen"), boxes)
        rgb = None
        if sh.get("screen"):  # the true figure's colour (truelook.py): despilled, the matte's own crop
            from .greenkey import screen_matte
            rgb = screen_matte(img)[1]
        for k, dd in enumerate(parts):
            pname = f"{name}-{k + 1}"
            cv2.imwrite(str(d / f"{pname}.png"), (dd * 255).astype(np.uint8))
            meta[pname] = {"sheet": name, "kind": sh.get("kind", "pose"), **anchors(dd)}
            if rgb is not None:
                y0, y1, x0, x1 = boxes[k]
                cv2.imwrite(str(d / f"{pname}.rgb.png"), rgb[y0:y1, x0:x1])
                meta[pname]["screen"] = str(sh["screen"])
            if sh.get("boost"):
                meta[pname]["boost"] = list(sh["boost"])
            areas.setdefault(name, []).append(meta[pname]["area"])
        log(f"poses: {name}: {len(parts)} {sh.get('kind', 'pose')}s")
    videos = {}
    for name, v in (spec.get("video") or {}).items():
        if not isinstance(v, dict):  # a note
            continue
        frames = video_frames(proj_root, v)
        fps = float(v.get("fps") or (1 / (frames[1][1] - frames[0][1]) if len(frames) > 1 else 30.0))
        rgbs: dict | None = {} if v.get("screen") else None
        mats, info = video_split(frames, v, name, rgbs)
        for rs in v.get("rescale") or []:
            # the model drifted (the figure shrank or grew from a frame on): those frames scaled back by `factor`
            # about a clip point (x, y), so the clip keeps one body size
            k, (ax, ay) = float(rs["factor"]), rs["about"]
            for i, (f, t, dd, ox, oy) in enumerate(mats):
                if rs.get("from", 0) <= f <= rs.get("to", 10 ** 9):
                    d2 = cv2.resize(dd, (max(1, int(round(dd.shape[1] * k))), max(1, int(round(dd.shape[0] * k)))),
                                    interpolation=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA)
                    mats[i] = (f, t, np.clip(d2, 0, 1), int(round(ax + (ox - ax) * k)), int(round(ay + (oy - ay) * k)))
                    if rgbs is not None and f in rgbs:  # its colour with it
                        rgbs[f] = cv2.resize(rgbs[f], (d2.shape[1], d2.shape[0]),
                                             interpolation=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA)
        track = []
        for f, t, dd, ox, oy in mats:
            pname = f"{name}-{f:03d}"
            cv2.imwrite(str(d / f"{pname}.png"), (np.clip(dd, 0, 1) * 255).astype(np.uint8))
            a = anchors(dd)
            meta[pname] = {"sheet": name, "kind": "pose", **a,
                           "video": {"src": name, "f": f, "t": round(t, 4), "ox": ox, "oy": oy}}
            if rgbs is not None and f in rgbs:  # the true figure's colour (truelook.py), the matte's own crop
                cv2.imwrite(str(d / f"{pname}.rgb.png"), rgbs[f])
                meta[pname]["screen"] = str(v["screen"])
            track.append({"f": f, "t": round(t, 4), "name": pname, "x": ox + a["cx"],
                          "y": oy + (a["bbox"][1] + a["bbox"][3]) / 2, "low": oy + a["bbox"][3],
                          "h": a["bbox"][3] - a["bbox"][1]})
        if (not info["block"] and "ground" not in v) or v.get("ground") == "feet":
            # no guide block (or a block off the ground: a meter cabinet on the wall, a pole, a bar): the ground is
            # where his planted feet are
            info["ground"] = float(np.percentile([r["low"] for r in track], 90))
        for (f, t, dd, ox, oy), r in zip(mats, track):
            r["feet"] = ground_feet(dd, ox, oy, float(info["ground"]))
        hr = v.get("height_frames", [0, 10])
        hs = [r["h"] for r in track if hr[0] <= r["f"] < hr[1]] or [r["h"] for r in track]
        # a running figure's bbox is about 425 px on a 450 px body (the R1 drawings); height_px says what those
        # frames measure on a 450 px body when they are not running (an upright walk: about 440)
        scale = float(spec["ref_height"]) * (float(v.get("height_px", 425)) / 450) / float(np.median(hs))
        if v.get("scale"):  # set by its guide (a box drawn at the street box's size): street px per clip px
            scale = float(v["scale"])
        for r in track:
            meta[r["name"]]["scale"] = scale
        videos[name] = {"path": v["path"], "fps": fps, "scale": scale, **info, "frames": track}
        if v.get("boxed"):  # a front view inside a box prop: the box sets his size (the canon's area does not apply)
            videos[name]["boxed"] = True
        log(f"poses: {name}: {len(track)} video frames ({v['path']}), figure scale {scale:.3f}")
    ref = spec.get("reference") or next(n for n, s in spec["sheets"].items() if s.get("height"))
    s_ref = float(spec["ref_height"]) / float(spec["sheets"][ref]["height"])
    med_ref = float(np.median(areas[ref]))
    for pname, m in meta.items():
        if "video" in m:
            continue
        sh = spec["sheets"][m["sheet"]]
        if m["kind"] != "pose":
            m["scale"] = 1.0
        elif sh.get("height"):
            m["scale"] = float(spec["ref_height"]) / float(sh["height"])
        else:
            m["scale"] = s_ref * float(np.sqrt(med_ref / np.median(areas[m["sheet"]])))
    # tags: cycles give move, phase and cycle index; roles add their role names; explicit tags win
    for move, cycles in spec["cycles"].items():
        phases = spec["phases"].get(move, [])
        for ci, cyc in enumerate(cycles):
            for pi, pn in enumerate(cyc):
                if pn in meta:
                    t = meta[pn].setdefault("tags", {})
                    t.setdefault("move", move)
                    if pi < len(phases):
                        t.setdefault("phase", phases[pi])
                    t.setdefault("cycle", [])
                    t["cycle"].append([ci, pi])
    for role, names in spec["roles"].items():
        for pn in ([names] if isinstance(names, str) else names):
            if pn in meta:
                meta[pn].setdefault("tags", {}).setdefault("roles", []).append(role)
    for pn, t in spec["tags"].items():
        if pn in meta:
            meta[pn].setdefault("tags", {}).update(t)
    unknown = [pn for role, names in spec["roles"].items() for pn in ([names] if isinstance(names, str) else names)
               if pn not in meta]
    unknown += [pn for cyc in spec["cycles"].values() for c in cyc for pn in c if pn not in meta]
    seqs = {}
    for k, v in spec["sequences"].items():
        if not isinstance(v, dict):
            continue
        if "video" in v:  # a run of video frames: its frames by number, keys by frame number
            if v["video"] not in videos:
                raise SystemExit(f"sequence {k}: unknown video {v['video']!r}")
            f0, f1 = v.get("range", [0, 10 ** 9])
            fr = [r for r in videos[v["video"]]["frames"] if f0 <= r["f"] <= f1]
            nums = [r["f"] for r in fr]
            keys = {kk: int(np.argmin(np.abs(np.array(nums) - int(fv)))) for kk, fv in (v.get("keys") or {}).items()}
            seqs[k] = {**v, "drawings": [r["name"] for r in fr], "keys": keys, "frames": nums}
            if v.get("cycle") and v.get("travel") == "time":
                # travel: time (a clip shot with a tracking camera: no foot plants to measure, the torso does not
                # travel): `step` clip px a step, spread evenly over the cycle's frames by frame time
                if not v.get("step"):
                    raise SystemExit(f"sequence {k}: travel: time needs step (clip px a step)")
                seqs[k].update(time_travel(nums, int(f1) if f1 < 10 ** 8 else None, float(v["step"]),
                                           int(v.get("steps", 2))))
                log(f"poses: {k}: travel by time, step {float(v['step']):.0f} clip px")
            elif v.get("cycle") and v.get("travel") != "torso":  # stride-true travel from the foot plants
                # (travel: torso: the torso's own travel, where the feet cannot be told apart in the clip)
                fps_ = videos[v["video"]]["fps"]
                m0, m1 = v.get("measure", [f0, f1])
                st = stride(videos[v["video"]]["frames"], f0, f1, m0, m1, int(v.get("steps", 2)))
                seqs[k].update(st)
                log(f"poses: {k}: step {st['step']:.0f} clip px ({st['step'] * videos[v['video']]['scale']:.0f} street px)"
                    f", plants at {st['plants']}")
        elif "drawings" in v:
            seqs[k] = v
    unknown += [pn for sq in seqs.values() for pn in sq["drawings"] if pn not in meta]
    phases = {k: v for k, v in spec["stride_phase"].items() if not k.startswith("note")}
    if unknown:
        raise SystemExit(f"poses.spec.json names poses that the sheets do not have: {sorted(set(unknown))}")
    bad = [f"{t}: {n}" for t, n in spec["motion"].items() if n not in seqs]
    if bad:
        raise SystemExit(f"poses.spec.json motion names sequences that do not exist: {bad}")
    for name, sq in seqs.items():
        for key, i in (sq.get("keys") or {}).items():
            if not 0 <= int(i) < len(sq["drawings"]):
                raise SystemExit(f"sequence {name}: key {key} = {i} is not a drawing index")
    write_json(d / "poses.json", {"version": 1, "ref_height": spec["ref_height"], "poses": meta,
                                  "cycles": spec["cycles"], "phases": spec["phases"], "roles": spec["roles"],
                                  "sequences": seqs, "motion": spec["motion"], "stride_phase": phases,
                                  "videos": videos, "canon_waive": spec.get("canon_waive") or {}}, nd=3)
    contact(d, d / "contact.png")
    log(f"poses: {len(meta)} poses and stamps, contact sheet {d / 'contact.png'}")
    return meta


class PoseLib:
    """The built library: mattes (loaded on demand), anchors, roles and cycles."""

    def __init__(self, d: Path):
        self.d = Path(d)
        f = self.d / "poses.json"
        if not f.exists():
            raise SystemExit(f"pose library not built: run `aimv paint poses` ({f} missing)")
        j = read_json(f)
        self.meta: dict[str, dict] = j["poses"]
        self.cycles: dict[str, list[list[str]]] = j.get("cycles", {})
        self.phases: dict[str, list[str]] = j.get("phases", {})
        self.roles: dict[str, str | list[str]] = j.get("roles", {})
        self.sequences: dict[str, dict] = j.get("sequences", {})  # motion M0: ordered drawings with keys
        self.motion: dict[str, str] = j.get("motion", {})  # template -> its default sequence
        self.stride_phase: dict[str, int] = j.get("stride_phase", {})
        self.videos: dict[str, dict] = j.get("videos", {})  # video sprites: fps, block, ground, the figure's track
        self.ref_height = float(j.get("ref_height", 450.0))
        self._sprites: dict[str, np.ndarray] = {}
        cf = self.d / "canon.json"  # drawings with canon errors (canon.py): clip sequences never show them
        self.canon_bad = {n for n, v in (read_json(cf).get("drawings", {}) if cf.exists() else {}).items()
                          if any(fl["level"] == "error" for fl in v.get("flags", []))}

    def __contains__(self, name: str) -> bool:
        return name in self.meta

    def m(self, name: str) -> dict:
        if name not in self.meta:
            raise KeyError(f"unknown pose {name!r}")
        return self.meta[name]

    def role(self, role: str, override=None) -> list[str]:
        """Pose names for a template role (a move can override with its own names)."""
        v = override if override is not None else self.roles.get(role)
        if v is None:
            raise KeyError(f"no poses for role {role!r}: add it to poses.spec.json roles")
        return [v] if isinstance(v, str) else list(v)

    def sprite(self, name: str) -> np.ndarray:
        if name not in self._sprites:
            d = cv2.imread(str(self.d / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if d is None:
                raise SystemExit(f"missing pose matte {self.d / name}.png")
            d = d.astype(np.float32) / 255
            b = self.meta[name].get("boost")
            if b:  # dry-brushed sheets: same paint load as the others
                d = np.clip(d * b[0], 0, 1) ** b[1]
            self._sprites[name] = d
        return self._sprites[name]

    def anchor(self, name: str, kind: str) -> tuple[float, float]:
        """Anchor point in sprite px: foot (torso x, lowest foot y), bottom, top, right, left, br (hands planted),
        center (bbox centre), cxc (torso x, bbox centre y), bc (bbox centre x, its bottom: a front view on a floor)."""
        m = self.m(name)
        x0, y0, x1, y1 = m["bbox"]
        pts = {"foot": (m["cx"], m["bottom"][1]), "bottom": m["bottom"], "top": m["top"], "right": m["right"],
               "left": m["left"], "br": m["bottomright"], "center": ((x0 + x1) / 2, (y0 + y1) / 2),
               "cxc": (m["cx"], (y0 + y1) / 2), "bc": ((x0 + x1) / 2, y1)}
        if kind not in pts:
            raise KeyError(f"unknown anchor {kind!r} (one of {', '.join(ANCHORS)})")
        return float(pts[kind][0]), float(pts[kind][1])

    def size(self, name: str, scale: float = 1.0) -> tuple[float, float]:
        """Painted body size (bbox) in street px."""
        m = self.m(name)
        x0, y0, x1, y1 = m["bbox"]
        s = m["scale"] * scale
        return (x1 - x0) * s, (y1 - y0) * s

    def torso_offset(self, name: str, anchor: str, scale: float = 1.0, mirror: bool = False) -> tuple[float, float]:
        """Street px from a pose's anchor to its torso (torso x, bbox centre y) at a scale (on top of its own)."""
        m = self.m(name)
        ax, ay = self.anchor(name, anchor)
        s = m["scale"] * scale
        dx = (m["cx"] - ax) * s
        y0, y1 = m["bbox"][1], m["bbox"][3]
        return (-dx if mirror else dx), ((y0 + y1) / 2 - ay) * s

    def sequence(self, name: str) -> dict:
        if name not in self.sequences:
            raise KeyError(f"unknown sequence {name!r} (known: {', '.join(self.sequences) or 'none'})")
        return self.sequences[name]

    def foot_offset(self, name: str) -> float:
        """Street px from the torso x to the lowest foot (positive: the foot is ahead): the stance foot of a run pose."""
        m = self.m(name)
        return (m["bottom"][0] - m["cx"]) * m["scale"]


def contact(d: Path, out: Path) -> Path:
    """Every pose at render scale on a common baseline, with its anchors (top blue, bottom red, right green)."""
    meta = read_json(d / "poses.json")["poses"]
    tiles = []
    for pname, m in meta.items():
        if "video" in m:  # video frames get their own sheets (contact-<video>.png)
            continue
        dd = cv2.imread(str(d / f"{pname}.png"), 0).astype(np.float32) / 255
        s = m["scale"] * (0.5 if m["kind"] == "pose" else 0.25)
        t = cv2.resize(dd, (max(1, int(dd.shape[1] * s)), max(1, int(dd.shape[0] * s))), interpolation=cv2.INTER_AREA)
        c = np.full((320, 300, 3), 255, np.uint8)
        th, tw = min(t.shape[0], 280), min(t.shape[1], 300)
        y0 = 300 - th
        c[y0:300, :tw] = (255 - t[-th:, :tw, None] * 255).astype(np.uint8)
        cv2.line(c, (0, 300), (300, 300), (0, 160, 255), 1)
        for key, col in (("top", (255, 0, 0)), ("bottom", (0, 0, 255)), ("right", (0, 200, 0))):
            x, y = m[key]
            cv2.circle(c, (int(x * s), int(y0 + (y - (dd.shape[0] - th / s)) * s)), 4, col, -1)
        tags = m.get("tags", {})
        label = pname + (f" {tags['phase']}" if "phase" in tags else "")
        cv2.putText(c, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        roles = ",".join(tags.get("roles", []))[:40]
        if roles:
            cv2.putText(c, roles, (4, 316), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        tiles.append(c)
    while len(tiles) % 8:
        tiles.append(np.full((320, 300, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(str(out), np.vstack(rows))
    return out
