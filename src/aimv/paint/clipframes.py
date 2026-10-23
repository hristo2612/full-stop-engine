"""`aimv paint clipframes`: start and end frames for an obstacle clip (motion M2: video sprites), sized to the real
obstacle, so the video model's figure meets a guide block as tall and wide, relative to him, as the obstacle he will
vault on the street (a waist-high guide block against a chest-high cabinet puts his hands and head inside it).

    aimv paint clipframes <project> <obstacle> --move vault [--cycle run_SW] [--street main]

writes projects/<p>/paint/clipframes/<obstacle>/<move>-start.png and <move>-end.png (1920x1080): the plain white
wall, the pale blue guide shape (RGB 172, 203, 236: the key colour video sprites key out) centred at x 970, sized as
the obstacle is relative to a 430 px figure (smaller when a wide obstacle needs the room or a bar would be above the
frame) standing on the ground line at y 900; the figure painted from the run
cycle's first drawing, running in from the left (start) and away on the right (end). <move>-frames.json has the
geometry (block box, shapes, ground, scale) for the video entry in poses.spec.json.

The shape follows the move: a pole (pole_swing, or any obstacle with a pole contact) is a vertical bar from the top
of the frame to the ground, as wide as the pipe; a hang (hang_release, swing_release, hang: an obstacle with a bar or
beam contact) is a horizontal bar at the bar's height with nothing under it, as long as the real one (at most 1000
px); anything else is a block from the obstacle's top to the ground (or floating at its own bottom when its box ends
above the ground: a cabinet hung on the wall), one block per solid part when the obstacle has several (a row of
bollards). A pole or bar clip has no block standing on the ground: give its video entry "ground": "feet".

    --and fire_escape          add another obstacle's shape where it stands against this one (the beam a climb
                               ends on)
    --start-pose hang          the figure in the start (or --end-pose: end) frame: a pose or a role; role hang hangs
                               him from the bar by his hands
    obstacle wall              no obstacle: a plain wall (a brake into a walk)
    a gap (the alley)          no shape: coiled at its near edge (start), running past its far edge (end)
    an edge                    no shape: the end frame stops him with his front on it (a brake into a crouch)
    --box 400x450x34           (obstacle wall) a front view inside a box: its outline is the guide (keyed out),
                               760 px tall; --scale-by BX1-6 gives both frames that pose's body scale
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..util import write_json

W, H = 1920, 1080
GROUND = 900
BLOCK_CX = 970
FIG_H = 430
KEY = (172, 203, 236)  # RGB
WALL = (235, 234, 230)
INK = (20, 20, 22)


def _wall(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (0, 0), 3)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) * 2.4 / max(n.std(), 1e-6)
    return np.clip(np.float32(WALL)[None, None, :] + n[..., None], 0, 255)


def _paint(img: np.ndarray, d: np.ndarray, x0: int, y0: int) -> None:
    h, w = d.shape
    a0, b0 = max(y0, 0), max(x0, 0)
    a1, b1 = min(y0 + h, H), min(x0 + w, W)
    if a1 <= a0 or b1 <= b0:
        return
    m = d[a0 - y0:a1 - y0, b0 - x0:b1 - x0][..., None]
    img[a0:a1, b0:b1] = img[a0:a1, b0:b1] * (1 - m) + np.float32(INK) * m


def _shapes(o, move: str, k: float, ref_x: float, feet: float, main: bool) -> tuple[str, list[list[float]]]:
    """An obstacle's guide shapes in frame px (street x ref_x at BLOCK_CX, the ground line `feet` at GROUND): a pole,
    a bar or blocks (see the module doc). The main obstacle's bar is centred; an added one keeps its place against
    the main obstacle."""
    X = lambda x: BLOCK_CX + (x - ref_x) * k
    fy = lambda y: GROUND - (feet - y) * k
    if move == "pole_swing" or ("pole" in o.contacts and move not in ("vault", "leap_off", "step_up", "slide")):
        w = max((o.x1 - o.x0) * k, 24.0)
        cx = X(o.contact("pole").x if o.contact("pole").x is not None else o.cx)
        return "pole", [[cx - w / 2, 0.0, cx + w / 2, float(GROUND)]]
    if move in ("hang_release", "swing_release", "hang") or ({"bar", "beam"} & set(o.contacts) and move != "slide"):
        c = o.contacts["bar" if "bar" in o.contacts else "beam"]
        x0_, x1_ = (c.x0 if c.x0 is not None else o.x0), (c.x1 if c.x1 is not None else o.x1)
        L = min((x1_ - x0_) * k, 1000.0)
        th = max(16.0, 14 * k)
        a = BLOCK_CX - L / 2 if main else X(x0_)
        b = min(a + L, W - 20.0)
        return "bar", [[a, fy(c.y) - th / 2, b, fy(c.y) + th / 2]]
    # the solid part of the obstacle: from its top to the ground, or floating at its own bottom when its box ends
    # above the ground (a cabinet hung on the wall); a row of bollards: one block per part
    top = next((o.contacts[cn].y for cn in ("top", "step") if cn in o.contacts and o.contacts[cn].y is not None), o.y0)
    solid = [p_["box"] for p_ in o.occlude if p_.get("solid", o.solid)] or [o.box]
    if len(solid) > 1 and o.kind == "bollard":
        return "block", [[X(b[0]), fy(b[1]), X(b[2]), float(GROUND)] for b in solid]
    bottom = min(o.y1, feet)
    if bottom > feet - 30:
        bottom = feet
    return "block", [[X(o.x0), fy(top), X(o.x1), fy(bottom)]]


def _pose_of(lib, spec: str) -> str:
    """A pose name, or the first drawing of a role."""
    if spec in lib:
        return spec
    try:
        return lib.role(spec)[0]
    except KeyError as e:
        raise SystemExit(f"clipframes: {spec!r} is neither a pose nor a role") from e


def make(lib, scene, obstacle: str, move: str, out_dir: Path, cycle: str | None = None, extra: tuple = (),
         start_pose: str | None = None, end_pose: str | None = None, box: tuple | None = None,
         fill: float = 0.88, scale_by: str | None = None, open_end: str | None = None,
         open_start: str | None = None, box_cx: float | None = None, start_x: float | None = None,
         end_x: float | None = None, box_h: float | None = None) -> dict:
    """Start and end frames for a clip of `move` at `obstacle` ("wall": no obstacle, plain wall), with the shapes of
    the `extra` obstacles too (where they stand against it: a pipe and the beam it leads to). start_pose, end_pose:
    a pose or role for the figure in either frame instead of the run cycle's (role "hang": hanging from the bar by
    the hands). A gap (the alley): no shape, he starts at its near edge and lands past its far edge; an edge: no
    shape, the end frame's figure stops with his front on it. box: (w, h, line) street px: a front view inside the
    box's outline (the guide, keyed out), 760 px tall, its bottom on the ground line, the figures fitted inside it
    (fill); scale_by: a pose that fills the box (his head near the top, his reach on its sides), the one body scale
    for both frames (a clip where he stands up must not shrink him). In the box a pose can say where it is:
    "BX2-1@right" pressed against the inner right side (left: the left side), "SW-061@out" outside the box on its
    right, standing on the ground line, at the same body scale (out of the box). open_end "right": the end frame's
    box has no right side (he broke out through it). Chained clips (one clip's end frame the next one's start) come
    out identical when given the same box, scale_by and pose. Obstacle "box" is the plain wall too. open_start: the
    start frame's box without that side too. box_cx: the box's centre x in the frame (default the middle; a box in
    the left third leaves room for a flight out of it). "YB-068@clip": a clip drawing where its own clip had it,
    moved with the box (its clip's guide box was drawn at the middle); "SW-061@far": outside the box at the body
    scale, a stride from the frame's right edge. start_x, end_x (out of the box): where the figure's left edge is in
    that frame (frame px), instead of its run-up and run-out (a short walk to a stop: the reach). box_h: the box's
    height in the frame (760 by default; smaller makes the figure smaller, leaving room to run out of it: v7's
    breakout)."""
    o = None if obstacle in ("wall", "box", "-") else scene.obstacle(obstacle)
    seq = lib.sequence(cycle or lib.motion.get("run", ""))
    names = seq["drawings"]
    fig = names[0]
    m = lib.m(fig)
    fh = (m["bbox"][3] - m["bbox"][1]) * m["scale"]  # the figure's height on the street (px)
    fw = (m["bbox"][2] - m["bbox"][0]) * m["scale"]
    k = FIG_H / fh  # street px to frame px
    if box is not None:  # a front view in a box: the box 760 px tall (box_h), so the figure in it has detail
        k = float(box_h or 760.0) / box[1]
    elif o is not None and move not in ("pole_swing", "hang_release", "swing_release", "hang") and o.x1 - o.x0 > 900:
        # a wide obstacle (a row of bollards): a smaller figure, so the frame holds a run-up and a run-out too
        k = min(k, (W - 80) / ((o.x1 - o.x0) + 2 * (fw + 0.8 * fh)))
    ref = o.cx if o is not None else 0.0
    if o is not None and o.kind == "edge":
        ref = o.contact("edge").x
    feet = scene.feet_at(ref)

    bcx = float(box_cx) if box_cx is not None else float(BLOCK_CX)

    def all_shapes(k):
        if box is not None:
            bw, bh, t = (v * k for v in box)
            x0, y1 = bcx - bw / 2, GROUND - 10 * k  # the box stands a little above the ground line
            x1, y0 = x0 + bw, y1 - bh
            return "box", [[x0, y0, x1, y0 + t], [x0, y1 - t, x1, y1], [x0, y0, x0 + t, y1], [x1 - t, y0, x1, y1]]
        if o is not None and o.kind in ("gap", "edge"):
            return o.kind, []
        shape, shapes = ("wall", []) if o is None else _shapes(o, move, k, ref, feet, True)
        for name in extra:
            shapes += _shapes(scene.obstacle(name), "", k, ref, feet, False)[1]
        return shape, shapes
    shape, shapes = all_shapes(k)
    tops = [(r[1] + r[3]) / 2 for r in shapes if r[2] - r[0] > 4 * (r[3] - r[1])]
    if tops and min(tops) < 60:  # a bar above the frame: a smaller figure, the bar 60 px from the top
        k *= (GROUND - 60) / (GROUND - min(tops))
        shape, shapes = all_shapes(k)
    shapes = [[int(round(v)) for v in r] for r in shapes]
    block = [min(r[0] for r in shapes), min(r[1] for r in shapes), max(r[2] for r in shapes),
             max(r[3] for r in shapes)] if shapes else [BLOCK_CX, GROUND, BLOCK_CX, GROUND]
    bars = [r for r in shapes if r[2] - r[0] > 4 * (r[3] - r[1])]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    figs = {}
    for which, spec, side in (("start", start_pose, "left"), ("end", end_pose, "right")):
        spec, _, align = (spec or "").partition("@")
        pose = _pose_of(lib, spec) if spec else (fig if side == "left" else names[len(names) // 2])
        img = _wall(3)
        for n_, r in enumerate(shapes):
            if box is not None and n_ == 3 and (open_end if which == "end" else open_start) == "right":
                continue  # the box's right side: broken through
            img[max(r[1], 0):r[3], max(r[0], 0):r[2]] = np.float32(KEY)
        d0 = lib.sprite(pose)
        mm = lib.m(pose)
        s = mm["scale"] * k
        d = cv2.resize(d0, (max(1, int(round(d0.shape[1] * s))), max(1, int(round(d0.shape[0] * s)))),
                       interpolation=cv2.INTER_AREA)
        bx = mm["bbox"]
        w_fig = (bx[2] - bx[0]) * s
        if box is not None:  # inside the box: fitted to its inner size, standing on its inner bottom, centred
            t = box[2] * k
            iw, ih = box[0] * k - 2 * t, box[1] * k - 2 * t
            if scale_by:  # one body scale: that pose fills the box (its height 0.96 of it, its reach the box's width)
                ms = lib.m(_pose_of(lib, scale_by))
                s2 = min(0.96 * ih / max((ms["bbox"][3] - ms["bbox"][1]) * ms["scale"], 1),
                         box[0] * k / max((ms["bbox"][2] - ms["bbox"][0]) * ms["scale"], 1)) * mm["scale"]
            else:
                fit = min(1.15, fill * min(iw / max((bx[2] - bx[0]) * s, 1), ih / max((bx[3] - bx[1]) * s, 1)))
                s2 = s * fit
            d = cv2.resize(d0, (max(1, int(round(d0.shape[1] * s2))), max(1, int(round(d0.shape[0] * s2)))),
                           interpolation=cv2.INTER_AREA)
            ax, ay = lib.anchor(pose, "bc")
            floor = GROUND - 10 * k - t - 2
            bx0_, bx1_ = bcx - box[0] * k / 2, bcx + box[0] * k / 2
            if align == "right":  # pressed against the inner right side
                x_, y_ = bx1_ - t - 2 - bx[2] * s2, floor - bx[3] * s2
            elif align == "left":
                x_, y_ = bx0_ + t + 2 - bx[0] * s2, floor - bx[3] * s2
            elif align == "out":  # out of the box on its right, on the ground line
                x_, y_ = bx1_ + 40 - bx[0] * s2, GROUND - bx[3] * s2
                x_ = min(x_, W - 30 - bx[2] * s2)
            elif align == "far":  # a stride from the frame's right edge, on the ground line
                x_, y_ = W - 60 - bx[2] * s2, GROUND - bx[3] * s2
            elif align == "clip" and "video" in mm:  # where its clip had it, moved with the box, at clip size
                d = d0
                x_, y_ = mm["video"]["ox"] + (bcx - BLOCK_CX), float(mm["video"]["oy"])
            else:
                x_, y_ = bcx - ax * s2, floor - ay * s2
            _paint(img, d, int(round(x_)), int(round(y_)))
        elif spec == "hang" and bars:  # hanging from the (first) bar by the hands, a little in from its left end
            r = bars[0]
            ax, ay = lib.anchor(pose, "top")
            hx = r[0] + max(60.0, 0.15 * (r[2] - r[0]))
            _paint(img, d, int(round(hx - ax * s)), int(round((r[1] + r[3]) / 2 - ay * s)))
        else:
            # start: his front a stride and a half before the shapes; end: his back as far past them
            gap = 1.6 * fh * k
            if o is None:
                left = 240.0 if side == "left" else W - 240.0 - w_fig
            elif o.kind == "gap":  # take-off at its near edge, the landing a little past its far edge
                left = (BLOCK_CX + (o.x0 - ref) * k - w_fig - 8 * k) if side == "left" else \
                    (BLOCK_CX + (o.x1 - ref) * k + 0.6 * fh * k)
            elif o.kind == "edge":  # running in from far left; stopped with his front on the edge
                left = 120.0 if side == "left" else BLOCK_CX - w_fig - 8 * k
            else:
                left = (block[0] - gap - w_fig) if side == "left" else (block[2] + gap)
            at_x = start_x if which == "start" else end_x
            if at_x is not None:
                left = float(at_x)
            left = float(np.clip(left, 40, W - 40 - w_fig))
            _paint(img, d, int(round(left - bx[0] * s)), int(round(GROUND - bx[3] * s)))
        f = out_dir / f"{move}-{which}.png"
        cv2.imwrite(str(f), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        paths[which] = str(f)
        figs[which] = pose
    top = block[1]
    info = {"obstacle": obstacle, "move": move, "size": [W, H], "ground": GROUND, "shape": shape, "block": block,
            "shapes": shapes, "extra": list(extra), "box": list(box) if box else None, "open_end": open_end,
            "open_start": open_start, "box_cx": bcx,
            "key": list(KEY), "figure": fig, "figures": figs, "figure_height": int(round(fh * k)),
            "street_to_frame": round(k, 4),
            "obstacle_px": {"width": round((block[2] - block[0]) / k, 1), "height": round((GROUND - top) / k, 1),
                            "clearance": round((GROUND - block[3]) / k, 1)},
            "frames": paths}
    write_json(out_dir / f"{move}-frames.json", info)
    return info
