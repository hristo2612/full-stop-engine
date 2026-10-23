"""Camera and light hints at set times, from the choreography (a prop: out of the runner's chain): a camera punch
on a word, a street light knocked out on a kick as he runs past it (Full Stop's final sprint: punches on "No speech"
and "No long reply", the last red light dead on the kick after he passes it, as the alley's at 57.6).

    hints: {punch: [[time, 0.06], ...],        the camera punches in by that share on the photo at time
            shake: [[time, 6], ...],           px of hand-held shake on that photo
            knock: {at: time, light: id, flecks: 45},   the light flares on the photo before, then dies: its glow
                                                and its lamp (the light's parallax `layer`) dark from then on, pale
                                                glass flecks flying from the lamp (as the J2 leap's `knock`)
            burst: {at: time, x, y, dirx: 1, power: 1.2, flecks: 50, splats: 50, spread: 60}
                                                a burst of wet paint at (x, y) on that photo (props.Burst: black
                                                flecks thrown, splatter landing on the wall round it; Full Stop's
                                                tear off the wall, v7). A list of them is fine
            frame: {from: time, to: time, zoom: [z0, z1], dx: [a, b], dy: [a, b],
                    zoom_v: [..], dx_v: [..], dy_v: [..], cx, cy, follow_v, ease, centre, curve}}
                                                a camera move on every photo from `from` to `to`, eased: the crop
                                                `zoom` times the follow framing's (below 1: closer), its centre dx px
                                                ahead of him and dy px below the follow framing's (positive: his feet
                                                higher in the frame, room below them); the _v values for the 9:16
                                                (camera.py; without them the 9:16 takes the 16:9's, muted). Full
                                                Stop's ending: from the dead stop the camera pushes in until he is
                                                half the frame at the edge, the drop and the street below in frame.
                                                `cx`, `cy`: a fixed centre (street px) instead of following him
                                                (`follow_v: true`: the 9:16 still follows). Under the spring camera
                                                write the values as steps (one number): the spring eases into them,
                                                dx and dy as offsets of its target, cx and cy as the target it
                                                settles on; `ease: s` or `[in, back]` sets how long the zoom change
                                                into the stretch and back out after it takes (at least a beat, and
                                                no photo above the camera QA's 4%). A zoom written as [z0, z1] is a
                                                push (or a pull) the spring camera follows exactly, photo by photo,
                                                geometrically (each photo the same share closer at a steady pace),
                                                from wherever the zoom is when the stretch starts; `curve` shapes
                                                it and the other pairs: smooth (default: eased in and out) or in
                                                (eased in and gathering speed all the way, settling onto its end
                                                value over the stretch's last tenth: a push that lands on a hit).
                                                `centre: c` (0 to 1, or [c0, c1]; spring camera): how far the frame
                                                moves from the follow framing onto his figure's middle (no lead,
                                                his middle at the frame's middle height; the vertical spring as
                                                stiff as the horizontal). Full Stop v8f: the push in as he sprints
                                                at the alley, tightest on the take-off. A dy (or dy_v) pair that
                                                changes is a tilt: the vertical spring follows it as stiffly as the
                                                horizontal while it runs (film-v8: easing into verse 2's eye-level
                                                framing inside p6's brake). `aspect: 9:16` (or 16:9): that aspect's
                                                camera only, the other's exactly as without the stretch.
"""

from __future__ import annotations

import numpy as np

from .choreo import ChoreoError, Ctx
from .moves import template


CURVES = ("smooth", "in")
IN_SETTLE = 0.1  # `curve: in` settles onto its end value over this share of the stretch


def ease_curve(u: float, curve: str = "smooth") -> float:
    """How far along (0 to 1) a frame stretch's values are at u (0 to 1) of it. smooth: a smoothstep (eased in and
    out). in: its pace rising steadily from 0 (eased in, gathering speed) to the top at 1 - IN_SETTLE, then easing to
    0 on its end value (a cosine) over the last IN_SETTLE: a push that lands on a hit. Its fastest photo is twice
    its mean pace."""
    u = min(max(float(u), 0.0), 1.0)
    if curve == "in":
        a = 1.0 - IN_SETTLE
        tot = a * a / 2 + a * IN_SETTLE / 2
        if u <= a:
            return u * u / 2 / tot
        w = (u - a) / IN_SETTLE
        return (a * a / 2 + a / 2 * IN_SETTLE * (w + np.sin(np.pi * w) / np.pi)) / tot
    return u * u * (3 - 2 * u)


@template("hints", prop=True)
def hints(c: Ctx) -> None:
    """Camera punches and shakes, a light knocked out at set times, a camera move over a stretch (prop; see
    hints.py)."""
    used = False
    for key in ("punch", "shake"):
        for i, item in enumerate(c.arg(key) or []):
            if not isinstance(item, list | tuple) or len(item) != 2:
                raise ChoreoError(f"hints: {key} item {i + 1} is not [time, value]")
            q = c.at(item[0], key=f"{key}[{i + 1}]")
            c.cam(q, **{key: float(item[1])})
            c.photo(q)
            used = True
    kn = c.arg("knock")
    if kn:
        if not isinstance(kn, dict) or "at" not in kn or "light" not in kn:
            raise ChoreoError("hints: write knock as {at, light}")
        lid = str(kn["light"])
        L = next((lt for lt in c.ch.scene.lights if lt.get("id") == lid), None)
        if L is None:
            raise ChoreoError(f"hints: knock: no street light {lid!r} (obstacles.json lights)")
        q = c.at(kn["at"], key="knock.at")
        s = c.ch.step_at(q)
        c.light(q - s, lid, 2.2)  # a last flare
        c.light_out(q, lid)
        rs = np.random.default_rng([c.mv.seed, c.mv.local_i, 79])
        c.burst(q, float(L["x"]), float(L["y"]), dirx=1.0, seed=int(rs.integers(0, 2 ** 31)), power=0.8,
                flecks=int(kn.get("flecks", 45)), splats=0, spread=40, albedo=0.6)
        c.photo(q)
        used = True
    bs = c.arg("burst")
    if bs:
        for j, b in enumerate(bs if isinstance(bs, list) else [bs]):
            if not isinstance(b, dict) or not all(k in b for k in ("at", "x", "y")):
                raise ChoreoError("hints: write burst as {at, x, y} (dirx, power, flecks, splats, spread optional)")
            q = c.at(b["at"], key="burst.at")
            rb = np.random.default_rng([c.mv.seed, c.mv.local_i, 83, j])
            c.burst(q, c.coord(b["x"], "x"), c.coord(b["y"], "y"), dirx=float(b.get("dirx", 1.0)),
                    seed=int(rb.integers(0, 2 ** 31)), power=float(b.get("power", 1.2)),
                    flecks=int(b.get("flecks", 50)), splats=int(b.get("splats", 50)),
                    spread=float(b.get("spread", 60)))
            c.photo(q)
        used = True
    fr = c.arg("frame")
    if fr:
        if not isinstance(fr, dict) or "from" not in fr or "to" not in fr:
            raise ChoreoError("hints: write frame as {from, to, zoom: [z0, z1], dx: [a, b], dy: [a, b]}")
        q0, q1 = c.at(fr["from"], key="frame.from"), c.at(fr["to"], key="frame.to")
        if q1 <= q0:
            raise ChoreoError("hints: frame's to must be after its from")
        pairs = {}
        for k in ("zoom", "dx", "dy", "zoom_v", "dx_v", "dy_v", "cx", "cy", "cx_v", "centre"):
            v = fr.get(k)
            if v is None:
                continue
            if isinstance(v, int | float):
                v = [v, v]
            if not isinstance(v, list | tuple) or len(v) != 2:
                raise ChoreoError(f"hints: frame.{k} is [from, to] (or one number)")
            pairs[k] = (float(v[0]), float(v[1]))
        if "zoom" in pairs and min(pairs["zoom"]) <= 0 or "zoom_v" in pairs and min(pairs["zoom_v"]) <= 0:
            raise ChoreoError("hints: frame zooms above 0")
        if "centre" in pairs and not all(0 <= v <= 1 for v in pairs["centre"]):
            raise ChoreoError("hints: frame.centre is 0 to 1 (or [c0, c1])")
        curve = str(fr.get("curve", "smooth"))
        if curve not in CURVES:
            raise ChoreoError(f"hints: frame.curve is one of {', '.join(CURVES)}")
        # a zoom pair that changes: a push the spring camera follows photo by photo (from where the zoom is when
        # the stretch starts), not a step it eases into
        ramp = any(k in pairs and abs(pairs[k][1] - pairs[k][0]) > 1e-9 for k in ("zoom", "zoom_v"))
        def _moves(k):
            return k in pairs and abs(pairs[k][1] - pairs[k][0]) > 1e-9
        # (the aspects a tilt moves: dy the 16:9's, and the 9:16's too unless it has its own dy_v)
        yramp = (["16:9"] if _moves("dy") else []) + (["9:16"] if _moves("dy_v") or ("dy_v" not in pairs
                                                                                    and _moves("dy")) else [])
        if fr.get("aspect") is not None and str(fr["aspect"]) not in ("16:9", "9:16"):
            raise ChoreoError("hints: frame.aspect is 16:9 or 9:16 (that aspect's camera only)")
        ez = fr.get("ease")  # s: the spring camera's zoom change into the stretch, and back after it
        if ez is not None:
            ez = (ez, ez) if isinstance(ez, int | float) else tuple(ez)
            if len(ez) != 2 or min(float(v) for v in ez) <= 0:
                raise ChoreoError("hints: frame.ease is seconds above 0, or [in, back]")
            ez = (float(ez[0]), float(ez[1]))
        s0, s1 = c.song(q0), c.song(q1)
        # the photos of the stretch, on the segments' own grid (Choreo.photos: each segment's steps from its bar line)
        qs = sorted({q for sg in c.ch.segments for q in c.ch.grid.steps(sg.q0, sg.q1, sg.rate)
                     if s0 - 1e-6 <= q <= s1 + 1e-6})
        for q in (qs if not c.dry else []):
            u = (q - s0) / max(s1 - s0, 1e-6)
            e = ease_curve(u, curve)
            h = {k: (b if e >= 1 else a if e <= 0 else a * (b / a) ** e) if k in ("zoom", "zoom_v") else
                 a + (b - a) * e for k, (a, b) in pairs.items()}
            h.setdefault("dx", 0.0)
            h["free"] = h["frame"] = True
            if ramp:
                h["zr"], h["ze"] = round(s0, 4), e  # (the push's own id and how far along it is)
            if yramp:  # a dy pair that changes: a tilt the vertical spring follows as stiffly as the horizontal
                h["yr"] = yramp  # (in the aspects it moves)
            if ez is not None:
                h["ease"] = ez
            if fr.get("follow_v"):
                h["follow_v"] = True
            if fr.get("aspect") is not None:  # one aspect's own camera move (the other's camera as without it)
                h["aspect"] = str(fr["aspect"])
            c.ch.cam_hints.setdefault(round(q, 4), {}).update(h)
        used = True
    if not used:
        raise ChoreoError("hints: give punch, shake, knock, burst or frame")
