"""The camera: one framing per photograph (prototype 6's plan_camera), with named presets per segment, and a separate
9:16 pass that frames the runner on its own instead of cropping the 16:9.

16:9 (follow): the crop is CH0 street px high (the figure, about 450 px, fills about half the frame), feet at 76% of
the height, a little lead in the run direction, vertical follow (85%) on climbs and hangs, MUTO-step smoothing (62%
of the way to the target per photo), kick punches (3.5%, 8.5% on an impact), a small roll swing on strong snares,
a little hand-held noise. Presets: follow, close (x0.85), wide (x1.25), hold (the segment's first framing), stop (a
wide establishing frame that pushes in photo by photo while the song stands still), box (tight on a box prop, almost
still: it steps closer as the box shrinks, a slow push over a span, and room for something beside it from a time),
frame (a set stretch of street: x [x0, x1], the crop that wide, at the ground's height or cy; an establishing
shot).
Templates add per-photo hints: the leap zooms out and snaps for the flight, the landing punches in, the pound and
the strikes in the box punch harder on each hit (punch), hits and the explosion shake the camera a little (shake),
and the leap from a clip frames its arc (cx, eased photo by photo) in a way the 9:16 does not take (follow_v).

9:16: its own crop (half the 16:9 width, feet at 80% of the height, at most the street's height), the same
smoothing, punches and hints (zoom-outs are muted by the street height), then a hard rule: the figure's box stays
inside the frame with a 5% margin (the crop widens up to the street height when a pose is too wide). With motion
M0 the 16:9 crop also keeps him inside (moved only), outside the box and frame presets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CH0 = 860.0  # 16:9 crop height in street px at zoom 1
CW0 = CH0 * 16 / 9
V_BASE = 0.5  # 9:16 crop width as a fraction of CW0
JOIN_BLEND = 4.0  # 16ths: the camera eases across a join between segment files over about a beat
XMAX_EASE = 1.0  # s: a segment's `x_max` shift eases in and out over this round the photos that need it
XMAX_HOLD = 0.8  # s: and inside such a segment holds each photo's need this long either side (a steady offset: the
#                  camera's own small moves stay)
PRESET_ZOOM = {"follow": 1.0, "close": 0.85, "wide": 1.25, "hold": 1.0, "stop": 1.0, "box": 1.0, "frame": 1.0,
               "spring": 1.0}
SPRING_W = 6.0  # rad/s: the spring camera's horizontal stiffness (critically damped); vertical a third of it


@dataclass
class Cam:
    cx: float
    cy: float
    cw: float  # crop width in street px
    ch: float  # crop height in street px
    rot: float  # roll, degrees
    look: float = 0.0  # the box camera's look push (0 to 1): the frame may crop him below his shoulders
    base: tuple | None = None  # (cx, cy, cw) before the punches and the hand-held nudge (the QA's snap rule)

    def as_list(self) -> list[float]:
        return [round(self.cx, 2), round(self.cy, 2), round(self.cw, 2), round(self.ch, 2), round(self.rot, 3)]


def kick_punch(t: float, kicks: list[tuple[float, float]], impacts: list[float]) -> float:
    p = 0.0
    for kt, ks in kicks:
        dt = t - kt
        if -0.03 <= dt < 0.25:
            amp = 0.085 if any(abs(kt - it) < 0.02 for it in impacts) else 0.035 * ks
            p = max(p, amp * math.exp(-max(dt, 0) / 0.09))
    return p


def snare_roll(t: float, snares: list[tuple[float, float]]) -> float:
    r = 0.0
    for k, (st, ss) in enumerate(snares):
        dt = t - st
        if -0.02 <= dt < 0.3:
            r = (0.7 if k % 2 else -0.7) * ss * math.exp(-max(dt, 0) / 0.12)
    return r


def plan(ch, photos, fig_boxes: dict, aspect: str = "16:9") -> list[Cam]:
    """Framings for `photos` (every photo of the choreography, in order: smoothing runs through all of them).
    fig_boxes: photo q -> (x0, y0, x1, y1) of the figure shown in that photo."""
    sc = ch.scene
    H, W, feet = sc.H, sc.W, sc.feet
    vertical = aspect == "9:16"
    g = ch.grid
    kicks = list(g.hits.get("kick", []))
    impacts = [t for t, _ in g.hits.get("impact", [])]
    snares = [(t, s) for t, s in g.hits.get("snare", []) if s > 0.85]
    u = ch.u  # crop sizes and offsets are written for a 450 px figure
    w_max = (H - 60) * 9 / 16 if vertical else W
    base = (CW0 * V_BASE if vertical else CW0) * u
    # figure box at each photo: the latest painted pose at or before it
    keys = sorted(fig_boxes)
    def fig(q):
        i = int(np.searchsorted(keys, q + 1e-6, side="right")) - 1
        return fig_boxes[keys[max(i, 0)]] if keys else (0, 0, 100, 100)

    def crop_h(cw):
        return cw * 16 / 9 if vertical else cw * 9 / 16

    def ground_cy(cw, x=0.0):
        chh = crop_h(cw)
        return min(sc.feet_at(x) - (0.30 if vertical else 0.26) * chh, H - chh / 2 - 2)

    cams: list[Cam] = []
    cur = None
    # joins between segment files: the framing eases from the last one before into the new one over JOIN_BLEND
    # 16ths (unless the new file's first segment says `cut: true`)
    joins = [s.q0 for s in ch.segments if s.join and not s.camera.get("cut")]
    blend_from = None
    rx_prev, v_fig = None, 0.0
    a_x = 0.62  # MUTO-step smoothing: 62% of the way to the target per photo
    spr = None  # the spring camera's state: x, vx, y, vy, the eased width (from, to, t0, duration), t, his speed
    t_prev = None
    sprung_prev = False  # (v8b) the photo before was framed by the spring: a spring stretch after another framing
    #                      starts afresh from where that one was, not from the state an earlier spring left
    tq, tx_ = [], []  # his torso x photo by photo (the spring follows it: steadier than the drawings' boxes)
    if any(s_.camera.get("preset") == "spring" for s_ in ch.segments):
        tor = {}
        for e in ch.events:
            if e.layer in ("wall", "air") and e.pose is not None and ch.lib.m(e.pose)["kind"] == "pose":
                o_ = ch.lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)
                tor[round(e.q, 4)] = (e.x + o_[0], e.y + o_[1])
        tq = sorted(tor)
        tx_ = [tor[k] for k in tq]
    tt_ = np.array([g.t(q_) for q_ in tq]) if tq else None  # (soft_bounds: his torso where it will be)
    txa_ = np.array([p_[0] for p_ in tx_]) if tq else None
    tya_ = np.array([p_[1] for p_ in tx_]) if tq else None

    def torso_x(q, dflt):
        i = int(np.searchsorted(tq, q + 1e-6, side="right")) - 1
        return tx_[i][0] if i >= 0 else dflt

    def torso_lift(q, x):
        """How far his torso is above a runner's on the ground there (0 when he runs, crouches or rolls)."""
        i = int(np.searchsorted(tq, q + 1e-6, side="right")) - 1
        return max(0.0, (sc.feet_at(x) - 0.5 * 450 * u) - tx_[i][1]) if i >= 0 else 0.0
    for n, ph in enumerate(photos):
        seg = ch.segments[ph.seg]
        cam = dict(ch.camera_defaults)
        cam.update(seg.camera)
        preset = cam.get("preset", "follow")
        hint = ch.cam_hints.get(round(ph.q, 4), {})
        if hint.get("frame") and hint.get("aspect", aspect) != aspect:  # the other aspect's own camera move: none
            hint = {k: v for k, v in hint.items() if k in ("punch", "shake")}  # of it here (its punches stay)
        bx0, by0, bx1, by1 = fig(ph.q)
        rx, ry = (bx0 + bx1) / 2, (by0 + by1) / 2
        if rx_prev is not None:  # how fast he travels (px per photo), smoothed
            v_fig = 0.5 * v_fig + 0.5 * (rx - rx_prev)
        rx_prev = rx
        zoom = cam.get("zoom", PRESET_ZOOM.get(preset, 1.0))
        zoom = 1.0 if isinstance(zoom, list) else float(zoom)  # the stop preset takes [from, to]
        snap = bool(cam.get("snap")) and abs(ph.q - seg.q0) < 1e-6
        free = False
        sprung = False
        if preset == "spring":
            # v7: a critically damped spring after his torso plus a look-ahead of his speed (so it does not lag),
            # vertical three times softer; any zoom change eased over at least a beat; no snaps
            dt = (ph.t - t_prev) if t_prev is not None else 0.0
            fr_ = hint if hint.get("frame") else {}  # a `hints: frame` stretch: offsets and a fixed target too
            v_ = "_v" if vertical else ""  # (the 9:16's own values where given)
            z_ = float(fr_.get("zoom", hint.get("zoom", 1.0)))  # (the 9:16 without its own: muted, as the follow)
            tw_t = base * zoom * float(fr_.get("zoom_v", 1 + (z_ - 1) * 0.5) if vertical and fr_ else z_)
            tw_t = min(tw_t, w_max) if fr_ else tw_t  # (eased to what the street allows, not cut off at it)
            pi_ = getattr(ch, "paint_in", None)  # v8 paint-in (reveal.py): the 9:16 frames ahead of him, wider, so the
            fv_ = pi_.frame_v if vertical and pi_ is not None and pi_.t0 <= ph.t <= pi_.t1 else None  # front is in it
            if fv_:
                tw_t *= float(fv_.get("zoom", 1.3))
            w_ = float(cam.get("stiff", SPRING_W)) * (1.5 if vertical else 1.0)  # the narrow 9:16 keeps closer
            fresh = spr is None or (abs(ph.q - seg.q0) < 1e-6 and cam.get("cut")) or not sprung_prev
            rx_t = torso_x(ph.q, rx)
            if fresh:  # already moving with him: his speed from the next photo
                v0_ = 0.0
                if n + 1 < len(photos):
                    v0_ = (torso_x(photos[n + 1].q, rx_t) - rx_t) / max(photos[n + 1].t - ph.t, 1e-6)
                spr = {"x": rx_t, "vx": v0_, "y": None, "vy": 0.0, "w": (tw_t, tw_t, ph.t, 0.0), "v": v0_}
            v_s = (rx_t - spr["rx"]) / dt if dt > 0 and "rx" in spr else spr["v"]
            spr["v"] = 0.6 * spr["v"] + 0.4 * v_s if dt > 0 else spr["v"]
            spr["rx"] = rx_t
            # the frame on his whole figure (its box's middle, a trailing leg included) rather than his torso: the
            # box's offset from the torso, smoothed (the drawings' boxes jump about)
            spr["off"] = (rx - rx_t) if "off" not in spr else 0.8 * spr["off"] + 0.2 * (rx - rx_t)
            w0, w1, wt0, wd = spr["w"]
            e_ = 1.0 if wd <= 0 else float(np.clip((ph.t - wt0) / wd, 0, 1))
            cur_w = w0 + (w1 - w0) * e_ * e_ * (3 - 2 * e_)
            ez = fr_.get("ease") or spr.get("back")  # a frame's own ease (s): in, and back after it ends
            zr_ = fr_.get("zr")
            if zr_ is not None:  # a push (a frame's zoom [z0, z1]): followed photo by photo, from where the zoom was
                if spr.get("zr") != zr_:  # when it began, blended onto the push by its end
                    spr["zk"] = cur_w / max(tw_t, 1e-6)
                tw_t = tw_t * spr["zk"] ** (1 - float(fr_.get("ze", 1.0)))
                spr["w"] = (tw_t, tw_t, ph.t, 0.0)
                cur_w = tw_t
            elif abs(tw_t - w1) > 1e-6:  # a new zoom: eased from where it is over a beat or more, and long enough
                beat = g.t(ph.q + 4) - ph.t  # that no photo changes it by more than 3.5% (a smoothstep's steepest
                big = 1.5 * abs(tw_t - cur_w) / max(cur_w, 1e-6) * (g.t(ph.q + 1) - ph.t) / 0.035  # is 1.5 its mean)
                if ez:  # a frame's own ease: at most the camera QA's 4% on its steepest photo (exact)
                    s_ = np.linspace(0, 1, 101)
                    big = float(np.max(6 * s_ * (1 - s_) * abs(tw_t - cur_w) / (cur_w + (tw_t - cur_w) * s_ * s_ * (
                        3 - 2 * s_)))) * (g.t(ph.q + 1) - ph.t) / 0.039
                d_ = float(ez[0] if fr_.get("ease") else ez[1]) if ez else float(cam.get("ease", 0.0))
                spr["w"] = (cur_w, tw_t, ph.t, max(beat, big, d_))
            spr["back"] = fr_.get("ease")
            spr["zr"] = zr_
            tw = cur_w
            look_ = float(cam.get("look", 2.0 / w_))  # s of his speed: 2/stiffness cancels the spring's lag
            L_ = look_ * spr["v"]  # the look-ahead (it cancels the spring's lag); with soft_bounds where his torso
            if cam.get("soft_bounds") and tt_ is not None and len(tt_) > 1:  # really is then (it stops at a wall: the
                L_ = float(np.interp(ph.t + look_, tt_, txa_)) - rx_t  # frame slows with him)
            lead_ = (float(cam.get("lead", 0.07)) * (0.4 if vertical else 1.0) if not fv_ else
                     0.5 - float(fv_.get("at", 0.3))) * tw  # (paint-in, 9:16: him at `at`)
            tx = rx_t + spr["off"] + L_ + lead_
            # vertical: the ground's framing, raised as his torso rises above a runner's (a jump, a climb), from his
            # torso track (the drawings' boxes change height from drawing to drawing)
            ty = ground_cy(tw, rx_t) - float(cam.get("follow_y", 0.85)) * torso_lift(ph.q, rx_t)
            # a frame's offsets (dx ahead, dy down) or a fixed target (cx, cy; the 9:16 follows him with follow_v),
            # weighed in over its ease and back out after it (a step in the spring's target would jolt it)
            fb0, fb1, ft0, fd = spr.get("fb", (0.0, 0.0, ph.t, 0.0))
            e2 = 1.0 if fd <= 0 else float(np.clip((ph.t - ft0) / fd, 0, 1))
            b_ = fb0 + (fb1 - fb0) * e2 * e2 * (3 - 2 * e2)
            fl = spr["fl"] = fr_ or spr.get("fl")
            if fl:
                fix_ = not (vertical and fl.get("follow_v"))
                cxk_ = "cx_v" if vertical and "cx_v" in fl else "cx"  # (v8b: the 9:16's own fixed target)
                ox = float(fl[cxk_]) - tx if "cx" in fl and fix_ else float(fl.get("dx" + v_, fl.get("dx", 0) * (
                    0.3 if vertical else 1.0)))  # (the 9:16 without its own dx: muted, as the follow)
                oy = float(fl["cy"]) - ty if "cy" in fl and fix_ else float(fl.get("dy" + v_, fl.get("dy", 0)))
                c_ = float(fl.get("centre", 0.0))
                if c_ > 0:  # centred on him: no lead, his figure's middle at the frame's middle height (his torso
                    if not ("cx" in fl and fix_):  # as far above a runner's as it is); nor the half photo the spring
                        ox -= c_ * (lead_ + 0.5 * spr["v"] * dt)  # runs ahead (it chases each photo's target over
                        # the interval before it)
                    if not ("cy" in fl and fix_):  # (a fall coming: where he will be by the time the vertical
                        lift_ = torso_lift(ph.q, rx_t)  # spring gets there, so the landing stays in frame)
                        if tt_ is not None and len(tt_) > 1:
                            t_a = ph.t + 2.0 / w_
                            ya_ = float(np.interp(t_a, tt_, tya_))
                            xa_ = float(np.interp(t_a, tt_, txa_))
                            lift_ = min(lift_, max(0.0, sc.feet_at(xa_) - 0.5 * 450 * u - ya_))
                        oy += c_ * (sc.feet_at(rx_t) - 0.5 * 450 * u - lift_ - ty)
                if bool(fr_) != (fb1 > 0.5):  # it begins or ends: eased over its ease, a beat at least, and long
                    n_ = math.sqrt(150 * max(abs(ox), abs(oy)) / max(tw, 1e-6))  # enough (photos) that the pan
                    ezf = fl.get("ease")  # changes by no more than about 4% of the frame in one photo
                    spr["fb"] = (b_, 1.0 if fr_ else 0.0, ph.t, max(g.t(ph.q + 4) - ph.t, n_ * (g.t(ph.q + 1) - ph.t),
                                                                     float(ezf[0 if fr_ else 1]) if ezf else 0.0))
                tx, ty = tx + b_ * ox, ty + b_ * oy
            walled = False
            if cam.get("soft_bounds"):
                # where the spring will sit (its target less the look-ahead, which only cancels its lag): kept where
                # the 9:16 holds his figure over the spring's lag (this photo's box and the next ones' within
                # 1/stiffness s) and where the world can fill the crop (low headroom under the world's top), so the
                # spring eases there and out again instead of the keep-inside rule and the final clamp shoving it
                ax_, ay_ = tx - L_, ty
                if vertical:
                    bb = [bx0, by0, bx1, by1]
                    for ph_ in photos[n + 1:]:
                        if ph_.t > ph.t + 1.0 / w_:
                            break
                        bq_ = fig(ph_.q)
                        bb = [min(bb[0], bq_[0]), min(bb[1], bq_[1]), max(bb[2], bq_[2]), max(bb[3], bq_[3])]
                    ax_, ay_, _ = _keep_inside(ax_, ay_, min(tw, w_max), bb, w_max, widen=False)
                if sc.world is not None:  # (the lowest the frame must go here or where he will be over the next
                    sh_ = []  # 0.45 s: the spring is down under a drop in the world's top by the time he gets there)
                    for tau_ in (0.0, 0.15, 0.3, 0.45):
                        d_ = (float(np.interp(ph.t + tau_, tt_, txa_)) - rx_t) if tt_ is not None and len(tt_) > 1 \
                            else spr["v"] * tau_
                        sh_.append(_clamp_y(sc, ax_ + d_, ay_, min(tw, w_max), crop_h(min(tw, w_max)))[1] - ay_)
                    ay_ += max(sh_, key=abs) if min(sh_) >= 0 or max(sh_) <= 0 else sh_[0]
                # (held, or he is about to leave the frame where the spring is now: the vertical spring as stiff as
                # the horizontal, so it gets there in time: a fall, a low top)
                ny_ = _keep_inside(spr["x"], spr["y"], min(tw, w_max), bb, w_max, widen=False)[1] \
                    if vertical and spr["y"] is not None else spr["y"]
                sy_ = spr["y"] if spr["y"] is not None else ty  # (only while the spring is on the wrong side of
                walled = (ay_ - ty) * (ay_ - sy_) > 4 or (ny_ is not None and abs(ny_ - sy_) > 2)  # where it must be)
                tx, ty = ax_ + L_, ay_
            # (towards a fixed cy the vertical spring stiffens to the horizontal's: the target no longer jumps with him)
            # (and on him, centred: as stiff as far as it is centred)
            st_ = 1.0 if fl and ("cy" in fl and not (vertical and fl.get("follow_v")) or
                                 ("9:16" if vertical else "16:9") in (fl.get("yr") or ())) else \
                float(fl.get("centre", 0.0)) if fl else 0.0
            wy = w_ / 3 * (1 + 2 * b_ * st_)
            if cam.get("soft_bounds"):  # (stiffened over about a quarter second, not at once; soft again when free,
                wk_ = spr.get("wy", wy)  # so a release eases out slowly)
                wy = spr["wy"] = wk_ + (w_ - wk_) * min(1.0, dt / 0.25) if walled else wy
            if spr["y"] is None:
                spr["y"] = ty
            if dt > 0:
                n_sub = max(1, int(math.ceil(dt / 0.01)))
                h_ = dt / n_sub
                for _ in range(n_sub):  # semi-implicit Euler on x'' = w^2 (target - x) - 2 w x'
                    spr["vx"] += h_ * (w_ * w_ * (tx - spr["x"]) - 2 * w_ * spr["vx"])
                    spr["x"] += h_ * spr["vx"]
                    spr["vy"] += h_ * (wy * wy * (ty - spr["y"]) - 2 * wy * spr["vy"])
                    spr["y"] += h_ * spr["vy"]
                if cam.get("soft_bounds") and sc.world is not None:  # the spring itself stays where the world can
                    spr["x"], spr["y"] = _clamp_y(sc, spr["x"], spr["y"], min(tw, w_max), crop_h(min(tw, w_max)))
            if fresh and cams and not cam.get("cut"):  # out of another framing (the box camera): from where that
                bx_, by_, bw_ = cams[-1].base or (cams[-1].cx, cams[-1].cy, cams[-1].cw)  # one was, eased after him
                vx_ = 0.0  # (v8b, the director at 78.8: moving as that one was, so the frame does not stop and jump)
                if len(cams) > 1 and dt > 0:
                    b0_ = cams[-2].base or (cams[-2].cx, cams[-2].cy, cams[-2].cw)
                    vx_ = (bx_ - b0_[0]) / max(photos[n - 1].t - photos[n - 2].t, 1e-6)
                spr["x"], spr["vx"], spr["y"], spr["vy"] = bx_ + vx_ * dt, vx_, by_, 0.0
                beat = g.t(ph.q + 4) - ph.t
                big = 1.5 * abs(tw_t - bw_) / max(bw_, 1e-6) / 0.035 * (g.t(ph.q + 1) - ph.t)
                spr["w"] = (bw_, tw_t, ph.t, max(beat, big))
                tw = bw_
            elif fresh:  # where the spring settles at his speed (the look-ahead cancels its lag)
                spr["x"], spr["vx"] = tx - look_ * spr["v"], spr["v"]
            cur = [spr["x"], spr["y"], tw]
            sprung = True
        elif preset == "box":
            tx, ty, tw = _box_target(ch, cam, ph.q, vertical, u)
            free = True
        elif preset == "frame":
            # a set stretch of street (the row of boxes before he walks in); 9:16 as wide as it can, on its middle
            x0f, x1f = (float(v) for v in cam["x"])
            tw = (x1f - x0f) if not vertical else w_max
            tx = (x0f + x1f) / 2
            ty = float(cam["cy"]) if "cy" in cam else ground_cy(min(tw, w_max), tx)
            free = True
        elif preset == "stop":
            # wide on the stop, then a slow push in, photo by photo
            uu = np.clip((ph.q - seg.q0) / max(seg.q1 - seg.q0 - 1, 1), 0, 1)
            uu = uu * uu * (3 - 2 * uu)
            z0, z1 = cam.get("zoom", [1.24, 1.0]) if isinstance(cam.get("zoom"), list) else (1.24, 1.0)
            l0, l1 = cam.get("lead", [170, 110]) if isinstance(cam.get("lead"), list) else (170, 110)
            cw = base * (z0 + (z1 - z0) * uu) if not vertical else base * (1.08 - 0.08 * uu)
            lead = (l0 + (l1 - l0) * uu) * (0.2 if vertical else 1.0) * u
            tx = rx + lead + float(cam.get("drift", 1.0)) * u * (ph.q - seg.q0) * (0 if vertical else 1)
            ty, tw = ground_cy(cw, rx), cw
            fy = float(cam.get("follow_y", 0.0))  # a freeze mid-air: follow him up (0: the ground line, as p6)
            if fy:
                ty += fy * min(0.0, ry - (sc.feet_at(rx) - 0.5 * 430 * u))
            free = True
        elif "cx" in hint or "dx" in hint:
            # a template's framing (the leap's flight): figure-relative x, the ground-relative y, snap
            z = float(hint.get("zoom", 1.0))
            tw = base * (1 + (z - 1) * (0.5 if vertical else 1.0))
            if vertical and "zoom_v" in hint:  # the 9:16's own zoom (a framing both aspects share: the ending)
                tw = base * float(hint["zoom_v"])
            # follow_v: a set 16:9 frame (the leap's wide shot) that the 9:16 does not take: it follows him
            fixed = "cx" in hint and not (vertical and hint.get("follow_v"))
            tx = (float(hint["cx_v" if vertical and "cx_v" in hint else "cx"]) if fixed
                  else rx + float(hint.get("dx", 0)) * (0.3 if vertical else 1.0))
            if vertical and "dx_v" in hint and not fixed:
                tx = rx + float(hint["dx_v"])
            ty = ground_cy(tw, rx) + float(hint.get("dy", 0))
            if vertical and "dy_v" in hint:
                ty = ground_cy(tw, rx) + float(hint["dy_v"])
            if "cy" in hint and fixed:  # (a frame's fixed target)
                ty = float(hint["cy"])
            free = bool(hint.get("free", True))
        else:
            tw = base * zoom * float(hint.get("zoom", 1.0))
            lead = float(cam.get("lead", 0.07)) * (0.4 if vertical else 1.0) * tw
            tx = rx + lead
            # keep up: aim ahead by the lag the smoothing would leave at his speed, so a fast run stays centred
            # (motion M0 default; legacy framing does not)
            if cam.get("keep_up", ch.motion == "m0"):
                tx += max(v_fig, 0.0) * (1 - a_x) / a_x
            ry_ground = sc.feet_at(rx) - 0.5 * 430 * u
            ty = ground_cy(tw, rx) + float(cam.get("follow_y", 0.85)) * (ry - ry_ground)
            if cam.get("eye") is not None:  # at eye level: the frame's middle at that share of his height
                ty = sc.feet_at(rx) - float(cam["eye"]) * 430 * u
            if preset == "hold" and cams and not (abs(ph.q - seg.q0) < 1e-6):
                tx, ty, tw = cur
        t_prev = ph.t
        snap = snap or bool(hint.get("snap"))
        tw = min(tw, w_max)
        if sprung:
            cur = [cur[0], cur[1], min(cur[2], w_max)]
        elif cur is None or snap:
            cur = [tx, ty, tw]
        else:
            cur = [cur[0] + a_x * (tx - cur[0]), cur[1] + 0.5 * (ty - cur[1]), cur[2] + 0.7 * (tw - cur[2])]
            if not free and not vertical and cam.get("band", True):
                # keep him in the central 9:16 band of the 16:9 (about +-0.12 of the crop width)
                cur[0] = float(np.clip(cur[0], rx - 0.12 * cur[2], rx + 0.14 * cur[2]))
        regions = None
        look = None
        if preset == "box":  # the box and every stencil already painted stay in frame, whatever the smoothing did
            pr = ch.props.get(str(cam.get("box", "box")))
            # 16:9 holds the stencils painted so far; 9:16 holds them all from the start (a steady frame that has
            # room for every rule, instead of stepping out on each one)
            regions = pr.regions(ph.q, every=vertical) if pr is not None else None
            mf = float(cam.get("margin", 0.05))  # the look crops about 3% a side and jitters
            if regions:
                cur = list(_contain(cur[0], cur[1], cur[2], regions, vertical, mf, w_max))
            look = _box_look(g, cam, ph.q, vertical)
        cx, cy, cw = cur
        keep = cam.get("keep", True) is not False  # (keep: false, a framing on something else: the cold open's splat)
        if vertical and keep:
            cx, cy, cw = _keep_inside(cx, cy, cw, (bx0, by0, bx1, by1), w_max)
            if sprung and cam.get("soft_bounds"):  # (the spring carries on from where the rule put it, instead of
                spr["x"], spr["y"] = cx, cy  # springing back next photo)
            cur = [cx, cy, cw]
        if look:  # the look's push toward its region, eased in and out over the box framing (on top of the
            # smoothed framing, which carries on underneath, so the push leaves it where it would have been)
            e, (lx, ly, lw), _ = look
            cx, cy, cw = cx + e * (lx - cx), cy + e * (ly - cy), cw + e * (lw - cw)
            regions = None  # the punches and nudges move it freely (the region's crop has its own margin)
        elif ch.motion == "m0" and preset not in ("box", "frame") and not (vertical and preset == "spring") and keep:
            # (the 9:16 spring keeps him inside by its own rule above; this one's ratio is the 16:9's)
            # motion M0: 16:9 keeps him inside too (moved, not widened: a jump up onto a cabinet outruns the
            # smoothing)
            cx, cy, cw = _keep_inside(cx, cy, cw, (bx0, by0, bx1, by1), w_max, widen=False, margin=0.05, ratio=9 / 16)
            cur = [cx, cy, cw]
        cw_p = cw * (1 - kick_punch(ph.t, kicks, impacts)) * (1 - float(hint.get("punch", 0.0)))
        rng = np.random.default_rng(1000 * ch.seed + ph.k)
        rot = snare_roll(ph.t, snares) + rng.normal(0, 0.15)
        cx_p = cx + rng.normal(0, 3 * u)
        cy_p = cy + rng.normal(0, 3 * u)
        if hint.get("shake"):  # a hit shakes the camera a little (a template's hint: the pound, the explosion)
            sk = float(hint["shake"]) * u
            cx_p += rng.normal(0, sk)
            cy_p += rng.normal(0, 0.6 * sk)
            rot += rng.normal(0, 0.5)
        chh = crop_h(cw_p)
        if vertical and keep:  # the punch must not push him out either
            cx_p, cy_p, _ = _keep_inside(cx_p, cy_p, cw_p, (bx0, by0, bx1, by1), w_max, widen=False)
        if regions:  # a punch or the hand-held nudge must not cut a stencil either (it gives way first)
            cx_p, cy_p, cw_p = _contain(cx_p, cy_p, cw_p, regions, vertical, mf, w_max)
            chh = crop_h(cw_p)
        if cam.get("x_min") is not None:  # a set left edge for the crop (the street's own rule, pinned: the cold
            cx_p = max(cx_p, float(cam["x_min"]) + cw_p / 2 + 30)  # open's framing whatever the street's end does)
        # (`x_max`, a right edge: an eased shift over the whole plan, _x_max)
        cx_p, cy_p = sc.clamp_crop(cx_p, cy_p, cw_p, chh)  # only street or world pixels in the crop
        jq = next((q0 for q0 in joins if q0 - 1e-6 <= ph.q < q0 + JOIN_BLEND - 1e-6), None)
        if jq is not None and cams:
            if blend_from is None or blend_from[0] != jq:
                # the framing on the last photo before the join, carried on at the pace it was moving (a camera
                # following him does not stop dead at the join and then catch up)
                p1_, p0_ = cams[-1], cams[-2] if len(cams) > 1 else cams[-1]
                b1_, b0_ = p1_.base or (p1_.cx, p1_.cy, p1_.cw), p0_.base or (p0_.cx, p0_.cy, p0_.cw)
                blend_from = (jq, p1_, (b1_[0] - b0_[0], b1_[1] - b0_[1], b1_[2] - b0_[2]), len(cams))
            pc, st_ = blend_from[1], ch.step_at(jq)
            m_ = len(cams) + 1 - blend_from[3]  # photos since the last one before the join
            gx, gy, gw = (pc.cx + m_ * blend_from[2][0], pc.cy + m_ * blend_from[2][1], pc.cw + m_ * blend_from[2][2])
            v = float(np.clip((ph.q - jq + st_) / (JOIN_BLEND + st_), 0, 1))
            e = v * v * (3 - 2 * v)
            cx_p, cy_p, cw_p = gx + e * (cx_p - gx), gy + e * (cy_p - gy), gw + e * (cw_p - gw)
            chh = crop_h(cw_p)
            cx_p, cy_p = sc.clamp_crop(cx_p, cy_p, cw_p, chh)
        cams.append(Cam(cx_p, cy_p, cw_p, chh, rot, look[0] if look else 0.0, (cx, cy, cw)))
        sprung_prev = sprung
    return _x_max(ch, photos, cams, fig)


def _x_max(ch, photos, cams: list[Cam], fig) -> list[Cam]:
    """A segment's `x_max` (v8): its photos' crops never reach right of it (30 px in). Each photo needs moving left
    by how far its crop's right edge passes that; the frames move by the smallest smooth shift that covers every
    photo's need: inside those segments each need held XMAX_HOLD s either side (a steady offset, so the camera's
    own small moves, punches and nudges stay) and eased out over XMAX_EASE s after that; outside them eased in and
    out over XMAX_EASE s, never so far that his figure leaves the frame."""
    need = np.zeros(len(cams))
    pinned = np.zeros(len(cams), bool)
    for k, (ph, c) in enumerate(zip(photos, cams)):
        xm = ch.segments[ph.seg].camera.get("x_max") if ph.seg is not None else None
        if xm is not None:
            pinned[k] = True
            need[k] = max(0.0, c.cx + c.cw / 2 - (float(xm) - 30))
    if not need.any():
        return cams
    ts = np.array([ph.t for ph in photos])
    idx = np.nonzero(need)[0]
    out = []
    for k, (ph, c) in enumerate(zip(photos, cams)):
        d = np.abs(ts[idx] - ts[k])
        if pinned[k]:  # (inside: the plateau, then the ease)
            d = np.maximum(d - XMAX_HOLD, 0.0)
        d = d / XMAX_EASE
        f = np.where(d < 1, 1 - d * d * (3 - 2 * d), 0.0)
        sh = float((need[idx] * f).max())
        if sh <= 1e-6:
            out.append(c)
            continue
        if not pinned[k]:  # (the ease outside the pinned segments: his figure stays in the frame)
            fx1 = fig(ph.q)[2]
            sh = min(sh, max(0.0, c.cx + c.cw / 2 - 0.06 * c.cw - fx1))
        b = c.base or (c.cx, c.cy, c.cw)
        out.append(Cam(c.cx - sh, c.cy, c.cw, c.ch, c.rot, c.look, (b[0] - sh, b[1], b[2])))
    return out


def _box_target(ch, cam: dict, q: float, vertical: bool, u: float) -> tuple[float, float, float]:
    """The box preset's framing at song 16th q: (cx, cy, crop width). The box takes `fill` of the crop height (0.5;
    in 9:16 `fill_w` of the width, 0.6); the crop follows the box's size by `track` (0.5: a step closer on each rule,
    but less than the box shrinks); push: [from, to, factor] eases the crop down by factor over that span (a slow push
    in, held until room starts); room: {from, x, margin}: from then on the 16:9 frame also holds x (the red light
    beside the box); steps: [from, to, factor]: a step closer on every bar between them (the batter: the camera steps
    in with each attack), the last bar's width `factor` of the first's. plan() then widens and moves the crop just enough to hold the box and every stencil painted so
    far (margin: 0.05 of the frame a side, for the look's crop and jitter), punches and nudges included. look: {from,
    to, back, region, margin} (_box_look): one slow push toward a region (his hood and the red light), blended over
    all that (nothing else is held in frame while it lasts)."""
    g = ch.grid
    pr = ch.props.get(str(cam.get("box", "box")))
    if pr is None:
        raise ValueError(f"camera box: no prop {cam.get('box', 'box')!r}")
    key, k0 = pr.geometry(q), pr.keys[0]
    x0, y0, x1, y1 = key.box
    bw = key.w * (1 + 2 * key.bulge)
    track = float(cam.get("track", 0.5))
    hh = (k0.h / float(cam.get("fill", 0.5))) * (key.h / k0.h) ** track
    push = cam.get("push")
    room = cam.get("room")
    q_room = g.resolve(room["from"])[0] if room else math.inf
    if push and q < q_room:
        qa, qb = g.resolve(push[0])[0], g.resolve(push[1])[0]
        v = float(np.clip((q - qa) / max(qb - qa, 1e-6), 0, 1))
        hh *= 1 + (float(push[2]) - 1) * v * v * (3 - 2 * v)
    cx, cy = key.cx, (y0 + y1) / 2 - 0.04 * hh
    if vertical:
        tw = max(bw / float(cam.get("fill_w", 0.6)), (key.h / 0.62) * 9 / 16)
        return cx, cy, tw
    tw = hh * 16 / 9
    if room and q >= q_room - 1e-6:
        rx = float(room["x"])
        mg = float(room.get("margin", 160)) * u
        a, b = min(x0 - 0.3 * key.w, rx - mg), max(x1 + 0.3 * key.w, rx + mg)
        tw = max(tw, (b - a) * 1.06)
        cx = (a + b) / 2
    st = cam.get("steps")
    if st:  # a step closer on every bar from `from` to `to` (factor: the last bar's width against the first's)
        qa, qb = g.resolve(st[0])[0], g.resolve(st[1])[0]
        nb = max(int(round((qb - qa) / 16)), 1)
        k = int(np.clip(math.floor((q - qa) / 16 + 1e-6), 0, nb - 1)) if q >= qa else 0
        tw *= 1 + (float(st[2]) - 1) * (k / max(nb - 1, 1) if q >= qa else 0.0)
    return cx, cy, tw


def _box_look(g, cam: dict, q: float, vertical: bool):
    """The box preset's look: {from, to, back, region: [x0, y0, x1, y1], margin}: one slow eased push from the box
    framing at `from` to the crop round `region` (street px: his hood and the red light) at `to`, eased back out to
    the box framing by `back` (default: held to the segment's end). Returns (weight 0..1, (cx, cy, crop width) of the
    region's crop, region) at song 16th q, or None outside it."""
    lk = cam.get("look")
    if not lk:
        return None
    qa, qp = g.resolve(lk["from"])[0], g.resolve(lk["to"])[0]
    qb = g.resolve(lk["back"])[0] if lk.get("back") is not None else math.inf
    if not qa <= q < qb:
        return None
    v = (q - qa) / max(qp - qa, 1e-6) if q < qp else (1 - (q - qp) / max(qb - qp, 1e-6) if qb < math.inf else 1.0)
    v = float(np.clip(v, 0, 1))
    e = v * v * (3 - 2 * v)
    if e <= 0:
        return None
    x0, y0, x1, y1 = (float(a) for a in lk["region"])
    k = 1 - 2 * float(lk.get("margin", 0.08))
    ratio = 9 / 16 if vertical else 16 / 9  # width over height
    lw = max((x1 - x0) / k, (y1 - y0) / k * ratio)
    return e, ((x0 + x1) / 2, (y0 + y1) / 2, lw), (x0, y0, x1, y1)


def _contain(cx: float, cy: float, cw: float, regions: list, vertical: bool, margin: float,
             w_max: float) -> tuple[float, float, float]:
    """Widen (never narrow) and move a crop as little as possible so every region (x0, y0, x1, y1) is inside it with
    `margin` of the frame to spare on every side."""
    ux0, uy0 = min(r[0] for r in regions), min(r[1] for r in regions)
    ux1, uy1 = max(r[2] for r in regions), max(r[3] for r in regions)
    ratio = 9 / 16 if vertical else 16 / 9  # width over height
    k = 1 - 2 * margin
    need = max((ux1 - ux0) / k, (uy1 - uy0) / k * ratio)
    cw = min(max(cw, need), w_max)
    ch = cw / ratio
    lo, hi = ux1 + margin * cw - cw / 2, ux0 - margin * cw + cw / 2
    cx = (ux0 + ux1) / 2 if lo > hi else float(np.clip(cx, lo, hi))
    lo, hi = uy1 + margin * ch - ch / 2, uy0 - margin * ch + ch / 2
    cy = (uy0 + uy1) / 2 if lo > hi else float(np.clip(cy, lo, hi))
    return cx, cy, cw


def _clamp_y(sc, cx: float, cy: float, cw: float, chh: float, margin: float = 30, reach: int = 2400):
    """A crop's centre moved only up or down (the nearest first, in 16 px steps) until the world can fill it (a
    low top: the frame goes down under it, not sideways); the scene's own clamp if no height at this x can."""
    hw, hh = cw / 2 + margin, chh / 2 + margin
    for d in range(0, reach + 1, 16):
        for y in ((cy + d, cy - d) if d else (cy,)):
            if sc.street.valid_box(cx - hw, y - hh, cx + hw, y + hh):
                return float(cx), float(y)
    return sc.clamp_crop(cx, cy, cw, chh, margin)


def _keep_inside(cx, cy, cw, box, w_max, widen=True, margin=0.05, ratio=16 / 9):
    """Move (and if needed widen) a crop (9:16 by default; ratio: its height over its width) so the figure's box is
    inside it with a margin."""
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    if widen:
        need_w = max(bw / (1 - 2 * margin), bh / (1 - 2 * margin) / ratio)
        if need_w > cw:
            cw = min(need_w, w_max)
    ch = cw * ratio
    m = margin * cw
    lo, hi = bx1 + m - cw / 2, bx0 - m + cw / 2
    cx = (bx0 + bx1) / 2 if lo > hi else float(np.clip(cx, lo, hi))
    mv = margin * ch
    lo, hi = by1 + mv - ch / 2, by0 - mv + ch / 2
    cy = (by0 + by1) / 2 if lo > hi else float(np.clip(cy, lo, hi))
    return cx, cy, cw


def framing_stats(cams: list[Cam], boxes: list[tuple], vertical: bool) -> dict:
    """How well the frames hold him: for 16:9 the torso offset from the centre (fraction of the frame width) and how
    often it leaves the central 9:16 band (15.8% each side); for 9:16 the share of the figure box outside the frame."""
    off, out_band, outside = [], 0, []
    for c, (bx0, by0, bx1, by1) in zip(cams, boxes):
        rx = (bx0 + bx1) / 2
        off.append(abs(rx - c.cx) / c.cw)
        if vertical:
            fx0, fx1 = c.cx - c.cw / 2, c.cx + c.cw / 2
            fy0, fy1 = c.cy - c.ch / 2, c.cy + c.ch / 2
            area = max((bx1 - bx0) * (by1 - by0), 1)
            ix = max(0, min(bx1, fx1) - max(bx0, fx0)) * max(0, min(by1, fy1) - max(by0, fy0))
            outside.append(1 - ix / area)
        elif off[-1] > c.ch * 9 / 16 / 2 / c.cw:  # the torso centre outside the central 9:16 band
            out_band += 1
    st = {"max_offset": max(off, default=0.0), "mean_offset": float(np.mean(off)) if off else 0.0}
    if vertical:
        st["max_outside"] = max(outside, default=0.0)
        st["photos_cut"] = int(sum(o > 0.001 for o in outside))
    else:
        st["photos_leaving_band"] = out_band
    return st
