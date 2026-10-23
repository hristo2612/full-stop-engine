"""Automatic QA of a choreography, photo by photo (`aimv paint check` and every render).

Rules (errors stop a review copy unless --force; warnings are listed):

    solid       part of the figure is hidden behind a solid obstacle (a cabinet, a bin, an AC unit: scene.SOLID_KINDS),
                more than 0.5% of its paint and 40 px; the top 14 px of a solid part are a touch band (hands and feet
                may rest on its top); a move's `behind: <obstacle>` allows it
    head        the head (the top 20% of the figure's box) hidden: behind a solid obstacle (over 10% of its paint) an
                error; behind a thin occluder or a parallax layer (a pipe, a pole: over 15%) a warning, as paint may
                go behind them
    ground      the feet below the ground line (the lowest paint more than 2 px under the wall's foot): the paint is
                cut there
    no_paint    over 25% of the figure where paint cannot go (glass, a gap, the pavement)
    frame       the figure cut by the 16:9 or 9:16 frame (over 2% of its box outside what the look keeps)
    backtrack   the torso goes back by more than 12 px between two photos (an error: the director's note on v1); in a box
                he pulls back to hit it again by design (inside, pound)
    lurch       the travel per photo changes by more than 35% (a warning: quick moves do this on purpose)
    repeat      a drawing repeats in a move that is not a cycle (a warning)
    hold        one drawing up on more than 2 photos in a row outside a true freeze (a warning; a move's `still:
                true` marks a deliberate stillness)
    canon       the drawing shown has a canon error (canon.py, paint/poses/canon.json: scale, smear, colour, shoes)
    crossing    a plate object (a pipe or pole: a thin occluder or a parallax layer) across more than 4% of him while
                he stays put on the ground (his torso within 40 px over 4 photos, his feet or body on the ground line:
                a hold, a crouch, a freeze; a slow flight past a pole is not one): an error (the director's
                note on v4: the pole right in front of him before the leap); not the obstacle his move is on (the
                pipe he climbs)
    feet        contact (v7): in every grounded photo (a parkour photo on the ground, a clip frame whose feet are on
                its clip's floor) his lowest paint within FEET_TOL px of the surface under it (the level's feet line,
                or the prop's top he stands on); an error
    takeoff     his take-off foot past the edge he leaves from (a parkour jump): an error
    inside      in flight, his paint inside a solid obstacle's box below its top, or below the feet line of the
                ground under him (inside a roof or the pavement): an error
    grip        hanging (a parkour clip's hold frames): his hands (the top of his paint) more than HOLD_TOL px from the
                bar or beam: an error
    fist        in the box (inside, pound): his paint past the outer edge of the box's side lines (a punch through
                the wall): an error
    reveal      v8 paint-in (reveal.py): a grounded photo whose feet are on world not painted in yet: an error
    camera      a photo-to-photo camera jump of more than CAM_JUMP of the frame width outside a cut (a snap): a
                warning

Every flagged figure is checked once, at the first photo that shows it. render() turns the result into
<name>-qa.json, <name>-qa.jpg (every flagged photo, the hidden paint tinted red, labelled with time and rule) and
<name>-moves.jpg (every move from 4 photos before to 4 after, labelled).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .check import HANDOVER_DV, figure_track, motion_stats
from .choreo import Choreo

LOOK_CROP = 1 / 1.06  # the MUTO look keeps the middle 1/1.06 of every photo
MOVES_PER_LINE = 120  # photos per line of the moves sheet (a longer move wraps onto more lines)
STILL_N, STILL_PX, CROSS = 4, 40, 0.04  # he stays put: 4 photos within 40 px; a plate object over 4% of him
FEET_TOL, HOLD_TOL, FIST_TOL = 6.0, 20.0, 4.0  # px for a 450 px figure: feet on the surface, hands on a bar, a fist
TOUCH_TOL = 25.0  # px (0.1 m) a flight photo touching a surface (flags touch) may reach into it: its skin smear
CAM_JUMP = 0.04  # a camera jump over 4% of the frame width in one photo (outside a cut) is a snap


def _layers_mask(ch: Choreo, x0: int, y0: int, w: int, h: int, cx: float) -> np.ndarray:
    """The parallax layers (the red light pole) over a street box, where they stand for a camera at cx."""
    out = np.zeros((h, w), np.float32)
    for L in ch.scene.layers:
        d = ch.scene.layer(L)
        dx = int(round(-d["parallax"] * (cx - d["x"])))
        a0 = d["a0"] + dx
        al = d["alpha"]
        o = int(d.get("y0", 0))  # the layer's first row (0 on the street; the world's layers start above it)
        c0, c1 = max(a0, x0), min(a0 + al.shape[1], x0 + w)
        r0, r1 = max(y0, o), min(y0 + h, o + al.shape[0])
        if c1 > c0 and r1 > r0:
            out[r0 - y0:r1 - y0, c0 - x0:c1 - x0] = np.maximum(out[r0 - y0:r1 - y0, c0 - x0:c1 - x0],
                                                              al[r0 - o:r1 - o, c0 - a0:c1 - a0])
    return out


def run(ch: Choreo, hist, photos, cams: dict) -> dict:
    """QA every figure the photos show. cams: aspect -> [Cam] for `photos`. Returns {"errors", "warnings", "flags":
    [...], "masks": {k: (x0, y0, mask)}} (masks: the hidden paint, for the QA sheet; not in the json)."""
    from . import canon as CN
    from . import truelook as TL

    sc, lib, g = ch.scene, ch.lib, ch.grid
    u = ch.u
    flags, masks = [], {}
    drawings = CN.load(lib.d)
    first_k: dict[int, int] = {}
    shown = {}
    for i, e in enumerate(ch.events):
        if e.layer in ("wall", "air") and e.pose is not None and lib.m(e.pose)["kind"] == "pose":
            shown[round(e.q, 4)] = i
    pk = {round(p.q, 4): n for n, p in enumerate(photos)}
    # where he stays put: the torso within STILL_PX over STILL_N photos (a hold, a crouch, a freeze)
    tr = figure_track(ch)
    txy = {round(r["q"], 4): (r["x"], r["y"]) for r in tr}
    painted_q = sorted(txy)
    still_q = set()
    for a_ in range(len(painted_q)):
        win = [txy[qq] for qq in painted_q[a_:a_ + STILL_N]]
        if len(win) == STILL_N and painted_q[a_ + STILL_N - 1] - painted_q[a_] <= 4 * STILL_N and \
                max(abs(p_[0] - win[0][0]) for p_ in win) <= STILL_PX * u:
            still_q.update(painted_q[a_:a_ + STILL_N])
    for q, i in sorted(shown.items()):
        if q not in pk:
            continue
        n = pk[q]
        ph = photos[n]
        e = ch.events[i]
        mv = ch.moves[e.move] if 0 <= e.move < len(ch.moves) else None
        where = mv.where if mv else ""
        d, fx0, fy0 = hist.place(i)
        X0, Y0 = int(math.floor(fx0)), int(math.floor(fy0))
        h, w = d.shape
        paint = d > 0.2
        tot = int(paint.sum())
        if tot == 0:
            continue
        base = {"k": ph.k, "t": round(ph.t, 3), "addr": g.addr(ph.q), "move": where, "pose": e.pose, "tag": e.tag}
        for cf in CN.errors_of(drawings, e.pose):
            flags.append({**base, "rule": "canon", "level": "error", "detail": f"{e.pose}: {cf['rule']}: {cf['why']}"})
        skip = tuple(e.flags.get("behind", ()))
        true_ = TL.is_true(ch, e)  # he is true (truelook.py): solid and in front of everything, no paint rules
        # solid occluders
        solid, touch = sc.occluders(X0, X0 + w, Y0, Y0 + h, solid=True, skip=skip)
        hid = paint & (solid > 0.5) & (touch < 0.5)
        nh = int(hid.sum())
        if nh > max(40, 0.005 * tot) and not true_:
            flags.append({**base, "rule": "solid", "level": "error",
                          "detail": f"{nh / tot:.1%} of him hidden behind a solid obstacle"})
            masks.setdefault(ph.k, []).append((X0, Y0, hid))
        # the head behind any occluder or layer
        ys = np.nonzero(paint.any(1))[0]
        hy1 = ys.min() + max(1, int(round(0.2 * (ys.max() - ys.min() + 1))))
        head = paint.copy()
        head[hy1:] = False
        allo, _ = sc.occluders(X0, X0 + w, Y0, Y0 + h, solid=None, skip=skip)
        allo = np.maximum(allo, _layers_mask(ch, X0, Y0, w, h, cams["16:9"][n].cx if "16:9" in cams else
                                             X0 + w / 2))
        hs = head & (solid > 0.5)  # behind a solid obstacle: an error
        ha = head & (allo > 0.5)  # behind a pipe, a pole, a ladder: paint may go there, a warning
        if true_:
            pass
        elif head.sum() and hs.sum() / head.sum() > 0.10:
            flags.append({**base, "rule": "head", "level": "error",
                          "detail": f"{hs.sum() / head.sum():.0%} of his head hidden behind a solid obstacle"})
            masks.setdefault(ph.k, []).append((X0, Y0, hs))
        elif head.sum() and ha.sum() / head.sum() > 0.15:
            flags.append({**base, "rule": "head", "level": "warn",
                          "detail": f"{ha.sum() / head.sum():.0%} of his head behind a thin occluder"})
            masks.setdefault(ph.k, []).append((X0, Y0, ha))
        # a plate object (a pipe, a pole: thin occluders and parallax layers) across him while he stays put
        low_ = Y0 + int(np.percentile(np.nonzero(paint)[0], 99.5))
        if q in still_q and low_ > sc.feet_at(fx0 + w / 2) - 40 * u:  # standing, crouched, lying: not mid-air
            own = tuple(str(mv.args[k_]) for k_ in ("obstacle", "pipe", "bar", "onto", "beam") if mv and k_ in mv.args)
            occ_x, _ = sc.occluders(X0, X0 + w, Y0, Y0 + h, solid=None, skip=skip + own)  # not what he climbs
            occ_x = np.maximum(occ_x, _layers_mask(ch, X0, Y0, w, h, cams["16:9"][n].cx if "16:9" in cams else
                                                   X0 + w / 2))
            cross = paint & (occ_x > 0.5) & ~(solid > 0.5)
            if cross.sum() > CROSS * tot:
                flags.append({**base, "rule": "crossing", "level": "error",
                              "detail": f"{cross.sum() / tot:.0%} of him behind a pipe or pole while he stays put"})
                masks.setdefault(ph.k, []).append((X0, Y0, cross))
        # feet below the ground, paint where paint cannot go (wall paint only)
        if e.layer == "wall":
            low = Y0 + int(np.percentile(np.nonzero(d > 0.4)[0], 99.5)) if (d > 0.4).any() else Y0 + h
            wb = sc.wall_base_at(fx0 + w / 2)
            if low > wb - 4:
                flags.append({**base, "rule": "ground", "level": "error",
                              "detail": f"his feet {low - (wb - 6):.0f} px below the ground line"})
            _, pnt = sc.masks(X0, X0 + w, Y0, Y0 + h)  # his box's rows (the world's above the street)
            reg = pnt
            dd = (d[:reg.shape[0], :reg.shape[1]] > 0.3)
            if dd.sum() and not true_:
                fr = float((dd & (reg < 0.5)).sum() / dd.sum())
                if fr > 0.25:
                    flags.append({**base, "rule": "no_paint", "level": "error",
                                  "detail": f"{fr:.0%} of him where paint cannot go"})
        flags.extend(_contact(ch, hist, i, e, mv, d, X0, Y0, paint, base, masks, ph))
        first_k[i] = ph.k
    # camera snaps: its motion changing by more than CAM_JUMP of the frame width from one photo to the next (a pan
    # that starts or stops at once), or its zoom changing by more than CAM_JUMP in one photo, outside a cut
    for a, cl in cams.items():
        bs = [c.base or (c.cx, c.cy, c.cw) for c in cl]  # without the punches and the hand-held nudge
        for n_ in range(2, len(cl)):
            p1 = photos[n_]
            seg = ch.segments[p1.seg]
            cut = any(abs(photos[m].q - ch.segments[photos[m].seg].q0) < 1e-6 and
                      (ch.segments[photos[m].seg].camera.get("cut") or ch.segments[photos[m].seg].camera.get("snap"))
                      for m in (n_ - 1, n_))
            b0, b1, b2 = bs[n_ - 2], bs[n_ - 1], bs[n_]
            jump = max(abs((b2[0] - b1[0]) - (b1[0] - b0[0])), abs((b2[1] - b1[1]) - (b1[1] - b0[1]))) / max(b1[2], 1)
            dz = abs(b2[2] - b1[2]) / max(b1[2], 1)
            if not cut and (jump > CAM_JUMP or dz > CAM_JUMP):
                flags.append({"k": p1.k, "t": round(p1.t, 3), "addr": g.addr(p1.q), "move": "", "pose": "", "tag": "",
                              "rule": "camera", "level": "warn", "aspect": a,
                              "detail": f"the {a} camera's pan changes by {jump:.1%} of the frame (zoom {dz:.1%}) "
                                        f"in one photo"})
    # the frame: every photo, both formats, what the look keeps
    keys = sorted(hist.fig_box)
    for a, cl in cams.items():
        for ph, c in zip(photos, cl):
            j = int(np.searchsorted(keys, ph.q + 1e-6, side="right")) - 1
            if j < 0 or ch.segments[ph.seg].camera.get("keep", True) is False:  # (a deliberate framing elsewhere)
                continue
            bx0, by0, bx1, by1 = hist.fig_box[keys[j]]
            if getattr(c, "look", 0.0) > 0:  # the box camera's look pushes in on his hood: his top third stays in
                by1 = by0 + 0.33 * (by1 - by0)
            hw, hh_ = c.cw / 2 * LOOK_CROP, c.ch / 2 * LOOK_CROP
            ix = max(0, min(bx1, c.cx + hw) - max(bx0, c.cx - hw)) * max(0, min(by1, c.cy + hh_) - max(by0, c.cy - hh_))
            out = 1 - ix / max((bx1 - bx0) * (by1 - by0), 1)
            if out > 0.02:
                flags.append({"k": ph.k, "t": round(ph.t, 3), "addr": g.addr(ph.q), "move": "", "pose": "",
                              "tag": "", "rule": "frame", "level": "error",
                              "detail": f"{out:.0%} of him outside the {a} frame", "aspect": a})
    # motion
    st = motion_stats(ch)
    for d_, r in st["backtracks"]:
        flags.append({"k": r["k"], "t": round(r["t"], 3), "addr": g.addr(r["q"]), "move": ch.moves[r["move"]].where,
                      "pose": r["pose"], "tag": r["tag"], "rule": "backtrack", "level": "error",
                      "detail": f"back {-d_:.0f} px"})
    for c_, ratio, r in st["lurches"]:
        flags.append({"k": r["k"], "t": round(r["t"], 3), "addr": g.addr(r["q"]), "move": ch.moves[r["move"]].where,
                      "pose": r["pose"], "tag": r["tag"], "rule": "lurch", "level": "warn",
                      "detail": f"travel changes by {c_:.0f} px ({ratio:.0%})"})
    for r in st.get("handovers", []):  # (v8b) a clip handing over to a run, or a run to a clip, at a jump in speed
        if abs(r["dv"]) > HANDOVER_DV:
            flags.append({"k": r["k"], "t": round(r["t"], 3), "addr": g.addr(r["q"]),
                          "move": ch.moves[r["move"]].where if 0 <= r["move"] < len(ch.moves) else "",
                          "pose": r["pose"], "tag": r["tag"], "rule": "handover", "level": "warn",
                          "detail": f"{r['dir']} at {r['v0']:.1f} to {r['v1']:.1f} m/s"})
    for r in st["repeats"]:
        flags.append({"k": r["k"], "t": round(r["t"], 3), "addr": g.addr(r["q"]), "move": ch.moves[r["move"]].where,
                      "pose": r["pose"], "tag": r["tag"], "rule": "repeat", "level": "warn",
                      "detail": f"{r['pose']} again"})
    for r in st["holds"]:
        flags.append({"k": r["k"], "t": round(r["t"], 3), "addr": g.addr(r["q"]),
                      "move": ch.moves[r["move"]].where if 0 <= r["move"] < len(ch.moves) else "",
                      "pose": r["pose"], "tag": r["tag"], "rule": "hold", "level": "warn",
                      "detail": f"{r['pose']} held on {r['n']} photos"})
    flags.sort(key=lambda f: (f["k"], f["rule"]))
    errs = sum(f["level"] == "error" for f in flags)
    per = {}
    for f in flags:
        per.setdefault(f["rule"], [0, 0])[0 if f["level"] == "error" else 1] += 1
    return {"errors": errs, "warnings": len(flags) - errs, "per_rule": {k: {"errors": v[0], "warnings": v[1]}
                                                                         for k, v in per.items()},
            "flags": flags, "masks": masks}


def _contact(ch: Choreo, hist, i: int, e, mv, d: np.ndarray, X0: int, Y0: int, paint: np.ndarray, base: dict,
             masks: dict, ph) -> list[dict]:
    """The contact rules (v7) for one shown figure: feet, takeoff, inside, grip, fist."""
    sc, lib = ch.scene, ch.lib
    u = ch.u
    out = []
    ys, xs = np.nonzero(d > 0.4)
    if not len(ys):
        return out
    low = Y0 + float(np.percentile(ys, 99.5))
    lowx = X0 + float(np.median(xs[ys >= np.percentile(ys, 99.5) - 2]))
    kind = e.flags.get("kind", "")
    grounded = bool(e.flags.get("ground"))
    if not grounded and e.flags.get("kind") is None and e.layer == "wall":
        # a clip frame whose feet are on its clip's floor (a video drawing in a move that is not parkour)
        m = lib.m(e.pose)
        vd = m.get("video") or {}
        vid = lib.videos.get(vd.get("src")) if vd else None
        if vid is not None and mv is not None and mv.kind not in ("inside", "pound", "burst", "box", "pose"):
            r = next((r_ for r_ in vid["frames"] if r_["name"] == e.pose), None)
            grounded = r is not None and float(vid["ground"]) - float(r["low"]) < 6
    if grounded:
        surf = e.flags.get("surface")
        surf = float(surf) if surf is not None else sc.feet_at(lowx)
        tol = FEET_TOL * u * (2 if kind == "roll" else 1) + (0 if not e.jitter else 6 * u)  # (a roll's lowest
        # paint is a tumbling back or shoulder: twice the tolerance)
        if abs(low - surf) > tol:
            out.append({**base, "rule": "feet", "level": "error",
                        "detail": f"his feet {low - surf:+.0f} px off the surface (tolerance {tol:.0f})"})
        if getattr(ch, "paint_in", None) is not None:  # v8: he never stands on world not yet painted in
            from . import reveal as RV
            rv = RV.revealed(ch, ph.t, lowx, surf - 2 * u)
            if rv < 0.5:
                out.append({**base, "rule": "reveal", "level": "error",
                            "detail": f"he stands on world not painted in yet (x {lowx:.0f}, {rv:.0%} revealed)"})
    if e.flags.get("takeoff") and e.flags.get("edge") is not None:
        p_ = np.percentile(ys, 99.6)  # the planted foot's sole: its lowest 8 px of paint (as the drawing's bottom
        footx = X0 + float(np.mean(xs[(ys >= p_ - 8) & (ys <= p_)]))  # anchor the solver plants; not its toe)
        over = footx - float(e.flags["edge"])
        if over > 2 * u:
            out.append({**base, "rule": "takeoff", "level": "error",
                        "detail": f"his take-off foot {over:.0f} px past the edge"})
    if kind == "flight" or e.flags.get("flight"):
        bad = np.zeros_like(paint)
        h, w = paint.shape
        for o in sc.obstacles.values():
            if not o.solid or o.kind == "gap" or o.id == e.flags.get("on"):  # (not the prop he stands on)
                continue
            top = o.contacts["top"].y if "top" in o.contacts and o.contacts["top"].y is not None else o.y0
            top += TOUCH_TOL * u * e.flags.get("touch", 0)
            c0, c1 = int(max(o.x0 - X0, 0)), int(min(o.x1 - X0, w))
            r0, r1 = int(max(top + 4 * u - Y0, 0)), int(min(o.y1 - Y0, h))
            if c1 > c0 and r1 > r0:
                bad[r0:r1, c0:c1] |= paint[r0:r1, c0:c1]
        gaps = [o for o in sc.obstacles.values() if o.kind == "gap"]
        for cx_ in range(0, w, 8) if not e.flags.get("ground") else []:  # (a grounded skin: the feet rule's)
            x_ = X0 + cx_
            if any(o.x0 < x_ < o.x1 for o in gaps):
                continue
            f_ = sc.feet_at(x_)
            r0 = int(max(f_ + (4 + TOUCH_TOL * e.flags.get("touch", 0)) * u - Y0, 0))  # (a flight photo touching
            # down or still leaving: its skin's smear may reach into the ground a little)
            if r0 < h:
                bad[r0:, cx_:cx_ + 8] |= paint[r0:, cx_:cx_ + 8]
        if bad.sum() > 30:
            out.append({**base, "rule": "inside", "level": "error",
                        "detail": f"in flight {int(bad.sum())} px of him inside an obstacle or the ground"})
            masks.setdefault(ph.k, []).append((X0, Y0, bad))
    hold = e.flags.get("hold")
    if hold:
        o = sc.obstacles.get(str(hold))
        if o is not None and ({"bar", "beam"} & set(o.contacts)):
            top_y = Y0 + float(np.percentile(ys, 0.5))
            top_x = X0 + float(np.median(xs[ys <= np.percentile(ys, 0.5) + 2]))
            dist = o.contact("bar", "beam").dist(top_x, top_y)
            if dist > HOLD_TOL * u:
                out.append({**base, "rule": "grip", "level": "error",
                            "detail": f"his hands {dist:.0f} px from {o.id}'s bar"})
    if mv is not None and mv.kind in ("inside", "pound"):
        pr = ch.props.get(str(mv.args.get("box", "box")))
        k = pr.key_at(e.q) if pr is not None else None
        if k is not None:
            x0, y0, x1, y1 = k.box
            rows = Y0 + np.arange(paint.shape[0])
            inb = (rows > y0) & (rows < y1)
            lim_l = x0 - k.bow(rows) - FIST_TOL * u  # (his paint may reach into the line, the wall; not past it)
            lim_r = x1 + k.bow(rows) + FIST_TOL * u
            cols = X0 + np.arange(paint.shape[1])
            through = paint & inb[:, None] & ((cols[None, :] < lim_l[:, None]) | (cols[None, :] > lim_r[:, None]))
            if through.sum() > 20:
                out.append({**base, "rule": "fist", "level": "error",
                            "detail": f"{int(through.sum())} px of him through the box's side"})
                masks.setdefault(ph.k, []).append((X0, Y0, through))
    return out


def lines(res: dict) -> list[str]:
    out = [f"QA: {res['errors']} errors, {res['warnings']} warnings"
           + (" (" + ", ".join(f"{k} {v['errors']}/{v['warnings']}" for k, v in sorted(res["per_rule"].items())) + ")"
              if res["per_rule"] else "")]
    for f in res["flags"]:
        if f["level"] == "error":
            out.append(f"QA ERROR photo {f['k']} {f['t']:.3f}s {f['addr']} {f['rule']}: {f['detail']} "
                       f"({f['move']} {f['pose']})".rstrip())
    return out


def banner(res: dict, sheet: str = "") -> str:
    if res["errors"]:
        msg = f"QA FAILED: {res['errors']} ERRORS" + (f"   see {sheet}" if sheet else "")
    else:
        msg = f"QA PASSED: 0 errors, {res['warnings']} warnings"
    bar = "#" * (len(msg) + 8)
    return f"{bar}\n##  {msg}  ##\n{bar}"


def to_photo(c, size: tuple[int, int], out_size: tuple[int, int], x: np.ndarray, y: np.ndarray):
    """Street px to the pixels of a rendered photo (the camera, then the look's 1.06 crop)."""
    W, H = size
    s = W / c.cw
    ang = math.radians(c.rot)
    dx, dy = (x - c.cx) * s, (y - c.cy) * s
    rx = dx * math.cos(ang) + dy * math.sin(ang)
    ry = -dx * math.sin(ang) + dy * math.cos(ang)
    LW, LH = out_size
    return rx / LOOK_CROP * (LW / W) + LW / 2, ry / LOOK_CROP * (LH / H) + LH / 2


def sheets(ch: Choreo, res: dict, jobs, work, name: str, out_dir, sizes: dict, looks: dict, aspects) -> dict:
    """<name>-qa.jpg and <name>-moves.jpg from the rendered photos in work/<aspect>/."""
    from PIL import Image, ImageDraw

    from .maps import _font
    out = {}
    font, small = _font(22), _font(16)
    jk = {j.k: j for j in jobs}
    a0 = aspects[0]
    tag = a0.replace(":", "x")
    tiles = []
    for k in sorted({f["k"] for f in res["flags"]}):
        if k not in jk:
            continue
        fl = [f for f in res["flags"] if f["k"] == k]
        a = next((f["aspect"] for f in fl if f.get("aspect") in aspects), a0)
        f_ = work / a.replace(":", "x") / f"{k:05d}.jpg"
        if not f_.exists():
            continue
        im = np.asarray(Image.open(f_).convert("RGB")).astype(np.float32)
        LH, LW = im.shape[:2]
        tint = np.zeros((LH, LW), np.float32)
        for X0, Y0, m in res["masks"].get(k, []):
            ys, xs = np.nonzero(m)
            if len(xs):
                px, py = to_photo(jk[k].cams[a], sizes[a], (LW, LH), X0 + xs + 0.5, Y0 + ys + 0.5)
                ok = (px >= 0) & (px < LW) & (py >= 0) & (py < LH)
                cv2.circle  # (points are dense enough at this scale)
                tint[py[ok].astype(int), px[ok].astype(int)] = 1.0
        tint = cv2.dilate(tint, np.ones((5, 5), np.uint8))
        im = im * (1 - 0.65 * tint[..., None]) + np.float32([255, 30, 30]) * 0.65 * tint[..., None]
        pim = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
        pim.thumbnail((960, 960))
        d = ImageDraw.Draw(pim)
        head = f"photo {k}  {jk[k].t:.3f}s  {ch.grid.addr(jk[k].q)}"
        rows = [head] + [f"{f['level'].upper()} {f['rule']}: {f['detail']}" for f in fl]
        hgt = 8 + 26 * len(rows)
        d.rectangle([0, 0, max(d.textlength(r, font=small) for r in rows) + 16, hgt], fill=(0, 0, 0))
        for li, r in enumerate(rows):
            col = (255, 255, 255) if li == 0 else ((255, 90, 80) if r.startswith("ERROR") else (255, 210, 90))
            d.text((8, 4 + 26 * li), r, fill=col, font=small if li else font)
        tiles.append(pim)
    qa_jpg = out_dir / f"{name}-qa.jpg"
    if tiles:
        cols = 3
        tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
        sheet = Image.new("RGB", (cols * tw, math.ceil(len(tiles) / cols) * th), (14, 14, 16))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    else:
        sheet = Image.new("RGB", (960, 120), (14, 30, 14))
        ImageDraw.Draw(sheet).text((20, 40), "QA: nothing flagged", fill=(140, 255, 140), font=font)
    parts = _save_tall(sheet, qa_jpg, th if tiles else sheet.height, 88)
    out["qa_sheet"] = str(qa_jpg)
    if len(parts) > 1:
        out["qa_sheets"] = [str(p_) for p_ in parts]
    # every move, 4 photos before to 4 after
    order = [j.k for j in jobs]
    rows = []
    for mv in ch.moves:
        if mv.kind in ("run", "freeze", "walk") or mv.prop or mv.state.q0 is None:
            continue
        ks = sorted({j.k for j in jobs if mv.state.q0 - 1e-6 <= j.q <= mv.state.q1 + 1e-6})
        if not ks:
            continue
        i0, i1 = order.index(ks[0]), order.index(ks[-1])
        allk = order[max(0, i0 - 4):i1 + 5]
        for c0 in range(0, len(allk), MOVES_PER_LINE):  # a long move wraps (a JPEG is at most 65,535 px wide)
            rows.append((mv, allk[c0:c0 + MOVES_PER_LINE], c0 == 0))
    if rows:
        tw, th = (320, 180) if a0 == "16:9" else (160, 284)
        ncol = max(len(r[1]) for r in rows)
        lab = 220
        sheet = Image.new("RGB", (lab + ncol * (tw + 4), len(rows) * (th + 24)), (14, 14, 16))
        d = ImageDraw.Draw(sheet)
        bad = {f["k"] for f in res["flags"] if f["level"] == "error"}
        for ri, (mv, ks, first) in enumerate(rows):
            y = ri * (th + 24)
            d.text((8, y + 8), mv.where if first else "  (on)", fill=(255, 220, 90), font=small)
            for ci, k in enumerate(ks):
                f_ = work / tag / f"{k:05d}.jpg"
                if not f_.exists():
                    continue
                im = Image.open(f_)
                im.thumbnail((tw, th))
                x = lab + ci * (tw + 4)
                sheet.paste(im, (x, y))
                inside = mv.state.q0 - 1e-6 <= jk[k].q <= mv.state.q1 + 1e-6
                d.text((x + 3, y + th + 3), f"{ch.grid.addr(jk[k].q)} k{k}",
                       fill=(255, 90, 80) if k in bad else ((255, 255, 255) if inside else (150, 150, 150)), font=small)
        mv_jpg = out_dir / f"{name}-moves.jpg"
        parts = _save_tall(sheet, mv_jpg, th + 24, 86)
        out["moves_sheet"] = str(mv_jpg)
        if len(parts) > 1:
            out["moves_sheets"] = [str(p_) for p_ in parts]
    return out


def _save_tall(sheet, path, row_h: int, quality: int) -> list:
    """Save a sheet as a JPEG, split into parts at whole rows (row_h px) when it is taller than a JPEG can be (65,500
    px) or a part would pass PIL's image size guard (about 179 Mpx): <path>, then <stem>-2.jpg, -3.jpg ... Returns
    the files written."""
    lim = min(65000, 150_000_000 // max(sheet.width, 1))
    if sheet.height <= lim:
        sheet.save(path, quality=quality)
        return [path]
    step = max(1, lim // max(int(row_h), 1)) * int(row_h)
    out = []
    for i, y0 in enumerate(range(0, sheet.height, step)):
        p_ = path if i == 0 else path.with_name(f"{path.stem}-{i + 1}{path.suffix}")
        sheet.crop((0, y0, sheet.width, min(y0 + step, sheet.height))).save(p_, quality=quality)
        out.append(p_)
    return out


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "masks"}
