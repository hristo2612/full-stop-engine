"""`aimv paint check`: validate a choreography and summarise it (photos per section, travel, street still needed).

Errors (a render refuses them): unknown templates, obstacles, poses or roles, bad times, segments that overlap or
leave gaps. Warnings: contacts that miss the grid (a named hit more than 25 ms from its photo, or a pose between two
photos of a slower segment), anchors that miss their obstacle's contact line, a slide whose figure is taller than
the gap under the obstacle, paint landing in no-paint zones or off the street, poses painted during a freeze, and
9:16 frames that cut the figure.
"""

from __future__ import annotations

import numpy as np

from ..util import read_json
from .camera import CW0, framing_stats, plan
from .choreo import Choreo, Issue

LOOK_CROP = 1 / 1.06  # the MUTO look keeps the middle 1/1.06 of every photo (aimv.look.muto._photo)
from .render import History
from . import truelook as TL

# street px per second of song by section kind, for the part of the song not choreographed yet (prototype 6's
# estimate: runs and instrumentals about 900 px/s, verses at walking pace about 500, stops, bridge, outro about 200)
SPEED = {"chorus": 900, "instrumental": 900, "drop": 900, "postchorus": 900, "build": 900, "verse": 500,
         "prechorus": 500, "intro": 200, "bridge": 200, "break": 200, "outro": 200, "other": 400}
PLATE_PX = 2300  # new street per outpainted plate (3840 px wide, 40% kept)


def _overlap(a, b, gap: float = 0.0) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def prop_layout(ch: Choreo, photos, cams16, cams9) -> list[Issue]:
    """The stencils round a box: none overlaps another, the box (as it is at any time while both are there) or a
    parallax layer (the light pole, at that photo's parallax), and each is wholly inside both formats' frames (what
    the look keeps of them) on every photo of a box camera segment from the one it appears in."""
    out = []
    g = ch.grid
    for pid, pr in ch.props.items():
        n = len(pr.stencils)
        boxes = [pr.stencil_box(i) for i in range(n)]
        end = pr.shatter.q if pr.shatter else float("inf")
        names = [" / ".join(st.lines) for st in pr.stencils]
        for i in range(n):
            for j in range(i + 1, n):
                if _overlap(boxes[i], boxes[j], 6):
                    out.append(Issue("warn", f"{pid} stencils", f"{names[i]!r} overlaps {names[j]!r}"))
            keys = sorted(pr.keys, key=lambda k: k.q)
            for a, k in enumerate(keys):
                k_end = keys[a + 1].q if a + 1 < len(keys) else end
                if k_end > pr.stencils[i].q and k.q < end and _overlap(boxes[i], pr.outer(k), 6):
                    out.append(Issue("warn", f"{pid} stencils", f"{names[i]!r} overlaps the box painted at "
                                                             f"{g.addr(k.q)}"))
                    break
        cut: dict = {}
        for ph, c16, c9 in zip(photos, cams16, cams9):
            if ph.q >= end or ch.segments[ph.seg].camera.get("preset") != "box":
                continue
            for i in range(n):
                if pr.stencils[i].q > ph.q + 1e-6:
                    continue
                for tag, c in (("16:9", c16), ("9:16", c9)):
                    hw, hh = c.cw / 2 * LOOK_CROP * 0.99, c.ch / 2 * LOOK_CROP * 0.99
                    b = boxes[i]
                    if b[0] < c.cx - hw or b[2] > c.cx + hw or b[1] < c.cy - hh or b[3] > c.cy + hh:
                        cut.setdefault((i, tag), ph)
                for L in ch.scene.layers:
                    dx = -float(L.get("parallax", 0.1)) * (c16.cx - float(L.get("x", 0)))
                    for part in L["parts"]:
                        x0, y0, x1, y1 = part["box"]
                        if _overlap(boxes[i], (x0 + dx, y0, x1 + dx, y1), 8):
                            cut.setdefault((i, f"behind {L['id']}"), ph)
        for (i, tag), ph in cut.items():
            what = f"is behind {tag[7:]}" if tag.startswith("behind") else f"is cut by the {tag} frame"
            out.append(Issue("warn", f"photo {ph.k} {g.addr(ph.q)}", f"stencil {names[i]!r} {what}"))
    return out


CYCLIC = ("run", "walk", "hang", "inside", "pound")  # templates whose drawings repeat by design (cycles, holds)


def figure_track(ch: Choreo) -> list[dict]:
    """The figure photo by photo: for every photo that paints a new figure pose, its torso position, drawing, move
    and tag (the photos in between show the same pose: holds and freezes)."""
    lib = ch.lib
    by_q: dict[float, object] = {}
    for e in ch.events:
        if e.layer in ("wall", "air") and e.pose is not None and lib.m(e.pose)["kind"] == "pose":
            by_q[round(e.q, 4)] = e  # the last one at a photo is the one shown
    out = []
    for ph in ch.photos():
        e = by_q.get(round(ph.q, 4))
        if e is None:
            continue
        dx, dy = lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)
        out.append({"k": ph.k, "q": ph.q, "t": ph.t, "x": e.x + dx, "y": e.y + dy, "pose": e.pose,
                    "move": e.move, "kind": e.flags.get("kind") or (ch.moves[e.move].kind if 0 <= e.move < len(ch.moves)
                                                                    else ""),  # a parkour photo says what it is
                    "tag": e.tag, "sheet": lib.m(e.pose)["sheet"],
                    "back": bool(e.flags.get("back")) or  # (v8c: a route clip's `sway` frames, as shot round a pole)
                            (bool(ch.moves[e.move].args.get("back")) if 0 <= e.move < len(ch.moves) else False)})
    return out


def motion_stats(ch: Choreo, t0: float = -1e9, t1: float = 1e9) -> dict:
    """How the figure moves from photo to photo (painted photos only, consecutive on the grid): backtracks (the torso
    goes back by more than 12 px), lurches (the travel per photo changes by more than 35% of its mean between two
    photos while he travels at least 20 px a photo), the largest travel per photo, repeated drawings (a drawing used
    twice in one move that is not a cycle, or on two painted photos in a row), runs that mix sheets."""
    u = ch.u
    tr = [r for r in figure_track(ch) if t0 <= r["t"] < t1]
    step = {}
    for a, b in zip(tr, tr[1:]):
        if b["k"] == a["k"] + 1:
            step[b["k"]] = b["x"] - a["x"]
    backs, lurches, travel, in_run = [], [], [], []
    for a, b in zip(tr, tr[1:]):
        d = step.get(b["k"])
        if d is None:
            continue
        travel.append((abs(d), b))
        if d < -12 * u and b["kind"] not in ("pound", "inside") and b["tag"] not in ("burst.box", "burst.windup") \
                and not b.get("back"):
            # in the box he pulls back to hit it again (the pound, the strikes, the burst's wind-up); a clip with
            # back: true steps back as it should (a bounce off a wall)
            backs.append((d, b))
        dp = step.get(a["k"])
        if dp is not None and dp >= 20 * u and d >= 20 * u:
            ch_ = abs(d - dp)
            lurches.append((ch_, ch_ / (0.5 * (d + dp)), b))
            if a["kind"] == b["kind"] == "run" and a["move"] == b["move"]:
                in_run.append(ch_ / (0.5 * (d + dp)))
    reps = []
    seen: dict[tuple, int] = {}
    for r in tr:
        key = (r["move"], r["pose"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2 and r["kind"] not in CYCLIC and r["kind"] not in ("slide", "climb", "box"):
            reps.append(r)
    have = {r["k"] for r in reps}
    for a, b in zip(tr, tr[1:]):
        if b["k"] == a["k"] + 1 and a["pose"] == b["pose"] and b["kind"] not in ("inside", "slide") \
                and b["k"] not in have:  # a slide glides on one drawing
            reps.append(b)
    cross = []
    for mv in ch.moves:
        if mv.kind == "run":
            sheets = {r["sheet"] for r in tr if r["move"] == mv.i}
            if len(sheets) > 1:
                cross.append((mv, sorted(sheets)))
    big = [x for x in lurches if x[1] > 0.35]
    holds = _holds(ch, t0, t1)
    return {"photos": len(tr), "backtracks": backs, "lurches": big, "holds": holds,
            "handovers": _handovers(ch, tr),
            "max_lurch_px": max((x[0] for x in lurches), default=0.0),
            "max_lurch_ratio": max((x[1] for x in lurches), default=0.0),
            "max_travel_px": max((x[0] for x in travel), default=0.0),
            "max_run_lurch_ratio": max(in_run, default=0.0),
            "repeats": reps, "cross_sheet_runs": cross}


HANDOVER_DV = 1.5  # m/s: his speed may change by at most this where a clip hands over to a run (or a run to a clip)


def _is_run(r: dict) -> bool:
    return r["kind"] == "run" or r["tag"] in ("run", "pk.run")


def _is_clip(r: dict) -> bool:
    return r["kind"] == "clip" or "clip" in str(r["tag"])


def _handovers(ch: Choreo, tr: list[dict]) -> list[dict]:
    """Every hand-over between a clip and a run on consecutive photos (v8b, the director's notes at 107.0 and 158.7: from
    a mantle or a squat "suddenly running"): his torso speed over the two photo intervals before it and the two after
    it (m/s: a clip's torso surges with each step as shot),
    and how much it changes. [{k, t, q, pose, move, tag, dir (clip-run or run-clip), v0, v1, dv}]; the QA warns when
    the change is over HANDOVER_DV."""
    P = 250.0 * ch.u  # px a metre
    out = []
    for i in range(3, len(tr) - 2):
        a, b = tr[i - 1], tr[i]
        if b["k"] != a["k"] + 1:
            continue
        d = "clip-run" if _is_clip(a) and _is_run(b) else "run-clip" if _is_run(a) and _is_clip(b) else None
        if d is None:
            continue
        p0, p2 = tr[i - 3], tr[i + 2] if i + 2 < len(tr) else None  # (two intervals a side: a clip's torso surges
        if p2 is None or p0["k"] != a["k"] - 2 or p2["k"] != b["k"] + 2:  # with each step as shot)
            continue
        v0 = (a["x"] - p0["x"]) / max(a["t"] - p0["t"], 1e-6) / P
        v1 = (p2["x"] - b["x"]) / max(p2["t"] - b["t"], 1e-6) / P
        out.append({"k": b["k"], "t": b["t"], "q": b["q"], "pose": b["pose"], "move": b["move"], "tag": b["tag"],
                    "dir": d, "v0": round(v0, 2), "v1": round(v1, 2), "dv": round(v1 - v0, 2)})
    return out


HOLD_MAX = 2  # photos one drawing may stay up (review note: "you are using single frames again")


def _holds(ch: Choreo, t0: float = -1e9, t1: float = 1e9) -> list[dict]:
    """Drawings held on more than HOLD_MAX photos in a row (the same drawing on consecutive photos, painted again or
    simply not replaced): [{k, q, t, n, pose, move, tag}] at the hold's first photo. A true freeze (a song stop or a
    freeze move with nothing painted in it) holds by design, and so does a move marked `still: true` (a deliberate
    stillness: the silence on "stop")."""
    lib = ch.lib
    shown: dict[float, object] = {}
    for e in ch.events:
        if e.layer in ("wall", "air") and e.pose is not None and lib.m(e.pose)["kind"] == "pose":
            shown[round(e.q, 4)] = e
    painted = sorted(shown)
    frz = [(a, b) for a, b in ch.freezes if not any(a + 1e-6 < q < b - 1e-6 for q in painted)]
    out, cur, run = [], None, []

    def close():
        if cur is not None and len(run) > HOLD_MAX and not cur.flags.get("still"):
            ph = run[0]
            if t0 <= ph.t < t1:
                out.append({"k": ph.k, "q": ph.q, "t": ph.t, "n": len(run), "pose": cur.pose, "move": cur.move,
                            "tag": cur.tag})

    for ph in ch.photos():
        e = shown.get(round(ph.q, 4))
        if any(a - 1e-6 <= ph.q <= b + 1e-6 for a, b in frz):
            close()
            cur, run = None, []
            continue
        if e is None or (cur is not None and e.pose == cur.pose):
            if cur is not None:
                run.append(ph)
            continue
        close()
        cur, run = e, [ph]
    close()
    return out


def motion_issues(ch: Choreo, st: dict) -> list[Issue]:
    g = ch.grid
    out = []
    w = lambda r: f"{ch.moves[r['move']].where} {g.addr(r['q'])}" if 0 <= r["move"] < len(ch.moves) else g.addr(r["q"])
    for d, r in st["backtracks"]:
        out.append(Issue("warn", w(r), f"backtracks {-d:.0f} px ({r['pose']})"))
    for c, ratio, r in st["lurches"]:
        out.append(Issue("warn", w(r), f"lurches: the travel per photo changes by {c:.0f} px ({ratio:.0%})"))
    for r in st["repeats"]:
        out.append(Issue("warn", w(r), f"repeats {r['pose']}"))
    for r in st.get("holds", []):
        out.append(Issue("warn", w(r), f"holds {r['pose']} on {r['n']} photos (at most {HOLD_MAX})"))
    for mv, sheets in st["cross_sheet_runs"]:
        out.append(Issue("warn", mv.where, f"the run mixes sheets ({', '.join(sheets)})"))
    return out


def check(p, ch: Choreo) -> dict:
    """Run every check; returns {"issues": [...], "summary": {...}, "lines": [...]} (lines: the report)."""
    issues: list[Issue] = list(ch.issues)
    if ch.errors:
        return {"issues": issues, "summary": {}, "lines": [str(i) for i in issues]}
    g, sc, lib = ch.grid, ch.scene, ch.lib
    photos = ch.photos()
    pq = np.array([ph.q for ph in photos])
    hist = History(ch)
    # contacts on the grid
    for q, hit_t, where in ch.hit_contacts:
        d = abs(g.t(q) - hit_t)
        if d > 0.025:
            issues.append(Issue("warn", where, f"the photo is {d * 1000:.0f} ms from the hit ({hit_t:.3f} s)"))
    for e in ch.events:
        if len(pq) and np.min(np.abs(pq - e.q)) > 1e-4:
            issues.append(Issue("warn", f"move {e.move + 1} ({e.tag})", f"{g.addr(e.q)} is between photos "
                                f"(segment rate {ch.step_at(e.q):g} 16ths): it first shows on the next photo"))
    # freezes
    for a, b in ch.freezes:
        for e in ch.events:
            if 0 <= e.move < len(ch.moves) and ch.moves[e.move].args.get("pause") is False:
                continue  # set to the song's silence on purpose (pause: false): the stop in p6, set to the words
            if a < e.q < b - 1e-6:
                issues.append(Issue("warn", f"move {e.move + 1} ({e.tag})", f"painted at {g.addr(e.q)}, inside the "
                                    f"freeze {g.addr(a)} to {g.addr(b)}"))
    # runs too slow for the foot lock: the drawings move the torso a fixed distance over the planted foot
    cyc = lib.cycles.get("run") or []
    ph = lib.phases.get("run") or []
    if cyc and "contact" in ph and "toeoff" in ph:
        ic, it = ph.index("contact"), ph.index("toeoff")
        st = float(np.median([(lib.foot_offset(c[ic]) - lib.foot_offset(c[it])) / (it - ic) for c in cyc
                              if len(c) == len(ph)]))
        for mv in ch.moves:
            ev = [e for e in ch.events if e.move == mv.i and e.tag == "run"]
            if ev and "video" in lib.m(ev[0].pose):  # a run from a clip: its own feet
                continue
            if mv.kind == "run" and len(ev) > 2:
                v = (ev[-1].x - ev[0].x) / (ev[-1].q - ev[0].q)
                if v < 0.5 * st:
                    issues.append(Issue("warn", mv.where, f"runs {v:.0f} px per 16th, too slow for the run drawings "
                                        f"(about {st:.0f} per photo over the planted foot): the feet slide; give it "
                                        f"more street or fewer photos"))
    # contacts, clearance, no-paint, off the street
    n_np = 0
    for i, e in enumerate(ch.events):
        if e.layer not in ("wall", "air"):
            continue
        where = f"move {e.move + 1} ({e.tag}) {g.addr(e.q)}"
        if e.contact:
            oid, cname, anchor = e.contact[:3]
            rel = e.contact[3] if len(e.contact) > 3 else "on"
            o = sc.obstacles.get(oid)
            c = o.contacts.get(cname) if o else None
            if c is None:
                issues.append(Issue("warn", where, f"contact {oid}.{cname} not found"))
            elif rel == "past":
                if (c.x if c.x is not None else c.x1) > e.x:
                    issues.append(Issue("warn", where, f"lands before {oid}.{cname} (x {e.x:.0f})"))
            elif cname == "under":
                m = lib.m(e.pose)
                s = m["scale"] * e.scale
                top = e.y - (lib.anchor(e.pose, e.anchor)[1] - m["top"][1]) * s
                if top < c.y - 2:
                    issues.append(Issue("warn", where, f"{e.pose} is {c.y - top:.0f} px too tall to slide under "
                                                       f"{oid} (its underside is at y {c.y:.0f})"))
            else:
                dist = c.dist(e.x, e.y)
                if dist > 25:
                    issues.append(Issue("warn", where, f"{anchor} anchor is {dist:.0f} px off {oid}.{cname}"))
        if e.layer != "wall":
            continue
        d, x0, y0 = hist.place(i)
        h, w = d.shape
        if not sc.inside(x0, y0, x0 + w, y0 + h):
            issues.append(Issue("warn", where, f"{e.pose} is partly off the street (x {x0:.0f} to {x0 + w:.0f})"))
            continue
        if TL.is_true(ch, e):  # a true figure is solid, not paint: no paint rules (as QA's no_paint)
            continue
        _, pnt = sc.masks(int(x0), int(x0) + w, int(y0), int(y0) + h)
        reg = pnt
        dd = d[:reg.shape[0], :reg.shape[1]] > 0.3
        if dd.sum() > 0:
            frac = float((dd & (reg < 0.5)).sum() / dd.sum())
            if frac > 0.25:
                n_np += 1
                issues.append(Issue("warn", where, f"{frac:.0%} of {e.pose} lands where paint cannot go "
                                                   f"(no-paint zone, gap or pavement)"))
    # framing
    boxes = hist.figure_boxes()
    keys = sorted(boxes)
    fb = [boxes[keys[max(int(np.searchsorted(keys, ph.q + 1e-6, side="right")) - 1, 0)]] for ph in photos] \
        if keys else []
    cams16 = plan(ch, photos, boxes, "16:9")
    cams9 = plan(ch, photos, boxes, "9:16")
    st16 = framing_stats(cams16, fb, False)
    st9 = framing_stats(cams9, fb, True)
    for ph, c, b in zip(photos, cams9, fb):
        fx0, fx1 = c.cx - c.cw / 2, c.cx + c.cw / 2
        fy0, fy1 = c.cy - c.ch / 2, c.cy + c.ch / 2
        if b[0] < fx0 - 1 or b[2] > fx1 + 1 or b[1] < fy0 - 1 or b[3] > fy1 + 1:
            issues.append(Issue("warn", f"photo {ph.k} {g.addr(ph.q)}", "the 9:16 frame cuts the figure"))
    issues += prop_layout(ch, photos, cams16, cams9)
    mst = motion_stats(ch)
    issues += motion_issues(ch, mst)
    from . import qa as QA
    qres = QA.run(ch, hist, photos, {"16:9": cams16, "9:16": cams9})
    # summary
    secs = read_json(p.bd / "sections.json")["sections"]
    dur = float(read_json(p.bd / "meta.json").get("duration", secs[-1]["t1"]))
    per_sec = []
    for s in secs:
        n = sum(1 for ph in photos if s["t0"] <= ph.t < s["t1"])
        if n:
            per_sec.append((s["label"], n))
    figs = [e for e in ch.events if e.layer in ("wall", "air")]
    xs = [e.x for e in sorted(figs, key=lambda e: e.q)]
    travel = float(np.sum(np.clip(np.diff(xs), 0, None))) if len(xs) > 1 else 0.0
    cam_end = max((c.cx + c.cw / 2 for c in cams16), default=0.0)
    t_a, t_b = (photos[0].t, photos[-1].t) if photos else (0.0, 0.0)
    rest = 0.0
    rest_rows = []
    for s in secs:
        a, b = s["t0"], s["t1"]
        free = max(0.0, min(b, t_a) - a) + max(0.0, b - max(a, t_b))
        if free > 0.05:
            px = free * SPEED.get(s["kind"], 400)
            rest += px
            rest_rows.append((s["label"], free, px))
    need_now = max(0.0, cam_end - sc.bounds[2])  # the street's right end, or the world's
    cam_start = min((c.cx - c.cw / 2 for c in cams16), default=0.0)
    need_left = max(0.0, sc.X0 - cam_start)
    total_need = max(xs, default=0.0) + rest + CW0
    summary = {"photos": len(photos), "events": len(ch.events), "moves": len(ch.moves),
               "covered_s": [round(t_a, 3), round(t_b, 3)], "song_s": dur, "photos_per_section": per_sec,
               "travel_px": round(travel), "x_range": [round(min(xs, default=0)), round(max(xs, default=0))],
               "street_px": sc.W - sc.X0, "street_x": [sc.X0, sc.W], "camera_reaches_px": round(cam_end),
               "camera_from_px": round(cam_start), "street_short_px": round(need_now),
               "street_short_left_px": round(need_left),
               "rest_px_estimate": round(rest), "song_street_px_estimate": round(total_need),
               "plates_more_estimate": int(np.ceil(max(0.0, total_need - sc.bounds[2]) / PLATE_PX)),
               "framing_16x9": st16, "framing_9x16": st9}
    errs = [i for i in issues if i.level == "error"]
    warns = [i for i in issues if i.level == "warn"]
    lines = [f"{ch.name}: {len(ch.moves)} moves, {len(ch.events)} painted things, {len(photos)} photos "
             f"({t_a:.2f} to {t_b:.2f} s of {dur:.0f} s), {len(errs)} errors, {len(warns)} warnings"]
    lines += [str(i) for i in errs + warns]
    lines.append("photos per section: " + ", ".join(f"{a} {n}" for a, n in per_sec))
    lines.append(f"travel: {travel:,.0f} px (figure x {summary['x_range'][0]:,} to {summary['x_range'][1]:,}); "
                 f"street x {sc.X0:,} to {sc.W:,} ({sc.W - sc.X0:,} px"
                 + (f"; the world to x {sc.bounds[2]:,}, y {sc.Y0:,} to {sc.Y1:,}" if sc.world is not None else "")
                 + f"); the camera sees x {cam_start:,.0f} to "
                 f"{cam_end:,.0f}"
                 + (f": {need_now:,.0f} px of street missing on the right" if need_now > 0 else "")
                 + (f": {need_left:,.0f} px of street missing on the left" if need_left > 0 else ""))
    lines.append(f"motion: {len(mst['backtracks'])} backtracks, {len(mst['lurches'])} lurches (max change "
                 f"{mst['max_lurch_px']:.0f} px, {mst['max_lurch_ratio']:.0%}; inside runs at most "
                 f"{mst['max_run_lurch_ratio']:.0%}), max travel {mst['max_travel_px']:.0f} px "
                 f"a photo, {len(mst['repeats'])} repeated drawings, {len(mst['cross_sheet_runs'])} runs mixing sheets")
    summary["motion"] = {k: (len(v) if isinstance(v, list) else round(v, 3)) for k, v in mst.items()}
    summary["qa"] = {"errors": qres["errors"], "warnings": qres["warnings"], "per_rule": qres["per_rule"]}
    lines += QA.lines(qres)
    lines.append(f"framing: 16:9 torso offset max {st16['max_offset']:.1%} of the width, "
                 f"{st16['photos_leaving_band']} photos leave the centre 9:16 band; 9:16 cuts the figure in "
                 f"{st9['photos_cut']} photos (max {st9['max_outside']:.0%} of it outside)")
    if rest_rows:
        lines.append(f"not choreographed yet: {sum(r[1] for r in rest_rows):.0f} s, about {rest:,.0f} px of street "
                     f"at section speeds (" + ", ".join(f"{a} {s:.0f} s" for a, s, _ in rest_rows) + ")")
        lines.append(f"whole song: the street must reach about x {total_need:,.0f}; it reaches x {sc.bounds[2]:,}: "
                     f"about "
                     f"{summary['plates_more_estimate']} more plates at {PLATE_PX:,} px each")
    return {"issues": issues, "summary": summary, "lines": lines, "qa": qres}
