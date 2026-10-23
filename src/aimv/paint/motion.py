"""Motion M0: the move templates rewritten for smooth, readable motion (the engine default; `motion: legacy` keeps
prototype 6's placements in moves.py).

What changed against the legacy templates (the director's notes on segment 1 v1: two drawings repeating in the runs,
rough jumps and vaults, abrupt transitions, the figure stepping back):

- Runs play one cycle strictly in drawing order (R1-1 to R1-8: two steps, one step per beat, the contacts on the
  beats), never mixing sheets. Their travel is even: a constant speed, eased in and out of the moves on either side
  (each photo's travel changes by at most about 15%), never backwards. The foot lock is gone (the drawings' foot
  positions are too uneven for it).
- Every move plays its own sequence (poses.spec.json `sequences`, defaults in `motion`): its drawings in order,
  the key drawings on the move's hits and contacts, the drawings between two keys sampled one per photo (fewer
  photos: some are skipped; a drawing repeats only when a move has more photos than drawings).
- Flights follow one smooth arc between their key positions, with eased spacing: faster at take-off and landing,
  slower at the top.
- Transitions: a run ends on the drawing before the move's first one in the step (its stride phase) and picks up
  after the move's last one; where the beat and the phase disagree, the run skips a drawing or two spread over its
  length (still in order). The take-off point of a move is where the run before it ends (pre_x).

Positions are the figure's torso (poses.PoseLib.torso_offset): torso x moves forward photo by photo.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .choreo import ChoreoError, Ctx, lerp
from .grid import rate_of
from .moves import LEAP_ARC, _ease, _start_after_prev, _vary, crouch as _crouch_legacy, leap as _leap_legacy

M0: dict = {}
RUN_SPEED = 88.0  # px per 16th for a 450 px figure


def m0(name: str):
    def deco(fn):
        M0[name] = fn
        fn.template = name
        return fn
    return deco


# ---------------------------------------------------------------------------------------------- helpers

def sequence(c: Ctx, template: str) -> tuple[str, dict]:
    """The move's sequence: its `sequence:` argument, else the library's default for the template."""
    name = c.arg("sequence") or c.ch.lib.motion.get(template)
    if not name:
        raise ChoreoError(f"no sequence for {template} (poses.spec.json motion)")
    try:
        return str(name), c.ch.lib.sequence(str(name))
    except KeyError as e:
        raise ChoreoError(str(e.args[0])) from e


def between(seq: dict, a: str | None, b: str | None) -> list[str]:
    """The drawings strictly between keys a and b (None: the sequence's start or end)."""
    d, k = seq["drawings"], seq.get("keys") or {}
    i = k[a] + 1 if a is not None else 0
    j = k[b] if b is not None else len(d)
    return d[i:j]


def key(seq: dict, name: str) -> str:
    return seq["drawings"][seq["keys"][name]]


def pick(pool: list[str], n: int) -> list[str]:
    """n drawings from pool in order: evenly spaced when there are more than n; each one once when n is the pool's
    size; when n is larger a drawing repeats (the check says so)."""
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return [pool[min(int(k * len(pool) / n), len(pool) - 1)] for k in range(n)]
    if n == 1:
        return [pool[len(pool) // 2]]
    return [pool[int(i)] for i in np.round(np.linspace(0, len(pool) - 1, n))]


def video_track(c: Ctx, seq: dict) -> tuple[dict, list[dict]]:
    """The clip of a video sequence and its frames' track (frame, time, pose, torso x, y, lowest point) in clip
    px, restricted to the sequence's frames (not its `skip` frames nor drawings with canon errors)."""
    vid = c.ch.lib.videos[seq["video"]]
    want = set(seq["frames"]) - set(seq.get("skip") or [])
    bad = getattr(c.ch.lib, "canon_bad", set())  # a drawing with a canon error: the neighbouring frame shows
    return vid, [r for r in vid["frames"] if r["f"] in want and r["name"] not in bad]


def video_at(c: Ctx, seq: dict, q_key: float, key: str, q0: float, q1: float,
             speed: float = 1.0) -> list[tuple[float, dict]]:
    """Photos from q0 to q1 (move time) with the clip frame each one shows when the key frame lands on q_key: the
    clip plays at `speed` times its own speed (frame = key frame + seconds since the key x fps x speed, in move
    time: the song's stops take no clip time), one frame per photo; photos the clip does not cover are left out."""
    vid, tr = video_track(c, seq)
    fk = seq["frames"][seq["keys"][key]]
    tk = c.ch.grid.t(q_key)  # move time: a song stop in between takes no clip time (the move pauses through it)
    by_f = {r["f"]: r for r in tr}
    fs = sorted(by_f)
    out = []
    s = c.ch.step_at(q_key)
    q = q_key - math.floor((q_key - q0) / s + 1e-6) * s
    while q <= q1 + 1e-6:
        fv = fk + (c.ch.grid.t(q) - tk) * vid["fps"] * speed
        if fs[0] - 0.5 <= fv <= fs[-1] + 0.5:
            f = min(fs, key=lambda x: abs(x - fv))
            out.append((q, by_f[f]))
        q += s
    return out


def ease_u(s: float, ease: float = 0.3) -> float:
    """Arc parameter for an even time step s in [0, 1]: faster at both ends, slower in the middle (the top)."""
    return s + ease * math.sin(2 * math.pi * s) / (2 * math.pi)


def arc(n: int, a: tuple[float, float], b: tuple[float, float], apex: float, ease: float = 0.3) -> list[tuple]:
    """n points strictly between a and b (x, y) on a parabolic arc that rises `apex` px above the straight line at
    its middle, with eased spacing. Returns (x, y, u)."""
    out = []
    for k in range(n):
        u = ease_u((k + 1) / (n + 1), ease)
        out.append((lerp(a[0], b[0], u), lerp(a[1], b[1], u) - 4 * apex * u * (1 - u), u))
    return out


def tors(c: Ctx, pose: str, x: float, y: float, anchor: str, scale: float = 1.0, mirror: bool = False):
    """Torso (x, y) of a pose placed with its anchor at (x, y)."""
    dx, dy = c.ch.lib.torso_offset(pose, anchor, scale * float(c.args.get("scale", 1.0)), mirror)
    return x + dx, y + dy


def ground(c: Ctx, pose: str, tx: float) -> tuple[float, float]:
    """The foot anchor x for a pose standing on the ground with its torso at tx (foot anchor x is the torso x)."""
    return tx, c.feet


def ramp_from(G: int, va: float | None, v: float, limit: float = 0.15) -> np.ndarray:
    """G gaps starting near va and settling at v (no fixed end)."""
    if va is None or G <= 0:
        return np.full(max(G, 0), v)
    va = float(np.clip(va, 0.8 * v, 1.25 * v))  # as gaps(): the middle of the run keeps its own speed
    m = min(max(int(math.ceil(math.pi / 2 * abs(va - v) / (limit * v))), 1), G)
    k = np.arange(G, dtype=np.float64)
    ra = np.where(k < m, 0.5 * (1 + np.cos(np.pi * (k + 0.5) / m)), 0.0)
    return v + (va - v) * ra


def recover_dx(c: Ctx, last_step: float) -> float:
    """The step from a landing to the recovery drawing: half the flight's last step (80 to 150 px), so the figure
    slows down over two photos instead of stopping dead."""
    return float(np.clip(0.5 * last_step, 80 * c.u, 150 * c.u))


def gaps(G: int, total: float, va: float | None, vb: float | None, limit: float = 0.15) -> np.ndarray:
    """G gaps that sum to total: a constant speed in the middle, eased from va at the start and to vb at the end
    (cosine ramps long enough that the travel per photo changes by about `limit` of the middle speed at most)."""
    if G <= 0:
        return np.zeros(0)
    vm = total / G
    # ease toward the neighbours' steps, but only within -20% to +25% of the run's own average, so the middle of
    # the run stays even (a faster neighbour step then reads as a push into the move)
    va = vm if va is None else float(np.clip(va, 0.8 * vm, 1.25 * vm))
    vb = vm if vb is None else float(np.clip(vb, 0.8 * vm, 1.25 * vm))
    k = np.arange(G, dtype=np.float64)
    g = np.full(G, vm)
    for _ in range(6):
        if vm <= 1e-6:
            break
        ma = min(max(int(math.ceil(math.pi / 2 * abs(va - vm) / (limit * vm))), 0), G // 2)
        mb = min(max(int(math.ceil(math.pi / 2 * abs(vb - vm) / (limit * vm))), 0), G // 2)
        ra = np.where(k < ma, 0.5 * (1 + np.cos(np.pi * (k + 0.5) / max(ma, 1))), 0.0) if ma else np.zeros(G)
        rb = np.where(k >= G - mb, 0.5 * (1 - np.cos(np.pi * (k - (G - mb) + 0.5) / max(mb, 1))), 0.0) if mb \
            else np.zeros(G)
        rest = (1 - ra - rb).sum()
        if rest < 0.5:  # too short to hold a middle speed: a straight ramp scaled to the total
            g = np.linspace(va, vb, G)
            g *= total / max(g.sum(), 1e-9)
            return g
        vm_new = (total - va * ra.sum() - vb * rb.sum()) / rest
        if vm_new < 0.35 * vm:  # the ends are much faster than the middle can be: meet them only halfway
            va, vb = 0.5 * (va + vm), 0.5 * (vb + vm)
            continue
        g = vm_new + (va - vm_new) * ra + (vb - vm_new) * rb
        if abs(vm_new - vm) < 0.01:
            break
        vm = vm_new
    return g


# ---------------------------------------------------------------------------------------------- run

@m0("run")
def run(c: Ctx) -> None:
    """Run (M0): one cycle in drawing order (sequence: default the library's run cycle; cycle: the step to start
    with), one step per beat with the contacts on the beats (offset/contacts as before), ending on the drawing before
    the next move's first one and starting after the previous move's last one (a drawing or two skipped, spread
    over the run, when the beat and the phase disagree). Travel is even, eased from the previous move's last step to
    the next move's first (about 15% change per photo at most). from/to, x: [x0, x1], from_x, speed, ease, every as
    before; foot_lock is ignored."""
    q0 = c.time("from") if c.has("from") else _start_after_prev(c)
    c.st.q0 = q0
    c.st.run = True
    step = c.ch.step_at(q0)
    every = rate_of(c.arg("every", step))
    xs_arg = c.arg("x")
    from_x = c.coord(c.arg("from_x"), "x") if c.has("from_x") else None
    if isinstance(xs_arg, list) and len(xs_arg) == 2:
        from_x = c.coord(xs_arg[0], "x")
    if c.has("to"):
        q1 = c.time("to", ref=q0)
    elif c.next is not None and c.next.q0 is not None:
        q1 = c.next.q0 - c.ch.step_at(c.next.q0)
    else:
        if c.dry:  # the end comes from the next move; the start is known (the move before aims at it)
            if from_x is not None:
                c.st.tx0 = c.st.tx1 = c.st.x0 = c.st.x1 = from_x
            return
        raise ChoreoError("needs to: (nothing after it to run to)")
    c.arg("foot_lock")
    c.arg("max_lurch")
    qs = list(np.arange(q0, q1 + 1e-6, every))
    n = len(qs)
    if n < 1:
        raise ChoreoError(f"from {c.ch.grid.addr(q0)} to {c.ch.grid.addr(q1)} is empty")
    _, seq = sequence(c, "run")
    if "video" in seq:
        if not c.has("every"):  # the photos as the segments step them (a run on into a slower segment: the silence)
            qs, q = [], q0
            while q <= q1 + 1e-6:
                qs.append(float(q))
                q += c.ch.step_at(q)
        return _run_video(c, seq, qs, every, from_x, xs_arg)
    D = seq["drawings"]
    L = len(D)
    per = L // int(seq.get("steps", 1))  # drawings per step (one step per beat)
    dpp = max(1, int(round(per * every / 4)))  # drawings per photo
    contacts = str(c.arg("contacts", "beats"))
    if contacts not in ("beats", "and"):
        raise ChoreoError(f"contacts: beats or and, not {contacts}")
    off = float(c.arg("offset", 2 if contacts == "and" else 0))
    prev, nxt = c.prev, c.next
    beat_ph = [int(round(((q + off) % 4) / 4 * per)) % per for q in qs]  # the phase that puts contacts on beats
    # where the cycle starts: after the previous move's last drawing (or the previous run's), else on the beat
    if prev is not None and prev.run:
        i0 = int(prev.cycle)
    else:
        ph = (prev.ph1 + 1) % per if prev is not None and prev.ph1 is not None and not c.has("offset") \
            else beat_ph[0]
        st = c.arg("cycle")
        i0 = ph + per * (int(st) if st is not None else 0)
    end_ph = (nxt.ph0 - 1) % per if nxt is not None and nxt.ph0 is not None and not nxt.run else None
    # skips (a drawing passed over, still in order): first onto the beat grid, at the end onto the next move's
    # phase; a run too short for both only matches the next move
    adv = [dpp] * (n - 1)
    d1 = (beat_ph[0] - i0) % per
    d2 = (end_ph - beat_ph[-1]) % per if end_ph is not None else 0
    if n - 1 >= 2 * per and d1 + d2 <= n - 1:
        for j in range(d1):
            adv[j] += 1
        for j in range(d2):
            adv[n - 2 - j] += 1
    elif end_ph is not None and n > 1:
        d = (end_ph - (i0 + dpp * (n - 1))) % per
        for j in range(d):  # spread over the run
            adv[min(n - 2, max(0, int(round((j + 1) * (n - 1) / (d + 1))) - 1))] += 1
    idx = [i0]
    for a_ in adv:
        idx.append(idx[-1] + a_)
    names = [D[i % L] for i in idx]
    # travel: torso x from the previous move's last pose to the next move's first (or its take-off point)
    speed = float(c.arg("speed", RUN_SPEED * c.u))
    va = prev.v_out if prev is not None and prev.v_out is not None and prev.v_out > 0 else None
    vb = None
    if isinstance(xs_arg, list) and len(xs_arg) == 2:
        start, end, G, first_gap = from_x, c.coord(xs_arg[1], "x"), n - 1, False
    else:
        if from_x is not None:
            start, first_gap = from_x, False
        elif prev is not None and prev.tx1 is not None:
            start, first_gap = prev.tx1, True
        elif nxt is not None and (nxt.pre_x is not None or nxt.tx0 is not None):
            start, first_gap = None, False  # run up to the next move at its speed
        else:
            raise ChoreoError("needs x: [x0, x1] or from_x: (nothing before it to start from)")
        G = n - 1 + (1 if first_gap else 0)
        if start is None:
            G_end = G + (0 if nxt.pre_x is not None else 1)
            start = (nxt.pre_x if nxt.pre_x is not None else nxt.tx0) - speed * every * G_end
        if nxt is not None and nxt.pre_x is not None and nxt.pre_x - start > 0.4 * speed * every * max(G, 1):
            end = nxt.pre_x  # the move's take-off point (when it leaves the run room to run)
            vb = nxt.v_in
        elif nxt is not None and nxt.tx0 is not None:
            end, G = nxt.tx0, G + 1
            vb = nxt.v_in
        else:
            end = None
    if end is None:  # nothing fixes the end: settle from the last move's step to the run's speed
        g = ramp_from(G, va if first_gap else None, speed * every)
        end = start + float(g.sum())
    total = end - start
    if G > 0 and total <= 0:
        raise ChoreoError(f"would run backwards ({start:.0f} to {end:.0f}): give it more time or move a neighbour")
    if G > 0:
        g = gaps(G, total, va if first_gap else None, vb)
        ease = str(c.arg("ease", "none"))
        if ease != "none":
            w = _ease(G, ease)
            g = g * w * (total / max((g * w).sum(), 1e-9))
        pos = start + np.concatenate([[0.0], np.cumsum(g)])
        xs = pos[1:n + 1] if first_gap else pos[:n]
    else:
        xs = np.array([start] * n)
    for q, p, x in zip(qs, names, xs):
        c.wall(q, p, float(x), tag="run")
    c.st.cycle = (idx[-1] + dpp) % L
    c.st.exit_gap = speed * every


def _ramp_time(c: Ctx, qs: list) -> list[float]:
    """The run's own clock at each photo: its move time, or with `ramp: [[time, factor], ...]` the time integrated
    at that speed factor (piecewise linear between the points, held before the first and after the last): 0.2 is
    slow motion at a fifth of the speed, the legs and the travel together (the run through a song stop)."""
    rp = c.arg("ramp")
    if not rp:
        return [float(q) for q in qs]
    pts = sorted((c.at(t, key="ramp"), float(f)) for t, f in rp)
    if any(f <= 0 for _, f in pts):
        raise ChoreoError("ramp: every factor above 0 (1: the run's own speed)")
    tq, fv = [p[0] for p in pts], [p[1] for p in pts]
    out, e = [], float(qs[0])
    for i, q in enumerate(qs):
        if i:
            a, b = float(qs[i - 1]), float(q)
            m = np.linspace(a, b, 9)  # the factor integrated over the step (trapezoids)
            f = np.interp(m, tq, fv)
            e += float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(m)))
        out.append(e)
    return out


def _run_video(c: Ctx, seq: dict, qs: list, every: float, from_x, xs_arg) -> None:
    """A run from a clip's cycle (run): stride-true. The legs cycle on our beat (the cycle's `steps` steps over as
    many beats, the contacts on the beats), and the travel comes from the stride (poses.stride: the step measured
    where the stance foot stays planted), scaled to the figure: a step on the wall is the clip's step, travelled
    evenly, and each photo shows the cycle frame whose stride-true place is nearest (the feet slip only by that
    difference, choreo.notes has the largest); per_beat: steps a beat (1; 2: a step an 8th). It starts after the
    previous move (or at
    from_x), else it runs up to the next move; between two placed moves its stride is fitted to meet the next one
    (at most `fit`, 0.1: 10% longer or shorter, the feet slipping by that much); x: [x0, x1] stretches it to fit
    (the feet then slide), and so does `speed` (px per photo)."""
    vid, tr = video_track(c, seq)
    S = int(seq.get("steps", 2))
    sv = c.ch.lib.m(tr[0]["name"])["scale"] * float(c.args.get("scale", 1.0))
    fps = vid["fps"]
    f0 = tr[0]["f"]
    if "travel" in seq and seq.get("step"):
        tf = dict(zip(seq["frames"], seq["travel"]))  # by frame: a frame left out (canon) keeps the others' places
        T = [float(tf[r["f"]]) for r in tr]
        period, period_f = float(seq["period"]), int(seq.get("period_f", tr[-1]["f"] + 1 - f0))
    else:  # no plants measured: the torso's own travel
        T = [r["x"] - tr[0]["x"] for r in tr]
        period, period_f = T[-1] * len(T) / max(len(T) - 1, 1), tr[-1]["f"] + 1 - f0
    off = float(c.arg("offset", 0))
    n = len(qs)
    CL = 4 * S / float(c.arg("per_beat", 1))  # 16ths a cycle: a step a beat (per_beat 2: a step an 8th, a quick jog)
    eq = _ramp_time(c, qs)  # ramp: slow motion (legs and travel together) where its factor is below 1
    beat0 = int(divmod(eq[0] + off, CL)[0])
    names, X, ynat = [], [], []
    slip = []
    for q in eq:
        cyc_f, rem = divmod(q + off, CL)  # one division: the cycle and the place in it agree (a non-whole CL)
        cyc_i = int(cyc_f) - beat0
        # where this photo is in the cycle, by distance: the stride rate (the clip's doubled frames make its own
        # timing unreliable, the torso is the clock); the frame shown is the one whose stride-true place is nearest
        target = rem / CL * period
        cands = [(abs(T[k] - target), k, 0) for k in range(len(tr))] + [(abs(period - target), 0, 1)]
        _, k, wrap = min(cands)
        names.append(tr[k]["name"])
        X.append((target + cyc_i * period) * sv)
        slip.append((T[k] + wrap * period - target) * sv)  # how far its planted foot is off its plant
        ynat.append((tr[k]["y"] - vid["ground"]) * sv)
    X = np.array(X)
    if not c.dry:
        c.ch.notes[c.mv.where] = {"step_px": round(float(seq.get("step", 0)) * sv, 1),
                                  "px_per_photo": round(period * sv / (CL / every), 1),
                                  "foot_slip_max_px": round(float(np.max(np.abs(slip))), 1) if slip else 0.0}
    v_nat = period * sv / (CL / every)  # px per photo
    prev, nxt = c.prev, c.next
    if isinstance(xs_arg, list) and len(xs_arg) == 2:
        start, end = from_x, c.coord(xs_arg[1], "x")
        xsr = start + (X - X[0]) * ((end - start) / max(X[-1] - X[0], 1e-6)) if n > 1 else np.array([start])
    else:
        if c.has("speed"):
            X = X * float(c.arg("speed")) / v_nat
            v_nat = float(c.arg("speed"))
        elif nxt is not None and nxt.tx0 is not None and n > 1 and (from_x is not None or (
                prev is not None and prev.tx1 is not None)):
            # between two placed moves: the stride a little longer or shorter (at most `fit`, 10%) so the run
            # meets the next move (the feet slip by that much)
            a0 = from_x if from_x is not None else prev.tx1
            gaps_ = (X[-1] - X[0]) + v_nat * (1 if from_x is not None else 2)
            fit = float(c.arg("fit", 0.1))
            a = float(np.clip((nxt.tx0 - a0) / max(gaps_, 1e-6), 1 - fit, 1 + fit))
            X, v_nat = X * a, v_nat * a
            if not c.dry and abs(a - 1) > 1e-3:
                c.ch.notes[c.mv.where]["stride_fit"] = round(a, 3)
        if from_x is not None:
            start = from_x
        elif prev is not None and prev.tx1 is not None:
            start = prev.tx1 + v_nat
        elif nxt is not None and nxt.tx0 is not None:
            start = nxt.tx0 - v_nat - (X[-1] - X[0])
        else:
            raise ChoreoError("needs x: [x0, x1] or from_x: (nothing before it to start from)")
        xsr = start + (X - X[0])
        if nxt is not None and nxt.tx0 is not None and not c.dry:
            miss = nxt.tx0 - (xsr[-1] + v_nat)
            if abs(miss) > 0.5 * v_nat:
                c.ch.warn(c.mv.where, f"the stride-true run ends {abs(miss):.0f} px {'short of' if miss > 0 else 'past'} "
                                      f"the next move: give it {'more' if miss > 0 else 'less'} time or move the next move")
    for q, name, x, y in zip(qs, names, xsr, ynat):
        c.wall(q, name, float(x), c.feet + float(y), anchor="cxc", tag="run")
    c.st.exit_gap = v_nat


# ---------------------------------------------------------------------------------------------- vault

@m0("vault")
def vault(c: Ctx) -> None:
    """Vault over an obstacle with a `top` contact (M0), its sequence in order (default vault_PK2), the key
    drawing on `plant_on`.

    Sheet sequences: drawings up to the `takeoff` key run in on the ground (none: the run's last step is the
    take-off), the drawings to the `plant` key fly an eased arc to the hands' plant on the top, the drawings to the
    `push` key keep the hands on the top while the body passes over, the drawings to the `land` key fall on an arc
    to the landing just past it, the rest stride out. land_on: default as many photos as the sequence has drawings.

    Video sequences (frames of a clip: video sprites, motion transfer): the clip plays at its own speed with its key
    frame (`key`, default `plant`: where his weight is on the block) on `plant_on`, `speed` times as fast as the
    clip (default the sequence's `speed`, else 1: compressed so a move reads quick), with the clip's own travel; the
    figure is scaled to the body size and the clip's guide block is put on the obstacle (video_move: its edges on
    the obstacle's edges, his weight lifted onto the top, solid occluders cleared). land_on: the sequence's land
    key on that photo (the clip's pace between the two); from, to: limit the photos; splash: a key that lands with
    a splash."""
    o = c.ob("obstacle")
    top = o.contact("top").y
    _, seq = sequence(c, "vault")
    p = c.time("plant_on")
    s = c.ch.step_at(p)
    if "video" in seq:
        lo = c.time("from", ref=p) if c.has("from") else None
        hi = c.time("to", ref=p) if c.has("to") else None
        pins = {str(c.arg("key", "plant")): p}
        if c.has("land_on"):
            pins["land"] = c.time("land_on", ref=p)
        video_move(c, seq, pins, o, float(c.arg("speed")) if c.has("speed") else None,
                   "vault", q0=lo, q1=hi, label=f"{seq.get('label', 'vault')} {o.id}", splash=c.arg("splash"))
        return
    k = seq["keys"]
    D = seq["drawings"]
    x0, x1, w, f, u = o.x0, o.x1, o.x1 - o.x0, c.feet, c.u
    sc = float(c.args.get("scale", 1.0))
    ground_pre = D[:k["takeoff"] + 1] if "takeoff" in k else []
    air_pre = between(seq, "takeoff" if "takeoff" in k else None, "plant")
    mid = between(seq, "plant", "push") if "push" in k else []
    post_air = between(seq, "push" if "push" in k else "plant", "land")
    after = between(seq, "land", None)
    plant = key(seq, "plant")
    push_q = p + (len(mid) + 1) * s if "push" in k else None
    land = c.time("land_on", ref=p) if c.has("land_on") else (push_q or p) + (len(post_air) + 1) * s
    hx = x0 + min(0.35 * w, max(0.2 * w, 70 * u))
    pt = tors(c, plant, hx, top + 5 * u, "br")
    last = pt
    if push_q is not None:
        # the push: the hands behind him on the top, the torso a step or two on from the plant (by the torso, so
        # the figure moves evenly; with some drawings the hand ends near the top's far edge)
        push = key(seq, "push")
        dxh = c.ch.lib.torso_offset(push, "left", sc)[0]
        utx = pt[0] + float(np.clip(x1 - 12 * u + dxh - pt[0], 90 * u, 150 * u)) * (len(mid) + 1) ** 0.5
        lx_hand = utx - dxh
        ut = tors(c, push, lx_hand, top + 5 * u, "left")
        last = ut
    land_pose = key(seq, "land")
    n_post = int(round((land - (push_q or p)) / s)) - 1
    last_x = last[0]
    # the landing clears the obstacle's far edge (paint goes behind an obstacle in front of the wall)
    lt_x = max(x1 + 140 * u, min(last_x + (n_post + 1) * 120 * u, max(x1 + 140 * u, last_x + (n_post + 1) * 70 * u)))
    lt = tors(c, land_pose, lt_x, f, "foot")
    # the take-off: a step and a half before the obstacle
    n_pre = len(air_pre)
    to_x = min(x0 - 150 * u, pt[0] - (n_pre + 1) * 105 * u)
    run_y = f - 230 * u
    t0q = p - (n_pre + 1) * s  # the take-off photo
    if ground_pre:  # the sequence's own run-in, on the ground, ending with the take-off drawing at to_x
        for j_, name in enumerate(ground_pre):
            back = len(ground_pre) - 1 - j_
            c.wall(t0q - back * s, name, to_x - back * 92 * u, f, tag="vault.runin" if back else "vault.takeoff")
        a0 = tors(c, ground_pre[-1], to_x, f, "foot")
    else:
        a0 = (to_x, run_y)
    for (x, y, uu), name, q in zip(arc(n_pre, a0, pt, 45 * u), air_pre, [p - (n_pre - j_) * s for j_ in range(n_pre)]):
        c.wall(q, name, x, y, anchor="cxc", rot=-6 * (1 - uu), tag="vault")
    c.wall(p, plant, hx, top + 5 * u, anchor="br", keeper=True, tag="vault.plant", contact=(o.id, "top", "br"))
    c.stamp(p, c.pose("droplets"), hx, top - 30 * u, 230 * u, tag="vault")
    for j_, name in enumerate(mid):  # the body over the hands: torso between the plant's and the push's
        v = (j_ + 1) / (len(mid) + 1)
        c.wall(p + (j_ + 1) * s, name, lerp(pt[0], ut[0], v), lerp(pt[1], ut[1], v) - 20 * u, anchor="cxc",
               tag="vault.over")
    if push_q is not None:
        c.wall(push_q, push, lx_hand, top + 5 * u, anchor="left", tag="vault.over", contact=(o.id, "top", "left"))
    names = pick(post_air, n_post)
    for (x, y, uu), name, j_ in zip(arc(n_post, last, lt, 30 * u), names, range(n_post)):
        c.wall((push_q or p) + (j_ + 1) * s, name, x, y, anchor="cxc", rot=4 * uu, tag="vault")
    c.wall(land, land_pose, lt_x, f, keeper=True, tag="vault.land")
    rdx = recover_dx(c, (lt_x - last_x) / (n_post + 1))
    for j_, name in enumerate(after):
        c.wall(land + (j_ + 1) * s, name, lt_x + (j_ + 1) * rdx, f, tag="vault.out")
    c.label(p, hx, top, f"vault {o.id}")
    c.st.pre_x = None if ground_pre else to_x
    c.st.entry_gap = 60.0 * u
    c.st.exit_gap = 115.0 * u


def _matte_box(c: Ctx, name: str, x: float, y: float, anchor: str = "cxc") -> tuple[np.ndarray, int, int]:
    """A pose's paint (density over 0.2) where it lands with its anchor at (x, y), in street px."""
    lib = c.ch.lib
    m = lib.m(name)
    s = m["scale"] * float(c.args.get("scale", 1.0))
    d0 = lib.sprite(name)
    d = cv2.resize(d0, (max(1, int(round(d0.shape[1] * s))), max(1, int(round(d0.shape[0] * s)))),
                   interpolation=cv2.INTER_AREA)
    ax, ay = lib.anchor(name, anchor)
    return d > 0.2, int(round(x - ax * s)), int(round(y - ay * s))


def _hidden(c: Ctx, name: str, x: float, y: float) -> tuple[int, float, float]:
    """How much of a pose at (x, y) a solid obstacle hides (touch band excluded): pixels, and the lift and the step
    back that would clear it."""
    paint, X0, Y0 = _matte_box(c, name, x, y)
    h, w = paint.shape
    solid, touch = c.ch.scene.occluders(X0, X0 + w, Y0, Y0 + h, solid=True)
    hid = paint & (solid > 0.5) & (touch < 0.5)
    if not hid.any():
        return 0, 0.0, 0.0
    ys, xs = np.nonzero(hid)
    # the occluder's top over those columns (its first solid row) and its left edge over those rows
    cols = np.unique(xs)
    tops = [np.nonzero(solid[:, cx_] > 0.5)[0].min() for cx_ in cols]
    lift = float(max(ys.max() - min(tops) - 13, 1))
    rows = np.unique(ys)
    lefts = [np.nonzero(solid[ry, :] > 0.5)[0].min() for ry in rows]
    back = float(xs.max() - min(lefts) + 2)
    return int(hid.sum()), lift, back


# ---------------------------------------------------------------------------------------------- video moves

def key_frame(seq: dict, name: str) -> int:
    """The clip frame number of a sequence key (first and last: its first and last frame)."""
    if name == "first":
        return int(seq["frames"][0])
    if name == "last":
        return int(seq["frames"][-1])
    if name not in (seq.get("keys") or {}):
        raise ChoreoError(f"the sequence has no key {name!r} (keys: {', '.join(seq.get('keys') or {}) or 'none'})")
    return int(seq["frames"][seq["keys"][name]])


def video_keyed(c: Ctx, seq: dict, pins: list[tuple[float, int]], q0: float | None = None, q1: float | None = None,
                speed: float = 1.0, alternate: bool = False) -> list[tuple[float, dict]]:
    """Photos and the clip frame each one shows, with frames pinned to move times (pins: [(q, frame)]): piecewise
    linear between pins, `speed` times the clip's own speed before the first and after the last (move time: a song
    stop takes no clip time), one frame per photo (the nearest, moving on when the clip has; the sequence's `skip`
    frames and drawings with canon errors never show; where no drawing is near that moment, the nearest one is put
    where the clip has him then). Photos run from q0 to q1 (default: as far as the sequence's frames reach).
    alternate: where the clip is near still (its doubled frames dropped), a drawing never shows on more than two
    photos in a row: the third shows its neighbour in the clip instead (he breathes rather than freezes)."""
    vid, tr = video_track(c, seq)
    by_f = {r["f"]: r for r in tr}
    fs = sorted(by_f)
    g, fps = c.ch.grid, vid["fps"]
    pins = sorted(pins)
    if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(pins, pins[1:])):
        raise ChoreoError("the keys must come in the clip's order, each on a later photo")
    T = [(g.t(q), float(f)) for q, f in pins]

    def fv(q: float) -> float:
        t = g.t(q)
        if t <= T[0][0]:
            return T[0][1] + (t - T[0][0]) * fps * speed
        if t >= T[-1][0]:
            return T[-1][1] + (t - T[-1][0]) * fps * speed
        for (ta, fa), (tb, fb) in zip(T, T[1:]):
            if ta <= t <= tb:
                return fa + (fb - fa) * (t - ta) / max(tb - ta, 1e-9)
        return T[-1][1]

    out = []
    qk = pins[0][0]
    q = qk
    while (q0 is None or q - c.ch.step_at(q) >= q0 - 1e-6) and fv(q - c.ch.step_at(q)) >= fs[0] - 0.5:
        q -= c.ch.step_at(q)
    if q0 is not None:
        q = max(q, q0)
    while (q1 is None or q <= q1 + 1e-6) and fv(q) <= fs[-1] + 0.5:
        v = fv(q)
        if v >= fs[0] - 0.5:
            pin = next((f for qq, f in pins if abs(qq - q) < 1e-6), None)
            f = pin if pin in by_f else min(fs, key=lambda x: abs(x - v))
            if out and f <= out[-1][1]["f"] and v > out[-1][1]["f"] + 0.25:
                # the clip moved on but the nearest frame is the last one shown (a frame left out): the next one
                f = next((x for x in fs if x > out[-1][1]["f"]), f)
            if alternate and len(out) >= 2 and out[-1][1]["f"] == f and out[-2][1]["f"] == f and len(fs) > 1:
                i = fs.index(f)
                nb = [fs[j] for j in (i - 1, i + 1) if 0 <= j < len(fs)]
                f = min(nb, key=lambda x: abs(x - v))
            r = by_f[f]
            if abs(f - v) > 1.5 or (abs(f - v) > 0.25 and vid["ground"] - r["low"] >= 15):
                # no drawing near this moment (frames left out), or in the air between two drawings (a slow clip):
                # the drawing nearest, where the clip has him at that moment (his feet planted keep their own place)
                r = dict(r, **_track_at(vid["frames"], v))
            out.append((q, r))
        q += c.ch.step_at(q)
    return out


def _track_at(frames: list[dict], v: float) -> dict:
    """The clip's torso x, y and low point at frame time v (between two tracked frames: linear)."""
    fs = [r["f"] for r in frames]
    j = int(np.searchsorted(fs, v))
    if j <= 0 or j >= len(fs):
        r = frames[min(max(j, 0), len(fs) - 1)]
        return {"x": r["x"], "y": r["y"], "low": r["low"]}
    a, b = frames[j - 1], frames[j]
    w = (v - a["f"]) / max(b["f"] - a["f"], 1e-9)
    return {k: a[k] + (b[k] - a[k]) * w for k in ("x", "y", "low")}


def _ramp(v: float) -> float:
    v = min(max(v, 0.0), 1.0)
    return v * v * (3 - 2 * v)


def _hold_ramp(n: int, on: list[int], grounded: list[bool]) -> list[float]:
    """0..1 per photo: 1 on the `on` photos (his weight on a top, his hands on a bar), 0 while his feet are on the
    ground before and after, rising and falling smoothly in between."""
    if not on:
        return [0.0] * n
    a, b = on[0], on[-1]
    g0 = max([k for k in range(n) if grounded[k] and k < a], default=-1)
    g1 = min([k for k in range(n) if grounded[k] and k > b], default=n)
    out = []
    for k in range(n):
        if k < a:
            out.append(_ramp((k - g0) / max(a - g0, 1)) if k > g0 else 0.0)
        elif k <= b:
            out.append(1.0)
        else:
            out.append(1 - _ramp((k - b) / max(g1 - b, 1)) if k < g1 else 0.0)
    return out


def video_move(c: Ctx, seq: dict, pins: dict, o=None, speed: float | None = None, tag: str = "clip",
               x_pins: dict | None = None, bottom: dict | None = None, stretch: list | None = None,
               q0: float | None = None, q1: float | None = None, label: str | None = None,
               splash: str | None = None, alternate: bool = False) -> list:
    """A move played from a clip (frames of a video sequence), one frame per photo, with the clip's own heights
    and travel scaled to the figure.

    pins {key: move time}: key frames on photos (piecewise linear between them, `speed` times the clip's speed
    outside them). The street place comes from the clip's guide shape when there is an obstacle `o`:
    - a pole (a pipe): his x relative to the pole is the clip's, scaled (the clip's tallest guide shape); a climb
      clip with a beam too: his hands reach the street's beam above the pole while he hangs from it;
    - a bar (a beam or a bar contact; its widest guide shape): the clip's bar on the street's bar: his hands reach
      it while he hangs (the photos between the catch and the release keys, or where his top is at the bar), the
      height easing in from the ground and back down to the landing;
    - a block: its edges on the obstacle's edges (the travel over it stretched to the obstacle's width), his pace
      kept (each step at least 80% of the last, never back), and a lift: the obstacle's height over the block's on
      the photos where his weight is on the block (his low point above 70% of the block's height; or the
      sequence's `on` frames, [first, last] or "none"), rising from the ground and falling back smoothly; then
      every photo is cleared of the solid occluders (touch band excluded): before the top he steps back when his
      pace allows, else he is
      lifted, never sinking again on the way up (a step back of at most 60 px: a bigger overlap is a lift); under
      an obstacle (an `under` contact: a slide) he is never lifted. x_offset shifts the clip along the obstacle.
    Without an obstacle: x_pins {key: torso x} place key frames (between two pinned keys the travel is stretched to
    fit, outside them stride-true), else it starts a step after the previous move. stretch [key_a, key_b, factor]
    (or a list of them): the travel between two keys times factor (a longer flight, a shorter get-up). bottom
    {key: y}: that frame's low point on y (the
    others shift with it, easing in from the ground before and back out by his next step after; held to the end
    when no step follows). splash: a key whose photo gets a landing splash."""
    vid, tr = video_track(c, seq)
    u, f = c.u, c.feet
    sv = c.ch.lib.m(tr[0]["name"])["scale"] * float(c.args.get("scale", 1.0))
    gv = float(vid["ground"])
    sp = float(speed if speed is not None else seq.get("speed", 1.0))
    pin_list = [(q, key_frame(seq, k)) for k, q in pins.items()]
    shots = video_keyed(c, seq, pin_list, q0, q1, sp, alternate)
    if not shots:
        raise ChoreoError("the clip has no frames on these photos")
    n = len(shots)
    R = [r for _, r in shots]
    sts = [stretch] if stretch and not isinstance(stretch[0], list) else (stretch or [])
    kf = {k: key_frame(seq, k) for k in list(pins) + list(x_pins or {}) + list(bottom or {}) +
          ([splash] if splash else []) + [str(v) for st in sts for v in st[:2]]}
    idx = {k: min(range(n), key=lambda i: abs(R[i]["f"] - fr)) for k, fr in kf.items()}
    grounded = [gv - r["low"] < 15 for r in R]
    xs = np.zeros(n)
    ys = np.array([f + (r["y"] - gv) * sv for r in R])
    shape = None
    pre = np.zeros(n)  # bottom pins: his low point on a set height (applied before the clearing)
    if bottom:
        pts = sorted([(idx[k], c.coord(v, "y") - (f + (R[idx[k]]["low"] - gv) * sv)) for k, v in bottom.items()])
        g0 = max([k for k in range(n) if grounded[k] and k < pts[0][0]], default=-1)
        g1 = min([k for k in range(n) if grounded[k] and k > pts[-1][0]], default=None)
        for k in range(n):
            if k >= pts[-1][0]:  # held to the end, or eased back to the clip's own height by his next step
                pre[k] = pts[-1][1] if g1 is None else pts[-1][1] * (1 - _ramp((k - pts[-1][0]) /
                                                                                 max(g1 - pts[-1][0], 1)))
            elif k >= pts[0][0]:
                (ia, va), (ib, vb) = next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= k <= b[0])
                pre[k] = lerp(va, vb, (k - ia) / max(ib - ia, 1))
            elif k > g0:
                pre[k] = pts[0][1] * _ramp((k - g0) / max(pts[0][0] - g0, 1))
        # never below the ground line on the way in or out (his low point at most on it)
        pre = np.minimum(pre, np.array([(gv - r["low"]) * sv for r in R]))
        for k, v in pts:  # the pinned photos themselves keep their pin
            pre[k] = v
        ys = ys + pre
    if o is not None:
        bx0, by0, bx1, by1 = vid["block"] or (0, 0, 0, 0)
        sh_ = vid.get("shapes") or [list(vid["block"] or (0, 0, 0, 0))]
        if "pole" in o.contacts and "under" not in o.contacts:
            shape = "pole"
            px = o.contact("pole").x
            pole = max(sh_, key=lambda b: (b[3] - b[1]) / max(b[2] - b[0], 1))  # the tallest, thinnest shape
            xs = np.array([px + (r["x"] - (pole[0] + pole[2]) / 2) * sv for r in R])
            if len(sh_) > 1:  # a climb: the beam it leads to sets his height once he hangs from it
                bar = max(sh_, key=lambda b: (b[2] - b[0]) / max(b[3] - b[1], 1))
                bar_c = (bar[1] + bar[3]) / 2
                beam = next((ob for ob in c.ch.scene.obstacles.values() if {"beam", "bar"} & set(ob.contacts)
                             and ob.x0 <= px + (bar[2] - (pole[0] + pole[2]) / 2) * sv
                             and ob.x1 >= px + (bar[0] - (pole[0] + pole[2]) / 2) * sv), None)
                if beam is not None:
                    off = (beam.contact("beam", "bar").y - f) - (bar_c - gv) * sv
                    on = [k for k, r in enumerate(R) if r["y"] - r["h"] / 2 < bar_c + 40]
                    ys = ys + off * np.array(_hold_ramp(n, on, grounded))
        elif {"bar", "beam"} & set(o.contacts) and "under" not in o.contacts:
            shape = "bar"
            bc = o.contact("bar", "beam")
            bar = max(sh_, key=lambda b: (b[2] - b[0]) / max(b[3] - b[1], 1))
            bx0, by0, bx1, by1 = bar
            bar_c = (by0 + by1) / 2
            off = (bc.y - f) - (bar_c - gv) * sv  # the street bar against the scaled clip bar
            a_, b_ = seq.get("keys", {}).get("catch"), seq.get("keys", {}).get("release")
            if a_ is not None and b_ is not None:
                fa, fb = seq["frames"][a_], seq["frames"][b_]
                on = [k for k, r in enumerate(R) if fa <= r["f"] <= fb]
            else:
                on = [k for k, r in enumerate(R) if r["y"] - r["h"] / 2 < bar_c + 40]
            w = _hold_ramp(n, on, grounded)
            ys = ys + off * np.array(w)
            bar_x0 = bc.x0 if bc.x0 is not None else o.x0
            xs = np.array([bar_x0 + (r["x"] - bx0) * sv for r in R])
        else:
            shape = "block"
            under = "under" in o.contacts  # a slide: cleared by stepping back, never lifted into the obstacle
            top = o.contact("under").y if under and "top" not in o.contacts and "step" not in o.contacts \
                else o.contact("top", "step").y
            bh = gv - by0
            shift = float(c.arg("x_offset", 0.0))
            lift_top = (f - top) - bh * sv
            if seq.get("on") == "none":  # hops: never his weight on a top
                on = []
            elif seq.get("on"):
                fa, fb = seq["on"]
                on = [k for k, r in enumerate(R) if fa <= r["f"] <= fb]
            else:
                on = [k for k, r in enumerate(R) if gv - r["low"] > 0.7 * bh]
            w = _hold_ramp(n, on, grounded)
            lifts = [lift_top * v for v in w]
            prev_v = c.prev.v_out if c.prev is not None and c.prev.v_out else 70 * u
            pace = bool(c.arg("pace", True))
            out_x: list[float] = []
            out_y: list[float] = []
            out_l: list[float] = []
            t_first = on[0] if on else n
            for k, r in enumerate(R):
                if r["x"] < bx0:
                    x = o.x0 + (r["x"] - bx0) * sv
                elif r["x"] > bx1:
                    x = o.x1 + (r["x"] - bx1) * sv
                else:
                    x = lerp(o.x0, o.x1, (r["x"] - bx0) / max(bx1 - bx0, 1))
                x += shift
                L = 0.0 if under else lifts[k]
                y = ys[k] - L
                if out_x and pace:  # his pace: at least 80% of his last step, never back
                    last = out_x[-1] - out_x[-2] if len(out_x) > 1 else prev_v
                    x = max(x, out_x[-1] + max(0.8 * last, 20 * u))
                elif out_x:  # slowed down (pace: false): the clip's own travel, never back
                    x = max(x, out_x[-1])
                for _ in range(4):  # clear the solid obstacles
                    nh, need_lift, need_back = _hidden(c, r["name"], x, y)
                    if not nh:
                        break
                    floor_x = out_x[-1] + max(0.8 * (out_x[-1] - out_x[-2] if len(out_x) > 1 else prev_v), 20 * u) \
                        if out_x else -1e9
                    room = x - need_back >= floor_x - (1e-6 if grounded[k] else 0.0)
                    if k < t_first and need_back <= 60 * u and room:
                        x -= need_back  # a small step back (the take-off a little earlier), else a lift
                    elif under:
                        break  # under an obstacle: never lifted into it (the QA says if it still shows)
                    else:
                        y -= need_lift
                        L += need_lift
                if k < t_first and out_l and not grounded[k] and L < out_l[-1] and k > 0 and not grounded[k - 1]:
                    y -= out_l[-1] - L  # on the way up he does not sink again
                    L = out_l[-1]
                out_x.append(x)
                out_y.append(y)
                out_l.append(L)
            xs, ys = np.array(out_x), np.array(out_y)
            if not c.dry:
                c.ch.notes[c.mv.where] = {"lift_on_top_px": round(lift_top, 1), "lifts": [round(v, 1) for v in out_l],
                                          "speed": sp}
    if shape in (None, "bar"):
        pinned = sorted([(idx[k], c.coord(v, "x")) for k, v in (x_pins or {}).items()])
        if shape is None:
            rx = np.array([r["x"] for r in R])
            if not c.arg("back", False):  # turning on the spot: no step back (back: true: the clip's own steps
                rx = np.maximum.accumulate(rx)  # back too: a bounce off a wall, sliding down against it)
            if not pinned:
                if c.prev is not None and c.prev.tx1 is not None:
                    gap = c.prev.exit_gap or c.prev.v_out or 90 * u
                    pinned = [(0, c.prev.tx1 + gap)]
                elif c.dry:  # planned before the move before it has a place: its time only
                    c.st.q0, c.st.q1 = shots[0][0], shots[-1][0]
                    return []
                else:
                    raise ChoreoError("needs x: {key: torso x} (nothing before it to start from)")
            xs = np.zeros(n)
            for k in range(n):
                before = [p for p in pinned if p[0] <= k]
                after = [p for p in pinned if p[0] >= k]
                if before and after and before[-1][0] != after[0][0]:
                    (ia, xa), (ib, xb) = before[-1], after[0]
                    xs[k] = lerp(xa, xb, (rx[k] - rx[ia]) / (rx[ib] - rx[ia]) if rx[ib] != rx[ia] else 0.0)
                else:
                    ia, xa = before[-1] if before else after[0]
                    xs[k] = xa + (rx[k] - rx[ia]) * sv
        elif pinned:  # a bar: move the whole swing to a catch point
            k0, x0 = pinned[0]
            xs = xs + (x0 - xs[k0])
    if shape in ("pole", "bar"):  # swaying round a pole or under a bar: never a step back
        xs = np.maximum.accumulate(xs)
    gb = c.arg("ground_body")
    if gb:  # a slide: his body (hip, back, lower leg) lies on the ground line, not just his lowest point
        ka, kb = (idx[str(k_)] if str(k_) in idx else
                  min(range(n), key=lambda i: abs(R[i]["f"] - key_frame(seq, str(k_)))) for k_ in gb[:2])
        sc_ = sv / c.ch.lib.m(R[0]["name"])["scale"]
        dys = np.zeros(n)
        for k in range(max(ka - 2, 0), min(kb + 3, n)):
            dys[k] = _body_drop(c, R[k]["name"], float(xs[k]), float(ys[k]), sc_)
        w = np.array([1.0 if ka <= k <= kb else (max(0.0, 1 - (ka - k) / 3) if k < ka else max(0.0, 1 - (k - kb) / 3))
                      for k in range(n)])
        ys = ys + dys * w
    for st in ([stretch] if stretch and not isinstance(stretch[0], list) else (stretch or [])):
        ka, kb, fac = idx[st[0]], idx[st[1]], float(st[2])
        d = np.zeros(n)
        for k in range(n):
            if ka < k <= kb:
                d[k] = (fac - 1) * (xs[k] - xs[ka])
            elif k > kb:
                d[k] = (fac - 1) * (xs[kb] - xs[ka])
        xs = xs + d
    keyname = {kf[k]: k for k in pins}
    events = []
    for k, (q, r) in enumerate(shots):
        kn = keyname.get(r["f"])
        events.append(c.wall(q, r["name"], float(xs[k]), float(ys[k]), anchor="cxc", keeper=kn is not None,
                             tag=f"{tag}.{kn}" if kn else tag))
    if splash:
        k = idx[splash]
        q = shots[k][0]
        lx = float(xs[k])
        c.stamp(q, c.pose("splash"), lx - 110 * u, f - 130 * u, 560 * u, flags={"splash": True, "primer": True},
                tag=f"{tag}.land")
        c.stamp(q, c.pose("droplets"), lx + 90 * u, f - 280 * u, 460 * u, tag=f"{tag}.land")
    if label:
        k = idx.get(next(iter(pins)), 0)
        c.label(shots[k][0], float(xs[k]), float(ys[k]) - 200 * u, label)
    c.st.entry_gap = float(xs[1] - xs[0]) if n > 1 else 60.0 * u
    c.st.exit_gap = float(np.clip(xs[-1] - xs[-2], 60 * u, 130 * u)) if n > 1 else 90.0 * u
    return events


def _body_drop(c: Ctx, name: str, x: float, y: float, scale: float) -> float:
    """How far to lower a lying drawing (anchor cxc at x, y, scale on top of its own) so its body rests on the ground
    line: the 80th percentile of its columns' lowest paint 8 px under the feet line, its very lowest paint never
    past the wall's foot (the QA's ground rule). Never raises it."""
    lib = c.ch.lib
    m = lib.m(name)
    d = lib.sprite(name)
    s = m["scale"] * scale
    cy = (m["bbox"][1] + m["bbox"][3]) / 2
    ys, xs = np.nonzero(d > 0.4)
    if not len(ys):
        return 0.0
    col_low = np.zeros(d.shape[1]) - 1
    np.maximum.at(col_low, xs, ys)
    col_low = col_low[col_low >= 0]
    body = y + (np.percentile(col_low, 80) - cy) * s
    lowest = y + (np.percentile(ys, 99.5) - cy) * s
    f = c.ch.scene.feet_at(x)
    wb = c.ch.scene.wall_base_at(x)
    return float(max(0.0, min((f + 8 * c.u) - body, (wb - 13) - lowest)))  # the photo jitter: a few px


def _video_args(c: Ctx, seq: dict, pins: dict, o, tag: str, label: str, splash: str | None = None) -> None:
    """A template's video branch: its own pins plus the common clip arguments (speed, from, to, x, stretch,
    bottom, splash; keys: more of the clip's keys pinned to times, the clip breathing between neighbouring frames
    where it is stretched; arc: {from, to, height}: the flight between two pinned keys on one smooth arc)."""
    q0 = c.time("from", ref=min(pins.values())) if c.has("from") else None
    q1 = c.time("to", ref=min(pins.values())) if c.has("to") else None
    st = c.arg("stretch")
    xp = c.arg("x_keys")
    for kk, v in (c.arg("keys") or {}).items():  # more of the clip's keys pinned to their times
        pins[str(kk)] = c.at(v, ref=min(pins.values()), key=f"keys.{kk}")
    evs = video_move(c, seq, pins, o, float(c.arg("speed")) if c.has("speed") else None, tag,
                     dict(xp) if isinstance(xp, dict) else None, c.arg("bottom"), list(st) if st else None, q0, q1,
                     label, c.arg("splash", splash), alternate=bool(c.arg("keys")))
    ar = c.arg("arc")
    if ar and evs:  # the flight between two pinned keys on one smooth arc (slowed through a song stop: no float)
        ar = dict(ar) if isinstance(ar, dict) else {}
        ka, kb = str(ar.get("from", "takeoff")), str(ar.get("to", "land"))
        if ka not in pins or kb not in pins:
            raise ChoreoError(f"arc: pin both {ka} and {kb}")
        arc_flight(c, evs, pins[ka], pins[kb], float(ar.get("height", 0.0)) * c.u)


@m0("clip")
def clip(c: Ctx) -> None:
    """Play a clip (a video sequence) on the wall: its frames one per photo, the clip's own heights and travel
    scaled to the figure (see video_move). sequence (required); keys: {key: time} (key names from the sequence,
    or first and last; default: its first frame on `from`, else one photo after the previous move); speed (the
    sequence's, else 1) outside the keys; from, to: limit the photos; obstacle: put the clip's guide shape on it (a
    pole, a bar, a block); x: {key: torso x} (back: true lets him step back between them as the clip does: a
    bounce off a wall); bottom: {key: y of its low point}; stretch: [key_a, key_b, factor];
    splash: a key that lands with a splash; arc: {from: takeoff, to: land, height} puts the flight between those
    two pinned keys on one smooth arc (arc_flight: a rise and a fall, no floating top); ground_body: [key_a,
    key_b]: a slide, his body lying on the ground line between those keys (eased in and out); label."""
    name = c.require("sequence")
    try:
        seq = c.ch.lib.sequence(str(name))
    except KeyError as e:
        raise ChoreoError(str(e.args[0])) from e
    if "video" not in seq:
        raise ChoreoError(f"{name} is not a video sequence")
    q0 = c.time("from") if c.has("from") else None
    q1 = c.time("to", ref=q0) if c.has("to") else None
    pins = {k: c.at(v, ref=q0, key=f"keys.{k}") for k, v in (c.arg("keys") or {}).items()}
    if not pins:
        start = q0 if q0 is not None else (c.prev.q1 + c.ch.step_at(c.prev.q1) if c.prev is not None and
                                           c.prev.q1 is not None else None)
        if start is None:
            raise ChoreoError("needs keys: or from: (it is the first move)")
        pins = {"first": start}
    o = c.ob("obstacle") if c.has("obstacle") else None
    st = c.arg("stretch")
    evs = video_move(c, seq, pins, o, float(c.arg("speed")) if c.has("speed") else None, str(c.arg("tag", "clip")),
                     c.arg("x") if isinstance(c.arg("x"), dict) else None, c.arg("bottom"),
                     list(st) if st else None, q0, q1, c.arg("label") or seq.get("label"), c.arg("splash"))
    ar = c.arg("arc")
    if ar and evs:  # a free leap: the flight between two pinned keys on one smooth arc (a rise and a fall)
        ar = dict(ar) if isinstance(ar, dict) else {}
        ka, kb = ar.get("from", "takeoff"), ar.get("to", "land")
        if ka not in pins or kb not in pins:
            raise ChoreoError(f"arc: pin both {ka} and {kb} in keys")
        arc_flight(c, evs, pins[ka], pins[kb], float(ar.get("height", 0.0)) * c.u)


# ---------------------------------------------------------------------------------------------- leap (J2)

@m0("leap")
def leap(c: Ctx) -> None:
    """Leap over a gap (M0, J2; J1 stays as before): its sequence (default leap_PK1: take-off, the flight drawings,
    landing, recovery) in order. The take-off is on `at` (when he leaves the wall), or later when the flight would
    otherwise need more photos than the sequence has flight drawings; the flight is one eased arc as a flat paint
    skin in the air; landing on `land_on` (default the next snare), the recovery a photo later."""
    style = str(c.arg("style", c.ch.jump)).upper()
    if style == "J1":
        return _leap_legacy(c)
    g = c.ob("gap", ("gap",))
    _, seq = sequence(c, "leap")
    at = c.time("at")
    s = c.ch.step_at(at)
    land = c.time("land_on", default="snare", ref=at)
    if land <= at:
        raise ChoreoError(f"land_on {c.ch.grid.addr(land)} is not after at {c.ch.grid.addr(at)}")
    if "video" in seq:
        return _leap_video(c, g, seq, at, land, s)
    near, far = g.x0, g.x1
    u, f = c.u, c.feet
    tx = c.coord(c.arg("takeoff_x"), "x") if c.has("takeoff_x") else near - 90 * u
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else far + 300 * u
    glow = g.glow or [1.0, 0.55, 0.22]
    pool = between(seq, "takeoff", "land")
    tko = max(at, land - (len(pool) + 1) * s)
    n = int(round((land - tko) / s)) - 1
    tk = key(seq, "takeoff")
    c.wall(tko, tk, tx, keeper=True, flags={"takeoff": True, "burst": "J2", "edge": near + 2, "glow": glow},
           tag="leap.takeoff")
    c.stamp(tko, c.pose("spray"), tx - 50 * u, f - 216 * u, 420 * u, tag="leap.spray")
    a = tors(c, tk, tx, f, "foot")
    lp = key(seq, "land")
    b = tors(c, lp, lx, f, "foot")
    apex = (f - 400 * u) - (a[1] + b[1]) / 2
    rots = [r for _, _, r in LEAP_ARC]
    for j, ((x, y, uu), name) in enumerate(zip(arc(n, a, b, max(apex, 60 * u)), pick(pool, n))):
        q = tko + (j + 1) * s
        c.air(q, name, x, y, anchor="cxc", rot=float(np.interp(uu, np.linspace(0, 1, len(rots)), rots)),
              flags={"flight": True, "glow": glow}, tag="leap.flight")
    zooms = [1.36, 1.36, 1.34, 1.30, 1.24]
    for j, q in enumerate([tko] + [tko + (k + 1) * s for k in range(n)]):
        c.cam(q, zoom=float(np.interp(j, np.linspace(0, n, len(zooms)), zooms)), dx=40 * u, dy=-25 * u, snap=True,
              free=True)
    c.wall(land, lp, lx, keeper=True, flags={"halo": True}, tag="leap.land", contact=(g.id, "far", "foot", "past"))
    c.stamp(land, c.pose("splash"), lx - 125 * u, f - 141 * u, 720 * 0.95 * u, flags={"splash": True, "primer": True},
            tag="leap.splash")
    c.stamp(land, c.pose("droplets"), lx + 110 * u, f - 296 * u, 560 * 0.95 * u, tag="leap.splash")
    c.stamp(land, c.pose("drips"), lx - 140 * u, f - 236 * u, 400 * 0.95 * u, tag="leap.splash")
    pts = arc(n, a, b, max(apex, 60 * u))
    rdx = recover_dx(c, b[0] - (pts[-1][0] if pts else a[0]))
    for j, name in enumerate(between(seq, "land", None)):
        c.wall(land + (j + 1) * s, name, lx + (j + 1) * rdx, tag="leap.recover")
    c.cam(land, zoom=0.97, snap=True)
    c.cam(land + s, zoom=0.99)
    c.label(tko, tx, f, "take-off (J2)")
    c.label(land, lx, f, "landing")
    c.st.entry_gap = 90.0 * u
    c.st.exit_gap = 130.0 * u


def arc_flight(c: Ctx, evs: list, qa: float, qb: float, lift: float = 0.0) -> None:
    """Re-place a clip's flight (the photos strictly between move times qa and qb) on one smooth eased arc from the
    photo on qa to the photo on qb (faster at take-off and landing, slower at the top), as high as the clip's own
    top plus `lift`: the clip's frames are uneven in time (doubled frames dropped) and a stretched flight would hold
    its top for many photos (it floats); the drawings stay the clip's."""
    fl = [e for e in evs if qa + 1e-6 < c.clock.fwd(e.q) < qb - 1e-6]
    e_a = next((e for e in evs if abs(c.clock.fwd(e.q) - qa) < 1e-6), None)
    e_b = next((e for e in evs if abs(c.clock.fwd(e.q) - qb) < 1e-6), None)
    if not fl or e_a is None or e_b is None:
        return
    lib = c.ch.lib
    dy = [lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)[1] for e in (e_a, e_b)]
    y_top = min(e.y + lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)[1] for e in fl) - lift
    a, b = (e_a.x, e_a.y + dy[0]), (e_b.x, e_b.y + dy[1])
    for e, (x, y, _) in zip(fl, arc(len(fl), a, b, max((a[1] + b[1]) / 2 - y_top, 0.0))):
        e.x, e.y = x, y - lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)[1]


def _leap_video(c: Ctx, g, seq: dict, at: float, land: float, s: float) -> None:
    """The J2 leap from a clip (video_move): its `takeoff` key (his feet leave the ground) on `at`, its `land` key
    (touch-down) on `land_on`, the frames between them stretched over the photos (the flight, not the crouch), the
    clip's own pace after the landing (`speed`) until `to`. Torso x: takeoff_x (default 45 px before the near edge)
    and land_x (250 px past the far side), the flight's travel spread like the clip's in between. height (px, 0)
    lifts the flight on top of the clip's own arc (a parabola, highest halfway), so it has real height over the gap.
    The flight photos are the flat paint skin in the air (the gap's glow on its rim), with a ghost trail of the
    photos before as pale silhouettes (ghosts: their fade in 16ths, 3; 0 for none); the take-off photo leaves the
    wall as the first of them and the landing photo is the last (the skin hitting the far wall: no primer, no primed
    outline round him). A splash at the take-off and on the landing; back on the wall the primer patch under him
    grows in over the `grow` photos from the landing (4) instead of appearing at full size. The camera eases
    throughout: wider over the take-off photos to hold the whole arc and the gap (16:9; 9:16 follows him), panning
    with him along the arc so he lands near the frame's middle, then closing in to the follow framing over `settle`
    photos (5).

    A clip that starts before the take-off (the coil at the edge: edge-launch) plays from `from`, with more of its
    keys pinned to their words (keys: {lift: 41.4.3, coil: 42.4.1}); those photos are on the wall. The take-off can
    be the biggest hit: burst: {flecks, splats, power} throws paint from his feet as he leaves the wall; hang: the
    16ths the whole arc's ghost skins hang over the gap after the landing (0); splash: the landing splash's size
    (1); knock: a street light he flies past is shattered as he leaves (pale glass flecks, dark from then on)."""
    u, f = c.u, c.feet
    near, far = g.x0, g.x1
    glow = g.glow or [1.0, 0.55, 0.22]
    tx = c.coord(c.arg("takeoff_x"), "x") if c.has("takeoff_x") else near - 45 * u
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else far + 250 * u
    q1 = c.time("to", ref=at) if c.has("to") else None
    q0 = c.time("from", ref=at) if c.has("from") else at
    pins = {"takeoff": at, "land": land}
    for kk, v in (c.arg("keys") or {}).items():
        pins[str(kk)] = c.at(v, ref=at, key=f"keys.{kk}")
    evs = video_move(c, seq, pins, None, float(c.arg("speed")) if c.has("speed") else None,
                     "leap", {"takeoff": tx, "land": lx}, None, None, q0, q1, None, None, alternate=q0 < at)
    if not evs:
        return
    ghosts = float(c.arg("ghosts", 3.0))
    top = f
    arc_flight(c, evs, at, land, float(c.arg("height", 0.0)) * u)
    for e in evs:
        qm = c.clock.fwd(e.q)
        if at - 1e-6 < qm < land + 1e-6:  # leaving the wall, over the gap and hitting the far wall: the flat
            e.layer = "air"                 # dark skin (painted on a dark shutter he would vanish on the landing)
            e.flags = dict(e.flags, flight=True, glow=glow, ghosts=ghosts if qm < land - 1e-6 else 0.0,
                           hang=[float(c.clock.inv(land)), float(c.arg("hang", 0))] if c.arg("hang") else None)
            m = c.ch.lib.m(e.pose)
            top = min(top, e.y - (m["bbox"][3] - m["bbox"][1]) * m["scale"] * e.scale / 2)
    # back on the wall: the primer under him arrives gradually (no pale patch appears from nothing round the landing):
    # the landing photo is still the skin, then the patches of the next wall photos start small round him and grow to
    # their usual size by the `grow`-th photo from the landing
    walls = sorted([e for e in evs if e.layer == "wall" and c.clock.fwd(e.q) > land + 1e-6], key=lambda e: e.q)
    n_grow = int(c.arg("grow", 4))
    for j, e in enumerate(walls[:n_grow - 1]):
        e.flags = dict(e.flags, patch_grow=[j + 1, n_grow])
    c.stamp(at, c.pose("splash"), tx - 40 * u, f - 110 * u, 430 * u, flags={"splash": True, "primer": True},
            tag="leap.splash")
    c.stamp(at, c.pose("spray"), tx - 70 * u, f - 216 * u, 420 * u, tag="leap.spray")
    k_s = float(c.arg("splash", 1.0))
    c.stamp(land, c.pose("splash"), lx - 110 * u * k_s, f - 130 * u, 520 * u * k_s, flags={"splash": True},
            tag="leap.land")
    c.stamp(land, c.pose("droplets"), lx + 90 * u * k_s, f - 280 * u * k_s, 460 * u * k_s, tag="leap.land")
    c.stamp(land, c.pose("drips"), lx - 140 * u * k_s, f - 236 * u, 380 * u * k_s, tag="leap.splash")
    rs = np.random.default_rng([c.mv.seed, c.mv.local_i, 77])
    if c.arg("burst"):  # the take-off as an explosion of paint from his feet
        b = dict(c.arg("burst")) if isinstance(c.arg("burst"), dict) else {}
        c.burst(at, tx - 30 * u, f - 90 * u, dirx=1.0, seed=int(rs.integers(0, 2 ** 31)),
                power=float(b.get("power", 1.4)), flecks=int(b.get("flecks", 50)), splats=int(b.get("splats", 70)),
                spread=float(b.get("spread", 50)))
    if c.arg("knock"):  # he goes on red: the light he flies past is shattered as he leaves
        lid = str(c.arg("knock"))
        L = next((lt for lt in c.ch.scene.lights if lt.get("id") == lid), None)
        if L is None:
            raise ChoreoError(f"knock: no street light {lid!r} (obstacles.json lights)")
        c.light(at - s, lid, 2.2)  # a last flare on the coil's end
        c.light_out(at, lid)
        c.burst(at, float(L["x"]), float(L["y"]), dirx=1.0, seed=int(rs.integers(0, 2 ** 31)), power=0.8,
                flecks=45, splats=0, spread=40, albedo=0.6)
    # the camera, eased throughout (no snap): it widens over the first photos to hold the whole arc and the gap
    # (16:9; the crop as high as the arc needs), pans with him along the arc so he lands near the frame's middle,
    # and closes in over the photos after the landing, back to the follow framing by the time `settle` photos
    # have passed (the 9:16 follows him all along)
    from .camera import CW0
    cw = max(lx - tx + 560 * u, (f - top + 60 * u) / 0.70 * 16 / 9)
    zoom = min(cw / (CW0 * u), (c.ch.scene.H - 80) * 16 / 9 / (CW0 * u))
    n = max(int(round((land - at) / s)), 1)
    lead = 0.07 * CW0 * u + 40 * u  # about where the follow framing puts its centre ahead of him
    c0, c1 = tx + lead, lx + lead
    ramp = [0.2, 0.45, 0.7, 0.9]  # the widening over the take-off photos (with the impact's own punch on the first)
    for k in range(n):
        v = k / n
        sm = v * v * (3 - 2 * v)
        z = 1 + (zoom - 1) * (ramp[k] if k < len(ramp) else 1.0)
        c.cam(at + k * s, cx=c0 + (c1 - c0) * sm, zoom=z, free=True, follow_v=True)
    by_q = {round(c.clock.fwd(e.q), 4): e for e in walls}
    settle = int(c.arg("settle", 5))
    for j in range(settle):
        e = by_q.get(round(land + j * s, 4))
        x = (e.x if e is not None else lx) + lead
        w = (j + 1) / settle
        c.cam(land + j * s, cx=x, zoom=zoom + (1 - zoom) * (0.12 + 0.88 * w * w * (3 - 2 * w)) if j < settle - 1
              else 1.0, free=True, follow_v=True)
    c.label(at, tx, f, "take-off (J2)")
    c.label(land, lx, f, "landing")


# ---------------------------------------------------------------------------------------------- swings and steps

@m0("swing_release")
def swing_release(c: Ctx) -> None:
    """Kick off a beam and release on `release_on` (M0): its sequence (default release_PK3) in order, the flight
    an eased arc along the wall, landing on `land_on` (default the next kick). bar, x, land_x as before. A video
    sequence plays the whole hang from its clip (video_move: the clip's bar on the bar), its `release` key on
    release_on (catch_on and land_on pin the catch and the landing too; catch_x: where his hands catch the bar)."""
    o = c.ob("bar", ("beam", "ledge", "ladder", "pipe"))
    by = o.contact("beam", "bar").y
    _, seq = sequence(c, "swing_release")
    r = c.time("release_on")
    if "video" in seq:  # the whole swing from a clip: the catch, the swing, the release on release_on, the landing
        pins = {"release": r}
        if c.has("land_on"):
            pins["land"] = c.time("land_on", ref=r)
        if c.has("catch_on"):
            pins["catch"] = c.time("catch_on", ref=r)
        if c.has("catch_x") and not c.has("x_keys"):
            c.args["x_keys"] = {"catch": c.arg("catch_x")}
        _video_args(c, seq, pins, o, "release", f"swing {o.id}", "land" if "land" in seq.get("keys", {}) else None)
        return
    s = c.ch.step_at(r)
    land = c.time("land_on", default="kick", ref=r)
    c.st.q0, c.st.q1 = r, land
    u, f = c.u, c.feet
    xr = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(260 * u)
    if xr is None:
        return
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else xr + 730 * u
    rel = key(seq, "release")
    c.wall(r, rel, xr, by + 53 * u, anchor="right", keeper=True, tag="release")
    a = tors(c, rel, xr, by + 53 * u, "right")
    lp = key(seq, "land")
    b = tors(c, lp, lx, f, "foot")
    n = int(round((land - r) / s)) - 1
    for j, ((x, y, uu), name) in enumerate(zip(arc(n, a, b, 50 * u), pick(between(seq, "release", "land"), n))):
        c.wall(r + (j + 1) * s, name, x, y, anchor="cxc", rot=-8 + 12 * uu, tag="release.flight")
    c.wall(land, lp, lx, keeper=True, tag="release.land")
    c.stamp(land, c.pose("droplets"), lx + 20 * u, f - 86 * u, 380 * u, tag="release.land")
    c.stamp(land, c.pose("splash"), lx - 5 * u, f - 76 * u, 300 * u, tag="release.land")
    pts = arc(n, a, b, 50 * u)
    rdx = recover_dx(c, b[0] - (pts[-1][0] if pts else a[0]))
    for j, name in enumerate(between(seq, "land", None)):
        c.wall(land + (j + 1) * s, name, lx + (j + 1) * rdx, tag="release.recover")
    c.label(r, xr, by, "release")
    c.st.exit_gap = 40.0 * u


@m0("pole_swing")
def pole_swing(c: Ctx) -> None:
    """Swing round a pipe (M0): its sequence (default pole_PK3: reach, grip, release, flight, landing, recovery) in
    order; the reach on the photo before `grab_on`, the release on the photo after; the flight an eased arc; land on
    `land_on` (default six photos after the grab). span scales the flight (1.0). A video sequence plays the clip
    round the pipe (video_move: his x against the pole as in the clip), its `grab` key on grab_on and/or its `land`
    key on land_on; stretch: [release, land, factor] flies further."""
    o = c.ob("pipe", ("pipe", "pole"))
    px = o.contact("pole").x
    _, seq = sequence(c, "pole_swing")
    if "video" in seq:  # from a clip: the run-in, the grab, the swing round the pipe, the landing
        pins = {}
        if c.has("grab_on"):
            pins["grab"] = c.time("grab_on")
        if c.has("land_on"):
            pins["land"] = c.time("land_on", ref=pins.get("grab"))
        if not pins:
            raise ChoreoError("needs grab_on: or land_on:")
        _video_args(c, seq, pins, o, "pole_swing", f"pole swing {o.id}",
                    "land" if "land" in seq.get("keys", {}) else None)
        return
    g = c.time("grab_on")
    s = c.ch.step_at(g)
    land = c.time("land_on", ref=g) if c.has("land_on") else g + 6 * s
    f, u = c.feet, c.u
    sp = float(c.arg("span", 1.0))
    reach, grip, rel = key(seq, "reach"), key(seq, "grip"), key(seq, "release")
    gt = tors(c, grip, px + 4 * u, f - 461 * u, "top")
    # the reach a step (at most 150 px) before the grip by the torso: the hands reach round the pipe
    dxr0 = c.ch.lib.torso_offset(reach, "right", float(c.args.get("scale", 1.0)))[0]
    rtx = max(px - 16 * u + dxr0, gt[0] - 130 * u)
    c.wall(g - s, reach, rtx - dxr0, f - 436 * u, anchor="right", tag="pole_swing")
    c.wall(g, grip, px + 4 * u, f - 461 * u, anchor="top", rot=15, keeper=True, tag="pole_swing.grip",
           contact=(o.id, "pole", "top"))
    rt = tors(c, reach, rtx - dxr0, f - 436 * u, "right")
    dxr = c.ch.lib.torso_offset(rel, "right", float(c.args.get("scale", 1.0)))[0]
    rtx = float(np.clip(px + 279 * sp * u + dxr, gt[0] + 90 * u, gt[0] + 150 * u))
    xr = rtx - dxr
    c.wall(g + s, rel, xr, f - 476 * u, anchor="right", tag="pole_swing")
    a = tors(c, rel, xr, f - 476 * u, "right")
    lp = key(seq, "land")
    lx = max(px + 809 * sp * u, a[0] + 200 * u)
    b = tors(c, lp, lx, f, "foot")
    n = int(round((land - g) / s)) - 2
    for j, ((x, y, uu), name) in enumerate(zip(arc(n, a, b, 60 * u), pick(between(seq, "release", "land"), n))):
        c.wall(g + (j + 2) * s, name, x, y, anchor="cxc", rot=-6 + 9 * uu, keeper=(j == n - 2),
               tag="pole_swing.flight")
    c.wall(land, lp, lx, keeper=True, tag="pole_swing.land")
    pts = arc(n, a, b, 60 * u)
    rdx = recover_dx(c, b[0] - (pts[-1][0] if pts else a[0]))
    for j, name in enumerate(between(seq, "land", None)):
        c.wall(land + (j + 1) * s, name, lx + (j + 1) * rdx, tag="pole_swing")
    c.label(g, px, f - 461 * u, f"pole swing {o.id}")
    c.st.pre_x = rt[0] - 110 * u
    c.st.entry_gap = 105.0 * u
    c.st.exit_gap = 85.0 * u


def _onto(c: Ctx, template: str, tag: str) -> None:
    """step_up and leap_off (M0): jump onto a step or top on `at`, land on it, push off from its far end, fly an
    eased arc (the first flight photo near the top of it), land on `land_on`, recover."""
    o = c.ob("onto")
    top = o.contact("step", "top")
    sy = top.y
    tx0 = top.x0 if top.x0 is not None else o.x0
    tx1 = top.x1 if top.x1 is not None else o.x1
    _, seq = sequence(c, template)
    a = c.time("at")
    if "video" in seq:  # from a clip: its `key` (default on: his foot on the top) on `at`, landing on land_on
        pins = {str(c.arg("key", "on")): a}
        if c.has("land_on"):
            pins["land"] = c.time("land_on", ref=a)
        _video_args(c, seq, pins, o, tag, f"{tag.replace('_', ' ')} {o.id}",
                    "land" if tag == "leap_off" and "land" in seq.get("keys", {}) else None)
        return
    s = c.ch.step_at(a)
    pool = between(seq, "push", "land")
    land = c.time("land_on", ref=a) if c.has("land_on") else a + (3 + len(pool)) * s
    if land < a + 4 * s - 1e-6:
        raise ChoreoError("land_on: needs at least four photos after at")
    f, u = c.feet, c.u
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else tx1 + (560 if tag == "leap_off" else 359) * u
    jx = c.coord(c.arg("jump_x"), "x") if c.has("jump_x") else tx0 - 76 * u
    peak = float(c.arg("height", 230 if tag == "leap_off" else 150)) * u
    jp, on, push, lp = key(seq, "jump"), key(seq, "on"), key(seq, "push"), key(seq, "land")
    on_x = max((tx0 + tx1) / 2, tx1 - 110 * u, jx + 90 * u)  # on the step, a step before its far end
    jx = max(jx, on_x - 140 * u)  # the jump a step before the top, whatever jump_x says (even steps)
    c.wall(a + s, on, on_x, sy, keeper=True, tag=f"{tag}.on", contact=(o.id, top.name, "foot"))
    px_ = max(tx1 - 6 * u, on_x + 80 * u)
    c.wall(a, jp, jx, f - 20 * u, tag=tag)
    c.wall(a + 2 * s, push, px_, sy - 4 * u, tag=f"{tag}.push")
    pa = tors(c, push, px_, sy - 4 * u, "foot")
    n = int(round((land - a) / s)) - 3
    if not c.has("land_x") and tag == "leap_off":  # far enough to fly, each photo a step of about 150 px
        lx = max(tx1 + 160 * u, pa[0] + (n + 1) * 150 * u)
    b = tors(c, lp, lx, f, "foot")
    for j, ((x, y, uu), name) in enumerate(zip(arc(n, pa, b, peak), pick(pool, n))):
        c.wall(a + (3 + j) * s, name, x, y, anchor="cxc", rot=6 - 10 * uu, keeper=(j == 0), tag=f"{tag}.flight")
    c.wall(land, lp, lx, keeper=True, flags={"halo": True} if tag == "leap_off" else {}, tag=f"{tag}.land")
    if tag == "leap_off":
        c.stamp(land, c.pose("splash"), lx - 110 * u, f - 130 * u, 600 * u, flags={"splash": True, "primer": True},
                tag="leap_off.land")
        c.stamp(land, c.pose("droplets"), lx + 90 * u, f - 280 * u, 480 * u, tag="leap_off.land")
    pts = arc(n, pa, b, peak)
    rdx = recover_dx(c, b[0] - (pts[-1][0] if pts else pa[0]))
    for j, name in enumerate(between(seq, "land", None)):
        c.wall(land + (j + 1) * s, name, lx + (j + 1) * rdx, tag=f"{tag}.recover")
    c.label(a + s, on_x, sy, f"{tag.replace('_', ' ')} {o.id}")
    if tag == "leap_off" and n:
        c.label(a + 3 * s, pa[0], sy - peak, "highest point")
    c.st.pre_x = jx - 85 * u
    c.st.entry_gap = 70.0 * u
    c.st.exit_gap = 110.0 * u


@m0("step_up")
def step_up(c: Ctx) -> None:
    """Leap onto a step (M0): at, land_on (default as many photos as the sequence needs), land_x, jump_x."""
    _onto(c, "step_up", "step_up")


@m0("leap_off")
def leap_off(c: Ctx) -> None:
    """Jump onto a top, push off, fly, land (M0): onto, at, land_on, land_x, jump_x, height (230). A video sequence
    plays its clip over the obstacle (video_move), its `key` (default on) on `at`, its land key on land_on (default
    the clip's own pace, `speed`), landing with a splash."""
    _onto(c, "leap_off", "leap_off")


# ---------------------------------------------------------------------------------------------- slide, climb, brake

@m0("slide")
def slide(c: Ctx) -> None:
    """Slide under an obstacle (M0): drop in on `at`, glide (the slide drawing moving, slowing a little), rise on
    `up` (default six photos later), then run."""
    o = c.ob("under")
    o.contact("under")
    _, seq = sequence(c, "slide")
    q_in = c.time("at")
    s = c.ch.step_at(q_in)
    q_up = c.time("up", ref=q_in) if c.has("up") else q_in + 6 * s
    n = int(round((q_up - q_in) / s))
    if n < 3:
        raise ChoreoError("up: needs at least three photos after at")
    w, u = o.x1 - o.x0, c.u
    if w >= n * 100 * u:
        xi, xr = o.x0 + 17 * u, o.x1
    else:
        span = n * 110 * u
        cen = o.cx + 0.1 * span
        xi, xr = cen - span / 2, cen + span / 2
    wts = np.array([1.15 - 0.3 * k / max(n - 1, 1) for k in range(n)])
    xs = xi + np.concatenate([[0.0], np.cumsum(wts)]) / wts.sum() * (xr - xi)
    c.wall(q_in, key(seq, "in"), xs[0], keeper=True, tag="slide")
    glide = key(seq, "glide")
    for k in range(1, n):
        v = _vary(c, k - 1, k, mirror=False, rot=1.5, off=2.0 * u)
        c.wall(q_in + k * s, glide, xs[k] + v.get("ox", 0), rot=v.get("rot", 0.0), keeper=(k == 2 and n > 4),
               tag="slide.under", contact=(o.id, "under", "top"))
    c.wall(q_up, key(seq, "up"), xr + 30 * u, keeper=True, tag="slide.rise")
    c.label(q_in, xi, c.feet, f"slide {o.id}")
    c.st.entry_gap = 110.0 * u
    c.st.exit_gap = 90.0 * u


@m0("climb")
def climb(c: Ctx) -> None:
    """Climb a pipe (M0): the reach on the photo before `grab`, then hand over hand (the sequence's drawings
    between grip and lunge, alternating) every `every` (8ths), lunging to a beam on `lunge` (with `to`). The run
    before ends at the take-off. pipe, rise, hands_y, until as before."""
    o = c.ob("pipe", ("pipe", "pole", "ladder"))
    px = o.contact("pole").x
    _, seq = sequence(c, "climb")
    g = c.time("grab")
    s = c.ch.step_at(g)
    every = rate_of(c.arg("every", "8ths"))
    u = c.u
    rise = float(c.arg("rise", 48.3 * u))
    hy = c.coord(c.arg("hands_y"), "y") if c.has("hands_y") else c.feet - 556 * u
    reach = key(seq, "reach")
    c.wall(g - s, reach, px - 10 * u, hy + 60 * u, anchor="right", tag="climb")
    rt = tors(c, reach, px - 10 * u, hy + 60 * u, "right")
    beam = c.ob("to", ("beam", "ledge", "ladder")) if c.has("to") else None
    if beam is not None:
        lunge = c.time("lunge", default="snare", ref=g + 4 * every - 0.5)
    else:
        lunge = c.time("until", ref=g) if c.has("until") else g + 4 * every
    cl = [key(seq, "grip")] + between(seq, "grip", "lunge")
    for k, q in enumerate(np.arange(g, lunge - 1e-6, every)):
        name = cl[k % len(cl)]
        v = _vary(c, k // len(cl), k, mirror=False, rot=3.0, off=5.0 * u)
        c.wall(q, name, px + 4 * u + v.get("ox", 0), hy - k * rise + v.get("oy", 0), anchor="top",
               rot=v.get("rot", 0.0), keeper=(k % 2 == 0), tag="climb.grip", contact=(o.id, "pole", "top"))
    if beam is not None:
        bc = beam.contact("beam", "bar")
        c.wall(lunge, key(seq, "lunge"), px + 188 * u, bc.y + 15 * u, anchor="right", keeper=True,
               tag="climb.lunge", contact=(beam.id, bc.name, "right"))
    c.label(g, px, hy, f"climb {o.id}")
    c.st.pre_x = rt[0] - 120 * u
    c.st.entry_gap = 122.0 * u


@m0("brake")
def brake(c: Ctx) -> None:
    """Brake hard (M0): the brake drawing once on `at` (default the next kick), held (no second drawing slid
    along). x: default 80 px after the previous move."""
    q = c.time("at", default="kick")
    hold = float(c.arg("hold", 2))
    c.st.q0, c.st.q1 = q, q + hold
    x = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(80 * c.u)
    if x is None:
        return
    c.wall(q, c.pose("brake"), x, keeper=True, tag="brake")
    c.label(q, x, c.feet, "brake")
    c.st.entry_gap = 80.0


@m0("crouch")
def crouch(c: Ctx) -> None:
    """Stop at an edge and crouch (M0): as before (brake-in, crouch), the drawings from the crouch sequence."""
    _crouch_legacy(c)


# ---------------------------------------------------------------------------------------------- the box

@m0("burst")
def burst(c: Ctx) -> None:
    """Break out of the box (M0): as the legacy burst (the shatter, the arc), but the flight plays the burst
    sequence's drawings in order (out, then the flight drawings) on an eased arc, and lands with its landing
    drawing. A video sequence (keys out, apex, land) plays its clip instead (_burst_video)."""
    from .props import Shatter
    from .moves import _box

    pr = _box(c)
    _, seq = sequence(c, "burst")
    at = c.time("at")
    s = c.ch.step_at(at)
    if "video" in seq:
        return _burst_video(c, pr, seq, at, s)
    land = c.time("land_on", default="impact", ref=at)
    if land <= at:
        raise ChoreoError(f"land_on {c.ch.grid.addr(c.song(land))} is not after at")
    key_ = pr.geometry(c.song(at) - 0.01)
    u, f = c.u, c.feet
    bx0, by0, bx1, by1 = key_.box
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else bx1 + 760 * u
    y_a = (by0 + by1) / 2
    top = min(f - float(c.arg("height", 300)) * u, y_a - 40 * u)
    out = key(seq, "out")
    n = int(round((land - at) / s))
    names = [out] + pick(between(seq, "out", "land"), n - 1)
    lp = key(seq, "land")
    a = (key_.side_x(y_a, "right") - 0.2 * key_.w, y_a)
    b = tors(c, lp, lx, f, "foot")
    pts = [a] + [(x, y) for x, y, _ in arc(n - 1, a, b, max(0.0, (a[1] + b[1]) / 2 - top))]
    for k, (q, name, (x, y)) in enumerate(zip([at + j * s for j in range(n)], names, pts)):
        c.wall(q, name, x, y, anchor="cxc", rot=-12 + 14 * k / max(n - 1, 1), keeper=(k in (0, n - 1)),
               tag="burst.flight")
    c.stamp(at, c.pose("spray"), bx1, y_a, 380 * u, flags={"splash": True}, tag="burst.spray")
    c.wall(land, lp, lx, keeper=True, flags={"halo": True}, tag="burst.land")
    c.stamp(land, c.pose("splash"), lx - 110 * u, f - 120 * u, 560 * u, flags={"splash": True, "primer": True},
            tag="burst.land")
    c.stamp(land, c.pose("drips"), lx - 120 * u, f - 200 * u, 320 * u, tag="burst.land")
    if not c.dry:
        sy = y_a - 0.08 * key_.h
        pr.shatter = Shatter(c.song(at), key_.side_x(sy, "right") + key_.thick / 2, sy, 1.0,
                             int(np.random.default_rng([c.mv.seed, c.mv.local_i, 3]).integers(0, 2 ** 31)),
                             float(c.arg("fly", 0.65)), float(c.arg("power", 1.0)), int(c.arg("splats", 0)),
                             float(c.arg("stencils", 0.0)))
    c.cam(at, punch=float(c.arg("punch", 0.1)), shake=float(c.arg("shake", 0.0)))
    if c.has("flicker"):  # the explosion knocks a light out for a photo (the red pedestrian light)
        c.light(at, str(c.arg("flicker")), 0.05)
        c.light(at + s, str(c.arg("flicker")), 0.5)
    c.label(at, bx1, by0, "shatter")
    c.label(land, lx, f, "landing")
    c.st.exit_gap = 110.0 * u


def _burst_video(c: Ctx, pr, seq: dict, at: float, s: float) -> None:
    """The burst from a clip (video_move): its `out` key (he breaks through the box's right side) on `at`, its
    `apex` key on `apex_on` (default the last photo before the song's next stop, so the stop holds him mid-air with
    the shards; without a stop the clip's own pace), the `land` key on `land_on` (default the clip's pace after the
    apex, `speed`); out at the box's right side, landing at land_x (the travel in between stretched to it, stride-
    true after; apex_x places the top of it too: clear of the red light), with a splash. The shatter and the camera
    punch as in the drawn burst; power (1: more and faster flecks), splats (paint thrown across the wall), stencils
    (the share of the letters that break off and fly) and shake (px) make the explosion bigger; flicker: a light
    knocked out for a photo. frame: box plays a front-view clip in the box instead (_burst_boxed)."""
    from .props import Shatter

    u, f = c.u, c.feet
    key_ = pr.geometry(c.song(at) - 0.01)
    bx0, by0, bx1, by1 = key_.box
    y_a = (by0 + by1) / 2
    if c.arg("frame") == "box":
        _burst_boxed(c, pr, seq, at, s, key_)
    else:
        _burst_side(c, seq, at, key_)
    c.stamp(at, c.pose("spray"), bx1, y_a, 380 * u, flags={"splash": True}, tag="burst.spray")
    if not c.dry:
        sy = y_a - 0.08 * key_.h
        pr.shatter = Shatter(c.song(at), key_.side_x(sy, "right") + key_.thick / 2, sy, 1.0,
                             int(np.random.default_rng([c.mv.seed, c.mv.local_i, 3]).integers(0, 2 ** 31)),
                             float(c.arg("fly", 0.65)), float(c.arg("power", 1.0)), int(c.arg("splats", 0)),
                             float(c.arg("stencils", 0.0)))
    c.cam(at, punch=float(c.arg("punch", 0.1)), shake=float(c.arg("shake", 0.0)))
    if c.has("flicker"):  # the explosion knocks a light out for a photo (the red pedestrian light)
        c.light(at, str(c.arg("flicker")), 0.05)
        c.light(at + s, str(c.arg("flicker")), 0.5)
    c.label(at, bx1, by0, "shatter")


def _burst_pins(c: Ctx, seq: dict, at: float, out: str) -> dict:
    """The burst's pins: `out` on at, apex (apex_on, else the last photo before the song's next stop), land
    (land_on), and with `from` an earlier key (windup_on: the clip's `windup`)."""
    pins = {out: at}
    if c.has("apex_on"):
        pins["apex"] = c.time("apex_on", ref=at)
    elif "apex" in seq.get("keys", {}):
        st = next(((a, b) for a, b in c.clock.stops if a > c.song(at) + 1e-6), None)
        if st is not None:
            pins["apex"] = c.clock.fwd(st[0]) - c.ch.step_at(st[0] - 1)
    if c.has("land_on"):
        pins["land"] = c.time("land_on", ref=at)
    if c.has("windup_on") and "windup" in seq.get("keys", {}):
        pins["windup"] = c.time("windup_on", ref=at)
    return pins


def _burst_boxed(c: Ctx, pr, seq: dict, at: float, s: float, key_) -> None:
    """The burst from a front-view clip in the box (frame: box; box-breakout): every frame put where the clip's
    guide box puts it on the street box (as in the box: one scale, his place in the box), its `through` key (the fist
    or shoulder through the right side) on `at`, the wind-up before it from `from` (windup_on), apex and land as the
    side-view burst. After `through` he leaves the box's frame: apex_x and land_x (torso x) move him there (never a
    step back), and his low point on the landing is put on the ground line (the clip's box floats above its own floor
    line), eased in from the break-through. He grows to the runner's size (measured against `run_sequence`, run_SW)
    over `grow` photos (4) after the landing, or with `grow_from: <key>` from that key to the landing, so the run
    after him takes over at full size.

    leap: {sequence, keys: {key: time}, land: time} plays a second clip from the photo after `at` (box-leap: the
    dive out of the box, its guide box anywhere in its frame): the first clip up to the break-through, the second
    from there, both put on the street box the same way, its keys pinned (its `land` key on `land`, default
    land_on), so the flight is slowed through a song stop instead of held (the burst with pause: false); its `x:
    {key: torso x}` places keys (past the red light's pole quickly, then the slow part where nothing crosses him);
    its `height` (px) lifts the flight on a parabola from the break-through to the landing (a high arc)."""
    from .moves import _in_box_place

    u = c.u
    out = "through" if "through" in seq.get("keys", {}) else "out"
    pins = _burst_pins(c, seq, at, out)
    q0 = c.time("from", ref=at) if c.has("from") else at
    q1 = c.time("to", ref=at) if c.has("to") else None
    lp = dict(c.arg("leap")) if isinstance(c.arg("leap"), dict) else None
    lib = c.ch.lib
    place = []  # [q, track row, x, y, scale, sequence]
    if lp:
        pins.pop("apex", None)
        land = pins.pop("land", None)
        shots = video_keyed(c, seq, sorted((q, key_frame(seq, k)) for k, q in pins.items()), q0, at,
                            float(c.arg("speed", seq.get("speed", 1.0))))
        try:
            seq2 = lib.sequence(str(lp["sequence"]))
        except KeyError as e:
            raise ChoreoError(str(e.args[0])) from e
        pins2 = {str(k): c.at(v, ref=at, key=f"leap.keys.{k}") for k, v in (lp.get("keys") or {}).items()}
        if "land" in seq2.get("keys", {}):
            pins2["land"] = c.at(lp["land"], ref=at, key="leap.land") if "land" in lp else land
        shots2 = video_keyed(c, seq2, sorted((q, key_frame(seq2, k)) for k, q in pins2.items() if q is not None),
                             at + s, q1, float(lp.get("speed", 1.0)), alternate=True)
        for q, r in shots:
            place.append([q, r, *_in_box_place(c, pr, seq, r, key_), seq])
        for q, r in shots2:
            place.append([q, r, *_in_box_place(c, pr, seq2, r, key_), seq2])
        pins = {**pins, **{k: v for k, v in pins2.items() if v is not None}}
    else:
        seq2 = seq
        shots = video_keyed(c, seq, sorted((q, key_frame(seq, k)) for k, q in pins.items()), q0, q1,
                            float(c.arg("speed", seq.get("speed", 1.0))))
        for q, r in shots:
            place.append([q, r, *_in_box_place(c, pr, seq, r, key_), seq])
    if not place:
        raise ChoreoError("the clip has no frames on these photos")
    n = len(place)
    k_of = {k: min(range(n), key=lambda i: abs(place[i][0] - q)) for k, q in pins.items()}
    i_out = k_of[out]
    # offsets after the break-through: x to apex_x / land_x, the landing's low point onto the ground line
    dx_pts, dy_pts = [(i_out, 0.0)], [(i_out, 0.0)]
    for k, arg in (("apex", "apex_x"), ("land", "land_x")):
        if k in k_of and c.has(arg):
            dx_pts.append((k_of[k], c.coord(c.arg(arg), "x") - place[k_of[k]][2]))
    for k, xv in ((lp or {}).get("x") or {}).items():  # the leap's own torso x pins (past a pole quickly, then slow)
        if str(k) in k_of:
            dx_pts = [p_ for p_ in dx_pts if p_[0] != k_of[str(k)]] + [(k_of[str(k)], c.coord(xv, "x") -
                                                                         place[k_of[str(k)]][2])]
    if "land" in k_of:
        i = k_of["land"]
        q, r, x, y, sc, sq = place[i]
        m = lib.m(r["name"])
        half = (m["bbox"][3] - m["bbox"][1]) / 2 * m["scale"] * sc
        land_x = x + (dx_pts[-1][1] if dx_pts[-1][0] == i else 0.0)
        dy_pts.append((i, (c.ch.scene.feet_at(land_x) + 4 * u) - (y + half)))
    dx_pts.sort()
    dy_pts.sort()

    def off(pts, i):
        if i <= pts[0][0]:
            return 0.0 if i < pts[0][0] else pts[0][1]
        for (ia, va), (ib, vb) in zip(pts, pts[1:]):
            if ia <= i <= ib:
                return va + (vb - va) * _ramp((i - ia) / max(ib - ia, 1))
        return pts[-1][1]

    grow = int(c.arg("grow", 4))
    ratio = None
    if "land" in k_of:  # the runner's size against his size in the box (from the clip's own running frames)
        vid2 = lib.videos[seq2["video"]]
        f_run = key_frame(seq2, "sprint") if "sprint" in seq2.get("keys", {}) else place[k_of["land"]][1]["f"]
        run_h = [r_["h"] for r_ in vid2["frames"] if r_["f"] >= f_run]
        rs = str(c.arg("run_sequence", "run_SW"))
        if rs in lib.sequences and run_h:
            bb = vid2["block"]
            s_box = (key_.box[3] - key_.box[1]) / max(bb[3] - bb[1], 1)  # street px per clip px in the box
            hs = [(lib.m(nm)["bbox"][3] - lib.m(nm)["bbox"][1]) * lib.m(nm)["scale"]
                  for nm in lib.sequence(rs)["drawings"]]
            ratio = float(np.median(hs)) / max(float(np.median(run_h)) * s_box, 1.0)
    g0 = k_of.get(str(c.arg("grow_from"))) if c.has("grow_from") else None
    lift = float((lp or {}).get("height", 0.0)) * u  # the leap's arc lifted (a parabola from out to the landing)
    for i, (q, r, x, y, sc, sq) in enumerate(place):  # the clip's place, the pins, the lift and the growth
        x += off(dx_pts, i)
        y += off(dy_pts, i)
        if lift and "land" in k_of and i_out < i < k_of["land"]:
            v = (i - i_out) / max(k_of["land"] - i_out, 1)
            y -= lift * 4 * v * (1 - v)
        m = lib.m(r["name"])
        if ratio and "land" in k_of:
            il = k_of["land"]
            if g0 is not None:
                j = 0.0 if i <= g0 else min(1.0, (i - g0) / max(il - g0, 1))
            else:
                j = min(i - il, grow) / grow if i > il else 0.0
            g_ = 1 + (ratio - 1) * _ramp(j)
            half = (m["bbox"][3] - m["bbox"][1]) / 2 * m["scale"] * sc
            y -= (g_ - 1) * half  # grown about his feet
            sc *= g_
        place[i][2], place[i][3], place[i][4] = x, y, sc
    rots = _leap_path(c, lp, place, k_of, at) if lp and (lp.get("path") or lp.get("tilt")) else [0.0] * n
    x_last = -math.inf
    for i, (q, r, x, y, sc, sq) in enumerate(place):
        m = lib.m(r["name"])
        if "land" in k_of and i >= k_of["land"]:  # running on: never below the ground line
            low = y + (m["bbox"][3] - m["bbox"][1]) / 2 * m["scale"] * sc
            y -= max(0.0, low - (c.ch.scene.feet_at(x) + 4 * u))
        if i > i_out:  # out of the box: never a step back
            x = max(x, x_last)
        x_last = x
        kn = next((k for k, ii in k_of.items() if ii == i and abs(place[ii][0] - pins[k]) < 1e-6), None)
        c.wall(q, r["name"], x, y, anchor="cxc", scale=sc, rot=rots[i], keeper=kn is not None,
               tag=f"burst.{kn}" if kn else ("burst.box" if i < i_out else "burst"))
        place[i][2], place[i][3] = x, y
    if "land" in k_of:
        q, r, x, y, sc, sq = place[k_of["land"]]
        f = c.ch.scene.feet_at(x)
        k_s = float(c.arg("splash", 1.0))
        c.stamp(q, c.pose("splash"), x - 110 * u * k_s, f - 130 * u, 560 * u * k_s,
                flags={"splash": True, "primer": True}, tag="burst.land")
        c.stamp(q, c.pose("droplets"), x + 90 * u * k_s, f - 280 * u, 460 * u * k_s, tag="burst.land")
        c.label(q, x, f, "landing")
    c.st.entry_gap = 60.0 * u
    c.st.exit_gap = float(np.clip(place[-1][2] - place[-2][2], 60 * u, 130 * u)) if n > 1 else 90.0 * u


def _pchip(xk, yk, x) -> tuple[np.ndarray, np.ndarray]:
    """Monotone cubic (Fritsch-Carlson) through (xk, yk) at x: (values, slopes). It never overshoots its points: a
    curve through a rise, a top and a fall keeps its top where the points put it."""
    xk, yk, x = np.asarray(xk, float), np.asarray(yk, float), np.asarray(x, float)
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros(len(xk))
    if len(xk) == 2:
        m[:] = d[0]
    else:
        for k in range(1, len(xk) - 1):
            if d[k - 1] * d[k] > 0:
                w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        for k, (a, b, ha, hb) in ((0, (d[0], d[1], h[0], h[1])), (-1, (d[-1], d[-2], h[-1], h[-2]))):
            e = ((2 * ha + hb) * a - ha * b) / (ha + hb)  # the end slope (three points), kept monotone
            m[k] = 0.0 if e * a <= 0 else (3 * a if a * b < 0 and abs(e) > abs(3 * a) else e)
    i = np.clip(np.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    t = (x - xk[i]) / h[i]
    y = ((2 * t ** 3 - 3 * t ** 2 + 1) * yk[i] + (t ** 3 - 2 * t ** 2 + t) * h[i] * m[i] +
         (-2 * t ** 3 + 3 * t ** 2) * yk[i + 1] + (t ** 3 - t ** 2) * h[i] * m[i + 1])
    dy = ((6 * t ** 2 - 6 * t) * yk[i] / h[i] + (3 * t ** 2 - 4 * t + 1) * m[i] +
          (-6 * t ** 2 + 6 * t) * yk[i + 1] / h[i] + (3 * t ** 2 - 2 * t) * m[i + 1])
    return y, dy


def _slant(lib, name: str) -> float:
    """A drawing's own slant in degrees (its long axis, nose up positive): the launch drawings already lean up, the
    dive drawings lie level. 0 for a drawing that is not clearly long (a tuck) or stands more than 60 degrees from
    level (upright: no nose to tilt)."""
    d = lib.sprite(name)
    ys, xs = np.nonzero(d > 0.3)
    if len(xs) < 20:
        return 0.0
    cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    w, v = np.linalg.eigh(cov)
    vx, vy = v[:, int(np.argmax(w))]
    if vx < 0:
        vx, vy = -vx, -vy
    a = float(math.degrees(math.atan2(-vy, vx)))
    return a if w.max() > 2.25 * max(w.min(), 1e-6) and abs(a) <= 60 else 0.0


def _leap_path(c: Ctx, lp: dict, place: list, k_of: dict, at: float) -> list[float]:
    """The burst leap's `path` and `tilt` (frame: box). path: [[key or time, x, y], ...]: torso places (x) and body
    centres (y) at those photos (a key of either clip, or a time; x or y null: where the clip and the pins put him),
    one smooth monotone curve in time through them replacing his course from the first to the last (up from the
    burst, over the red light, the dive down). tilt: {to, max, gain, ease}: every drawing turned along the course,
    nose up rising, level at the top, nose down diving (gain times the course's angle, less the drawing's own slant,
    at most `max` degrees, 60), fading in over `ease` photos (2) from the first path point and let go from the `to`
    key (default dive_end) to the landing, so the tuck lands upright. Moves place in place; returns the rotations."""
    n = len(place)
    qs = np.array([p[0] for p in place], float)
    pts: dict[int, tuple] = {}
    for j, p in enumerate(lp.get("path") or []):
        if not isinstance(p, list | tuple) or len(p) != 3:
            raise ChoreoError(f"leap.path[{j + 1}]: write it as [key or time, x, y]")
        a, xv, yv = p
        i = k_of[str(a)] if str(a) in k_of else int(np.argmin(np.abs(qs - c.at(a, ref=at, key=f"leap.path[{j + 1}]"))))
        pts[i] = (place[i][2] if xv is None else c.coord(xv, "x"), place[i][3] if yv is None else c.coord(yv, "y"))
    ii = sorted(pts)
    if len(ii) >= 2:
        span = range(ii[0], ii[-1] + 1)
        qk = [qs[i] for i in ii]
        xs_, _ = _pchip(qk, [pts[i][0] for i in ii], qs[ii[0]:ii[-1] + 1])
        ys_, _ = _pchip(qk, [pts[i][1] for i in ii], qs[ii[0]:ii[-1] + 1])
        for k, i in enumerate(span):
            place[i][2], place[i][3] = float(xs_[k]), float(ys_[k])
    rots = [0.0] * n
    tl = lp.get("tilt")
    if not tl:
        return rots
    tl = {} if tl is True else dict(tl)
    lib = c.ch.lib
    i0 = ii[0] if ii else k_of.get("out", 0)
    i_land = k_of.get("land", ii[-1] if ii else n - 1)
    to = str(tl.get("to", "dive_end"))
    i_to = min(k_of.get(to, i_land - 2), i_land - 1)  # upright again on the landing
    ease = max(int(tl.get("ease", 2)), 1)
    most, gain = float(tl.get("max", 60.0)), float(tl.get("gain", 1.0))
    for i in range(i0, min(i_land + 1, n)):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        ang = math.degrees(math.atan2(-(place[b][3] - place[a][3]), place[b][2] - place[a][2] + 1e-6))
        w = min(1.0, (i - i0) / ease)
        if i > i_to:
            w *= max(0.0, (i_land - i) / max(i_land - i_to, 1))
        rots[i] = w * float(np.clip(gain * ang - _slant(lib, place[i][1]["name"]), -most, most))
    return rots


def _burst_side(c: Ctx, seq: dict, at: float, key_) -> None:
    """The burst from a side-view clip (burst_BU): video_move with its keys pinned, out at the box's right side."""
    bx0, by0, bx1, by1 = key_.box
    y_a = (by0 + by1) / 2
    pins = _burst_pins(c, seq, at, "out")
    pins.pop("windup", None)
    xk = {"out": key_.side_x(y_a, "right") - 0.2 * key_.w}
    if c.has("land_x"):
        xk["land"] = c.coord(c.arg("land_x"), "x")
    if c.has("apex_x") and "apex" in pins:
        xk["apex"] = c.coord(c.arg("apex_x"), "x")
    q1 = c.time("to", ref=at) if c.has("to") else None
    video_move(c, seq, pins, None, float(c.arg("speed")) if c.has("speed") else None, "burst", xk, None,
               None, at, q1, None, "land" if "land" in seq.get("keys", {}) else None)
