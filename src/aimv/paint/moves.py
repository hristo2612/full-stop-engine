"""Move templates: each places its poses from an obstacle and a hit (prototype 6's hand placements, made relative to
the obstacles' contacts so they work on any street).

A template is a function of a Ctx (choreo.py), registered with @template("name"). It reads its arguments with
c.time() (a grid time), c.ob() (an obstacle), c.arg(), places poses with c.wall() / c.air() / c.stamp(), and can
freeze the paint clock (c.freeze), ask for camera framing (c.cam) or an extra photo off the grid (c.photo). Poses are
named by role (c.pose("takeoff")), which the project maps to its drawings in poses.spec.json; a move can override
any role with `poses: {role: name}`. Times inside a template step by c.step (the segment's photo step: one 16th in a
run, an 8th in calmer parts), so a template keeps its rhythm on any photo rate.

To add a template (box escape, slip down a wall, peel off the wall, float ...): write a function here, decorate it,
read its arguments, place poses relative to the obstacle's contacts, set c.st.entry_gap if the previous run should
stop short of its first pose, and document it in docs/PAINT.md. Street-px constants are written for a
450 px figure (prototype 6) and multiplied by c.u (the library's ref_height / 450), so templates fit any figure size.
"""

from __future__ import annotations

import math

import numpy as np

from .choreo import ChoreoError, Ctx, lerp
from .grid import rate_of

TEMPLATES: dict = {}
RUN_SPEED = 88.0  # px per 16th for a 450 px figure: prototype 6's runs (83 to 90)


def template(name: str, prop: bool = False):
    """Register a template. prop=True: it paints the set (the box), is placed before the runner's moves and does not
    take part in their chain (no time or street extent of its own)."""
    def deco(fn):
        TEMPLATES[name] = fn
        fn.template = name
        fn.prop = prop
        return fn
    return deco


def _vary(c: Ctx, k: int, seed: int, mirror: bool = True, rot: float = 4.0, off: float = 7.0) -> dict:
    """Small per-photo variation for a repeated drawing (k: how many times it was used before in this move)."""
    if not c.ch.vary or k == 0 or not c.arg("vary", True):
        return {}
    rng = np.random.default_rng([c.mv.seed, c.mv.local_i, seed, k])  # its own file's seed and numbering
    out = {"rot": float(rng.uniform(-rot, rot)), "ox": float(rng.uniform(-off, off)),
           "oy": float(rng.uniform(-off, off) * 0.6)}
    if mirror and k % 2 == 1:
        out["mirror"] = True
    return out


def _start_after_prev(c: Ctx) -> float:
    if c.prev is None or c.prev.q1 is None:
        raise ChoreoError("needs from: (it is the first move)")
    return c.prev.q1 + c.ch.step_at(c.prev.q1)


# ---------------------------------------------------------------------------------------------- locomotion

def _run_cycle(c: Ctx, cyc_name: str = "run") -> list[list[str]]:
    over = (c.args.get("poses") or {}).get(cyc_name)
    if over:
        return [list(s) for s in over]
    cyc = c.ch.lib.cycles.get(cyc_name)
    if not cyc:
        raise ChoreoError(f"no {cyc_name} cycle in poses.spec.json cycles")
    return cyc


def _ease(n: int, mode: str) -> np.ndarray:
    """Relative step lengths for n steps: constant, or slowing down at the end (out) / speeding up (in)."""
    w = np.ones(n)
    prof = [0.97, 0.88, 0.74]  # prototype 6's last strides into the crouch
    if mode in ("out", "inout"):
        for j, f in enumerate(reversed(prof[-min(3, n):])):
            w[n - 1 - j] = f
    if mode in ("in", "inout"):
        for j, f in enumerate(reversed(prof[-min(3, n):])):
            w[j] = f
    return w


@template("run")
def run(c: Ctx) -> None:
    """Run strides on the ground. from/to: times (default: right after the previous move, up to the next one);
    x: [x0, x1] street x of the torso (numbers or obstacle coordinates; default from the neighbours, or speed);
    from_x: the start x alone;
    speed: px per 16th when nothing fixes the end (88); ease: none|out|in|inout; every: pose rate (default the
    photo step); contacts: beats (default) or and (the foot lands on the "and"); offset: the same in 16ths; cycle:
    stride to start with (default: continues the previous run); foot_lock: keep the stance foot planted between beat
    contacts (default: the choreography's foot_lock)."""
    q0 = c.time("from") if c.has("from") else _start_after_prev(c)
    c.st.q0 = q0
    step = c.ch.step_at(q0)
    every = rate_of(c.arg("every", step))
    if c.has("to"):
        q1 = c.time("to", ref=q0)
    elif c.next is not None and c.next.q0 is not None:
        q1 = c.next.q0 - c.ch.step_at(c.next.q0)
    else:
        if c.dry:  # the end comes from the next move; the start is known already (the move before aims at it)
            xs = c.arg("x")
            if isinstance(xs, list) and len(xs) == 2:
                c.st.x0, c.st.x1 = c.coord(xs[0], "x"), c.coord(xs[1], "x")
            elif c.has("from_x"):
                c.st.x0 = c.st.x1 = c.coord(c.arg("from_x"), "x")
            return
        raise ChoreoError("needs to: (nothing after it to run to)")
    qs = list(np.arange(q0, q1 + 1e-6, every))
    if len(qs) < 1:
        raise ChoreoError(f"from {c.ch.grid.addr(q0)} to {c.ch.grid.addr(q1)} is empty")
    n = len(qs)
    speed = float(c.arg("speed", RUN_SPEED * c.u))
    xs = c.arg("x")
    if xs is not None:
        if not isinstance(xs, list) or len(xs) != 2:
            raise ChoreoError("x: give [x0, x1]")
        x0, x1 = c.coord(xs[0], "x"), c.coord(xs[1], "x")
    else:
        if c.has("from_x"):
            x0 = c.coord(c.arg("from_x"), "x")
        elif c.prev is not None and c.prev.x1 is not None:
            x0 = c.prev.x1 + (c.prev.exit_gap if c.prev.exit_gap is not None else speed * every)
        else:
            raise ChoreoError("needs x: [x0, x1] or from_x: (nothing before it to start from)")
        if c.next is not None and c.next.x0 is not None:
            x1 = c.next.x0 - (c.next.entry_gap if c.next.entry_gap is not None else speed * every)
        else:
            x1 = x0 + speed * every * (n - 1)
    if n > 1 and x1 < x0:
        raise ChoreoError(f"would run backwards ({x0:.0f} to {x1:.0f}): give it more time or move a neighbour")
    w = _ease(n - 1, str(c.arg("ease", "none"))) if n > 1 else np.ones(0)
    plan = np.concatenate([[x0], x0 + np.cumsum(w) / max(w.sum(), 1e-9) * (x1 - x0)]) if n > 1 else np.array([x0])
    cycles = _run_cycle(c)
    start = c.arg("cycle")
    if start is None:
        start = c.prev.cycle if c.prev is not None and c.prev.cycle else 0
    contacts = str(c.arg("contacts", "beats"))
    if contacts not in ("beats", "and"):
        raise ChoreoError(f"contacts: beats or and, not {contacts}")
    # 16ths: shifts the contacts off the beats (2: on the "and")
    off = float(c.arg("offset", 2 if contacts == "and" else 0))
    beat0 = math.floor((q0 + off) / 4)
    names, phases = [], []
    ph_names = c.ch.lib.phases.get("run", [])
    for q in qs:
        s = (int(start) + int(math.floor((q + off) / 4)) - beat0) % len(cycles)
        stride = cycles[s]
        pos = int(round(((q + off) % 4) / 4 * len(stride))) % len(stride)
        names.append(stride[pos])
        phases.append(ph_names[pos] if pos < len(ph_names) and len(stride) == len(ph_names) else "")
    x = plan.copy()
    lock = c.arg("foot_lock", c.ch.foot_lock)
    if lock and n > 1:
        x = foot_lock(plan, names, phases, [c.ch.lib.foot_offset(p) for p in names], [q + off for q in qs],
                      every, float(c.arg("max_lurch", 2.0)))
    for q, p, xx in zip(qs, names, x):
        c.wall(q, p, xx, tag="run")
    c.st.cycle = (int(start) + int(math.floor((qs[-1] + off) / 4)) - beat0 + 1) % len(cycles)
    c.st.exit_gap = speed * every


def foot_lock(plan: np.ndarray, names: list[str], phases: list[str], foot: list[float], qs: list[float],
              every: float, max_lurch: float = 2.0) -> np.ndarray:
    """Torso x per pose so the stance foot stays where it touched down, from the contact through the passing pose
    to the toe-off (the drawings set how far the torso moves over the foot); the flight pose and the next contact
    go back to the plan. A lurch is limited to max_lurch times the planned step, the torso to 0.9 of a step ahead of
    the plan (a quarter in the last photos, so the run still meets the next move), and it never moves back. A run
    slower than a quarter of the drawings' stance length per photo cannot hold the foot fully."""
    stance = ("contact", "passing", "toeoff")
    x = plan.astype(float).copy()
    n = len(x)
    for j in range(1, n):
        v = plan[j] - plan[j - 1]
        same_stride = math.floor(qs[j] / 4) == math.floor(qs[j - 1] / 4) and abs(qs[j] - qs[j - 1] - every) < 1e-6
        if same_stride and phases[j - 1] in stance and phases[j] in stance[1:]:
            d_lock = foot[j - 1] - foot[j]
            lo, hi = 0.35 * v, max_lurch * v
            x[j] = x[j - 1] + float(np.clip(d_lock, lo, hi)) if v > 0 else x[j - 1]
            # at most most of a step ahead of the plan (the flight photo still moves on), and close to the plan at
            # the end of the run, so it still meets the next move
            x[j] = min(x[j], plan[j] + (0.9 if j < n - 2 else 0.25) * v)
        else:
            x[j] = max(plan[j], x[j - 1] + 0.25 * v)
    return x


@template("walk")
def walk(c: Ctx) -> None:
    """Walk: one pose per `every` (default beats), alternating the walk cycle. from/to: times; x: start x (default
    118 px after the previous move); stride: px per pose (170); keepers: true (every pose stays as a ghost)."""
    every = rate_of(c.arg("every", "beats"))
    if c.has("from"):
        q0 = c.time("from")
    elif c.prev is not None and c.prev.q1 is not None:
        q0 = math.ceil(_start_after_prev(c) / every - 1e-6) * every
    elif c.dry:
        return
    else:
        raise ChoreoError("needs from:")
    c.st.q0 = q0
    if c.has("to"):
        q1 = c.time("to", ref=q0)
    elif c.next is not None and c.next.q0 is not None:
        q1 = c.next.q0 - every
    else:
        if c.dry:
            return
        raise ChoreoError("needs to:")
    c.st.q1 = q1
    x0 = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(118 * c.u)
    if x0 is None:
        return
    stride = float(c.arg("stride", 170 * c.u))
    cyc = [p for s in _run_cycle(c, "walk") for p in s]
    keep = bool(c.arg("keepers", True))
    for k, q in enumerate(np.arange(q0, q1 + 1e-6, every)):
        c.wall(q, cyc[k % len(cyc)], x0 + stride * k, keeper=keep, tag="walk")
    c.st.exit_gap = stride


@template("brake")
def brake(c: Ctx) -> None:
    """Brake hard: the brake pose at `at` (default: the next kick), held for `hold` 16ths (2) with a little slide.
    x: default 80 px after the previous move."""
    q = c.time("at", default="kick")
    hold = float(c.arg("hold", 2))
    c.st.q0, c.st.q1 = q, q + hold
    c.st.entry_gap = 80.0 * c.u
    x = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(80 * c.u)
    if x is None:
        return
    c.wall(q, c.pose("brake"), x, keeper=True, tag="brake")
    if hold > 0:
        c.wall(q + hold, c.pose("brake"), x + 22 * c.u, tag="brake")
    c.label(q, x, c.feet, "brake")
    c.st.entry_gap = 80.0


@template("crouch")
def crouch(c: Ctx) -> None:
    """Stop at an edge and crouch: brake-in poses on the two photos before `at`, the crouch on `at` (a keeper).
    edge: an obstacle with an `edge` contact (his front stops 8 px before it), or x: the crouch's torso x."""
    q = c.time("at")
    s = c.ch.step_at(q)
    name = c.pose("crouch")
    if c.has("edge"):
        o = c.ob("edge")
        ex = o.contact("edge", "near").x
        m = c.ch.lib.m(name)
        x = ex - 8 * c.u - (m["bbox"][2] - m["cx"]) * m["scale"]
    elif c.has("x"):
        x = c.coord(c.arg("x"), "x")
    else:
        raise ChoreoError("needs edge: or x:")
    if c.arg("brake_in", True):
        bi = c.poses("brake_in")
        c.wall(q - 2 * s, bi[0], x - 38 * c.u, tag="crouch")
        c.wall(q - s, bi[min(1, len(bi) - 1)], x - 12 * c.u, tag="crouch")
        c.st.entry_gap = 48.0 * c.u
    c.wall(q, name, x, keeper=True, tag="crouch")
    c.label(q, x, c.feet, "crouch")


@template("freeze")
def freeze(c: Ctx) -> None:
    """The song stops: the pose holds (nothing is painted, the paint does not dry or drip) while the photos keep
    coming (rain, light and grain still change). from: default the previous move's last photo; to: required."""
    q0 = c.time("from") if c.has("from") else (c.prev.q1 if c.prev is not None and c.prev.q1 is not None else None)
    if q0 is None:
        raise ChoreoError("needs from:")
    q1 = c.time("to", ref=q0)
    if q1 is None:
        raise ChoreoError("needs to:")
    c.freeze(q0, q1)
    if c.prev is not None:
        c.st.x0 = c.st.x1 = c.prev.x1
    c.st.entry_gap = c.st.exit_gap = None


# ---------------------------------------------------------------------------------------------- jumps

# J2 flight keyframes (prototype 6 over the alley): fraction of the way from take-off to landing x, centre height
# above the feet line, rotation. n flight photos sample this curve.
LEAP_ARC = [(0.230, -316.0, 8.0), (0.412, -396.0, 0.0), (0.593, -336.0, -6.0), (0.757, -231.0, -12.0)]


def _sample(table: list[tuple], n: int) -> list[tuple]:
    if n == len(table):
        return table
    us = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    src = np.linspace(0, 1, len(table))
    cols = list(zip(*table))
    return [tuple(float(np.interp(u, src, col)) for col in cols) for u in us]


@template("leap")
def leap(c: Ctx) -> None:
    """Jump between buildings over a gap. gap: a gap obstacle; at: when he leaves the wall (J2) or lands (J1, the
    impact); land_on: J2 landing (default: the next snare after `at`); style: J2 (default: the choreography's jump)
    or J1; lead: J1 take-off this many photos before the impact (1 or 2); takeoff_x, land_x: torso x (default 90 px
    before the near edge, 300 px past the far side)."""
    g = c.ob("gap", ("gap",))
    style = str(c.arg("style", c.ch.jump)).upper()
    at = c.time("at")
    s = c.ch.step_at(at)
    near, far = g.x0, g.x1
    u = c.u
    tx = c.coord(c.arg("takeoff_x"), "x") if c.has("takeoff_x") else near - 90 * u
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else far + 300 * u
    f = c.feet
    glow = g.glow or [1.0, 0.55, 0.22]
    if style == "J1":
        c.arg("land_on")  # J1 lands on the impact itself
        lead = int(c.arg("lead", c.ch.j1_lead))
        land = at
        tko = at - lead * s
        big = 1.25
        c.wall(tko, c.pose("takeoff"), tx, keeper=True, flags={"takeoff": True, "burst": "J1", "edge": near + 2},
               tag="leap.takeoff")
        c.stamp(tko, c.pose("burst"), near, f - 236 * u, 760 * u, tag="leap.burst")
        c.stamp(tko, c.pose("droplets"), near + 50 * u, f - 336 * u, 560 * u, tag="leap.burst")
        smear_q = tko + 0.5 * s if lead == 1 else at - s
        if lead == 1:
            c.photo(smear_q)
        e = c.smear(smear_q)
        e.flags = {"x0": near, "y": f, "body": c.pose("flight"), "stamp": c.pose("smear"), "gap": [near, far],
                   "u": u}
        for q in np.arange(tko, at - 1e-6, 0.5 * s):
            c.cam(q, zoom=1.42, cx=near + (160 if q < smear_q else 460) * u, snap=True, free=True)
    else:
        land = c.time("land_on", default="snare", ref=at)
        if land <= at:
            raise ChoreoError(f"land_on {c.ch.grid.addr(land)} is not after at {c.ch.grid.addr(at)}")
        tko = at - s
        big = 0.95
        c.wall(tko, c.pose("takeoff"), tx, keeper=True, flags={"takeoff": True, "burst": "J2", "edge": near + 2,
                                                                "glow": glow}, tag="leap.takeoff")
        c.stamp(tko, c.pose("spray"), tx - 50 * u, f - 216 * u, 420 * u, tag="leap.spray")
        qs = list(np.arange(at, land - 1e-6, s))
        fl = c.poses("flight")
        for k, (q, (uu, dy, rot)) in enumerate(zip(qs, _sample(LEAP_ARC, len(qs)))):
            pose = fl[min(int(k * len(fl) / len(qs)), len(fl) - 1)] if len(qs) != len(fl) else fl[k]
            c.air(q, pose, lerp(tx, lx, uu), f + dy * u, rot=rot, flags={"flight": True, "glow": glow},
                  tag="leap.flight")
        zooms = _sample([(1.36,), (1.36,), (1.34,), (1.30,), (1.24,)], len(qs) + 1)
        for q, (z,) in zip([tko] + qs, zooms):
            c.cam(q, zoom=z, dx=40 * u, dy=-25 * u, snap=True, free=True)
    c.wall(land, c.pose("land"), lx, keeper=True, flags={"halo": True}, tag="leap.land",
           contact=(g.id, "far", "foot", "past"))
    c.stamp(land, c.pose("splash"), lx - 125 * u, f - 141 * u, 720 * big * u, flags={"splash": True, "primer": True},
            tag="leap.splash")
    c.stamp(land, c.pose("droplets"), lx + 110 * u, f - 296 * u, 560 * big * u, tag="leap.splash")
    c.stamp(land, c.pose("drips"), lx - 140 * u, f - 236 * u, 400 * big * u, tag="leap.splash")
    rec = c.poses("recover")
    c.wall(land + 2 * s, rec[0], lx + 60 * u, tag="leap.recover")
    c.wall(land + 3 * s, rec[min(1, len(rec) - 1)], lx + 130 * u, tag="leap.recover")
    c.cam(land, zoom=0.97, snap=True)
    c.cam(land + s, zoom=0.99)
    c.label(tko, tx, f, f"take-off ({style})")
    c.label(land, lx, f, "landing")
    c.st.entry_gap = 90.0 * u
    c.st.exit_gap = 130.0 * u


# ---------------------------------------------------------------------------------------------- obstacles

@template("vault")
def vault(c: Ctx) -> None:
    """Kong vault over an obstacle with a `top` contact: approach, jump, reach, hands planted on the top on
    `plant_on`, legs through, drop, land four photos later."""
    o = c.ob("obstacle")
    top = o.contact("top").y
    p = c.time("plant_on")
    s = c.ch.step_at(p)
    x0, x1, w, f, u = o.x0, o.x1, o.x1 - o.x0, c.feet, c.u
    c.wall(p - 3 * s, c.pose("vault_approach"), x0 - 201 * u, tag="vault")
    c.wall(p - 2 * s, c.pose("jump"), x0 - 126 * u, f - 20 * u, tag="vault")
    c.wall(p - s, c.pose("reach"), x0 + 36 * u, top - 17 * u, anchor="right", tag="vault")
    c.wall(p, c.pose("vault_plant"), x0 + 0.332 * w, top + 5 * u, anchor="br", keeper=True, tag="vault.plant",
           contact=(o.id, "top", "br"))
    c.stamp(p, c.pose("droplets"), x0 + 0.352 * w, top - 30 * u, 230 * u, tag="vault")
    c.wall(p + s, c.pose("vault_over"), x0 + 0.708 * w, top + 5 * u, anchor="left", tag="vault.over",
           contact=(o.id, "top", "left"))
    c.wall(p + 2 * s, c.pose("drop"), x1 + 86 * u, f - 111 * u, tag="vault")
    c.wall(p + 3 * s, c.pose("drop"), x1 + 131 * u, f - 26 * u, rot=3, tag="vault")
    c.wall(p + 4 * s, c.pose("land_vault"), x1 + 126 * u, keeper=True, tag="vault.land")
    c.label(p, x0 + 0.332 * w, top, f"vault {o.id}")
    c.st.entry_gap = 60.0 * u
    c.st.exit_gap = 115.0 * u


@template("slide")
def slide(c: Ctx) -> None:
    """Drop into a slide under an obstacle with an `under` contact (its underside) at `at`, slide, and pop up on
    `up` (default six photos later). The slide spans the obstacle, or is centred on a narrow one."""
    o = c.ob("under")
    o.contact("under")  # an obstacle without an underside cannot be slid under
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
    xs = np.linspace(xi, xr, n + 1)
    c.wall(q_in, c.pose("slide_in"), xs[0], keeper=True, tag="slide")
    for k in range(1, n - 1):
        c.wall(q_in + k * s, c.pose("slide"), xs[k], keeper=(k == 2 and n > 4), tag="slide.under",
               contact=(o.id, "under", "top"))
    c.wall(q_up - s, c.pose("slide_out"), xs[n - 1], tag="slide")
    rise = c.poses("rise")
    c.wall(q_up, rise[0], xr, keeper=True, tag="slide.rise")
    if len(rise) > 1:
        c.wall(q_up + s, rise[1], xr + 80 * u, tag="slide.rise")
    c.label(q_in, xi, c.feet, f"slide {o.id}")
    c.st.entry_gap = 110.0 * u
    c.st.exit_gap = 90.0 * u


@template("climb")
def climb(c: Ctx) -> None:
    """Climb a pipe (`pole` contact): jump and reach, grab on `grab`, climb every `every` (8ths) rising `rise` px
    per step, then (with `to`: a beam obstacle) lunge to its `beam` contact on `lunge` (default: the next snare after
    the last climb step)."""
    o = c.ob("pipe", ("pipe", "pole", "ladder"))
    px = o.contact("pole").x
    g = c.time("grab")
    s = c.ch.step_at(g)
    every = rate_of(c.arg("every", "8ths"))
    u = c.u
    rise = float(c.arg("rise", 48.3 * u))
    hy = c.coord(c.arg("hands_y"), "y") if c.has("hands_y") else c.feet - 556 * u
    f = c.feet
    c.wall(g - 2 * s, c.pose("jump"), px - 122 * u, f - 20 * u, tag="climb")
    c.wall(g - s, c.pose("reach"), px - 10 * u, hy + 60 * u, anchor="right", tag="climb")
    beam = c.ob("to", ("beam", "ledge", "ladder")) if c.has("to") else None
    if beam is not None:
        lunge = c.time("lunge", default="snare", ref=g + 4 * every - 0.5)
    else:
        lunge = c.time("until", ref=g) if c.has("until") else g + 4 * every
    cl = c.poses("climb")
    used: dict[str, int] = {}
    for k, q in enumerate(np.arange(g, lunge - 1e-6, every)):
        name = cl[k % len(cl)]
        v = _vary(c, used.get(name, 0), k, mirror=False, rot=3.0, off=5.0 * u)
        used[name] = used.get(name, 0) + 1
        c.wall(q, name, px + 4 * u + v.get("ox", 0), hy - k * rise + v.get("oy", 0), anchor="top",
               rot=v.get("rot", 0.0), mirror=v.get("mirror", False), keeper=(k % 2 == 0), tag="climb.grip",
               contact=(o.id, "pole", "top"))
    if beam is not None:
        by = beam.contact("beam", "bar").y
        c.wall(lunge, c.pose("reach"), px + 188 * u, by + 15 * u, anchor="right", keeper=True, tag="climb.lunge",
               contact=(beam.id, beam.contact("beam", "bar").name, "right"))
    c.label(g, px, hy, f"climb {o.id}")
    c.st.entry_gap = 122.0 * u


# hang pattern (prototype 6 under the fire escape): 16ths after `from`, role, rotation, x step, y offset
HANG = [(0, "hang", 0.0, 0.0, 0.0), (2, "hang", -8.0, 50.0, 0.0), (3, "swing", 0.0, 105.0, 0.0),
        (5, "hang", -14.0, 150.0, 0.0), (6, "swing", 10.0, 200.0, 3.0)]


@template("hang")
def hang(c: Ctx) -> None:
    """Hang from a beam (`beam` contact) and swing along it: from (default right after the previous move) to `to`
    (default: up to the next move), starting at x (default 10 px past the previous move), moving `traverse` px per
    16th (33). Hands (the top anchor) stay on the beam."""
    o = c.ob("beam", ("beam", "ledge", "ladder", "pipe"))
    by = o.contact("beam", "bar").y
    q0 = c.time("from") if c.has("from") else _start_after_prev(c)
    if c.has("to"):
        q1 = c.time("to", ref=q0)
    elif c.next is not None and c.next.q0 is not None:
        q1 = c.next.q0 - c.ch.step_at(c.next.q0)
    else:
        q1 = q0 + 6
    c.st.q0, c.st.q1 = q0, q1
    u = c.u
    x0 = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(10 * u)
    if x0 is None:
        return
    trav = float(c.arg("traverse", 33.3 * u)) / u  # in the pattern's units (a 450 px figure)
    used: dict[str, int] = {}
    k = 0
    while True:
        cyc, j = divmod(k, len(HANG))
        dq, role, rot, dxs, dy = HANG[j]
        dq += 7 * cyc  # the pattern repeats every 7 16ths (a little higher swing each time)
        q = q0 + dq
        if q > q1 + 1e-6:
            break
        name = c.pose(role)
        v = _vary(c, used.get(name, 0), k, mirror=(role == "hang"), rot=3.0, off=6.0 * u)
        used[name] = used.get(name, 0) + 1
        amp = 1.0 + 0.15 * cyc
        c.wall(q, name, x0 + (dxs + 233 * cyc) * (trav / 33.3) * u + v.get("ox", 0), by + (5 + dy) * u,
               anchor="top", rot=rot * amp + v.get("rot", 0.0), mirror=v.get("mirror", False),
               keeper=(role == "swing" and j == 2), tag="hang", contact=(o.id, o.contact("beam", "bar").name, "top"))
        k += 1
    c.label(q0, x0, by, f"hang {o.id}")


# swing-release flight keyframes (prototype 6, off the fire escape): fraction from release to landing x, role
# index into flight, anchor, height above the feet line, rotation
RELEASE_ARC = [(0.219, 0, "center", -536.0, -2.0), (0.438, 0, "center", -451.0, -12.0),
               (0.630, 1, "center", -336.0, 0.0), (0.795, 2, "foot", -96.0, 0.0), (0.904, 2, "foot", -16.0, 4.0)]


@template("swing_release")
def swing_release(c: Ctx) -> None:
    """Kick off a beam and release on `release_on`, fly along the wall, land on `land_on` (default: the next kick)
    with a splash. bar: the beam (for the release height); x: release x (default 260 px past the previous move);
    land_x: default 730 px further."""
    o = c.ob("bar", ("beam", "ledge", "ladder", "pipe"))
    by = o.contact("beam", "bar").y
    r = c.time("release_on")
    s = c.ch.step_at(r)
    land = c.time("land_on", default="kick", ref=r)
    c.st.q0, c.st.q1 = r, land
    u = c.u
    xr = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(260 * u)
    if xr is None:
        return
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else xr + 730 * u
    f = c.feet
    c.wall(r, c.pose("release"), xr, by + 53 * u, anchor="right", keeper=True, tag="release")
    qs = list(np.arange(r + s, land - 1e-6, s))
    fl = c.poses("flight")
    table = RELEASE_ARC if len(qs) == len(RELEASE_ARC) else [
        (u, int(round(fi)), "center" if u < 0.7 else "foot", dy, rot)
        for (u, fi, dy, rot) in _sample([(a, b, d, e) for a, b, _, d, e in RELEASE_ARC], len(qs))]
    for q, (uu, fi, anc, dy, rot) in zip(qs, table):
        c.wall(q, fl[min(int(fi), len(fl) - 1)], lerp(xr, lx, uu), f + dy * u, anchor=anc, rot=rot,
               tag="release.flight")
    c.wall(land, c.pose("land"), lx, keeper=True, tag="release.land")
    c.stamp(land, c.pose("droplets"), lx + 20 * u, f - 86 * u, 380 * u, tag="release.land")
    c.stamp(land, c.pose("splash"), lx - 5 * u, f - 76 * u, 300 * u, tag="release.land")
    c.label(r, xr, by, "release")
    c.st.exit_gap = 40.0 * u


@template("step_up")
def step_up(c: Ctx) -> None:
    """Leap onto a step (`step` contact of the obstacle, e.g. a lit doorstep) at `at`, land on it two photos later,
    push off, land on the ground on `land_on` (default six photos after `at`)."""
    o = c.ob("onto")
    sy = o.contact("step", "top").y
    a = c.time("at")
    s = c.ch.step_at(a)
    land = c.time("land_on", ref=a) if c.has("land_on") else a + 6 * s
    x0, x1, f, u = o.x0, o.x1, c.feet, c.u
    if land < a + 5 * s:
        raise ChoreoError("land_on: needs at least five photos after at")
    c.wall(a, c.pose("jump"), x0 - 76 * u, f - 20 * u, keeper=True, tag="step_up")
    c.wall(a + s, c.pose("flight", 2), x0 + 64 * u, sy - 37 * u, tag="step_up")
    c.wall(a + 2 * s, c.pose("land_small"), x0 + 174 * u, sy, keeper=True, tag="step_up.on",
           contact=(o.id, "step", "foot"))
    c.wall(a + 3 * s, c.pose("takeoff"), x0 + 284 * u, sy - 10 * u, tag="step_up")
    air = list(np.arange(a + 4 * s, land - 1e-6, s))
    for k, q in enumerate(air):
        if k == 0:
            c.wall(q, c.pose("flight", 0), x1 + 129 * u, f - 336 * u, anchor="center", rot=6, keeper=True,
                   tag="step_up")
        else:
            c.wall(q, c.pose("flight", 2), x1 + (129 + k * 140 / max(len(air) - 1, 1)) * u, f - 76 * u, tag="step_up")
    c.wall(land, c.pose("land_small"), x1 + 359 * u, keeper=True, tag="step_up.land")
    c.wall(land + s, c.pose("recover_small"), x1 + 439 * u, tag="step_up")
    c.label(a + 2 * s, x0 + 174 * u, sy, f"step {o.id}")
    c.st.entry_gap = 60.0 * u
    c.st.exit_gap = 90.0 * u


@template("pole_swing")
def pole_swing(c: Ctx) -> None:
    """Swing round a pipe (`pole` contact): reach, grab on `grab_on`, release, fly, land on `land_on` (default six
    photos after the grab). span: scales how far the flight carries (1.0; less when an obstacle follows close)."""
    o = c.ob("pipe", ("pipe", "pole"))
    px = o.contact("pole").x
    g = c.time("grab_on")
    s = c.ch.step_at(g)
    land = c.time("land_on", ref=g) if c.has("land_on") else g + 6 * s
    f, u = c.feet, c.u
    sp = float(c.arg("span", 1.0))
    fl = c.poses("flight")
    c.wall(g - s, c.pose("reach"), px - 16 * u, f - 436 * u, anchor="right", tag="pole_swing")
    c.wall(g, c.pose("swing"), px + 4 * u, f - 461 * u, anchor="top", rot=15, keeper=True, tag="pole_swing.grip",
           contact=(o.id, "pole", "top"))
    c.wall(g + s, c.pose("release"), px + 279 * sp * u, f - 476 * u, anchor="right", tag="pole_swing")
    arc = [(439, 0, "center", -366.0, -6.0, False), (569, 1, "center", -296.0, 0.0, False),
           (679, 2, "foot", -76.0, 0.0, True), (749, 2, "foot", -11.0, 3.0, False)]
    qs = list(np.arange(g + 2 * s, land - 1e-6, s))
    if len(qs) != len(arc):
        arc = [arc[min(int(k * len(arc) / max(len(qs), 1)), len(arc) - 1)] for k in range(len(qs))]
    for q, (dx, fi, anc, dy, rot, kp) in zip(qs, arc):
        c.wall(q, fl[min(fi, len(fl) - 1)], px + dx * sp * u, f + dy * u, anchor=anc, rot=rot, keeper=kp,
               tag="pole_swing.flight")
    c.wall(land, c.pose("land_small"), px + 809 * sp * u, keeper=True, tag="pole_swing.land")
    c.wall(land + s, c.pose("recover_small"), px + 884 * sp * u, tag="pole_swing")
    c.label(g, px, f - 461 * u, f"pole swing {o.id}")
    c.st.entry_gap = 105.0 * u
    c.st.exit_gap = 85.0 * u


# ---------------------------------------------------------------------------------------------- the box (opening)

def _hits(c: Ctx, on: str, t0: float, t1: float, least: float | None = None) -> list[float]:
    """Song times of the hits `on` (snare, kick, hat, impact) or grid steps (beats, 8ths, 16ths) in [t0, t1]."""
    g = c.ch.grid
    if on in ("beats", "8ths", "16ths", "bars", "half-bars"):
        return [g.t(q) for q in g.steps(g.q(t0) - 1e-6, g.q(t1) + 1e-6, rate_of(on))]
    if on not in g.hits:
        raise ChoreoError(f"crack on {on!r}: snare, kick, hat, impact, beats, 8ths or 16ths")
    th = g.thresholds.get(on, 0.0) if least is None else float(least)
    return [t for t, st in g.hits[on] if st >= th and t0 - 0.02 <= t <= t1 + 0.02]


def _lines(text) -> list[str]:
    if isinstance(text, list):
        return [str(t) for t in text]
    return [t.strip() for t in str(text).replace("\n", "/").split("/") if t.strip()]


@template("cam", prop=True)
def cam(c: Ctx) -> None:
    """A camera move over a span, on top of the follow framing: from, to, zoom (the crop's width over the follow
    width: 1.7 pulls back), dy (street px: negative frames higher, more of the wall above him, a tilt up), dx (his
    lead, px; default the follow's 0.07 of the crop width), ease (16ths: it eases in from `from` and back out before
    `to`, 4). Each photo's hint is then smoothed by the camera like any framing (review note: the walkway's pull-back)."""
    from .camera import CW0

    q0 = c.time("from")
    q1 = c.time("to", ref=q0)
    if q1 <= q0:
        raise ChoreoError("cam: to must come after from")
    z, dy, ease = float(c.arg("zoom", 1.0)), float(c.arg("dy", 0.0)) * c.u, max(float(c.arg("ease", 4)), 1.0)
    dx = c.arg("dx")
    for q in np.arange(q0, q1, 1.0):
        k = min(1.0, (q - q0 + 1) / ease, (q1 - q) / ease)
        k = k * k * (3 - 2 * k)
        zq = 1 + (z - 1) * k
        c.cam(float(q), zoom=zq, dy=dy * k, dx=float(dx) * c.u if dx is not None else 0.07 * CW0 * c.u * zq,
              free=False)


@template("blink", prop=True)
def blink(c: Ctx) -> None:
    """Street lights blinking on the music: the antennas' red warning lights on the kicks as he passes, a lit sign
    flickering. lights (a light id from obstacles.json, or a list), on (kick, snare, hat, impact, beats, 8ths, or a
    list of times), from, to, dim (0.08: the light's glow and its lamp's lit pixels on those photos; a light with a
    `layer` dims its lamp there), min (the hits' least strength), every (1: each hit, 2: every other one), stagger
    (16ths: each next light in the list blinks that much later, a chase along the roofs)."""
    ids = c.arg("lights")
    ids = [str(ids)] if isinstance(ids, str) else [str(i) for i in (ids or [])]
    known = {str(L.get("id")) for L in c.ch.scene.lights}
    bad = [i for i in ids if i not in known]
    if not ids or bad:
        raise ChoreoError(f"blink: lights {bad or ids} are not in obstacles.json lights ({', '.join(sorted(known))})")
    g = c.ch.grid
    q0 = c.song(c.time("from"))
    q1 = c.song(c.time("to", ref=q0))
    on = c.arg("on", "kick")
    if isinstance(on, list):
        qs = [c.song(c.at(v, key="on")) for v in on]
    else:
        qs = [g.snap(g.q(t)) for t in _hits(c, str(on), g.t(q0), g.t(q1), c.arg("min"))]
    qs = qs[::max(int(c.arg("every", 1)), 1)]
    dim, stagger = float(c.arg("dim", 0.08)), float(c.arg("stagger", 0))
    for j, lid in enumerate(ids):
        for q in qs:
            c.light(c.clock.fwd(q + j * stagger), lid, dim)


@template("stencil", prop=True)
def stencil(c: Ctx) -> None:
    """Words stencilled on the wall in black spray paint, without a box (WAIT by the second red light, verse 2's
    pitch lines): at, text ("A BETTER/FACE": "/" breaks the lines), x (the block's left edge, centre or right edge by
    align), y (its top), h (cap height, 58 px for a 450 px figure), align (left, center, right), rot, punch (0.03: the
    camera punches in as it lands; 0 for none), photo (true: a photo exactly on it), id (stencil1, stencil2 ...),
    albedo (black spray by default; 0.7: white primer letters, where black would not read: a dark shop fascia), fade:
    {at, over} (or a time): it fades out from `at` over `over` 16ths (2) and is gone (a lyric line after it is sung);
    runs (true; false: no paint runs down the letters, P7's STOP).
    Drawn like the box's stencils (the same letters, the spray burst, runs), one mark per stencil.
    smear: {at, length (500 px), over (3 16ths), rows: [y0, y1] (street y; default all of it), dir (right)}: a body
    running through it from `at` drags its letters along in streaks over `over` 16ths, and they stay smeared."""
    from .props import BoxProp, Stencil

    u = c.u
    n = sum(1 for p in c.ch.props if str(p).startswith("stencil"))
    pid = str(c.arg("id", f"stencil{n + 1}"))
    if pid in c.ch.props:
        raise ChoreoError(f"stencil: a prop {pid!r} already exists (give it its own id)")
    if not c.has("text") or not c.has("x") or not c.has("y"):
        raise ChoreoError("stencil: give at, text, x and y (the block's top)")
    rng = np.random.default_rng([c.mv.seed, c.mv.local_i, 107])
    q = c.time("at")
    align = str(c.arg("align", "left"))
    if align not in ("left", "center", "right"):
        raise ChoreoError("stencil: align is left, center or right")
    st = Stencil(c.song(q), _lines(c.arg("text")), c.coord(c.arg("x"), "x"), c.coord(c.arg("y"), "y"),
                 float(c.arg("h", 58 * u)), align, float(c.arg("rot", 0.0)), int(rng.integers(0, 2 ** 31)))
    if c.has("albedo"):  # white letters (primer) where black would not read: verse 2's dark shop fascias
        st.albedo = float(c.arg("albedo"))
    if c.arg("runs", True) is False:  # (v8b: no paint runs at all: P7's STOP, a run down the S read as "$TOP")
        st.runs = False
    if c.has("fade"):  # fade: {at, over}: the line fades out from `at` over `over` 16ths (2) and is gone after
        fd = c.arg("fade")
        fd = {"at": fd} if not isinstance(fd, dict) else fd
        st.fade = (c.song(c.at(fd["at"], key="fade.at")), float(fd.get("over", 2)))
    sm = c.arg("smear")
    if sm:
        if not isinstance(sm, dict) or "at" not in sm:
            raise ChoreoError("stencil: write smear as {at, length, over, rows, dir}")
        side = str(sm.get("dir", "right"))
        if side not in ("right", "left"):
            raise ChoreoError("stencil: smear dir is right or left")
        rows = sm.get("rows")
        st.smear = (c.song(c.at(sm["at"], key="smear.at")), float(sm.get("length", 500 * u)), float(sm.get("over", 3)),
                    (c.coord(rows[0], "y"), c.coord(rows[1], "y")) if rows else None, 1 if side == "right" else -1)
    c.ch.props[pid] = BoxProp(pid, [], [st], seed=int(rng.integers(0, 2 ** 31)))
    if c.arg("photo", False):
        c.photo(q)
    if float(c.arg("punch", 0.03)):
        c.cam(q, punch=float(c.arg("punch", 0.03)))


@template("box", prop=True)
def box(c: Ctx) -> None:
    """The box (board B1): a rectangle of thick white primer lines on the wall with drips, (re)painted on its keys.
    id (box); x: its centre; bottom: its outer bottom edge (default 26 px above the ground line); keys: [{at, w, h,
    thick, bulge, bulge_at, bulge_h, x, bottom}], each keeping the previous key's values unless it changes them
    (shrink a step on each rule, repaint it thicker, push its sides out locally: bulge as a share of the width, at
    bulge_at of the height from the top, spread bulge_h); the first photo before the first key shows the bare wall.
    stencils: [{at, text ("KEEP/IT DOWN" or a list of lines), x, y (the block's top), h (cap height, 58), align
    (left, center, right), rot, punch (0.03: the camera punches in as it lands; 0 for none), albedo (none: black
    spray; 0.7: white primer letters, on dark brick)}], each landing with a
    spray burst (a cloud of black mist round the letters settling in half a second) and runs that grow;
    coordinates may use the box (box.x0-30). cracks: [{on: snare|kick|beats|8ths,
    from, to, at: [x, y], size (40), grow (12), min (least strength)}] or [{at: time, x, y, size}]. photo: true
    adds a photo exactly on each key and stencil (so they hit their words). hits: [{on: snare, from, to, min, side
    (right), bulge (14 px), y (0.45 of the height from the top), flakes (8), grow (0: the bulge and flakes grow by
    that share each hit)}, or {at, ...} for one hit]: hits from inside, the side giving and springing back, primer
    flakes falling. peel_from: the stencils start to crack and peel (half speed) before the shatter. A key's
    sides ({top|bottom|left|right: time, ...}, v8b): only those sides, each painted on its own time (the cold open's
    box closing on him side by side); otherwise all four on its key. A key's
    paint_in (seconds): it is painted in with a brush sweep (a third of it on its key); so is the figure (figure.at:
    when, figure.paint_in). A key's drop ({height (1.6 times the box), over (0.26 s)} or true): v8, it slams down onto
    that key from above, a photo at a time and ever faster, each falling outline gone at once as the next one is
    painted (the villain catching him); the key itself lands as any key (its hits: the flakes knocked off). (figure.at:
    when, figure.paint_in). old (0 to 1, default 0): every key
    painted long ago (dry, faded, rain-washed; a key's own `old` overrides it). figure: someone else in the box, a
    drawing ({pose, fill (0.9), align, mirror, old (0.85)} or just the pose name) painted grey and old, fitted to
    the first key and still the whole film."""
    from .props import BoxKey, BoxProp, Bump, Crack, Figure, Stencil

    pid = str(c.arg("id", "box"))
    u = c.u
    x = c.coord(c.require("x"), "x")
    bottom = c.coord(c.arg("bottom"), "y") if c.has("bottom") else c.ch.scene.wall_base_at(x) - 26 * u
    old0 = float(c.arg("old", 0.0))
    cur = {"w": 340 * u, "h": 400 * u, "thick": 24 * u, "bulge": 0.0, "bulge_at": 0.5, "bulge_h": 0.12, "x": x,
           "bottom": bottom}
    rng = np.random.default_rng([c.mv.seed, c.mv.local_i, 101])
    photo = bool(c.arg("photo", False))
    keys = []
    for j, k in enumerate(c.require("keys")):
        if not isinstance(k, dict) or "at" not in k:
            raise ChoreoError(f"keys[{j + 1}]: write it as {{at: ..., w: ..., h: ...}}")
        q = c.at(k["at"], key=f"keys[{j + 1}].at")
        for f in ("w", "h", "thick", "bulge", "bulge_at", "bulge_h"):
            if f in k:
                cur[f] = float(k[f])
        if "x" in k:
            cur["x"] = c.coord(k["x"], "x")
        if "bottom" in k:
            cur["bottom"] = c.coord(k["bottom"], "y")
        dr = k.get("drop")
        if dr:  # v8: it slams down from above onto this key, a photo at a time, falling ever faster (the villain)
            dr = {} if dr is True else dict(dr)
            hgt = float(dr.get("height", 1.6 * cur["h"]))
            over = float(dr.get("over", 0.26))
            g_ = c.ch.grid
            qs_ = c.song(q)
            st_ = c.ch.step_at(qs_)
            n_ = max(1, int(round(over / max(g_.t(qs_) - g_.t(qs_ - st_), 1e-3))))
            for i_ in range(n_, 0, -1):
                s_ = (n_ - i_) / n_  # 0 at its first photo, falling as under gravity onto the key
                keys.append(BoxKey(qs_ - i_ * st_, cur["x"], cur["bottom"] - hgt * (1 - s_ * s_), cur["w"], cur["h"],
                                   cur["thick"], 0.0, int(rng.integers(0, 2 ** 31)), 0.5, 0.12, 0.0, 0.0, drop=True,
                                   bright=True))
        sides = ()
        if k.get("sides") is not None:  # v8b: each side painted at its own time ({right: 2.75s, top: 3.1, ...})
            sd = k["sides"]
            if not isinstance(sd, dict) or not sd or any(n not in ("top", "bottom", "left", "right") for n in sd):
                raise ChoreoError(f"keys[{j + 1}].sides: {{top|bottom|left|right: time, ...}}, not {sd!r}")
            g_, t_k = c.ch.grid, c.ch.grid.t(c.song(q))
            sides = tuple(sorted(((n, g_.t(c.song(c.at(v, key=f"keys[{j + 1}].sides.{n}"))) - t_k)
                                  for n, v in sd.items()), key=lambda nv: nv[1]))
            if sides[0][1] < -1e-6:
                raise ChoreoError(f"keys[{j + 1}].sides: no side before its key ({k['at']})")
        keys.append(BoxKey(c.song(q), cur["x"], cur["bottom"], cur["w"], cur["h"], cur["thick"], cur["bulge"],
                           int(rng.integers(0, 2 ** 31)), cur["bulge_at"], cur["bulge_h"],
                           float(k.get("old", old0)), float(k.get("paint_in", 0.0)), bright=bool(dr), sides=sides))
        if photo:
            c.photo(q)
    if not keys:
        raise ChoreoError("keys: give at least one {at, w, h}")
    keys.sort(key=lambda k: k.q)
    pr = BoxProp(pid, keys, seed=int(rng.integers(0, 2 ** 31)))
    c.ch.props[pid] = pr
    fg = c.arg("figure")
    if fg:
        fg = {"pose": fg} if isinstance(fg, str) else dict(fg)
        if "pose" not in fg:
            raise ChoreoError("figure: give its pose ({pose: OT1-2} or the pose name)")
        name = str(fg["pose"])
        if name not in c.ch.lib:
            raise ChoreoError(f"figure: unknown pose {name!r}")
        k0 = keys[0]
        ix0, iy0, ix1, iy1 = k0.inner
        sc = float(fg["scale"]) if "scale" in fg else _fit(c, [name], ix1 - ix0, iy1 - iy0, float(fg.get("fill", 0.9)),
                                                          float(fg.get("max_scale", 1.6)))
        fx, fy = _place_in_box(c, name, k0, sc, str(fg.get("align", "center")))
        fq = c.song(c.at(fg["at"], key="figure.at")) if "at" in fg else None
        pr.figure = Figure(name, fx, fy, sc, bool(fg.get("mirror", False)), float(fg.get("old", 0.85)),
                           int(rng.integers(0, 2 ** 31)), fq, float(fg.get("paint_in", 0.0)))
    for j, st in enumerate(c.arg("stencils") or []):
        if not isinstance(st, dict) or "at" not in st or "text" not in st:
            raise ChoreoError(f"stencils[{j + 1}]: write it as {{at, text, x, y, h}}")
        q = c.at(st["at"], key=f"stencils[{j + 1}].at")
        align = str(st.get("align", "left"))
        if align not in ("left", "center", "right"):
            raise ChoreoError(f"stencils[{j + 1}]: align is left, center or right")
        pr.stencils.append(Stencil(c.song(q), _lines(st["text"]), c.coord(st.get("x", x), "x"),
                                   c.coord(st.get("y", keys[0].box[1] - 160 * u), "y"), float(st.get("h", 58 * u)),
                                   align, float(st.get("rot", 0.0)), int(rng.integers(0, 2 ** 31)),
                                   albedo=float(st["albedo"]) if st.get("albedo") is not None else None))
        if photo:
            c.photo(q)
        if float(st.get("punch", 0.03)):  # each rule lands like a hit: the camera punches in a little
            c.cam(q, punch=float(st.get("punch", 0.03)))
    g = c.ch.grid
    for j, hb in enumerate(c.arg("hits") or []):  # hits from inside: the side gives and primer flakes fall
        if isinstance(hb, dict) and "at" in hb:  # one hit at a set time
            times = [g.t(c.song(c.at(hb["at"], key=f"hits[{j + 1}].at")))]
        elif isinstance(hb, dict) and "on" in hb and "from" in hb and "to" in hb:
            t0 = g.t(c.song(c.at(hb["from"], key=f"hits[{j + 1}].from")))
            t1 = g.t(c.song(c.at(hb["to"], key=f"hits[{j + 1}].to")))
            times = _hits(c, str(hb["on"]), t0, t1, hb.get("min"))
        else:
            raise ChoreoError(f"hits[{j + 1}]: write it as {{on: snare, from, to, side, bulge, y, flakes}} or "
                              f"{{at, ...}}")
        for n, t in enumerate(times):
            grow = 1 + float(hb.get("grow", 0.0)) * n
            pr.bumps.append(Bump(g.q(t), str(hb.get("side", "right")), float(hb.get("bulge", 14)) * u * grow,
                                 float(hb.get("y", 0.45)), int(hb.get("flakes", 8) * grow),
                                 int(rng.integers(0, 2 ** 31))))
    if c.has("peel_from"):
        pr.peel_from = c.song(c.time("peel_from"))
    for j, cr in enumerate(c.arg("cracks") or []):
        at = cr.get("at")
        if isinstance(at, list) and len(at) == 2:
            px, py = c.coord(at[0], "x"), c.coord(at[1], "y")
        else:
            px, py = c.coord(cr.get("x", x), "x"), c.coord(cr.get("y", keys[0].box[1]), "y")
        size, grow = float(cr.get("size", 40 * u)), float(cr.get("grow", 12 * u))
        if "on" in cr:
            t0 = g.t(c.song(c.at(cr["from"], key=f"cracks[{j + 1}].from")))
            t1 = g.t(c.song(c.at(cr["to"], key=f"cracks[{j + 1}].to")))
            times = [g.q(t) for t in _hits(c, str(cr["on"]), t0, t1, cr.get("min"))]
        else:
            times = [c.song(c.at(at if not isinstance(at, list) else cr["time"], key=f"cracks[{j + 1}].at"))]
        for n, q in enumerate(times):
            pr.cracks.append(Crack(float(round(q)), px, py, size + grow * n, int(rng.integers(0, 2 ** 31))))
    c.label(keys[0].q, x, keys[0].box[1], f"box {pid}")


def _box(c: Ctx):
    pid = str(c.arg("box", "box"))
    if pid not in c.ch.props:
        raise ChoreoError(f"box: no prop {pid!r} (a box template with id: {pid} must come first)")
    return c.ch.props[pid]


def _fit(c: Ctx, names: list[str], iw: float, ih: float, fill: float, most: float) -> float:
    """The scale (on top of each pose's own) that fits the drawings in an inner box of iw x ih (fill of it)."""
    sizes = [c.ch.lib.size(n) for n in names]
    bw, bh = max(w for w, _ in sizes), max(h for _, h in sizes)
    return float(min(most, fill * min(iw / max(bw, 1), ih / max(bh, 1))))


def _place_in_box(c: Ctx, name: str, key, scale: float, align: str) -> tuple[float, float]:
    """x, y of the bottom-centre anchor (bc) that stands the drawing on the box's inner bottom edge, centred or
    pressed against a side (touching it where the drawing reaches furthest, the bulge included)."""
    ix0, iy0, ix1, iy1 = key.inner
    m = c.ch.lib.m(name)
    half = (m["bbox"][2] - m["bbox"][0]) / 2 * m["scale"] * scale
    y = iy1 - 2 * c.u
    if align in ("right", "left"):
        ey = _edge_point(c, name, 0.0, y, scale, align)[1]  # the height of the hand, fist or shoulder at the side
        sx = key.side_x(ey, align)
        return (sx - 2 * c.u - half, y) if align == "right" else (sx + 2 * c.u + half, y)
    return (ix0 + ix1) / 2, y


def _boxed(key) -> dict:
    """Paint flags for a figure repainted inside a box: a light trail (no keepers: the ghosts and paint-outs wash
    away instead of piling up in one place) and its primer patch kept inside the box."""
    x0, y0, x1, y1 = key.box
    b = key.bulge * key.w
    return {"trail": "light", "clip": [x0 - b, y0, x1 + b, y1], "patch": 0.2}


def _edge_point(c: Ctx, name: str, x: float, y: float, scale: float, side: str = "right") -> tuple[float, float]:
    """Street position of a drawing's `side` anchor (the fist, the shoulder) when its bc anchor is at x, y."""
    m = c.ch.lib.m(name)
    s = m["scale"] * scale
    ax, ay = c.ch.lib.anchor(name, side)
    bx, by = c.ch.lib.anchor(name, "bc")
    return x + (ax - bx) * s, y + (ay - by) * s


@template("inside")
def inside(c: Ctx) -> None:
    """The figure inside a box (box: the prop id): a pose on every photo from `from` to `to`, the role from a
    schedule (roles: [[time, role], {at, role, align, fill, scale, crack}...]); a role with several drawings
    alternates them (breathing). Each pose is scaled to fit the box as it is at that photo (fill: its share of the
    inner height or width, 0.88; max_scale: 1.15) and stands on the box's inner bottom edge, centred or pressed
    against a side (align: right or left). crack: {on: snare|kick|beats, min, size, grow} grows a crack where the
    pose touches the side (its right or left edge) on those hits. push: px a held pose (one drawing) leans into
    its side on every other photo.

    A schedule entry can play a clip instead of a role ({at, clip: <video sequence>, keys: {key: time}, speed}): a
    front view of him in the clip's guide box (its outline keyed out), every frame fitted to the box as it is on
    that photo (the clip's box on the street box: one body scale per photo, so he shrinks with the box on the
    rules), the keys pinned to their times (piecewise linear between them, `speed` times the clip's own speed
    outside them), the first and last frame held where the clip does not reach. strike: {on: kick|snare, min,
    hit: <frame list>, back: <frame list>, push, crack: {size, grow}, punch: [first, last], shake, flicker}
    replaces the clip's own timing with the music's: a photo on a hit shows the next `hit` frame (a punch, a
    shove: a list of clip frames, or the name of the sequence's list), every other photo the next `back` frame
    (drawn back), so each punch lands on a hit and the in-betweens are dropped (`tail: [from, to]`: the photos after
    the last hit play those frames instead, ending on `to`: the hand-over to the next clip); push leans him into the side on the hit, crack
    grows a crack where his fist or shoulder meets the side, punch and shake hit the camera (growing over the
    hits), flicker dims a light on each hit (the red pedestrian light). strike segments: [[approach, hit], ...]
    (clip frames) plays the clip in pieces between the hits instead, each piece ending on its hit frame on the hit
    (a step back and a slam, a recoil and a punch, cycling), `tail: [from, to]` after the last hit (the wind-up that
    hands over to the next clip). frames: [clip frames] plays those frames one a photo from the entry's `at`, in
    any order (a clip backwards: box-mouth's fold reversed, he unfolds from the tight ball into the hand over his
    mouth), the last one held. start: <key> starts a keyed entry at that key's frame instead of the clip's first (the
    clip carrying on where a frames entry left it). blows: {keys: [pinned keys], push, crack, punch, shake, flicker} on a keyed clip:
    the photos of those keys are the hits (a shoulder slam, a kick), with the strike's effects. lean: [key_a, key_b, degrees, px]: between those two keys he leans (rotated,
    shifted) toward something outside the box: the hood turned toward the red light; when both keys are pinned
    (keys), by time: leaning in over the first half, held, let go over the last photos (the stare), else over the
    clip frames between them."""
    from .motion import key_frame
    from .props import Crack

    pr = _box(c)
    q0 = c.time("from")
    q1 = c.time("to", ref=q0)
    sched = []
    for j, e in enumerate(c.require("roles")):
        if isinstance(e, list) and len(e) >= 2:
            e = {"at": e[0], "role": e[1], **(e[2] if len(e) > 2 and isinstance(e[2], dict) else {})}
        if not isinstance(e, dict) or "at" not in e or ("role" not in e and "clip" not in e):
            raise ChoreoError(f"roles[{j + 1}]: write it as [time, role], {{at, role, ...}} or {{at, clip, keys}}")
        sched.append((c.at(e["at"], key=f"roles[{j + 1}].at"), e))
    sched.sort(key=lambda r: r[0])
    fill0, most0 = float(c.arg("fill", 0.88)), float(c.arg("max_scale", 1.15))
    last, k = None, 0
    g = c.ch.grid
    ncrack: dict[int, int] = {}
    clips: dict[int, dict] = {}  # entry -> {photo q: clip frame row}
    for q in _photo_qs(c, q0, q1):  # every photo, the extra ones on the words too
        j = max([i for i, (t, _) in enumerate(sched) if t <= q + 1e-6], default=0)
        e = sched[j][1]
        if j != last:
            k = 0
        key = pr.geometry(c.song(q))
        if "clip" in e:
            if j not in clips:
                end = sched[j + 1][0] - 1e-3 if j + 1 < len(sched) else q1
                clips[j] = _clip_photos(c, e, sched[j][0], end, j)
            r, push = clips[j][round(q, 4)]
            name = r["name"]
            x, y, sc = _in_box_place(c, pr, e["_seq"], r, key)
            side = str(e.get("side", "right"))
            x += (push if side == "right" else -push) * c.u
            rot = 1.2 if push else 0.0
            if e.get("lean"):  # lean toward something outside the box (the red light) between two keys
                la, lb, deg = e["lean"][:3]
                kq = e.get("keys") or {}
                w = 0.0
                if la in kq and lb in kq:  # pinned: by time, leaning in over the first half, held, let go at the end
                    qa_, qb_ = (c.at(kq[k_], ref=sched[j][0], key=f"lean.{k_}") for k_ in (la, lb))
                    v = (q - qa_) / max(qb_ - qa_, 1e-6)
                    if 0 <= v <= 1:
                        u_ = min(1.0, v / 0.5)
                        w = u_ * u_ * (3 - 2 * u_) * min(1.0, (1 - v) / 0.12 + 0.0)
                else:  # by clip frame
                    fa, fb = (key_frame(e["_seq"], str(k_)) for k_ in (la, lb))
                    if fa <= r["f"] <= fb:
                        w = min(1.0, (r["f"] - fa + 1) / 4, (fb - r["f"] + 1) / 4)
                rot += float(deg) * w
                x += float(e["lean"][3] if len(e["lean"]) > 3 else 0.0) * w * c.u
            if c.arg("wall", True):  # v7 (the director at 32.5): the box's sides stop him, a fist into the line at most
                m_ = c.ch.lib.m(name)  # (a margin for the jitter): the whole figure moved back by what would go through
                ss_ = m_["scale"] * sc
                lims, exs = {}, {}
                for sd_, sgn in (("right", 1), ("left", -1)):
                    exs[sd_] = ((m_["bbox"][2] if sgn > 0 else m_["bbox"][0]) - m_["cx"]) * ss_
                    ay_ = y + (c.ch.lib.anchor(name, sd_)[1] - c.ch.lib.anchor(name, "cxc")[1]) * ss_
                    lims[sd_] = key.side_x(ay_, sd_) + sgn * (key.thick - 12 * c.u)
                room, span = lims["right"] - lims["left"], exs["right"] - exs["left"]
                if span > room > 0:  # wider than the box (both hands on the sides): a little smaller, centred
                    k_ = room / span
                    sc *= k_
                    x = (lims["left"] + lims["right"]) / 2 - (exs["right"] + exs["left"]) / 2 * k_
                else:
                    for sd_, sgn in (("right", 1), ("left", -1)):
                        over = sgn * (x + exs[sd_] - lims[sd_])
                        if over > 0:
                            x -= sgn * over
            c.wall(q, name, x, y, anchor="cxc", scale=sc, rot=rot, tag="inside", flags=_boxed(key))
            if j != last:
                c.label(q, x, key.box[1], str(e.get("label") or e["_seq"].get("label") or e["clip"]))
            st = e.get("strike") or e.get("blows")
            if st and r.get("_hit") is not None and not c.dry:
                n = r["_hit"]
                m = c.ch.lib.m(name)
                ss = m["scale"] * sc
                ax, ay = c.ch.lib.anchor(name, side)
                tx, ty = c.ch.lib.anchor(name, "cxc")
                py = y + (ay - ty) * ss
                if st.get("crack") is not None:
                    crk = dict(st["crack"]) if isinstance(st["crack"], dict) else {}
                    px = key.side_x(py, side) + (3 if side == "right" else -3) * c.u
                    pr.cracks.append(Crack(c.song(q), px, py, float(crk.get("size", 30 * c.u)) +
                                           float(crk.get("grow", 14 * c.u)) * n,
                                           int(np.random.default_rng([c.mv.seed, c.mv.local_i, int(q), n, 5]).integers(
                                               0, 2 ** 31))))
                nh = max(e["_nhits"] - 1, 1)
                if st.get("punch"):
                    p0, p1 = st["punch"]
                    c.cam(q, punch=float(lerp(float(p0), float(p1), n / nh)))
                if st.get("shake"):
                    c.cam(q, shake=float(st["shake"]) * (0.5 + 0.5 * n / nh))
                if st.get("flicker"):
                    c.light(q, str(st["flicker"]), float(st.get("dim", 0.15)))
        else:
            names = c.poses(str(e["role"]))
            name = names[k % len(names)]
            ix0, iy0, ix1, iy1 = key.inner
            iw = key.width_at(iy0 + 0.45 * (iy1 - iy0))
            sc = float(e["scale"]) if "scale" in e else _fit(c, names, iw, iy1 - iy0, float(e.get("fill", fill0)),
                                                              float(e.get("max_scale", most0)))
            align = str(e.get("align", "center"))
            x, y = _place_in_box(c, name, key, sc, align)
            push = float(e.get("push", 0.0)) * c.u  # a held push: every other photo leans into the side a little
            if push and len(names) == 1 and k % 2 == 1:
                x += -push if align == "left" else push
            c.wall(q, name, x, y, anchor="bc", scale=sc, rot=(1.5 if push and k % 2 == 1 else 0.0), tag="inside",
                   flags=_boxed(key))
            if j != last:
                c.label(q, x, key.box[1], str(e["role"]))
            crk = e.get("crack")
            if crk and not c.dry:  # the hits since the previous photo of this role (each on the first photo after it)
                crk = {"on": crk} if isinstance(crk, str) else dict(crk)
                t_b = g.t(c.song(q)) + 0.03
                t_a = g.t(c.song(q - c.ch.step_at(q))) + 0.03 if j == last else g.t(c.song(sched[j][0])) - 0.03
                hits = [h for h in _hits(c, str(crk.get("on", "snare")), t_a, t_b, crk.get("min")) if t_a < h <= t_b]
                for _ in hits:
                    side = "left" if align == "left" else "right"
                    px, py = _edge_point(c, name, x, y, sc, side)
                    px = key.side_x(py, side) + (3 if side == "right" else -3) * c.u
                    n = ncrack.get(j, 0)
                    ncrack[j] = n + 1
                    pr.cracks.append(Crack(c.song(q), px, py, float(crk.get("size", 34 * c.u)) +
                                           float(crk.get("grow", 14 * c.u)) * n,
                                           int(np.random.default_rng([c.mv.seed, c.mv.local_i, int(q), n]).integers(
                                               0, 2 ** 31))))
        last = j
        k += 1
    cx = pr.keys[0].cx
    c.st.x0 = c.st.x1 = cx


def _photo_qs(c: Ctx, qa: float, qb: float) -> list[float]:
    """Every photo from move time qa to qb: the segments' steps and the extra photos (keys and stencils on their
    words) already asked for."""
    qs, q = [], qa
    while q <= qb + 1e-6:
        qs.append(round(q, 4))
        q += c.ch.step_at(q)
    for qe in c.ch.extra_photos:
        qm = c.clock.fwd(qe)
        if qa - 1e-6 <= qm <= qb + 1e-6:
            qs.append(round(qm, 4))
    return sorted(set(qs))


def _clip_photos(c: Ctx, e: dict, qa: float, qb: float, j: int) -> dict:
    """The clip frame (track row) and the push (px) on every photo of an inside schedule entry from qa to qb (move
    time): the clip's own timing with its keys pinned, or with `strike` the hit frames on the hits and the back
    frames between them."""
    from .motion import key_frame, video_keyed, video_track

    try:
        seq = c.ch.lib.sequence(str(e["clip"]))
    except KeyError as ex:
        raise ChoreoError(str(ex.args[0])) from ex
    if "video" not in seq:
        raise ChoreoError(f"clip {e['clip']}: not a video sequence")
    e["_seq"] = seq
    vid, tr = video_track(c, seq)
    if not vid.get("block"):
        raise ChoreoError(f"clip {e['clip']}: its video has no guide box (a front view in a box outline)")
    g = c.ch.grid
    qs = _photo_qs(c, qa, qb)
    out: dict = {}
    if e.get("frames"):  # set frames, one a photo (any order: a clip played backwards), the last one held after
        by_f = {r["f"]: r for r in tr}
        fs = sorted(by_f)
        want = [by_f[min(fs, key=lambda x: abs(x - int(f)))] for f in e["frames"]]
        return {round(q, 4): (want[min(i, len(want) - 1)], 0.0) for i, q in enumerate(qs)}
    st = e.get("strike")
    if st and st.get("segments"):
        return _strike_segments(c, e, seq, tr, qs, qa, qb)
    if st:
        by_f = {r["f"]: r for r in tr}
        fs = sorted(by_f)

        def frames(v) -> list:  # a list of clip frames, or the name of the sequence's list (nearest kept frame)
            lst = v if isinstance(v, list) else seq.get(str(v), [])
            return [by_f[min(fs, key=lambda x: abs(x - int(f)))] for f in lst]

        hit, back = frames(st.get("hit", "hit")), frames(st.get("back", "back"))
        if not hit or not back:
            raise ChoreoError(f"clip {e['clip']}: strike needs hit and back frames (lists, or the names of the "
                              f"sequence's frame lists)")
        t0, t1 = g.t(c.song(qa)) - 0.03, g.t(c.song(qb)) + 0.03
        hits = _hits(c, str(st.get("on", "snare")), t0, t1, st.get("min"))
        on = [any(abs(g.t(c.song(q)) - h) <= 0.04 for h in hits) for q in qs]  # a hit within half a 16th
        last = max([i for i, v in enumerate(on) if v], default=-1)
        tail = st.get("tail")
        nh = nb = 0
        for i, q in enumerate(qs):
            if on[i]:
                out[round(q, 4)] = (dict(hit[nh % len(hit)], _hit=nh), float(st.get("push", 0.0)))
                nh += 1
            elif tail and i > last >= 0:  # after the last hit: the tail [from, to], ending on its last frame
                n = len(qs) - 1 - last
                v = int(tail[0]) + (int(tail[1]) - int(tail[0])) * (i - last) / n
                out[round(q, 4)] = (by_f[min(fs, key=lambda x: abs(x - v))], 0.0)
            else:
                out[round(q, 4)] = (back[nb % len(back)], 0.0)
                nb += 1
        e["_nhits"] = nh
        return out
    pins = [(c.at(v, ref=qa, key=f"roles[{j + 1}].keys.{kk}"), key_frame(seq, kk))
            for kk, v in (e.get("keys") or {}).items()]
    f_start = key_frame(seq, str(e["start"])) if e.get("start") else int(seq["frames"][0])
    if not any(f == f_start for _, f in pins):  # the clip starts with the entry (at its `start` key, or its first frame)
        pins.append((qa, f_start))
    pins.sort()
    shots = dict((round(q, 4), r) for q, r in video_keyed(c, seq, pins, qa, qb, float(e.get("speed", 1.0)),
                                                         alternate=True))
    first = min(tr, key=lambda r: r["f"])
    tail = sorted(tr, key=lambda r: r["f"])[-3:]  # past its end he keeps at it: the last three frames to and fro
    loop = [tail[1], tail[0], tail[1], tail[2]] if len(tail) == 3 else tail[::-1]
    n_after = 0
    for q in qs:
        r = shots.get(round(q, 4))
        if r is None and shots and q < min(shots):  # before the clip's first frame: held
            r = first
        elif r is None:
            r = loop[n_after % len(loop)]
            n_after += 1
        out[round(q, 4)] = (r, 0.0)
    bl = e.get("blows")
    if bl:  # the blows: the photos of these pinned keys hit the box (crack, push, camera), one after another
        pk = {kk: round(c.at(v, ref=qa, key=f"blows.{kk}"), 4) for kk, v in (e.get("keys") or {}).items()}
        n = 0
        for kk in bl.get("keys", []):
            if pk.get(kk) in out:
                r, _ = out[pk[kk]]
                out[pk[kk]] = (dict(r, _hit=n), float(bl.get("push", 0.0)))
                n += 1
        e["_nhits"] = n
    return out


def _strike_segments(c: Ctx, e: dict, seq: dict, tr: list, qs: list, qa: float, qb: float) -> dict:
    """strike with segments: the clip played in pieces between the hits, so each hit lands on its frame: before the
    first hit the clip from its start to the first segment's hit frame; between two hits the next segment
    [approach frame, hit frame] (cycling through them: a step back and a slam, a recoil and a punch), spread over
    the photos up to the hit; after the last hit the `tail` [from, to] (the wind-up that hands over to the next
    clip). Every photo shows the clip frame nearest its place in the piece; no drawing on more than two photos in
    a row."""
    st = e["strike"]
    g = c.ch.grid
    segs = [tuple(int(v) for v in sg) for sg in st["segments"]]
    fs = sorted(r["f"] for r in tr)
    by_f = {r["f"]: r for r in tr}
    t0, t1 = g.t(c.song(qa)) - 0.03, g.t(c.song(qb)) + 0.03
    hits = _hits(c, str(st.get("on", "snare")), t0, t1, st.get("min"))
    hit_i = []
    for h in hits:
        j = min(range(len(qs)), key=lambda i: abs(g.t(c.song(qs[i])) - h))
        if abs(g.t(c.song(qs[j])) - h) <= 0.045 and (not hit_i or j > hit_i[-1]):
            hit_i.append(j)
    pieces = []  # (first photo, last photo, from frame, to frame, ends on a hit)
    start = 0
    for n_, j in enumerate(hit_i):
        a, h = segs[n_ % len(segs)]
        if n_ == 0:
            a = fs[0]
        pieces.append((start, j, a, h, True))
        start = j + 1
    if start < len(qs):
        tail = st.get("tail")
        a, b = (int(tail[0]), int(tail[1])) if tail else ((segs[(len(hit_i) - 1) % len(segs)][1] if hit_i
                                                          else fs[0]), fs[-1])
        pieces.append((start, len(qs) - 1, a, b, False))
    out: dict = {}
    shown: list = []
    nh = 0
    for p0, p1, a, b, on_hit in pieces:
        n = p1 - p0 + 1
        for j in range(n):
            v = a + (b - a) * ((j + 1) / n if on_hit else (j / max(n - 1, 1) if n > 1 else 1.0))
            f = min(fs, key=lambda x: abs(x - v))
            if len(shown) >= 2 and shown[-1] == f and shown[-2] == f:  # no third photo on one drawing
                i = fs.index(f)
                f = fs[min(i + 1, len(fs) - 1)] if b >= a else fs[max(i - 1, 0)]
            shown.append(f)
            q = qs[p0 + j]
            if on_hit and j == n - 1:
                out[round(q, 4)] = (dict(by_f[f], _hit=nh), float(st.get("push", 0.0)))
                nh += 1
            else:
                out[round(q, 4)] = (by_f[f], 0.0)
    e["_nhits"] = nh
    return out


def _in_box_place(c: Ctx, pr, seq: dict, r: dict, key) -> tuple[float, float, float]:
    """Where a front-view clip frame goes in the box as it is now: the clip's guide box (outer) on the street box,
    one scale from their heights. Returns (torso x, bbox-centre y, scale on top of the frame's own) for anchor cxc."""
    vid = c.ch.lib.videos[seq["video"]]
    bx0, by0, bx1, by1 = vid["block"]
    x0, y0, x1, y1 = key.box
    s = (y1 - y0) / max(by1 - by0, 1)
    return (key.cx + (r["x"] - (bx0 + bx1) / 2) * s, y1 + (r["y"] - by1) * s,
            s / c.ch.lib.m(r["name"])["scale"])


@template("clip")
def clip(c: Ctx) -> None:
    """Play a clip (a video sequence) on the wall (motion.clip; the same in both engines)."""
    from .motion import clip as _clip

    _clip(c)


@template("pound")
def pound(c: Ctx) -> None:
    """Pound the box's side (box: the prop id): a strike on every `every` (8ths) from `from` to `to`, the wind-up a
    16th before each (an extra photo when the segment is slower), the fist reaching the box's inner `side` (right),
    a crack growth at the fist on every strike (size, growing by `grow` a strike, so the cracks spread across the
    stencils), and the camera punching harder as it builds (punch: [first, last] share of the frame; shake: px of
    hand-held shake on each strike). body: a set size for both drawings (default: fit them in the box by fill)."""
    from .props import Crack

    pr = _box(c)
    q0 = c.time("from")
    q1 = c.time("to", ref=q0)
    every = rate_of(c.arg("every", "8ths"))
    strikes = list(np.arange(q0, q1 + 1e-6, every))
    p0, p1 = c.arg("punch", [0.02, 0.09])
    size, grow = float(c.arg("size", 30 * c.u)), float(c.arg("grow", 24 * c.u))
    fill = float(c.arg("fill", 0.92))
    fixed = float(c.arg("body")) if c.has("body") else None  # a set size (as the brace before it), else fit
    wind, hit = c.pose("windup"), c.pose("strike")
    side = str(c.arg("side", "right"))
    rng = np.random.default_rng([c.mv.seed, c.mv.local_i, 7])
    have = {round(q, 4) for q in c.ch.grid.steps(q0 - 1, q1 + 1, c.ch.step_at(q0))}
    for k, q in enumerate(strikes):
        key = pr.geometry(c.song(q))
        ix0, iy0, ix1, iy1 = key.inner
        sc = fixed if fixed is not None else _fit(c, [wind, hit], key.width_at((iy0 + iy1) / 2), iy1 - iy0, fill, 1.15)
        x, y = _place_in_box(c, hit, key, sc, side)
        xw = x - (8 if side == "right" else -8) * c.u
        if round(c.song(q - 1), 4) not in have:
            c.photo(q - 1)
        c.wall(q - 1, wind, xw, y, anchor="bc", scale=sc, mirror=side == "left", tag="pound.windup", flags=_boxed(key))
        c.wall(q, hit, x, y, anchor="bc", scale=sc, mirror=side == "left", tag="pound.strike", flags=_boxed(key))
        fx, fy = _edge_point(c, hit, x, y, sc, side)
        if not c.dry:
            pr.cracks.append(Crack(c.song(q), key.side_x(fy, side) + (3 if side == "right" else -3) * c.u, fy,
                                   size + grow * k, int(rng.integers(0, 2 ** 31))))
        c.cam(q, punch=float(lerp(p0, p1, k / max(len(strikes) - 1, 1))), shake=float(c.arg("shake", 0.0)))
    c.label(q0, pr.keys[0].cx, pr.keys[0].box[1], "pound")
    c.st.x0 = c.st.x1 = pr.keys[0].cx


@template("burst")
def burst(c: Ctx) -> None:
    """Break out of the box (box: the prop id) on `at`: the outline shatters into flying primer and paint shards and
    he bursts out through its right side shoulder first, airborne along a rising arc (height: the arc's top, 300 px
    above the ground line, and at least a little above the box's middle) until `land_on` (default: the next impact),
    where he lands at land_x (default 760 px past
    the box) into a run. With pause on, a song stop during the flight freezes him mid-burst, the shards hanging
    round him; the shards fall after the stop. The camera punches in on the shatter."""
    from .props import Shatter

    pr = _box(c)
    at = c.time("at")
    s = c.ch.step_at(at)
    land = c.time("land_on", default="impact", ref=at)
    if land <= at:
        raise ChoreoError(f"land_on {c.ch.grid.addr(c.song(land))} is not after at")
    key = pr.geometry(c.song(at) - 0.01)
    u, f = c.u, c.feet
    bx0, by0, bx1, by1 = key.box
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else bx1 + 760 * u
    top = f - float(c.arg("height", 300)) * u
    fly = c.poses("break_out")
    qs = list(np.arange(at, land - 1e-6, s))
    n = len(qs)
    x_a, x_b = key.side_x((by0 + by1) / 2, "right") - 0.2 * key.w, lx - 200 * u
    y_a = (by0 + by1) / 2
    top = min(top, y_a - 40 * u)  # it always rises out of the box
    used: dict[str, int] = {}
    for k, q in enumerate(qs):
        v = k / max(n - 1, 1)
        name = fly[k % len(fly)]
        var = _vary(c, used.get(name, 0), k, mirror=False, rot=2.5, off=4.0 * u)
        used[name] = used.get(name, 0) + 1
        c.wall(q, name, x_a + (x_b - x_a) * v ** 0.8 + var.get("ox", 0), y_a + (top - y_a) * (1 - (1 - v) ** 2),
               anchor="center", rot=-12 + 14 * v + var.get("rot", 0.0), keeper=(k in (0, n - 1)),
               tag="burst.flight")
    c.stamp(at, c.pose("spray"), bx1, y_a, 380 * u, flags={"splash": True}, tag="burst.spray")
    c.wall(land, c.pose("break_land"), lx, keeper=True, flags={"halo": True}, tag="burst.land")
    c.stamp(land, c.pose("splash"), lx - 110 * u, f - 120 * u, 560 * u, flags={"splash": True, "primer": True},
            tag="burst.land")
    c.stamp(land, c.pose("drips"), lx - 120 * u, f - 200 * u, 320 * u, tag="burst.land")
    if not c.dry:
        sy = y_a - 0.08 * key.h
        pr.shatter = Shatter(c.song(at), key.side_x(sy, "right") + key.thick / 2, sy, 1.0,
                             int(np.random.default_rng([c.mv.seed, c.mv.local_i, 3]).integers(0, 2 ** 31)),
                             float(c.arg("fly", 0.65)))
    c.cam(at, punch=float(c.arg("punch", 0.1)))
    c.label(at, bx1, by0, "shatter")
    c.label(land, lx, f, "landing")
    c.st.exit_gap = 110.0 * u


@template("leap_off")
def leap_off(c: Ctx) -> None:
    """Jump onto an obstacle's top (a `step` or `top` contact: a bin lid, a low wall) at `at`, crouch on it a photo
    later, push off on the next and fly: the first flight photo is the highest point (a song stop right after it
    freezes him there), landing on `land_on` (default five photos after `at`) at land_x (default 560 px past the
    top's right end), with a splash, then recovering. height: the highest point above the top (230 px); jump_x:
    where he jumps from (90 px before the top)."""
    o = c.ob("onto")
    top = o.contact("step", "top")
    ty, tx0, tx1 = top.y, top.x0, top.x1
    a = c.time("at")
    s = c.ch.step_at(a)
    land = c.time("land_on", ref=a) if c.has("land_on") else a + 5 * s
    if land < a + 4 * s - 1e-6:
        raise ChoreoError("land_on: needs at least four photos after at")
    f, u = c.feet, c.u
    lx = c.coord(c.arg("land_x"), "x") if c.has("land_x") else tx1 + 560 * u
    peak = ty - float(c.arg("height", 230)) * u
    jx = c.coord(c.arg("jump_x"), "x") if c.has("jump_x") else tx0 - 90 * u
    c.wall(a, c.pose("jump"), jx, f - 20 * u, tag="leap_off")
    c.wall(a + s, c.pose("land_small"), (tx0 + tx1) / 2, ty, keeper=True, tag="leap_off.on",
           contact=(o.id, top.name, "foot"))
    c.wall(a + 2 * s, c.pose("takeoff"), tx1 - 6 * u, ty - 4 * u, tag="leap_off.push")
    fl = c.poses("flight")
    air = list(np.arange(a + 3 * s, land - 1e-6, s))
    for k, q in enumerate(air):
        v = k / max(len(air), 1)
        x = tx1 + 150 * u + (lx - 120 * u - tx1 - 150 * u) * v
        y = peak + (f - 260 * u - peak) * v * v
        c.wall(q, fl[min(k, len(fl) - 1)], x, y, anchor="center", rot=6 - 10 * v, keeper=(k == 0),
               tag="leap_off.flight")
    c.wall(land, c.pose("land"), lx, keeper=True, flags={"halo": True}, tag="leap_off.land")
    c.stamp(land, c.pose("splash"), lx - 110 * u, f - 130 * u, 600 * u, flags={"splash": True, "primer": True},
            tag="leap_off.land")
    c.stamp(land, c.pose("droplets"), lx + 90 * u, f - 280 * u, 480 * u, tag="leap_off.land")
    rec = c.poses("recover")
    c.wall(land + s, rec[0], lx + 70 * u, tag="leap_off.recover")
    c.label(a + s, (tx0 + tx1) / 2, ty, f"leap off {o.id}")
    c.label(air[0] if air else land, tx1 + 150 * u, peak, "highest point")
    c.st.entry_gap = 70.0 * u
    c.st.exit_gap = 110.0 * u


# ---------------------------------------------------------------------------------------------- by hand

@template("pose")
def pose(c: Ctx) -> None:
    """One pose placed by hand: at, pose (a name or a role), x, y (default the feet line), anchor (foot), rot,
    layer (wall or air), keeper, mirror, flipv. strokes: [{at, box: [x0, y0, x1, y1], angle}, ...]: the pose is
    painted by brush strokes slamming onto the wall one by one (props.stroke_masks; Full Stop's cold open): each
    stroke on its time reveals its share of him (box: shares of his painted box, angle: its slant), the last one
    whatever is left, over a broad white primer swath laid first (a roller pass, wider and longer than the stroke,
    staying on the wall); over (px, 20): the brush running past his silhouette, staying on the wall with its
    splatter; splats (40), drops (90: wet drops thrown off along the stroke, already flying on its photo); a stroke
    may set its own over, splats, drops and sweep (0 to 1: the brush's path running on past him, a long sweep);
    no primer patch. drip_in: {to, streaks, lead, amp} (v8b, cold open B): instead of appearing whole the pose is
    formed by paint running down the wall from `at` to `to`: a ragged front runs from his top to his feet (`amp`,
    0.07 of his height, its raggedness), everything above it painted, `streaks` (24) thin runs leading it by up to
    `lead` (0.3) of his height; no primer patch."""
    q = c.time("at")
    name = str(c.require("pose"))
    if name not in c.ch.lib:
        name = c.pose(name)
    x = c.coord(c.require("x"), "x")
    y = c.coord(c.arg("y", "feet"), "y")
    layer = c.arg("layer", "wall")
    kw = dict(anchor=c.arg("anchor", "foot" if layer == "wall" else "center"), rot=float(c.arg("rot", 0.0)),
              keeper=bool(c.arg("keeper", False)), mirror=bool(c.arg("mirror", False)),
              flipv=bool(c.arg("flipv", False)), tag=str(c.arg("label", "pose")))
    if layer == "air":
        c.air(q, name, x, y, flags={"flight": True}, **kw)
        return
    fl = None
    if c.has("strokes"):  # painted by brush strokes slamming onto the wall, one by one (the cold open)
        st = []
        for j, s_ in enumerate(c.arg("strokes")):
            if not isinstance(s_, dict) or "at" not in s_ or "box" not in s_ or len(s_["box"]) != 4:
                raise ChoreoError(f"pose: strokes[{j + 1}] is {{at, box: [x0, y0, x1, y1] (shares of him), angle}}")
            sq = c.at(s_["at"], key=f"strokes[{j + 1}].at")
            if sq < q - 1e-6:
                raise ChoreoError(f"pose: strokes[{j + 1}] lands before the pose is up")
            st.append([c.ch.grid.t(c.song(sq)), [float(v) for v in s_["box"]], float(s_.get("angle", 0.0)),
                       s_.get("over"), s_.get("splats"), s_.get("drops"), s_.get("sweep")])
            c.photo(sq)
        fl = {"strokes": st, "over": float(c.arg("over", 20.0)), "splats": int(c.arg("splats", 40)),
              "drops": int(c.arg("drops", 90)), "seed": int(np.random.default_rng([c.mv.seed, c.mv.local_i]).integers(
                  0, 2 ** 31)), "patch": 0.0}
    if c.has("drip_in"):  # formed by paint running down the wall (cold open B, v8b)
        di = c.arg("drip_in")
        if fl is not None or not isinstance(di, dict) or "to" not in di:
            raise ChoreoError("pose: drip_in is {to, streaks, lead, amp} (and not with strokes)")
        q1 = c.at(di["to"], key="drip_in.to")
        if q1 <= q + 1e-6:
            raise ChoreoError("pose: drip_in.to must come after the pose's at")
        fl = {"drip_in": {"t0": c.ch.grid.t(c.song(q)), "t1": c.ch.grid.t(c.song(q1)),
                          "streaks": int(di.get("streaks", 24)), "lead": float(di.get("lead", 0.3)),
                          "amp": float(di.get("amp", 0.07)),
                          "seed": int(np.random.default_rng([c.mv.seed, c.mv.local_i, 3]).integers(0, 2 ** 31))},
              "patch": 0.0}
    c.wall(q, name, x, y, flags=fl, **kw) if fl else c.wall(q, name, x, y, **kw)


@template("stamp")
def stamp(c: Ctx) -> None:
    """A paint stamp (splash, spray, drips ...): at, stamp (a name or a role), x, y (its centre), w (width in px),
    splash (droplets fly off), primer (a primer patch under it; keep: it stays on the wall instead of fading), swath
    (its shape painted in white primer instead of black, staying on the wall)."""
    q = c.time("at")
    name = str(c.require("stamp"))
    if name not in c.ch.lib:
        name = c.pose(name)
    c.stamp(q, name, c.coord(c.require("x"), "x"), c.coord(c.require("y"), "y"), float(c.require("w")),
            flags={"splash": bool(c.arg("splash", False)), "swath": bool(c.arg("swath", False)),
                   "primer": "keep" if c.arg("primer") == "keep" else bool(c.arg("primer", False))}, tag="stamp")


@template("screens", prop=True)
def screens(c: Ctx) -> None:
    """Shop windows' screens switched off (prop; Full Stop's verse 2: obstacles.json `screens`, built by `python -m
    aimv.paint.screens`): off: {group: time}, each window dark from its time, or order: [group, ...] with from (the
    first) and every (a rate: beats, one window a beat along the row); fade (16ths, 0.5). The street in a window's
    band turns to its off version (the screens dead, their glow gone from the pavement and the road), and the paint
    round it dims with its light (the stencils' glow and contrast drop)."""
    groups = {str(g["id"]) for g in (c.ch.scene.spec.get("screens") or {}).get("groups", [])}
    if not groups:
        raise ChoreoError("screens: obstacles.json has no `screens` groups")
    fade = float(c.arg("fade", 0.5))
    offs = {}
    if c.has("off"):
        for gid, at in dict(c.arg("off")).items():
            offs[str(gid)] = c.at(at, key=f"off.{gid}")
    else:
        q0 = c.time("from")
        step = rate_of(c.arg("every", "beats"))
        for i, gid in enumerate(c.require("order")):
            offs[str(gid)] = q0 + i * step
    for gid, q in offs.items():
        if gid not in groups:
            raise ChoreoError(f"screens: no group {gid!r} (obstacles.json screens: {', '.join(sorted(groups))})")
        c.screen_off(q, gid, fade)


from . import marks  # noqa: E402,F401  (the `marks` prop template: painted strokes in stages, marks.py)
from . import screenlevels  # noqa: E402,F401  (the `screen_cues` prop: screens lit and killed on cues, screenlevels.py)
from . import reveal  # noqa: E402,F401  (the `paint_in` prop template: the world painted in on the beat, reveal.py)
from . import windows  # noqa: E402,F401  (the `windows` prop template: windows lit along his path, windows.py)
from . import doors  # noqa: E402,F401  (the `door` prop: a door opening for him and shutting behind him, doors.py)
from . import hints  # noqa: E402,F401  (the `hints` prop template: camera punches and a light knocked out at set times, hints.py)
from . import truelook  # noqa: E402,F401  (the `true` prop: he becomes true, truelook.py)
from . import skinlook  # noqa: E402,F401  (the `skin_look` prop: the solid skin after the wall, skinlook.py)
from . import parkour  # noqa: E402,F401  (the `parkour` template: runs and jumps by physics, parkour.py)
