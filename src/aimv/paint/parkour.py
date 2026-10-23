"""Parkour (v7, review note): his runs and jumps by physics instead of keys pinned to beats.

A `parkour` move is a stretch of his route: the items he meets along it in order (runs, jumps over gaps, drops,
hops over props, vaults, flips, clips played in real time), each placed in space by the world (obstacles.json:
the feet lines of ground.levels, the gaps, the obstacles' boxes and top contacts). The solver walks him through
them in real time:

- one speed per gait (GAITS, m/s; the figure's standing height, the library's ref_height, is H_M = 1.8 m, so 450
  street px is 1.8 m and a metre is 250 px), with the accelerations a body can make (A_UP, A_DOWN) between speeds;
- stride-true runs always: the leg phase is the distance travelled over the cycle's measured length (no per_beat,
  no ramps); on the approach to a take-off the last steps lengthen or shorten a little (at most STRIDE_ADJ) so the
  take-off falls at the right point of the stride;
- ballistic jumps: one gravity (G), the take-off's horizontal speed kept through the air, the take-off foot at or
  before the edge, the landing where the landing drawing's feet meet a surface (a level's feet line or an
  obstacle's top), never forced onto a beat; the flight's drawings by normalized flight time between the clip's
  take-off and land keys; the vertical speed from `rise` (the apex above the take-off, m) or the least that
  clears the gap or the prop (plus CLEAR_M);
- the landing crouch as deep as the fall (the land phase's drawings over an absorb time that grows with the
  fall); a roll after a fall over ROLL_M or a landing faster than ROLL_VY; momentum kept through vaults and hops
  (a small loss, then the run speeds up again); `land: {clips: [[sequence, frame0, frame1], ...], v_out, brake}`
  on a jump or drop (no roll) plays those frames of other clips instead, in order, at their own speed (a deep
  landing crouch and the rise out of it): his speed falls from the landing's to `v_out` (m/s, 2.8) with the time
  constant `brake` (0.12 s), each drawing's lowest foot on the landing surface, and the run takes over from there;
- procedural flips: one tuck drawing turned by the engine between the take-off and land drawings, one direction
  (`dir: front` turns forward: clockwise for a runner going right), the full turn.

Music sync comes from placement, not from bending time: `at` on an item anchors its event (the take-off of a
jump, the hand plant of a vault, the arrival at a run waypoint, a clip's start) to a song time. Between two
anchors the solver finds one speed factor for the span (every gait in it times the same factor), within TOL of
the gaits; a span that needs more is an error that says how much route it is short or long (the world adds or
removes roof, or the route changes), and it is placed at the needed speed so it can be watched. The song's stops
do not pause a parkour move (pause: false), and its photos should be 16ths: real time, no slow motion.

It emits the same Evs as the other moves (wall poses on the feet line while grounded, air skins in flight,
stamps for the splashes), so render, QA and the camera keep working; a move's `skin` turns its wall poses into
the J2 paint skin (the hook moves uses after he leaves the wall). ch.notes[move] gets the solved spans (speed,
factor, slack), flights and landings; ch.pk[q] what the debug overlay draws on each photo.

    - parkour: {from: 51.1, from_x: 420, v0: carry, gait: sprint, to: 58.3, route: [
        {run: {gait: run, x: bench.x0-600, at: 44.2}},          a waypoint (anchored: he is there then)
        {hop: {obstacle: bench, sequence: dash_BD}},
        {vault: {obstacle: meter, sequence: vault_ME}},
        {jump: {gap: alley, sequence: leap_AL, at: 57.607s, rise: 0.4}},
        {drop: {x: 21400, sequence: leap_AL}},                 off an edge onto a lower surface (roll: auto)
        {jump: {gap: alleyP4, rise: 0, land: {clips: [[gapdrop_VG2, 66, 68], [bigland_GL, 43, 46]], v_out: 2.8}}},
                                                                the landing from other clips' frames (below)
        {flip: {edge: 61200, sequence: flip_FL, tuck: FL-048, dir: front, at: 167.31s}},
        {clip: {sequence: getup_FG, from: push, to: run}}]}   a clip in real time
        {clip: {sequence: bracket_P2B, obstacle: bracketP2b, from: takeoff, flight: [takeoff, catch],
                edge: -15385, ground: start}}                    its flight to the bar ballistic (below)

A clip's `flight: [k1, k2]` (v8, P2b's bracket) makes the part of it from key k1 (the take-off) to k2 (the catch)
a ballistic flight: k1's frame with its foot at `edge` (EDGE_PX before it; without an edge where the clip puts it),
k2's frame where the clip's placement puts it (a bar's catch), his horizontal speed his take-off speed (the run's,
when the clip starts at k1) and the time in the air the distance over it; the torso on one gravity's arc between the
two frames' heights, the drawings between them by normalized flight time; the clip as shot (in real time) before k1
and after k2. Keys may be clip frame numbers too (`to: 57`: a hand-over between two clips on any drawn frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .choreo import ChoreoError, Ctx, FLIGHTS
from .moves import template

H_M = 1.8  # his standing height (the library's ref_height), m
G = 9.81  # m/s2, one gravity for the film
GAITS = {"walk": 1.4, "jog": 3.5, "run": 5.5, "sprint": 7.0}  # m/s
CYCLES = {"walk": "walk_WL", "jog": "run_RW", "run": "run_SW", "sprint": "run_SW"}  # default run cycle per gait
A_UP, A_DOWN = 4.0, 6.0  # m/s2: speeding up, braking
TOL = 0.10  # a span's speed factor within 1 +- TOL
STRIDE_ADJ = 0.08  # the approach's steps lengthen or shorten by at most this
TAKEOFF_PHASE = 0.2  # the run cycle's phase (0: a contact; two steps a cycle) where he leaves the ground
ROLL_M, ROLL_VY = 2.5, 7.0  # a roll after a fall over 2.5 m or landing faster than 7 m/s
CLEAR_M = 0.15  # a gap jump's least rise clears the far edge by this
AUTO_CYCLES = ("run_SW", "run_RW", "run_SP", "run_RN")  # `cycle: auto` picks among these (those in the library)
CADENCE_HI, CADENCE_LO, CADENCE_ERR = 1.15, 0.8, 1.35  # a run's cadence over natural: a warning; over 1.35 an error
DRIVE = ("drop_P5D", 43, 70)  # v8b: the drive steps a run starts with after a get-up (out of a deep crouch, rising
#                               with a strong forward lean into four drive steps, 1 to 4.2 m/s, the arms pumping, the
#                               steps lengthening; moves' pick)
DRIVE_V = 3.0  # m/s: a run that starts slower than this right after a clip starts with the drive
DRIVE_GAIN = 1.5  # m/s: ... when its speed is to build by more than this (no drive into a walk or a jog at its speed)


def natural_cadence(v: float) -> float:
    """A runner's natural steps a second at v m/s (PLAN-v8: 3.0 at 3.5 m/s, 3.2 at 5, 3.5 at 7)."""
    return 2.55 + 0.13 * v


HOP_CLEAR_M = 0.25  # a hop's feet clear the prop's top by this at the top of the arc (not a monster jump)
JUMP_RISE_M = 0.45  # a running jump's natural rise (0.4 to 0.6 m at a sprint), unless more is needed or asked
EDGE_PX = 8.0  # the take-off foot this far before the edge (px for a 450 px figure)


@dataclass
class Phase:
    kind: str  # ground | takeoff | flight | absorb | land | roll | vault | clip
    t0: float
    t1: float
    x0: float  # torso x at t0
    v0: float  # horizontal speed at t0, px/s
    data: dict = field(default_factory=dict)

    def x(self, t: float) -> float:
        d = self.data
        tau = min(max(t - self.t0, 0.0), self.t1 - self.t0)
        if self.kind == "ground":
            vt, a = d["vt"], d["a"]
            tr = abs(vt - self.v0) / a if a > 0 else 0.0
            s = math.copysign(a, vt - self.v0)
            if tau <= tr:
                return self.x0 + self.v0 * tau + 0.5 * s * tau * tau
            return self.x0 + self.v0 * tr + 0.5 * s * tr * tr + vt * (tau - tr)
        if self.kind in ("vault", "clip", "land", "roll") and "xs" in d:
            return float(np.interp(tau, d["ts"], d["xs"]))
        return self.x0 + self.v0 * tau

    def v(self, t: float) -> float:
        d = self.data
        if self.kind == "ground":
            tau = min(max(t - self.t0, 0.0), self.t1 - self.t0)
            vt, a = d["vt"], d["a"]
            tr = abs(vt - self.v0) / a if a > 0 else 0.0
            return self.v0 + math.copysign(a, vt - self.v0) * tau if tau <= tr else vt
        if self.kind in ("vault", "clip", "land", "roll") and "xs" in d and len(d["ts"]) > 1:
            dt = 1 / 60
            return (self.x(t + dt) - self.x(t - dt)) / (2 * dt)
        return self.v0


def _ground_time(x0: float, v0: float, vt: float, a: float, L: float) -> tuple[float, float]:
    """Time to cover L px on the ground from speed v0 toward vt (accel a), and the speed at the end."""
    if L <= 0:
        return 0.0, v0
    tr = abs(vt - v0) / a if a > 0 else 0.0
    s = math.copysign(a, vt - v0)
    dr = v0 * tr + 0.5 * s * tr * tr
    if dr >= L:  # still changing speed when it ends
        if abs(s) < 1e-9:
            return L / max(v0, 1e-6), v0
        disc = v0 * v0 + 2 * s * L
        if disc < 0:  # braking to a stop before L (cannot happen for positive vt): crawl at vt
            return L / max(vt, 1e-6), vt
        t = (-v0 + math.sqrt(disc)) / s
        return t, v0 + s * t
    return tr + (L - dr) / max(vt, 1e-6), vt


class Route:
    """The parsed items, the world's geometry they need, and the clips' tracks."""

    def __init__(self, c: Ctx):
        self.c, self.ch, self.lib, self.sc = c, c.ch, c.ch.lib, c.ch.scene
        self.u = c.u
        self.P = self.lib.ref_height / H_M  # px per m
        self.g = G * self.P
        self.gaps = [o for o in self.sc.obstacles.values() if o.kind == "gap"]
        self.items = []
        for k, it in enumerate(c.arg("route") or []):
            if not isinstance(it, dict) or len(it) != 1:
                raise ChoreoError(f"route item {k + 1}: one of run, jump, drop, hop, vault, flip, clip with its args")
            kind, a = next(iter(it.items()))
            a = dict(a or {})
            if kind not in ("run", "jump", "drop", "hop", "vault", "flip", "clip"):
                raise ChoreoError(f"route item {k + 1}: unknown {kind!r} (run, jump, drop, hop, vault, flip, clip)")
            if kind in ("jump", "drop", "hop", "flip") and a.get("flight") is not None and a["flight"] not in FLIGHTS:
                raise ChoreoError(f"route item {k + 1} ({kind}): flight is {' or '.join(FLIGHTS)}, not {a['flight']!r}")
            self.items.append((kind, a))

    # ------------------------------------------------------------------ geometry
    def gait_v(self, gait) -> float:
        if isinstance(gait, int | float):
            return float(gait) * self.P
        if str(gait) not in GAITS:
            raise ChoreoError(f"gait {gait!r}: one of {', '.join(GAITS)} or a speed in m/s")
        return GAITS[str(gait)] * self.P

    def in_gap(self, x: float) -> bool:
        return any(o.x0 < x < o.x1 for o in self.gaps)

    def surface(self, x: float, above: float) -> float | None:
        """The highest surface at x at or below street y `above` (a level's feet line or an obstacle's top
        contact), None over a gap with nothing below it."""
        best = None
        for o in self.sc.obstacles.values():
            if o.kind == "gap" or "top" not in o.contacts:
                continue
            k = o.contacts["top"]
            if k.y is not None and k.x0 is not None and k.x0 <= x <= k.x1 and k.y >= above - 1e-6:
                best = k.y if best is None else min(best, k.y)
        if not self.in_gap(x):
            f = self.sc.feet_at(x)
            if f >= above - 1e-6:
                best = f if best is None else min(best, f)
        return best

    def coord(self, spec, axis="x") -> float:
        return self.c.coord(spec, axis)

    # ------------------------------------------------------------------ clips
    def seq(self, name: str) -> dict:
        try:
            return self.lib.sequence(str(name))
        except KeyError as e:
            raise ChoreoError(str(e.args[0])) from e

    def track(self, seq: dict) -> tuple[dict, list, float]:
        from .motion import video_track
        vid, tr = video_track(self.c, seq)
        if not tr:
            raise ChoreoError(f"sequence {seq.get('video')}: no frames")
        return vid, tr, self.lib.m(tr[0]["name"])["scale"]

    def key(self, seq: dict, name) -> int:
        """A key's clip frame number: a key name, or (v8) a clip frame number itself (one of the drawn frames: a
        hand-over between two clips on any frame)."""
        from .motion import key_frame
        if isinstance(name, int) and not isinstance(name, bool):
            if name not in seq["frames"]:
                near = min(seq["frames"], key=lambda f: abs(f - name))
                raise ChoreoError(f"sequence {seq.get('video')}: no drawn frame {name} (the nearest is {near})")
            return int(name)
        return key_frame(seq, str(name))

    def foot(self, name: str) -> tuple[float, float]:
        """Street px from a drawing's torso (cxc anchor: torso x, bbox middle) to its lowest foot (the bottom
        anchor): where its planted foot is."""
        dx, dy = self.lib.torso_offset(name, "bottom")
        return -dx, -dy


class Solver:
    """The route walked in real time: phases from the start state, with one speed factor per span."""

    def __init__(self, R: Route, t0: float, x0: float, v0: float, gait, t_end: float):
        self.R, self.t0, self.x0, self.v0, self.gait0, self.t_end = R, t0, x0, v0, gait, t_end
        self.anchors = []  # (item index, song time)
        c = R.c
        for i, (kind, a) in enumerate(R.items):
            if kind == "clip" and a.get("keys") is not None and "_pins" not in a:
                R.items[i] = (kind, _clip_pins(R, i, a))
        for i, (kind, a) in enumerate(R.items):
            if a.get("at") is not None:
                q = c.at(a["at"], key=f"route {i + 1} at")
                self.anchors.append((i, c.ch.grid.t(q)))
        self.f = [1.0] * (len(self.anchors) + 1)
        self.tol = TOL
        self.errors: list[str] = []
        self.on_top = None  # (obstacle id, top y, x0, x1) of a prop's top he stands on, else the ground's feet line
        self.clip_flights: list[dict] = []  # (a clip's flight: Solver.clip)

    def span_of(self, i: int) -> int:
        """The span an item's run-up belongs to: the first anchor at or after it."""
        for k, (ia, _) in enumerate(self.anchors):
            if i <= ia:
                return k
        return len(self.anchors)

    def simulate(self, upto: int | None = None) -> tuple[list[Phase], dict]:
        """Walk the route (up to item `upto`'s event if given). Returns the phases and each item's event time."""
        R = self.R
        u = R.u
        t, x, v = self.t0, self.x0, self.v0
        gait = self.gait0
        self.on_top = None
        self.clip_flights = []
        ph: list[Phase] = []
        ev: dict[int, float] = {}
        evx: dict[int, float] = {}

        brake = [None]  # (v8b: a run item's `brake`, m/s2: how hard he slows to its gait, until the next run item)
        accel = [None]  # (v8b: a run item's `accel`, m/s2: how hard he speeds up to its gait, likewise)

        def ground_to(x1: float, k: int, fac_gait, foot_dx: float = 0.0):
            nonlocal t, x, v
            vt = self.f[k] * R.gait_v(fac_gait)
            a = ((accel[0] if accel[0] is not None else A_UP) if vt >= v
                 else (brake[0] if brake[0] is not None else A_DOWN)) * R.P
            dt_, v1 = _ground_time(x, v, vt, a, x1 - x)
            if x1 > x + 1e-6:
                ph.append(Phase("ground", t, t + dt_, x, v, {"vt": vt, "a": a, "gait": fac_gait, "span": k,
                                                             "top": self.on_top}))
                if self.on_top and x1 + foot_dx > self.on_top[3] + 40 * R.u:  # off its top (his planted foot past
                    # its end, not his torso: a take-off drawing's torso is well ahead of its foot): a drop or a jump
                    self.errors.append(f"he runs off {self.on_top[0]}'s top at x {self.on_top[3]:.0f}: give the "
                                       f"route a drop or a jump from its edge")
                    self.on_top = None
            t, x, v = t + dt_, max(x, x1), v1

        for i, (kind, a) in enumerate(R.items):
            k = self.span_of(i)
            if kind == "run":
                if a.get("gait") is not None:
                    gait = a["gait"]
                brake[0] = float(a["brake"]) if a.get("brake") is not None else None
                accel[0] = float(a["accel"]) if a.get("accel") is not None else None
                if a.get("x") is not None:
                    ground_to(R.coord(a["x"]), k, gait)
                ev[i], evx[i] = t, x
            elif kind in ("jump", "drop", "flip", "hop"):
                plan = self.takeoff_plan(i, kind, a, self.f[k] * R.gait_v(gait))
                ground_to(plan["tx"], k, gait, plan["fx"])
                ev[i], evx[i] = t, x
                if upto is not None and i == upto:
                    return ph, ev
                t, x, v = self.flight(ph, i, kind, a, plan, t, x, v)
            elif kind == "vault":
                self.on_top = None
                plan = self.vault_plan(a)
                ground_to(plan["x_in"], k, gait)
                ev[i], evx[i] = t, x
                if upto is not None and i == upto:
                    return ph, ev
                t, x, v = self.vault(ph, a, plan, t, x, v)
            elif kind == "clip":
                self.t_clip = t
                plan = self.clip_plan(a, x)
                if plan["x_in"] is not None:
                    ground_to(plan["x_in"], k, gait)
                kr = self.clip_rate(a, plan, v)
                Ta = next((T_ for ia_, T_ in self.anchors if ia_ == i), None)

                def key_t(v_, plan=plan, x=x, kr=kr):  # the key's time from the clip's start (a key in or after
                    if plan.get("flight") and plan["f_key"] is not None and plan["f_key"] > plan["frames"][  # its
                            plan["flight"]["i1"]]["f"]:  # flight: from the flight, at its own time in the air)
                        ts_, _, _ = self.clip_track(plan, x, v_, kr)
                        ik_ = min(range(len(plan["frames"])),
                                  key=lambda k_: abs(plan["frames"][k_]["f"] - plan["f_key"]))
                        return ts_[ik_]
                    return plan["t_key"] / kr
                if Ta is not None and (plan["boxed"] or a.get("wait")) and t + key_t(0.0) < Ta - 1e-6:
                    # nothing to run before it (he is in the box, or stands still before it): he waits in its first
                    # frame for its key's time (v8c: a key after a real flight, the landing on its impact, timed
                    # through that flight)
                    w_ = Ta - key_t(0.0) - t
                    ph.append(Phase("clip", t, t + w_, x, 0.0, dict(plan, ts=[0.0, w_], xs=[x, x],
                                                                    frames=[plan["frames"][0]] * 2,
                                                                    boxed=[plan["boxed"][0]] * 2 if plan["boxed"]
                                                                    else None)))
                    t, v = t + w_, 0.0
                ev[i], evx[i] = t + key_t(v), x  # (a key in or after the flight: its time from the flight)
                if upto is not None and i == upto:
                    return ph, ev
                t, x, v = self.clip(ph, a, plan, t, x, v, kr)
            if upto is not None and i == upto:
                return ph, ev
        k = len(self.anchors)
        if t < self.t_end:  # on to the move's end at the gait's speed
            vt = self.f[k] * R.gait_v(gait)
            ph.append(Phase("ground", t, self.t_end, x, v, {"vt": vt, "a": (A_UP if vt >= v else A_DOWN) * R.P,
                                                          "gait": gait, "span": k, "top": self.on_top}))
        self.evx = evx
        return ph, ev

    # ------------------------------------------------------------------ jumps
    def takeoff_plan(self, i: int, kind: str, a: dict, v_run: float) -> dict:
        """Where he leaves the ground: the take-off drawing's torso x (its foot at or before the edge)."""
        R = self.R
        seq = R.seq(a.get("sequence") or {"jump": "leap_AL", "drop": "leap_AL", "flip": "flip_FL",
                                           "hop": "dash_BD"}[kind])
        vid, tr, sv = R.track(seq)
        by_f = {r["f"]: r for r in tr}
        f_to = R.key(seq, "takeoff") if "takeoff" in seq.get("keys", {}) else tr[0]["f"]
        r_to = by_f.get(f_to) or min(tr, key=lambda r: abs(r["f"] - f_to))
        fx, fy = R.foot(r_to["name"])
        if kind == "hop":  # take off so the arc's top is over the prop's middle (onto: its near edge)
            o = R.sc.obstacle(str(a["obstacle"]))
            edge = None
            if a.get("onto"):
                vy, tx = self._hop_onto(a, o, seq, tr, r_to, sv, v_run)
            else:
                vy = self._vy_hop(a, o, seq, tr, r_to, sv, v_run)
                tx = o.cx - v_run * vy / R.g
        else:
            if a.get("gap") is not None:
                edge = R.sc.obstacle(str(a["gap"])).x0
            elif a.get("edge") is not None:
                edge = R.coord(a["edge"])
            elif a.get("x") is not None:
                edge = R.coord(a["x"])
            else:
                raise ChoreoError(f"route {i + 1} {kind}: give gap, edge or x (where he leaves the ground)")
            tx = edge - EDGE_PX * R.u - fx
        return {"seq": seq, "vid": vid, "tr": tr, "sv": sv, "r_to": r_to, "edge": edge, "tx": tx, "fx": fx,
                "fy": fy}

    def _vy_hop(self, a: dict, o, seq: dict, tr: list, r_to: dict, sv: float, v_run: float) -> float:
        """A hop's vertical take-off speed: `rise` (m) if asked; else the least (scanned up from 0.1 m) whose arc,
        with the clip's drawings by normalized flight time, keeps every drawing's feet over the prop's x0..x1 at
        least HOP_CLEAR_M above its top (onto: the landing drawing's feet 0.1 m over its near edge, landing on its
        top). The take-off puts the arc's top over the prop's middle (onto: over its near edge)."""
        R = self.R
        if a.get("rise") is not None:
            return math.sqrt(2 * R.g * float(a["rise"]) * R.P)
        top = o.contacts["top"].y if "top" in o.contacts and o.contacts["top"].y is not None else o.y0
        keys = seq.get("keys", {})
        f_to = r_to["f"]
        f_ld = R.key(seq, "land") if "land" in keys else tr[-1]["f"]
        fy_to = R.foot(r_to["name"])[1]
        Y0 = R.sc.feet_at(o.x0 - 1) - fy_to  # his torso at the take-off, on the ground before it
        onto = False
        x_mid = o.cx
        h = 0.1
        for _ in range(80):
            vy = math.sqrt(2 * R.g * h * R.P)
            T = 2 * vy / R.g + 0.05  # (level to level, near enough for the drawings' timing)
            x_to = x_mid - v_run * vy / R.g
            ok = True
            for tau in np.linspace(0, T, 48):
                X = x_to + v_run * tau
                uu = tau / T
                r_ = min(tr, key=lambda rr: abs(rr["f"] - (f_to + uu * (f_ld - f_to))))
                feet = Y0 - vy * tau + 0.5 * R.g * tau * tau + (r_["low"] - r_["y"]) * sv
                if onto:
                    if abs(X - o.x0) <= v_run / 96 and feet > top - 0.1 * R.P:
                        ok = False
                elif o.x0 <= X <= o.x1 and feet > top - HOP_CLEAR_M * R.P:
                    ok = False
            if ok:
                return vy
            h += 0.05
        return math.sqrt(2 * R.g * h * R.P)

    def _onto_geom(self, o, seq: dict, tr: list, r_to: dict) -> dict:
        """A hop onto a prop's top: its top (y, x0, x1), the landing drawing's foot from its torso, the take-off torso
        y on the ground before it, and the landing margins (0.15 m in from the near edge, 0.3 m short of the far)."""
        R = self.R
        k_ = o.contacts["top"]
        top, x0 = float(k_.y), float(k_.x0 if k_.x0 is not None else o.x0)
        x1 = float(k_.x1 if k_.x1 is not None else o.x1)
        f_ld = R.key(seq, "land") if "land" in seq.get("keys", {}) else tr[-1]["f"]
        lfx, lfy = R.foot(min(tr, key=lambda rr: abs(rr["f"] - f_ld))["name"])
        L = x1 - x0
        return {"top": top, "x0": x0, "x1": x1, "lfx": lfx, "lfy": lfy, "c": 0.1 * R.P,
                "Y0": R.sc.feet_at(x0 - 1) - R.foot(r_to["name"])[1],
                "m0": min(0.15 * R.P, 0.25 * L), "m1": min(0.3 * R.P, 0.25 * L)}

    def _onto_times(self, vy: float, H0: float, c: float):
        """When the landing foot (H0 below the top at the take-off) is c over the top rising and falling, and when it
        comes down on it: (ta, tb, tl), or None if the arc does not get c over it."""
        g = self.R.g
        d1 = vy * vy - 2 * g * (H0 + c)
        if d1 < 0:
            return None
        return (vy - math.sqrt(d1)) / g, (vy + math.sqrt(d1)) / g, (vy + math.sqrt(vy * vy - 2 * g * H0)) / g

    def _hop_onto(self, a: dict, o, seq: dict, tr: list, r_to: dict, sv: float, v_run: float) -> tuple[float, float]:
        """A hop onto a prop's top, planned: (vertical speed, take-off torso x). The least rise (`rise` m if asked;
        else scanned up from the landing drawing's foot just 0.15 m over the top's height) whose foot passes the
        top's near edge at least 0.1 m over it and comes down on the top between its margins, as near the edge as
        that allows (a step up, not a hurdle)."""
        R = self.R
        G = self._onto_geom(o, seq, tr, r_to)
        H0 = G["Y0"] + G["lfy"] - G["top"]
        v = max(v_run, 1e-6)
        L = G["x1"] - G["x0"]
        h = float(a["rise"]) * R.P if a.get("rise") is not None else max(H0 + G["c"], 0.0) + 0.05 * R.P
        for _ in range(80):
            vy = math.sqrt(2 * R.g * max(h, 1.0))
            tt = self._onto_times(vy, H0, G["c"])
            if tt is not None:
                ta, tb, tl = tt
                d = max(G["m0"], v * (tl - tb))  # how far in he lands: the edge passed as late as it can be
                if d <= min(v * (tl - ta), L - G["m1"]) or a.get("rise") is not None:
                    te = tl - min(d, v * (tl - ta)) / v
                    return vy, G["x0"] - G["lfx"] - v * te
            if a.get("rise") is not None:
                break
            h += 0.05 * R.P
        vy = math.sqrt(2 * R.g * max(h, 1.0))
        return vy, G["x0"] - G["lfx"] - v * vy / R.g

    def _hop_onto_from(self, o, plan: dict, x: float, v: float, Y0: float):
        """The least vertical speed for a hop onto a prop's top from where he leaves (torso x, speed v, torso y Y0):
        the landing foot passes the near edge 0.1 m over it and comes down between the top's margins. None if no
        arc does (too near, or too fast for the top's length: every arc that clears the edge flies over it)."""
        R = self.R
        G = self._onto_geom(o, plan["seq"], plan["tr"], plan["r_to"])
        H0 = Y0 + G["lfy"] - G["top"]
        te = (G["x0"] - (x + G["lfx"])) / max(v, 1e-6)  # when his landing foot is over the near edge
        if te <= 0:
            return None
        h = max(H0 + G["c"], 0.0)
        for _ in range(120):
            vy = math.sqrt(2 * R.g * max(h, 1.0))
            tt = self._onto_times(vy, H0, G["c"])
            if tt is not None and tt[0] <= te <= tt[1]:
                d = v * (tt[2] - te)
                if d > (G["x1"] - G["x0"]) - G["m1"]:
                    return None  # (higher only lands further)
                if d >= G["m0"]:
                    return vy
            h += 0.02 * R.P
        return None

    def flight(self, ph, i, kind, a, plan, t, x, v):
        """The flight from the take-off (torso at x, speed v) to the landing, then the absorb or the roll."""
        R = self.R
        top_before = self.on_top
        u = R.u
        r_to = plan["r_to"]
        fx, fy = plan["fx"], plan["fy"]
        x_foot = x + fx
        y_feet = self.on_top[1] if self.on_top and self.on_top[2] - 20 <= x_foot <= self.on_top[3] + 20 \
            else R.sc.feet_at(x_foot)  # (a take-off from a prop's top he stands on)
        Y0 = y_feet - fy  # torso y at take-off
        seq = plan["seq"]
        f_land = R.key(seq, "land") if "land" in seq.get("keys", {}) else plan["tr"][-1]["f"]
        by_f = {r["f"]: r for r in plan["tr"]}
        r_ld = by_f.get(f_land) or min(plan["tr"], key=lambda r: abs(r["f"] - f_land))
        air = _air_spec(R, i, kind, a)
        if air:  # (`air`: the drawings in the air from clips' runs of frames; the landing drawing is their last)
            r_ld = air[-1][0][-1]
        lfx, lfy = R.foot(r_ld["name"])
        vx = v
        onto = R.sc.obstacle(str(a["obstacle"])) if kind == "hop" and a.get("onto") else None
        if onto is not None:  # from where he really leaves (the run-up may end past the planned take-off)
            vy = self._hop_onto_from(onto, plan, x, v, Y0) if a.get("rise") is None else None
            if vy is None:
                vy, tx_p = self._hop_onto(a, onto, plan["seq"], plan["tr"], r_to, plan["sv"], v)
                if a.get("rise") is None:
                    self.errors.append(
                        f"route {i + 1} hop onto {onto.id}: from x {x:.0f} at {v / R.P:.1f} m/s no arc comes down on "
                        f"its top (x {onto.x0:.0f} to {onto.x1:.0f}); it needs the take-off at about x {tx_p:.0f}, "
                        f"{x - tx_p:.0f} px before where he is: give it more run-up (or less speed)")
        elif kind == "hop":
            vy = self._vy_hop(a, R.sc.obstacle(str(a["obstacle"])), plan["seq"], plan["tr"], r_to, plan["sv"], v)
        elif a.get("rise") is not None:
            vy = math.sqrt(2 * R.g * float(a["rise"]) * R.P)
        else:
            vy = self._vy_clear(a, plan, x, Y0, vx, lfx, lfy)
        # the landing: the first moment after the top the landing drawing's feet meet a surface
        dt = 1 / 240
        tau, land = 0.0, None
        apex_y = Y0 - vy * vy / (2 * R.g)
        while tau < 6.0:
            tau += dt
            X = x + vx * tau
            Yf = Y0 - vy * tau + 0.5 * R.g * tau * tau + lfy
            if tau > vy / R.g:  # coming down
                s = R.surface(X + lfx, Yf - (vy * dt + R.g * tau * dt) - 2)
                if s is not None and Yf >= s:
                    land = (tau, X, s)
                    break
        if land is None:  # it falls short (an infeasible span's speed): put down past the gap, reported
            far = max((o.x1 for o in R.gaps if o.x0 <= x + vx * 0.3 <= o.x1 + 1), default=x + vx * 0.5)
            T = max((far + 40 * u - x - lfx) / max(vx, 1e-6), 2 * vy / R.g)
            sL = R.surface(far + 40 * u, -1e9) or Y0 + lfy
            land = (T, x + vx * T, sL)
            self.errors.append(f"route {i + 1} {kind}: at {vx / R.P:.1f} m/s it falls short of any surface (put "
                               f"down past {far:.0f})")
        T, XL, sL = land
        # landed on a prop's top (not the ground's feet line): he stands and runs on it until he leaves it
        self.on_top = None
        for o_ in R.sc.obstacles.values():
            k_ = o_.contacts.get("top")
            if o_.kind != "gap" and k_ is not None and k_.y is not None and k_.x0 is not None \
                    and k_.x0 <= XL + lfx <= k_.x1 and abs(k_.y - sL) < 1.0 and abs(R.sc.feet_at(XL + lfx) - sL) > 1.0:
                self.on_top = (o_.id, float(k_.y), float(k_.x0), float(k_.x1))
        if onto is not None and (self.on_top is None or self.on_top[0] != onto.id) and not any(
                f"route {i + 1} hop onto" in m for m in self.errors):
            self.errors.append(f"route {i + 1} hop onto {onto.id}: lands at x {XL + lfx:.0f}, not on its top "
                               f"(x {onto.x0:.0f} to {onto.x1:.0f})")
        fall = (sL - (apex_y + lfy)) / R.P  # m from the top of the arc to the landing surface
        vy_land = (-vy + R.g * T) / R.P
        info = {"i": i, "kind": kind, "t_to": t, "x_to": x, "vx": vx, "vy": vy, "Y0": Y0, "T": T, "x_land": XL,
                "y_land": sL, "fall_m": round(fall, 2), "vy_land": round(vy_land, 2), "apex": apex_y,
                "edge": plan["edge"], "foot_to": x_foot, "lfx": lfx, "lfy": lfy}
        if kind != "hop" and plan["edge"] is not None and x_foot > plan["edge"] + 1e-6:
            info["over_edge"] = x_foot - plan["edge"]
        # how far his feet clear what he jumps (the drawing's own feet on the arc): a hop's prop top, a gap's far edge
        f_to_, f_ld_ = r_to["f"], r_ld["f"]
        clear = None
        if kind == "hop" or a.get("gap") is not None:
            o_ = R.sc.obstacle(str(a["obstacle"] if kind == "hop" else a["gap"]))
            top_ = (o_.contacts["top"].y if "top" in o_.contacts else o_.y0) if kind == "hop" else None
            for tau_ in np.linspace(0, T, 60):
                X_ = x + vx * tau_
                r_ = min(plan["tr"], key=lambda rr: abs(rr["f"] - (f_to_ + tau_ / max(T, 1e-6) * (f_ld_ - f_to_))))
                feet_ = Y0 - vy * tau_ + 0.5 * R.g * tau_ * tau_ + (r_["low"] - r_["y"]) * plan["sv"]
                if kind == "hop" and o_.x0 <= X_ <= o_.x1:
                    c_ = (top_ - feet_) / R.P
                elif kind != "hop" and abs(X_ - o_.x1) <= vx / 120:
                    s_ = R.surface(o_.x1 + 2, -1e9)
                    c_ = (s_ - feet_) / R.P if s_ is not None else None
                else:
                    continue
                if c_ is not None:
                    clear = c_ if clear is None else min(clear, c_)
        if onto is not None:  # onto a top: how far his landing foot is over its near edge as it passes it
            G_ = self._onto_geom(onto, plan["seq"], plan["tr"], r_to)
            te_ = (G_["x0"] - (x + G_["lfx"])) / max(vx, 1e-6)
            clear = (vy * te_ - 0.5 * R.g * te_ * te_ - (Y0 + G_["lfy"] - G_["top"])) / R.P if te_ > 0 else None
        info["clear_m"] = None if clear is None else round(clear, 2)
        info["rise_m"] = round(vy * vy / (2 * R.g) / R.P, 2)
        ph.append(Phase("takeoff", t, t, x, vx, {"r": r_to, "y_feet": y_feet, "plan": plan, "a": a,
                                                 "top": top_before}))
        ph.append(Phase("flight", t, t + T, x, vx, {"plan": plan, "info": info, "a": a, "r_land": r_ld,
                                                    "top": top_before, "top_land": self.on_top, "air": air}))
        t, x = t + T, XL
        roll = a.get("roll", "auto")
        rolled = roll is True or (roll == "auto" and (fall > ROLL_M or vy_land > ROLL_VY))
        loss = 0.10 if rolled else min(0.05 + 0.03 * max(fall, 0.0), 0.3)
        v1 = vx * (1 - loss)
        if rolled:
            rs = R.seq(a.get("roll_sequence", "roll_RL"))
            vid, tr, sv = R.track(rs)
            fa, fb = R.key(rs, "land"), R.key(rs, a.get("roll_to", "run"))  # (v8b: `roll_to`, its last key)
            frs = [r for r in tr if fa <= r["f"] <= fb]
            rate = float(a.get("roll_speed", 1.0))  # (v8b: played this much faster than shot)
            T_r = (fb - fa) / vid["fps"] / max(rate, 1e-6)
            if a.get("roll_v") is not None:  # (v8b, P7: up out of the roll at this speed, slowing evenly to it)
                v_e = min(float(a["roll_v"]) * R.P, v1)
                ts_ = [(r["f"] - fa) / vid["fps"] / max(rate, 1e-6) for r in frs] + [T_r]
                xs_ = [x + v1 * tt - 0.5 * (v1 - v_e) / max(T_r, 1e-6) * tt * tt for tt in ts_]
                ph.append(Phase("roll", t, t + T_r, x, v1, {"frames": frs, "vid": vid, "sv": sv, "y": sL, "ts": ts_,
                                                            "xs": xs_}))
                t, x, v1 = t + T_r, xs_[-1], v_e
            else:
                ph.append(Phase("roll", t, t + T_r, x, v1, {"frames": frs, "vid": vid, "sv": sv, "y": sL}))
                t, x = t + T_r, x + v1 * T_r
            info["roll"] = True
        elif a.get("land"):  # the landing from other clips' frames (a deep crouch and the rise out of it)
            la = a["land"] if isinstance(a["land"], dict) else {"clips": a["land"]}
            frs, ts, t_acc = [], [], 0.0
            for part in la.get("clips") or []:
                if not (isinstance(part, list | tuple) and len(part) == 3):
                    raise ChoreoError(f"route {i + 1} {kind}: land clips are [sequence, first frame, last frame]")
                rs = R.seq(str(part[0]))
                vid_, tr_, sv_ = R.track(rs)
                sel = [r for r in tr_ if int(part[1]) <= r["f"] <= int(part[2])]
                if not sel:
                    raise ChoreoError(f"route {i + 1} {kind}: land: {part[0]} has no frames {part[1]} to {part[2]}")
                for r in sel:
                    frs.append(dict(r, sv=sv_, gv=vid_["ground"]))
                    ts.append(t_acc + (r["f"] - sel[0]["f"]) / vid_["fps"])
                t_acc = ts[-1] + 1 / vid_["fps"]
            vo = float(la.get("v_out", 2.8)) * R.P
            tau_b = max(float(la.get("brake", 0.12)), 1e-3)
            vo = min(vo, v1)
            T_l = t_acc
            xs_ = [x + vo * tt + (v1 - vo) * tau_b * (1 - math.exp(-tt / tau_b)) for tt in ts + [T_l]]
            ph.append(Phase("land", t, t + T_l, x, v1, {"frames": frs, "ts": ts + [T_l], "xs": xs_, "y": sL,
                                                        "top": self.on_top}))
            info["land"] = [str(p_[0]) for p_ in la.get("clips")]
            t, x, v1 = t + T_l, xs_[-1], vo
        else:
            f_run = R.key(seq, "run") if "run" in seq.get("keys", {}) else plan["tr"][-1]["f"]
            frs = [r for r in plan["tr"] if f_land <= r["f"] <= f_run] or [r_ld]
            T_a = float(np.clip(0.10 + 0.05 * max(fall, 0.0), 0.12, 0.40))
            # the deepest crouch of the land phase, held longer the bigger the fall
            ph.append(Phase("absorb", t, t + T_a, x, v1, {"frames": frs, "y": sL, "fall": fall, "top": self.on_top}))
            t, x = t + T_a, x + v1 * T_a
        info["loss"] = loss
        self.flights.append(info)
        return t, x, v1

    def _vy_clear(self, a, plan, x, Y0, vx, lfx, lfy) -> float:
        """The least vertical take-off speed whose arc clears the gap's far edge (and a prop under the arc) by
        CLEAR_M: scanned upward from a running jump's natural rise (JUMP_RISE_M)."""
        R = self.R
        far = None
        if a.get("gap") is not None:
            far = R.sc.obstacle(str(a["gap"])).x1
        h = JUMP_RISE_M
        for _ in range(60):
            vy = math.sqrt(2 * R.g * h * R.P)
            if far is None:
                return vy
            tau = (far + 2 * EDGE_PX * R.u - lfx - x) / max(vx, 1e-6)
            if tau <= 0:
                return vy
            Yf = Y0 - vy * tau + 0.5 * R.g * tau * tau + lfy
            s = R.surface(far + 2 * EDGE_PX * R.u, -1e9)
            if s is None or Yf <= s - CLEAR_M * R.P:
                return vy
            h += 0.05
        return math.sqrt(2 * R.g * h * R.P)

    # ------------------------------------------------------------------ vaults and clips
    def vault_plan(self, a: dict) -> dict:
        R = self.R
        o = R.sc.obstacle(str(a["obstacle"]))
        seq = R.seq(a.get("sequence", "vault_AC"))
        vid, tr, sv = R.track(seq)
        blk = vid.get("block") or None
        keys = seq.get("keys", {})
        f_a = R.key(seq, "plant") if "plant" in keys else tr[0]["f"]
        f_b = R.key(seq, "land") if "land" in keys else tr[-1]["f"]
        lead = int(a.get("lead", 3))  # clip frames shown before the plant (the gather)
        frs = [r for r in tr if r["f"] >= f_a - lead * 2 and r["f"] <= f_b]
        bx0, bx1 = (blk[0], blk[2]) if blk else (frs[0]["x"], frs[-1]["x"])
        if blk:  # it starts on a frame whose torso is still clear of the obstacle (the run-up stays in front of it)
            sx0 = float(np.clip((o.x1 - o.x0) / max(bx1 - bx0, 1e-6), 0.8 * sv, 1.25 * sv))
            clear = [r for r in tr if r["f"] <= f_a and o.x0 + (r["x"] - bx0) * sx0 <= o.x0 - 150 * R.u]
            if clear:
                frs = [r for r in tr if clear[-1]["f"] <= r["f"] <= f_b]
        sx = (o.x1 - o.x0) / max(bx1 - bx0, 1e-6) if blk else sv
        sx = float(np.clip(sx, 0.8 * sv, 1.25 * sv))  # the travel over it fitted to its width (within 25%)
        xs = [o.x0 + (r["x"] - bx0) * sx for r in frs]
        xs = list(np.maximum.accumulate(np.array(xs)))
        gv = vid["ground"]
        f = R.sc.feet_at(o.x0)
        top = o.contacts["top"].y if "top" in o.contacts else o.y0
        off = ((top - f) - ((blk[1] - gv) * sv)) if blk else 0.0
        return {"o": o, "seq": seq, "frames": frs, "xs": xs, "x_in": xs[0], "x_out": xs[-1], "sv": sv, "gv": gv,
                "off": off, "f_a": f_a, "f_b": f_b, "behind": a.get("behind")}

    def vault(self, ph, a, plan, t, x, v):
        vv = v * (1 - float(a.get("loss", 0.05)))
        xs = np.array(plan["xs"])
        ts = (xs - xs[0]) / max(vv, 1e-6)
        T = float(ts[-1])
        ph.append(Phase("vault", t, t + T, x, vv, {"ts": list(ts), "xs": list(xs), "plan": plan}))
        return t + T, float(xs[-1]), vv

    def clip_plan(self, a: dict, x_now: float) -> dict:
        """A clip played in real time (its own fps times `speed`) from key `from` to key `to`, its travel scaled
        to the figure and never back. Placed by `obstacle` (its guide shape on the obstacle: a pole's line, a bar's
        middle, a block's near side) or `x: {key: torso x}`, the run before it running up to its first frame; else
        it starts where he is. `key`: the frame an `at` anchors (default its first). `dy`: px to move the whole clip
        down (a climb cut for a taller wall: its top on the lower roof's level)."""
        R = self.R
        seq = R.seq(a["sequence"])
        vid, tr, sv = R.track(seq)
        fa = R.key(seq, a["from"]) if a.get("from") is not None else tr[0]["f"]
        fb = R.key(seq, a["to"]) if a.get("to") is not None else tr[-1]["f"]
        frs = [r for r in tr if fa <= r["f"] <= fb]
        if not frs:
            raise ChoreoError(f"clip {a['sequence']}: no frames from {a.get('from')} to {a.get('to')}")
        sp = float(a["speed"]) if isinstance(a.get("speed"), int | float) else 1.0
        ts = [(r["f"] - fa) / (vid["fps"] * sp) for r in frs]
        fk = R.key(seq, a["key"]) if a.get("key") is not None else fa
        t_key = (fk - fa) / (vid["fps"] * sp)
        if a.get("_pins"):  # several keys pinned to song times (`keys`): the frames between two pins played to fit
            pf = np.array([p_[0] for p_ in a["_pins"]], float)  # (a stick held, a flight at its own pace), as shot
            pt = np.array([p_[1] for p_ in a["_pins"]], float)  # (times `speed`) before the first and after the last
            if pf[0] < fa or pf[-1] > fb:
                raise ChoreoError(f"clip {a['sequence']}: its keys (frames {int(pf[0])} to {int(pf[-1])}) must lie "
                                  f"within its frames {fa} to {fb}")
            fps_ = vid["fps"] * sp

            def t_of(f_):
                if f_ <= pf[0]:
                    return pt[0] - (pf[0] - f_) / fps_
                if f_ >= pf[-1]:
                    return pt[-1] + (f_ - pf[-1]) / fps_
                return float(np.interp(f_, pf, pt))
            ts = [t_of(r["f"]) - t_of(fa) for r in frs]
            t_key = t_of(pf[0]) - t_of(fa)
        gv = vid["ground"]
        rel = np.array([(r["x"] - frs[0]["x"]) * sv for r in frs])
        x_in, yoff, hold, grip = None, 0.0, None, set()
        yoff_bar = anchor_x = None  # (a bar's height: the bar less the clip's bar over its ground, from the ground)
        if a.get("obstacle") is not None:
            o = R.sc.obstacle(str(a["obstacle"]))
            shapes = vid.get("shapes") or ([vid["block"]] if vid.get("block") else [])
            if not shapes:
                raise ChoreoError(f"clip {a['sequence']}: the clip has no guide shape to put on {o.id}")
            align = a.get("align") or ("pole" if "pole" in o.contacts else "bar" if {"bar", "beam"} & set(o.contacts)
                                       else "block")  # (a slide under a ladder: align: block)
            if align == "pole":  # the tallest, thinnest shape on the pole's line
                sh = max(shapes, key=lambda b: (b[3] - b[1]) / max(b[2] - b[0], 1))
                anchor_x, clip_x = o.contacts["pole"].x, (sh[0] + sh[2]) / 2
                if a.get("beam") is not None and len(shapes) > 1:  # a climb that ends on a beam: the clip's bar on
                    bo = R.sc.obstacle(str(a["beam"]))           # it from its `reach` (or `top`) key on
                    bar = max(shapes, key=lambda b_: (b_[2] - b_[0]) / max(b_[3] - b_[1], 1))
                    yoff_bar = bo.contact("bar", "beam").y - (((bar[1] + bar[3]) / 2) - gv) * sv
                    kk = next((k_ for k_ in ("reach", "top", "hang") if k_ in seq.get("keys", {})), None)
                    hold = (R.key(seq, kk) if kk else fa, fb)
                    grip = {r["f"] for r in frs if abs((r["y"] - r["h"] / 2) - (bar[1] + bar[3]) / 2) < 30}
                    a = dict(a, obstacle=bo.id)
            elif align == "bar":  # the widest, flattest on the bar, at its catch x
                sh = max(shapes, key=lambda b: (b[2] - b[0]) / max(b[3] - b[1], 1))
                bc = o.contact("bar", "beam")
                anchor_x = R.coord(a["catch_x"]) if a.get("catch_x") is not None else o.x0 + (sh[2] - sh[0]) * sv / 2
                clip_x = (sh[0] + sh[2]) / 2
                yoff_bar = bc.y - (((sh[1] + sh[3]) / 2) - gv) * sv
                ka = next((k_ for k_ in ("catch", "grab") if k_ in seq.get("keys", {})), None)
                kb = next((k_ for k_ in ("release",) if k_ in seq.get("keys", {})), None)
                hold = (R.key(seq, ka) if ka else fa, R.key(seq, kb) if kb else fb)
                bar_y = (sh[1] + sh[3]) / 2  # (frames whose top is at the clip's bar: his hands on it)
                grip = {r["f"] for r in frs if abs((r["y"] - r["h"] / 2) - bar_y) < 30}
            else:  # a block: its near side on the obstacle's (the clip's registered block when it has one: v8b, P7's
                sh = vid.get("block") or shapes[0]  # clips list the floor first among their shapes)
                anchor_x, clip_x = o.x0, sh[0]
            x_in = anchor_x + (frs[0]["x"] - clip_x) * sv
            if a.get("place") == "here":  # where the clip before left him (a climb into a swing): the obstacle
                x_in = None               # only sets the bar's height
        elif isinstance(a.get("x"), dict) and a["x"]:
            pins = []
            for kk, xv in a["x"].items():
                fr = R.key(seq, str(kk))
                r = min(frs, key=lambda r_: abs(r_["f"] - fr))
                pins.append((r["x"], R.coord(xv)))
            pins.sort()
            cx_ = np.array([r_["x"] for r_ in frs], dtype=float)
            if len(pins) > 1 and pins[-1][0] > pins[0][0]:  # two keys placed: the travel between them stretched to fit
                (ca, xa_), (cb, xb_) = pins[0], pins[-1]  # (a slide that must clear the whole ledge)
                k_ = (xb_ - xa_) / (cb - ca)
                street = np.where(cx_ < ca, xa_ + (cx_ - ca) * sv, np.where(cx_ > cb, xb_ + (cx_ - cb) * sv,
                                                                           xa_ + (cx_ - ca) * k_))
                rel = street - street[0]
                x_in = float(street[0])
            else:
                x_in = pins[0][1] - (pins[0][0] - frs[0]["x"]) * sv
        if a.get("glide") is not None:  # v8b (the duct slide): from key g1 to key g2 one smooth deceleration
            gl_ = a["glide"]
            if not isinstance(gl_, list | tuple) or len(gl_) != 2:
                raise ChoreoError(f"clip {a['sequence']}: glide: [first key, last key], not {gl_!r}")
            g1, g2 = R.key(seq, gl_[0]), R.key(seq, gl_[1])
            j1 = min(range(len(frs)), key=lambda k_: abs(frs[k_]["f"] - g1))
            j2 = min(range(len(frs)), key=lambda k_: abs(frs[k_]["f"] - g2))
            if not 0 <= j1 < j2 < len(frs):
                raise ChoreoError(f"clip {a['sequence']}: glide {gl_[0]} to {gl_[1]} must lie within its frames, "
                                  f"in order")
            rel = np.array(rel, dtype=float)
            T_, D_ = ts[j2] - ts[j1], float(rel[j2] - rel[j1])
            jb = max(j2 - 2, j1)  # its own pace into g2 (as shot)
            v1_ = max(float(rel[j2] - rel[jb]) / max(ts[j2] - ts[jb], 1e-6), 0.0) if j2 > jb else 0.0
            v0_ = max(2 * D_ / max(T_, 1e-6) - v1_, v1_)  # (the distance over the time fixes the start speed)
            ac_ = (v0_ - v1_) / max(T_, 1e-6)
            for k_ in range(j1 + 1, j2):
                tau = ts[k_] - ts[j1]
                rel[k_] = rel[j1] + v0_ * tau - 0.5 * ac_ * tau * tau
        sway_i = None  # (v8c) `sway: true` (all its frames) or a key (up to its frame): x as shot, back as well
        if a.get("sway") not in (None, False):  # as forward
            if a["sway"] is True:
                sway_i = len(frs) - 1
            else:
                fs_ = R.key(seq, a["sway"])
                sway_i = min(range(len(frs)), key=lambda k_: abs(frs[k_]["f"] - fs_))
        flight = None
        if a.get("flight") is not None:  # v8: from its take-off to its catch a ballistic flight (Solver.clip_track)
            fl_ = a["flight"]
            if not isinstance(fl_, list | tuple) or len(fl_) != 2:
                raise ChoreoError(f"clip {a['sequence']}: flight: [take-off key, catch key], not {fl_!r}")
            f1, f2 = R.key(seq, fl_[0]), R.key(seq, fl_[1])
            if not (frs[0]["f"] <= f1 < f2 <= frs[-1]["f"]):
                raise ChoreoError(f"clip {a['sequence']}: flight {fl_[0]} to {fl_[1]} (frames {f1} to {f2}) must lie "
                                  f"within the clip's frames {frs[0]['f']} to {frs[-1]['f']}, in order")
            i1 = min(range(len(frs)), key=lambda k_: abs(frs[k_]["f"] - f1))
            i2 = min(range(len(frs)), key=lambda k_: abs(frs[k_]["f"] - f2))
            edge = R.coord(a["edge"]) if a.get("edge") is not None else None
            fx_t = R.foot(frs[i1]["name"])[0]
            x_c = x_in + rel[i2] if x_in is not None else None  # the catch where the placement puts it
            if edge is not None:  # the take-off drawing's foot at the edge, the frames before it behind it as shot
                x_in = edge - EDGE_PX * R.u - fx_t - rel[i1]
            flight = {"i1": i1, "i2": i2, "x_c": x_c, "edge": edge, "fx": fx_t, "free": bool(a.get("ballistic"))}
        if yoff_bar is not None:  # the bar's height over the ground the clip's heights stand on (its own `ground`)
            gr_ = a.get("ground")
            if gr_ == "start":
                g_ref = R.sc.feet_at(max(x_now, x_in) if x_in is not None else x_now)
            elif gr_:
                lv_ = next((l_ for l_ in R.sc.levels if l_["id"] == str(gr_)), None)
                if lv_ is None:
                    raise ChoreoError(f"clip ground: no level {gr_!r} (ground.levels: "
                                      f"{', '.join(l_['id'] for l_ in R.sc.levels)})")
                g_ref = float(np.interp(anchor_x, lv_["px"], lv_["py"]))
            else:
                g_ref = R.sc.feet_at(anchor_x)
            yoff = yoff_bar - g_ref
        boxed = None
        if a.get("box") is not None:  # a front-view clip in a box prop: its guide box on the box as it is then
            from .moves import _in_box_place
            pr = self.R.ch.props.get(str(a["box"]))
            if pr is None:
                raise ChoreoError(f"clip: no box prop {a['box']!r}")
            key_ = pr.geometry(self.R.ch.grid.q(self.t_clip if hasattr(self, "t_clip") else self.t0))
            boxed = [_in_box_place(self.R.c, pr, seq, r, key_) for r in frs]
            x_in = boxed[0][0]
        return {"seq": seq, "vid": vid, "sv": sv, "frames": frs, "ts": ts, "rel": rel, "x_in": x_in, "t_key": t_key,
                "yoff": yoff, "dy": float(a.get("dy", 0.0)) * R.u, "hold": hold, "gv": gv, "obstacle": a.get("obstacle"),
                "boxed": boxed, "grip": grip, "flight": flight,
                "behind": a.get("behind"), "ground": a.get("ground"),
                "f_key": fk if a.get("key") is not None else None,
                "sway": sway_i,
                "body": a.get("body"),
                "lift": _lift_spec(a),
                "light_from": R.key(seq, str(a["light_from"])) if a.get("light_from") is not None else None}

    def clip_rate(self, a: dict, plan: dict, v: float) -> float:
        """How much faster than its own pace a clip plays: `speed` (a number: that), or auto (the default for a
        clip that travels): his horizontal speed through it is his speed coming in, less `loss` (5%), so a vault or
        a swing keeps his momentum instead of slowing him to the clip's pace (within `min_rate` to `max_rate`, 0.6 to 2,
        times its own: a step-up shot at 2 m/s may take max_rate: 3.5)."""
        sp = a.get("speed", "auto")
        if sp != "auto" or a.get("_pins"):
            return 1.0  # (a number, or keys pinned to times, is in the clip's timing already: clip_plan)
        D, T0 = float(plan["rel"][-1]) if len(plan["rel"]) else 0.0, plan["ts"][-1] if plan["ts"] else 0.0
        if D < 60 * self.R.u or T0 <= 0 or v <= 0:  # a clip that stays put (the breakout's tumble) plays as shot
            return 1.0
        return float(np.clip(v * (1 - float(a.get("loss", 0.05))) / (D / T0), float(a.get("min_rate", 0.6)),
                             float(a.get("max_rate", 2.0))))

    def clip_track(self, plan: dict, x: float, v: float, kr: float = 1.0) -> tuple[list, list, dict | None]:
        """The clip's frames' times (from its start) and torso x when its first frame is at x and he comes in at
        speed v: as shot (its pace times kr), or with its `flight` ballistic from the take-off frame to the catch
        frame at his take-off speed (the time in the air is the distance over it; the drawings between by
        normalized flight time), the frames after the catch as shot from it."""
        rel = plan["rel"]
        ts0 = [tt / kr for tt in plan["ts"]]
        fl = plan.get("flight")
        if plan["boxed"]:
            return ts0, [b[0] for b in plan["boxed"]], None
        def fwd(xs_):  # never back, but for a clip's frames up to its `sway` (v8c, pipe B: a climber's torso swaying
            k_ = plan.get("sway")  # round the pole it is placed on, as shot from it; never back from there on)
            if k_ is None:
                return list(np.maximum.accumulate(xs_))
            xs_ = np.asarray(xs_, float)
            return list(xs_[:k_ + 1]) + list(np.maximum.accumulate(xs_[k_:])[1:])
        if not fl:
            return ts0, fwd(x + rel), None
        i1, i2 = fl["i1"], fl["i2"]
        x_t, t_t = x + rel[i1], ts0[i1]
        if i1 == 0:  # it starts at the take-off: his speed coming in
            vx = v
        else:  # his pace in the clip into the take-off
            j = max(i1 - 2, 0)
            vx = (rel[i1] - rel[j]) / max(ts0[i1] - ts0[j], 1e-6)
        x_c = fl["x_c"] if fl["x_c"] is not None else x_t + (rel[i2] - rel[i1])
        D = x_c - x_t
        T = D / vx if vx > 1e-6 and D > 0 else max(ts0[i2] - ts0[i1], 1e-3)
        if fl.get("free"):  # `ballistic` (v8 P5's lache): from the take-off frame at its own velocity in the clip
            T, x_c = self._free_flight(plan, i1, i2, x, x_t, ts0, vx)  # (both axes) to where that arc meets a
            D = x_c - x_t                                                # surface: the landing frame there
        u_ = [(plan["ts"][k] - plan["ts"][i1]) / max(plan["ts"][i2] - plan["ts"][i1], 1e-9) for k in range(len(rel))]
        ts, xs = [], []
        for k in range(len(rel)):
            if k <= i1:
                ts.append(ts0[k])
                xs.append(x + rel[k])
            elif k < i2:
                ts.append(t_t + T * u_[k])
                xs.append(x_t + D * u_[k])
            else:
                ts.append(t_t + T + ts0[k] - ts0[i2])
                xs.append(x_c + rel[k] - rel[i2])
        info = {"i1": i1, "i2": i2, "t_t": t_t, "T": T, "vx": vx, "D": D, "x_t": x_t, "x_c": x_c}
        return ts, fwd(np.array(xs)), info

    def _free_flight(self, plan: dict, i1: int, i2: int, x: float, x_t: float, ts0: list, vx: float):
        """A clip's ballistic flight from its take-off frame (a release) at the clip's own velocity there (its last
        two frames into it, as shot), on one gravity, to where the landing frame's feet meet a surface (a level's
        feet line or a prop's top; none over a gap). Returns (time in the air, the landing frame's torso x)."""
        R, frs, sv = self.R, plan["frames"], plan["sv"]
        j = max(i1 - 2, 0)
        dt_ = max(ts0[i1] - ts0[j], 1e-6)
        vy = (frs[i1]["y"] - frs[j]["y"]) * sv / dt_  # (street y down: up is negative)
        _, y_t, _ = _clip_place(R.c, R, dict(plan, xs=[x + plan["rel"][0]]), frs[i1], x_t, x_t, {})
        h_t, h_l = ((frs[k_]["low"] - frs[k_]["y"]) * sv for k_ in (i1, i2))  # torso over feet: take-off, landing
        hit = ballistic_landing(R.surface, x_t, y_t, vx, vy, h_t, h_l, R.g, R.u)
        if hit is None:
            raise ChoreoError(f"clip {plan['seq'].get('label', '')}: its ballistic flight meets no surface within 3 s")
        return hit

    def clip(self, ph, a, plan, t, x, v, kr: float = 1.0):
        ts, xs, cf = self.clip_track(plan, x, v, kr)
        plan = dict(plan, ts=ts, rate=kr)
        T = ts[-1] if ts else 0.0
        if cf is not None:  # the flight (the notes' clip_flights; an error when its foot leaves past the edge or
            fl = plan["flight"]  # it cannot fly: no speed or no distance)
            i_ = next((ia for ia, (_, a_) in enumerate(self.R.items) if a_ is a), None)
            rec = {"i": i_, "kind": "clip", "seq": a["sequence"], "t_to": t + cf["t_t"], "x_to": cf["x_t"],
                   "T": cf["T"], "vx_mps": cf["vx"] / self.R.P, "x_catch": cf["x_c"]}
            if fl["edge"] is not None and cf["x_t"] + fl["fx"] > fl["edge"] + 1e-6:
                rec["over_edge"] = cf["x_t"] + fl["fx"] - fl["edge"]
            if cf["vx"] <= 1e-6 or cf["D"] <= 0:
                self.errors.append(f"route {(i_ or 0) + 1} clip {a['sequence']}: its flight cannot fly (speed "
                                   f"{cf['vx'] / self.R.P:.1f} m/s, {cf['D']:.0f} px to the catch)")
            self.clip_flights.append(rec)
            plan["flight"] = dict(fl, **cf)
        ph.append(Phase("clip", t, t + T, x, v, dict(plan, xs=xs)))
        v_out = (xs[-1] - xs[-3]) / max(ts[-1] - ts[-3], 1e-6) if len(xs) > 2 else v
        return t + T, xs[-1], max(v_out, 0.0)

    # ------------------------------------------------------------------ the spans
    def solve(self) -> list[dict]:
        """One speed factor per span, in order, so each anchored event lands on its time (bisection: a faster
        span reaches its event sooner). Returns the spans' report."""
        rep = []
        for k, (ia, T) in enumerate(self.anchors):
            lo, hi = 0.2, 3.0

            def t_at(fk):
                self.f[k] = fk
                self.flights = []
                _, ev = self.simulate(upto=ia)
                return ev[ia]
            if abs(t_at(hi) - t_at(lo)) < 1e-4:  # no running in the span (a clip waits for its key): no factor
                fk = 1.0
            elif t_at(hi) > T:  # too far even at three times the gait
                fk = hi
            elif t_at(lo) < T:
                fk = lo
            else:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if t_at(mid) > T:
                        lo = mid
                    else:
                        hi = mid
                fk = 0.5 * (lo + hi)
            row = {"span": k, "anchor": ia, "t": round(T, 3), "factor": round(fk, 4)}
            tol = self.tol
            if abs(fk - 1) > tol:  # how much route it is short or long: the time it misses by at the nearest
                fb = 1 + tol if fk > 1 else 1 - tol  # allowed factor, times his speed there
                tb = t_at(fb)
                ph_, _ = self.simulate(upto=ia)
                gl = next((p for p in reversed(ph_) if p.kind == "ground"), None)
                vb = gl.data["vt"] if gl is not None else fb * self.R.gait_v(self.gait0)
                row["miss_s"] = round(T - tb, 3)  # positive: he gets there too early (route too short)
                row["miss_px"] = round((T - tb) * vb)
                if abs(T - tb) < 0.012:  # the nearest allowed speed meets its anchor within 12 ms (a short run
                    fk = row["factor"] = round(fb, 4)  # before a clip barely moves the time): that, not an error
                    row.pop("miss_s"), row.pop("miss_px")
            self.f[k] = fk
            rep.append(row)
        self.flights = []
        self.errors = []
        self.phases, self.ev = self.simulate()
        return rep


SLOPE_MIN = 0.02  # a level's feet line steeper than this (dy/dx, about 1.1 degrees) is a slope (a beam, a ramp)


def _slope_feet(R: Route, x: float, fdx: float, y: float) -> float:
    """A run drawing's feet line on a sloped level (v8 P6's beam, 5.6 degrees): the line at the drawing's lowest
    point (his torso x plus its foot offset), not at his torso, so a trailing toe does not float above the slope nor
    a reaching foot sink into it (tan(slope) times the offset). Flat and gently sloping lines (under SLOPE_MIN) keep
    the torso's (exact on the flat), and so do a foot past the level's end and a stride over an air level."""
    xf = x + fdx
    for lv in reversed(R.sc.levels):
        if lv["x0"] <= x < lv["x1"]:
            if "(air)" in str(lv["id"]) or not (lv["x0"] <= xf < lv["x1"]):  # (an air level over a gap is only the
                return y  # camera's: no ground, as before; P6's beam over its light well is a real one)
            px, py = lv["px"], lv["py"]
            i = int(np.clip(np.searchsorted(px, min(x, xf), side="right") - 1, 0, max(len(px) - 2, 0)))
            j = int(np.clip(np.searchsorted(px, max(x, xf), side="right") - 1, 0, max(len(px) - 2, 0)))
            seg = [abs((py[k + 1] - py[k]) / max(px[k + 1] - px[k], 1e-6)) for k in range(i, j + 1) if k + 1 < len(px)]
            if not seg or max(seg) < SLOPE_MIN:
                return y
            return float(np.interp(xf, px, py))
    return y


def _tuck_turn(R: Route, a: dict, name: str) -> float:
    """How far round the flip a flip step's tuck drawing is drawn (degrees, in the flip's direction): the step's
    `tuck_turn`, else its sequence's `turn` for the key whose drawing the tuck is (poses.spec.json, e.g. flip_FL's
    turn: {tuck: 180}: FL-048 head down, half way round), else 0 (drawn upright)."""
    if a.get("tuck_turn") is not None:
        return float(a["tuck_turn"])
    try:
        seq = R.seq(str(a["sequence"]))
    except Exception:
        return 0.0
    turns = seq.get("turn") or {}
    for key_, deg in turns.items():
        k_ = (seq.get("keys") or {}).get(key_)
        if k_ is not None and 0 <= int(k_) < len(seq.get("drawings") or []) and seq["drawings"][int(k_)] == name:
            return float(deg)
    return 0.0


def _lift_spec(a: dict) -> list:
    """A route clip's `lift: [[first frame, last frame, px], ...]`: those clip frames moved down by px (street px)."""
    out = []
    for k, it in enumerate(a.get("lift") or []):
        if not (isinstance(it, list | tuple) and len(it) == 3):
            raise ChoreoError(f"clip {a.get('sequence')}: lift item {k + 1} is [first frame, last frame, px]")
        out.append((int(it[0]), int(it[1]), float(it[2])))
    return out


def _air_spec(R: Route, i: int, kind: str, a: dict) -> list | None:
    """A jump's `air: [[sequence, first frame, last frame, seconds], ...]` (v8 P6's big leap: flipleap_GF's own
    turn, its opening, then roll_RL's drop): each run of drawings over its seconds, in order, the lot scaled to the
    flight's own time; the last drawing is the landing drawing. None without it."""
    spec = a.get("air")
    if not spec:
        return None
    out = []
    for k, it in enumerate(spec):
        if not (isinstance(it, list | tuple) and len(it) == 4 and float(it[3]) > 0):
            raise ChoreoError(f"route {i + 1} {kind}: air item {k + 1} is [sequence, first frame, last frame, seconds]")
        vid_, tr_, sv_ = R.track(R.seq(str(it[0])))
        sel = [r for r in tr_ if int(it[1]) <= r["f"] <= int(it[2])]
        if not sel:
            raise ChoreoError(f"route {i + 1} {kind}: air: {it[0]} has no drawn frames {it[1]} to {it[2]}")
        out.append((sel, float(it[3])))
    return out


def _air_drawing(air: list, uu: float) -> dict:
    """The drawing at uu (0 to 1) of the flight from the `air` runs (each its share of the time, by frame number)."""
    tot = sum(sec for _, sec in air)
    acc = 0.0
    for k_, (sel, sec) in enumerate(air):
        w = sec / tot
        if uu <= acc + w + 1e-9 or k_ == len(air) - 1:
            v = float(np.clip((uu - acc) / max(w, 1e-9), 0, 1))
            fv = sel[0]["f"] + v * (sel[-1]["f"] - sel[0]["f"])
            return min(sel, key=lambda r_: abs(r_["f"] - fv))
        acc += w
    return air[-1][0][-1]


def _look(a: dict | None) -> str | None:
    """A jump, drop, hop or flip step's own `flight` (skin or paint: v8, the director on P2b v8f: the leap between
    buildings in v7's J2 skin while the rest of the part flies as paint), for its take-off, its strides over the edge
    and its flight; None: the move's or the file's."""
    fl_ = (a or {}).get("flight")
    return fl_ if isinstance(fl_, str) else None


def _clip_pins(R: Route, i: int, a: dict) -> dict:
    """A route clip's `keys: {key: time, ...}` (v8 P5: the stacks, a landing on every kick): its keys as (frame,
    song time) pairs in frame order, the first of them the clip's anchor (its `key` and `at`), the frames between
    two pins played to fit them (Solver.clip_plan)."""
    c = R.c
    kp = a["keys"]
    where = f"route {i + 1} clip {a.get('sequence')}"
    if not isinstance(kp, dict) or len(kp) < 2:
        raise ChoreoError(f"{where}: keys is {{key: time, ...}} with two or more keys")
    if a.get("key") is not None or a.get("at") is not None:
        raise ChoreoError(f"{where}: keys anchors the clip on its first key: drop key and at")
    seq = R.seq(a["sequence"])
    pins = sorted((R.key(seq, k_), c.ch.grid.t(c.at(v_, key=f"{where} keys.{k_}")), k_, v_) for k_, v_ in kp.items())
    fl_ = a.get("flight")  # (a flight after the last pin: its time from the flight, as a release's)
    if fl_ is not None and (not isinstance(fl_, list | tuple) or len(fl_) != 2 or R.key(seq, fl_[0]) < pins[-1][0]):
        raise ChoreoError(f"{where}: with keys, its flight must start at or after its last pinned key (no pin "
                          f"inside the flight: its time comes from the flight)")
    for (f0, t0, k0, _), (f1, t1, k1, _) in zip(pins, pins[1:]):
        if f1 <= f0 or t1 <= t0:
            raise ChoreoError(f"{where}: its keys' times must rise with their frames ({k0} frame {f0} at "
                              f"{t0:.3f} s, {k1} frame {f1} at {t1:.3f} s)")
    return dict(a, _pins=[(p_[0], p_[1]) for p_ in pins], key=pins[0][2], at=pins[0][3])


def _stride(R: Route, seq: dict) -> tuple[list[str], np.ndarray, float]:
    """A run cycle's drawings, their phases (0..1, stride-true travel over the cycle) and the cycle's length in
    street px."""
    vid, tr, sv = R.track(seq)
    names = [r["name"] for r in tr]
    if "travel" in seq and seq.get("period"):
        tf = dict(zip(seq["frames"], seq["travel"]))
        T = np.array([float(tf[r["f"]]) for r in tr])
        period = float(seq["period"])
    else:
        T = np.array([r["x"] - tr[0]["x"] for r in tr])
        period = T[-1] * len(T) / max(len(T) - 1, 1)
    return names, T / period, period * sv


@template("parkour")
def parkour(c: Ctx) -> None:
    """His route by physics (parkour.py): from (time; default right after the previous move), from_x (torso x;
    default where the previous move left him plus its speed), v0 (speed at the start, m/s, or carry: the previous
    move's; default carry, else the gait's), gait (walk, jog, run, sprint, or m/s; sprint), cycle (a run cycle
    sequence per gait: {sprint: run_SW}, or auto: by natural cadence, from `auto_cycles` if given), to (the last photo;
    default up to the next move), tol (0.10: a span's speed
    within 10% of its gaits), route: [items] (the module's docstring). Its photos should be 16ths (real time)."""
    ch, g = c.ch, c.ch.grid
    R = Route(c)
    u = R.u
    q0 = c.time("from") if c.has("from") else None
    if q0 is None:
        if c.prev is None or c.prev.q1 is None:
            raise ChoreoError("needs from: (it is the first move)")
        q0 = c.prev.q1 + ch.step_at(c.prev.q1)
    t0 = g.t(q0)
    gait = c.arg("gait", "sprint")
    vg = R.gait_v(gait)
    v0a = c.arg("v0", "carry")
    if v0a == "carry":
        v0 = c.prev.speed if c.prev is not None and getattr(c.prev, "speed", None) else vg
    else:
        v0 = float(v0a) * R.P
    if c.has("from_x"):
        x0 = c.coord(c.arg("from_x"), "x")
    elif c.prev is not None and c.prev.tx1 is not None:
        x0 = c.prev.tx1 + v0 * (t0 - g.t(c.prev.q1))
    else:
        raise ChoreoError("needs from_x: (nothing before it to start from)")
    if c.has("to"):
        q1 = c.time("to", ref=q0)
    elif c.next is not None and c.next.q0 is not None:
        q1 = c.next.q0 - ch.step_at(c.next.q0)
    else:
        raise ChoreoError("needs to: (nothing after it to run to)")
    tol = float(c.arg("tol", TOL))
    cyc_arg = dict(c.arg("cycle") or {}) if isinstance(c.arg("cycle"), dict) else (
        {gk: c.arg("cycle") for gk in GAITS} if c.arg("cycle") and c.arg("cycle") != "auto" else {})
    if c.arg("cycle") == "auto":  # v8: each run stretch's cycle by its natural cadence (AUTO_CYCLES)
        cyc_arg["_auto"] = True
        if c.has("auto_cycles"):  # the cycles it may pick from (a cycle with a drawing to leave out)
            ac = c.arg("auto_cycles")
            if not isinstance(ac, list) or not ac or not all(str(n) in c.ch.lib.sequences for n in ac):
                raise ChoreoError(f"auto_cycles: a list of run cycles in the library, not {ac!r}")
            cyc_arg["_auto_from"] = [str(n) for n in ac]
    dr_ = c.arg("drive", "auto")
    if dr_ not in (False, "auto") and not (isinstance(dr_, list) and len(dr_) == 3 and str(dr_[0]) in c.ch.lib.sequences):
        raise ChoreoError(f"drive: auto, false or [sequence, first frame, last frame], not {dr_!r}")
    cyc_arg["_drive"] = None if dr_ is False else (tuple(dr_) if isinstance(dr_, list) else DRIVE)
    S = Solver(R, t0, x0, v0, gait, g.t(q1))
    S.tol = tol
    S.flights = []
    spans = S.solve()
    phases = S.phases
    # each span's report: its speed, factor, and how far the route is from fitting its gaits within tol
    out_spans = []
    t_prev, x_prev = t0, x0
    for sp in spans:
        ia = sp["anchor"]
        fk = sp["factor"]
        x_ev = S.evx.get(ia, x_prev)
        T = sp["t"] - t_prev
        D = x_ev - x_prev
        kind, a = R.items[ia]
        row = {"from": round(t_prev, 3), "to": sp["t"], "item": f"{ia + 1} {kind}", "x0": round(x_prev),
               "x1": round(x_ev), "px": round(D), "factor": fk, "ok": abs(fk - 1) <= tol + 1e-6}
        if not row["ok"]:
            miss = sp.get("miss_px", 0)
            row["short_px" if miss > 0 else "long_px"] = abs(miss)
            if not c.dry:
                if miss > 0:
                    msg = (f"span {row['from']:.2f} to {row['to']:.2f} s (to route item {ia + 1}, {kind}): {D:,.0f} px "
                           f"in {T:.2f} s is too little: even {(1 - tol) * 100:.0f}% of its gait gets there "
                           f"{sp['miss_s']:.2f} s early, so add about {miss:,.0f} px of route before it (or start "
                           f"{sp['miss_s']:.2f} s later); placed at {fk:.2f} times its gait")
                else:
                    msg = (f"span {row['from']:.2f} to {row['to']:.2f} s (to route item {ia + 1}, {kind}): {D:,.0f} px "
                           f"in {T:.2f} s is too far: even {(1 + tol) * 100:.0f}% of its gait gets there "
                           f"{-sp['miss_s']:.2f} s late, so remove about {-miss:,.0f} px of route before it (or start "
                           f"{-sp['miss_s']:.2f} s earlier); placed at {fk:.2f} times its gait")
                ch.err(c.mv.where, msg)
        out_spans.append(row)
        t_prev, x_prev = sp["t"], x_ev
    for msg in S.errors if not c.dry else []:
        ch.err(c.mv.where, msg)
    # a solid prop on the running line that no item gets him over: he would run through it
    behind = c.arg("behind") or []
    behind = [behind] if isinstance(behind, str) else list(behind)
    for P in phases if not c.dry else []:
        if P.kind != "ground" or P.t1 - P.t0 < 1e-6:
            continue
        xa, xb = P.x0, P.x(P.t1)
        for o in R.sc.obstacles.values():
            if not o.solid or o.id in behind or o.x1 < xa - 40 * u or o.x0 > xb + 40 * u:
                continue
            if P.data.get("top") and P.data["top"][0] == o.id:  # (running on its top)
                continue
            f_ = R.sc.feet_at(o.cx)
            if xb < o.x0 + 0.25 * (o.x1 - o.x0) or xa > o.x1:  # (his torso stops short of it: a vault starts)
                continue
            if o.y1 >= f_ - 30 * u and o.y0 < f_ - 30 * u:  # it stands on his ground and sticks up
                ch.err(c.mv.where, f"he runs through {o.id} ({o.kind}, x {o.x0:.0f} to {o.x1:.0f}) between "
                                   f"{P.t0:.2f} and {P.t1:.2f} s: give the route a hop, vault or jump over it")
    for fl in S.flights + S.clip_flights:
        if fl.get("over_edge") and not c.dry:
            ch.err(c.mv.where, f"route {fl['i'] + 1}: the take-off foot is {fl['over_edge']:.0f} px past the edge")
    # a clip in the box that breaks out of it (shatter: the box explodes on its `key` frame: the shards and the
    # splatter, the stencils breaking off, a spray of paint, the camera punched and shaken, a light flickering)
    for i, (kind, a) in enumerate(R.items):
        if kind == "clip" and a.get("box") is not None and a.get("shatter") is not None and not c.dry:
            from .props import Shatter
            sh = dict(a["shatter"]) if isinstance(a["shatter"], dict) else {}
            pr = ch.props[str(a["box"])]
            qk = float(round(g.q(S.ev[i])))
            key_ = pr.geometry(qk - 0.01)
            bx0, by0, bx1, by1 = key_.box
            y_a = (by0 + by1) / 2
            sy = y_a - 0.08 * key_.h
            pr.shatter = Shatter(qk, key_.side_x(sy, "right") + key_.thick / 2, sy, 1.0,
                                 int(np.random.default_rng([c.mv.seed, c.mv.local_i, 3]).integers(0, 2 ** 31)),
                                 float(sh.get("fly", 0.65)), float(sh.get("power", 1.0)), int(sh.get("splats", 0)),
                                 float(sh.get("stencils", 0.0)))
            c.stamp(qk, c.pose("spray"), bx1, y_a, 380 * u, flags={"splash": True}, tag="pk.shatter")
            c.cam(qk, punch=float(sh.get("punch", 0.1)), shake=float(sh.get("shake", 0.0)))
            if sh.get("flicker"):
                c.light(qk, str(sh["flicker"]), 0.05)
                c.light(qk + 1, str(sh["flicker"]), 0.5)
            c.label(qk, bx1, by0, "shatter")
    # the paint of a jump (the J2 look): a splash where he leaves the wall and where he lands (splash: its size, 1;
    # false: none), burst: {flecks, splats, power, spread} throws paint from his feet as he leaves, knock: a street
    # light he flies past is shattered as he leaves (pale glass flecks, dark from then on)
    for fl in S.flights if not c.dry else []:
        kind, a = R.items[fl["i"]]
        if kind == "hop" or a.get("splash") is False:
            continue
        k_s = float(a.get("splash", 1.0))
        q_to = g.q(fl["t_to"])
        q_ld = float(math.ceil(g.q(fl["t_to"] + fl["T"]) - 1e-6))
        fx_, fy_ = fl["foot_to"], R.sc.feet_at(fl["foot_to"])
        lx, lf = fl["x_land"], fl["y_land"]
        c.stamp(q_to, c.pose("splash"), fx_ - 40 * u, fy_ - 110 * u, 430 * u * k_s,
                flags={"splash": True, "primer": True}, tag="pk.splash")
        c.stamp(q_to, c.pose("spray"), fx_ - 70 * u, fy_ - 216 * u, 420 * u * k_s, tag="pk.spray")
        c.stamp(q_ld, c.pose("splash"), lx - 110 * u * k_s, lf - 130 * u, 520 * u * k_s, flags={"splash": True},
                tag="pk.land")
        c.stamp(q_ld, c.pose("droplets"), lx + 90 * u * k_s, lf - 280 * u * k_s, 460 * u * k_s, tag="pk.land")
        rs = np.random.default_rng([c.mv.seed, c.mv.local_i, 77, fl["i"]])
        if a.get("burst"):
            b_ = dict(a["burst"]) if isinstance(a["burst"], dict) else {}
            c.burst(q_to, fx_ - 30 * u, fy_ - 90 * u, dirx=1.0, seed=int(rs.integers(0, 2 ** 31)),
                    power=float(b_.get("power", 1.4)), flecks=int(b_.get("flecks", 50)),
                    splats=int(b_.get("splats", 70)), spread=float(b_.get("spread", 50)))
        if a.get("knock"):
            lid = str(a["knock"])
            L = next((lt for lt in R.sc.lights if lt.get("id") == lid), None)
            if L is None:
                raise ChoreoError(f"knock: no street light {lid!r} (obstacles.json lights)")
            c.light(q_to - 1, lid, 2.2)
            c.light_out(q_to, lid)
            c.burst(q_to, float(L["x"]), float(L["y"]), dirx=1.0, seed=int(rs.integers(0, 2 ** 31)), power=0.8,
                    flecks=45, splats=0, spread=40, albedo=0.6)
    # ---------------------------------------------------------------- the photos
    photos = [p for p in ch.photos() if q0 - 1e-6 <= p.q <= q1 + 1e-6]
    if not photos:
        raise ChoreoError(f"no photos from {g.addr(q0)} to {g.addr(q1)}")
    runs = _emit(c, R, S, phases, photos, cyc_arg, gait)
    tail = next((p for p in reversed(phases) if p.kind == "ground"), phases[-1] if phases else None)
    c.st.speed = tail.v(g.t(q1)) if tail is not None else v0
    if not c.dry:
        ch.notes[c.mv.where] = {"spans": out_spans, "start": {"t": round(t0, 3), "x": round(x0), "v_mps":
                                round(v0 / R.P, 2)},
                                "flights": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in fl.items()
                                             if k in ("i", "kind", "t_to", "x_to", "T", "x_land", "fall_m", "vy_land",
                                                      "loss", "roll", "land", "clear_m", "rise_m")} for fl in S.flights],
                                "clip_flights": [{k: (round(v, 2) if isinstance(v, float) else v)
                                                  for k, v in fl.items()} for fl in S.clip_flights],
                                "px_per_m": round(R.P, 1), "gravity_px": round(R.g), "runs": runs}
        for r_ in runs if cyc_arg.get("_auto") else []:  # the cadence (cycle: auto): the legs at a runner's natural
            # rate for his speed
            if r_["ratio"] > CADENCE_ERR:
                ch.err(c.mv.where, f"run {r_['t0']:.2f} to {r_['t1']:.2f} s at {r_['v_mps']:.1f} m/s on {r_['cycle']}: "
                                   f"{r_['cadence']:.2f} steps a second, {r_['ratio']:.2f} times the natural "
                                   f"{r_['natural']:.2f} (legs spinning): a longer-stride cycle or less speed")
            elif r_["ratio"] > CADENCE_HI or r_["ratio"] < CADENCE_LO:
                ch.warn(c.mv.where, f"run {r_['t0']:.2f} to {r_['t1']:.2f} s at {r_['v_mps']:.1f} m/s on "
                                    f"{r_['cycle']}: {r_['cadence']:.2f} steps a second, {r_['ratio']:.2f} times the "
                                    f"natural {r_['natural']:.2f}")


def _settle(c: Ctx, R: Route, r: dict, name: str, x: float, y: float, gv: float, sv: float, x_prev: float | None,
            lift_ok: bool = True) -> tuple[float, float]:
    """Contact for a clip or vault frame: a frame whose feet are on its clip's floor stands with its lowest point on
    the feet line here (not the clip's own floor, a few px off); a frame a solid obstacle would hide steps back
    (feet down, a little, never behind his last place) or is lifted clear (in the air)."""
    from .motion import _hidden
    grounded = float(gv) - float(r["low"]) < 15
    if grounded:
        y = R.sc.feet_at(x) - (float(r["low"]) - float(r["y"])) * sv
    for _ in range(3):
        nh, need_lift, need_back = _hidden(c, name, x, y)
        if not nh:
            break
        if grounded and need_back <= 140 * R.u and (x_prev is None or x - need_back >= x_prev):
            x -= need_back
        elif lift_ok:
            y -= need_lift
        else:
            break
    return x, y


def _solid_under(R: Route, x: float, y: float) -> str | None:
    """The solid obstacle whose top is the surface he stands on or leaves at foot x (within 2 px of y): a level laid
    on a solid box (P2b's awnings), his ground there like a prop's top he landed on (the QA's inside rule)."""
    for o in R.sc.obstacles.values():
        if not o.solid or o.kind == "gap":
            continue
        k = o.contacts.get("top")
        ty = k.y if k is not None and k.y is not None else o.y0
        x0, x1 = (k.x0, k.x1) if k is not None and k.x0 is not None else (o.x0, o.x1)
        if abs(ty - y) <= 2 * R.u and x0 - 20 * R.u <= x <= x1 + 20 * R.u:
            return o.id
    return None



def ballistic_landing(surface, x_t: float, y_t: float, vx: float, vy: float, h_t: float, h_l: float, g: float,
                      u: float = 1.0, t_max: float = 3.0) -> tuple[float, float] | None:
    """Where a ballistic flight lands (Solver._free_flight): the torso from (x_t, y_t) at (vx, vy) on gravity g (street
    y down), h_t and h_l its take-off and landing frames' torso over feet. Coming down, onto a surface below where he
    left (or, once he has risen more than 10 px, below his highest; `surface(x, above)`: the highest at x at or below
    `above`), his landing frame's feet on it, coming onto it from above (v8c, p5c's step-off: a landing frame that stands
    taller than the take-off one is not down at once on the parapet he steps off, its feet already level with it).
    Returns (time in the air, the landing frame's torso x), None when it meets no surface within t_max."""
    yf0 = y_t + h_t
    tt, apex = 0.0, yf0
    prev = y_t + h_l  # (his landing frame's feet where he leaves)
    while tt < t_max:
        tt += 0.002
        xx, yy = x_t + vx * tt, y_t + vy * tt + 0.5 * g * tt * tt
        apex = min(apex, yy + h_t)
        if vy + g * tt > 0:
            above = apex if yf0 - apex > 10 * u else yf0 + 1
            s_ = surface(xx, above)
            if s_ is not None and yy + h_l >= s_ and prev < s_ + 1e-6:
                return tt, xx
        prev = yy + h_l
    return None

def _clip_place(c: Ctx, R: Route, d: dict, r: dict, x: float, last_x: float | None, fl_: dict
                ) -> tuple[float, float, bool]:
    """A clip frame's place (torso x, y; whether its hands grip the bar): on the ground its heights stand on (under
    each frame by default; `start`: the one under its first frame, a climb that crosses onto a higher level keeps
    rising from where it began; or a level's own feet line), moved down by `dy`, a bar's height on the bar it hangs
    from (its hands on it while they grip), a slide's body on the ground, or settled on the feet line and clear of
    solids. Adds the grip to the flags `fl_`."""
    gr = d.get("ground")
    if gr == "start":
        gy = R.sc.feet_at(d["xs"][0])
    elif gr:
        lv = next((l_ for l_ in R.sc.levels if l_["id"] == str(gr)), None)
        if lv is None:
            raise ChoreoError(f"clip ground: no level {gr!r} (ground.levels: "
                              f"{', '.join(l_['id'] for l_ in R.sc.levels)})")
        gy = float(np.interp(x, lv["px"], lv["py"]))
    else:
        gy = R.sc.feet_at(x)
    y = gy + (r["y"] - d["gv"]) * d["sv"]
    y += d.get("dy", 0.0)  # (`dy`: the clip moved down that many px: a climb cut for a taller wall)
    for f0_, f1_, px_ in d.get("lift") or ():  # (`lift`: a run of the clip's frames moved down: its stance on a
        if f0_ <= r["f"] <= f1_:  # raised top drawn standing above the line, P6's precision stick and crate step)
            y += px_
    gripped = False
    if d["hold"]:  # hanging from a bar: the clip's bar on the street's, eased in and out
        fa_, fb_ = d["hold"]
        w = 1.0 if fa_ <= r["f"] <= fb_ else max(0.0, 1 - min(abs(r["f"] - fa_), abs(r["f"] - fb_)) / 6)
        y += d["yoff"] * w
        bo_ = R.sc.obstacles.get(str(d["obstacle"]))
        bc_ = bo_.contact("bar", "beam") if bo_ is not None and ({"bar", "beam"} & set(bo_.contacts)) else None
        on_bar = bc_ is not None and (bc_.x0 is None or bc_.x0 - 20 <= x <= bc_.x1 + 20)
        if fa_ <= r["f"] <= fb_ and r["f"] in d.get("grip", ()) and on_bar:
            fl_["hold"] = d["obstacle"]
            gripped = True
            m_ = c.ch.lib.m(r["name"])  # his hands (the top of his paint, as the grip rule reads it) on it
            sp_ = c.ch.lib.sprite(r["name"])
            ys_ = np.nonzero(sp_ > 0.4)[0]
            t_row = float(np.percentile(ys_, 0.5)) if len(ys_) else m_["bbox"][1]
            top_ = y + (t_row - (m_["bbox"][1] + m_["bbox"][3]) / 2) * m_["scale"]
            y += bc_.y - top_
    if d.get("body") == "ground":  # lying (a slide): the body on the ground line, not just his lowest point
        from .motion import _body_drop
        y += _body_drop(c, r["name"], x, y, 1.0)
    elif not gripped and d.get("obstacle") is not None and not d.get("ground"):
        x, y = _settle(c, R, r, r["name"], x, y, d["gv"], d["sv"], last_x)
    return x, y, gripped


def _emit(c: Ctx, R: Route, S: Solver, phases: list[Phase], photos, cyc_arg: dict, gait0) -> list[dict]:
    """One Ev per photo, sampled from the phases at the photo's time. Returns the run stretches' cadence report."""
    ch, g, lib = c.ch, c.ch.grid, c.ch.lib
    u = R.u
    stride_cache: dict = {}

    def stride_n(name):
        if name not in stride_cache:
            stride_cache[name] = _stride(R, R.seq(name))
        return stride_cache[name]

    def gait_cycle(gk):
        # a gait's cycle: `cycle:` per gait, else the gait's own (a speed in m/s: a walk below 2, a jog below 4.5)
        dflt = CYCLES.get(str(gk)) or ("walk_WL" if float(gk) < 2.0 else "run_RW" if float(gk) < 4.5 else "run_SW")
        return cyc_arg.get(str(gk)) or dflt

    def stride(gk):
        return stride_n(gait_cycle(gk))

    def auto_cycle(v_px):
        """The cycle whose cadence at this speed is nearest a runner's natural one (2.55 + 0.13 v steps a second)."""
        v = v_px / R.P
        if v < 2.0:
            return gait_cycle("walk")
        nat = natural_cadence(v)
        best = None
        for name in cyc_arg.get("_auto_from") or AUTO_CYCLES:
            if name not in lib.sequences:
                continue
            step = stride_n(name)[2] / 2 / R.P
            r_ = abs(math.log(v / max(step, 1e-6) / nat))
            if best is None or r_ < best[0]:
                best = (r_, name)
        return best[1] if best else gait_cycle(v)

    def drive_for(s):
        """(v8b, the director's notes at 107.0 and 158.7: never full speed from a crouch or a mantle) a run stretch that
        starts slow right after a clip (a get-up) and has to build speed starts with the drive: the drive clip's
        drawings (DRIVE, or the move's `drive: [sequence, first, last]`) in real time from the frame where the clip
        is as fast as he is, his torso still where the solver's acceleration puts him; then the cycle takes over in
        the stride phase whose planted foot matches the drive's last drawing. None when it does not apply (or the
        stretch is too short for it)."""
        spec = cyc_arg.get("_drive")
        if spec is None or s["after"] != "clip" or str(spec[0]) not in lib.sequences:
            return None
        v0, vt = s["v0"] / R.P, s["vt"] / R.P
        if v0 >= DRIVE_V or vt - v0 < DRIVE_GAIN:
            return None
        seq = R.seq(str(spec[0]))
        vid, tr_, sv = R.track(seq)
        fa, fb = R.key(seq, spec[1]), R.key(seq, spec[2])
        frs = [r for r in tr_ if fa <= r["f"] <= fb]
        if len(frs) < 3:
            return None
        fps = float(vid["fps"])
        n_ = len(frs)  # its speed frame by frame, over about a step either side (a drive's torso surges each step)

        def sp(i):
            a_, b_ = max(0, i - 3), min(n_ - 1, i + 3)
            return (frs[b_]["x"] - frs[a_]["x"]) * sv / R.P * fps / max(frs[b_]["f"] - frs[a_]["f"], 1)
        i0 = next((i for i in range(n_ - 1) if sp(i) >= v0), n_ - 2)  # from where it is as fast as he is
        frs = frs[i0:]
        T = (frs[-1]["f"] - frs[0]["f"]) / fps
        if s["t1"] - s["t0"] < T + 0.3 or len(frs) < 2:
            return None
        return {"ts": [(r["f"] - frs[0]["f"]) / fps for r in frs], "names": [r["name"] for r in frs], "T": T}

    # the leg phase: the distance travelled on the ground, run stretch by run stretch; each stretch that ends in a
    # take-off has its last steps lengthened or shortened (at most STRIDE_ADJ) so it leaves at TAKEOFF_PHASE
    stretches = []  # [t0, t1, x0, x1, phase0, k_end, window px, gait]
    cur = None
    for k, p in enumerate(phases):
        if p.kind == "ground":
            if cur is None:
                cur = {"t0": p.t0, "x0": p.x0, "ph0": getattr(c.prev, "phase", None) if not stretches and c.prev
                       is not None else 0.0, "gait": p.data["gait"], "after": phases[k - 1].kind if k else None,
                       "v0": p.v0, "vt": p.data["vt"], "P0": p}
                if k == 0 and cur["ph0"] is not None and getattr(c.prev, "tx1", None) is not None:
                    cur["gap"] = p.x0 - c.prev.tx1  # (the move before ended its last photo that far back)
            cur["t1"], cur["x1"] = p.t1, p.x(p.t1)
        else:
            if cur is not None:
                cur["takeoff"] = p.kind in ("takeoff", "vault")
                cur["fx_to"] = R.foot(p.data["r"]["name"])[0] if p.kind == "takeoff" else None
                stretches.append(cur)
                cur = None
    if cur is not None:
        cur["takeoff"] = False
        stretches.append(cur)
    for s in stretches:
        v_ = (s["x1"] - s["x0"]) / max(s["t1"] - s["t0"], 1e-6)  # its mean speed
        s["drive"] = drive_for(s)
        if s["drive"] is not None and v_ < 2.0 * R.P:  # (v8c: a short stretch that is mostly its drive: the cycle
            t_a = s["t0"] + s["drive"]["T"]            # after it by the speed after the drive, not a walk at a run's
            v_ = (s["x1"] - s["P0"].x(t_a)) / max(s["t1"] - t_a, 1e-6)  # pace: p5b's first steps to the wall)
        s["cyc"] = auto_cycle(v_) if cyc_arg.get("_auto") and s["t1"] - s["t0"] > 1e-3 else gait_cycle(s["gait"])
        names, phs, Lc = stride_n(s["cyc"])
        s["Lc"] = Lc
        if s["drive"] is not None:  # the cycle after the drive: in the phase whose planted foot is where the
            dl = s["drive"]["names"][-1]  # drive's last drawing has it (ahead of or behind his torso)
            fx_d = R.foot(dl)[0]
            j_ = int(np.argmin([abs(R.foot(n_)[0] - fx_d) for n_ in names]))
            x_d = s["P0"].x(s["t0"] + s["drive"]["T"])
            s["ph0"] = float(phs[j_]) - (x_d - s["x0"]) / Lc
            s.pop("gap", None)
        if 0 < s.get("gap", 0.0) < 2 * Lc:  # the carried phase advanced over the ground from the move before's
            s["ph0"] = float(s["ph0"]) + s["gap"] / Lc  # last photo to this one's start (no stride held at a join)
        ph0 = float(s["ph0"] or 0.0)
        L = s["x1"] - s["x0"]
        s["k"], s["W"] = 1.0, 0.0
        if s["takeoff"] and L > 0.2 * Lc:
            W = min(3.0 * Lc, L)  # six steps
            nat = ph0 + L / Lc
            # the take-off's phase: his last contact plants the take-off foot where the take-off drawing has it (the
            # contact drawing's foot ahead of its torso, the take-off drawing's behind: the torso travels between)
            tph = TAKEOFF_PHASE
            if s.get("fx_to") is not None:
                fo_c = R.foot(names[int(np.argmin(phs))])[0]
                tph = ((fo_c - s["fx_to"]) / Lc) % 0.5
            cands = [math.floor(nat) + tph + h for h in (-1, -0.5, 0, 0.5, 1)]
            best = min(cands, key=lambda tg: abs(tg - nat))
            # phase gained over the window with its rate scaled by (1 + e * w), w ramping 0 to 1 over its first half
            e = (best - nat) * Lc / (0.75 * W)
            s["k"], s["W"] = float(np.clip(e, -STRIDE_ADJ, STRIDE_ADJ)), W

    def leg_phase(s, x):
        d = x - s["x0"]
        L = s["x1"] - s["x0"]
        ph = float(s["ph0"] or 0.0) + d / s["Lc"]
        if s["W"] > 0:
            a0 = L - s["W"]
            if d > a0:  # in the window: the extra phase gained so far
                y = d - a0
                half = s["W"] / 2
                if y <= half:  # the integral of a weight ramping 0 to 1 over the window's first half
                    gain = y * y / (2 * half)
                else:
                    gain = half / 2 + (y - half)
                ph += s["k"] * gain / s["Lc"]
        return ph

    def stretch_at(t):
        return next((s for s in stretches if s["t0"] - 1e-9 <= t <= s["t1"] + 1e-9), None)

    pk = getattr(ch, "pk", None)
    if pk is None:
        ch.pk = pk = {}
    last_phase = 0.0
    last_x = None
    # the last run photos before a take-off whose body is over the edge already (the take-off drawing's torso is
    # well ahead of its planted foot): the flight's skins from there, as the take-off photo (J2)
    edges = [(p.t0, p.data["plan"].get("edge"), p.data) for p in phases if p.kind == "takeoff"
             and p.data["plan"].get("edge") is not None]
    for p_ in photos:
        t, q = p_.t, p_.q
        P = next((p for p in phases if p.t0 - 1e-9 <= t < p.t1 - 1e-9), None)
        if P is None:
            P = next((p for p in phases if abs(p.t0 - t) < 1e-6 and p.kind == "takeoff"), None)
        if P is None:  # before the first phase or after the last one
            P = phases[-1] if phases and t >= phases[-1].t1 - 1e-9 else (phases[0] if phases else None)
        if P is None:
            continue
        tko = next((p for p in phases if p.kind == "takeoff" and abs(p.t0 - t) < 0.5 / 1000), None)
        info = {"t": round(t, 3), "kind": P.kind, "v_mps": round(P.v(t) / R.P, 2), "gait": str(P.data.get("gait",
                                                                                                        gait0))}
        span = P.data.get("span")
        if span is not None:
            info["span"], info["factor"] = span, round(S.f[span], 4)
            info["ok"] = abs(S.f[span] - 1) <= S.tol + 1e-6
        if tko is not None:  # the take-off photo: the clip's take-off drawing, its foot at the edge
            r = tko.data["r"]
            # (the flight's first skin: leaving the wall, the J2 paint skin in the air, its foot still on the edge)
            fa_ = tko.data.get("a") or {}
            glow = [1.0, 0.55, 0.22]
            if fa_.get("gap") is not None and R.sc.obstacle(str(fa_["gap"])).glow:
                glow = R.sc.obstacle(str(fa_["gap"])).glow
            fl_t = {"kind": "jump", "ground": True, "takeoff": True, "flight": True, "glow": glow,
                    "ghosts": float(fa_.get("ghosts", 3)), "surface": tko.data["y_feet"],
                    "edge": tko.data["plan"].get("edge")}
            if tko.data.get("top"):  # off a prop's top
                fl_t.update(on=tko.data["top"][0], behind=[tko.data["top"][0]])
            else:  # off a level laid on a solid box (an awning): that box is his ground, as a prop's top would be
                sid_ = _solid_under(R, tko.x0 + R.foot(r["name"])[0], tko.data["y_feet"])
                if sid_ is not None:
                    fl_t["on"] = sid_
            e = c._ev(q, "air", r["name"], tko.x0, tko.data["y_feet"], anchor="foot", jitter=False, keeper=True,
                      tag="pk.takeoff", flags=fl_t, flight_look=_look(fa_))
            info.update(foot=[tko.x0 + R.foot(r["name"])[0], tko.data["y_feet"]], surface=tko.data["y_feet"])
        elif P.kind == "ground":
            s = stretch_at(t)
            gk = P.data["gait"]
            names, phs, Lc = stride_n(s["cyc"]) if s is not None else stride(gk)
            x = P.x(t)
            ph = leg_phase(s, x) if s is not None else last_phase
            last_phase = ph
            fr = ph % 1.0
            j = int(np.argmin([min(abs(fr - v), 1 - abs(fr - v)) for v in phs]))
            drv = s.get("drive") if s is not None else None
            if drv is not None and t - s["t0"] < drv["T"] - 1e-6:  # the get-up's drive steps (v8b)
                names = list(names)
                j = len(names)
                names.append(drv["names"][int(np.argmin([abs(t - s["t0"] - v_) for v_ in drv["ts"]]))])
            tp_ = P.data.get("top")  # on a prop's top he landed on (his foot on it), else the ground's feet line
            on_ = tp_ is not None and tp_[2] - 20 <= x + R.foot(names[j])[0] <= tp_[3] + 20
            nx_ = next((p for p in phases if p.t0 >= P.t1 - 1e-9 and p.kind != "ground"), None)
            if tp_ is not None and not on_:  # the strides before a take-off from its top stay on it (a leading foot
                # reaching past its rim)
                on_ = nx_ is not None and nx_.kind == "takeoff" and (nx_.data.get("top") or [None])[0] == tp_[0]
            y0_ = tp_[1] if on_ else R.sc.feet_at(x)  # (the line at his torso; on a slope at his lowest point)
            y = y0_ if on_ else _slope_feet(R, x, R.foot(names[j])[0], y0_)
            if not on_ and nx_ is not None and nx_.kind == "takeoff" and nx_.data["plan"].get("edge") is not None:
                # before a take-off from a level's edge (a step down): his foot is on the upper level, not where
                # his torso is (past the edge); a leading foot reaching past it is in the air
                yu = R.sc.feet_at(min(x + R.foot(names[j])[0], nx_.data["plan"]["edge"] - 1))
                y = yu if abs(yu - y0_) > 0.1 * R.P else y
            elif not on_ and nx_ is not None and nx_.kind == "clip" and nx_.data.get("ground") not in (None, "start") \
                    and nx_.t0 - t < 0.35:  # (v8 P5's stacks) the last strides into a clip that stands on a level
                lv_ = next((l_ for l_ in R.sc.levels if l_["id"] == str(nx_.data["ground"])), None)  # run on it,
                if lv_ is not None:  # not on the ground under his torso (past a step down's edge, where the clip's
                    yu = float(np.interp(x, lv_["px"], lv_["py"]))  # first frames still stand on the upper roof)
                    y = yu if abs(yu - y0_) > 0.1 * R.P else y
            ov_ = next(((ed, dd) for t_to, ed, dd in edges if 0 < t_to - t < 0.25 and x > ed), None)
            over = ov_[0] if ov_ is not None else None
            fl_r = {"kind": "run", "ground": True, "surface": y}
            if on_:  # standing on the prop: its own box is no obstacle to him (the QA's solid and inside rules)
                fl_r.update(on=tp_[0], behind=[tp_[0]])
            if over is not None:  # his body over the edge already: the flight's first skins
                if not on_:  # (off a level laid on a solid box: that box is his ground)
                    sid_ = _solid_under(R, x + R.foot(names[j])[0], y)
                    if sid_ is not None:
                        fl_r["on"] = sid_
                c._ev(q, "air", names[j], x, y, anchor="foot", jitter=False, tag="pk.run",
                      flags=dict(fl_r, flight=True, glow=[1.0, 0.55, 0.22], ghosts=3.0),
                      flight_look=_look(ov_[1].get("a")))
            else:
                c.wall(q, names[j], x, y, anchor="foot", jitter=False, tag="pk.drive" if drv is not None and
                       j == len(names) - 1 and t - s["t0"] < drv["T"] - 1e-6 else "pk.run", flags=fl_r)
            last_x = x
            info.update(foot=[x, y], surface=y, span=span)
        elif P.kind == "flight":
            d = P.data
            fi, plan = d["info"], d["plan"]
            tau = t - P.t0
            uu = tau / max(P.t1 - P.t0, 1e-6)
            X = P.x0 + fi["vx"] * tau
            Y = fi["Y0"] - fi["vy"] * tau + 0.5 * R.g * tau * tau
            f_to, f_ld = plan["r_to"]["f"], d["r_land"]["f"]
            fv = f_to + uu * (f_ld - f_to)
            r = min(plan["tr"], key=lambda r_: abs(r_["f"] - fv))
            if d.get("air"):  # the air's runs of frames, each over its share of the flight, in order
                r = _air_drawing(d["air"], uu)
            rot = 0.0
            name = r["name"]
            if P.data["info"]["kind"] == "flip":
                a = d["a"]
                sgn = -1.0 if str(a.get("dir", "front")) == "front" else 1.0
                w = float(np.clip((uu - 0.08) / 0.84, 0, 1))
                rot = sgn * 360.0 * (w * w * (3 - 2 * w))
                if a.get("tuck") and 0.08 < uu < 0.92:
                    name = str(a["tuck"])
                    # the tuck is drawn part way round the turn (flip_FL's FL-048 head down, half way): it shows
                    # as drawn when the turn reaches its own angle, so the turn reads in phase, no snap at its ends
                    rot -= sgn * _tuck_turn(R, a, name)
                rot = ((rot + 180) % 360) - 180
            glow = [1.0, 0.55, 0.22]
            gp = d["a"].get("gap")
            if gp is not None and R.sc.obstacle(str(gp)).glow:
                glow = R.sc.obstacle(str(gp)).glow
            # the drawing's torso (cxc anchor) on the arc
            fl_f = {"kind": "flight", "flight": True, "glow": glow, "ghosts": float(d["a"].get("ghosts", 3))}
            fx_, fy_ = R.foot(name)
            m_ = lib.m(name)  # his lowest paint (the skin's brushed rim reaches a little past the drawing's box)
            b_ = m_["bbox"]
            lo_ = (max(float(b_[3]), float(m_["bottom"][1])) + 0.015 * (b_[3] - b_[1]) - (b_[1] + b_[3]) / 2) * m_["scale"]
            s_ = R.surface(X + fx_, Y + lo_ - 0.3 * R.P) if abs(rot) < 1 else None
            touch = s_ is not None and Y + lo_ > s_  # a drawing whose feet reach lower than the landing drawing's
            if touch:  # meets the surface a moment early (or still on it at the take-off): on it, not through it
                Y = s_ - lo_
            for tp_ in (d.get("top"), d.get("top_land")):  # just off a prop's top or coming down on it: his feet
                if tp_ and tp_[2] - 20 <= X + fx_ <= tp_[3] + 20 and Y + fy_ <= tp_[1] + 6 * u \
                        and Y + fy_ > tp_[1] - 0.5 * R.P:  # at its rim are no flight through it (QA inside)
                    fl_f.update(on=tp_[0], behind=[tp_[0]])
            if touch or tau < 0.06 or P.t1 - t < 0.06:  # (touching, or a moment off the ground (a trailing foot
                fl_f.update(touch=True)  # still at the rim): the QA's inside rule allows its skin 0.1 m into it)
            c.air(q, name, X, Y, anchor="cxc", rot=rot, jitter=False, tag=f"pk.{fi['kind']}", flags=fl_f,
                  flight_look=_look(d["a"]))
            arc = [(P.x0 + fi["vx"] * tt, fi["Y0"] - fi["vy"] * tt + 0.5 * R.g * tt * tt)
                   for tt in np.linspace(0, P.t1 - P.t0, 24)]
            info.update(arc=[[round(a_, 1), round(b_, 1)] for a_, b_ in arc], flight=fi["i"], clear_m=fi.get("clear_m"),
                        rise_m=fi.get("rise_m"))
        elif P.kind in ("absorb", "roll"):
            d = P.data
            frs = d["frames"]
            uu = (t - P.t0) / max(P.t1 - P.t0, 1e-6)
            x = P.x(t)
            if P.kind == "absorb":
                # the deepest crouch (the lowest torso of the land phase) reached early and held longer for a big fall
                k_low = int(np.argmin([r["low"] - r["y"] for r in frs]))
                hold = float(np.clip(0.2 + 0.1 * d["fall"], 0.2, 0.5))
                if uu < 0.3:
                    fi_ = uu / 0.3 * k_low
                elif uu < 0.3 + hold:
                    fi_ = k_low
                else:
                    fi_ = k_low + (uu - 0.3 - hold) / max(0.7 - hold, 1e-6) * (len(frs) - 1 - k_low)
                r = frs[int(np.clip(round(fi_), 0, len(frs) - 1))]
                fl_l = {"kind": "land", "ground": True, "surface": d["y"]}
                if d.get("top"):  # landed on a prop's top
                    fl_l.update(on=d["top"][0], behind=[d["top"][0]])
                c.wall(q, r["name"], x, d["y"], anchor="foot", jitter=False, tag="pk.land", flags=fl_l)
                info.update(foot=[x, d["y"]], surface=d["y"])
            else:
                r = frs[int(np.clip(round(uu * (len(frs) - 1)), 0, len(frs) - 1))]
                # each drawing's own lowest paint on the landing surface (a roll is in contact all through; a clip's
                # floor can be another level than the one he rolls on: a drop filmed from the higher roof)
                y = d["y"] + (r["y"] - r["low"]) * d["sv"]
                c.wall(q, r["name"], x, y, anchor="cxc", jitter=False, tag="pk.roll",
                       flags={"kind": "roll", "ground": True, "surface": d["y"]})
        elif P.kind == "land":  # the landing from other clips' frames: each drawing's lowest paint on the surface
            d = P.data
            tau = t - P.t0
            k_ = int(np.clip(np.searchsorted(d["ts"], tau + 1e-6) - 1, 0, len(d["frames"]) - 1))
            r = d["frames"][k_]
            x = P.x(t)
            fl_l = {"kind": "land", "ground": True, "surface": d["y"]}
            if d.get("top"):
                fl_l.update(on=d["top"][0], behind=[d["top"][0]])
            # (its lowest paint, as the QA's feet rule measures it, on the surface: drawings from other clips, whose
            # tracked low point and foot anchor can each be a few px off it)
            m_ = lib.m(r["name"])
            ys_ = np.nonzero(lib.sprite(r["name"]) > 0.4)[0]
            off_ = (float(np.percentile(ys_, 99.5)) - lib.anchor(r["name"], "foot")[1]) * m_["scale"]
            c.wall(q, r["name"], x, d["y"] - off_, anchor="foot", jitter=False, tag="pk.land", flags=fl_l)
            info.update(foot=[x, d["y"]], surface=d["y"])
            last_x = x
        elif P.kind == "vault":
            d = P.data
            pl = d["plan"]
            x = P.x(t)
            k_ = int(np.argmin(np.abs(np.array(pl["xs"]) - x)))
            r = pl["frames"][k_]
            f = R.sc.feet_at(x)
            w = 1.0 if pl["f_a"] <= r["f"] <= pl["f_b"] else 0.0
            y = f + (r["y"] - pl["gv"]) * pl["sv"] + pl["off"] * w
            x, y = _settle(c, R, r, r["name"], x, y, pl["gv"], pl["sv"], last_x)
            fv_ = {"kind": "vault", "obstacle": pl["o"].id}
            if pl.get("behind"):
                fv_["behind"] = [pl["behind"]] if isinstance(pl["behind"], str) else list(pl["behind"])
            c.wall(q, r["name"], x, y, anchor="cxc", jitter=False, tag="pk.vault", flags=fv_)
            last_x = x
        elif P.kind == "clip":
            d = P.data
            tau = t - P.t0
            k_ = int(np.argmin(np.abs(np.array(d["ts"]) - tau)))
            r = d["frames"][k_]
            x = P.x(t)
            fl_ = {"kind": "clip", "seq": str(d["seq"].get("video"))}
            if d.get("sway") is not None and k_ <= d["sway"]:  # (v8c) its torso as shot round the pole, back as well
                fl_["back"] = True  # as forward: not a backtrack
            if d.get("light_from") is not None and r["f"] >= d["light_from"]:  # no keepers from that key on: its
                fl_["trail"] = "light"  # paint-outs wash away (a crouch he tears himself out of leaves no figure)
            if d.get("behind"):  # obstacles this item may pass behind (a step into a doorway)
                fl_["behind"] = [d["behind"]] if isinstance(d["behind"], str) else list(d["behind"])
            if d.get("boxed"):  # in the box: its own place and scale from the box (_in_box_place); the wind-up in
                bx_, by_, bs_ = d["boxed"][k_]  # the box is his (it pulls back: the backtrack rule's inside); once
                fk_ = d.get("f_key")            # out, a frame on the clip's floor stands on the street's feet line
                if fk_ is not None and r["f"] < fk_:
                    fl_["kind"] = "inside"
                elif float(d["gv"]) - float(r["low"]) < 15:
                    sc_ = c.ch.lib.m(r["name"])["scale"] * bs_
                    m_ = c.ch.lib.m(r["name"])
                    low_ = (m_["bottom"][1] - (m_["bbox"][1] + m_["bbox"][3]) / 2) * sc_
                    by_ = R.sc.feet_at(bx_) - low_
                c.wall(q, r["name"], bx_, by_, anchor="cxc", scale=bs_, jitter=False, tag="pk.clip", flags=fl_)
                last_x = bx_
                if not c.dry:
                    pk[round(q, 4)] = info
                continue
            cf = d.get("flight")
            if cf and "T" in cf and d["ts"][cf["i1"]] < tau < d["ts"][cf["i2"]]:  # in its flight: the torso on one
                # gravity's arc from the take-off frame's place to the catch frame's (each as the clip places it)
                _, y_t, _ = _clip_place(c, R, d, d["frames"][cf["i1"]], d["xs"][cf["i1"]], None, {})
                _, y_c, _ = _clip_place(c, R, d, d["frames"][cf["i2"]], d["xs"][cf["i2"]], None, {})
                tf_, T_ = tau - d["ts"][cf["i1"]], cf["T"]
                y = y_t + (y_c - y_t) * tf_ / T_ + 0.5 * R.g * tf_ * (tf_ - T_)
                fl_["flight"] = True
            else:
                x, y, _ = _clip_place(c, R, d, r, x, last_x, fl_)
            c.wall(q, r["name"], x, y, anchor="cxc", jitter=False, tag="pk.clip", flags=fl_)
            last_x = x
        if not c.dry:
            pk[round(q, 4)] = info
    # the stride phase where the move ends (the next parkour move carries on from it)
    c.st.phase = last_phase % 1.0
    # the cadence report: each run stretch's speed, cycle, steps a second against the natural, photos a cycle
    runs = []
    for s in stretches:
        T_ = s["t1"] - s["t0"]
        if T_ < 0.3:
            continue
        v_ = (s["x1"] - s["x0"]) / T_
        Lc_ = stride_n(s["cyc"])[2]
        vm = v_ / R.P
        cad = vm / max(Lc_ / 2 / R.P, 1e-6)
        nat = natural_cadence(vm)
        n_ph = sum(1 for p_ in photos if s["t0"] - 1e-9 <= p_.t <= s["t1"] + 1e-9)
        runs.append({"t0": round(s["t0"], 3), "t1": round(s["t1"], 3), "v_mps": round(vm, 2), "cycle": s["cyc"],
                     "step_m": round(Lc_ / 2 / R.P, 2), "cadence": round(cad, 2), "natural": round(nat, 2),
                     "ratio": round(cad / nat, 2) if vm >= 2.0 else 1.0,
                     "photos_per_cycle": round(n_ph / max(T_ * v_ / Lc_, 1e-6), 1)})
    return runs
