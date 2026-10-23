"""The choreography: a hand-edited YAML file that places the runner's poses on the street's obstacles and on the
song's grid. Reference: docs/PAINT.md ("Choreography format").

    street: main            streets/<name>
    seed: 11
    jump: J2                default leap style (J2 free flight, J1 paint-explosion jump cut)
    primer: keepers         all | keepers | none | ghosts (primer patches; see render.py)
    foot_lock: true         runs keep the stance foot planted between beat contacts
    vary: true              repeated single drawings (hang, climb) get small per-photo variations
    hits: {snare: 0.85}     strength thresholds for named hits
    pause: true             a move that spans one of the song's stops holds its pose through it (default true)
    ink: 1.0                the dark paint's albedo multiplier (figure, stencils, cracks; below 1 reads blacker)
    window: [55.06s, 69.0s] default render range
    include: [opening, p6]  other choreographies joined in front of this one's own segments and moves
    segments:               photo rate and camera preset; they tile the choreography without gaps or overlaps
      - {from: 41.1.3, to: 42.4.4, rate: 16ths, camera: stop}
    moves:                  in time order; each is one template with its arguments
      - crouch: {at: 41.1.3, edge: edgeA}
      - freeze: {from: 41.1.3, to: 42.4.4}
      - leap: {gap: alley, at: impact, land_on: snare}
      - run                 (no arguments: fills the time and the street to the next move)

Templates live in moves.py. Placement runs in two passes: a dry pass gives every move its time and street extent,
then each move is placed knowing where the previous one ended and where the next one starts (so a bare `run` can
fill the gap). Every pose becomes an Ev; the renderer, the camera, the map and the check all read the Ev list.
Prop templates (the box) are placed first and stay out of that chain: they paint the set, not the runner.

Song stops (pause): the song's stops that no `freeze` move covers stop the move clock. Templates work in move time,
so a vault planted just before a stop holds its pose through the stop (no new paint, the paint does not dry, the
photos keep coming) and goes on over the top on the stop's end impact. A time given inside such a stop shows at its
end.

Composition (include): a film is one continuous choreography made of segment files. The includer's top-level
settings win, then the first included file's; each file's camera defaults go into its own segments, and each file
keeps its own seed and move numbering for its random choices, so a segment paints the same in the film as alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from .grid import GridError, SongGrid, rate_of
from .poses import PoseLib
from .scene import Obstacle, Scene

TOP_KEYS = {"street", "seed", "jump", "primer", "primer_fade", "foot_lock", "vary", "hits", "window", "segments",
            "moves", "note", "camera", "j1_lead", "poses", "include", "pause", "ink", "motion", "flight"}
COMMON_ARGS = {"wash", "stream", "primer_clip", "poses", "dx", "dy", "scale", "label", "keepers", "note", "pause", "behind", "trail", "still",
               "boost", "ink", "skin", "glow", "grow", "fill_holes", "flight", "trail_in"}
FLIGHTS = ("skin", "paint")  # his look in the air: the J2 paint skin (default) or paint on the wall, primer and all
CAMERA_PRESETS = ("follow", "stop", "wide", "close", "hold", "box", "frame", "spring")


class ChoreoError(ValueError):
    pass


@dataclass
class Ev:
    """One painted thing at one photo: a pose on the wall, a pose in the air, a stamp, or the J1 whip photo."""

    q: float  # 16th index (the photo it first appears in)
    layer: str  # wall | air | stamp | smear
    pose: str | None = None
    x: float = 0.0
    y: float = 0.0
    anchor: str = "foot"
    rot: float = 0.0
    scale: float = 1.0
    w: float | None = None  # stamp width in street px
    flipv: bool = False
    mirror: bool = False
    keeper: bool = False
    jitter: bool = True
    flags: dict = field(default_factory=dict)  # takeoff, burst, halo, splash, primer, flight
    move: int = -1
    tag: str = ""
    contact: tuple | None = None  # (obstacle, contact, anchor[, relation]): on (touches), past, or below (under)


@dataclass
class Segment:
    q0: float
    q1: float
    rate: float  # photo step in 16ths
    camera: dict  # {"preset": ..., params}
    label: str = ""
    join: bool = False  # the first segment of an included file after the first (the camera eases across the join)


@dataclass
class Issue:
    level: str  # error | warn
    where: str
    text: str

    def __str__(self) -> str:
        return f"{self.level.upper():5} {self.where}: {self.text}"


@dataclass
class MoveState:
    """What a placed (or dry-planned) move occupies: its time span and where on the street it starts and ends."""

    q0: float | None = None
    q1: float | None = None
    x0: float | None = None
    x1: float | None = None
    entry_gap: float | None = None  # how far before x0 the previous pose should be
    exit_gap: float | None = None
    cycle: int = 0  # run cycle position after this move (runs continue the cycle)
    # motion M0: the figure's torso x at the move's first and last pose, the travel of its first and last photo,
    # the stride phase of its first and last drawing, and where the move before it should end (its take-off)
    tx0: float | None = None
    tx1: float | None = None
    v_in: float | None = None
    v_out: float | None = None
    ph0: int | None = None
    ph1: int | None = None
    pre_x: float | None = None
    run: bool = False
    speed: float | None = None  # parkour: his horizontal speed at the end, px/s (the next parkour move carries it)
    phase: float | None = None  # parkour: the run cycle's phase at the end (0..1)


@dataclass
class Move:
    i: int
    kind: str
    args: dict
    where: str
    plan: MoveState = field(default_factory=MoveState)
    state: MoveState = field(default_factory=MoveState)
    src: int = 0  # the file it comes from (include): its seed and numbering
    seed: int = 11
    local_i: int = 0  # its index among its own file's moves
    prop: bool = False  # a prop template (the box): placed first, outside the runner's chain


class Clock:
    """The move clock: song 16ths with the song's stops taken out. stops: [(a, b)], a the first photo of a stop, b
    the photo it ends on (its end impact). fwd maps song to move time (a time inside a stop collapses to its start),
    inv maps back (move time at a stop's start shows on its end)."""

    def __init__(self, stops: list[tuple[float, float]] | None = None):
        self.stops = sorted(stops or [])

    def fwd(self, q: float | None) -> float | None:
        if q is None or not self.stops:
            return q
        s = 0.0
        for a, b in self.stops:
            if q >= b - 1e-6:
                s += b - a
            elif q >= a - 1e-6:
                return a - s
            else:
                break
        return q - s

    def inv(self, m: float | None) -> float | None:
        if m is None or not self.stops:
            return m
        s = 0.0
        for a, b in self.stops:
            if m + s >= a - 1e-6:
                s += b - a
            else:
                break
        return m + s

    def inside(self, q: float) -> tuple[float, float] | None:
        """The stop that song time q falls inside (strictly after its first photo), or None."""
        for a, b in self.stops:
            if a + 1e-6 < q < b - 1e-6:
                return a, b
        return None

    def state(self, st: MoveState | None) -> MoveState | None:
        """A neighbour's state in move time."""
        if st is None or not self.stops:
            return st
        return replace(st, q0=self.fwd(st.q0), q1=self.fwd(st.q1))


@dataclass
class Photo:
    k: int  # global photo index in the choreography
    q: float
    t: float
    seg: int


class Choreo:
    def __init__(self, path: Path, grid: SongGrid, scene: Scene, lib: PoseLib, data: dict):
        self.path, self.grid, self.scene, self.lib = path, grid, scene, lib
        self.name = path.stem
        self.data = data
        self.seed = int(data.get("seed", 11))
        self.jump = str(data.get("jump", "J2")).upper()
        self.j1_lead = int(data.get("j1_lead", 1))
        self.primer = str(data.get("primer", "keepers"))
        self.primer_fade = dict(data.get("primer_fade") or {})
        self.foot_lock = data.get("foot_lock", True)
        self.vary = data.get("vary", True)
        self.camera_defaults = dict(data.get("camera") or {})
        # body unit: template geometry is written for a 450 px figure (prototype 6); other projects scale by this
        self.u = lib.ref_height / 450.0
        self.issues: list[Issue] = []
        self.segments: list[Segment] = []
        self.moves: list[Move] = []
        self.events: list[Ev] = []
        self.freezes: list[tuple[float, float]] = []
        self.cam_hints: dict[float, dict] = {}
        self.light_hints: dict[float, dict] = {}  # photo q -> {light id: strength factor} (a flicker)
        self.lights_out: dict[str, float] = {}  # light id -> the song 16th it is knocked out (dark from then on)
        self.screens_off: dict[str, tuple[float, float]] = {}  # screen group -> (song 16th it goes dark, fade 16ths)
        self.bursts: list = []  # props.Burst: paint explosions without a box (a take-off, a light shattered)
        self.extra_photos: list[float] = []
        self.labels: list[tuple[float, float, float, str]] = []
        self.hit_contacts: list[tuple[float, float, str]] = []  # (q, hit time, where): for the on-grid check
        self.window: tuple[float, float] | None = None
        self.pause = bool(data.get("pause", True))
        self.ink = float(data.get("ink", 1.0))  # dark paint's albedo multiplier (below 1: closer to black)
        self.motion = str(data.get("motion", "m0"))  # m0 (the motion overhaul) or legacy (prototype 6's placements)
        self.flight = str(data.get("flight", "skin"))  # v8: skin (J2, the default) or paint (the wall part's flights)
        if self.flight not in FLIGHTS:
            raise ChoreoError(f"flight: {self.flight!r}, one of {', '.join(FLIGHTS)}")
        self.clock = Clock()  # identity until the song's stops are known (load)
        self.stops: list[tuple[float, float]] = []  # the song stops that pause moves (song 16ths)
        self._active: Clock | None = None  # the clock of the template being placed (step_at reads move time)
        self.props: dict = {}  # prop id -> prop (props.BoxProp): the set the templates paint
        self.notes: dict = {}  # what templates measured (a video run's step, px per photo, foot slip)
        self.sources: list[dict] = list(data.get("_sources") or [{"name": path.stem, "seed": self.seed}])

    # ------------------------------------------------------------------ issues
    def err(self, where: str, text: str) -> None:
        self.issues.append(Issue("error", where, text))

    def warn(self, where: str, text: str) -> None:
        self.issues.append(Issue("warn", where, text))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    # ------------------------------------------------------------------ grid helpers
    def step_at(self, q: float) -> float:
        if self._active is not None:  # inside a template: q is move time
            q = self._active.inv(q)
        for s in self.segments:
            if s.q0 - 1e-6 <= q < s.q1 - 1e-6:
                return s.rate
        return 1.0

    def segment_at(self, q: float) -> int | None:
        for i, s in enumerate(self.segments):
            if s.q0 - 1e-6 <= q < s.q1 - 1e-6:
                return i
        return None

    @property
    def q_start(self) -> float:
        return self.segments[0].q0 if self.segments else 0.0

    @property
    def q_end(self) -> float:
        return self.segments[-1].q1 if self.segments else 0.0

    def photos(self) -> list[Photo]:
        """Every photo of the choreography: each segment's steps, plus the extra photos templates asked for."""
        qs = []
        for si, s in enumerate(self.segments):
            for q in self.grid.steps(s.q0, s.q1, s.rate):
                qs.append((q, si))
        have = {round(q, 4) for q, _ in qs}
        for q in self.extra_photos:
            if round(q, 4) not in have and self.segment_at(q) is not None:
                qs.append((q, self.segment_at(q)))
                have.add(round(q, 4))  # a key and a stencil on the same word ask for the same photo
        qs.sort()
        return [Photo(k, q, self.grid.t(q), si) for k, (q, si) in enumerate(qs)]

    def paint_clock_back(self, t: float, dt: float) -> float:
        """The song time dt seconds of paint clock before t (the freezes do not count)."""
        cur, left = t, dt
        for q0, q1 in sorted(self.freezes, reverse=True):
            a, b = self.grid.t(q0), self.grid.t(q1) - 0.004
            if a >= cur:
                continue
            if b < cur:
                if cur - b >= left:
                    return cur - left
                left -= cur - b
            cur = a
        return cur - left

    def paint_time(self, t0: float, dt: float) -> float:
        """The song time when dt seconds of paint clock have passed after song time t0 (freezes do not count)."""
        t, left = t0, dt
        for q0, q1 in sorted(self.freezes):
            a, b = self.grid.t(q0), self.grid.t(q1) - 0.004
            if b <= t:
                continue
            if a > t:
                if a - t >= left:
                    return t + left
                left -= a - t
            t = max(t, b)
        return t + left

    def paint_clock(self, t: float) -> float:
        """Song time with the freezes taken out (paint does not dry, drip or wash while the song stops)."""
        out = t
        for q0, q1 in self.freezes:
            a, b = self.grid.t(q0), self.grid.t(q1) - 0.004
            out -= max(0.0, min(t, b) - a)
        return out


# ---------------------------------------------------------------------------------------------- the move context

class Ctx:
    """What a template sees: its arguments, the grid, the scene, the poses, and helpers that place Evs."""

    def __init__(self, ch: Choreo, mv: Move, prev: MoveState | None, nxt: MoveState | None, dry: bool,
                 feet: float | None = None):
        # the template works in move time (song time without the stops it pauses through; identity without them)
        # (a parkour move runs in real time through the song's stops unless it says pause: true)
        self.clock = ch.clock if mv.args.get("pause", ch.pause and mv.kind != "parkour") else Clock()
        prev, nxt = self.clock.state(prev), self.clock.state(nxt)
        self.ch, self.mv, self.prev, self.next, self.dry = ch, mv, prev, nxt, dry
        self.args = dict(mv.args)
        self.used: set[str] = set(COMMON_ARGS)
        self.out: list[Ev] = []
        self.st = MoveState()
        self.feet = ch.scene.feet if feet is None else feet  # the ground line where this move happens
        self.u = ch.u
        self._ref = prev.q1 if prev is not None and prev.q1 is not None else (prev.q0 if prev else None)

    # -------------------------------------------------------------- arguments
    def arg(self, key: str, default=None):
        self.used.add(key)
        return self.args.get(key, default)

    def has(self, key: str) -> bool:
        self.used.add(key)
        return key in self.args and self.args[key] is not None

    def require(self, key: str):
        self.used.add(key)
        if key not in self.args or self.args[key] is None:
            raise ChoreoError(f"needs {key}:")
        return self.args[key]

    def time(self, key: str, default=None, ref: float | None = None, snap: bool = True) -> float | None:
        """Resolve a time argument to a 16th index (snapped to the 16th grid unless snap=False). Named hits are
        remembered for the on-grid check."""
        self.used.add(key)
        return self.at(self.args.get(key, default), ref, snap, key)

    def at(self, spec, ref: float | None = None, snap: bool = True, key: str = "time") -> float | None:
        """Resolve any time spec (one nested in an argument, such as a box key's `at`) to move time."""
        if spec is None:
            return None
        r = ref if ref is not None else self._ref
        rs = self.clock.inv(r)  # the reference in song time
        try:
            q, hit_t = self.ch.grid.resolve(spec, rs)
        except GridError as e:
            raise ChoreoError(f"{key}: {e}") from e
        if isinstance(spec, str) and spec.strip()[:1] in ("+", "-"):  # relative: counted in move time
            qm = r + (q - rs)
            return float(round(qm) if snap else qm)
        qs = round(q) if snap else q
        if hit_t is not None and not self.dry:
            self.ch.hit_contacts.append((qs, hit_t, f"{self.mv.where} {key}: {spec}"))
        if not self.dry and self.clock.inside(qs):
            a, b = self.clock.inside(qs)
            self.ch.warn(self.mv.where, f"{key}: {self.ch.grid.addr(qs)} is inside the song stop {self.ch.grid.addr(a)} "
                                        f"to {self.ch.grid.addr(b)}: it shows on the stop's end")
        return float(self.clock.fwd(qs))

    def ob(self, key: str, kinds: tuple[str, ...] | None = None) -> Obstacle:
        oid = self.require(key)
        try:
            o = self.ch.scene.obstacle(str(oid))
        except KeyError as e:
            raise ChoreoError(f"{key}: {e.args[0]}") from e
        if kinds and o.kind not in kinds:
            self.ch.warn(self.mv.where, f"{key}: {oid} is a {o.kind}, expected {' or '.join(kinds)}")
        return o

    def coord(self, spec, axis: str) -> float:
        """A coordinate: a number, 'feet' (the ground line), or '<obstacle>.<x0|x1|cx|y0|y1|contact>[+-N]'."""
        if isinstance(spec, int | float):
            return float(spec)
        m = re.match(r"^\s*([A-Za-z_][\w]*)(?:\.([A-Za-z_]\w*))?\s*([+\-]\s*\d+(?:\.\d+)?)?\s*$", str(spec))
        if not m:
            raise ChoreoError(f"bad coordinate {spec!r}")
        name, fld, off = m.group(1), m.group(2), m.group(3)
        d = float(off.replace(" ", "")) if off else 0.0
        if name == "feet" and fld is None:
            return self.feet + d
        if name in self.ch.props:
            try:
                return float(self.ch.props[name].coord(fld, axis)) + d
            except KeyError as e:
                raise ChoreoError(f"coordinate {spec!r}: {e.args[0]}") from e
        try:
            o = self.ch.scene.obstacle(name)
        except KeyError as e:
            raise ChoreoError(f"coordinate {spec!r}: {e.args[0]}") from e
        if fld is None:
            return (o.cx if axis == "x" else (o.y0 + o.y1) / 2) + d
        if fld in ("x0", "x1", "y0", "y1", "cx"):
            return getattr(o, fld) + d
        if fld in o.contacts:
            c = o.contacts[fld]
            v = c.x if axis == "x" and c.x is not None else c.y if axis == "y" and c.y is not None else None
            if v is None:
                v = (c.x0 + c.x1) / 2 if axis == "x" and c.x0 is not None else (c.y0 + c.y1) / 2
            return float(v) + d
        raise ChoreoError(f"coordinate {spec!r}: {name} has no {fld}")

    def x_after_prev(self, gap: float) -> float | None:
        """The previous move's last x plus gap; None while planning when the previous move has no x yet."""
        if self.prev is not None and self.prev.x1 is not None:
            return self.prev.x1 + gap
        if self.dry:
            return None
        raise ChoreoError("needs x: (nothing before it to start from)")

    @property
    def step(self) -> float:
        return self.ch.step_at(self.st.q0 if self.st.q0 is not None else (self._ref or 0.0))

    # -------------------------------------------------------------- poses
    def pose(self, role: str, i: int = 0) -> str:
        over = (self.args.get("poses") or {}).get(role)
        try:
            names = self.ch.lib.role(role, over)
        except KeyError as e:
            raise ChoreoError(str(e.args[0])) from e
        return names[i % len(names)]

    def poses(self, role: str) -> list[str]:
        over = (self.args.get("poses") or {}).get(role)
        try:
            return self.ch.lib.role(role, over)
        except KeyError as e:
            raise ChoreoError(str(e.args[0])) from e

    # -------------------------------------------------------------- placing
    def _ev(self, q: float, layer: str, pose: str | None, x: float, y: float, **kw) -> Ev:
        dx, dy = float(self.args.get("dx", 0.0)), float(self.args.get("dy", 0.0))
        sk = self.args.get("skin")
        fm = str(kw.pop("flight_look", None) or self.args.get("flight", self.ch.flight))  # (a route step's own wins)
        if fm not in FLIGHTS:
            raise ChoreoError(f"flight: {fm!r}, one of {', '.join(FLIGHTS)}")
        if layer == "air" and pose is not None and fm == "paint" and sk in (None, False):
            # v8 (the director on P2b): in the air he stays paint on the wall, his primer patch behind him, as he is
            # running (no J2 skin); a move's `skin` still takes him off the wall (the free section after the tear-off)
            layer = "wall"
        if sk not in (None, False) and layer == "wall" and pose is not None and \
                (sk is True or q >= self.at(sk, key="skin") - 1e-6):
            # skin: true (or a time: from then on): off the wall for good, the J2 skin in the air (render's skin: a soft
            # shadow on the wall behind him, a trail of pale skins), never paint on the wall (the free section)
            layer = "air"
            fl = dict(kw.pop("flags", None) or {}, skin=True, ghosts=float(self.args.get("ghosts", 3)))
            if self.args.get("glow"):
                fl["glow"] = [float(v) for v in self.args["glow"]]
            kw["flags"] = fl
        gr = self.args.get("grow")
        if gr and layer == "wall" and pose is not None:
            # grow: [time, n]: back on the wall at `time` (a landing), the primer under its next n wall poses arrives
            # gradually, small round him and growing (patch_grow, as the J2 leap's landing)
            g_at, g_n = gr if isinstance(gr, list | tuple) else (None, gr)
            if g_at is None or q >= self.at(g_at, key="grow") - 1e-6:
                self._grow_k = getattr(self, "_grow_k", 0) + 1
                if self._grow_k <= int(g_n):
                    kw["flags"] = dict(kw.pop("flags", None) or {}, patch_grow=[self._grow_k, int(g_n) + 1])
        kp = self.args.get("keepers")
        if kp not in (None, False) and layer == "wall" and pose is not None:
            # keepers: true (every wall pose of the move stays as a pale ghost) or N (every Nth): a longer trail
            self._keep_k = getattr(self, "_keep_k", 0) + 1
            if kp is True or self._keep_k % max(int(kp), 1) == 0:
                kw["keeper"] = True
        e = Ev(q=float(self.clock.inv(q)), layer=layer, pose=pose, x=float(x) + dx, y=float(y) + dy, move=self.mv.i,
               **kw)
        if self.args.get("behind"):  # obstacles this move may pass behind (the QA lets it)
            b = self.args["behind"]
            e.flags = dict(e.flags, behind=[b] if isinstance(b, str) else list(b))
        if self.args.get("trail") and layer in ("wall", "air"):  # trail: light: its ghosts wash away completely
            e.flags = dict(e.flags, trail=str(self.args["trail"]))
        if self.args.get("still") and layer in ("wall", "air"):  # a deliberate stillness: holds are not flagged
            e.flags = dict(e.flags, still=True)
        wa = self.args.get("wash")
        if wa is not None and wa is not True and layer == "wall" and pose is not None:  # (v8c: no rain wash on its
            if wa is False or (isinstance(wa, dict) and q >= self.at(wa.get("from", 0), key="wash.from") - 1e-6):
                e.flags = dict(e.flags, nowash=True)  # paint: a pose held as paint stays crisp, p5c's crouch)
        pc = self.args.get("primer_clip")
        if pc is not None and layer in ("wall", "stamp") and pose is not None:
            # (v8c, p5e: his swath trimmed so the step out of the wall leaves it) the primer this move lays (its poses'
            # patches and paint-outs, a stamp's primer) kept inside [x0, y0, x1, y1] (null: open on that side)
            if not isinstance(pc, list | tuple) or len(pc) != 4:
                raise ChoreoError(f"primer_clip: [x0, y0, x1, y1] (null for an open side), not {pc!r}")
            bx = [(-1e9 if k_ < 2 else 1e9) if v_ is None else float(v_) for k_, v_ in enumerate(pc)]
            e.flags = dict(e.flags, primer_clip=bx)
        if self.args.get("boost") and layer in ("wall", "air"):  # its paint denser: [gain, gamma]
            e.flags = dict(e.flags, boost=[float(v) for v in self.args["boost"]])
        if self.args.get("fill_holes") and layer in ("wall", "air"):  # its matte's interior holes up to N px closed
            e.flags = dict(e.flags, fill_holes=int(self.args["fill_holes"]))
        if self.args.get("ink") and layer == "wall":  # its dark paint's albedo times this (below 1: blacker)
            e.flags = dict(e.flags, ink=float(self.args["ink"]))
        if layer in ("wall", "air") and "scale" in self.args:
            e.scale *= float(self.args["scale"])
        if pose is not None and pose not in self.ch.lib:
            raise ChoreoError(f"unknown pose {pose!r}")
        self.out.append(e)
        q0 = self.st.q0
        self.st.q0 = q if q0 is None else min(q0, q)
        self.st.q1 = q if self.st.q1 is None else max(self.st.q1, q)
        if layer in ("wall", "air") and pose is not None:
            if self.st.x0 is None:
                self.st.x0 = e.x
            self.st.x1 = e.x
        return e

    def wall(self, q, pose, x, y=None, anchor="foot", **kw) -> Ev:
        return self._ev(q, "wall", pose, x, self.feet if y is None else y, anchor=anchor, **kw)

    def air(self, q, pose, x, y, anchor="center", **kw) -> Ev:
        return self._ev(q, "air", pose, x, y, anchor=anchor, **kw)

    def stamp(self, q, pose, x, y, w, **kw) -> Ev:
        return self._ev(q, "stamp", pose, x, y, anchor="center", w=float(w), **kw)

    def smear(self, q) -> Ev:
        return self._ev(q, "smear", None, 0, 0)

    def freeze(self, q0: float, q1: float) -> None:
        if not self.dry:
            self.ch.freezes.append((self.clock.inv(q0), self.clock.inv(q1)))
        self.st.q0 = q0 if self.st.q0 is None else min(self.st.q0, q0)
        self.st.q1 = q1 if self.st.q1 is None else max(self.st.q1, q1)

    def cam(self, q: float, **hint) -> None:
        if not self.dry:
            self.ch.cam_hints.setdefault(round(self.clock.inv(q), 4), {}).update(hint)

    def light(self, q: float, light: str, factor: float) -> None:
        """Dim (or boost) one of the street's lights on the photo at q: its glow on the wall and paint, and the lamp
        in its parallax layer (the light's `layer`): a flicker."""
        if not self.dry:
            self.ch.light_hints.setdefault(round(self.clock.inv(q), 4), {})[light] = float(factor)

    def light_out(self, q: float, light: str) -> None:
        """Knock a street light out at q (move time): its glow and its lamp dark from then on."""
        if not self.dry:
            self.ch.lights_out[light] = float(self.clock.inv(q))

    def screen_off(self, q: float, group: str, fade: float = 0.5) -> None:
        """Switch a window's screens off at q (move time): the street's off version in its band (screens.py)."""
        if not self.dry:
            self.ch.screens_off[group] = (float(self.clock.inv(q)), float(fade))

    def burst(self, q: float, x: float, y: float, **kw) -> None:
        """A paint explosion at (x, y) on q (move time): props.Burst's flecks and splatter."""
        if not self.dry:
            from .props import Burst
            self.ch.bursts.append(Burst(float(self.clock.inv(q)), float(x), float(y), **kw))

    def photo(self, q: float) -> None:
        if not self.dry:
            self.ch.extra_photos.append(self.clock.inv(q))

    def label(self, q: float, x: float, y: float, text: str) -> None:
        if not self.dry:
            self.ch.labels.append((self.clock.inv(q), x, y, text))

    def song(self, q: float) -> float:
        """Song time of a move time (for props keyed in song time)."""
        return float(self.clock.inv(q))

    def torso(self, e: Ev) -> tuple[float, float]:
        dx, dy = self.ch.lib.torso_offset(e.pose, e.anchor, e.scale, e.mirror)
        return e.x + dx, e.y + dy

    def finish(self) -> MoveState:
        """The move's state in song time, with the torso track of its figure (motion M0)."""
        st = replace(self.st, q0=self.clock.inv(self.st.q0), q1=self.clock.inv(self.st.q1))
        lib = self.ch.lib
        figs = sorted([e for e in self.out if e.layer in ("wall", "air") and e.pose is not None
                       and lib.m(e.pose)["kind"] == "pose"], key=lambda e: e.q)
        if figs:
            xs = [self.torso(e)[0] for e in figs]
            st.tx0, st.tx1 = xs[0], xs[-1]
            st.v_in = xs[1] - xs[0] if len(xs) > 1 else None
            st.v_out = xs[-1] - xs[-2] if len(xs) > 1 else None
            st.ph0, st.ph1 = lib.stride_phase.get(figs[0].pose), lib.stride_phase.get(figs[-1].pose)
        elif self.prev is not None and self.mv.kind == "freeze":  # the figure stays as the move before left it
            p = self.prev
            st.tx0 = st.tx1 = p.tx1
            st.v_in = st.v_out = p.v_out
            st.ph0 = st.ph1 = p.ph1
            st.cycle, st.run = p.cycle, p.run
        return st


# ---------------------------------------------------------------------------------------------- loading

def _parse_moves(raw) -> list[tuple[str, dict]]:
    out = []
    for k, m in enumerate(raw or []):
        if isinstance(m, str):
            out.append((m.strip(), {}))
        elif isinstance(m, dict) and len(m) == 1:
            kind, a = next(iter(m.items()))
            if a is None:
                a = {}
            if not isinstance(a, dict):
                raise ChoreoError(f"move {k + 1} ({kind}): arguments must be a mapping, got {a!r}")
            out.append((str(kind).strip(), dict(a)))
        else:
            raise ChoreoError(f"move {k + 1}: write it as '- template: {{args}}' or '- template'")
    return out


def _words(v):
    """YAML 1.1 reads the keys on and off as booleans ({on: snare} is {True: snare}): back to words."""
    if isinstance(v, dict):
        return {({True: "on", False: "off"}[k] if isinstance(k, bool) else k): _words(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_words(x) for x in v]
    return v


def read(path: Path) -> dict:
    try:
        data = _words(yaml.safe_load(Path(path).read_text()) or {})
    except yaml.YAMLError as e:
        raise ChoreoError(f"{Path(path).name}: YAML error: {e}") from e
    if not isinstance(data, dict):
        raise ChoreoError(f"{Path(path).name}: the top level must be a mapping")
    return data


def _include_path(path: Path, name) -> Path:
    f = Path(str(name))
    if not f.suffix:
        f = f.with_suffix(".yaml")
    return f if f.is_absolute() else path.parent / f


def expand(path: Path, data: dict, _seen: frozenset = frozenset()) -> dict:
    """Join the included choreographies (`include: [opening, p6]`: names or paths relative to this file) and this
    file's own segments and moves, in that order, into one choreography. Every move is tagged with its source file
    (_move_src, _sources: name and seed). Top-level settings: this file's win, then the first included file's; hits
    thresholds are merged; each included file's camera defaults go into its own segments."""
    path = Path(path)
    names = data.get("include") or []
    names = [names] if isinstance(names, str) else list(names)
    own_seed = data.get("seed")
    out = {k: v for k, v in data.items() if k not in ("include", "segments", "moves")}
    sources: list[dict] = []
    segs: list = []
    moves: list = []
    msrc: list[int] = []
    hits: dict = {}
    for name in names:
        f = _include_path(path, name)
        if not f.exists():
            raise ChoreoError(f"{path.name}: include {name!r}: {f} not found")
        if f.resolve() in _seen or f.resolve() == path.resolve():
            raise ChoreoError(f"{path.name}: include cycle through {f.name}")
        sub = expand(f, read(f), _seen | {path.resolve()})
        base = len(sources)
        sources += sub["_sources"]
        cam = dict(sub.get("camera") or {})
        for i_sg, sg in enumerate(sub.get("segments") or []):
            sg = dict(sg)
            if i_sg == 0 and segs:  # a join between two files: the camera eases across it (camera.plan)
                sg["_join"] = True
            c = sg.get("camera", "follow")
            c = {"preset": c} if isinstance(c, str) else dict(c)
            sg["camera"] = {**cam, **c}
            segs.append(sg)
        moves += list(sub.get("moves") or [])
        msrc += [base + k for k in sub["_move_src"]]
        hits.update(sub.get("hits") or {})
        for k, v in sub.items():
            if k not in out and k not in ("segments", "moves", "window", "camera", "hits", "note", "_sources",
                                          "_move_src"):
                out[k] = v
    own = data.get("moves") or []
    sources.append({"name": path.stem, "seed": int(own_seed if own_seed is not None else out.get("seed", 11))})
    segs += list(data.get("segments") or [])
    moves += list(own)
    msrc += [len(sources) - 1] * len(own)
    hits.update(data.get("hits") or {})
    if hits:
        out["hits"] = hits
    out["segments"], out["moves"], out["_sources"], out["_move_src"] = segs, moves, sources, msrc
    return out


def _auto_stops(ch: Choreo) -> list[tuple[float, float]]:
    """The song's stops inside the choreography that no freeze move covers (song 16ths: first photo, end photo)."""
    g = ch.grid
    out = []
    frz = [(mv.plan.q0, mv.plan.q1) for mv in ch.moves if mv.kind == "freeze" and mv.plan.q0 is not None]
    for st in g.stops:
        t1 = float(st.get("end_impact") or st["t1"])
        a = float(math.ceil(g.q(float(st["t0"])) - 0.1))
        b = float(round(g.q(t1)))
        if b <= a or a < ch.q_start or b > ch.q_end:
            continue
        cover = max((max(0.0, min(b, f1 + 1) - max(a, f0)) for f0, f1 in frz), default=0.0)
        if cover < 0.5 * (b - a):
            out.append((a, b))
    return out


def _run(ch: Choreo, mv: Move, c: Ctx, M) -> None:
    from . import motion as MO

    ch._active = c.clock if c.clock.stops else None
    fn = MO.M0.get(mv.kind) if ch.motion == "m0" else None
    try:
        (fn or M.TEMPLATES[mv.kind])(c)
    finally:
        ch._active = None


def _feet(ch: Choreo, *xs) -> float:
    """The ground line at the first known x (the move's planned middle, else the previous move's end)."""
    for x in xs:
        if x is not None:
            return ch.scene.feet_at(x)
    return ch.scene.feet


def load(path: Path, grid: SongGrid, scene: Scene, lib: PoseLib, overrides: dict | None = None) -> Choreo:
    """Parse, place and validate a choreography. Placement problems become issues (errors stop a render).
    overrides: top-level settings that replace the file's (the CLI's --jump, --primer ...)."""
    from . import moves as M  # registers the templates

    path = Path(path)
    raw_data = read(path)
    data = expand(path, raw_data)
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    grid.thresholds.update(data.get("hits") or {})
    ch = Choreo(path, grid, scene, lib, data)
    for k in raw_data:
        if k not in TOP_KEYS:
            ch.warn("top", f"unknown key {k!r} (ignored)")
    if ch.jump not in ("J1", "J2"):
        ch.err("top", f"jump must be J1 or J2, not {ch.jump}")
    if ch.primer not in ("all", "keepers", "none", "ghosts"):
        ch.err("top", f"primer must be all, keepers, none or ghosts, not {ch.primer}")
    if ch.motion not in ("m0", "legacy"):
        ch.err("top", f"motion must be m0 or legacy, not {ch.motion}")
    # segments
    for k, s in enumerate(data.get("segments") or []):
        where = f"segment {k + 1}"
        try:
            q0, _ = grid.resolve(s["from"])
            q1, _ = grid.resolve(s["to"])
            rate = rate_of(s.get("rate", "16ths"))
        except (GridError, KeyError) as e:
            ch.err(where, f"{e}")
            continue
        cam = s.get("camera", "follow")
        cam = {"preset": cam} if isinstance(cam, str) else dict(cam)
        cam.setdefault("preset", "follow")
        if cam["preset"] not in CAMERA_PRESETS:
            ch.err(where, f"unknown camera preset {cam['preset']!r} ({', '.join(CAMERA_PRESETS)})")
        if cam["preset"] == "frame" and not (isinstance(cam.get("x"), list) and len(cam["x"]) == 2
                                             and float(cam["x"][1]) > float(cam["x"][0])):
            ch.err(where, "camera frame: give x: [x0, x1] (the stretch of street it shows)")
        if q1 <= q0:
            ch.err(where, f"ends before it starts ({grid.addr(q0)} to {grid.addr(q1)})")
        ch.segments.append(Segment(q0, q1, rate, cam, s.get("label", ""), bool(s.get("_join"))))
    ch.segments.sort(key=lambda s: s.q0)
    if not ch.segments:
        ch.err("segments", "no segments: give at least one {from, to, rate}")
    for a, b in zip(ch.segments, ch.segments[1:]):
        if b.q0 < a.q1 - 1e-6:
            ch.err("segments", f"overlap: {grid.addr(a.q0)} to {grid.addr(a.q1)} and {grid.addr(b.q0)} to "
                               f"{grid.addr(b.q1)}")
        elif b.q0 > a.q1 + 1e-6:
            ch.err("segments", f"gap between {grid.addr(a.q1)} and {grid.addr(b.q0)} (no photos there)")
    if data.get("window"):
        try:
            w0, _ = grid.resolve(data["window"][0])
            w1, _ = grid.resolve(data["window"][1])
            ch.window = (grid.t(w0), grid.t(w1))
        except (GridError, IndexError, TypeError) as e:
            ch.err("window", str(e))
    # moves
    try:
        raw = _parse_moves(data.get("moves"))
    except ChoreoError as e:
        ch.err("moves", str(e))
        raw = []
    msrc = data.get("_move_src") or [0] * len(raw)
    multi = len(ch.sources) > 1
    count: dict[int, int] = {}
    for k, (kind, a) in enumerate(raw):
        src = msrc[k] if k < len(msrc) else 0
        li = count.get(src, 0)
        count[src] = li + 1
        where = (f"{ch.sources[src]['name']} " if multi else "") + f"move {li + 1} ({kind})"
        if kind not in M.TEMPLATES:
            ch.err(where, f"unknown template (known: {', '.join(sorted(M.TEMPLATES))})")
            continue
        # (v8c: a move's `stream: n` gives it the random stream of the n-th move of its file (from 0), so a variant
        # that adds or drops moves before it paints it exactly as the file it came from: cold open C3's crouch)
        li_ = int(a["stream"]) if isinstance(a, dict) and a.get("stream") is not None else li
        ch.moves.append(Move(len(ch.moves), kind, a, where, src=src, seed=int(ch.sources[src]["seed"]),
                             local_i=li_, prop=bool(getattr(M.TEMPLATES[kind], "prop", False))))
    if ch.errors:
        return ch
    # props first (the set: the box, its stencils): they do not take part in the runner's chain
    for mv in [m for m in ch.moves if m.prop]:
        c = Ctx(ch, mv, None, None, dry=False)
        try:
            _run(ch, mv, c, M)
        except (ChoreoError, KeyError) as e:
            ch.err(mv.where, str(e.args[0]) if e.args else str(e))
            continue
        unknown = set(mv.args) - c.used
        if unknown:
            ch.warn(mv.where, f"unknown argument(s) {', '.join(sorted(unknown))} (ignored)")
        mv.plan = mv.state = c.finish()
        ch.events.extend(c.out)
    figs = [m for m in ch.moves if not m.prop]

    def dry_pass() -> None:
        prev = None
        for mv in figs:
            c = Ctx(ch, mv, prev, None, dry=True, feet=_feet(ch, prev.x1 if prev else None))
            try:
                _run(ch, mv, c, M)
            except ChoreoError:
                pass  # reported in pass 2
            except KeyError:
                pass
            mv.plan = c.finish()
            prev = mv.plan

    # pass 1: dry, to learn every move's time and extent (and which song stops a freeze covers)
    dry_pass()
    if ch.pause:
        ch.stops = _auto_stops(ch)
        if ch.stops:
            ch.clock = Clock(ch.stops)
            ch.freezes.extend(ch.stops)  # the song stops: the paint does not dry while it stands still
            dry_pass()
    # pass 2: place
    prev = None
    for n, mv in enumerate(figs):
        nxt = figs[n + 1].plan if n + 1 < len(figs) else None
        mid = (mv.plan.x0 + mv.plan.x1) / 2 if mv.plan.x0 is not None and mv.plan.x1 is not None else None
        c = Ctx(ch, mv, prev, nxt, dry=False, feet=_feet(ch, mid, prev.x1 if prev else None,
                                                         nxt.x0 if nxt is not None else None))
        try:
            _run(ch, mv, c, M)
        except ChoreoError as e:
            ch.err(mv.where, str(e))
            prev = mv.plan
            continue
        except KeyError as e:
            ch.err(mv.where, str(e.args[0]) if e.args else str(e))
            prev = mv.plan
            continue
        unknown = set(mv.args) - c.used
        if unknown:
            ch.warn(mv.where, f"unknown argument(s) {', '.join(sorted(unknown))} (ignored)")
        mv.state = c.finish()
        if mv.plan.q0 is not None and mv.state.q0 is not None and abs(mv.plan.q0 - mv.state.q0) > 1e-6:
            ch.warn(mv.where, f"its time moved from {grid.addr(mv.plan.q0)} to {grid.addr(mv.state.q0)} once the "
                              f"move before it was placed: give it an explicit time (a bar.beat, or a hit with @ or >)")
        try:
            _trail_in(ch, mv, c.out)
        except ChoreoError as e:
            ch.err(mv.where, str(e))
        ch.events.extend(c.out)
        prev = mv.state
    ch.events.sort(key=lambda e: e.q)  # stable: a move's own order stays within a photo
    ch.freezes.sort()
    _order_check(ch)
    return ch


def _trail_in(ch: Choreo, mv: Move, out: list[Ev]) -> None:
    """A move's `trail_in: <seconds>` (v8b, the climb after verse 2): its trail fades in over its first seconds
    instead of switching on with its first photo. Each figure event from the move's first wall or air pose until that
    many seconds later carries flags trail_w, 0 at the first pose rising linearly to 1, and the renderer blends its
    ghost and paint-out between the light trail's (washing away, as verse 2's run before it) and its own (render.py's
    Mark.trail_w); the events, their timing and every random choice stay as they are."""
    secs = mv.args.get("trail_in")
    if secs in (None, False):
        return
    try:
        secs = float(secs)
    except (TypeError, ValueError):
        raise ChoreoError(f"trail_in: {secs!r}, seconds (a number above 0)") from None
    if secs <= 0:
        raise ChoreoError(f"trail_in: {secs}, seconds (a number above 0)")
    figs = [e for e in out if e.layer in ("wall", "air") and e.pose is not None]
    if not figs:
        return
    g = ch.grid
    t0 = min(g.t(e.q) for e in figs)
    for e in figs:
        w = (g.t(e.q) - t0) / secs
        if w < 1.0:
            e.flags = dict(e.flags, trail_w=round(max(w, 0.0), 4))


def _order_check(ch: Choreo) -> None:
    last = None
    for mv in ch.moves:
        st = mv.state
        if st.q0 is None or mv.prop:
            continue
        if last is not None and last.state.q1 is not None and st.q0 < last.state.q1 - 1e-6 and mv.kind != "freeze" \
                and last.kind != "freeze":
            ch.warn(mv.where, f"starts at {ch.grid.addr(st.q0)}, before the previous move ends "
                              f"({ch.grid.addr(last.state.q1)})")
        if mv.kind != "freeze":
            last = mv


def open_choreo(p, name: str, overrides: dict | None = None) -> Choreo:
    """Load projects/<p>/paint/choreo/<name>.yaml (or a .yaml path) with its street, scene and poses."""
    from . import PaintDirs

    pd = PaintDirs(p)
    f = pd.choreo_file(name)
    if not f.exists():
        raise SystemExit(f"No choreography: {f}")
    data = expand(f, read(f))
    street = str((overrides or {}).get("street") or data.get("street", "main"))
    scene = Scene(pd.street(street), pd.cache)
    lib = PoseLib(pd.poses)
    grid = SongGrid.from_project(p)
    return load(f, grid, scene, lib, overrides)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u
