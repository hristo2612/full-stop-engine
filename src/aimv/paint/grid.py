"""The song grid for the choreography: 16th-note indices, bar.beat addresses, named hits.

Internally every time is a 16th index q (float): q = 4 * beat index + 16th within the beat, beat index 0 being the
first beat of beats.json. Subdivisions are linear between neighbouring beats (as in `aimv.look.muto.grid`), so
q -> seconds is exact on the breakdown's grid and q can be fractional (a 32nd is q + 0.5).

Addresses in choreography files are 1-based like the rest of our docs (TIMELINE.md counts bars from 1):

    43          bar 43, beat 1              (YAML numbers are bars: 43.2 is bar 43 beat 2, never seconds)
    43.2        bar 43, beat 2
    43.2.3      bar 43, beat 2, third 16th
    57.607s     seconds
    impact      a named hit: impact, kick, snare, hat, vocal (vocal entries), stop (a stop's start), stop.end
                (its return), or a section label (Verse 2, Bridge): the first one after the reference time
    snare@44.2  the snare nearest bar 44 beat 2
    snare>44.2  the first snare at or after bar 44 beat 2
    +2, -1      relative to the reference time, in 16ths; any of the above can take an offset (impact-1, 44.2+0.5)

Kicks and snares are filtered by strength (choreography `hits:` thresholds) so `snare` means a backbeat, not a ghost.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..project import Project
from ..util import read_json

RATES = {"32nds": 0.5, "16ths": 1.0, "8ths": 2.0, "beats": 4.0, "half-bars": 8.0, "bars": 16.0}
HIT_KINDS = ("impact", "kick", "snare", "hat", "vocal", "stop", "stop.end")
DEFAULT_THRESHOLDS = {"kick": 0.5, "snare": 0.85, "hat": 0.3, "impact": 0.0, "vocal": 0.0}

_TIME = re.compile(r"^\s*(?P<base>.*?)\s*(?P<off>[+\-]\s*\d+(?:\.\d+)?)?\s*$")


class GridError(ValueError):
    pass


@dataclass
class SongGrid:
    beats: np.ndarray  # beat times (s)
    beat_addr: list[tuple[int, int]]  # (bar, beat), 0-based as in beats.json
    hits: dict[str, list[tuple[float, float]]]  # kind -> [(t, strength)], sorted
    stops: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @classmethod
    def from_project(cls, p: Project, thresholds: dict | None = None) -> "SongGrid":
        b = read_json(p.bd / "beats.json")
        beats = np.array([x["t"] for x in b["beats"]], np.float64)
        addr = [(int(x["bar"]), int(x["beat"])) for x in b["beats"]]
        ev = read_json(p.bd / "hits.json")["events"]
        hits = {k: sorted((float(e["t"]), float(e.get("s", 1.0))) for e in ev.get(k, []))
                for k in ("kick", "snare", "hat", "impact")}
        hits["vocal"] = sorted((float(e["t"]), float(e.get("s", 1.0))) for e in ev.get("vocal_entry", []))
        sec = read_json(p.bd / "sections.json")["sections"] if (p.bd / "sections.json").exists() else []
        th = dict(DEFAULT_THRESHOLDS)
        th.update(thresholds or {})
        return cls(beats, addr, hits, list(ev.get("stop", [])), sec, th)

    # ------------------------------------------------------------------ q <-> seconds
    def t(self, q: float) -> float:
        """Seconds of 16th index q (extrapolated with the edge beat length outside the grid)."""
        b = q / 4.0
        n = len(self.beats)
        k = int(math.floor(b))
        if k < 0:
            return float(self.beats[0] + b * (self.beats[1] - self.beats[0]))
        if k >= n - 1:
            return float(self.beats[-1] + (b - (n - 1)) * (self.beats[-1] - self.beats[-2]))
        return float(self.beats[k] + (self.beats[k + 1] - self.beats[k]) * (b - k))

    def q(self, t: float) -> float:
        """16th index of time t (inverse of t())."""
        n = len(self.beats)
        if t <= self.beats[0]:
            return 4.0 * (t - self.beats[0]) / (self.beats[1] - self.beats[0])
        if t >= self.beats[-1]:
            return 4.0 * ((n - 1) + (t - self.beats[-1]) / (self.beats[-1] - self.beats[-2]))
        k = int(np.searchsorted(self.beats, t, side="right")) - 1
        return 4.0 * (k + (t - self.beats[k]) / (self.beats[k + 1] - self.beats[k]))

    def beat_index(self, bar: int, beat: int) -> int:
        """Beat index of a 1-based bar and beat."""
        key = (bar - 1, beat - 1)
        if not hasattr(self, "_addr_index"):
            self._addr_index = {a: i for i, a in enumerate(self.beat_addr)}
        if key in self._addr_index:
            return self._addr_index[key]
        first_bar, last_bar = self.beat_addr[0][0], self.beat_addr[-1][0]
        if first_bar <= bar - 1 <= last_bar or beat > 4:
            n = sum(1 for a in self.beat_addr if a[0] == bar - 1) or 4
            raise GridError(f"bar {bar} has {n} beats: {bar}.{beat} does not exist")
        # past the end (or before the start) of the grid: assume 4/4 from the last known downbeat
        last_beat = self.beat_addr[-1][1]
        return len(self.beat_addr) - 1 + (bar - 1 - last_bar) * 4 + (beat - 1 - last_beat)

    def addr(self, q: float) -> str:
        """1-based address of q: bar.beat.16th (with a fraction when q is not on a 16th)."""
        qi = int(math.floor(q + 1e-6))
        frac = q - qi
        k, s = divmod(qi, 4)
        if 0 <= k < len(self.beat_addr):
            bar, beat = self.beat_addr[k]
        else:
            last = len(self.beat_addr) - 1
            bar0, beat0 = self.beat_addr[last if k > last else 0]
            d = k - (last if k > last else 0)
            tot = bar0 * 4 + beat0 + d
            bar, beat = divmod(tot, 4)
        a = f"{bar + 1}.{beat + 1}.{s + 1}"
        return a + (f"+{frac:.2f}" if frac > 0.01 else "")

    # ------------------------------------------------------------------ hits
    def hit_times(self, kind: str) -> list[float]:
        if kind == "stop":
            return [float(s["t0"]) for s in self.stops]
        if kind == "stop.end":
            return [float(s.get("end_impact") or s["t1"]) for s in self.stops]
        if kind not in self.hits:
            raise GridError(f"unknown hit {kind!r} (known: {', '.join(HIT_KINDS)})")
        th = self.thresholds.get(kind, 0.0)
        return [t for t, s in self.hits[kind] if s >= th]

    def section_start(self, label: str) -> list[float]:
        lab = label.strip().lower()
        return [float(s["t0"]) for s in self.sections if s.get("label", "").lower() == lab]

    def _named(self, name: str) -> list[float] | None:
        n = name.strip().lower()
        if n in HIT_KINDS:
            return self.hit_times(n)
        sec = self.section_start(name)
        return sec or None

    # ------------------------------------------------------------------ parsing
    def parse_addr(self, s: str) -> float:
        """q of a bare address: '43', '43.2', '43.2.3', '57.6s'."""
        s = s.strip()
        if s.endswith("s"):
            try:
                return self.q(float(s[:-1]))
            except ValueError as e:
                raise GridError(f"bad seconds {s!r}") from e
        parts = s.split(".")
        if not all(p.isdigit() for p in parts) or not 1 <= len(parts) <= 3:
            raise GridError(f"bad grid address {s!r} (bar, bar.beat, bar.beat.16th or seconds like 57.6s)")
        bar = int(parts[0])
        beat = int(parts[1]) if len(parts) > 1 else 1
        six = int(parts[2]) if len(parts) > 2 else 1
        if bar < 1 or beat < 1 or not 1 <= six <= 4:
            raise GridError(f"grid address {s!r} out of range (bars and beats from 1, 16ths 1 to 4)")
        return 4.0 * self.beat_index(bar, beat) + (six - 1)

    def resolve(self, spec, ref: float | None = None) -> tuple[float, float | None]:
        """Resolve a time spec to (q, hit_t): hit_t is the exact hit time for named hits (for the on-grid check),
        else None. ref is the reference 16th index for relative specs and for 'the next hit'."""
        if isinstance(spec, bool) or spec is None:
            raise GridError(f"bad time {spec!r}")
        if isinstance(spec, int | float):
            s = repr(float(spec)) if isinstance(spec, float) else str(spec)
            if isinstance(spec, float) and s.endswith(".0"):
                s = s[:-2]
            return self.parse_addr(s), None
        s = str(spec).strip()
        m = _TIME.match(s)
        if not m:
            raise GridError(f"bad time {spec!r}")
        base, off = m.group("base").strip(), m.group("off")
        dq = float(off.replace(" ", "")) if off else 0.0
        if not base:
            if ref is None:
                raise GridError(f"relative time {spec!r} needs a reference (an earlier time in the move)")
            return ref + dq, None
        hit_t = None
        sel = re.match(r"^([A-Za-z][A-Za-z .0-9]*?)\s*([@>])\s*(.+)$", base)
        if sel:
            name, op, where = sel.group(1), sel.group(2), sel.group(3)
            times = self._named(name)
            if times is None:
                raise GridError(f"unknown hit or section {name!r} in {spec!r}")
            wq = self.parse_addr(where)
            wt = self.t(wq)
            if not times:
                raise GridError(f"no {name} in the song")
            if op == "@":
                hit_t = min(times, key=lambda x: abs(x - wt))
            else:
                after = [x for x in times if x >= wt - 0.03]
                if not after:
                    raise GridError(f"no {name} at or after {where}")
                hit_t = after[0]
        elif re.match(r"^[A-Za-z]", base):
            times = self._named(base)
            if times is None:
                raise GridError(f"unknown hit or section {base!r} (hits: {', '.join(HIT_KINDS)})")
            rt = self.t(ref) if ref is not None else -1e9
            after = [x for x in times if x > rt + 0.03]
            if not after:
                raise GridError(f"no {base} after {self.addr(ref) if ref is not None else 'the start'}")
            hit_t = after[0]
        else:
            return self.parse_addr(base) + dq, None
        return self.q(hit_t) + dq, hit_t

    # ------------------------------------------------------------------ photo steps
    def steps(self, q0: float, q1: float, rate: float, origin: float | None = None) -> list[float]:
        """Photo indices in [q0, q1) every `rate` 16ths, aligned to origin (default: the bar line at or before q0)."""
        if origin is None:
            origin = math.floor(q0 / 16.0) * 16.0
        k0 = math.ceil((q0 - origin) / rate - 1e-6)
        out = []
        q = origin + k0 * rate
        while q < q1 - 1e-6:
            out.append(round(q, 6))
            q += rate
        return out

    def snap(self, q: float, rate: float = 1.0) -> float:
        return round(q / rate) * rate

    def nearest_hit(self, kind: str, t: float) -> float | None:
        ts = self.hit_times(kind)
        if not ts:
            return None
        i = bisect.bisect_left(ts, t)
        cand = [ts[j] for j in (i - 1, i) if 0 <= j < len(ts)]
        return min(cand, key=lambda x: abs(x - t))


def rate_of(name) -> float:
    """Photo step in 16ths of a rate name (16ths, 8ths, beats, half-bars, bars, 32nds) or a number of 16ths."""
    if isinstance(name, int | float):
        return float(name)
    key = str(name).strip().lower()
    aliases = {"16th": "16ths", "8th": "8ths", "beat": "beats", "half-bar": "half-bars", "halfbars": "half-bars",
               "bar": "bars", "32nd": "32nds"}
    key = aliases.get(key, key)
    if key not in RATES:
        raise GridError(f"unknown rate {name!r} (one of {', '.join(RATES)})")
    return RATES[key]
