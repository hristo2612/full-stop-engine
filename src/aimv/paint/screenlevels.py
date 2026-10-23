"""Screens at levels over time (v8, the treatment's E: verse 2's wall of glowing screens): every screen's own light
(its face, its spill on the wall and its streaked reflection on the wet ground) is stored apart from the street
(world.py's screens8: <street>/screens8/<id>.npz, float16 linear RGB, the face at full resolution over face_box, the
glow at half resolution over glow_box), and the street with screen i at level k is street_lin - (1 - k) * contrib_i:
0 dead (dark glass), 1 as built, above 1 a flare.

    screen_cues: {dir: screens8,                          (the street's folder of them, default screens8)
                  on: [[A1, 68.56s], [A2, 69.43s], ...],   lit from dead on that photo: in over `rise` photos (2) with an
                                                           overshoot (1.3) on its first lit photo
                  flare: [[B1, 71.94s], ...],              to `flare_k` (1.6) and back over `flare_s` (0.25 s)
                  pulse: {from: 76.87s, to: 79.0s, on: 8ths, hi: [1.0, 1.2], lo: [0.7, 0.6]},   every lit screen: hi
                                                           on each hit, lo between, both eased from their first value
                                                           to their second over the stretch
                  kill: [[A, 79.02s], [D2, 79.53s, 2]],    a group or a screen dies (over that many photos, default 1),
                                                           flickering once to `flicker` (0.4) on the photo before;
                        [D4, D3]                           or dies with smeared screen D3
                  smear: [[D3], [D2, 79.22s]],             smeared black by him: a wet black stroke across its face at
                                                           his hand's height (`smear_y`: the share of the face from its
                                                           top, 0.5), its front where his body is at the face's height
                                                           as he passes (the stroke stays, wet, dripping); with a
                                                           time: from his hand at that time, spreading back to the
                                                           face's left edge over `smear_n` photos (2); the screen dies
                                                           `smear_die` photos (4) after its stroke is complete
                  before: dead}                            before its `on` (dead, or lit for a screen with no `on`)

A screen's light on the paint (obstacles.json lights with the screen's id in `screen`, or `scr<id>`) follows its
level (the light hints). The paint round a screen dims with its light (render.py scales the wall's light map by the
change in the street's brightness)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from .choreo import ChoreoError, Ctx
from .moves import template


@template("screen_cues", prop=True)
def screen_cues(c: Ctx) -> None:
    """Screens lit, flared, pulsed and killed on cues (prop; see screenlevels.py)."""
    ch, g = c.ch, c.ch.grid
    d = Path(ch.scene.d) / str(c.arg("dir", "screens8"))
    f = d / "screens.json"
    if not f.exists():
        raise ChoreoError(f"screen_cues: no {f}")
    spec = json.loads(f.read_text())
    scr = {s["id"]: s for s in spec["screens"]}
    groups: dict[str, list[str]] = {}
    for s in spec["screens"]:
        groups.setdefault(str(s.get("group", s["id"])), []).append(s["id"])

    def ids(name):
        name = str(name)
        if name in scr:
            return [name]
        if name in groups:
            return list(groups[name])
        raise ChoreoError(f"screen_cues: no screen or group {name!r} (screens: {', '.join(scr)})")

    def tt(spec_, key):
        return g.t(c.song(c.at(spec_, key=key)))

    cue = {sid: {"on": None, "flares": [], "kill": None, "smear": None, "kill_with": None} for sid in scr}
    for i, it in enumerate(c.arg("on") or []):
        for sid in ids(it[0]):
            cue[sid]["on"] = tt(it[1], f"on[{i + 1}]")
    for i, it in enumerate(c.arg("flare") or []):
        for sid in ids(it[0]):
            cue[sid]["flares"].append(tt(it[1], f"flare[{i + 1}]"))
    sn = int(c.arg("smear_n", 2))
    for i, it in enumerate(c.arg("kill") or []):
        if len(it) == 2 and str(it[1]) in scr:  # dies with another screen's smear
            for sid in ids(it[0]):
                cue[sid]["kill_with"] = str(it[1])
            continue
        n = int(it[2]) if len(it) > 2 else 1
        for sid in ids(it[0]):
            cue[sid]["kill"] = (tt(it[1], f"kill[{i + 1}]"), n)
    for i, it in enumerate(c.arg("smear") or []):
        it = [it] if isinstance(it, str) else list(it)
        for sid in ids(it[0]):
            ts = tt(it[1], f"smear[{i + 1}]") if len(it) > 1 else None
            cue[sid]["smear"] = {"t": ts}
    for sid, cu in cue.items():
        w = cu.get("kill_with")
        if w is not None and cue[w]["smear"] is None:
            raise ChoreoError(f"screen_cues: kill {sid} with {w}, but {w} is not smeared")
    pulse = c.arg("pulse")
    pl = None
    if pulse:
        hits = pulse.get("on", "8ths")
        t0, t1 = tt(pulse["from"], "pulse.from"), tt(pulse["to"], "pulse.to")
        step = {"8ths": 2, "16ths": 1, "beats": 4}.get(str(hits))
        if step is None:
            ht = [t for t, s in g.hits.get(str(hits), []) if t0 <= t <= t1]
        else:
            q = round(g.q(t0) / step) * step
            ht = []
            while g.t(q) <= t1 + 1e-6:
                if g.t(q) >= t0 - 1e-6:
                    ht.append(g.t(q))
                q += step
        hi, lo = pulse.get("hi", [1.0, 1.0]), pulse.get("lo", [0.7, 0.7])
        hi = [hi, hi] if isinstance(hi, int | float) else hi
        lo = [lo, lo] if isinstance(lo, int | float) else lo
        pl = {"t0": t0, "t1": t1, "hits": ht, "hi": [float(v) for v in hi], "lo": [float(v) for v in lo]}
    if c.dry:
        return
    ch.screen_cues = {"dir": str(d), "spec": spec, "cue": cue, "pulse": pl, "rise": int(c.arg("rise", 2)),
                      "over": float(c.arg("overshoot", 1.3)), "flare_k": float(c.arg("flare_k", 1.6)),
                      "flare_s": float(c.arg("flare_s", 0.25)), "flicker": float(c.arg("flicker", 0.4)),
                      "before": str(c.arg("before", "dead")), "smear_n": sn,
                      "smear_y": float(c.arg("smear_y", 0.5)), "smear_die": int(c.arg("smear_die", 4)), "track": {},
                      "face": {s_["id"]: s_["face"] for s_ in spec["screens"]}}
    _light_hints(ch, scr)


def _light_hints(ch, sids) -> None:
    """Each screen's light on the paint follows its level, photo by photo (the screens sids)."""
    lights = {L.get("screen") or (L.get("id", "")[3:] if str(L.get("id", "")).startswith("scr") else None): L.get("id")
              for L in ch.scene.lights}
    for ph in ch.photos():
        for sid in sids:
            lid = lights.get(sid)
            if lid:
                k = level(ch, sid, ph.t)
                d = ch.light_hints.setdefault(round(ph.q, 4), {})
                if abs(k - 1) > 1e-3:
                    d[lid] = float(k)
                else:
                    d.pop(lid, None)


def _step(ch) -> float:
    g = ch.grid
    return g.t(g.resolve("1.1.2")[0]) - g.t(g.resolve("1.1.1")[0])


def level(ch, sid: str, t: float) -> float:
    """Screen sid's level at song time t (1 when there are no cues)."""
    sc = getattr(ch, "screen_cues", None)
    if sc is None or sid not in sc["cue"]:
        return 1.0
    cu = sc["cue"][sid]
    st = _step(ch)
    on = cu["on"]
    if on is None:
        k = 1.0
    elif t < on - 1e-6:
        return 0.0
    else:
        n = (t - on) / st  # photos since it lit
        r = max(1, sc["rise"])
        if n < 0.5:
            k = sc["over"]  # (the first lit photo: an overshoot)
        elif n < r:
            k = sc["over"] + (1 - sc["over"]) * (n / r)
        else:
            k = 1.0
    for tf in cu["flares"]:
        if tf - 1e-6 <= t < tf + sc["flare_s"]:
            u = (t - tf) / sc["flare_s"]
            k = max(k, 1 + (sc["flare_k"] - 1) * (1 - u) ** 2)
    pl = sc["pulse"]
    if pl and pl["t0"] - 1e-6 <= t <= pl["t1"] + 1e-6 and k > 0.5:
        u = (t - pl["t0"]) / max(pl["t1"] - pl["t0"], 1e-6)
        hi = pl["hi"][0] + (pl["hi"][1] - pl["hi"][0]) * u
        lo = pl["lo"][0] + (pl["lo"][1] - pl["lo"][0]) * u
        on_hit = any(abs(t - h) < 0.4 * st for h in pl["hits"])
        k = k * (hi if on_hit else lo)
    kl = cu["kill"]
    if cu.get("smear") is not None or cu.get("kill_with") is not None:  # dies after its smear (or another's)
        td = smear_death(sc, cu.get("kill_with") or sid, st)
        kl = (td, 1) if td is not None else None
    if kl is not None:
        tk, n = kl
        if t >= tk - 1e-6:
            k = k * max(0.0, 1 - (t - tk + st) / (n * st)) if n > 1 else 0.0
        elif t >= tk - st - 1e-6:
            k = k * sc["flicker"]  # (a last flicker on the photo before)
    return float(k)


def smear_death(sc: dict, sid: str, st: float) -> float | None:
    """When smeared screen sid dies: `smear_die` photos (4) after its stroke is complete (a timed smear: spread back
    to the face's edge; else: his body past the face's right edge, or the last photo it was smeared on). None: not
    known yet (the smears are tracked when the photos are rendered)."""
    sm, die = sc["cue"][sid]["smear"], sc["smear_die"]
    if sm["t"] is not None:
        return sm["t"] + (max(1, sc["smear_n"]) - 1 + die) * st
    tr = sc["track"].get(sid)
    if not tr:
        return None
    fx1 = sc["face"][sid][2]
    t_end = next((a[0] for a in tr if a[2] >= fx1 - 1), tr[-1][0])
    return t_end + die * st


def extent(sc: dict, sid: str, t: float) -> tuple[float, float]:
    """The smear stroke's street x extent [L, R] on screen sid at song time t (L == R: none yet)."""
    tr = sc["track"].get(sid)
    if not tr:
        return 0.0, 0.0
    j = int(np.searchsorted([a[0] for a in tr], t + 1e-6)) - 1
    if j < 0:
        return 0.0, 0.0
    return tr[j][1], tr[j][2]


class ScreenLights:
    """The screens' stored light, applied to a photo's street box at the levels of its time (cached per screen)."""

    def __init__(self, ch, hist=None):
        self.ch = ch
        self.sc = getattr(ch, "screen_cues", None)
        self._cache: dict[str, tuple] = {}
        if self.sc is not None and hist is not None and not self.sc["track"]:
            self._track(hist)

    def _track(self, hist) -> None:
        """Where each smear's stroke is on every photo: its front follows the front of his body at the face's
        height (the matte of his figure on that photo) once it reaches the face, and never goes back; a timed smear
        starts at his hand at its time and spreads back to the face's left edge over smear_n photos. Then the
        screens' lights on the paint follow the smears."""
        sc, ch = self.sc, self.ch
        st = _step(ch)
        n = max(1, sc["smear_n"])
        ev = {round(q, 4): i for q, i in getattr(hist, "fig_ev", {}).items()}
        photos = ch.photos()
        touched = []
        for s in sc["spec"]["screens"]:
            sm = sc["cue"][s["id"]]["smear"]
            if sm is None:
                continue
            fx0, fy0, fx1, fy1 = s["face"]
            tr, L, R, t_start, c0 = [], None, None, None, None
            for ph in photos:
                if sm["t"] is not None and ph.t < sm["t"] - 1e-6:
                    continue
                i = ev.get(round(ph.q, 4))
                front = None
                if i is not None:
                    bx = hist.fig_box[hist.events[i].q]
                    if bx[2] >= fx0 - 2 and bx[0] <= fx1 + 900:
                        d, X0, Y0 = hist.place(i)
                        r0, r1 = max(0, int(fy0 - Y0)), min(d.shape[0], int(fy1 - Y0))
                        if r1 > r0:
                            cols = np.nonzero((d[r0:r1] > 0.4).any(axis=0))[0]
                            if len(cols):
                                front = float(X0 + cols.max())
                    elif bx[0] > fx1 + 900 and R is not None and R >= fx1:
                        break
                if L is None:
                    if sm["t"] is not None:  # timed: from his hand (or the face's middle, nothing of him there)
                        c0 = float(np.clip(front if front is not None else (fx0 + fx1) / 2, fx0, fx1))
                        L, R, t_start = c0, c0, ph.t
                    elif front is not None and front >= fx0:
                        L, R, t_start = float(fx0), float(fx0), ph.t
                    else:
                        continue
                if front is not None:
                    R = max(R, float(min(front, fx1)))
                if sm["t"] is not None:
                    p = min(1.0, (ph.t - t_start + st) / (n * st))
                    L = c0 - (c0 - fx0) * p
                    R = max(R, min(fx1, c0 + 0.08 * (fy1 - fy0)))  # (the palm's width)
                tr.append((ph.t, float(L), float(R)))
            sc["track"][s["id"]] = tr
            touched.append(s["id"])
        if touched:
            _light_hints(ch, [s["id"] for s in sc["spec"]["screens"]
                              if s["id"] in touched or sc["cue"][s["id"]].get("kill_with")])

    @property
    def active(self) -> bool:
        return self.sc is not None

    def _load(self, sid: str):
        if sid not in self._cache:
            s = next(x for x in self.sc["spec"]["screens"] if x["id"] == sid)
            z = np.load(Path(self.sc["dir"]).parent / s["file"]) if not Path(s["file"]).is_absolute() else np.load(s["file"])
            face, fb = z["face"].astype(np.float32), [int(v) for v in z["face_box"]]
            glow, gb = z["glow"].astype(np.float32), [int(v) for v in z["glow_box"]]
            glow = cv2.resize(glow, (gb[2] - gb[0], gb[3] - gb[1]), interpolation=cv2.INTER_LINEAR)
            self._cache[sid] = (face, fb, glow, gb)
        return self._cache[sid]

    def apply(self, lin: np.ndarray, x0: int, y0: int, t: float) -> np.ndarray:
        """The street box (linear, top-left street x0, y0) with every screen at its level at song time t."""
        if self.sc is None:
            return lin
        h, w = lin.shape[:2]
        out = None
        for s in self.sc["spec"]["screens"]:
            gb = s["glow_box"]
            if gb[2] <= x0 or gb[0] >= x0 + w or gb[3] <= y0 or gb[1] >= y0 + h:
                continue
            k = level(self.ch, s["id"], t)
            smeared = self.sc["cue"][s["id"]]["smear"] is not None and np.diff(extent(self.sc, s["id"], t))[0] >= 2
            if abs(k - 1) < 1e-4 and not smeared:
                continue
            if out is None:
                out = lin.copy()
            face, fb, glow, gb_ = self._load(s["id"])
            for img, b in ((glow, gb_), (face, fb)) if abs(k - 1) >= 1e-4 else ():
                a0, a1 = max(b[0], x0), min(b[2], x0 + w)
                c0, c1 = max(b[1], y0), min(b[3], y0 + h)
                if a1 <= a0 or c1 <= c0:
                    continue
                out[c0 - y0:c1 - y0, a0 - x0:a1 - x0] -= (1 - k) * img[c0 - b[1]:c1 - b[1], a0 - b[0]:a1 - b[0]]
            if smeared:
                out = self._smear(out, s, x0, y0, t)
        if out is None:
            return lin
        return np.clip(out, 0, None)

    def _smear(self, out: np.ndarray, s: dict, x0: int, y0: int, t: float) -> np.ndarray:
        """His wet black stroke across screen s's face at his hand's height (its extent from the track), the
        leading edge feathered like a dragged brush, its lower edge dripping, staying."""
        from .reveal import _n1
        L, R = extent(self.sc, s["id"], t)
        if R - L < 2:
            return out
        fx0, fy0, fx1, fy1 = s["face"]
        fw, fh = fx1 - fx0, fy1 - fy0
        yc = fy0 + self.sc["smear_y"] * fh
        half = 0.26 * fh
        lo = L - (0.03 * fw if L <= fx0 + 1 else 0.0)
        xs = np.arange(int(lo), int(R) + 1, dtype=np.float64)
        h, w = out.shape[:2]
        cols = (xs - x0).astype(int)
        ok = (cols >= 0) & (cols < w)
        if len(xs) < 2 or not ok.any():
            return out
        seed = zlib.crc32(s["id"].encode()) % 10000
        tilt = 0.06 * fh * (xs - fx0) / max(fw, 1)  # (a slight slope down, as a hand drags)
        top = yc - half * (1 + 0.2 * _n1(xs, 0.3 * fh, seed)) + tilt
        bot = yc + half * (1 + 0.25 * _n1(xs, 0.25 * fh, seed + 1)) + tilt
        tip = np.clip((R - xs) / (0.12 * fh), 0, 1) ** 0.6  # the leading edge thins out
        mid = (top + bot) / 2
        top, bot = mid - (mid - top) * (0.35 + 0.65 * tip), mid + (bot - mid) * (0.35 + 0.65 * tip)
        rows = np.arange(y0, y0 + h, dtype=np.float64)[:, None]
        tp, bt = top[None, ok], bot[None, ok]
        band = np.clip((rows - tp) / 2 + 0.5, 0, 1) * np.clip((bt - rows) / 2 + 0.5, 0, 1)
        v = np.clip((rows - tp) / np.maximum(bt - tp, 1), 0, 1)  # across the stroke, 0 at its top
        # dense black with the fingers' streaks along it: thin gaps where the paint ran out, more of them at the
        # leading edge, where the stroke thins out
        sl = rows - tilt[None, ok]
        gap = (np.clip((_n1(sl, 2.2 * self.ch.u, seed + 2) - 0.5) / 0.25, 0, 1)
               * (0.5 + 0.5 * np.clip(_n1(xs[ok], 0.5 * fh, seed + 5)[None, :] + 0.3, 0, 1)))
        m = band * (1 - 0.8 * gap * (1.6 - tip[None, ok]) / 1.6) * np.clip(0.35 + tip[None, ok], 0, 1)
        tr = self.sc["track"][s["id"]]
        t0 = tr[0][0]
        rng = np.random.default_rng(seed)
        for _ in range(int(0.025 * fw) + 3):  # drips from its lower edge, fixed places, running down the glass
            fxp = fx0 + rng.random() * fw
            ln_full = (0.12 + 0.45 * rng.random()) * fh
            if not (L <= fxp <= R - 0.1 * fh):
                continue
            # (when the stroke got there: its age there sets the drip's length)
            ta = next((a[0] for a in tr if a[2] >= fxp and a[1] <= fxp), t0)
            ln = ln_full * min(1.0, max(0.0, t - ta) / 1.2)
            j = min(int(round(fxp - xs[0])), len(bot) - 1)
            r0 = int(bot[j] - y0)
            r1 = int(min(h, r0 + ln))
            jc = int(np.searchsorted(cols[ok], int(fxp - x0)))
            if 0 <= jc < m.shape[1] and r1 > max(r0, 0):
                m[max(r0, 0):r1, max(jc - 1, 0):jc + 2] = np.maximum(m[max(r0, 0):r1, max(jc - 1, 0):jc + 2], 0.9)
        # wet paint: a thin sheen along its upper part (the street's light on it), so it reads on dead glass too
        sheen = np.exp(-((v - 0.18) / 0.05) ** 2) * np.clip(_n1(xs[ok], 0.08 * fh, seed + 3)[None, :], 0, 1) ** 2
        c0 = cols[ok]
        sub = out[:, c0]
        paint = np.float32([0.006, 0.0058, 0.0055]) + np.float32([0.035, 0.037, 0.04]) * sheen[..., None]
        out[:, c0] = sub * (1 - m[..., None]) + paint * m[..., None]
        return out
