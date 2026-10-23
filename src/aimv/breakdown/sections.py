"""sections: song sections on the bar grid, from the lyric tags and the aligned lines.

Each lyric tag becomes a section placed by its lines' word times. Where the vocal is absent for two bars or
more between tagged sections (and before the first / after the last), an instrumental section fills the gap:
a tag with no lines (`[Drop - instrumental]`) names it, otherwise it is an Intro, Outro or Instrumental
(refined to Break or Drop by loudness). Boundaries sit on downbeats: a section starts on the downbeat at or
before its first word, or on the next one when the first words are a pickup (within half a bar of it).
Sections tile 0..duration with no gaps.

`plan()` is shared with the align step so that `section` indices in words.json / lines.json match this file.
Suno's own section tags and novelty peaks (song/suno/) are reported as checks, they do not move boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..util import load_audio, read_json, write_json
from . import lyrics as lyr

NAME = "sections"
MIN_INSTR_BARS = 2
PICKUP_BARS = 0.5


def inputs(p):
    here = Path(__file__).parent
    return [p.bd / "lines.json", p.bd / "words.json", p.bd / "beats.json", p.bd / "meta.json", p.bd / "audio/mix.wav",
            p.song_dir / "lyrics.txt", p.suno_dir / "aligned_lyrics.json", p.suno_dir / "novelty_sections.json",
            here / "lyrics.py"]


def outputs(p):
    return [p.bd / "sections.json"]


# ---------------------------------------------------------------- planning (pure, unit tested)


class _Grid:
    """Downbeats as boundaries 0..n: boundary k is bar k's start; boundary n is the song end."""

    def __init__(self, bars: list[dict], duration: float):
        if not bars:  # no beat grid: fall back to 2 s cells so the tiling still works
            bars = [{"t0": t, "t1": min(t + 2.0, duration)} for t in np.arange(0.0, duration, 2.0)]
        self.starts = np.array([float(b["t0"]) for b in bars])
        self.ends = np.array([float(b.get("t1", b["t0"])) for b in bars])
        self.n = len(bars)
        self.duration = duration

    def time(self, k: int) -> float:
        if k <= 0:
            return 0.0
        if k >= self.n:
            return self.duration
        return float(self.starts[k])

    def bar_of(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, self.n - 1))

    def bar_len(self, k: int) -> float:
        k = int(np.clip(k, 0, self.n - 1))
        return float(self.ends[k] - self.starts[k]) or 2.0

    def start_boundary(self, t: float) -> int:
        """Downbeat at or before t, or the next one when t is a pickup into it."""
        k = self.bar_of(t)
        if t < self.starts[0]:
            return 0
        if k + 1 < self.n and self.starts[k + 1] - t <= PICKUP_BARS * self.bar_len(k):
            return k + 1
        return k

    def end_boundary(self, t: float) -> int:
        """The downbeat after the bar that contains t."""
        if t >= self.duration:
            return self.n
        return min(self.n, self.bar_of(t) + 1)


def _auto(kind: str) -> dict:
    label = {"intro": "Intro", "outro": "Outro"}.get(kind, "Instrumental")
    return {"label": label, "kind": kind, "tag": "", "lyric": None}


def plan(L: lyr.Lyrics, line_times: list[tuple[float, float] | None], bars: list[dict],
         duration: float) -> list[dict]:
    """Tile the song into sections.

    line_times[i] is (t0, t1) of lyric line i, or None when it was not sung. Returns sections with label, kind,
    tag, t0, t1, bar0, bar1 (bar1 exclusive: the next section's bar0), lines, vocal, lyric (lyric section
    index or None)."""
    g = _Grid(bars, duration)
    # Vocal groups in lyric order: sections with at least one sung line.
    groups = []
    for s in L.sections:
        ts = [line_times[li] for li in s.lines if line_times[li] is not None]
        if ts:
            groups.append({"sec": s, "v0": min(t[0] for t in ts), "v1": max(t[1] for t in ts)})
    if not groups:
        return [{"label": "Instrumental", "kind": "instrumental", "tag": "", "lyric": None, "t0": 0.0,
                 "t1": duration, "bar0": 0, "bar1": g.n, "lines": [], "vocal": False}]

    starts = []
    for gi, gr in enumerate(groups):
        b = g.start_boundary(gr["v0"])
        if starts and b <= starts[-1]:
            b = starts[-1] + 1 if starts[-1] + 1 <= g.bar_of(gr["v1"]) else starts[-1]
        starts.append(b)

    # Empty lyric sections (tags with no lines) sit between vocal groups in lyric order.
    def empties_between(a: int | None, b: int | None) -> list[lyr.Section]:
        lo = groups[a]["sec"].i if a is not None else -1
        hi = groups[b]["sec"].i if b is not None else len(L.sections)
        return [s for s in L.sections[lo + 1:hi] if not s.lines]

    out: list[dict] = []

    def emit(b0: int, b1: int, info: dict, lines: list[int], vocal: bool):
        if b1 <= b0:
            return
        if out and out[-1]["bar1"] != b0:
            out[-1]["bar1"] = b0
        out.append({**info, "bar0": b0, "bar1": b1, "lines": lines, "vocal": vocal})

    def fill(b0: int, b1: int, tags: list[lyr.Section], default: str):
        """Instrumental bars [b0, b1): split evenly between the empty tags, or one auto section."""
        if b1 <= b0:
            return
        if not tags:
            emit(b0, b1, _auto(default), [], False)
            return
        tags = tags[: b1 - b0]
        cuts = [b0 + round(k * (b1 - b0) / len(tags)) for k in range(len(tags) + 1)]
        for s, c0, c1 in zip(tags, cuts, cuts[1:]):
            emit(c0, c1, {"label": s.label, "kind": s.kind, "tag": s.tag, "lyric": s.i}, [], False)

    for gi, gr in enumerate(groups):
        s = gr["sec"]
        info = {"label": s.label, "kind": s.kind, "tag": s.tag, "lyric": s.i}
        b0 = starts[gi]
        # Lead-in before the first vocal group.
        if gi == 0 and b0 > 0:
            pre = empties_between(None, 0)
            if b0 >= MIN_INSTR_BARS and (pre or s.kind != "intro"):
                fill(0, b0, pre, "intro")
            else:
                b0 = 0
        nb = starts[gi + 1] if gi + 1 < len(groups) else g.n
        end = min(max(g.end_boundary(gr["v1"]), b0 + 1), nb)
        tags = empties_between(gi, gi + 1 if gi + 1 < len(groups) else None)
        last = gi + 1 == len(groups)
        if nb - end >= MIN_INSTR_BARS and (tags or not last or s.kind != "outro"):
            emit(b0, end, info, list(s.lines), True)
            fill(end, nb, tags, "outro" if last else "instrumental")
        else:
            emit(b0, nb, info, list(s.lines), True)

    # Times from bar boundaries; the first section starts at 0 and the last ends at the song end.
    for k, s in enumerate(out):
        s["t0"] = g.time(s["bar0"]) if k else 0.0
        s["t1"] = g.time(s["bar1"]) if k + 1 < len(out) else duration
    for a, b in zip(out, out[1:]):
        a["t1"] = b["t0"]
    out[-1]["bar1"] = g.n
    return out


# ---------------------------------------------------------------- checks against Suno


def suno_tags(data: dict) -> list[dict]:
    """Section tags from Suno's aligned_words with the time of the first sung word after each tag."""
    words = (data or {}).get("aligned_words") or []
    tags, cur, in_tag, pending = [], [], False, []
    for w in words:
        raw = str(w.get("word") or "")
        start = w.get("start_s")
        text = []
        for ch in raw:
            if ch == "[":
                in_tag, cur = True, []
            elif ch == "]":
                in_tag = False
                pending.append("".join(cur).strip())
            elif in_tag:
                cur.append(ch)
            else:
                text.append(ch)
        if pending and lyr.normalize("".join(text)) and start is not None:
            # Only the last of several stacked tags is sung; earlier ones were empty (instrumental) sections.
            tags.append({"tag": pending[-1], "t": float(start)})
            pending = []
    return [t for t in tags if not re.fullmatch(r"end", t["tag"], flags=re.I)]


def _checks(p, out: list[dict], words_meta: dict, log) -> dict:
    checks: dict = {}
    offset = (words_meta.get("suno_offset_ms") or 0.0) / 1000.0
    al = p.suno("aligned_lyrics")
    if al:
        st = suno_tags(al)
        voc = [s for s in out if s["vocal"]]
        rows = []
        k = 0
        for s in voc:
            label = lyr.kind_for(s["tag"].split(" - ")[0]) if s["tag"] else s["kind"]
            for j in range(k, len(st)):
                if lyr.kind_for(st[j]["tag"].split(" - ")[0]) == label:
                    rows.append({"section": s["label"], "suno_tag": st[j]["tag"], "suno_t": st[j]["t"] - offset,
                                 "first_word": s.get("_v0"), "diff": (s.get("_v0") or 0) - (st[j]["t"] - offset)})
                    k = j + 1
                    break
        if rows:
            d = np.abs([r["diff"] for r in rows])
            checks["suno_tags"] = {"matched": len(rows), "of": len(voc), "median_s": float(np.median(d)),
                                   "max_s": float(d.max()), "rows": rows}
            log(f"suno tags: {len(rows)}/{len(voc)} vocal sections matched, median {np.median(d):.2f}s apart")
    nov = p.suno("novelty_sections")
    if isinstance(nov, dict) and nov.get("peak_times"):
        peaks = np.array(nov["peak_times"], float) - offset
        inner = [s["t0"] for s in out[1:]]
        if inner:
            near = [float(np.min(np.abs(peaks - t))) for t in inner]
            checks["novelty"] = {"boundaries": len(inner), "within_2s": int(sum(x <= 2.0 for x in near)),
                                 "median_s": float(np.median(near))}
            labels = nov.get("segment_labels") or []
            if len(labels) == len(peaks) + 1:
                edges = np.concatenate([[0.0], peaks, [np.inf]])
                for s in out:
                    mid = (s["t0"] + s["t1"]) / 2
                    s["_novelty"] = labels[int(np.searchsorted(edges, mid, side="right") - 1)]
                checks["novelty"]["labels"] = [{"section": s["label"], "label": s.get("_novelty")} for s in out]
            log(f"suno novelty: {checks['novelty']['within_2s']}/{len(inner)} boundaries within 2 s of a peak")
    return checks


# ---------------------------------------------------------------- run


def _loudness(p) -> tuple[np.ndarray, float]:
    """Mix loudness per 50 ms frame, dB mapped to 0..1 over the song (p5 = 0, p99 = 1)."""
    x, sr = load_audio(p.bd / "audio/mix.wav", mono=True)
    hop = int(sr * 0.05)
    n = x.shape[1] // hop
    fr = x[0, : n * hop].reshape(n, hop).astype(np.float64)
    dbv = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    lo, hi = np.percentile(dbv, 5), np.percentile(dbv, 99)
    return np.clip((dbv - lo) / max(hi - lo, 1e-6), 0, 1), 0.05


def run(p, log):
    lines = read_json(p.bd / "lines.json")
    beats = read_json(p.bd / "beats.json")
    meta = read_json(p.bd / "meta.json")
    words_meta = read_json(p.bd / "words.json") if (p.bd / "words.json").exists() else {}
    duration = float(meta["duration"])
    L = lyr.parse_file(p.song_dir / "lyrics.txt")
    lt = [(ln["t0"], ln["t1"]) if ln.get("t0") is not None and ln.get("sung", True) else None
          for ln in lines["lines"]]
    if len(lt) != len(L.lines):
        raise RuntimeError("lines.json does not match lyrics.txt; re-run align")
    out = plan(L, lt, beats.get("bars", []), duration)

    line_sec = {li: k for k, s in enumerate(out) for li in s["lines"]}
    mismatch = [ln["i"] for ln in lines["lines"] if ln.get("section") != line_sec.get(ln["i"])]
    if mismatch:
        log(f"warning: lines {mismatch[:8]} carry a different section index in lines.json; re-run align")

    env, hop = _loudness(p)
    for s in out:
        a, b = int(s["t0"] / hop), max(int(s["t0"] / hop) + 1, int(s["t1"] / hop))
        s["energy"] = float(env[a:b].mean()) if b <= len(env) else float(env[a:].mean() if a < len(env) else 0)
        ts = [lt[li] for li in s["lines"] if lt[li] is not None]
        s["_v0"] = min(t[0] for t in ts) if ts else None

    # An untagged instrumental stretch quieter than both neighbours is a break. Drops are not guessed here:
    # loudness alone cannot tell a drop from a build, so those stay "instrumental" unless a tag says so.
    for k, s in enumerate(out):
        if s["lyric"] is None and s["kind"] == "instrumental" and 0 < k < len(out) - 1:
            if s["energy"] <= min(out[k - 1]["energy"], out[k + 1]["energy"]) - 0.05:
                s["kind"], s["label"] = "break", "Break"

    checks = _checks(p, out, words_meta, log)
    sections = [{"i": k, "label": s["label"], "kind": s["kind"], "tag": s["tag"], "t0": s["t0"], "t1": s["t1"],
                 "bar0": s["bar0"], "bar1": s["bar1"], "lines": s["lines"], "energy": s["energy"],
                 "vocal": s["vocal"]} for k, s in enumerate(out)]
    write_json(p.bd / "sections.json", {"version": 1, "sections": sections, "checks": checks})
    for s in sections:
        log(f"{s['i']:2d} {s['label']:<14} {s['kind']:<12} bars {s['bar0']:3d}-{s['bar1']:3d} "
            f"{s['t0']:7.2f}-{s['t1']:7.2f}  energy {s['energy']:.2f} {'vocal' if s['vocal'] else ''}")
