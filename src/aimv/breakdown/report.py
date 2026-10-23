"""report: the index (song.json), a readable TIMELINE.md, a visual report.html and an audible check.wav.

The two QA artefacts answer different questions:
  report.html  "is every word, beat and section where it should be?" (by eye, with playback)
  check.wav    "does the grid sit on the music?" (by ear: clicks on beats, ticks on word starts)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..util import load_audio, read_json, save_audio, write_json

NAME = "report"
CONFIG_KEYS = ("title",)
TEMPLATE = Path(__file__).with_name("report_template.html")
PEAKS_PER_S = 100  # waveform resolution in the html (10 ms buckets)

DATA_FILES = ["meta", "beats", "words", "lines", "sections", "hits", "envelopes"]


def inputs(p):
    return [p.bd / f"{n}.json" for n in DATA_FILES] + [p.bd / "audio/mix.wav", p.bd / "stems/vocals.wav", TEMPLATE]


def outputs(p):
    return [p.bd / "song.json", p.bd / "TIMELINE.md", p.bd / "report.html", p.bd / "check.wav"]


def _load(p) -> dict:
    out = {}
    for n in DATA_FILES:
        f = p.bd / f"{n}.json"
        out[n] = read_json(f) if f.exists() else None
    return out


def _mmss(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:06.3f}"


def _bar(i) -> str:
    """Bars are 0-based in the JSON and 1-based for people."""
    return "?" if i is None else str(int(i) + 1)


def _qa(d: dict) -> list[str]:
    qa = []
    if not d["beats"]:
        qa.append("beats.json missing")
    else:
        ph = d["beats"].get("phase_check", {})
        if ph.get("shifted_ms"):
            qa.append(f"beat grid shifted {ph['shifted_ms']:+.0f} ms onto the kicks ({ph.get('note', '')})")
        if ph.get("note") and not ph.get("shifted_ms"):
            qa.append(f"beat phase: {ph['note']}")
    if not d["words"]:
        qa.append("words.json missing (no lyrics.txt?)")
    else:
        words = d["words"]["words"]
        flagged = [w for w in words if w.get("flag")]
        if flagged:
            kinds = {}
            for w in flagged:
                kinds[w["flag"]] = kinds.get(w["flag"], 0) + 1
            qa.append(f"{len(flagged)} of {len(words)} words flagged: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
        agr = d["words"].get("suno_agreement")
        if agr and agr.get("p90_ms") and agr["p90_ms"] > 150:
            qa.append(f"local and Suno word times disagree (p90 {agr['p90_ms']:.0f} ms)")
    for n in ("sections", "hits", "envelopes"):
        if not d[n]:
            qa.append(f"{n}.json missing")
    return qa


def _song_json(p, d: dict, qa: list[str]) -> dict:
    meta = d["meta"] or {}
    beats = d["beats"] or {}
    words = (d["words"] or {}).get("words", [])
    secs = (d["sections"] or {}).get("sections", [])
    hits = (d["hits"] or {}).get("counts", {})
    return {
        "version": 1,
        "title": p.config.get("title", p.root.name),
        "duration": meta.get("duration"),
        "lufs_integrated": meta.get("lufs_integrated"),
        "key": meta.get("key"),
        "meter": beats.get("meter"),
        "bpm_median": beats.get("bpm_median"),
        "bpm_range": beats.get("bpm_range"),
        "counts": {
            "beats": len(beats.get("beats", [])),
            "bars": len(beats.get("bars", [])),
            "sections": len(secs),
            "lines": len((d["lines"] or {}).get("lines", [])),
            "words": len(words),
            "words_flagged": sum(1 for w in words if w.get("flag")),
            **{f"hits_{k}": v for k, v in hits.items()},
        },
        "sections": [{"label": s["label"], "kind": s["kind"], "t0": s["t0"], "t1": s["t1"],
                      "bar0": s.get("bar0"), "bar1": s.get("bar1")} for s in secs],
        "files": {
            "soundtrack": "audio/mix.wav",
            "stems": sorted(str(f.relative_to(p.bd)) for f in (p.bd / "stems").glob("*.wav")),
            **{n: f"{n}.json" for n in DATA_FILES if d[n] is not None},
            "timeline": "TIMELINE.md", "report": "report.html", "check": "check.wav",
        },
        "qa": qa,
    }


def _timeline(p, d: dict, song: dict) -> str:
    beats = d["beats"] or {}
    secs = (d["sections"] or {}).get("sections", [])
    lines = (d["lines"] or {}).get("lines", [])
    words = (d["words"] or {}).get("words", [])
    out = [f"# {song['title']}: timeline", "", "Bars are counted from 1 here (from 0 in the JSON files).", ""]
    out.append(f"- Duration: {_mmss(song['duration'] or 0)} ({song['duration']:.3f} s)")
    if beats:
        lo, hi = beats.get("bpm_range", [None, None])
        out.append(f"- Tempo: median {beats['bpm_median']:.2f} BPM (5th to 95th percentile {lo:.2f} to {hi:.2f}), {beats['meter']}/4, "
                   f"{song['counts']['bars']} bars, first downbeat {beats['bars'][0]['t0']:.3f} s" if beats.get("bars") else "")
    k = song.get("key") or {}
    out.append(f"- Key: {k.get('local')}" + (f" (Suno: {k['suno']})" if k.get("suno") else ""))
    out.append(f"- Loudness: {song['lufs_integrated']:.1f} LUFS")
    out.append(f"- Words: {song['counts']['words']} ({song['counts']['words_flagged']} flagged)")
    if song["qa"]:
        out += ["", "## QA", ""] + [f"- {q}" for q in song["qa"]]
    out += ["", "## Sections", "", "| # | Section | Kind | Start | End | Bars | Vocal |", "|---|---|---|---|---|---|---|"]
    for s in secs:
        out.append(f"| {s['i']} | {s['label']} | {s['kind']} | {_mmss(s['t0'])} | {_mmss(s['t1'])} | "
                   f"{_bar(s.get('bar0'))} to {_bar(s.get('bar1'))} | {'yes' if s.get('vocal') else ''} |")
    out += ["", "## Lines", ""]
    cur = None
    for ln in lines:
        if ln.get("section") != cur:
            cur = ln.get("section")
            label = next((s["label"] for s in secs if s["i"] == cur), f"Section {cur}")
            out += ["", f"### {label}", ""]
        flags = [words[i]["flag"] for i in ln.get("words", []) if i < len(words) and words[i].get("flag")]
        note = f"  ({len(flags)} flagged)" if flags else ""
        t0, t1 = ln.get("t0"), ln.get("t1")
        when = f"{_mmss(t0)} to {_mmss(t1)}" if t0 is not None else "not found"
        out.append(f"- `{when}` bar {_bar(ln.get('bar0'))}: {ln['text']}{note}")
    return "\n".join(out) + "\n"


def _click(sr: int, freq: float, dur: float, gain: float) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (gain * np.sin(2 * np.pi * freq * t) * np.exp(-t / (dur / 5))).astype(np.float32)


def _check_wav(p, d: dict) -> None:
    x, sr = load_audio(p.bd / "audio/mix.wav")
    out = x * 0.5  # -6 dB so the clicks sit on top
    n = out.shape[1]

    def place(times, snd, chans=(0, 1)):
        for t in times:
            i = int(round(t * sr))
            if 0 <= i < n:
                m = min(len(snd), n - i)
                for c in chans:
                    out[c, i:i + m] += snd[:m]

    if d["beats"]:
        beats = d["beats"]["beats"]
        place([b["t"] for b in beats if b["beat"] != 0], _click(sr, 1500, 0.04, 0.35))
        place([b["t"] for b in beats if b["beat"] == 0], _click(sr, 2500, 0.06, 0.55))
    if d["words"]:
        # Word starts on the right channel only, so they can be told apart from the beat clicks.
        place([w["t0"] for w in d["words"]["words"] if w.get("t0") is not None], _click(sr, 5000, 0.02, 0.3), chans=(1,))
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    save_audio(p.bd / "check.wav", out, sr, subtype="PCM_16")


def _peaks(path: Path) -> list[float]:
    x, sr = load_audio(path, mono=True)
    hop = sr // PEAKS_PER_S
    nb = x.shape[1] // hop
    v = np.abs(x[0, : nb * hop]).reshape(nb, hop).max(axis=1)
    v = v / max(v.max(), 1e-9)
    return np.round(v, 3).tolist()


def _encode_preview(src: Path, dst: Path) -> bool:
    # FLAC, not mp3/aac: lossy codecs add encoder padding that would shift playback against the data.
    if not shutil.which("ffmpeg"):
        return False
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "flac", "-sample_fmt", "s16", str(dst)], check=True)
    return True


def _html(p, d: dict, song: dict) -> str:
    _encode_preview(p.bd / "audio/mix.wav", p.bd / "audio/preview-mix.flac")
    _encode_preview(p.bd / "check.wav", p.bd / "audio/preview-check.flac")
    data = {
        "title": song["title"],
        "song": song,
        "duration": song["duration"],
        "peaks": {"mix": _peaks(p.bd / "audio/mix.wav"),
                  "vocals": _peaks(p.bd / "stems/vocals.wav") if (p.bd / "stems/vocals.wav").exists() else []},
        "peaksPerS": PEAKS_PER_S,
        "beats": (d["beats"] or {}).get("beats", []),
        "bars": (d["beats"] or {}).get("bars", []),
        "tempo": (d["beats"] or {}).get("tempo_curve", []),
        "sections": (d["sections"] or {}).get("sections", []),
        "lines": (d["lines"] or {}).get("lines", []),
        "gaps": (d["lines"] or {}).get("gaps", []),
        "words": (d["words"] or {}).get("words", []),
        "hits": (d["hits"] or {}).get("events", {}),
        "audio": {"mix": "audio/preview-mix.flac", "check": "audio/preview-check.flac"},
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("__TITLE__", song["title"]).replace("__DATA__", blob)


def run(p, log):
    d = _load(p)
    qa = _qa(d)
    song = _song_json(p, d, qa)
    write_json(p.bd / "song.json", song)
    (p.bd / "TIMELINE.md").write_text(_timeline(p, d, song))
    _check_wav(p, d)
    (p.bd / "report.html").write_text(_html(p, d, song))
    for q in qa:
        log(f"QA: {q}")
    log(f"open {p.bd / 'report.html'}")
