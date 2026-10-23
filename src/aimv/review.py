"""`aimv review <project>`: a local page for watching a render and leaving notes on it.

Notes are either about a part of the video (a point or a range of song time, with a snapshot of the frame) or general
(the whole film), and carry the bar address, section and segment they fall in. Older notes may also carry a kind (fix,
idea, like, question) and drawn marks. They live
in projects/<p>/review/: notes.json (the record, read by the lead), notes.md (the same, readable, regenerated on every
change) and frames/<id>.jpg. Deleted notes move to the file's "deleted" list; nothing is erased.

Videos are the review copies in paint/renders (`*-review-16x9.mp4`, `*-review-9x16.mp4`). A render of a window (for
example review-final-144.00-180.00) is mapped to song time by its steps log's t0. Segments come from
review/segments.json when it exists (id, label, t0, t1), else the song's sections stand in.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from bisect import bisect_right
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from .serve import RangeHandler

PAGE = Path(__file__).with_name("review_page.html")
KINDS = ("fix", "idea", "like", "question")
_lock = threading.Lock()


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write_atomic(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


class Song:
    """The song's grid for addresses: bars, beats, sections, segments, lyric lines."""

    def __init__(self, root: Path):
        bd = root / "breakdown"
        beats = _read(bd / "beats.json", {})
        self.bars = [b["t0"] for b in beats.get("bars", [])]
        self.beats = [(b["t"], b["bar"], b["beat"]) for b in beats.get("beats", [])]
        self.beat_t = [b[0] for b in self.beats]
        self.sections = [{"label": s["label"], "kind": s["kind"], "t0": s["t0"], "t1": s["t1"]}
                         for s in _read(bd / "sections.json", {}).get("sections", [])]
        song = _read(bd / "song.json", {})
        self.duration = float(song.get("duration") or song.get("duration_s") or
                              (self.sections[-1]["t1"] if self.sections else 180.0))
        words = _read(bd / "words.json", {}).get("words", [])
        self.lines = []
        for ln in _read(bd / "lines.json", {}).get("lines", []):
            if ln.get("t0") is None:
                continue
            ws = [{"text": words[i]["text"], "t0": words[i]["t0"], "t1": words[i]["t1"]}
                  for i in ln.get("words", []) if i < len(words) and words[i].get("t0") is not None]
            self.lines.append({"text": ln["text"], "t0": ln["t0"], "t1": ln["t1"], "words": ws})
        seg = _read(root / "review" / "segments.json", None)
        self.segments = (seg.get("segments", []) if isinstance(seg, dict) else seg) or [
            {"id": str(i + 1), "label": s["label"], "t0": s["t0"], "t1": s["t1"]} for i, s in enumerate(self.sections)]

    def address(self, t: float) -> str:
        """bar.beat.16th, 1-based (the project's convention, e.g. 43.2.1)."""
        if not self.beats:
            return ""
        i = max(0, bisect_right(self.beat_t, t) - 1)
        bt, bar, beat = self.beats[i]
        nxt = self.beat_t[i + 1] if i + 1 < len(self.beat_t) else bt + 0.343
        six = min(3, max(0, int((t - bt) / max(nxt - bt, 1e-6) * 4)))
        return f"{bar + 1}.{beat + 1}.{six + 1}"

    def _span(self, items, t):
        for s in items:
            if s["t0"] <= t < s["t1"]:
                return s
        return items[-1] if items and t >= items[-1]["t0"] else None

    def section(self, t: float) -> str:
        s = self._span(self.sections, t)
        return s["label"] if s else ""

    def segment(self, t: float) -> dict | None:
        return self._span(self.segments, t)


def _videos(root: Path) -> list[dict]:
    out = []
    rd = root / "paint" / "renders"
    for f in sorted(rd.glob("*-review-*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True):
        m = re.match(r"(.+)-review-(16x9|9x16)\.mp4$", f.name)
        if not m:
            continue
        steps = _read(rd / f"{m.group(1)}-steps.json", {})
        t0 = float(steps.get("t0", 0.0) or 0.0)
        t1 = steps.get("t1")
        out.append({"name": f.name, "url": f"/paint/renders/{f.name}", "base": m.group(1),
                    "aspect": m.group(2).replace("x", ":"), "t0": t0, "t1": float(t1) if t1 is not None else None,
                    "mtime": f.stat().st_mtime})
    newest = {}
    for x in out:
        newest[x["base"]] = max(newest.get(x["base"], 0), x["mtime"])
    out.sort(key=lambda x: (-newest[x["base"]], x["aspect"] != "16:9"))  # newest render first, 16:9 before 9:16
    return out


def _fmt(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


class Notes:
    def __init__(self, root: Path, song: Song):
        self.dir = root / "review"
        self.path = self.dir / "notes.json"
        self.song = song

    def load(self) -> dict:
        d = _read(self.path, None) or {"version": 1, "notes": [], "deleted": []}
        d.setdefault("notes", [])
        d.setdefault("deleted", [])
        return d

    def _enrich(self, n: dict) -> dict:
        if n.get("scope") == "overall" or n.get("t0") is None:
            n.update(scope="overall", t0=None, t1=None, bar=None, section=None, segment=None)
            return n
        t0 = float(n["t0"])
        t1 = float(n["t1"]) if n.get("t1") not in (None, "") else None
        if t1 is not None and t1 < t0:
            t0, t1 = t1, t0
        if t1 is not None and t1 - t0 < 0.05:
            t1 = None
        seg = self.song.segment(t0)
        n.update(t0=round(t0, 3), t1=round(t1, 3) if t1 is not None else None,
                 scope="range" if t1 is not None else "point",
                 bar=self.song.address(t0) + (f" to {self.song.address(t1)}" if t1 is not None else ""),
                 section=self.song.section(t0), segment=f"{seg['id']} {seg['label']}" if seg else None)
        return n

    def _frame(self, nid: str, b64: str | None) -> str | None:
        if not b64:
            return None
        raw = base64.b64decode(b64.split(",", 1)[-1])
        name = f"frames/{nid}-{int(time.time())}.jpg"
        _write_atomic(self.dir / name, raw)
        return name

    def save(self, d: dict) -> None:
        _write_atomic(self.path, json.dumps(d, indent=1, ensure_ascii=False).encode())
        _write_atomic(self.dir / "notes.md", self.markdown(d).encode())

    def create(self, body: dict) -> dict:
        with _lock:
            d = self.load()
            nid = secrets.token_hex(4)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            n = {"id": nid, "kind": body.get("kind") if body.get("kind") in KINDS else "fix",
                 "text": (body.get("text") or "").strip(), "status": "open", "t0": body.get("t0"),
                 "t1": body.get("t1"), "scope": body.get("scope"), "video": body.get("video"),
                 "marks": body.get("marks") or [], "created": now, "updated": now}
            n["frame"] = self._frame(nid, body.get("frame"))
            d["notes"].append(self._enrich(n))
            self.save(d)
            return n

    def update(self, nid: str, body: dict) -> dict | None:
        with _lock:
            d = self.load()
            n = next((x for x in d["notes"] if x["id"] == nid), None)
            if n is None:
                return None
            for k in ("text", "status", "t0", "t1", "scope", "marks"):
                if k in body:
                    n[k] = body[k].strip() if k == "text" else body[k]
            if body.get("kind") in KINDS:
                n["kind"] = body["kind"]
            if body.get("frame"):
                n["frame"] = self._frame(nid, body["frame"])
                n["video"] = body.get("video", n.get("video"))
            n["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._enrich(n)
            self.save(d)
            return n

    def delete(self, nid: str) -> bool:
        with _lock:
            d = self.load()
            n = next((x for x in d["notes"] if x["id"] == nid), None)
            if n is None:
                return False
            d["notes"].remove(n)
            n["deleted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            d["deleted"].append(n)
            self.save(d)
            return True

    def markdown(self, d: dict) -> str:
        notes = d["notes"]
        lines = ["# Review notes", "",
                 f"{len(notes)} notes ({sum(n['status'] == 'open' for n in notes)} open). Times are song time; the "
                 "bar address is bar.beat.16th. Written by `aimv review`; edit notes in the review page.", ""]

        def item(n):
            when = "" if n["scope"] == "overall" else (
                f"{_fmt(n['t0'])}" + (f" to {_fmt(n['t1'])}" if n.get("t1") is not None else "") + f" (bar {n['bar']}"
                + (f", {n['section']}" if n.get("section") else "") + ") ")
            done = " (done)" if n["status"] == "done" else ""
            kind = f"**{n['kind']}** " if n.get("kind") not in (None, "fix") else ""  # older notes carry a kind
            out = [f"- {kind}{when}{n['text'] or '(no text)'}{done}"]
            if n.get("frame"):
                out.append(f"  ![{n['id']}]({n['frame']})")
            if n.get("video"):
                out.append(f"  <sub>{n['id']}, on {n['video']}</sub>")
            return out

        overall = [n for n in notes if n["scope"] == "overall"]
        if overall:
            lines += ["## General", ""]
            for n in overall:
                lines += item(n)
            lines.append("")
        timed = sorted((n for n in notes if n["scope"] != "overall"), key=lambda n: n["t0"])
        seg = None
        for n in timed:
            if n.get("segment") != seg:
                seg = n.get("segment")
                lines += ["", f"## Segment {seg}" if seg else "## (no segment)", ""]
            lines += item(n)
        return "\n".join(lines).rstrip() + "\n"


class ReviewHandler(RangeHandler):
    root: Path
    song: Song
    notes: Notes

    def _json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/meta":
            s = self.song
            return self._json({"project": self.root.name, "duration": s.duration, "bars": s.bars,
                               "beats": [[t, bar, beat] for t, bar, beat in s.beats], "sections": s.sections,
                               "segments": s.segments, "lines": s.lines, "videos": _videos(self.root)})
        if path == "/api/notes":
            return self._json(self.notes.load())
        if path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        if not (path.startswith("/paint/renders/") or path.startswith("/review/")):
            return self._json({"error": "not found"}, 404)
        return super().do_GET()

    def do_HEAD(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/paint/renders/") or path.startswith("/review/"):
            return super().do_HEAD()
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/notes":
            return self._json({"error": "not found"}, 404)
        try:
            return self._json(self.notes.create(self._body()), 201)
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)

    def do_PUT(self):
        m = re.match(r"^/api/notes/([0-9a-f]+)$", self.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        try:
            n = self.notes.update(m.group(1), self._body())
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        return self._json(n) if n else self._json({"error": "no such note"}, 404)

    def do_DELETE(self):
        m = re.match(r"^/api/notes/([0-9a-f]+)$", self.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        return self._json({"ok": True}) if self.notes.delete(m.group(1)) else self._json({"error": "no such note"}, 404)


def run(root: Path, port: int = 8770, open_page: bool = True) -> None:
    song = Song(root)
    handler = type("H", (ReviewHandler,), {"root": root, "song": song, "notes": Notes(root, song)})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(root)))
    url = f"http://127.0.0.1:{port}/"
    print(f"review page for {root.name} at {url} (ctrl-c to stop); notes in {root / 'review'}")
    if open_page:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
