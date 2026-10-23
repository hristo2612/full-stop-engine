"""aimv review: the song grid's addresses and the notes store (create, update, delete into the deleted list, the
Markdown). Nothing is served."""

import json

from aimv.review import Notes, Song


def project(tmp_path):
    bd = tmp_path / "breakdown"
    bd.mkdir()
    beats = [{"i": i, "t": i * 0.5, "bar": i // 4, "beat": i % 4} for i in range(64)]
    bars = [{"i": b, "t0": b * 2.0, "t1": b * 2.0 + 2.0, "beats": 4} for b in range(16)]
    (bd / "beats.json").write_text(json.dumps({"beats": beats, "bars": bars}))
    (bd / "sections.json").write_text(json.dumps({"sections": [
        {"label": "Verse 1", "kind": "verse", "t0": 0.0, "t1": 16.0},
        {"label": "Chorus", "kind": "chorus", "t0": 16.0, "t1": 32.0}]}))
    (bd / "song.json").write_text(json.dumps({"duration": 32.0}))
    (tmp_path / "review").mkdir()
    (tmp_path / "review" / "segments.json").write_text(json.dumps({"segments": [
        {"id": "1", "label": "Boxed", "t0": 0.0, "t1": 10.0}, {"id": "2", "label": "Out", "t0": 10.0, "t1": 32.0}]}))
    return tmp_path


def test_song_addresses(tmp_path):
    s = Song(project(tmp_path))
    assert s.duration == 32.0
    assert s.address(0.0) == "1.1.1"
    assert s.address(2.0) == "2.1.1"
    assert s.address(2.5 + 0.13) == "2.2.2"  # a quarter of the way into beat 2: the second 16th
    assert s.section(17.0) == "Chorus" and s.segment(12.0)["label"] == "Out"


def test_notes_store(tmp_path):
    root = project(tmp_path)
    notes = Notes(root, Song(root))
    a = notes.create({"text": "the pole crosses him", "kind": "fix", "t0": 12.6, "t1": 11.0})
    assert a["scope"] == "range" and a["t0"] == 11.0 and a["t1"] == 12.6  # swapped into order
    assert a["segment"] == "2 Out" and a["bar"].startswith("6.3.1 to 7.2.")
    b = notes.create({"text": "too dark overall", "kind": "nope", "scope": "overall", "t0": 5.0})
    assert b["scope"] == "overall" and b["t0"] is None and b["kind"] == "fix"
    c = notes.create({"text": "nice", "kind": "like", "t0": 3.0, "frame": "data:image/jpeg;base64,/9j/AA=="})
    assert c["scope"] == "point" and (root / "review" / c["frame"]).read_bytes() == b"\xff\xd8\xff\x00"
    notes.update(a["id"], {"status": "done", "text": " fixed "})
    assert notes.delete(b["id"]) and not notes.delete("nothere")
    d = json.loads((root / "review" / "notes.json").read_text())
    assert [n["id"] for n in d["notes"]] == [a["id"], c["id"]] and d["deleted"][0]["id"] == b["id"]
    assert d["notes"][0]["status"] == "done" and d["notes"][0]["text"] == "fixed"
    md = (root / "review" / "notes.md").read_text()
    assert "## Segment 1 Boxed" in md and "## Segment 2 Out" in md and "(done)" in md and "too dark" not in md
