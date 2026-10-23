"""Section planning: tiling, bar snapping, pickups, instrumental gaps, tags without lines; plus vocal gaps."""

import numpy as np
import pytest

from aimv.breakdown import align
from aimv.breakdown.lyrics import parse
from aimv.breakdown.sections import plan, suno_tags

BAR = 2.0  # 120 BPM in 4/4


def bars(n, start=0.25):
    return [{"i": k, "t0": start + k * BAR, "t1": start + (k + 1) * BAR, "beats": 4} for k in range(n)]


def downbeat(k, start=0.25):
    return start + k * BAR


def check_tiling(out, duration, grid):
    assert out[0]["t0"] == 0.0 and out[0]["bar0"] == 0
    assert out[-1]["t1"] == pytest.approx(duration) and out[-1]["bar1"] == len(grid)
    for a, b in zip(out, out[1:]):
        assert a["t1"] == pytest.approx(b["t0"])  # no gaps, no overlaps
        assert a["bar1"] == b["bar0"]
        assert a["t1"] > a["t0"]
        assert any(abs(b["t0"] - g["t0"]) < 1e-9 for g in grid)  # inner boundaries on downbeats


LYR = """[Intro - spoken]
Ladies and gentlemen

[Verse 1 - spoken]
Line one of the verse
Line two of the verse

[Chorus - sung]
Chorus line one
Chorus line two

[Outro - spoken]
Goodbye
"""


def times_for(L, spans):
    return [spans.get(li) for li in range(len(L.lines))]


def test_tiles_whole_song_on_downbeats():
    L = parse(LYR)
    g = bars(30)
    dur = 61.0
    lt = times_for(L, {0: (4.5, 7.0), 1: (14.4, 17.0), 2: (17.2, 21.0), 3: (22.3, 26.0), 4: (26.3, 29.5),
                       5: (40.4, 42.0)})
    out = plan(L, lt, g, dur)
    check_tiling(out, dur, g)
    labels = [(s["label"], s["kind"], s["vocal"]) for s in out]
    assert labels == [
        ("Intro", "intro", True),            # lead-in folded into the tagged intro
        ("Instrumental", "instrumental", False),
        ("Verse 1", "verse", True),
        ("Chorus", "chorus", True),
        ("Instrumental", "instrumental", False),
        ("Outro", "outro", True),             # runs to the end of the song
    ]
    assert out[2]["lines"] == [1, 2] and out[3]["lines"] == [3, 4]
    assert out[0]["tag"] == "Intro - spoken" and out[1]["tag"] == "" and out[1]["lyric"] is None


def test_section_starts_at_or_before_first_word_with_pickups():
    L = parse("[Verse]\nOne\n[Chorus]\nTwo\n[Bridge]\nThree\n")
    g = bars(20)
    # chorus: first word 0.3 s before downbeat 6 (a pickup) -> starts on downbeat 6
    # bridge: first word 0.4 s after downbeat 10 -> starts on downbeat 10
    lt = [(downbeat(1) + 0.1, downbeat(4)), (downbeat(6) - 0.3, downbeat(8)), (downbeat(10) + 0.4, downbeat(12))]
    out = plan(L, lt, g, 40.5)
    by = {s["label"]: s for s in out}
    assert by["Chorus"]["bar0"] == 6
    assert by["Bridge"]["bar0"] == 10
    assert by["Verse"]["t0"] == 0.0  # a lead-in shorter than 2 bars joins the first section


def test_short_gap_is_not_an_instrumental():
    L = parse("[Verse]\nOne\n[Chorus]\nTwo\n")
    g = bars(12)
    # verse ends in bar 3, chorus starts at downbeat 5: only bar 4 is free (< 2 bars)
    lt = [(downbeat(0) + 0.2, downbeat(3) + 1.0), (downbeat(5) + 0.1, downbeat(10) + 1.0)]
    out = plan(L, lt, g, 24.5)
    assert [s["label"] for s in out] == ["Verse", "Chorus"]
    assert out[0]["bar1"] == 5


def test_tags_without_lines_name_the_gap():
    L = parse("[Intro - solo piano]\n\n[Verse 1]\nWords here\n\n[Drop - instrumental]\n\n[Break - strings]\n\n"
              "[Chorus]\nMore words\n[Outro - piano fades]\n\n[End]\n")
    g = bars(40)
    lt = [(downbeat(4) + 0.1, downbeat(8)), (downbeat(20) + 0.1, downbeat(24))]
    out = plan(L, lt, g, 80.5)
    check_tiling(out, 80.5, g)
    got = [(s["label"], s["kind"], s["bar0"], s["bar1"]) for s in out]
    assert got == [
        ("Intro", "intro", 0, 4),
        ("Verse 1", "verse", 4, 9),
        ("Drop", "drop", 9, 15),
        ("Break", "break", 15, 20),
        ("Chorus", "chorus", 20, 25),
        ("Outro", "outro", 25, 40),
    ]
    assert out[2]["tag"] == "Drop - instrumental" and not out[2]["vocal"]


def test_auto_intro_and_outro_when_untagged():
    L = parse("[Verse]\nWords\n[Chorus]\nMore\n")
    g = bars(20)
    lt = [(downbeat(3) + 0.1, downbeat(6)), (downbeat(7) + 0.1, downbeat(10) + 0.5)]
    out = plan(L, lt, g, 40.5)
    assert [(s["label"], s["kind"]) for s in out] == [("Intro", "intro"), ("Verse", "verse"), ("Chorus", "chorus"),
                                                       ("Outro", "outro")]
    check_tiling(out, 40.5, g)


def test_unsung_section_is_dropped_and_its_lines_unassigned():
    L = parse("[Verse]\nA\n[Chorus]\nB\n[Verse 2]\nC\n")
    g = bars(20)
    lt = [(downbeat(1), downbeat(3)), None, (downbeat(4) + 0.2, downbeat(6))]
    out = plan(L, lt, g, 40.5)
    assert [s["label"] for s in out] == ["Verse", "Verse 2", "Outro"]
    assert all(1 not in s["lines"] for s in out)


def test_no_beat_grid_still_tiles():
    L = parse("[Verse]\nA\n[Chorus]\nB\n")
    out = plan(L, [(1.0, 5.0), (12.3, 15.0)], [], 20.0)
    assert out[0]["t0"] == 0.0 and out[-1]["t1"] == 20.0
    for a, b in zip(out, out[1:]):
        assert a["t1"] == b["t0"]


def test_no_vocals_at_all():
    L = parse("[Verse]\nA\n")
    out = plan(L, [None], bars(10), 20.5)
    assert len(out) == 1 and out[0]["kind"] == "instrumental" and out[0]["t1"] == 20.5


def test_suno_tags_from_export_tokens():
    data = {"aligned_words": [
        {"word": "[Intro - ", "start_s": 22.739}, {"word": "solo piano]\n\n\n", "start_s": 22.80},
        {"word": "[Verse 1 - emotive]\nI ", "start_s": 22.862}, {"word": "was ", "start_s": 23.08},
        {"word": "line\n\n[", "start_s": 87.15}, {"word": "Drop - instrumental]\n\n\n", "start_s": 120.83},
        {"word": "[Breakdown - piano]\nEvery ", "start_s": 120.87}, {"word": "again\n\n[", "start_s": 197.9},
        {"word": "End]\n\n\n", "start_s": 198.6},
    ]}
    tags = suno_tags(data)
    # empty sections stacked before a sung word are dropped; End is not a section
    assert [(t["tag"], round(t["t"], 2)) for t in tags] == [("Verse 1 - emotive", 22.86), ("Breakdown - piano", 120.87)]


def test_find_gaps_uses_vocal_energy():
    hop = align.ENV_HOP
    n = int(10.0 / hop)
    db = np.full(n, -80.0)
    db[int(1.0 / hop):int(2.0 / hop)] = -20.0   # word A
    db[int(2.0 / hop):int(2.6 / hop)] = -30.0   # reverb tail / held sound after A's CTC end
    db[int(4.0 / hop):int(5.0 / hop)] = -20.0   # word B
    db[int(5.3 / hop):int(5.6 / hop)] = -25.0   # an ad-lib between B and C that is not in the lyrics
    db[int(6.5 / hop):int(7.0 / hop)] = -20.0   # word C
    words = [{"t0": 1.0, "t1": 1.6}, {"t0": 4.0, "t1": 5.0}, {"t0": 6.5, "t1": 7.0}]
    gaps = align.find_gaps(words, db, thr=-50.0, duration=10.0)
    spans = [(round(g["t0"], 2), round(g["t1"], 2)) for g in gaps]
    assert spans == [(0.0, 1.0), (2.6, 4.0), (5.0, 5.3), (5.6, 6.5), (7.0, 10.0)]
    assert all(g["dur"] >= align.GAP_MIN for g in gaps)
