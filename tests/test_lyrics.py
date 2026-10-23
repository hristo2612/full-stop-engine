"""Lyrics parsing, normalisation, word matching and the Suno timing comparison (align step helpers)."""

import numpy as np
import pytest

from aimv.breakdown import align
from aimv.breakdown.lyrics import kind_for, normalize, number_to_words, parse

EV = """[Intro - spoken, deadpan female voice, room tone]
Ladies. Gentlemen. Agents.

[Verse 1 - spoken, deadpan]
Look one. A G I era. Not unreasonable.
A G I. C S I: Miami.

[Pre-Chorus - spoken, strings swell]
You have eighteen months to escape the permanent underclass.

[Chorus - sung, female vocal, trance lift]
Lock in, baby, foot down on the gas
(It's so over?) WE'RE SO BACK!

[Pre-Chorus - spoken, strings higher]
Lock in.

[Final Chorus - sung, key change up, euphoric]
Zero months, there is no underclass

[End]
This line comes after the end tag.
"""


def test_sections_labels_kinds_tags():
    L = parse(EV)
    got = [(s.label, s.kind, s.delivery) for s in L.sections]
    assert got == [
        ("Intro", "intro", "spoken"),
        ("Verse 1", "verse", "spoken"),
        ("Pre-Chorus", "prechorus", "spoken"),
        ("Chorus", "chorus", "sung"),
        ("Pre-Chorus 2", "prechorus", "spoken"),  # repeats without a number get one
        ("Final Chorus", "chorus", "sung"),
    ]
    assert L.sections[1].tag == "Verse 1 - spoken, deadpan"
    assert [len(s.lines) for s in L.sections] == [1, 2, 1, 2, 1, 1]


def test_end_tag_stops_parsing():
    L = parse(EV)
    assert all("after the end" not in ln.text for ln in L.lines)


def test_words_keep_text_and_get_norm():
    L = parse(EV)
    w = L.words
    assert [x.text for x in w[:3]] == ["Ladies.", "Gentlemen.", "Agents."]
    assert [x.norm for x in w[:3]] == ["ladies", "gentlemen", "agents"]
    assert all(x.line == 0 and x.section == 0 for x in w[:3])
    # line and word indices are consistent
    for ln in L.lines:
        assert all(L.words[k].line == ln.i for k in ln.words)


def test_backing_vocals_in_parentheses():
    L = parse(EV)
    ln = next(x for x in L.lines if x.text.startswith("(It's"))
    ws = L.line_words(ln.i)
    assert [(x.text, x.backing) for x in ws] == [
        ("It's", True), ("so", True), ("over?", True), ("WE'RE", False), ("SO", False), ("BACK!", False)]
    assert ws[0].norm == "it's" and ws[3].norm == "we're"
    assert ln.text == "(It's so over?) WE'RE SO BACK!"  # line text as written


def test_acronym_letters_are_spoken_names_for_mms():
    L = parse(EV)
    ln = L.line_words(1)  # Look one. A G I era. Not unreasonable.
    assert [(x.norm, x.say) for x in ln[2:5]] == [("a", "ay"), ("g", "jee"), ("i", "eye")]
    ln2 = L.line_words(2)  # A G I. C S I: Miami.
    assert [x.say for x in ln2[3:6]] == ["see", "es", "eye"]
    assert ln2[-1].say == "miami"


def test_single_letter_words_stay_words():
    L = parse("[Verse]\nI was there and a bird saw me\nNot A I. S I.\n")
    ws = L.words
    assert ws[0].norm == "i" and ws[0].say == "i"
    assert next(x for x in ws if x.text == "a").say == "a"
    assert [x.say for x in L.line_words(1)] == ["not", "ay", "eye", "es", "eye"]


def test_lines_before_any_tag_and_inline_tags():
    L = parse("Hello there\n[Chorus]\nSing it [whispered] loud\n")
    assert L.sections[0].label == "Lyrics" and L.sections[0].kind == "other"
    assert [x.norm for x in L.line_words(1)] == ["sing", "it", "loud"]


def test_empty_sections_are_kept():
    L = parse("[Intro - piano]\n\n[Verse 1]\nOne line\n\n[Drop - instrumental]\n\n[Outro]\nBye\n")
    assert [(s.label, len(s.lines)) for s in L.sections] == [("Intro", 0), ("Verse 1", 1), ("Drop", 0), ("Outro", 1)]


@pytest.mark.parametrize("label,kind", [
    ("Intro", "intro"), ("Verse 2", "verse"), ("Pre-Chorus", "prechorus"), ("Pre Chorus", "prechorus"),
    ("Prechorus", "prechorus"), ("Chorus", "chorus"), ("Final Chorus", "chorus"), ("Hook", "chorus"),
    ("Post-Chorus", "postchorus"), ("Bridge", "bridge"), ("Break", "break"), ("Breakdown", "break"),
    ("Interlude", "break"), ("Drop", "drop"), ("Drop 2", "drop"), ("Build", "build"), ("Build-up", "build"),
    ("Outro", "outro"), ("Instrumental", "instrumental"), ("Guitar Solo", "instrumental"), ("Spoken Word", "verse"),
    ("Applause", "other"),
])
def test_kind_for(label, kind):
    assert kind_for(label) == kind


@pytest.mark.parametrize("raw,norm", [
    ("Eighteen", "eighteen"), ("unreasonable.", "unreasonable"), ("Jensen’s", "jensen's"),
    ('"We\'ve', "we've"), ("Zero-day.", "zero day"), ("11.2", "eleven point two"), ("3rd", "third"),
    ("1,000,000", "one million"), ("2026", "two thousand twenty six"), ("25%", "twenty five percent"),
    ("café", "cafe"), ("...", ""), ("rock'n'roll", "rock'n'roll"), ("'til", "til"),
])
def test_normalize(raw, norm):
    assert normalize(raw) == norm


def test_number_to_words():
    assert number_to_words(0) == "zero"
    assert number_to_words(17) == "seventeen"
    assert number_to_words(42) == "forty two"
    assert number_to_words(100) == "one hundred"
    assert number_to_words(12_900_000_000) == "twelve billion nine hundred million"


def test_match_tokens_merges_and_repeats():
    lyr_words = ["feel", "the", "a", "g", "i", "escape", "velocity", "it's", "so", "over", "we're", "so", "back",
                 "look", "seven"]
    heard = ["feel", "the", "agi", "escape", "velocity", "it's", "so", "over", "we're", "so", "back",
             "it's", "so", "over", "we're", "so", "back", "look", "seven"]
    m = align.match_tokens(lyr_words, heard)
    pairs = {tuple(a): (tuple(b), round(s, 2)) for a, b, s in m}
    assert pairs[(2, 3, 4)] == ((2,), 1.0)  # "A G I" = "AGI"
    # the earlier repeat is matched, the extra one stays unmatched
    assert pairs[(7,)][0] == (5,)
    assert pairs[(13,)][0] == (17,)
    used = {j for _, b, _ in m for j in b}
    assert used.isdisjoint({11, 12, 13, 14, 15, 16})


def test_match_tokens_leaves_unsung_line_unmatched():
    lyr_words = ["lock", "in", "purple", "monkeys", "dance", "six", "months"]
    heard = ["lock", "in", "six", "months"]
    m = align.match_tokens(lyr_words, heard)
    matched = {i for a, _, s in m if s >= 0.8 for i in a}
    assert matched == {0, 1, 5, 6}


# ---------------------------------------------------------------- Suno timings (synthetic export)

LYR = """[Intro - piano]

[Verse 1 - spoken]
Look one. A G I era.
(It's so over?) We're so back

[Drop - instrumental]

[Chorus - sung]
Eighteen months to go
[End]
"""


def _suno_fixture(our_t0, offset=0.2, off_word=None):
    """aligned_lyrics.json shaped like a real export: raw tokens with glued tags and newlines."""
    raw = ["[Intro - ", "piano]\n\n", "[Verse 1 - ", "spoken]\nLook ", "one. ", "A ", "G ", "I ", "era.\n",
           "(It's ", "so ", "over?) ", "We're ", "so ", "back\n\n[", "Drop - ", "instrumental]\n\n",
           "[Chorus - ", "sung]\nEighteen ", "months ", "to ", "go\n[", "End]\n"]
    word_of = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 18: 12, 19: 13,
               20: 14, 21: 15}
    out = []
    t_last = 0.0
    for k, tok in enumerate(raw):
        if k in word_of:
            i = word_of[k]
            t = our_t0[i] + offset + (0.6 if i == off_word else 0.01 * ((i % 3) - 1))
            end = t + 0.3 if "[" not in tok else t + 20.0  # Suno's end spans the instrumental after a line
        else:
            t = t_last + 0.001  # empty-section tags are squeezed into an instant
            end = t + 0.001
        t_last = t
        out.append({"word": tok, "success": True, "start_s": t, "end_s": end, "p_align": 0.99})
    lines = [{"text": "Look one. A G I era.", "start_s": out[3]["start_s"], "end_s": out[8]["end_s"],
              "section": "Intro - piano", "words": [{"text": "Look", "start_s": out[3]["start_s"], "end_s": 1}]}]
    return {"aligned_words": out, "aligned_lyrics": lines, "hoot_cer": 0.1, "is_streamed": False,
            "waveform_data": []}


def _our_words(L, t0s, confs=None):
    return [{"i": w.i, "text": w.text, "norm": w.norm, "line": w.line, "t0": t, "t1": t + 0.3,
             "conf": (confs or {}).get(w.i, 0.9), "flag": None, "suno_t0": None}
            for w, t in zip(L.words, t0s)]


def test_suno_words_strip_tags():
    L = parse(LYR)
    t0s = [10.0 + 0.5 * k for k in range(len(L.words))]
    sw = align.suno_words(_suno_fixture(t0s))
    assert [s["norm"] for s in sw] == [w.norm for w in L.words]
    assert "piano" not in [s["norm"] for s in sw] and "drop" not in [s["norm"] for s in sw]


def test_compare_suno_offset_agreement_and_flags():
    L = parse(LYR)
    t0s = [10.0 + 0.5 * k for k in range(len(L.words))]
    words = _our_words(L, t0s, confs={13: 0.05})
    words[13]["flag"] = "low_conf"
    res = align.compare_suno(words, L, _suno_fixture(t0s, offset=0.2, off_word=7))
    assert res is not None
    assert res["offset_ms"] == pytest.approx(200, abs=15)
    assert res["median_ms"] < 20
    assert res["matched"] == len(L.words)
    assert words[7]["flag"] == "disagree"  # 600 ms off
    assert sum(w["flag"] == "disagree" for w in words) == 1
    assert words[13]["flag"] is None  # low_conf cleared: Suno agrees within 100 ms
    # suno_t0 is on our clock (offset removed)
    assert words[0]["suno_t0"] == pytest.approx(t0s[0], abs=0.02)


def test_compare_suno_falls_back_to_aligned_lyrics_lines():
    L = parse("[Verse]\nHold the gravity line\n")
    t0s = [5.0, 5.4, 5.8, 6.3]
    data = {"aligned_lyrics": [
        {"text": "[Verse]", "start_s": 5.1, "end_s": 5.1, "section": "Verse", "words": [{"text": "[Verse]"}]},
        {"text": "Hold the gravity line", "start_s": 5.1, "end_s": 7, "section": "Verse",
         "words": [{"text": "Hold", "start_s": 5.1, "end_s": 5.4}, {"text": " the", "start_s": 5.5, "end_s": 5.8},
                   {"text": " gravity", "start_s": 5.9, "end_s": 6.3}, {"text": " line", "start_s": 6.4, "end_s": 7}]},
    ]}
    words = _our_words(L, t0s)
    res = align.compare_suno(words, L, data)
    assert res["matched"] == 4 and res["offset_ms"] == pytest.approx(100, abs=1)


def test_compare_suno_without_words_returns_none():
    L = parse(LYR)
    words = _our_words(L, [1.0] * len(L.words))
    assert align.compare_suno(words, L, {"aligned_words": [], "aligned_lyrics": []}) is None


# ---------------------------------------------------------------- Suno guidance, breaths, unsung lines

def test_compare_suno_skips_guided_words():
    L = parse(LYR)
    t0s = [10.0 + 0.5 * k for k in range(len(L.words))]
    words = _our_words(L, t0s)
    words[7]["t0"] += 0.6  # pretend word 7 was placed by Suno's guidance elsewhere: no flag, no count
    res = align.compare_suno(words, L, _suno_fixture(t0s, offset=0.2, off_word=7), guided={7})
    assert res["matched"] == len(L.words) - 1 and res["guided"] == 1
    assert words[7]["flag"] is None and words[7]["suno_t0"] is not None


def test_suno_guides_only_unsure_words_far_from_suno():
    L = parse(LYR)
    t0s = [10.0 + 0.5 * k for k in range(len(L.words))]
    raw = {w.i: t + 0.2 for w, t in zip(L.words, t0s)}
    raw[3] += 1.0   # Suno says word 3 is a second later than us
    raw[5] += 1.0   # and word 5 too, but our CTC is sure about word 5
    raw[9] += 0.15  # a small disagreement is left alone
    words = _our_words(L, t0s, confs={3: 0.1, 5: 0.95, 9: 0.05})
    g = align._suno_guides(L, words, raw, [True] * len(L.lines), T=10_000, have={})
    assert set(g) == {3}
    s, nxt = g[3]
    assert s == pytest.approx(t0s[3] + 1.0, abs=0.01)
    assert nxt == pytest.approx(s + 0.2, abs=0.01)  # the next word starts before it: at least 200 ms span


def test_viterbi_word_window_moves_a_word():
    # Two one-letter words "a" "b"; the emissions show "a" twice (frames 10 and 60) and "b" at frame 80.
    L = parse("[Verse]\na b\n")
    lab = {"a": 1, "b": 2}
    T = 100
    E = np.full((T, 3), -20.0)
    E[:, 0] = -0.01  # blank everywhere
    E[10, :] = [-5.0, -0.1, -20.0]
    E[60, :] = [-5.0, -1.0, -20.0]  # a weaker "a" later
    E[80, :] = [-5.0, -20.0, -0.1]
    gcol = np.maximum(E[:, 0], -3.0)
    free = align.viterbi_align(E, gcol, L, lab, "en", [None], [False])
    assert free[0]["f"][0] == 10
    guided = align.viterbi_align(E, gcol, L, lab, "en", [None], [False], word_windows={0: (55, 65, 55, 75)})
    assert guided[0]["f"][0] == 60 and guided[1]["f"][0] == 80
    clash = {0: (55, 65, 55, 75), 1: (5, 8, 5, 8)}  # "b" forced before "a": no path
    assert align.viterbi_align(E, gcol, L, lab, "en", [None], [False], word_windows=clash) is None


def test_dip_onset_steps_over_a_breath():
    hop = align.ENV_HOP
    n = int(2.0 / hop)
    db = np.full(n, -25.0)
    vo = np.full(n, 10.0)
    def span(a, b, level, voicing):
        db[int(a / hop):int(b / hop)] = level
        vo[int(a / hop):int(b / hop)] = voicing
    span(0.40, 0.60, -60.0, 0.0)   # silence after the previous word
    span(0.60, 0.85, -32.0, -15.0)  # a breath: noise, unvoiced
    span(0.85, 0.92, -48.0, 0.0)   # short dip
    span(0.92, 1.30, -24.0, 12.0)  # the word, voiced
    without = align._dip_onset(db, 0.30, 1.0)
    with_vo = align._dip_onset(db, 0.30, 1.0, vo=vo)
    assert without == pytest.approx(0.60, abs=0.02)   # deepest dip: the breath's start
    assert with_vo == pytest.approx(0.92, abs=0.02)   # the word after the breath
    vo[int(0.60 / hop):int(0.85 / hop)] = 8.0          # the same shape, voiced: a syllable, keep the first dip
    assert align._dip_onset(db, 0.30, 1.0, vo=vo) == pytest.approx(0.60, abs=0.02)


def test_voicing_separates_tone_from_noise():
    sr = align.SR
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, sr)
    b = np.array([1.0, -1.0])  # first difference: tilt the noise towards high frequencies, like a breath
    noise = np.convolve(noise, b, mode="same")
    v = align.voicing(np.concatenate([tone, noise]).astype(np.float32))
    half = len(v) // 2
    assert np.median(v[:half - 20]) > 10 and np.median(v[half + 20:]) < 0


def test_line_with_nothing_heard_and_nothing_found_is_not_sung():
    L = parse("FULL STOP\n[Verse 1]\nEverybody has a plan\n")
    words = [{"i": w.i, "text": w.text, "norm": w.norm, "t0": 0.5 + 0.4 * w.i, "t1": 0.8 + 0.4 * w.i,
              "conf": 0.02 if w.line == 0 else 0.9, "flag": None} for w in L.words]
    db = np.full(int(5 / align.ENV_HOP), -20.0)  # vocal sound everywhere (a pad), so the sound test alone fails
    support = [{"n": 2, "heard": 0, "sound": 2.0, "need": 0.5}, {"n": 4, "heard": 4, "sound": 2.0, "need": 1.0}]
    sung = align._sung_lines(L, words, [0.0, 1.0], True, db, -50.0, support)
    assert sung == [False, True]
    assert words[0]["t0"] is None and words[2]["t0"] is not None
