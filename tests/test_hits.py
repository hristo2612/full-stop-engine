import json

import numpy as np
import pytest
import scipy.signal as sps

from aimv.breakdown import hits as H

SR = 48000
BEAT = 0.5  # 120 BPM


def _noise(n, lo, hi, rng):
    sos = sps.butter(4, [lo, min(hi, 0.45 * SR)], "bandpass", fs=SR, output="sos")
    return sps.sosfilt(sos, rng.standard_normal(n))


def _kick(rng):
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 110 * np.exp(-t / 0.03)  # pitch drop
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)
    click = rng.standard_normal(t.size) * np.exp(-t / 0.002) * 0.2
    return 0.8 * body + click


def _clap(rng):
    t = np.arange(int(0.3 * SR)) / SR
    return 0.5 * _noise(t.size, 1000, 8000, rng) * np.exp(-t / 0.06)


def _hat(rng):
    t = np.arange(int(0.08 * SR)) / SR
    return 0.3 * _noise(t.size, 7000, 16000, rng) * np.exp(-t / 0.012)


def _add(x, sig, t):
    i = int(round(t * SR))
    n = min(sig.size, x.size - i)
    x[i:i + n] += sig[:n]


def _drum_song(rng, bars=8, quiet_bars=(5, 6)):
    """Four-on-the-floor kick, clap on 2 and 4, hats on the off-beat 8ths. `quiet_bars` have no drums,
    only a -40 dB bass line leaking into the stem (must not become kicks)."""
    dur = bars * 4 * BEAT + 1.0
    x = rng.standard_normal(int(dur * SR)) * 1e-4  # noise floor (-80 dB)
    kicks, claps, hats = [], [], []
    for bar in range(bars):
        t0 = 0.5 + bar * 4 * BEAT
        if bar in quiet_bars:
            for k in range(8):
                tt = np.arange(int(0.24 * SR)) / SR
                _add(x, 0.01 * np.sin(2 * np.pi * (55 + 10 * (k % 3)) * tt) * np.hanning(tt.size), t0 + k * BEAT / 2)
            continue
        for b in range(4):
            tb = t0 + b * BEAT
            _add(x, _kick(rng), tb)
            kicks.append(tb)
            if b in (1, 3):
                _add(x, _clap(rng), tb)
                claps.append(tb)
            _add(x, _hat(rng), tb + BEAT / 2)
            hats.append(tb + BEAT / 2)
    return x, np.array(kicks), np.array(claps), np.array(hats)


def _match(found, truth, tol):
    """Number of truth events with a found event within tol, and number of found events matching nothing."""
    hit = sum(bool(np.any(np.abs(found - t) <= tol)) for t in truth)
    extra = sum(not np.any(np.abs(truth - f) <= tol) for f in found)
    return hit, extra


@pytest.fixture(scope="module")
def drums():
    rng = np.random.default_rng(1)
    x, k, c, h = _drum_song(rng)
    return x, k, c, h, H.drum_hits(x, SR)


def test_kicks_on_every_beat_and_none_from_bass_bleed(drums):
    _, kicks, _, _, ev = drums
    t, s = ev["kick"]
    hit, extra = _match(t, kicks, 0.01)
    assert hit == len(kicks)
    assert extra == 0  # the quiet bars' bass line is bleed, not kicks
    assert np.all((s > 0) & (s <= 1))


def test_clap_on_a_kick_is_a_snare_but_the_kick_click_is_not(drums):
    _, kicks, claps, _, ev = drums
    t, _ = ev["snare"]
    hit, extra = _match(t, claps, 0.015)
    assert hit == len(claps)
    assert extra == 0  # kicks on 1 and 3 excite the snare band but leave no tail


def test_hats_off_the_beat_only(drums):
    _, kicks, _, hats, ev = drums
    t, _ = ev["hat"]
    hit, extra = _match(t, hats, 0.01)
    assert hit >= 0.9 * len(hats)
    assert extra == 0
    assert np.min(np.abs(t[:, None] - kicks[None, :])) > 0.04


def test_vocal_entries_after_a_bar_of_silence_only():
    rng = np.random.default_rng(2)
    dur = 14.0
    x = rng.standard_normal(int(dur * SR)) * 1e-4
    tt = np.arange(int(dur * SR)) / SR
    voice = sum(np.sin(2 * np.pi * 220 * k * tt) / k for k in range(1, 12)) * 0.1
    gate = np.zeros_like(tt)
    for a, b in ((1.0, 3.0), (3.5, 5.0), (8.0, 10.0), (10.4, 12.0)):  # gaps: 0.5 s, 3 s, 0.4 s
        gate[(tt >= a) & (tt < b)] = 1.0
    gate = np.convolve(gate, np.ones(240) / 240, mode="same")  # 5 ms fades
    x += voice * gate
    x += voice * 0.003 * ((tt >= 5.6) & (tt < 6.4))  # bleed 50 dB down inside the long gap
    t, s = H.vocal_entries(x, SR, lambda _t: 2.0)
    assert len(t) == 2
    assert abs(t[0] - 1.0) < 0.03 and abs(t[1] - 8.0) < 0.03
    assert np.all((s > 0) & (s <= 1))


def _riser_song(rng):
    """Groove for 4 bars, a 4-bar noise riser with the drums out, then a loud drop on bar 8."""
    dur = 12 * 4 * BEAT
    n = int(dur * SR)
    x = rng.standard_normal(n) * 1e-4
    tt = np.arange(n) / SR
    beats = np.arange(0, dur, BEAT)
    for tb in beats:
        if tb < 8.0 or tb >= 16.0:
            _add(x, _kick(rng) * (1.0 if tb >= 16.0 else 0.6), tb)
    bass = 0.2 * np.sin(2 * np.pi * 55 * tt)
    x += bass * ((tt < 8.0) | (tt >= 16.0)) * np.where(tt >= 16.0, 1.5, 1.0)
    # riser: noise whose level climbs 40 dB and whose low cut opens upward over bars 4..7
    seg = (tt >= 8.0) & (tt < 16.0)
    r = _noise(n, 300, 16000, rng)
    prog = np.clip((tt - 8.0) / 8.0, 0, 1)
    x += r * seg * 0.3 * 10 ** (-2 * (1 - prog))
    # drop: crash noise and everything louder
    crash = _noise(n, 2000, 16000, rng) * np.exp(-np.clip(tt - 16.0, 0, None) / 0.8) * (tt >= 16.0) * 0.3
    x += crash
    down = np.zeros(beats.size, bool)
    down[::4] = True
    return x, beats, down


def test_noise_riser_into_a_drop_gives_one_impact_and_one_riser():
    rng = np.random.default_rng(3)
    x, beats, down = _riser_song(rng)
    kicks = H.drum_hits(x, SR)["kick"][0]
    imp_t, imp_s = H.impacts(x, SR, beats, down, kicks)
    assert len(imp_t) == 1
    assert abs(imp_t[0] - 16.0) < 0.02
    assert imp_s[0] == pytest.approx(1.0)
    rs = H.risers(x, SR, beats, down, imp_t)
    assert len(rs) == 1
    assert rs[0]["t1"] == pytest.approx(16.0, abs=1e-6)
    assert 8.0 - 1e-6 <= rs[0]["t0"] <= 12.0
    assert rs[0]["impact"] is True


def test_steady_groove_has_no_impacts_or_risers():
    rng = np.random.default_rng(4)
    dur = 12.0
    x = rng.standard_normal(int(dur * SR)) * 1e-4
    beats = np.arange(0, dur, BEAT)
    for tb in beats:
        _add(x, _kick(rng), tb)
    down = np.zeros(beats.size, bool)
    down[::4] = True
    assert len(H.impacts(x, SR, beats, down, beats)[0]) == 0
    assert H.risers(x, SR, beats, down, np.array([])) == []


def test_run_writes_the_contract(tmp_path):
    from aimv.project import Project
    from aimv.util import save_audio, write_json

    rng = np.random.default_rng(5)
    drums, *_ = _drum_song(rng, bars=6, quiet_bars=())
    n = drums.size
    tt = np.arange(n) / SR
    vox = sum(np.sin(2 * np.pi * 220 * k * tt) / k for k in range(1, 8)) * 0.05 * ((tt > 4.5) & (tt < 7))
    root = tmp_path / "song"
    (root / "breakdown/stems").mkdir(parents=True)
    (root / "breakdown/audio").mkdir(parents=True)
    save_audio(root / "breakdown/stems/drums.wav", np.stack([drums, drums]), SR)
    save_audio(root / "breakdown/stems/vocals.wav", np.stack([vox, vox]), SR)
    save_audio(root / "breakdown/stems/instrumental.wav", np.stack([drums, drums]), SR)
    save_audio(root / "breakdown/audio/mix.wav", np.stack([drums + vox, drums + vox]), SR)
    beats = 0.5 + np.arange(0, 6 * 4) * BEAT
    write_json(root / "breakdown/beats.json", {
        "version": 1, "meter": 4,
        "beats": [{"i": i, "t": float(b), "bar": i // 4, "beat": i % 4, "bpm": 120.0} for i, b in enumerate(beats)],
        "bars": [{"i": k, "t0": float(beats[4 * k]), "t1": float(beats[4 * k] + 2.0), "beats": 4} for k in range(6)],
    })
    p = Project.open(root)
    H.run(p, lambda m: None)
    out = json.loads((root / "breakdown/hits.json").read_text())
    assert out["version"] == 1
    ev = out["events"]
    assert set(ev) == {"kick", "snare", "hat", "vocal_entry", "impact", "riser", "stop", "roll"}
    assert out["counts"] == {k: len(v) for k, v in ev.items()}
    assert out["counts"]["kick"] == 24 and out["counts"]["snare"] == 12
    for k, v in ev.items():
        for e in v:
            assert 0 < e["s"] <= 1
            for key in ("t", "t0", "t1"):
                if key in e:
                    assert e[key] == round(e[key], 3)
    assert [e["t"] for e in ev["kick"]] == sorted(e["t"] for e in ev["kick"])
    assert len(ev["vocal_entry"]) == 1 and abs(ev["vocal_entry"][0]["t"] - 4.5) < 0.03


def test_stems_of_bleed_only_give_no_drums_and_no_vocal_entries():
    """A song without drums or vocals still gets drum and vocal stems from the separator: faint bleed."""
    rng = np.random.default_rng(6)
    x, *_ = _drum_song(rng, bars=4, quiet_bars=())
    bleed = x * 10 ** (-45 / 20)
    mix_db = -12.0
    ev = H.drum_hits(bleed, SR, mix_db=mix_db)
    assert all(len(ev[k][0]) == 0 for k in ("kick", "snare", "hat"))
    tt = np.arange(int(8 * SR)) / SR
    murmur = np.sin(2 * np.pi * 220 * tt) * 10 ** (-50 / 20) * (tt > 2)
    assert len(H.vocal_entries(murmur, SR, lambda _t: 2.0, mix_db=mix_db)[0]) == 0
    assert len(H.vocal_entries(murmur, SR, lambda _t: 2.0)[0]) == 1  # without the mix it cannot tell


def _break_kick(rng):
    """A breakbeat kick through a producer's high-pass (4th order at 200 Hz): thump and click, no sub."""
    t = np.arange(int(0.2 * SR)) / SR
    f = 120 + 130 * np.exp(-t / 0.01)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.04)
    click = _noise(t.size, 500, 6000, rng) * np.exp(-t / 0.002) * 0.3
    hp = sps.butter(4, 200, "highpass", fs=SR, output="sos")
    return sps.sosfilt(hp, 0.8 * body + click)


def _break_snare(rng):
    """The same break's snare, through the same high-pass: 220 Hz body and a 1-8 kHz rattle."""
    t = np.arange(int(0.25 * SR)) / SR
    body = np.sin(2 * np.pi * 220 * t) * np.exp(-t / 0.03)
    rattle = _noise(t.size, 1000, 8000, rng) * np.exp(-t / 0.07)
    hp = sps.butter(4, 200, "highpass", fs=SR, output="sos")
    return sps.sosfilt(hp, 0.3 * body + 0.3 * rattle)


def test_high_passed_break_gives_kicks_on_the_kicks_and_snares_on_the_snares():
    """Two-step break with its sub filtered away: kicks on 1 and the "and" of 3, snares on 2 and 4."""
    rng = np.random.default_rng(7)
    beat = 60 / 175
    bars = 6
    x = rng.standard_normal(int((bars * 4 * beat + 1) * SR)) * 1e-4
    kicks, snares = [], []
    for bar in range(bars):
        t0 = 0.3 + bar * 4 * beat
        for k in (0.0, 2.5):
            _add(x, _break_kick(rng), t0 + k * beat)
            kicks.append(t0 + k * beat)
        for s in (1.0, 3.0):
            _add(x, _break_snare(rng), t0 + s * beat)
            snares.append(t0 + s * beat)
    ev = H.drum_hits(x, SR)
    kt, ks = ev["kick"]
    hit, extra = _match(kt, np.array(kicks), 0.015)
    assert hit >= len(kicks) - 1 and extra == 0
    assert np.all((ks > 0) & (ks <= 1))
    hit, extra = _match(ev["snare"][0], np.array(snares), 0.015)
    assert hit >= len(snares) - 1 and extra == 0


def test_a_kick_with_a_quiet_breakbeat_top_is_not_also_a_snare():
    """Sub kicks on 1 and the "and" of 3; the second one is layered with a rattle 14 dB under the snares
    on 2 and 4. Only the real snares are snares."""
    rng = np.random.default_rng(8)
    beat = 60 / 175
    bars = 8
    x = rng.standard_normal(int((bars * 4 * beat + 1) * SR)) * 1e-4
    snares = []
    for bar in range(bars):
        t0 = 0.3 + bar * 4 * beat
        _add(x, _kick(rng), t0)
        _add(x, _kick(rng), t0 + 2.5 * beat)
        top = _break_snare(rng) * 0.2
        _add(x, top, t0 + 2.5 * beat)
        for s in (1.0, 3.0):
            _add(x, _break_snare(rng), t0 + s * beat)
            snares.append(t0 + s * beat)
    hit, extra = _match(H.drum_hits(x, SR)["snare"][0], np.array(snares), 0.015)
    assert hit >= len(snares) - 1 and extra == 0


def test_vocal_entries_follow_the_words_not_chops_in_the_gap():
    rng = np.random.default_rng(9)
    dur = 16.0
    tt = np.arange(int(dur * SR)) / SR
    x = rng.standard_normal(tt.size) * 1e-4
    voice = sum(np.sin(2 * np.pi * 220 * k * tt) / k for k in range(1, 12)) * 0.1
    chop = sum(np.sin(2 * np.pi * 330 * k * tt) / k for k in range(1, 8)) * 0.08
    sung = ((tt >= 1.0) & (tt < 4.0)) | ((tt >= 10.0) & (tt < 13.0)) | ((tt >= 14.2) & (tt < 15.5))
    chopped = (tt >= 5.0) & (tt < 9.6) & ((tt * 4) % 1 < 0.6)  # stabs every 250 ms in the gap
    x += voice * sung + chop * chopped
    words = [(1.0, 2.0), (2.0, 4.0), (10.0, 13.0), (14.2, 15.5)]
    t_stem, _ = H.vocal_entries(x, SR, lambda _t: 2.0)
    assert len(t_stem) == 1  # the chops fill the gap: the stem alone hears no silence
    t, s = H.vocal_entries(x, SR, lambda _t: 2.0, words=words)
    assert len(t) == 2 and abs(t[0] - 1.0) < 0.03 and abs(t[1] - 10.0) < 0.03
    t2, _ = H.vocal_entries(x, SR, lambda _t: 2.0, words=words, gap_bars=0.5)
    assert len(t2) == 3 and abs(t2[2] - 14.2) < 0.03  # a 1.2 s pause counts when half a bar is enough


def _groove(rng, dur, beat, silent=(), hats_only=()):
    """Kick on every beat, a bass note and hats on the off-beats; `silent` spans are cut to nothing,
    `hats_only` spans keep only the hats. Returns (audio, drum hit times that sound)."""
    n = int(dur * SR)
    x = rng.standard_normal(n) * 1e-4
    tt = np.arange(n) / SR
    bass = 0.15 * np.sin(2 * np.pi * 55 * tt)
    hits = []
    for tb in np.arange(0.0, dur - 0.5, beat):
        _add(x, _kick(rng), tb)
        _add(x, _hat(rng), tb + beat / 2)
        hits += [tb, tb + beat / 2]
    x += bass
    for a, b in hats_only:
        m = (tt >= a) & (tt < b)
        keep = np.zeros(n)
        for tb in np.arange(0.0, dur, beat):
            if a <= tb + beat / 2 < b:
                _add(keep, _hat(rng), tb + beat / 2)
        x[m] = keep[m] + rng.standard_normal(m.sum()) * 1e-4
    for a, b in silent:
        m = (tt >= a) & (tt < b)
        x[m] = rng.standard_normal(m.sum()) * 1e-4
    hits = np.array(hits)
    gone = np.zeros(hits.size, bool)
    for a, b in silent:
        gone |= (hits >= a) & (hits < b)
    for a, b in hats_only:
        gone |= (hits >= a) & (hits < b) & (np.abs(((hits - a) / beat) % 1 - 0.5) > 0.1)
    return x, hits[~gone]


def test_a_dropout_that_comes_back_is_a_stop_but_holes_and_a_fade_out_are_not():
    rng = np.random.default_rng(10)
    beat = 0.5
    dur = 24.0
    # 2 beats of nothing at 8.0-9.0 s, then the groove slams back on the downbeat at 9.0;
    # 13.0-14.0 keeps the hats running (a hole in the groove); everything stops for good at 20.0
    x, drums = _groove(rng, dur, beat, silent=[(8.0, 9.0), (20.0, dur)], hats_only=[(13.0, 14.0)])
    beats = np.arange(0.0, dur, beat)
    sp = H.stops(x, SR, beats, np.array([9.0]), drums=drums)
    assert len(sp) == 1
    st = sp[0]
    assert abs(st["t0"] - 8.0) < 0.06 and abs(st["t1"] - 9.0) < 0.03
    assert st["end_impact"] == 9.0
    assert 0 < st["s"] <= 1


def test_drum_rolls_into_a_downbeat_but_a_steady_eighth_groove_is_not_a_roll():
    beat = 0.5
    beats = np.arange(0.0, 40.0, beat)
    down = np.zeros(beats.size, bool)
    down[::4] = True
    # quarter kicks, then 8ths for bars 4-5 (8.0-12.0 s) into the downbeat at 12.0, quarters again;
    # bars 12-19 (24-40 s): 8ths all the way (a groove)
    kicks = np.concatenate([np.arange(0.0, 8.0, beat), np.arange(8.0, 12.0, beat / 2), np.arange(12.0, 24.0, beat),
                            np.arange(24.0, 40.0, beat / 2)])
    snares = np.arange(0.5, 40.0, 2 * beat)
    rl = H.rolls(kicks, snares, beats, down)
    assert len(rl) == 1
    r = rl[0]
    assert abs(r["t0"] - 8.0) < 1e-6 and abs(r["t1"] - 12.0) < 1e-6
    assert r["voice"] == "kick" and r["div"] == 8 and 0 < r["s"] <= 1
