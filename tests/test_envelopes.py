import json
import math

import numpy as np
import pytest

from aimv.breakdown import envelopes as E

SR = 48000
FPS = 24


def test_frame_count_is_ceil_of_duration():
    assert E.n_frames(10 * SR, SR, FPS) == 240
    assert E.n_frames(10 * SR + 1, SR, FPS) == 241
    assert E.n_frames(int(306.4 * SR), SR, FPS) == math.ceil(306.4 * 24)


def test_rms_db_and_mapping():
    t = np.arange(SR) / SR
    sine = np.sin(2 * np.pi * 440 * t)
    b = E.frame_bounds(SR, SR, FPS, E.n_frames(SR, SR, FPS))
    d = E.frame_rms_db(np.stack([sine, sine]), b)
    assert np.allclose(d, -3.01, atol=0.05)
    assert E.db_to_unit(np.array([-80.0, -60.0, -30.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_follower_attacks_in_one_frame_and_releases_over_150ms():
    v = np.r_[np.zeros(5), np.ones(5), np.zeros(30)]
    y = E.follower(v, FPS)
    assert y[5] == 1.0
    k = int(round(0.15 * FPS))  # frames after the drop: about one time constant
    assert y[9 + k] == pytest.approx(math.exp(-k / (FPS * 0.15)), abs=1e-9)
    assert y[-1] < 0.01


def test_pulse_peaks_in_the_frame_that_contains_the_hit():
    frames = 100
    y = E.pulses(np.array([1.01]), np.array([0.8]), frames, FPS, tau=0.1)
    k = int(1.01 * FPS)  # frame 24 covers [1.0, 1.0417)
    assert y[k] == pytest.approx(0.8)
    assert np.all(y[:k] == 0)
    assert y[k + 1] == pytest.approx(0.8 * math.exp(-((k + 1.5) / FPS - 1.01) / 0.1))
    assert y[k + 1] < y[k] and y[k + 5] < 0.1


def test_overlapping_pulses_take_the_max_and_stay_in_range():
    y = E.pulses(np.array([0.5, 0.52, 0.6]), np.array([1.0, 0.5, 0.9]), 50, FPS, tau=0.14)
    assert y.max() <= 1.0 and y.min() >= 0.0
    assert y[int(0.5 * FPS)] == 1.0


def test_run_end_to_end(tmp_path):
    from aimv.project import Project
    from aimv.util import save_audio, write_json

    rng = np.random.default_rng(0)
    dur = 4.3
    n = int(dur * SR)
    t = np.arange(n) / SR
    root = tmp_path / "song"
    bd = root / "breakdown"
    (bd / "stems").mkdir(parents=True)
    (bd / "audio").mkdir(parents=True)
    tone = 0.5 * np.sin(2 * np.pi * 110 * t) * (t > 2.0)              # bass enters at 2 s
    hiss = 0.05 * rng.standard_normal(n) * (t > 3.0)                    # bright noise from 3 s
    mix = tone + hiss
    save_audio(bd / "audio/mix.wav", np.stack([mix, mix]), SR)
    for s, x in (("vocals", np.zeros(n)), ("drums", hiss), ("bass", tone), ("other", np.zeros(n))):
        save_audio(bd / f"stems/{s}.wav", np.stack([x, x]), SR)
    write_json(bd / "hits.json", {"version": 1, "events": {
        "kick": [{"t": 1.01, "s": 0.9}], "snare": [{"t": 2.5, "s": 0.5}], "hat": [],
        "vocal_entry": [], "impact": [], "riser": []}, "counts": {}})
    p = Project.open(root)
    E.run(p, lambda m: None)
    text = (bd / "envelopes.json").read_text()
    assert "\n " not in text and '"series":{"mix":[' in text  # compact
    out = json.loads(text)
    frames = math.ceil(dur * FPS)
    assert out["version"] == 1 and out["fps"] == FPS and out["frames"] == frames
    S = out["series"]
    assert list(S) == ["mix", "vocals", "drums", "bass", "other", "kick", "snare", "hat", "centroid"]
    for k, v in S.items():
        assert len(v) == frames
        assert all(0 <= x <= 1 for x in v)
        assert all(x == round(x, 3) for x in v)
    assert S["kick"][24] == 0.9 and S["kick"][23] == 0
    assert S["snare"][60] == 0.5
    assert max(S["vocals"]) == 0 and max(S["hat"]) == 0
    assert S["bass"][47] == 0 and S["bass"][49] == pytest.approx((-3 - 6 + 60) / 60, abs=0.01)
    # brightness: the tone alone is dark, the hiss on top is bright
    assert np.mean(S["centroid"][55:70]) < np.mean(S["centroid"][80:100])
    assert S["centroid"][10] == 0  # silence
