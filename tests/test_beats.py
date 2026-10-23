import numpy as np

from aimv.breakdown.beats import _bar_labels, _fit, _phase_check


def _grid(n=200, bpm0=130.0, bpm1=134.0, start=0.3):
    periods = 60.0 / np.linspace(bpm0, bpm1, n)
    return start + np.concatenate([[0], np.cumsum(periods[:-1])]), periods


def test_fit_removes_frame_quantisation_and_keeps_drift():
    true, periods = _grid()
    quantised = np.round(true / 0.02) * 0.02  # beat_this reports beats on a 20 ms frame grid
    fitted, period = _fit(quantised)
    assert np.abs(fitted - true).max() < 0.008
    assert abs(60 / period[10] - 130.2) < 0.6
    assert abs(60 / period[-10] - 133.8) < 0.6


def test_phase_check_moves_grid_from_offbeat_onto_kicks():
    beats, periods = _grid()
    kicks = beats + 0.0
    offbeat_grid = beats + periods / 2  # the BLISS failure: grid locked to the hats
    res = _phase_check(offbeat_grid, periods, kicks)
    shifted = offbeat_grid + res["shifted_ms"] / 1000
    assert res["note"].startswith("grid was on the off-beat")
    assert np.median(np.abs(shifted - beats)) < 0.01


def test_phase_check_corrects_small_constant_offset_only_when_consistent():
    beats, periods = _grid()
    res = _phase_check(beats - 0.02, periods, beats)
    assert abs(res["shifted_ms"] - 20) < 2
    res2 = _phase_check(beats, periods, beats + 0.004)
    assert res2["shifted_ms"] == 0.0


def test_bar_labels_with_pickup():
    bar, beat = _bar_labels(10, np.array([2, 6]), 4)
    assert list(bar) == [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert list(beat) == [2, 3, 0, 1, 2, 3, 0, 1, 2, 3]


def test_repair_fills_half_time_stretch():
    from aimv.breakdown.beats import _repair

    true = np.arange(400) * 0.343
    halftime = np.concatenate([true[:100], true[100:160:2], true[160:]])  # tracker skipped every other beat
    rep, filled, dropped = _repair(halftime)
    assert filled == 30 and dropped == 0
    assert np.abs(rep - true).max() < 1e-6


def test_phase_downbeats_votes_away_short_wobble_keeps_real_change():
    from aimv.breakdown.beats import _phase_downbeats

    tracked = np.array([0, 4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48])  # 18 is a glitch
    assert list(_phase_downbeats(52, tracked, 4)) == list(range(0, 52, 4))
    real = np.array([0, 4, 8, 12, 16, 17, 21, 25, 29, 33, 37, 41, 45])  # phase moves at 17 and stays
    out = list(_phase_downbeats(49, real, 4))
    assert 21 in out and 41 in out and 20 not in out
