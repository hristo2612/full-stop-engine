"""Windows lighting up along his path (aimv/paint/windows.py): the window masks and labels from a cell and its lit
twin, each window's time from his path, the blend, and the `windows` prop."""

from types import SimpleNamespace

import numpy as np

from aimv.paint import windows as WN
from test_paint import _with, world  # noqa: F401  (test_paint's module fixture)


def _cell():
    dark = np.full((120, 200, 3), 60, np.uint8)
    lit = dark.copy()
    lit[20:50, 20:50] = 230  # window 1
    lit[20:50, 140:170] = 230  # window 2
    lit[80:84, 60:64] = 230  # a fleck (re-drawn rain): too small to be a window
    return dark, lit


def test_masks_and_labels_find_each_window_and_skip_flecks():
    dark, lit = _cell()
    mask, labels = WN._mask_and_labels(dark, lit, core=0.06, soft=0.02, spill=6)
    assert mask[35, 35] > 0.9 and mask[35, 155] > 0.9 and mask[100, 100] < 0.05
    assert mask[82, 62] < 0.5  # the fleck is not a window
    a, b = labels[35, 35], labels[35, 155]
    assert a > 0 and b > 0 and a != b


class _Lib:
    def m(self, name):
        return {"kind": "pose", "scale": 1.0, "cx": 5.0, "bbox": [0, 0, 10, 100]}

    def anchor(self, name, kind):
        return 5.0, 100.0


def _windows(track):
    """A Windows over one fake cell (two windows, at x 100 and 400) and a choreography whose poses walk `track`."""
    evs = [SimpleNamespace(q=q, layer="wall", pose="P", x=x, y=0.0, scale=1.0, anchor="foot", move=0)
           for q, x in track]
    ch = SimpleNamespace(events=evs, lib=_Lib(), moves=[SimpleNamespace(prop=False)])
    w = WN.Windows.__new__(WN.Windows)
    w.ch = ch
    w.cfg = {"q0": 0.0, "q1": 100.0, "origin": 0.0, "every": 4.0, "lead": 0.0, "spread": 1e9, "fade": 0.0,
             "cells": None}
    w.info = {"cells": {"C": {"box": [0, -50, 500, 50], "windows": [
        {"label": 1, "cx": 100.0, "cy": 0.0, "box": [90, -10, 110, 10]},
        {"label": 2, "cx": 400.0, "cy": 0.0, "box": [390, -10, 410, 10]}]}}}
    img = np.ones((100, 500, 3), np.float32)
    m = np.zeros((100, 500), np.float32)
    lab = np.zeros((100, 500), np.int32)
    m[40:60, 90:110], lab[40:60, 90:110] = 1, 1
    m[40:60, 390:410], lab[40:60, 390:410] = 1, 2
    w._cache = {"C": (img, m, lab)}
    w._times = None
    return w


def test_each_window_lights_on_the_beat_after_he_passes_it():
    w = _windows([(0, 0.0), (5, 150.0), (9, 300.0), (13, 450.0)])
    t = w.times()["C"]
    assert t[1] == 8.0 and t[2] == 16.0  # passed at 5 and 13: the next beats (every 4 16ths)
    lin = np.zeros((100, 500, 3), np.float32)
    E = np.ones((100, 500), np.float32)
    out, E2 = w.blend(lin.copy(), E.copy(), 0, -50, 10.0)  # at 10: window 1 on, window 2 still dark
    assert out[50, 100].min() > 0.99 and out[50, 400].max() < 0.01 and E2[50, 100] > 1.0
    out, _ = w.blend(lin.copy(), E.copy(), 0, -50, 16.0)
    assert out[50, 400].min() > 0.99


def test_a_window_he_never_passes_stays_dark():
    w = _windows([(0, 0.0), (5, 150.0)])
    t = w.times()["C"]
    assert t[1] == 8.0 and np.isinf(t[2])


def test_the_windows_prop_needs_lit_cells(world):  # noqa: F811
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}]}
    ch = _with(world, "vwin", dict(base, moves=[{"windows": {"from": 4.1, "to": 5.1}},
                                                {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]))
    assert any("no `lit` cells" in i.text for i in ch.errors)


def test_flicker_moves_only_the_warm_lights_a_little_each_photo_anchored_to_the_street():
    ch = SimpleNamespace(seed=3, flickers=[{"q0": 10.0, "q1": 20.0, "amount": 0.1, "cell": 10, "thr": 0.2}])
    base = np.full((80, 120, 3), 0.02, np.float32)  # dark wall
    base[10:30, 10:40] = (0.9, 0.5, 0.2)  # a warm lit window
    base[50:70, 60:90] = (0.3, 0.4, 0.9)  # a bright blue sign: not a warm light
    a = WN.flicker(ch, base.copy(), 1000, 500, 12.0, 7)
    b = WN.flicker(ch, base.copy(), 1000, 500, 12.0, 8)
    assert np.array_equal(a[60:, :50], base[60:, :50]) and np.array_equal(a[50:70, 60:90], base[50:70, 60:90])
    ra, rb = a[10:30, 10:40] / base[10:30, 10:40], b[10:30, 10:40] / base[10:30, 10:40]
    assert np.abs(ra - 1).max() <= 0.1 + 1e-6 and np.abs(ra - rb).max() > 1e-3  # faint, and new each photo
    assert np.array_equal(WN.flicker(ch, base.copy(), 1000, 500, 25.0, 7), base)  # outside its stretch: nothing
    # anchored to the street: the same light, seen from a crop 7 px to the right, flickers the same
    c = WN.flicker(ch, base[:, 7:].copy(), 1007, 500, 12.0, 7)
    assert np.allclose(c[10:30, 3:33], a[10:30, 10:40], atol=1e-6)


def test_the_flicker_prop_checks_its_args(world):  # noqa: F811
    for name, args, why in (("vfl1", {"from": 4.3}, "give from and to"), ("vfl2", {"from": 4.3, "to": 4.4, "amount": 2},
                                                                          "amount between")):
        ch = _with(world, name, {"street": "main", "seed": 3, "motion": "m0",
                                 "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
                                 "moves": [{"flicker": args}, {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]})
        assert any(why in i.text for i in ch.errors), (name, [str(i) for i in ch.issues])
    ch = _with(world, "vfl3", {"street": "main", "seed": 3, "motion": "m0",
                               "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
                               "moves": [{"flicker": {"from": 4.3, "to": 5.1, "amount": 0.08}},
                                         {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]})
    assert not ch.errors and ch.flickers[0]["amount"] == 0.08 and ch.flickers[0]["cell"] == 10
