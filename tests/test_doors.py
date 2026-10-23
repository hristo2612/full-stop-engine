"""A door opening for him and shutting behind him (aimv/paint/doors.py, v8b's outro): the angle over time (pushed
open, held, the closer's sweep and latch zone, the snap and the latch's rebound), the leaf's quad, the blend (the
open twin through the doorway, the leaf at its angle, the spill), the light, the occlusion once he is inside, and the
`door` prop."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aimv.paint import doors as DR
from test_paint import _with, world  # noqa: F401  (test_paint's module fixture)

CFG = {"o0": 10.0, "o1": 10.2, "angle": 80.0, "s0": 11.0, "s1": 13.0, "latch": 1.5, "inside": 10.15}


def test_the_angle_opens_holds_closes_and_latches():
    a = lambda t: DR.angle(CFG, t)  # noqa: E731
    assert a(9.9) == 0.0 and a(10.2) == pytest.approx(80.0) and a(10.6) == 80.0
    assert 0 < a(10.05) < a(10.1) < a(10.2) and a(10.1) > 40  # pushed: quick, easing out
    ts = np.linspace(11.0, 13.0, 41)
    vs = [a(t) for t in ts]
    assert all(b <= c + 1e-9 for b, c in zip(vs[1:], vs))  # the closer only closes
    assert a(11.0 + 0.75 * 2) == pytest.approx(12.0) and a(11.0 + 0.96 * 2) == pytest.approx(3.0)
    assert a(13.0) == pytest.approx(0.0, abs=1e-9) and 0 < a(13.06) <= 1.5 and a(13.2) == 0.0  # the latch's rebound
    assert DR.openness(CFG, 9.0) == 0.0 and DR.openness(CFG, 10.5) == pytest.approx(1.0)


def test_the_leaf_narrows_toward_its_hinge_and_its_free_edge_recedes():
    q = DR.leaf_quad([100, 0, 200, 300], 60, "left", 0.12)
    assert q[0].tolist() == [100, 0] and q[3].tolist() == [100, 300]  # the hinge stays
    assert q[1][0] == pytest.approx(100 + 100 * math.cos(math.radians(60)), abs=1e-3)
    assert q[2][1] - q[1][1] < 300  # the free edge shorter
    r = DR.leaf_quad([100, 0, 200, 300], 60, "right", 0.12)
    assert r[0].tolist() == [200, 0] and r[1][0] == pytest.approx(150, abs=1e-3)


def _doors():
    """Doors over one fake door: box [0, 0, 300, 200], the leaf (and the doorway) x 100 to 200, y 20 to 200; the open
    twin bright, the closed world dark grey, the spill outside the leaf."""
    d = DR.Doors.__new__(DR.Doors)
    d.cfg = {"D": dict(CFG)}
    d.spec = {"D": {"id": "D", "leaf": [100, 20, 200, 200], "hinge": "left", "light": "spill", "persp": 0.12}}
    d.info = {"D": {"box": [0, 0, 300, 200]}}
    opn = np.full((200, 300, 3), 0.8, np.float32)
    closed = np.full((200, 300, 3), 0.05, np.float32)
    closed[20:200, 100:200] = 0.2  # the leaf's own paint, lighter than the wall
    spill = np.ones((200, 300), np.float32)
    spill[20:200, 100:200] = 0
    d._cache = {"D": (opn, closed, spill)}
    return d


def test_the_blend_shows_the_doorway_the_leaf_and_the_spill():
    d = _doors()
    lin0 = np.full((200, 300, 3), 0.05, np.float32)
    lin0[20:200, 100:200] = 0.2
    lin, _ = d.blend(lin0.copy(), np.ones((200, 300), np.float32), 0, 0, 9.0)
    assert np.allclose(lin, lin0)  # shut: the world as built
    lin, E = d.blend(lin0.copy(), np.ones((200, 300), np.float32), 0, 0, 10.6)
    assert lin[100, 190, 0] == pytest.approx(0.8)  # the doorway shows the open twin past the leaf
    w = 100 * math.cos(math.radians(80))
    assert lin[100, 101, 0] < 0.3 and lin[100, int(100 + w) + 3, 0] > 0.7  # the leaf narrow at its hinge
    assert lin[100, 50, 0] == pytest.approx(0.8) and E[100, 50] > 1.0  # the spill out of the doorway
    half = d.blend(lin0.copy(), np.ones((200, 300), np.float32), 0, 0, 12.5)[0]
    assert 0.05 < half[100, 50, 0] < 0.8  # closing: the spill fading with the angle
    assert d.lights(9.0) == {"spill": 0.0} and d.lights(10.6)["spill"] == pytest.approx(1.0)


def test_inside_he_is_seen_only_through_the_doorway_less_the_leaf():
    d = _doors()
    A = np.ones((200, 300), np.float32)
    assert np.allclose(d.occlude(A, 0, 0, 10.1), 1.0)  # not inside yet
    B = d.occlude(A, 0, 0, 10.6)
    assert B[100, 50] == 0 and B[10, 150] == 0 and B[100, 190] == pytest.approx(1.0)  # the wall, the lintel; the doorway
    assert B[100, 101] < 0.5  # behind the open leaf at its hinge


def test_the_door_prop_sets_its_times_and_checks_them(world):  # noqa: F811
    import shutil
    src = world["street"]
    dd = src.parent / "doorst"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["doors"] = [{"id": "doorT", "cell": "C", "open": "x.png", "leaf": [800, 300, 900, 600]}]
    (dd / "obstacles.json").write_text(json.dumps(ob))
    base = {"street": "doorst", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}]}
    mv = {"door": {"id": "doorT", "open": {"from": "4.2", "to": "4.2.3", "angle": 75}, "shut": {"from": 5.1,
                                                                                             "to": "5.3"},
                   "inside": "4.2.2"}}
    ch = _with(world, "vdoor", dict(base, moves=[mv]))
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    c = ch.doors["doorT"]
    assert c["o0"] == pytest.approx(g.t(g.resolve("4.2")[0])) and c["angle"] == 75.0
    assert c["inside"] == pytest.approx(g.t(g.resolve("4.2.2")[0]))
    bad = dict(mv["door"], shut={"from": "4.1", "to": "5.3"})
    assert any("door" in str(i) for i in _with(world, "vdoor2", dict(base, moves=[{"door": bad}])).errors)
    assert any("no door" in str(i) for i in _with(world, "vdoor3", dict(
        base, moves=[{"door": dict(mv["door"], id="nope")}])).errors)


def test_once_gone_nothing_of_him_shows_at_the_door():
    d = _doors()
    d.cfg["D"]["gone"] = 11.5
    A = np.ones((200, 300), np.float32)
    assert d.occlude(A, 0, 0, 11.0)[100, 190] == pytest.approx(1.0)
    assert not d.occlude(A, 0, 0, 11.6)[0:200, 0:300].any()
