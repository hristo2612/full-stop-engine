"""The skin over voids (render.History): a wall pose with sky or a void behind most of him becomes the J2 skin, with a
soft shadow on the wall behind it (never on the void), on test_world's synthetic world."""

import json
import shutil
from types import SimpleNamespace

import numpy as np

from aimv.paint.render import SKIN_VOID, History
from aimv.paint.scene import Scene

from test_world import B, GX, GY, STEP, _street_to_tex, built  # noqa: F401  (its synthetic street and world)


def test_a_pose_over_a_void_is_a_skin_with_a_shadow_on_the_wall(built, tmp_path):
    sd = tmp_path / "s"
    shutil.copytree(built["sd"], sd)
    ox, oy = _street_to_tex(built["sd"])
    rx0, ry0 = GX - ox + 2 * STEP, GY + B[0] - oy  # the roof band's cell
    o = json.loads((sd / "obstacles.json").read_text())
    o["voids"] = [{"id": "sky", "box": [rx0, ry0, rx0 + 400, ry0 + 250]}]
    (sd / "obstacles.json").write_text(json.dumps(o))
    sc = Scene(sd, tmp_path / "cache")
    fake = SimpleNamespace(ch=SimpleNamespace(scene=sc), u=1.0)
    d = np.zeros((120, 60), np.float32)
    d[10:110, 15:45] = 1.0  # a figure's matte
    ev = lambda x: SimpleNamespace(flags={}, x=x)
    over = lambda x, y: History._over_void(fake, ev(x), d, x, y)
    assert over(rx0 + 100, ry0 + 20)  # all of him over the void
    assert not over(rx0 + 600, ry0 + 20)  # over the cell's wall
    assert not over(rx0 + 380, ry0 + 20 + 250 - 120 * (1 - SKIN_VOID) + 10)  # mostly over the wall below it
    assert not History._over_void(fake, SimpleNamespace(flags={"takeoff": True}, x=rx0 + 100), d, rx0 + 100, ry0)
    assert not over(100, 100)  # the plain street (left of the world) never turns into skins
    # the shadow: down and right of him, on the wall only
    item = SimpleNamespace(x0=rx0 + 300, y0=ry0 + 150, flags={"skin": True})
    fake.air_mask = lambda it: d
    out = np.ones((400, 600, 3), np.float32)
    sh = History.skin_shadow(fake, out, [item], rx0, ry0, 600, 400)
    v = sc.void(rx0, rx0 + 600, ry0, ry0 + 400)
    dark = 1 - sh[..., 0]
    assert dark.max() > 0.2 and dark[v > 0.5].max() < 1e-6  # a shadow, never on the void
    ys, xs = np.nonzero(dark > 0.1)
    assert xs.mean() > 300 + 30 and ys.mean() > 150 + 60  # offset down and right of him


def test_skin_and_paint_swap_only_when_the_background_holds():
    from aimv.paint.render import skin_hysteresis
    qs = list(range(20))
    raw = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    states, fades = skin_hysteresis(qs, [bool(r) for r in raw], hold=3, fade=2)
    # a single photo of sky (3) or of wall (11) does not flip it; a stretch that holds 3 photos does
    assert [int(s) for s in states] == [0] * 7 + [1] * 9 + [0] * 4
    # the primer cross-fades over the 2 paint photos on each side of the skin stretch
    assert fades[5:7] == [2 / 3, 1 / 3] and fades[16:18] == [1 / 3, 2 / 3] and fades[0] == fades[19] == 1.0
    # after a long gap it starts afresh from the next photos' majority
    s2, _ = skin_hysteresis([0, 1, 2, 40, 41, 42], [False, False, False, True, True, False], hold=3)
    assert s2 == [False, False, False, True, True, True]


def test_a_skin_has_a_pale_rim_on_its_outline():
    import math as _m
    from aimv.paint import render as R
    d = np.zeros((160, 80), np.float32)
    d[10:150, 15:65] = 1.0
    d[60:80, 30:50] = 0.0  # a pale gap inside him: no rim round it
    item = SimpleNamespace(kind="sil", x0=0, y0=0, glow=(1.0, 0.55, 0.22), flags={"skin": True})
    h = SimpleNamespace(air={1.0: [item]}, _air_q=[1.0], u=1.0, _drops=np.zeros((0, 6)),
                        ch=SimpleNamespace(paint_clock_back=lambda t, s: t, freezes=[], grid=None), air_mask=lambda it: d,
                        held_skins=lambda q: [])
    A, col = R.History.air_layer(h, 0.0, 1.0, 0, 0, 80, 160)
    edge, mid, hole_edge = col[80, 14:20].mean(1).max(), col[40, 40].mean(), col[58:82, 27:33].mean(2).max()
    assert edge > 0.08 and mid < 0.02 and hole_edge < 0.03 and A[40, 40] > 0.9
    item.flags = {}
    A2, col2 = R.History.air_layer(h, 0.0, 1.0, 0, 0, 80, 160)
    assert col2[80, 14:20].mean(1).max() < 0.05  # a flight skin without `skin` keeps the old rim
