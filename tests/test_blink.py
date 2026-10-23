"""The blink prop template (street lights blinking on the music), held skins, and QA's view of a world layer's rows,
on test_paint's synthetic project and test_world's synthetic world."""

import json
import shutil
from types import SimpleNamespace

import numpy as np

from aimv.paint.render import History

from test_paint import _with, world  # noqa: F401  (its synthetic project: song grid, poses, street)


def test_blink_dims_lights_on_the_kicks(world):
    root = world["root"]
    sd = root / "paint" / "streets" / "blink"
    if not sd.exists():
        shutil.copytree(root / "paint" / "streets" / "main", sd)
        o = json.loads((sd / "obstacles.json").read_text())
        o["lights"] = [dict(o["lights"][0], id="lamp"), {"id": "sign", "x": 2000, "y": 300, "sigma": 200}]
        (sd / "obstacles.json").write_text(json.dumps(o))
    d = {"street": "blink", "seed": 3, "motion": "m0", "pause": False,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"blink": {"lights": ["lamp", "sign"], "from": 4.1, "to": "5.4.4", "on": "kick",
                              "stagger": 1}},
                   {"blink": {"lights": "lamp", "on": ["6.2"], "from": 6.1, "to": 6.4, "dim": 0.3}},
                   {"run": {"from": 4.1, "to": 6.4, "x": [900, 2400]}}]}
    ch = _with(world, "blink", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    kicks = [g.snap(g.q(t)) for t, s in g.hits["kick"] if g.t(g.resolve(4.1)[0]) - 0.02 <= t <= g.t(g.resolve("5.4.4")[0])
             and s >= g.thresholds.get("kick", 0)]
    assert kicks
    for q in kicks:
        assert ch.light_hints[round(q, 4)]["lamp"] == 0.08
        assert ch.light_hints[round(q + 1, 4)]["sign"] == 0.08  # one 16th later: the chase
    assert ch.light_hints[round(g.resolve("6.2")[0], 4)]["lamp"] == 0.3


def test_a_held_skin_stays_until_the_next_figure():
    skin = SimpleNamespace(kind="sil", flags={"skin": True})
    flight = SimpleNamespace(kind="sil", flags={})
    h = SimpleNamespace(fig_box={10.0: 1, 14.0: 1, 20.0: 1}, air={10.0: [skin], 14.0: [], 20.0: [flight]})
    held = lambda q: History.held_skins(h, q)
    assert held(11.0) == [skin] and held(13.0) == [skin]  # held after its photo
    assert held(10.0) == []  # on its own photo it is shown anyway
    assert held(15.0) == []  # a wall pose came after it
    assert held(21.0) == [] and held(9.0) == []  # a flight is not held; nothing before the first


def test_qa_reads_a_world_layers_rows():
    from aimv.paint import qa
    al = np.zeros((100, 50), np.float32)
    al[10:20, 5:15] = 1.0  # rows -490 .. -480 in street y (the layer's rows start at -500)
    L = {"id": "lamp", "parts": []}
    d = {"a0": 1000, "alpha": al, "parallax": 0.0, "x": 1025.0, "y0": -500}
    sc = SimpleNamespace(layers=[L], layer=lambda L_: d)
    m = qa._layers_mask(SimpleNamespace(scene=sc), 990, -495, 40, 30, 1025.0)
    ys, xs = np.nonzero(m > 0.5)
    assert ys.min() == -490 + 495 and ys.max() == -481 + 495 and xs.min() == 15 and xs.max() == 24


def test_cam_pulls_back_and_tilts_with_eased_hints(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"cam": {"from": 5.1, "to": 6.1, "zoom": 1.7, "dy": -200, "ease": 4}},
                   {"run": {"from": 4.1, "to": 6.4, "x": [900, 2400]}}]}
    ch = _with(world, "camhint", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    hint = lambda a: ch.cam_hints.get(round(g.resolve(a)[0], 4), {})
    assert not hint("4.4.4") and not hint("6.1")
    mid = hint("5.3")
    assert mid["zoom"] == 1.7 and mid["dy"] == -200 * ch.u and mid["free"] is False and mid["dx"] > 0
    assert 1.0 < hint("5.1")["zoom"] < hint("5.1.3")["zoom"] < 1.7  # eased in
    assert 1.0 < hint("5.4.4")["zoom"] < 1.7  # and out
