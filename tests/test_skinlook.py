"""The solid skin after the wall (aimv/paint/skinlook.py, Full Stop v7's pick A)."""

import numpy as np

from aimv.paint import skinlook as SL
from test_paint import _with, world  # noqa: F401  (test_paint's module fixture: a small street, poses and song)

BASE = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}]}


def test_the_skin_look_paints_skins_opaque_with_a_pale_rim_only_in_its_stretch(world):  # noqa: F811
    from aimv.paint.render import History

    moves = [{"skin_look": {"from": 4.3}},
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 600, "skin": True}},
             {"pose": {"at": 4.4, "pose": "RUN-1", "x": 900, "skin": True}}]
    ch = _with(world, "vskinlook", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    assert len(ch.skin_looks) == 1 and ch.skin_looks[0]["look"] == {}
    hist = History(ch)
    qs = sorted(q for q, its in hist.air.items() if any(it.flags.get("skin") for it in its))
    assert len(qs) == 2
    for q, inside in zip(qs, (False, True)):
        it = next(i for i in hist.air[q] if i.flags.get("skin"))
        x0, y0 = int(it.x0) - 20, int(it.y0) - 20
        w, h = 500, 600
        A0, c0 = hist.air_layer(ch.grid.t(q), q, x0, y0, w, h)
        A1, c1 = SL.apply(hist, q, A0, c0, x0, y0, w, h)
        if not inside:  # before the stretch: unchanged (the alley leap's skin)
            assert np.array_equal(A0, A1) and np.array_equal(c0, c1)
            continue
        s, wh, stk, ring = SL.figure(hist, it, SL.SKIN_LOOK, ch.u)
        core = (s > 0.99) & (ring < 0.01) & (wh < 0.01)
        ys, xs = np.nonzero(core)
        pick = (ys[len(ys) // 2] + 20, xs[len(xs) // 2] + 20)
        assert A1[pick] > 0.99 and A0[pick] < A1[pick] + 1e-6  # opaque
        assert c1[pick].max() < 0.04  # black paint (strokes at most dark grey)
        ry, rx = np.unravel_index(int(np.argmax(ring)), ring.shape)
        assert ring.max() > 0.3 and c1[ry + 20, rx + 20].min() > 0.06  # the pale rim (blended at its peak)


def test_the_skin_look_paints_flights_too(world):  # noqa: F811
    """A flight's J2 silhouette (no skin flag: a leap, a hop, a flip) inside the stretch is painted as skin A."""
    from aimv.paint.render import History

    moves = [{"skin_look": {"from": 4.1}}, {"pose": {"at": 4.3, "pose": "RUN-1", "x": 800, "layer": "air"}}]
    ch = _with(world, "vskinfl", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    hist = History(ch)
    q, it = next((q, i) for q, its in hist.air.items() for i in its if i.kind == "sil")
    assert not it.flags.get("skin")
    x0, y0, w, h = int(it.x0) - 20, int(it.y0) - 20, 500, 600
    A1, c1 = hist.air_layer(ch.grid.t(q), q, x0, y0, w, h)
    s, wh, stk, ring = SL.figure(hist, it, SL.SKIN_LOOK, ch.u, SL.screen_scale(w, h))
    core = (s > 0.99) & (ring < 0.01) & (wh < 0.01)
    ys, xs = np.nonzero(core)
    pick = (ys[len(ys) // 2] + 20, xs[len(xs) // 2] + 20)
    assert A1[pick] > 0.99 and c1[pick].max() < 0.04  # opaque black paint


def test_the_skin_look_checks_its_args(world):  # noqa: F811
    for name, args, why in (("vskl1", {}, "give from"), ("vskl2", {"from": 4.3, "to": 4.2}, "after from")):
        ch = _with(world, name, dict(BASE, moves=[{"skin_look": args}, {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]))
        assert any(why in i.text for i in ch.errors), (name, [str(i) for i in ch.issues])


def test_the_rim_stays_readable_on_screen(world):  # noqa: F811
    from aimv.paint.render import History

    moves = [{"skin_look": {"from": 4.1}}, {"skin_look": {"from": 4.3, "to": 5.1, "trail": 0.2}}]
    moves += [{"pose": {"at": f"4.{b}", "pose": "RUN-1", "x": 600 + 150 * b, "skin": True}} for b in (1, 2, 3, 4)]
    ch = _with(world, "vskinrim", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    assert SL.span_at(ch, ch.skin_looks[1]["q0"])["look"].get("trail") == 0.2  # the later, shorter stretch wins
    assert SL.span_at(ch, ch.skin_looks[0]["q0"])["look"] == {}
    L, u = SL.SKIN_LOOK, ch.u
    # on screen: at least 3 px at the normal 16:9 framing (1529 px of street across 1920), 4 px zoomed out to 1.8x
    for cw, px in ((1529, 3.0), (2752, 4.0)):
        scr = SL.screen_scale(cw + 13, int(cw * 9 / 16) + 13)
        assert abs(scr - 1920 / cw) < 1e-6 and SL.rim_width(L, u, scr) * scr >= px - 1e-6
    assert SL.rim_width(L, u, 3.0) == SL.rim_width(L, u, None)  # close up: rim_px as drawn
    hist = History(ch)
    qs = sorted(q for q, its in hist.air.items() if any(it.flags.get("skin") for it in its))
    it = next(i for i in hist.air[qs[0]] if i.flags.get("skin"))
    s0, _, _, r0 = SL.figure(hist, it, L, u, 1920 / 1529)
    s1, _, _, r1 = SL.figure(hist, it, L, u, 1920 / 2752)
    assert r1.sum() > 1.5 * r0.sum()  # the rim band wider when zoomed out


def test_a_flights_echoes_are_four_soft_smears_from_35_to_8_percent(world):  # noqa: F811
    from aimv.paint.render import History

    L = SL.SKIN_LOOK
    assert abs(SL.echo_alpha(L, 1) - 0.35) < 1e-6 and abs(SL.echo_alpha(L, 4) - 0.08) < 1e-6
    assert SL.echo_alpha(L, 2) > SL.echo_alpha(L, 3) > SL.echo_alpha(L, 4) and SL.echo_alpha(L, 5) == 0.0
    moves = [{"skin_look": {"from": 4.1}}]
    moves += [{"pose": {"at": f"4.{b}", "pose": "RUN-1", "x": 600 + 150 * b, "layer": "air"}} for b in (1, 2, 3, 4)]
    moves += [{"pose": {"at": "5.1", "pose": "RUN-1", "x": 1400, "skin": True}}]
    ch = _with(world, "vskintrail", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    hist = History(ch)
    qs = sorted(q for q, its in hist.air.items() if any(i.kind == "sil" for i in its))
    q = qs[3]
    it = next(i for i in hist.air[q] if i.kind == "sil")
    sk = next(i for i in hist.air[qs[-1]] if i.kind == "sil")
    assert SL.trail_owns(hist, q, it) and not SL.trail_owns(hist, qs[-1], sk)  # flights, not the run's skins
    x0, y0, w, h = int(it.x0) - 500, int(it.y0) - 40, 1000, 640
    A0, c0 = hist.air_layer(ch.grid.t(q), q, x0, y0, w, h)
    ch.skin_looks[0]["look"]["trail"] = 0.0
    A1, c1 = hist.air_layer(ch.grid.t(q), q, x0, y0, w, h)
    echo = (A0 - A1) > 0.02
    assert echo.any() and A0[echo].max() < 0.45  # the echoes show, and none is near solid
