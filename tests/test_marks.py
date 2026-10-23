"""The `marks` prop (aimv/paint/marks.py): painted strokes in stages, each stage a stamp at its time, out of the
runner's chain (the smear down the far wall, his word, the dot)."""

from test_paint import _with, world  # noqa: F401  (test_paint's module fixture: a small street, poses and song)

BASE = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}]}


def test_marks_places_each_stage_as_a_stamp_out_of_the_chain(world):  # noqa: F811
    moves = [{"marks": {"stages": [[4.1, "SPL-1", 600, 300, 200], ["4.1.3", "SPL-2", 610, 320, 220]]}},
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}},
             {"pose": {"at": 4.3, "pose": "RUN-2", "x": 760}}]
    ch = _with(world, "vmarks", dict(BASE, moves=moves))
    assert not ch.errors
    st = sorted([e for e in ch.events if e.layer == "stamp" and e.tag == "mark"], key=lambda e: e.q)
    assert [e.pose for e in st] == ["SPL-1", "SPL-2"] and [e.w for e in st] == [200.0, 220.0]
    assert st[0].q < st[1].q and (st[0].x, st[0].y) == (600.0, 300.0)
    assert not st[0].flags.get("primer") and not st[0].flags.get("splash")
    # a prop: the strokes overlap the poses in time without an order warning, and the poses keep their chain
    assert not any("before the previous move ends" in i.text for i in ch.issues)
    assert [e.pose for e in ch.events if e.layer == "wall"] == ["RUN-1", "RUN-2"]


def test_marks_rejects_a_pose_an_unknown_stamp_and_a_bad_stage(world):  # noqa: F811
    for name, stages, why in (("vmarks1", [[4.1, "RUN-1", 600, 300, 200]], "is a pose"),
                              ("vmarks2", [[4.1, "NOPE-1", 600, 300, 200]], "unknown stamp"),
                              ("vmarks3", [[4.1, "SPL-1", 600]], "is not [at, stamp, x, y, w]")):
        ch = _with(world, name, dict(BASE, moves=[{"marks": {"stages": stages}}]))
        assert any(why in i.text for i in ch.errors), (name, [str(i) for i in ch.issues])


def test_a_mark_bigger_than_the_wash_texture_paints(world):  # noqa: F811
    """A stroke stage wider and taller than render's 1500 px rain-wash texture (the word, the smear) paints and washes
    without an error (render tiles the texture)."""
    import numpy as np

    from aimv.paint.render import History

    moves = [{"marks": {"stages": [[4.1, "SPL-1", 900, 400, 1700]]}}, {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]
    ch = _with(world, "vmarks_big", dict(BASE, moves=moves))
    assert not ch.errors
    hist = History(ch)
    i = next(k for k, e in enumerate(hist.events) if e.layer == "stamp" and e.tag == "mark")
    d, x0, y0 = hist.place(i)
    assert d.shape[1] > 1500  # the fixture's stamp scaled to 1700 px wide: past the 1500 px texture
    t = ch.grid.t(ch.grid.resolve(5.4)[0])  # well after it: the rain has washed it (page > 0.3)
    pm, cv, wt = hist.layers(t, int(x0) - 10, int(y0) - 10, d.shape[1] + 20, d.shape[0] + 20)
    assert np.isfinite(pm).all() and cv.max() > 0.3


def test_a_stage_can_carry_its_own_flags(world):  # noqa: F811
    moves = [{"marks": {"stages": [[4.1, "SPL-1", 600, 300, 200, {"splash": True}], [4.2, "SPL-2", 600, 300, 200]]}},
             {"pose": {"at": 4.3, "pose": "RUN-1", "x": 700}}]
    ch = _with(world, "vmarks_fl", dict(BASE, moves=moves))
    assert not ch.errors
    st = sorted([e for e in ch.events if e.layer == "stamp" and e.tag == "mark"], key=lambda e: e.q)
    assert st[0].flags["splash"] and not st[0].flags["primer"] and not st[1].flags["splash"]


def test_grow_makes_the_primer_under_a_landing_grow_in(world):  # noqa: F811
    """choreo's common `grow: [time, n]`: the move's first n wall poses from `time` get patch_grow [k, n + 1]."""
    moves = [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600, "grow": ["4.1.2", 3]}}]
    ch = _with(world, "vgrow0", dict(BASE, moves=moves))
    assert "patch_grow" not in ch.events[0].flags  # before its time
    moves = [{"walk": {"from": 4.1, "to": 5.1, "x": 600, "grow": [4.2, 2]}}]
    ch = _with(world, "vgrow", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    walls = sorted([e for e in ch.events if e.layer == "wall"], key=lambda e: e.q)
    q2 = ch.grid.resolve(4.2)[0]
    grown = [e.flags.get("patch_grow") for e in walls if e.q >= q2 - 1e-6]
    assert grown[:3] == [[1, 3], [2, 3], None]
    assert all("patch_grow" not in e.flags for e in walls if e.q < q2 - 1e-6)


def test_hints_punch_the_camera_and_knock_a_light_out(world):  # noqa: F811
    """The `hints` prop (hints.py): camera punches at set times, a street light knocked out (flare, dark, flecks)."""
    moves = [{"hints": {"punch": [[4.2, 0.07]], "knock": {"at": 4.3, "light": "lamp"}}},
             {"pose": {"at": 4.1, "pose": "RUN-1", "x": 700}}]
    import json as _json
    obs = world["street"] / "obstacles.json"
    d = _json.loads(obs.read_text())
    saved = obs.read_text()
    d["lights"] = [dict(d["lights"][0], id="lamp")]
    obs.write_text(_json.dumps(d))
    try:
        ch = _with(world, "vhints", dict(BASE, moves=moves))
    finally:
        obs.write_text(saved)
    assert not ch.errors, [str(i) for i in ch.errors]
    q2, q3 = ch.grid.resolve(4.2)[0], ch.grid.resolve(4.3)[0]
    assert ch.cam_hints[round(q2, 4)]["punch"] == 0.07
    assert ch.lights_out["lamp"] == q3 and len(ch.bursts) == 1
    assert ch.light_hints[round(q3 - ch.step_at(q3), 4)]["lamp"] > 1  # the flare on the photo before


def test_keepers_makes_a_moves_wall_poses_keepers(world):  # noqa: F811
    """choreo's common `keepers`: true (every wall pose a keeper) or N (every Nth): a longer trail of pale ghosts."""
    base_moves = lambda kp: [{"walk": {"from": 4.1, "to": 5.1, "x": 600, "keepers": False}},
                             {"pose": {"at": 5.2, "pose": "RUN-1", "x": 1400, **({"keepers": kp} if kp else {})}}]
    ch = _with(world, "vkeep", dict(BASE, segments=[{"from": 4.1, "to": 6.1, "rate": "16ths"}],
                                    moves=base_moves(True)))
    assert [e.keeper for e in ch.events if e.pose == "RUN-1"] == [True]
    assert not any(e.keeper for e in ch.events if e.tag == "walk")
    seg = [{"from": 4.1, "to": 6.1, "rate": "16ths"}]
    clip = {"sequence": "flat_SPR", "keys": {"a": 4.1, "b": 4.4}, "x": {"a": 800}}
    ch = _with(world, "vkeep2", dict(BASE, segments=seg, moves=[{"clip": clip}]))
    plain = sum(e.keeper for e in ch.events if e.layer == "wall")
    ch = _with(world, "vkeep3", dict(BASE, segments=seg, moves=[{"clip": dict(clip, keepers=2)}]))
    walls = [e for e in sorted(ch.events, key=lambda e: e.q) if e.layer == "wall"]
    assert all(e.keeper for e in walls[1::2])  # every 2nd
    assert plain < sum(e.keeper for e in walls) < len(walls)


def test_hints_frame_moves_the_camera_over_a_stretch(world):  # noqa: F811
    """The `hints` prop's frame (hints.py): on every photo of the stretch an eased camera hint (zoom, dx, dy, free),
    from the first values to the last; none outside it."""
    moves = [{"hints": {"frame": {"from": 4.2, "to": 5.1, "zoom": [1.8, 1.0], "dx": [400, 100], "dy": [200, 150],
                                  "zoom_v": [1.2, 0.7]}}},
             {"run": {"from": 4.1, "to": 5.4, "x": [600, 1400]}}]
    ch = _with(world, "vhints_frame", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    g = ch.grid
    q0, q1 = g.resolve(4.2)[0], g.resolve(5.1)[0]
    hs = {q: h for q, h in ch.cam_hints.items() if "dx" in h}
    assert hs and min(hs) == q0 and max(hs) == q1
    assert hs[q0]["zoom"] == 1.8 and hs[q1]["zoom"] == 1.0 and hs[q0]["dx"] == 400 and hs[q1]["dy"] == 150
    assert hs[q1]["zoom_v"] == 0.7 and all(h["free"] for h in hs.values())
    mid = sorted(hs)[len(hs) // 2]
    assert 1.0 < hs[mid]["zoom"] < 1.8
    assert set(hs) <= {round(p.q, 4) for p in ch.photos()}  # on the photos themselves
    # a stretch starting between photos, in a segment that starts between beats (seconds): still on the photos
    t_mid = g.t(g.resolve("5.2.2")[0]) + 0.013  # between 16ths
    t_a = g.t(g.resolve("4.3.1")[0]) + 0.007
    seg = [{"from": 4.1, "to": f"{t_mid:.3f}s", "rate": "8ths"}, {"from": f"{t_mid:.3f}s", "to": 6.1, "rate": "16ths"}]
    moves = [{"hints": {"frame": {"from": f"{t_a:.3f}s", "to": 5.4, "zoom": [1.5, 1.0]}}},
             {"run": {"from": 4.1, "to": 5.4, "x": [600, 1400]}}]
    ch = _with(world, "vhints_frame2", dict(BASE, segments=seg, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    hs = {q for q, h in ch.cam_hints.items() if "dx" in h}
    assert hs and hs <= {round(p.q, 4) for p in ch.photos()}


def test_hints_burst_throws_paint_at_a_time(world):  # noqa: F811
    """The `hints` prop's burst (hints.py): a props.Burst at (x, y) on that photo, one per item."""
    moves = [{"hints": {"burst": [{"at": 4.2, "x": 700, "y": 300, "power": 1.5, "flecks": 30},
                                  {"at": 4.3, "x": 720, "y": 320}]}},
             {"pose": {"at": 4.1, "pose": "RUN-1", "x": 700}}]
    ch = _with(world, "vhints_burst", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    assert len(ch.bursts) == 2
    b = ch.bursts[0]
    assert (b.x, b.y, b.power, b.flecks) == (700.0, 300.0, 1.5, 30)
    assert any(abs(q - b.q) < 1e-6 for q in ch.extra_photos)
    ch = _with(world, "vhints_burst2", dict(BASE, moves=[{"hints": {"burst": {"at": 4.2}}},
                                                         {"pose": {"at": 4.1, "pose": "RUN-1", "x": 700}}]))
    assert any("write burst as" in i.text for i in ch.errors)
