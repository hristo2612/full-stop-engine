"""He becomes true (aimv/paint/truelook.py): the green key, the solid figure's look and the `true` prop."""

import cv2
import numpy as np
import pytest

from aimv.paint import truelook as TL
from test_paint import _with, world  # noqa: F401  (test_paint's module fixture: a small street, poses and song)

BASE = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}]}
GREEN = (75, 208, 80)  # BGR of a chroma screen, as the true sheets'


def _figure_on_green(h=240, w=320):
    """A dark figure (a hoodie: near-black with a lighter fold), a white shoe flash and a skin-coloured hand on green."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = GREEN
    cv2.ellipse(img, (160, 120), (40, 90), 0, 0, 360, (22, 20, 21), -1)
    cv2.line(img, (150, 60), (170, 170), (60, 58, 60), 5)  # a fold
    cv2.rectangle(img, (140, 200), (190, 212), (235, 235, 235), -1)  # the shoe's white
    cv2.circle(img, (208, 80), 9, (120, 150, 205), -1)  # a hand
    return img


def test_screen_matte_keys_green_keeps_the_figure_its_whites_and_hand():
    img = _figure_on_green()
    a, clean = TL.screen_matte(img)
    assert a[10, 10] == 0 and a[120, 160] == 1 and a[206, 165] == 1 and a[80, 208] == 1
    assert a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    # despill: green never above the mean of red and blue (+6) on the figure; the hoodie stays neutral
    f = clean.astype(int)
    assert (f[..., 1] <= (f[..., 0] + f[..., 2]) / 2 + 6 + 1).all()
    assert abs(int(clean[100, 185, 1]) - 20) <= 3  # off the fold


def test_cut_sheet_finds_the_figures_in_reading_order():
    sheet = np.zeros((500, 900, 3), np.uint8)
    sheet[:] = GREEN
    centres = [(150, 120), (450, 130), (750, 110), (150, 380), (450, 370), (750, 390)]
    for k, (x, y) in enumerate(centres):
        cv2.ellipse(sheet, (x, y), (30 + 3 * k, 80), 0, 0, 360, (20, 20, 20), -1)
    parts = TL.cut_sheet(sheet, 6)
    assert len(parts) == 6
    xs = [b[0] + b[2] / 2 for _, _, b in parts]
    ys = [b[1] + b[3] / 2 for _, _, b in parts]
    assert all(abs(x - c[0]) < 3 and abs(y - c[1]) < 3 for x, y, c in zip(xs, ys, centres))
    a, bgr, (x0, y0, w, h) = parts[0]
    assert a.shape == bgr.shape[:2] == (h, w) and a.max() == 1


def _blob(h=200, w=90):
    a = np.zeros((h, w), np.float32)
    cv2.ellipse(a, (w // 2, h // 2), (w // 2 - 6, h // 2 - 6), 0, 0, 360, 1.0, -1)
    rgb = np.full((h, w, 3), 0.012, np.float32)
    return rgb, a


def test_the_solid_figure_is_near_black_with_a_rim_on_top_and_back():
    plate = np.full((400, 400, 3), 0.05, np.float32)  # a city glow behind him
    rgb, a = _blob()
    out = TL.compose(plate, plate, [(rgb, a, 150, 100, 1.0)], {"rain": 0, "grain": 0.0, "haze": 0.0})
    core = out[200, 195]
    assert core.max() < 0.01 and abs(core[0] - core[2]) < 0.002  # deep and neutral (sRGB about 8 to 18)
    top = out[107:112, 190:200].mean(0).mean(0)  # just inside his top edge
    bottom = out[288:293, 190:200].mean(0).mean(0)  # just inside his bottom edge
    assert top[0] > 4 * max(bottom[0], 1e-4) and top[0] > top[2]  # a warm rim on top, little underneath
    back = out[200, 156:161].mean(0)  # facing right: his back is on the left
    front = out[200, 229:234].mean(0)
    assert back[0] > front[0]


def test_the_rim_reads_against_a_dark_background_and_is_wider_there():
    rgb, a = _blob()
    wid = []
    for lum in (0.0005, 0.05):
        plate = np.full((400, 400, 3), lum, np.float32)
        out = TL.compose(plate, plate, [(rgb, a, 150, 100, 1.0)], {"rain": 0, "grain": 0.0, "haze": 0.0,
                                                                  "bloom_k": 0.0, "soft": 0.0})
        col = out[100:200, 195, 0]  # down through his top edge
        assert col.max() > 0.025 > 40 * lum or lum > 0.01  # his outline reads even against the dark
        wid.append(int((col > 0.5 * col.max()).sum()))
    assert wid[0] >= wid[1]


def test_after_images_contact_shadow_and_held_photos():
    plate = np.full((400, 600, 3), 0.05, np.float32)
    rgb, a = _blob()
    L = {"rain": 0, "grain": 0.0, "haze": 0.0}
    figs = [(rgb, a, 40, 100, 1.0), (rgb, a, 200, 100, 1.0), (rgb, a, 360, 100, 1.0)]
    out = TL.compose(plate, plate, figs, L)
    g1, g2 = out[200, 245, 0], out[200, 85, 0]  # the newest after-image darker than the older
    assert g1 < g2 < 0.05
    # grounded: a soft shadow on the surface under his feet; in the air none
    on = TL.compose(plate, plate, [(rgb, a, 200, 100, 1.0)], L, feet_y=300)
    air = TL.compose(plate, plate, [(rgb, a, 200, 50, 1.0)], L, feet_y=300)
    assert on[303, 245, 0] < 0.045 and abs(air[303, 245, 0] - 0.05) < 1e-4


def test_the_true_prop_marks_the_stretch(world):  # noqa: F811
    moves = [{"true": {"from": 4.3, "look": {"after": [0.2]}}},
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}},
             {"pose": {"at": 4.4, "pose": "RUN-2", "x": 760}}]
    ch = _with(world, "vtrue", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    assert len(ch.true) == 1 and ch.true[0]["look"] == {"after": [0.2]} and ch.true[0]["q1"] == float("inf")
    walls = sorted([e for e in ch.events if e.layer == "wall"], key=lambda e: e.q)
    assert [TL.is_true(ch, e) for e in walls] == [False, True]
    for name, args, why in (("vtrue1", {}, "give from"), ("vtrue2", {"from": 4.3, "to": 4.2}, "after from"),
                            ("vtrue3", {"from": 4.3, "look": {"nope": 1}}, "unknown look keys")):
        ch = _with(world, name, dict(BASE, moves=[{"true": args}, {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]))
        assert any(why in i.text for i in ch.errors), (name, [str(i) for i in ch.issues])


def test_place_rgba_matches_the_matte_placement_and_add_shows_the_photos(world):  # noqa: F811
    """place_rgba puts the drawing's RGB exactly where History.place puts its matte (scale, anchor, mirror, flip,
    rotation); add records true photos without paint marks, and shown picks the current photo, its after-images on
    its own photo and none on a held photo."""
    from aimv.paint.render import History

    moves = [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}},
             {"pose": {"at": 4.2, "pose": "RUN-2", "x": 680, "mirror": True}},
             {"pose": {"at": 4.3, "pose": "RUN-1", "x": 760, "rot": 12}},
             {"pose": {"at": 4.4, "pose": "RUN-2", "x": 840, "flipv": True}}]
    ch = _with(world, "vtrue_place", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    hist = History(ch)
    idx = [i for i, e in enumerate(hist.events) if e.layer == "wall"]
    assert len(idx) == 4
    for i in idx:
        d, x0, y0 = hist.place(i)
        rgb, a, rx0, ry0 = TL.place_rgba(hist, i)
        assert a.shape == d.shape and rgb.shape == d.shape + (3,)
        assert (rx0, ry0) == pytest.approx((x0, y0), abs=1e-3)
        assert np.abs(a - np.clip(d, 0, 1)).max() < 0.02
    # add: true photos without marks
    n_marks = len(hist.marks)
    for i in idx:
        TL.add(hist, i, hist.events[i], ch.lib.m(hist.events[i].pose))
    assert len(hist.marks) == n_marks and len(hist.true_q) == 4
    ch.true = [{"q0": 0.0, "q1": float("inf"), "look": {}}]
    qs = sorted(hist.true_q)
    assert TL.shown(hist, qs[0] - 1) == []
    assert TL.shown(hist, qs[3]) == [hist.true_q[qs[1]][0], hist.true_q[qs[2]][0], hist.true_q[qs[3]][0]]
    assert TL.shown(hist, qs[3] + 0.5) == [hist.true_q[qs[3]][0]]  # held: no after-images


def test_only_the_socks_and_midsoles_stay_white():
    """The canon shoes in the true look: the flash and the laces go dark with him (no brand's mark), the thick white
    midsole along the sole and the short white sock stay (the paint figure's signature)."""
    h, w = 220, 300
    a = np.zeros((h, w), np.float32)
    rgb = np.full((h, w, 3), 0.01, np.float32)
    a[20:120, 110:170] = 1  # the leg (dark trousers)
    a[120:150, 110:170] = 1
    rgb[120:150, 110:170] = 0.8  # the sock: across the whole leg, reaching both sides of his outline
    a[150:200, 60:260] = 1  # the shoe
    rgb[180:196, 60:260] = 0.8  # the midsole: a thick band 4 px above the sole's edge
    cv2.line(rgb, (100, 170), (220, 160), (0.85, 0.85, 0.85), 4)  # the flash: a thin stroke inside the upper
    keep = TL.shoe_whites(rgb, a, TL.TRUE_LOOK, 1.0)
    assert keep[135, 140] and keep[190, 160]  # the sock, the midsole
    assert not keep[165, 160] and not keep[60, 140]  # the flash, the trousers
    fig = TL.body_colour(rgb, TL.TRUE_LOOK, keep.astype(np.float32))
    assert fig[190, 160, 0] > 0.2 and fig[165, 160, 0] < 0.02


def test_true_insert_crops_its_frames_to_each_photo_camera(world, tmp_path):  # noqa: F811
    """`true_insert` (truelook.py): a full-frame clip in place of a stretch's photos, its frame by time, cropped from
    the insert's locked camera to the photo's own camera; outside the stretch and for the other aspect, nothing."""
    from types import SimpleNamespace

    clip = world["root"] / "boards" / "insert.mp4"
    wr = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for k in range(30):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:, :320] = (0, 0, 200)  # red left half (BGR), blue right half
        img[:, 320:] = (200, 0, 0)
        cv2.putText(img, str(k), (20, 60), 0, 1.5, (255, 255, 255), 3)
        wr.write(img)
    wr.release()
    cam16 = [1000.0, 500.0, 640.0, 360.0, 0.0]
    moves = [{"true_insert": {"clip": "boards/insert.mp4", "from": 4.1, "to": 5.1, "aspect": "16:9", "camera": cam16}},
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]
    ch = _with(world, "vtrue_ins", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    ins = ch.true_inserts[0]
    t0, t1 = ins["t0"], ins["t1"]
    assert TL.insert_frame_index(ins, t0) == 0 and TL.insert_frame_index(ins, t1) == 29
    r = SimpleNamespace(ch=ch)
    full = SimpleNamespace(cx=1000.0, cy=500.0, cw=640.0, ch=360.0, rot=0.0)
    img = TL.insert_photo(r, 1, t0 + 0.01, 0.0, full, (640, 360))
    assert img is not None and img.shape == (360, 640, 3)
    assert img[200, 100, 0] > 0.6 and img[200, 540, 2] > 0.6  # red stays left, blue right (sRGB float, RGB)
    # a camera zoomed in on the right half: all blue
    right = SimpleNamespace(cx=1160.0, cy=500.0, cw=320.0, ch=180.0, rot=0.0)
    img = TL.insert_photo(r, 1, t0 + 0.01, 0.0, right, (640, 360))
    assert img[180, 50, 2] > 0.6 and img[180, 50, 0] < 0.3
    # a camera reaching past the insert's left edge: the engine's own photo fills what the insert lacks (feathered)
    r.photo = lambda k, t, q, cam, size: np.full((size[1], size[0], 3), (0.0, 0.8, 0.0), np.float32)
    edge = SimpleNamespace(cx=700.0, cy=500.0, cw=320.0, ch=180.0, rot=0.0)
    img = TL.insert_photo(r, 1, t0 + 0.01, 0.0, edge, (640, 360))
    assert img[180, 20, 1] > 0.6 and img[180, 20, 0] < 0.3  # the filled part (left of the insert's frame)
    assert img[180, 620, 0] > 0.6 and img[180, 620, 1] < 0.3  # the insert's red half
    # the other aspect, and after the stretch: the photo renders as usual
    assert TL.insert_photo(r, 1, t0 + 0.01, 0.0, full, (360, 640)) is None
    assert TL.insert_photo(r, 1, t1 + 0.01, 0.0, full, (640, 360)) is None


def test_nothing_paints_in_a_true_stretch(world):  # noqa: F811
    """A template's paint stamps (a take-off splash, drips) inside a true stretch are dropped: no splash mark."""
    from aimv.paint.render import History

    moves = [{"true": {"from": 4.1}},
             {"stamp": {"at": 4.2, "stamp": "SPL-1", "x": 700, "y": 300, "w": 300, "splash": True}},
             {"pose": {"at": 4.3, "pose": "RUN-1", "x": 760}}]
    ch = _with(world, "vtrue_stamp", dict(BASE, moves=moves))
    assert not ch.errors, [str(i) for i in ch.errors]
    hist = History(ch)
    assert not [m for m in hist.marks if m.kind in ("splash", "primer", "pose")]
    assert len(hist.true_q) == 1
