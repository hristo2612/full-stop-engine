"""aimv paint canon: the character drift check on synthetic drawings (a painted body, a smeared hand, brush streaks
fenced by paint, white shoe parts, a figure in pieces, colour in a clip frame) and the per-clip rules. Nothing is
paid."""

import cv2
import numpy as np

from aimv.paint import canon as C

H = 400


def body(shoes: bool = True) -> np.ndarray:
    """A crude standing figure, 400 px tall: head, torso, two legs, two shoes; white sock bands above the shoes and a
    white midsole stripe inside them when `shoes`."""
    d = np.zeros((460, 260), np.float32)
    cv2.circle(d, (130, 60), 30, 1.0, -1)
    cv2.rectangle(d, (95, 90), (165, 250), 1.0, -1)
    cv2.rectangle(d, (100, 250), (125, 410), 1.0, -1)
    cv2.rectangle(d, (135, 250), (160, 410), 1.0, -1)
    cv2.rectangle(d, (95, 410), (140, 430), 1.0, -1)
    cv2.rectangle(d, (130, 410), (175, 430), 1.0, -1)
    if shoes:
        d[388:398, 100:125] = 0.0  # socks between cuff and shoe
        d[388:398, 135:160] = 0.0
        d[420:424, 100:136] = 0.0  # midsole stripes inside the shoes
        d[420:424, 136:170] = 0.0
    return d


def test_clean_body():
    m = C.metrics(body(), 1.0)
    assert m["smear"] == 0.0
    assert m["pieces"] == 0.0
    assert m["shoes"] > 1.0
    assert 380 < m["height"] < 420


def test_smeared_hand_counts_but_fenced_streaks_do_not():
    d = body()
    blob = np.zeros_like(d)
    cv2.circle(blob, (205, 150), 22, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 9) * 0.55  # a grey ghost of a hand fading into the wall
    smeared = np.maximum(d, blob)
    streaked = d.copy()
    streaked[120:220, 110:150] = 0.4  # a light dry-brush area inside the torso, dark paint all round it
    assert C.metrics(smeared, 1.0)["smear"] > C.SMEAR_ERR
    assert C.metrics(streaked, 1.0)["smear"] == 0.0


def test_shoes_and_pieces():
    assert C.metrics(body(shoes=False), 1.0)["shoes"] < 0.2 * C.metrics(body(), 1.0)["shoes"]
    d = body()
    cv2.circle(d, (235, 40), 18, 1.0, -1)  # a loose scribble away from the body
    cv2.rectangle(d, (5, 300), (40, 380), 1.0, -1)
    assert C.metrics(d, 1.0)["pieces"] > C.PIECES


def test_colour_ignores_paint_wall_and_the_guide_block():
    d = body()
    frame = np.full((d.shape[0], d.shape[1], 3), 235, np.uint8)
    frame[d > 0.5] = (20, 20, 20)
    assert C.colour(frame, d, 0, 0, key=(172, 203, 236)) == 0.0
    frame[40:80, 110:150] = (172, 203, 236)[::-1]  # the pale blue block behind the head: ignored
    assert C.colour(frame, d, 0, 0, key=(172, 203, 236)) == 0.0
    frame[40:80, 110:150] = (110, 150, 210)  # skin (BGR): a face showed up
    assert C.colour(frame, d, 0, 0, key=(172, 203, 236)) > C.COLOUR


def test_judge_flags_a_clip():
    clean = C.metrics(body(), 1.0)
    canon = {"area": clean["area"]}
    names = [f"X-{i:03d}" for i in range(12)]
    met = {n: dict(clean) for n in names}
    met["X-004"] = dict(clean, area=clean["area"] * 1.7)  # grew toward the camera
    met["X-005"] = dict(clean, smear=1.5)
    met["X-006"] = dict(clean, colour=0.05)
    met["X-007"] = dict(clean, shoes=0.0)
    summary, flags = C.judge("X", names, met, canon, clip=True)
    rules = {n: {(lv, r) for lv, r, _ in flags[n]} for n in names}
    assert ("error", "scale") in rules["X-004"]
    assert ("error", "smear") in rules["X-005"]
    assert ("error", "colour") in rules["X-006"]
    assert rules["X-007"] == {("warning", "shoes")}
    assert not rules["X-003"] and summary["errors"] == 3
    # most of the clip loses the shoes: an error on each of them
    for n in names[3:9]:
        met[n] = dict(clean, shoes=0.0)
    summary, flags = C.judge("X", names, met, canon, clip=True)
    assert summary["shoes_gone"] and ("error", "shoes") in {(lv, r) for lv, r, _ in flags["X-008"]}
    # a sheet gets the scale and pieces rules only
    _, flags = C.judge("X", names, met, canon, clip=False)
    assert all(r in ("scale", "pieces") for n in names for _, r, _ in flags[n])


def test_judge_boxed_clips_and_waivers():
    clean = C.metrics(body(), 1.0)
    canon = {"area": clean["area"]}
    names = [f"X-{i:03d}" for i in range(8)]
    small = dict(clean, area=clean["area"] * 0.4)  # folded small in a box: under the canon's 0.55
    met = {n: dict(small) for n in names}
    met["X-003"] = dict(small, smear=1.5)
    _, flags = C.judge("X", names, met, canon, clip=True)
    assert ("error", "scale") in {(lv, r) for lv, r, _ in flags["X-001"]}
    summary, flags = C.judge("X", names, met, canon, clip=True, boxed=True,
                             waive={"X-003": {"smear": "a highlight on his sleeve"}})
    assert not flags["X-001"]  # the box sets his size; his clip's median still holds
    assert [(lv, r) for lv, r, _ in flags["X-003"]] == [("warning", "smear")]
    assert "waived by eye: a highlight on his sleeve" in flags["X-003"][0][2] and summary["errors"] == 0
