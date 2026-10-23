"""aimv paint: grid addressing, street tiles (and prepending on the left with stable coordinates), the choreography
and its templates, pauses in song stops, composition (include), the box prop (keys, cracks, stencils, shatter), foot
lock, the camera.

Everything runs on a small synthetic world built once per module: a 120 BPM breakdown, a street stitched from plates
cut out of one texture, an obstacle registry, and a pose library split from drawn sprite sheets. Nothing is paid.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from aimv.paint import poses as PS
from aimv.paint import props as PR
from aimv.paint import street as ST
from aimv.paint.camera import framing_stats, plan
from aimv.paint.check import check, figure_track, motion_stats
from aimv.paint.choreo import Clock, open_choreo
from aimv.paint.grid import GridError, SongGrid, rate_of
from aimv.paint.moves import foot_lock
from aimv.paint.render import History, Renderer, plan_jobs
from aimv.paint.scene import Scene
from aimv.project import Project

BEAT = 0.5  # 120 BPM, first beat at 0.5 s
NBEATS = 200
PLATE_W, PLATE_H, KEEP = 1200, 860, 0.4


def beat_t(bar, beat=1):
    return 0.5 + ((bar - 1) * 4 + (beat - 1)) * BEAT


# ---------------------------------------------------------------------------------------------- the synthetic world

def _breakdown(root):
    bd = root / "breakdown"
    (bd / "audio").mkdir(parents=True)
    beats = [{"i": k, "t": 0.5 + k * BEAT, "bar": k // 4, "beat": k % 4, "bpm": 120.0} for k in range(NBEATS)]
    (bd / "beats.json").write_text(json.dumps({"version": 1, "beats": beats, "bars": []}))
    stop = (beat_t(3, 2), beat_t(4, 1) - 0.02)
    kick, snare = [], []
    for k in range(NBEATS):
        t = 0.5 + k * BEAT
        if stop[0] <= t < stop[1] + 0.03:
            continue
        (kick if k % 2 == 0 else snare).append({"t": t, "s": 0.9 if k % 2 == 0 else 1.0})
        if k % 4 == 2:
            snare.append({"t": t + 0.75 * BEAT, "s": 0.5})  # a ghost snare: not a backbeat
    snare.sort(key=lambda e: e["t"])
    ev = {"kick": kick, "snare": snare, "hat": [], "vocal_entry": [], "impact": [{"t": beat_t(4, 1), "s": 0.8}],
          "riser": [], "roll": [], "stop": [{"t0": stop[0], "t1": stop[1], "s": 0.9, "end_impact": beat_t(4, 1)}]}
    (bd / "hits.json").write_text(json.dumps({"version": 1, "events": ev}))
    secs = [{"i": 0, "label": "Verse 1", "kind": "verse", "t0": 0.0, "t1": beat_t(5)},
            {"i": 1, "label": "Chorus", "kind": "chorus", "t0": beat_t(5), "t1": beat_t(30)},
            {"i": 2, "label": "Outro", "kind": "outro", "t0": beat_t(30), "t1": 0.5 + NBEATS * BEAT}]
    (bd / "sections.json").write_text(json.dumps({"version": 1, "sections": secs}))
    (bd / "meta.json").write_text(json.dumps({"version": 1, "duration": 0.5 + NBEATS * BEAT}))
    import soundfile as sf
    sf.write(str(bd / "audio" / "mix.wav"), np.zeros(int(8000 * (0.5 + NBEATS * BEAT)), np.float32), 8000)


def _texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for sigma, amp in ((1.2, 40), (4, 45), (12, 40)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        img += (n / (n.std() + 1e-6) * amp)[..., None] * np.float32([1.0, 0.85, 0.7])
    img += np.float32([95, 80, 70])
    img[640:] *= 0.55  # the pavement
    return np.clip(img, 0, 255).astype(np.uint8)


def _figure(img, ox, oy, f1=(0, 0), f2=(0, 0), arms="down", horizontal=False, short=False):
    c = (0, 0, 0)
    if horizontal:  # a slide: the body along the ground
        cv2.line(img, (ox + 70, oy + 325), (ox + 235, oy + 345), c, 70)
        cv2.circle(img, (ox + 55, oy + 320), 28, c, -1)
        cv2.line(img, (ox + 230, oy + 350), (ox + 260, oy + 390), c, 30)
        cv2.line(img, (ox + 100, oy + 330), (ox + 150, oy + 290), c, 20)
        return
    top = 150 if short else 110
    cv2.circle(img, (ox + 165, oy + top - 30), 30, c, -1)
    cv2.line(img, (ox + 150, oy + top), (ox + 150, oy + 250), c, 55)
    hands = {"down": (ox + 210, oy + 300), "up": (ox + 170, oy + 25), "forward": (ox + 255, oy + top + 40),
             "plant": (ox + 240, oy + 395)}[arms]
    cv2.line(img, (ox + 150, oy + top + 20), hands, c, 22)
    for dx, dy in (f1, f2):
        foot = (ox + 150 + dx, oy + 400 - dy)
        cv2.line(img, (ox + 150, oy + 250), foot, c, 34)
        cv2.ellipse(img, (foot[0] + 8, foot[1] - 4), (24, 9), 0, 0, 360, c, -1)


BLOCK = (560, 520, 760, 720)  # the guide block in the synthetic clip (x0, y0, x1, y1); its bottom is the ground
VID_ON = range(20, 29)  # frames where the clip's figure stands on the block


def _clip(root):
    """A 1280x720 24 fps clip: a figure runs left to right over a pale blue block (on it in frames 20 to 28), every
    sixth frame doubled (as models do), a faint grey scribble far behind him and one touching his feet."""
    f = root / "boards" / "clip.mp4"
    w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280, 720))
    prev = None
    for k in range(48):
        if k % 6 == 5 and prev is not None:
            w.write(prev)
            continue
        img = np.full((720, 1280, 3), 236, np.uint8)
        cv2.rectangle(img, BLOCK[:2], (BLOCK[2] - 1, BLOCK[3] - 1), (236, 203, 172), -1)
        cv2.line(img, (40, 300), (70, 400), (175, 175, 175), 3)  # far scribble
        x = 60 + 22 * k
        lift = 200 if k in VID_ON else 0
        a, b = ((70, 0), (-50, 60)) if k % 4 < 2 else ((-60, 40), (60, 40))
        _figure(img, x, 320 - lift, a, b, arms="forward")
        cv2.line(img, (x + 100, 722 - lift), (x + 125, 700 - lift), (170, 170, 170), 2)  # faint, at his feet
        w.write(img)
        prev = img
    w.release()
    return "boards/clip.mp4"


POLE = (600, 670)  # the guide pole in the synthetic pole clip (x0, x1), full height


def _pole_clip(root):
    """A 1280x720 30 fps clip: a figure moves right past a pale blue pole; where they overlap the pole hides the
    middle of him (two pieces, one on either side)."""
    f = root / "boards" / "pole.mp4"
    w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for k in range(40):
        img = np.full((720, 1280, 3), 236, np.uint8)
        a, b = ((70, 0), (-50, 60)) if k % 4 < 2 else ((-60, 40), (60, 40))
        _figure(img, 220 + 16 * k, 300, a, b, arms="forward")
        cv2.rectangle(img, (POLE[0], 0), (POLE[1] - 1, 719), (236, 203, 172), -1)
        w.write(img)
    w.release()
    return "boards/pole.mp4"


SPR_STEP, SPR_PER = 160, 8  # the synthetic sprint: a step of 160 px every 8 frames (30 fps), 20 px a frame


def _sprint(root):
    """A 30 fps sprint with truly planted feet: the torso moves 20 px a frame; each step's foot lands 160 px on and
    stays put for 5 frames (stance), then 3 frames of flight; frames 6, 14, ... are doubled with a little noise."""
    f = root / "boards" / "sprint.mp4"
    w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    rng = np.random.default_rng(2)
    prev = None
    for k in range(64):
        if k % 8 == 6 and prev is not None:
            img = np.clip(prev.astype(int) + rng.integers(-3, 4, prev.shape), 0, 255).astype(np.uint8)
            w.write(img)
            prev = img
            continue
        img = np.full((720, 1280, 3), 236, np.uint8)
        tx = 100 + 20 * k
        hip = (tx, 520)
        cv2.circle(img, (tx + 10, 380), 30, (0, 0, 0), -1)
        cv2.line(img, (tx, 410), hip, (0, 0, 0), 50)
        cv2.line(img, (tx, 430), (tx + 60, 470), (0, 0, 0), 18)
        s_ = k // SPR_PER
        ph = k % SPR_PER
        plant = 100 + SPR_STEP * s_ + 40
        if ph < 5:  # stance: the foot on its plant
            cv2.line(img, hip, (plant, 690), (0, 0, 0), 30)
            cv2.ellipse(img, (plant + 10, 690), (28, 10), 0, 0, 360, (0, 0, 0), -1)
            cv2.line(img, hip, (tx - 60, 640), (0, 0, 0), 26)
        else:  # flight: both feet up
            cv2.line(img, hip, (tx + 50, 650), (0, 0, 0), 30)
            cv2.line(img, hip, (tx - 70, 630), (0, 0, 0), 26)
        w.write(img)
        prev = img
    w.release()
    return "boards/sprint.mp4"


def _sheets(root):
    d = root / "boards"
    d.mkdir()
    run = np.full((880, 1200, 3), 255, np.uint8)
    cyc = [((70, 0), (-50, 60)), ((0, 0), (40, 70)), ((-70, 0), (40, 60)), ((-60, 40), (60, 40))]
    for k, (a, b) in enumerate(cyc + [((75, 0), (-45, 55)), ((5, 0), (35, 75)), ((-75, 0), (35, 55)),
                                      ((-55, 45), (55, 45))]):
        _figure(run, (k % 4) * 300, (k // 4) * 440, a, b, arms="forward")
    cv2.imwrite(str(d / "RUN.png"), run)
    key = np.full((880, 1200, 3), 255, np.uint8)
    poses = [dict(f1=(-45, 0), f2=(45, 0), short=True), dict(f1=(-60, 30), f2=(40, 80), arms="forward"),
             dict(f1=(-80, 20), f2=(20, 60), arms="forward"), dict(f1=(-60, 120), f2=(-20, 140), arms="plant"),
             dict(f1=(-10, 0), f2=(10, 0), arms="up"), dict(f1=(-70, 0), f2=(60, 0)), dict(horizontal=True),
             dict(f1=(-50, 0), f2=(50, 0))]
    for k, kw in enumerate(poses):
        _figure(key, (k % 4) * 300, (k // 4) * 440, **kw)
    cv2.imwrite(str(d / "KEY.png"), key)
    spl = np.full((420, 600, 3), 255, np.uint8)
    rng = np.random.default_rng(4)
    for c in range(2):
        cv2.circle(spl, (150 + 300 * c, 210), 70, (0, 0, 0), -1)
        for _ in range(60):
            cv2.circle(spl, (int(150 + 300 * c + rng.normal(0, 70)), int(210 + rng.normal(0, 60))),
                       int(rng.uniform(2, 8)), (0, 0, 0), -1)
    cv2.imwrite(str(d / "SPL.png"), spl)


ROLES = {
    "crouch": "KEY-1", "brake_in": ["KEY-6", "KEY-1"], "brake": "KEY-6", "takeoff": "KEY-2",
    "flight": ["KEY-3", "KEY-3", "KEY-2", "KEY-2"], "land": "KEY-6", "recover": ["KEY-8", "RUN-2"], "jump": "KEY-2",
    "land_small": "KEY-6", "recover_small": "KEY-8", "land_vault": "KEY-6", "reach": "KEY-3",
    "vault_approach": "RUN-1", "vault_plant": "KEY-4", "vault_over": "KEY-4", "drop": "KEY-2", "slide_in": "KEY-1",
    "slide": "KEY-7", "slide_out": "KEY-1", "rise": ["KEY-8", "RUN-1"], "climb": ["KEY-5", "KEY-5"], "hang": "KEY-5",
    "swing": "KEY-5", "release": "KEY-3", "splash": "SPL-1", "burst": "SPL-2", "smear": "SPL-2", "spray": "SPL-2",
    "drips": "SPL-1", "droplets": "SPL-2",
    "curled": "KEY-1", "breathe": ["KEY-1", "KEY-8"], "stand_bent": "KEY-5", "press_sides": "KEY-6", "look": "KEY-8",
    "brace": "RUN-1", "windup": "RUN-2", "strike": "KEY-3", "break_out": "KEY-2", "break_land": "KEY-6"}

OBSTACLES = {
    "edgeA": {"kind": "edge", "box": [800, 0, 840, 640], "contacts": {"edge": {"x": 840}}},
    "alley": {"kind": "gap", "box": [840, 0, 1210, 640], "contacts": {"near": {"x": 840}, "far": {"x": 1210}}},
    "ac": {"kind": "ac", "box": [1700, 400, 1830, 490], "contacts": {"top": {"y": 400}}, "occlude": "rect"},
    "ledge": {"kind": "ledge", "box": [2050, 420, 2380, 440], "contacts": {"under": {"y": 440}}, "occlude": "rect"},
    "pipe": {"kind": "pipe", "box": [2700, 0, 2712, 640], "contacts": {"pole": {"x": 2706}}, "occlude": "rect"},
    "beam": {"kind": "beam", "box": [2740, 120, 3300, 135], "contacts": {"beam": {"y": 132}}},
    "door": {"kind": "door", "box": [3700, 300, 3900, 640], "contacts": {"step": {"y": 570}}},
    "pipe2": {"kind": "pipe", "box": [4300, 0, 4310, 640], "contacts": {"pole": {"x": 4305}}},
}

SEQUENCES = {
    "run_RUN": {"drawings": [f"RUN-{k}" for k in range(1, 9)], "cycle": True, "steps": 2},
    "vault_KEY": {"drawings": ["KEY-3", "KEY-4", "KEY-2", "KEY-6"], "keys": {"plant": 1, "push": 2, "land": 3}},
    "leap_KEY": {"drawings": ["KEY-2", "KEY-3", "KEY-5", "KEY-4", "KEY-6", "KEY-8"],
                 "keys": {"takeoff": 0, "land": 4, "recover": 5}},
    "release_KEY": {"drawings": ["KEY-3", "KEY-2", "KEY-5", "KEY-4", "KEY-6"], "keys": {"release": 0, "land": 4}},
    "pole_KEY": {"drawings": ["KEY-3", "KEY-5", "KEY-2", "KEY-4", "KEY-7", "KEY-1", "KEY-8", "KEY-6"],
                 "keys": {"reach": 0, "grip": 1, "release": 2, "land": 7}},
    "step_KEY": {"drawings": ["KEY-2", "KEY-6", "KEY-3", "KEY-5", "KEY-4", "KEY-7", "KEY-1", "KEY-8"],
                 "keys": {"jump": 0, "on": 1, "push": 2, "land": 6, "recover": 7}},
    "slide_KEY": {"drawings": ["KEY-1", "KEY-7", "KEY-8"], "keys": {"in": 0, "glide": 1, "up": 2}},
    "climb_KEY": {"drawings": ["KEY-3", "KEY-5", "KEY-8", "KEY-2"], "keys": {"reach": 0, "grip": 1, "lunge": 3}},
    "burst_KEY": {"drawings": ["KEY-2", "KEY-3", "KEY-5", "KEY-6"], "keys": {"out": 0, "land": 3}},
    "crouch_KEY": {"drawings": ["KEY-6", "KEY-8", "KEY-1"], "keys": {"crouch": 2}}}
MOTION = {"run": "run_RUN", "vault": "vault_KEY", "leap": "leap_KEY", "swing_release": "release_KEY",
          "pole_swing": "pole_KEY", "step_up": "step_KEY", "leap_off": "step_KEY", "slide": "slide_KEY",
          "climb": "climb_KEY", "burst": "burst_KEY", "crouch": "crouch_KEY"}
STRIDE_PHASE = {**{f"RUN-{k}": (k - 1) % 4 for k in range(1, 9)}, "KEY-3": 3, "KEY-2": 2, "KEY-6": 0, "KEY-1": 0,
                "KEY-8": 2}

CHOREO = {
    "street": "main", "seed": 3, "jump": "J2", "primer": "ghosts", "window": ["4.1", "10.4"], "motion": "legacy",
    "segments": [{"from": 2.1, "to": 3.1, "rate": "16ths"},
                 {"from": 3.1, "to": "3.4.4", "rate": "16ths", "camera": {"preset": "stop", "snap": True}},
                 {"from": "3.4.4", "to": 11.1, "rate": "16ths"}],
    "moves": [{"run": {"from": 2.1, "from_x": 200}}, {"crouch": {"at": 3.1, "edge": "edgeA"}},
              {"freeze": {"to": "3.4.4"}}, {"leap": {"gap": "alley", "at": "impact", "land_on": "snare"}}, "run",
              {"vault": {"obstacle": "ac", "plant_on": 5.2}}, {"slide": {"under": "ledge", "at": "5.3.3", "up": 6.1}},
              {"climb": {"pipe": "pipe", "grab": 6.2, "to": "beam"}}, {"hang": {"beam": "beam"}},
              {"swing_release": {"bar": "beam", "release_on": 7.2, "land_on": 7.3}}, "run",
              {"step_up": {"onto": "door", "at": 8.2, "land_on": "8.3.3"}}, "run",
              {"pole_swing": {"pipe": "pipe2", "grab_on": 9.2}}, {"run": {"to": "9.4.4"}},
              {"brake": {"at": 10.1}}, {"walk": {"to": 10.3}}],
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("paintworld") / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"version": 1, "slug": "proj", "fps": 24}))
    _breakdown(root)
    _sheets(root)
    clip = _clip(root)
    sprint = _sprint(root)
    pole = _pole_clip(root)
    tex = _texture(PLATE_W + 6 * int(PLATE_W * (1 - KEEP)), PLATE_H, seed=1)
    step = int(PLATE_W * (1 - KEEP))
    plates = []
    for n in range(7):
        f = root / "boards" / f"P{n + 1}.png"
        cv2.imwrite(str(f), tex[:, n * step:n * step + PLATE_W])
        plates.append(f"boards/P{n + 1}.png")
    sd = root / "paint" / "streets" / "main"
    sd.mkdir(parents=True)
    spec = {"version": 1, "plates": plates[:6], "keep": KEEP, "tile": 1024}
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)
    first = json.loads((sd / "build.json").read_text())
    spec["plates"] = plates
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)  # appended a plate: an incremental build
    (sd / "obstacles.json").write_text(json.dumps({
        "version": 1, "ground": {"wall_base": 640, "feet": 632, "kerb_top": 720, "kerb_bot": 745},
        "obstacles": OBSTACLES, "no_paint": [{"id": "glass", "box": [5200, 0, 6000, 860]}],
        "lights": [{"x": 900, "y": 300, "sigma": 200}]}))
    pd = root / "paint" / "poses"
    pd.mkdir(parents=True)
    (pd / "poses.spec.json").write_text(json.dumps({
        "version": 1, "ref_height": 225, "reference": "RUN",
        "sheets": {"RUN": {"path": "boards/RUN.png", "n": 8, "cols": 4, "kind": "pose", "height": 360},
                   "KEY": {"path": "boards/KEY.png", "n": 8, "cols": 4, "kind": "pose"},
                   "SPL": {"path": "boards/SPL.png", "n": 2, "cols": 2, "kind": "stamp"}},
        "cycles": {"run": [["RUN-1", "RUN-2", "RUN-3", "RUN-4"], ["RUN-5", "RUN-6", "RUN-7", "RUN-8"]],
                   "walk": [["KEY-8"], ["KEY-6"]]},
        "phases": {"run": ["contact", "passing", "toeoff", "flight"]}, "roles": ROLES,
        "video": {"VID": {"path": clip, "key": [172, 203, 236], "height_frames": [0, 8]},
                  "SPR": {"path": sprint, "height_frames": [0, 64]},
                  "POL": {"path": pole, "key": [172, 203, 236], "bridge": True, "ground": "feet",
                          "height_frames": [0, 8], "height_px": 430}},
        "sequences": {**SEQUENCES,
                      "vault_VID": {"video": "VID", "range": [8, 40], "keys": {"plant": 24}, "label": "climb-over"},
                      "run_VID": {"video": "VID", "range": [0, 16], "cycle": True, "steps": 2, "keys": {"contact": 0}},
                      "run_SPR": {"video": "SPR", "range": [16, 31], "measure": [0, 52], "cycle": True, "steps": 2,
                                  "keys": {"contact": 16}},
                      "vault_FAST": {"video": "VID", "range": [8, 40], "keys": {"plant": 24}, "speed": 2.0},
                      "flat_SPR": {"video": "SPR", "range": [0, 63], "keys": {"a": 16, "b": 40}},
                      "pole_POL": {"video": "POL", "range": [0, 39], "keys": {"grab": 20, "land": 34}},
                      "vault_DRAWN": {"drawings": ["RUN-1", "RUN-2", "RUN-3", "KEY-2", "KEY-3", "KEY-4", "KEY-5",
                                                   "KEY-7", "KEY-8", "KEY-6", "RUN-5", "RUN-6"],
                                      "keys": {"takeoff": 3, "plant": 5, "push": 7, "land": 9}}},
        "motion": MOTION, "stride_phase": STRIDE_PHASE}))
    PS.build(root, pd, log=lambda m: None)
    cd = root / "paint" / "choreo"
    cd.mkdir()
    (cd / "t.yaml").write_text(yaml.safe_dump(CHOREO, sort_keys=False))
    return {"root": root, "p": Project.open(root), "tex": tex, "street": sd, "first_build": first}


def _choreo(world, **over):
    return open_choreo(world["p"], "t", over or None)


def _with(world, name, data):
    f = world["root"] / "paint" / "choreo" / f"{name}.yaml"
    f.write_text(yaml.safe_dump(data, sort_keys=False))
    return open_choreo(world["p"], name)


# ---------------------------------------------------------------------------------------------- grid addressing

def test_grid_addresses_are_one_based_bar_beat_sixteenth(world):
    g = SongGrid.from_project(world["p"])
    q, hit = g.resolve(4.2)  # a YAML number is a bar.beat, never seconds
    assert q == 4 * ((4 - 1) * 4 + 1) and hit is None and g.t(q) == pytest.approx(beat_t(4, 2))
    q, _ = g.resolve("4.2.3")
    assert g.t(q) == pytest.approx(beat_t(4, 2) + 2 * BEAT / 4)
    assert g.addr(q) == "4.2.3"
    assert g.resolve(7)[0] == g.resolve("7.1.1")[0] == g.resolve(7.1)[0]
    assert g.t(g.resolve("7.25s")[0]) == pytest.approx(7.25)
    assert g.resolve("4.2.3+1")[0] == q + 1 and g.resolve("4.2.3-0.5")[0] == q - 0.5
    assert g.resolve("+2", ref=40.0)[0] == 42.0
    for bad in ("4.5", "4.2.5", "0.1", "x.y", "4.2.3.1"):
        with pytest.raises(GridError):
            g.resolve(bad)
    with pytest.raises(GridError):
        g.resolve("+2")  # relative without a reference


def test_grid_named_hits_next_nearest_and_thresholds(world):
    g = SongGrid.from_project(world["p"])
    q, hit = g.resolve("impact")
    assert hit == pytest.approx(beat_t(4, 1))
    # the next backbeat snare after bar 5 beat 2: beat 4 (the ghost snare on the "a" of beat 3 is too weak)
    q, hit = g.resolve("snare", ref=g.resolve(5.2)[0])
    assert hit == pytest.approx(beat_t(5, 4))
    assert g.resolve("snare@5.3")[1] in (pytest.approx(beat_t(5, 2)), pytest.approx(beat_t(5, 4)))
    assert g.resolve("snare>5.3")[1] == pytest.approx(beat_t(5, 4))
    assert g.resolve("kick", ref=g.resolve(5.1)[0])[1] == pytest.approx(beat_t(5, 3))
    assert g.resolve("impact-1")[0] == pytest.approx(g.q(beat_t(4, 1)) - 1)
    assert g.resolve("Chorus")[1] == pytest.approx(beat_t(5))
    assert g.resolve("stop")[1] == pytest.approx(beat_t(3, 2))
    g.thresholds["snare"] = 0.4  # ghosts count now
    assert g.resolve("snare", ref=g.resolve(5.3)[0])[1] == pytest.approx(beat_t(5, 3) + 0.75 * BEAT)
    assert rate_of("8ths") == 2 and rate_of("beats") == 4 and rate_of(3) == 3
    with pytest.raises(GridError):
        rate_of("quavers")


# ---------------------------------------------------------------------------------------------- street

def test_street_stitches_plates_into_the_texture(world):
    s = ST.Street(world["street"])
    full = s.window(0, s.W)
    tex = world["tex"][..., ::-1]
    b = json.loads((world["street"] / "build.json").read_text())["strip"]
    x0, y0 = b["crop"][0], b["crop"][1] - ST.OY
    ref = tex[y0:y0 + s.H, x0:x0 + s.W]
    assert full.shape == ref.shape
    assert np.abs(full.astype(int) - ref.astype(int)).mean() < 1.0


def test_street_window_across_a_tile_boundary_equals_the_untiled_strip(world):
    s = ST.Street(world["street"], cache_tiles=2)
    tiles = [cv2.imread(str(world["street"] / "tiles" / f"t{k:04d}.png"))[..., ::-1] for k in range(s.n)]
    untiled = np.concatenate(tiles, axis=1)
    assert untiled.shape[1] == s.W and all(t.shape[1] == 1024 for t in tiles[:-1])
    for x0, x1 in ((1000, 1100), (1023, 1025), (500, 3600), (2048 - 7, 2048 + 900), (s.W - 30, s.W)):
        assert np.array_equal(s.window(x0, x1), untiled[:, x0:x1])
    # outside the street: edge replicated
    assert np.array_equal(s.window(-3, 2)[:, 0], untiled[:, 0])
    # the light maps are per window, yet a window equals the same columns of a wider one
    sc = Scene(world["street"], world["root"] / "paint" / ".cache")
    a = sc.span(1000, 1100)
    b = sc.span(0, 3000)
    for k in ("lin", "relief", "spec", "e_wall", "tint", "occ", "paint"):
        assert np.allclose(getattr(a, k), getattr(b, k)[:, 1000:1100], atol=1e-4), k


def test_street_append_keeps_earlier_tiles(world):
    first = world["first_build"]["strip"]["tiles"]
    now = json.loads((world["street"] / "build.json").read_text())["strip"]["tiles"]
    assert len(now) > len(first)
    plate7_x = json.loads((world["street"] / "build.json").read_text())["strip"]["plates"][6]["offset"][0]
    unchanged = [k for k in first if (int(k) + 1) * 1024 <= plate7_x - 16]
    assert unchanged and all(first[k] == now[k] for k in unchanged)


def test_street_edits_match_the_prototype_cross_fades(tmp_path):
    rng = np.random.default_rng(0)
    strip = rng.integers(0, 255, (40, 3000, 3), np.uint8)
    d = tmp_path / "s"
    tw = ST.TileWriter(d / "strip", 512, 40, "s")
    for a in range(0, 3000, 700):
        tw.push(strip[:, a:a + 700])
    tw.close()
    spec = {"tile": 512, "edit_feather": 96,
            "edits": [{"cut": [600, 900]}, {"insert": 2000, "src": [1500, 1800], "mirror": True, "stretch": 1.5}]}
    info = ST.render_street(d, spec, {"size": [3000, 40]})
    out = np.concatenate([cv2.imread(str(d / "tiles" / f"t{k:04d}.png")) for k in range(len(info["tiles"]))], 1)

    def join(a, b, f=96):  # edit_strip.py's cross-fade
        w = np.linspace(0, 1, f, dtype=np.float32)[None, :, None]
        return np.concatenate([a[:, :-f], a[:, -f:] * (1 - w) + b[:, :f] * w, b[:, f:]], axis=1)
    s = strip.astype(np.float32)
    h = 48
    cut = join(s[:, :600 + h], s[:, 900 - h:])
    at = 2000 - 300
    m = cv2.resize(s[:, 1500 - h:1800 + h][:, ::-1], (int(round((300 + 2 * h) * 1.5)), 40),
                   interpolation=cv2.INTER_CUBIC)
    ref = np.clip(join(join(cut[:, :at + h], m), cut[:, at - h:]) + 0.5, 0, 255).astype(np.uint8)
    assert out.shape == ref.shape
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    pcs = ST.pieces(3000, spec["edits"], 96)
    assert ST.xmap(100, pcs, 96) == 100 and ST.xmap(700, pcs, 96) is None and ST.xmap(950, pcs, 96) == 650


# ---------------------------------------------------------------------------------------------- choreography

def test_choreography_parses_and_validates(world):
    ch = _choreo(world)
    assert not ch.issues, [str(i) for i in ch.issues]
    kinds = [m.kind for m in ch.moves]
    assert kinds[:4] == ["run", "crouch", "freeze", "leap"] and kinds.count("run") == 5
    assert ch.freezes and ch.grid.addr(ch.freezes[0][1]) == "3.4.4"
    assert any(e.layer == "air" for e in ch.events)
    photos = ch.photos()
    assert [p.k for p in photos] == list(range(len(photos)))
    assert all(b.q - a.q == pytest.approx(1) for a, b in zip(photos, photos[1:]))


@pytest.mark.parametrize("change, text", [
    (lambda d: d["moves"].append({"backflip": {}}), "unknown template"),
    (lambda d: d["moves"][5]["vault"].update(obstacle="bin"), "unknown obstacle"),
    (lambda d: d["moves"][5]["vault"].update(poses={"vault_plant": "XX-1"}), "unknown pose"),
    (lambda d: d["segments"][1].update({"to": 4.2}), "overlap"),
    (lambda d: d["segments"][2].update({"from": 4.2}), "gap between"),
    (lambda d: d["moves"][5]["vault"].update(plant_on="5.7"), "does not exist"),
    (lambda d: d.update(primer="lots"), "primer must be"),
])
def test_choreography_errors(world, change, text):
    d = json.loads(json.dumps(CHOREO))
    change(d)
    ch = _with(world, "bad", d)
    assert any(text in i.text and i.level == "error" for i in ch.issues), [str(i) for i in ch.issues]


def test_choreography_warns_on_unknown_arguments_and_off_grid_hits(world):
    d = json.loads(json.dumps(CHOREO))
    d["moves"][5]["vault"]["hands"] = "top"
    d["moves"][6]["slide"]["at"] = "kick@5.3.3"  # nearest kick to 5.3.3 is on beat 3: 2 16ths early
    ch = _with(world, "warn", d)
    assert any("unknown argument" in i.text for i in ch.issues)
    r = check(world["p"], ch)
    assert not [i for i in r["issues"] if i.level == "error"]
    assert r["summary"]["photos"] > 100 and r["summary"]["travel_px"] > 3000


# ---------------------------------------------------------------------------------------------- templates

def _anchor_xy(hist, i, e):
    """Where the event's anchor really is: the placed matte's own anchor (after scale, jitter, rotation)."""
    d, x0, y0 = hist.place(i)
    s = hist._placed[i][0]
    lib = hist.lib
    if e.rot == 0 and not e.mirror:
        ax, ay = lib.anchor(e.pose, e.anchor)
        return x0 + ax * s, y0 + ay * s
    return e.x, e.y


def test_templates_put_anchors_on_their_obstacle_contacts(world):
    ch = _choreo(world)
    hist = History(ch)
    sc = ch.scene
    seen = set()
    for i, e in enumerate(ch.events):
        if not e.contact or e.layer == "stamp":
            continue
        oid, cname, anchor = e.contact[:3]
        rel = e.contact[3] if len(e.contact) > 3 else "on"
        c = sc.obstacles[oid].contacts[cname]
        x, y = _anchor_xy(hist, i, e)
        seen.add(e.tag)
        if rel == "past":
            assert x > c.x, e.tag
        elif cname == "under":  # the slide fits under the underside
            m = ch.lib.m(e.pose)
            s = hist._placed[i][0]
            top = y - (ch.lib.anchor(e.pose, e.anchor)[1] - m["top"][1]) * s
            assert top >= c.y - 3, (e.tag, top, c.y)
        else:
            assert c.dist(x, y) < 14, (e.tag, x, y, c)  # hands grip 15 px (a 450 px figure) below a bar
    assert {"vault.plant", "vault.over", "slide.under", "climb.grip", "climb.lunge", "hang", "step_up.on",
            "pole_swing.grip", "leap.land"} <= seen
    # the take-off is at the gap's near edge, the flight over the gap, the landing on the far wall
    tk = next(e for e in ch.events if e.tag == "leap.takeoff")
    fl = [e for e in ch.events if e.tag == "leap.flight"]
    g = sc.obstacles["alley"]
    assert g.x0 - 100 < tk.x < g.x0 and len(fl) == 4 and all(g.x0 < e.x < g.x1 + 60 for e in fl)
    # timing: the vault's hands on its beat, the leap's flight from the impact to the snare
    plant = next(e for e in ch.events if e.tag == "vault.plant")
    assert ch.grid.addr(plant.q) == "5.2.1"
    assert ch.grid.addr(fl[0].q) == "4.1.1" and ch.grid.addr(next(e for e in ch.events if e.tag == "leap.land").q) == "4.2.1"


def test_hang_and_climb_repeats_are_varied(world):
    ch = _choreo(world)
    hang = [e for e in ch.events if e.tag == "hang"]
    assert len(hang) >= 4 and any(e.mirror for e in hang)
    climb = [e for e in ch.events if e.tag == "climb.grip"]
    assert len({(round(e.rot, 3), round(e.x, 2)) for e in climb}) == len(climb)
    plain = _choreo(world, vary=False)
    assert not any(e.mirror for e in plain.events if e.tag in ("hang", "climb.grip"))


# ---------------------------------------------------------------------------------------------- foot lock

def test_foot_lock_keeps_the_stance_foot_planted():
    plan_x = np.arange(12) * 30.0
    names = ["c", "p", "t", "f"] * 3
    phases = ["contact", "passing", "toeoff", "flight"] * 3
    foot = [40.0, 0.0, -40.0, 0.0] * 3  # the drawings move the torso 40 px over the foot per photo
    qs = [16.0 + k for k in range(12)]
    x = foot_lock(plan_x, names, phases, foot, qs, 1.0)
    for s in (0, 4):  # contact, passing and toe-off of each stride: the foot does not move
        planted = [x[s + k] + foot[s + k] for k in range(3)]
        assert max(planted) - min(planted) < 1e-6
    assert all(np.diff(x) > 0)  # and the runner never moves back
    slid = [plan_x[k] + foot[k] for k in range(3)]
    assert max(slid) - min(slid) == pytest.approx(20)  # without the lock it slides 10 px per photo


def test_runs_slide_less_with_the_foot_lock(world):
    def slide(ch):  # how far the stance foot moves on the wall during the first run (a normal running speed)
        tot = 0.0
        ev = [e for e in ch.events if e.move == 0]
        lib = ch.lib
        for a, b in zip(ev, ev[1:]):
            pa = lib.m(a.pose).get("tags", {}).get("phase")
            pb = lib.m(b.pose).get("tags", {}).get("phase")
            if b.q - a.q == 1 and math.floor(a.q / 4) == math.floor(b.q / 4) and pa in ("contact", "passing") \
                    and pb in ("passing", "toeoff"):
                tot += abs((b.x + lib.foot_offset(b.pose)) - (a.x + lib.foot_offset(a.pose)))
        return tot
    locked, free = slide(_choreo(world)), slide(_choreo(world, foot_lock=False))
    assert free > 10 and locked < 0.25 * free
    # a run too slow for its drawings cannot hold the foot: the check says so
    r = check(world["p"], _choreo(world))
    assert any("too slow" in i.text for i in r["issues"])


# ---------------------------------------------------------------------------------------------- camera

def test_vertical_camera_frames_him_on_its_own(world):
    ch = _choreo(world)
    hist = History(ch)
    photos = ch.photos()
    boxes = hist.figure_boxes()
    cams9 = plan(ch, photos, boxes, "9:16")
    keys = sorted(boxes)
    fb = [boxes[keys[max(int(np.searchsorted(keys, p.q + 1e-6, side="right")) - 1, 0)]] for p in photos]
    st = framing_stats(cams9, fb, True)
    assert st["photos_cut"] == 0 and st["max_outside"] == 0.0
    assert all(abs(c.ch - c.cw * 16 / 9) < 1e-6 and c.ch <= ch.scene.H for c in cams9)
    # it is not a crop of the 16:9: during the leap the 16:9 zooms out, the 9:16 follows the figure
    cams16 = plan(ch, photos, boxes, "16:9")
    flight = [k for k, p in enumerate(photos) if round(p.q, 4) in {round(e.q, 4) for e in ch.events
                                                                     if e.tag == "leap.flight"}]
    assert flight and max(abs(cams9[k].cx - cams16[k].cx) for k in flight) > 1


def test_render_preview_photos_both_aspects(world):
    ch = _choreo(world)
    hist = History(ch)
    jobs, _ = plan_jobs(ch, hist, ["16:9", "9:16"])
    r = Renderer(ch, hist)
    j = next(j for j in jobs if ch.grid.addr(j.q) == "4.1.3")  # mid-flight
    a = r.photo(j.k, j.t, j.q, j.cams["16:9"], (512, 288))
    b = r.photo(j.k, j.t, j.q, j.cams["9:16"], (288, 512))
    assert a.shape == (288, 512, 3) and b.shape == (512, 288, 3)
    assert np.isfinite(a).all() and 0.01 < a.mean() < 0.9
    # the same photo renders the same whatever else was rendered before it
    r2 = Renderer(ch, History(ch))
    assert np.allclose(a, r2.photo(j.k, j.t, j.q, j.cams["16:9"], (512, 288)), atol=1e-5)


# ---------------------------------------------------------------------------------------------- prepending on the left

def _left_world(tmp_path, n_left=2, edits=None):
    """A street of two plates, then the same street with n_left plates prepended (all cut from one texture)."""
    root = tmp_path / "lw"
    (root / "boards").mkdir(parents=True)
    step = int(PLATE_W * (1 - KEEP))
    tex = _texture(PLATE_W + (n_left + 1) * step, PLATE_H, seed=5)
    o = n_left * step  # P1's left edge in the texture
    cv2.imwrite(str(root / "boards" / "P1.png"), tex[:, o:o + PLATE_W])
    cv2.imwrite(str(root / "boards" / "P2.png"), tex[:, o + step:o + step + PLATE_W])
    for k in range(1, n_left + 1):
        lp = tex[:, o - k * step:o - k * step + PLATE_W].astype(np.int16)
        if k == 1:  # a re-rendered overlap is never a pixel copy of the plate it came from
            lp[:, PLATE_W - int(PLATE_W * KEEP):] += 6
        cv2.imwrite(str(root / "boards" / f"L{k}.png"), np.clip(lp, 0, 255).astype(np.uint8))
    sd = root / "paint" / "streets" / "main"
    sd.mkdir(parents=True)
    spec = {"version": 1, "plates": ["boards/P1.png", "boards/P2.png"], "keep": KEEP, "tile": 512}
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)
    before = json.loads((sd / "build.json").read_text())
    shas = {f.name: f.read_bytes() for f in (sd / "tiles").glob("t*.png")}
    spec["left"] = [f"boards/L{k}.png" for k in range(1, n_left + 1)]
    spec["edits"] = edits or []
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)
    return {"root": root, "sd": sd, "tex": tex, "o": o, "before": before, "tiles_before": shas, "step": step}


def test_make_canvas_left_puts_the_previous_plates_left_part_on_the_right(tmp_path):
    img = np.random.default_rng(1).integers(0, 255, (40, 100, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "p.png"), img)
    ST.make_canvas(tmp_path / "p.png", tmp_path / "c.png", 0.4, left=True)
    c = cv2.imread(str(tmp_path / "c.png"))
    assert np.array_equal(c[:, 60:], img[:, :40]) and (c[:, :60] == 128).all()


def test_prepending_keeps_coordinates_and_tiles(tmp_path):
    w = _left_world(tmp_path)
    sd, tex, o = w["sd"], w["tex"], w["o"]
    after = json.loads((sd / "build.json").read_text())
    # right of x 0: the same size; the strip tiles are untouched; the street tiles change only where the first left
    # plate's seam lies inside its overlap with the first plate (the join patch), and the tiles as the right side
    # made them (light_tiles) keep their old hashes, so the light maps stay as they were
    assert after["street"]["size"] == w["before"]["street"]["size"]
    xa, xb = after["strip"]["left"]["patch"]["x"]
    assert xa == 0 and 0 < xb < int(PLATE_W * KEEP)
    seam = after["strip"]["left"]["seams"][0]["x_mean"]
    assert 0.25 * PLATE_W * KEEP < seam < 0.75 * PLATE_W * KEEP  # inside the overlap, away from x 0
    changed = [k for k, v in after["street"]["tiles"].items() if v != w["before"]["street"]["tiles"][k]]
    assert set(changed) <= {str(k) for k in range(xb // 512 + 1)}
    light = {**after["street"]["tiles"], **after["street"]["light_tiles"]}
    assert light == w["before"]["street"]["tiles"]
    s = ST.Street(sd)
    assert all(np.array_equal(s.light_rgb(int(k)), cv2.imread(str(sd / "tiles" / ST.tile_name("t", int(k))))[..., ::-1])
               for k in after["street"]["tiles"] if k not in changed)
    old0 = cv2.imdecode(np.frombuffer(w["tiles_before"]["t0000.png"], np.uint8), cv2.IMREAD_COLOR)
    new0 = cv2.imread(str(sd / "tiles" / "t0000.png"))
    assert np.array_equal(old0[:, xb:], new0[:, xb:]) and not np.array_equal(old0[:, :xb], new0[:, :xb])
    assert np.array_equal(s.light_rgb(0), old0[..., ::-1])
    assert s.X0 == -after["street"]["left"]["size"][0] < -1.5 * w["step"] and s.k_min < 0
    # x 0 is still the first plate's left edge; left of it is the texture left of the first plate
    y0 = after["strip"]["crop"][1] - ST.OY
    full = s.window(s.X0, s.W)[..., ::-1].astype(int)
    ref = tex[y0:y0 + s.H, o + s.X0:o + s.W].astype(int)
    assert full.shape == ref.shape
    assert np.abs(full[:, -s.X0:] - ref[:, -s.X0:]).mean() < 1.5
    assert np.abs(full[:, :-s.X0] - ref[:, :-s.X0]).mean() < 3.0  # the first left plate's overlap is off by 6
    # a window across x 0 and one entirely left of it read the same columns as a wide window
    assert np.array_equal(s.window(-300, 300), s.window(s.X0, s.W)[:, -s.X0 - 300:-s.X0 + 300])
    assert np.array_equal(s.window(s.X0 + 10, s.X0 + 50), s.window(s.X0, s.W)[:, 10:50])
    assert ST.street_x(sd, -100) == -100 and ST.street_x(sd, 200) == 200


def test_edits_left_of_x0_move_only_the_left_side(tmp_path):
    w = _left_world(tmp_path, edits=[{"insert": -400, "src": [-900, -500], "mirror": True}])
    sd = w["sd"]
    after = json.loads((sd / "build.json").read_text())
    assert {k: v for k, v in after["street"]["tiles"].items() if k != "0"} == \
        {k: v for k, v in w["before"]["street"]["tiles"].items() if k != "0"}  # right of x 0: only the join
    lw = after["strip"]["left"]["size"][0]
    assert after["street"]["left"]["size"][0] == lw + 400  # the insert's width (the cross-fades overlap)
    assert ST.street_x(sd, -200) == -200  # right of the insert: as it was
    assert ST.street_x(sd, -700) == -700 - 400  # left of it: 400 px further left
    with pytest.raises(SystemExit):
        ST.load_spec(_spec_with(sd, [{"cut": [-100, 100]}]))  # an edit across x 0 is refused


def _spec_with(sd, edits):
    spec = json.loads((sd / "street.json").read_text())
    spec["edits"] = edits
    d = sd.parent / "bad"
    d.mkdir(exist_ok=True)
    (d / "street.json").write_text(json.dumps(spec))
    return d


def test_scene_left_of_x0_keeps_the_right_sides_maps(tmp_path):
    w = _left_world(tmp_path)
    sd = w["sd"]
    (sd / "obstacles.json").write_text(json.dumps({
        "version": 1, "ground": {"wall_base": 640, "feet": 632, "dy": [[-600, -12], [0, 0]]},
        "obstacles": {"crate": {"kind": "box", "box": [-500, 500, -380, 640], "contacts": {"top": {"y": 500}},
                                "occlude": "rect"}}}))
    sc = Scene(sd, w["root"] / "paint" / ".cache")
    # the light maps right of x 0 are the right side's alone (as before prepending)
    import shutil
    solo = tmp_path / "solo"
    shutil.copytree(sd, solo)
    b = json.loads((solo / "build.json").read_text())
    b["street"].pop("left")
    (solo / "build.json").write_text(json.dumps(b))
    sc_solo = Scene(solo, tmp_path / "cache2")
    a, r = sc.span(0, 600), sc_solo.span(0, 600)
    for k in ("e_wall", "tint", "relief"):
        assert np.allclose(getattr(a, k)[:, 60:], getattr(r, k)[:, 60:], atol=1e-5), k
    # left of x 0: occluders, the ground offset (paint stops at the raised ground line)
    occ, pnt = sc.masks(-520, -360)
    assert occ[560, 40:140].mean() > 0.9 and occ[560, :10].mean() < 0.1
    assert pnt[640 - 12 - 10, 0] > 0.9 and pnt[640 - 12, 0] < 0.5
    assert sc.feet_at(-700) == 620 and sc.feet_at(100) == 632
    out = sc.preview(tmp_path / "masks.jpg")
    assert out.exists()


# ---------------------------------------------------------------------------------------------- song stops pause moves

def test_clock_takes_the_stops_out():
    c = Clock([(36, 48), (60, 64)])
    assert c.fwd(35) == 35 and c.fwd(36) == 36 and c.fwd(40) == 36 and c.fwd(48) == 36 and c.fwd(62) == 48
    assert c.inv(35) == 35 and c.inv(36) == 48 and c.inv(47) == 59 and c.inv(48) == 64
    assert all(c.fwd(c.inv(m)) == m for m in range(30, 70))
    assert c.inside(40) == (36, 48) and c.inside(36) is None and c.inside(48) is None


def test_a_move_across_a_song_stop_holds_through_it(world):
    # the synthetic song stops from bar 3 beat 2 (16th 36) to the impact on bar 4 (16th 48)
    d = {"street": "main", "seed": 3, "primer": "ghosts", "motion": "legacy",
         "segments": [{"from": 2.1, "to": 11.1, "rate": "16ths"}],
         "moves": [{"run": {"from": 2.1, "from_x": 1100, "to": "3.1.2"}}, {"vault": {"obstacle": "ac", "plant_on": "3.1.4"}},
                   "run", {"slide": {"under": "ledge", "at": "5.3.3", "up": 6.1}}]}
    ch = _with(world, "pause", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    assert ch.stops == [(36.0, 48.0)] and (36.0, 48.0) in ch.freezes
    vault = [e for e in ch.events if e.move == 1 and e.layer == "wall"]
    tags = {e.tag: g.addr(e.q) for e in vault}
    assert tags["vault.plant"] == "3.1.4"  # hands planted just before the stop
    assert tags["vault.over"] == "4.1.1"  # the next drawing comes on the stop's end impact
    assert not [e for e in ch.events if 36 < e.q < 48]  # nothing painted while the song stands still
    land = next(e for e in vault if e.tag == "vault.land")
    assert land.q == 48 + 3  # the move's own rhythm carries on after the stop
    # without pause the vault goes on through the stop
    d["pause"] = False
    ch2 = _with(world, "nopause", d)
    assert ch2.stops == [] and any(36 < e.q < 48 for e in ch2.events)


def test_pause_leaves_a_choreography_with_its_own_freeze_alone(world):
    a = _choreo(world)
    d = json.loads(json.dumps(CHOREO))
    d["pause"] = False
    b = _with(world, "p6like", d)
    assert a.stops == []  # the stop is covered by the freeze move
    assert [(e.q, e.pose, e.x, e.y) for e in a.events] == [(e.q, e.pose, e.x, e.y) for e in b.events]


def test_paint_clock_and_droplets_hang_in_a_freeze(world):
    ch = _choreo(world)
    a, b = ch.grid.t(ch.freezes[0][0]), ch.grid.t(ch.freezes[0][1])
    assert ch.paint_clock(b + 0.2) == pytest.approx(ch.paint_clock(a) + 0.2, abs=0.01)
    assert ch.paint_time(a - 0.1, 0.3) == pytest.approx(b - 0.004 + 0.2, abs=0.01)
    assert ch.paint_clock_back(b + 0.1, 0.3) == pytest.approx(a - 0.2, abs=0.01)


# ---------------------------------------------------------------------------------------------- composition

def test_include_joins_segment_files_that_paint_as_they_do_alone(world):
    seg_a = {"street": "main", "seed": 5, "primer": "ghosts", "segments": [{"from": 5.1, "to": 7.1, "rate": "16ths"}],
             "moves": [{"run": {"from": 5.1, "from_x": 1200, "to": "6.4.4"}}]}
    seg_b = {"seed": 9, "primer": "ghosts", "segments": [{"from": 7.1, "to": 9.1, "rate": "16ths", "camera": "close"}],
             "camera": {"lead": 0.05},
             "moves": [{"run": {"from": 7.1, "from_x": 2900, "cycle": 0, "to": "7.4.4"}},
                       {"step_up": {"onto": "door", "at": 8.2, "land_on": "8.3.3"}}]}
    _with(world, "sega", seg_a)
    b = _with(world, "segb", seg_b)
    film = _with(world, "film", {"include": ["sega", "segb"], "window": ["5.1", "9.1"]})
    assert not film.errors, [str(i) for i in film.issues]
    assert [s["name"] for s in film.sources] == ["sega", "segb", "film"]
    assert film.seed == 5 and film.primer == "ghosts"  # settings: the first included file's
    assert film.segments[1].camera == {"lead": 0.05, "preset": "close"}  # each file's camera defaults
    fb = [e for e in film.events if film.moves[e.move].src == 1]
    assert [(e.q, e.pose, round(e.x, 3)) for e in fb] == [(e.q, e.pose, round(e.x, 3)) for e in b.events]
    # the paint of segment b is placed (jitter, scale) exactly as when it renders alone
    hf, hb = History(film), History(b)
    first = film.events.index(fb[0])
    for j in range(len(b.events)):
        assert hf._placed[first + j] == hb._placed[j]
    assert "segb move 2 (step_up)" in [m.where for m in film.moves]


def test_include_cycles_are_refused(world):
    cd = world["root"] / "paint" / "choreo"
    (cd / "cyc1.yaml").write_text(yaml.safe_dump({"include": ["cyc2"]}))
    (cd / "cyc2.yaml").write_text(yaml.safe_dump({"include": ["cyc1"]}))
    from aimv.paint.choreo import ChoreoError
    with pytest.raises(ChoreoError):
        open_choreo(world["p"], "cyc1")


# ---------------------------------------------------------------------------------------------- the box

BOX = {"street": "main", "seed": 4, "primer": "ghosts",
       "segments": [{"from": 1.1, "to": 3.1, "rate": "beats", "camera": {"preset": "box", "snap": True, "push": [2.1, 3.1, 0.9]}},
                    {"from": 3.1, "to": 6.1, "rate": "8ths", "camera": {"preset": "box", "room": {"from": 3.1, "x": 1000}}},
                    {"from": 6.1, "to": 9.1, "rate": "16ths", "camera": "follow"}],
       "moves": [{"box": {"id": "box", "x": 600, "photo": True,
                          "keys": [{"at": 1.2, "w": 200, "h": 260, "thick": 14},
                                   {"at": "1.4.3", "w": 180, "h": 230},
                                   {"at": 3.1, "h": 300, "w": 220, "bulge": 0.15}],
                          "stencils": [{"at": "1.4.3", "text": "KEEP/IT DOWN", "x": "box.x0-60", "y": 420, "h": 24,
                                        "align": "right"}],
                          "cracks": [{"on": "snare", "from": 2.1, "to": 2.4, "at": [690, 500]}]}},
                 {"inside": {"box": "box", "from": 1.2, "to": "4.4.4",
                             "roles": [[1.2, "breathe"], [3.1, "stand_bent", {"fill": 0.95}],
                                       [4.1, "brace", {"align": "right", "crack": "snare", "push": 6}]]}},
                 {"pound": {"box": "box", "from": 5.1, "to": "5.4.3", "every": "8ths", "punch": [0.02, 0.08]}},
                 {"burst": {"box": "box", "at": 6.1, "land_on": 6.3, "land_x": 1100}},
                 {"run": {"to": "7.2.4"}},
                 {"leap_off": {"onto": "door", "at": "7.3.1", "land_on": "7.4.2"}}]}


@pytest.fixture(scope="module")
def boxed_world(world):
    return {"world": world}


@pytest.fixture(scope="module")
def boxed(world):
    return _with(world, "boxed", BOX)


def test_box_keys_inherit_and_step(boxed):
    pr = boxed.props["box"]
    g = boxed.grid
    assert not boxed.errors, [str(i) for i in boxed.issues]
    assert [g.addr(k.q) for k in pr.keys] == ["1.2.1", "1.4.3", "3.1.1"]
    assert [(k.w, k.h, k.thick) for k in pr.keys] == [(200, 260, 14), (180, 230, 14), (220, 300, 14)]
    assert pr.key_at(g.resolve(1.1)[0]) is None and pr.key_at(g.resolve("1.4.3")[0]).w == 180
    assert pr.geometry(0).w == 200  # before it is painted, the camera uses the first key
    k = pr.keys[2]
    assert k.width_at(k.bottom - k.h / 2) > k.width_at(k.bottom - 2) + 50  # the push widens it at its height
    from aimv.paint.props import BoxKey
    hand = BoxKey(0, 0, 1000, 400, 380, 30, 0.04, 1, bulge_at=0.2, bulge_h=0.1)
    assert hand.bow(1000 - 380 + 0.2 * 380) == pytest.approx(16)  # a few percent, where the hands press
    assert hand.bow(1000 - 380 * 0.5) < 0.5 and hand.bow(1000 - 5) < 0.01  # elsewhere the sides stay straight
    assert pr.key_at(pr.shatter.q) is None  # gone once it shatters
    # a photo exactly on each key and stencil (1.4.3 is between the beat photos)
    assert any(abs(p.q - g.resolve("1.4.3")[0]) < 1e-6 for p in boxed.photos())
    m, x0, y0 = PR.outline(k)
    ys, xs = np.nonzero(m > 0.5)
    assert abs((x0 + xs.min()) - (k.box[0] - k.bulge * k.w)) < 12 and abs((y0 + ys.max()) - k.bottom) < 14


def test_inside_fits_the_box_and_cracks_on_the_snares(boxed):
    hist = History(boxed)
    pr = boxed.props["box"]
    ins = [(i, e) for i, e in enumerate(boxed.events) if e.tag == "inside"]
    assert [boxed.lib.m(e.pose)["sheet"] for _, e in ins[:2]] == ["KEY", "KEY"] and ins[0][1].pose != ins[1][1].pose
    for i, e in ins:
        k = pr.geometry(e.q)
        d, x0, y0 = hist.place(i)
        ys, xs = np.nonzero(d > 0.3)
        assert y0 + ys.max() <= k.bottom + 4 and y0 + ys.min() >= k.box[1] - 6, e.pose
        bul = k.bulge * k.w
        assert x0 + xs.min() >= k.box[0] - bul - 6 and x0 + xs.max() <= k.box[2] + bul + 6, e.pose
        assert e.flags.get("trail") == "light"
    # the role schedule, and the crack growths: two snares in bar 2 (from the box), two per bar while bracing
    g = boxed.grid
    assert {boxed.lib.m(e.pose)["sheet"] + str(g.addr(e.q)[:1]) for _, e in ins if g.addr(e.q).startswith("4.")} == {"RUN4"}
    on_brace = [c for c in pr.cracks if g.addr(c.q).startswith("4.")]
    assert len([c for c in pr.cracks if g.addr(c.q).startswith("2.")]) == 2 and len(on_brace) == 2
    assert on_brace[1].size > on_brace[0].size


def test_pound_strikes_on_the_8ths_with_wind_ups_cracks_and_punches(boxed):
    g = boxed.grid
    strikes = [e for e in boxed.events if e.tag == "pound.strike"]
    winds = [e for e in boxed.events if e.tag == "pound.windup"]
    assert len(strikes) == 8 and [g.addr(e.q) for e in strikes[:3]] == ["5.1.1", "5.1.3", "5.2.1"]
    assert all(w.q == s.q - 1 for w, s in zip(winds, strikes))
    photos = {round(p.q, 4) for p in boxed.photos()}
    assert all(round(w.q, 4) in photos for w in winds)  # the wind-ups get their own photos in an 8ths segment
    punches = [boxed.cam_hints[round(e.q, 4)]["punch"] for e in strikes]
    assert punches[0] == pytest.approx(0.02) and punches[-1] == pytest.approx(0.08) and punches == sorted(punches)
    pc = [c for c in boxed.props["box"].cracks if g.addr(c.q).startswith("5.")]
    assert len(pc) == 8 and pc[-1].size > pc[0].size


def test_burst_shatters_the_box_and_the_shards_hang_through_a_freeze(boxed):
    pr = boxed.props["box"]
    g = boxed.grid
    assert g.addr(pr.shatter.q) == "6.1.1"
    fl = [e for e in boxed.events if e.tag == "burst.flight"]
    land = next(e for e in boxed.events if e.tag == "burst.land")
    assert g.addr(fl[0].q) == "6.1.1" and g.addr(land.q) == "6.3.1" and land.x == 1100
    assert min(e.y for e in fl) < fl[0].y - 20  # rising out of the box
    hist = History(boxed)
    kinds = {m.kind for m in hist.marks}
    assert {"box", "chip", "crack", "stencil", "remains", "spatter"} <= kinds
    (t0, pc0, shards), = hist.shards
    assert len(shards) > 20 and any(s.albedo > 0.5 for s in shards) and any(s.albedo < 0.1 for s in shards)
    a, _ = hist.shard_layer(t0 + 0.05, 300, 0, 900, 860)
    assert a.max() > 0.5
    # shards move on the paint clock: a freeze holds them still
    boxed.freezes.append((g.q(t0 + 0.2), g.q(t0 + 0.6)))
    try:
        a1, _ = hist.shard_layer(t0 + 0.25, 300, 0, 900, 860)
        a2, _ = hist.shard_layer(t0 + 0.5, 300, 0, 900, 860)
        assert np.allclose(a1, a2)
    finally:
        boxed.freezes.pop()
    # remains and spatter stay on the wall afterwards; replaced keys and cracks are marked with their end
    box_marks = [m for m in hist.marks if m.kind == "box"]
    assert box_marks[0].end == pytest.approx(g.t(pr.keys[1].q)) and box_marks[-1].end == pytest.approx(t0)


def test_stencils_are_cut_with_bridges_and_laid_out(boxed):
    g = PR.stencil_glyph("O", 120, 10)
    cnts, hier = cv2.findContours((g > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    assert not any(h[3] >= 0 for h in hier[0])  # no counter left floating: the bridges hold it
    plain = PR.stencil_glyph("L", 120, 10)
    cnts, hier = cv2.findContours((plain > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 1  # letters without a counter are not cut
    st = PR.Stencil(0, ["KEEP", "IT DOWN"], 500, 100, 40, "right", 0, 1)
    paint, hard, x0, y0 = PR.stencil_mask(st)
    ys, xs = np.nonzero(hard > 0.5)
    assert abs((x0 + xs.max()) - 500) < 4 and abs((y0 + ys.min()) - 100) < 4
    assert 2 * 40 < ys.max() - ys.min() < 3 * 40  # two lines
    assert (paint > 0.02).sum() > (hard > 0.5).sum() * 1.2  # the overspray halo
    peeled = PR.peel(hard, paint, 12, 3)
    assert peeled.sum() < paint.sum() * 0.9 and PR.peel(hard, paint, 0, 3) is paint
    s = boxed.props["box"].stencils[0]
    assert s.lines == ["KEEP", "IT DOWN"] and s.x == boxed.props["box"].keys[0].box[0] - 60


def test_box_camera_steps_closer_and_makes_room(boxed, boxed_world):
    hist = History(boxed)
    photos = boxed.photos()
    cams = plan(boxed, photos, hist.figure_boxes(), "16:9")
    g = boxed.grid
    pr = boxed.props["box"]
    w = {g.addr(p.q): c for p, c in zip(photos, cams)}
    big, small = w["1.3.1"].cw, w["2.1.1"].cw  # before and after the box shrinks
    k0, k1 = pr.keys[0], pr.keys[1]
    assert small < big and small / big > k1.h / k0.h  # closer, but less than the box shrinks
    assert w["2.4.1"].cw < w["2.1.1"].cw * 0.97  # the push
    assert abs(w["1.3.1"].cx - 600) < 40 and w["3.2.1"].cx > 650  # room for x 1000 from bar 3
    assert w["3.2.1"].cx + w["3.2.1"].cw / 2 > 1000
    # the box and every stencil already painted are inside every box photo in both formats, 5% from the edges
    cams9 = plan(boxed, photos, hist.figure_boxes(), "9:16")
    for p, c16, c9 in zip(photos, cams, cams9):
        if boxed.segments[p.seg].camera.get("preset") != "box":
            continue
        for c in (c16, c9):
            for x0, y0, x1, y1 in pr.regions(p.q):
                assert c.cx - 0.451 * c.cw <= x0 and x1 <= c.cx + 0.451 * c.cw, (g.addr(p.q), c)
                assert c.cy - 0.451 * c.ch <= y0 and y1 <= c.cy + 0.451 * c.ch, (g.addr(p.q), c)
    # a stencil far from the box widens the frame to hold it
    d = json.loads(json.dumps(BOX))
    d["moves"][0]["box"]["stencils"][0].update(x="box.x0-300", y=150)
    far = _with(boxed_world["world"], "boxfar", d)
    cf = plan(far, far.photos(), History(far).figure_boxes(), "16:9")
    wf = {g.addr(p.q): c for p, c in zip(far.photos(), cf)}
    sb = far.props["box"].stencil_box(0)
    assert wf["2.1.1"].cw > w["2.1.1"].cw and wf["2.1.1"].cx - 0.45 * wf["2.1.1"].cw <= sb[0]


def test_leap_off_stands_on_the_top_and_flies_from_it(boxed):
    g = boxed.grid
    door = boxed.scene.obstacles["door"]
    on = next(e for e in boxed.events if e.tag == "leap_off.on")
    fl = [e for e in boxed.events if e.tag == "leap_off.flight"]
    land = next(e for e in boxed.events if e.tag == "leap_off.land")
    assert on.y == door.contact("step").y and door.x0 < on.x < door.x1
    assert fl and fl[0].y < on.y - 100 * boxed.u and g.addr(land.q) == "7.4.2" and land.x > door.x1


def test_stencil_layout_check_reports_overlaps_cuts_and_layers(world):
    from aimv.paint.check import prop_layout
    d = json.loads(json.dumps(BOX))
    d["moves"][0]["box"]["stencils"] = [
        {"at": "1.4.3", "text": "KEEP/IT DOWN", "x": "box.x0-60", "y": 420, "h": 24, "align": "right"},
        {"at": 2.1, "text": "WAIT YOUR TURN", "x": "box.x0-80", "y": 430, "h": 24, "align": "right"},  # on KEEP
        {"at": 2.1, "text": "DON'T", "x": "box.cx", "y": 450, "h": 24, "align": "center"}]  # on the box
    ch = _with(world, "badlayout", d)
    ch.scene.layers.append({"id": "pole", "parallax": 0.0, "x": 0.0,
                            "parts": [{"box": [300, 0, 330, 860]}]})  # a pole through KEEP IT DOWN
    try:
        photos = ch.photos()
        hist = History(ch)
        c16 = plan(ch, photos, hist.figure_boxes(), "16:9")
        c9 = plan(ch, photos, hist.figure_boxes(), "9:16")
        texts = " | ".join(i.text for i in prop_layout(ch, photos, c16, c9))
        assert "'KEEP / IT DOWN' overlaps 'WAIT YOUR TURN'" in texts
        assert "overlaps the box" in texts and "is behind pole" in texts
    finally:
        ch.scene.layers.pop()
    good = _with(world, "goodlayout", BOX)
    photos = good.photos()
    hist = History(good)
    assert prop_layout(good, photos, plan(good, photos, hist.figure_boxes(), "16:9"),
                       plan(good, photos, hist.figure_boxes(), "9:16")) == []


def test_ink_darkens_the_paint_only_when_asked(world):
    d = json.loads(json.dumps(BOX))
    a = _with(world, "ink1", d)
    d["ink"] = 0.7
    b = _with(world, "ink07", d)
    assert a.ink == 1.0 and b.ink == 0.7
    ha, hb = History(a), History(b)
    t = a.grid.t(a.grid.resolve(2.1)[0])
    Pa, Ca, _ = ha.layers(t, 400, 300, 400, 400)
    Pb, Cb, _ = hb.layers(t, 400, 300, 400, 400)
    fig = (Ca > 0.95) & (Pa / np.maximum(Ca, 1e-4) < 0.03)  # where the figure's dark paint covers the wall
    assert fig.sum() > 100
    ratio = float(np.median(Pb[fig] / Pa[fig]))
    assert 0.68 < ratio < 0.75


# ---------------------------------------------------------------------------------------------- motion M0

def _m0(world):
    d = json.loads(json.dumps(CHOREO))
    d["motion"] = "m0"
    return _with(world, "m0", d)


def test_m0_places_the_choreography_and_moves_evenly(world):
    ch = _m0(world)
    assert not ch.errors, [str(i) for i in ch.issues]
    st = motion_stats(ch)
    assert st["backtracks"] == [] and st["cross_sheet_runs"] == []
    assert st["max_run_lurch_ratio"] <= 0.2  # inside a run the travel per photo changes by about 15% at most
    old = motion_stats(_choreo(world))
    assert st["max_lurch_px"] < old["max_lurch_px"] and len(st["lurches"]) < len(old["lurches"])


def test_m0_runs_play_one_cycle_in_order_with_contacts_on_the_beats(world):
    ch = _m0(world)
    seq = SEQUENCES["run_RUN"]["drawings"]
    for mv in ch.moves:
        if mv.kind != "run":
            continue
        ev = [e for e in ch.events if e.move == mv.i and e.tag == "run"]
        idx = [seq.index(e.pose) for e in ev]
        steps = [(b - a) % 8 for a, b in zip(idx, idx[1:])]
        assert all(1 <= d <= 2 for d in steps), (mv.where, idx)  # in order (a drawing passed over now and then)
        assert {ch.lib.m(e.pose)["sheet"] for e in ev} == {"RUN"}
        if len(ev) >= 12:  # long enough to settle on the beat grid: contacts on the beats in its middle
            mid = ev[4:-4]
            assert all((e.pose in ("RUN-1", "RUN-5")) == (int(round(e.q)) % 4 == 0) for e in mid), mv.where


def test_m0_transitions_match_the_stride_phase(world):
    ch = _m0(world)
    moves = [m for m in ch.moves if m.kind != "freeze"]
    for a, b in zip(moves, moves[1:]):
        if a.kind == "run" and b.state.ph0 is not None and b.kind != "run":
            last = [e for e in ch.events if e.move == a.i and e.tag == "run"][-1]
            assert STRIDE_PHASE[last.pose] == (b.state.ph0 - 1) % 4, (a.where, b.where)


def test_m0_moves_play_their_own_drawings_without_repeats(world):
    ch = _m0(world)
    for mv in ch.moves:
        if mv.kind in ("vault", "leap", "step_up", "pole_swing"):
            figs = [e.pose for e in ch.events if e.move == mv.i and e.layer in ("wall", "air")]
            seq = SEQUENCES[MOTION[mv.kind]]["drawings"]
            assert len(figs) == len(set(figs)), (mv.where, figs)
            assert [seq.index(p) for p in figs] == sorted(seq.index(p) for p in figs), (mv.where, figs)


def test_m0_flights_are_eased_arcs(world):
    ch = _m0(world)
    tr = [r for r in figure_track(ch) if ch.moves[r["move"]].kind == "leap"]
    xs = [r["x"] for r in tr]
    d = np.diff(xs)
    assert (d > 0).all()
    fl = [r for r in tr if r["tag"] in ("leap.takeoff", "leap.flight", "leap.land")]
    steps = np.diff([r["x"] for r in fl])
    assert steps[0] > steps[len(steps) // 2] and steps[-1] > steps[len(steps) // 2]  # slower at the top
    ys = [r["y"] for r in fl]
    assert min(ys[1:-1]) < min(ys[0], ys[-1]) - 20  # it rises above both ends


def test_m0_vault_plants_on_its_hit_and_never_steps_back(world):
    ch = _m0(world)
    pl = next(e for e in ch.events if e.tag == "vault.plant")
    assert ch.grid.addr(pl.q) == "5.2.1" and pl.pose == "KEY-4"
    vt = [r for r in figure_track(ch) if ch.moves[r["move"]].kind == "vault"]
    assert all(b["x"] > a["x"] for a, b in zip(vt, vt[1:]))


def test_check_warns_on_legacy_motion(world):
    r = check(world["p"], _choreo(world))
    texts = " | ".join(i.text for i in r["issues"])
    assert "lurches" in texts and "repeats" in texts
    assert "motion:" in " ".join(r["lines"])


def test_legacy_motion_is_unchanged_by_m0(world):
    a = _choreo(world)
    d = json.loads(json.dumps(CHOREO))
    b = _with(world, "legacy2", d)
    assert [(e.q, e.pose, e.x, e.y) for e in a.events] == [(e.q, e.pose, e.x, e.y) for e in b.events]
    assert a.motion == "legacy" and _m0(world).motion == "m0"


def test_gaps_ease_between_neighbour_steps():
    from aimv.paint.motion import gaps, pick
    g = gaps(20, 2000.0, 160.0, 60.0)
    assert g.sum() == pytest.approx(2000.0) and (g > 0).all()
    assert np.abs(np.diff(g)).max() <= 0.2 * 100  # about 15% of the middle speed per photo
    assert pick(["a", "b", "c"], 3) == ["a", "b", "c"] and pick(["a", "b", "c", "d", "e"], 3) == ["a", "c", "e"]
    assert pick(["a", "b"], 3) == ["a", "a", "b"]


# ---------------------------------------------------------------------------------------------- sequences (part 2)

def test_sheets_split_in_reading_order_over_three_rows():
    img = np.full((900, 1200, 3), 255, np.uint8)
    for k in range(12):  # three rows of four, a figure's size grows with its number
        r, c_ = divmod(k, 4)
        cv2.circle(img, (150 + 300 * c_, 150 + 300 * r), 72 + 6 * k, (0, 0, 0), -1)
    parts = PS.split(img, 12, 4, "pose", "T")
    sizes = [int((p > 0.5).sum()) for p in parts]
    assert sizes == sorted(sizes)


def test_video_sprites_are_keyed_matted_and_deduplicated(world):
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    v = lib.videos["VID"]
    assert v["fps"] == pytest.approx(24) and v["block"] == pytest.approx(list(BLOCK), abs=4)
    assert v["ground"] == pytest.approx(BLOCK[3], abs=4)
    fr = v["frames"]
    assert len(fr) == 48 - 8  # the doubled frames are gone
    for r in fr:
        m = lib.m(r["name"])
        ox = m["video"]["ox"]
        assert ox > 90 + 22 * r["f"] - 60  # the far scribble is not part of him
        assert m["bbox"][2] - m["bbox"][0] < 330  # nor the faint one at his feet
    on = [r for r in fr if r["f"] in VID_ON]
    assert all(abs(r["low"] - BLOCK[1]) < 30 for r in on)  # his feet on the block (keyed out, not in the matte)
    assert all(r["low"] > BLOCK[3] - 40 for r in fr if r["f"] < 18)  # running on the ground (feet up in a stride)


def test_video_vault_plays_the_clip_with_its_key_on_the_hit_and_on_the_top(world):
    d = json.loads(json.dumps(BOX))
    d.update(motion="m0", segments=[{"from": 4.1, "to": 7.1, "rate": "16ths"}])
    d["moves"] = [{"run": {"from": 4.1, "sequence": "run_VID"}},
                  {"vault": {"obstacle": "ac", "plant_on": 5.2, "sequence": "vault_VID"}},
                  {"run": {"to": "6.4.4"}}]
    ch = _with(world, "vvault", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = [e for e in ch.events if ch.moves[e.move].kind == "vault"]
    plant = next(e for e in ev if e.tag == "vault.plant")
    assert g.addr(plant.q) == "5.2.1" and plant.pose == "VID-024"
    # the clip plays at its own speed: 24 fps against 16ths at 120 BPM (8 a second): 3 frames a photo
    fs = [int(e.pose.split("-")[1]) for e in sorted(ev, key=lambda e: e.q)]
    assert all(2 <= b - a <= 4 for a, b in zip(fs, fs[1:]))
    # on the block his lowest point is on the obstacle's top: lifted by what the obstacle has over the scaled
    # block, so the clip's own offset from the block's top stays (the synthetic feet reach a little into it)
    hist = History(ch)
    o = ch.scene.obstacles["ac"]
    v = ch.lib.videos["VID"]
    sv = ch.lib.m("VID-000")["scale"]
    by_name = {r["name"]: r for r in v["frames"]}
    for i, e in enumerate(ch.events):
        if e in ev and int(e.pose.split("-")[1]) in VID_ON:
            dm, x0, y0 = hist.place(i)
            ys, _ = np.nonzero(dm > 0.5)
            want = o.contact("top").y + (by_name[e.pose]["low"] - v["block"][1]) * sv
            assert y0 + np.percentile(ys, 99.6) <= want + 8, e.pose  # (lifted more where it had to clear the AC)
    xs = [r["x"] for r in figure_track(ch) if ch.moves[r["move"]].kind == "vault"]
    assert xs[0] < o.x0 < o.x1 < xs[-1]


def test_sprint_stride_is_measured_from_the_planted_feet(world):
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    sq = lib.sequence("run_SPR")
    assert sq["step"] == pytest.approx(SPR_STEP, abs=3) and sq["period"] == pytest.approx(2 * SPR_STEP, abs=6)
    assert sq["period_f"] == 2 * SPR_PER
    assert all(d == pytest.approx(SPR_STEP, abs=3) for d in sq["steps_measured"])
    # the doubled frames (with noise) are gone, and every frame's place in the cycle moves on
    assert len(sq["frames"]) < 16 and all(b > a for a, b in zip(sq["travel"], sq["travel"][1:]))


def test_video_run_is_stride_true(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"run": {"from": 4.1, "from_x": 1300, "to": "5.4.4", "sequence": "run_SPR"}}]}
    ch = _with(world, "vrun", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ev = [e for e in ch.events if e.tag == "run"]
    assert len(ev) == 32
    names = [e.pose for e in ev]
    assert names[:8] == names[8:16] and names[0] == "SPR-016"  # 8 photos a cycle, the touch-down on the bar line
    sv = ch.lib.m("SPR-016")["scale"]
    d_ = np.diff([r["x"] for r in figure_track(ch)])
    # a step on the wall is the clip's step (one step per beat, 4 photos), travelled evenly
    assert d_ == pytest.approx(np.full(31, SPR_STEP * sv / 4), abs=0.01)
    note = ch.notes["move 1 (run)"]
    assert note["step_px"] == pytest.approx(SPR_STEP * sv, abs=4) and note["foot_slip_max_px"] < 0.2 * SPR_STEP * sv
    # stance photos of one step keep the foot where it landed
    hist = History(ch)
    feet = {}
    for i, e in enumerate(ch.events):
        if e.tag != "run":
            continue
        dm, x0, y0 = hist.place(i)
        ys, xs = np.nonzero(dm > 0.5)
        low = ys > ys.max() - 6
        if ys.max() + y0 > ch.scene.feet - 6:  # on the ground
            feet.setdefault(int(e.q) // 4, []).append(x0 + xs[low].mean())
    spreads = [max(v) - min(v) for v in feet.values() if len(v) > 1]
    assert spreads and max(spreads) < 0.2 * SPR_STEP * sv


def test_video_vault_speed_compresses_the_clip(world):
    def frames(speed):
        d = json.loads(json.dumps(BOX))
        d.update(motion="m0", segments=[{"from": 4.1, "to": 7.1, "rate": "16ths"}])
        arg = {"obstacle": "ac", "plant_on": 5.2, "sequence": "vault_FAST"}
        if speed is not None:
            arg["speed"] = speed
        d["moves"] = [{"run": {"from": 4.1, "sequence": "run_SPR"}}, {"vault": arg}, {"run": {"to": "6.4.4"}}]
        ch = _with(world, f"vfast{speed}", d)
        assert not ch.errors, [str(i) for i in ch.issues]
        ev = sorted([e for e in ch.events if ch.moves[e.move].kind == "vault"], key=lambda e: e.q)
        assert next(e for e in ev if e.tag == "vault.plant").pose == "VID-024"
        return [int(e.pose.split("-")[1]) for e in ev], ch
    fast, ch = frames(None)  # the sequence's own speed: 2
    slow, _ = frames(1.0)
    # 24 fps, 8 photos a second: 3 frames a photo at speed 1, 6 at speed 2 (doubled frames are gone)
    assert np.median(np.diff(slow)) == pytest.approx(3, abs=1) and np.median(np.diff(fast)) == pytest.approx(6, abs=1)
    assert len(fast) < len(slow)
    xs = [r["x"] for r in figure_track(ch) if ch.moves[r["move"]].kind == "vault"]
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_follow_camera_keeps_up_with_a_fast_run(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"run": {"from": 4.1, "from_x": 300, "to": "6.4.4", "speed": 140}}]}
    lag = {}
    for keep in (True, False):
        d["segments"][0]["camera"] = {"preset": "follow", "keep_up": keep, "band": False}
        ch = _with(world, f"keep{keep}", d)
        photos = ch.photos()
        boxes = History(ch).figure_boxes()
        cams = plan(ch, photos, boxes, "16:9")
        tr = {round(r["q"], 4): r["x"] for r in figure_track(ch)}
        lead = 0.07 * cams[20].cw
        lag[keep] = np.mean([tr[round(p.q, 4)] + lead - c.cx for p, c in zip(photos[16:30], cams[16:30])])
    assert abs(lag[True]) < 15 and lag[False] > 60  # 140 px a photo: about 86 px behind without keeping up


def test_drawn_sequence_runs_in_flies_plants_and_strides_out(world):
    d = json.loads(json.dumps(CHOREO))
    d["motion"] = "m0"
    d["moves"] = [{"run": {"from": 2.1, "from_x": 900, "to": "4.4.4"}},
                  {"vault": {"obstacle": "ac", "plant_on": 5.2, "sequence": "vault_DRAWN"}}, {"run": {"to": "6.4.4"}}]
    d["segments"] = [{"from": 2.1, "to": 7.1, "rate": "16ths"}]
    ch = _with(world, "drawn", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if ch.moves[e.move].kind == "vault" and e.layer == "wall"], key=lambda e: e.q)
    assert [e.pose for e in ev] == SEQUENCES_DRAWN
    p = next(e for e in ev if e.tag == "vault.plant")
    assert g.addr(p.q) == "5.2.1" and [round(e.q - p.q) for e in ev] == list(range(-5, 7))
    o = ch.scene.obstacles["ac"]
    runin = [e for e in ev if e.tag in ("vault.runin", "vault.takeoff")]
    assert len(runin) == 4 and all(e.y == ch.scene.feet for e in runin)
    land = next(e for e in ev if e.tag == "vault.land")
    assert land.x > o.x1 and all(e.x > land.x for e in ev if e.tag == "vault.out")
    xs = [r["x"] for r in figure_track(ch) if ch.moves[r["move"]].kind == "vault"]
    assert all(b > a for a, b in zip(xs, xs[1:]))


SEQUENCES_DRAWN = ["RUN-1", "RUN-2", "RUN-3", "KEY-2", "KEY-3", "KEY-4", "KEY-5", "KEY-7", "KEY-8", "KEY-6", "RUN-5",
                   "RUN-6"]


# ---------------------------------------------------------------------------------------------- solid obstacles, QA

@pytest.fixture(scope="module")
def tall(world):
    """A second street on the same tiles with a tall cabinet (taller than him, standing on the ground) and a thin
    pipe, for the obstacle QA."""
    import shutil
    src = world["street"]
    d = src.parent / "tall"
    d.mkdir(exist_ok=True)
    for f in ("street.json", "build.json"):
        shutil.copy(src / f, d / f)
    if not (d / "tiles").exists():
        shutil.copytree(src / "tiles", d / "tiles")
    (d / "obstacles.json").write_text(json.dumps({
        "version": 1, "ground": {"wall_base": 640, "feet": 632, "kerb_top": 720, "kerb_bot": 745},
        "obstacles": {
            "cab": {"kind": "cabinet", "box": [1700, 330, 1860, 632], "contacts": {"top": {"y": 330}},
                    "occlude": [{"box": [1700, 330, 1860, 632], "method": "rect"},
                                {"box": [1770, 200, 1780, 330], "method": "rect", "solid": False}]},
            "pipe": {"kind": "pipe", "box": [1100, 0, 1130, 640], "contacts": {"pole": {"x": 1115}},
                     "occlude": "rect"}}}))
    return d


def _tall_choreo(world, name, moves, segs=None):
    d = {"street": "tall", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": segs or [{"from": 4.1, "to": 7.1, "rate": "16ths"}], "moves": moves}
    return _with(world, name, d)


def test_obstacles_are_solid_or_thin_by_kind_and_part(world, tall):
    sc = Scene(tall, world["root"] / "paint" / ".cache")
    assert sc.obstacles["cab"].solid and not sc.obstacles["pipe"].solid
    solid, touch = sc.occluders(1690, 1870, 150, 640, solid=True)
    thin, _ = sc.occluders(1690, 1870, 150, 640, solid=False)
    assert solid[250 - 150, 1775 - 1690] == 0 and thin[250 - 150, 1775 - 1690] == 1  # the part said thin
    assert solid[400 - 150, 1780 - 1690] == 1 and touch[335 - 150, 1780 - 1690] == 1 and touch[400 - 150, 1780 - 1690] == 0
    w_ = Scene(world["street"], world["root"] / "paint" / ".cache")
    assert w_.obstacles["ac"].solid and not w_.obstacles["pipe"].solid


def test_qa_flags_hidden_hands_and_heads_and_feet(world, tall, tall_hist=None):
    from aimv.paint import qa as QA
    moves = [{"pose": {"at": 4.1, "pose": "KEY-3", "x": 1650, "anchor": "foot"}},  # reaching into the cabinet
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 1115}},  # running past the thin pipe: his head behind it
             {"pose": {"at": 4.3, "pose": "RUN-2", "x": 600, "y": "feet+40"}},  # feet under the ground line
             {"pose": {"at": 4.4, "pose": "KEY-3", "x": 1650, "behind": "cab"}}]  # allowed
    ch = _tall_choreo(world, "qa1", moves, [{"from": 4.1, "to": 5.1, "rate": "16ths"}])
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    photos = ch.photos()
    cams = {a: plan(ch, photos, hist.figure_boxes(), a) for a in ("16:9", "9:16")}
    res = QA.run(ch, hist, photos, cams)
    by = {}
    for f in res["flags"]:
        by.setdefault(f["addr"], []).append((f["rule"], f["level"]))
    assert ("solid", "error") in by["4.1.1"]
    assert all(lv != "error" for r, lv in by.get("4.2.1", [])) and ("head", "warn") in by["4.2.1"]
    assert ("ground", "error") in by["4.3.1"]
    assert "4.4.1" not in by or all(r != "solid" for r, _ in by["4.4.1"])
    assert res["errors"] >= 2 and res["masks"]
    assert "QA FAILED" in QA.banner(res) and "QA PASSED" in QA.banner({"errors": 0, "warnings": 1})


def test_video_vault_clears_a_tall_solid_obstacle(world, tall):
    from aimv.paint import qa as QA
    moves = [{"run": {"from": 4.1, "sequence": "run_SPR"}},
             {"vault": {"obstacle": "cab", "plant_on": 5.2, "sequence": "vault_FAST"}},
             {"run": {"to": "6.4.4", "sequence": "run_SPR"}}]
    ch = _tall_choreo(world, "tallvault", moves)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    photos = ch.photos()
    cams = {a: plan(ch, photos, hist.figure_boxes(), a) for a in ("16:9", "9:16")}
    res = QA.run(ch, hist, photos, cams)
    bad = [f for f in res["flags"] if f["rule"] in ("solid", "head", "backtrack") and f["level"] == "error"]
    assert not bad, bad
    # his weight lands on the top: on the plant photo his low point is on the cabinet's top (the touch band)
    i = next(i for i, e in enumerate(ch.events) if e.tag == "vault.plant")
    dm, x0, y0 = hist.place(i)
    ys = np.nonzero(dm > 0.5)[0]
    assert abs((y0 + np.percentile(ys, 99.6)) - 330) < 16
    note = ch.notes["move 2 (vault)"]
    assert note["lift_on_top_px"] > 50 and note["lifts"][0] == 0.0  # lifted on the top, not on the ground
    on = [k for k, v in enumerate(note["lifts"]) if v >= note["lift_on_top_px"] - 1]
    up = note["lifts"][:on[0] + 1]
    assert all(b >= a - 1e-6 for a, b in zip(up, up[1:]))  # he rises and does not sink on the way up


def test_clipframes_size_the_block_like_the_obstacle(world, tall):
    from aimv.paint import clipframes
    from aimv.paint.poses import PoseLib
    lib = PoseLib(world["root"] / "paint" / "poses")
    sc = Scene(tall, world["root"] / "paint" / ".cache")
    out = world["root"] / "paint" / "clipframes" / "cab"
    info = clipframes.make(lib, sc, "cab", "vault", out, "run_RUN")
    img = cv2.imread(info["frames"]["start"])[..., ::-1]
    bx0, by0, bx1, by1 = info["block"]
    m = lib.m("RUN-1")
    fh = (m["bbox"][3] - m["bbox"][1]) * m["scale"]
    assert (by1 - by0) / 430 == pytest.approx((632 - 330) / fh, rel=0.02) and by1 == 900
    assert (bx1 - bx0) / 430 == pytest.approx(160 / fh, rel=0.02)
    assert np.abs(img[(by0 + by1) // 2, (bx0 + bx1) // 2].astype(int) - np.array(clipframes.KEY)).max() < 3
    dark = img.mean(2) < 90
    ys, xs = np.nonzero(dark)
    assert xs.max() < bx0 and abs(ys.max() - 900) < 6 and 380 < ys.max() - ys.min() < 470  # him, before it
    end = cv2.imread(info["frames"]["end"])[..., ::-1]
    assert np.nonzero(end.mean(2) < 90)[1].min() > bx1
    # a box hung on the wall (the AC on the first street) floats at its height
    w_ = Scene(world["street"], world["root"] / "paint" / ".cache")
    info2 = clipframes.make(lib, w_, "ac", "vault", world["root"] / "paint" / "clipframes" / "ac", "run_RUN")
    assert info2["block"][3] < 900 - 50


def test_render_writes_qa_sheets_and_holds_back_review_copies(world, tall):
    from aimv.paint.render import render
    moves = [{"pose": {"at": 4.1, "pose": "KEY-3", "x": 1650, "anchor": "foot"}},
             {"pose": {"at": 4.2, "pose": "RUN-1", "x": 1300}}]
    ch = _tall_choreo(world, "qarender", moves, [{"from": 4.1, "to": 4.3, "rate": "16ths"}])
    out = world["root"] / "paint" / "renders"
    st = render(world["p"], ch, out, aspects=("16:9",), preview=True, workers=1, review=True, name="qar",
                log=lambda m: None)
    assert st["qa"]["errors"] >= 1 and "review" not in st
    qa = json.loads((out / "qar-qa.json").read_text())
    assert qa["errors"] == st["qa"]["errors"] and any(f["rule"] == "solid" for f in qa["flags"])
    assert (out / "qar-qa.jpg").exists() and (out / "qar-moves.jpg").exists()
    st2 = render(world["p"], ch, out, aspects=("16:9",), preview=True, workers=1, review=True, force=True,
                 name="qar2", log=lambda m: None)
    assert st2["review"] and Path(st2["review"][0]).exists()


# ---------------------------------------------------------------------------------------------- segment 1 v3

def test_street_window_and_patch_paste_an_aligned_edit_where_it_changed(world):
    import shutil
    d = world["street"].parent / "patched"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(world["street"], d)
    out = d / "patches" / "w.png"
    info = ST.export_window(d, 1500, 2600, out, world["root"])
    H = ST.Street(d).H
    assert info["window"] == [1500, 2600] and info["size"] == [1100, H] and info["edit"]["window"] == [1500, 2600]
    assert json.loads(out.with_suffix(".json").read_text())["edit"]["patch"] == "paint/streets/patched/patches/w-edit.png"
    win = cv2.imread(str(out))
    assert win.shape[:2] == (H, 1100)
    # the edit: a centred crop (as an image model's 4:3 crop), upscaled, with a dark box painted on it
    ed = cv2.resize(win[:, 50:1050], None, fx=1.07, fy=1.07, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(ed, (int(450 * 1.07), int(300 * 1.07)), (int(590 * 1.07), int(460 * 1.07)), (20, 20, 20), -1)
    cv2.imwrite(str(d / "patches" / "w-edit.png"), ed)
    spec = json.loads((d / "street.json").read_text())
    spec["edits"] = list(spec.get("edits", [])) + [info["edit"]]
    (d / "street.json").write_text(json.dumps(spec))
    ST.build(world["root"], d, log=lambda m: None)
    b = json.loads((d / "build.json").read_text())["street"]
    pt = b["patches"][0]
    assert pt["inliers"] >= 12 and pt["median_px"] < 2 and pt["scale"] == pytest.approx(1 / 1.07, abs=0.01)
    cx0, cy0, cx1, cy1 = pt["changed"]
    assert abs(cx0 - 2000) < 30 and abs(cx1 - 2140) < 30 and abs(cy0 - 300) < 30 and abs(cy1 - 460) < 30
    new, old = ST.Street(d).window(1500, 2600).astype(int), ST.Street(world["street"]).window(1500, 2600).astype(int)
    assert new[380, 570].max() < 60  # the box is in the street
    diff = np.abs(new - old).max(2)
    assert diff[:, :380].max() <= 3 and diff[:, 760:].max() <= 3  # nothing else changed (the texture is no box)
    assert b["tiles"]["0"] == json.loads((world["street"] / "build.json").read_text())["street"]["tiles"]["0"]


def test_a_patch_with_align_none_is_pasted_where_it_is(world):
    import shutil
    d = world["street"].parent / "exact"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(world["street"], d)
    st = ST.Street(d)
    win = cv2.cvtColor(st.window(1500, 2600), cv2.COLOR_RGB2BGR)
    ed = np.full_like(win, 128)  # nothing like the street (SIFT would find nothing): only align none can place it
    ed[300:302, 600:700] = (255, 255, 255)  # a 2 px line, to see it land to the pixel
    (d / "patches").mkdir(exist_ok=True)
    cv2.imwrite(str(d / "patches" / "exact.png"), ed)
    spec = json.loads((d / "street.json").read_text())
    spec["edits"] = list(spec.get("edits", [])) + [{"patch": "paint/streets/exact/patches/exact.png",
                                                    "window": [1500, 2600], "region": [2050, 280, 2250, 320],
                                                    "mask": "full", "feather": 2, "thresh": 1.0, "align": "none"}]
    (d / "street.json").write_text(json.dumps(spec))
    ST.build(world["root"], d, log=lambda m: None)
    pt = json.loads((d / "build.json").read_text())["street"]["patches"][-1]
    assert pt["inliers"] == 0 and pt["scale"] == 1.0
    new = ST.Street(d).window(1500, 2600).astype(int)
    assert new[300:302, 2110 - 1500:2190 - 1500].min() > 240  # the line, to the pixel
    assert new[296:298, 2110 - 1500:2190 - 1500].max() < 250 and new[304:306, 2110 - 1500:2190 - 1500].max() < 250
    spec["edits"][-1]["align"] = "nope"
    (d / "street.json").write_text(json.dumps(spec))
    with pytest.raises(SystemExit):
        ST.build(world["root"], d, log=lambda m: None)


def test_clipframes_draw_poles_bars_rows_extra_shapes_and_hanging_starts(world):
    from aimv.paint import clipframes
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    sc = Scene(world["street"], world["root"] / "paint" / ".cache")
    out = world["root"] / "paint" / "clipframes"
    pole = clipframes.make(lib, sc, "pipe", "pole_swing", out / "pipe", "run_RUN")
    assert pole["shape"] == "pole" and pole["block"][1] == 0 and pole["block"][3] == 900
    bar = clipframes.make(lib, sc, "beam", "hang_release", out / "beam", "run_RUN", (), "hang")
    k = bar["street_to_frame"]
    bx0, by0, bx1, by1 = bar["block"]
    assert bar["shape"] == "bar" and (by0 + by1) / 2 == pytest.approx(900 - (632 - 132) * k, abs=2)
    img = cv2.imread(bar["frames"]["start"])[..., ::-1]
    ys, xs = np.nonzero(img.mean(2) < 90)
    assert abs(ys.min() - (by0 + by1) / 2) < 25 and bx0 < xs.min() < bx1  # hanging from it by the hands
    climb = clipframes.make(lib, sc, "pipe", "climb", out / "climb", "run_RUN", ("beam",), None, "hang")
    assert len(climb["shapes"]) == 2
    beam = next(r for r in climb["shapes"] if r[2] - r[0] > 4 * (r[3] - r[1]))
    assert beam[0] == pytest.approx(clipframes.BLOCK_CX + (2740 - 2706) * climb["street_to_frame"], abs=2)
    wall = clipframes.make(lib, sc, "wall", "brake_walk", out / "wall", "run_RUN", (), None, "KEY-8")
    assert wall["shapes"] == [] and wall["figures"]["end"] == "KEY-8" and Path(wall["frames"]["end"]).exists()


def test_video_pieces_split_by_a_pole_join_and_the_pole_sets_his_place(world):
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    v = lib.videos["POL"]
    assert len(v["shapes"]) == 1 and abs(v["shapes"][0][0] - POLE[0]) <= 4 and abs(v["shapes"][0][2] - POLE[1]) <= 4
    both = 0
    for r in v["frames"]:
        cols = np.nonzero((lib.sprite(r["name"]) > 0.3).any(0))[0] + lib.m(r["name"])["video"]["ox"]
        both += cols.min() < POLE[0] - 20 and cols.max() > POLE[1] + 20
    assert both >= 3  # the pieces on either side of the pole stay one figure
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"run": {"from": 4.1, "from_x": 3500, "sequence": "run_SPR"}},
                   {"pole_swing": {"pipe": "pipe2", "sequence": "pole_POL", "land_on": 5.3}},
                   {"run": {"to": "6.4.4", "sequence": "run_SPR"}}]}
    ch = _with(world, "vpole", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    land = next(e for e in ch.events if e.tag == "pole_swing.land")
    assert ch.grid.addr(land.q) == "5.3.1" and land.pose == "POL-034"
    r = next(r for r in v["frames"] if r["name"] == land.pose)
    sv = lib.m(land.pose)["scale"]
    assert land.x == pytest.approx(4305 + (r["x"] - (v["shapes"][0][0] + v["shapes"][0][2]) / 2) * sv, abs=1)


def test_clip_template_pins_keys_in_time_and_place_and_its_low_point(world):
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    seg = [{"from": 4.1, "to": 7.1, "rate": "16ths"}]
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": seg,
         "moves": [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.2, "b": 5.2}, "x": {"a": 1000, "b": 2200},
                             "bottom": {"b": 500}}}]}
    ch = _with(world, "vclip", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ev = sorted([e for e in ch.events if e.layer == "wall"], key=lambda e: e.q)
    a = next(e for e in ev if e.tag == "clip.a")
    b = next(e for e in ev if e.tag == "clip.b")
    assert ch.grid.addr(a.q) == "4.2.1" and ch.grid.addr(b.q) == "5.2.1"
    assert a.x == pytest.approx(1000) and b.x == pytest.approx(2200)
    assert all(y >= x - 1e-6 for x, y in zip([e.x for e in ev], [e.x for e in ev][1:]))
    dm, x0, y0 = History(ch).place(ch.events.index(b))
    assert abs(y0 + np.nonzero(dm > 0.4)[0].max() - 500) < 8
    d["moves"] = [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.2, "b": 5.2}, "x": {"a": 1000},
                            "stretch": ["a", "b", 2.0]}}]
    ch = _with(world, "vclip2", d)
    a = next(e for e in ch.events if e.tag == "clip.a")
    b = next(e for e in ch.events if e.tag == "clip.b")
    tr = {r["name"]: r for r in lib.videos["SPR"]["frames"]}
    sv = lib.m(a.pose)["scale"]
    assert b.x - a.x == pytest.approx(2 * (tr[b.pose]["x"] - tr[a.pose]["x"]) * sv, rel=0.02)


def test_video_moves_pause_through_a_song_stop(world):
    # the world's stop runs from 3.2 to 4.1: a vault keyed on the photo before it picks up after it where it held
    d = {"street": "main", "seed": 3, "motion": "m0", "pause": True,
         "segments": [{"from": 2.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"run": {"from": 2.1, "from_x": 1150, "sequence": "run_SPR"}},
                   {"vault": {"obstacle": "ac", "sequence": "vault_FAST", "plant_on": "3.1.4"}},
                   {"run": {"to": "4.4.4", "sequence": "run_SPR"}}]}
    ch = _with(world, "vpause", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ev = sorted([e for e in ch.events if ch.moves[e.move].kind == "vault"], key=lambda e: e.q)
    fs = [int(e.pose.split("-")[1]) for e in ev]
    assert any(e.q >= ch.grid.resolve(4.1)[0] - 1e-6 for e in ev)  # it goes on after the stop
    assert all(0 < b - a <= 8 for a, b in zip(fs, fs[1:]))  # no clip time lost in the stop


def test_video_run_fits_its_stride_to_reach_the_next_move(world):
    seg = [{"from": 4.1, "to": 7.1, "rate": "16ths"}]
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": seg,
            "moves": [{"run": {"from": 4.1, "from_x": 1000, "to": "4.4.4", "sequence": "run_SPR"}}]}
    ch = _with(world, "vfit0", base)
    v = ch.notes["move 1 (run)"]["px_per_photo"]
    target = 1000 + 1.06 * v * 16  # 15 gaps to the run's last photo and one to the next move: 6% further
    d = dict(base, moves=[{"run": {"from": 4.1, "from_x": 1000, "sequence": "run_SPR"}},
                          {"clip": {"sequence": "flat_SPR", "keys": {"first": "5.1.1"}, "x": {"first": target}}}])
    ch = _with(world, "vfit1", d)
    assert ch.notes["move 1 (run)"]["stride_fit"] == pytest.approx(1.06, abs=0.01)
    assert not [i for i in ch.issues if "stride-true run ends" in i.text]


def test_canon_errors_keep_clip_drawings_out_and_fail_the_qa(world, tall):
    from aimv.paint import qa as QA
    cf = world["root"] / "paint" / "poses" / "canon.json"
    bad = {"level": "error", "rule": "smear", "why": "a smeared hand"}
    cf.write_text(json.dumps({"drawings": {"VID-027": {"flags": [bad]}, "RUN-1": {"flags": [bad]}}}))
    try:
        d = json.loads(json.dumps(BOX))
        d.update(motion="m0", segments=[{"from": 4.1, "to": 7.1, "rate": "16ths"}])
        d["moves"] = [{"run": {"from": 4.1, "sequence": "run_SPR"}},
                      {"vault": {"obstacle": "ac", "plant_on": 5.2, "sequence": "vault_VID"}},
                      {"run": {"to": "6.4.4"}}]
        ch = _with(world, "vcanon", d)
        shown = [e.pose for e in ch.events if ch.moves[e.move].kind == "vault"]
        assert "VID-027" not in shown and len(shown) > 5
        moves = [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}}, {"pose": {"at": 4.2, "pose": "RUN-2", "x": 700}}]
        ch = _tall_choreo(world, "qacanon", moves, [{"from": 4.1, "to": 4.3, "rate": "16ths"}])
        hist = History(ch)
        photos = ch.photos()
        res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
        canon = [f for f in res["flags"] if f["rule"] == "canon"]
        assert len(canon) == 1 and canon[0]["pose"] == "RUN-1" and canon[0]["level"] == "error"
    finally:
        cf.unlink()


def test_old_boxes_hold_a_grey_figure_still_and_dry(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "other", "x": 1500, "old": 0.9, "figure": "KEY-1",
                            "keys": [{"at": 4.1, "w": 200, "h": 260, "thick": 16}]}},
                   {"box": {"id": "his", "x": 2200, "keys": [{"at": 4.1, "w": 200, "h": 260, "old": 0.5},
                                                             {"at": 5.1, "thick": 24}]}},
                   {"pose": {"at": 4.2, "pose": "RUN-1", "x": 600}}]}
    ch = _with(world, "oldbox", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    other, his = ch.props["other"], ch.props["his"]
    assert other.figure.pose == "KEY-1" and other.keys[0].old == 0.9
    assert [k.old for k in his.keys] == [0.5, 0.0]
    ix0, iy0, ix1, iy1 = other.keys[0].inner
    assert ix0 < other.figure.x < ix1 and other.figure.y == pytest.approx(iy1 - 2, abs=1)
    hist = History(ch)
    fig = [m for m in hist.marks if m.kind == "figure"]
    assert len(fig) == 1
    g = ch.grid
    s1 = hist._fx_state(fig[0], g.t(50), ch.paint_clock(g.t(50)))
    s2 = hist._fx_state(fig[0], g.t(90), ch.paint_clock(g.t(90)))
    assert np.array_equal(s1[0], s2[0]) and s1[2] == 0.0 and 0.03 < s1[1] < 0.3  # still, dry, grey
    box = {m.t: m for m in hist.marks if m.kind == "box" and m.x0 > 2000}
    old, fresh = box[min(box)], box[max(box)]
    so = hist._fx_state(old, g.t(56), ch.paint_clock(g.t(56)))
    sf = hist._fx_state(fresh, g.t(64) + 0.01, ch.paint_clock(g.t(64) + 0.01))
    assert so[2] == 0.0 and sf[2] > 0.3  # the waiting outline is dry, the repaint wet


def test_frame_camera_shows_a_set_stretch_of_street(world):
    d = {"street": "main", "seed": 3, "motion": "m0",
         "segments": [{"from": 4.1, "to": 4.2, "rate": "beats",
                       "camera": {"preset": "frame", "x": [1000, 2000], "snap": True}},
                      {"from": 4.2, "to": 5.1, "rate": "beats"}],
         "moves": [{"run": {"from": 4.2, "from_x": 600, "to": "4.4.4"}}]}
    ch = _with(world, "framecam", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    photos = ch.photos()
    cams = plan(ch, photos, hist.figure_boxes(), "16:9")
    assert cams[0].cw == pytest.approx(1000, rel=0.1) and cams[0].cx == pytest.approx(1500, abs=30)
    d["segments"][0]["camera"] = {"preset": "frame"}
    assert any("camera frame" in str(i) for i in _with(world, "framecam2", d).errors)


# ---------------------------------------------------------------------------------------------- demo v3

BXC_BOX = (440, 80, 840, 680)  # the pale blue box outline in the synthetic in-box clip (outer), 30 px line


def _box_clip(root):
    """A 1280x720 30 fps front view inside a pale blue box outline: a figure standing on its inner bottom, drifting
    2 px a frame, its arm out to the right wall on frames 2, 3, 8, 9, ... (punches) and drawn back otherwise."""
    f = root / "boards" / "boxclip.mp4"
    w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    x0, y0, x1, y1 = BXC_BOX
    for k in range(48):
        img = np.full((720, 1280, 3), 236, np.uint8)
        cv2.rectangle(img, (x0 + 15, y0 + 15), (x1 - 16, y1 - 16), (236, 203, 172), 30)
        cx = 600 + 2 * k
        cv2.circle(img, (cx, 330), 34, (0, 0, 0), -1)
        cv2.line(img, (cx, 360), (cx, 520), (0, 0, 0), 60)
        cv2.line(img, (cx, 520), (cx - 40, 648), (0, 0, 0), 30)
        cv2.line(img, (cx, 520), (cx + 40, 648), (0, 0, 0), 30)
        reach = (805, 400) if k % 6 in (2, 3) else (cx + 70, 470)
        cv2.line(img, (cx, 390), reach, (0, 0, 0), 22)
        w.write(img)
    w.release()
    return "boards/boxclip.mp4"


@pytest.fixture(scope="module")
def boxclip(world):
    """The world's pose library with the in-box clip (boxed, scaled by its guide box, its frames from 40 on drawn
    1.2 times larger again by `rescale`) and two more video sequences: box_BXC (keys, hit and back frame lists) and
    leap_SPR (the sprint as a flight between a take-off and a landing key)."""
    root = world["root"]
    pd = root / "paint" / "poses"
    spec = json.loads((pd / "poses.spec.json").read_text())
    spec["video"]["BXC"] = {"path": _box_clip(root), "key": [172, 203, 236], "ground": "feet", "boxed": True,
                            "scale": 0.5, "rescale": [{"from": 40, "factor": 1.2, "about": [640, 650]}]}
    spec["sequences"]["box_BXC"] = {"video": "BXC", "range": [0, 47], "keys": {"mid": 20, "end": 38},
                                    "hit": [2, 8, 14, 20, 26, 32], "back": [0, 4, 6, 10, 16, 22]}
    spec["sequences"]["leap_SPR"] = {"video": "SPR", "range": [0, 63], "keys": {"takeoff": 16, "land": 40}}
    spec["sequences"]["launch_SPR"] = {"video": "SPR", "range": [0, 63], "keys": {"coil": 8, "takeoff": 16, "land": 40}}
    spec["sequences"]["roll_SPR"] = {"video": "SPR", "range": [0, 63], "keys": {"land": 40, "run": 60}}
    spec["sequences"]["leap_BXC"] = {"video": "BXC", "range": [0, 47], "keys": {"launch": 8, "dive": 14, "land": 34,
                                                                               "sprint": 40}}
    spec["sequences"]["burst_BXC"] = {"video": "BXC", "range": [0, 47], "keys": {"windup": 20, "through": 26,
                                                                                 "apex": 32, "land": 38}}
    spec["canon_waive"] = {"BXC-005": {"smear": "checked by eye"}}
    (pd / "poses.spec.json").write_text(json.dumps(spec))
    PS.build(root, pd, log=lambda m: None)
    return PS.PoseLib(pd)


def test_rescale_keeps_one_body_size_and_boxed_clips_keep_their_flags(boxclip):
    tr = {r["f"]: r for r in boxclip.videos["BXC"]["frames"]}
    assert boxclip.videos["BXC"]["boxed"] and boxclip.videos["BXC"]["scale"] == 0.5
    assert np.allclose(boxclip.videos["BXC"]["block"], BXC_BOX, atol=4)  # the codec softens its edge a little
    before = np.median([r["h"] for f, r in tr.items() if f < 40])
    after = np.median([r["h"] for f, r in tr.items() if f >= 40])
    assert after == pytest.approx(1.2 * before, rel=0.04)
    assert abs(np.median([r["low"] for f, r in tr.items() if f >= 40]) -
               np.median([r["low"] for f, r in tr.items() if f < 40])) < 6  # scaled about his feet
    assert json.loads((boxclip.d / "poses.json").read_text())["canon_waive"]["BXC-005"]


def test_inside_plays_a_clip_fitted_to_the_box_and_strikes_on_the_hits(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 6.1, "rate": "8ths"}, {"from": 6.1, "to": 8.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14},
                                                             {"at": 5.1, "w": 160, "h": 240}]}},
                   {"inside": {"box": "box", "from": 4.1, "to": "7.4.4", "roles": [
                       {"at": 4.1, "clip": "box_BXC", "keys": {"mid": 4.3, "end": 5.3}},
                       {"at": 6.1, "clip": "box_BXC", "strike": {"on": "snare", "hit": "hit", "back": "back",
                                                                  "push": 6, "crack": {"size": 20, "grow": 8},
                                                                  "punch": [0.02, 0.06], "shake": 3,
                                                                  "flicker": "lamp"}}]}}]}
    ch = _with(world, "inclip", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    lib, g = ch.lib, ch.grid
    pr = ch.props["box"]
    ev = sorted([e for e in ch.events if e.tag == "inside"], key=lambda e: e.q)
    assert all(e.pose.startswith("BXC-") and e.anchor == "cxc" for e in ev)
    by = {g.addr(e.q): e for e in ev}
    assert by["4.3.1"].pose == "BXC-020" and by["5.3.1"].pose == "BXC-038"  # the keys on their photos
    for e in ev:  # one body scale per photo: the clip's box on the box as it is then (it shrinks at 5.1)
        k = pr.geometry(e.q)
        bb = lib.videos["BXC"]["block"]
        assert e.scale * lib.m(e.pose)["scale"] == pytest.approx(k.h / (bb[3] - bb[1]), rel=1e-6)
        assert k.box[0] < e.x < k.box[2]
    # after the clip's end he keeps at it: the last frames to and fro, never the same drawing twice in a row
    tail = [e.pose for e in ev if g.addr(e.q) >= "5.3.3" and e.q < g.resolve(6.1)[0]]
    assert tail and all(a != b for a, b in zip(tail, tail[1:]))
    # strike: a photo on each snare shows the next hit frame (and leans in), every other photo a back frame
    a6, a8 = g.resolve(6.1)[0], g.resolve(8.1)[0]
    snares = [g.q(t) for t, s in g.hits["snare"] if s >= 0.85 and a6 - 0.5 <= g.q(t) <= a8]
    hit_f = [f"BXC-{f:03d}" for f in (2, 8, 14, 20, 26, 32)]
    on = [e for e in ev if any(abs(e.q - q) < 0.5 for q in snares)]
    assert on and [e.pose for e in on] == hit_f[:len(on)]
    assert all(e.rot != 0 for e in on)
    off = [e for e in ev if e.q >= g.resolve(6.1)[0] and e not in on]
    assert off and all(e.pose not in hit_f for e in off)
    crk = [c for c in pr.cracks if c.q >= g.resolve(6.1)[0] - 1e-6]
    assert len(crk) == len(on) and crk[-1].size > crk[0].size
    assert all(abs(c.x - (pr.geometry(c.q).side_x(c.y, "right") + 3 * ch.u)) < 1 for c in crk)  # the right wall
    hints = [ch.cam_hints[round(e.q, 4)] for e in on]
    assert all("punch" in h and "shake" in h for h in hints) and hints[-1]["punch"] > hints[0]["punch"]
    assert all(ch.light_hints[round(e.q, 4)] == {"lamp": 0.15} for e in on)
    st = motion_stats(ch)
    assert st["backtracks"] == []  # pulling back to hit again is not a backtrack


def test_leap_video_flies_an_arc_in_the_air_with_a_pale_ghost_trail(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"leap": {"gap": "alley", "sequence": "leap_SPR", "at": 4.2, "land_on": 5.1, "height": 80,
                             "to": "5.2.2"}}]}
    ch = _with(world, "vleap", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if e.layer in ("wall", "air") and e.pose.startswith("SPR-")], key=lambda e: e.q)
    q0, q1 = g.resolve(4.2)[0], g.resolve(5.1)[0]
    fl = [e for e in ev if q0 < e.q < q1]
    tko = next(e for e in ev if e.q == q0)
    land = next(e for e in ev if e.q == q1)
    assert tko.pose == "SPR-016" and land.pose == "SPR-040" and tko.layer == "air"  # he leaves the wall, dark
    assert fl and all(e.layer == "air" and e.flags.get("ghosts") == 3.0 for e in fl)
    xs, ys = [tko.x] + [e.x for e in fl] + [land.x], [e.y for e in fl]
    assert all(b > a for a, b in zip(xs, xs[1:]))
    top = int(np.argmin(ys))  # one smooth arc: up, then down
    assert all(b < a for a, b in zip(ys[:top + 1], ys[1:top + 1]))
    assert all(b > a for a, b in zip(ys[top:], ys[top + 1:]))
    assert any(s.layer == "stamp" and s.q == q0 for s in ch.events)  # a splash at the take-off too
    assert land.layer == "air" and not land.flags.get("halo")  # the skin hits the far wall: no primed outline
    # the camera eases throughout: wider over the take-off photos, panning with him, closing in after the landing
    hs = [ch.cam_hints[round(q, 4)] for q in np.arange(q0, q1 + 5 - 1e-6, 1.0)]
    assert not any(h.get("snap") for h in hs) and all("cx" in h and h["follow_v"] for h in hs)
    zs = [h["zoom"] for h in hs]
    assert 1.0 < zs[0] < zs[3] and zs[-1] == 1.0 and max(zs) > 1.0
    assert all(b >= a - 1e-6 for a, b in zip([h["cx"] for h in hs], [h["cx"] for h in hs][1:]))
    k = int(q1 - q0)
    assert all(abs(b - a) / a <= 0.12 for a, b in zip(zs[k - 1:], zs[k:]))  # no jump in on the landing
    # back on the wall the primer arrives gradually: the patches under the first photos grow to full size
    walls = sorted([e for e in ev if e.layer == "wall" and e.q > q1], key=lambda e: e.q)
    assert [e.flags.get("patch_grow") for e in walls[:4]] == [[1, 4], [2, 4], [3, 4], None]
    hist0 = History(ch)
    pat = {m.q: m.w * m.h for m in hist0.marks if m.kind == "primer"}
    assert q1 not in pat  # no primer under the landing photo
    sizes = [pat[e.q] / max(hist0.place(ch.events.index(e))[0].size, 1) for e in walls[:4]]
    assert sizes[0] < 0.7 * sizes[3] and sizes[0] < sizes[1] < sizes[3]
    hist = History(ch)
    mid = fl[len(fl) // 2].q
    A, _ = hist.air_layer(g.t(mid), mid, 0, 0, 3000, 860)
    for qq in hist.air:
        for it in hist.air[qq]:
            it.flags["ghosts"] = 0.0
    A0, _ = hist.air_layer(g.t(mid), mid, 0, 0, 3000, 860)
    assert (A > 0.05).sum() > 1.3 * (A0 > 0.05).sum()  # the photos before still hang there, pale


def test_clip_arc_rises_and_falls(world, boxclip):
    seg = [{"from": 4.1, "to": 7.1, "rate": "16ths"}]
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": seg,
         "moves": [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.2, "b": 5.2}, "x": {"a": 1000, "b": 2200},
                             "arc": {"from": "a", "to": "b", "height": 120}}}]}
    ch = _with(world, "varc", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ev = sorted([e for e in ch.events if e.layer == "wall"], key=lambda e: e.q)
    a, b = ch.grid.resolve(4.2)[0], ch.grid.resolve(5.2)[0]
    ys = [e.y for e in ev if a < e.q < b]
    top = int(np.argmin(ys))
    assert 0 < top < len(ys) - 1 and min(ys) < min(e.y for e in ev if e.q in (a, b)) - 120 * ch.u * 0.9
    # strictly up, then strictly down: no floating at one height
    assert all(q < p for p, q in zip(ys[:top + 1], ys[1:top + 1]))
    assert all(q > p for p, q in zip(ys[top:], ys[top + 1:]))


def test_box_paint_in_hits_mist_flakes_and_words_for_yaml_on(world):
    text = """street: main
seed: 3
motion: m0
primer: ghosts
segments:
  - {from: 4.1, to: 7.1, rate: 8ths}
moves:
  - box:
      id: box
      x: 1500
      photo: true
      figure: {pose: KEY-1, at: 4.1.3, paint_in: 0.3}
      keys:
        - {at: 4.1, w: 200, h: 260, thick: 16, paint_in: 0.3}
      stencils:
        - {at: 4.3, text: KEEP IT DOWN, x: 1250, y: 250, h: 24, align: right}
        - {at: 4.3, text: WAIT, x: 1560, y: 250, h: 24}
      hits:
        - {on: snare, from: 5.1, to: 5.4.4, bulge: 10, flakes: 6, grow: 0.5}
        - {at: 6.1, bulge: 30, flakes: 12}
      peel_from: 5.1
  - pose: {at: 4.2, pose: RUN-1, x: 600}
"""
    f = world["root"] / "paint" / "choreo" / "v3box.yaml"
    f.write_text(text)
    ch = open_choreo(world["p"], "v3box")
    assert not ch.errors, [str(i) for i in ch.issues]
    pr, g = ch.props["box"], ch.grid
    assert len(pr.bumps) == 3 and pr.bumps[1].amp == pytest.approx(1.5 * pr.bumps[0].amp)
    assert pr.bumps[2].amp == pytest.approx(30 * ch.u) and pr.peel_from == g.resolve(5.1)[0]
    assert len([p for p in ch.photos() if abs(p.q - g.resolve(4.3)[0]) < 1e-6]) == 1  # one photo for both
    hist = History(ch)
    box = next(m for m in hist.marks if m.kind == "box")
    t0 = g.t(box.q if box.q else g.resolve(4.1)[0])
    on_key = hist._fx_state(box, t0, ch.paint_clock(t0))[0].sum()
    later = hist._fx_state(box, t0 + 0.5, ch.paint_clock(t0 + 0.5))[0].sum()
    assert 0.1 * later < on_key < 0.8 * later  # painted in with a sweep, a third done on its key
    fig = next(m for m in hist.marks if m.kind == "figure")
    assert fig.t == pytest.approx(g.t(g.resolve("4.1.3")[0]))
    mist = [m for m in hist.marks if m.kind == "mist"]
    st = [m for m in hist.marks if m.kind == "stencil"]
    assert len(mist) == len(st) == 2 and mist[0].w > st[0].w
    ts = mist[0].t
    assert hist._fx_state(mist[0], ts + 0.05, ch.paint_clock(ts + 0.05))[0].max() > 0.5
    assert hist._fx_state(mist[0], ts + 0.7, ch.paint_clock(ts + 0.7)) is None  # settled
    assert sum(len(s) for t, _, s in hist.shards if t < g.t(g.resolve(6.1)[0]) + 0.1) >= 6 * 3
    assert ch.cam_hints[round(g.resolve(4.3)[0], 4)]["punch"] == pytest.approx(0.03)  # a rule lands like a hit


def test_light_trail_drops_keepers_and_the_flicker_dims_a_light(world, boxclip):
    import shutil
    src = world["street"]
    dd = src.parent / "lit"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["lights"] = [{"id": "lamp", "x": 900, "y": 300, "sigma": 300, "strength": 0.8, "paint": 1.0}]
    (dd / "obstacles.json").write_text(json.dumps(ob))
    d = {"street": "lit", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.2, "b": 5.2}, "x": {"a": 700, "b": 1300},
                             "trail": "light"}},
                   {"pose": {"at": "5.4", "pose": "RUN-1", "x": 1400}}]}
    ch = _with(world, "vlight", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    keys = [i for i, e in enumerate(ch.events) if e.tag.startswith("clip.")]
    assert keys and all(not m.keeper for m in hist.marks if m.kind == "pose" and m.gen[1] in keys)
    q = ch.grid.resolve(4.3)[0]
    jobs, _ = plan_jobs(ch, hist, ["16:9"])
    j = next(j for j in jobs if abs(j.q - q) < 1e-6)
    r = Renderer(ch, hist)
    a = r.photo(j.k, j.t, j.q, j.cams["16:9"], (480, 270))
    ch.light_hints[round(q, 4)] = {"lamp": 0.1}
    b = r.photo(j.k, j.t, j.q, j.cams["16:9"], (480, 270))
    assert np.abs(a - b).max() > 0.02  # the glow on the paint dimmed


# ---------------------------------------------------------------------------------------------- demo v4

def _lit_street(world):
    import shutil
    src = world["street"]
    dd = src.parent / "lit"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["lights"] = [{"id": "lamp", "x": 900, "y": 300, "sigma": 300, "strength": 0.8, "paint": 1.0}]  # glows on paint
    (dd / "obstacles.json").write_text(json.dumps(ob))
    return "lit"


def test_holds_are_flagged_unless_still_or_frozen(world):
    seg = [{"from": 4.1, "to": 5.1, "rate": "16ths"}]
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": seg,
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}}, {"pose": {"at": "4.2", "pose": "RUN-2", "x": 700}},
                   {"pose": {"at": "4.2.2", "pose": "RUN-3", "x": 790}}]}
    ch = _with(world, "vhold", d)
    holds = motion_stats(ch)["holds"]
    assert [(h["pose"], h["n"]) for h in holds] == [("RUN-1", 4), ("RUN-3", 11)]
    d["moves"][0]["pose"]["still"] = True
    assert [h["pose"] for h in motion_stats(_with(world, "vhold2", d))["holds"]] == ["RUN-3"]
    from aimv.paint import qa as QA
    hist = History(ch)
    photos = ch.photos()
    res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
    assert {f["pose"] for f in res["flags"] if f["rule"] == "hold"} == {"RUN-1", "RUN-3"}


def test_strike_segments_land_each_piece_on_its_hit(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"inside": {"box": "box", "from": 4.1, "to": "6.4.4", "roles": [
                       {"at": 4.1, "clip": "box_BXC", "strike": {"on": "snare", "segments": [[4, 14], [16, 26]],
                                                                  "tail": [30, 44], "crack": {"size": 20}}}]}}]}
    ch = _with(world, "vseg", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if e.tag == "inside"], key=lambda e: e.q)
    fr = [int(e.pose.split("-")[1]) for e in ev]
    a4, b6 = g.resolve(4.1)[0], g.resolve("6.4.4")[0]
    snares = [g.q(t) for t, s in g.hits["snare"] if s >= 0.85 and a4 <= g.q(t) <= b6]
    on = [i for i, e in enumerate(ev) if any(abs(e.q - q) < 0.5 for q in snares)]
    tr = sorted(r["f"] for r in boxclip.videos["BXC"]["frames"])
    near = lambda v: min(tr, key=lambda x: abs(x - v))
    assert [fr[i] for i in on] == [near(14), near(26)] * (len(on) // 2) + [near(14)] * (len(on) % 2)
    for a_, b_ in zip(on, on[1:]):  # between two hits the next piece plays forward to its hit frame
        piece = fr[a_ + 1:b_ + 1]
        assert all(y >= x for x, y in zip(piece, piece[1:]))
    tail = fr[on[-1] + 1:]
    assert tail and tail[-1] == near(44) and all(y >= x for x, y in zip(tail, tail[1:]))
    assert all(not (a_ == b_ == c_) for a_, b_, c_ in zip(fr, fr[1:], fr[2:]))  # never three in a row
    assert len(ch.props["box"].cracks) == len(on)


def test_burst_from_a_front_view_clip_breaks_out_lands_and_grows(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts", "pause": False,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"burst": {"box": "box", "at": 5.1, "sequence": "burst_BXC", "frame": "box", "from": "4.4.3",
                              "windup_on": "4.4.3", "land_on": 5.3, "land_x": 1800, "to": "5.4.4",
                              "run_sequence": "run_SPR", "power": 2.0, "splats": 30, "flicker": "x"}}]}
    ch = _with(world, "vbbox", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g, lib = ch.grid, ch.lib
    ev = sorted([e for e in ch.events if e.layer == "wall" and e.pose and e.pose.startswith("BXC-")], key=lambda e: e.q)
    by = {g.addr(e.q): e for e in ev}
    assert by["4.4.3"].tag == "burst.windup" and by["5.1.1"].tag == "burst.through" and by["5.3.1"].tag == "burst.land"
    assert by["4.4.4"].tag == "burst.box"  # in the box before the break-through: pulling back is no backtrack
    pr = ch.props["box"]
    k = pr.keys[0]
    from aimv.paint.moves import _in_box_place
    x, y, sc = _in_box_place(type("C", (), {"ch": ch, "u": ch.u})(), pr, lib.sequence("burst_BXC"),
                             next(r for r in lib.videos["BXC"]["frames"] if r["name"] == by["4.4.3"].pose), k)
    assert by["4.4.3"].x == pytest.approx(x) and by["4.4.3"].scale == pytest.approx(sc)  # placed by the box
    land = by["5.3.1"]
    assert land.x == pytest.approx(1800, abs=1)
    m = lib.m(land.pose)
    low = land.y + (m["bbox"][3] - m["bbox"][1]) / 2 * m["scale"] * land.scale
    assert low == pytest.approx(ch.scene.feet_at(land.x) + 4 * ch.u, abs=1.5)  # his landing on the ground line
    after = [e for e in ev if e.q > land.q]
    hh = lambda nm, scl: (lib.m(nm)["bbox"][3] - lib.m(nm)["bbox"][1]) * lib.m(nm)["scale"] * scl
    run_h = np.median([hh(nm, 1.0) for nm in lib.sequence("run_SPR")["drawings"]])
    box_h = np.median([hh(e.pose, land.scale) for e in [land] + after])
    assert after and hh(after[-1].pose, after[-1].scale) == pytest.approx(
        hh(after[-1].pose, land.scale) * run_h / box_h, rel=0.15)  # grown to the runner's size
    assert pr.shatter is not None and pr.shatter.q == g.resolve(5.1)[0]
    assert motion_stats(ch)["backtracks"] == []


def test_launch_keys_burst_knock_and_hanging_ghosts(world, boxclip):
    street = _lit_street(world)
    d = {"street": street, "seed": 3, "motion": "m0", "primer": "ghosts", "pause": False,
         "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"leap": {"gap": "alley", "sequence": "launch_SPR", "from": "4.1.2", "keys": {"coil": "4.1.4"},
                             "at": 4.2, "land_on": 5.1, "hang": 4, "splash": 1.5, "knock": "lamp",
                             "burst": {"flecks": 20, "splats": 25}, "to": "5.2.2", "trail": "light"}}]}
    ch = _with(world, "vlaunch", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    q0, q1 = g.resolve(4.2)[0], g.resolve(5.1)[0]
    ev = sorted([e for e in ch.events if e.layer in ("wall", "air") and e.pose.startswith("SPR-")], key=lambda e: e.q)
    pre = [e for e in ev if e.q < q0]
    assert pre and all(e.layer == "wall" for e in pre) and any(e.tag == "leap.coil" for e in pre)
    assert ch.lights_out == {"lamp": q0} and ch.light_hints[round(q0 - 1, 4)] == {"lamp": 2.2}
    assert len(ch.bursts) == 2 and {b.albedo for b in ch.bursts} == {0.02, 0.6}  # paint at his feet, glass
    hist = History(ch)
    assert sum(len(sh) for t, _, sh in hist.shards) >= 20 + 45
    assert any(m.kind == "spatter" for m in hist.marks)
    big = [s for s in ch.events if s.layer == "stamp" and s.q == q1 and s.tag == "leap.land"]
    assert big and max(s.w for s in big) == pytest.approx(520 * 1.5 * ch.u)
    # after the landing the whole arc's ghosts hang over the gap for `hang` 16ths, then fade and go
    A1, _ = hist.air_layer(g.t(q1 + 2), q1 + 2, 0, 0, 3000, 860)
    A2, _ = hist.air_layer(g.t(q1 + 14), q1 + 14, 0, 0, 3000, 860)
    assert (A1 > 0.05).sum() > 2000 and (A2 > 0.05).sum() < 0.05 * (A1 > 0.05).sum()
    # the knocked-out light: its glow on the paint is gone from the take-off on
    jobs, _ = plan_jobs(ch, hist, ["16:9"])
    j = next(j for j in jobs if abs(j.q - (q0 + 1)) < 1e-6)
    r = Renderer(ch, hist)
    a = r.photo(j.k, j.t, j.q, j.cams["16:9"], (480, 270))
    ch.lights_out.clear()
    b = r.photo(j.k, j.t, j.q, j.cams["16:9"], (480, 270))
    assert np.abs(a - b).max() > 0.02


def test_clipframes_chain_a_box_and_break_it_open(world, tmp_path):
    from aimv.paint import clipframes as CF
    lib = PS.PoseLib(world["root"] / "paint" / "poses")
    sc = Scene(world["street"], world["root"] / ".cache")
    a = CF.make(lib, sc, "box", "one", tmp_path, start_pose="KEY-1", end_pose="KEY-5@right", box=(200, 260, 16),
                scale_by="KEY-5")
    b = CF.make(lib, sc, "box", "two", tmp_path, start_pose="KEY-5@right", end_pose="RUN-1@out", box=(200, 260, 16),
                scale_by="KEY-5", open_end="right")
    e1, s2 = cv2.imread(a["frames"]["end"]), cv2.imread(b["frames"]["start"])
    assert np.array_equal(e1, s2)  # one clip's end frame is the next one's start
    end = cv2.imread(b["frames"]["end"])
    key = np.all(np.abs(end.astype(int) - [236, 203, 172]) < 10, axis=2)  # the guide (BGR)
    x0, y0, x1, y1 = b["block"]
    assert key[y0 + 30:y1 - 30, x0:x0 + 10].all() and not key[y0 + 60:y1 - 60, x1 - 10:x1].any()  # right side open
    dark = end.mean(2) < 90
    assert dark[:, x1:].sum() > 5000 and not dark[:, x0 + 30:x1 - 30].any()  # he is out of the box on its right
    rightp = cv2.imread(a["frames"]["end"]).mean(2) < 90
    ys, xs = np.nonzero(rightp[:, x0:x1])
    assert x0 + xs.max() > x1 - 0.05 * (x1 - x0) - b["box"][2] * b["street_to_frame"]  # against the right side


def test_dedupe_thresholds_keep_subtle_motion():
    m = np.zeros((200, 100), np.float32)
    m[20:190, 30:70] = 1.0
    n = m.copy()
    n[20:50, 70:85] = 1.0  # a small change: a hood turning (overlap 0.94)
    prev = (0, 0.0, m, 10, 10)
    assert PS._same(prev, n, 10, 10)  # the default: dropped as a double
    assert not PS._same(prev, n, 10, 10, 0.97, 0.97)  # strict: kept


# ---------------------------------------------------------------------------------------------- demo v5

def test_crossing_flags_a_pipe_across_him_while_he_stays_put(world, tall):
    from aimv.paint import qa as QA
    still = [{"pose": {"at": f"4.1.{k}", "pose": "RUN-1" if k % 2 else "RUN-2", "x": 1115}} for k in range(1, 5)]
    ch = _tall_choreo(world, "vcross", still, [{"from": 4.1, "to": 4.2, "rate": "16ths"}])
    hist = History(ch)
    photos = ch.photos()
    res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
    assert any(f["rule"] == "crossing" and f["level"] == "error" for f in res["flags"])
    moving = [{"pose": {"at": f"4.1.{k}", "pose": "RUN-1", "x": 900 + 120 * k}} for k in range(1, 5)]
    ch = _tall_choreo(world, "vcross2", moving, [{"from": 4.1, "to": 4.2, "rate": "16ths"}])
    hist = History(ch)
    photos = ch.photos()
    res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
    assert not any(f["rule"] == "crossing" for f in res["flags"])


def test_body_drop_lays_a_slide_on_the_ground(world):
    from aimv.paint.motion import _body_drop
    ch = _choreo(world)
    c = type("C", (), {"ch": ch, "u": ch.u})()
    lib = ch.lib
    m = lib.m("KEY-7")  # the horizontal slide drawing
    s = m["scale"]
    half = (m["bbox"][3] - m["bbox"][1]) / 2 * s
    f = ch.scene.feet_at(1000)
    y = f - half - 60  # floating 60 px above the feet line
    dy = _body_drop(c, "KEY-7", 1000.0, y, 1.0)
    assert dy > 40
    d = lib.sprite("KEY-7")
    ys, _ = np.nonzero(d > 0.4)
    lowest = y + dy + (np.percentile(ys, 99.5) - (m["bbox"][1] + m["bbox"][3]) / 2) * s
    assert lowest <= ch.scene.wall_base_at(1000) - 12  # never past the wall's foot
    assert _body_drop(c, "KEY-7", 1000.0, y + dy + 30, 1.0) == 0.0  # never raises it


def test_neutral_light_keeps_white_primer_white(world, boxclip):
    street = _lit_street(world)
    sd = world["street"].parent / street
    ob = json.loads((sd / "obstacles.json").read_text())
    d = {"street": street, "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 4.3, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 900, "y": 500}}]}
    out = []
    for light in ({"id": "lamp", "x": 900, "y": 300, "sigma": 300, "strength": 0.8, "color": [1.0, 0.1, 0.1],
                   "paint": 1.0, "neutral": 0.0},
                  {"id": "lamp", "x": 900, "y": 300, "sigma": 300, "strength": 0.8, "color": [1.0, 0.1, 0.1],
                   "paint": 0.0, "neutral": 1.0}):
        ob["lights"] = [light]
        (sd / "obstacles.json").write_text(json.dumps(ob))
        ch = _with(world, "vneutral", d)
        hist = History(ch)
        jobs, _ = plan_jobs(ch, hist, ["16:9"])
        img = Renderer(ch, hist).photo(jobs[0].k, jobs[0].t, jobs[0].q, jobs[0].cams["16:9"], (480, 270))
        out.append(img)
    dif = out[0] - out[1]  # where the paint is: the red glow and cast gone from it
    sel = np.abs(dif).sum(-1) > 0.02
    assert sel.sum() > 200
    assert dif[..., 0][sel].mean() > dif[..., 1][sel].mean() + 0.01  # it was redder than it is now


def test_burst_leap_plays_a_second_clip_on_an_arc_and_grows(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts", "pause": False,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"burst": {"box": "box", "at": 5.1, "sequence": "burst_BXC", "frame": "box", "from": "4.4.3",
                              "windup_on": "4.4.3", "run_sequence": "run_SPR", "grow_from": "dive",
                              "leap": {"sequence": "leap_BXC", "keys": {"launch": "5.1.2", "dive": "5.1.4"},
                                       "land": "5.4.1", "x": {"dive": 1700, "land": 1900}, "height": 80},
                              "to": "5.4.1"}}]}
    ch = _with(world, "vbleap", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g, lib = ch.grid, ch.lib
    ev = sorted([e for e in ch.events if e.layer == "wall" and e.pose and e.pose.startswith("BXC-")], key=lambda e: e.q)
    by = {g.addr(e.q): e for e in ev}
    assert by["5.1.1"].tag == "burst.through" and by["5.1.4"].tag == "burst.dive" and by["5.4.1"].tag == "burst.land"
    assert by["5.1.4"].x == pytest.approx(1700, abs=1) and by["5.4.1"].x == pytest.approx(1900, abs=1)
    after = [e for e in ev if e.q > g.resolve(5.1)[0]]
    assert all(b.x >= a.x for a, b in zip(after, after[1:]))  # never a step back out of the box
    mid = after[len(after) // 2]
    land = by["5.4.1"]
    assert mid.y < land.y - 30  # lifted on its arc
    vid = lib.videos["BXC"]
    s_box = 300 / (vid["block"][3] - vid["block"][1])
    run_h = np.median([(lib.m(n)["bbox"][3] - lib.m(n)["bbox"][1]) * lib.m(n)["scale"]
                       for n in lib.sequence("run_SPR")["drawings"]])
    ratio = run_h / (np.median([r["h"] for r in vid["frames"] if r["f"] >= 40]) * s_box)
    assert land.scale / by["5.1.4"].scale == pytest.approx(ratio, rel=0.03)  # grown to his running size by landing


def test_blows_hit_on_their_keys_and_the_box_camera_steps_in(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 8.1, "rate": "16ths",
                       "camera": {"preset": "box", "box": "box", "steps": [5.1, 8.1, 0.85]}}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"inside": {"box": "box", "from": 4.1, "to": "7.4.4", "roles": [
                       {"at": 4.1, "clip": "box_BXC", "keys": {"mid": 5.2, "end": 7.2},
                        "blows": {"keys": ["mid", "end"], "push": 5, "crack": {"size": 20, "grow": 10},
                                  "punch": [0.02, 0.05], "shake": 3}}]}}]}
    ch = _with(world, "vblows", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    cr = ch.props["box"].cracks
    assert [g.addr(c_.q) for c_ in cr] == ["5.2.1", "7.2.1"] and cr[1].size > cr[0].size
    assert {"punch", "shake"} <= set(ch.cam_hints[round(g.resolve(5.2)[0], 4)])
    hist = History(ch)
    photos = ch.photos()
    cams = plan(ch, photos, hist.figure_boxes(), "16:9")
    w_bar = {}
    for ph, cm in zip(photos, cams):
        w_bar.setdefault(g.addr(ph.q).split(".")[0], []).append(cm.cw)
    assert np.median(w_bar["7"]) < np.median(w_bar["5"]) * 0.95  # a step closer each bar


# ---------------------------------------------------------------------------------------------- v5 review

def test_strike_takes_hit_and_back_frame_lists_and_a_tail(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"inside": {"box": "box", "from": 4.1, "to": "6.4.4", "roles": [
                       {"at": 4.1, "clip": "box_BXC", "strike": {"on": "snare", "hit": [14, 26], "back": [5, 21],
                                                                  "tail": [30, 44], "crack": {"size": 20}}}]}}]}
    ch = _with(world, "vlists", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if e.tag == "inside"], key=lambda e: e.q)
    fr = [int(e.pose.split("-")[1]) for e in ev]
    a4, b6 = g.resolve(4.1)[0], g.resolve("6.4.4")[0]
    snares = [g.q(t) for t, s in g.hits["snare"] if s >= 0.85 and a4 <= g.q(t) <= b6]
    on = [i for i, e in enumerate(ev) if any(abs(e.q - q) < 0.5 for q in snares)]
    tr = sorted(r["f"] for r in boxclip.videos["BXC"]["frames"])
    near = lambda v: min(tr, key=lambda x: abs(x - v))
    assert len(on) >= 2
    assert [fr[i] for i in on] == [near((14, 26)[k % 2]) for k in range(len(on))]  # inline lists: nearest kept frame
    backs = [fr[i] for i in range(on[-1]) if i not in on]
    assert backs and backs == [near((5, 21)[k % 2]) for k in range(len(backs))]
    tail = fr[on[-1] + 1:]  # after the last hit: the tail, ending on its last frame (the hand-over)
    assert tail and tail[-1] == near(44) and all(y >= x for x, y in zip(tail, tail[1:]))
    assert len(ch.props["box"].cracks) == len(on)


def test_box_camera_look_pushes_in_on_its_region_and_eases_back(world, boxclip):
    base = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
            "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                      {"inside": {"box": "box", "from": 4.1, "to": "8.4.4",
                                  "roles": [{"at": 4.1, "clip": "box_BXC"}]}}]}
    region = [1420, 250, 1700, 420]
    out = {}
    for name, look in (("vnolook", None), ("vlook", {"from": 5.1, "to": 6.1, "back": 7.1, "region": region})):
        cam = {"preset": "box", "box": "box", **({"look": look} if look else {})}
        ch = _with(world, name, dict(base, segments=[{"from": 4.1, "to": 9.1, "rate": "8ths", "camera": cam}]))
        assert not ch.errors, [str(i) for i in ch.issues]
        photos = ch.photos()
        out[name] = (ch, photos, plan(ch, photos, History(ch).figure_boxes(), "16:9"))
    ch, photos, cams = out["vlook"]
    flat = out["vnolook"][2]
    g = ch.grid
    qa_, qp, qb = (g.resolve(a)[0] for a in (5.1, 6.1, 7.1))
    for ph, a, b in zip(photos, cams, flat):
        if ph.q < qa_ or ph.q >= qb:  # outside it the box framing, untouched (the push sits on top of it)
            assert a.look == 0 and (a.cx, a.cy, a.cw) == pytest.approx((b.cx, b.cy, b.cw), abs=1e-6)
    ws = [(ph.q, a.look) for ph, a in zip(photos, cams) if qa_ <= ph.q < qb]
    up = [w for q, w in ws if q <= qp]
    down = [w for q, w in ws if q >= qp]
    assert all(y >= x for x, y in zip(up, up[1:])) and all(y <= x for x, y in zip(down, down[1:]))
    peak = next(a for ph, a in zip(photos, cams) if abs(ph.q - qp) < 1e-6)
    flat_peak = next(b for ph, b in zip(photos, flat) if abs(ph.q - qp) < 1e-6)
    assert peak.look == 1.0 and peak.cw < 0.6 * flat_peak.cw  # pushed in on the region
    x0, y0, x1, y1 = region
    assert peak.cx - peak.cw / 2 <= x0 and peak.cx + peak.cw / 2 >= x1  # the region stays in frame
    assert peak.cy - peak.ch / 2 <= y0 and peak.cy + peak.ch / 2 >= y1


def test_burst_leap_path_goes_through_its_points_and_tilts_along_it(world, boxclip):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts", "pause": False,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "box", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 300, "thick": 14}]}},
                   {"burst": {"box": "box", "at": 5.1, "sequence": "burst_BXC", "frame": "box", "from": "4.4.3",
                              "windup_on": "4.4.3", "run_sequence": "run_SPR", "grow_from": "dive",
                              "leap": {"sequence": "leap_BXC", "keys": {"launch": "5.1.2", "dive": "5.1.4"},
                                       "land": "5.4.1", "x": {"land": 1900},
                                       "path": [["through", None, None], ["5.2.1", 1700, 300],
                                                ["5.3.1", 1800, 330], ["land", None, None]],
                                       "tilt": {"max": 50}},
                              "to": "5.4.1"}}]}
    ch = _with(world, "vbpath", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if e.layer == "wall" and e.pose and e.pose.startswith("BXC-")], key=lambda e: e.q)
    by = {g.addr(e.q): e for e in ev}
    assert (by["5.2.1"].x, by["5.2.1"].y) == pytest.approx((1700, 300), abs=1)  # through its points
    assert (by["5.3.1"].x, by["5.3.1"].y) == pytest.approx((1800, 330), abs=1)
    assert by["5.4.1"].x == pytest.approx(1900, abs=1)
    fl = [e for e in ev if g.resolve(5.1)[0] <= e.q <= g.resolve("5.4.1")[0]]
    assert all(b.x >= a.x for a, b in zip(fl, fl[1:]))
    assert min(e.y for e in fl) >= 300 - 1  # a monotone curve: no overshoot above its top
    rise = [e.rot for e in fl if g.resolve("5.1.2")[0] <= e.q < g.resolve("5.2.1")[0]]
    fall = [e.rot for e in fl if g.resolve("5.3.2")[0] <= e.q < g.resolve("5.4.1")[0]]
    assert max(rise) > 5 and min(fall) < -5  # nose up rising, nose down diving
    assert all(abs(e.rot) <= 50 + 1e-6 for e in fl)
    assert by["5.1.1"].rot == 0 and by["5.4.1"].rot == 0  # the break-through and the touchdown upright


def test_red_lights_default_to_no_glow_on_the_paint():
    from aimv.paint.render import light_opt
    red, white = {"color": [1.0, 0.07, 0.05]}, {"color": [1.0, 0.95, 0.85]}
    assert (light_opt(red, "paint"), light_opt(red, "neutral")) == (0.0, 1.0)  # white primer stays white
    assert (light_opt({}, "paint"), light_opt({}, "neutral")) == (0.0, 1.0)  # no colour: the red light
    assert (light_opt(white, "paint"), light_opt(white, "neutral")) == (1.0, 0.0)
    assert light_opt(dict(red, paint=0.5), "paint") == 0.5  # set: as set


def test_skin_move_argument_makes_its_wall_poses_air_skins(world):
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}]}
    ch = _with(world, "vskin", dict(base, moves=[{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600, "skin": True}}]))
    ev = [e for e in ch.events if e.pose == "RUN-1"]
    assert len(ev) == 1 and ev[0].layer == "air" and ev[0].flags.get("skin") is True
    ch = _with(world, "vskin0", dict(base, moves=[{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}}]))
    assert [e.layer for e in ch.events if e.pose == "RUN-1"] == ["wall"]


# ---------------------------------------------------------------------------------------------- verse 2: screens, glass

def _screens_street(world):
    """A copy of the street whose plate P6 has an `off` twin (a dark block where its screens were), one screens group
    over it and a shop window of glass."""
    import shutil
    src = world["street"]
    dd = src.parent / "scr"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    for sub in ("tiles", "strip"):
        if not (dd / sub).exists():
            shutil.copytree(src / sub, dd / sub)
    root = world["root"]
    P = cv2.imread(str(root / "boards" / "P6.png"))
    P[250:550, 500:900] = (P[250:550, 500:900] * 0.3).astype(np.uint8)
    cv2.imwrite(str(root / "boards" / "P6-off.png"), P)
    build = json.loads((dd / "build.json").read_text())
    x6 = int(round(next(p["offset"][0] for p in build["strip"]["plates"] if p["path"].endswith("P6.png"))))
    x6 -= build["strip"]["crop"][0]
    ob = json.loads((src / "obstacles.json").read_text())
    ob["screens"] = {"plates": {"boards/P6.png": "boards/P6-off.png"}, "top": 200, "feather": 40,
                     "groups": [{"id": "S", "x": [x6 + 400, x6 + 1000]}]}
    ob["glass"] = [{"id": "S", "box": [x6 + 450, 240, x6 + 950, 600]}]
    (dd / "obstacles.json").write_text(json.dumps(ob))
    return "scr", x6


def test_screens_off_bands_match_the_street_and_switch_off(world):
    from aimv.paint import screens as SCR
    from aimv.paint import street as STR
    name, x6 = _screens_street(world)
    dd = world["street"].parent / name
    info = SCR.build(world["root"], dd, log=lambda m: None)
    g = info["groups"]["S"]
    off = cv2.cvtColor(cv2.imread(str(dd / "screens" / "S.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
    on = STR.Street(dd, world=False).window(g["x0"], g["x1"]).astype(np.float32)
    dark = np.zeros(on.shape[:2], bool)
    dark[250 - 3:550 + 3, x6 + 500 - 3 - g["x0"]:x6 + 900 + 3 - g["x0"]] = True  # where P6-off is darkened
    assert np.abs(off - on)[~dark].mean() < 2.0  # the off street lines up with the street everywhere else
    inner = np.zeros_like(dark)
    inner[270:530, x6 + 520 - g["x0"]:x6 + 880 - g["x0"]] = True
    assert (off[inner].mean() + 1) / (on[inner].mean() + 1) == pytest.approx(0.3, abs=0.06)
    S = SCR.Screens(dd, json.loads((dd / "obstacles.json").read_text())["screens"])
    x0, y0, w, h = x6 + 300, 0, 900, 860
    lin = np.full((h, w, 3), 0.5, np.float32)
    E = np.ones((h, w), np.float32)
    a, b, d = S.blend(lin.copy(), E.copy(), x0, y0, {"S": (10.0, 0.0)}, 9.0)  # before its time: untouched
    assert np.array_equal(a, lin) and np.array_equal(b, E) and d.max() == 0
    a, b, d = S.blend(lin.copy(), E.copy(), x0, y0, {"S": (10.0, 0.0)}, 10.0)
    assert d[600, x6 + 700 - x0] == 1.0 and d[100, x6 + 700 - x0] == 0.0 and d[600, 20] == 0.0  # its band, from top
    assert b[400, x6 + 700 - x0] < 0.6 and b[400, 20] == 1.0  # the paint's light drops where the screens died
    assert np.allclose(a[100], 0.5) and not np.allclose(a[600, x6 + 700 - x0], 0.5)


def test_screens_template_switches_windows_off_one_a_beat(world):
    name, _ = _screens_street(world)
    d = {"street": name, "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"screens": {"order": ["S"], "from": 4.3, "fade": 1}},
                   {"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}}]}
    ch = _with(world, "vscreens", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    assert ch.screens_off == {"S": (pytest.approx(ch.grid.resolve(4.3)[0]), 1.0)}
    d["moves"][0] = {"screens": {"off": {"T": 4.3}}}
    assert any("no group 'T'" in str(i) for i in _with(world, "vscreens2", d).issues)


def test_paint_on_glass_is_a_cut_out_without_primer_or_ghosts(world):
    name, x6 = _screens_street(world)
    gx = x6 + 700
    d = {"street": name, "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": gx, "y": 420}},
                   {"pose": {"at": "4.1.3", "pose": "RUN-3", "x": gx + 60, "y": 420}}]}
    ch = _with(world, "vglass", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    r = Renderer(ch, hist)
    x0, y0, w, h = gx - 300, 150, 600, 500
    G = r._glass(x0, y0, w, h)
    assert G is not None and G[300 - y0, gx - x0] == 1.0 and G[700 - y0 if 700 - y0 < h else h - 1, 0] == 0.0
    t1 = ch.grid.t(ch.grid.resolve("4.1.1")[0]) + 0.01
    Pm, Cv, _ = hist.layers(t1, x0, y0, w, h)
    Pg, Cg, _ = hist.layers(t1, x0, y0, w, h, glass=G)
    on = G > 0.99
    fig = Pg / np.maximum(Cg, 1e-4)
    assert (Cv[on] > 0.2).sum() > (Cg[on] > 0.2).sum() * 1.3  # no primer patch round him on the glass
    assert fig[on & (Cg > 0.5)].max() < 0.1  # what is left there is his black paint
    t2 = ch.grid.t(ch.grid.resolve("4.1.3")[0]) + 0.01
    _, C2, _ = hist.layers(t2, x0, y0, w, h, glass=G)
    first = hist.mask(next(m for m in hist.marks if m.kind == "pose")) > 0.5  # the first pose's own paint
    m0 = next(m for m in hist.marks if m.kind == "pose")
    ys, xs = np.nonzero(first)
    ys, xs = ys + int(round(m0.y0)) - y0, xs + int(round(m0.x0)) - x0
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    ys, xs = ys[keep], xs[keep]
    own2 = np.zeros((h, w), bool)
    m1 = [m for m in hist.marks if m.kind == "pose"][1]
    y1s, x1s = np.nonzero(hist.mask(m1) > 0.05)
    y1s, x1s = y1s + int(round(m1.y0)) - y0, x1s + int(round(m1.x0)) - x0
    ok = (y1s >= 0) & (y1s < h) & (x1s >= 0) & (x1s < w)
    own2[y1s[ok], x1s[ok]] = True
    sel = on[ys, xs] & ~own2[ys, xs]
    assert sel.sum() > 50 and C2[ys[sel], xs[sel]].mean() < 0.01  # the first pose left no ghost on the glass
    assert C2[ys[sel], xs[sel]].max() < 0.1  # (the second one's drips at most)
    img = r.photo(0, t1, ch.grid.resolve("4.1.1")[0], plan(ch, ch.photos(), hist.figure_boxes(), "16:9")[0],
                  (480, 270))
    assert np.isfinite(img).all()


def test_a_stencil_turned_a_quarter_stands_up_the_pillar():
    flat = PR.Stencil(0.0, ["A BETTER FACE"], 1000, 400, 44, "center", 0.0, 5)
    up = PR.Stencil(0.0, ["A BETTER FACE"], 1000, 400, 44, "center", -90.0, 5)
    pf, hf, xf, yf = PR.stencil_mask(flat)
    pu, hu, xu, yu = PR.stencil_mask(up)
    assert hf.shape[1] > 3 * hf.shape[0] and hu.shape[0] > 3 * hu.shape[1]  # the whole line, standing up
    assert hu.sum() == pytest.approx(hf.sum(), rel=0.02)  # nothing cropped
    assert abs((xu + hu.shape[1] / 2) - 1000) < 2 and yu < 400  # centred on x, its top at y (less the padding)
    tilt = PR.Stencil(0.0, ["KEEP"], 1000, 400, 44, "left", 1.0, 5)  # a small tilt: as before
    assert PR.stencil_mask(tilt)[1].shape == PR.stencil_mask(PR.Stencil(0.0, ["KEEP"], 1000, 400, 44, "left", 0.0,
                                                                         5))[1].shape


def test_white_stencil_letters_paint_primer_white(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"stencil": {"at": 4.1, "text": "A BETTER FACE", "x": 900, "y": 200, "h": 40, "albedo": 0.7,
                                "punch": 0}},
                   {"stencil": {"at": 4.1, "text": "A BETTER FACE", "x": 900, "y": 400, "h": 40, "punch": 0}},
                   {"pose": {"at": 4.1, "pose": "RUN-1", "x": 1600}}]}
    ch = _with(world, "vwhite", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    t = ch.grid.t(ch.grid.resolve(4.3)[0])
    Pm, Cv, _ = hist.layers(t, 700, 150, 700, 400)
    alb = Pm / np.maximum(Cv, 1e-4)
    top, low = Cv[40:120] > 0.6, Cv[240:320] > 0.6
    assert top.sum() > 200 and low.sum() > 200
    assert alb[40:120][top].mean() > 0.5 and alb[240:320][low].mean() < 0.05  # white letters, black letters


def test_dead_glass_takes_primer_for_what_is_painted_after(world):
    name, x6 = _screens_street(world)
    gx = x6 + 700
    d = {"street": name, "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": gx, "y": 420}},
                   {"screens": {"off": {"S": 4.3}, "fade": 0}}]}
    ch = _with(world, "vdead", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    r = Renderer(ch, hist)
    x0, y0, w, h = gx - 300, 150, 600, 500
    Gw = r._glass(x0, y0, w, h, per=True)
    assert len(Gw) == 1 and Gw[0][1] == pytest.approx(ch.grid.t(ch.grid.resolve(4.3)[0]))
    t1 = ch.grid.t(ch.grid.resolve("4.1.1")[0]) + 0.01
    _, C0, _ = hist.layers(t1, x0, y0, w, h)  # as on a wall
    _, Clit, _ = hist.layers(t1, x0, y0, w, h, glass=[(Gw[0][0], math.inf)])  # the screens never die
    _, Cdead, _ = hist.layers(t1, x0, y0, w, h, glass=[(Gw[0][0], t1 - 1.0)])  # painted after they died
    on = Gw[0][0] > 0.99
    assert (C0[on] > 0.2).sum() > (Clit[on] > 0.2).sum() * 1.3
    assert np.allclose(Cdead, C0)  # dead glass: the primer behind him, as on the wall


def test_a_cycle_takes_more_steps_a_beat(world):
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "8ths"}]}
    out = {}
    for name, pb in (("vpb1", 1), ("vpb2", 1.75)):
        mv = {"sequence": "run_SPR", "from": 4.1, "to": "5.4.3", "from_x": 600}
        if pb != 1:
            mv["per_beat"] = pb
        ch = _with(world, name, dict(base, moves=[{"run": mv}]))
        assert not ch.errors, [str(i) for i in ch.issues]
        ev = sorted([e for e in ch.events if e.tag == "run"], key=lambda e: e.q)
        out[name] = (ev[-1].x - ev[0].x, [e.pose for e in ev])
    assert out["vpb2"][0] == pytest.approx(1.75 * out["vpb1"][0], rel=0.02)  # stride-true: 1.75 times the travel
    poses = out["vpb2"][1]
    assert all(not (a == c and b == d) for a, b, c, d in zip(poses, poses[1:], poses[2:], poses[3:]))  # no flip-flop


def test_the_camera_eases_across_a_join_between_files(world):
    cd = world["root"] / "paint" / "choreo"
    a = {"street": "main", "seed": 3, "motion": "m0",
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths", "camera": {"preset": "follow", "zoom": 1.0}}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 600}}]}
    out = {}
    for name, cam in (("jb", {"preset": "follow", "zoom": 2.0}), ("jc", {"preset": "follow", "zoom": 2.0, "cut": True})):
        b = {"street": "main", "seed": 3, "motion": "m0",
             "segments": [{"from": 5.1, "to": 6.1, "rate": "16ths", "camera": cam}],
             "moves": [{"pose": {"at": 5.1, "pose": "RUN-2", "x": 700}}]}
        (cd / "ja.yaml").write_text(yaml.safe_dump(a, sort_keys=False))
        (cd / f"{name}.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
        ch = _with(world, f"{name}film", {"include": ["ja", name], "street": "main"})
        assert not ch.errors, [str(i) for i in ch.issues]
        assert [s.join for s in ch.segments] == [False, True]
        photos = ch.photos()
        cams = plan(ch, photos, History(ch).figure_boxes(), "16:9")
        q5 = ch.grid.resolve(5.1)[0]
        out[name] = [c.cw for ph, c in zip(photos, cams) if q5 - 1 <= ph.q <= q5 + 6]
    w = out["jb"]  # the photo before the join, the four across it, then on
    assert w[1] > w[0] * 1.05 and w[2] > w[1] and w[3] > w[2] and w[4] > w[3]  # eased out to the wider frame
    assert w[1] < 0.6 * w[5]  # not a jump: the first photo after the join is still near the old frame
    c = out["jc"]
    assert c[1] > 0.6 * c[5]  # `cut: true`: no ease, the new frame at once (as fast as the smoothing goes)


def test_the_join_ease_keeps_a_moving_camera_moving(world, boxclip):
    cd = world["root"] / "paint" / "choreo"
    run = {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "4.4.4",
           "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}
    a = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths", "camera": "spring"}],
         "moves": [{"parkour": run}]}
    b = {"street": "main", "seed": 3, "segments": [{"from": 5.1, "to": 6.1, "rate": "16ths", "camera": "spring"}],
         "moves": [{"parkour": dict(run, **{"from": 5.1, "v0": "carry", "to": "5.4.4"})}]}
    b["moves"][0]["parkour"].pop("from_x")
    (cd / "jma.yaml").write_text(yaml.safe_dump(a, sort_keys=False))
    (cd / "jmb.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
    ch = _with(world, "jmfilm", {"include": ["jma", "jmb"], "street": "main"})
    assert not ch.errors, [str(i) for i in ch.issues]
    photos = ch.photos()
    cams = plan(ch, photos, History(ch).figure_boxes(), "16:9")
    k = next(i for i, p in enumerate(photos) if abs(p.q - ch.grid.resolve(5.1)[0]) < 1e-6)
    dx = [cams[i].cx - cams[i - 1].cx for i in range(k - 2, k + 4)]
    assert min(dx) > 0.6 * max(dx), dx  # it carries on at his pace across the join (no stop for a photo)


def test_a_stencil_fades_out_and_is_gone(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"stencil": {"at": 4.1, "text": "TURN IT OFF", "x": 900, "y": 300, "h": 40, "punch": 0,
                                "fade": {"at": 5.1, "over": 4}}},
                   {"pose": {"at": 4.1, "pose": "RUN-1", "x": 1600}}]}
    ch = _with(world, "vfade", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    g = ch.grid
    cov = [hist.layers(g.t(g.resolve(a)[0]) + 0.01, 700, 250, 500, 150)[1].sum() for a in ("4.4", "5.1.3", "5.3")]
    assert cov[0] > 500 and 0 < cov[1] < 0.8 * cov[0] and cov[2] == 0  # there, fading, gone


# ---------------------------------------------------------------------------------------------- the true look: green

def _green_frame(h=400, w=600, figs=((150, 120),), seed=0):
    """A frame on chroma green with dark upright figures (a head and a body) at (cx, top)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 190, 50)  # BGR green
    for cx, top in figs:
        cv2.circle(img, (cx, top + 25), 22, (30, 28, 26), -1)
        cv2.rectangle(img, (cx - 35, top + 45), (cx + 35, top + 230), (45, 40, 38), -1)
        cv2.rectangle(img, (cx - 30, top + 230), (cx + 30, top + 250), (210, 210, 210), -1)  # pale shoes
    return img


def test_a_green_sheet_splits_by_the_key_and_keeps_its_colour():
    img = _green_frame(figs=((150, 60), (450, 80)))
    boxes = []
    parts = PS.split(img, 2, 2, "pose", "TST", screen="green", boxes=boxes)
    assert len(parts) == 2 and len(boxes) == 2
    for dd, (y0, y1, x0, x1) in zip(parts, boxes):
        assert dd.shape == (y1 - y0, x1 - x0)
        assert dd.max() > 0.9 and (dd > 0.5).sum() > 8000  # the whole figure, the pale shoes too (not paint density)
    assert boxes[0][2] < 300 < boxes[1][2]  # reading order
    with pytest.raises(SystemExit):
        PS.split(img, 2, 2, "pose", "TST", screen="blue")


def test_a_green_clip_gives_its_frames_colour_crops():
    frames = [(f, f / 30, _green_frame(figs=((150 + 20 * f, 80),))) for f in range(4)]
    rgbs: dict = {}
    mats, info = PS.video_split(frames, {"screen": "green", "dedupe": {"iou": 0.99, "still": 0.99}}, "TSV", rgbs)
    assert len(mats) == 4 and sorted(rgbs) == [0, 1, 2, 3]
    for f, t, mat, ox, oy in mats:
        assert rgbs[f].shape[:2] == mat.shape and rgbs[f].shape[2] == 3
        assert rgbs[f][..., 1].astype(int).max() <= max(int(rgbs[f][..., 0].max()), int(rgbs[f][..., 2].max())) + 7


def test_canon_skips_the_paint_rules_for_a_screen_source():
    from aimv.paint import canon as CN
    names = [f"S-{k}" for k in range(8)]
    met = {n: {"area": 1000.0, "smear": 0.01, "shoes": 0.2, "colour": 0.0, "pieces": 0.0} for n in names}
    met["S-3"].update(smear=0.9, colour=0.5)
    canon = {"area": 1000.0}
    _, fl = CN.judge("S", names, met, canon, clip=True)
    assert {r for _, r, _ in fl["S-3"]} >= {"smear", "colour"}
    _, fl = CN.judge("S", names, met, canon, clip=True, screen=True)
    assert not any(r in ("smear", "colour", "shoes") for _, r, _ in fl["S-3"])


def test_time_travel_spreads_a_tracked_cycle_evenly():
    st = PS.time_travel([10, 11, 13, 14, 16, 17, 19], 19, 300.0, 2)
    assert st["step"] == 300.0 and st["period"] == 600.0 and st["period_f"] == 10
    assert st["travel"] == [0.0, 60.0, 180.0, 240.0, 360.0, 420.0, 540.0]  # by frame time: in order, even
    assert PS.time_travel([10, 12], None, 100.0, 2)["period_f"] == 3  # no range end: the last kept frame's next
    with pytest.raises(SystemExit):
        PS.time_travel([10, 12], 12, 0.0, 2)


def test_a_green_clips_guide_stays_out_of_the_screen_colour():
    from aimv.paint.greenkey import screen_colour
    img = _green_frame(figs=((300, 80),))
    img[:, :200] = (236, 203, 172)  # a pale blue guide block down the whole left edge (BGR)
    km = np.zeros(img.shape[:2], bool)
    km[:, :200] = True
    assert list(screen_colour(img, exclude=km)) == [40, 190, 50]  # the green, not a blend with the guide
    frames = [(f, f / 30, img.copy()) for f in range(2)]
    mats, _ = PS.video_split(frames, {"screen": "green", "key": [172, 203, 236]}, "TSG", {})
    assert mats and all(m.shape[1] < 200 for _, _, m, _, _ in mats)  # the guide keyed out, only the figure kept


def test_renderer_photo_shows_the_true_insert_inside_its_stretch(world):
    from aimv.paint import truelook as TL
    from aimv.paint.camera import Cam
    clip = world["root"] / "boards" / "insert-r.mp4"
    wr = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(30):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:, :320] = (0, 0, 200)  # red left half (BGR), blue right half
        img[:, 320:] = (200, 0, 0)
        wr.write(img)
    wr.release()
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"true_insert": {"clip": "boards/insert-r.mp4", "from": 4.1, "to": 5.1, "aspect": "16:9",
                                    "camera": [1000.0, 500.0, 640.0, 360.0, 0.0]}},
                   {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]}
    ch = _with(world, "vins_r", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    r = Renderer(ch, History(ch))
    ins = ch.true_inserts[0]
    cam = Cam(1000.0, 500.0, 640.0, 360.0, 0.0)
    t_in = ins["t0"] + 0.01
    a = r.photo(1, t_in, ch.grid.q(t_in), cam, (640, 360))
    assert np.allclose(a, TL.insert_photo(r, 1, t_in, ch.grid.q(t_in), cam, (640, 360)))  # the insert itself
    assert a[200, 100, 0] > 0.6 and a[200, 540, 2] > 0.6
    t_out = ins["t1"] + 0.2
    b = r.photo(2, t_out, ch.grid.q(t_out), cam, (640, 360))  # after the stretch: the street as usual
    assert not (b[200, 100, 0] > 0.6 and b[200, 540, 2] > 0.6)


def test_an_insert_cuts_to_a_full_frame_clip_at_its_own_speed_synced_by_one_frame(world):
    from aimv.paint import truelook as TL
    from aimv.paint.camera import Cam
    clip = world["root"] / "boards" / "insert-cut.mp4"
    wr = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (360, 640))  # a 9:16 clip
    for i in range(60):
        img = np.full((640, 360, 3), 4 * i, np.uint8)  # each frame its own grey
        img[:60] = (0, 0, 220)  # a red band along its top (BGR)
        wr.write(img)
    wr.release()
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"insert": {"clip": "boards/insert-cut.mp4", "from": 4.2, "to": 4.4, "aspect": "16:9",
                               "sync": [30, 4.3]}},
                   {"pose": {"at": 4.2, "pose": "RUN-1", "x": 700}}]}
    ch = _with(world, "vins_cut", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ins = ch.true_inserts[0]
    g = ch.grid
    t43 = g.t(g.resolve(4.3)[0])
    assert TL.insert_frame_index(ins, t43) == 30  # the sync frame on its time
    assert TL.insert_frame_index(ins, t43 + 0.2) == 36  # then at the clip's own 30 fps
    r = Renderer(ch, History(ch))
    cam = Cam(1000.0, 500.0, 640.0, 360.0, 0.0)
    a = r.photo(3, t43, g.q(t43), cam, (640, 360))
    assert abs(float(np.median(a[100:260, 200:440])) - 120 / 255) < 0.08  # frame 30's grey fills the photo
    assert a[:30, :, 0].mean() < 0.6  # covered: scaled to the 16:9's width, cropped round its middle (no red band)
    out = r.photo(9, g.t(g.resolve(4.4)[0]) + 0.05, g.resolve(4.4)[0] + 0.5, cam, (640, 360))
    assert abs(float(np.median(out[100:260, 200:440])) - float(np.median(a[100:260, 200:440]))) > 0.02  # after it
    d["moves"][0]["insert"]["sync"] = [5, 4.3]  # (from 4.2 would need frame -5)
    ch = _with(world, "vins_cut2", d)
    assert any("needs frames" in str(i) for i in ch.issues)


def test_a_still_lamp_is_round_over_its_own_plate_and_the_true_figure_goes_over_it(world):
    from aimv.paint.camera import Cam
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"true": {"from": 4.1}}, {"pose": {"at": 4.1, "pose": "RUN-1", "x": 900}}]}
    ch = _with(world, "vlamp", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    i = next(iter(hist.true_q.values()))[-1]
    from aimv.paint import truelook as TL
    _, a, fx0, fy0 = TL.place_rgba(hist, i)
    dist = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)  # the figure's deepest point: the lamp's centre
    cx, cy = int(round(fx0)) + int(px), int(round(fy0)) + int(py)
    sc = ch.scene
    box = [cx - 150, cy - 150, cx + 150, cy + 150]  # full alpha within 60 px of its centre
    sc.layers += [{"id": "vlamp-head", "parallax": 0.0, "reach": 8, "x": cx, "parts": [{"box": box}]},
                  {"id": "vlamp-pole", "parallax": 0.0, "reach": 8, "x": cx, "parts": [{"box": box}]}]
    sc.lights.append({"id": "vlamp", "x": cx, "y": cy, "sigma": 160, "color": [1, 0.08, 0.06], "layer": "vlamp-head"})
    try:
        head, pole = sc.layer(sc.layers[-2]), sc.layer(sc.layers[-1])
        assert head["bg"] is head["fg"] and pole["bg"] is pole["fg"]  # nothing rebuilt behind a layer that never moves
        o = lambda L, x, y: float(L["alpha"][y - L["y0"], x - L["a0"]])
        assert o(head, cx, cy) > 0.99 and o(head, box[0] + 2, box[1] + 2) == 0 and o(head, cx, box[1] + 1) < 0.01
        assert o(pole, box[0] + 4, box[1] + 4) > 0.9  # a layer that is not a light's lamp keeps its rect
        head["fg"][...] = (1.0, 0.0, 0.0)  # the lamp's plate bright red (and so the street behind it)
        head.pop("fg_lin", None)
        sc.layers.pop()
        r = Renderer(ch, hist)
        q = next(iter(hist.true_q))
        img = r.photo(0, ch.grid.t(q), q, Cam(float(cx), float(cy), 640.0, 360.0, 0.0), (640, 360))
        assert img[180, 320, 0] < 0.4  # his body over the lamp
        ys, xs = np.nonzero(a < 0.01)
        near = [(y, x) for y, x in zip(ys, xs) if 10 < math.hypot(y - py, x - px) < 55]
        y, x = near[0]
        assert img[180 + y - py, 320 + x - px, 0] > 0.8  # the lamp beside him
    finally:
        sc.layers[:] = [L for L in sc.layers if not str(L["id"]).startswith("vlamp")]
        sc.lights.pop()


def test_a_parallax_layer_rebuilds_the_street_behind_it_with_the_walls_texture(world):
    sc = _with(world, "vkerb", {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 5.1}]}).scene
    sc.layers.append({"id": "vkerb-pole", "parallax": 0.1, "x": 1200, "parts": [{"box": [1180, 100, 1220, 600]}]})
    try:
        d = sc.layer(sc.layers[-1])
    finally:
        sc.layers.pop()
    row, c = d["bg"][300], 1200 - d["a0"]
    grad = lambda a: float(np.abs(np.diff(a, axis=0)).mean())
    assert grad(row[c - 30:c + 30]) > 0.5 * grad(row[:c - 60])  # textured like the wall, not a flat blend
    assert np.abs(row[c - 40] - d["fg"][300, c - 40]).max() < 1e-6  # the street beside the gap untouched


def test_check_skips_the_no_paint_warning_for_a_true_figure(world):
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}]}
    pose = {"pose": {"at": 4.2, "pose": "RUN-1", "x": 2000, "y": 800}}  # down on the pavement (never paintable)
    msgs = lambda ch: [str(i) for i in check(world["p"], ch)["issues"] if "paint cannot go" in str(i)]
    assert msgs(_with(world, "vnp_paint", dict(base, moves=[pose])))  # paint: warned
    assert not msgs(_with(world, "vnp_true", dict(base, moves=[{"true": {"from": 4.1}}, pose])))  # true: not


def test_template_framing_takes_its_own_9x16_zoom_and_offsets(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 1600}}]}
    ch = _with(world, "vzoomv", d)
    photos = ch.photos()
    q = round(photos[4].q, 4)
    boxes = History(ch).figure_boxes()
    out = {}
    for name, hint in (("plain", {"dx": 0.0, "zoom": 1.0, "snap": True}),
                       ("v", {"dx": 0.0, "zoom": 1.0, "snap": True, "zoom_v": 0.7, "dx_v": 40.0, "dy_v": -30.0})):
        ch.cam_hints[q] = hint
        out[name] = {a: plan(ch, photos, boxes, a)[4] for a in ("16:9", "9:16")}
    a, b = out["plain"]["16:9"], out["v"]["16:9"]
    assert (a.cx, a.cy, a.cw) == pytest.approx((b.cx, b.cy, b.cw))  # 16:9 ignores the _v keys
    pv, vv = out["plain"]["9:16"], out["v"]["9:16"]
    assert vv.cw < pv.cw * 0.8 and vv.cx != pytest.approx(pv.cx)  # the 9:16's own zoom (closer) and offset


def test_fill_holes_closes_a_matte_gap_only_when_asked(world):
    from aimv.paint.render import fill_holes
    ring = np.zeros((60, 60), np.float32)
    ring[10:50, 10:50] = 0.9
    ring[25:35, 25:35] = 0.0  # a 10 x 10 gap inside the paint (the wall showing through)
    assert fill_holes(ring, 200)[30, 30] > 0.8 and fill_holes(ring, 50)[30, 30] == 0.0  # only gaps up to n px
    assert np.array_equal(fill_holes(ring, 200)[:, :8], ring[:, :8])  # the outside untouched
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 4.3, "rate": "16ths"}]}
    for name, extra, want in (("vfh0", {}, None), ("vfh1", {"fill_holes": 400}, 400)):
        ch = _with(world, name, dict(d, moves=[{"pose": {"at": 4.1, "pose": "RUN-1", "x": 900, **extra}}]))
        e = next(e for e in ch.events if e.pose == "RUN-1")
        assert e.flags.get("fill_holes") == want  # the flag only with the arg; the sprite as before without it


def test_a_pose_painted_by_brush_strokes_appears_stroke_by_stroke(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 900, "still": True, "over": 14,
                             "strokes": [{"at": 4.1, "box": [0.2, 0.1, 0.8, 0.6], "angle": -15},
                                         {"at": 4.2, "box": [0.0, 0.5, 1.0, 1.0], "angle": 20},
                                         {"at": 4.3, "box": [0.3, 0.0, 0.9, 0.3], "angle": 10, "over": 30,
                                          "sweep": 0.8}]}}]}
    ch = _with(world, "vstrokes", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    e = next(e for e in ch.events if e.pose == "RUN-1")
    assert len(e.flags["strokes"]) == 3 and e.flags["patch"] == 0.0
    hist = History(ch)
    g = ch.grid
    m = next(m for m in hist.marks if m.kind == "pose")
    assert m.id in hist.reveal
    box = hist.fig_box[e.q]
    x0, y0, w, h = int(box[0]) - 60, int(box[1]) - 60, int(box[2] - box[0]) + 120, int(box[3] - box[1]) + 160
    cov = []
    for a in ("4.1.1", "4.1.3", "4.2.1", "4.3.1", "4.4.1"):
        _, C, _ = hist.layers(g.t(g.resolve(a)[0]) + 0.001, x0, y0, w, h)
        cov.append(float(C.sum()))
    assert 0 < cov[0] < cov[2] < cov[3]  # stroke by stroke
    full = hist.mask(m)
    _, C, _ = hist.layers(g.t(g.resolve("4.4.1")[0]), x0, y0, w, h)
    assert C.sum() >= 0.95 * full.sum()  # whole by the last stroke (with the overshoot round him)
    over = [mk for mk in hist.marks if mk.kind == "spatter"]
    assert len(over) == 3 and all(hist.fx[mk.id]["mask"].max() > 0.5 for mk in over)  # the overshoot stays
    bands = [mk for mk in hist.marks if mk.kind == "remains"]
    assert len(bands) == 3 and all(b.t < o.t for b, o in zip(bands, over))  # a white swath under each stroke first
    assert all(hist.fx[b.id].get("swath") for b in bands)
    assert all(hist.fx[b.id]["mask"].sum() > 1.5 * hist.fx[o.id]["mask"].sum() for b, o in zip(bands, over))  # broad
    assert e.flags["strokes"][2][3] == 30 and e.flags["strokes"][2][6] == 0.8  # a stroke's own over and sweep
    sw_m, al, _ = hist._fx_state(bands[0], g.t(g.resolve("4.4.1")[0]), ch.paint_clock(g.t(g.resolve("4.4.1")[0])))
    assert al >= 0.75 and sw_m.max() > 0.85  # bright white, staying
    t0 = g.t(g.resolve(4.1)[0])
    assert any(abs(dr[0] - (t0 - 0.06)) < 1e-6 for dr in hist.drops)  # drops already flying on the first photo
    d["moves"][0]["pose"]["strokes"][1]["at"] = 3.4
    assert any("before the pose" in str(i) for i in _with(world, "vstrokes2", d).issues)


def test_a_clip_with_back_steps_back_as_its_clip_does(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.2, "b": 5.2}, "x": {"a": 700, "b": 1300},
                             "back": True, "trail": "light"}}]}
    ch = _with(world, "vback", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    ev = sorted([e for e in ch.events if e.tag.startswith("clip")], key=lambda e: e.q)
    assert ev and ev[0].x != ev[-1].x
    assert all(r.get("back") for r in figure_track(ch))  # its steps back are not QA backtracks
    d["moves"][0]["clip"].pop("back")
    assert not any(r.get("back") for r in figure_track(_with(world, "vback0", d)))


# ---------------------------------------------------------------------------------------------- v7: physics

def _pk(world, name, route, from_x, gait="sprint", seg_to=6.1, to="5.4.4", extra=None, camera="follow"):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": seg_to, "rate": "16ths", "camera": camera}],
         "moves": [{"parkour": {"from": 4.1, "from_x": from_x, "v0": 7.0, "gait": gait, "to": to, "route": route,
                                "cycle": "run_SPR", **(extra or {})}}]}
    return _with(world, name, d)


def _pk_alley_x(world, at=5.1, v_mps=7.0):
    """A start x that makes the run to the alley's take-off at `at` its gait's speed (the test figure: 125 px/m)."""
    g = SongGrid.from_project(world["p"])
    T = g.t(g.resolve(at)[0]) - g.t(g.resolve(4.1)[0])
    return 840 - v_mps * 125.0 * T - 40


def test_parkour_jump_is_ballistic_and_lands_past_the_gap_on_its_anchor(world, boxclip):
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk1", route, _pk_alley_x(world))
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    sp = ch.notes["move 1 (parkour)"]["spans"][0]
    assert sp["ok"] and 0.9 <= sp["factor"] <= 1.1
    evs = [e for e in ch.events if e.layer in ("wall", "air") and e.pose]
    tko = [e for e in evs if e.tag == "pk.takeoff"]
    assert len(tko) == 1 and abs(tko[0].q - g.resolve(5.1)[0]) < 1e-6  # the take-off photo on its anchor
    fl = [e for e in evs if e.tag == "pk.jump"]
    assert len(fl) >= 3 and all(e.layer == "air" and e.anchor == "cxc" for e in fl)
    ys = np.array([e.y for e in fl])
    dt = g.t(fl[1].q) - g.t(fl[0].q)
    acc = np.diff(ys, 2) / dt ** 2  # one gravity: 9.81 m/s2 at 125 px/m
    assert np.allclose(acc, 9.81 * 125, rtol=0.02)
    vx = np.diff([e.x for e in fl]) / dt
    assert np.ptp(vx) < 1.0  # his take-off speed kept through the air
    land = ch.notes["move 1 (parkour)"]["flights"][0]
    assert land["x_land"] > 1210  # past the far edge, where his feet met the ground
    run = [e for e in evs if e.tag == "pk.run"]
    assert all(abs(e.y - ch.scene.feet_at(e.x)) < 1e-6 for e in run)  # on the feet line
    assert len({e.pose for e in run}) > 4  # stride-true drawings, not one held


def test_parkour_span_too_short_says_how_much_route_it_needs(world, boxclip):
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk2", route, 640)
    msg = " ".join(str(i) for i in ch.errors)
    assert "too little" in msg and "px of route before it" in msg
    sp = ch.notes["move 1 (parkour)"]["spans"][0]
    assert not sp["ok"] and sp["short_px"] > 1000


def test_parkour_flip_turns_one_way_the_whole_turn(world, boxclip):
    route = [{"flip": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "dir": "front", "tuck": "RUN-3",
                       "rise": 1.2}}]
    ch = _pk(world, "vpk3", route, _pk_alley_x(world))
    assert not ch.errors, [str(i) for i in ch.issues]
    fl = [e for e in ch.events if e.tag == "pk.flip"]
    rot = np.unwrap(np.radians([e.rot for e in fl]))
    assert len(fl) >= 5 and np.all(np.diff(rot) <= 1e-6)  # front: clockwise on screen, never back
    assert np.degrees(rot[0] - rot.min()) > 250  # most of the full turn shown in the air
    assert sum(e.pose == "RUN-3" for e in fl) >= len(fl) - 2  # the tuck drawing turned by the engine


def test_parkour_jump_air_plays_runs_of_clip_frames_in_order_over_the_flight(world, boxclip):
    """v8 (P6's big leap): a jump's `air: [[sequence, first, last, seconds], ...]` shows those runs of drawings in
    order over the flight, each its share of the time (scaled to the flight's own), the last drawing landing."""
    lib = boxclip
    seqs = {n: lib.sequence(n) for n in ("leap_SPR", "leap_BXC")}
    fa = [lib.videos[seqs["leap_SPR"]["video"]]["frames"][k]["f"] for k in (2, 8)]
    fb = [lib.videos[seqs["leap_BXC"]["video"]]["frames"][k]["f"] for k in (0, 5)]
    air = [["leap_SPR", fa[0], fa[1], 0.3], ["leap_BXC", fb[0], fb[1], 0.1]]
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "air": air, "roll": False}}]
    ch = _pk(world, "vair1", route, _pk_alley_x(world))
    assert not ch.errors, [str(i) for i in ch.issues]
    fl = sorted((e for e in ch.events if e.tag == "pk.jump"), key=lambda e: e.q)
    vid_a, vid_b = seqs["leap_SPR"]["video"], seqs["leap_BXC"]["video"]
    src = [vid_a if e.pose in seqs["leap_SPR"]["drawings"] else vid_b for e in fl]
    assert src[0] == vid_a and src[-1] == vid_b  # the first run, then the second
    k = src.index(vid_b)
    assert all(v == vid_b for v in src[k:]) and 0.55 < k / len(src) < 0.95  # in order, about its share (3:1)
    bad = _pk(world, "vair2", [{"jump": dict(route[0]["jump"], air=[["leap_SPR", 1, 2]])}], _pk_alley_x(world))
    assert any("air item 1" in str(i) for i in bad.errors)


def test_parkour_clip_lift_moves_a_run_of_its_frames_down(world, barclip):
    """v8 (P6's precision stick, the crate step): a route clip's `lift: [[first, last, px]]` moves those frames down
    by px street px, the others where they were."""
    def pk(name, clip):
        d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 300, "v0": 0, "gait": "jog", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": "jog"}}]}}]}
        return _with(world, name, d)
    base = {"sequence": "swing_SPB", "speed": 1.0, "ground": "start", "wait": True, "keys": {"takeoff": "4.3", "release": "5.2"}}
    a, b = pk("vlift1", base), pk("vlift2", dict(base, lift=[[30, 40, 25]]))
    assert not a.errors and not b.errors, [str(i) for i in b.issues]
    tr = {r["name"]: r["f"] for r in barclip.videos["SPB"]["frames"]}
    ea = {round(e.q, 4): e for e in a.events if e.tag == "pk.clip"}
    eb = {round(e.q, 4): e for e in b.events if e.tag == "pk.clip"}
    moved = [q for q in ea if 30 <= tr[ea[q].pose] <= 40]
    assert moved and all(abs(eb[q].y - ea[q].y - 25) < 1e-6 for q in moved)
    assert all(abs(eb[q].y - ea[q].y) < 1e-6 for q in ea if not 30 <= tr[ea[q].pose] <= 40)


def test_parkour_flip_shows_a_tuck_drawn_mid_turn_as_drawn_at_the_apex(world, boxclip):
    """A tuck drawn part way round the flip (flip_FL's FL-048 head down: `turn: {tuck: 180}` on its sequence, or the
    step's `tuck_turn`) is turned that much less, so it shows as drawn at the middle of the turn and upright at its
    ends (no half-turn out of phase, no 180-degree snap where the take-off and landing drawings meet it)."""
    from aimv.paint.parkour import _tuck_turn
    base = {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "dir": "front", "tuck": "RUN-3", "rise": 1.2}
    a = _pk(world, "vpk3a", [{"flip": base}], _pk_alley_x(world))
    b = _pk(world, "vpk3b", [{"flip": dict(base, tuck_turn=180)}], _pk_alley_x(world))
    assert not a.errors and not b.errors, [str(i) for i in b.issues]
    ta = {round(e.q, 4): e for e in a.events if e.tag == "pk.flip" and e.pose == "RUN-3"}
    tb = {round(e.q, 4): e for e in b.events if e.tag == "pk.flip" and e.pose == "RUN-3"}
    assert ta and set(ta) == set(tb)
    assert all(abs(((tb[q].rot - ta[q].rot) % 360) - 180) < 1e-6 for q in ta)  # turned half a turn less
    qs = sorted(tb)
    mid = tb[qs[len(qs) // 2]]
    assert abs(mid.rot) < 60  # at the middle of the turn: as drawn
    assert abs(tb[qs[0]].rot) > 120 and abs(tb[qs[-1]].rot) > 120  # at its ends half a turn from as drawn: upright
    rot = np.unwrap(np.radians([e.rot for e in b.events if e.tag == "pk.flip"]))
    assert np.all(np.diff(rot[1:-1]) <= 1e-6)  # still one way through the tuck

    class _R:  # the library's own property: the flip sequence's turn for the key whose drawing the tuck is
        def seq(self, name):
            return {"keys": {"takeoff": 0, "tuck": 1}, "drawings": ["FL-032", "FL-048"], "turn": {"tuck": 180}}
    assert _tuck_turn(_R(), {"sequence": "flip_FL"}, "FL-048") == 180
    assert _tuck_turn(_R(), {"sequence": "flip_FL"}, "FL-050") == 0
    assert _tuck_turn(_R(), {"sequence": "flip_FL", "tuck_turn": 90}, "FL-048") == 90


def test_contact_qa_flags_floating_feet_and_a_takeoff_past_the_edge(world, boxclip):
    from aimv.paint import qa as QA
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk4", route, _pk_alley_x(world))
    hist = History(ch)
    photos = ch.photos()
    cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
    assert not [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] in ("feet", "takeoff")]
    run = [e for e in ch.events if e.tag == "pk.run"]
    run[3].y -= 30  # floating
    tko = next(e for e in ch.events if e.tag == "pk.takeoff")
    tko.x += 80  # the take-off foot past the edge
    hist = History(ch)
    rules = {f["rule"] for f in QA.run(ch, hist, photos, cams)["flags"] if f["level"] == "error"}
    assert {"feet", "takeoff"} <= rules


def test_contact_qa_flags_a_fist_through_the_box(boxed):
    from aimv.paint import qa as QA
    hist = History(boxed)
    photos = boxed.photos()
    cams = {"16:9": plan(boxed, photos, hist.figure_boxes(), "16:9")}
    before = [f for f in QA.run(boxed, hist, photos, cams)["flags"] if f["rule"] == "fist"]
    e = next(e for e in boxed.events if e.move >= 0 and boxed.moves[e.move].kind == "pound" and e.pose)
    x0 = e.x
    e.x += 400  # through the box's right side
    try:
        hist = History(boxed)
        after = [f for f in QA.run(boxed, hist, photos, cams)["flags"] if f["rule"] == "fist"]
    finally:
        e.x = x0
    assert len(after) > len(before)


def test_spring_camera_pans_smoothly_and_eases_its_zoom(world, boxclip):
    from aimv.paint import qa as QA
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths", "camera": {"preset": "spring"}},
                      {"from": 5.1, "to": 6.1, "rate": "16ths", "camera": {"preset": "spring", "zoom": 0.8}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "5.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    ch = _with(world, "vpk5", d)
    hist = History(ch)
    photos = ch.photos()
    for a in ("16:9", "9:16"):
        cams = plan(ch, photos, hist.figure_boxes(), a)
        snaps = [f for f in QA.run(ch, hist, photos, {a: cams})["flags"] if f["rule"] == "camera"]
        assert not snaps, [(f["addr"], f["detail"]) for f in snaps[:4]]
    cams = plan(ch, photos, hist.figure_boxes(), "16:9")
    w = np.array([c.base[2] for c in cams])
    k = next(i for i, p in enumerate(photos) if abs(p.q - ch.grid.resolve(5.1)[0]) < 1e-6)
    # eased over at least a beat, and long enough that no photo changes it by more than 4%
    assert w[k + 3] > w[k - 1] * 0.85 and w[k + 10] < w[k - 1] * 0.82 and np.all(np.abs(np.diff(w)) / w[:-1] <= 0.04)


def test_spring_camera_takes_a_frame_hints_offsets_target_and_ease(world, boxclip):
    from aimv.paint import qa as QA
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "6.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    plain = _with(world, "vpk5h0", d)
    d["moves"].append({"hints": {"frame": {"from": 4.3, "to": 5.4, "zoom": 1.25, "zoom_v": 1.15, "dx": 150, "dx_v": 30,
                                           "cy": 440, "follow_v": True, "ease": [1.6, 1.2]}}})
    ch = _with(world, "vpk5h1", d)
    photos = ch.photos()
    hist, hist0 = History(ch), History(plain)
    c, c0 = plan(ch, photos, hist.figure_boxes(), "16:9"), plan(plain, photos, hist0.figure_boxes(), "16:9")
    v, v0 = plan(ch, photos, hist.figure_boxes(), "9:16"), plan(plain, photos, hist0.figure_boxes(), "9:16")
    for a, cams in (("16:9", c), ("9:16", v)):
        snaps = [f for f in QA.run(ch, hist, photos, {a: cams})["flags"] if f["rule"] == "camera"]
        assert not snaps, [(f["addr"], f["detail"]) for f in snaps[:4]]
    t = np.array([p.t for p in photos])
    k0, k1 = (int(np.argmin(np.abs(t - ch.grid.t(ch.grid.resolve(a)[0])))) for a in ("4.3", "5.4"))
    w, w0 = np.array([x.base[2] for x in c]), np.array([x.base[2] for x in c0])
    k_in = int(np.argmin(np.abs(t - (t[k0] + 1.6))))
    assert w[k0 + 6] < 1.15 * w0[k0 + 6] and abs(w[k_in] / w0[k_in] - 1.25) < 0.01  # eased in over its 1.6 s
    k_out = int(np.argmin(np.abs(t - (t[k1 + 1] + 1.2))))
    assert w[k1 + 4] > 1.12 * w0[k1 + 4] and abs(w[k_out] / w0[k_out] - 1.0) < 0.01  # back over 1.2 s
    assert abs(c[k1].base[1] - 440) < 15 and abs(c0[k1].base[1] - 440) > 60  # settled on the fixed cy
    assert abs((c[k1].base[0] - c0[k1].base[0]) - 150) < 40  # dx ahead of the spring's own target
    wv, wv0 = np.array([x.base[2] for x in v]), np.array([x.base[2] for x in v0])
    assert abs(wv[k1] / wv0[k1] - 1.15) < 0.01  # the 9:16's own zoom
    assert abs(v[k1].base[1] - 440) > 25 and abs((v[k1].base[0] - v0[k1].base[0]) - 30) < 15  # it follows him


def test_frame_push_curve_in_gathers_speed_and_settles_on_its_end():
    """hints.ease_curve: `in` eases in and gathers speed, fastest just before the end (twice its mean pace), and
    settles onto its end value; `smooth` is the smoothstep as before."""
    from aimv.paint.hints import ease_curve
    u = np.linspace(0, 1, 401)
    e = np.array([ease_curve(x, "in") for x in u])
    assert e[0] == 0 and abs(e[-1] - 1) < 1e-9 and np.all(np.diff(e) >= -1e-12)
    pace = np.diff(e) / np.diff(u)
    k = int(np.argmax(pace))
    assert 0.85 < u[k] < 0.95 and abs(pace.max() - 2.0) < 0.05
    assert pace[0] < 0.01 and pace[-1] < 0.05 and pace[200] < pace[300] < pace[350]
    assert np.allclose([ease_curve(x) for x in u], u * u * (3 - 2 * u))


def test_spring_camera_follows_a_frame_push_centred_on_him(world, boxclip):
    """A frame's zoom pair (hints.py) is a push the spring camera follows photo by photo, geometrically, to its end
    value on the stretch's last photo; `centre` takes the lead away and puts his middle at the frame's middle."""
    from aimv.paint.camera import CW0, V_BASE
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "5.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    plain = _with(world, "vpk5p0", d)
    d["moves"].append({"hints": {"frame": {"from": 4.3, "to": 5.4, "zoom": [1.0, 0.7], "zoom_v": [1.0, 0.85],
                                           "centre": 1, "curve": "in"}}})
    ch = _with(world, "vpk5p1", d)
    assert not ch.errors, [str(i) for i in ch.errors]
    g = ch.grid
    q0, q1 = g.resolve(4.3)[0], g.resolve(5.4)[0]
    hs = {q: h for q, h in ch.cam_hints.items() if h.get("frame")}
    assert hs[round(q0, 4)]["zoom"] == 1.0 and hs[round(q1, 4)]["zoom"] == 0.7 and hs[round(q1, 4)]["centre"] == 1
    assert all("zr" in h for h in hs.values()) and hs[round(q1, 4)]["ze"] == 1.0
    photos = ch.photos()
    hist, hist0 = History(ch), History(plain)
    t = np.array([p.t for p in photos])
    k0, k1 = (int(np.argmin(np.abs(t - g.t(q)))) for q in (q0, q1))
    for a, b0, z1 in (("16:9", CW0, 0.7), ("9:16", CW0 * V_BASE, 0.85)):
        c = plan(ch, photos, hist.figure_boxes(), a)
        z = np.array([x.base[2] for x in c]) / (b0 * ch.u)
        zh = np.array([hs[round(photos[k].q, 4)]["zoom" if a == "16:9" else "zoom_v"] for k in range(k0, k1 + 1)])
        assert np.allclose(z[k0:k1 + 1], zh, rtol=1e-3) and abs(z[k1] - z1) < 1e-3  # followed exactly
        assert np.all(np.diff(z[k0:k1 + 1]) <= 1e-9)  # one push, never back out
        lr = np.log(z[k0:k1 + 1])
        assert np.argmax(-np.diff(lr)) > 0.6 * (k1 - k0)  # gathering speed: its fastest photos late
    c, c0 = plan(ch, photos, hist.figure_boxes(), "16:9"), plan(plain, photos, hist0.figure_boxes(), "16:9")
    fb, fb0 = hist.figure_boxes(), hist0.figure_boxes()
    def off(cams, boxes, k):
        keys = sorted(boxes)
        i = int(np.searchsorted(keys, photos[k].q + 1e-6, side="right")) - 1
        b = boxes[keys[i]]
        return ((b[0] + b[2]) / 2 - cams[k].base[0]) / cams[k].base[2], ((b[1] + b[3]) / 2 - cams[k].base[1])
    o = np.array([off(c, fb, k)[0] for k in range(k1 - 6, k1 + 1)])
    o0 = np.array([off(c0, fb0, k)[0] for k in range(k1 - 6, k1 + 1)])
    assert abs(o.mean()) < 0.03 and o0.mean() < -0.06  # centred; the plain spring leads him
    oy = np.array([off(c, fb, k)[1] for k in range(k1 - 6, k1 + 1)]) / (0.7 * CW0 * ch.u * 9 / 16)
    assert abs(oy.mean()) < 0.06  # his middle at the frame's middle height
    bad = _with(world, "vpk5p2", dict(d, moves=d["moves"][:1] + [{"hints": {"frame": {"from": 4.3, "to": 5.4,
                                                                                      "zoom": [1.0, 0.7],
                                                                                      "curve": "bounce"}}}]))
    assert any("frame.curve" in str(i) for i in bad.errors)


def test_spring_camera_follows_a_frame_tilt_stiffly(world, boxclip):
    """A frame's dy pair that changes (a tilt: film-v8's ease into verse 2's framing inside p6's brake) is followed
    by the vertical spring as stiffly as the horizontal: by the stretch's end the frame is at the full offset; a dy
    step keeps the soft vertical spring as ever."""
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "5.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    plain = _with(world, "vtilt0", d)
    d["moves"].append({"hints": {"frame": {"from": 4.3, "to": 5.1, "dy": [0, -40]}}})
    ch = _with(world, "vtilt1", d)
    assert not ch.errors, [str(i) for i in ch.errors]
    assert all(h.get("yr") == ["16:9", "9:16"] for h in ch.cam_hints.values() if h.get("frame"))
    photos = ch.photos()
    c = plan(ch, photos, History(ch).figure_boxes(), "16:9")
    c0 = plan(plain, photos, History(plain).figure_boxes(), "16:9")
    k = next(i for i, p in enumerate(photos) if abs(p.q - ch.grid.resolve(5.1)[0]) < 1e-6)
    assert abs((c[k].base[1] - c0[k].base[1]) + 40) < 12  # near the full tilt by its end
    d["moves"][-1] = {"hints": {"frame": {"from": 4.3, "to": 5.1, "dy": -40}}}  # a step: the soft spring, as ever
    st = _with(world, "vtilt2", d)
    assert not any(h.get("yr") for h in st.cam_hints.values())
    d["moves"][-1] = {"hints": {"frame": {"from": 4.3, "to": 5.1, "dy": 0, "dy_v": [0, 60]}}}  # the 9:16 only
    v9 = _with(world, "vtilt3", d)
    assert all(h.get("yr") == ["9:16"] for h in v9.cam_hints.values() if h.get("frame"))
    c3 = plan(v9, photos, History(v9).figure_boxes(), "16:9")
    assert all(abs(x.base[1] - y.base[1]) < 1e-6 for x, y in zip(c3, c0))  # the 16:9 as without the tilt
    # `aspect: 9:16`: the stretch is that aspect's only; the 16:9 exactly as without it (its weight, its ease)
    d["moves"][-1] = {"hints": {"frame": {"from": 4.3, "to": 5.1, "dy_v": [0, 60], "zoom": 1.3, "aspect": "9:16"}}}
    a9 = _with(world, "vtilt4", d)
    c4 = plan(a9, photos, History(a9).figure_boxes(), "16:9")
    assert all(abs(x.cx - y.cx) + abs(x.cy - y.cy) + abs(x.cw - y.cw) < 1e-9 for x, y in zip(c4, c0))
    v4, v0_ = plan(a9, photos, History(a9).figure_boxes(), "9:16"), plan(plain, photos, History(plain).figure_boxes(), "9:16")
    assert max(abs(x.base[1] - y.base[1]) for x, y in zip(v4, v0_)) > 20  # the 9:16's tilt
    d["moves"][-1] = {"hints": {"frame": {"from": 4.3, "to": 5.1, "dy_v": 60, "aspect": "4:3"}}}
    assert any("aspect" in str(i) for i in _with(world, "vtilt5", d).errors)


def test_spring_soft_bounds_eases_under_a_low_world_top_instead_of_snapping(world, boxclip):
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "6.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    ch = _with(world, "vpk5s0", d)
    hist = History(ch)
    photos = ch.photos()
    free = plan(ch, photos, hist.figure_boxes(), "9:16")
    xr = free[len(free) // 2].cx  # the low top ends here (the world's cloud cells begin)
    lim = float(np.median([c.cy for c in free])) + 120  # under it the crop must sit 120 px lower

    def clamp(cx, cy, cw, chh, margin=30):
        return (float(cx), float(max(cy, lim))) if cx - cw / 2 < xr else (float(cx), float(cy))

    def valid(x0, y0, x1, y1):  # (the same world, box by box: as the scene's clamp sees it)
        return x0 + 30 >= xr or (y0 + y1) / 2 >= lim - 1e-6
    sc = ch.scene
    w0 = sc.world
    sc.world, sc.clamp_crop, sc.street.valid_box = object(), clamp, valid
    try:
        hard = plan(ch, photos, hist.figure_boxes(), "9:16")
        ch.camera_defaults = {"soft_bounds": True}
        soft = plan(ch, photos, hist.figure_boxes(), "9:16")
    finally:
        sc.world = w0
        del sc.clamp_crop, sc.street.valid_box
        ch.camera_defaults = {}

    def worst(y, cams):  # the largest photo-to-photo change of a vertical pan, as a share of the width
        return float(np.max(np.abs(np.diff(np.array(y), 2)) / np.array([c.cw for c in cams[1:-1]])))
    # the final clamp snaps the plain spring (the hand-held nudge, the same in both, taken out); the soft one eases
    # (its spring, before the nudge) and leaves the final clamp at most the nudge to clip
    assert worst([c.cy - f.cy for c, f in zip(hard, free)], hard) > 0.08
    assert worst([c.base[1] for c in soft], soft) < 0.03
    assert max(abs(c.cy - c.base[1]) for c in soft) < 12  # (of the 120 px step)
    assert framing_stats(soft, [hist.figure_boxes()[max(k for k in hist.figure_boxes() if k <= p.q + 1e-6)]
                                for p in photos], True)["max_outside"] < 0.02  # and he stays in the frame


def test_debug_overlay_draws_the_physics_on_a_photo(world, boxclip):
    from aimv.paint.render import debug_overlay, plan_jobs, solver_notes
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk6", route, _pk_alley_x(world))
    hist = History(ch)
    jobs, _ = plan_jobs(ch, hist, ["16:9"])
    r = Renderer(ch, hist)
    j = next(j for j in jobs if abs(j.q - ch.grid.resolve("5.1.2")[0]) < 1e-6)  # in the air
    img = r.photo(j.k, j.t, j.q, j.cams["16:9"], (640, 360))
    dbg = debug_overlay(r, img, j, "16:9", (640, 360))
    assert dbg.shape == img.shape and np.abs(dbg - img).max() > 0.3
    assert ch.pk[round(j.q, 4)].get("arc")  # the flight's solved arc is there to draw
    assert "move 1 (parkour)" in solver_notes(ch, 0, 100)


def test_a_video_entry_can_set_its_guide_block_by_hand():
    img = np.full((120, 200, 3), 240, np.uint8)
    img[60:120, :] = (236, 203, 172)  # a roof guide (BGR of the key) across the whole frame
    img[30:90, 90:110] = 20  # the figure
    frames = [(f, f / 30, img.copy()) for f in range(2)]
    _, auto = PS.video_split(frames, {"key": [172, 203, 236]}, "BLK", {})
    assert auto["block"][0] == 0 and auto["block"][2] >= 190  # the whole frame's width: no use to a vault
    _, info = PS.video_split(frames, {"key": [172, 203, 236], "block": [80, 60, 140, 100], "shapes": [[80, 60, 140, 100]]},
                             "BLK", {})
    assert info["block"] == [80, 60, 140, 100] and info["shapes"] == [[80, 60, 140, 100]]


def test_a_qa_sheet_taller_than_a_jpeg_can_be_is_split_at_whole_rows(tmp_path):
    from PIL import Image
    from aimv.paint.qa import _save_tall
    sheet = Image.new("RGB", (300, 70000), (10, 10, 10))
    parts = _save_tall(sheet, tmp_path / "f-qa.jpg", 540, 80)
    assert [p.name for p in parts] == ["f-qa.jpg", "f-qa-2.jpg"]
    h = [Image.open(p).height for p in parts]
    assert sum(h) == 70000 and h[0] % 540 == 0 and h[0] <= 65000
    assert _save_tall(Image.new("RGB", (300, 900)), tmp_path / "g.jpg", 540, 80) == [tmp_path / "g.jpg"]
    wide = _save_tall(Image.new("RGB", (2880, 60000)), tmp_path / "w.jpg", 540, 80)  # (each part under 150 Mpx)
    assert len(wide) == 2 and all(Image.open(p).height * 2880 <= 150_000_000 for p in wide)


def test_a_stamps_primer_keep_stays_on_the_wall_while_primer_true_fades(world, boxclip):
    out = {}
    for pr in (True, "keep"):
        d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "5.4.4",
                                    "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}},
                       {"stamp": {"at": "4.1.2", "stamp": "SPL-1", "x": 300, "y": 450, "w": 200, "primer": pr}}]}
        ch = _with(world, f"vstk{pr}", d)
        hist = History(ch)
        t0, t1 = ch.grid.t(ch.grid.resolve("4.1.3")[0]), ch.grid.t(ch.grid.resolve("5.3.1")[0])
        ts = ch.grid.t(ch.grid.resolve("4.1.2")[0])
        prim = [m for m in hist.marks if m.kind == "primer" and abs(m.t - (ts - 0.0015)) < 1e-6 and m.w >= 200]
        assert len(prim) == 1 and prim[0].keeper == (pr == "keep")  # (the stamp's patch, not the poses')
        m = prim[0]
        _, C0, _ = hist.layers(t0, int(m.x0), int(m.y0), m.w, m.h)
        _, C1, _ = hist.layers(t1, int(m.x0), int(m.y0), m.w, m.h)
        out[pr] = (float(C0.mean()), float(C1.mean()))
    assert out[True][1] < 0.6 * out[True][0], out  # true: it fades as he paints on
    assert out["keep"][1] > 0.8 * out["keep"][0] and out["keep"][1] > 1.5 * out[True][1], out  # keep: it stays (the
    # rain's wash only)



def test_screen_cues_light_flare_and_kill_a_screen_from_its_stored_light(world, boxclip):
    from aimv.paint import screenlevels as SL
    d = world["street"] / "screens8"
    d.mkdir(exist_ok=True)
    face = np.full((100, 200, 3), 0.3, np.float16)
    glow = np.full((150, 250, 3), 0.05, np.float16)
    np.savez(d / "S1.npz", face=face, face_box=np.int32([1000, 200, 1200, 300]), glow=glow,
             glow_box=np.int32([900, 100, 1400, 400]))
    (d / "screens.json").write_text(json.dumps({"version": 1, "screens": [
        {"id": "S1", "group": "G", "box": [1000, 200, 1200, 300], "face": [1000, 200, 1200, 300],
         "glow_box": [900, 100, 1400, 400], "file": "screens8/S1.npz"}]}))
    dd = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
          "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 300}},
                    {"screen_cues": {"on": [["S1", 4.2]], "flare": [["S1", 4.4]], "kill": [["G", 5.3, 2]]}}]}
    ch = _with(world, "vscr", dd)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    T = lambda a: g.t(g.resolve(a)[0])
    assert SL.level(ch, "S1", T("4.1.3")) == 0.0  # dead before it lights
    assert SL.level(ch, "S1", T(4.2)) > 1.2  # the overshoot on its first lit photo
    assert SL.level(ch, "S1", T("4.3.1")) == pytest.approx(1.0)
    assert SL.level(ch, "S1", T(4.4)) > 1.5  # the flare
    assert 0 < SL.level(ch, "S1", T(5.3)) < 1 and SL.level(ch, "S1", T("5.3.3")) == 0.0  # dying over 2 photos
    sl = SL.ScreenLights(ch)
    lin = np.full((400, 600, 3), 0.4, np.float32)
    dead = sl.apply(lin, 900, 50, T("4.1.3"))
    assert dead[200, 200].mean() == pytest.approx(0.4 - 0.3 - 0.05, abs=0.01)  # face and glow taken out
    assert np.allclose(sl.apply(lin, 900, 50, T("4.3.1")), lin)  # as built


def test_camera_x_min_and_x_max_pin_the_crops_edges(world):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 4.3, "rate": "16ths",
                                                     "camera": {"preset": "follow", "zoom": 0.6, "x_min": 40}}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 60}}]}
    ch = _with(world, "vxmin", d)
    hist = History(ch)
    photos = ch.photos()
    c = plan(ch, photos, hist.figure_boxes(), "16:9")
    assert all(abs(k.cx - k.cw / 2 - 30 - 40) < 1e-6 for k in c)  # held at its own left edge (he is further left)
    d["segments"][0]["camera"].pop("x_min")
    ch2 = _with(world, "vxmin2", d)
    c2 = plan(ch2, ch2.photos(), History(ch2).figure_boxes(), "16:9")
    assert all(k.cx < j.cx for k, j in zip(c2, c))  # without it: the street's own end (further left) holds it
    # x_max: the crop's right edge never right of it (he is right of it: held at the edge, 30 px in), by the
    # smallest shift, eased out over the next segment's first photos (never cutting him there)
    d["segments"] = [{"from": 4.1, "to": 4.3, "rate": "16ths", "camera": {"preset": "follow", "zoom": 0.6,
                                                                       "x_max": 1400}},
                     {"from": 4.3, "to": 5.1, "rate": "16ths", "camera": {"preset": "follow", "zoom": 0.6}}]
    d["moves"][0]["pose"]["x"] = 1700
    ch3 = _with(world, "vxmax", d)
    ph3 = ch3.photos()
    c3 = plan(ch3, ph3, History(ch3).figure_boxes(), "16:9")
    d["segments"][0]["camera"].pop("x_max")
    ch4 = _with(world, "vxmax0", d)
    c4 = plan(ch4, ch4.photos(), History(ch4).figure_boxes(), "16:9")
    first = [k for k, p_ in enumerate(ph3) if p_.seg == 0]
    assert first and all(c3[k].cx + c3[k].cw / 2 <= 1400 - 30 + 1e-6 for k in first)
    assert any(abs(c3[k].cx + c3[k].cw / 2 - (1400 - 30)) < 1e-6 for k in first)  # (no further than it needs)
    later = [k for k, p_ in enumerate(ph3) if p_.seg == 1]
    sh = [c4[k].cx - c3[k].cx for k in later]
    assert sh[0] > 0 and sh[-1] < 1e-6  # eased out over the next segment's first photos
    fb = History(ch3).figure_boxes()
    fx1 = fb[max(q for q in fb if q <= ph3[later[0]].q + 1e-6)][2]
    room = [max(0.0, c4[k].cx + c4[k].cw / 2 - 0.06 * c4[k].cw - fx1) for k in later]
    assert all(s_ <= r_ + 1e-6 for s_, r_ in zip(sh, room))  # never moved so far that his figure leaves the frame


def test_a_smear_follows_his_body_across_a_screen_and_the_screen_dies_after_it(world, boxclip):
    from aimv.paint import screenlevels as SL
    d = world["street"] / "screens8"
    d.mkdir(exist_ok=True)
    scr = []
    for sid, x0 in (("S2", 1400), ("S3", 1700)):
        np.savez(d / f"{sid}.npz", face=np.full((140, 200, 3), 0.3, np.float16),
                 face_box=np.int32([x0, 430, x0 + 200, 570]), glow=np.full((10, 10, 3), 0.0, np.float16),
                 glow_box=np.int32([x0, 430, x0 + 10, 440]))
        scr.append({"id": sid, "group": sid, "box": [x0, 430, x0 + 200, 570], "face": [x0, 430, x0 + 200, 570],
                    "glow_box": [x0, 430, x0 + 10, 440], "file": f"screens8/{sid}.npz"})
    (d / "screens.json").write_text(json.dumps({"version": 1, "screens": scr}))
    dd = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
          "moves": [{"parkour": {"from": 4.1, "from_x": 900, "v0": 7.0, "gait": "sprint", "to": "4.3.4",
                                 "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}},
                    {"screen_cues": {"smear": [["S2"]], "kill": [["S3", "S2"]]}}]}
    ch = _with(world, "vsmear", dd)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    sl = SL.ScreenLights(ch, hist)
    sc = ch.screen_cues
    tr = sc["track"]["S2"]
    assert tr and all(a[1] == 1400 for a in tr)  # from the face's left edge
    R = [a[2] for a in tr]
    assert all(b >= a for a, b in zip(R, R[1:])) and R[-1] == 1600  # never back, across the whole face
    for t_, L_, R_ in tr:  # its front is where the front of his body is (at the face's height)
        q = next(p.q for p in ch.photos() if abs(p.t - t_) < 1e-6)
        if q in hist.fig_box:
            assert R_ <= min(hist.fig_box[q][2], 1600) + 1
    st = SL._step(ch)
    mid = next(a for a in tr if 1450 < a[2] < 1590)
    assert SL.level(ch, "S2", mid[0]) == pytest.approx(1.0)  # still lit while it is smeared
    td = SL.smear_death(sc, "S2", st)
    t_full = next(a[0] for a in tr if a[2] >= 1599)
    assert td == pytest.approx(t_full + 4 * st)
    assert SL.level(ch, "S2", td) == 0.0 and SL.level(ch, "S3", td) == 0.0  # S3 dies with it
    assert SL.level(ch, "S2", td - 2 * st) == pytest.approx(1.0) and SL.level(ch, "S3", td - 2 * st) == 1.0
    lin = np.full((300, 800, 3), 0.4, np.float32)
    out = sl.apply(lin, 1300, 350, mid[0])
    band = out[150, int(mid[2]) - 1300 - 25]  # (y 500: the face's middle) behind its front: black paint
    ahead = out[150, int(mid[2]) - 1300 + 25]
    assert band.mean() < 0.08 and ahead.mean() == pytest.approx(0.4)
    late = sl.apply(lin, 1300, 350, td + 0.5)
    assert late[150, 200].mean() < 0.08  # the stroke stays on the dead screen
    # a timed smear: from where his body is at its time, spreading back to the face's left edge over smear_n photos
    dd["moves"][1] = {"screen_cues": {"smear": [["S2", "4.1.3"]]}}
    ch = _with(world, "vsmear2", dd)
    hist = History(ch)
    SL.ScreenLights(ch, hist)
    sc = ch.screen_cues
    tr = sc["track"]["S2"]
    ts = ch.grid.t(ch.grid.resolve("4.1.3")[0])
    assert tr[0][0] == pytest.approx(ts) and tr[0][1] > 1400 and tr[1][1] == 1400
    assert SL.smear_death(sc, "S2", st) == pytest.approx(ts + 5 * st)


def test_a_box_key_with_drop_slams_down_from_above_a_photo_at_a_time(world, boxclip):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 300}},
                   {"box": {"id": "trap", "x": 1500, "keys": [{"at": 5.1, "w": 200, "h": 300, "thick": 14,
                                                                "drop": {"height": 600, "over": 0.4}}]}}]}
    ch = _with(world, "vdrop", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    pr = ch.props["trap"]
    falls = [k for k in pr.keys if k.drop]
    land = [k for k in pr.keys if not k.drop]
    assert len(land) == 1 and len(falls) >= 3
    assert all(a.q < b.q for a, b in zip(pr.keys, pr.keys[1:]))  # a photo at a time, then the key
    bt = [k.bottom for k in pr.keys]
    assert bt[0] == pytest.approx(land[0].bottom - 600) and all(a < b for a, b in zip(bt, bt[1:]))  # falling down
    steps = np.diff(bt)
    assert all(b > a for a, b in zip(steps, steps[1:]))  # ever faster
    hist = History(ch)
    g = ch.grid
    k0 = falls[0]
    x0, y0, x1, y1 = k0.box
    on = hist.layers(g.t(k0.q) + 0.01, int(x0), int(y0), int(x1 - x0), int(y1 - y0))[1].mean()
    off = hist.layers(g.t(land[0].q) + 0.3, int(x0), int(y0), int(x1 - x0), int(y1 - y0))[1].mean()
    assert on > 0.02 and off < 0.2 * on  # each falling outline shows on its photo and is gone after


def test_a_stamp_with_swath_is_painted_in_primer_and_stays(world, boxclip):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "5.4.4",
                                "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}},
                   {"marks": {"stages": [["4.1.2", "SPL-1", 2000, 450, 200, {"swath": True}],
                                         ["4.1.2", "SPL-1", 2400, 450, 200]]}}]}
    ch = _with(world, "vswath", d)
    hist = History(ch)
    ts = ch.grid.t(ch.grid.resolve("4.1.2")[0])
    sw = [m for m in hist.marks if m.kind == "remains" and abs(m.t - (ts - 0.0015)) < 1e-6]
    black = [m for m in hist.marks if m.kind == "splash" and abs(m.t - (ts - 0.0005)) < 1e-6]
    assert len(sw) == 1 and hist.fx[sw[0].id].get("swath") and sw[0].keeper  # the swath stage: primer, kept
    assert len(black) == 1 and black[0].x0 > sw[0].x0 + 300  # the other stage: black as before
    m = sw[0]
    t1 = ch.grid.t(ch.grid.resolve("5.3.1")[0])
    P0, C0, _ = hist.layers(ts + 0.01, int(m.x0), int(m.y0), m.w, m.h)
    P1, C1, _ = hist.layers(t1, int(m.x0), int(m.y0), m.w, m.h)
    sel = C0 > 0.5
    assert sel.any() and (P0[sel] / C0[sel]).mean() > 0.6  # white primer, not black paint
    assert C1.mean() > 0.85 * C0.mean()  # it stays (only the rain's wash)

def test_paint_in_reveals_the_world_on_the_beat_ahead_of_him_and_qa_checks_his_feet(world, boxclip):
    from aimv.paint import qa as QA
    from aimv.paint import reveal as RV
    route = [{"run": {"gait": "run"}}]

    def pk(name, pin):
        d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 400, "v0": 5.5, "gait": "run", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": route}}, {"paint_in": pin}]}
        return _with(world, name, d)
    ch = pk("vrv1", {"from": 4.1, "front": 700, "on": ["beats"], "ahead": [0.2, 0.4], "dur": 2.5})
    assert not ch.errors, [str(i) for i in ch.issues]
    pl = RV.plan(ch)
    g = ch.grid
    t0 = g.t(g.resolve(4.1)[0])
    assert len(pl["strokes"]) > 3 and all(b.b >= a.b - 1 for a, b in zip(pl["strokes"][1:], pl["strokes"][2:]))
    assert RV.revealed(ch, t0, 300, 400) > 0.99 and RV.revealed(ch, t0, 1600, 400) < 0.01  # primer ahead at first
    t1 = g.t(g.resolve(5.3)[0])
    xh = float(np.interp(t1, pl["tt"], pl["tx"]))
    assert RV.revealed(ch, t1, xh, 600) > 0.99 and RV.revealed(ch, t1, xh + 3000, 600) < 0.01  # (painted in under
    # him by then, not the whole street)
    hist = History(ch)
    photos = ch.photos()
    cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
    assert not [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] == "reveal"]
    R, K = RV.layers(ch, t0 + 0.01, 0, 0, 2400, 700)  # the border: wet black paint on it
    assert R[:, :500].mean() > 0.99 and R[:, 900:].mean() < 0.01 and K[:, 600:900].max() > 0.5
    r = Renderer(ch, hist)
    j = next(p for p in photos if p.t >= t0)
    img = r.photo(j.k, j.t, j.q, cams["16:9"][photos.index(j)], (640, 360))
    assert img[:, 560:].mean() > img[:, :120].mean() + 0.1  # pale primer ahead, the darker plate behind him
    # nothing painted (no hit strong enough): he runs onto primer, an error
    ch = pk("vrv2", {"from": 4.1, "front": 700, "on": ["impact"], "min": 2.0})
    hist = History(ch)
    photos = ch.photos()
    cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
    assert [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] == "reveal" and f["level"] == "error"]


def test_parkour_cycle_auto_picks_the_cycle_by_natural_cadence_and_reports_it(world, boxclip):
    from aimv.paint import parkour as PK
    assert abs(PK.natural_cadence(5.0) - 3.2) < 1e-9
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 400, "v0": 4.5, "gait": 4.5, "to": "5.4.4", "cycle": "auto",
                                "route": [{"run": {"gait": 4.5}}]}}]}
    old = PK.AUTO_CYCLES
    PK.AUTO_CYCLES = ("run_SPR",)
    try:
        ch = _with(world, "vcad", d)
    finally:
        PK.AUTO_CYCLES = old
    runs = ch.notes["move 1 (parkour)"]["runs"]
    assert runs and runs[0]["cycle"] == "run_SPR" and runs[0]["v_mps"] > 4
    r0 = runs[0]
    assert abs(r0["cadence"] - r0["v_mps"] / r0["step_m"]) < 0.02 and r0["natural"] == round(PK.natural_cadence(r0["v_mps"]), 2)
    assert r0["photos_per_cycle"] > 0


def test_parkour_auto_cycles_limits_the_choice_and_a_clip_dy_moves_it_down(world, boxclip):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 400, "v0": 4.5, "gait": 4.5, "to": "5.4.4", "cycle": "auto",
                                "auto_cycles": ["run_SPR"], "route": [{"run": {"gait": 4.5}}]}}]}
    ch = _with(world, "vcad2", d)  # (the test world's cycle has a short stride: its cadence check fails, as is)
    assert {r["cycle"] for r in ch.notes["move 1 (parkour)"]["runs"]} == {"run_SPR"}
    d["moves"][0]["parkour"]["auto_cycles"] = ["no_such_cycle"]
    ch = _with(world, "vcad3", d)
    assert any("auto_cycles" in str(i) for i in ch.issues)
    ys = {}
    for dy in (0, 40):
        route = [{"clip": {"sequence": "flat_SPR", "from": "a", "to": "b", "dy": dy}}, {"run": {"gait": "sprint"}}]
        ch = _pk(world, f"vpkdy{dy}", route, 900)
        ys[dy] = {round(e.q, 3): e.y for e in ch.events if e.tag == "pk.clip"}
    assert ys[0] and set(ys[0]) == set(ys[40])
    assert all(abs(ys[40][q] - ys[0][q] - 40 * ch.u) < 1e-6 for q in ys[0])  # the whole clip, that much lower


def test_parkour_join_carries_the_stride_phase_on_over_the_ground_between_moves(world, boxclip):
    d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "4.4.4",
                                "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}},
                   {"parkour": {"from": 5.1, "v0": "carry", "gait": "sprint", "to": "5.4.4",
                                "route": [{"run": {"gait": "sprint"}}], "cycle": "run_SPR"}}]}
    ch = _with(world, "vpk17", d)
    run = sorted([e for e in ch.events if e.tag == "pk.run"], key=lambda e: e.q)
    k = next(i for i, e in enumerate(run) if e.move == 1)
    a, b = run[k - 1], run[k]
    assert b.x - a.x > 50 and a.pose != b.pose  # he moved on, and so did his legs (no stride held at the join)
    # about the same step through the cycle as between any two photos of the run
    fr = [int(e.pose.split("-")[1]) for e in run]
    steps = [fr[i + 1] - fr[i] for i in range(len(fr) - 1) if i + 1 != k and fr[i + 1] > fr[i]]
    assert min(steps) - 1 <= fr[k] - fr[k - 1] <= max(steps) + 1, (fr[k - 1], fr[k], steps)


def test_parkour_roll_puts_each_drawing_on_the_landing_surface_and_qa_checks_it(world, boxclip):
    from aimv.paint import qa as QA
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "roll": True, "roll_sequence": "roll_SPR"}}]
    ch = _pk(world, "vpk16", route, _pk_alley_x(world))
    rolls = [(i, e) for i, e in enumerate(ch.events) if e.tag == "pk.roll"]
    assert rolls and all(e.flags.get("ground") and e.flags.get("surface") is not None for _, e in rolls)
    hist = History(ch)
    for i, e in rolls:  # (its lowest paint on the surface it landed on, whatever floor the clip itself had)
        d, x0, y0 = hist.place(i)
        ys = np.nonzero(d > 0.4)[0]
        assert abs(y0 + float(np.percentile(ys, 99.5)) - e.flags["surface"]) < 12
    photos = ch.photos()
    cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
    assert not [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] == "feet" and "roll" in f["tag"]]
    i, e = rolls[len(rolls) // 2]
    e.y += 60  # sunk into the roof
    hist = History(ch)
    assert [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] == "feet" and "roll" in f["tag"]]


def test_parkour_land_plays_other_clips_frames_as_the_landing_and_slows_him_to_v_out(world, boxclip):
    from aimv.paint import qa as QA
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "roll": False,
                       "land": {"clips": [["leap_SPR", 40, 46], ["roll_SPR", 48, 52]], "v_out": 2.0}}}]
    ch = _pk(world, "vpk_land", route, _pk_alley_x(world))
    assert not ch.errors, [str(i) for i in ch.issues]
    fl = ch.notes["move 1 (parkour)"]["flights"][0]
    assert fl["land"] == ["leap_SPR", "roll_SPR"] and not fl.get("roll")
    lands = sorted([(e.q, i, e) for i, e in enumerate(ch.events) if e.tag == "pk.land" and e.layer != "stamp"])
    assert len(lands) >= 3
    fr = [int(e.pose.split("-")[1]) for _, _, e in lands]
    assert fr == sorted(fr) and fr[0] >= 40 and fr[-1] <= 52  # those frames, in order (not the splash stamps)
    xs = [e.x for _, _, e in lands]
    dx = np.diff(xs)
    assert all(d > 0 for d in dx) and dx[-1] < dx[0]  # on, slowing down
    g = ch.grid
    run = sorted([e for e in ch.events if e.tag == "pk.run" and g.t(e.q) > g.t(lands[-1][0])], key=lambda e: e.q)
    v_first = (run[1].x - run[0].x) / (g.t(run[1].q) - g.t(run[0].q)) / (ch.lib.ref_height / 1.8)
    assert v_first < 3.6  # the run takes over near v_out (2 m/s), building from there
    hist = History(ch)
    for _, i, e in lands:  # each drawing's lowest paint on the surface it landed on
        d, x0, y0 = hist.place(i)
        ys = np.nonzero(d > 0.4)[0]
        assert abs(y0 + float(np.percentile(ys, 99.5)) - e.flags["surface"]) < 12
    photos = ch.photos()
    cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
    assert not [f for f in QA.run(ch, hist, photos, cams)["flags"] if f["rule"] == "feet" and "land" in f["tag"]]


def test_parkour_hop_clears_the_prop_by_a_quarter_metre_and_a_gap_jump_rises_naturally(world, boxclip):
    route = [{"hop": {"obstacle": "ac", "sequence": "leap_SPR"}}]
    ch = _pk(world, "vpk7", route, 900, to="4.4.4")
    assert not ch.errors, [str(i) for i in ch.issues]
    fl = ch.notes["move 1 (parkour)"]["flights"][0]
    assert 0.2 <= fl["clear_m"] <= 0.4  # the least rise whose every drawing over it clears it: not a monster jump
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk8", route, _pk_alley_x(world))
    fl = ch.notes["move 1 (parkour)"]["flights"][0]
    assert 0.4 <= fl["rise_m"] <= 0.7 and fl["clear_m"] > 0


def test_parkour_clip_keeps_his_momentum_and_the_takeoff_is_the_first_skin(world, boxclip):
    route = [{"clip": {"sequence": "flat_SPR", "from": "a", "to": "b"}}, {"run": {"gait": "sprint"}}]
    ch = _pk(world, "vpk9", route, 200, to="4.4.4")
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    cl = [e for e in ch.events if e.tag == "pk.clip"]
    t = [g.t(e.q) for e in cl]
    v = (cl[-1].x - cl[0].x) / (t[-1] - t[0])
    assert abs(v - 0.95 * 7.0 * 125) < 0.12 * 7.0 * 125  # through the clip at his speed coming in, less 5%
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    ch = _pk(world, "vpk10", route, _pk_alley_x(world))
    tko = next(e for e in ch.events if e.tag == "pk.takeoff")
    assert tko.layer == "air" and tko.flags.get("flight") and tko.flags.get("takeoff")  # the flight's first skin


def test_parkour_clip_ground_start_keeps_its_heights_from_where_it_began(world, boxclip):
    import shutil
    src = world["street"]
    dd = src.parent / "lvl"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["ground"]["levels"] = [{"id": "roofT", "x": [780, 4000], "feet": 400}]  # a raised ground from x 780
    (dd / "obstacles.json").write_text(json.dumps(ob))
    out = {}
    for gr in (None, "start", "roofT"):
        clip = {"sequence": "flat_SPR", "from": "a", "to": "b", "speed": 1.0}
        if gr:
            clip["ground"] = gr
        d = {"street": "lvl", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 700, "v0": 7.0, "gait": "sprint", "to": "4.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}]}}]}
        ch = _with(world, f"vpk11{gr or 'n'}", d)
        assert not ch.errors, [str(i) for i in ch.issues]
        out[gr] = [(e.x, e.y) for e in ch.events if e.tag == "pk.clip"]
    late = [i for i, (x, _) in enumerate(out[None]) if x > 800]
    assert late
    k = late[-1]
    assert out[None][k][1] < out["start"][k][1] - 150  # per frame: lifted onto the raised ground
    assert abs(out["roofT"][k][1] - out[None][k][1]) < 1e-6  # the level's own line
    assert abs(out["start"][0][1] - out[None][0][1]) < 1e-6  # (both start on the street)


def test_parkour_clip_light_from_drops_the_keepers_from_its_key_on(world, boxclip):
    out = {}
    for lf in (None, "b"):
        clip = {"sequence": "flat_SPR", "from": "a", "speed": 1.0}
        if lf:
            clip["light_from"] = lf
        d = {"street": "main", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 700, "v0": 7.0, "gait": "sprint", "to": "4.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}]}}]}
        ch = _with(world, f"vpklf{lf or 'n'}", d)
        assert not ch.errors, [str(i) for i in ch.issues]
        out[lf] = [(e.pose, e.flags.get("trail")) for e in ch.events if e.tag == "pk.clip"]
    assert out[None] and all(t is None for _, t in out[None])
    tr = [t for _, t in out["b"]]
    k = tr.index("light")  # from its key on: a light trail (no keepers); before it, as without
    assert 0 < k and all(t is None for t in tr[:k]) and all(t == "light" for t in tr[k:])


@pytest.fixture(scope="module")
def barclip(world, boxclip):
    """The sprint again as a bar clip (SPB: a hand-set guide bar above his head, x 600 to 1000, y 300 to 310, his
    feet on 700) with take-off, catch and release keys, and a street (barT) with a raised deck to x 2000 (feet 500)
    and a bar at y 280 past it, over the street (632)."""
    import shutil
    root = world["root"]
    pd = root / "paint" / "poses"
    spec = json.loads((pd / "poses.spec.json").read_text())
    spec["video"]["SPB"] = dict(spec["video"]["SPR"], shapes=[[600, 300, 1000, 310]], ground=700)
    spec["sequences"]["swing_SPB"] = {"video": "SPB", "range": [0, 63], "keys": {"takeoff": 16, "catch": 32,
                                                                                  "release": 48}}
    (pd / "poses.spec.json").write_text(json.dumps(spec))
    PS.build(root, pd, log=lambda m: None)
    src = world["street"]
    dd = src.parent / "barT"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["ground"]["levels"] = [{"id": "deckT", "x": [0, 2000], "feet": 500}]
    ob["obstacles"] = {"edgeT": {"kind": "edge", "box": [1960, 500, 2000, 520], "contacts": {"edge": {"x": 2000}}},
                       "barT": {"kind": "beam", "box": [2300, 270, 2560, 290],
                                "contacts": {"bar": {"y": 280, "x0": 2300, "x1": 2560}}}}
    (dd / "obstacles.json").write_text(json.dumps(ob))
    return PS.PoseLib(pd)


def test_parkour_clip_pins_several_keys_to_times(world, barclip):
    """v8 (P5's stacks): a route clip's `keys: {key: time, ...}` pins each key's frame to its time (the first one
    anchors the clip), the frames between two pins played to fit them (squeezed or held), as shot before the first
    and after the last; their times must rise with their frames, and keys replaces key and at."""
    lib = barclip
    tr = {r["name"]: r for r in lib.videos["SPB"]["frames"]}

    def pk(name, clip):
        d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 300, "v0": 0, "gait": "jog", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": "jog"}}]}}]}
        return _with(world, name, d)

    pins = {"takeoff": "4.3", "catch": "4.4", "release": "5.2"}
    ch = pk("vpins1", {"sequence": "swing_SPB", "speed": 1.0, "ground": "start", "wait": True, "keys": pins})
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    shown = {round(e.q, 4): tr[e.pose]["f"] for e in ch.events if e.tag == "pk.clip"}
    drawn = sorted(r["f"] for r in lib.videos["SPB"]["frames"])
    for k_, f_ in (("takeoff", 16), ("catch", 32), ("release", 48)):
        want = min(drawn, key=lambda d_: abs(d_ - f_))
        assert abs(shown[round(g.resolve(pins[k_])[0], 4)] - want) <= 1, (k_, shown)
    # held between catch and release (two beats for 16 frames), squeezed between takeoff and catch (one beat)
    qa, qb = g.resolve("4.4")[0], g.resolve("5.2")[0]
    mid = min(shown, key=lambda q_: abs(q_ - (qa + qb) / 2))
    assert abs(shown[mid] - 40) <= 3
    # the times must rise with the frames; keys replaces key and at
    bad = pk("vpins2", {"sequence": "swing_SPB", "speed": 1.0, "wait": True,
                        "keys": {"takeoff": "4.4", "catch": "4.3"}})
    assert any("must rise with their frames" in str(i) for i in bad.errors)
    bad = pk("vpins3", {"sequence": "swing_SPB", "speed": 1.0, "wait": True, "key": "catch", "at": "4.3",
                        "keys": {"takeoff": "4.3", "catch": "4.4"}})
    assert any("drop key and at" in str(i) for i in bad.errors)


def test_parkour_run_on_a_slope_puts_each_drawing_s_lowest_point_on_the_line(world, boxclip):
    """v8 (P6's beam, 5.6 degrees): on a sloped level the run's feet line is taken at each drawing's lowest point
    (torso x plus its foot offset), so no trailing toe floats and no reaching foot sinks; on the flat (and under
    SLOPE_MIN) it stays the torso's, as before."""
    import shutil
    from aimv.paint.parkour import Route
    src = world["street"]
    dd = src.parent / "rampT"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["ground"]["levels"] = [{"id": "ramp", "x": [0, 4000], "feet": [[0, 632], [4000, 232]]}]  # 10%: about 5.7 deg
    ob["obstacles"] = {}
    (dd / "obstacles.json").write_text(json.dumps(ob))

    def runs(street, name):
        d = {"street": street, "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 400, "v0": 7.0, "gait": "sprint", "to": "4.4.4",
                                    "cycle": "run_SPR", "route": [{"run": {"gait": "sprint"}}]}}]}
        ch = _with(world, name, d)
        assert not ch.errors, [str(i) for i in ch.issues]
        return ch, [e for e in ch.events if e.tag == "pk.run"]
    ch, ev = runs("rampT", "vslope1")
    lib, sc = ch.lib, ch.scene
    offs = []
    for e in ev:
        fdx = -lib.torso_offset(e.pose, "bottom")[0]
        assert abs(e.flags["surface"] - sc.feet_at(e.x + fdx)) < 0.5  # the line at its lowest point
        offs.append(abs(fdx))
    assert max(offs) > 15  # (drawings whose lowest point is well off the torso: the case that matters)
    flat, fe = runs("main", "vslope2")
    assert all(abs(e.flags["surface"] - flat.scene.feet_at(e.x)) < 1e-6 for e in fe)  # flat: the torso's line


def test_parkour_run_into_a_clip_on_a_level_stays_on_it_past_its_edge(world, barclip):
    """v8 (P5's stacks): the last strides into a clip that stands on a level (`ground: <level>`) run on that level,
    not on the ground under his torso when it is past the level's edge (a step down: the clip's first frames still
    on the upper roof)."""
    def pk(name, ground):
        clip = {"sequence": "swing_SPB", "from": "takeoff", "to": "catch", "speed": 1.0, "x": {"takeoff": 2150},
                **({"ground": ground} if ground else {})}
        d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 1200, "v0": 3.5, "gait": "jog", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": "jog"}}]}}]}
        return _with(world, name, d)

    ch = pk("vlvl1", "deckT")
    assert not ch.errors, [str(i) for i in ch.issues]
    t0 = min(e.q for e in ch.events if e.tag == "pk.clip")
    past = [e for e in ch.events if e.tag == "pk.run" and e.q < t0 and e.x > 2010]
    assert past and all(abs(e.flags["surface"] - 500) < 1 for e in past), [(e.x, e.flags["surface"]) for e in past]
    plain = pk("vlvl2", None)  # (without a level: the ground under his torso, the street's 632)
    t0 = min(e.q for e in plain.events if e.tag == "pk.clip")
    assert any(e.flags["surface"] > 600 for e in plain.events if e.tag == "pk.run" and e.q < t0 and e.x > 2010)


def test_parkour_clip_ballistic_flight_lands_where_its_arc_meets_the_ground(world, barclip):
    """v8 (P5's lache): `flight` with `ballistic: true` flies from the take-off frame at the clip's own velocity
    there (both axes) on one gravity and puts the landing frame where that arc meets a surface (off the deck's end:
    down on the street 132 px below), instead of where the placement puts it."""
    clip = {"sequence": "swing_SPB", "from": "takeoff", "to": "release", "speed": 1.0,
            "flight": ["takeoff", "catch"], "edge": 2000, "ballistic": True}
    d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 1300, "v0": 3.5, "gait": "jog", "to": "5.4.4",
                                "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": "jog"}}]}}]}
    ch = _with(world, "vfree1", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    cf = ch.notes["move 1 (parkour)"]["clip_flights"][0]
    g = 9.81 * barclip.ref_height / 1.8
    T = cf["T"]
    assert abs(cf["x_catch"] - (cf["x_to"] + cf["vx_mps"] * barclip.ref_height / 1.8 * T)) < 2  # at its own speed
    assert 0.6 * (2 * 132 / g) ** 0.5 < T < 1.6 * (2 * 132 / g) ** 0.5  # about the time to fall 132 px
    tr = {r["name"]: r for r in barclip.videos["SPB"]["frames"]}
    land = [e for e in ch.events if e.tag == "pk.clip" and tr[e.pose]["f"] == 32]
    assert land and land[0].x > 2000  # down on the street, past the deck
    # with keys pinned up to the flight's take-off (a swing to its release), the flight's time from the flight
    pinned = dict(clip, **{"from": 8, "keys": {8: "4.3.3", "takeoff": "4.4"}})
    ch2 = _with(world, "vfree2", dict(d, moves=[{"parkour": dict(d["moves"][0]["parkour"], route=[
        {"clip": pinned}, {"run": {"gait": "jog"}}])}]))
    assert not ch2.errors, [str(i) for i in ch2.issues]
    cf2 = ch2.notes["move 1 (parkour)"]["clip_flights"][0]
    assert abs(cf2["t_to"] - ch2.grid.t(ch2.grid.resolve("4.4")[0])) < 0.02 and cf2["T"] > 0.05
    bad = _with(world, "vfree3", dict(d, moves=[{"parkour": dict(d["moves"][0]["parkour"], route=[
        {"clip": dict(pinned, keys={8: "4.3.3", "catch": "4.4.3"})}, {"run": {"gait": "jog"}}])}]))
    assert any("no pin inside the flight" in str(i) for i in bad.errors)


def test_parkour_clip_wait_lands_a_key_after_its_ballistic_flight_on_its_time(world, barclip):
    """v8c (P5's step-off): a clip that waits (`wait: true`) for a `key` in or after its ballistic flight (the
    landing pinned on the impact) waits so that the key, timed through the real flight, lands on its anchor: the
    standing still before it absorbs the difference, the flight is never stretched."""
    clip = {"sequence": "swing_SPB", "from": 8, "to": "release", "speed": 1.0, "flight": ["takeoff", "catch"],
            "edge": 2000, "ballistic": True, "wait": True, "key": "catch", "at": "5.2"}
    d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 1300, "v0": 3.5, "gait": "jog", "to": "5.4.4",
                                "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": "jog"}}]}}]}
    ch = _with(world, "vwaitfl1", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    cf = ch.notes["move 1 (parkour)"]["clip_flights"][0]
    Ta = ch.grid.t(ch.grid.resolve("5.2")[0])
    assert abs(cf["t_to"] + cf["T"] - Ta) < 0.011, (cf, Ta)  # the catch, after the real flight, on its anchor
    g = 9.81 * barclip.ref_height / 1.8
    assert 0.6 * (2 * 132 / g) ** 0.5 < cf["T"] < 1.6 * (2 * 132 / g) ** 0.5  # (the flight as before: real)


def test_parkour_clip_sway_plays_its_torso_as_shot_back_as_well_as_forward():
    """v8c (the director on pipe B: his hands must close on the pipe): a route clip's torso never goes back (a run's
    rule), except with `sway` (true, or a key: up to its frame), where its frames keep their x as shot from its
    placement, back as well as forward (a climber's torso swaying round the pole it is placed on, his hands staying
    on it)."""
    import numpy as np
    from aimv.paint.parkour import Solver
    plan = {"rel": np.array([0.0, 30.0, 12.0, -8.0, 20.0]), "ts": [0.0, 0.1, 0.2, 0.3, 0.4], "boxed": None,
            "flight": None}
    _, xs, _ = Solver.clip_track(None, plan, 100.0, 0.0)
    assert xs == [100.0, 130.0, 130.0, 130.0, 130.0]  # (never back)
    _, xs, _ = Solver.clip_track(None, dict(plan, sway=4), 100.0, 0.0)
    assert xs == [100.0, 130.0, 112.0, 92.0, 120.0]  # as shot (sway: true, all its frames)
    _, xs, _ = Solver.clip_track(None, dict(plan, sway=2), 100.0, 0.0)
    assert xs == [100.0, 130.0, 112.0, 112.0, 120.0]  # as shot up to its sway key's frame, never back from there


def test_parkour_ballistic_landing_comes_down_onto_a_surface_from_above():
    """v8c (p5c's step-off): a ballistic flight lands where its landing frame's feet come down onto a surface from
    above. A step off a parapet (feet line 0 to x 10, then a roof 100 px lower) whose take-off frame stands 3 px above
    the parapet's line and whose landing frame stands 10 px taller lands on the lower roof after a real fall, not at
    once on the parapet he steps off (its landing frame's feet already below that line there)."""
    from aimv.paint.parkour import ballistic_landing

    def surface(x, above):  # (the parapet to x 10, the lower roof past it; the highest at or below `above`)
        ys = [y for x0, x1, y in ((-1e9, 10.0, 0.0), (-1e9, 1e9, 100.0)) if x0 <= x <= x1 and y >= above - 1e-6]
        return min(ys) if ys else None

    g = 1000.0
    tt, xx = ballistic_landing(surface, 7.0, -50.0, 100.0, 0.0, 47.0, 57.0, g)
    assert xx > 10 and abs(tt - ((2 * (100 - 7) / g) ** 0.5)) < 0.01, (tt, xx)  # its feet (y -50 + 57) fall 93 px
    # a flight that leaves the ground going up and comes down on the same level still lands there, as before
    tt2, xx2 = ballistic_landing(lambda x, above: 0.0 if 0.0 >= above - 1e-6 else None, 0.0, -50.0, 100.0, -300.0,
                                 50.0, 50.0, g)
    assert abs(tt2 - 0.6) < 0.01 and abs(xx2 - 60) < 1.5
    assert ballistic_landing(lambda x, above: None, 0.0, -50.0, 100.0, 0.0, 50.0, 50.0, g) is None  # over a gap


def test_parkour_clip_keys_by_frame_bar_height_from_its_ground_and_a_ballistic_flight(world, barclip):
    """v8 (P2b): a clip's keys may be clip frame numbers; a bar clip's height is measured from the ground its heights
    stand on (`ground: start` on the deck: the bar still on the street's bar); `flight: [takeoff, catch]` flies him
    from the take-off (its foot at `edge`) to the catch (where the bar puts it) at his take-off speed, the torso on
    one gravity's arc."""
    lib = barclip
    tr = {r["name"]: r for r in lib.videos["SPB"]["frames"]}

    def pk(name, clip, from_x=1300, v0=3.5, gait="jog"):
        d = {"street": "barT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": from_x, "v0": v0, "gait": gait, "to": "5.4.4",
                                    "cycle": "run_SPR", "route": [{"clip": clip}, {"run": {"gait": gait}}]}}]}
        return _with(world, name, d)

    def bar_line(e):  # where the clip's guide bar is for this photo's drawing (street y)
        r = tr[e.pose]
        return e.y - (r["y"] - 305) * lib.m(e.pose)["scale"]

    # keys by frame number: the same clip as by name; a frame that is not drawn is an error
    base = {"sequence": "swing_SPB", "obstacle": "barT", "catch_x": 2400, "speed": 1.0}
    a = pk("vbar1", dict(base, **{"from": "takeoff", "to": "release"}))
    b = pk("vbar2", dict(base, **{"from": 16, "to": 48}))
    assert not a.errors and not b.errors, [str(i) for i in a.issues + b.issues]
    ea = [(round(e.q, 3), e.pose, round(e.x, 3), round(e.y, 3)) for e in a.events if e.tag == "pk.clip"]
    assert ea and ea == [(round(e.q, 3), e.pose, round(e.x, 3), round(e.y, 3)) for e in b.events
                         if e.tag == "pk.clip"]
    bad = pk("vbar3", dict(base, **{"from": 16, "to": 999}))
    assert any("no drawn frame 999" in str(i) for i in bad.errors)
    # the bar's height from the clip's own ground: per frame (the street under the bar) or from its first frame (on
    # the deck): either way the hanging frames (catch to release) have the clip's bar on the street's (y 280)
    for gr in (None, "start"):
        clip = dict(base, **{"from": "takeoff", "to": "release"}, **({"ground": gr} if gr else {}))
        ch = pk(f"vbar4{gr or 'n'}", clip)
        assert not ch.errors, [str(i) for i in ch.issues]
        hang = [e for e in ch.events if e.tag == "pk.clip" and 32 <= tr[e.pose]["f"] <= 48
                and 700 - tr[e.pose]["low"] >= 15]  # (off the clip's floor, as a hanging frame is: not settled)
        assert hang and all(abs(bar_line(e) - 280) < 1.0 for e in hang), (gr, [bar_line(e) for e in hang])
    # the flight: the take-off frame's foot on the deck's end, his run's speed through the air to the catch frame
    # where the bar puts it (as without the flight), the torso on the arc (equal photo steps: second differences of
    # its height g dt^2)
    fl = pk("vbar5", dict(base, **{"from": "takeoff", "to": "release", "flight": ["takeoff", "catch"],
                                   "edge": 2000, "ground": "start"}))
    assert not fl.errors, [str(i) for i in fl.issues]
    note = fl.notes["move 1 (parkour)"]
    cf = note["clip_flights"][0]
    nm_to = next(n for n, r in tr.items() if r["f"] == 16)
    fx = -lib.torso_offset(nm_to, "bottom")[0]
    assert abs(cf["x_to"] + fx - (2000 - 8 * fl.u)) < 0.5  # its foot 8 px before the edge
    assert abs(cf["vx_mps"] - 3.5) < 0.35  # his take-off speed (the run's)
    nm_c = next(n for n, r in tr.items() if r["f"] == 32)
    sv = lib.m(nm_c)["scale"]
    x_c = 2400 + (tr[nm_c]["x"] - 800) * sv  # the catch where the bar alone puts it
    assert abs(cf["x_catch"] - x_c) < 0.5
    assert abs(cf["T"] - (x_c - cf["x_to"]) / (cf["vx_mps"] * 125.0)) < 0.01  # time in the air: distance over speed
    g = fl.grid
    air = sorted([e for e in fl.events if e.tag == "pk.clip" and e.flags.get("flight")], key=lambda e: e.q)
    assert len(air) >= 4
    dt = g.t(air[1].q) - g.t(air[0].q)
    d2 = [air[k + 1].y - 2 * air[k].y + air[k - 1].y for k in range(1, len(air) - 1)]
    assert all(abs(v - 9.81 * 125.0 * dt * dt) < 0.5 for v in d2), d2
    xs = [e.x for e in air]
    assert all(abs((xs[k + 1] - xs[k]) - cf["vx_mps"] * 125.0 * dt) < 1.0 for k in range(len(xs) - 1))
    hang = [e for e in fl.events if e.tag == "pk.clip" and 32 <= tr[e.pose]["f"] <= 48 and not e.flags.get("flight")]
    assert hang and all(abs(bar_line(e) - 280) < 1.0 for e in hang)
    # its foot past the edge (the run-up starts beyond it): an error
    late = pk("vbar6", dict(base, **{"from": "takeoff", "flight": ["takeoff", "catch"], "edge": 2000}), from_x=2100)
    assert any("past the edge" in str(i) for i in late.errors)


def test_a_gap_with_the_wall_behind_it_is_paintable_and_no_void(world):
    """v8 (P2b's awnings): a gap obstacle with `"wall": true` is a gap for the route only: paint goes on the wall
    behind it and it is no void; a plain gap (the alley) stays unpaintable and a void."""
    import shutil
    src = world["street"]
    dd = src.parent / "gapW"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["obstacles"]["slotW"] = {"kind": "gap", "box": [3000, 300, 3150, 640], "wall": True,
                                "contacts": {"near": {"x": 3000}, "far": {"x": 3150}}}
    (dd / "obstacles.json").write_text(json.dumps(ob))
    sc = Scene(dd, world["root"] / "paint" / ".cache")
    assert sc.obstacle("slotW").wall and not sc.obstacle("alley").wall
    ids = {z["id"] for z in sc.no_paint}
    assert "alley" in ids and "slotW" not in ids
    vids = {v["id"] for v in sc.void_parts()}
    assert "alley" in vids and "slotW" not in vids
    _, pnt = sc.masks(2950, 3200)
    assert pnt[100:500, 60:140].min() > 0.9  # the wall behind it
    _, pa = sc.masks(900, 1150)
    assert pa[100:500, 20:230].max() < 0.1  # the alley: no paint


def test_flight_paint_keeps_him_on_the_wall_in_the_air_and_skin_still_takes_him_off(world, boxclip):
    """v8 (the director on P2b): `flight: paint` (the choreography's, or a move's own) paints his flights on the wall, his
    primer patch behind him as when he runs (no J2 skin); a move's `skin` still makes him the free skin; the default
    is the J2 skin as before."""
    route = [{"jump": {"gap": "alley", "sequence": "leap_SPR", "at": 5.1}}]
    def air_poses(ch):
        return [e for e in ch.events if e.layer == "air" and e.pose and e.tag.startswith("pk.")]
    base = _pk(world, "vflt0", route, _pk_alley_x(world))
    assert not base.errors and air_poses(base)  # the J2 skin in the air (the default)
    d = {"street": "main", "seed": 3, "flight": "paint", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": _pk_alley_x(world), "v0": 7.0, "gait": "sprint",
                                "to": "5.4.4", "route": route, "cycle": "run_SPR"}}]}
    ch = _with(world, "vflt1", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    assert not air_poses(ch)
    fl = [e for e in ch.events if e.layer == "wall" and e.flags.get("flight")]
    assert fl and any(e.tag == "pk.jump" for e in fl)  # the flight's drawings, on the wall
    hist = History(ch)
    marks = [m for m in hist.marks if m.kind == "primer"]
    assert any(abs(m.t - ch.grid.t(e.q)) < 0.01 for m in marks for e in fl)  # a primer patch under them
    d["moves"][0]["parkour"]["skin"] = True  # a move's skin: off the wall still
    assert air_poses(_with(world, "vflt2", d))
    d["moves"][0]["parkour"].pop("skin")
    d["moves"][0]["parkour"]["flight"] = "skin"  # a move's own flight wins over the file's
    assert air_poses(_with(world, "vflt3", d))
    d["moves"][0]["parkour"].pop("flight")
    # a jump step's own flight wins over the file's (P2b v8g: the leap between buildings as v7's J2 skin): its
    # take-off and flight in the air, not on the wall; the landing back on the wall
    d["moves"][0]["parkour"]["route"] = [{"jump": dict(route[0]["jump"], flight="skin")}]
    st = _with(world, "vflt5", d)
    assert not st.errors, [str(i) for i in st.issues]
    ap = air_poses(st)
    assert any(e.tag == "pk.jump" for e in ap) and any(e.tag == "pk.takeoff" for e in ap)
    assert not [e for e in st.events if e.layer == "wall" and e.flags.get("flight")]
    assert any(e.tag == "pk.land" and e.layer == "wall" for e in st.events)
    d["moves"][0]["parkour"]["route"] = [{"jump": dict(route[0]["jump"], flight="air")}]
    assert any("flight" in str(i) for i in _with(world, "vflt6", d).errors)
    d["moves"][0]["parkour"]["route"] = route
    d["flight"] = "air"
    with pytest.raises(ValueError, match="flight"):
        _with(world, "vflt4", d)


def test_parkour_take_off_from_a_level_on_a_solid_box_has_it_as_his_ground(world, boxclip):
    """v8 (P2b's awnings): a level laid on a solid box's top (not a prop he landed on): the take-off photo off its
    edge has that box as its ground (`on`), so the QA's inside rule leaves it to the ground line's tolerance."""
    import shutil
    src = world["street"]
    dd = src.parent / "ledgeL"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["obstacles"] = {"ledgeT": {"kind": "ledge", "box": [1200, 560, 2600, 632], "contacts": {"top": {"y": 560}}}}
    ob["ground"]["levels"] = [{"id": "ledge top", "x": [1200, 2600], "feet": 560}]
    (dd / "obstacles.json").write_text(json.dumps(ob))
    d = {"street": "ledgeL", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 1900, "v0": 5.0, "gait": "run", "to": "5.4.4",
                                "cycle": "run_SPR",
                                "route": [{"drop": {"edge": 2600, "sequence": "leap_SPR", "at": 4.3}}]}}]}
    ch = _with(world, "vledge", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    tko = next(e for e in ch.events if e.tag == "pk.takeoff")
    assert tko.flags.get("on") == "ledgeT"
    skins = [e for e in ch.events if e.tag == "pk.run" and e.flags.get("flight")]
    assert all(e.flags.get("on") == "ledgeT" for e in skins)


def test_parkour_lands_on_a_props_top_runs_on_it_and_drops_off_its_edge(world, boxclip):
    import shutil
    src = world["street"]
    dd = src.parent / "wallT"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["obstacles"] = {"wallT": {"kind": "box", "box": [1400, 560, 2600, 632], "contacts": {"top": {"y": 560}},
                                 "occlude": "rect"}}  # a low wall alone on the street
    (dd / "obstacles.json").write_text(json.dumps(ob))

    def pk(name, route):
        d = {"street": "wallT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": 700, "v0": 5.0, "gait": "run", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": route}}]}
        return _with(world, name, d)
    ch = pk("vpk12", [{"hop": {"obstacle": "wallT", "sequence": "leap_SPR", "onto": True}},
                      {"drop": {"edge": "wallT.x1", "sequence": "leap_SPR"}}])
    assert not ch.errors, [str(i) for i in ch.issues]  # (the drop off its own edge: his foot on it, not off it)
    fl0 = ch.notes["move 1 (parkour)"]["flights"][0]
    assert fl0["fall_m"] < 1.0 and 1400 < fl0["x_land"] < 1700  # the landing drawing's feet came down on the top
    from aimv.paint import qa as QA
    hist = History(ch)
    photos = ch.photos()
    res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
    bad = [(f["addr"], f["rule"], f["tag"], f["detail"]) for f in res["flags"] if f["level"] == "error" and f["rule"] in ("feet", "inside", "solid")]
    assert not bad, bad
    fls = ch.notes["move 1 (parkour)"]["flights"]
    assert len(fls) == 2 and fls[1]["kind"] == "drop"
    land = [e for e in ch.events if e.tag == "pk.land"]
    assert land and abs(land[0].y - 560) < 1e-6  # the hop's absorb on the top, not the ground (632)
    on_top = [e for e in ch.events if e.tag == "pk.run" and 1500 <= e.x <= 2400]
    assert on_top and all(abs(e.y - 560) < 1e-6 for e in on_top)  # running on it
    ch = pk("vpk13", [{"hop": {"obstacle": "wallT", "sequence": "leap_SPR", "onto": True}}, {"run": {"x": 3000}}])
    assert any("runs off wallT's top" in str(i) for i in ch.errors)


def test_parkour_onto_steps_up_early_runs_on_to_the_rim_and_reports_a_late_take_off(world, boxclip):
    import shutil
    src = world["street"]
    dd = src.parent / "wallT"
    dd.mkdir(exist_ok=True)
    for name in ("street.json", "build.json"):
        shutil.copy(src / name, dd / name)
    if not (dd / "tiles").exists():
        shutil.copytree(src / "tiles", dd / "tiles")
    ob = json.loads((src / "obstacles.json").read_text())
    ob["obstacles"] = {"wallT": {"kind": "box", "box": [1400, 560, 2600, 632], "contacts": {"top": {"y": 560}},
                                 "occlude": "rect"}}
    (dd / "obstacles.json").write_text(json.dumps(ob))

    def pk(name, route, x0=700):
        d = {"street": "wallT", "seed": 3, "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
             "moves": [{"parkour": {"from": 4.1, "from_x": x0, "v0": 5.0, "gait": "run", "to": "5.4.4",
                                    "cycle": "run_SPR", "route": route}}]}
        return _with(world, name, d)
    ch = pk("vpk14", [{"hop": {"obstacle": "wallT", "sequence": "leap_SPR", "onto": True}},
                      {"drop": {"edge": 2600, "sequence": "leap_SPR"}}])
    assert not ch.errors, [str(i) for i in ch.issues]
    notes = ch.notes["move 1 (parkour)"]
    fl0 = notes["flights"][0]
    wall_m = 72 / notes["px_per_m"]
    assert 0.09 <= fl0["clear_m"] < 0.3 and fl0["rise_m"] < wall_m + 0.35  # a step up: feet just over the rim
    assert fl0["x_land"] < 1400 + 0.3 * 1200  # down early on the top, with room to run along it
    t_land, t_drop = fl0["t_to"] + fl0["T"], notes["flights"][1]["t_to"]
    run = [e for e in ch.events if e.tag == "pk.run" and t_land < ch.grid.t(e.q) < t_drop - 1e-6]
    assert len(run) >= 2 and all(abs(e.y - 560) < 1e-6 for e in run)  # every stride on the top, the last too
    from aimv.paint import qa as QA
    hist = History(ch)
    photos = ch.photos()
    res = QA.run(ch, hist, photos, {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")})
    assert not [f for f in res["flags"] if f["level"] == "error" and f["rule"] in ("feet", "inside", "solid")]
    ch = pk("vpk15", [{"hop": {"obstacle": "wallT", "sequence": "leap_SPR", "onto": True}}], x0=1395)
    assert any("no arc comes down on its top" in str(i) for i in ch.errors), [str(i) for i in ch.issues]


def test_trail_in_fades_a_moves_trail_in_from_the_light_trail(world):
    """A move's `trail_in: <seconds>` (v8b, the climb after verse 2's light trail): its figure events carry trail_w,
    0 at its first pose rising to 1 over those seconds; the renderer blends their ghosts and paint-outs from the light
    trail's (washing away) to their own, so the trail fades in. Events, keepers and every random choice stay as they
    are; past the ramp the paint is unchanged."""
    seg = [{"from": 4.1, "to": 6.1, "rate": "16ths"}]
    run = {"from": 4.1, "to": 5.4, "x": [600, 1400]}
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts", "segments": seg, "moves": [{"run": run}]}
    a = _with(world, "vtrin0", d)
    b = _with(world, "vtrin1", dict(d, moves=[{"run": dict(run, trail_in=0.5)}]))
    assert not a.errors and not b.errors, [str(i) for i in b.issues]
    fa = [e for e in a.events if e.layer == "wall" and e.pose]
    fb = [e for e in b.events if e.layer == "wall" and e.pose]
    assert [(e.q, e.pose, e.x, e.y, e.keeper) for e in fa] == [(e.q, e.pose, e.x, e.y, e.keeper) for e in fb]
    g = b.grid
    t0 = g.t(fb[0].q)
    ws = [e.flags.get("trail_w", 1.0) for e in fb]
    assert ws[0] == 0.0 and ws == sorted(ws) and 0 < ws[2] < 1
    for e, w in zip(fb, ws):
        dt = g.t(e.q) - t0
        assert w == (1.0 if dt >= 0.5 else pytest.approx(dt / 0.5, abs=1e-3))
    ha, hb = History(a), History(b)
    pa = [m for m in ha.marks if m.kind in ("pose", "paintout")]
    pb = [m for m in hb.marks if m.kind in ("pose", "paintout")]
    assert [(m.kind, m.x0, m.y0, m.keeper) for m in pa] == [(m.kind, m.x0, m.y0, m.keeper) for m in pb]
    first = next(m for m in pb if m.kind == "pose")
    assert first.keeper and first.trail_w == 0.0  # on the beat: a keeper, but its trail not in yet
    t = first.t + 0.6
    age = hb.frames_since(first, t) * hb.frame_dt
    assert hb.ghost_strength(first, t) == pytest.approx(0.33 * math.exp(-age / 0.12))  # as the light trail
    assert ha.ghost_strength(pa[pb.index(first)], t) > 0.2  # its own: a ghost that stays
    late = [(ma, mb) for ma, mb in zip(pa, pb) if mb.kind == "pose" and mb.trail_w >= 1.0]
    assert late and all(ha.ghost_strength(ma, t + 1) == hb.ghost_strength(mb, t + 1) for ma, mb in late)
    outs = [m for m in pb if m.kind == "paintout"]
    assert all(m.trail_w == hb.marks[m.of].trail_w for m in outs)
    bad = _with(world, "vtrin2", dict(d, moves=[{"run": dict(run, trail_in=-1)}]))
    assert any("trail_in" in str(i) for i in bad.errors)


def test_a_pose_drip_in_is_formed_top_down_and_a_stage_carries_its_drops_and_drips(world):
    """v8b, cold open B: a pose's `drip_in: {to}` forms him by paint running down the wall from `at` to `to` (none
    of him before, more of him photo by photo, whole by `to`, no primer patch); a marks stage's own `drops`, `lead`
    (already flying on its photo) and `drips: [n, length]`."""
    d = {"street": "main", "seed": 3, "motion": "m0", "primer": "ghosts",
         "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}],
         "moves": [{"marks": {"stages": [[4.1, "SPL-1", 900, 300, 300,
                                           {"splash": True, "drops": 90, "lead": 0.07, "drips": [12, 150]}]]}},
                   {"pose": {"at": 4.1, "pose": "RUN-1", "x": 900, "still": True,
                             "drip_in": {"to": "4.4.4", "streaks": 20, "lead": 0.3}}}]}
    ch = _with(world, "vdripin", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    e = next(e for e in ch.events if e.pose == "RUN-1")
    assert e.flags["patch"] == 0.0 and e.flags["drip_in"]["streaks"] == 20
    hist = History(ch)
    g = ch.grid
    m = next(m for m in hist.marks if m.kind == "pose")
    assert hist.reveal[m.id]["kind"] == "drip"
    box = hist.fig_box[e.q]
    x0, y0, w, h = int(box[0]) - 40, int(box[1]) - 40, int(box[2] - box[0]) + 80, int(box[3] - box[1]) + 120
    t0, t1 = g.t(g.resolve(4.1)[0]), g.t(g.resolve("4.4.4")[0])
    full = float((hist.mask(m) > 0.02).sum())
    areas, tops = [], []
    for u_ in (0.1, 0.4, 0.7, 1.0):
        mk = hist._dripped(hist.reveal[m.id], t0 + u_ * (t1 - t0))
        areas.append(float((mk > 0.3).sum()))
        tops.append(int(np.nonzero(mk.max(axis=1) > 0.3)[0].max()) if mk.max() > 0.3 else -1)
    assert areas == sorted(areas) and areas[0] < 0.6 * areas[-1] and areas[-1] >= 0.95 * full  # top down, whole
    assert tops == sorted(tops)  # the paint's lowest point runs down
    assert hist._dripped(hist.reveal[m.id], t0 - 0.05) is None
    sp = next(mk for mk in hist.marks if mk.kind == "splash")
    assert sp.drips[0] == 12 and abs(sp.drips[1] - 150 * ch.u) < 1e-6  # the stage's own drips
    assert sum(1 for dr in hist.drops if abs(dr[0] - (t0 - 0.07)) < 1e-6) == 90  # its drops already flying
    d["moves"][1]["pose"]["drip_in"] = {"streaks": 3}
    assert any("drip_in" in str(i) for i in _with(world, "vdripin2", d).errors)


def test_camera_keep_false_frames_off_him_and_the_frame_rule_skips_it(world):
    """v8b: a segment camera's `keep: false` (a framing on something else: the cold open's splat, the hood close-up)
    leaves the keep-inside rule out, so the eye level holds even when his figure leaves the frame, and QA's frame rule
    does not flag those photos."""
    from aimv.paint import qa as QA
    base = {"street": "main", "seed": 3, "motion": "m0",
            "moves": [{"pose": {"at": 4.1, "pose": "RUN-1", "x": 900, "still": True}}]}
    out = {}
    for keep in (True, False):
        cam = {"preset": "follow", "zoom": 0.45, "eye": 1.6, "snap": True, **({} if keep else {"keep": False})}
        d = dict(base, segments=[{"from": 4.1, "to": 4.3, "rate": "16ths", "camera": cam}])
        ch = _with(world, f"vkeep{keep}", d)
        assert not ch.errors, [str(i) for i in ch.issues]
        hist = History(ch)
        photos = ch.photos()
        cams = {"16:9": plan(ch, photos, hist.figure_boxes(), "16:9")}
        res = QA.run(ch, hist, photos, cams)
        out[keep] = (cams["16:9"][0].cy, sum(f["rule"] == "frame" for f in res["flags"]))
    assert out[False][0] < out[True][0] - 20  # up at the eye level asked for, not pulled back down onto him
    assert out[False][1] == 0


def test_a_clip_to_run_handover_reports_the_speed_change(world, boxclip):
    """v8b: motion_stats' handovers: every clip handing over to a run (and a run to a clip) on consecutive photos,
    with his torso speed over the two photos before and after; the QA warns over HANDOVER_DV (1.5 m/s)."""
    from aimv.paint.check import HANDOVER_DV
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"clip": {"sequence": "flat_SPR", "keys": {"a": 4.1, "b": 4.4}, "x": {"a": 600, "b": 700}}},
                   {"run": {"from": "4.4.2", "to": 5.3, "x": [720, 2400]}}]}
    ch = _with(world, "vhandover", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hs = motion_stats(ch)["handovers"]
    h = next(h for h in hs if h["dir"] == "clip-run")
    assert h["v1"] - h["v0"] > HANDOVER_DV and h["dv"] == pytest.approx(h["v1"] - h["v0"], abs=0.02)


def test_parkour_starts_a_run_after_a_slow_clip_with_the_drive(world, boxclip):
    """v8b (never full speed from a crouch or a mantle): a run stretch that starts slower than DRIVE_V right after
    a clip and has to build speed starts with the drive's drawings in real time (its torso where the solver's
    acceleration puts him), then the cycle; `drive: false` leaves it out; a fast exit gets none."""
    route = [{"clip": {"sequence": "flat_SPR", "from": "a", "to": "b"}}, {"run": {"gait": "sprint"}}]
    drv = ["flat_SPR", 20, 40]
    ch = _pk(world, "vdrive1", route, 200, to="5.4.4", extra={"v0": 1.0, "drive": drv})
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    ev = sorted([e for e in ch.events if e.tag in ("pk.clip", "pk.drive", "pk.run")], key=lambda e: e.q)
    tags = [e.tag for e in ev]
    i0 = tags.index("pk.drive")
    assert tags[i0 - 1] == "pk.clip" and "pk.run" in tags[i0:]  # clip, then the drive, then the cycle
    dr = [e for e in ev if e.tag == "pk.drive"]
    assert len(dr) >= 3 and all(e.pose.startswith("SPR") for e in dr)
    xs = [e.x for e in ev[i0:]]
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:]))  # on along the solver's track
    ch = _pk(world, "vdrive2", route, 200, to="5.4.4", extra={"v0": 1.0, "drive": False})
    assert not any(e.tag == "pk.drive" for e in ch.events)
    ch = _pk(world, "vdrive3", route, 200, to="5.4.4", extra={"drive": drv})  # (in at 7 m/s: no get-up)
    assert not any(e.tag == "pk.drive" for e in ch.events)
    bad = _pk(world, "vdrive4", route, 200, to="5.4.4", extra={"drive": ["nope", 1, 2]})
    assert any("drive" in str(i) for i in bad.errors)


def test_a_spring_stretch_after_another_framing_starts_where_that_one_was(world, boxclip):
    """v8b (the director at 78.8, "a sudden camera shift"): a spring stretch after a stretch framed otherwise (a follow)
    starts from where that framing was, moving at its pace, not from the state an earlier spring stretch left: the
    pan carries on smoothly across both hand-overs."""
    route = [{"run": {"gait": "run"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 4.3, "rate": "16ths", "camera": {"preset": "spring"}},
                      {"from": 4.3, "to": 5.1, "rate": "16ths", "camera": {"preset": "follow", "lead": 0.14}},
                      {"from": 5.1, "to": 6.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 5.5, "gait": "run", "to": "5.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    ch = _with(world, "vsprfresh", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    hist = History(ch)
    photos = ch.photos()
    cams = plan(ch, photos, hist.figure_boxes(), "16:9")
    k = next(i for i, p in enumerate(photos) if abs(p.q - ch.grid.resolve(5.1)[0]) < 1e-6)
    xs = np.array([c.base[0] for c in cams])
    step = np.diff(xs)
    # across the hand-over back to the spring: no jump (each photo's pan within 40% of the photo before's)
    assert all(abs(step[i] - step[i - 1]) <= 0.4 * abs(step[i - 1]) + 4 for i in range(k - 2, k + 4))


def test_a_clip_glide_is_one_smooth_deceleration_between_its_keys(world, boxclip):
    """v8b (the duct slide): a parkour clip's `glide: [k1, k2]` lays its travel from k1 to k2 as one constant
    deceleration (its distance and times as they were: pinned or as shot), from the speed that distance needs down
    to the clip's own pace into k2."""
    base = {"sequence": "flat_SPR", "from": "a", "to": "b", "speed": 1.0, "x": {"a": 400, "b": 900}}
    out = {}
    for gl in (None, ["a", "b"]):
        route = [{"clip": dict(base, **({"glide": gl} if gl else {}))}, {"run": {"gait": "run"}}]
        ch = _pk(world, f"vglide{bool(gl)}", route, 200, to="5.4.4", extra={"v0": 4.0})
        assert not ch.errors, [str(i) for i in ch.issues]
        g = ch.grid
        cl = sorted([e for e in ch.events if e.tag == "pk.clip"], key=lambda e: e.q)
        out[bool(gl)] = ([g.t(e.q) for e in cl], [e.x for e in cl])
    (t0, x0), (t1, x1) = out[False], out[True]
    assert t0 == t1 and abs(x0[0] - x1[0]) < 2.0 and abs(x0[-1] - x1[-1]) < 2.0  # the same keys, times, ends
    v = np.diff(x1) / np.diff(t1)
    assert np.all(np.diff(v[:-1]) <= 1.0)  # slowing down all the way (no surge)
    bad = _pk(world, "vglide2", [{"clip": dict(base, glide=["a"])}], 200, to="5.4.4")
    assert any("glide" in str(i) for i in bad.errors)


def test_parkour_roll_to_speed_and_out_speed_shorten_and_slow_the_roll(world, boxclip):
    """v8b (P7, the big leap's landing): a jump's `roll_to` ends the roll at that key of its roll clip, `roll_speed`
    plays it faster, and `roll_v` (m/s) slows him evenly through it to that speed (instead of rolling on at his
    landing speed); the run after it starts from there."""
    base = {"gap": "alley", "sequence": "leap_SPR", "at": 5.1, "roll": True, "roll_sequence": "roll_SPR"}
    out = {}
    for name, extra in (("plain", {}), ("short", {"roll_speed": 1.5, "roll_v": 2.0})):
        ch = _pk(world, f"vroll{name}", [{"jump": dict(base, **extra)}, {"run": {"gait": "run"}}],
                 _pk_alley_x(world), to="6.1")
        assert not ch.errors, [str(i) for i in ch.issues]
        g = ch.grid
        r = sorted([e for e in ch.events if e.tag == "pk.roll"], key=lambda e: e.q)
        out[name] = (g.t(r[-1].q) - g.t(r[0].q), [e.x for e in r], [g.t(e.q) for e in r])
    assert out["short"][0] < out["plain"][0] * 0.8  # quicker
    xs, ts = np.array(out["short"][1]), np.array(out["short"][2])
    v = np.diff(xs) / np.diff(ts)
    assert v[-1] < v[0] * 0.7 and v[-1] / 125.0 < 3.5  # slowing toward 2 m/s (the test figure: 125 px a metre)


def test_parkour_run_brake_spreads_the_slowdown(world, boxclip):
    """v8b (P7, into the vaults): a run item's `brake` (m/s2) is how hard he slows to its gait, so the slowdown into a
    move spreads over the last strides instead of the solver's hard A_DOWN in one."""
    out = {}
    for b in (None, 2.0):
        route = [{"run": {"gait": "sprint", "x": 900}}, {"run": dict({"gait": 3.0, "x": 1900}, **({"brake": b} if b else {}))}]
        ch = _pk(world, f"vbrake{b}", route, 200, to="5.4.4")
        g = ch.grid
        ev = sorted([e for e in ch.events if e.tag == "pk.run"], key=lambda e: e.q)
        xs, ts = np.array([e.x for e in ev]), np.array([g.t(e.q) for e in ev])
        v = np.diff(xs) / np.diff(ts) / 125.0
        k = int(np.searchsorted(xs, 900))
        out[b] = v[k:k + 12]
    # the gentle brake still slowing several photos on, the hard one already at its gait
    assert out[2.0][4] > out[None][4] + 0.5


def test_parkour_run_accel_sets_how_hard_he_speeds_up(world, boxclip):
    """v8b (P7, out of the vaults): a run item's `accel` (m/s2) is how hard he speeds up to its gait (the solver's
    A_UP, 4, otherwise), so a run between two close moves can build its speed sooner and leave room to ease off into
    the next one."""
    out = {}
    for a in (None, 8.0):
        route = [{"run": {"gait": "sprint", "x": 400}}, {"run": {"gait": 3.0, "x": 900}},
                 {"run": dict({"gait": "sprint", "x": 1900}, **({"accel": a} if a else {}))}]
        ch = _pk(world, f"vaccel{a}", route, 200, to="5.4.4")
        g = ch.grid
        ev = sorted([e for e in ch.events if e.tag in ("pk.run", "pk.drive")], key=lambda e: e.q)
        xs, ts = np.array([e.x for e in ev]), np.array([g.t(e.q) for e in ev])
        v = np.diff(xs) / np.diff(ts) / 125.0
        k = int(np.searchsorted(xs, 900))
        out[a] = v[k:k + 8]
    # the hard one up to speed several photos sooner
    assert out[8.0][3] > out[None][3] + 0.5
    assert max(out[8.0]) <= 7.0 * 1.1 + 0.3


def test_spring_camera_frame_cx_v_is_the_9x16s_own_fixed_target(world, boxclip):
    """v8b (P7's ending): a frame's `cx_v` is the 9:16's own fixed target where the 16:9 takes `cx` (the 9:16 holds
    the door and the STOP sign whole while the 16:9 pushes toward the door)."""
    route = [{"run": {"gait": "sprint"}}]
    d = {"street": "main", "seed": 3,
         "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths", "camera": {"preset": "spring"}}],
         "moves": [{"parkour": {"from": 4.1, "from_x": 200, "v0": 7.0, "gait": "sprint", "to": "6.4.4",
                                "route": route, "cycle": "run_SPR"}}]}
    plain = _with(world, "vpkcxv0", d)
    photos = plain.photos()
    t = np.array([p.t for p in photos])
    k1 = int(np.argmin(np.abs(t - plain.grid.t(plain.grid.resolve("5.4")[0]))))
    x1 = plan(plain, photos, History(plain).figure_boxes(), "16:9")[k1].base[0]
    out = {}
    for name, extra in (("vpkcxv1", {}), ("vpkcxv2", {"cx_v": x1 - 120})):
        d2 = dict(d, moves=d["moves"] + [{"hints": {"frame": dict({"from": 4.3, "to": 5.4, "cx": x1, "ease": [0.4, 0.4]},
                                                                  **extra)}}])
        ch = _with(world, name, d2)
        hb = History(ch).figure_boxes()
        out[name] = (plan(ch, photos, hb, "16:9")[k1].base[0], plan(ch, photos, hb, "9:16")[k1].base[0])
    assert abs(out["vpkcxv2"][0] - out["vpkcxv1"][0]) < 1.0  # the 16:9 as without it
    assert abs(out["vpkcxv1"][1] - x1) < 40  # the 9:16 on cx without its own
    assert abs(out["vpkcxv2"][1] - (x1 - 120)) < 40  # and on its own cx_v with it


def test_box_key_sides_are_painted_one_by_one(world):
    """v8b (the cold open C3): a key's `sides` paints only those sides, each from its own time (wet and dripping from
    then), with the strokes of the full outline: he slams into its right side, the top and bottom follow on the
    beats and the left side closes on "box"."""
    from aimv.paint.props import BoxKey
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}],
         "moves": [{"box": {"id": "his", "x": 2200, "keys": [
             {"at": 4.1, "w": 200, "h": 260, "sides": {"right": 4.1, "top": 4.2, "bottom": 4.3, "left": 5.1}}]}},
                   {"pose": {"at": 4.2, "pose": "RUN-1", "x": 600}}]}
    ch = _with(world, "boxsides", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    g = ch.grid
    k = ch.props["his"].keys[0]
    assert [n for n, _ in k.sides] == ["right", "top", "bottom", "left"]
    assert k.sides[0][1] == 0.0 and k.sides[3][1] == pytest.approx(g.t(g.resolve(5.1)[0]) - g.t(g.resolve(4.1)[0]))
    # the parts are the full outline's own strokes
    full, x0, y0 = PR.outline(BoxKey(k.q, k.cx, k.bottom, k.w, k.h, k.thick, 0.0, k.seed))
    parts, px0, py0 = PR.outline(k, parts=True)
    union = 1 - np.prod([1 - parts[n] for n in parts], axis=0)
    assert (px0, py0) == (x0, y0) and np.abs(union - full).max() < 1e-5
    hist = History(ch)
    marks = sorted([m for m in hist.marks if m.kind == "box"], key=lambda m: m.t)
    assert len(marks) == 4 and [round(m.t, 3) for m in marks] == [round(g.t(g.resolve(a)[0]), 3)
                                                                    for a in ("4.1", "4.2", "4.3", "5.1")]
    t = g.t(g.resolve("4.2.2")[0])
    shown = [m for m in marks if m.t <= t + 1e-6]
    assert len(shown) == 2  # the right side and the top so far
    st = hist._fx_state(marks[1], t, ch.paint_clock(t))
    xs = np.nonzero(st[0].max(axis=0) > 0.5)[0] + marks[1].x0
    assert xs.min() < k.cx - k.w / 2 + k.thick and xs.max() > k.cx + k.w / 2 - k.thick  # the top spans the width
    assert st[2] > hist._fx_state(marks[0], t, ch.paint_clock(t))[2]  # the newer side is the wetter


def test_move_stream_keeps_its_random_stream_when_moves_are_added_before_it(world):
    """v8c (cold open C3): a move's `stream: n` paints it with the random stream of its file's n-th move, so a variant
    that drops or adds moves before it (C3's row boxes) paints it exactly as the file it came from (C's crouch, its
    primer swath)."""
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 5.1, "rate": "16ths"}]}
    pose = {"pose": {"at": 4.2, "pose": "RUN-1", "x": 600}}
    a = _with(world, "stream0", dict(base, moves=[pose]))
    extra = {"box": {"id": "other", "x": 1500, "keys": [{"at": 4.1, "w": 200, "h": 260}]}}
    b = _with(world, "stream1", dict(base, moves=[extra, pose]))
    c = _with(world, "stream2", dict(base, moves=[extra, {"pose": dict(pose["pose"], stream=0)}]))
    for ch in (a, b, c):
        assert not ch.errors, [str(i) for i in ch.issues]
    assert [m.local_i for m in c.moves] == [0, 0] and b.moves[1].local_i == 1
    def primer(ch):
        h = History(ch)
        return [m.gen for m in h.marks if m.kind == "primer"]
    assert primer(c) == primer(a) and primer(b) != primer(a)


def test_move_wash_false_keeps_held_paint_crisp(world):
    """v8c (p5c, the director: "still as paint"): a move's `wash: false` (or `wash: {from: time}` for its poses from then
    on) keeps the rain from washing its paint: a pose held on the wall for seconds stays as dark as when painted."""
    base = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}]}
    out = {}
    for name, extra in (("wash0", {}), ("wash1", {"wash": False})):
        ch = _with(world, name, dict(base, moves=[{"pose": dict({"at": 4.2, "pose": "RUN-1", "x": 600}, **extra)}]))
        assert not ch.errors, [str(i) for i in ch.issues]
        hist = History(ch)
        m = next(m for m in hist.marks if m.kind == "pose")
        assert m.nowash == bool(extra)
        g = ch.grid
        t0 = g.t(g.resolve(4.2)[0])
        cov = [float(hist.layers(t0 + dt, int(m.x0), int(m.y0), int(m.w), int(m.h))[1].sum()) for dt in (0.35, 3.0)]
        out[name] = cov[1] / cov[0]  # how much of it is left after 3 s against just after the wash would begin
    assert out["wash1"] > out["wash0"] + 0.01  # his paint held crisp (only the primer under it still washes)


def test_move_primer_clip_keeps_its_primer_inside_the_box(world):
    """v8c (p5e, the director: "turning him into one ... while he is still in the wall"): a move's `primer_clip: [x0, y0,
    x1, y1]` (null: open) keeps the primer it lays inside that box: its poses' patches, their paint-outs and a
    stamp's primer, so a step can carry him off the swath. Without it they reach past."""
    base = {"street": "main", "seed": 3, "primer": "ghosts", "motion": "m0",
            "segments": [{"from": 4.1, "to": 6.1, "rate": "16ths"}]}

    def moves(extra):
        return [{"pose": dict({"at": a, "pose": "RUN-1", "x": x}, **extra)}
                for a, x in ((4.2, 600), (4.3, 640), (4.4, 680), (5.1, 720))] + \
            [{"marks": dict({"stages": [[4.1, "SPL-1", 640, 300, 300, {"primer": "keep"}]]}, **extra)}]

    reach = {}
    for name, extra in (("pclip0", {}), ("pclip1", {"primer_clip": [None, None, 650, None]})):
        ch = _with(world, name, dict(base, moves=moves(extra)))
        assert not ch.errors, [str(i) for i in ch.issues]
        hist = History(ch)
        kinds, far = set(), 0.0
        for m in hist.marks:
            if m.kind not in ("primer", "paintout"):
                continue
            mk = hist.mask(m)
            cols = np.nonzero((mk > 0.05).any(0))[0]
            if len(cols):
                kinds.add(m.kind)
                far = max(far, m.x0 + cols.max())
        assert kinds == {"primer", "paintout"}, kinds
        reach[name] = far
    assert reach["pclip0"] > 700 and reach["pclip1"] <= 650 + 20, reach  # (its dry edge wanders a few px)
