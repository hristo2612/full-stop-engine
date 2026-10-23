"""The 2D world round the street (aimv.paint.world): cells placed by their overlap, seams, the overlay tiles, windows
across rows, the scene's maps and masks there, the camera's clamp, levels, voids, the canvas tool, and the 1D street
left exactly as it was. Everything is cut from one synthetic texture (nothing paid, no network)."""

import json
import shutil

import cv2
import numpy as np
import pytest

from aimv.paint import street as ST
from aimv.paint import world as WD
from aimv.paint.scene import Scene

PW, PH = 800, 400  # plates and j0/B cells
KEEP = 0.4
STEP = int(PW * (1 - KEEP))  # 480: the street's plate step and the world's column step
N_PLATES = 7  # P1 .. P7; P7 is the anchor
GY = 700  # texture row of street level's top (anchor y 0)
GX = (N_PLATES - 1) * STEP  # texture column of the anchor's left edge
J1 = (-300, 500)  # the j1 row: y and height (its bottom 200 rows overlap j0)
B = (-550, 400)  # the roof band: y and height (its bottom 150 rows overlap j1)
# id, column, row, jitter (the model's drift off the grid, anchor px), gain
CELLS = [("T1", 1, "j0", (2, -1), 1.0), ("T2", 2, "j0", (-1, 2), 1.0), ("U1", 1, "j1", (3, 1), 1.0),
         ("U2", 2, "j1", (0, -2), 1.06), ("R2", 2, "B", (1, 1), 1.0)]
ROWS = {"j0": (0, PH), "j1": J1, "B": B}


def _texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for sigma, amp in ((1.2, 30), (4, 40), (12, 40), (40, 25)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        img += (n / (n.std() + 1e-6) * amp)[..., None] * np.float32([1.0, 0.85, 0.7])
    img += np.float32([95, 85, 75])
    for _ in range(w * h // 9000):  # windows, pipes, signs: corners for the features
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))),
                      [float(v) for v in rng.uniform(20, 230, 3)], -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _make(root, with_world=True, cells=None, edits=None):
    """A street of N_PLATES plates on the texture's rows GY .. GY + PH, and the world's cells round its last plate."""
    tex = _texture(GX + 3 * STEP + PW + 40, GY + PH + 20, seed=3)
    (root / "boards").mkdir(parents=True, exist_ok=True)
    plates = []
    for n in range(N_PLATES):
        f = root / "boards" / f"P{n + 1}.png"
        cv2.imwrite(str(f), tex[GY:GY + PH, n * STEP:n * STEP + PW])
        plates.append(f"boards/P{n + 1}.png")
    spec = {"version": 1, "plates": plates, "keep": KEEP, "tile": 512, "edits": edits or []}
    world = []
    for cid, col, row, (jx, jy), gain in (cells if cells is not None else CELLS):
        y, h = ROWS[row]
        w = PW
        x0, y0 = GX + col * STEP + jx, GY + y + jy
        img = np.clip(tex[y0:y0 + h, x0:x0 + w].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / "boards" / f"{cid}.png"), img)
        world.append({"id": cid, "path": f"boards/{cid}.png", "at": [col * STEP, y]})
    if with_world:
        spec["world"] = {"anchor": plates[-1], "cells": world}
    sd = root / "paint" / "streets" / "main"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "street.json").write_text(json.dumps(spec))
    (sd / "obstacles.json").write_text(json.dumps({
        "version": 1, "ground": {"wall_base": 360, "feet": 352},
        "obstacles": {"crate": {"kind": "box", "box": [300, 250, 380, 352], "occlude": "rect",
                                "contacts": {"top": {"y": 250}}}},
        "lights": [{"x": 500, "y": 200, "sigma": 150}]}))
    ST.build(root, sd, log=lambda m: None)
    return {"root": root, "sd": sd, "tex": tex, "spec": spec}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("world2d") / "proj"
    w = _make(root)
    flat_root = tmp_path_factory.mktemp("flat") / "proj"
    w["flat"] = _make(flat_root, with_world=False)
    return w


def _street_to_tex(sd):
    """Street px -> texture px for this test street: the strip's crop (its first plate is the texture's left edge)."""
    b = json.loads((sd / "build.json").read_text())["strip"]
    return -b["crop"][0], GY + b["crop"][1] - ST.OY


def test_cells_are_placed_by_their_overlap_with_the_world(built):
    info = json.loads((built["sd"] / "world" / "world.json").read_text())
    A = np.float64(info["anchor"]["A"])
    ox, oy = _street_to_tex(built["sd"])
    assert abs(A[0, 2] + ox - GX) < 0.5 and abs(A[1, 2] + oy - GY) < 0.5  # the anchor is the last plate
    by = {c["id"]: c for c in info["cells"]}
    for cid, col, row, (jx, jy), _ in CELLS:
        c = by[cid]
        T = np.float64(c["T"])
        tx, ty = GX + col * STEP + jx - ox, GY + ROWS[row][0] + jy - oy  # where the texture says it is
        assert abs(T[0, 2] - tx) < 0.5 and abs(T[1, 2] - ty) < 0.5, (cid, T[:, 2], tx, ty)
        assert abs(c["scale"] - 1) < 2e-3 and c["inliers"] >= 12
        assert c["nominal_dxy"] == pytest.approx([jx, jy], abs=0.6)
    sides = {c["id"]: sorted(s["side"] for s in c["seams"]) for c in info["cells"]}
    assert sides["T1"] == ["left"] and sides["T2"] == ["left"]
    assert sides["U2"] == ["bottom", "left"]  # an L of neighbours: both seams
    assert sides["R2"] == ["bottom"]
    for c in info["cells"]:  # every seam in the middle of its overlap
        for s in c["seams"]:
            b = s["band"]
            lo, hi = (b[0], b[2]) if s["side"] in ("left", "right") else (b[1], b[3])
            assert lo + 0.25 * (hi - lo) < s["at"] < lo + 0.75 * (hi - lo)


def test_the_world_reads_as_the_texture_across_rows(built):
    s = ST.Street(built["sd"])
    ox, oy = _street_to_tex(built["sd"])
    x0, x1 = GX - ox - 200, GX - ox + 2 * STEP + PW - 40
    y0, y1 = GY + B[0] - oy + 10, s.H
    rgb, v = s.window(x0, x1, y0, y1, valid=True)
    ref = built["tex"][y0 + oy:y1 + oy, x0 + ox:x1 + ox][..., ::-1].astype(int)
    d = np.abs(rgb.astype(int) - ref).mean(2)
    # the cells without a gain drift match the texture; U2 (6 % brighter) is matched near its seams
    assert v.mean() > 0.6 and np.median(d[v]) < 2.0
    # invalid: above the street left of the world's cells, where no cell is
    assert not v[:-y0, :150].any()
    assert v[-50:-4, :].all()  # street level: the street, then the cells right of it (a cell's edge rows aside)
    # the overlay leaves the street's own tiles alone (they are the 1D street's, as built without the world)
    flat = json.loads((built["flat"]["sd"] / "build.json").read_text())["street"]["tiles"]
    assert json.loads((built["sd"] / "build.json").read_text())["street"]["tiles"] == flat
    # left of the world: the plain street, edge-replicated outside it as always
    f = ST.Street(built["flat"]["sd"])
    assert np.array_equal(s.window(-30, 600), f.window(-30, 600))
    assert np.array_equal(s.window(100, 500, 20, 300), f.window(100, 500)[20:300])


def test_gain_is_matched_at_the_seams(built):
    info = json.loads((built["sd"] / "world" / "world.json").read_text())
    s = ST.Street(built["sd"])
    ox, oy = _street_to_tex(built["sd"])
    u2 = next(c for c in info["cells"] if c["id"] == "U2")
    for sm in u2["seams"]:
        b = sm["band"]
        if sm["side"] == "left":
            xa, xb, ya, yb = int(sm["at"]) + 20, int(sm["at"]) + 120, b[1] + 20, b[3] - 20
        else:
            xa, xb, ya, yb = b[0] + 20, b[2] - 20, int(sm["at"]) - 120, int(sm["at"]) - 20
        rgb = s.window(xa, xb, ya, yb).astype(float)
        ref = built["tex"][ya + oy:yb + oy, xa + ox:xb + ox][..., ::-1].astype(float)
        assert abs(rgb.mean() / ref.mean() - 1) < 0.015  # the 6 % drift is gone next to the seam


def test_tiles_have_hashes_and_adding_a_cell_keeps_the_others(tmp_path):
    w = _make(tmp_path / "p", cells=CELLS[:4])
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    for k, h in before["tiles"].items():
        t = cv2.imread(str(sd / "world" / f"w{k}.png"), cv2.IMREAD_UNCHANGED)
        assert t.shape == (512, 512, 4) and set(np.unique(t[..., 3])) <= {0, 255}
    spec = json.loads((sd / "street.json").read_text())
    cid, col, row, (jx, jy), gain = CELLS[4]
    y, h = ROWS[row]
    x0, y0 = GX + col * STEP + jx, GY + y + jy
    cv2.imwrite(str(w["root"] / "boards" / f"{cid}.png"), w["tex"][y0:y0 + h, x0:x0 + PW])
    spec["world"]["cells"].append({"id": cid, "path": f"boards/{cid}.png", "at": [col * STEP, y]})
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    after = json.loads((sd / "world" / "world.json").read_text())
    assert any("4 cells kept" in m for m in logs)
    assert len(after["cells"]) == 5 and after["cells"][:4] == before["cells"]
    T = 512
    rx0, ry0, rx1, ry1 = after["cells"][4]["box"]
    reach = {(ix, iy) for ix in range((rx0 - 8) // T, (rx1 + 8) // T + 1) for iy in range((ry0 - 8) // T, (ry1 + 8) // T + 1)}
    for k, h in before["tiles"].items():
        if WD._wkey(k) not in reach:
            assert after["tiles"][k] == h, k
    assert len(after["tiles"]) > len(before["tiles"])


def test_scene_maps_and_masks_in_two_dimensions(built):
    sc = Scene(built["sd"], built["root"] / "paint" / ".cache")
    flat = Scene(built["flat"]["sd"], built["flat"]["root"] / "paint" / ".cache")
    assert sc.Y0 < B[0] + 10 and sc.Y1 >= sc.H and sc.xw is not None
    # the plain street, far enough left of the world: the same maps and masks as without the world
    a, b = sc.span(0, 300), flat.span(0, 300)
    for k in ("lin", "relief", "spec", "e_wall", "tint", "occ", "paint"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert all(np.array_equal(u, v) for u, v in zip(sc.masks(0, 400), flat.masks(0, 400)))
    # in the world, any rows: everything finite, paint only where there are pixels (and above the pavement)
    x0 = sc.xw + 100
    sp = sc.span(x0, x0 + 900, -400, 300)
    assert (sp.y0, sp.y1) == (-400, 300) and sp.lin.shape == (700, 900, 3)
    for k in ("lin", "relief", "e_wall", "tint", "paint"):
        assert np.isfinite(getattr(sp, k)).all(), k
    _, v = sc.street.window(x0, x0 + 900, -400, 300, valid=True)
    far = cv2.erode((~v).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0  # 5 px from any pixel
    assert sp.paint[far].max() < 0.01 and sp.paint[v].mean() > 0.8
    assert sp.covers(x0 + 10, x0 + 20, -100, 0) and not sp.covers(x0 + 10, x0 + 20, -500, 0)
    # a span with the default rows across the world's first column is the street's height, its maps blended
    sp2 = sc.span(sc.xw - 300, sc.xw + 300)
    assert sp2.lin.shape[0] == sc.H and np.isfinite(sp2.e_wall).all()
    # masks across rows: the crate's occluder on the street, nothing above
    occ, pnt = sc.masks(280, 400, -100, 400)
    assert occ[100 + 300, 40] > 0.9 and occ[:100].max() == 0 and pnt[:100].max() == 0
    out = sc.preview(built["root"] / "masks.jpg")
    assert out.exists()


def test_world_cells_above_the_streets_left_part_keep_the_world_start_and_get_their_own_light(built, tmp_path):
    from types import SimpleNamespace
    from aimv.paint.render import History
    sd = tmp_path / "streets" / "main"
    shutil.copytree(built["sd"], sd)
    wj = sd / "world" / "world.json"
    info = json.loads(wj.read_text())
    ov = info["overlay"]
    plain = Scene(sd, tmp_path / ".cache")
    assert plain.xw == min(ov[0], plain.W) and plain.bands == []  # without the keys: as before
    # a band of cells above the street's left part (world.py writes these keys when it has `floor` cells): the
    # world's first column stays where the world replaces the street
    band = [0, -500, 600, 0]
    info["xw"], info["bands"] = ov[0], [band]
    wj.write_text(json.dumps(info))
    sc = Scene(sd, tmp_path / ".cache")
    assert sc.xw == plain.xw and sc.bands == [tuple(band)]
    assert sc.street.street_only(100, 500, 0, sc.H)  # the street's own rows: the plain street, its 1D path
    a, b = sc.span(100, 500), plain.span(100, 500)
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("lin", "e_wall", "tint"))
    # rows above the street there: the band's own light maps, not the world's maps replicated from its far edge
    real = sc.lf_world()
    sc._lf_world = dict(real, ox=np.int32(900))  # (the world's maps starting right of the band)
    sp = sc.span(100, 500, -300, 300)
    lb = sc.lf_band(tuple(band))
    want = sc._upw(lb["loc"], int(lb["ox"]), int(lb["oy"]), 100, -300, 400, 600)
    gotw = sc._upw(real["loc"], 900, int(real["oy"]), 100, -300, 400, 600)
    assert not np.allclose(want[:300], gotw[:300])
    sc2 = Scene(sd, tmp_path / ".cache")
    sc2._lf_world = dict(real, ox=np.int32(900))
    sc2.bands = []
    sp_noband = sc2.span(100, 500, -300, 300)
    assert not np.allclose(sp.e_wall[:300], sp_noband.e_wall[:300])  # the band's maps are what lights it
    assert np.allclose(sp.e_wall[300:], sp_noband.e_wall[300:])  # (the street's rows: the street's own)
    # the skin check: left of the world's start, a pose reaching above the street, over no pixels, is a skin;
    # the same pose on the street's rows is the plain street
    h = SimpleNamespace(ch=SimpleNamespace(scene=sc))
    e = SimpleNamespace(flags={}, x=300.0)
    d = np.ones((120, 60), np.float32)
    assert History._over_void(h, e, d, 270.0, 100.0) is False
    assert History._over_void(h, e, d, 270.0, -700.0) is True


def test_an_occluder_reaching_above_the_streets_left_part_masks_its_street_rows(built, tmp_path):
    """v8 (P2b's barrier, a prop standing on a scaffold deck above the street): an occluder part left of the world
    that reaches above the street's top row occludes the street's rows it covers (the street tile's masks)."""
    sd = tmp_path / "streets" / "main"
    shutil.copytree(built["sd"], sd)
    before, _ = Scene(sd, tmp_path / ".cache0").masks(250, 450)
    ob = json.loads((sd / "obstacles.json").read_text())
    ob["obstacles"]["tallT"] = {"kind": "other", "box": [300, -200, 360, 300],
                                "occlude": [{"box": [300, -200, 360, 300], "method": "rect"}]}
    (sd / "obstacles.json").write_text(json.dumps(ob))
    sc = Scene(sd, tmp_path / ".cache")
    assert sc.street.Y0 < 0 and 360 < sc._x1d()  # (a street with a world above it; the part in its 1D tiles)
    occ, _ = sc.masks(250, 450)
    assert occ[20:280, 60:100].min() > 0.9  # its street rows
    assert np.array_equal(occ[330:], before[330:]) and np.array_equal(occ[:, :40], before[:, :40])  # nothing else


def test_a_floor_cell_above_the_streets_left_part_leaves_its_rows_and_the_world_start(tmp_path):
    w = _make(tmp_path / "p")
    sd, root, tex = w["sd"], w["root"], w["tex"]
    before = json.loads((sd / "world" / "world.json").read_text())
    assert "xw" not in before and "bands" not in before  # no floor cells: world.json as before
    ox, oy = _street_to_tex(sd)
    # a cell over the street's left part (far left of the world): 600 rows, its bottom 200 on the street's top
    # rows, made 8 % brighter than the street
    sx0, sy0, cw, ch = 400, -400, 800, 600
    img = tex[sy0 + oy:sy0 + oy + ch, sx0 + ox:sx0 + ox + cw].astype(np.float32) * 1.08
    cv2.imwrite(str(root / "boards" / "L1.png"), np.clip(img, 0, 255).astype(np.uint8))
    A = np.float64(before["anchor"]["A"])
    u, v = np.linalg.solve(A[:, :2], np.float64([sx0, sy0]) - A[:, 2])  # its at: the anchor's map to that box
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["cells"].append({"id": "L1", "path": "boards/L1.png", "at": [u + 4, v - 3], "floor": 0})
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)
    info = json.loads((sd / "world" / "world.json").read_text())
    rec = next(r for r in info["cells"] if r["id"] == "L1")
    assert rec["floor"] == 0 and rec["inliers"] >= 12 and rec["median_px"] < 1.5  # placed on the street's rows
    assert abs(rec["box"][0] - sx0) <= 2 and abs(rec["box"][1] - sy0) <= 2
    wt = WD.WorldTiles(sd)
    _, al = wt.read(sx0 - 50, sx0 + cw + 50, 0, sy0 + ch + 50)
    assert not al.any()  # nothing on the street's rows: the street keeps them
    rgb, al = wt.read(sx0 + 40, sx0 + cw - 40, sy0 + 20, 0)
    assert al.all()  # above the street: the cell
    got = rgb[-60:].astype(float).mean((0, 1))  # its rows just above the street: toned to the street's
    want = tex[oy - 60:oy, sx0 + ox + 40:sx0 + ox + cw - 40][..., ::-1].astype(float).mean((0, 1))
    assert np.abs(got / want - 1).max() < 0.03
    assert info["xw"] == before["overlay"][0] and info["overlay"][0] < info["xw"]  # the world's start stays
    assert info["bands"] == [[rec["box"][0], rec["box"][1], rec["box"][2], 0]]
    sc = Scene(sd, tmp_path / ".cache")
    assert sc.xw == before["overlay"][0] and sc.bands == [tuple(info["bands"][0])]
    # paint may go on the cell (its rows above the street: the world's masks there), not beside it; the street's
    # rows keep their own masks
    occ, pnt = sc.masks(sx0 - 200, sx0 + cw + 100, sy0 - 50, 300)
    j0, j1 = 200 + 40, 200 + cw - 40
    assert pnt[50 + 20:50 - sy0 - 20, j0:j1].min() > 0.9  # on the cell
    assert pnt[:50 - sy0, :150].max() == 0 and pnt[:40].max() == 0  # left of it and above it: no paint
    o1, p1 = sc.masks(sx0 - 200, sx0 + cw + 100)
    assert np.array_equal(pnt[50 - sy0:], p1[:300]) and np.array_equal(occ[50 - sy0:], o1[:300])


def test_camera_clamp_keeps_crops_on_valid_pixels(built):
    sc = Scene(built["sd"], built["root"] / "paint" / ".cache")
    flat = Scene(built["flat"]["sd"], built["flat"]["root"] / "paint" / ".cache")
    cw, ch = 300.0, 168.75
    old = lambda s, x, y: (float(np.clip(x, s.X0 + cw / 2 + 30, s.W - cw / 2 - 30)),
                           float(np.clip(y, ch / 2 + 30, s.H - ch / 2 - 30)))
    rng = np.random.default_rng(1)
    for x, y in zip(rng.uniform(-200, 1500, 40), rng.uniform(-300, 600, 40)):
        assert flat.clamp_crop(x, y, cw, ch) == old(flat, x, y)  # the plain street: its own rule
        if x < 1000:  # left of the world: the same answer with the world there
            assert sc.clamp_crop(x, y, cw, ch) == old(sc, x, y)
    for x, y in zip(rng.uniform(sc.xw - 200, sc.bounds[2] + 300, 60), rng.uniform(-900, 700, 60)):
        cx, cy = sc.clamp_crop(x, y, cw, ch)
        assert sc.inside(cx - cw / 2 - 30, cy - ch / 2 - 30, cx + cw / 2 + 30, cy + ch / 2 + 30), (x, y, cx, cy)
        if sc.inside(x - cw / 2 - 30, y - ch / 2 - 30, x + cw / 2 + 30, y + ch / 2 + 30):
            assert (cx, cy) == (x, y)
    # a crop up on the roof band's cell stays there (the world has pixels above the street)
    ox, oy = _street_to_tex(built["sd"])
    rx, ry = GX - ox + 2 * STEP + PW / 2, GY + B[0] + B[1] / 2 - oy
    assert sc.clamp_crop(rx, ry, cw, ch) == (rx, ry)
    assert sc.span_rows(10, 300) == (None, None) and sc.span_rows(-200, 100)[0] <= -200
    # a crop wider than any valid area round it stays where it was asked (inside the bounds), never jumping away
    bx, by = sc.clamp_crop(rx, ry, 2400.0, 1350.0)  # taller than the world: no all-valid crop anywhere
    hw = 1200 + 30
    assert bx == float(np.clip(rx, sc.bounds[0] + hw, sc.bounds[2] - hw)) and by == (sc.bounds[1] + sc.bounds[3]) / 2


def test_levels_give_the_ground_a_step_per_x(built, tmp_path):
    sd = tmp_path / "s"
    shutil.copytree(built["sd"], sd)
    o = json.loads((sd / "obstacles.json").read_text())
    o["ground"]["levels"] = [{"id": "walkway", "x": [3500, 3900], "feet": 120},
                             {"id": "roof", "x": [3900, 4400], "feet": [[3900, -400], [4400, -410]]},
                             {"id": "far street", "x": [4400, 5000], "feet": 330, "street": True}]
    o["voids"] = [{"id": "sky", "box": [4000, -560, 4300, -450], "method": "rect", "except": [[4200, -560, 4300, -450]]}]
    (sd / "obstacles.json").write_text(json.dumps(o))
    sc = Scene(sd, tmp_path / "cache")
    assert sc.feet_at(3000) == sc.feet and sc.feet_at(3600) == 120 and sc.feet_at(4150) == pytest.approx(-405)
    assert sc.level_at(3600) == "walkway" and sc.level_at(100) == "street"
    assert sc.wall_base_at(3600) == sc.wall_base  # the pavement stays the street's
    # a street level: the world's own street, drifted; feet and pavement move together there
    assert sc.feet_at(4600) == 330 and sc.wall_base_at(4600) == sc.wall_base + 330 - sc.feet
    assert sc.ground_dy(np.array([100.0, 4600.0])).tolist() == [0.0, 330 - sc.feet]
    v = sc.void(4050, 4100, -540, -470)
    assert v.min() == 1.0
    assert sc.void(4210, 4290, -540, -470).max() == 0.0  # a structure standing in the void (its except box)
    _, pnt = sc.masks(4000, 4200, -560, -450)
    assert pnt[10:-10, 10:-10].max() < 0.05  # voids are never paintable
    measured = sc.measure_line(3500, 3900, lambda x: 120.0)
    assert all(abs(y - 120) <= 60 for _, y in measured)


def test_canvas_holds_the_known_world_in_the_new_cells_frame(built, tmp_path):
    info = json.loads((built["sd"] / "world" / "world.json").read_text())
    # an existing cell's rect: all of it known, its own pixels where it won its seams
    out = tmp_path / "c.png"
    r = WD.canvas(built["root"], built["sd"], "T2", (2 * STEP, 0, 2 * STEP + PW, PH), out)
    c = cv2.imread(str(out)).astype(int)
    assert r["size"] == [PW, PH] and r["known"] > 0.95
    cell = cv2.imread(str(built["root"] / "boards" / "T2.png")).astype(int)
    assert np.median(np.abs(c - cell)[:, PW // 2:]) < 3  # T2's right half is T2 in the world
    # a cell right of the world: its left part known, grey elsewhere
    r = WD.canvas(built["root"], built["sd"], "T3", (3 * STEP, 0, 3 * STEP + PW, PH), out)
    c = cv2.imread(str(out))
    assert r["sides"] == ["left"] and 0.3 < r["known"] < 0.45 and (c[:, -100:] == 128).all()
    ox, oy = _street_to_tex(built["sd"])
    ref = built["tex"][GY + 3:GY + PH - 3, GX + 3 * STEP + 3:GX + 3 * STEP + 200]  # the texture there
    t2 = next(cc for cc in info["cells"] if cc["id"] == "T2")
    dx, dy = t2["nominal_dxy"]  # the canvas follows T2's drift (where the build will look for T3)
    sh = np.float32([[1, 0, -dx], [0, 1, -dy]])
    exp = cv2.warpAffine(built["tex"], sh, (built["tex"].shape[1], built["tex"].shape[0]))
    ref = exp[GY + 3:GY + PH - 3, GX + 3 * STEP + 3:GX + 3 * STEP + 200]
    assert np.median(np.abs(c[3:-3, 3:200].astype(int) - ref.astype(int))) < 3


def test_the_anchor_follows_the_street_edits(tmp_path):
    w = _make(tmp_path / "p", cells=CELLS[:1], edits=[{"cut": [500, 700]}])
    info = json.loads((w["sd"] / "world" / "world.json").read_text())
    flat_root = tmp_path / "q"
    v = _make(flat_root, cells=CELLS[:1])
    plain = json.loads((v["sd"] / "world" / "world.json").read_text())
    shift = np.float64(plain["anchor"]["A"])[0, 2] - np.float64(info["anchor"]["A"])[0, 2]
    assert shift == pytest.approx(200, abs=0.5)  # the cut's width: everything right of it moved left
    assert np.float64(plain["cells"][0]["T"])[0, 2] - np.float64(info["cells"][0]["T"])[0, 2] == pytest.approx(shift,
                                                                                                             abs=0.5)
    # an anchor inside a cut is refused
    spec = json.loads((w["sd"] / "street.json").read_text())
    spec["edits"] = [{"cut": [GX + 100, GX + 300]}]
    (w["sd"] / "street.json").write_text(json.dumps(spec))
    with pytest.raises(SystemExit):
        ST.build(w["root"], w["sd"], log=lambda m: None)


def test_the_strip_ends_where_a_drifted_last_plate_covers_most_rows(tmp_path):
    root = tmp_path / "p"
    (root / "boards").mkdir(parents=True)
    tex = _texture(4 * STEP + PW, PH + 40, seed=9)
    plates = []
    for n in range(4):  # the last plate drifted 8 px up: it never covers the strip's bottom rows
        dy = 20 - (8 if n == 3 else 0)
        cv2.imwrite(str(root / "boards" / f"P{n + 1}.png"), tex[dy:dy + PH, n * STEP:n * STEP + PW])
        plates.append(f"boards/P{n + 1}.png")
    sd = root / "paint" / "streets" / "main"
    sd.mkdir(parents=True)
    (sd / "street.json").write_text(json.dumps({"version": 1, "plates": plates[:3], "keep": KEEP, "tile": 512,
                                                "edge_cover": 0.95}))
    ST.build(root, sd, log=lambda m: None)
    first = json.loads((sd / "build.json").read_text())["strip"]
    (sd / "street.json").write_text(json.dumps({"version": 1, "plates": plates, "keep": KEEP, "tile": 512,
                                                "edge_cover": 0.95}))
    ST.build(root, sd, log=lambda m: None)  # used to fail: no column of the last plate covers every row
    b = json.loads((sd / "build.json").read_text())["strip"]
    assert b["size"][0] > first["size"][0] + STEP and b["size"][1] == first["size"][1]
    s = ST.Street(sd)
    end = s.window(s.W - 300, s.W)
    assert end[-8:].min() > 0 or end[-8:].mean() > 20  # the rows it does not cover are filled, not black


def test_a_patch_over_a_large_area_keeps_its_own_tone():
    """An edit that replaces a block wider than the tone blur's reach (a whole shop front made plain wall) is pasted
    in its own colour, not a gain clipped from an empty ratio (dark rectangles)."""
    rng = np.random.default_rng(2)
    base = np.clip(_texture(1400, 900, seed=7).astype(np.float32), 0, 255).astype(np.uint8)
    edit = base.copy()
    flat = np.clip(np.float32([150, 140, 130]) + cv2.GaussianBlur(rng.normal(0, 6, (700, 800)).astype(np.float32),
                                                                   (0, 0), 3)[..., None], 0, 255).astype(np.uint8)
    edit[100:800, 300:1100] = flat  # a 800 x 700 px block replaced
    r = ST.patch_layer(base, edit)
    c = r["layer"][350:550, 600:800].reshape(-1, 3).mean(0)
    assert np.all(np.abs(c / flat[250:450, 300:500].reshape(-1, 3).mean(0) - 1) < 0.03), c
    assert r["alpha"][450, 700] > 0.99


def test_a_layer_beyond_the_world_is_left_out(built, tmp_path):
    sd = tmp_path / "s"
    shutil.copytree(built["sd"], sd)
    o = json.loads((sd / "obstacles.json").read_text())
    far = 10 ** 6  # far right of the street and the world (a retired cell's lamp)
    o["layers"] = [{"id": "lost", "parallax": 0.1, "parts": [{"box": [far, -100, far + 50, 100], "method": "rect"}]},
                   {"id": "here", "parallax": 0.1, "parts": [{"box": [400, 100, 450, 300], "method": "rect"}]}]
    o["lights"] = [{"id": "lostlight", "x": far, "y": 0, "layer": "lost"}]
    (sd / "obstacles.json").write_text(json.dumps(o))
    sc = Scene(sd, tmp_path / "cache")
    assert [L["id"] for L in sc.layers] == ["here"]
    assert sc.span(300, 700).lin.shape[1] == 400  # spans (and their layers) still build


def _sky_region(x0, x1, **kw):
    r = {"x": [x0, x1], "up": 60, "src": 30, "taper": [1, 1], "seam": 8, "darken": 0.18, "borrow": [], "paint_out": []}
    r.update(kw)
    return r


def test_the_sky_continues_the_open_sky_upward():
    rng = np.random.default_rng(5)
    H, W, t = 200, 120, 100
    img = np.zeros((H, W, 3), np.float32)
    img[t:] = 60 + rng.normal(0, 8, (H - t, W, 3)).astype(np.float32)  # dark textured sky under the top edge
    top = np.full(W, t)
    top[:10] = -1  # columns the world does not define stay out
    out, added = WD.sky_band(img, top, _sky_region(0, W), 0, 0)
    assert not added[:, :10].any() and not added[t:].any()  # nothing below the edge, nothing where it has no top
    assert added[t - 60:t, 10:].all() and not added[:t - 60].any()  # `up` rows above it
    assert np.allclose(out[t:], img[t:])  # the world itself untouched
    # away from the seam the band repeats the column's own rows (a shift, never mirrored), darker going up
    d = 20
    want = img[t + (30 - d % 30) % 30, 10:] * (1 - 0.18 * d / 60)
    assert np.abs(out[t - d, 10:] - want).mean() < 6
    assert out[t - 59, 10:].mean() < out[t - 1, 10:].mean()
    # tapered ends: the band shortens toward the region's edges
    out2, added2 = WD.sky_band(img, np.full(W, t), _sky_region(0, W, taper=[1, 60]), 0, 0)
    assert added2[:, W - 5].sum() < added2[:, W // 3].sum()


def test_the_sky_step_can_be_redone_and_undone(tmp_path):
    T = 64
    ov = WD.Overlay(tmp_path, T)
    rng = np.random.default_rng(1)
    bgr = (50 + rng.normal(0, 6, (100, 200, 3))).clip(0, 255).astype(np.uint8)
    ov.write(0, 40, bgr, np.ones((100, 200), bool))  # the cells: rows 40 to 140
    r = _sky_region(0, 200, up=30, src=20)
    n1 = WD.apply_sky(ov, r)
    t1 = {k: ov.get(k).copy() for k in ov.keys()}
    assert n1 == 30 * 199  # the region's first column is its taper's zero
    assert all(set(np.unique(t[..., 3])) <= {0, 255, WD.SKY_ALPHA} for t in t1.values())
    WD.clear_sky(ov, [r])
    n2 = WD.apply_sky(ov, r)  # redone over its own pixels: the same
    assert n2 == n1 and all(np.array_equal(ov.get(k), t1[k]) for k in t1)
    WD.clear_sky(ov, [r])  # undone: only the cells' pixels left
    _, al = ov.read(0, 0, 200, 200)
    assert al[40:140].all() and not al[:40].any()


def test_a_sky_in_the_spec_is_added_without_placing_the_cells_again(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["sky"] = [{"x": [rx0 + 40, rx1 - 40], "up": 80, "src": 40, "taper": [20, 20]}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    after = json.loads((sd / "world" / "world.json").read_text())
    assert any("cells kept" in m for m in logs) and any("world: sky over" in m for m in logs)
    assert after["cells"] == before["cells"] and after["sky"][0]["up"] == 80
    assert after["bounds"][1] <= before["bounds"][1] - 70  # the world reaches higher
    wt = WD.WorldTiles(sd)
    xm = (rx0 + rx1) // 2
    _, al = wt.read(xm, xm + 1, ry0 - 120, ry0 + 20)
    assert al[120 - 75:120, 0].all() and not al[:120 - 85, 0].any()
    spec["world"]["sky"] = []  # and taken away again
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(w["root"], sd, log=lambda m: None)
    again = json.loads((sd / "world" / "world.json").read_text())
    assert again["tiles"] == before["tiles"] and again["bounds"] == before["bounds"]


def test_the_sky_leaves_columns_that_already_reach_higher():
    H, W = 300, 60
    img = np.full((H, W, 3), 60, np.float32)
    top = np.full(W, 200)
    top[50:] = 20  # a sky row's cell to the right reaches far higher
    defined = np.arange(H)[:, None] >= top[None, :]
    defined[100:200, 52:55] = False  # a hole at the high cell's edge, below its top
    _, added = WD.sky_band(img, top, _sky_region(0, W, up=80, src=30, taper=[1, 1]), 0, 0, defined)
    assert added[:, 5:50].any(0).all() and not (added & defined).any()
    assert added[120:200, 52:55].all() and not added[:, 56:].any()  # the hole filled, nothing else there


def _kit(root, size=(200, 120), real_h=60):
    """A one-prop kit: a solid block cut-out (RGBA, a transparent margin) and its props.json."""
    kd = root / "boards" / "kit"
    kd.mkdir(parents=True, exist_ok=True)
    w, h = size
    im = np.zeros((h, w, 4), np.uint8)
    im[10:, 10:w - 10, :3] = (40, 160, 220)
    im[10:, 10:w - 10, 3] = 255
    cv2.imwrite(str(kd / "box1.png"), im)
    (kd / "props.json").write_text(json.dumps({"box1": {"real_h": real_h, "base_row": h - 1,
                                                        "top": {"row": 10, "col0": 10, "col1": w - 11}}}))
    return "boards/kit"


def test_props_are_drawn_at_their_real_size_and_undone(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["props_kit"] = _kit(w["root"])
    x, base = rx0 + 300, ry0 + 300
    spec["world"]["props"] = [{"id": "b1", "kit": "box1", "x": x, "base": base}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    after = json.loads((sd / "world" / "world.json").read_text())
    assert any("prop b1" in m for m in logs) and after["cells"] == before["cells"]
    pp = after["placed_props"][0]
    bx0, by0, bx1, by1 = pp["box"]
    assert by1 == base and by1 - by0 == 55  # 110 of its 120 rows are solid, at 60 / 120 scale
    assert abs(bx0 - (x + 5)) <= 1 and abs(pp["top"]["y"] - (base - 55)) <= 1
    assert (sd / pp["mask"]).exists()
    wt = WD.WorldTiles(sd)
    rgb, al = wt.read(bx0 + 5, bx1 - 5, by0 + 5, by1 - 5)
    assert al.all() and rgb[..., 2].mean() > rgb[..., 0].mean()  # the block's blue (graded, but bluer than red)
    ov = WD.Overlay(sd / "world", 512, start={WD._wkey(k): v for k, v in after["tiles"].items()})
    raw = ov.read_alpha(bx0 + 5, by0 + 5, bx1 - 5, by1 - 5)
    assert (raw == WD.PROP_ALPHA).all()
    spec["world"]["props"] = []  # taken away: the tiles are as they were
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(w["root"], sd, log=lambda m: None)
    again = json.loads((sd / "world" / "world.json").read_text())
    assert again["tiles"] == before["tiles"] and not (sd / "world" / "props" / "under.npz").exists()



def test_a_floor_prop_writes_only_above_the_street_and_its_foot_continues_it(tmp_path):
    w = _make(tmp_path / "p")
    sd, root, tex = w["sd"], w["root"], w["tex"]
    before = json.loads((sd / "world" / "world.json").read_text())
    ox, oy = _street_to_tex(sd)
    sx0, sy0, cw, ch = 400, -400, 800, 600  # a floor cell over the street's left part (as above)
    cv2.imwrite(str(root / "boards" / "L1.png"), tex[sy0 + oy:sy0 + oy + ch, sx0 + ox:sx0 + ox + cw])
    A = np.float64(before["anchor"]["A"])
    u, v = np.linalg.solve(A[:, :2], np.float64([sx0, sy0]) - A[:, 2])
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["cells"].append({"id": "L1", "path": "boards/L1.png", "at": [u, v], "floor": 0})
    spec["world"]["props_kit"] = _kit(root, real_h=120)
    x, base = sx0 + 300, 60  # a block standing on the street's rows, 120 px tall: half of it above the street
    spec["world"]["props"] = [{"id": "f1", "kit": "box1", "x": x, "base": base, "floor": 0}]
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(root, sd, log=lambda m: None)
    info = json.loads((sd / "world" / "world.json").read_text())
    pp = info["placed_props"][0]
    bx0, by0, bx1, by1 = pp["box"]
    assert by1 == base and by0 < -40  # placed whole (over the cell and the street)
    ov = WD.Overlay(sd / "world", 512, start={WD._wkey(k): v for k, v in info["tiles"].items()})
    assert (ov.read_alpha(bx0 + 5, by0 + 5, bx1 - 5, -5) == WD.PROP_ALPHA).all()  # above the street: the prop
    assert not ov.read_alpha(bx0 - 20, 0, bx1 + 20, base + 20).any()  # the street's rows: nothing of it
    assert info["xw"] == before["overlay"][0]
    win, comp, foot = WD.prop_foot(sd, root, "f1")
    fx0, fy0 = win[0], win[1]
    ys, xs = np.nonzero(foot)
    assert len(ys) and fy0 + ys.min() == 0 and fy0 + ys.max() <= base + 70  # the foot, its contact line: the street's rows
    above, _ = WD.WorldTiles(sd).read(bx0 + 10, bx1 - 10, -6, 0)
    below = comp[-fy0:-fy0 + 6, bx0 + 10 - fx0:bx1 - 10 - fx0][..., ::-1]
    assert np.abs(above.astype(float).mean((0, 1)) - below.astype(float).mean((0, 1))).max() < 6  # one prop


def test_highlights_pull_a_props_brights_down_and_keep_its_darks():
    rng = np.random.default_rng(1)
    plate = np.full((200, 300, 3), 0.15, np.float32) + rng.normal(0, 0.03, (200, 300, 3)).astype(np.float32)
    bgr = np.zeros((80, 100, 3), np.float32)
    bgr[:, :50] = 0.05  # a dark half (a fan) and a pale half (the casing), a ramp in each
    bgr[:, 50:] = np.linspace(0.6, 0.9, 50)[None, :, None]
    a = np.ones((80, 100), np.float32)
    near = np.ones((200, 300), bool)
    L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[..., 0]
    g1, g9 = WD._prop_grade(bgr, a, plate, near), WD._prop_grade(bgr, a, plate, near, highlights=0.9)
    assert np.allclose(L(g9)[:, :50], L(g1)[:, :50], atol=0.3)  # the darks as they were
    hi1, hi9 = np.percentile(L(g1), 98), np.percentile(L(g9), 98)
    assert 0.86 * hi1 < hi9 < 0.93 * hi1  # the brights about 10% down


def test_warm_shifts_a_props_mid_tones_toward_orange_and_keeps_its_blacks():
    rng = np.random.default_rng(2)
    plate = np.full((200, 300, 3), 0.15, np.float32) + rng.normal(0, 0.03, (200, 300, 3)).astype(np.float32)
    bgr = np.zeros((80, 100, 3), np.float32)
    bgr[:, :50] = 0.02
    bgr[:, 50:] = np.linspace(0.5, 0.8, 50)[None, :, None]
    a = np.ones((80, 100), np.float32)
    near = np.ones((200, 300), bool)
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    g0, g1 = lab(WD._prop_grade(bgr, a, plate, near)), lab(WD._prop_grade(bgr, a, plate, near, warm=1.0))
    assert np.allclose(g1[:, :50], g0[:, :50], atol=0.5)  # the blacks as they were
    db, da = (g1[:, 60:, 2] - g0[:, 60:, 2]).mean(), (g1[:, 60:, 1] - g0[:, 60:, 1]).mean()
    assert 5 < db < 9 and 1.5 < da < 4  # the pale half toward orange

def test_a_rail_in_front_of_a_prop_stays_in_front():
    rng = np.random.default_rng(0)
    plate = np.full((300, 400, 3), 0.2, np.float32) + rng.normal(0, 0.005, (300, 400, 3)).astype(np.float32)
    plate[150:154] = 0.6  # a thin bright rail across the window
    prop = np.zeros((100, 120, 4), np.float32)
    prop[..., :3] = (0.1, 0.3, 0.5)
    prop[..., 3] = 1
    out, a = WD.prop_composite(plate, prop, 140, 100, rng, front_rails=True)
    assert np.allclose(out[150:154, 150:250], plate[150:154, 150:250], atol=0.02)  # the rail over it
    assert not np.allclose(out[110:140, 150:250], plate[110:140, 150:250], atol=0.02)  # the prop elsewhere
    assert (a[48:56] == 0).all() and (a[10:40] == 1).all()
    out2, _ = WD.prop_composite(plate, prop, 140, 100, rng)  # without it the prop covers the rail
    assert not np.allclose(out2[150:154, 150:250], plate[150:154, 150:250], atol=0.02)
    assert out2[201:204, 150:250].mean() < plate[201:204, 150:250].mean() - 0.02  # its contact line on the surface


def test_an_insert_opens_the_world_and_takes_its_patch(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    X, L = rx0 + 300, 260
    wt0 = WD.WorldTiles(sd)
    a0, al0 = wt0.read(X + 40, X + 200, ry0 + 20, ry0 + 300)
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["inserts"] = [{"x": X, "length": L}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    after = json.loads((sd / "world" / "world.json").read_text())
    assert any("insert at x" in m for m in logs) and after["inserts"][0]["length"] == L
    assert after["bounds"][2] == before["bounds"][2] + L
    wt = WD.WorldTiles(sd)
    a1, al1 = wt.read(X + L + 40, X + L + 200, ry0 + 20, ry0 + 300)  # what was right of X is L px further right
    assert al1.all() and np.abs(a1.astype(int) - a0.astype(int)).mean() < 2
    g, gal = wt.read(X, X + L, ry0 + 20, ry0 + 300)
    assert gal.all()  # the gap holds the placeholder (the world left of it, mirrored)
    left, _ = wt.read(X - L, X, ry0 + 20, ry0 + 300)
    assert np.abs(g.astype(int) - left[:, ::-1].astype(int)).mean() < 2
    # the grid: a point right of X is L px further right, and back
    u = after["cells"][-1]["at"][0] + 600
    x, y = WD.grid_to_street(after, u, after["cells"][-1]["at"][1] + 100)
    x0, _ = WD.grid_to_street(before, u, after["cells"][-1]["at"][1] + 100)
    assert abs(x - x0 - L) < 1e-6 and abs(WD.street_to_grid(after, x, y)[0] - u) < 1e-3
    # a patch: the window round the gap with a new middle, aligned and pasted
    win = [X - 150, ry0 + 10, X + L + 150, ry0 + 320]
    base, _ = wt.read(win[0], win[2], win[1], win[3])
    patch = base[..., ::-1].copy()  # (BGR)
    patch[60:250, 150 + 30:150 + L - 30] = (20, 200, 40)
    cv2.imwrite(str(w["root"] / "boards" / "ins.png"), patch)
    spec["world"]["inserts"] = [{"x": X, "length": L, "patches": [{"path": "boards/ins.png", "window": win,
                                                                   "region": [X, win[1], X + L, win[3]]}]}]
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(w["root"], sd, log=lambda m: None)
    wt2 = WD.WorldTiles(sd)
    mid, _ = wt2.read(X + 80, X + L - 80, ry0 + 100, ry0 + 200)
    assert mid[..., 1].mean() > 150 and mid[..., 0].mean() < 60  # the patch's green (RGB) is in the gap


def test_a_clean_insert_takes_neither_tone_nor_pixels_from_its_placeholder(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    X, L = rx0 + 300, 260
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["inserts"] = [{"x": X, "length": L}]
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(w["root"], sd, log=lambda m: None)
    win = [X - 150, ry0 + 10, X + L + 150, ry0 + 320]
    base, _ = WD.WorldTiles(sd).read(win[0], win[2], win[1], win[3])
    ph = base[:, 150:150 + L].astype(int)  # the placeholder
    plate = base[..., ::-1].astype(int)  # (BGR) a plate whose new middle is the placeholder 20 darker: close enough
    plate[:, 150:150 + L] -= 20  # to it that a tone match from it would undo the difference
    cv2.imwrite(str(w["root"] / "boards" / "ins.png"), np.clip(plate, 0, 255).astype(np.uint8))
    want = np.clip(ph - 20, 0, 255)
    got = {}
    for clean in (False, True):  # its region ends in the middle of the gap, against the placeholder
        spec["world"]["inserts"] = [{"x": X, "length": L, **({"clean": True} if clean else {}), "patches": [
            {"path": "boards/ins.png", "window": win, "region": [X - 100, win[1], X + 130, win[3]]}]}]
        (sd / "street.json").write_text(json.dumps(spec))
        logs = []
        ST.build(w["root"], sd, log=logs.append)
        assert any("(clean)" in m for m in logs) == clean
        got[clean], _ = WD.WorldTiles(sd).read(X, X + L, win[1], win[3])
    rows = slice(40, 260)
    for clean, tol in ((True, 3), (False, None)):
        mid = np.abs(got[clean][rows, 40:110].astype(int) - want[rows, 40:110]).mean()
        edge = np.abs(got[clean][rows, 122:129].astype(int) - want[rows, 122:129]).mean()
        if clean:  # the plate as it is, in the middle and up to its region's edge
            assert mid < tol and edge < tol
        else:  # the old way: toned back toward the placeholder, half of it at the region's edge
            assert mid > 8 and edge > 8


def test_a_cut_takes_the_world_out_and_moves_what_is_right_of_it_left(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    X, L = rx0 + 200, 180
    wt0 = WD.WorldTiles(sd)
    keep0, kal0 = wt0.read(X - 150, X - 10, ry0 + 20, ry0 + 300)  # left of the cut
    moved0, mal0 = wt0.read(X + L + 20, X + L + 200, ry0 + 20, ry0 + 300)  # right of it
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["inserts"] = [{"x": X, "length": -L}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    after = json.loads((sd / "world" / "world.json").read_text())
    assert any("cut at x" in m for m in logs) and after["inserts"][0]["length"] == -L
    assert after["overlay"][2] == before["overlay"][2] - L
    wt = WD.WorldTiles(sd)
    keep1, kal1 = wt.read(X - 150, X - 10, ry0 + 20, ry0 + 300)
    assert kal1.all() and np.abs(keep1.astype(int) - keep0.astype(int)).mean() < 1  # left of it: as it was
    moved1, mal1 = wt.read(X + 20, X + 200, ry0 + 20, ry0 + 300)  # what was right of it, L px further left
    assert mal1.all() and np.abs(moved1.astype(int) - moved0.astype(int)).mean() < 2
    # the grid: a point right of the cut is L px further left, and back; left of it, as it was
    u = after["cells"][-1]["at"][0] + 600
    v = after["cells"][-1]["at"][1] + 100
    x, y = WD.grid_to_street(after, u, v)
    x0, _ = WD.grid_to_street(before, u, v)
    assert abs(x - (x0 - L)) < 1e-6 and abs(WD.street_to_grid(after, x, y)[0] - u) < 1e-3
    assert abs(WD.street_before_inserts(after, X - 50) - (X - 50)) < 1e-9
    assert abs(WD.street_before_inserts(after, X + 50) - (X + 50 + L)) < 1e-9


def test_a_cuts_patch_with_align_none_is_pasted_where_it_is_in_its_own_tone(tmp_path):
    """v8 P5: a cut whose patches rebuild the whole stretch: a plate with nothing of the world under its window to
    find is placed as it is (align none, its window's size) and keeps its own tone (tone false); SIFT would match
    stray points (P5b to P5e: 6 to 12 inliers, scale 1.01)."""
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    X, L = rx0 + 200, 180
    win = [X - 160, ry0 + 10, X + 200, ry0 + 330]
    H, Wd = win[3] - win[1], win[2] - win[0]
    yy, xx = np.mgrid[0:H, 0:Wd]
    plate = np.zeros((H, Wd, 3), np.uint8)  # a new stretch: nothing like the texture under it
    plate[...] = (40, 90, 160)
    plate[((xx // 23) + (yy // 31)) % 2 == 0] = (200, 60, 30)
    cv2.imwrite(str(w["root"] / "boards" / "cutp.png"), plate)
    spec = json.loads((sd / "street.json").read_text())
    region = [X - 100, win[1], X + 140, win[3]]
    spec["world"]["inserts"] = [{"x": X, "length": -L, "clean": True, "patches": [
        {"path": "boards/cutp.png", "window": win, "region": region, "align": "none", "tone": False}]}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    assert any("cutp.png: placed as it is, its own tone" in m for m in logs)
    got, gal = WD.WorldTiles(sd).read(X - 60, X + 100, win[1] + 60, win[3] - 60)  # inside the region, past its feather
    want = plate[60:H - 60, 100:260, ::-1]  # (RGB)
    assert gal.all() and np.abs(got.astype(int) - want.astype(int)).mean() < 2  # in place, its own colours
    # the wrong size is refused
    cv2.imwrite(str(w["root"] / "boards" / "cutp.png"), plate[:-10])
    with pytest.raises(SystemExit, match="align none needs it at its window's size"):
        ST.build(w["root"], sd, log=lambda m: None)


def test_a_props_change_after_an_insert_does_not_open_it_twice(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    X, L = rx0 + 300, 260
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["inserts"] = [{"x": X, "length": L}]
    (sd / "street.json").write_text(json.dumps(spec))
    ST.build(w["root"], sd, log=lambda m: None)
    once = json.loads((sd / "world" / "world.json").read_text())
    wt = WD.WorldTiles(sd)
    a1, _ = wt.read(X + L + 40, X + L + 200, ry0 + 20, ry0 + 300)
    spec["world"]["props_kit"] = _kit(w["root"])  # a props-only change: the cells and the insert stay
    spec["world"]["props"] = [{"id": "b1", "kit": "box1", "x": rx0 + 40, "base": ry0 + 200}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    twice = json.loads((sd / "world" / "world.json").read_text())
    assert not any("insert at x" in m for m in logs) and twice["bounds"] == once["bounds"]
    a2, _ = WD.WorldTiles(sd).read(X + L + 40, X + L + 200, ry0 + 20, ry0 + 300)
    assert np.abs(a2.astype(int) - a1.astype(int)).mean() < 1  # nothing moved again


def test_a_grade_lifts_its_core_to_its_luma_once_and_leaves_the_rest(tmp_path):
    w = _make(tmp_path / "p")
    sd = w["sd"]
    before = json.loads((sd / "world" / "world.json").read_text())
    rx0, ry0, rx1, ry1 = next(c for c in before["cells"] if c["id"] == "R2")["box"]
    x0, x1, y0, y1 = rx0 + 150, rx0 + 450, ry0 + 100, ry0 + 260
    luma = lambda rgb: (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean()
    core0, _ = WD.WorldTiles(sd).read(x0, x1, y0, y1)
    far0, _ = WD.WorldTiles(sd).read(rx1 - 120, rx1 - 20, ry0 + 20, ry0 + 120)
    target = float(luma(core0.astype(np.float32))) + 25
    spec = json.loads((sd / "street.json").read_text())
    spec["world"]["grade"] = [{"x": [x0, x1], "y": [y0, y1], "falloff": 120, "luma": target}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    assert any("grade over" in m for m in logs)
    core1, _ = WD.WorldTiles(sd).read(x0, x1, y0, y1)
    far1, _ = WD.WorldTiles(sd).read(rx1 - 120, rx1 - 20, ry0 + 20, ry0 + 120)
    assert abs(float(luma(core1.astype(np.float32))) - target) < 2.0  # the core at its luma
    assert np.array_equal(far0, far1)  # past the falloff, untouched
    assert core1[..., 0].astype(int).mean() - core0[..., 0].astype(int).mean() > \
        core1[..., 2].astype(int).mean() - core0[..., 2].astype(int).mean()  # warm: more red than blue added
    spec["world"]["props_kit"] = _kit(w["root"])  # a props-only change: the grade is not applied again
    spec["world"]["props"] = [{"id": "b1", "kit": "box1", "x": rx1 - 300, "base": ry1 - 40}]
    (sd / "street.json").write_text(json.dumps(spec))
    logs = []
    ST.build(w["root"], sd, log=logs.append)
    assert not any("grade over" in m for m in logs)
    core2, _ = WD.WorldTiles(sd).read(x0, x1, y0, y1)
    assert np.array_equal(core1, core2)
