"""A door that opens for him and shuts behind him (Full Stop's outro, P7 "The exit": the stair house's door on the
roof; v8b). The world's cell has an open twin (the same photograph with the door open and the light inside on:
boards/38-p7/P7d-open.png, registered with P7d); the leaf itself is cut from the closed world and drawn hinged on its
jamb at the door's angle, photo by photo.

Spec (obstacles.json `doors`): [{"id": "doorP7", "cell": "P7d" (a world cell; or "cell": null and "patch":
{"path": the plate, "window": [x0, y0, x1, y1]}: a patch placed by a pure translation at scale 1, street = plate px +
(x0, y0)), "open": "boards/38-p7/P7d-open-noleaf.png" (the open twin, its own leaf painted out: the leaf is drawn),
"open_region": [x0, y0, x1, y1] (street px: the open twin is used only inside it, feathered over FEATHER px; its rain
elsewhere was re-rendered),
"leaf": [x0, y0, x1, y1] (street px: the closed leaf as the world shows it, which is also the doorway's opening),
"hinge": "left" (or "right"), "light": "spillP7" (a scene light the door scales by how far it is open; its own
strength is the door's fully open one), "persp": 0.12 (how much shorter the free edge looks at 90 degrees), "dir":
"doors"}].

`build` (python -m aimv.paint.doors <project> [street]) writes, per door, in street px over its cell's placed box
(world/world.json: its transform T, cell px to street px):
- doors/<id>.png: the open twin warped onto the street and gain-matched to the world there (sRGB);
- doors/<id>-closed.png: the world as built over the same box (the leaf is cut from it);
- doors/<id>-spill.png: where the open twin is brighter than the world outside the leaf (the light thrown out of
  the doorway onto the roof and the jambs), feathered;
and doors/doors.json: each door's box.

The `door` prop (the choreography) sets its times: `open: {from, to, angle}` (degrees, 80: pushed open, quick and
easing), held open by the closer until `shut: {from, to}`: the closer's sweep (most of the way, easing into the latch
zone), the latch zone (slower), and the snap shut on `to`, then a small rebound (`latch` degrees, 1.5) over the next
0.12 s; `inside`: when he crosses the threshold (from then on his figure in the air is seen only through the
doorway, less the leaf); `gone`: from then on nothing of him shows in the door's box (the last drawing would be held
on the photos after it).

The renderer (Doors.blend, next to the windows): inside the leaf's box the open twin where the leaf is not, and the
leaf drawn at width W cos(angle) from its hinge, its free edge shorter by persp sin(angle) (it recedes), darker as it
turns from the light (0.55 + 0.45 cos); the spill blended in by how far it is open (sin(angle) over sin(the open
angle)); the door's light scaled the same (Doors.lights); Doors.occlude hides his skins outside the opening once he
is inside."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .scene import luminance, srgb2lin

LATCH_T = 0.12  # s: the latch's rebound
FEATHER = 24  # px: the open region's soft edge
SWEEP, LATCH_ZONE = (0.75, 12.0), (0.96, 3.0)  # the closer: (share of its time, degrees left) at the sweep's end and
#                                                 the latch zone's end; the snap shuts the rest


def angle(cfg: dict, t: float) -> float:
    """The door's angle (degrees, 0 shut) at song time t."""
    o0, o1, A = cfg["o0"], cfg["o1"], cfg["angle"]
    s0, s1 = cfg["s0"], cfg["s1"]
    if t <= o0:
        return 0.0
    if t < o1:  # pushed open: quick, easing out
        u = (t - o0) / max(o1 - o0, 1e-6)
        return A * (1 - (1 - u) ** 2)
    if t < s0:
        return A
    if t < s1:  # the closer
        u = (t - s0) / max(s1 - s0, 1e-6)
        (u1, a1), (u2, a2) = SWEEP, LATCH_ZONE
        if u < u1:
            e = u / u1
            return A + (a1 - A) * e * e * (3 - 2 * e)
        if u < u2:
            return a1 + (a2 - a1) * (u - u1) / (u2 - u1)
        return a2 * (1 - (u - u2) / (1 - u2))
    dt = t - s1
    if dt < LATCH_T:  # the latch: a small rebound off the frame
        return float(cfg.get("latch", 1.5)) * math.sin(math.pi * dt / LATCH_T)
    return 0.0


def openness(cfg: dict, t: float) -> float:
    """How far open, 0 to 1: sin(angle) over sin(the open angle) (the light through the doorway)."""
    a = angle(cfg, t)
    return float(np.clip(math.sin(math.radians(a)) / max(math.sin(math.radians(cfg["angle"])), 1e-6), 0, 1))


def leaf_quad(leaf: list, a: float, hinge: str = "left", persp: float = 0.12) -> np.ndarray:
    """The leaf's four corners (street px: hinge top, free top, free bottom, hinge bottom) at angle a (degrees)."""
    x0, y0, x1, y1 = (float(v) for v in leaf)
    W, H = x1 - x0, y1 - y0
    r = math.radians(a)
    sh = 0.5 * persp * math.sin(r) * H
    if hinge == "right":
        xf = x1 - W * math.cos(r)
        return np.float32([[x1, y0], [xf, y0 + sh], [xf, y1 - sh], [x1, y1]])
    xf = x0 + W * math.cos(r)
    return np.float32([[x0, y0], [xf, y0 + sh], [xf, y1 - sh], [x0, y1]])


def build(proj_root: Path, d: Path, log=print) -> dict:
    from .world import WorldTiles

    ob = json.loads((d / "obstacles.json").read_text())
    doors = ob.get("doors") or []
    if not doors:
        raise SystemExit("doors: obstacles.json has no `doors`")
    wj = d / "world" / "world.json"
    cells = {c["id"]: c for c in json.loads(wj.read_text())["cells"]} if wj.exists() else {}
    wt = WorldTiles(d)
    info = {"version": 1, "doors": {}}
    for dr in doors:
        out = d / str(dr.get("dir", "doors"))
        out.mkdir(parents=True, exist_ok=True)
        if dr.get("cell"):
            c = cells.get(dr["cell"])
            if c is None:
                raise SystemExit(f"doors: no world cell {dr['cell']!r}")
            plate, T = c["path"], np.float64(c["T"])
        elif dr.get("patch"):  # a patch placed by a pure translation at scale 1 (its window's top left)
            plate = dr["patch"]["path"]
            wx0, wy0 = (float(v) for v in dr["patch"]["window"][:2])
            T = np.float64([[1, 0, wx0], [0, 1, wy0]])
        else:
            raise SystemExit(f"doors: {dr['id']}: give a cell or a patch")
        shut = cv2.cvtColor(cv2.imread(str(proj_root / plate)), cv2.COLOR_BGR2RGB)
        opn = cv2.cvtColor(cv2.imread(str(proj_root / dr["open"])), cv2.COLOR_BGR2RGB)
        if opn.shape != shut.shape:
            opn = cv2.resize(opn, (shut.shape[1], shut.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = shut.shape[:2]
        if dr.get("open_region"):  # (only round the door: the box is the open region and its feather)
            ox0, oy0, ox1, oy1 = (float(v) for v in dr["open_region"])
            x0, y0 = int(math.floor(ox0)) - 2 * FEATHER, int(math.floor(oy0)) - 2 * FEATHER
            x1, y1 = int(math.ceil(ox1)) + 2 * FEATHER, int(math.ceil(oy1)) + 2 * FEATHER
        else:
            corners = np.float64([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]) @ T.T
            x0, y0 = int(math.floor(corners[:, 0].min())), int(math.floor(corners[:, 1].min()))
            x1, y1 = int(math.ceil(corners[:, 0].max())), int(math.ceil(corners[:, 1].max()))
        M = T.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        opn_s = cv2.warpAffine(opn, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        shut_s = cv2.warpAffine(shut, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
        world, al = wt.read(x0, x1, y0, y1)  # the world as built here: the closed door, the stitch's gain
        ok = (al & valid).astype(np.float32)
        wl, sl = srgb2lin(world), srgb2lin(shut_s)
        g = np.clip((cv2.GaussianBlur(wl * ok[..., None], (0, 0), 30) + 1e-3) /
                    (cv2.GaussianBlur(sl * ok[..., None], (0, 0), 30) + 1e-3), 0.6, 1.6)
        opn_l = np.clip(srgb2lin(opn_s) * g, 0, 1)
        lx0, ly0, lx1, ly1 = (int(round(v)) for v in dr["leaf"])
        leafm = np.zeros(size[::-1], np.float32)
        leafm[max(ly0 - y0, 0):max(ly1 - y0, 0), max(lx0 - x0, 0):max(lx1 - x0, 0)] = 1
        dlt = cv2.GaussianBlur(luminance(opn_l) - luminance(wl), (0, 0), 3)
        core = np.clip(dlt / max(float(dr.get("spill_core", 0.05)), 1e-4), 0, 1) * ok * (1 - leafm)
        # only the light thrown out of the doorway: the bright regions that touch it (not the open twin's re-drawn
        # rain and grain on the wall round it)
        n_, lab_ = cv2.connectedComponents((cv2.morphologyEx((core > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                                             np.ones((5, 5), np.uint8))), 8)
        near_ = cv2.dilate(leafm.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        keep_ = np.zeros(n_, bool)
        keep_[np.unique(lab_[near_ & (lab_ > 0)])] = True
        keep_[0] = False
        spill = cv2.GaussianBlur(core * keep_[lab_], (0, 0), 4)
        if dr.get("open_region"):  # the open twin only inside its region, feathered
            reg = np.zeros(size[::-1], np.float32)
            ox0, oy0, ox1, oy1 = (int(round(v)) for v in dr["open_region"])
            reg[max(oy0 - y0, 0):max(oy1 - y0, 0), max(ox0 - x0, 0):max(ox1 - x0, 0)] = 1
            reg = cv2.GaussianBlur(reg, (0, 0), FEATHER / 2)
            spill = spill * reg

        def srgb8(a):
            return np.clip(255 * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055) + 0.5, 0,
                           255).astype(np.uint8)
        cv2.imwrite(str(out / f"{dr['id']}.png"), cv2.cvtColor(srgb8(opn_l), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{dr['id']}-closed.png"), cv2.cvtColor(world, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{dr['id']}-spill.png"), np.clip(spill * 255 + 0.5, 0, 255).astype(np.uint8))
        info["doors"][dr["id"]] = {"box": [x0, y0, x1, y1]}
        (out / "doors.json").write_text(json.dumps(info, indent=1) + "\n")
        log(f"doors: {dr['id']} over street x {x0} to {x1}, y {y0} to {y1}")
    return info


class Doors:
    """The built doors, loaded on demand, and the choreography's times (ch.doors: id -> {o0, o1, angle, s0, s1,
    latch, inside})."""

    def __init__(self, d: Path, ch):
        self.ch = ch
        self.cfg = dict(getattr(ch, "doors", None) or {})
        self.spec = {dr["id"]: dr for dr in (ch.scene.spec.get("doors") or [])}
        self.info: dict = {}
        self._cache: dict = {}
        for did in self.cfg:
            dr = self.spec.get(did) or {}
            f = Path(d) / str(dr.get("dir", "doors")) / "doors.json"
            if not f.exists():
                raise SystemExit(f"doors: {f} missing: run `python -m aimv.paint.doors <project>`")
            self.info[did] = json.loads(f.read_text())["doors"][did]
        self.d = Path(d)

    @property
    def active(self) -> bool:
        return bool(self.cfg)

    def _load(self, did: str):
        if did not in self._cache:
            dr = self.spec[did]
            base = self.d / str(dr.get("dir", "doors")) / did
            o = cv2.imread(f"{base}.png")
            c = cv2.imread(f"{base}-closed.png")
            s = cv2.imread(f"{base}-spill.png", cv2.IMREAD_GRAYSCALE)
            if o is None or c is None or s is None:
                raise SystemExit(f"doors: {base}.png missing: run `python -m aimv.paint.doors`")
            self._cache[did] = (srgb2lin(cv2.cvtColor(o, cv2.COLOR_BGR2RGB)), srgb2lin(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)),
                                s.astype(np.float32) / 255)
        return self._cache[did]

    def lights(self, t: float) -> dict:
        """{light id: factor}: each door's light scaled by how far it is open."""
        return {self.spec[did]["light"]: openness(cfg, t) for did, cfg in self.cfg.items()
                if self.spec.get(did, {}).get("light")}

    def _masks(self, did: str, a: float, bx0: int, by0: int, size: tuple) -> tuple:
        """(the opening's mask, the leaf's mask, the leaf drawn) over the door's box at angle a."""
        dr, cfg = self.spec[did], self.cfg[did]
        w, h = size
        lx0, ly0, lx1, ly1 = (float(v) for v in dr["leaf"])
        om = np.zeros((h, w), np.float32)
        om[int(round(ly0 - by0)):int(round(ly1 - by0)), int(round(lx0 - bx0)):int(round(lx1 - bx0))] = 1
        quad = leaf_quad(dr["leaf"], a, dr.get("hinge", "left"), float(dr.get("persp", 0.12)))
        src = np.float32([[lx0, ly0], [lx1, ly0], [lx1, ly1], [lx0, ly1]]) - np.float32([bx0, by0])
        if dr.get("hinge", "left") == "right":
            src = np.float32([[lx1, ly0], [lx0, ly0], [lx0, ly1], [lx1, ly1]]) - np.float32([bx0, by0])
        dst = quad - np.float32([bx0, by0])
        P = cv2.getPerspectiveTransform(src, dst)
        _, closed, _ = self._load(did)
        leaf_img = cv2.warpPerspective(closed * om[..., None], P, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        qm = cv2.warpPerspective(om, P, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        shade = 0.55 + 0.45 * math.cos(math.radians(a))
        return om, qm, leaf_img * shade

    def blend(self, lin: np.ndarray, E: np.ndarray, x0: int, y0: int, t: float):
        """lin (linear) and E of street box (x0, y0) with every door as it is at song time t."""
        if not self.active:
            return lin, E
        h, w = lin.shape[:2]
        for did, cfg in self.cfg.items():
            bx0, by0, bx1, by1 = self.info[did]["box"]
            c0, c1, r0, r1 = max(x0, bx0), min(x0 + w, bx1), max(y0, by0), min(y0 + h, by1)
            if c1 <= c0 or r1 <= r0:
                continue
            a = angle(cfg, t)
            if a <= 1e-3:
                continue  # shut: the world as built
            opn, _, spill = self._load(did)
            om, qm, leaf = self._masks(did, a, bx0, by0, (bx1 - bx0, by1 - by0))
            s = openness(cfg, t)
            src = (slice(r0 - by0, r1 - by0), slice(c0 - bx0, c1 - bx0))
            sl = (slice(r0 - y0, r1 - y0), slice(c0 - x0, c1 - x0))
            reg = lin[sl]
            sp = (spill[src] * s)[..., None]
            reg = reg * (1 - sp) + opn[src] * sp  # the light thrown out of the doorway
            o_, q_ = om[src][..., None], qm[src][..., None]
            door = opn[src] * (1 - q_) + leaf[src]  # (the leaf drawn premultiplied: its own pixels only)
            lin[sl] = reg * (1 - o_) + door * o_
            E[sl] = E[sl] * (1 + 0.5 * spill[src] * s)
        return lin, E

    def occlude(self, A: np.ndarray, x0: int, y0: int, t: float) -> np.ndarray:
        """His figure in the air (A, the air layer's coverage over street box (x0, y0)) once he is inside: seen only
        through the doorway's opening, less the leaf."""
        if not self.active:
            return A
        h, w = A.shape
        for did, cfg in self.cfg.items():
            if cfg.get("inside") is None or t < cfg["inside"] - 1e-6:
                continue
            bx0, by0, bx1, by1 = self.info[did]["box"]
            c0, c1, r0, r1 = max(x0, bx0), min(x0 + w, bx1), max(y0, by0), min(y0 + h, by1)
            if c1 <= c0 or r1 <= r0:
                continue
            sl = (slice(r0 - y0, r1 - y0), slice(c0 - x0, c1 - x0))
            A = A.copy()
            if cfg.get("gone") is not None and t >= cfg["gone"] - 1e-6:  # gone inside: nothing of him here any more
                A[sl] = 0  # (not even the last drawing held, a sliver at the jamb)
                continue
            om, qm, _ = self._masks(did, angle(cfg, t), bx0, by0, (bx1 - bx0, by1 - by0))
            src = (slice(r0 - by0, r1 - by0), slice(c0 - bx0, c1 - bx0))
            A[sl] = A[sl] * np.clip(om[src] - qm[src], 0, 1)
        return A


if __name__ == "__main__":
    from ..project import Project

    p = Project.open(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "main"
    build(Path(p.root), Path(p.root) / "paint" / "streets" / name)


# ---------------------------------------------------------------------------------------------- the template
from .choreo import ChoreoError, Ctx  # noqa: E402
from .moves import template  # noqa: E402


@template("door", prop=True)
def door(c: Ctx) -> None:
    """A door opening for him and shutting behind him (prop; obstacles.json `doors`, built by `python -m
    aimv.paint.doors`): id, open: {from, to, angle (80)}, shut: {from, to}, latch (degrees of the rebound, 1.5),
    inside (when he crosses the threshold; from then on he is seen only through the doorway), gone (from then on
    nothing of him shows in the door's box: the last drawing is not held at the jamb)."""
    spec = {dr["id"]: dr for dr in (c.ch.scene.spec.get("doors") or [])}
    did = str(c.require("id"))
    if did not in spec:
        raise ChoreoError(f"door: no door {did!r} in obstacles.json doors ({', '.join(spec) or 'none'})")
    op, sh = c.arg("open"), c.arg("shut")
    if not isinstance(op, dict) or not isinstance(sh, dict) or not all(k in op for k in ("from", "to")) or \
            not all(k in sh for k in ("from", "to")):
        raise ChoreoError("door: open: {from, to, angle} and shut: {from, to}")
    g = c.ch.grid
    tt = {k: g.t(c.song(c.at(v, key=f"door.{k}"))) for k, v in
          (("o0", op["from"]), ("o1", op["to"]), ("s0", sh["from"]), ("s1", sh["to"]))}
    if not tt["o0"] < tt["o1"] <= tt["s0"] < tt["s1"]:
        raise ChoreoError("door: open.from < open.to <= shut.from < shut.to")
    a = float(op.get("angle", 80.0))
    if not 5 <= a <= 120:
        raise ChoreoError(f"door: open.angle {a} (degrees, 5 to 120)")
    ins, gone = c.arg("inside"), c.arg("gone")
    doors = dict(getattr(c.ch, "doors", None) or {})
    doors[did] = dict(tt, angle=a, latch=float(c.arg("latch", 1.5)),
                      inside=g.t(c.song(c.at(ins, key="door.inside"))) if ins is not None else None,
                      gone=g.t(c.song(c.at(gone, key="door.gone"))) if gone is not None else None)
    c.ch.doors = doors
