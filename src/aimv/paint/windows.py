"""Windows lighting up along his path (Full Stop's final chorus): the world's cells come with a lit twin (the same
photograph with every window lit warm from inside: boards/19-lit/<cell>-lit.png), and each window switches on at its
own time, a beat at a time as he passes (the choreography's `windows` prop). The reverse of verse 2's screens
(screens.py), on the world's cells (world.py) instead of the 1D street.

Spec (obstacles.json `lit`): {"cells": {"G2u": "boards/19-lit/G2u-lit.png", ...}, "dir": "lit", "core": 0.06,
"soft": 0.02, "spill": 24}.

`build` (python -m aimv.paint.windows <project> [street]) writes, per cell, in street px over the cell's placed box
(world/world.json: its transform T, cell px to street px):
- lit/<cell>.png: the lit twin warped onto the street and gain-matched to the world there (sRGB);
- lit/<cell>-mask.png: where the lit twin is brighter than the cell (the windows and the warm spill round them,
  feathered; everything else, where the edit model only re-drew the rain and the grain, stays the world's);
- lit/<cell>-labels.png (uint16): which window each masked pixel belongs to (the window cores are the strongly lit
  blobs; the spill round them goes to the nearest core);
and lit/lit.json: each cell's box and its windows (label, centre, box).

The renderer (Windows.blend, from Renderer.photo next to the screens) blends a window's lit pixels in from its time,
over `fade` 16ths. Times come from the choreography's `windows` prop (ch.lit): with `follow`, each window lights on
the first beat (`every`) after his torso passes its centre minus `lead` px, one beat later per `spread` px of height
between the window and his feet (the light spreading up and down the facade from him); windows he never passes
stay dark."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .scene import luminance, srgb2lin


def _cell_info(d: Path) -> dict:
    w = json.loads((d / "world" / "world.json").read_text())
    return {c["id"]: c for c in w["cells"]}


def _mask_and_labels(dark: np.ndarray, lit: np.ndarray, core: float, soft: float, spill: int):
    """In cell px: the window mask (0..1, feathered) and the label of each masked pixel's window."""
    a = luminance(srgb2lin(dark))
    b = luminance(srgb2lin(lit))
    dlt = cv2.GaussianBlur(b - a, (0, 0), 1.5)
    cm = (dlt > core).astype(np.uint8)
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(cm, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 60  # windows, not flecks of re-drawn rain
    core_m = keep[lab]
    # the spill round each window: brighter than the cell by `soft`, within `spill` px of a core
    near = cv2.dilate(core_m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                         (2 * spill + 1, 2 * spill + 1))) > 0
    soft_m = np.clip((dlt - soft) / max(core - soft, 1e-4), 0, 1) * near
    mask = np.maximum(core_m.astype(np.float32), soft_m.astype(np.float32))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
    # each pixel's nearest core (the labels of the kept cores only)
    inv = (~core_m).astype(np.uint8)
    _, near_lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core_m)
    pix_lab = np.zeros(int(near_lab.max()) + 1, np.int32)
    pix_lab[near_lab[ys, xs]] = lab[ys, xs]
    labels = pix_lab[near_lab].astype(np.int32)
    labels[mask < 0.02] = 0
    return mask, labels


def build(proj_root: Path, d: Path, log=print) -> dict:
    from .world import WorldTiles

    ob = json.loads((d / "obstacles.json").read_text())
    spec = ob.get("lit")
    if not spec:
        raise SystemExit("windows: obstacles.json has no `lit`")
    out = d / str(spec.get("dir", "lit"))
    out.mkdir(parents=True, exist_ok=True)
    cells = _cell_info(d)
    wt = WorldTiles(d)
    info = {"version": 1, "cells": {}}
    for cid, lit_path in spec["cells"].items():
        c = cells.get(cid)
        if c is None:
            raise SystemExit(f"windows: no world cell {cid!r}")
        dark = cv2.cvtColor(cv2.imread(str(proj_root / c["path"])), cv2.COLOR_BGR2RGB)
        lit = cv2.cvtColor(cv2.imread(str(proj_root / lit_path)), cv2.COLOR_BGR2RGB)
        if lit.shape != dark.shape:
            lit = cv2.resize(lit, (dark.shape[1], dark.shape[0]), interpolation=cv2.INTER_AREA)
        mask, labels = _mask_and_labels(dark, lit, float(spec.get("core", 0.06)), float(spec.get("soft", 0.02)),
                                         int(spec.get("spill", 24)))
        T = np.float64(c["T"])
        h, w = dark.shape[:2]
        corners = np.float64([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]) @ T.T
        x0, y0 = int(math.floor(corners[:, 0].min())), int(math.floor(corners[:, 1].min()))
        x1, y1 = int(math.ceil(corners[:, 0].max())), int(math.ceil(corners[:, 1].max()))
        M = T.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        lit_s = cv2.warpAffine(lit, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dark_s = cv2.warpAffine(dark, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask_s = cv2.warpAffine(mask, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        lab_s = cv2.warpAffine(labels.astype(np.float32), M, size, flags=cv2.INTER_NEAREST, borderValue=0)
        valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
        # the world here (the cell as placed, its seams with its neighbours): the gain of the stitch on this cell
        world, al = wt.read(x0, x1, y0, y1)
        ok = (al & valid).astype(np.float32)
        wl, dl = srgb2lin(world), srgb2lin(dark_s)
        num = cv2.GaussianBlur(wl * ok[..., None], (0, 0), 30)
        den = cv2.GaussianBlur(dl * ok[..., None], (0, 0), 30)
        g = np.clip((num + 1e-3) / (den + 1e-3), 0.6, 1.6)
        lit_l = np.clip(srgb2lin(lit_s) * g, 0, 1)
        mask_s = mask_s * ok  # only where the world shows this cell (its own pixels, not a neighbour's)
        lit8 = np.clip(255 * np.where(lit_l <= 0.0031308, 12.92 * lit_l, 1.055 * lit_l ** (1 / 2.4) - 0.055) + 0.5,
                       0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{cid}.png"), cv2.cvtColor(lit8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{cid}-mask.png"), np.clip(mask_s * 255 + 0.5, 0, 255).astype(np.uint8))
        lab16 = lab_s.astype(np.uint16)
        cv2.imwrite(str(out / f"{cid}-labels.png"), lab16)
        wins = []
        for k in np.unique(lab16):
            if k == 0:
                continue
            ys, xs = np.nonzero((lab16 == k) & (mask_s > 0.5))
            if len(xs) < 40:
                continue
            wins.append({"label": int(k), "cx": round(float(xs.mean()) + x0, 1), "cy": round(float(ys.mean()) + y0, 1),
                         "box": [int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1]})
        info["cells"][cid] = {"box": [x0, y0, x1, y1], "windows": wins}
        log(f"windows: {cid}: {len(wins)} windows over street x {x0} to {x1}, y {y0} to {y1}")
    (out / "lit.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


class Windows:
    """The built lit cells, loaded on demand; times(ch) resolves each window's time from the choreography's
    `windows` prop (ch.lit) and his path; blend switches the lit windows in."""

    def __init__(self, d: Path, ch):
        self.ch = ch
        self.cfg = getattr(ch, "lit", None)
        spec = ch.scene.spec.get("lit") or {}
        self.dir = Path(d) / str(spec.get("dir", "lit"))
        f = self.dir / "lit.json"
        self.info = json.loads(f.read_text()) if (self.cfg and f.exists()) else None
        if self.cfg and self.info is None:
            raise SystemExit(f"windows: {f} missing: run `python -m aimv.paint.windows <project>`")
        self._cache: dict = {}
        self._times: dict | None = None

    @property
    def active(self) -> bool:
        return bool(self.cfg and self.info)

    def times(self) -> dict:
        """cell -> array (indexed by label) of the 16th each window lights (inf: never)."""
        if self._times is not None:
            return self._times
        ch, cfg = self.ch, self.cfg
        lib = ch.lib
        track = []  # (q, torso x, feet y) of his poses, in time order
        for e in sorted(ch.events, key=lambda e: e.q):
            if e.layer not in ("wall", "air") or e.pose is None or lib.m(e.pose)["kind"] != "pose":
                continue
            mv = ch.moves[e.move] if 0 <= e.move < len(ch.moves) else None
            if mv is None or mv.prop:
                continue
            m = lib.m(e.pose)
            s = m["scale"] * e.scale
            ax, ay = lib.anchor(e.pose, e.anchor)
            track.append((e.q, e.x + (m["cx"] - ax) * s, e.y + (m["bbox"][3] - ay) * s))
        q0, q1 = float(cfg["q0"]), float(cfg["q1"])
        track = [t for t in track if q0 - 1e-6 <= t[0] <= q1 + 1e-6]
        step = float(cfg.get("every", 4.0))
        lead, spread = float(cfg.get("lead", 0.0)), float(cfg.get("spread", 700.0))
        origin = float(cfg.get("origin", q0))
        out = {}
        tq = np.array([t[0] for t in track]) if track else np.zeros(0)
        tx = np.maximum.accumulate(np.array([t[1] for t in track])) if track else np.zeros(0)
        ty = np.array([t[2] for t in track]) if track else np.zeros(0)
        for cid, c in self.info["cells"].items():
            if cfg.get("cells") and cid not in cfg["cells"]:
                continue
            n = max([wn["label"] for wn in c["windows"]], default=0) + 1
            arr = np.full(n, np.inf)
            for wn in c["windows"]:
                k = int(np.searchsorted(tx, wn["cx"] - lead)) if len(tx) else 0
                if k >= len(tx):
                    continue  # he never passes it
                q = tq[k] + math.floor(abs(wn["cy"] - ty[k]) / max(spread, 1.0)) * step
                arr[wn["label"]] = origin + math.ceil((q - origin) / step - 1e-6) * step
            out[cid] = arr
        self._times = out
        return out

    def _load(self, cid: str):
        if cid not in self._cache:
            img = cv2.imread(str(self.dir / f"{cid}.png"))
            m = cv2.imread(str(self.dir / f"{cid}-mask.png"), cv2.IMREAD_GRAYSCALE)
            lab = cv2.imread(str(self.dir / f"{cid}-labels.png"), cv2.IMREAD_UNCHANGED)
            if img is None or m is None or lab is None:
                raise SystemExit(f"windows: {self.dir / cid}.png missing: run `python -m aimv.paint.windows`")
            self._cache[cid] = (srgb2lin(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), m.astype(np.float32) / 255,
                                lab.astype(np.int32))
        return self._cache[cid]

    def blend(self, lin: np.ndarray, E: np.ndarray, x0: int, y0: int, q: float):
        """lin (h x w x 3, linear) and E (h x w, the paint's light) of street box (x0, y0) with every window lit
        whose time has come (faded in over `fade` 16ths); E brightens a little round the lit windows."""
        if not self.active:
            return lin, E
        h, w = lin.shape[:2]
        fade = float(self.cfg.get("fade", 1.0))
        times = self.times()
        for cid, arr in times.items():
            if not np.any(arr <= q + 1e-6):
                continue
            bx0, by0, bx1, by1 = self.info["cells"][cid]["box"]
            c0, c1, r0, r1 = max(x0, bx0), min(x0 + w, bx1), max(y0, by0), min(y0 + h, by1)
            if c1 <= c0 or r1 <= r0:
                continue
            img, m, lab = self._load(cid)
            src = (slice(r0 - by0, r1 - by0), slice(c0 - bx0, c1 - bx0))
            a = np.clip((q - arr) / max(fade, 1e-6) + 1e-6, 0, 1) if fade > 0 else (arr <= q + 1e-6).astype(float)
            a[0] = 0.0
            wgt = m[src] * a[np.clip(lab[src], 0, len(a) - 1)].astype(np.float32)
            if not wgt.any():
                continue
            sl = (slice(r0 - y0, r1 - y0), slice(c0 - x0, c1 - x0))
            lin[sl] = lin[sl] * (1 - wgt[..., None]) + img[src] * wgt[..., None]
            glow = cv2.GaussianBlur(wgt, (0, 0), 25)
            E[sl] = E[sl] * (1 + 0.6 * glow)
        return lin, E


def flicker(ch, lin: np.ndarray, x0: int, y0: int, q: float, k: int) -> np.ndarray:
    """The city's lights flickering faintly on every photo of a `flicker` stretch (Full Stop's last hold, 177.7 to
    180: time stops for him, not for the city): the plate's warm bright pixels (lit windows, street lamps; linear
    luminance over `thr`, red over blue) each brighten or dim by up to `amount`, a new draw every photo (k), per
    `cell` px of street (anchored to the street, so the camera's nudge does not move it; cubed, so most lights
    barely move and a few dip). lin: the plate box at street (x0, y0), linear; returned changed in place."""
    sp = next((f for f in getattr(ch, "flickers", None) or [] if f["q0"] - 1e-6 <= q < f["q1"] - 1e-6), None)
    if sp is None:
        return lin
    h, w = lin.shape[:2]
    thr, amt, cell = float(sp["thr"]), float(sp["amount"]), max(2, int(sp["cell"]))
    lum = luminance(lin)
    m = np.clip((lum - thr) / thr, 0, 1) * (lin[..., 0] > 1.15 * lin[..., 2])
    if not m.any():
        return lin
    gx0, gy0 = x0 // cell - 1, y0 // cell - 1
    gx1, gy1 = (x0 + w) // cell + 2, (y0 + h) // cell + 2
    ix, iy = np.meshgrid(np.arange(gx0, gx1, dtype=np.float64), np.arange(gy0, gy1, dtype=np.float64))
    v = np.sin(ix * 12.9898 + iy * 78.233 + (k + 1000 * int(ch.seed)) * 37.719) * 43758.5453
    v = (v - np.floor(v)) * 2 - 1
    v = (v ** 3).astype(np.float32)
    big = cv2.resize(v, ((gx1 - gx0) * cell, (gy1 - gy0) * cell), interpolation=cv2.INTER_LINEAR)
    ox, oy = x0 - gx0 * cell, y0 - gy0 * cell
    g = 1 + amt * big[oy:oy + h, ox:ox + w] * m.astype(np.float32)
    lin *= g[..., None]
    return lin


if __name__ == "__main__":
    from ..project import Project

    p = Project.open(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "main"
    build(Path(p.root), Path(p.root) / "paint" / "streets" / name)


# ---------------------------------------------------------------------------------------------- the template
from .choreo import ChoreoError, Ctx  # noqa: E402
from .grid import rate_of  # noqa: E402
from .moves import template  # noqa: E402


@template("windows", prop=True)
def windows(c: Ctx) -> None:
    """Windows lighting up along his path (prop; obstacles.json `lit`, built by `python -m aimv.paint.windows`):
    from, to (the stretch of his path that lights them), every (beats: the grid they switch on), lead (px: a window
    lights once his torso is this far short of its centre), spread (700 px of height a beat), fade (16ths, 1),
    cells (default all). Each window lights on the first beat after he passes it (Windows.times)."""
    spec = c.ch.scene.spec.get("lit") or {}
    if not spec.get("cells"):
        raise ChoreoError("windows: obstacles.json has no `lit` cells")
    q0, q1 = c.time("from"), c.time("to")
    if q0 is None or q1 is None or q1 <= q0:
        raise ChoreoError("windows: give from and to (to after from)")
    cells = c.arg("cells")
    if cells:
        bad = [x for x in cells if x not in spec["cells"]]
        if bad:
            raise ChoreoError(f"windows: no lit cell {', '.join(bad)} (obstacles.json lit: {', '.join(spec['cells'])})")
    c.ch.lit = {"q0": float(c.song(q0)), "q1": float(c.song(q1)), "origin": float(c.song(q0)),
                "every": rate_of(c.arg("every", "beats")), "lead": float(c.arg("lead", 0.0)),
                "spread": float(c.arg("spread", 700.0)), "fade": float(c.arg("fade", 1.0)),
                "cells": list(cells) if cells else None}


@template("flicker", prop=True)
def flicker_prop(c: Ctx) -> None:
    """The city's lights flickering faintly on every photo from `from` to `to` (prop; windows.flicker): amount (the
    most a light brightens or dims, 0.1), cell (street px a light's draw covers, 10), thr (linear luminance of a
    light, 0.2)."""
    q0, q1 = c.time("from"), c.time("to")
    if q0 is None or q1 is None or q1 <= q0:
        raise ChoreoError("flicker: give from and to (to after from)")
    amt, cell, thr = float(c.arg("amount", 0.1)), int(c.arg("cell", 10)), float(c.arg("thr", 0.2))
    if not 0 < amt < 1 or cell < 2 or thr <= 0:
        raise ChoreoError("flicker: amount between 0 and 1, cell 2 px or more, thr above 0")
    fl = list(getattr(c.ch, "flickers", None) or [])
    fl.append({"q0": float(c.song(q0)), "q1": float(c.song(q1)), "amount": amt, "cell": cell, "thr": thr})
    c.ch.flickers = fl
