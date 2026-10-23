"""Screens switched off (Full Stop's verse 2): shop windows full of glowing screens on the street plates, each window
going dark at its own time (the choreography's `screens` move).

The plates come in pairs: V1.png with the screens on, V1-off.png the same photograph with every screen dead (no
faces, no glow on the pavement or in the road). `build` makes, for every group in obstacles.json `screens`, the
street with that window's screens off: each street pixel of the group's band is read from the off plate the strip
took it from (the plate whose on version matches the street there best, both warped by the strip's own transforms
from build.json), gain-matched to the strip (the stitch's exposure fades), and saved as screens/<id>.png (sRGB, the
band's columns, all rows). screens/<id>-light.png holds the ratio of the off street's light to the on street's
(smoothed, x 40000 in uint16): the paint round the window dims with it (the stencils' glow and contrast drop).
The renderer blends a band in from `top` down, feathered over `feather` px at its sides and top.

Run: python -m aimv.paint.screens <project> [street]."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import street as ST
from .scene import luminance, srgb2lin

LIGHT_SCALE = 40000.0  # the light ratio's uint16 scale


def _plate_maps(T: list, crop: list, off: float, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plate pixel coordinates of street pixels (xs, ys): street x -> strip column (x + off) -> canvas (crop's
    origin, the stitch's OY margin) -> the plate (the inverse of its transform)."""
    M = np.vstack([np.float64(T), [0, 0, 1]])
    M[1, 2] += ST.OY
    Mi = np.linalg.inv(M)
    cx = xs + off + crop[0]
    cy = ys + crop[1]
    mx = Mi[0, 0] * cx + Mi[0, 1] * cy + Mi[0, 2]
    my = Mi[1, 0] * cx + Mi[1, 1] * cy + Mi[1, 2]
    return mx.astype(np.float32), my.astype(np.float32)


def _strip_offset(spec: dict, strip_info: dict, x: float) -> float:
    """Strip column minus street x for street columns near x (a plain piece of the edited street)."""
    F = int(spec["edit_feather"])
    ps = ST.pieces(strip_info["size"][0], ST._side_edits(spec, 1), F)
    for s in range(0, strip_info["size"][0], 50):
        X = ST.xmap(s, ps, F)
        if X is not None and abs(X - x) < 25:
            return float(s - X)
    raise SystemExit(f"screens: street x {x} is not on a plain piece of the strip")


def off_band(proj_root: Path, d: Path, x0: int, x1: int, plates: dict, log=print) -> tuple[np.ndarray, np.ndarray]:
    """The street's columns [x0, x1) (all rows) with the screens off: (RGB uint8 image, light ratio float32)."""
    spec = ST.load_spec(d)
    build = json.loads((d / "build.json").read_text())
    strip = build["strip"]
    st = ST.Street(d, world=False)
    on_street = st.window(x0, x1).astype(np.float32) / 255
    H, w = on_street.shape[:2]
    off = _strip_offset(spec, strip, (x0 + x1) / 2)
    ys, xs = np.mgrid[0:H, x0:x1].astype(np.float64)
    errs, offs, ons = [], [], []
    for p in strip["plates"]:
        if p["path"] not in plates:
            continue
        A = cv2.cvtColor(cv2.imread(str(proj_root / p["path"])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        B = cv2.cvtColor(cv2.imread(str(proj_root / plates[p["path"]])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if A.shape != B.shape:
            raise SystemExit(f"screens: {plates[p['path']]} is not the size of {p['path']}")
        mx, my = _plate_maps(p["T"], strip["crop"], off, xs, ys)
        ok = (mx >= 0) & (mx <= A.shape[1] - 1) & (my >= 0) & (my <= A.shape[0] - 1)
        if not ok.any():
            continue
        a = cv2.remap(A, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(B, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        e = cv2.GaussianBlur(np.abs(a - on_street).mean(2), (0, 0), 6)
        e[~ok] = np.inf
        # the stitch's gain on this plate here (exposure fades between plates): the street over the on plate
        g = (cv2.GaussianBlur(on_street, (0, 0), 24) + 0.02) / (cv2.GaussianBlur(a, (0, 0), 24) + 0.02)
        errs.append(e)
        offs.append(np.clip(b * np.clip(g, 0.6, 1.6), 0, 1))
        ons.append(p["path"])
    if not errs:
        raise SystemExit(f"screens: no off plate covers street x {x0} to {x1}")
    E = np.stack(errs)
    wts = 1.0 / (np.where(np.isfinite(E), E, 1e3) + 0.01) ** 4  # the plate the strip took the pixel from, softly
    wts /= wts.sum(0, keepdims=True)
    img = (np.stack(offs) * wts[..., None]).sum(0)
    log(f"screens: x {x0} to {x1} from {', '.join(Path(p).stem for p in ons)}")
    lo, ln = luminance(srgb2lin(img)), luminance(srgb2lin(on_street))
    ratio = np.clip((cv2.GaussianBlur(lo, (0, 0), 30) + 1e-3) / (cv2.GaussianBlur(ln, (0, 0), 30) + 1e-3), 0.05, 1.5)
    return (np.clip(img * 255 + 0.5, 0, 255)).astype(np.uint8), ratio.astype(np.float32)


def build(proj_root: Path, d: Path, log=print) -> dict:
    """screens/<id>.png and <id>-light.png for every group of obstacles.json `screens` (see the module docstring);
    returns screens/screens.json's content."""
    ob = json.loads((d / "obstacles.json").read_text())
    sc = ob.get("screens")
    if not sc:
        raise SystemExit("screens: obstacles.json has no `screens`")
    out = d / str(sc.get("dir", "screens"))
    out.mkdir(parents=True, exist_ok=True)
    fe = int(sc.get("feather", 60))
    info = {"top": int(sc.get("top", 0)), "feather": fe, "groups": {}}
    for g in sc["groups"]:
        a, b = int(g["x"][0]) - fe, int(g["x"][1]) + fe
        img, ratio = off_band(proj_root, d, a, b, sc["plates"], log)
        cv2.imwrite(str(out / f"{g['id']}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{g['id']}-light.png"), np.clip(ratio * LIGHT_SCALE, 0, 65535).astype(np.uint16))
        info["groups"][g["id"]] = {"x": [int(g["x"][0]), int(g["x"][1])], "x0": a, "x1": b}
    (out / "screens.json").write_text(json.dumps(info, indent=1))
    return info


class Screens:
    """The built screen bands of a street, loaded on demand: blend(lin, E, x0, y0, offs, q) switches the windows off
    whose time has come (offs: id -> (song 16th, fade 16ths), the choreography's screens_off; q: the photo's)."""

    def __init__(self, d: Path, sc: dict | None):
        self.sc = sc or {}
        self.dir = Path(d) / str(self.sc.get("dir", "screens"))
        f = self.dir / "screens.json"
        self.info = json.loads(f.read_text()) if f.exists() else None
        self._cache: dict[str, tuple] = {}

    def _load(self, gid: str):
        if gid not in self._cache:
            img = cv2.imread(str(self.dir / f"{gid}.png"))
            lt = cv2.imread(str(self.dir / f"{gid}-light.png"), cv2.IMREAD_UNCHANGED)
            if img is None or lt is None:
                raise SystemExit(f"screens: {self.dir / gid}.png missing: run `python -m aimv.paint.screens`")
            self._cache[gid] = (srgb2lin(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), lt.astype(np.float32) / LIGHT_SCALE)
        return self._cache[gid]

    def mask(self, gid: str, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """The group's blend weight over street box [x0, x0 + w) x [y0, y0 + h): 1 in its band from `top` down,
        feathered over `feather` px at its sides and top."""
        g = self.info["groups"][gid]
        fe, top = float(self.info["feather"]), float(self.info["top"])
        xs = np.arange(x0, x0 + w, dtype=np.float32)
        ys = np.arange(y0, y0 + h, dtype=np.float32)
        mx = np.clip(np.minimum(xs - (g["x"][0] - fe / 2), (g["x"][1] + fe / 2) - xs) / fe + 0.5, 0, 1)
        my = np.clip((ys - top) / fe + 0.5, 0, 1)
        mx, my = mx * mx * (3 - 2 * mx), my * my * (3 - 2 * my)
        return (my[:, None] * mx[None, :]).astype(np.float32)

    def blend(self, lin: np.ndarray, E: np.ndarray, x0: int, y0: int, offs: dict, q: float):
        """lin (h x w x 3, linear) and E (h x w, the paint's light) of street box (x0, y0) with every window off
        whose time has come; returns (lin, E, weight of the darkness: h x w)."""
        h, w = lin.shape[:2]
        dark = np.zeros((h, w), np.float32)
        if not self.info:
            return lin, E, dark
        for gid, (q_off, fade) in offs.items():
            g = self.info["groups"].get(gid)
            if g is None or q < q_off - 1e-6 or g["x1"] <= x0 or g["x0"] >= x0 + w:
                continue
            a = 1.0 if fade <= 0 else float(np.clip((q - q_off) / fade + 1e-6, 0, 1))
            img, ratio = self._load(gid)
            c0, c1 = max(x0, g["x0"]), min(x0 + w, g["x1"])
            r0, r1 = max(y0, 0), min(y0 + h, img.shape[0])
            if c1 <= c0 or r1 <= r0:
                continue
            m = self.mask(gid, c0, r0, c1 - c0, r1 - r0) * a
            sl = (slice(r0 - y0, r1 - y0), slice(c0 - x0, c1 - x0))
            src = (slice(r0, r1), slice(c0 - g["x0"], c1 - g["x0"]))
            lin[sl] = lin[sl] * (1 - m[..., None]) + img[src] * m[..., None]
            E[sl] = E[sl] * (1 - m + m * ratio[src])
            dark[sl] = np.maximum(dark[sl], m)
        return lin, E, dark


if __name__ == "__main__":
    from ..project import Project

    p = Project.open(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "main"
    build(Path(p.root), Path(p.root) / "paint" / "streets" / name)
