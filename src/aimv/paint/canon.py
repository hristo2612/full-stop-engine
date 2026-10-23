"""The canon check: every drawing in the pose library against the runner's canon (its model sheet),
so a drawing that drifted never reaches a render unnoticed (`aimv paint canon`; the QA gate reads canon.json).

A video model only sees a clip's first and last frames (both built from canon drawings), so the drawings in between
can drift: the figure grows toward the camera, a hand smears into a grey ghost, skin or a face shows, the white shoe
parts vanish, the figure melts into pieces. The check measures each drawing (its paint-density matte in paint/poses
and, for clip drawings, the clip frame it came from) against the canon sheet (R1) and against its own clip.

Rules (an error keeps the drawing out of a QA-passing render; warnings are listed):

    scale     body size off: the painted area at street scale outside 0.55 to 1.45 times the canon's median, or a clip
              drawing outside 0.55 to 1.35 times its own clip's median (the figure grew toward the camera or shrank)
    smear     clip drawings: a soft grey zone that fades straight into the wall (a hand or foot smeared by motion
              blur, a melt). Dry-brush streaks inside the figure are fenced off by darker paint and do not count. An
              error over max(6 x the clip's median, 0.6), a warning over max(6 x the median, 0.3) (per 1000 of the
              body height squared)
    colour    clip drawings: skin or any other colour inside the figure (a face, bare skin, coloured clothes), over
              1.5% of its pixels in the clip frame (the pale blue guide block is ignored)
    shoes     clip drawings: the white shoe parts (socks, midsole, flash: light shapes held between paint above and
              below at the feet) below 35% of the clip's first and last drawings (a warning); when over 30% of a clip's drawings
              miss them, an error on each of those (the clip lost the canon shoes)
    pieces    the figure split into big pieces (over 8% of its paint away from the main body): a melt or scribbles
              (a warning)

Sheets (stills from an image model) get the scale and pieces rules only: they are checked by eye against the model
sheet (boards/16-model MS1 turnaround, MS2 details) before they are cut.

A clip of him inside a box prop (poses.spec.json video `boxed: true`) skips the canon's area: the box sets his size
(he is folded small in it); his clip's median still applies. `canon_waive` in poses.spec.json ({drawing: {rule:
reason}}) turns errors that were checked by eye and found clean (a dry-brush highlight on his back that the smear rule
reads as a smear) into warnings that say so.

Output: paint/poses/canon.json ({"canon": the reference medians, "sources": per sheet or clip, "drawings": per drawing
metrics and flags}) and canon.jpg (every source as rows of thumbnails, flagged drawings outlined: red error, amber
warning, with the rule under them).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..util import read_json, write_json

REFERENCE = "R1"
STAMPS = ("SP1",)  # splash stamps, not the runner
OTHERS = ("OT1",)  # other people (the boxed figures): scale only

SCALE_CANON = (0.55, 1.45)
SCALE_CLIP = (0.55, 1.35)
SMEAR_X, SMEAR_ERR, SMEAR_WARN = 6.0, 0.6, 0.3
COLOUR = 0.015
SHOES_REL, SHOES_CLIP = 0.35, 0.30
PIECES = 0.08


def matte(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"canon: cannot read {path}")
    return m.astype(np.float32) / 255.0


def _outside(mask: np.ndarray) -> np.ndarray:
    """The parts of a boolean mask connected (4-way) to the image border."""
    _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, border[border > 0]) & mask


def smear(d: np.ndarray, H: float) -> float:
    """The largest soft zone on the figure's edge that fades straight into the wall: mid density (0.1 to 0.6), more
    than 1% of the body height from both solid paint (over 0.75) and bare wall (under 0.05), and reachable from
    outside the figure without crossing paint denser than 0.6. Per 1000 of H squared."""
    r = max(2.0, 0.01 * H)
    d = np.pad(d, 8)
    solid = (d > 0.75).astype(np.uint8)
    wall = (d < 0.05).astype(np.uint8)
    ds = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
    dw = cv2.distanceTransform(1 - wall, cv2.DIST_L2, 3)
    mid = ((d > 0.1) & (d < 0.6) & (ds > r) & (dw > r)).astype(np.uint8)
    out = _outside(d < 0.6)
    k, lab, st, _ = cv2.connectedComponentsWithStats(mid, 8)
    best = 0
    for i in range(1, k):
        a = st[i, cv2.CC_STAT_AREA]
        if a > best and (out & (lab == i)).sum() > 0.5 * a:
            best = a
    return float(best) / (H * H) * 1000


def shoes(d: np.ndarray, y0: float, H: float) -> float:
    """White shoe parts: light pixels at the feet held between paint above and below within 3.5% of the body height
    (a sock between cuff and shoe, a flash or midsole inside the shoe's outline; the gap between two legs is held
    only left and right, so it does not count). The feet: the
    bottom 16% of the figure plus boxes round its left- and rightmost paint in the lower half (a foot kicked up
    behind him). Per 1000 of H squared."""
    r = max(int(round(0.035 * H)), 3)
    dark, light = d > 0.5, d < 0.3
    region = np.zeros_like(dark)
    region[int(y0 + 0.84 * H):, :] = True
    lo = int(y0 + 0.45 * H)
    ys, xs = np.nonzero(dark[lo:, :])
    if len(xs):
        ys = ys + lo
        hh = max(int(round(0.07 * H)), 2)
        for x in (np.percentile(xs, 0.5), np.percentile(xs, 99.5)):
            sel = np.abs(xs - x) < hh
            if sel.any():
                cy = int(np.median(ys[sel]))
                region[max(cy - hh, 0):cy + hh, max(int(x) - 2 * hh, 0):int(x) + 2 * hh] = True
    above, below = np.zeros_like(dark), np.zeros_like(dark)
    for k in range(1, r + 1):
        above[k:, :] |= dark[:-k, :]
        below[:-k, :] |= dark[k:, :]
    held = light & above & below & region
    return float(held.sum()) / (H * H) * 1000


def metrics(d: np.ndarray, scale: float) -> dict:
    """Shape metrics of one density matte (0 wall .. 1 solid paint) drawn at `scale` (sprite px to street px)."""
    paint = d > 0.25
    n = int(paint.sum())
    if n == 0:
        return {"area": 0.0, "height": 0.0, "smear": 0.0, "shoes": 0.0, "pieces": 1.0}
    ys, _ = np.nonzero(paint)
    y0, y1 = np.percentile(ys, 0.2), np.percentile(ys, 99.8)
    H = max(float(y1 - y0), 1.0)
    k = max(int(round(0.02 * H)) | 1, 3)
    m = cv2.morphologyEx(paint.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    areas = sorted((int(a) for a in st[1:, cv2.CC_STAT_AREA]), reverse=True)
    away = sum(a for a in areas[1:] if a > 0.01 * n) / max(sum(areas), 1)
    return {"area": round(n * scale * scale, 1), "height": round(H * scale, 1), "smear": round(smear(d, H), 3),
            "shoes": round(shoes(d, y0, H), 3), "pieces": round(float(away), 4)}


def colour(frame: np.ndarray, d: np.ndarray, ox: int, oy: int, key=None) -> float:
    """Share of the figure's pixels in a clip frame (BGR) that carry colour: skin, a face, coloured clothes. Black
    emulsion and the white wall are neutral; the pale blue guide block (key, RGB) is ignored."""
    h, w = d.shape
    crop = frame[oy:oy + h, ox:ox + w]
    fig = d > 0.25
    if crop.shape[:2] != d.shape or not fig.any():
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0] * 2, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    col = (sat > 0.28) & (val > 0.22)
    if key is not None:
        kh = float(cv2.cvtColor(np.uint8([[[key[2], key[1], key[0]]]]), cv2.COLOR_BGR2HSV)[0, 0, 0]) * 2
        col &= np.abs(((hue - kh + 180) % 360) - 180) > 25
    return float((col & fig).sum()) / float(fig.sum())


def _frames(proj_root: Path, path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(proj_root / path))
    out, f, last = {}, 0, max(wanted, default=-1)
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        if f in wanted:
            out[f] = img
        f += 1
    cap.release()
    return out


def judge(src: str, names: list[str], met: dict, canon: dict, clip: bool, boxed: bool = False,
          waive: dict | None = None, screen: bool = False) -> tuple[dict, dict]:
    """Flags for every drawing of one source (a sheet or a clip, names in time order). boxed: a clip of him inside a
    box prop (the box sets his size, so the canon's area does not apply; his clip's median still does). waive:
    {drawing: {rule: reason}}, flags checked by eye and found clean (a dry-brush highlight the smear rule reads as a
    smear): those errors become warnings that say so. screen: a source shot on chroma green (the true figure:
    photoreal, not paint): the paint's rules (smear, colour, shoes) do not apply. Returns (source summary, {name:
    [(level, rule, why)]})."""
    ms = [met[n] for n in names]
    med_area = float(np.median([m["area"] for m in ms]))
    med_smear = float(np.median([m["smear"] for m in ms]))
    flags: dict[str, list] = {n: [] for n in names}
    ends = [m["shoes"] for m in ms[:3] + ms[-3:]] if clip and len(ms) >= 6 else []
    ref_shoes = float(np.mean(ends)) if ends else 0.0
    low = [n for n in names if clip and not screen and src not in OTHERS and ref_shoes > 0 and
           met[n]["shoes"] < SHOES_REL * ref_shoes]
    gone = clip and len(low) > SHOES_CLIP * len(names)
    for n in names:
        m, fl = met[n], flags[n]
        a = m["area"] / canon["area"]
        if not boxed and not SCALE_CANON[0] <= a <= SCALE_CANON[1]:
            fl.append(("error", "scale", f"body area {a:.2f}x the canon's"))
        if clip:
            c = m["area"] / med_area
            if not SCALE_CLIP[0] <= c <= SCALE_CLIP[1]:
                fl.append(("error", "scale", f"body area {c:.2f}x its clip's median"))
            base = SMEAR_X * max(med_smear, 0.02)
            if screen:
                pass
            elif m["smear"] > max(base, SMEAR_ERR):
                fl.append(("error", "smear", f"smeared edge {m['smear']:.2f} (clip median {med_smear:.2f})"))
            elif m["smear"] > max(base, SMEAR_WARN):
                fl.append(("warning", "smear", f"soft edge {m['smear']:.2f} (clip median {med_smear:.2f})"))
            if m.get("colour", 0.0) > COLOUR and not screen:
                fl.append(("error", "colour", f"{m['colour']:.1%} of the figure in colour"))
            if n in low:
                why = f"white shoe parts {m['shoes']:.2f} (clip ends {ref_shoes:.2f})"
                fl.append(("error", "shoes", why + f"; {len(low)}/{len(names)} of the clip miss them") if gone
                          else ("warning", "shoes", why))
        if m["pieces"] > PIECES:
            fl.append(("warning", "pieces", f"{m['pieces']:.0%} of the paint away from the body"))
        wv = (waive or {}).get(n) or {}
        flags[n] = [("warning", r, f"{w}; waived by eye: {wv[r]}") if lv == "error" and r in wv else (lv, r, w)
                    for lv, r, w in fl]
    summary = {"n": len(names), "clip": clip, "area": round(med_area, 1), "smear": round(med_smear, 3),
               "shoes_ends": round(ref_shoes, 3), "shoes_missing": len(low), "shoes_gone": bool(gone),
               "errors": sum(any(f[0] == "error" for f in flags[n]) for n in names)}
    return summary, flags


def run(proj_root: Path, poses_dir: Path, log=print) -> dict:
    """Check every drawing of the pose library; write canon.json and canon.jpg next to poses.json."""
    lib = read_json(poses_dir / "poses.json")
    P, V = lib["poses"], lib.get("videos", {})
    met: dict[str, dict] = {}
    by_src: dict[str, list[str]] = {}
    for name, p in P.items():
        if p.get("kind") != "pose" or p["sheet"] in STAMPS:
            continue
        met[name] = metrics(matte(poses_dir / f"{name}.png"), float(p.get("scale", 1.0)))
        by_src.setdefault(p["sheet"], []).append(name)
    for src, names in by_src.items():
        names.sort(key=lambda n: (P[n].get("video", {}).get("t", 0.0), n))
        v = V.get(src)
        if not v:
            continue
        frames = _frames(proj_root, v["path"], {P[n]["video"]["f"] for n in names if "video" in P[n]})
        for n in names:
            vv = P[n].get("video")
            if vv and vv["f"] in frames:
                met[n]["colour"] = round(colour(frames[vv["f"]], matte(poses_dir / f"{n}.png"), vv["ox"], vv["oy"],
                                                v.get("key")), 4)
    if REFERENCE not in by_src:
        raise SystemExit(f"canon: the reference sheet {REFERENCE} is not in the pose library")
    canon = {"reference": REFERENCE, "area": float(np.median([met[n]["area"] for n in by_src[REFERENCE]]))}
    drawings, sources = {}, {}
    for src in sorted(by_src):
        clip = src in V and all("video" in P[n] for n in by_src[src])
        summary, flags = judge(src, by_src[src], met, canon, clip, bool(V.get(src, {}).get("boxed")),
                               lib.get("canon_waive"), any(P[n].get("screen") for n in by_src[src]))
        sources[src] = summary
        for n in by_src[src]:
            drawings[n] = {"source": src, **met[n],
                           "flags": [{"level": lv, "rule": r, "why": w} for lv, r, w in flags[n]]}
    errors = sum(any(f["level"] == "error" for f in d["flags"]) for d in drawings.values())
    warnings = sum(bool(d["flags"]) for d in drawings.values()) - errors
    res = {"version": 1, "canon": canon,
           "rules": {"scale_canon": SCALE_CANON, "scale_clip": SCALE_CLIP, "smear": [SMEAR_X, SMEAR_ERR, SMEAR_WARN],
                     "colour": COLOUR, "shoes": [SHOES_REL, SHOES_CLIP], "pieces": PIECES},
           "errors": errors, "warnings": warnings, "sources": sources, "drawings": drawings}
    write_json(poses_dir / "canon.json", res, nd=4)
    sheet(poses_dir, res, poses_dir / "canon.jpg")
    for src, s in sources.items():
        bad = [n for n in by_src[src] if any(f["level"] == "error" for f in drawings[n]["flags"])]
        if bad:
            log(f"  {src}: {len(bad)}/{s['n']} drawings with errors: " + ", ".join(
                f"{n} ({', '.join(sorted({f['rule'] for f in drawings[n]['flags'] if f['level'] == 'error'}))})"
                for n in bad))
    log(f"canon: {len(drawings)} drawings, {errors} with errors, {warnings} with warnings only -> "
        f"{poses_dir / 'canon.json'}, {poses_dir / 'canon.jpg'}")
    return res


def load(poses_dir: Path) -> dict:
    """canon.json's drawings ({} when the check has not run): {name: {"flags": [...], ...}}."""
    f = poses_dir / "canon.json"
    return read_json(f).get("drawings", {}) if f.exists() else {}


def errors_of(drawings: dict, pose: str) -> list[dict]:
    """The canon errors of one drawing (none when it was not checked)."""
    return [f for f in drawings.get(pose, {}).get("flags", []) if f["level"] == "error"]


def sheet(poses_dir: Path, res: dict, out: Path, th: int = 96, per_row: int = 24) -> Path:
    """Every source as rows of thumbnails (white paint on dark, so drips and holes show), flagged drawings outlined
    (red: error, amber: warning) with the rule written under them."""
    by_src: dict[str, list[str]] = {}
    for n, d in res["drawings"].items():
        by_src.setdefault(d["source"], []).append(n)
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    for src in sorted(by_src):
        names = sorted(by_src[src])
        for i in range(0, len(names), per_row):
            cells = []
            for n in names[i:i + per_row]:
                d = cv2.imread(str(poses_dir / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
                s = th / max(d.shape)
                t = cv2.resize(d, (max(int(d.shape[1] * s), 1), max(int(d.shape[0] * s), 1)), interpolation=cv2.INTER_AREA)
                cell = np.full((th + 30, th + 8, 3), 32, np.uint8)
                y, x = (th - t.shape[0]) // 2 + 4, (th + 8 - t.shape[1]) // 2
                cell[y:y + t.shape[0], x:x + t.shape[1]] = np.dstack([t] * 3)
                fl = res["drawings"][n]["flags"]
                if fl:
                    c = (40, 40, 230) if any(f["level"] == "error" for f in fl) else (40, 170, 240)
                    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, th + 7), c, 2)
                    cv2.putText(cell, ",".join(sorted({f["rule"] for f in fl}))[:16], (2, th + 25), font, 0.32, c, 1)
                cv2.putText(cell, n.split("-", 1)[1], (2, th + 14), font, 0.33, (200, 200, 200), 1)
                cells.append(cell)
            row = np.hstack(cells)
            lab = np.full((row.shape[0], 70, 3), 20, np.uint8)
            cv2.putText(lab, src, (6, 40), font, 0.6, (230, 230, 230), 1)
            rows.append(np.hstack([lab, row]))
    W = max(r.shape[1] for r in rows)
    img = np.vstack([np.pad(r, ((0, 2), (0, W - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
