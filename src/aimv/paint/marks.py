"""Painted marks in time with his arm: strokes in stages (the smear down the far wall, his word, the dot).

A stroke sheet is a `stamp` sheet in poses.spec.json (one cell per stage, each cell the stroke painted up to there,
the drips it has grown so far included). The `marks` prop places it stage by stage, one stamp at each stage's time, so
the mark grows with his arm. Every stage covers the ones before it (the newest paint on top, wet), and the whole mark
stays on the wall for the rest of the film (a stamp is a splash mark: it settles at 62% of its density and washes a
little in the rain). As a prop it is placed before the runner's moves and stays out of their chain (it has no time or
street extent of its own), so its strokes may overlap the clips that paint them.

A stamp is placed by its paint's bbox centre (x, y, street px) and its cell width (w, street px), as `stamp` does; a
tool that draws the sheet knows each stage's crop and centre.
"""

from __future__ import annotations

from .choreo import ChoreoError, Ctx
from .moves import template


@template("marks", prop=True)
def marks(c: Ctx) -> None:
    """Strokes in stages (prop): stages: [[at, stamp, x, y, w], ...], each a time, a stamp drawing (a cell of a stroke
    sheet), the centre of its paint (x, y, street px or obstacle coordinates) and its width (w, street px), and
    optionally its own flags as a sixth item ({splash: true}: droplets fly off it, `drops` of them (40), `lead` s
    already in the air on its photo (0); drips: [n, length px] its own runs of paint; {swath: true}: the stamp's shape
    painted in white primer instead of black, staying on the wall; {primer: true}, or {primer: keep}:
    the primer patch stays on the wall like the stroke instead of fading as he paints on); primer (false: no
    primer patch under any stage); photo (false; true puts a photo exactly on each stage); label (mark)."""
    stages = c.require("stages")
    if not isinstance(stages, list) or not stages:
        raise ChoreoError("marks: stages is a list of [at, stamp, x, y, w]")
    tag = str(c.arg("label", "mark"))
    primer = c.arg("primer", False)
    primer = "keep" if primer == "keep" else bool(primer)  # (keep: the primer patch stays, as the mark does)
    photo = bool(c.arg("photo", False))
    for i, s in enumerate(stages):
        if not isinstance(s, list | tuple) or len(s) not in (5, 6) or (len(s) == 6 and not isinstance(s[5], dict)):
            raise ChoreoError(f"marks: stage {i + 1} is not [at, stamp, x, y, w] (or with its flags: {{splash: true}})")
        at, name, x, y, w = s[:5]
        fl = dict(s[5]) if len(s) == 6 else {}
        q = c.at(at, key=f"stages[{i + 1}]")
        if q is None:
            raise ChoreoError(f"marks: stage {i + 1} has no time")
        name = str(name)
        if name not in c.ch.lib:
            raise ChoreoError(f"marks: unknown stamp {name!r} (add its sheet to poses.spec.json and run `aimv paint "
                              f"poses`)")
        if c.ch.lib.m(name)["kind"] != "stamp":
            raise ChoreoError(f"marks: {name!r} is a pose, not a stamp (its sheet needs kind: stamp)")
        if float(w) <= 0:
            raise ChoreoError(f"marks: stage {i + 1} needs a width above 0")
        c.stamp(q, name, c.coord(x, "x"), c.coord(y, "y"), float(w),
                flags={"swath": bool(fl.get("swath", False)),
                       "primer": "keep" if fl.get("primer", primer) == "keep" else bool(fl.get("primer", primer)),
                       "splash": bool(fl.get("splash", False)),
                       **({"drops": int(fl["drops"])} if "drops" in fl else {}),
                       **({"lead": float(fl["lead"])} if "lead" in fl else {}),
                       **({"drips": [int(fl["drips"][0]), float(fl["drips"][1])]} if "drips" in fl else {})},
                tag=tag)
        if photo:
            c.photo(q)
