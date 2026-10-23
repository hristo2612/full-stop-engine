"""The standalone stencil (a prop without a box: WAIT by the second red light, verse 2's pitch lines) and its smear
(a body running through it drags the letters along), on test_paint's synthetic project."""

import numpy as np

from aimv.paint import props as PR
from aimv.paint.render import History

from test_paint import _with, world  # noqa: F401  (its synthetic project: song grid, poses, street)


def test_stencil_smear_drags_the_letters_along():
    st = PR.Stencil(0.0, ["WAIT"], 100, 100, 60)
    paint, hard, x0, y0 = PR.stencil_mask(st)
    L = 200
    pad = np.pad(paint, ((0, 0), (0, L)))
    w = paint.shape[1]
    half = PR.stencil_smear(pad, L, 0.5, seed=3)
    full = PR.stencil_smear(pad, L, 1.0, seed=3)
    right = lambda a: a[:, w + 20:].sum()
    assert right(pad) == 0 and 0 < right(half) < right(full)  # it reaches past the letters, further when it is done
    assert full[:, :w].sum() < pad[:, :w].sum() * 1.1  # the letters thin where their paint was pulled away
    rows = (paint.shape[0] // 2, paint.shape[0])
    band = PR.stencil_smear(pad, L, 1.0, rows, seed=3)
    assert np.array_equal(band[:rows[0]], pad[:rows[0]]) and right(band[rows[0]:]) > 0
    left = PR.stencil_smear(np.pad(paint, ((0, 0), (L, 0))), L, 1.0, seed=3, right=False)
    assert left[:, :L - 20].sum() > 0


def test_stencil_template_paints_words_and_smears_them(world):
    d = {"street": "main", "seed": 3, "motion": "m0", "segments": [{"from": 4.1, "to": 7.1, "rate": "16ths"}],
         "moves": [{"stencil": {"at": 4.2, "text": "WAIT", "x": 1500, "y": 150, "h": 80,
                                "smear": {"at": 5.2, "length": 300, "over": 4}}},
                   {"run": {"from": 4.1, "to": 6.4, "x": [900, 2400]}}]}
    ch = _with(world, "wait", d)
    assert not ch.errors, [str(i) for i in ch.issues]
    pr = ch.props["stencil1"]
    assert not pr.keys and len(pr.stencils) == 1 and pr.stencils[0].lines == ["WAIT"]
    assert pr.stencils[0].smear[1] == 300
    hist = History(ch)
    marks = [m for m in hist.marks if m.kind == "stencil"]
    assert len(marks) == 1  # one mark per stencil, the smear included
    g = ch.grid
    x0, y0, w, h = 1400, 100, 900, 250
    cov = lambda t: hist.layers(t, x0, y0, w, h)[1]
    before, on, smeared = cov(g.t(g.resolve(4.1)[0])), cov(g.t(g.resolve(5.1)[0])), cov(g.t(g.resolve(6.1)[0]))
    bx0, by0, bx1, by1 = pr.stencil_box(0)
    ink = (slice(int(by0) - y0, int(by1) - y0), slice(int(bx0) - x0, int(bx1) - x0))
    beyond = (slice(int(by0) - y0, int(by1) - y0), slice(int(bx1) - x0 + 30, int(bx1) - x0 + 250))
    assert before[ink].max() == 0 and on[ink].mean() > 0.1
    assert on[beyond].max() < 0.05 and smeared[beyond].mean() > 0.05  # dragged right past its letters
