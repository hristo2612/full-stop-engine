# aimv paint: the painted-runner side-scroller

A runner who exists only as black emulsion paint on the walls of a long night street does parkour over the real
objects on the wall (pipes, AC units, ledges, ladders), one continuous sideways shot with no cuts. Every
photograph is one painted pose, held on the song's grid like stitched photographs (after Blu's MUTO): a pose per
16th in runs, per 8th in calmer parts, per beat or slower in slow-downs. Paint goes behind the real objects, older
poses stay on the wall as ghosts, and the song's stops are his freezes while the photos keep coming.

This is prototype 6 made into a tool that works for any project.
It spends nothing: plates and sprite sheets are made with `aimv board` (paid, logged) before you start; everything
here runs locally and is deterministic.

```
aimv paint street  <project> [--street main] [--canvas P6]   plates to a tiled street (or the canvas for the next plate;
                   [--canvas L6 --left]                       --left: for the next plate prepended on the left)
                   [--canvas G11 --rect X0 Y0 X1 Y1]          a world cell's canvas (anchor px; see The world)
aimv paint scene   <project> [--street main]                 check the obstacles, cache the masks, write scene-masks.jpg
aimv paint poses   <project>                                 sprite sheets and clips to the pose library, contact sheet,
                                                             and the canon check
aimv paint window  <project> --from X --to X [--out W.png]   a street window at full resolution (for a patch edit)
aimv paint check   <project> <choreo>                        validate a choreography and summarise it
aimv paint render  <project> <choreo> [--from T --to T] [--aspect 16:9|9:16|both] [--preview] [--only 3,4,5]
aimv paint still   <project> <choreo> --at T [--aspect ...] [--preview] [--raw]
aimv paint map     <project> <choreo>                        the street with obstacles and every pose as a dot
aimv paint contact <project> <choreo> --video V [--times ...] [--against OTHER.mp4 --against-t0 T]
aimv paint clipframes <project> <obstacle> --move vault [--cycle run_SW] [--and beam] [--start-pose hang]
                   [--box 400x450x34 --scale-by BX1-6]       start and end frames for an obstacle clip (or a front view
                   [--open right] [--open-start right]       in the box; a chain of them; the box anywhere in the frame)
                   [--box-x 420] [--folder box] [--start-x X --end-x X]
python -m aimv.paint.screens <project> [street]              the screens-off bands (obstacles.json `screens`)
```

Times on the command line are seconds (`57.6`), a grid address with a 16th (`43.2.1`) or a named hit (`impact`).
`--jump J1|J2`, `--primer all|keepers|none|ghosts`, `--motion m0|legacy`, `--no-foot-lock` and `--no-vary` override the choreography's
settings for one run (for comparing).

## Data layout

```
projects/<p>/paint/
  streets/<name>/street.json      the street spec: plates and game-level edits (hand-written)
  streets/<name>/obstacles.json   ground lines, obstacles and contacts, occluders, no-paint zones, layers, lights
  streets/<name>/build.json       what the last build made: plate transforms, seams, tile hashes (written)
  streets/<name>/strip/s0000.png  the stitched strip, 2048 px wide tiles (written); s-0001.png ... left of x 0
  streets/<name>/tiles/t0000.png  the edited street, 2048 px wide tiles (written); t-0001.png ... left of x 0
  streets/<name>/world/          the world round the street: RGBA overlay tiles w{ix}_{iy}.png, world.json,
                                  valid8.png (written; see The world); world.jpg, world-seams.jpg beside them
  streets/<name>/scene-masks.jpg  occluders red, layers blue, no-paint grey, boxes yellow, contacts green (written)
  poses/poses.spec.json           sheets, scale, cycles, roles, tags (hand-written)
  poses/<sheet>-<k>.png           density mattes (written), poses/poses.json (anchors, tags), poses/contact.png
  choreo/<name>.yaml              the choreography (hand-written); a film is one file that includes segment files
  renders/                        videos, steps logs, stills, contact sheets, maps (written)
  .cache/                         per-tile occluder and paint masks, the low-frequency light map, parallax layers
                                  (regenerable, small, safe to delete)
```

Nothing holds the whole street in memory: tiles are read by windows (a few tiles at a time), the light maps are
computed per window of about 4,600 px from the tiles plus a 1/8 overview, masks are cached per tile by content hash,
and the paint history makes each pose's matte on demand.

## Workflow

1. **Plates.** Make the first plate with `aimv board` (text to image, 21:9 4k, the look reference). For each next
   plate: `aimv paint street <p> --canvas P2` writes `canvas-P2.png` next to the last plate (its right 40% on the
   left, flat grey elsewhere); a board frame with that canvas as its only reference and a prompt that keeps the left
   part and continues the street into the grey (name the obstacles you want: AC unit, low ledge, drainpipe, fire
   escape with ladder, lit doorway, long render wall, shop window). The model keeps the overlap within about 1 px.
   Plates can also be prepended on the left (a street that starts earlier): `aimv paint street <p> --canvas L1 --left`
   writes `canvas-L1.png` next to the leftmost plate (its left 40% on the right, grey elsewhere); outpaint leftward.
2. **Street.** List the plates in `street.json`, run `aimv paint street`. Check `seams.jpg`. Add game-level edits
   (cut a span, insert a mirrored or stretched copy) and run it again. Appending plates only rewrites the tiles right
   of the new plate's overlap; earlier tiles keep their hashes, so their caches stay valid. Prepending (`left`) never
   touches anything right of x 0: street x 0 stays the first plate's left edge, the new street has negative x, and
   obstacles, choreographies, steps logs and caches of the old street stay valid. To add an obstacle to a stretch of
   street (a bench, bollards): `aimv paint window <p> --from -9750 --to -7450 --out patches/bench.png` writes the
   window at full resolution and full height with its placement (`bench.json`, holding the edit line); an image edit
   adds the object; save it as `bench-edit.png`, add the edit line to `edits` and run `aimv paint street`.
3. **Obstacles.** Mark boxes and contacts in `obstacles.json` (street px; open a tile in an image viewer for
   coordinates), run `aimv paint scene` and look at `scene-masks.jpg`: paint must go behind every real object.
4. **Poses.** Sprite sheets via `aimv board` (black emulsion poses on plain primer, plain shoes, clear hand and foot
   contacts, explicit cycles). List them in `poses.spec.json`, map roles, run `aimv paint poses`, check `contact.png`.
5. **Choreography.** Write `choreo/<name>.yaml` (below). `aimv paint check` until it has no errors, then
   `aimv paint map` to see every pose on the street.
6. **Preview.** `aimv paint render <p> <choreo> --preview --from 59 --to 62` (half size, seconds), or
   `--only 40,41,42` (just those photos, kept as JPEGs), or `aimv paint still --at 44.2.1`.
7. **Render.** `aimv paint render <p> <choreo>` renders the choreography's window (or `--from/--to`) in 16:9
   (1920x1080) and 9:16 (1080x1920) with the song slice, then `aimv paint contact` for a labelled contact sheet.

## Street spec (`street.json`)

```json
{"version": 1,
 "plates": ["boards/07-street/P1.png", "boards/07-street/P2.png"],
 "left": ["boards/09-opening/L1.png", "boards/09-opening/L2.png"],
 "keep": 0.4,
 "seam_feather": 24, "gain_fade": 1400,
 "edits": [{"cut": [5203, 6100]},
           {"insert": 11100, "src": [10400, 11800], "mirror": true, "stretch": 1.5},
           {"insert": -6100, "src": [-7650, -6100], "mirror": true}],
 "edit_feather": 96,
 "tile": 2048}
```

| Key | Meaning |
|---|---|
| `plates` | project-relative paths, left to right; every plate after the first was outpainted from the previous one |
| `left` | plates prepended on the left, nearest first; each outpainted leftward from the plate to its right (its left `keep` on the canvas's right). They get negative street x. Every join is an ordinary seam in the middle of its overlap, so the first left plate's seam lies a few hundred px right of x 0 (away from the first plate's own left border): the columns it changes there are kept as a join patch (`strip/join-0.png`) that street tile 0 is rendered with; the strip tiles right of x 0 are never rewritten, and the street tiles as the right side's plates made them stay in `tiles/pure/` for its light maps. Stitched in a mirrored frame, where prepending is appending |
| `keep` | the fraction of each plate taken from the previous one (the canvas); the overlap SIFT matches in |
| `seam_feather`, `gain_fade` | stitching: per join a SIFT + RANSAC similarity, a low-frequency gain match faded out over `gain_fade` px of the new plate, a minimum-difference seam in the middle of the overlap feathered over `seam_feather` px |
| `edits` | in strip px (the stitched strip before edits). `cut: [x0, x1]` removes the columns; `insert: x` puts a copy of `src: [x0, x1]` there, optionally `mirror`ed and `stretch`ed horizontally. Every join is cross-faded over `edit_feather` px. Edits left of x 0 (negative strip px) lay their pieces out leftward from x 0, so they move only the street further left; an edit may not straddle x 0 |
| `edits` (patches) | `{"patch": "paint/streets/main/patches/bench-edit.png", "window": [-9750, -7450]}` pastes an edited copy of a street window back in place, in street px (after the cuts and inserts, so nothing moves). It is aligned to the street by SIFT (a similarity: a resized or shifted edit, a 4:3 crop upscaled, still fits; `build.json` records the inliers, the residual and the scale), its tone matched with a low-frequency gain from the pixels the edit left alone, and pasted only where it differs (`mask: diff`, the default: changed pixels over `thresh` 28, opened, closed, small specks dropped, grown by `grow` 24 px and feathered; `mask: full` pastes the whole window, feathered at its border). `region: [x0, y0, x1, y1]` (street px) limits what it may change: an edit model also nudges sills, cornices and rain, so give the object's box. `"align": "none"` places it as it is, without SIFT (it must be its window's size): for an exact edit of the built street, which SIFT could only match against the unpatched street under it (the patches are aligned against that), or a generated patch made on a canvas of the built street. `build.json` records the box it changed (`changed`) |
| `edge_cover` | the far left end keeps the columns where at least this share of the rows is covered (0.98); lower it when a far plate shrank through the joins (its missing top and bottom rows are filled from the nearest covered row) |
| `tile` | tile width (2048) |

Obstacle coordinates are street px (after edits). `aimv paint street` prints the plate offsets and seams; the
`pieces` in `build.json` give the strip-to-street mapping (`street.street_x(d, strip_x)` on either side of x 0).
Light maps right of x 0 are computed from the right side's own tiles (without the join patch: the same values and
cache as before any plate was prepended); the street left of x 0 gets its own, blended into the right side's edge
values over 2048 px. So prepending changes the old street only between x 0 and the first left seam.

## Obstacle registry (`obstacles.json`)

```json
{"version": 1,
 "ground": {"wall_base": 1212, "feet": 1196, "kerb_top": 1370, "kerb_bot": 1420,
            "dy": [[-2600, -20], [-1400, -12], [-500, -9], [0, 0]]},
 "obstacles": {
   "ac": {"kind": "ac", "box": [4246, 765, 4499, 947], "contacts": {"top": {"y": 765}},
          "occlude": [{"box": [4240, 758, 4506, 953], "method": "rect"},
                      {"box": [4262, 945, 4300, 992], "method": "dark"}], "note": "AC unit, top 1.75 m"},
   "pipeB": {"kind": "pipe", "box": [5575, 0, 5630, 1212], "contacts": {"pole": {"x": 5602}}, "occlude": "dark"},
   "alley": {"kind": "gap", "box": [1960, 0, 2700, 1212], "glow": [1.0, 0.55, 0.22],
             "contacts": {"near": {"x": 1960}, "far": {"x": 2700}}}},
 "no_paint": [{"id": "shop glass", "box": [13126, 0, 14320, 1643]}],
 "layers": [{"id": "redlight", "kind": "pole", "parallax": 0.12, "x": 2002, "reach": 140,
             "parts": [{"box": [1965, 340, 2040, 1310], "method": "dark", "vmax": 0.4},
                       {"box": [2018, 430, 2142, 642], "method": "rect"}]}],
 "lights": [{"id": "red", "x": 2080, "y": 535, "sigma": 420, "strength": 0.6, "color": [1.0, 0.07, 0.05]}],
 "look": {"tint_gain": [1.0, 1.0, 1.0]}}
```

- `ground`: `wall_base` (wall meets pavement), `feet` (the line painted feet stand on, just above it), the kerb.
  `dy` (optional): the ground line's offset along x, piecewise linear and constant beyond its ends (outpainted plates
  drift a little in scale, so their ground is not level). Each move stands on the ground where it happens, the
  paintable wall ends at the offset ground line, and the camera and reflections follow it.
- `kind`: edge, gap, ledge, pipe, beam, ac, step, ladder, door, glass, wall, pole, box, sign, shutter, window, other.
  A `gap` is never paintable and gives the J2 air layer its glow (`glow`, linear RGB); with `"wall": true` (v8: the
  gaps between P2b's awnings and from deck 2's end to A1, the brick going on behind them) it is a gap only for the
  route (no ground there): paint goes on the wall behind it and it is no void.
- `contacts`: lines or points the runner touches. `{"y": 765}` is horizontal (x0, x1 default to the box),
  `{"x": 5602}` vertical (y0, y1 default to the box), `{"pt": [x, y]}` a point. Names the templates look for:
  `edge` (take-off edge), `near`/`far` (a gap's sides), `top` (plant hands, stand on), `under` (the underside to
  slide under), `pole` (a pipe to climb or swing on), `beam` or `bar` (to grab and hang), `step` (to land on).
- `solid` (default by kind: box, cabinet, ac, bin, bench, step, ledge, door and sign are solid; pipe, pole, ladder,
  railing, beam, wire and shutter are thin): no part of the figure may be hidden behind a solid obstacle (unless the
  move says `behind: <obstacle>`), and within its occluders he must be above its top (hands and feet may rest in the
  top 14 px). Paint may pass behind thin ones. An occluder part can say `"solid": false` (an AC's brackets, a
  meter's conduit, a door lamp).
- `occlude`: the parts of the object in front of the wall (paint goes behind them): `rect` (the whole box), `dark`
  (black iron inside the box: dark and not warm, `vmax` 0.30), `mask` (a PNG the size of the box, white in front,
  path relative to the street folder). A string (`"occlude": "dark"`) uses the obstacle's box.
- `no_paint`: boxes where paint never goes (glass, above a railing, the pavement is never paintable anyway).
- `glass`: `[{id, box}]` shop window glass he may paint over (Full Stop's verse 2: the windows full of glowing
  screens). Paint on glass is a black cut-out against what is behind it: no primer patch under a pose, no paint-out,
  no cover, no wall texture or sheen, no light's glow on it, and a pose leaves no ghost there (only the pose of its
  own photo shows, so he moves across the glass as a silhouette), the brush's dry gaps closed up (`GLASS_SOLID`: a
  solid black cut-out against the faces); the screens' light glows softly round his edges (`render.GLASS_GLOW`:
  spread, strength, light blur). A window whose screens are off (`screens`, its group id the glass's id) is dead
  glass: what he paints there after they died takes primer and wall shading as on the wall, so the black figure
  still reads on the dark glass (the marks painted while it was lit stay as they were).
- `screens`: `{plates: {plate: off plate}, groups: [{id, x: [x0, x1]}], top, feather, dir}`: windows whose screens
  can be switched off (the choreography's `screens` move). Each plate has an `off` twin, the same photograph with
  every screen dead (no faces, no glow on the pavement or in the road). `python -m aimv.paint.screens <project>`
  builds `<dir>/<id>.png` per group: the street's columns of the group's band with the screens off, each pixel read
  from the off plate the strip took it from (the plate whose on version matches the street there best, warped by the
  strip's own transforms in build.json, gain-matched to the strip), and `<id>-light.png`, the off street's light over
  the on street's (smoothed). From a group's time the renderer blends its band in, from `top` down, feathered over
  `feather` px at its sides and top, and the paint's light there drops with it (the stencils on the pillars lose
  their glow and contrast). Off plates aligned with their on plates within half a pixel (as V1-off to V4-off are)
  need nothing else.
- `layers`: objects well in front of the wall that move with parallax (the street behind them is rebuilt row by
  row: the wall's own texture either side mirrored into the gap and cross-faded, so the part the shift uncovers
  reads as wall, never as a flat streaky panel); `reach` is the most they shift (px). A parallax-0 layer never moves
  against the plate: nothing is rebuilt behind it (the street behind is the plate itself).
- `lights`: coloured glows on the paint (the red pedestrian light). `id` names one for a flicker (a template's
  light hint dims it on a photo: the strikes in the box, the explosion) or a knock-out (dark from a photo on: the
  leap's `knock`), `layer` names the parallax layer holding its lamp, whose lit (red and pink) pixels dim with it;
  a dead lamp loses all its lit pixels (inside its layer's `rect` parts everything brighter than the casing becomes a
  dark lens), its glowing figure too, and its red bloom on the wall round it. `paint` scales its glow on the paint,
  `neutral` (0 to 1) takes the plate's light colour out of the paint round it. A red light (red over three times its
  green and blue; a light with no `color` is red) defaults to `paint: 0, neutral: 1` (the director: the white paint looked
  pink by the alley's light, then by the box's: every red light in the film), so its glow stays on the plate and the
  lamp; any other light defaults to `paint: 1, neutral: 0` (`render.light_opt`). Set either to override.
- `look.tint_gain`: a colour cast on the light-colour map. Prototype 6 had about [0.97, 1.10, 1.0] by accident (a
  float32 mean over the street), which made primer read mint-grey; the default is neutral.

## The world

The street can grow in two dimensions: plates above it, below it and to its right (cells), for a film that leaves
street level (Full Stop's climb, rooftops, fall and gaps). The 1D
street is the special case: without a `world` section nothing changes, and with one the street left of the world is
read, masked, lit and framed exactly as before (byte-identical renders).

Coordinates stay street px: x as on the street, y down from the street's top row, so everything above the street has
negative y (Full Stop's roofs are near y -1600). Obstacles, contacts, levels and voids in the world use them. Cells
are given in anchor px: the anchor plate's own pixels, its top-left (0, 0), the grid the cells were outpainted on.

### The spec

```json
"world": {"anchor": "boards/11-verse2/V4.png",
          "cells": [{"id": "T1", "path": "boards/18-world/T1.png", "at": [2304, 0]},
                    {"id": "T2u", "path": "boards/18-world/T2u.png", "at": [4608, -1501], "from": ["T2", "T1"]}],
          "seam_feather": 24, "gain_fade": 1400}
```

| Key | Meaning |
|---|---|
| `anchor` | one of `plates` (right of x 0); its strip transform, the strip's crop and the street edits (a cut before it moves it left with the street) map anchor px to street px. Its left half must not be cut, mirrored or stretched |
| `cells` | in the order they were made; each needs `path` and `at` (its nominal top-left, anchor px); `id` defaults to the file stem |
| `from` | the cells (or the anchor, by its file stem) it was outpainted from; default every placed cell and the anchor its grid box overlaps |
| `place` | `sift` (default) or `nominal` (no feature match: the prediction below as it is, for a cell with nothing to match) |
| `floor` | a street row (0): a cell above the street, over its left part (left of the world). The street is its neighbour below: it is placed by SIFT on the street's rows it overlaps from 64 rows under its floor (a street patch may feather its seam into the rows above that; the placement stays put when that patch is redone), its tone matched to them along the floor, and it writes nothing at or below the floor (the street keeps its own rows; the seam is the floor itself). world.json then has `bands` (the floor cells' boxes, grouped, down to their floor) and `xw` (the world's first column outside them): the street left of `xw` keeps its 1D paths, and each band gets its own light maps. Full Stop: P2b's U cells over the brick building, above the scaffold (boards/33-p2b) |
| `seam_feather`, `gain_fade` | as for the strip (default the street's own) |
| `sky` | regions where the world's top edge is too low for him: `[{"x": [x0, x1], "up": 320, "src": 150, "taper": [400, 1800], "borrow": [[from_x, dx]], "paint_out": [[x0, x1, y_low]]}]`, the open sky continued `up` px above the edge (the sky step below) |
| `props`, `props_kit` | foreground props on his line: `[{"id": "acR1", "kit": "ac-low-a", "x": 30750, "base": -1602, "flip": false, "shadow": 0.65, "front_rails": false, "front_cols": [[x0, x1]], "highlights": 1.0, "warm": 0}]`, cut-outs from `props_kit` (default boards/26-props/cut, with its props.json), drawn by the prop step below. `"floor": 0` for a prop over the street's left part that stands on the street's rows (as the floor cells): it is graded to the cells and the street behind it and writes nothing at or below its floor; its foot there (with its contact line) goes into the street as a patch: `world.prop_foot(street_dir, project, id)` gives the composite and the foot's mask over the same plate, the same grading |
| `inserts` | `[{"x": 23500, "length": 2300, "patches": [{"path": "boards/27-street-v7/O5.png", "window": [x0, y0, x1, y1], "region": [x0, y0, x1, y1]}]}]`: more world at x (where the cells put it): everything right of it moves `length` px right, in every row; the insert step below. `"clean": true` keeps the placeholder out of its patches. A negative `length` is a cut: [x, x - length) leaves the world and everything right of it moves left; its patches make the seam. A patch's `"align": "none"` places it as it is (it must be its window's size) and `"tone": false` keeps its own tone: for plates made on a canvas of the world after the insert whose window holds nothing of the world under it (a cut whose patches rebuild a whole stretch) |
| `grade` | warm glows lifted out of the sky's own clouds: `[{"x": [x0, x1], "y": [y0, y1], "falloff": 600, "below": 600, "luma": 30, "color": [255, 125, 55], "texture": 1.0}]`, the core box's mean 8-bit luma brought to `luma` (the grade step below) |

### The build

`aimv paint street` builds the world after the street (`world.py`):

- **Prediction.** A cell is predicted from the placed neighbour its grid box overlaps most: its grid offset from that
  neighbour applied to the neighbour's placement, so the drift of the cells before it is carried along. Full Stop's
  cells drift up to 330 px off the grid by the last column; off the neighbour that predicts them, at most 6 px.
- **Placement.** SIFT + RANSAC similarity against the world composed so far (the street and the cells before it),
  in the cell's overlap with its `from` neighbours, matches gated to 80 px of the prediction; refused (with the
  reason) when the match is more than 80 px off, scaled by more than 2% or turned by more than 0.75 deg.
- **Seams.** One minimum-difference seam per side, in the middle of that side's overlap: vertical with a neighbour
  on the left or right, horizontal with one above or below; an L of neighbours gets both, and the cell wins only past
  every seam. Feathered over `seam_feather` px. A neighbour that only touches a corner joins the band of the side it
  touches most.
- **Gain.** Per side a low-frequency gain along the seam (the median ratio across the middle of the overlap), faded
  out over `gain_fade` px into the cell; where two sides meet, their gains are averaged by their fades.
- **Sky.** After the cells, each `sky` region continues the open sky above the world's top edge: every column repeats its own band of `src` rows under the edge upward (a shift, so the rain keeps its slant), up to `up` px, tapered to nothing over `taper` [left, right] px at the region's ends (the camera's frame top eases down there, never steps), the first `seam` (40) rows cross-faded from a reflection of the edge, `darken` (0.18) going up. `paint_out` paints structures out of the source band first (an antenna's tip), `borrow` takes the band from `dx` px away from street x `from_x` on (where far buildings reach into it). Its pixels have alpha 254, so the step is cleared and redone whenever the cells or the regions change, without placing the cells again. Full Stop: N5, the tallest roof, ended 50 px above his head (`up` 320 over x 55296 to 63488).
- **Inserts.** After the cells, before the sky: at each insert's x the overlay opens `length` px in every row, everything right of it moving right (no cell is placed again, nothing appears twice), the gap filled with the world left of it mirrored (a placeholder) and then with each patch: an image of the world round the gap (the post-insert world, its middle new: a canvas with real world either side and the middle grey), aligned by SIFT on its known parts and pasted in `region` like the street's patches (mask full). A `clean` insert's patches take nothing from the placeholder: their tone match uses only the real world in their window (and the patches before them), and over the placeholder they paste whole up to their region's edge, feathered only into real world or an earlier patch (without it the tone follows the mirrored placeholder, and a region's feathered edge lets it show through: mirrored windows and tone steps in a sky). A cut (a negative length) takes the columns [x, x - length) out of every row and moves everything right of them left (no placeholder: its patches, made from a canvas of the world after the cut with a grey band over the seam, make the seam clean). A patch with `align: none` is placed at its window as it is, and one with `tone: false` takes no gain from the world under it: SIFT can only match stray points in a window that is all new (v8 P5's cells over the cut: 6 to 12 inliers and a 1% scale), and the tone match would pull a new stretch toward whatever the cut left under it. The cells' placements in world.json stay where the cells put them; `grid_to_street` and `street_to_grid` add the inserts and cuts (`insert_shift`), and the sky regions, the props and obstacles.json are in the coordinates after them. Any change to the inserts places the world again (they are in the params). Full Stop: 2,300 px in T1's pale wall at 23,500 for segment 4's real-speed run (the solver asked for 2,100); on the v8 street, 4,600 px at 69,870 for P4's drop between buildings (`clean`, four plates), and a 6,770 px cut at 41,700 for P5 (chorus 2's roofs rebuilt shorter: six plates, `align: none`, `tone: false`).
- **Grade.** After the inserts, before the sky: each `grade` region screens the cells' own pixels toward `color`, weight 1 in its core box falling off smoothly to 0 over `falloff` px outside it (`below` px under it), strongest at the core's foot (the underside of a cloud bank lit by the city below, half as strong at its top) and following the clouds' texture (`texture` 1: cloud detail lighter than its surroundings takes more, the dark gaps less) and lifting what is darker than `luma` fully and nothing brighter than 1.4 times it (the horizon band below stays), with the strength whose mean 8-bit luma over the core is `luma`. It changes the cells' pixels, so the grade is in the params: any change places the world again; a props or sky build leaves it alone. Full Stop: the big leap's flip against the dark upper sky (x 122,800 to 124,200, y -3,100 to -2,800: luma 14 to 31 in M15s and L1s, the city's light on the underside of a lower cloud bank).
- **Props.** Last, each prop in `props` is drawn into the overlay: its kit cut-out (RGBA, keyed off chroma green, trimmed; props.json gives its real height at 250 px/m, its base row and its top surface) at its real size, its base row on `base` and its left edge at `x`. It is graded to the plate behind it (in Lab: black level and median to the plate's, the spread x1.25 so it reads in front, the plate's colour cast, at most 0.8 of its own chroma, a 5% veil), with the plate's grain, rain streaks at the plates' angle, a wet glint on its top silhouette, ambient occlusion at its foot and a soft dark contact line on the surface along its base (`shadow`). `front_rails` keeps a guard rail in front of it (rows whose median is a thin bright line), `front_cols` keeps street columns in front (tubes). `highlights` (1) pulls its highlights down: its lightness times this at its own 95th percentile and above, eased from 1 at its median (0.9: 10% down, for a pale unit that comes out too bright and clean). `warm` (0) adds a cast toward the city's sodium light on top of the plate's (at 1: a* +3, b* +8 from its median lightness up, eased in from its black level), for a prop nearer the warm lights below. Its pixels have alpha 253; what was under them is kept in world/props/under.npz, so the step is undone and redone whenever the cells, the sky or the props change. world.json `placed_props` has each one's silhouette box, top surface and matte (world/props/<id>-mask.png): obstacles.json registers them (kind, box, contacts.top, vault, step, prop; an occluder mask where paint must pass behind it).
- **Storage.** An overlay, never a rewrite: `strip/` and `tiles/` stay as the street build made them. `world/` holds
  RGBA tiles `w{ix}_{iy}.png` on the street's own tile grid (tile (ix, iy) covers x [ix * tile, (ix + 1) * tile) and
  y [iy * tile, (iy + 1) * tile)), alpha 255 where the world defines the pixel (the cells, and the street pixels a
  seam changed: the anchor's overlap with the first cells); `world.json` (the anchor's map, every cell's placement,
  its seams, the tile hashes, the bounds); `valid8.png` (street and world at 1/8, 255 where a block is all valid).
  `build.json` notes the world (`world.tiles`, a hash of the tile hashes).
- **Incremental.** Cells added at the end of an unchanged list are placed on top of the last build and only rewrite
  the tiles they reach; other tiles keep their hashes (and their scene caches). A changed cell, anchor or street
  under the world rebuilds it.
- **Sheets.** `world.jpg` (the world at 1/8: every cell's box, every seam in red) and `world-seams.jpg` (every seam at
  full resolution, 600 px crops labelled with the cell, the side and its neighbours). The log gives per cell its
  place, how far it is off the grid and off its predicting neighbour, its scale, inliers and seams.

The strip's right end: a long chain of plates drifts (Full Stop's V4 sits about 11 px higher than P1 and is turned
0.08 deg), so the last plate may not cover every row of the strip's crop. The strip then ends where `edge_cover` of
the rows are covered (as on the left), the missing rows filled from the nearest covered row.

### Canvas for a new cell

`aimv paint street <p> --canvas G11 --rect X0 Y0 X1 Y1 [--out F]` writes the outpaint canvas for a cell at
`at: [X0, Y0]` of size (X1 - X0) x (Y1 - Y0) (anchor px, the cells' grid): every known pixel of the street and the
world in it, resampled into the frame the build will predict for that cell (from the placed cell it overlaps most),
flat grey (128) elsewhere. It prints how much is known and on which sides. Default location: next to the last cell.
A board frame with that canvas as its only reference then continues the world into the grey.

### Reading the world

- `Street.window(x0, x1, y0, y1, valid=False, fill=True)`: any rows; the street (rows 0 to H) and the overlay.
  Pixels neither holds are invalid: filled from the nearest valid pixel (for blurs) unless `fill=False`, and
  `valid=True` also returns the mask. Without rows, a window the world does not reach is the plain street's (edge
  replicated past its ends, as always). `Street.bounds` (x0, y0, x1, y1), `Y0`, `Y1`, `xw` (the world's first
  column), `valid_box`, `overview2d(scale, box)`.
- `Scene.span(x0, x1, y0, y1)`: the maps over any rows (`Span.y0`, `y1`); `Scene.masks(x0, x1, y0, y1)`. Street
  tiles that end (with their 16 px mask margin) left of the world keep their own masks; everything else uses 2D mask
  tiles (`occ2-*`, `paint2-*` in the cache), made from the street and the world as they are there: paintable is valid,
  above the pavement line, outside the no-paint zones and the voids.
- Light maps: the street's own (1D) maps never see the world. The world gets 2D maps at 1/8 (`lfw-*.npz`): loc and
  col as on the street but normalized over the valid pixels, level a geometric mean of the wall's luminance (sigma
  900 px along x, 600 along y), norm the tint's wide average (sigma 3000 by 1500). They blend into the street's maps
  over the 2048 px left of the world's first column; rows above and below the street are the world's.
- World cells above the street's left part (v8, P2b's scaffold): world.py's `floor` cells write nothing in the
  street's rows, and world.json then says where the world really starts (`xw`, counting only cells without a floor)
  and where those cells are (`bands`: one box per group). Left of `xw` a box inside the street's rows is still the
  plain street (its 1D path, its own maps), so the cold open and the box are unchanged; a box reaching above the
  street there takes the 2D path (the street and the cells), and each band has its own 2D light maps
  (`Scene.lf_band`, `lfb*-*.npz`, the band a little wider each side, rows from the world's top to the street's foot)
  in place of the world's maps where those do not reach; its rows above the street take the world's mask tiles
  (`Scene.masks`: paint goes on the cells, never beside them or above them), so his paint shows on the brick above
  the scaffold. Without the keys nothing changes.
- Parallax layers whose parts reach above or below the street hold the world's rows round them (`L["y0"]`;
  `scene.layer_rows`, `scene.rows_of` read them by street y).
- Cache files are written atomically (render workers share the cache).

### Grounds per level

```json
"ground": {"wall_base": 1212, "feet": 1196, "dy": [...],
           "levels": [{"id": "T street", "x": [21690, 30975], "feet": [[21690, 1196], [21950, 1182], ...], "street": true},
                      {"id": "walkway", "x": [24532, 27799], "feet": [[24532, 190], [27799, 186]]},
                      {"id": "roofs", "x": [28950, 45960], "feet": [[28668, -1493], ..., [47143, -1558]]}]}
```

- `feet_at(x)` is the level holding x (the last one listed wins where two overlap; `feet` a y, or a polyline
  [[x, y], ...] linear between its points), else the street's feet plus `dy`: a step at each level's end.
  `level_at(x)` names it ("street" outside every level).
- Everything that stands on the ground goes through `feet_at`: a move's `c.feet` (at the move's x), motion's
  landings, the camera's ground framing and its vertical follow. A run on the walkway stands on the walkway.
- `wall_base_at(x)` stays the street's pavement (wall base plus `dy`): the wet-pavement reflection, puddles, the
  paintable wall's bottom, droplets landing, the QA ground rule. The wall below a raised level is still wall.
- `"street": true` makes a level the street's own ground there: the world's street level, drifted with its cells
  (Full Stop's last street cells sit more than 100 px above the 1D street's feet line). It sets `ground_dy` over its
  x range, so the pavement (wall base, kerb, reflections, the paintable bottom) moves with its feet line.
- A level's x range is where he is, not where its object is: a move stands on the level at its planned middle, so a
  climb from one level to the next is split where its clips' middles fall (Full Stop: the walkway starts at the wall
  run's tube, the ledge holds the climb clip, the roofs start at the pull-over).
- `aimv paint scene` with levels writes `levels.jpg` (each level's stretch of world, its feet line green, the
  strongest horizontal edge measured near it every 64 px red: a parapet's top, a walkway's boards) and `levels.json`
  (the measured lines, to fit the feet polylines to). `Scene.measure_line(x0, x1, y_at)` measures any line.

### Voids

Where only sky or a void is behind him (between rooftop structures, a gap between roofs, the open space at a wide
gap) he is the J2 skin, never paint. `voids` in obstacles.json mark them:

```json
"voids": [{"id": "sky", "box": [27100, -2700, 45600, -1610], "method": "sky"},
          {"id": "gap2", "box": [45600, -2700, 46100, 1650]},
          {"id": "lot", "box": [...], "method": "poly", "poly": [[x, y], ...]}]
```

Methods: `rect` (the box, the default), `poly` (a polygon in street px), `mask` (a PNG the size of the box, white =
void), `sky` (inside the box, the smooth pixels joined to its top edge: open sky and low-textured far buildings;
`smooth` 0.02, `vmax` 0.5). `except: [[x0, y0, x1, y1], ...]` cuts boxes out of a void: the structures standing in
it (a stair house, a water tank, a billboard back) are walls he is painted on. `Scene.void(x0, x1, y0, y1)` is the
voids, the gaps and every invalid pixel; voids are never paintable. scene-masks.jpg shows them purple.

The skin over voids (render.py): a wall pose in the world (right of its first column, or left of it reaching above
the street, by a band's cells; not a take-off photo) with
more than `SKIN_VOID` (0.25) of its paint over `Scene.void` is drawn as the J2 skin instead: the event becomes an
air event (`skin`, `ghosts` 3: a trail of pale skins), no primer patch, no paint-out, no droplets, a thin soft pale
rim on its outline (`SKIN_RIM`: 6 px, the city's glow catching the paint's edge, so he reads against a dark skyline;
never round the pale gaps inside him), and a soft shadow (`SKIN_SHADOW`: 22, 30 px down and right, blur 9, 0.38) on
the wall behind it, never on the void. Skin and paint swap with hysteresis (`skin_hysteresis`): only where the new
background holds `SKIN_HOLD` (3) photos, and the primer patches of the `SKIN_FADE` (2) paint photos next to a skin
stretch are cross-faded (a third, two thirds), so running along a roof past structures does not flicker. A skin held
over photos with no new figure (a held clip frame, a slow-down) stays, as paint would. QA's wall rules skip skins.
A move's own air events with `skin` (and `ghosts`, `glow`) draw the same way.

Lamps in the plate that blink (the roofs' red warning lights): a small parallax-0 layer round each lamp and a light
naming it (`layer`); the `blink` template dims the light and its lamp's red pixels on the music. In a light's
parallax-0 layer each `rect` part (the lamp head) is a soft round falloff from the box's centre (full within 40% of
its radius, gone at the box's edge), so the lamp and its glow dim in a round soft area, never a square.

### Maps and checks in 2D

`aimv paint scene` (scene-masks.jpg) and `aimv paint map` draw the street and the world in two dimensions when there
is a world: invalid areas dark, y marks, the levels' feet lines (cyan), the voids (purple); scene-masks.jpg is
wrapped in rows of 5,000 px of street, each cut to the rows it has.

The camera never shows an invalid pixel: `Scene.clamp_crop(cx, cy, cw, ch)` moves a crop's centre as little as
possible so the crop (30 px to spare) is all street or world. On the plain street it is the street's own rule (each
axis clamped into the street); in the world the nearest valid centre within 4,000 px (valid8 at 1/8, conservative;
exact on the street), then moved back toward the asked centre along the line and each axis while it stays valid.
A crop wider or taller than any valid area there (a wide frame over a gap whose cells do not reach that far) stays
where it was asked, inside the world's bounds, and shows the nearest valid pixels where it has none: keep wide
frames inside the cells (`aimv paint street` writes world.jpg to check them against).
`Scene.span_rows(y0, y1)` gives the rows a renderer's span should hold (the street's 0 to H while the crop stays in
them).

The renderer, camera, QA and check read the world through these (render.py: spans by rows, crop rows clamped to
`Y0`/`Y1`, layers by their rows; camera.py: `clamp_crop`; qa.py and check.py: masks with rows, `inside`, layers by
their rows, the world's right end in the street reach).

Coordinates on the cells' grid (anchor px, as `at` and `boards/18-world/make.py`, what clip frames are drawn on) and
street px convert through the cell that holds the point, its drift included: `world.grid_to_street(winfo, u, v)`
and `world.street_to_grid(winfo, x, y)` (winfo: world/world.json). Full Stop's cells drift up to 330 px off the grid
by the last column, so a grid number is never a street number.

## Pose library (`poses.spec.json`)

```json
{"version": 1, "ref_height": 450, "reference": "R1",
 "sheets": {"R1": {"path": "boards/05-sprites/R1.png", "n": 8, "cols": 4, "kind": "pose", "height": 670},
            "PK1": {"path": "boards/08-parkour/PK1.png", "n": 6, "cols": 3, "kind": "pose", "boost": [1.12, 0.72]},
            "SP1": {"path": "boards/08-parkour/SP1.png", "n": 6, "cols": 3, "kind": "stamp"}},
 "cycles": {"run": [["R1-1", "R1-3", "R1-7", "R1-4"]], "walk": [["PK4-5"], ["PK4-6"]]},
 "phases": {"run": ["contact", "passing", "toeoff", "flight"]},
 "roles": {"crouch": "K1-1", "flight": ["PK1-2", "PK1-3", "PK1-4", "PK1-4"]},
 "tags": {"PK1-1": {"move": "leap", "phase": "takeoff"}}}
```

A sheet or a video entry may say `"screen": "green"`: drawings or a clip shot on flat chroma green (Full Stop's true
figure, review note: boards/23-true-sheet and its clips). They are keyed by `greenkey.screen_matte` (the CrCb
distance to the screen's own green, its shadows guarded, the colour despilled) instead of the paint density, a
clip's guide shapes are painted with the screen's colour so the key takes them out too, and the build writes each
drawing's despilled colour crop as `<pose>.rgb.png` beside its matte `<pose>.png`; the screen's colour is the
border median without the guide shapes (`screen_colour(img, exclude=...)`: a roof block along the bottom and up a
side can be most of the border) (a clip's `rescale` scales it
with the matte); `truelook.py` composites him from those. Their poses carry `"screen"` in poses.json. A `true_insert`
prop (truelook.py: a full-frame clip in place of a stretch's photos, one per aspect) is taken first in
`Renderer.photo`: inside its stretch each photo is the insert's frame at its time, cropped from the insert's locked
camera to the photo's own camera, with the photo's rain over it (the MUTO look follows as for every photo); outside
it, and for the other aspect, the photo renders as usual. A true figure (inside a `true` prop's stretch, `truelook.is_true`) is solid,
not paint, and in front of the whole world: `Renderer.photo` composites him after the plate and everything on it
(the parallax layers with their lamps, blinks and knocked-out lights, the lit windows, signs, screens and glass),
so only the photo's rain, haze and grain go over him. He is not paint: QA skips its solid, head and no_paint rules and `check` its "lands where paint cannot go" warning (the
off-the-street warning, ground, frame and backtrack still apply).

Poses are named `<sheet>-<k>` (k along rows from the top left). The matte is the paint density in linear light (1
minus pose/primer reflectance), so brush texture stays partial. `height` (standing figure height in sheet px) sets a
sheet's scale to `ref_height`; sheets without one match the `reference` sheet's median painted area. `boost` (gain,
gamma) evens out dry-brushed sheets. `ref_height` is the body height in street px: template geometry is written for
450 px and scales with it. Anchors: `foot` (torso x, lowest foot), `bottom`, `top`, `left`, `right`, `br` (bottom
right: hands planted), `center` (bbox centre), `cxc` (torso x, bbox centre y), `bc` (bbox centre x, its bottom: a
front view standing or sitting on a floor). Cycles and phases tag the run
poses (the foot lock reads the phases); roles are what the templates ask for:

| Role | Used by | Full Stop |
|---|---|---|
| `crouch`, `brake_in` (2) | crouch | K1-1; K1-6, K1-7 |
| `brake` | brake | PK4-4 |
| `takeoff`, `flight` (list), `land`, `recover` (2) | leap, step_up, swing_release, pole_swing | PK1-1; PK1-2, PK1-3, PK1-4, PK1-4; PK1-5; PK1-6, R2-3 |
| `jump`, `land_small`, `recover_small`, `land_vault` | vault, climb, step_up, pole_swing | K1-5, K1-6, K1-8, K1-7 |
| `reach`, `vault_approach`, `vault_plant`, `vault_over`, `drop` | vault, climb | PK3-1, PK2-1, PK2-2, PK2-3, PK2-4 |
| `slide_in`, `slide`, `slide_out`, `rise` (2) | slide | PK4-1, PK4-2, PK4-3; K1-3, K1-4 |
| `climb` (2), `hang`, `swing`, `release` | climb, hang, swing_release, pole_swing | PK3-2, PK3-3; PK3-4; PK3-5; PK3-6 |
| `splash`, `burst`, `smear`, `spray`, `drips`, `droplets` (stamps) | leap, vault, swing_release, burst, leap_off | SP1-1 to SP1-6 |
| `curled`, `breathe` (2), `head_up`, `curled_tight`, `hand_mouth`, `stand_bent`, `press_sides`, `look`, `brace` | inside (any role works there; demo v3 plays the clips box_XC, box_XM, box_XS, box_XG instead) | BX1-1, BX1-1/BX1-3, BX1-2, BX1-3, BX1-4, BX1-5, BX1-6, BX2-4, BX2-1 |
| `windup`, `strike` | pound | BX2-2, BX2-3 |
| `break_out`, `break_land` | burst | BX2-5, BX2-6 |

Full Stop's runner sheets wear the canon shoes (dark running shoes with a curved white flash, white midsole and laces,
short white socks): PK1 to PK4 and BX1, BX2 point at the shoe-only edits in `boards/12-canon`; the old paths are kept
in `poses.spec.json` under `history`.

### Sequences and video sprites (motion M0)

A move plays its own **sequence**: an ordered list of drawings with named keys, the drawings that land on the move's
hits and contacts. `motion` names each template's default; a move picks another with `sequence: <name>`.

```json
"sequences": {
  "run_R1": {"drawings": ["R1-1", "R1-2", "R1-3", "R1-4", "R1-5", "R1-6", "R1-7", "R1-8"], "cycle": true, "steps": 2},
  "vault_PK2": {"drawings": ["PK2-1", "PK2-2", "PK2-3", "PK2-4", "PK2-6"], "keys": {"plant": 1, "push": 2, "land": 4}},
  "vault_MV1": {"drawings": ["MV1-1", "...", "MV1-12"], "keys": {"takeoff": 3, "plant": 5, "push": 7, "land": 9}},
  "vault_VW": {"video": "VW", "range": [24, 72], "keys": {"plant": 45}, "label": "climb-over"},
  "run_SW": {"video": "SW", "range": [54, 69], "measure": [30, 77], "cycle": true, "steps": 2, "keys": {"contact": 54}},
  "vault_VF": {"video": "VF", "range": [18, 42], "keys": {"plant": 29}, "speed": 1.75}},
"motion": {"run": "run_R1", "vault": "vault_PK2", "leap": "leap_PK1"},
"stride_phase": {"R1-1": 0, "R1-2": 1, "R1-3": 2, "R1-4": 3, "PK2-1": 3, "PK2-6": 0},
"video": {"VW": {"path": "tests/motion-02/clips/m2-vault-wan.mp4", "key": [172, 203, 236]},
          "VM": {"path": "...m3-mc.mp4", "key": [172, 203, 236], "crop": [0, 4, 1920, 1084], "until": 3.2},
          "RW": {"path": "...m2-run-wan.mp4", "flat": true, "ground": 908}}
```

- A **cycle** (`cycle: true`) has `steps` steps, one step per beat: R1's eight drawings are two steps of contact,
  passing, push and flight, 8 photos a cycle at 16ths.
- The drawings between two keys are the in-betweens. They are sampled one per photo in order: some are skipped
  when there are fewer photos than drawings, and one repeats only when there are more photos than drawings (`check`
  says so).
- `stride_phase` says where a drawing sits in a run step (0 contact, 1 passing, 2 push, 3 flight). A run ends on
  the drawing before a move's first drawing and picks up after its last one.
- Sheets split in reading order, over any number of rows (`n` / `cols`).
- **Video sprites** (`video`): every frame of a clip is matted like a sheet drawing. A pale guide block or pole is
  keyed out by colour (`key`, RGB; `tol` 42) and painted over with the wall. Only the figure's own blob is kept:
  the largest dark component near where it was a frame before, plus small dense bits touching it. Faint scribbles,
  smears elsewhere and footprints at the feet (`flat`) are dropped.
  - Frames the model doubled are dropped, noise and all: the overlap is over 0.85, or over 0.7 with the centroid
    within 2 px (a new pose moves the limbs by tens of px).
  - Without a guide block, the ground is where his feet usually are (the 75th percentile of the lowest points).
  - Each frame stores the parts of the figure on the ground line (`feet`: a planted foot, or a toe landing or
    leaving).
  - `crop` and `until` trim the clip; `below` blanks everything under a line.
  - Each frame is a pose named `<video>-<frame>` and stores where the figure was in the clip. Each video stores the
    block's box, the ground line (the block's bottom, or `ground`) and the figure's scale: a running figure's box is
    about 425 px on a 450 px body, measured over `height_frames`.
  - Video frames get no contact sheet tile.
  - `scale`: street px per clip px, set by the clip's guide instead of measured from his height (an in-box clip:
    a box outline drawn 760 px tall for a 420 px box is 420 / 760).
  - `boxed: true`: a front view of him inside a box prop. The box sets his size, so the canon check skips the
    canon's area for it (his clip's median still applies).
  - `rescale: [{from, to, factor, about: [x, y]}]`: the model drifted (box-charge drew him 13% smaller from frame
    58 on): those frames scaled back by `factor` about a clip point, so the clip keeps one body size.
  - `dedupe: {iou, still}`: the overlaps above which a frame counts as a double of the last kept one (0.85; 0.7 with
    the centroid within 2 px). A clip with subtle motion needs them higher (0.97): box-turn's hood turning and
    edge-launch's head lifting in the crouch were dropped as doubles at the defaults.
  - `height_px`: what the `height_frames` measure on a 450 px body when they are not a run (an upright walk: 440).
  - `ground: "feet"`: the ground is where his planted feet are even with a guide shape (a block on the wall, a pole,
    a bar), the 90th percentile of the lowest points.
  - `bridge: true` (a pole or bar in front of him): the pieces of him on either side of the guide shape join into
    one figure through it.
  - Each video stores its guide shapes one by one (`shapes`: a pole and the beam it leads to, a row of posts).
  - A sequence's `skip` frames never show, nor do drawings with canon errors (canon.json): the neighbouring frame
    shows instead.
  - A sequence can carry lists of clip frames by name for a strike in the box (`hit`: the punches or the furthest
    shoves, `back`: drawn back), which `inside`'s `strike` plays on the hits.
- A **video sequence** (`video`, `range` of frames, keys by frame number) plays at `speed` times the clip's own
  speed (default 1): the frame shown on a photo is the key frame plus the seconds since the key's hit (move time:
  a song stop the move pauses through takes no clip time), times the frame rate and the speed. A move clip played faster reads quick: the director's note on M2 (review note) is that
  obstacles and jumps should all be quick. The quick climb-over (`vault_VF`) plays at 1.75, so the last stride
  before take-off to the first stride after landing takes about a beat, with his weight on the top for 1 or 2
  photos. A move's `speed:` overrides the sequence's.
- A **video cycle** (`cycle: true`, `range` from one touch-down to just before the same foot's next) is
  **stride-true**. Its foot plants are measured over `measure`: a foot on the ground that stays within 70 px over
  consecutive frames. The step is the cycle's own (from its first plant to the same foot's next, over `steps`
  steps), else the median step.
  - Each cycle frame gets its place in the cycle (`travel`, clip px). On stance frames that is the plant's place
    plus the torso's offset from the planted foot; flight frames move on evenly in time.
  - The legs cycle on our beat (one step per beat). The travel per step is the clip's step scaled to the figure,
    covered evenly.
  - Each photo shows the frame whose stride-true place is nearest its own, so the feet slip only by that
    difference. `choreo.notes` has the step, the px per photo and the largest slip.
  - The Wan sprint (`run_SW`) gives a step of 333 clip px (348 street px), 87 px a photo at 16ths. Its feet slip at
    most 31 px.
  - A video run starts after the previous move (or at `from_x`), else runs up to the next move. `check` warns when
    its stride-true end misses the next move by more than half a photo's travel. `x: [x0, x1]` and `speed` stretch
    it (the feet then slide).
  - Between two placed moves its stride is fitted to meet the next one, by at most `fit` (0.1: 10% longer or
    shorter; `choreo.notes` has `stride_fit`).
  - `per_beat` (a run move's: 1) takes that many steps a beat: 1.75 is a quicker jog at the same stride (verse 2's
    run to the crates); a value whose photos land on the same two phases by turns (2 at 8ths) shows two drawings
    flip-flopping, so pick one that walks round the cycle.
  - `ramp: [[time, factor], ...]` (a run move's) is slow motion: the run's own clock runs at that factor of its
    speed (piecewise linear between the points, held before the first and after the last), the legs and the travel
    together, so it stays stride-true (`x: [x0, x1]` still fits it between its ends). Full Stop's p6: the sprint
    along building A eases from 1 at 40.2 to 0.27 by "Full" (55.06) and runs on through the song's silence, its
    last stride reaching the edge on the drop. A video run's photos follow the segments' rates (a run on into an 8ths
    segment is photographed at 8ths), unless the move sets `every`.
  - `"travel": "time"` with `"step"` (clip px a step, in the sequence spec) is for a clip shot with a tracking
    camera (the true sprint and lope, on green): neither the feet nor the torso travel there, so the cycle's
    `steps` steps of `step` are spread evenly over its frames by frame time (`poses.time_travel`), and the frames
    play in order.
  - `"travel": "torso"` (in the sequence spec) skips the plants and uses the torso's own travel, for a clip whose two
    feet merge on the ground line and so measure wrong (run_RW: 65 px a step measured, about 165 by the torso).

## Motion (M0)

`motion: m0` (the default) is the overhaul after the director's notes on segment 1 v1: two drawings repeating in the runs,
rough jumps and vaults, abrupt transitions, the figure stepping back. `motion: legacy` keeps prototype 6's
placements exactly (moves.py).

- **Runs** play one cycle strictly in drawing order, never mixing sheets.
  - One step falls on each beat, so the contacts are on the beats.
  - Travel is even: a constant speed, eased in and out of the neighbouring moves. Each photo's travel changes by
    about 15% at most, and the run never goes backwards.
  - A run eases toward a neighbour's step only within 0.8 to 1.25 times its own average, so its middle stays even.
  - The foot lock is gone: the drawings' foot positions are too uneven for it.
  - Where the beat and the stride phase disagree, a run skips a drawing or two, still in order. It first converges
    onto the beat grid, then matches the next move's phase. A run shorter than two steps only matches the next move.
  - With nothing before it, a run runs up to the next move at its speed.
  - A video cycle is spread over its steps' beats, at the clip's own average speed (the frames' own vertical bob is
    kept).
- **Moves** play their sequence's drawings in order and place them by the torso, so the figure moves forward photo
  by photo; only true contacts are anchored (hands on a top, a grip, feet on the ground).
  - Flights follow one parabolic arc between key positions with eased spacing: faster at take-off and landing,
    slower at the top.
  - Landings hand over to the run at the next stride phase; the first step after a landing is half the flight's
    last step (80 to 150 px).
  - A move's take-off point (`pre_x`) is where the run before it ends, when that leaves the run room.
  - The vault's landing clears the obstacle's far edge: paint goes behind an obstacle in front of the wall.
- **Vault with video sequences.** The figure is scaled to body size. The clip's block is put on the obstacle: its
  edges on the obstacle's edges, the travel over it stretched. A waist-high guide block against a chest-high cabinet
  would leave his hands and head inside it on the take-off, so his height comes from the clip plus a lift:
  - the lift is 0 while his feet are on the ground and the obstacle's height over the scaled block's while his
    weight is on the block, rising smoothly in between (he rises earlier on the take-off) and falling the same way
    after the top;
  - on the plant photo his low point is on the obstacle's top;
  - every photo is then checked against the solid occluders (`_hidden`): before the top he steps back when his
    pace allows (at least 80% of his last step, never back), else he is lifted; on and after the top he is lifted;
    on the way up he never sinks again;
  - `choreo.notes` has the lift on top and the lift on each photo.
  - Before his feet reach the top he stays left of the obstacle, and after leaving it right of it, so he isn't
    painted behind it. He still keeps his pace: at least 80% of his last step.
- **Leap (J2).** The take-off is on `at`, or later when the flight would need more photos than the sequence has
  flight drawings. J1 stays as before.
- **Video moves** (`motion.video_move`, segment 1 v3 and 2: every move from its clip). Key frames are pinned to
  photos (`keys: {key: time}`, or the template's own times); the clip plays piecewise linearly between pins and at
  `speed` outside them. One frame a photo, the nearest; when no drawing is near that moment (frames left out) or he
  is in the air between two drawings, the nearest drawing is put where the clip has him then (feet planted keep
  their own place). His place comes from the clip's guide shape on the street:
  - a **pole** (`pole_swing`, a climb): his x against the pole is the clip's (the clip's tallest shape); a climb
    clip's beam puts his hands on the street's beam while he hangs; never a step back;
  - a **bar** (`swing_release`, the awning): the clip's bar on the street's bar, his hands reaching it while he
    hangs (between the catch and release keys), the height easing in from the ground and back down; `catch_x`;
  - a **block** (`vault`, `leap_off`, `step_up`, a clip with `obstacle`): as for the vault above (edges on edges,
    the lift onto the top, the solid occluders cleared; a step back of at most 60 px, else a lift). Under an
    obstacle (an `under` contact: a slide) he is never lifted. `x_offset` shifts the clip along the obstacle; a
    sequence's `on: [first, last]` (or `"none"`: hops) says where his weight is on the top;
  - **none** (a walk, a sit, a free leap): `x: {key: torso x}` places key frames (between two pinned keys the travel
    is stretched to fit, outside them stride-true), else it starts a step after the previous move; turning on the
    spot never steps back.
  `stretch: [key_a, key_b, factor]` (or a list of them) scales the travel between two keys (a longer flight, a
  shorter get-up). `bottom: {key: y}` puts that frame's low point on y (a slide the clip floated, his sit on the
  box's inner bottom), easing in from the ground and back out by his next step, never below the ground line.
  `splash: <key>` lands with a splash. `from`/`to` trim the photos.
- **The J2 leap from a clip** (`leap` with a video sequence, Full Stop's `leap_AL`): its `takeoff` key (his feet
  leave the ground) on `at`, its `land` key on `land_on`, so the frames between are the flight stretched over the
  photos (three beats at the drop: 12 photos), the clip's own pace after the landing until `to`. `takeoff_x` and
  `land_x` place his torso (defaults 45 px before the near edge, 250 px past the far side). The flight is re-placed
  on one smooth eased arc as high as the clip's own top plus `height` (`motion.arc_flight`: the clip's frames are
  uneven in time, so its own travel lurches); the flight photos are the flat paint skin in the air with the gap's
  glow on its rim and a trail of pale ghost skins of the photos before (`ghosts`: their fade in 16ths, 3; 0 for
  none); the take-off photo leaves the wall past the edge; a splash at the take-off and on the landing. Back on
  the wall the primer arrives gradually: the patches under the first `grow` wall photos (4) start at 45% of their
  size round him and grow to full size (no large pale patch appears from nothing round the landing; the landing
  splash adds no primer of its own). The camera eases throughout, never snapping: it widens over the take-off
  photos to hold the whole arc and the gap (16:9; the 9:16 follows him), pans with him along the arc so he lands
  near the frame's middle, and closes in to the follow framing over `settle` photos (5) after the landing
  (the director's note on demo v3: the landing must not zoom in or turn white all of a sudden).
  A clip that starts before the take-off (edge-launch: the crouch at the edge through the song's silence, the head
  lifting to the red light, the coil) plays from `from` with more keys pinned to their words (`keys: {look:
  41.4.3, coil: 42.4.3}`), those photos on the wall (`pause: false` across the song stop; a near-still stretch
  breathes between neighbouring frames instead of holding one, as in the box). The take-off as the biggest hit:
  `burst: {flecks, splats, power, spread}` throws paint from his feet as he leaves the wall (a `props.Burst`: black
  flecks in the air, splatter landing on the wall over a few photos), `hang` (16ths) keeps the whole arc's ghost
  skins hanging over the gap after the landing before they fade, `splash` scales the landing splash, and `knock:
  <light id>` shatters a street light he flies past: it flares on the coil's last photo, then pale glass flecks fly
  and the light and its lamp stay dark (he goes on red: Full Stop's red pedestrian light at the alley).
- **A free leap from a clip** (`clip` with `arc: {from: takeoff, to: land, height}`): the flight between the two
  pinned keys on the same smooth arc, so it rises and falls instead of holding one height (bar 38's long leap).
- **The leap out of the box** (`burst` with `frame: box` and `leap: {sequence, keys, land, x, height}`, demo v5):
  the breakout clip plays up to its break-through on `at`, then a second front-view clip (box-leap: the dive out of
  the box, its guide box anywhere in its frame) from the next photo, put on the street box the same way, its keys
  pinned (`land` on the impact), so with `pause: false` the dive is slowed through the song's stop instead of held.
  `x: {key: torso x}` places its keys (past the red light's pole fast, then the slow part where nothing crosses
  him), `height` lifts the flight on a parabola, out of the box he never steps back, and `grow_from: <key>` grows
  him to his running size from that key to the landing, so the sprint takes over at full size on the next photo.
  `path: [[key or time, x, y], ...]` (v5 review) replaces his course from its first point to its last with one
  smooth monotone curve in time through those places (torso x, body centre y; null: where the clip and the pins put
  him; `motion._pchip`, so the top stays where its point is): up from the burst and over the red light, a slow top
  over it through the stop, the dive down onto the impact. `tilt: {to, max, gain, ease}` turns every drawing along
  the course (nose up rising, level at the top, nose down diving): `gain` (1) times the course's angle less the
  drawing's own slant (its long axis: the launch drawings already lean up), at most `max` degrees (60), fading in
  over `ease` photos (2) from the first path point and let go from the `to` key (default `dive_end`) to the
  landing, so the tuck lands upright.
- **Slow motion through a stop** (the director's note on v4: slow-downs, not pauses): any video move with `pause: false`
  and its keys pinned on either side of a song stop plays its frames slowed through it. A block move (`leap_off`,
  `vault`) takes `keys` (more of its clip's keys pinned), `arc: {from, to}` (the flight on one smooth arc) and
  `pace: false` (the clip's own travel, never back, instead of keeping at least 80% of his last step): the leap off
  the wheelie bin, its top in the middle of the third stop. Strict `dedupe` on the clip keeps every drawing for the
  slow photos, and a drawing never shows on more than two photos in a row.
- **The breakout from a front-view clip** (`burst` with `frame: box`, Full Stop's `box_YB`): every frame put where
  the clip's guide box puts it on the street box (as in the box), the wind-up from `from` (`windup_on`), the
  `through` key (his fist through the right side) on `at` with the shatter, the apex before the next song stop and
  the landing on `land_on` as before. After the break-through he leaves the box's frame: `apex_x` and `land_x`
  move his torso there (eased from the break-through), his low point on the landing goes onto the ground line
  (the clip's box floats above its own floor line) and never below it after, and from the landing he grows to the
  runner's size over `grow` photos (4; measured against `run_sequence`, run_SW) so the run after him takes over
  without a jump.
- The templates' video branches: `vault` (`key`, default plant, on `plant_on`; `land_on` pins the sequence's land
  key), `leap_off`/`step_up` (`key`, default on, on `at`; `land_on`), `pole_swing` (`grab_on` and/or `land_on`),
  `swing_release` (`release_on`; `catch_on`, `land_on`, `catch_x`), `burst` (`out` on `at`; `apex` on `apex_on`,
  default the last photo before the song's next stop, so the stop holds him mid-air; `land_on`; `land_x`, `apex_x`).
  `clip` plays any video sequence (below).
- `check` adds a motion line and warnings:
  - backtracks: the torso goes back by more than 12 px (the pound's wind-up excepted);
  - lurches: the travel per photo changes by more than 35% between two photos while he moves at least 20 px;
  - repeated drawings: twice in one move that is not a cycle, or on two painted photos in a row (a slide's glide
    excepted);
  - runs that mix sheets.
  `check.motion_stats` gives the numbers (`max_run_lurch_ratio`: inside runs only).

### Physics (`parkour.py`, v7)

review note (the director's film-v6 notes: runs that speed up and slow down, stops dead before jumps, floaty jumps
stretched onto beats, slow motion everywhere, loose contact). A `parkour` move walks him through a stretch of his
route by physics instead of keys pinned to beats, and emits the same Evs (wall poses on the feet line while grounded,
air skins in flight, stamps), so render, QA and the camera work as before.

- **Scale.** His standing height (the library's `ref_height`, 450 px) is 1.8 m: 250 px a metre. One gravity, 9.81
  m/s2 (2,452 px/s2).
- **Gaits** (m/s): walk 1.4, jog 3.5, run 5.5, sprint 7.0 (150 px per 16th), or a number. Speed changes at most 4
  m/s2 up and 6 m/s2 down. Default cycles: sprint and run `run_SW`, jog `run_RW`, walk `walk_WL` (`cycle:` picks).
- **Stride-true always**: the leg phase is the distance travelled over the cycle's measured length. Before a
  take-off the last six steps lengthen or shorten (at most 8%) so he leaves at the stride's push-off.
- **Jumps** (`jump`, `drop`, `flip`, `hop`) are ballistic: the take-off foot (the drawing's lowest point) at or 8 px
  before the edge, the take-off's horizontal speed kept, the vertical speed from `rise` (the apex above the take-off,
  m) if the yaml asks (the big moments), else a running jump's natural rise (0.45 m) or more if that is needed to
  clear the gap's far edge by 0.15 m; a hop rises only so far that every drawing's feet over the prop clear its
  top by 0.25 m (scanned along the arc: not a monster jump); `hop: {..., onto: true}` instead lands on the prop's top
  (a step up: the least rise whose landing drawing's feet pass its near edge 0.1 m over it, coming down on the
  top as near that edge as it allows, at least 0.15 m in and 0.3 m short of its end, so he runs along it; solved
  again from where he really leaves, and an error with the take-off x it needs when no arc from there comes down
  on the top: too little run-up). Landing on a prop's `top` (any jump), he stands and runs on it until a `drop` or
  `jump` from its edge (`edge: 27796`, the top's own end) takes him off, every stride on the top up to it (a leading
  foot reaching past its rim too); running past its end without one is an error ("runs off p_wallB's top"). Before
  a take-off from a level's edge (a step down) his strides stay on the upper level. A flight photo whose drawing
  reaches the surface a moment early is put on it, and the flight photos just off a top or coming down on one, or
  within 0.06 s of the take-off or the landing, may reach 0.1 m into the surface (QA `inside`: the skin's smear, a
  trailing foot at the rim). The run's last contact plants the take-off foot where the take-off drawing has it
  (the stride's phase measured from the drawings), and the take-off photo is the flight's first skin (J2) with its
  foot on the edge (off a level laid on a solid box's top, v8's awnings: that box is his ground there, as a prop's top
  he landed on would be: `on`, so the QA's inside rule leaves it to the ground line's tolerance). A jump's paint: a splash where he leaves and where he lands (`splash`: its size, 1; false: none),
  `burst: {flecks, splats, power, spread}` (paint thrown from his feet), `knock: <light>` (a light he flies past
  shattered as he leaves). He lands where the landing drawing's feet
  meet a surface (a level's feet line or a prop's `top`), whenever that is. The flight's drawings go by normalized
  flight time between the clip's `takeoff` and `land` keys, the torso on the arc.
- **Landings**: the land phase's drawings over 0.12 to 0.4 s, the deepest crouch held longer the bigger the fall,
  speed kept less 5% plus 3% a metre of fall; after a fall over 2.5 m or a landing faster than 7 m/s, the roll
  (`roll_RL`, `roll_sequence:`), 10% lost; each roll drawing's own lowest paint on the landing surface (a roll is in
  contact all through, whatever floor the clip had: a drop filmed from the higher roof), checked by the `feet` rule.
  `land: {clips: [[sequence, first frame, last frame], ...], v_out: 2.8, brake: 0.12}` on a jump or drop without a
  roll (v8, P4's drop into the courtyard) plays those frames of other clips instead, in order, each at its own
  speed (a deep landing crouch from one clip, the rise out of it from another), his speed falling from the
  landing's to `v_out` (m/s) with the time constant `brake` (s), each drawing's lowest paint on the landing surface;
  the run takes over from `v_out`. `auto_cycles: [...]` limits `cycle: auto` to those cycles (v8: run_SP left out
  for its SP-099, whose trailing toe sinks under the roof). A clip item's `dy` (px) moves the whole clip down: a
  climb cut for a taller wall, its top on the lower roof (v8: wallclimb_VW3, cut for 812 px, up W4's 717).
- **Vaults** (`vault`): the clip's frames by the share of the obstacle crossed (its guide block on the obstacle, the
  travel over it fitted to the obstacle within 25%), at the run speed less 5% (`loss`), then speeding up again.
- **Clips** (`clip`): a clip from key `from` to `to`, its travel scaled to him, placed by `obstacle` (a pole's line,
  a bar's middle at `catch_x`, a block's near side (the clip's registered `block` when it has one, else its first shape; v8b, P7's clips list the floor first); `align: block|pole|bar` overrides: a slide under a ladder) or `x:
  {key: torso x}`, the run before it running up to its first frame; `place: here` starts it where the clip before
  left him, the obstacle only setting the bar's height (a climb into a swing); `beam: <obstacle>` puts a pole climb's
  hang on the beam it reaches. Its pace keeps his momentum (`speed: auto`, the default for a clip that travels: his
  speed through it is his speed coming in less 5%, within `min_rate` to `max_rate`, 0.6 to 2, times its own); `speed:
  1` plays it as shot. On a
  bar the frames whose top is at the clip's bar carry the bar's height and the `grip` rule. `box: <prop>` plays a
  front-view clip in a box prop (its guide box on the box, one scale: the breakout), waiting in its first frame until
  its `key` meets its `at`; `shatter: {fly, power, splats, stencils, punch, shake, flicker}` explodes the box on that
  key (the shards, the splatter, the stencils breaking off, a spray, the camera punched and shaken, a light
  flickering). `behind: <obstacle>` (also on a vault) lets that item pass behind it (a step into a doorway). `ground: start` stands
  every frame's heights on the ground under the clip's first frame (a climb that crosses onto a higher level keeps
  rising from where it began), `ground: <level id>` on that level's feet line; by default each frame stands on the
  ground under it. `light_from: <key>` gives its frames from that key on a light trail (no keepers: their
  paint-outs wash away; the crouch he tears himself out of leaves no figure). v8 (P2b): keys (`from`, `to`, `key`)
  may be clip frame numbers too (`to: 57`: a hand-over between two clips on whichever drawn frame matches best); a
  bar clip's height is measured from the ground its heights stand on (`ground: start` or a level: the bar still on
  the street's bar when the clip starts on a deck and swings out over the street). `flight: [k1, k2]` makes the part
  from key k1 (the take-off) to k2 (the catch) ballistic: k1's frame with its foot `EDGE_PX` before `edge` (without
  an edge where the clip puts it), k2's where the clip's placement puts it (a bar's catch), his horizontal speed his
  take-off speed (the run's when the clip starts at k1, else the clip's own pace into it), the time in the air the
  distance over it, the torso on one gravity's arc between the two frames' heights and the drawings between them by
  normalized flight time; before k1 and after k2 the clip as shot. ch.notes[move]["clip_flights"] reports each (its
  take-off time and x, time in the air, speed, the catch x); its foot past the edge is an error. Full Stop: the
  leap off deck 2's open end to the iron bracket (bracket_P2B).
- **Flips** (`flip`): one tuck drawing (`tuck:`) turned by the engine between the take-off and land drawings, one
  direction (`dir: front`: clockwise for a runner going right; `back`), the whole turn, eased.
- **Music sync from placement**: `at` on an item anchors its event (a jump's take-off, a vault's first frame, a run
  waypoint's arrival, a clip's `key` frame) to a time. Between anchors the solver finds one speed factor for the
  span (all its gaits times it); within `tol` (0.10) the span is fine, beyond it the move has an error saying how
  much route the span is short or long (the time it misses by at the nearest allowed factor, times his speed: "add
  about 6,000 px of route before it, or start 3.4 s later") and is placed at the needed speed so it can be watched.
- A solid prop on his running line that no item gets him over is an error ("he runs through utility").
- A parkour move ignores the song's stops (no pause) and should have 16th photos: real time, no slow motion.
- `skin: true` turns its wall poses into the J2 paint skin (the hook for the figure after he leaves the wall).

    - parkour: {from: 51.1, from_x: 420, v0: carry, gait: sprint, to: 58.3, route: [
        {run: {gait: run, x: bench.x0-600}},                  a waypoint (with `at`: he is there then)
        {vault: {obstacle: meter, sequence: vault_ME}},
        {hop: {obstacle: bench, sequence: dash_BD}},
        {clip: {sequence: pole_PS, obstacle: pipeA2}},
        {jump: {gap: alley, sequence: leap_AL, at: 57.607s, rise: 0.4}},
        {drop: {x: 21400}},                                   off an edge onto a lower surface (roll: auto)
        {flip: {edge: 61200, sequence: flip_FL, tuck: FL-048, dir: front, at: 167.31s}}]}

`ch.notes[move]` (and the render's QA json, `solver`) holds each span (its time, the route's px, its speed factor,
ok, and short or long px), the flights (take-off, time in the air, landing x, fall, landing speed, loss, roll, rise,
feet clearance) and the start; `aimv paint render --debug` draws it all on the photos (see Outputs).

### QA (`qa.py`)

Every `aimv paint check` and every render checks each figure the photos show, once, at the first photo that shows
it:

| Rule | Level | What |
|---|---|---|
| `solid` | error | part of him hidden behind a solid obstacle (over 0.5% of his paint and 40 px; the top 14 px of a solid part are a touch band; `behind: <obstacle>` allows it) |
| `head` | error / warning | the top 20% of his box hidden: behind a solid obstacle (over 10%) an error, behind a thin occluder or a parallax layer (over 15%) a warning |
| `ground` | error | his feet below the ground line (the lowest paint more than 2 px under the wall's foot: the paint is cut there) |
| `no_paint` | error | over 25% of him where paint cannot go |
| `frame` | error | over 2% of his box outside what the look keeps of the 16:9 or 9:16 frame (while the box camera's `look` pushes in, only the top third of his box: his hood and shoulders) |
| `backtrack` | error | the torso back by more than 12 px between two photos (in a box, pulling back to hit it again is not one) |
| `lurch`, `repeat` | warning | as in Motion |
| `crossing` | error | a plate object (a pipe or pole: a thin occluder or a parallax layer) across more than 4% of him while he stays put on the ground (his torso within 40 px over 4 photos, his feet or body on the ground line: a crouch, a hold, a freeze; a slow flight past a pole is not one; the obstacle his move is on, the pipe he climbs, does not count). the director's note on v4: the drainpipe right in front of him in the crouch before the leap, since patched out of the plate |
| `handover` | warning | v8b (the director's notes at 107.0 and 158.7): a clip handing over to a run, or a run to a clip, on consecutive photos where his torso speed over the two photo intervals before it and the two after it (a clip's torso surges with each step as shot) differs by more than 1.5 m/s (check.HANDOVER_DV; `motion_stats(ch)["handovers"]` lists every hand-over with both speeds) |
| `hold` | warning | one drawing up on more than 2 photos in a row (review note: "you are using single frames again"), outside a true freeze (a song stop or freeze move with nothing painted in it); a move's `still: true` marks a deliberate stillness |
| `canon` | error | the drawing shown has a canon error (`canon.py`, `paint/poses/canon.json`): `<pose>: <rule>: <why>` |
| `feet` | error | contact (v7): in every grounded photo (a parkour photo on the ground, a clip frame whose feet are on its clip's floor) his lowest paint more than 6 px from the surface under it (the level's feet line or the prop's top he stands on; 12 px for a jittered drawing or a roll) |
| `takeoff` | error | his take-off foot (the sole: the mean of its lowest 8 px of paint, as the drawing's bottom anchor the solver plants) more than 2 px past the edge he leaves from (a parkour jump) |
| `inside` | error | in flight, his paint inside a solid obstacle's box below its top, or under the feet line of the ground below him (inside a roof or the pavement); a flight photo flagged `touch` (at the take-off or the landing) may reach 25 px (0.1 m) into it |
| `grip` | error | hanging from a bar (a parkour clip's hold frames), his hands (the top of his paint) more than 20 px from the bar |
| `fist` | error | in the box (inside, pound), his paint past the outer edge of the box's side lines (a punch through the wall, the director's note at 32.5) |
| `reveal` | error | v8 paint-in (reveal.py): a grounded photo (running, landing, standing) whose feet are on world not painted in yet |
| `camera` | warning | the camera's pan changing by more than 4% of the frame width from one photo to the next (a pan that starts or stops at once), or its zoom by more than 4% in one photo, outside a cut (the punches and the hand-held nudge do not count) |

A render writes `<name>-qa.json` and `<name>-qa.jpg` (every flagged photo, the hidden paint tinted red, labelled with
time, rule and figures) and `<name>-moves.jpg` (every move from 4 photos before to 4 after, every photo labelled,
flagged ones in red; a move longer than 120 photos wraps onto more lines; a sheet taller than a JPEG can be, 65,500 px (or a part over 150 Mpx),
is split at whole rows into `<name>-qa-2.jpg`, `-3.jpg` ... and `<name>-moves-2.jpg` ...), and prints `QA PASSED` or `QA FAILED: N ERRORS` in a banner. `--review` makes the small
review copies (1600x900 CRF 27, 720x1280 CRF 26), but not when the QA has errors unless `--force`. `check` and
`render` exit with 1 on QA errors.

### Canon check (`canon.py`)

`aimv paint canon <project>` checks every drawing of the pose library for character drift and writes
`paint/poses/canon.json` and `canon.jpg` (every sheet and clip as rows of thumbnails, flagged drawings outlined red
for errors and amber for warnings). A video model only sees a clip's first and last frames, which are built from
canon drawings, so the drawings in between are where he drifts. The check compares each drawing with the canon sheet
(R1) and with its own clip:

| Rule | Level | What |
|---|---|---|
| `scale` | error | the painted area at street scale outside 0.55 to 1.45 times the canon's, or (clips) outside 0.55 to 1.35 times its clip's median: he grew toward the camera, or part of him is missing |
| `smear` | error / warning | (clips) a soft grey zone on his edge that fades straight into the wall (motion blur, a melt); dry-brush streaks fenced by darker paint do not count. An error over max(6 x the clip's median, 0.6), a warning over max(6 x the median, 0.3), per 1000 of the body height squared |
| `colour` | error | (clips) skin or any colour in over 1.5% of his pixels in the clip frame (a face, bare skin); the guide block is ignored |
| `shoes` | warning / error | (clips) the white shoe parts (light shapes held between paint above and below at the feet: socks, midsole) below 35% of the clip's first and last drawings; an error on each when over 30% of the clip misses them |
| `pieces` | warning | over 8% of his paint away from the main body (a melt, scribbles) |

Sheets get `scale` and `pieces` only: they are stills, checked by eye against the model sheet
(`boards/16-model`: MS1 turnaround, MS2 details) before they are cut. A `boxed` clip skips the canon's area (the
box sets his size). A source shot on chroma green (`"screen": "green"`: the true figure, photoreal, not paint)
skips the paint's rules (smear, colour, shoes); scale and pieces still apply. `canon_waive` in poses.spec.json
(`{drawing: {rule: reason}}`) turns errors that were checked
by eye and found clean into warnings that say so: the smear rule reads a dry-brush highlight on a sleeve, a back or
a thigh (fenced by darker paint, reached from outside through a thin light streak) as a smear, and a stricter rule
did not separate those from the real melts of the motion-control take. Full Stop waives twelve such drawings of the
v3 clips (brake-crouch, box-mouth, box-charge), each with its reason. Calibrated on Full Stop's library (749
drawings): it flags the known-bad clips (the Kling vault, the motion-control take) and the smeared hands of AW-042,
and is quiet on clean drawings. `aimv paint poses` runs it after every build, so canon.json always matches the
library; clip sequences skip drawings with canon errors and the QA fails a render that shows one.

### Clip frames for obstacle clips

`aimv paint clipframes <project> <obstacle> --move vault|leap_off|step_up ... [--cycle run_SW]` writes
`paint/clipframes/<obstacle>/<move>-start.png` and `-end.png` (1920x1080): the white wall, the pale blue guide
block (RGB 172, 203, 236) on the ground line at y 900, centred at x 970, sized as the obstacle is relative to a
430 px figure (a box hung on the wall floats at its height), and the figure painted from the run cycle's first
drawing, running in on the left and away on the right. `<move>-frames.json` has the geometry. Generate the clip
from these frames, then add it under `video` and a sequence in poses.spec.json.

The shape follows the move: a pole (`pole_swing`, or a pipe) is a vertical bar from the top of the frame to the
ground, as wide as the pipe; a hang (`hang_release`, `swing_release`, `hang`, or a beam or bar contact) is a
horizontal bar at the bar's height with nothing under it (at most 1000 px long; a bar above the frame makes the
figure smaller); a block otherwise, floating when the obstacle's box ends above the ground, one block per post for a
row of bollards (the figure smaller when a wide obstacle needs the room for a run-up and a run-out); a slide always
gets a block. `--and fire_escape` adds another obstacle's shape where it stands against this one (the beam a climb
ends on); `--start-pose` / `--end-pose` put another pose or role in either frame (role `hang`: hanging from the
bar by his hands); obstacle `wall` is the plain wall (a brake into a walk). Clips with a pole or bar need
`"ground": "feet"` (and `"bridge": true`) in their video entry. A video entry's `"block"` and `"shapes"` (clip px) set the guide block by hand where the
build's own (the box round all the guide pixels) is no use: a clip whose roof guide spans the frame.

A `gap` obstacle (the alley) gets no shape: the start frame has him coiled at its near edge, the end frame running
past its far edge. An `edge` gets no shape either: the end frame stops him with his front on it (a brake into a
crouch). `--box WxHxLINE` (street px, obstacle `wall`) draws a front view inside a box: the box's outline as the
pale blue guide (keyed out), 760 px tall, its bottom on the ground line, the start and end poses fitted inside it;
`--scale-by POSE` gives both frames one body scale (that pose fills the box: its height, its reach to the sides),
so a clip where he stands up does not shrink him. The clip's video entry then takes `"scale"` (box height / 760),
`"ground": "feet"` and `"boxed": true`.

For a chain of clips that hand over seamlessly (each clip's end frame is the next one's start frame), give every
clip the same `--box` and `--scale-by` and the shared pose: the frames come out identical. A pose in the box can
say where it stands: `BX2-1@right` pressed against the inner right side (`@left`), `SW-061@out` outside the box on
its right, on the ground line, at the same body scale (out of the box). `--open right` draws the end frame's box
without its right side (he broke out through it), `--folder box` writes the frames to `paint/clipframes/box/`
instead of the obstacle's folder (obstacle `box` is the plain wall too). Full Stop's v4 chain: `box/rise`,
`box/turn`, `box/charge`, `box/breakout`, and `edge/launch` (the alley's geometry: AL's start and end frames); v5's
`box/leap` (the dive out of the box) moves the box to the frame's left third (`--box-x 420`), opens its right side in
both frames (`--open-start right --open right`) and starts from a clip drawing where its own clip had it
(`YB-068@clip`: placed by its clip position, moved with the box) and ends on a sprint stride a stride from the
frame's right edge (`SW-061@far`, the body scale). Out of the box, `--start-x` / `--end-x` put the figure's left edge at that frame x instead
of its run-up or run-out (a short walk to a stop: verse 2's reach), and the command line's `[--start-x X] [--end-x
X]` set them. `--box-h 470` makes the box that tall in the frame instead of 760 (the figure smaller, room for him to run out of
it): v7's `box/smash` (the breakout: YB-049 pressed to the left side, SW-061 a stride from the frame's right edge,
the right side open in the end frame, the box 400 x 450 x 34 at x 360).

Comparing methods: `maps.grid_video` (renders of the same choreography side by side with labels and the song) and
`maps.photo_sheet` (photos in rows, cropped round the figure, the rows aligned on a key photo). The motion test
(`choreo/test-vault*.yaml`, `test-run-m0/m2.yaml`) used them for `renders/motion-02/`.

## Choreography format (`choreo/<name>.yaml`)

```yaml
street: main
seed: 11
jump: J2              # default leap style: J2 free flight (he leaves the wall as flat paint) or J1 jump cut
j1_lead: 1            # J1: take-off 1 or 2 photos before the impact
primer: ghosts        # all | keepers | none | ghosts
primer_fade: {}       # override the fade rules (see Paint model)
motion: m0            # m0 (default: the motion overhaul) or legacy (prototype 6's placements)
foot_lock: true       # legacy runs keep the stance foot planted between beat contacts (M0 ignores it)
pause: true           # a move across one of the song's stops holds its pose through it (default true)
ink: 1.0              # the dark paint's albedo multiplier: figure, splashes, stencils, cracks (0.8 reads blacker)
flight: skin          # his look in the air: skin (the J2 paint skin) or paint (v8, the director on P2b: the wall part's
                      # flights stay paint on the wall, his primer patch behind him, as he is when he runs; a move's
                      # own `flight` wins, and a move's `skin` still takes him off the wall, the free section)
vary: true            # repeated single drawings (hang, climb) get small per-photo variations
hits: {snare: 0.85, kick: 0.5}   # strength thresholds for named hits
include: [opening, p6]           # (a film) segment files joined in front of this file's own segments and moves
camera: {}            # defaults for every segment's camera
window: [55.06s, 69.0s]          # the default render range
segments:
  - {from: 41.1.3, to: 42.4.4, rate: 16ths, camera: {preset: stop, snap: true}, label: the big stop}
moves:
  - crouch: {at: 41.1.3, edge: edgeA}
  - freeze: {to: 42.4.4}
  - leap: {gap: alley, at: impact, land_on: snare}
  - run: {cycle: 1}
  - vault: {obstacle: ac, plant_on: 44.2}
```

### Times

| Spec | Meaning |
|---|---|
| `43` | bar 43, beat 1 (YAML numbers are bars: `43.2` is bar 43 beat 2, never seconds) |
| `43.2` | bar 43, beat 2 |
| `43.2.3` | bar 43, beat 2, the third 16th |
| `57.607s` | seconds |
| `impact`, `kick`, `snare`, `hat`, `vocal`, `stop`, `stop.end` | the first such hit after the reference time (`vocal` is a vocal entry, `stop` a stop's start, `stop.end` its return) |
| `Verse 2`, `Bridge` | a section label from sections.json: its start |
| `snare@44.2` | the snare nearest bar 44 beat 2 |
| `snare>44.2` | the first snare at or after bar 44 beat 2 |
| `"+2"`, `"-1"` | relative to the reference, in 16ths (quote them: YAML reads +2 as the number 2, which is bar 2) |
| `impact-1`, `44.2+0.5` | any of the above with an offset in 16ths |

Bars and beats count from 1, as in `breakdown/TIMELINE.md`. The reference is the previous move's last photo for a
move's first time, and the move's own earlier time after that. Times are snapped to the 16th grid; a named hit more
than 25 ms from its photo is a check warning. `kick` and `snare` only count hits at or above the `hits` thresholds,
so `snare` means a backbeat, not a ghost note.

### Song stops (pause)

The song stops dead several times (stop-time breaks in the choruses, the big stops before the drops). Every stop that
no `freeze` move covers stops the move clock: templates work in move time (song time with those stops taken out), so
a move that spans a stop holds its current drawing through it, mid-vault or mid-leap, the photos keep coming (rain,
light, grain change; the paint does not dry, drip or wash; droplets and shards hang in the air), and the move goes on
with its own rhythm from the stop's end impact. A time given inside such a stop shows at its end (`check` warns).
`pause: false` (top level, or on one move) turns it off: a move with `pause: false` is painted at song time inside
the stop (`check` does not warn about it), while the paint clock still stands still. Demo v3 sets the big stop
before the drop to the words that way (p6.yaml: the brake clip still on the kick, down into the crouch on "Full
stop", up to look at the red light on "nothing", lowering and loading on "left to", the deepest coil on "prove",
the leap on the impact). A stop covered by a `freeze` move is left to that move.

### Composition (include)

A film is one continuous choreography joined from segment files, in song order:

```yaml
# film.yaml
include: [opening, p6]    # names (choreo/<name>.yaml) or paths relative to this file
seed: 11
window: [0s, 56.5s]
```

The included files' segments and moves come first, in order, then the includer's own. Top-level settings: the
includer's win, then the first included file's; `hits` thresholds are merged; each file's `camera` defaults go into
its own segments. Every move keeps its file's seed and numbering for its random choices (jitter, keepers, primer
patches, per-photo variations), so a segment paints the same in the film as on its own; the chain still runs across
the join (a bare `run` at the end of one file runs to the first move of the next). Segments must tile the song.
At each join between two files the camera eases from the last framing before it (carried on at the pace it was
moving, so a camera following a run does not stop for a photo at the join) into the new file's (crop centre
and zoom, both aspects) over about a beat (`camera.JOIN_BLEND`, 4 16ths), eased in and out, so one segment's
camera hands over to the next without a jump (verse 2's eye-level follow into the climb's); `cut: true` in the new
file's first segment's camera keeps a deliberate cut.

### Segments

Segments tile the choreography (no gaps, no overlaps; `check` says so). `rate`: `32nds`, `16ths`, `8ths`,
`beats`, `half-bars`, `bars` (photos aligned to the bar line). `camera`: a preset name or a mapping with `preset`
and parameters:

| Preset | Framing | Parameters |
|---|---|---|
| `follow` | close on him: the crop about twice his height, feet at 76%, a little lead, vertical follow on climbs; with `keep_up` it aims ahead by the lag the smoothing leaves at his current speed, so a fast run stays centred at the lead | `zoom` (1.0), `lead` (0.07 of the width), `follow_y` (0.85), `band` (true: keep his torso in the central 9:16 band), `keep_up` (true with motion M0, false with legacy), `eye` (the frame's middle at that share of a 430 px figure's height above his feet: 0.8 is eye level, verse 2's straight-on walk past the screens), `keep` (true; v8b `false`: a framing on something else, the cold open's splat or his hood close up: the keep-inside rule leaves the frame where the eye level and the lead put it, in both formats, and QA's frame rule skips those photos) |
| `close` | follow at 0.85 | as follow |
| `wide` | follow at 1.25 (only briefly: the director wants the camera close) | as follow |
| `hold` | the segment's first framing, held | |
| `spring` | v7: a critically damped spring after his torso (and his figure's middle) with a look-ahead of his speed that cancels its lag, so a sprint stays framed without a step or a snap; vertical three times softer, rising with his torso above a runner's (a jump, a climb); any zoom change eased over at least a beat and long enough that no photo changes it by more than 3.5%; no snaps (a segment's `cut: true` starts it afresh; v8b: a spring stretch after a stretch framed otherwise, a follow or the box, starts from where that framing was and moving at its pace, never from the state an earlier spring stretch left: the director's "sudden camera shift" at 78.8, verse 2's walk into the run); the 9:16 spring 1.5 times stiffer and kept round him by its own rule. A `hints: frame` stretch (hints.py; values as steps, a zoom as a pair for a push) moves the spring's target: `dx`, `dy` as offsets, `cx`, `cy` as a fixed target it settles on (the vertical spring as stiff as the horizontal towards it; `follow_v: true`: the 9:16 keeps following), `zoom_v`, `dx_v`, `dy_v` the 9:16's own, and `cx_v` its own fixed target where the 16:9 takes `cx` (v8b, P7's ending: the 9:16 holds the door and the STOP sign whole while the 16:9 pushes toward the door) (without them it takes the 16:9's zoom half way and 0.3 of its dx, as the follow framing does, and its dy), each weighed in over the frame's `ease` (s, or `[in, back]`: the zoom change into the stretch and back out after it) and at least a beat, stretched where needed so no photo's pan or zoom changes by more than the camera QA's 4% (Full Stop's big leap: out to about 2x and held over the arc). A zoom written as a pair (`zoom: [z0, z1]`, `zoom_v` too) is a push or a pull the spring follows exactly, photo by photo, geometrically (each photo the same share closer at a steady pace), from wherever the zoom is when the stretch starts and onto z1 on its last photo; the frame's `curve` shapes it and its other pairs: `smooth` (the default smoothstep) or `in` (eased in and gathering speed all the way, fastest just before the end at twice its mean pace, settling onto its end value over the stretch's last tenth: a push that lands on a hit; keep its fastest photo under the QA's 4%: the log of z0/z1 at most 0.02 times the stretch's photos). `centre: c` (0 to 1, or a pair) moves the target from the follow framing onto his figure's middle: no lead (nor the half photo the spring runs ahead, as it chases each photo's target over the interval before it), his middle at the frame's middle height with his torso's lift, the vertical spring as stiff as the horizontal, and where a fall is coming (his torso lower 2/stiffness s on) the height he will be at by then, so a landing stays in the frame. A `dy` (or `dy_v`) pair that changes is a tilt: the vertical spring follows it as stiffly as the horizontal while it runs (film-v8: the ease into verse 2's eye-level framing inside p6's brake). A frame's `aspect: 9:16` (or 16:9) makes the stretch that aspect's only: the other aspect's camera is exactly as without it (film-v8: the 9:16's tilt into the tear-off). Hints later in the file override earlier ones photo by photo, so a push and a separate recentring (`dx: [750, 0], centre: [0, 1]` over the push's first part) can overlap (Full Stop v8f: the push in as he sprints at the alley, 1.7 to 0.8 over 54.26 to 57.6 s, curve in, centred from 55.5, tightest on the take-off). `soft_bounds: true` (opt-in; a file's top-level `camera: {soft_bounds: true}` sets it for its own segments): the spring's resting point (its target less the look-ahead) is kept where the world can fill the crop, moved only up or down and as low as it must be here or where he will be over the next 0.45 s (a drop in the world's top), and in 9:16 where it holds his figure over the spring's lag (this photo's box and the next ones' within 1/stiffness s); the look-ahead is his torso where it really will be (he stops at a wall: the frame slows with him); the vertical spring stiffens to the horizontal's over a quarter second while the frame must catch up (a low top, a fall) and is soft again when free (a release eases out); the keep-inside rule's push carries on in the spring instead of springing back. So the frame eases where the final clamp and the keep-inside rule would snap it (Full Stop's final chorus: the 9:16's pops from 28 to 6, none cut) | `zoom` (1.0), `lead` (0.07 of the width), `stiff` (6 rad/s), `look` (2/stiff s), `follow_y` (0.85), `ease` (s, at least), `soft_bounds` (false) |
| `stop` | a wide establishing frame that pushes in photo by photo while the song stands still | `zoom` ([1.24, 1.0]), `lead` ([170, 110] px), `drift` (1 px per 16th), `follow_y` (0: the ground line; 0.8 to follow a freeze mid-air) |
| `frame` | a set stretch of street (an establishing shot before he is there): the crop `x: [x0, x1]` wide at the ground's height (or `cy`); 9:16 as wide as it can on its middle | `x` (required), `cy` |
| `box` | tight on a box prop, almost still (`steps: [from, to, factor]`: a step closer on every bar between them): the box takes `fill` of the crop height, the crop steps closer as the box shrinks (by `track`, less than the box shrinks, so it still reads smaller), a slow push over a span, and room for something beside it. Every photo then widens and moves the crop just enough to hold the box and the stencils (16:9: those painted so far; 9:16: all of them from the start, a steady frame), `margin` of the frame clear on every side (the look crops about 3% a side), punches and nudges included | `box` (prop id), `fill` (0.5), `fill_w` (9:16: 0.6 of the width), `track` (0.5), `push: [from, to, factor]`, `room: {from, x, margin}` (16:9), `margin` (0.05), `look: {from, to, back, region: [x0, y0, x1, y1], margin}` (v5 review: one slow eased push from the box framing at `from` to the crop round `region`, street px, at `to`, eased back out by `back` (default: held to the segment's end); it sits on top of the framing, which carries on underneath; while it lasts nothing else is held in frame (the region's crop keeps `margin`, 0.08, a side), so it may crop him below the shoulders: the stare, his hood and the red light) |
| `stencil` (prop) | `at`, `text` ("A BETTER/FACE": "/" breaks the lines), `x` (the block's left edge, centre or right edge by `align`), `y` (its top), `h` (cap height, 58), `align` (left, center, right), `rot`, `punch` (0.03), `photo`, `id` (stencil1, stencil2 ...), `fade: {at, over}` (or a time: the line fades out from `at` over `over` 16ths, 2, and is gone: a lyric line after it is sung, verse 2), `albedo` (black spray by default; 0.7: white primer letters where black would not read, verse 2's dark shop fascias); `smear: {at, length (500 px), over (3 16ths), rows: [y0, y1] (street y, default all of it), dir (right)}`; `rot: 90` or `-90` stands the block up whole (a quarter turn, nothing cropped); `runs: false` (v8b) paints no runs down the letters (P7's STOP, where a run down the S read as a dollar sign) | words stencilled in black spray paint without a box, drawn like the box's stencils (the same cut letters, the spray burst as it lands, runs), one mark per stencil: WAIT by the second red light, verse 2's pitch lines. `smear`: a body running through it from `at` drags its letters along in streaks (each row's wet paint pulled up to `length` px, thinning out; the letters thinned where it was pulled from) over `over` 16ths, and they stay smeared |
| `blink` (prop) | `lights` (an id from obstacles.json `lights`, or a list), `on` (kick: snare, hat, impact, beats, 8ths, or a list of times), `from`, `to`, `dim` (0.08), `min` (the hits' least strength), `every` (1), `stagger` (16ths, 0) | street lights blinking on the music: on each hit the light's glow and its lamp's lit pixels (a light with a `layer`: its lamp in that layer, which can be a small parallax-0 layer round a lamp in the plate) dim to `dim` for that photo; `stagger` makes each next light in the list blink that many 16ths later (a chase along the roofs). Full Stop: the roofs' red warning lights on the kicks as he passes, the lit sign box |
| `cam` (prop) | `from`, `to`, `zoom` (1.0), `dy` (px, 0), `dx` (px; the follow's lead by default), `ease` (16ths, 4) | a camera move on top of the follow framing over a span: `zoom` pulls back (1.7), a negative `dy` frames higher (a tilt up: the wall towers over him), each eased in from `from` and out before `to`, then smoothed by the camera like any framing. Full Stop: the scaffolding walkway (bar 70). Any template's framing hint (`dx` or `cx`) may also carry `zoom_v`, `dx_v`, `dy_v`: the 9:16's own zoom (the crop is the base times it, not the muted 16:9 zoom) and offsets (not scaled), for a framing both aspects share (the ending's push-in) |
| `skin_look` (prop, skinlook.py) | `from`, `to` (the song's end), look overrides (`rim_px`, `strokes`, `ink`, `primer`, `rim_gain`, `rim_break`, `rim_min`, `rim_wide`, `wide_scale`, `trail`, `trail_col`, `trail_blur`) | v7 (the lead's pick A): after he leaves the wall, every skin on its photos painted opaque: flat black emulsion with the drawing's dry-brush strokes in dark grey, the white soles and socks as primer, a thin rough off-white rim (3 px); the ghosts, droplets and smears as before; outside it the skins as before (the alley leap). The rim follows the camera: never thinner than `rim_min` (3) px of the full-size frame, `rim_wide` (4) once the frame is zoomed out (below `wide_scale`, 1.05 screen px per street px: 16:9 past about 1.2x). `trail` (0: off) paints the skins of the last `ghosts` 16ths as a pale primer smear, `trail` times render.py's ghosts (the big leap's flip in the wide frame). Overlapping stretches: the one that starts last wins |
| `paint_in` (prop, reveal.py) | `from`, `to`, `front` (x: painted in up to there at `from`), `on` ([kick, snare]; also hat, impact, beats, 8ths), `min` (0.5), `ahead` ([0.33, 0.6] of a 16:9 screen at zoom 1), `dur` (2.5 photos), `reveals: [{at, x: [x0, x1], y, dur, dir: right or down, weight}]`, `grade: [[time, saturation, warmth]]`, `trail: {from, sat, warm, ramp}`, `frame_v` ({zoom: 1.3, at: 0.3}; false) | v8 (the treatment's A): the world ahead of him is pale primer with faint graphite pencil lines tracing the plate's main edges (an edge map of the plate), and on each hit (none inside the song's stops; the hit that ends a stop paints far enough to carry him to the next) a giant wet brush stroke sweeps the next chunk into existence in 2 or 3 photos: a few broad swaths a brush's width apart, each with a wet black leading edge and the brush's hair lines, bristle streaks running ahead onto the primer and splatter flying ahead, the plate showing through behind; where a swath stops its rounded end stays as the border's black rim, with drips running down the primer, until the next stroke paints over it. The front after each stroke is `ahead[0]` of a screen ahead of where he will be when the next stroke ends, at most `ahead[1]` ahead of him. `reveals` adds strokes at set times (dir down: a band swept top to bottom: the far roof painted while he flies). What is painted stays. The revealed world takes the `grade` (saturation, warmth) over time, and more of both where he has passed (`trail`, from its `from`, over `ramp` px behind him). The renderer lays it over the world (the plate, its lamps and parallax layers, the wall paint) before the figure in the air; QA's `reveal` rule checks his feet. Meanwhile the 9:16 spring camera frames ahead so the front is in it: `frame_v.zoom` times wider, his torso at `frame_v.at` of its width (the lead's call on P1: him at the left third, the front in the right third) |
| `insert` (prop, truelook.py) | `clip` (project path), `aspect` ("16:9" or "9:16"), `from`, `to`, `sync: [frame, time]` (default: its first frame on `from`), `focus: [x, y]` ([0.5, 0.5]) | v8 (the treatment's D: the straight-down drop): a hard cut to a full-frame clip of another view in place of the photos from `from` to `to`, for one aspect (one insert per aspect). The clip plays at its own speed, a frame per photo at its time (no slow motion), `sync` putting one frame on one time (the landing on its word); a stretch needing frames outside the clip is an error. The frame fills the photo (scaled to cover it, cropped round `focus`), with the photo's rain over it; the MUTO look follows as for every photo. A cut back out is the next segment's `snap: true` |
| `screen_cues` (prop, screenlevels.py) | `dir` (screens8), `on: [[screen or group, time]]`, `flare: [[id, time]]`, `pulse: {from, to, on (8ths, 16ths, beats or a hit kind), hi: [a, b], lo: [a, b]}`, `kill: [[id or group, time, photos]]` or `[[id, smeared id]]`, `smear: [[id], [id, time]]`, `smear_n` (2 photos), `smear_y` (0.5), `smear_die` (4 photos), `rise` (2 photos), `overshoot` (1.3), `flare_k` (1.6), `flare_s` (0.25 s), `flicker` (0.4) | v8 verse 2 (the treatment's E): a wall of screens, each at its own level over time from world.py's stored light per screen (`<street>/screens8`: its face at full resolution and its spill and wet-ground reflection at half, float16 linear; the street with screen i at level k is the street less (1 - k) times its light; 0 dead, 1 as built, above 1 a flare). Dead before its `on`, then lit over `rise` photos with an overshoot on the first; a `flare` peaks at `flare_k` and eases back over `flare_s`; in a `pulse` stretch every lit screen is `hi` on each hit and `lo` between (each eased from its first value to its second over the stretch); a `kill` flickers to `flicker` on the photo before, then dies over that many photos. A `smear` is his hand's wet black stroke across a screen's face at hand height (`smear_y` of the face from its top), dense black with the fingers' streaks, thinning at its leading edge, a thin sheen along it (so it still reads on the dead glass) and drips running down from it, staying: without a time its front follows the front of his body at the face's height photo by photo (from his figure's matte on each photo, never going back) once he reaches the face; with a time it starts at his hand then (the front of his body there) and spreads back to the face's left edge over `smear_n` photos. A smeared screen stays lit under its stroke and dies `smear_die` photos after the stroke is complete (with the flicker before); `kill: [[id, smeared id]]` dies with that smeared screen. The smears are tracked when the photos are rendered (render.History's `fig_ev`: the event of his figure on each photo), and the screens' lights on the paint follow them then. The screen's light on the paint (its obstacles.json light: `screen: <id>` or the id `scr<id>`) follows its level, and the wall's light on the paint round it scales with the street's brightness |
| `flicker` (prop, windows.py) | `from`, `to`, `amount` (0.1), `cell` (10 px), `thr` (0.2) | the city's lights flickering faintly on every photo: the plate's warm bright pixels (linear luminance over `thr`, red over blue) each brighten or dim by up to `amount`, a new draw every photo per `cell` px of street (cubed: most barely move, a few dip); for a held moment that is not a paused video (Full Stop's last hold). Renderer.photo calls `windows.flicker` on the plate box after the windows blend |
| `door` (prop, doors.py) | `id` (a door in obstacles.json `doors`), `open: {from, to, angle}` (degrees, 80), `shut: {from, to}`, `latch` (degrees, 1.5), `inside`, `gone` | v8b, the outro (P7 "The exit"): the stair house's door opens for him and shuts behind him. The street's `doors` (`id`, `cell`: the world cell it is on, `open`: the cell's open twin, the same photograph with the door open and the light inside on, `leaf: [x0, y0, x1, y1]` in street px: the closed leaf, which is also the doorway's opening, `hinge` (left), `light`: a scene light the door scales by how far it is open, its own strength the fully open one, `persp` (0.12)) are built by `python -m aimv.paint.doors <project> [street]` into doors/<id>.png (the open twin on the street, gain-matched to the world as windows.py does), doors/<id>-closed.png (the world as built: the leaf is cut from it), doors/<id>-spill.png (where the open twin is brighter outside the leaf: the light thrown out of the doorway) and doors/doors.json. The angle: pushed open from `open.from` to `open.to` (quick, easing out), held open by the closer until `shut.from`, the closer's sweep (three quarters of its time, easing into the latch zone at 12 degrees), the latch zone (to 3 degrees), the snap shut on `shut.to`, a rebound of `latch` degrees over 0.12 s. Each photo: through the doorway the open twin where the leaf is not, the leaf drawn from its hinge at width W cos(angle), its free edge shorter by `persp` sin(angle) (it recedes), darker as it turns from the light (0.55 + 0.45 cos); the spill blended in by how far it is open (sin(angle) over sin(the open angle)), the paint's light round it with it, and the door's light scaled the same. From `inside` (when he crosses the threshold) his figure in the air shows only through the doorway's opening, less the leaf; from `gone` nothing of him shows in the door's box (the last drawing would otherwise be held at the jamb) |
| `screens` (prop) | `off: {group: time}`, or `order: [group, ...]` with `from` (the first) and `every` (beats); `fade` (16ths, 0.5) | shop windows' screens switched off (obstacles.json `screens`): from its time each window's band turns to its off version (the screens dead, their glow gone from the pavement and the road), and the paint round it dims with its light. Full Stop's verse 2: "Turn it off", one window a beat along the row, ending at him |

Every preset takes `snap: true` (the segment's first photo jumps to its framing instead of easing in), `x_min`
(street x: the crop's left edge never left of it, 30 px of margin as the street's own rule: a framing pinned at the
street's end, whatever later street edits do to that end; v8's cold open, at main's end moved +4,100) and `x_max` (the
crop's right edge never right of it, 30 px in: the frames move left by the smallest smooth shift that covers every
photo of the segment, eased in and out over a second round them, so the camera's own motion, punches and nudges stay and
the ease never cuts him outside the segment; v8's box scenes kept off the downpipe and the scaffold that P2b's
street cuts brought closer). On top of
the preset: MUTO-step smoothing (62% of the way per photo), kick punches (3.5%, 8.5% on an impact), a small roll
on strong snares, a little hand-held noise. Templates add per-photo hints (the leap zooms out and snaps for the
flight, the landing punches in, the pound and the strikes in the box punch harder on every hit: `punch`; the strikes
and the explosion shake the camera a little: `shake`, px of hand-held shake on that photo; each stencil punches in
a little as it lands; the leap from a clip frames its arc with `cx`, eased photo by photo, which the 9:16 does not
take: `follow_v`).

### Moves

Moves are in time order, one template each: `- template: {args}` or a bare `- template`. Every move takes `poses:
{role: name}` (use other drawings), `dx`, `dy` (shift all its poses), `scale`, `vary: false`, `label`, `pause:
false` (painted at song time through a song stop: the stop in p6, set to the words in the silence), `behind: <obstacle>`
(may pass behind it), `still: true` (a deliberate stillness: the QA's hold rule leaves it alone), `ink` (its dark paint's albedo factor:
0.3 makes the coil before the drop as black as the flight skins, not pale on the red-lit primer), `boost: [gain,
gamma]` (its drawings' paint denser, the dry-brush streaks closed up), `fill_holes: N` (its drawings' interior holes
up to N px closed with the density round them: the matte's gaps where the wall showed through, a light speck in the
hood's opening that read as an eye) and `trail: light` (its ghosts and paint-outs wash away completely and none of its drawings is
a keeper: the walk-in past the row of boxed figures, the sit into the box, so no trail is left over them), `skin:
true` (or a time: from then on) makes every wall pose of the move the J2 skin in the air (a soft shadow on the wall
behind him, a trail of `ghosts` pale skins, 3; `glow: [r, g, b]` the rim colour), never paint on the wall: off the wall
for good (Full Stop's free section, from the peel-off), `grow: [time, n]` (back on the wall at `time`, a
landing: the primer under its next n wall poses arrives gradually, small round him and growing, as the J2 leap's
landing), `keepers: true` (every wall pose of the move a keeper: it fades slowly and leaves a pale paint-out, so the
trail grows longer) or `keepers: N` (every Nth), `trail_in: <seconds>` (v8b: its trail fades in over its first
seconds instead of switching on with its first photo; see the Paint model). YAML 1.1
reads the keys `on` and `off` as booleans; the choreography reads them back as words (`{on: snare}`). Positions
are street x of the anchor (numbers, `feet` for the ground line, or obstacle coordinates such as `ac.x0+20`,
`ac.top`, `pipeB.pole`). Template geometry comes from prototype 6's hand placements, relative to the obstacles'
contacts, and scales with the figure size.

| Template | Arguments (defaults) | What it paints |
|---|---|---|
| `parkour` (v7: see Physics) | `from` (right after the previous move), `from_x` (where the previous one left him plus his speed), `v0` (m/s, or `carry`: the previous move's), `gait` (sprint), `cycle` (a sequence, or per gait; `auto`: v8, each run stretch's cycle by cadence), `to` (up to the next move), `tol` (0.10), `route: [items]` (`run: {gait, x, at, brake, accel}`, `jump: {gap or edge or x, sequence, rise, at, roll, flight, air}`, `drop`, `hop: {obstacle, sequence, rise}`, `vault: {obstacle, sequence, loss, lead}`, `flip: {..., tuck, dir}`, `clip: {sequence, from, to, speed, obstacle, catch_x, x, key, at, keys, ground, dy, lift, flight, edge, ballistic, sway}`) | his route by physics: real-time runs at a gait's speed, stride-true, ballistic jumps landing where the arc meets a surface, landings and rolls by the fall, vaults and clips at his speed; anchors on the music, one speed factor per span, an error with the route's shortfall when a span cannot fit (not when the nearest allowed speed still meets its anchor within 12 ms: a short run before an anchored clip barely moves its time). A move that starts running takes the stride phase where the move before left it, advanced over the ground between that move's last photo and its own start (no stride held at a join). A jump's `air: [[sequence, first frame, last frame, seconds], ...]` (P6's big leap: flipleap_GF's own turn and opening, then roll_RL's drop) shows those runs of drawings in order over the flight, each its share of the time (all scaled to the flight's own time, which physics sets), the last drawing the landing drawing (its feet meet the roof); no procedural turn. A route clip's `lift: [[first frame, last frame, px], ...]` moves those clip frames down by px street px (P6: the precision stick and the crate step, drawn standing above raised tops). A flip's `tuck` drawn part way round the turn (flip_FL's FL-048, head down half way: its sequence's `turn: {tuck: 180}` in poses.spec.json, or the step's own `tuck_turn` in degrees) is turned that much less, so it shows as drawn at that point of the turn and upright at the turn's ends (no half turn out of phase, no snap where the take-off and landing drawings meet it); 0 by default. A jump, drop, hop or flip step's own `flight` (skin or paint) wins over the move's and the file's for that step's take-off photo, its strides over the edge and its flight (P2b v8g: the leap between buildings as v7's J2 skin, every other flight of the part paint on the wall); its landing is back on the wall as ever. A clip's `keys: {key: time, ...}` (v8, P5's stacks: a landing on every kick) pins several of its keys to song times: the first anchors the clip (in place of `key` and `at`), the frames between two pins play to fit them (a stick held longer, a flight squeezed), as shot (times `speed`) before the first and after the last; their times must rise with their frames, and with a `flight` no pin may fall inside it (pins up to its take-off: a swing to its release, then the flight's own time). A move's `primer_clip: [x0, y0, x1, y1]` (null: open on that side; v8c, p5e) keeps the primer it lays inside that street box: its poses' patches, their paint-outs and a stamp's primer, the sides cut with a dry brush edge (his swath trimmed so his step out of the wall carries him off it as he becomes real). A move's `wash: false` (or `wash: {from: time}`, its poses from then on; v8c) keeps the rain from washing its paint: a pose held as paint for seconds stays as black as when painted (p5c's crouch on the parapet through the first stop). A run item's `brake` (m/s2, v8b) is how hard he slows to its gait (the solver's A_DOWN, 6, otherwise), until the next run item: the slowdown into a vault spread over the last two strides (P7). A run item's `accel` (m/s2, v8b) is likewise how hard he speeds up to its gait (A_UP, 4, otherwise): a run between two close moves building its speed sooner out of a landing, so it has room to ease off into the next one (P7's runs after the hatch). A jump's roll (`roll: true` or auto) takes `roll_to` (the roll clip's key it ends on, `run`), `roll_speed` (played that much faster than shot, 1) and `roll_v` (m/s: he slows evenly through it to that speed instead of rolling on at his landing speed; v8b, P7: up out of the big leap's roll at about 2.6 m/s, into the drive). A clip's `glide: [k1, k2]` (v8b, the duct slide) lays its travel from key k1 to key k2 as one constant deceleration over the same distance and times (pinned by `x` or as shot), from the speed that distance needs down to the clip's own pace into k2: a slide that keeps the run's speed into its drop instead of the shot's slow approach. A clip's `flight` with `ballistic: true` (P5's lache: the release) flies from its take-off frame at the clip's own velocity there (both axes, from its two frames before it, as shot) on one gravity and puts the landing frame where that arc meets a surface (a level's feet line or a prop's top below where he left, or below his highest once he has risen; none over a gap; v8c: his landing frame's feet coming down onto it from above, so a landing frame that stands taller than the take-off one is not down at once on the parapet he steps off), instead of where the placement puts it; the frames after it follow from there. (v8c, p5c's step-off) Such a clip with `wait: true` and a single `key` in or after its flight (`key: land, at: <the impact>`) waits in its first frame so that the key, timed through the real flight, lands on its time: the standing still before it takes the difference, the flight is never stretched. A route clip's torso never goes back (a run's rule); with `sway: true` (all its frames) or `sway: <key>` (its frames up to that key) its frames keep their x as shot from its placement, back as well as forward, and never back from there on (v8c, the director on pipe B: a climber's torso swaying round the pole the clip is placed on, so his hands stay on it; those photos are not backtracks to QA). On a sloped level (its feet line steeper than SLOPE_MIN, 2%: P6's beam at 5.6 degrees) a run drawing's feet line is taken at its lowest point (his torso x plus the drawing's foot offset), not at his torso, so a trailing toe does not float above the slope nor a reaching foot sink into it; flat and gently sloping lines keep the torso's, exactly as before. The last strides (0.35 s) into a clip that stands on a level (`ground: <level>`) run on that level, not on the ground under his torso when it is already past the level's edge (P5's stacks: the clip starts with his torso over the step down, his foot still on the run-up roof). A run stretch that starts slower than DRIVE_V (3 m/s) right after a clip (a get-up: a mantle, a squat, a stick) and has to build by more than 1.5 m/s starts with the drive (v8b, the director's notes at 107.0 and 158.7: never full speed from a crouch or a mantle): the drive clip's drawings (DRIVE: drop_P5D 43 to 70, rising out of a deep crouch into four drive steps, 1 to 4.2 m/s, or the move's `drive: [sequence, first frame, last frame]`; `drive: false` for none) in real time from the frame where the clip is as fast as he is, his torso where the solver's acceleration puts him (A_UP), then the cycle in the stride phase whose planted foot matches the drive's last drawing; the drive's photos are tagged pk.drive; a stretch too short for the drive (its length plus 0.3 s) runs without it. `cycle: auto` (v8, PLAN-v8 A) picks each run stretch's cycle from run_SW, run_RW, run_SP and run_RN (those in the library) by its speed: the one whose cadence (speed over its step) is nearest a runner's natural 2.55 + 0.13 v steps a second (3.2 at 5 m/s); below 2 m/s a walk. (v8c) A short stretch that is mostly its drive takes its cycle by its speed after the drive, so it never walks at a run's pace (p5b's first steps to the wall). ch.notes[move]["runs"] reports every run stretch (speed, cycle, step, cadence, natural, ratio, photos a cycle); with `auto`, a cadence over 1.15 or under 0.8 times natural is a warning and over 1.35 an error |
| `run` (M0: see Motion; `sequence` picks the cycle, `cycle` the step to start with) | `from` (right after the previous move), `to` (right before the next), `x: [x0, x1]` or `from_x` (from the neighbours: the previous move's exit gap, the next move's entry gap; else `speed`), `speed` (88 px per 16th), `ease` (none, out, in, inout), `every` (the photo step), `contacts` (beats, or `and`), `offset` (the same in 16ths), `cycle` (stride to start with; continues the previous run), `foot_lock`, `max_lurch` (2.0) | run cycle strides, foot contacts on the beats; with the foot lock the stance foot stays where it touched down (a run slower than about a quarter of the drawings' stride per photo cannot hold it, and `check` says so) |
| `walk` | `from` (next beat after the previous move), `to`, `every` (beats), `x` (118 px after the previous move), `stride` (170), `keepers` (true) | the walk cycle, one pose per beat, a row of walking ghosts |
| `brake` | `at` (the next kick), `x` (80 px after the previous move), `hold` (2 16ths) | the brake pose, held with a little slide |
| `crouch` | `at`, `edge` (an obstacle with an `edge` contact) or `x`, `brake_in` (true) | two brake-in poses, then the crouch with his front 8 px before the edge |
| `freeze` | `from` (the previous move's last photo), `to` | nothing: the song stops, the pose holds, paint does not dry or drip, the photos keep coming |
| `leap` | `gap` (a gap obstacle), `at` (J2: when he leaves the wall; J1: the impact he lands on), `land_on` (J2: the next snare), `style` (the choreography's `jump`), `lead` (J1: 1 or 2), `takeoff_x` (90 px before the near edge), `land_x` (300 px past the far side); with a video `sequence`: `takeoff_x` (45 px before the near edge), `land_x` (250 px past), `height` (0), `ghosts` (3), `grow` (4), `settle` (5), `to`, `speed` | take-off at the edge; J2: the flight as a flat paint skin in the air (rim of the gap's glow, spray and drips falling), J1: take-off burst, a whip photo on the 32nd (or the 16th with `lead: 2`), landing splash on the impact; landing with splash stamps, two recovery poses; camera zoom-out and snap. A video sequence plays the clip's flight between its `takeoff` and `land` keys on one smooth arc with a pale ghost trail, the camera wide on the arc (Motion: the J2 leap from a clip) |
| `vault` | `obstacle` (a `top` contact), `plant_on`, `land_on`, `sequence` | legacy: approach, jump, reach, hands planted on the top on the hit, legs through, drop, land four photos later. M0: the sequence's run-in (to `takeoff`), an arc to the hands' plant on the top on the hit, the body over the hands (to `push`), an arc to the landing past the far edge (`land`), the strides out; a video sequence plays the clip round its key frame (see Motion) |
| `slide` | `under` (an `under` contact), `at` (the drop), `up` (six photos later) | drop into a slide, slide under (it spans the obstacle, or is centred on a narrow one), pop up on `up` |
| `climb` | `pipe` (a `pole` contact), `grab`, `every` (8ths), `rise` (48 px a step), `hands_y` (556 px above the feet), `to` (a beam), `lunge` (the next snare), `until` (without `to`) | jump, reach, grab on the hit, climb hand over hand, lunge to the beam |
| `hang` | `beam` (a `beam` or `bar` contact), `from`, `to` (up to the next move), `x` (10 px past the previous move), `traverse` (33 px per 16th) | hangs and swings along the beam (hands on it), a slightly bigger swing each 7 16ths |
| `swing_release` | `bar`, `release_on`, `land_on` (the next kick), `x` (260 px past the previous move), `land_x` (730 px further) | kicks off and releases on the hit, flies along the wall, lands with a splash |
| `step_up` | `onto` (a `step` contact), `at`, `land_on` (six photos later) | leaps onto the step, lands on it, pushes off, lands on the ground on the hit |
| `pole_swing` | `pipe` (a `pole` contact), `grab_on`, `land_on` (six photos later), `span` (1.0: how far the flight carries) | reach, grab and swing round the pipe, release, fly, land on the hit |
| `leap_off` | `onto` (a `step` or `top` contact: a bin lid), `at`, `land_on` (five photos later), `land_x` (560 px past the top), `jump_x`, `height` (230 px) | jump onto the top, crouch on it, push off; the first flight photo is the highest point (a song stop right after it freezes him there); land with a splash |
| `box` (prop) | `id` (box), `x` (centre), `bottom` (26 px above the ground), `old` (0), `figure` (`{pose, fill, align, mirror, old, at, paint_in}` or a pose name), `keys: [{at, w, h, thick, bulge, bulge_at, bulge_h, x, bottom, old, paint_in, drop, sides}]` (a key's `sides: {top|bottom|left|right: time}`, v8b: only those sides, each painted, wet and dripping from its own time, with the full outline's strokes: the cold open C3's box closing on him side by side), `stencils: [{at, text, x, y, h, align, rot, punch}]`, `cracks: [{on, from, to, at: [x, y], size, grow, min}]`, `hits: [{on, from, to, min, side, bulge, y, flakes, grow} or {at, ...}]`, `peel_from`, `photo` | the box of board B1: a rectangle of thick white primer lines with drips, repainted on each key (a key keeps the previous key's values unless it changes them; the replaced outline washes out). `bulge` pushes the sides out locally where his hands press (a share of the width, a Gaussian bump at `bulge_at` of the height from the top, spread `bulge_h`; a few percent keeps it a rectangle). The rules stencilled in black spray (text is ours: a bold condensed sans cut with stencil bridges, overspray, runs); crack growths on hits. Coordinates may use the box (`box.x0-30`, `box.cx`, `box.ix1`, from its first key). `photo: true` adds a photo exactly on each key and stencil time. `old` (0 to 1, per box or per key): painted long ago, dry from the first photo, its drips full length, faded and streaked by the rain. `figure`: someone else curled in it, a drawing fitted to the first key (`fill` 0.9) and painted in weathered grey paint, as old, still all film (the others in the row of boxes). `paint_in` (seconds, on a key or the figure; `figure.at` says when the figure is painted): painted in with a brush sweep from its top left, a third of it done on its own photo (the row of boxes painted in on the beats, each figure an 8th after its box). Each stencil lands with a spray burst (a speckled cloud of black mist round the letters, settling in about half a second) and its runs growing, and the camera punches in a little (`punch`, 0.03). `hits`: hits from inside, on hits (`on`, `min`) from `from` to `to` or once `at` a time: the side (`side`, right) gives by `bulge` px at `y` of the height from the top and springs back over a few photos, and `flakes` primer flakes fall from it to the wall's foot; `grow` makes each hit a share bigger than the last. `peel_from`: the stencils start to crack and peel (half as fast as after the shatter) from then (the kick roll). `check` reports stencils that overlap each other, the box (as it is at any time while both are there) or a parallax layer (the light pole, at each photo's parallax), or that a box-camera frame cuts in either format. A key's `drop` (v8, the box as the villain: `{height` (1.6 times the box), `over` (0.26 s)`}` or true) slams it down onto that key from above, a photo at a time and ever faster (as under gravity), each falling outline gone as the next is painted, no drips; the key lands as any key (its `hits` knock the flakes off) |
| `inside` | `box`, `from`, `to`, `roles: [[time, role], {at, role, align, fill, max_scale, scale, push, crack}, {at, clip, keys, speed, side, strike: {on, min, hit, back, push, crack: {size, grow}, punch: [first, last], shake, flicker, dim}}]`, `fill` (0.88), `max_scale` (1.15) | the figure inside the box, a pose per photo, the role from the schedule (a role with several drawings alternates: breathing), scaled to fit the box as it is at that photo, standing on its inner bottom, centred or pressed against a side (`align`, the bulge included); `push` leans a held pose into its side every other photo; `crack: {on: snare, min, size, grow}` grows a crack where the pose touches the side on those hits. A `clip` entry plays a front-view clip of him in the box (a video sequence with a guide box, `boxed`): every frame put in the box as it is on that photo (the clip's box outline on the street box, one scale from their heights, so he shrinks with it on the rules), its `keys` pinned to their times (the words), `speed` times the clip's speed outside them; past the clip's end its last three frames go to and fro (he keeps pressing), before its start its first frame holds. `strike` replaces the clip's timing with the music's: a photo on a hit shows the next `hit` frame (a list in the sequence: punches, shoves) and leans in by `push` px, every other photo the next `back` frame, so each punch lands on a hit and the in-betweens are dropped; `crack` grows a crack where his fist or shoulder meets the side (`side`) on each hit, bigger each time; `punch` and `shake` hit the camera, growing over the hits; `flicker` dims a light (its id) to `dim` (0.15) on each hit. `blows: {keys, push, crack, punch, shake, flicker}` on a keyed clip makes the photos of those pinned keys the hits (box-batter: a shoulder slam, a kick, a jump kick, a flying charge, one a bar on the snare). `strike: {segments: [[approach, hit], ...], tail: [from, to]}` plays the clip in pieces instead: before the first hit the clip from its start to the first segment's hit frame, between two hits the next segment (cycling: a step back and a slam, a recoil and a punch) spread over the photos up to the hit, which shows its hit frame, and after the last hit the `tail` (the wind-up that hands over to the next clip); no drawing on more than two photos in a row. `hit` and `back` take a list of clip frames as well as a list name (each the nearest kept frame), and `tail: [from, to]` works with them too: the photos after the last hit play those frames, ending on `to` (box-pound: a punch on every kick of the roll, pulled back between, its last photo box-breakout's first frame). `frames: [clip frames]` plays those frames one a photo from the entry's `at`, in any order, the last one held (a clip played backwards: box-mouth's fold reversed, so he unfolds from the tight ball into the hand over his mouth over 4 photos; the next entry carries on from there); `start: <key>` starts a keyed entry at that key's frame instead of the clip's first. `lean: [key_a, key_b, degrees, px]` leans him (rotated, shifted) on the clip frames between two keys, toward something outside the box (the hood turned to the red light). The inside template visits every photo, the extra ones on the words too, and a near-still stretch of a clip breathes between neighbouring frames instead of holding one. v7 (the director at 32.5: the punches went through the wall): the box's sides stop him: a frame whose paint would pass a side's line is moved back by that much (12 px inside its outer edge, for the jitter), so a fist meets the wall (`wall: false` turns it off; the `fist` rule checks it) |
| `pound` | `box`, `from`, `to`, `every` (8ths), `side` (right), `punch: [first, last]`, `shake` (px, 0), `size`, `grow`, `fill`, `body` (a set size instead of fitting) | a strike on every step (`strike`), the wind-up a 16th before (`windup`; an extra photo when the segment is slower), the fist on the side, a crack growth there on each strike (growing), the camera punching harder as it builds |
| `burst` | `box`, `at`, `land_on` (the next impact), `land_x` (760 px past the box), `height` (300), `fly` (0.65), `punch` (0.1), `power` (1), `splats` (0), `stencils` (0), `shake` (0), `flicker`; with a video `sequence`: `apex_on`, `apex_x`, `to`, `speed`; `frame: box` (a front-view clip in the box): `from`, `windup_on`, `grow` (4), `run_sequence` (run_SW) | the shatter: the outline and its cracks burst into flying white primer and black paint shards; he bursts out shoulder first along a rising arc (`break_out`) and lands into a run (`break_land`, splash). A song stop during the flight freezes him mid-burst with the shards hanging round him; they fall after it and settle as spatter at the wall's foot; the broken remains of the outline stay and wash. The explosion (Full Stop's "Full stop"): `power` throws more and faster black flecks, `splats` drops of paint thrown across the wall round the burst point (the far ones later, so it spreads over several photos), `stencils` the share of the stencils' letters (the nearest first) cut into pieces that break off and fly, `shake` px of camera shake, `flicker` knocks a light out for a photo |
| `clip` | `sequence` (a video sequence), `keys: {key: time}` (default its first frame on `from`, else after the previous move), `speed`, `from`, `to`, `obstacle` (its guide shape on that obstacle), `x: {key: x}`, `bottom: {key: y}`, `stretch`, `x_offset`, `splash`, `arc: {from, to, height}`, `back: true` (between its x pins he steps back as the clip does, not only forward: a bounce off a wall, the cold open's crash into the box; QA's backtrack rule leaves those steps alone), `ground_body: [key_a, key_b]` (a slide: his body, not his lowest point, lying on the ground line between those keys: the 80th percentile of the drawing's column bottoms 8 px under the feet line, its lowest paint never past the wall's foot, eased in and out), `tag` | the clip's frames one per photo (Motion: video moves): a walk's end, a sit into the box, a free leap (on one smooth arc with `arc`), hops over bollards, a slide, a brake into a crouch or a walk |
| `pose` | `at`, `pose` (a name or a role), `x`, `y` (feet), `anchor` (foot), `rot`, `layer` (wall or air), `keeper`, `mirror`, `flipv`; `strokes: [{at, box: [x0, y0, x1, y1], angle, over, splats, drops, sweep}, ...]` (a stroke's own over, splats, drops; `sweep` 0 to 1: the brush's path running on past him, a long sweep), `over` (px, 20), `splats` (40), `drops` (90) | one pose by hand. With `strokes` it is painted by giant wet brush strokes slamming onto the wall one by one (`props.stroke_masks`; Full Stop's cold open, review note): each stroke on its time reveals its share of him (`box`: shares of his painted box; `angle`: its slant), the last one whatever is left, so he is whole; each stroke lands on a broad bright white primer swath laid on the same photo (a roller pass, wider and longer than the stroke, so the black sits on clean white even on a dark wall; it stays on the wall as the residue behind him; `primer: false` for none), runs `over` px past his silhouette with bristle streaks and throws splatter off its end, and that overshoot stays on the wall (MUTO residue); wet drops fly off along the stroke, already in the air on its own photo (the first one is the thumbnail); each stroke's drips run from its own time; no primer patch under him. Mark it `still` (one drawing over many photos). `drip_in: {to, streaks, lead, amp}` (v8b, cold open B; not with strokes) forms him instead by paint running down the wall from `at` to `to`: a ragged front (`amp`, 0.07 of his height) runs from his top to his feet, everything above it painted, `streaks` (24) thin runs of paint leading it by up to `lead` (0.3) of his height (never past his own outline's foot in their column), quick at first and slowing as it reaches his feet, so he is whole on `to`; his drips grow from `at`; no primer patch |
| `stamp` | `at`, `stamp` (a name or a role), `x`, `y` (its centre), `w`, `splash`, `primer` | a paint stamp by hand; `primer: true` lays a primer patch under it that fades as he paints on, `primer: keep` one that stays on the wall like the stamp; `swath: true` paints the stamp's own shape in white primer (bright and wet, then staying) instead of black: a band or a block painted on the wall on its beat (v8's rules; a `marks` stage takes it as `{swath: true}`) |
| `marks` (prop) | `stages: [[at, stamp, x, y, w], ...]` (each a time, a cell of a stroke sheet, its paint's centre and its width in street px), `primer` (false; `keep`: the primer patch stays instead of fading, as a stage's own sixth item `{primer: keep}` too), `photo` (false: true puts a photo on each stage), `label`; a stage's own sixth item may also set `splash: true` (droplets fly off it: `drops` of them, 40, `lead` s already in the air on its photo, 0: the cold open's splat mid-burst on the first photo), `drips: [n, length px]` (its own runs of paint, instead of 6 to 13 of 40 to 110 px) and `swath: true` (painted in white primer) | painted strokes in stages, in time with his arm (marks.py): a stroke sheet (a `stamp` sheet, one cell per stage, each the stroke painted up to there with the drips it has grown) placed stage by stage, each stage a stamp over the ones before; the mark stays on the wall (settling at 62% density) for the rest of the film. A prop, so the strokes overlap the clips that paint them without joining the runner's chain. Full Stop's smear down the far wall and his word NO |

How moves fit together: a dry pass gives every move its times and street extent, then each move is placed knowing
where the previous one ended (its last x and exit gap) and where the next one starts (its first x and entry gap).
So a bare `run` fills the time and the street between two anchored moves. A move right after a bare `run` should
give its time explicitly (a bar.beat, or a hit with `@` or `>`): its own time decides the run's length, and `check`
warns when the two disagree.

Prop templates (`@template("box", prop=True)`) paint the set rather than the runner: they are placed before the
runner's moves and stay out of their chain (a prop has no time or street extent of its own). They register a prop on
the choreography (`c.ch.props[id]`); other templates act on it (inside, pound, burst) and the renderer paints it.

Adding a template (slip and slide down a wall, peel off the wall, float): write a function in
`src/aimv/paint/moves.py`:

```python
@template("slip")
def slip(c: Ctx) -> None:
    """Slip and slide down a wall from `at` to `land_on`, leaving a smear."""
    q0, q1 = c.time("at"), c.time("land_on", default="kick")
    x = c.coord(c.arg("x"), "x") if c.has("x") else c.x_after_prev(40 * c.u)
    for k, q in enumerate(np.arange(q0, q1 + 1e-6, c.ch.step_at(q0))):
        c.wall(q, c.pose("slip", k), x + 6 * k * c.u, c.feet - (300 - 40 * k) * c.u, rot=-4 * k)
    c.stamp(q1, c.pose("drips"), x, c.feet - 120 * c.u, 300 * c.u)
```

Then add its roles to the project's `poses.spec.json` and a row to the table above. The context gives `c.time`,
`c.ob` (an obstacle, with `.contact(name)`), `c.arg`, `c.coord`, `c.pose`/`c.poses` (roles), `c.wall`, `c.air`,
`c.stamp`, `c.freeze`, `c.cam` (camera hints: `zoom`, `snap`, `dx`, `dy`, `cx`, `free`, `punch`, `shake`,
`follow_v`), `c.light` (dim a street light by its id on a photo: a flicker), `c.light_out` (knock a light out from
a photo on), `c.burst` (a paint explosion at a point: `props.Burst`), `c.photo` (an
extra photo off the grid), `c.label` (for maps and contact sheets), `c.prev`/`c.next` (neighbours' extents), `c.u`
(body unit), `c.at` (a time nested in an argument), `c.song` (move time to song time, for props), `c.feet` (the
ground line where the move happens). Times inside a template are move time (see Song stops); the context maps them.

## Paint model

Every move paints from a random stream of its own (seeded by its file's seed and its place there), so a change to
one move leaves the jitter and the trail of the moves after it alone, and a segment paints the same in the film as
alone.

Per photo, in paint order: primer patches, then the poses (and splash stamps, and primer rolled over old trail),
each lit by the street: albedo x illumination x relief x light colour, plus the plate's wet speculars, light glows,
the paint reflected in the wet pavement. The newest pose is wet (deepest black, glossier, drips growing); older
poses fade per painted frame (keepers slowly: beats, landings, grabs; in-between poses fast), get rain-washed and
dry. Paint is masked by the occluders and the no-paint zones. Weather per photo: puddle ripples, rain streaks, now
and then a passing car's light. The J2 air layer and the J1 burst, whip photo and splash are as in prototype 6.

Primer modes (`primer`, fade rules in `primer_fade`):

| Mode | What | Fade rules (defaults) |
|---|---|---|
| `all` | a patch under every pose (prototype 6): at close framing they pile up into a long pale band | `floor` 0.5, `tau` 0.45 s |
| `keepers` | patches only under keepers; in-between poses go straight on the wall | `floor` 0.3, `tau` 0.6 |
| `none` | no patches: dark ghosts on the bare wall (low contrast on dark brick) | |
| `ghosts` (default) | a patch under the newest poses that fades away, and every pose painted out with primer (its silhouette, a little larger) two poses later: the trail is pale figures straight on the wall, as on the C4 board | `floor` 0, `tau` 0.2; paint-outs `out` 0.72, keepers stay at `out_keeper` 0.5 of that, others start at `out_other_start` 0.35 and wash away (`out_tau` 0.3 s), `out_after` 2 |

`floor` is what a patch fades to once its figure is a ghost, `tau` the time (in painted 16ths) to get there.

A figure repainted in one place (inside the box) would pile its ghosts into a smudge, so `inside` and `pound` mark
their poses `trail: light` (no keepers, not even a move's key drawings; their ghosts and paint-outs wash away
completely), keep their primer patch inside the box (`clip`) and thin (`patch` 0.2), so the box stays an outline on
the bare wall. Any move can ask for the light trail (`trail: light`): the sit into the box does, so the standing
drawings it starts from do not stay as a pale figure behind him in the box.

A move after a light-trail move can fade its own trail in (`trail_in: <seconds>`, v8b: the climb after verse 2's
run, whose trail would otherwise switch on with the hand-over's first photo): from its first figure pose until that
many seconds later each pose's ghost and paint-out are a blend of the light trail's (washing away) and its own (a
keeper's pale figure that stays, an in-between pose's faint floor), the share of its own rising linearly from 0 at the
first pose to 1 (the event flag `trail_w`). Keepers, positions and every random choice stay as they are; past the
ramp the paint is exactly as without it.

In the air (a J2 flight from a clip) the trail is pale too: the flat skins of the photos before stay as pale primer
silhouettes behind the newest one, fading over `ghosts` 16ths and gone after twice that.

`ink` (top level, default 1.0) scales the dark paint's albedo (the figure, splashes, stencils, cracks, black spatter):
below 1 the paint reads closer to black under the same light. Full Stop's film uses 0.8; p6.yaml alone keeps 1.0.

Props (props.py): the box outline is white primer, darker and glossy when (re)painted and drying paler, its drips
growing while wet; a replaced outline washes to a faint ghost. Cracks are black hairlines with a pale chipped edge
(visible on the black stencils too); each growth extends a few of the earlier tips from the same point and starts new
branches there. Stencils are black spray: hard edges with bridges, a speckled overspray halo, a few runs; after the
shatter they crack and peel a little each beat and wash like the ghosts, as do the broken remains of the outline
(with `peel_from` they start earlier, at half the pace). Each stencil lands with a spray burst: its own mark, a
speckled cloud of black mist round the letters (wider than their overspray) that settles in about half a second.
Shards (Voronoi cells of the outline, smaller near the burst point; the burst side flies, the rest stays as the
remains; black flecks from the cracks) fly in front of the wall lit by its light, slow down, hang still while the paint
clock stands still (a song stop), fall, and leave spatter at the wall's foot where they land. The explosion adds
splats thrown across the wall (`splats`, landing over several photos, the far ones last) and pieces of the stencils'
letters (Voronoi cells of the letters nearest the burst point) that fly as black shards, the letters left with holes.
A hit from inside (`hits`) bends the side out locally and lets it spring back over a few photos, and knocks small
primer flakes off it that flutter down to the wall's foot. A light hint (`c.light`, the strikes' and the explosion's
`flicker`) dims a street light's glow on that photo and the lit pixels of its lamp (the light's `layer`).

## Camera and the 9:16 cut

The 16:9 framing is the preset above. The 9:16 cut is its own pass, not a crop of the 16:9: half the 16:9 crop
width (at most the street's height), feet at 80% of the height, the same smoothing, punches and hints (zoom-outs
muted), then a hard rule that the figure's box stays inside the frame with a 5% margin (the crop widens up to the
street height for a wide pose). With motion M0 the 16:9 crop also keeps his box inside (moved only, outside the
box and frame presets): a jump up onto a cabinet outruns the smoothing. `check` reports the 16:9 torso offsets and
any 9:16 frame that cuts him.

## Outputs

`aimv paint render` writes `renders/<name>-16x9.mp4` and `-9x16.mp4` (24 fps, H.264 CRF 17, the song slice, each
photo held from its own song time with the MUTO look: jitter, exposure and white-balance flicker, grain, seeded per
photo so a photo looks the same in any range) and `<name>-steps.json` (every photo's time, address and framing,
render statistics). Photos render in parallel processes (up to 8) in contiguous chunks; `--preview` renders at half
size; `--only` keeps the photos as JPEGs instead of making a video; `--keep` also keeps raw PNGs.

`--debug` (v7) renders `<name>-debug-*` with the physics drawn on every photo, both aspects: the feet lines (green)
and the props' tops (cyan), the obstacle boxes (solid orange, others grey, named) and the gaps' edges (magenta), the
solved arc of the flight he is in (white), his lowest paint against the surface under it (a dot and a line, green
within the `feet` rule, red outside it), and a corner label: time, address, speed (m/s), gait, the drawing's tag,
feet offset and the span's speed factor (OUT beyond its tolerance); in flight the jump's rise and how far his feet
clear what he jumps (a hop's prop top, a gap's far edge). Every render's QA json carries `solver`: the
parkour moves' notes for its window (spans, factors, slack, flights).

## Costs

- Money: none in this tool. Inputs: street plates (GPT Image 2.5 4k 21:9, about USD 0.12 each, about 2,300 px of
  new street each) and sprite sheets (2k, about USD 0.08 each), through `aimv board`.
- The whole song needs about 110,000 px of street (`aimv paint check` estimates it from the sections not yet
  choreographed): about 45 plates, USD 6 to 9 with re-rolls, fewer with stretched and mirrored long walls.
- Time: about 0.23 s per photo per format in one process (look and JPEG included); 8 processes render the 14 s
  window (164 photos, both formats, then both videos) in about 27 s, so a 3-minute song (about 2,100 photos) takes
  about 5 to 6 minutes. Memory does not grow with the song: each process holds one street window of maps, a capped
  mask cache and the history without masks (about 1 GB at peak per process).
- Disk: about 5 MB per 2,048 px tile for the strip and again for the street (about 0.5 GB for the whole song), the
  pose library about 9 MB, renders about 100 MB per 14 s per format at CRF 17.
