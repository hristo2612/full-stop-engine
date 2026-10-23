# aimv: a beat-locked music video engine

This is the engine behind the music video for "Full Stop" by Main Thread: a three-minute, one-take side-scroller in
which a runner painted in black emulsion on the walls of a night street does parkour over the real objects on the
wall (pipes, AC units, ledges, ladders), every photograph held on the song's grid like the stitched photographs of
Blu's MUTO.

Nothing in the film is placed by guesswork. The song is analysed first (stems, beat grid, word timings, sections,
drum hits), and everything after that is written against that grid: a choreography in YAML says which move happens
on which bar, beat or snare, a physics solver turns the moves into stride-true runs and ballistic jumps, a spring
camera follows him, and the compositor paints each pose onto the street as wet paint behind the real objects, with
older poses left on the wall as ghosts.

The song itself was made with an AI music tool; that part is not in this repository. You bring your own song.

The code was written by Claude Code (Opus 5.5) over about a week, directed through creative notes on each render
rather than written by hand.

## What you need to supply

This repository is code only. To make a video you need:

- **A song**: a mixed master (`.wav`, `.mp3`, `.m4a` or `.flac`) and, for word timings, its lyrics as plain text.
- **A street**: a set of wide plates (images) that overlap left to right. The film's plates were generated with an
  image model and outpainted one from the next (`aimv paint street --canvas`), but any panorama you own works.
- **A pose library**: sprite sheets of the figure (one pose per cell on a plain ground) and, optionally, short
  clips of moves (vaults, climbs, wall runs) as video sprites. The film used painted figures on primer and some
  chroma-green clips. The pose library, plates and renders of "Full Stop" are not included.

## Pipeline stages

1. **Breakdown** (`aimv breakdown`): the song turned into timing data. Output contract: `docs/BREAKDOWN.md`.
   - ingest: loudness, true peak, key, a 48 kHz master and a 16 kHz mono copy for the models
   - separate: vocals and instrumental with a band-split RoFormer, then drums, bass and other with `htdemucs_ft`
     (both through `python-audio-separator`)
   - beats: beat and downbeat tracking with `beat_this`, cleaned into a bar grid
   - align: word and line times for the known lyrics by CTC forced alignment (wav2vec2 through torchaudio, or MMS
     for other languages), cross-checked with Whisper word timestamps
   - sections, hits (kicks, snares, hats, impacts, song stops), envelopes per frame, an HTML report and a click track
2. **Street and scene** (`aimv paint street`, `aimv paint scene`): plates stitched into a tiled street, obstacles,
   occluders and contacts marked in `obstacles.json`, masks cached per tile.
3. **Pose library** (`aimv paint poses`): sheets and clips to density mattes with anchors, cycles and roles, plus a
   contact sheet and a canon check that catches drawings that drifted off-model.
4. **Choreography** (`aimv paint check`, `aimv paint map`): the YAML plan validated against the grid and the street.
5. **Render** (`aimv paint render`, `aimv paint still`): every photo solved (physics, camera, paint history), composited
   with the wet-paint model and the MUTO look (jitter, exposure and white-balance flicker, grain), in 16:9 and 9:16,
   with the song slice. A QA pass flags backtracking, floating feet, paint where it cannot go and similar errors.
6. **Review** (`aimv review`): a local page to watch a render and leave time-coded notes. See `docs/REVIEW.md`.

`aimv muto` applies the stitched-photographs look to any video or folder of frames on the song's grid, without the
painted runner.

## Install

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run aimv --help
```

With pip:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/aimv --help
```

`ffmpeg` must be on your `PATH` for audio slices and video encoding. PyTorch is a large download; on Apple Silicon
the breakdown uses MPS where it can, else the CPU.

## Break down your own song

```bash
aimv new projects/my-song --audio path/to/master.wav --lyrics path/to/lyrics.txt --title "My Song"
aimv breakdown projects/my-song          # all steps, cached; --only align,sections or --force to redo
aimv serve projects/my-song              # opens breakdown/report.html with seeking
```

Steps are cached by input hash, so a re-run only redoes what changed. Models download on first use: the separation
models (about 1 GB) into `.models/` next to the source (override with `AIMV_MODELS_DIR`), Whisper and the CTC model
into their usual caches, and `beat_this` checkpoints into `~/.cache/torch/hub`.

The lyrics format is one line per sung line, `[Section]` tags on their own lines and `(parentheses)` for backing
vocals. Optional word timings or section tags exported from your song generator can be dropped into `song/suno/`
(see `docs/BREAKDOWN.md`); they are used only as cross-checks and guides, never required.

## How a choreography works

A choreography is a YAML file in `projects/<p>/paint/choreo/`. Times are written on the song's grid: `43.2.3` is bar
43, beat 2, third 16th; `snare@44.2` is the snare nearest bar 44 beat 2; `impact`, `kick`, `stop.end` and section
names such as `Verse 2` also work, and `57.6s` gives plain seconds. Segments tile the song and set the photo rate
(`16ths`, `8ths`, `beats` ...) and the camera preset; moves are templates in time order:

```yaml
street: main
seed: 31
jump: J2
primer: ghosts
hits: {snare: 0.85}
window: [60.34s, 65.9s]

segments:
  - {from: 45.1, to: 49.1, rate: 16ths, camera: {preset: follow, snap: true}, label: run, vault, run}

moves:
  - run: {from: 45.1}
  - vault: {obstacle: utility, plant_on: snare@47.2}
  - run: {to: 48.4.4}
```

A film is one file that `include`s its parts, rendered as one continuous shot. `examples/choreo/` has this test,
the film's climb (one physics-driven `parkour` move with a route of hops, a wall run, a tic-tac and a facade climb)
and a film skeleton. The full format, every move template, camera preset and prop, the paint model and the world
(streets that grow up and down) are in `docs/PAINT.md`.

```bash
aimv paint check  projects/my-song vault          # validate and summarise
aimv paint render projects/my-song vault --preview --from 45.1 --to 47.1
aimv paint render projects/my-song vault --aspect both
```

## External models and services

| What | Used for | Required |
|---|---|---|
| `python-audio-separator` (BS-RoFormer, htdemucs_ft) | stems | yes, for the breakdown |
| `beat_this` (CPJKU) | beats and downbeats | yes, for the breakdown |
| wav2vec2 / MMS via torchaudio, `ctc-forced-aligner` | word alignment | for word timings |
| `openai-whisper` (large-v3-turbo) | alignment cross-check | for word timings |
| OpenAI Images API (`OPENAI_API_KEY`) | plates and sprite sheets via `aimv board` | optional, paid |
| Higgsfield API (`HF_KEY`, `pip install -e .[higgsfield]`) | images and video clips via `aimv hf` and `aimv board` | optional, paid |

The paint engine and renderer run fully locally and spend nothing. The paid commands read keys only from the
environment or a `.env` file (`.env` in the working directory, or `AIMV_ENV_FILE`; see `.env.example`), check every
estimate against the project's cap in `project.json` (`budget.cap_usd`, default USD 25) before the request, and log
each request to `projects/<p>/ledger.jsonl`. `examples/project.json` shows a project file.

## Tests

```bash
uv run pytest -q
```

The tests build their own synthetic audio, plates and poses; they need no media, models or network.

## Layout

```
src/aimv/breakdown/   song analysis (phase 3)
src/aimv/paint/       street, scene, poses, choreography, physics, camera, paint model, render, QA
src/aimv/look/        the MUTO look on its own
src/aimv/gen/         optional paid generation: ledger and caps, OpenAI images, Higgsfield, choice boards
src/aimv/review.py    the review page
docs/                 BREAKDOWN.md, PAINT.md, REVIEW.md
examples/             a project file and choreographies
tools/render_jpg.py   render wrapper that keeps the JPEG work photos instead of raw PNGs
```

The docs and comments were written while the film was being made, so they refer to its parts (the cold open,
verse 2, "P7 The exit"), its plates and its review rounds as worked examples.

## License

MIT, see `LICENSE`.
