# aimv review: notes on a render

A local page for watching a render and leaving notes on it, so the director and the lead work from the same list.

```
aimv review <project> [--port 8770] [--no-open]
```

It serves `projects/<p>/` on 127.0.0.1 and opens the page. Nothing leaves the machine.

## The page

- **Video**: pick any review copy in `paint/renders` (`*-review-16x9.mp4`, `*-review-9x16.mp4`), newest first. A render
  of a window (e.g. `review-final-144.00-180.00`) is placed on the song by its steps log's `t0`, so every time on the
  page is song time. Click the video to play or pause.
- **Timeline**: the whole song with its segments and the notes on it.
  - Scroll on it with two fingers (or a mouse wheel) to scrub through the song.
  - Pinch to zoom in for fine scrubbing; "Show the whole song" zooms back out.
  - Click to jump, and drag (or press I and O) to mark a part.
- **A note** is either:
  - **This part**: the moment at the playhead, or the part you dragged. A snapshot of the frame is attached.
  - **General**: about the whole film.
- **The list** shows the open notes (or all of them), general notes first, then by time. Click a note's time to jump
  there; each note can be marked done, edited or deleted.

Keys (also under "Shortcuts", or press ?): Space play or pause, Left and Right one frame (held, the playhead moves
after 0.3 s and speeds up the longer the key is held, from 0.5 to 30 s of film per second), Shift with them one bar,
1 to 9 a segment's start, I and O the part's start and end, L loop the part, X clear it, G switch between
This part and General, N write a note, Cmd+Enter save, `[` and `]` slower or faster, Esc leave the text box, clear the
part or cancel an edit.

## Files (`projects/<p>/review/`)

| File | |
|---|---|
| `notes.json` | `{"notes": [...], "deleted": [...]}`; each note has `id`, `text`, `status` (open, done), `scope` (point, range, overall), `t0`, `t1` (song s), `bar`, `section`, `segment`, `video`, `frame`, `created`, `updated`. Older notes may also carry a `kind` (fix, idea, like, question) and drawn `marks`. Deleting moves a note to `deleted`; nothing is erased |
| `notes.md` | the same notes, general first, then by segment and time, with the frames, rewritten on every change: what the lead reads |
| `frames/<id>-<time>.jpg` | the snapshots at the render's own size |
| `segments.json` | optional: the production segments (`id`, `label`, `t0`, `t1`); without it the song's sections stand in |

The lead marks notes done as they are addressed (an edit of `notes.json` or the page's Done button); the page reloads
the notes whenever it gets focus.
