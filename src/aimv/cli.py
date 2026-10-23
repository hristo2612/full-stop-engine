"""Command line: `aimv breakdown <project>`, `aimv new <project> --audio ... --lyrics ...`, `aimv paint ...`."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _new(args) -> None:
    from .project import Project

    root = Path(args.project).expanduser().resolve()
    (root / "song").mkdir(parents=True, exist_ok=True)
    p = Project.open(root)
    if args.title:
        p.update_config(title=args.title)
    if args.fps:
        p.update_config(fps=args.fps)
    if args.audio:
        src = Path(args.audio).expanduser()
        dst = p.song_dir / f"master{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print(f"audio  -> {dst}")
    if args.lyrics:
        shutil.copy2(Path(args.lyrics).expanduser(), p.song_dir / "lyrics.txt")
        print(f"lyrics -> {p.song_dir / 'lyrics.txt'}")
    print(f"project ready: {root}")


def _breakdown(args) -> None:
    from . import breakdown
    from .project import Project

    p = Project.open(args.project)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    breakdown.run(p, only=only, force=args.force)


def _serve(args) -> None:
    from .project import Project
    from .serve import serve

    p = Project.open(args.project)
    serve(p.root, port=args.port, open_page=None if args.no_open else "breakdown/report.html")


def _review(args) -> None:
    from .project import Project
    from .review import run

    p = Project.open(args.project)
    run(p.root, port=args.port, open_page=not args.no_open)


def _board(args) -> None:
    from .gen import board
    from .project import Project

    p = Project.open(args.project)
    page = board.generate(p, args.name, workers=args.workers, dry_run=args.dry_run)
    print(f"board: {page}")


def _keys(args) -> None:
    from .gen import keys

    for k, ok in keys.available().items():
        print(f"{k:<20} {'set' if ok else 'missing'}")


def _ledger(args) -> None:
    from .gen import ledger
    from .project import Project

    p = Project.open(args.project)
    cap, phases = ledger.caps(p)
    t = ledger.totals(p)
    print(f"spent ${t['total_usd']:.2f} of ${cap:.2f}")
    for ph, usd in t["by_phase"].items():
        print(f"  {ph or '(none)':<12} ${usd:.2f}" + (f" of ${float(phases[ph]):.2f}" if ph in phases else ""))


def _hf(args) -> None:
    import json as _json

    from .gen import hf, ledger
    from .project import Project

    def need(what: str) -> str:
        if not args.target:
            raise SystemExit(f"aimv hf {args.action}: give {what}")
        return args.target

    if args.action == "upload":
        print(hf.upload(Path(need("a file to upload"))))
        return
    if args.action == "fetch":
        if not args.project:
            raise SystemExit("aimv hf fetch needs --project")
        p = Project.open(args.project)
        if not args.target:
            todo = ledger.pending(p)
            print(f"{len(todo)} request(s) without an outcome" + (":" if todo else ""))
            for e in todo:
                print(f"  {e['request_id']}  {e.get('status')}  {e.get('model')}  {(e.get('request') or {}).get('tag', '')}")
            return
        for f in hf.fetch(p, args.target, Path(args.out) if args.out else None):
            print(f)
        return

    endpoint = need("a model endpoint, e.g. bytedance/seedance-2.5/image-to-video")
    if args.action == "run" and not args.project:
        raise SystemExit("aimv hf run needs --project")
    raw = args.args or "{}"
    a = _json.loads(Path(raw[1:]).read_text() if raw.startswith("@") else raw)
    try:
        a, vid_s = hf.apply_files(a, args.file or [], uploader=None if args.action == "price" else hf.upload)
        input_s = args.input_video_seconds if args.input_video_seconds is not None else (vid_s or 0.0)
        if args.action == "price":
            sent = hf.apply_defaults(endpoint, a)
            added = {k: v for k, v in sent.items() if k not in a}
            info = hf.price(endpoint, sent, input_s)
            shown = {k: (v if k != "prompt" else f"<{len(v)} chars>") for k, v in sent.items()}
            print(f"request:      {_json.dumps(shown)}")
            if added:
                print(f"defaults:     {', '.join(f'{k}={_json.dumps(v)}' for k, v in added.items())} (set them to override)")
            print(f"our estimate: {'$%.3f' % info['usd'] if info['usd'] is not None else 'none, a run needs --usd'}"
                  f" ({info['source']})")
            print(f"higgsfield:   {hf.describe(info['higgsfield'])}")
            return
        p = Project.open(args.project)
        files = hf.run(p, endpoint, a, Path(args.out) if args.out else None, phase=args.phase, usd=args.usd,
                       tag=args.tag or "", input_video_s=input_s)
    except hf.Rejected as e:
        raise SystemExit(f"Higgsfield rejected the request (nothing charged): {e}")
    except ValueError as e:
        raise SystemExit(str(e))
    for f in files:
        print(f)


def _muto(args) -> None:
    from .look import muto
    from .project import Project

    p = Project.open(args.project)
    frames = sorted(Path(args.frames).glob("*.png")) if args.frames else None
    out = muto.render(p, Path(args.out), args.t0, args.t1, video=Path(args.video) if args.video else None,
                      video_t0=args.video_t0, frames=frames, division=args.division, hold_stops=not args.no_hold,
                      pan=(args.pan_x, args.pan_y), zoom=args.zoom, strength=args.strength, seed=args.seed)
    print(out)


def _time_arg(grid, v):
    """CLI times: seconds (57.6 or 57.6s), a grid address with a 16th (43.2.1), or a named hit (impact)."""
    if v is None:
        return None
    v = str(v).strip()
    try:
        return float(v[:-1] if v.endswith("s") else v)
    except ValueError:
        return grid.t(grid.resolve(v)[0])


def _paint(args) -> None:
    from .paint import PaintDirs
    from .project import Project

    p = Project.open(args.project)
    pd = PaintDirs(p)
    act = args.action
    if act == "street":
        from .paint import street

        d = pd.street(args.street)
        if args.canvas and args.rect:  # a world cell: the known world pixels in the rect, grey elsewhere
            from .paint import world

            info = world.canvas(p.root, d, args.canvas, tuple(args.rect), Path(args.out) if args.out else None)
            print(f"canvas for {args.canvas}: {info['canvas']} ({info['size'][0]} x {info['size'][1]}, "
                  f"{info['known']:.0%} known, on its {', '.join(info['sides']) or 'no'} side(s); street px "
                  f"{info['street_box']})")
            return
        if args.canvas:
            spec = street.load_spec(d)
            if args.left:  # the next plate on the left: from the leftmost plate so far
                prev = p.root / (spec["left"][-1] if spec["left"] else spec["plates"][0])
            else:
                prev = p.root / spec["plates"][-1]
            out = prev.with_name(f"canvas-{args.canvas}.png")
            street.make_canvas(prev, out, float(spec["keep"]), left=args.left)
            side = "left" if args.left else "right"
            print(f"canvas for {args.canvas}: {out} (the {side} {spec['keep']:.0%} of {prev.stem} on the "
                  f"{'right' if args.left else 'left'}, grey elsewhere)")
            return
        built = street.build(p.root, d)
        street.seams_preview(d, d / "seams.jpg")
        print(street.info_text(d))
        print(f"seams: {d / 'seams.jpg'}")
        if built.get("world"):
            from .paint import world

            print(f"world: {world.contact(d, d / 'world.jpg')} (cells and seams), "
                  f"{world.seams_sheet(d, d / 'world-seams.jpg')} (every seam at full resolution)")
        return
    if act == "window":
        from .paint import street

        if not (args.t_from and args.t_to):
            raise SystemExit("aimv paint window <project> --from X --to X [--out file.png] (street px)")
        d = pd.street(args.street)
        x0, x1 = int(round(float(args.t_from))), int(round(float(args.t_to)))
        out = Path(args.out) if args.out else d / "windows" / f"x{x0}_{x1}.png"
        info = street.export_window(d, x0, x1, out.resolve(), p.root.resolve())
        print(f"window x {x0} to {x1} ({info['size'][0]} x {info['size'][1]} px): {out}")
        print(f"placement: {out.with_suffix('.json')}")
        if info["patched"]:
            print(f"(already patched there: {', '.join(info['patched'])})")
        print(f"once edited, add to street.json edits: {json.dumps(info['edit'])}")
        return
    if act == "scene":
        from .paint.scene import Scene

        sc = Scene(pd.street(args.street), pd.cache)
        out = sc.preview(pd.street(args.street) / "scene-masks.jpg")
        print(f"{len(sc.obstacles)} obstacles, {len(sc.no_paint)} no-paint zones, {len(sc.layers)} layers, "
              f"{len(sc.lights)} lights on street x {sc.X0} to {sc.W} ({sc.W - sc.X0} x {sc.H} px)")
        if sc.world is not None:
            x0, y0, x1, y1 = sc.bounds
            print(f"world: x {x0} to {x1}, y {y0} to {y1}; {len(sc.levels)} levels, {len(sc.voids)} voids")
        print(f"masks: {out} (occluders red, layers blue, no-paint grey, voids purple, boxes yellow, contacts green, "
              f"ground lines cyan)")
        if sc.levels:
            sheet, measured = sc.levels_sheet(pd.street(args.street) / "levels.jpg")
            from .util import write_json

            write_json(pd.street(args.street) / "levels.json", {"measured": measured})
            print(f"levels: {sheet} (each level's feet line green, the edge measured near it red), the measured "
                  f"lines in {pd.street(args.street) / 'levels.json'}")
        return
    if act == "clipframes":
        from .paint import clipframes
        from .paint.poses import PoseLib
        from .paint.scene import Scene

        if not args.choreo:
            raise SystemExit("aimv paint clipframes <project> <obstacle> --move vault|leap_off|step_up|...")
        sc = Scene(pd.street(args.street), pd.cache)
        info = clipframes.make(PoseLib(pd.poses), sc, args.choreo, args.move or "vault",
                               pd.root / "clipframes" / (args.folder or args.choreo), args.cycle,
                               tuple(x for x in (args.also or "").split(",") if x), args.start_pose, args.end_pose,
                               tuple(float(v) for v in args.box.split("x")) if args.box else None,
                               scale_by=args.scale_by, open_end=args.open, open_start=args.open_start,
                               box_cx=float(args.box_x) if args.box_x else None,
                               start_x=float(args.start_x) if args.start_x else None,
                               end_x=float(args.end_x) if args.end_x else None,
                               box_h=float(args.box_h) if args.box_h else None)
        print(f"{info['obstacle']} ({info['obstacle_px']['width']:.0f} x {info['obstacle_px']['height']:.0f} street px): "
              f"guide block {info['block']} on a {info['figure_height']} px figure, ground y {info['ground']}")
        for k_, v in info["frames"].items():
            print(f"  {k_}: {v}")
        return
    if act == "poses":
        from .paint import poses

        pd.poses.mkdir(parents=True, exist_ok=True)
        poses.build(p.root, pd.poses)
        from .paint import canon

        canon.run(p.root, pd.poses)  # canon.json always matches the library (QA and the clip sequences read it)
        return
    if act == "canon":
        from .paint import canon

        canon.run(p.root, pd.poses)
        return
    if not args.choreo:
        raise SystemExit(f"aimv paint {act} needs a choreography name (projects/<p>/paint/choreo/<name>.yaml)")
    from .paint.choreo import open_choreo

    over = {"jump": args.jump, "primer": args.primer, "street": args.street if args.street != "main" else None,
            "motion": args.motion}
    if args.no_foot_lock:
        over["foot_lock"] = False
    if args.no_vary:
        over["vary"] = False
    ch = open_choreo(p, args.choreo, over)
    if act == "check":
        from .paint.check import check

        from .paint.qa import banner

        r = check(p, ch)
        print("\n".join(r["lines"]))
        if "qa" in r:
            print(banner(r["qa"]))
        if any(i.level == "error" for i in r["issues"]) or r.get("qa", {}).get("errors"):
            raise SystemExit(1)
        return
    if ch.errors:
        print("\n".join(str(i) for i in ch.errors))
        raise SystemExit("fix the choreography first (aimv paint check)")
    pd.renders.mkdir(parents=True, exist_ok=True)
    g = ch.grid
    if act == "map":
        from .paint.maps import street_map

        out = Path(args.out) if args.out else pd.renders / f"{ch.name}-map.jpg"
        print(street_map(ch, out))
        return
    aspects = {"16:9": ("16:9",), "9:16": ("9:16",), "both": ("16:9", "9:16")}[args.aspect]
    if act == "still":
        import cv2

        from .paint.render import SIZES, History, Renderer, look, plan_jobs

        t = _time_arg(g, args.at)
        if t is None:
            raise SystemExit("aimv paint still needs --at (seconds or bar.beat.16th)")
        hist = History(ch)
        jobs, _ = plan_jobs(ch, hist, list(aspects))
        j = max([j for j in jobs if j.t <= t + 1e-6], key=lambda j: j.t, default=jobs[0])
        r = Renderer(ch, hist)
        for a in aspects:
            size = SIZES[a] if not args.preview else tuple(v // 2 for v in SIZES[a])
            img = r.photo(j.k, j.t, j.q, j.cams[a], size)
            tag = a.replace(":", "x")
            out = Path(args.out) if args.out and len(aspects) == 1 else pd.renders / f"{ch.name}-still-{j.t:.3f}-{tag}.png"
            if args.raw:
                cv2.imwrite(str(out), cv2.cvtColor((img * 255 + 0.5).astype("uint8"), cv2.COLOR_RGB2BGR))
            else:
                look(img, j.k, a, preview=args.preview).save(out)
            print(f"photo {j.k} at {j.t:.3f} s ({g.addr(j.q)}): {out}")
        return
    if act == "render":
        from .paint.render import render

        only = [int(v) for v in args.only.split(",")] if args.only else None
        st = render(p, ch, pd.renders, _time_arg(g, args.t_from), _time_arg(g, args.t_to), aspects, args.preview,
                    only, args.workers, video=not args.no_video, keep=args.keep, overrides=over, name=args.name,
                    review=args.review, force=args.force, debug=args.debug)
        print(f"{st['photos']} photos in {st['wall_s']} s ({st['per_photo_s']} s per photo per process, "
              f"{st['workers']} processes)")
        for k, v in st.items():
            if k.startswith("framing"):
                print(f"  {k}: {v}")
        if st.get("photos_dir"):
            print(f"  photos: {st['photos_dir']}")
        if st.get("peek"):
            print(f"  peek: {st['peek']}")
        for k_ in ("qa_sheet", "moves_sheet"):
            if st.get(k_):
                print(f"  {k_.replace('_', ' ')}: {st[k_]}")
        for f_ in st.get("review", []):
            print(f"  review copy: {f_}")
        if st.get("qa", {}).get("errors") and not args.force:
            raise SystemExit(1)
        return
    if act == "contact":
        from .paint.maps import contact_sheet, default_shots

        if not args.video:
            raise SystemExit("aimv paint contact needs --video (a render) [--against another video]")
        video = Path(args.video)
        t0 = _time_arg(g, args.t_from) if args.t_from else (ch.window[0] if ch.window else 0.0)
        t1 = _time_arg(g, args.t_to) if args.t_to else (ch.window[1] if ch.window else t0 + 10)
        if args.times:
            shots = []
            for part in args.times.split(","):
                tt, _, lab = part.partition(":")
                tv = _time_arg(g, tt.strip())
                shots.append((tv, lab.strip() or g.addr(g.q(tv)).split("+")[0]))
        else:
            shots = default_shots(ch, t0, t1, args.n)
        out = Path(args.out) if args.out else video.with_name(video.stem + "-contact.jpg")
        contact_sheet(shots, video, t0, out, Path(args.against) if args.against else None,
                      _time_arg(g, args.against_t0) if args.against_t0 else None,
                      labels=(args.label_a, args.label_b))
        print(out)
        return
    raise SystemExit(f"unknown action {act}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aimv", description="AI music video pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new", help="create a project folder and copy the song in")
    n.add_argument("project")
    n.add_argument("--audio", help="the master take (wav/mp3/m4a/flac)")
    n.add_argument("--lyrics", help="lyrics.txt: one sung line per line, [Section] tags, (backing vocals)")
    n.add_argument("--title")
    n.add_argument("--fps", type=int)
    n.set_defaults(func=_new)

    b = sub.add_parser("breakdown", help="phase 3: stems, beat grid, word times, sections, hits, envelopes, report")
    b.add_argument("project")
    b.add_argument("--only", help="comma-separated steps: ingest,separate,beats,align,sections,hits,envelopes,report")
    b.add_argument("--force", action="store_true", help="re-run even when inputs are unchanged")
    b.set_defaults(func=_breakdown)

    v = sub.add_parser("serve", help="open the breakdown report in a browser (local server, seeking works)")
    v.add_argument("project")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-open", action="store_true")
    v.set_defaults(func=_serve)

    rv = sub.add_parser("review", help="watch a render and leave notes on it (local page; notes in projects/<p>/review)")
    rv.add_argument("project")
    rv.add_argument("--port", type=int, default=8770)
    rv.add_argument("--no-open", action="store_true")
    rv.set_defaults(func=_review)

    bo = sub.add_parser("board", help="generate a choice board of frames (projects/<p>/boards/<name>/board.json)")
    bo.add_argument("project")
    bo.add_argument("name")
    bo.add_argument("--workers", type=int, default=4)
    bo.add_argument("--dry-run", action="store_true", help="only estimate the cost and render what exists")
    bo.set_defaults(func=_board)

    k = sub.add_parser("keys", help="which API keys are set (environment or .env) (never prints values)")
    k.set_defaults(func=_keys)

    lg = sub.add_parser("ledger", help="spend so far against the project's caps")
    lg.add_argument("project")
    lg.set_defaults(func=_ledger)

    h = sub.add_parser("hf", help="Higgsfield API: price (free), upload (free), run (paid, capped, logged), fetch")
    h.add_argument("action", choices=["price", "run", "upload", "fetch"])
    h.add_argument("target", nargs="?",
                   help="model endpoint (price, run), a file (upload) or a request id (fetch; omit to list pending)")
    h.add_argument("--project", help="project folder (required for run and fetch)")
    h.add_argument("--args", help="model arguments as JSON, or @file.json")
    h.add_argument("--file", action="append", metavar="FIELD=PATH",
                   help="upload PATH and set that field; FIELD[] appends to an array. Repeatable. "
                        "e.g. image_url=plate.png, end_image_url=end.png, image_urls[]=ref.png, audio_urls[]=v.wav, "
                        "video_url=clip.mp4. price uses placeholder URLs and uploads nothing")
    h.add_argument("--input-video-seconds", dest="input_video_seconds", type=float,
                   help="seconds of video input for Seedance extend/edit/video refs (measured from --file videos "
                        "when not given)")
    h.add_argument("--out", help="output path stem, extension from the result (default gen/<endpoint>-<time>-<id>)")
    h.add_argument("--phase", default="misc", help="budget phase for the ledger")
    h.add_argument("--usd", type=float, help="explicit estimate for models without a cost model")
    h.add_argument("--tag", help="free label stored in the ledger")
    h.set_defaults(func=_hf)

    m = sub.add_parser("muto", help="stitched-photographs look: hold frames on the song grid, camera steps, flicker")
    m.add_argument("project")
    m.add_argument("--out", required=True)
    m.add_argument("--t0", type=float, required=True, help="song time where the output starts")
    m.add_argument("--t1", type=float, required=True)
    m.add_argument("--video", help="source clip (song time t0 maps to --video-t0 in the clip)")
    m.add_argument("--video-t0", type=float, default=0.0)
    m.add_argument("--frames", help="folder of numbered frames, one per step, instead of a video")
    m.add_argument("--division", type=int, default=4, help="steps per beat: 4 = 16ths, 2 = 8ths")
    m.add_argument("--no-hold", action="store_true", help="do not freeze content during stop spans")
    m.add_argument("--pan-x", type=float, default=0.0)
    m.add_argument("--pan-y", type=float, default=0.0)
    m.add_argument("--zoom", type=float, default=0.0)
    m.add_argument("--strength", type=float, default=1.0)
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(func=_muto)

    pa = sub.add_parser("paint", help="the painted-runner side-scroller: street, scene, poses, check, render, still, "
                                      "map, contact (docs/PAINT.md)")
    pa.add_argument("action", choices=["street", "scene", "poses", "check", "render", "still", "map", "contact",
                                       "clipframes", "window", "canon"])
    pa.add_argument("project")
    pa.add_argument("choreo", nargs="?", help="choreography name (paint/choreo/<name>.yaml) or a .yaml path; "
                                              "clipframes: the obstacle")
    pa.add_argument("--move", help="clipframes: the move the clip is for (vault, leap_off, step_up ...)")
    pa.add_argument("--cycle", help="clipframes: the run sequence the figure comes from (default: the library's)")
    pa.add_argument("--and", dest="also", help="clipframes: more obstacles to show (comma list), where they stand")
    pa.add_argument("--start-pose", help="clipframes: the figure in the start frame (a pose or role; hang: on the bar)")
    pa.add_argument("--end-pose", help="clipframes: the figure in the end frame (a pose or role; hang: on the bar)")
    pa.add_argument("--box", help="clipframes: a front view inside a box outline WxHxLINE (street px), obstacle wall")
    pa.add_argument("--scale-by", help="clipframes --box: the pose whose height fills the box (one body scale)")
    pa.add_argument("--open", help="clipframes --box: the end frame's box without this side (right: he broke out)")
    pa.add_argument("--open-start", help="clipframes --box: the start frame's box without this side too")
    pa.add_argument("--box-x", help="clipframes --box: the box's centre x in the frame (default the middle)")
    pa.add_argument("--box-h", help="clipframes --box: the box's height in the frame, px (760; smaller leaves room "
                                    "for him to run out of it)")
    pa.add_argument("--folder", help="clipframes: the folder under paint/clipframes (default: the obstacle)")
    pa.add_argument("--start-x", help="clipframes: the start frame's figure, its left edge at this frame x")
    pa.add_argument("--end-x", help="clipframes: the end frame's figure, its left edge at this frame x")
    pa.add_argument("--street", default="main", help="street name (paint/streets/<name>)")
    pa.add_argument("--canvas", metavar="PLATE", help="street: write the outpaint canvas for the next plate")
    pa.add_argument("--left", action="store_true", help="street --canvas: the next plate on the left (prepended)")
    pa.add_argument("--rect", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="street --canvas: a world cell's canvas over this rect (anchor px, as a cell's at)")
    pa.add_argument("--from", dest="t_from", help="start: seconds (57.6) or bar.beat.16th (43.1.1); window: street x")
    pa.add_argument("--to", dest="t_to", help="end: seconds or bar.beat.16th; window: street x")
    pa.add_argument("--at", help="still: song time (seconds or bar.beat.16th)")
    pa.add_argument("--aspect", choices=["16:9", "9:16", "both"], default="both")
    pa.add_argument("--preview", action="store_true", help="half size (the fast iteration loop)")
    pa.add_argument("--only", help="render: comma list of photo indices (keeps the photos, no video)")
    pa.add_argument("--workers", type=int, help="render processes (default: up to 8)")
    pa.add_argument("--keep", action="store_true", help="render: keep the photos (and raw PNGs)")
    pa.add_argument("--no-video", action="store_true")
    pa.add_argument("--name", help="render: output name stem")
    pa.add_argument("--debug", action="store_true",
                    help="render: draw the physics on every photo (feet lines, obstacle boxes, the flight's arc, his "
                         "feet coloured by the contact rule, speed, gait and span slack); named <name>-debug")
    pa.add_argument("--review", action="store_true",
                    help="render: small review copies (1600x900, 720x1280), not when the QA has errors")
    pa.add_argument("--force", action="store_true", help="render: review copies even when the QA has errors")
    pa.add_argument("--jump", choices=["J1", "J2"], help="override the choreography's jump style")
    pa.add_argument("--primer", choices=["all", "keepers", "none", "ghosts"], help="override the primer mode")
    pa.add_argument("--motion", choices=["m0", "legacy"], help="override the motion engine (m0 by default)")
    pa.add_argument("--no-foot-lock", action="store_true", help="runs slide like prototype 6")
    pa.add_argument("--no-vary", action="store_true", help="no per-photo variation of repeated drawings")
    pa.add_argument("--raw", action="store_true", help="still: without the MUTO look")
    pa.add_argument("--out")
    pa.add_argument("--video", help="contact: the rendered video")
    pa.add_argument("--against", help="contact: a second video, shown next to each frame")
    pa.add_argument("--against-t0", help="contact: song time where the second video starts (default --from)")
    pa.add_argument("--label-a", default="new")
    pa.add_argument("--label-b", default="before")
    pa.add_argument("--times", help="contact: comma list of time[:label] (default: the moves' key moments)")
    pa.add_argument("--n", type=int, default=16, help="contact: number of shots")
    pa.set_defaults(func=_paint)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
