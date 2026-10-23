"""`aimv paint render` keeping only the looked JPEG photos (no raw PNGs: the disk is nearly full): the CLI's render
without --keep, with the work dir kept at the end so the sheets and join strips can read the photos (an empty work
dir left by an aborted run is cleared; one with photos in it never is).
python render_jpg.py <same args as aimv paint render, without --keep>"""
import os, sys, types
from pathlib import Path


def _rmtree(p, *a, **k):
    p = Path(p)
    if any(f.is_file() for f in p.rglob("*")):
        print(f"(work dir kept: {p})")
        return
    for d in sorted([d for d in p.rglob("*") if d.is_dir()], key=lambda d: -len(d.parts)) + [p]:
        os.rmdir(d)


if __name__ == "__main__":
    import aimv.paint.render as RR
    RR.shutil = types.SimpleNamespace(rmtree=_rmtree)
    from aimv.cli import main
    sys.argv = ["aimv", "paint", "render"] + sys.argv[1:]
    main()
