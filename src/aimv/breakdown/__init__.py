"""Phase 3: song breakdown. See docs/BREAKDOWN.md for the output contract.

Every step is a module exposing:
    NAME: str
    def inputs(p: Project) -> list[Path]     files whose contents decide whether to re-run
    def outputs(p: Project) -> list[Path]    files the step writes (all must exist for a cache hit)
    def run(p: Project, log) -> None
    CONFIG_KEYS: tuple[str, ...]          optional: project.json sections the step reads (part of the cache key)
"""

from __future__ import annotations

import hashlib
import json
import importlib
import time
from pathlib import Path

from ..project import Project
from ..util import read_json, sha256_file, write_json

STEPS = ["ingest", "separate", "beats", "align", "sections", "hits", "envelopes", "report"]


def _module(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def _fingerprint(p: Project, mod) -> str:
    h = hashlib.sha256()
    h.update(Path(mod.__file__).read_bytes())
    h.update(str(p.fps).encode())
    cfg = p.config
    for k in getattr(mod, "CONFIG_KEYS", ()):  # only the settings this step reads
        h.update(json.dumps(cfg.get(k), sort_keys=True).encode())
    for f in mod.inputs(p):
        f = Path(f)
        h.update(str(f.relative_to(p.root) if f.is_relative_to(p.root) else f).encode())
        h.update(sha256_file(f).encode() if f.exists() else b"missing")
    return h.hexdigest()


def run(p: Project, only: list[str] | None = None, force: bool = False, log=print) -> None:
    p.bd.mkdir(parents=True, exist_ok=True)
    cache_path = p.bd / ".cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    steps = only or STEPS
    for name in steps:
        if name not in STEPS:
            raise SystemExit(f"Unknown step {name!r}. Steps: {', '.join(STEPS)}")
    for name in steps:
        mod = _module(name)
        fp = _fingerprint(p, mod)
        outs = [Path(o) for o in mod.outputs(p)]
        if not force and cache.get(name) == fp and all(o.exists() for o in outs):
            log(f"[{name}] up to date")
            continue
        log(f"[{name}] running")
        t = time.time()
        mod.run(p, lambda msg, _n=name: log(f"[{_n}] {msg}"))
        missing = [str(o) for o in outs if not o.exists()]
        if missing:
            raise RuntimeError(f"step {name} did not write: {missing}")
        cache[name] = _fingerprint(p, mod)  # inputs may include files this step just made
        write_json(cache_path, cache)
        log(f"[{name}] done in {time.time() - t:.1f}s")
