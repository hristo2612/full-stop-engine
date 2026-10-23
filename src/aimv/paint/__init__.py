"""`aimv paint`: the painted-runner side-scroller (prototype 6 made into a tool). See docs/PAINT.md.

A runner who exists only as black emulsion on the walls of a long street does parkour over real objects, one
painted pose per photograph, the photographs held on the song's grid (MUTO). Everything is ours and deterministic:
the street is stitched from plates, the obstacles are marked by hand, the poses come from sprite sheets, and a
hand-written choreography places the poses on the obstacles and the beats.

Per-project data lives in projects/<p>/paint/:

    streets/<name>/street.json      plates and game-level edits (the spec)
    streets/<name>/obstacles.json   ground lines, obstacles with contacts, occluders, no-paint zones, layers, lights
    streets/<name>/build.json       what the last build made (offsets, seams, tile hashes); written by `street`
    streets/<name>/strip/, tiles/   the stitched strip and the edited street as 2048 px wide PNG tiles
    poses/poses.spec.json           sheets to split, scale references, cycles, roles and tags
    poses/<name>.png, poses.json    the library (density mattes, anchors, tags); written by `poses`
    choreo/<name>.yaml              the choreography (hand-written)
    renders/                        videos, stills, contact sheets, street maps
    .cache/                         per-tile scene maps (regenerable, safe to delete)

Modules: grid (song addressing), street (plates to tiles, appended on the right or prepended on the left), world
(plates in two dimensions round the street), scene (obstacles, maps, levels, voids), poses (library), choreo and
moves (the choreography, composition, song-stop pauses and the move templates), props (the box: outline, cracks,
stencils, shatter), camera (16:9 and 9:16 plans), render (paint model, compositor, MUTO look, video), maps (street
map), check (validation and summary).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..project import Project


@dataclass
class PaintDirs:
    """Paths of one project's paint data."""

    p: Project

    @property
    def root(self) -> Path:
        return self.p.root / "paint"

    def street(self, name: str) -> Path:
        return self.root / "streets" / name

    @property
    def poses(self) -> Path:
        return self.root / "poses"

    @property
    def choreo(self) -> Path:
        return self.root / "choreo"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    @property
    def cache(self) -> Path:
        return self.root / ".cache"

    def choreo_file(self, name: str) -> Path:
        f = Path(name)
        if f.suffix in (".yaml", ".yml") and f.exists():
            return f.resolve()
        return self.choreo / f"{name}.yaml"
