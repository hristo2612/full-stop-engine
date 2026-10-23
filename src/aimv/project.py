"""A video project folder: projects/<slug>/ with project.json, song/, breakdown/, ..."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .util import read_json, write_json

AUDIO_EXTS = (".wav", ".flac", ".aiff", ".mp3", ".m4a")


@dataclass
class Project:
    root: Path

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Project folder not found: {root}")
        p = cls(root)
        if not p.config_path.exists():
            write_json(p.config_path, {"version": 1, "title": root.name, "slug": root.name, "fps": 24})
        return p

    @property
    def config_path(self) -> Path:
        return self.root / "project.json"

    @property
    def config(self) -> dict:
        return read_json(self.config_path)

    def update_config(self, **kv) -> None:
        cfg = self.config
        cfg.update(kv)
        write_json(self.config_path, cfg)

    @property
    def fps(self) -> int:
        return int(self.config.get("fps", 24))

    @property
    def song_dir(self) -> Path:
        return self.root / "song"

    @property
    def suno_dir(self) -> Path:
        return self.song_dir / "suno"

    @property
    def bd(self) -> Path:
        """Breakdown output folder."""
        return self.root / "breakdown"

    def song_audio(self) -> Path:
        """The master take: song/master.* first, else the only audio file in song/."""
        for ext in AUDIO_EXTS:
            cand = self.song_dir / f"master{ext}"
            if cand.exists():
                return cand
        found = sorted(f for f in self.song_dir.glob("*") if f.suffix.lower() in AUDIO_EXTS)
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"No audio in {self.song_dir} (expected master.wav)")
        raise SystemExit(f"Several audio files in {self.song_dir}; name the take master.wav")

    def lyrics_path(self) -> Path | None:
        f = self.song_dir / "lyrics.txt"
        return f if f.exists() else None

    def suno(self, name: str) -> dict | list | None:
        """Exported Suno JSON (song/suno/<name>.json) or None."""
        f = self.suno_dir / f"{name}.json"
        return read_json(f) if f.exists() else None
