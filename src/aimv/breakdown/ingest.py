"""ingest: decode the master take into the reference clock files.

breakdown/audio/mix.wav is THE soundtrack for every later phase. Renders must use it (not the mp3 in song/),
because decoders pad mp3/m4a differently and that would shift every timing in this folder.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pyloudnorm as pyln
import soxr

from ..util import load_audio, save_audio, sha256_file, write_json
from .dsp import estimate_key

NAME = "ingest"
SR = 48000


def inputs(p):
    return [p.song_audio(), p.suno_dir / "key.json"]


def outputs(p):
    return [p.bd / "audio/mix.wav", p.bd / "audio/mix16k.wav", p.bd / "meta.json"]


def _decode(src, dst) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (brew install ffmpeg)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:a:0", "-ac", "2",
         "-af", "aresample=48000:resampler=soxr:precision=28", "-c:a", "pcm_s24le", str(dst)],
        check=True,
    )


def run(p, log):
    src = p.song_audio()
    mix = p.bd / "audio/mix.wav"
    _decode(src, mix)
    x, sr = load_audio(mix)
    log(f"{src.name}: {x.shape[1] / sr:.2f}s, {x.shape[0]} ch")

    mono16 = soxr.resample(x.mean(axis=0), sr, 16000, quality="VHQ").astype(np.float32)
    save_audio(p.bd / "audio/mix16k.wav", mono16, 16000, subtype="FLOAT")

    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(x.T))
    up = soxr.resample(x.T, sr, sr * 4, quality="HQ")
    true_peak = float(20 * np.log10(max(np.abs(up).max(), 1e-12)))

    key_local, key_r = estimate_key(x.mean(axis=0), sr)
    suno_key = p.suno("key")
    key_suno = suno_key.get("key") if isinstance(suno_key, dict) else None
    if key_suno:
        key_suno = key_suno.replace("_", " ")  # Suno writes "D_minor"
    log(f"LUFS {lufs:.1f}, true peak {true_peak:.1f} dBTP, key {key_local} (r={key_r:.2f}) suno={key_suno}")

    write_json(p.bd / "meta.json", {
        "version": 1,
        "source": str(src.relative_to(p.root)),
        "source_sha256": sha256_file(src),
        "duration": x.shape[1] / sr,
        "sample_rate": sr,
        "channels": int(x.shape[0]),
        "lufs_integrated": lufs,
        "true_peak_db": true_peak,
        "key": {"local": key_local, "local_confidence": key_r, "suno": key_suno},
    })
