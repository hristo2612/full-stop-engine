"""Shared helpers: JSON with millisecond rounding, audio I/O, hashing, devices."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

MODELS_DIR = Path(os.environ.get("AIMV_MODELS_DIR") or Path(__file__).resolve().parents[2] / ".models")  # override with AIMV_MODELS_DIR


def _clean(obj: Any, nd: int) -> Any:
    if isinstance(obj, float | np.floating):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return round(v, nd)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_clean(x, nd) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _clean(v, nd) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(x, nd) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data: dict, nd: int = 3, compact: bool = False) -> None:
    """Write JSON with floats rounded to `nd` decimals and NaN/inf replaced by null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _clean(data, nd)
    if compact:
        text = json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(cleaned, indent=1, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_audio(path: Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio as float32, shape (channels, samples). Resamples with soxr when `sr` is given."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.T
    if mono:
        x = x.mean(axis=0, keepdims=True)
    if sr and sr != file_sr:
        x = soxr.resample(x.T, file_sr, sr, quality="VHQ").T.astype(np.float32)
        file_sr = sr
    return np.ascontiguousarray(x), file_sr


def save_audio(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.T if x.ndim == 2 else x
    sf.write(str(path), arr, sr, subtype=subtype)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def db(x: np.ndarray, floor: float = -120.0) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(x), 1e-12)), floor)
