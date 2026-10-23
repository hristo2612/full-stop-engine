"""separate: stems for alignment, hits and envelopes.

Two passes with python-audio-separator:
  1. vocals / instrumental with a band-split RoFormer (best vocal isolation; feeds word alignment and lip sync)
  2. drums / bass / other with htdemucs_ft, run on the instrumental so vocals do not leak into "other"

Models are configurable in project.json:
  "separate": {"vocal_model": "...ckpt", "drum_model": "htdemucs_ft.yaml"}
Suno's own 12 stems, if exported to song/suno/stems/, are kept as extra material but not relied on.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from ..util import MODELS_DIR, load_audio, save_audio

NAME = "separate"
CONFIG_KEYS = ("separate",)
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DRUM_MODEL = "htdemucs_ft.yaml"
STEMS = ["vocals", "instrumental", "drums", "bass", "other"]


def _cfg(p) -> dict:
    return p.config.get("separate", {})


def inputs(p):
    return [p.bd / "audio/mix.wav"]


def outputs(p):
    return [p.bd / f"stems/{s}.wav" for s in STEMS]


def _separate(src: Path, model: str, workdir: Path, log) -> dict[str, Path]:
    import logging

    from audio_separator.separator import Separator

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = workdir / Path(model).stem
    out.mkdir(parents=True, exist_ok=True)
    sep = Separator(model_file_dir=str(MODELS_DIR), output_dir=str(out), output_format="WAV",
                    log_level=logging.WARNING)
    sep.load_model(model_filename=model)
    files = sep.separate(str(src))
    stems = {}
    for f in files:
        f = Path(f) if Path(f).is_absolute() else out / f
        m = re.search(r"\(([^)]+)\)[^()]*$", f.name)
        if m:
            stems[m.group(1).strip().lower()] = f
    log(f"{model}: {', '.join(sorted(stems))}")
    return stems


def run(p, log):
    cfg = _cfg(p)
    vocal_model = cfg.get("vocal_model", VOCAL_MODEL)
    drum_model = cfg.get("drum_model", DRUM_MODEL)
    mix = p.bd / "audio/mix.wav"
    dst = p.bd / "stems"
    dst.mkdir(parents=True, exist_ok=True)
    _, sr = load_audio(mix)

    with tempfile.TemporaryDirectory(prefix="aimv-sep-") as tmp:
        tmp = Path(tmp)
        vs = _separate(mix, vocal_model, tmp, log)
        voc = vs.get("vocals")
        inst = vs.get("instrumental") or vs.get("other") or vs.get("no vocals")
        if not voc or not inst:
            raise RuntimeError(f"vocal model gave {list(vs)}; need vocals + instrumental")
        ds = _separate(inst, drum_model, tmp, log)
        found = {"vocals": voc, "instrumental": inst, "drums": ds.get("drums"), "bass": ds.get("bass"),
                 "other": ds.get("other")}
        missing = [k for k, v in found.items() if v is None]
        if missing:
            raise RuntimeError(f"drum model gave {list(ds)}; missing {missing}")
        for name, f in found.items():
            x, fsr = load_audio(f, sr=sr)  # keep every stem on the mix's sample rate and length
            save_audio(dst / f"{name}.wav", x, fsr)
        for extra, f in ds.items():  # 6-stem models also give guitar/piano
            if extra not in found:
                shutil.copy2(f, dst / f"{extra}.wav")
