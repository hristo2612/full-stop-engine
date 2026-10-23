"""envelopes: one value per video frame for the motion engine to drive shapes, scale and colour.

Frame convention: series[name][i] describes the audio heard while frame i is on screen, the interval
[i / fps, (i + 1) / fps), so a renderer that shows frame floor(t * fps) at song time t reads the value
for the sound playing during that frame. Loudness is the RMS over exactly that interval (equivalently, a
one-frame window centred on the frame centre (i + 0.5) / fps). A hit pulse peaks in the frame that
contains the hit and its decay is evaluated at frame centres. Visuals therefore lead the sound by at most
one frame, never lag it (a late picture is noticed sooner than an early one).

Series (all 0..1, 3 decimals):
  mix, vocals, drums, bass, other   RMS in dBFS mapped -60..0 dB to 0..1, then a follower with instant
                                    attack and a 150 ms release (time constant), applied to that 0..1 value
  kick, snare, hat                  decaying pulses from hits.json: s * exp(-dt / tau), the max where pulses
                                    overlap; tau 100 / 140 / 50 ms
  centroid                          spectral centroid of the mix (power-weighted over 150 Hz..20 kHz so kick
                                    and bass fundamentals do not pull it down on every beat; Hann window of
                                    two frames centred on the frame centre), log-frequency, mapped so the
                                    song's 2nd..98th percentile of audible frames spans 0..1, smoothed with a
                                    5-frame triangle; 0 where the mix is under -60 dBFS
"""

from __future__ import annotations

import math

import numpy as np

NAME = "envelopes"
STEMS = ["vocals", "drums", "bass", "other"]
DB_FLOOR = -60.0
RELEASE_S = 0.15
PULSE_TAU = {"kick": 0.10, "snare": 0.14, "hat": 0.05}
CENTROID_LO = 150.0


def inputs(p):
    return [p.bd / "audio/mix.wav", *[p.bd / f"stems/{s}.wav" for s in STEMS], p.bd / "hits.json"]


def outputs(p):
    return [p.bd / "envelopes.json"]


def n_frames(n_samples: int, sr: int, fps: int) -> int:
    """ceil(duration * fps), robust to float error when the duration is a whole number of frames."""
    return max(1, math.ceil(n_samples * fps / sr - 1e-9))


def frame_bounds(n_samples: int, sr: int, fps: int, frames: int) -> np.ndarray:
    """Sample index where each frame interval starts, plus the end: shape (frames + 1,)."""
    b = np.round(np.arange(frames + 1) * sr / fps).astype(np.int64)
    return np.clip(b, 0, n_samples)


def frame_rms_db(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """RMS in dBFS over each frame interval of a (channels, samples) or (samples,) signal."""
    p = (x.astype(np.float64) ** 2).mean(axis=0) if x.ndim == 2 else x.astype(np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(p)])
    n = np.maximum(np.diff(bounds), 1)
    ms = (cs[bounds[1:]] - cs[bounds[:-1]]) / n
    return 10.0 * np.log10(np.maximum(ms, 1e-12))


def db_to_unit(d: np.ndarray) -> np.ndarray:
    return np.clip((d - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0)


def follower(v: np.ndarray, fps: int, release_s: float = RELEASE_S) -> np.ndarray:
    """Instant attack, exponential release toward the input with time constant `release_s`."""
    a = math.exp(-1.0 / (fps * release_s))
    out = np.empty_like(v)
    y = 0.0
    for i, x in enumerate(v):
        y = x if x >= y else a * y + (1.0 - a) * x
        out[i] = y
    return out


def pulses(times: np.ndarray, strengths: np.ndarray, frames: int, fps: int, tau: float) -> np.ndarray:
    """Decaying pulses: peak s in the frame containing the hit, then s * exp(-(centre - t) / tau)."""
    out = np.zeros(frames)
    centre = (np.arange(frames) + 0.5) / fps
    span = int(math.ceil(tau * 7 * fps)) + 1  # exp(-7) < 0.001: rounds to 0
    for t, s in zip(times, strengths):
        k = int(math.floor(t * fps))
        if k >= frames or k < 0:
            continue
        j = np.arange(k, min(frames, k + span))
        v = s * np.exp(-np.maximum(centre[j] - t, 0.0) / tau)
        v[0] = s
        out[j] = np.maximum(out[j], v)
    return out


def centroid(x: np.ndarray, sr: int, fps: int, frames: int, mix_db: np.ndarray) -> tuple[np.ndarray, list]:
    """Normalised log-frequency spectral centroid per frame. Returns (series, [lo_hz, hi_hz])."""
    mono = x.mean(axis=0) if x.ndim == 2 else x
    n = 1 << int(math.ceil(math.log2(2 * sr / fps)))
    win = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= CENTROID_LO) & (freqs <= 20000.0)
    f = freqs[band]
    pad = np.concatenate([np.zeros(n, np.float32), mono.astype(np.float32), np.zeros(n, np.float32)])
    centres = np.round((np.arange(frames) + 0.5) * sr / fps).astype(np.int64) + n
    hz = np.zeros(frames)
    for c0 in range(0, frames, 512):
        c = centres[c0:c0 + 512]
        idx = c[:, None] - n // 2 + np.arange(n)[None, :]
        P = np.abs(np.fft.rfft(pad[idx] * win, axis=1))[:, band] ** 2
        tot = P.sum(axis=1)
        hz[c0:c0 + 512] = np.where(tot > 0, (P * f).sum(axis=1) / np.maximum(tot, 1e-30), 0.0)
    audible = (mix_db > -50.0) & (hz > 0)
    if not audible.any():
        return np.zeros(frames), [0.0, 0.0]
    lg = np.log2(np.maximum(hz, 1.0))
    lo, hi = np.percentile(lg[audible], [2, 98])
    v = np.clip((lg - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    tri = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    v = np.convolve(np.pad(v, 2, mode="edge"), tri / tri.sum(), mode="valid")
    v[mix_db <= DB_FLOOR] = 0.0
    return v, [float(2 ** lo), float(2 ** hi)]


def _compact(v: np.ndarray) -> list:
    """3 decimals, and 0 / 1 written as integers to keep the file small."""
    out = []
    for x in np.round(np.clip(v, 0.0, 1.0), 3).tolist():
        out.append(0 if x == 0 else 1 if x == 1 else x)
    return out


def run(p, log):
    from ..util import load_audio, read_json, write_json

    fps = p.fps
    mix, sr = load_audio(p.bd / "audio/mix.wav")
    n = mix.shape[1]
    frames = n_frames(n, sr, fps)
    bounds = frame_bounds(n, sr, fps, frames)

    series: dict[str, np.ndarray] = {}
    mix_db = frame_rms_db(mix, bounds)
    series["mix"] = follower(db_to_unit(mix_db), fps)
    cen, cen_hz = centroid(mix, sr, fps, frames, mix_db)
    del mix
    for s in STEMS:
        x, xsr = load_audio(p.bd / f"stems/{s}.wav", sr=sr)
        if x.shape[1] < n:
            x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
        series[s] = follower(db_to_unit(frame_rms_db(x[:, :n], bounds)), fps)
    hits = read_json(p.bd / "hits.json")["events"]
    for k, tau in PULSE_TAU.items():
        ev = hits.get(k, [])
        series[k] = pulses(np.array([e["t"] for e in ev], float), np.array([e["s"] for e in ev], float),
                           frames, fps, tau)
    series["centroid"] = cen

    order = ["mix", "vocals", "drums", "bass", "other", "kick", "snare", "hat", "centroid"]
    out = p.bd / "envelopes.json"
    write_json(out, {
        "version": 1,
        "fps": fps,
        "frames": frames,
        "sampling": "frame i covers [i/fps, (i+1)/fps); pulses peak in the frame containing the hit",
        "centroid_hz": cen_hz,
        "series": {k: _compact(series[k]) for k in order},
    }, compact=True)
    log(f"{frames} frames at {fps} fps, {out.stat().st_size / 1024:.0f} KB, centroid "
        f"{cen_hz[0]:.0f}..{cen_hz[1]:.0f} Hz")
