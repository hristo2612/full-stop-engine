"""Small DSP helpers shared by the breakdown steps (onsets, envelopes, key)."""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

HOP = 256  # at 48 kHz: 5.33 ms per frame


def stft_mag(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = HOP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT of a mono signal. Returns (mag[freq, frame], freqs, frame_times)."""
    import librosa

    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, freqs, times


def band_flux(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Half-wave rectified spectral flux of log magnitude inside [lo, hi) Hz, normalised to 0..1."""
    band = (freqs >= lo) & (freqs < hi)
    L = np.log1p(100.0 * S[band])
    d = np.diff(L, axis=1, prepend=L[:, :1])
    f = np.maximum(d, 0.0).sum(axis=0)
    m = np.percentile(f, 99.5) if f.size else 0.0
    return np.clip(f / m, 0.0, 1.5) if m > 0 else f


def pick_peaks(env: np.ndarray, times: np.ndarray, min_gap_s: float, thresh: float, rel: float = 0.0,
               win_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Peaks above an absolute threshold and above `rel` x the local running mean (adaptive).

    Returns (peak_times, peak_heights)."""
    if env.size < 3:
        return np.array([]), np.array([])
    dt = times[1] - times[0]
    dist = max(1, int(round(min_gap_s / dt)))
    idx, props = sps.find_peaks(env, height=thresh, distance=dist)
    if rel > 0 and idx.size:
        w = max(1, int(round(win_s / dt)))
        local = np.convolve(env, np.ones(w) / w, mode="same")
        keep = env[idx] >= rel * local[idx] + thresh * 0.5
        idx = idx[keep]
    return times[idx], env[idx]


def band_onsets(x: np.ndarray, sr: int, lo: float, hi: float, min_gap_s: float = 0.08,
                thresh: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Onset times and strengths for one frequency band of a mono signal."""
    S, freqs, times = stft_mag(x, sr)
    env = band_flux(S, freqs, lo, hi)
    return pick_peaks(env, times, min_gap_s=min_gap_s, thresh=thresh, rel=1.5)


def refine_attacks(x: np.ndarray, sr: int, times: np.ndarray, lo: float, hi: float,
                   search_s: float = 0.06, frac: float = 0.2) -> np.ndarray:
    """Move approximate onset times to the real attack in the waveform.

    STFT flux peaks fire early by up to half the analysis window (about 20 ms with n_fft 2048), which is
    enough to drag a beat grid off the beat. For each time, band-pass the signal, take its envelope, and
    return the first point inside [t - search, t + search] that reaches `frac` of the local peak."""
    if times.size == 0:
        return times
    sos = sps.butter(4, [max(lo, 20.0), min(hi, sr / 2 - 100)], "bandpass", fs=sr, output="sos")
    env = np.abs(sps.sosfiltfilt(sos, x))
    env = sps.sosfiltfilt(sps.butter(2, min(4 * hi, 400.0), "lowpass", fs=sr, output="sos"), env)
    out = times.copy()
    w = int(search_s * sr)
    for k, t in enumerate(times):
        a, e = max(0, int(t * sr) - w), min(len(env), int(t * sr) + w)
        if e - a < 8:
            continue
        seg = env[a:e]
        pk = int(np.argmax(seg))
        if seg[pk] <= 0:
            continue
        rise = np.nonzero(seg[: pk + 1] >= frac * seg[pk])[0]
        out[k] = (a + (rise[0] if rise.size else pk)) / sr
    return out


def nearest_offsets(events: np.ndarray, grid: np.ndarray, max_abs: float) -> np.ndarray:
    """Signed offset from each event to its nearest grid point, for events within `max_abs` of one."""
    if events.size == 0 or grid.size == 0:
        return np.array([])
    j = np.clip(np.searchsorted(grid, events), 1, len(grid) - 1)
    left, right = grid[j - 1], grid[j]
    near = np.where(np.abs(events - left) < np.abs(events - right), left, right)
    off = events - near
    return off[np.abs(off) <= max_abs]


KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(x: np.ndarray, sr: int) -> tuple[str, float]:
    """Krumhansl-Schmuckler key estimate from CQT chroma. Returns ("F minor", correlation)."""
    import librosa

    y = librosa.resample(x, orig_sr=sr, target_sr=22050) if sr != 22050 else x
    harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=22050).mean(axis=1)
    best = ("C major", -1.0)
    for k in range(12):
        for prof, mode in ((KS_MAJOR, "major"), (KS_MINOR, "minor")):
            r = float(np.corrcoef(chroma, np.roll(prof, k))[0, 1])
            if r > best[1]:
                best = (f"{NOTES[k]} {mode}", r)
    return best
