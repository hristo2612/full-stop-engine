"""beats: beat and downbeat grid with a fitted tempo, checked against the kick drum.

beat_this gives beats on a 20 ms frame grid. We fit each beat with a local linear regression over its
neighbours (removes the quantisation jitter, keeps real tempo drift), then check the phase against kick
onsets from the drum stem. BLISS lost a storyboard to a grid that had locked onto the hi-hats half a beat
early, so a consistent offset or an off-beat lock is corrected here and reported.
"""

from __future__ import annotations

import numpy as np

from ..util import load_audio, torch_device, write_json
from .dsp import band_onsets, nearest_offsets, refine_attacks

NAME = "beats"
FIT_HALF = 8          # beats each side in the local fit
KEEP_RAW_S = 0.025    # residual above this = a real deviation, keep the tracker's beat
KICK_BAND = (30.0, 150.0)


def inputs(p):
    return [p.bd / "audio/mix.wav", p.bd / "stems/drums.wav"]


def outputs(p):
    return [p.bd / "beats.json"]


def _track(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import Audio2Beats

    dev = torch_device()
    a2b = Audio2Beats(checkpoint_path="final0", device=dev if dev != "cuda" else "cuda", dbn=False)
    beats, downs = a2b(x, sr)
    return np.asarray(beats, float), np.asarray(downs, float)


def _fit(beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fitted_times, local_period) per beat."""
    n = len(beats)
    idx = np.arange(n)
    fitted = beats.copy()
    period = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - FIT_HALF), min(n, i + FIT_HALF + 1)
        if hi - lo < 4:
            period[i] = np.median(np.diff(beats)) if n > 1 else 0.5
            continue
        slope, icpt = np.polyfit(idx[lo:hi], beats[lo:hi], 1)
        period[i] = slope
        f = slope * i + icpt
        if abs(f - beats[i]) < KEEP_RAW_S:
            fitted[i] = f
    return fitted, period


def _repair(beats: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fill beats the tracker skipped and drop doubles, using the tempo of the surrounding beats.

    In drumless stretches (intros, breakdowns) beat_this can fall back to half time, reporting every other
    beat. Suno songs hold a near-steady tempo, so a gap that is a whole multiple of the local period is
    filled evenly. Returns (beats, filled, dropped)."""
    if beats.size < 8:
        return beats, 0, 0
    d = np.diff(beats)
    mode = np.median(d)
    centres = (beats[:-1] + beats[1:]) / 2
    good = (d > 0.8 * mode) & (d < 1.25 * mode)
    ref = np.empty_like(d)
    for i, c in enumerate(centres):
        near = good & (np.abs(centres - c) < 20.0)
        ref[i] = np.median(d[near]) if near.sum() >= 4 else mode
    out = [beats[0]]
    filled = dropped = 0
    for i, di in enumerate(d):
        ratio = di / ref[i]
        k = int(round(ratio))
        if ratio < 0.55 and i + 1 < len(d):  # a double detection: skip this beat
            dropped += 1
            continue
        if k >= 2 and abs(ratio - k) < 0.3:
            step = (beats[i + 1] - out[-1]) / k
            out.extend(out[-1] + step * np.arange(1, k))
            filled += k - 1
        out.append(beats[i + 1])
    return np.asarray(out), filled, dropped


def _phase_downbeats(n: int, down_idx: np.ndarray, meter: int, run: int = 4) -> np.ndarray:
    """Downbeats as every `meter`-th beat, with the phase taken by majority vote of the tracker's downbeats.

    A phase change survives only if the tracker holds it for several bars (a real odd bar in the song);
    shorter wobbles (half-time stretches, single glitches) are voted away."""
    if down_idx.size == 0:
        return np.arange(0, n, meter)
    ph = down_idx % meter
    sm = np.empty_like(ph)
    for k in range(len(ph)):
        win = ph[max(0, k - run): k + run + 1]
        sm[k] = np.bincount(win, minlength=meter).argmax()
    out = []
    for j in range(n):
        k = int(np.argmin(np.abs(down_idx - j)))
        if j % meter == sm[k]:
            out.append(j)
    return np.asarray(out, int)


def _phase_check(beats: np.ndarray, period: np.ndarray, kicks: np.ndarray) -> dict:
    """Measure how the kicks sit on the grid; return the shift (s) to apply and stats."""
    res = {"kicks": int(kicks.size), "kick_offset_ms": None, "kicks_on_beat_pct": None,
           "kicks_off_beat_pct": None, "shifted_ms": 0.0, "note": ""}
    if kicks.size < 32 or beats.size < 32:
        res["note"] = "too few kicks to check phase"
        return res
    half = np.median(period) / 2
    tol = min(0.05, half * 0.4)
    mids = beats[:-1] + np.diff(beats) / 2
    on = nearest_offsets(kicks, beats, tol)
    off = nearest_offsets(kicks, mids, tol)
    res["kicks_on_beat_pct"] = 100.0 * on.size / kicks.size
    res["kicks_off_beat_pct"] = 100.0 * off.size / kicks.size
    shift = 0.0
    if off.size > 1.5 * max(on.size, 1) and off.size > 0.3 * kicks.size:
        # The grid is on the off-beats (e.g. locked to hats): move it half a beat onto the kicks.
        shift = float(np.median(nearest_offsets(kicks, beats, half * 1.01)))
        res["note"] = "grid was on the off-beat; moved onto the kicks"
    elif on.size >= 16:
        med = float(np.median(on))
        res["kick_offset_ms"] = med * 1000
        if abs(med) >= 0.012:
            shift = med
            res["note"] = "consistent kick offset; grid moved onto the kicks"
    res["shifted_ms"] = shift * 1000
    return res


def _bar_labels(n: int, down_idx: np.ndarray, meter: int) -> tuple[np.ndarray, np.ndarray]:
    bar = np.full(n, -1)
    beat = np.zeros(n, int)
    for j in range(n):
        k = np.searchsorted(down_idx, j, side="right") - 1
        if k < 0:
            beat[j] = (meter - (down_idx[0] - j)) % meter if down_idx.size else j % meter
        else:
            bar[j] = k
            beat[j] = j - down_idx[k]
    return bar, beat


def run(p, log):
    x, sr = load_audio(p.bd / "audio/mix.wav", mono=True)
    raw, raw_down = _track(x[0], sr)
    log(f"tracker: {raw.size} beats, {raw_down.size} downbeats")

    rep, filled, dropped = _repair(raw)
    if filled or dropped:
        log(f"grid repair: filled {filled} skipped beats, dropped {dropped} doubles")
    beats, period = _fit(rep)

    drums, dsr = load_audio(p.bd / "stems/drums.wav", mono=True)
    kicks, _ = band_onsets(drums[0], dsr, *KICK_BAND, min_gap_s=0.1, thresh=0.15)
    kicks = refine_attacks(drums[0], dsr, kicks, *KICK_BAND)  # flux peaks fire ~18 ms early
    phase = _phase_check(beats, period, kicks)
    if phase["shifted_ms"]:
        beats = beats + phase["shifted_ms"] / 1000
        raw_down = raw_down + phase["shifted_ms"] / 1000
        log(f"phase: {phase['note']} ({phase['shifted_ms']:+.0f} ms)")
    log(f"phase: kicks on beat {phase['kicks_on_beat_pct']}%, off beat {phase['kicks_off_beat_pct']}%")

    # Map each downbeat to its nearest beat.
    down_idx = np.unique(np.clip(np.searchsorted(beats, raw_down - np.median(period) / 2), 0, len(beats) - 1))
    down_idx = np.array([i for i in down_idx if abs(beats[i] - raw_down[np.argmin(abs(raw_down - beats[i]))]) < np.median(period) / 2])
    per_bar = np.diff(down_idx)
    meter = int(np.bincount(per_bar).argmax()) if per_bar.size else 4
    down_idx = _phase_downbeats(len(beats), down_idx, meter)
    per_bar = np.diff(down_idx)
    # A 1-beat "bar" is an odd bar in the song (Suno intros do this), not a real downbeat to cut on:
    # fold it into the bar before, which then has meter + 1 beats.
    short = [k for k in range(1, len(down_idx) - 1) if down_idx[k + 1] - down_idx[k] < 2]
    if short:
        log(f"folded {len(short)} one-beat bars into the previous bar at " + ", ".join(f"{beats[down_idx[k]]:.2f}s" for k in short))
        down_idx = np.delete(down_idx, short)
        per_bar = np.diff(down_idx)
    odd = [(beats[down_idx[k]], int(n)) for k, n in enumerate(per_bar) if n != meter]
    if odd:
        log(f"{len(odd)} bars without {meter} beats (kept as tracked): " + ", ".join(f"{t:.2f}s={n}" for t, n in odd))

    bar, beat = _bar_labels(len(beats), down_idx, meter)
    bpm = 60.0 / period

    # Downbeat sanity: kick strength by position in bar (information only).
    pos_strength = np.zeros(meter)
    if kicks.size:
        j = np.clip(np.searchsorted(beats, kicks), 1, len(beats) - 1)
        nearest = np.where(np.abs(kicks - beats[j - 1]) < np.abs(kicks - beats[j]), j - 1, j)
        close = np.abs(kicks - beats[nearest]) < 0.05
        for nj in nearest[close]:
            pos_strength[beat[nj] % meter] += 1
    phase["kicks_by_beat_in_bar"] = pos_strength.tolist()

    bars = []
    for k, di in enumerate(down_idx):
        t0 = beats[di]
        t1 = beats[down_idx[k + 1]] if k + 1 < len(down_idx) else t0 + meter * period[di]
        bars.append({"i": k, "t0": t0, "t1": t1, "beats": int((down_idx[k + 1] if k + 1 < len(down_idx) else di + meter) - di)})

    curve = [{"t": b["t0"], "bpm": 60.0 * b["beats"] / (b["t1"] - b["t0"])} for b in bars]
    write_json(p.bd / "beats.json", {
        "version": 1,
        "method": "beat_this final0 (no DBN), local linear fit, kick phase check on drums stem",
        "meter": meter,
        "bpm_median": float(np.median(bpm)),
        "bpm_range": [float(np.percentile(bpm, 5)), float(np.percentile(bpm, 95))],
        "tempo_curve": curve,
        "beats": [{"i": i, "t": beats[i], "bar": int(bar[i]), "beat": int(beat[i]), "bpm": bpm[i]} for i in range(len(beats))],
        "bars": bars,
        "phase_check": phase,
    })
    log(f"{len(beats)} beats, {len(bars)} bars of {meter}, {np.percentile(bpm, 5):.1f} to {np.percentile(bpm, 95):.1f} BPM")
