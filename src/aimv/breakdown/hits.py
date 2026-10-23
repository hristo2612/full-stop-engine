"""hits: sound-event map for the motion engine (kick, snare, hat, vocal_entry, impact, riser).

The JS engine punches the camera on kicks, whips on fills, rides risers into impacts and drives cuts from
these events, so a false hit is worse than a missed one: every detector prefers silence when unsure.

  kick, snare, hat  stems/drums.wav: 11-band filterbank envelopes at 1 ms, band-limited onset novelty per
                    voice, kick-leak cross-suppression (see drum_hits), adaptive song-wide and local
                    level gates, and a vocal-bleed veto from stems/vocals.wav.
  vocal_entry       words.json (when the align step has run) and stems/vocals.wav: first singing after
                    >= 1 bar of vocal silence (see vocal_entries).
  impact            audio/mix.wav: energy steps over two beats either side of each beat (see impacts).
  riser             stems/instrumental.wav: steady per-beat climbs of level or brightness that end on
                    a downbeat (see risers).
  stop              stems/instrumental.wav: the music drops out for at least a beat and comes back
                    (see stops).
  roll              kicks and snares above: 8th-note-or-faster runs that lead into a downbeat (see rolls).

Times are seconds on the mix clock at the sound's attack (20% of the local peak of the band envelope,
the same definition the beat grid's kick phase check uses), not at a frame or analysis-window centre.
Strength s is 0..1 within each type: drum voices and vocal entries map their level in dB against a high
percentile of their own type (24 dB range); impacts and risers are scaled so the song's biggest is 1;
stops and rolls use absolute scales (see their functions) so a small stop stays small.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

NAME = "hits"
CONFIG_KEYS = ("hits",)
RATE = 1000  # envelope frames per second (1 ms)

# Filterbank for the drum stem. Index lists below pick each voice's bands.
BANDS = ((30, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 6000), (6000, 9000), (9000, 13000), (13000, 18000))
KICK_B = [0, 1]       # 30-120 Hz
BODY_B = [0, 1, 2, 3, 4]  # 30 Hz-1 kHz: a kick's whole body, what is left of it in a high-passed break
LOWMID_B = [2, 3, 4]  # 120 Hz-1 kHz
HI_B = [6, 7]         # 2-6 kHz
SNARE_B = [5, 6, 7]   # 1-6 kHz
HAT_B = [8, 9, 10]    # 6-18 kHz
MID_B = [5, 6]        # 1-4 kHz


def _db(p):
    return 10.0 * np.log10(np.maximum(p, 1e-12))


def band_power(x: np.ndarray, sr: int, bands=BANDS, rate: int = RATE, win_s: float = 0.0) -> np.ndarray:
    """Mean power per band, frame k centred on k / rate. Shape (bands, frames).

    Zero-phase Butterworth band-pass, then a moving average whose length follows the band (2.5 periods of
    the lower edge, 3 to 20 ms, at least `win_s`), so low bands do not ripple and high bands keep detail."""
    n = int(len(x) * rate / sr)
    c = np.round(np.arange(n) * (sr / rate)).astype(np.int64)
    out = np.zeros((len(bands), n))
    for i, (lo, hi) in enumerate(bands):
        hi = min(hi, 0.45 * sr)
        if lo >= hi:
            continue
        sos = sps.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        y = sps.sosfiltfilt(sos, x.astype(np.float64))
        cs = np.concatenate([[0.0], np.cumsum(y * y)])
        h = max(1, int(max(np.clip(2.5 / lo, 0.003, 0.02), win_s) * sr / 2))
        a = np.clip(c - h, 0, len(x))
        b = np.clip(c + h, 0, len(x))
        out[i] = (cs[b] - cs[a]) / np.maximum(b - a, 1)
    return out


class _Win:
    """Fast mean power of one envelope over windows [i + a, i + b) for many frame indices i."""

    def __init__(self, p: np.ndarray):
        self.n = len(p)
        self.cs = np.concatenate([[0.0], np.cumsum(p)])

    def mean(self, idx, a, b) -> np.ndarray:
        """a and b are frame offsets: ints, or arrays shaped like idx."""
        idx = np.asarray(idx, dtype=np.int64)
        lo = np.clip(idx + np.asarray(a, dtype=np.int64), 0, self.n)
        hi = np.clip(idx + np.asarray(b, dtype=np.int64), 0, self.n)
        return (self.cs[hi] - self.cs[lo]) / np.maximum(hi - lo, 1)

    def db(self, idx, a, b) -> np.ndarray:
        return _db(self.mean(idx, a, b))


def _novelty(L: np.ndarray, bands: list[int], lag: int) -> np.ndarray:
    """Coarse band-limited spectral flux: positive level change (dB) over `lag` frames, summed over bands."""
    d = L[bands, lag:] - L[bands, :-lag]
    return np.concatenate([np.zeros(lag), np.maximum(d, 0.0).sum(axis=0)])


def _local_ref(times: np.ndarray, levels: np.ndarray, half_s: float, q: float) -> np.ndarray:
    """For each event, the q-th percentile of the levels of events within +-half_s (itself included)."""
    out = np.empty_like(levels)
    lo = np.searchsorted(times, times - half_s)
    hi = np.searchsorted(times, times + half_s, side="right")
    for k in range(len(times)):
        out[k] = np.percentile(levels[lo[k]:hi[k]], q)
    return out


def _near(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """Mask over a: True where some element of sorted b lies within tol."""
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape, bool)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    if len(b) == 1:
        return np.abs(a - b[0]) <= tol
    return np.minimum(np.abs(a - b[j - 1]), np.abs(a - b[j])) <= tol


def _otsu(v: np.ndarray) -> tuple[float, float, float, float]:
    """Two-cluster split of 1-D values. Returns (threshold, low_mean, high_mean, high_fraction)."""
    s = np.sort(v)
    n = len(s)
    if n < 4:
        return float("inf"), float(np.mean(s)) if n else 0.0, float("inf"), 0.0
    cs = np.cumsum(s)
    k = np.arange(1, n)
    m0 = cs[:-1] / k
    m1 = (cs[-1] - cs[:-1]) / (n - k)
    score = k * (n - k) * (m1 - m0) ** 2
    i = int(np.argmax(score))
    return float((s[i] + s[i + 1]) / 2), float(m0[i]), float(m1[i]), float((n - i - 1) / n)


def _strength(levels: np.ndarray, ref: float, range_db: float = 24.0) -> np.ndarray:
    """dB level to 0..1 strength: `ref` (a high percentile of the type) maps to 1, ref - range_db to 0."""
    return np.clip(1.0 + (levels - ref) / range_db, 0.02, 1.0)


def _refine(x: np.ndarray, sr: int, t: np.ndarray, lo: float, hi: float, search_s: float) -> np.ndarray:
    """Move times to the real attack (20% of the local peak of the band envelope), via dsp.refine_attacks."""
    from .dsp import refine_attacks

    if t.size == 0:
        return t
    return refine_attacks(x, sr, np.asarray(t, float), lo, hi, search_s=search_s, frac=0.2)


# ---------------------------------------------------------------------------------------------------- drums

def loud_db(x: np.ndarray, sr: int) -> float:
    """95th percentile of the 100 ms RMS in dBFS: how loud the loud parts of a mono signal are."""
    n = max(1, int(0.1 * sr))
    m = len(x) // n
    if m == 0:
        return float(_db(np.mean(x.astype(np.float64) ** 2)))
    r = _db(np.mean(x[:m * n].astype(np.float64).reshape(m, n) ** 2, axis=1))
    return float(np.percentile(r, 95))


def drum_hits(x: np.ndarray, sr: int, diag: dict | None = None, vocals: np.ndarray | None = None,
              mix_db: float | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Kick, snare and hat onsets from a mono drum stem. Returns {voice: (times, strengths)}.

    kick   30-120 Hz candidates that jump >= 10 dB, carry the hit (sub within 20 dB of the 30 Hz-1 kHz
           level) and reach within 25 dB of the song's kicks (18 dB of the kicks around them), so bass
           bleed and the ripple inside a long kick body do not count. Where a
           break is high-passed (no sub kick within 1.5 s), kick-shaped hits in 30 Hz-1 kHz count too (see
           the code); their strength is their body level against the sub kicks' body level.
    snare  a clap or snare leaves a noise tail in 1-6 kHz that a kick's click does not. At a kick, the tail
           level minus the kick level splits all kicks into two clusters (kick alone vs kick + snare); only the
           upper cluster gets a snare, and only when the clusters are clearly apart and its tail is as loud
           as the snares around it (within 6 dB of their median, when there are any within 4 s). Away from
           kicks the onset must be loud for the song's snares, broadband (not a hat, not a 2 kHz tick) and
           strike (its first 30 ms no more than 3 dB under its tail: a swell is not a hit).
    hat    6-18 kHz onsets at least 40 ms from any kick or snare (a broadband kick or clap excites the hat
           band), brighter above 6 kHz than in 1-4 kHz, loud for the hats around them and for the song.
    vocals (optional, same rate) vetoes snares away from kicks and hats where the vocal stem is 10 dB or
           more louder than the drum stem in that band at that moment: a consonant ("t", "s") that leaked
           into the drum stem (leakage sits well under its source). Vocals sung on the beat over a snare of
           similar level veto nothing.
    mix_db (optional, loud_db of the mix): a voice whose typical level is more than 30 / 35 / 40 dB (kick /
           snare / hat) under the mix is separation bleed of a song without that drum, and is dropped.
    """
    diag = diag if diag is not None else {}
    P = band_power(x, sr)
    L = _db(P)
    L = np.maximum(L, L.max() - 90.0)  # ignore changes deep in the noise floor

    Pk, Ps, Ph = P[KICK_B].sum(0), P[SNARE_B].sum(0), P[HAT_B].sum(0)
    Wk, Ws, Wh = _Win(Pk), _Win(Ps), _Win(Ph)
    Wmid = _Win(P[MID_B].sum(0))
    Wsub = [_Win(P[b]) for b in SNARE_B]
    abs_floor = _db(np.mean(P.sum(0))) - 30.0  # 30 dB under the stem's average power
    if vocals is not None:
        Pv = band_power(vocals, sr, BANDS[SNARE_B[0]:])
        off = SNARE_B[0]
        Wvs = _Win(Pv[[b - off for b in SNARE_B]].sum(0))
        Wvh = _Win(Pv[[b - off for b in HAT_B]].sum(0))
        del Pv

    # ---- kick
    Wb = _Win(P[BODY_B].sum(0))
    cand, _ = sps.find_peaks(_novelty(L, KICK_B, 15), height=6.0, distance=70)
    lev = Wk.db(cand, 0, 60)
    jump = lev - Wk.db(cand, -40, -15)
    # the sub must be a real part of the hit, not the skirt of a 150-250 Hz thump or a snare's body
    ok = (jump >= 10.0) & (lev >= abs_floor) & (lev >= Wb.db(cand, 0, 40) - 20.0)
    k_idx, k_lev = cand[ok], lev[ok]
    k_ref = float(np.percentile(k_lev, 90)) if k_lev.size else 0.0
    keep = k_lev >= k_ref - 25.0
    k_idx, k_lev = k_idx[keep], k_lev[keep]
    if k_idx.size:
        keep = k_lev >= _local_ref(k_idx / RATE, k_lev, 2.0, 90) - 18.0
        k_idx, k_lev = k_idx[keep], k_lev[keep]
    if mix_db is not None and k_ref < mix_db - 30.0:
        k_idx, k_lev = k_idx[:0], k_lev[:0]
    diag["kick"] = {"candidates": int(cand.size), "ref_db": k_ref}
    kt = _refine(x, sr, k_idx / RATE - 0.01, 30.0, 150.0, 0.04)
    k_s = _strength(k_lev, k_ref)

    # ---- kicks of a high-passed break: no sub, the body (120 Hz-1 kHz) carries the hit. Only where no sub
    # kick is within 1.5 s (else the sub kicks already cover the groove), only kick-shaped hits: a quiet
    # 1-6 kHz tail 30-90 ms after the attack (a snare's rattle is not there), little 2-6 kHz in it, and
    # some low end left (60-250 Hz within 23 dB of the body: a rim, snap or wood block has none).
    kb_ref = float(np.percentile(Wb.db(k_idx, 0, 40), 90)) if k_idx.size else None
    cand, _ = sps.find_peaks(_novelty(L, LOWMID_B, 10), height=6.0, distance=70)
    cand = cand[~_near(cand / RATE, np.sort(kt), 1.5)]
    body = Wb.db(cand, 0, 40)
    ok = ((body - Wb.db(cand, -40, -10) >= 10.0) & (Ws.db(cand, 30, 90) - body <= -17.0)
          & (_Win(P[HI_B].sum(0)).db(cand, 0, 40) - body <= -6.0)
          & (_Win(P[1] + P[2]).db(cand, 0, 30) - body >= -23.0) & (body >= abs_floor))
    b_ref = kb_ref if kb_ref is not None else (float(np.percentile(body[ok], 90)) if ok.any() else 0.0)
    ok &= body >= b_ref - 15.0
    b_idx, b_lev = cand[ok], body[ok]
    if b_idx.size:
        keep = b_lev >= _local_ref(b_idx / RATE, b_lev, 2.0, 90) - 10.0
        b_idx, b_lev = b_idx[keep], b_lev[keep]
    if mix_db is not None and b_lev.size and b_ref < mix_db - 30.0:
        b_idx, b_lev = b_idx[:0], b_lev[:0]
    bt = _refine(x, sr, b_idx / RATE - 0.01, 120.0, 1000.0, 0.03)
    diag["kick"]["break_kicks"] = int(b_idx.size)

    # ---- snare at kicks: tail-over-kick level, two clusters
    k_tail = Ws.db(k_idx, 15, 70)
    r = k_tail - k_lev
    thr, m0, m1, frac_hi = _otsu(r)
    apart = (m1 - m0) >= 12.0 and frac_hi >= 0.03
    r_thr = max(thr, m0 + 12.0) if apart else float("inf")
    sk = r > r_thr
    diag["snare_at_kick"] = {"split_db": thr if apart else None, "low_mean": m0, "high_mean": m1,
                             "high_frac": frac_hi, "accepted": int(sk.sum())}

    # ---- snare away from kicks
    cand, _ = sps.find_peaks(_novelty(L, SNARE_B, 8), height=6.0, distance=50)
    # No kick attack from 35 ms before to 80 ms after: the tail window must not hear the kick, and a snare
    # onset just ahead of a kick is a flam the kick masks anyway.
    cand = cand[~_near(cand / RATE + 0.0225, np.sort(np.concatenate([kt, bt])), 0.0575)]
    tail = Ws.db(cand, 15, 70)
    jump = Ws.db(cand, 0, 30) - Ws.db(cand, -35, -8)
    sub = np.stack([w.db(cand, 5, 60) for w in Wsub])           # 1-2, 2-4, 4-6 kHz
    hat_vs_mid = Wh.db(cand, 5, 60) - Wmid.db(cand, 5, 60)
    spread = sub.max(0) - sub.min(0)
    # broadband in 1-6 kHz, and not mostly above 6 kHz (a hat); a bright snare may lean up to 9 dB toward
    # the top when its 1-6 kHz is even (within 8 dB)
    not_hat = (hat_vs_mid <= 6.0) | ((hat_vs_mid <= 9.0) & (spread <= 8.0))
    attack = Ws.db(cand, 0, 30)
    ok = (jump >= 8.0) & (spread <= 15.0) & not_hat & (tail >= abs_floor) & (attack >= tail - 3.0)
    if vocals is not None:
        ok &= Wvs.db(cand, 0, 40) < Ws.db(cand, 0, 40) + 10.0

    # A snare on a kick sounds like the snares around it: when at least two snares away from kicks are
    # within 4 s, its tail must be within 6 dB of their median. A quieter "tail" is the kick's own top
    # (a breakbeat kick layered over a sub kick), not a second drum.
    if sk.any() and ok.any():
        ft, fl = cand[ok] / RATE, tail[ok]
        for i in np.nonzero(sk)[0]:
            near = np.abs(ft - k_idx[i] / RATE) <= 4.0
            if near.sum() >= 2 and k_tail[i] < np.median(fl[near]) - 6.0:
                sk[i] = False
        diag["snare_at_kick"]["kept_after_local_check"] = int(sk.sum())
    # Both kinds together: loud for the song's snares (a soft kick's click passes the ratio test but is
    # no snare) and within 9 dB of the snares around it (a ghost note or bleed is not worth a camera move).
    s_idx = np.concatenate([k_idx[sk], cand[ok]])
    s_lev = np.concatenate([k_tail[sk], tail[ok]])
    at_kick = np.concatenate([np.ones(int(sk.sum()), bool), np.zeros(int(ok.sum()), bool)])
    o = np.argsort(s_idx)
    s_idx, s_lev, at_kick = s_idx[o], s_lev[o], at_kick[o]
    s_ref = float(np.percentile(s_lev, 90)) if s_lev.size else 0.0
    keep = s_lev >= s_ref - 15.0
    if mix_db is not None and s_ref < mix_db - 35.0:
        keep[:] = False
    s_idx, s_lev, at_kick = s_idx[keep], s_lev[keep], at_kick[keep]
    if s_idx.size:
        keep = s_lev >= _local_ref(s_idx / RATE, s_lev, 2.0, 90) - 9.0
        s_idx, s_lev, at_kick = s_idx[keep], s_lev[keep], at_kick[keep]
    f_idx = s_idx[~at_kick]
    diag["snare"] = {"ref_db": s_ref, "free": int(f_idx.size)}

    # ---- hats
    cand, _ = sps.find_peaks(_novelty(L, HAT_B, 5), height=6.0, distance=40)
    st = _refine(x, sr, s_idx / RATE - 0.005, 1000.0, 6000.0, 0.03)
    busy = np.sort(np.concatenate([kt, bt, st]))
    cand = cand[~_near(cand / RATE, busy, 0.04)]
    pk = Wh.db(cand, -2, 25)
    jump = pk - Wh.db(cand, -30, -6)
    bright = (pk - Wmid.db(cand, -2, 25)) >= -3.0
    ok = (jump >= 6.0) & bright & (pk >= abs_floor - 10.0)
    if vocals is not None:
        ok &= Wvh.db(cand, -2, 25) < pk + 10.0
    h_idx, h_lev = cand[ok], pk[ok]
    h_ref = float(np.percentile(h_lev, 90)) if h_lev.size else 0.0
    keep = h_lev >= h_ref - 24.0
    if mix_db is not None and h_ref < mix_db - 40.0:
        keep[:] = False
    h_idx, h_lev = h_idx[keep], h_lev[keep]
    if h_idx.size:
        keep = h_lev >= _local_ref(h_idx / RATE, h_lev, 2.0, 90) - 15.0
        h_idx, h_lev = h_idx[keep], h_lev[keep]
    diag["hat"] = {"ref_db": h_ref}

    # ---- attack times and strengths
    ht = _refine(x, sr, h_idx / RATE - 0.003, 6000.0, min(18000.0, 0.45 * sr), 0.02)
    out = {
        "kick": (np.concatenate([kt, bt]), np.concatenate([k_s, _strength(b_lev, b_ref)])),
        "snare": (st, _strength(s_lev, s_ref)),
        "hat": (ht, _strength(h_lev, h_ref)),
    }
    for k, (t, s) in out.items():
        o = np.argsort(t)
        out[k] = (t[o], s[o])
    return out


# ---------------------------------------------------------------------------------------------- vocals

VOX_BAND = ((150, 8000),)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def vocal_entries(v: np.ndarray, sr: int, bar_len, diag: dict | None = None, mix_db: float | None = None,
                  words: list[tuple[float, float]] | None = None,
                  gap_bars: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """First vocal onset after at least one bar (`gap_bars`) of vocal silence. Returns (times, strengths).

    `bar_len(t)` gives the bar length in seconds around t; `gap_bars` bars of silence make an entry.
    Stem levels are relative to the vocal stem's loud level (90th percentile of its audible part); voice
    is >= 20 dB under that.

    What counts as singing:
      - with `words` (the align step's [(t0, t1)], backing words included): the lyric words, each one
        extended while the stem keeps sounding right after it (a held note the aligner cut short; dips
        under 100 ms allowed, at most 3 s). The stem alone cannot tell the singer from vocal chops, pads
        or a synth that leaked into the vocal stem, and in a drop those fill the gaps between lines.
      - without words: the stem itself (voice runs of at least 50 ms; quieter sound such as reverb
        tails, bleed and breaths is silence).
    An entry is the first singing after a silence of at least `gap_bars` bars (the song's first voice
    always counts) that holds 100 ms of stem voice in its first half second. Its time is the word start
    (moved back along a rising stem level by at most 50 ms; for the song's first word, the stem's first
    voice when it starts within 1 s before) or, without words, where the stem first crosses 30 dB under
    its loud level, then moved to the attack.
    Strength = 0.6 x loudness of the first second (99th percentile of the stem = 1, 18 dB under = 0)
    + 0.4 x how long the voice was away (4 bars or more = 1). With `mix_db` (loud_db of the mix), a vocal
    stem whose loud level is more than 30 dB under the mix is bleed from an instrumental: no entries."""
    diag = diag if diag is not None else {}
    rate = 100
    L = _db(band_power(v, sr, VOX_BAND, rate=rate, win_s=0.03)[0])
    audible = L[L > L.max() - 60.0]
    if audible.size == 0 or L.max() < -80.0:
        return np.array([]), np.array([])
    ref = float(np.percentile(audible, 90))
    top = float(np.percentile(audible, 99))
    diag["vocal"] = {"ref_db": ref, "voice_db": ref - 20.0, "source": "words" if words else "stem"}
    if mix_db is not None and ref < mix_db - 30.0:
        return np.array([]), np.array([])
    n = len(L)
    loud = L >= ref - 20.0
    st, en = _runs(loud)
    keep = (en - st) >= 5
    st, en = st[keep], en[keep]

    if words:
        spans = []
        for t0, t1 in sorted(words):
            a = int(np.clip(round(t0 * rate), 0, n))
            b = int(np.clip(round(t1 * rate), a, n))
            limit, quiet = min(n, b + 3 * rate), 0
            while b < limit and quiet < 10:  # follow a held note past the word's end
                quiet = 0 if loud[b] else quiet + 1
                b += 1
            spans.append((a, b - quiet))
        st = np.array([x[0] for x in spans], int)
        en = np.maximum.accumulate(np.array([x[1] for x in spans], int))
        first_stem = int(np.nonzero(loud)[0][0]) if loud.any() else None
    times, levels, away = [], [], []
    prev_end = None
    for a, b in zip(st, en):
        t = a / rate
        gap = (a - prev_end) / rate if prev_end is not None else np.inf
        first = prev_end is None
        prev_end = max(b, prev_end or 0)
        if gap < gap_bars * bar_len(t):
            continue
        if words and first and first_stem is not None and 0 <= a - first_stem <= rate:
            a = first_stem
        if loud[max(0, a - 10):a + rate // 2].sum() < 10:
            continue  # nothing audible where the word should be: a misplaced or whispered word
        onset = a
        if words and not (first and a == first_stem):
            # the aligner snaps word starts to energy rises: only follow the rise itself back (at most 50 ms),
            # never into a pad or chop that was already sounding
            while onset > 0 and a - onset < 5 and ref - 30.0 <= L[onset - 1] < L[onset] - 1.0:
                onset -= 1
        else:
            while onset > 0 and a - onset < 30 and L[onset - 1] >= ref - 30.0:
                onset -= 1
        seg = L[a:a + rate]
        times.append(onset / rate)
        levels.append(float(_db(np.mean(10 ** (seg[seg >= ref - 20.0] / 10)))) if np.any(seg >= ref - 20.0)
                      else ref - 20.0)
        away.append(min(gap / bar_len(t) / 4.0, 1.0))
    t = np.array(times)
    if t.size:
        t = _refine(v, sr, np.maximum(t - 0.01, 0.0), 150.0, 8000.0, 0.02 if words else 0.04)
    s = 0.6 * _strength(np.array(levels), top, 18.0) + 0.4 * np.array(away)
    return t, np.clip(s, 0.02, 1.0)


# --------------------------------------------------------------------------------------------- impacts

MIX_BANDS = ((30, 150), (150, 600), (600, 2500), (2500, 7000), (7000, 16000))


def impacts(mix: np.ndarray, sr: int, beats: np.ndarray, downbeat: np.ndarray, kicks: np.ndarray,
            diag: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Big energy steps in the mix (drops, section hits), evaluated on beats. Returns (times, strengths).

    For every beat the mix is compared over the two beats after vs the two beats before, in 5 bands and
    in total. An impact needs either
      - the total to rise >= 6 dB with at least 2 bands rising >= 3 dB (one instrument coming back is not
        an impact), or
      - the low band (30-150 Hz) to rise >= 30 dB with the total up >= 3 dB (kick and bass return under
        a riser that was already as loud as the drop),
    and must land in a loud part of the song (post level within 6 dB of the song's median), so a stab in
    a quiet breakdown does not count. Score = max(total rise, low rise / 3) capped at 24 dB, plus how loud
    the landing is against the median (+-6 dB), so a drop into the loudest chorus outranks a short gap.
    Peaks are kept at least 2 bars apart; a downbeat next to the peak wins when it scores at least half
    as much (impacts belong on the one). The time is the kick attack when a kick is within 50 ms, else
    the beat. The first and last two beats are never impacts (no "before" or "after" to compare)."""
    diag = diag if diag is not None else {}
    if beats.size < 8:
        return np.array([]), np.array([])
    rate = 100
    P = band_power(mix, sr, MIX_BANDS, rate=rate, win_s=0.02)
    Wb = [_Win(p) for p in P]
    Wt = _Win(P.sum(0))
    period = np.gradient(beats)
    idx = np.round(beats * rate).astype(np.int64)
    w = np.maximum(np.round(2 * period * rate).astype(np.int64), 5)
    nb = len(beats)
    step = np.array([W.db(idx, 1, w) - W.db(idx, -w, -3) for W in Wb])
    post = Wt.db(idx, 1, w)
    total = post - Wt.db(idx, -w, -3)
    tot_p = P.sum(0)
    song_med = float(np.median(_db(tot_p[tot_p > 0]))) if np.any(tot_p > 0) else -60.0
    low = step[0]
    raw = np.minimum(np.maximum(total, low / 3.0), 24.0) + np.clip(post - song_med, -6.0, 6.0)
    broad = (total >= 6.0) & ((step >= 3.0).sum(0) >= 2)
    drop = (low >= 30.0) & (total >= 3.0)  # kick and bass return under a riser that was already loud
    ok = (broad | drop) & (post >= song_med - 6.0)
    ok &= (idx - w >= 0) & (idx + w <= P.shape[1])
    score = np.where(ok, raw, 0.0)

    chosen: list[int] = []
    for i in np.argsort(-score):
        if score[i] <= 0:
            break
        j = i
        if not downbeat[i]:
            near = [k for k in (i - 1, i + 1) if 0 <= k < nb and downbeat[k]]
            if near and raw[near[0]] >= 0.5 * raw[i] and total[near[0]] > 0:
                j = near[0]
        if any(abs(beats[j] - beats[c]) < 8 * period[j] * 0.99 for c in chosen):
            continue
        chosen.append(j)
    chosen.sort()
    t = beats[chosen].astype(float)
    if kicks.size and t.size:
        k = np.clip(np.searchsorted(kicks, t), 1, max(len(kicks) - 1, 1))
        k0 = np.clip(k - 1, 0, len(kicks) - 1)
        k1 = np.clip(k, 0, len(kicks) - 1)
        cand = np.where(np.abs(kicks[k0] - t) < np.abs(kicks[k1] - t), kicks[k0], kicks[k1])
        t = np.where(np.abs(cand - t) <= 0.05, cand, t)
    sc = np.maximum(raw[chosen], 0.0)
    diag["impact"] = {"song_median_db": song_med, "beats_passing": int(ok.sum())}
    s = sc / sc.max() if sc.size and sc.max() > 0 else sc
    return t, s


# ---------------------------------------------------------------------------------------------- risers

def _kendall(v: np.ndarray) -> float:
    n = len(v)
    if n < 3:
        return 0.0
    d = np.sign(v[None, :] - v[:, None])
    iu = np.triu_indices(n, 1)
    return float(d[iu].mean())


def risers(inst: np.ndarray, sr: int, beats: np.ndarray, downbeat: np.ndarray, impact_t: np.ndarray,
           diag: dict | None = None) -> list[dict]:
    """Spans of steadily rising energy or brightness that end on a downbeat. Returns [{t0, t1, s, ...}].

    Features of the instrumental stem (no vocals), one value per beat: level above 2.5 kHz (noise sweeps,
    snare rolls, cymbals, filter opens) and whole level (swells, rolls). A candidate is a span of 2 to 8
    bars ending on a downbeat (the climb may stop up to 2 beats early if those beats drop out: the
    silence before a drop). A feature must, over the span:
      - rise >= 6 dB from the first to the last quarter, every quarter-to-quarter step between 10% and
        60% of the rise and no beat-to-beat jump over 60% of it (a steady ramp, not a section change),
      - have Kendall tau >= 0.7 over its beats and end within 3 dB of its top,
      - contain no impact (the energy has not arrived yet),
      - stop climbing at the downbeat: unless an impact lands there, the next two beats must not keep
        rising (then the riser ends later).
    One-bar spans count only when strictly rising by >= 9 dB into an impact. The start moves forward to
    where the climb begins (the last beat within 15% of the rise of the bottom). Score = rise (capped at
    18 dB, so a fade-in from silence does not dwarf everything) x tau^2, x1.25 when ending on an impact;
    the best spans win and overlapping ones are dropped."""
    diag = diag if diag is not None else {}
    if beats.size < 8:
        return []
    rate = 50
    P = band_power(inst, sr, MIX_BANDS, rate=rate, win_s=0.04)
    floor = _db(np.percentile(P.sum(0), 95)) - 50.0
    feats = {"high": P[3] + P[4], "level": P.sum(0)}
    edges = np.round(beats * rate).astype(np.int64)
    vals = {}
    for name, f in feats.items():
        cs = np.concatenate([[0.0], np.cumsum(f)])
        a, b = np.clip(edges[:-1], 0, len(f)), np.clip(edges[1:], 0, len(f))
        vals[name] = np.maximum(_db((cs[b] - cs[a]) / np.maximum(b - a, 1)), floor)  # beat i = [b_i, b_i+1)

    imp = np.asarray(impact_t, float)

    def lands(t: float) -> bool:
        return bool(imp.size and np.min(np.abs(imp - t)) < 0.1)

    cands = []
    for di in np.nonzero(downbeat)[0]:
        for nbar in range(1, 9):
            start = di - 4 * nbar
            if start < 0:
                break
            for gap in (0, 1, 2):
                stop = di - gap
                if stop - start < 4:
                    continue
                if gap and vals["level"][stop:di].max() > vals["level"][stop - 1] - 6.0:
                    continue  # the beats skipped before the downbeat must drop out
                for name, v in vals.items():
                    seg = v[start:stop]
                    q = np.array([part.mean() for part in np.array_split(seg, 4)])
                    rise = float(q[-1] - q[0])
                    if rise < 6.0 or seg[-1] < seg.max() - 3.0:
                        continue
                    dq = np.diff(q)
                    if dq.min() < 0.1 * rise or dq.max() > 0.6 * rise or np.diff(seg).max() > 0.6 * rise:
                        continue  # one jump (a section change), not a ramp
                    tau = _kendall(seg)
                    if tau < 0.7:
                        continue
                    if nbar == 1 and not (tau > 0.99 and rise >= 9.0 and lands(beats[di])):
                        continue  # one bar is too short to tell a sweep from a fill, unless it is clean
                    if np.any((imp > beats[start] + 0.1) & (imp < beats[di] - 0.1)):
                        continue  # the energy already arrived inside the span
                    ends_on_impact = lands(beats[di])
                    after = v[di:di + 2]
                    if not ends_on_impact and after.size and after.mean() > seg[-1] + 1.0:
                        continue  # still climbing past this downbeat: not where it ends
                    # start where the climb starts: the last beat still within 15% of the rise of the bottom
                    sm = np.convolve(seg, np.ones(2) / 2, mode="same")
                    bottom = np.nonzero(sm <= q[0] + 0.15 * rise)[0]
                    first = int(bottom[-1]) if bottom.size and bottom[-1] < len(seg) - 4 else 0
                    cands.append({"t0": float(beats[start + first]), "t1": float(beats[di]),
                                  "score": min(rise, 18.0) * tau * tau * (1.25 if ends_on_impact else 1.0),
                                  "rise_db": rise, "tau": tau, "feature": name, "gap_beats": gap})
    cands.sort(key=lambda c: -c["score"])
    out: list[dict] = []
    for c in cands:
        if any(c["t0"] < o["t1"] - 1e-3 and o["t0"] < c["t1"] - 1e-3 for o in out):
            continue
        out.append(c)
    out.sort(key=lambda c: c["t0"])
    if out:
        m = max(c["score"] for c in out)
        for c in out:
            c["s"] = c["score"] / m
            c["impact"] = lands(c["t1"])
    diag["riser"] = {"candidates": len(cands)}
    return out


# ----------------------------------------------------------------------------------------------- stops

def stops(inst: np.ndarray, sr: int, beats: np.ndarray, impact_t: np.ndarray, drums: np.ndarray | None = None,
          diag: dict | None = None) -> list[dict]:
    """Spans where the instrumental drops out for at least a beat and then comes back. Returns
    [{t0, t1, s, end_impact, depth_db}].

    Level: broadband RMS of the instrumental stem (vocals may carry on over a stop), 10 ms frames.
    A frame is out when it is 20 dB under the music around it (90th percentile over +-8 s). Out runs
    split by a blip of at most a third of a beat that stays 10 dB under that reference are joined.
    A run is a stop when:
      - it lasts at least one beat (from where the music fell to where it comes back),
      - its median level is 24 dB or more under the louder of the bar before and the bar after (75th
        percentiles), and 10 dB or more under each of them (it fell, and it returns), and each of three
        bands (30-250 Hz, 250 Hz-2.5 kHz, 2.5-16 kHz) that carries the music around it drops by 15 dB or
        more (a hole in the bass line under running hats is the groove, not a stop),
      - something comes back: the bar after reaches within 25 dB of the song's loud level, and there
        was music before (so the song's start, a quiet intro or a fade-out ending are not stops),
      - no drum hit (`drums`: kick, snare and hat times) inside it, 80 ms from either edge.
    t0 is where the level last stood within 6 dB of the floor of the bar before (its 25th percentile:
    the cut, before any reverb tail; at most 2 beats back), t1 is where it first returns to within 10 dB
    of the bar after. end_impact is the impact within 150 ms of t1, else None.
    s = depth (dB, 40 = 1) x sqrt(beats / 8), capped at 1."""
    diag = diag if diag is not None else {}
    rate = 100
    PB = band_power(inst, sr, ((30, 250), (250, 2500), (2500, 16000)), rate=rate, win_s=0.03)
    L = _db(PB.sum(0))
    n = len(L)
    if n < rate or beats.size < 4:
        return []
    loud = float(np.percentile(L, 95))
    L = np.maximum(L, loud - 60.0)
    LB = np.maximum(_db(PB), loud - 60.0)
    from scipy.ndimage import percentile_filter

    ref = percentile_filter(L, 90, size=16 * rate + 1, mode="nearest")
    out = L < ref - 20.0
    period = np.median(np.diff(beats))

    def beat_at(t: float) -> float:
        j = int(np.clip(np.searchsorted(beats, t), 1, len(beats) - 1))
        return float(beats[j] - beats[j - 1]) if len(beats) > 1 else period

    st, en = _runs(out)
    runs: list[list[int]] = []
    for a, b in zip(st, en):
        if runs:
            pa, pb = runs[-1]
            if a - pb <= beat_at(a / rate) * rate / 3 and np.all(L[pb:a] < ref[pb:a] - 10.0):
                runs[-1][1] = b
                continue
        runs.append([int(a), int(b)])

    imp = np.asarray(impact_t, float)
    found = []
    for a, b in runs:
        p = beat_at(a / rate)
        bar = int(4 * p * rate)
        if a < bar // 2 or b >= n:
            continue  # nothing before it, or it never comes back
        pre = float(np.percentile(L[max(0, a - bar):a], 75))
        post = float(np.percentile(L[b:min(n, b + bar)], 75))
        mid = float(np.median(L[a:b]))
        depth = max(pre, post) - mid
        if depth < 24.0 or pre - mid < 10.0 or post - mid < 10.0 or post < loud - 25.0:
            continue
        # every band that carries the music drops out (a hole in the bass line under running hats is the
        # groove, not a stop); a band 30 dB under the music around it has nothing to drop
        drops = []
        for k in range(len(LB)):
            around = max(np.percentile(LB[k, max(0, a - bar):a], 75), np.percentile(LB[k, b:min(n, b + bar)], 75))
            if around >= max(pre, post) - 30.0:
                drops.append(around - np.median(LB[k, a:b]))
        if drops and min(drops) < 15.0:
            continue
        floor_pre = float(np.percentile(L[max(0, a - bar):a], 25))  # the groove between its hits
        a0 = a
        while a0 > max(0, a - int(2 * p * rate)) and L[a0 - 1] < floor_pre - 6.0:
            a0 -= 1
        b1 = b
        while b1 < n and L[b1] < post - 10.0:
            b1 += 1
        t0, t1 = a0 / rate, b1 / rate
        if t1 - t0 < 0.99 * p:
            continue
        if drums is not None and np.any((drums > t0 + 0.08) & (drums < t1 - 0.08)):
            continue
        end_imp = None
        if imp.size and np.min(np.abs(imp - t1)) <= 0.15:
            end_imp = float(imp[np.argmin(np.abs(imp - t1))])
        beats_long = (t1 - t0) / p
        found.append({"t0": t0, "t1": t1, "end_impact": end_imp, "depth_db": depth, "beats": beats_long,
                      "s": float(min(1.0, depth / 40.0) * min(1.0, np.sqrt(beats_long / 8.0)))})
    diag["stop"] = {"runs": len(runs), "found": len(found)}
    return found


# ----------------------------------------------------------------------------------------------- rolls

def rolls(kicks: np.ndarray, snares: np.ndarray, beats: np.ndarray, downbeat: np.ndarray,
          diag: dict | None = None) -> list[dict]:
    """Dense regular kick or snare runs (8ths or faster) that lead into a downbeat. Returns
    [{t0, t1, s, voice, div}].

    Per voice, consecutive hits closer than 0.6 beat form a run (one gap of up to 1.15 beats, a skipped
    or missed note, is bridged). A run is a roll when it has at least 4 hits over at least 2 beats and
    at most 4 bars, its last hit falls within a beat before a downbeat, and the voice does not carry on
    at that density past the downbeat (then it is a groove, not a roll). t0 is the first hit, t1 the
    downbeat it leads into. Kick and snare rolls into the same downbeat merge (voice "kick+snare").
    div is the densest regular subdivision in the roll, in notes per bar (8 = 8ths, 16 = 16ths).
    s = min(1, beats / 8) x min(1, hits per beat / 4) for the denser voice."""
    diag = diag if diag is not None else {}
    if beats.size < 8:
        return []
    downs = beats[downbeat]
    period = np.median(np.diff(beats))

    def beat_at(t: float) -> float:
        j = int(np.clip(np.searchsorted(beats, t), 1, len(beats) - 1))
        return float(beats[j] - beats[j - 1])

    found: dict[float, dict] = {}
    for voice, t in (("kick", np.sort(kicks)), ("snare", np.sort(snares))):
        if t.size < 4:
            continue
        ioi = np.diff(t)
        p = np.array([beat_at(x) for x in t[:-1]])
        dense = ioi <= 0.6 * p
        bridge = (~dense) & (ioi <= 1.15 * p)
        # a lone bridge between two dense IOIs joins them
        join = dense.copy()
        for i in np.nonzero(bridge)[0]:
            if 0 < i < len(dense) - 1 and dense[i - 1] and dense[i + 1]:
                join[i] = True
        st, en = _runs(join)
        for a, b in zip(st, en):  # IOIs a..b-1 -> hits a..b
            h0, h1 = t[a], t[b]
            pb = beat_at(h1)
            nxt = downs[downs > h1 - 0.05 * pb]
            if nxt.size == 0:
                continue
            d = float(nxt[0])
            if d - h1 > 1.05 * pb:
                continue  # does not lead into a downbeat
            nh = b - a + 1
            span_beats = (d - h0) / pb
            if nh < 4 or span_beats < 2.0 or span_beats > 16.0:
                continue
            after = t[(t > d - 0.05 * pb) & (t < d + 2.0 * pb)]
            if after.size >= 3 and np.median(np.diff(after)) <= 0.6 * pb:
                continue  # the same density carries on: a groove
            run_ioi = ioi[a:b][dense[a:b]]
            per_bar = 4.0 * pb / float(np.percentile(run_ioi, 25))
            div = min((8, 12, 16, 24, 32), key=lambda v: abs(v - per_bar))
            rate = nh / max(span_beats, 1e-6)
            s = min(1.0, span_beats / 8.0) * min(1.0, rate / 4.0)
            key = round(d, 3)
            if key in found:
                f = found[key]
                f["t0"] = min(f["t0"], float(h0))
                f["voice"] = "kick+snare"
                f["div"] = max(f["div"], div)
                f["s"] = max(f["s"], s)
            else:
                found[key] = {"t0": float(h0), "t1": d, "s": s, "voice": voice, "div": div}
    out = sorted(found.values(), key=lambda r: r["t0"])
    diag["roll"] = {"found": len(out)}
    return out


# ------------------------------------------------------------------------------------------------- step

def inputs(p):
    return [p.bd / "stems/drums.wav", p.bd / "stems/vocals.wav", p.bd / "stems/instrumental.wav",
            p.bd / "audio/mix.wav", p.bd / "beats.json", p.bd / "words.json"]


def outputs(p):
    return [p.bd / "hits.json"]


def _words(p) -> list[tuple[float, float]] | None:
    """Word spans from the align step (words.json), or None when it has not run or found no words."""
    from ..util import read_json

    f = p.bd / "words.json"
    if not f.exists():
        return None
    try:
        ws = read_json(f).get("words", [])
    except (ValueError, OSError):
        return None
    out = [(float(w["t0"]), float(w["t1"])) for w in ws
           if isinstance(w.get("t0"), int | float) and isinstance(w.get("t1"), int | float)]
    return out or None


def _grid(p) -> tuple[np.ndarray, np.ndarray, list]:
    from ..util import read_json

    bj = read_json(p.bd / "beats.json")
    beats = np.array([b["t"] for b in bj["beats"]], float)
    down = np.array([b["beat"] == 0 and b["bar"] >= 0 for b in bj["beats"]], bool)
    return beats, down, bj.get("bars", [])


def run(p, log):
    from ..util import load_audio, write_json

    beats, down, bars = _grid(p)
    bar_t0 = np.array([b["t0"] for b in bars], float)
    bar_d = np.array([b["t1"] - b["t0"] for b in bars], float)
    med_bar = float(np.median(bar_d)) if bar_d.size else 2.0

    # project.json {"hits": {"vocal_gap_bars": 0.75}} shortens the vocal silence that makes an entry (the
    # contract is 1 bar; at 170+ BPM a designed pause before a drop can be shorter than a bar)
    gap_bars = float(p.config.get("hits", {}).get("vocal_gap_bars", 1.0))

    def bar_len(t: float) -> float:
        if bar_t0.size == 0:
            return med_bar
        return float(bar_d[int(np.clip(np.searchsorted(bar_t0, t) - 1, 0, len(bar_d) - 1))])

    diag: dict = {}
    mix, msr = load_audio(p.bd / "audio/mix.wav", mono=True)
    mix_db = loud_db(mix[0], msr)
    vox, vsr = load_audio(p.bd / "stems/vocals.wav", mono=True)
    ve = vocal_entries(vox[0], vsr, bar_len, diag, mix_db=mix_db, words=_words(p), gap_bars=gap_bars)
    drums, sr = load_audio(p.bd / "stems/drums.wav", mono=True)
    same = vsr == sr and vox.shape[1] == drums.shape[1]
    dr = drum_hits(drums[0], sr, diag, vocals=vox[0] if same else None, mix_db=mix_db)
    del drums, vox
    im = impacts(mix[0], msr, beats, down, dr["kick"][0], diag)
    del mix
    inst, isr = load_audio(p.bd / "stems/instrumental.wav", mono=True)
    rs = risers(inst[0], isr, beats, down, im[0], diag)
    drum_t = np.sort(np.concatenate([dr[k][0] for k in ("kick", "snare", "hat")]))
    sp = stops(inst[0], isr, beats, im[0], drums=drum_t, diag=diag)
    del inst
    rl = rolls(dr["kick"][0], dr["snare"][0], beats, down, diag)

    def evs(t, s):
        return [{"t": float(a), "s": float(b)} for a, b in zip(t, s)]

    events = {
        "kick": evs(*dr["kick"]), "snare": evs(*dr["snare"]), "hat": evs(*dr["hat"]),
        "vocal_entry": evs(*ve), "impact": evs(*im),
        "riser": [{"t0": r["t0"], "t1": r["t1"], "s": r["s"]} for r in rs],
        "stop": [{"t0": x["t0"], "t1": x["t1"], "s": x["s"], "end_impact": x["end_impact"]} for x in sp],
        "roll": [{"t0": x["t0"], "t1": x["t1"], "s": x["s"], "voice": x["voice"], "div": x["div"]} for x in rl],
    }
    write_json(p.bd / "hits.json", {
        "version": 1,
        "method": "drums: 11-band filterbank onsets with kick-leak cross-suppression on stems/drums.wav; "
                  + ("vocal_entry: lyric word gaps (words.json) timed on the vocal stem; "
                     if diag.get("vocal", {}).get("source") == "words" else "vocal_entry: vocal stem gate; ")
                  + f"vocal gap {gap_bars:g} bar; impact: 5-band mix energy steps on beats; "
                  "riser: steady rise of instrumental level/brightness ending on a downbeat; "
                  "stop: instrumental dropout >= 1 beat that comes back; roll: 8th-or-faster kick/snare run "
                  "into a downbeat",
        "events": events,
        "counts": {k: len(v) for k, v in events.items()},
    })
    log(", ".join(f"{len(v)} {k}" for k, v in events.items()))
