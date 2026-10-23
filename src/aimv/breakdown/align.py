"""align: word and line times for the known lyrics, on the vocal stem.

Method:
  1. CTC emissions of breakdown/stems/vocals.wav (16 kHz mono) from wav2vec2 large-lv60k-960h (English ASR,
     torchaudio), 30 s windows with 2 s of context each side. MMS_FA (multilingual) is the option for lyrics
     that are not English (project.json "align": {"ctc_model": "mms"}).
  2. Whisper (openai-whisper large-v3-turbo) transcribes the same stem with word timestamps. Its words are
     matched to the lyric words (fuzzy, monotonic, merges like "AGI" = "A G I"). Whisper times are too loose to
     use as word times, so they only give each line a search window (+-2.5 s) and say whether a line was sung.
  3. Lines whisper did not hear, where whisper heard nothing else and the vocal stem has no room either, are
     marked not sung before alignment. One Viterbi pass then aligns all other lines over the whole song:
     lyric tokens in order, a garbage state between words and lines (absorbs ad-libs, repeats and backing
     vocals that are not in the lyrics), each line confined to its window, lines without whisper evidence
     skippable at a cost.
  4. Lines with no support after the pass (whisper heard nothing there and the CTC found next to nothing,
     or no vocal sound they could explain) are marked not sung (`sung: false`, null times): a title line
     above the first tag, a line Suno skipped.
  5. With Suno's own word timings (song/suno/aligned_lyrics.json): its starts are matched and put on our clock
     by the median offset over confident words. A word where our CTC confidence is under 0.5 and Suno is
     more than 250 ms away is aligned again inside Suno's place (first letter within -200..+150 ms of Suno's
     start) and takes Suno's start (`src: "suno"`). Passes repeat until nothing changes (at most 3).
  6. Word starts: the vocal-stem energy rise when the word follows a silence; else the end of the deepest level
     dip before the CTC start, stepping over unvoiced noise (a breath) to a later dip; else the CTC start less
     the song's measured CTC lag. A word whose letters straddle a vocal silence moves to its far side. Ends
     extend over the held vowel while the vocal stays above the silence floor.
  7. Flags: "low_conf" for CTC confidence < 0.15, or < 0.5 with whisper more than 0.5 s away (not for Suno-
     guided words). With Suno: "disagree" when a word not guided by Suno is more than 250 ms from it, and a
     low_conf word within 100 ms of Suno is cleared.

Measured on a second test song (spoken verses, sung trance choruses, chanted drop; no Suno export) against word
onsets read off the vocal spectrogram for 41 random words: spoken median 10 to 15 ms, worst 45 ms; sung and
chanted median 15 ms, all but two within 85 ms. On Full Stop (fast sung DnB, real Suno export) against 9
blind and 23 targeted readings: median 15 ms; the Suno guidance fixed 12 words that were 0.3 to 1.8 s off
(placed on a held vowel of the previous word, on backing pads or vocal chops). Rejected: MMS_FA emitted almost
only blanks on the sung choruses and hears spelled letters badly; whisper word starts were a median 90 to
170 ms early with outliers past 1 s; global CTC without garbage states smeared a chorus where Suno sang a line
pair three times; trusting Suno whenever it disagrees (it labels the stressed vowel, so "enough" and "should"
come out 130 to 200 ms late).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from ..util import MODELS_DIR, load_audio, read_json, torch_device, write_json
from . import lyrics as lyr
from . import sections as sec

NAME = "align"
CONFIG_KEYS = ("align",)
SR = 16000
HOP = 320                 # wav2vec2 frame: 20 ms
FRAME = HOP / SR
WIN_S, CTX_S = 30.0, 2.0  # emission windows
STAR = 3.0                # garbage log-prob per frame (cost of explaining audio as "not the lyrics")
SKIP_PER_CHAR = 6.0       # cost of skipping a line, per character
WINDOW_MARGIN = 2.5       # seconds around whisper's span of a line
ANCHOR_SIM = 0.8          # whisper word counts as evidence at this similarity
ENV_HOP = 0.005           # vocal energy envelope hop
LOW_CONF = 0.15
GAP_MIN = 0.25
DIP_DB = 6.0              # a level dip this deep between two words marks the boundary
DIP_LOOKBACK = 0.35
VOICED_DB = 0.0           # voicing score of a voiced frame (see voicing())
RISE_LIFT_DB = 12.0       # a word onset reaches this far above the silence floor within 80 ms
SUNO_DISAGREE = 0.25
SUNO_TRUST_BELOW = 0.5    # CTC confidence under which Suno's start wins a disagreement (see solve)
SUNO_EARLY, SUNO_LATE = 0.2, 0.15  # a Suno-guided word starts within [suno - 0.2, suno + 0.15] s

CTC_MODELS = {
    "en": ("WAV2VEC2_ASR_LARGE_LV60K_960H", "wav2vec2 large-lv60k-960h"),
    "mms": ("MMS_FA", "MMS_FA mms-300m"),
}


def _cfg(p) -> dict:
    return p.config.get("align", {})


def inputs(p):
    here = Path(__file__).parent
    return [p.bd / "stems/vocals.wav", p.song_dir / "lyrics.txt", p.suno_dir / "aligned_lyrics.json",
            p.bd / "beats.json", p.bd / "meta.json", here / "lyrics.py", here / "sections.py"]


def outputs(p):
    return [p.bd / "words.json", p.bd / "lines.json"]


# ---------------------------------------------------------------- acoustic model


def ctc_emissions(x: np.ndarray, model: str = "en") -> tuple[np.ndarray, dict[str, int], list[int]]:
    """Log-probs (frames, vocab) for 16 kHz mono audio. Returns (E, char->column, word-separator columns)."""
    import torch
    import torchaudio

    bundle = getattr(torchaudio.pipelines, CTC_MODELS[model][0])
    dev = torch_device()
    if model == "mms":
        net = bundle.get_model(with_star=False)
        labels = list(bundle.get_labels(star=None))
    else:
        net = bundle.get_model()
        labels = list(bundle.get_labels())
    net = net.to(dev).eval()
    n = len(x)
    T = n // HOP
    step, ctx = int(WIN_S * SR), int(CTX_S * SR)
    E = None
    with torch.inference_mode():
        for s in range(0, n, step):
            a, b = max(0, s - ctx), min(n, s + step + ctx)
            chunk = torch.from_numpy(np.ascontiguousarray(x[a:b])).to(dev)[None]
            # Unit variance per window, as both models were trained on (repeating the bundle's own step is harmless).
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-7)
            em, _ = net(chunk)
            em = torch.log_softmax(em.float(), dim=-1)[0].cpu().numpy()
            if E is None:
                E = np.full((T, em.shape[1]), -30.0, np.float32)
                E[:, 0] = 0.0
            f0, f1 = s // HOP, min(T, (s + step) // HOP)
            k0 = f0 - a // HOP
            seg = em[k0:k0 + (f1 - f0)]
            E[f0:f0 + len(seg)] = seg
    del net
    lab = {c.lower(): i for i, c in enumerate(labels) if c not in ("-", "|", "*")}
    seps = [i for i, c in enumerate(labels) if c == "|"]
    return E, lab, seps


# ---------------------------------------------------------------- whisper


def whisper_words(x: np.ndarray, model_name: str, language: str | None, log) -> list[dict]:
    """Whisper transcription with word timestamps, split into normalised tokens."""
    import torch
    import whisper
    import whisper.timing as wt

    dev = torch_device()
    if dev == "mps":
        # whisper's DTW casts to float64 on the model device, which MPS cannot do; do it on the CPU.
        wt.dtw = lambda m: wt.dtw_cpu(m.float().cpu().double().numpy())
    root = MODELS_DIR / "whisper"
    try:
        model = whisper.load_model(model_name, device=dev, download_root=str(root))
        r = model.transcribe(x, language=language, word_timestamps=True, condition_on_previous_text=False,
                             temperature=0.0, fp16=(dev == "cuda"), verbose=None)
    except (RuntimeError, NotImplementedError) as e:  # an op the GPU backend lacks: retry on the CPU
        log(f"whisper on {dev} failed ({e}); retrying on cpu")
        model = whisper.load_model(model_name, device="cpu", download_root=str(root))
        r = model.transcribe(x, language=language, word_timestamps=True, condition_on_previous_text=False,
                             temperature=0.0, fp16=False, verbose=None)
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    return whisper_tokens(r)


def whisper_tokens(result: dict) -> list[dict]:
    out = []
    for s in result.get("segments", []):
        for w in s.get("words", []) or []:
            parts = lyr.normalize(w["word"]).split()
            if not parts:
                continue
            n = sum(len(q) for q in parts)
            t, d = float(w["start"]), max(float(w["end"]) - float(w["start"]), 0.0)
            for q in parts:
                dd = d * len(q) / n
                out.append({"norm": q, "t0": t, "t1": t + dd, "p": float(w.get("probability", 1.0)),
                            "raw": w["word"].strip()})
                t += dd
    return out


# ---------------------------------------------------------------- text matching


def similarity(a: str, b: str) -> float:
    a, b = a.replace("'", ""), b.replace("'", "")
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def match_tokens(left: list[str], right: list[str], gap: float = 0.45, max_merge: int = 4,
                 merge_min: float = 0.85) -> list[tuple[list[int], list[int], float]]:
    """Monotonic fuzzy alignment of two word sequences (lyrics vs a transcript).

    Allows one-to-many merges ("agi" = "a g i", "datacenter" = "data center") when the joined strings are near
    identical. Returns [(left indices, right indices, similarity)] for matched groups, in order. On ties the
    earlier occurrence wins, so an extra repeat in the transcript stays unmatched at the end."""
    n, m = len(left), len(right)
    D = np.empty((n + 1, m + 1))
    D[0, :] = -gap * np.arange(m + 1)
    D[:, 0] = -gap * np.arange(n + 1)
    B = np.zeros((n + 1, m + 1, 2), np.int8)
    cache: dict[tuple[int, int, int, int], float] = {}

    def sim(i0, i1, j0, j1):
        k = (i0, i1, j0, j1)
        if k not in cache:
            cache[k] = similarity("".join(left[i0:i1]), "".join(right[j0:j1]))
        return cache[k]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, arg = D[i, j - 1] - gap, (0, 1)
            v = D[i - 1, j] - gap
            if v > best:
                best, arg = v, (1, 0)
            for a in range(1, min(max_merge, i) + 1):
                for b in range(1, min(max_merge, j) + 1):
                    if a > 1 and b > 1:
                        break
                    s = sim(i - a, i, j - b, j)
                    if (a > 1 or b > 1) and s < merge_min:
                        continue
                    v = D[i - a, j - b] + (2 * s - 1) * (a + b) / 2 - 0.05 * (a + b - 2)
                    if v > best:
                        best, arg = v, (a, b)
            D[i, j] = best
            B[i, j] = arg
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        a, b = int(B[i, j, 0]), int(B[i, j, 1])
        if a and b:
            out.append((list(range(i - a, i)), list(range(j - b, j)), sim(i - a, i, j - b, j)))
        i, j = i - a, j - b
    return out[::-1]


# ---------------------------------------------------------------- Viterbi


def _tokens_for(w: lyr.Word, lab: dict[str, int], model: str) -> list[list[int]]:
    """Token ids per sub-word. The English model writes spelled letters as letters; MMS hears their names."""
    subs = [w.say] if model == "mms" else w.norm.split()
    out = [[lab[c] for c in s if c in lab] for s in subs]
    return [t for t in out if t]


def viterbi_align(E: np.ndarray, gcol: np.ndarray, L: lyr.Lyrics, lab: dict[str, int], model: str,
                  windows: list[tuple[int, int] | None], skippable: list[bool],
                  active: list[bool] | None = None,
                  word_windows: dict[int, tuple[int, int, int, int]] | None = None) -> dict[int, dict] | None:
    """Forced alignment of the active lines with garbage gaps and optional line skips.

    `windows` confine each line; `word_windows` confine single words as frames (first token from, first token
    to, other tokens from, other tokens to). Returns {word index:
    {"f": [frames of its tokens], "lp": [log-probs]}} for aligned words, or None when the windows leave
    no path."""
    word_windows = word_windows or {}
    T = E.shape[0]
    Ex = np.concatenate([E, gcol[:, None]], axis=1)
    GC = Ex.shape[1] - 1
    lab_s, kind, word_of, lo, hi, gates = [], [], [], [], [], []

    def add(c, k, wi, w0, w1):
        lab_s.append(c), kind.append(k), word_of.append(wi), lo.append(w0), hi.append(w1)

    lines = [li for li in range(len(L.lines)) if active is None or active[li]]
    for li in lines:
        gates.append(len(lab_s))
        add(GC, "G", -1, 0, T)
        w0, w1 = windows[li] or (0, T)
        ws = L.line_words(li)
        for w in ws:  # the line window must hold its words' windows
            if w.i in word_windows:
                w0, w1 = min(w0, word_windows[w.i][2]), max(w1, word_windows[w.i][3])
        for wk, w in enumerate(ws):
            f0, f1, v0, v1 = word_windows.get(w.i, (w0, w1, w0, w1))
            subs = _tokens_for(w, lab, model) or [[GC]]
            for si, toks in enumerate(subs):
                for k, c in enumerate(toks):
                    first = si == 0 and k == 0
                    add(c, "t", w.i, f0 if first else v0, f1 if first else v1)
                    last_tok = k == len(toks) - 1
                    if not last_tok:
                        add(0, "b", w.i, v0, v1)
                    elif si < len(subs) - 1 or wk < len(ws) - 1:
                        add(GC, "w", w.i, w0, w1)  # the gap after a word follows the line window
    gates.append(len(lab_s))
    add(GC, "G", -1, 0, T)

    lab_a = np.array(lab_s)
    S = len(lab_a)
    kind_a = np.array(kind)
    lo_a, hi_a = np.array(lo), np.array(hi)
    a1 = np.ones(S, bool)
    a1[0] = False
    a2 = np.zeros(S, bool)
    a2[2:] = (kind_a[2:] == "t") & (kind_a[:-2] == "t") & (
        (kind_a[1:-1] == "w") | ((kind_a[1:-1] == "b") & (lab_a[2:] != lab_a[:-2])))
    nchar = np.array([max(1, sum(len(w.norm) for w in L.line_words(li))) for li in lines])
    sk = np.array([skippable[li] for li in lines], bool)
    gsrc, gdst = np.array(gates[:-1])[sk], np.array(gates[1:])[sk]
    pen = SKIP_PER_CHAR * nchar[sk]
    em = Ex[:, lab_a]
    NEG = -1e9
    bp = np.zeros((T, S), np.uint8)
    prev = np.full(S, NEG)
    prev[0] = em[0, 0]
    prev[1] = em[0, 1]
    prev[(lo_a > 0) | (hi_a < 0)] = NEG
    idx = np.arange(S)
    c1 = np.empty(S)
    c2 = np.empty(S)
    c3 = np.empty(S)
    for t in range(1, T):
        c1[0] = NEG
        c1[1:] = prev[:-1]
        c1[~a1] = NEG
        c2[:2] = NEG
        c2[2:] = prev[:-2]
        c2[~a2] = NEG
        c3.fill(NEG)
        if gsrc.size:
            c3[gdst] = prev[gsrc] - pen
        st = np.stack([prev, c1, c2, c3])
        arg = st.argmax(axis=0)
        bp[t] = arg
        cur = st[arg, idx] + em[t]
        cur[(lo_a > t) | (hi_a < t)] = NEG
        prev = cur
    s = S - 1 if prev[S - 1] >= prev[S - 2] else S - 2
    if prev[s] < NEG / 2:
        return None
    back = dict(zip(gdst.tolist(), gsrc.tolist()))
    out: dict[int, dict] = {}
    for t in range(T - 1, -1, -1):
        if kind[s] == "t" and word_of[s] >= 0:
            d = out.setdefault(word_of[s], {"f": [], "lp": []})
            d["f"].append(t)
            d["lp"].append(float(Ex[t, lab_s[s]]))
        a = bp[t, s]
        if a == 1:
            s -= 1
        elif a == 2:
            s -= 2
        elif a == 3:
            s = back[s]
    for d in out.values():
        d["f"].reverse()
        d["lp"].reverse()
    return out


# ---------------------------------------------------------------- vocal energy


def vocal_envelope(path: Path) -> tuple[np.ndarray, float]:
    """RMS level of the vocal stem in dBFS on a 5 ms hop (20 ms window)."""
    x, sr = load_audio(path, mono=True)
    x = x[0]
    hop = int(round(sr * ENV_HOP))
    win = hop * 4
    n = len(x) // hop
    pad = np.pad(x, (win // 2, win // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, win)[::hop][:n]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms), float(np.percentile(20.0 * np.log10(rms), 95))


def _rise_near(db: np.ndarray, t: float, thr: float, before: float = 0.15, after: float = 0.03,
               quiet: float = 0.06, attack: float = 0.08, lift: float = RISE_LIFT_DB) -> float | None:
    """Latest silence-to-sound crossing in [t - before, t + after] preceded by `quiet` s below `thr` and
    followed within `attack` s by a real word level (`lift` dB over the floor), so breaths do not count."""
    i0, i1 = max(0, int((t - before) / ENV_HOP)), min(len(db) - 1, int((t + after) / ENV_HOP))
    q, k = int(quiet / ENV_HOP), int(attack / ENV_HOP)
    for i in range(i1, i0 - 1, -1):
        if db[i] >= thr and i > 0 and db[i - 1] < thr and i - q >= 0 and np.all(db[i - q:i] < thr):
            if db[i:i + k].max() >= thr + lift:
                return i * ENV_HOP
    return None


def silent_runs(db: np.ndarray, t0: float, t1: float, thr: float, min_len: float) -> list[tuple[float, float]]:
    """Stretches inside [t0, t1] where the vocal level stays below `thr`, at least `min_len` long."""
    i0, i1 = max(0, int(np.ceil(t0 / ENV_HOP))), min(len(db), int(t1 / ENV_HOP))
    if i1 <= i0:
        return []
    quiet = db[i0:i1] < thr
    out = []
    k = 0
    while k < len(quiet):
        if quiet[k]:
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            a, b = (i0 + k) * ENV_HOP, (i0 + j) * ENV_HOP
            if b - a >= min_len:
                out.append((max(a, t0), min(b, t1)))
            k = j
        else:
            k += 1
    return out


# ---------------------------------------------------------------- Suno timings


def suno_words(data: dict) -> list[dict]:
    """Words from Suno's aligned_lyrics.json with tags removed. Start times only (ends are unreliable)."""
    words = data.get("aligned_words") if isinstance(data, dict) else None
    out = []
    if words:
        in_tag = False
        for w in words:
            raw = str(w.get("word") or w.get("text") or "")
            start = w.get("start_s", w.get("start"))
            if start is None:
                continue
            text = []
            for ch in raw:
                if ch == "[":
                    in_tag = True
                elif ch == "]":
                    in_tag = False
                elif not in_tag:
                    text.append(ch)
            if w.get("success") is False:
                continue
            for q in lyr.normalize("".join(text)).split():
                out.append({"norm": q, "t0": float(start), "p": float(w.get("p_align", 1.0) or 0.0)})
        return out
    for line in (data.get("aligned_lyrics") or []) if isinstance(data, dict) else []:
        if re.fullmatch(r"\s*\[.*\]\s*", str(line.get("text", ""))):
            continue
        for w in line.get("words") or []:
            start = w.get("start_s", w.get("start"))
            if start is None:
                continue
            for q in lyr.normalize(str(w.get("text") or w.get("word") or "")).split():
                out.append({"norm": q, "t0": float(start), "p": 1.0})
    return out


def suno_starts(L: lyr.Lyrics, data: dict) -> dict[int, float]:
    """Suno's start (its own clock) for each lyric word it has, by lyric word index."""
    sw = suno_words(data)
    if not sw:
        return {}
    raw: dict[int, float] = {}
    for li, si, s in match_tokens([w.norm.replace(" ", "") for w in L.words], [s["norm"] for s in sw]):
        if s >= ANCHOR_SIM:  # in a merged group ("A G I" = "AGI") only the first word has Suno's start
            raw[li[0]] = sw[si[0]]["t0"]
    return raw


def suno_offset(words: list[dict], raw: dict[int, float]) -> float | None:
    """Median of Suno start minus ours over confident words: decoder padding plus Suno's own label lag."""
    pairs = [(k, raw[k] - words[k]["t0"]) for k in raw if words[k]["t0"] is not None]
    if not pairs:
        return None
    good = [d for k, d in pairs if (words[k]["conf"] or 0) >= 0.5] or [d for _, d in pairs]
    return float(np.median(good))


def compare_suno(words: list[dict], L: lyr.Lyrics, data: dict, guided: set[int] | None = None) -> dict | None:
    """Fill suno_t0 and "disagree" flags in `words` (our output dicts). Returns the agreement summary.

    Words in `guided` took their start from Suno: they get suno_t0 but count neither for agreement nor for
    flags, since they agree by construction."""
    guided = guided or set()
    raw = suno_starts(L, data)
    offset = suno_offset([w if w["i"] not in guided else {**w, "t0": None} for w in words], raw)
    if offset is None:
        return None
    errs = []
    for k in raw:
        w = words[k]
        if w["t0"] is None:
            continue
        w["suno_t0"] = raw[k] - offset
        if k in guided:
            continue
        r = abs(w["t0"] - w["suno_t0"])
        errs.append(r)
        if r > SUNO_DISAGREE:
            w["flag"] = "disagree"
        elif w["flag"] == "low_conf" and r <= 0.1:
            w["flag"] = None  # two independent methods agree
    if not errs:
        return None
    errs = np.array(errs)
    return {"median_ms": float(np.median(errs) * 1000), "p90_ms": float(np.percentile(errs, 90) * 1000),
            "matched": len(errs), "guided": len(guided), "offset_ms": offset * 1000}


# ---------------------------------------------------------------- bars


def bar_index(bars: list[dict], t: float) -> int:
    if not bars:
        return 0
    starts = np.array([b["t0"] for b in bars])
    return int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(bars) - 1))


# ---------------------------------------------------------------- main


def run(p, log):
    lp = p.lyrics_path()
    if lp is None:
        raise SystemExit(f"{p.song_dir / 'lyrics.txt'} is missing; align needs the lyrics")
    cfg = _cfg(p)
    model = cfg.get("ctc_model", "en")
    L = lyr.parse_file(lp)
    if not L.words:
        raise SystemExit("lyrics.txt has no lyric lines")
    meta = read_json(p.bd / "meta.json") if (p.bd / "meta.json").exists() else {}
    beats = read_json(p.bd / "beats.json") if (p.bd / "beats.json").exists() else {"bars": []}
    vpath = p.bd / "stems/vocals.wav"
    x, _ = load_audio(vpath, sr=SR, mono=True)
    x = x[0]
    duration = float(meta.get("duration") or len(x) / SR)
    log(f"{len(L.sections)} sections, {len(L.lines)} lines, {len(L.words)} words; {duration:.1f}s of vocals")

    E, lab, seps = ctc_emissions(x, model)
    log(f"ctc emissions: {CTC_MODELS[model][1]}, {E.shape[0]} frames")
    wh: list[dict] = []
    if cfg.get("whisper", True):
        wh = whisper_words(x, cfg.get("whisper_model", "large-v3-turbo"), cfg.get("language", "en"), log)
    db, ref = vocal_envelope(vpath)
    res = solve(L, E, lab, seps, model, wh, db, ref, duration, beats.get("bars", []), p.suno("aligned_lyrics"), log,
                vo=voicing(x, len(db)))
    method = (f"{CTC_MODELS[model][1]} CTC Viterbi on vocals stem with garbage states"
              + (f", whisper {cfg.get('whisper_model', 'large-v3-turbo')} line windows" if wh else "")
              + (", Suno starts where the CTC was unsure and Suno disagreed" if res["stats"]["suno_guided"] else "")
              + ", starts snapped to vocal energy rises and dips")
    suno = res["suno"]
    write_json(p.bd / "words.json", {
        "version": 1, "method": method,
        "suno_agreement": {"median_ms": suno["median_ms"], "p90_ms": suno["p90_ms"]} if suno else None,
        "suno_offset_ms": suno["offset_ms"] if suno else None,
        "stats": res["stats"],
        "words": res["words"],
    })
    write_json(p.bd / "lines.json", {"version": 1, "lines": res["lines"], "gaps": res["gaps"], "extra": res["extra"]})
    st = res["stats"]
    log(f"{st['aligned']}/{st['words']} words aligned, {st['flagged']} flagged, {len(res['gaps'])} gaps, "
        f"{len(res['extra'])} vocal phrases not in the lyrics")


def solve(L: lyr.Lyrics, E: np.ndarray, lab: dict[str, int], seps: list[int], model: str, wh: list[dict],
          db: np.ndarray, ref: float, duration: float, bars: list[dict], suno_data: dict | None, log,
          vo: np.ndarray | None = None) -> dict:
    """Everything after the models: matching, Viterbi passes, refinement, flags, lines, gaps. `vo` is the
    voicing curve of the vocal stem (voicing()), used to step over breaths."""
    T = E.shape[0]
    anchors: dict[int, dict] = {}
    if wh:
        for li_, wi_, s in match_tokens([w.norm.replace(" ", "") for w in L.words], [w["norm"] for w in wh]):
            for k in li_:  # in a merged group only the first word's start is whisper's
                anchors[k] = {"t0": wh[wi_[0]]["t0"], "t1": wh[wi_[-1]]["t1"], "sim": s, "j": wi_,
                              "timed": k == li_[0]}
        good = sum(1 for a in anchors.values() if a["sim"] >= ANCHOR_SIM)
        log(f"whisper: {len(wh)} words, {good}/{len(L.words)} lyric words matched")

    windows: list[tuple[int, int] | None] = []
    evidence: list[float] = []
    for li in range(len(L.lines)):
        ws = L.line_words(li)
        ks = [k for k, w in enumerate(ws) if anchors.get(w.i, {}).get("sim", 0) >= ANCHOR_SIM]
        evidence.append(len(ks) / len(ws))
        if len(ks) >= min(2, len(ws)):
            a0 = anchors[ws[ks[0]].i]["t0"] - WINDOW_MARGIN - 0.4 * ks[0]
            a1 = anchors[ws[ks[-1]].i]["t1"] + WINDOW_MARGIN + 0.4 * (len(ws) - 1 - ks[-1])
            windows.append((max(0, int(a0 / FRAME)), min(T, int(a1 / FRAME) + 1)))
        else:
            windows.append(None)
    use_wh = bool(wh)
    skippable = [(not use_wh) or ev < 0.2 for ev in evidence]
    sil_thr = ref - 28.0
    support = _line_support(L, anchors, db, sil_thr) if use_wh else None
    absent = {li for li in range(len(L.lines)) if use_wh and evidence[li] < 0.2
              and support[li]["heard"] < 0.3 * support[li]["n"] and support[li]["sound"] < 0.5 * support[li]["need"]}
    active = [li not in absent for li in range(len(L.lines))]

    gcol = np.max(E[:, [0] + seps], axis=1) if seps else E[:, 0]
    gcol = np.maximum(gcol, -STAR)
    raw_suno = suno_starts(L, suno_data) if suno_data else {}

    # Pass 1: CTC with whisper windows. Pass 2 (when needed): without the lines found not sung, and with the
    # words where the CTC is unsure and Suno's own alignment disagrees confined to Suno's place.
    guides: dict[int, tuple[int, int]] = {}
    for it in range(3):
        al = viterbi_align(E, gcol, L, lab, model, windows, skippable, active, _guide_frames(guides, T))
        if al is None:  # windows left no path: drop the Suno guides
            log("guided alignment had no path; keeping the unguided one")
            guides = {}
            al = viterbi_align(E, gcol, L, lab, model, windows, skippable, active)
        out_words = _words_from_path(L, al, guides)
        line_sung = _sung_lines(L, out_words, evidence, use_wh, db, sil_thr, support)
        _refine_times(out_words, db, sil_thr, duration, lambda m: None, guides, vo)
        changed = False
        newly = [li for li, s_ in enumerate(line_sung) if not s_ and active[li]]
        if newly:
            for li in newly:
                active[li] = False
            changed = True
        if raw_suno:
            more = _suno_guides(L, out_words, raw_suno, line_sung, T, guides)
            if more:
                guides.update(more)
                changed = True
        if not changed:
            break
    if guides:
        log(f"{len(guides)} words re-aligned inside Suno's place (CTC unsure, Suno more than "
            f"{SUNO_DISAGREE * 1000:.0f} ms away)")
    skipped = [li for li, s_ in enumerate(line_sung) if not s_]
    if skipped:
        log(f"not sung (no evidence): lines {skipped}")
    out_words = _words_from_path(L, al, guides)
    _sung_lines(L, out_words, evidence, use_wh, db, sil_thr, support, force=line_sung)
    _refine_times(out_words, db, sil_thr, duration, log, guides, vo)

    bias = _whisper_bias(out_words, anchors)
    for w in out_words:
        if w["t0"] is None:
            continue
        a = anchors.get(w["i"])
        far = a is not None and a["timed"] and a["sim"] >= ANCHOR_SIM and abs(w["t0"] - (a["t0"] + bias)) > 0.5
        if w["i"] in guides:
            continue  # the start is Suno's (see _suno_guides); the CTC confidence says nothing about it
        if w["conf"] < LOW_CONF or (far and w["conf"] < 0.5):
            w["flag"] = "low_conf"

    suno = compare_suno(out_words, L, suno_data, set(guides)) if suno_data else None
    if suno:
        log(f"suno agreement: median {suno['median_ms']:.0f} ms, p90 {suno['p90_ms']:.0f} ms "
            f"({suno['matched']} words not guided by Suno, offset {suno['offset_ms']:+.0f} ms)")

    out_lines = []
    for li, line in enumerate(L.lines):
        ws = [out_words[k] for k in line.words]
        timed = [w for w in ws if w["t0"] is not None]
        t0 = min(w["t0"] for w in timed) if timed else None
        t1 = max(w["t1"] for w in timed) if timed else None
        out_lines.append({
            "i": li, "text": line.text, "section": None, "t0": t0, "t1": t1, "words": list(line.words),
            "bar0": bar_index(bars, t0) if t0 is not None else None,
            "bar1": bar_index(bars, t1 - 1e-3) if t1 is not None else None,
            "conf": float(np.mean([w["conf"] for w in ws])), "sung": line_sung[li],
        })

    # Section indices: the same plan the sections step writes.
    plan = sec.plan(L, [(ln["t0"], ln["t1"]) if ln["sung"] else None for ln in out_lines], bars, duration)
    line_sec = {li: k for k, s_ in enumerate(plan) for li in s_["lines"]}
    for ln in out_lines:
        ln["section"] = line_sec.get(ln["i"])
    for w in out_words:
        w["section"] = line_sec.get(w["line"])

    gaps = find_gaps(out_words, db, sil_thr, duration)
    extra = extra_vocals(wh, out_words, db, sil_thr)
    stats = {"words": len(out_words), "aligned": sum(w["t0"] is not None for w in out_words),
             "flagged": sum(w["flag"] is not None for w in out_words),
             "low_conf": sum(w["flag"] == "low_conf" for w in out_words),
             "disagree": sum(w["flag"] == "disagree" for w in out_words),
             "suno_guided": len(guides),
             "lines_not_sung": skipped,
             "whisper_matched": sum(1 for a in anchors.values() if a["sim"] >= ANCHOR_SIM),
             "whisper_bias_ms": bias * 1000,
             "conf_median": float(np.median([w["conf"] for w in out_words]))}
    return {"words": out_words, "lines": out_lines, "gaps": gaps, "extra": extra, "stats": stats, "suno": suno,
            "anchors": anchors}


def _words_from_path(L: lyr.Lyrics, al: dict[int, dict], guides: dict) -> list[dict]:
    out = []
    for w in L.words:
        d = al.get(w.i)
        rec = {"i": w.i, "text": w.text, "norm": w.norm, "line": w.line, "section": None, "t0": None, "t1": None,
               "conf": 0.0, "backing": w.backing, "suno_t0": None, "flag": "low_conf",
               "src": "suno" if w.i in guides else "ctc"}
        if d:
            rec["t0"] = d["f"][0] * FRAME
            rec["t1"] = (d["f"][-1] + 1) * FRAME
            rec["conf"] = float(np.exp(np.mean(d["lp"])))
            rec["flag"] = None
        out.append(rec)
    return out


def _suno_guides(L: lyr.Lyrics, words: list[dict], raw: dict[int, float], line_sung: list[bool], T: int,
                 have: dict) -> dict[int, tuple[float, float, float]]:
    """Frame windows around Suno's start for words where our CTC is unsure and Suno disagrees.

    Measured on Full Stop (a real export): where the two disagreed by more than 250 ms and our CTC confidence
    was under 0.5, Suno was right in every case checked on the spectrogram (words the CTC had placed on a held
    vowel of the previous word, on backing vocals or on vocal chops). Where our confidence was high, ours was
    usually right, so those are left alone and only flagged. Returns {word: (suno start on our clock, end of
    its span, start of the next word)}."""
    offset = suno_offset(words, raw)
    if offset is None:
        return {}
    keys = sorted(raw)
    out = {}
    for n, k in enumerate(keys):
        w = words[k]
        if k in have or w["t0"] is None or not line_sung[w["line"]] or w["conf"] >= SUNO_TRUST_BELOW:
            continue
        s = raw[k] - offset
        # Only gross disagreements: Suno labels the stressed vowel rather than the first consonant ("e-NOUGH",
        # "sh-OULD"), so its fine timing is up to about 200 ms late on such words.
        if abs(w["t0"] - s) <= SUNO_DISAGREE:
            continue
        nxt = raw[keys[n + 1]] - offset if n + 1 < len(keys) else s + 1.0
        out[k] = (s, min(max(nxt, s + 0.2), s + 2.0))
    return out


def _guide_frames(guides: dict[int, tuple[float, float]], T: int) -> dict[int, tuple[int, int, int, int]]:
    out = {}
    for k, (s, nxt) in guides.items():
        lo = max(0, int((s - SUNO_EARLY) / FRAME))
        # the word's later letters end before the next word's Suno start (50 ms margin)
        out[k] = (lo, min(T, int((s + SUNO_LATE) / FRAME) + 1), lo, min(T, int((nxt + 0.05) / FRAME) + 1))
    return out


def _line_support(L: lyr.Lyrics, anchors: dict, db: np.ndarray, thr: float) -> list[dict]:
    """Per line: how many whisper words fall between the neighbouring lines' matched words (`heard`), how much
    vocal sound is there (`sound`, s) and how much a line this long needs (`need`, about 60 ms per letter)."""
    good = {k: a for k, a in anchors.items() if a["sim"] >= ANCHOR_SIM}
    line_of = [w.line for w in L.words]
    out = []
    for li in range(len(L.lines)):
        before = [a for k, a in good.items() if line_of[k] < li]
        after = [a for k, a in good.items() if line_of[k] > li]
        j0 = max((a["j"][-1] for a in before), default=-1)
        j1 = min((a["j"][0] for a in after), default=None)
        t0 = max((a["t1"] for a in before), default=0.0)
        t1 = min((a["t0"] for a in after), default=len(db) * ENV_HOP)
        n = len(L.lines[li].words)
        heard = (j1 if j1 is not None else j0 + 1 + n) - j0 - 1
        i0, i1 = int(t0 / ENV_HOP), int(t1 / ENV_HOP)
        sound = float(np.sum(db[i0:i1] >= thr)) * ENV_HOP if i1 > i0 else 0.0
        out.append({"n": n, "heard": heard, "sound": sound, "need": 0.06 * sum(len(w.norm) for w in L.line_words(li))})
    return out


def _sung_lines(L: lyr.Lyrics, words: list[dict], evidence: list[float], use_wh: bool, db: np.ndarray,
                thr: float, support: list[dict] | None = None, force: list[bool] | None = None) -> list[bool]:
    """A line counts as not sung when the path skipped it, or when nothing supports it: whisper did not hear
    it and the CTC fit is poor, and either whisper heard nothing at all there while the CTC found almost
    nothing (vocal pads and chops make sound without words), or there is no vocal sound it could explain.
    `force` imposes an earlier decision. Times of lines not sung are cleared."""
    out = []
    for li in range(len(L.lines)):
        ws = [words[k] for k in L.lines[li].words]
        timed = [w for w in ws if w["t0"] is not None]
        if force is not None:
            sung = force[li] and bool(timed)
        elif not timed:
            sung = False
        else:
            conf = float(np.mean([w["conf"] for w in ws]))
            doubtful = evidence[li] < 0.2 and conf < (0.25 if use_wh else 0.05)
            if doubtful and support is not None and support[li]["heard"] < 0.3 * support[li]["n"] and conf < 0.1:
                pass  # nothing heard, nothing found
            elif doubtful:
                # Vocal sound between the neighbouring lines that only this line could explain.
                prev = [words[k]["t1"] for j in range(li) for k in L.lines[j].words if words[k]["t1"] is not None]
                nxt = [words[k]["t0"] for j in range(li + 1, len(L.lines)) for k in L.lines[j].words
                       if words[k]["t0"] is not None]
                a = max(prev) if prev else 0.0
                b = min(nxt) if nxt else len(db) * ENV_HOP
                i0, i1 = int(a / ENV_HOP), int(b / ENV_HOP)
                active = float(np.sum(db[i0:i1] >= thr)) * ENV_HOP if i1 > i0 else 0.0
                need = 0.06 * sum(len(w["norm"]) for w in ws)  # about 60 ms per letter of speech
                doubtful = active < 0.5 * need
            sung = not doubtful
        out.append(sung)
        if not sung:
            for w in ws:
                w.update(t0=None, t1=None, conf=0.0, flag="low_conf")
    return out


def _whisper_bias(words: list[dict], anchors: dict) -> float:
    d = [a["t0"] - w["t0"] for w in words if w["t0"] is not None and w["conf"] >= 0.5
         for a in [anchors.get(w["i"])] if a and a["timed"] and a["sim"] >= 0.95]
    return -float(np.median(d)) if d else 0.0


def _dip_onset(db: np.ndarray, a: float, b: float, min_depth: float = DIP_DB,
               vo: np.ndarray | None = None) -> float | None:
    """End of the deepest level dip in [a, b]: where the next word starts when there is no silence before it.

    With a voicing curve, when unvoiced noise (a breath) separates that dip from a later dip, the later one is
    the word's start: the breath belongs to no word."""
    i0, i1 = max(1, int(a / ENV_HOP)), min(len(db) - 1, int(b / ENV_HOP))
    if i1 - i0 < 3:
        return None
    seg = db[i0:i1]
    tail = db[i0:min(len(db), i1 + int(0.05 / ENV_HOP))]

    def depth(m):
        return min(seg[: m + 1].max(), tail[m:].max()) - seg[m]

    def onset(m):
        lo, d = seg[m], depth(m)
        k = i0 + m
        while k < len(db) - 1 and db[k] < lo + d / 2:
            k += 1
        return k

    m = int(np.argmin(seg))
    if depth(m) < min_depth:
        return None
    k = onset(m)
    if vo is not None:
        r = 5  # a later dip: the lowest point within 25 ms, deep enough
        later = [j for j in range(m + 1, len(seg)) if seg[j] <= seg[max(0, j - r):j + r + 1].min()
                 and depth(j) >= min_depth]
        for j in later:
            if i0 + j <= k:
                continue
            if np.median(vo[k:i0 + j]) < VOICED_DB:
                m, k = j, onset(j)
            else:
                break
    return k * ENV_HOP


def voicing(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """Voicing score of 16 kHz mono audio on the ENV_HOP grid: dB of the energy below 1 kHz over the energy
    from 2 to 7.5 kHz. Vowels and voiced consonants score above 0; breaths and unvoiced consonants below."""
    hop = int(round(SR * ENV_HOP))
    n_fft = 512
    pad = np.pad(x.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop]
    f = np.fft.rfftfreq(n_fft, 1.0 / SR)
    lo_b, hi_b = (f >= 80) & (f < 1000), (f >= 2000) & (f < 7500)
    win = np.hanning(n_fft)
    out = np.empty(len(frames))
    for c in range(0, len(frames), 4096):  # chunks keep memory small
        spec = np.abs(np.fft.rfft(frames[c:c + 4096] * win, axis=1)) ** 2
        out[c:c + 4096] = (10.0 * np.log10(spec[:, lo_b].mean(axis=1) + 1e-12)
                           - 10.0 * np.log10(spec[:, hi_b].mean(axis=1) + 1e-12))
    if n is not None:
        out = np.pad(out, (0, max(0, n - len(out))), mode="edge")[:n]
    return out


def _refine_times(words: list[dict], db: np.ndarray, thr: float, duration: float, log,
                  guides: dict | None = None, vo: np.ndarray | None = None) -> None:
    """Word starts: the energy rise after a silence when there is one; else the end of the level dip between the
    previous word and this one (vowel-initial words after a vowel come out of CTC late), skipping a breath
    when `vo` is given; else the CTC start less the song's measured CTC lag. Words guided by Suno take Suno's
    start. Ends extend over held sound."""
    timed = sorted((w for w in words if w["t0"] is not None), key=lambda w: w["i"])  # lyric order
    # A word whose letters straddle a vocal silence sits on the far side of it, next to the following word:
    # CTC tends to start a weak word on the tail of the previous sound.
    split = 0
    for w in timed:
        runs = silent_runs(db, w["t0"], w["t1"], thr, GAP_MIN)
        if runs and runs[-1][1] < w["t1"] - FRAME:
            w["t0"] = runs[-1][1]
            split += 1
    if split:
        log(f"{split} words straddled a vocal silence; moved to its far side")
    raw0 = {w["i"]: w["t0"] for w in timed}
    raw1 = {w["i"]: w["t1"] for w in timed}
    how: dict[int, str] = {}
    lags = []
    for w in timed:
        r = _rise_near(db, w["t0"], thr)
        if r is not None:
            lags.append(w["t0"] - r)
            w["t0"] = r
            how[w["i"]] = "rise"
    lag = float(np.clip(np.median(lags), 0.0, 0.06)) if len(lags) >= 5 else 0.03
    for k, w in enumerate(timed):
        if w["i"] in how:
            continue
        a = max(raw1[timed[k - 1]["i"]] if k else 0.0, raw0[w["i"]] - DIP_LOOKBACK)
        d = _dip_onset(db, a, raw0[w["i"]] + 0.02, vo=vo)
        if d is not None:
            w["t0"] = d
            how[w["i"]] = "dip"
        else:
            w["t0"] = max(0.0, w["t0"] - lag)
    for w in timed:  # Suno-guided words take Suno's start (offset-corrected); the CTC had next to nothing there
        if guides and w["i"] in guides:
            w["t0"] = guides[w["i"]][0]
    # Ends: extend over the held vowel while the vocal stays above the silence floor, up to the next word.
    for k, w in enumerate(timed):
        nxt = timed[k + 1]["t0"] if k + 1 < len(timed) else duration
        i = int(w["t1"] / ENV_HOP)
        stop = int(min(nxt, w["t1"] + 4.0, duration) / ENV_HOP)
        while i < stop and i < len(db) and db[i] >= thr:
            i += 1
        w["t1"] = max(w["t1"], min(i * ENV_HOP, nxt))
    # Strictly increasing starts (one frame apart at least), no overlaps, inside the song.
    for k, w in enumerate(timed):
        lo = timed[k - 1]["t0"] + FRAME if k else 0.0
        w["t0"] = min(max(w["t0"], lo), duration - FRAME * (len(timed) - k))
    for k, w in enumerate(timed):
        nxt = timed[k + 1]["t0"] if k + 1 < len(timed) else duration
        w["t1"] = min(max(w["t1"], w["t0"] + FRAME), nxt)
    n_rise = sum(v == "rise" for v in how.values())
    n_dip = sum(v == "dip" for v in how.values())
    log(f"word starts: {n_rise} on vocal onsets after silence, {n_dip} on level dips, "
        f"{len(timed) - n_rise - n_dip} CTC with {lag * 1000:.0f} ms lag removed")


def find_gaps(words: list[dict], db: np.ndarray, thr: float, duration: float) -> list[dict]:
    """Vocal silences >= GAP_MIN between consecutive words, checked against the vocal stem."""
    timed = sorted((w for w in words if w["t0"] is not None), key=lambda w: w["t0"])
    spans = []
    cur = 0.0
    for w in timed:
        if w["t0"] - cur >= GAP_MIN:
            spans.append((cur, w["t0"]))
        cur = max(cur, w["t1"])
    if duration - cur >= GAP_MIN:
        spans.append((cur, duration))
    out = []
    for a, b in spans:
        for s0, s1 in silent_runs(db, a, b, thr, GAP_MIN):
            out.append({"t0": s0, "t1": s1, "dur": s1 - s0})
    return out


def extra_vocals(wh: list[dict], words: list[dict], db: np.ndarray, thr: float) -> list[dict]:
    """Phrases whisper heard where no lyric word was placed (repeats, ad-libs Suno added)."""
    spans = sorted((w["t0"], w["t1"]) for w in words if w["t0"] is not None)
    starts = np.array([s[0] for s in spans]) if spans else np.array([])

    def covered(a, b):
        if not spans:
            return False
        k = np.searchsorted(starts, b + 0.25)
        return any(s1 > a - 0.25 for s0, s1 in spans[max(0, k - 8):k])

    runs, cur = [], []
    for w in wh:
        if covered(w["t0"], w["t1"]):
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and w["t0"] - cur[-1]["t1"] > 0.8:
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        a, b = r[0]["t0"], max(x["t1"] for x in r)
        i0, i1 = int(a / ENV_HOP), max(int(a / ENV_HOP) + 1, int(b / ENV_HOP))
        active = float(np.mean(db[i0:i1] >= thr)) if i1 <= len(db) else 0.0
        if np.mean([x["p"] for x in r]) < 0.4 or active < 0.3:
            continue
        text = []
        for x in r:
            if not text or text[-1] != x["raw"]:
                text.append(x["raw"])
        out.append({"t0": a, "t1": b, "text": " ".join(text)})
    return out
