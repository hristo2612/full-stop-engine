"""Parse a Suno lyrics field (song/lyrics.txt) into sections, lines and words.

Format (as typed into Suno):
    [Verse 1 - spoken, deadpan]       a section tag on its own line: label, then directions after " - "
    Look one. A G I era.              a sung or spoken line
    (It's so over?) WE'RE SO BACK!    words inside ( ) are backing vocals
    [End]                             stops parsing

Each word keeps its `text` as written (with punctuation) and gets a `norm`: lowercase letters, digits spelled
out, apostrophes kept. `say` is what the acoustic aligner is given: like `norm`, but single letters of a
spelled acronym ("A G I") become their spoken names ("ay jee eye"), since the audio says the names.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

TAG_RE = re.compile(r"^\s*\[([^\]]*)\]\s*$")
INLINE_TAG_RE = re.compile(r"\[[^\]]*\]")
KINDS = ("intro", "verse", "prechorus", "chorus", "postchorus", "bridge", "break", "drop", "build", "outro",
         "instrumental", "other")

# Order matters: "pre-chorus" and "post-chorus" contain "chorus", "breakdown" contains "break".
_KIND_RULES: list[tuple[str, str]] = [
    (r"\bpre[\s-]*chorus\b|\bprechorus\b|\bpre[\s-]*hook\b|\blift\b", "prechorus"),
    (r"\bpost[\s-]*chorus\b|\bpostchorus\b|\bpost[\s-]*hook\b", "postchorus"),
    (r"\bchorus\b|\brefrain\b|\bhook\b", "chorus"),
    (r"\bintro\b|\bopening\b", "intro"),
    (r"\boutro\b|\bcoda\b|\bending\b|\bfade[\s-]*out\b|\bend\b", "outro"),
    (r"\bverse\b|\brap\b|\bspoken\b", "verse"),
    (r"\bbridge\b", "bridge"),
    (r"\bdrop\b", "drop"),
    (r"\bbuild[\s-]*up\b|\bbuildup\b|\bbuild\b|\briser?\b", "build"),
    (r"\bbreak[\s-]*down\b|\bbreak\b|\binterlude\b", "break"),
    (r"\binstrumental\b|\bsolo\b|\binst\b", "instrumental"),
]

LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "jee", "h": "aitch", "i": "eye",
    "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "es", "t": "tee", "u": "you", "v": "vee", "w": "doubleyou", "x": "ex", "y": "why", "z": "zee",
}
# Single letters that are ordinary English words when they stand alone.
WORD_LETTERS = {"a", "i", "o"}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def number_to_words(n: int) -> str:
    """Cardinal number in words: 2026 -> 'two thousand twenty six'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    for size, name in _SCALES:
        if n >= size:
            hi, lo = divmod(n, size)
            return number_to_words(hi) + " " + name + ("" if lo == 0 else " " + number_to_words(lo))
    raise AssertionError(n)


def ordinal_words(n: int) -> str:
    words = number_to_words(n).split()
    last = words[-1]
    if last in _ORD:
        words[-1] = _ORD[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _spell_digits(s: str) -> str:
    """Replace digit runs inside a token with words: '11.2' -> 'eleven point two', '3rd' -> 'third'."""

    def dec(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        out = number_to_words(int(whole.replace(",", "")))
        if frac:
            out += " point " + " ".join(_ONES[int(d)] for d in frac)
        return " " + out + " "

    s = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: " " + ordinal_words(int(m.group(1))) + " ", s, flags=re.I)
    s = s.replace("%", " percent ").replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"(\d[\d,]*)(?:\.(\d+))?", dec, s)
    return s


def normalize(text: str) -> str:
    """Lowercase letters, spelled-out digits and apostrophes; words joined by single spaces."""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = _spell_digits(s).lower()
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"(^|\s)'+|'+(\s|$)", " ", s)  # quote marks, not apostrophes
    return " ".join(s.split())


def kind_for(label: str) -> str:
    low = label.lower()
    for pat, kind in _KIND_RULES:
        if re.search(pat, low):
            return kind
    return "other"


def delivery_for(tag: str) -> str:
    """'spoken', 'sung', 'chanted' or '' from a tag's directions."""
    low = tag.lower()
    for key, val in (("chant", "chanted"), ("spoken", "spoken"), ("speak", "spoken"), ("whisper", "spoken"),
                     ("rap", "spoken"), ("sung", "sung"), ("sing", "sung"), ("vocal", "sung")):
        if re.search(r"\b" + key, low):
            return val
    return ""


@dataclass
class Word:
    i: int
    text: str
    norm: str
    say: str
    backing: bool
    line: int
    section: int


@dataclass
class Line:
    i: int
    text: str
    section: int
    words: list[int] = field(default_factory=list)


@dataclass
class Section:
    i: int
    label: str
    kind: str
    tag: str
    delivery: str
    lines: list[int] = field(default_factory=list)


@dataclass
class Lyrics:
    sections: list[Section]
    lines: list[Line]
    words: list[Word]

    def line_words(self, li: int) -> list[Word]:
        return [self.words[k] for k in self.lines[li].words]


def _split_label(tag: str) -> str:
    label = re.split(r"\s+[-\u2013\u2014:]\s+|\s*:\s*", tag, maxsplit=1)[0].strip()  # "-", en or em dash
    return label or tag.strip()


def _say(norm: str, is_acronym_letter: bool) -> str:
    if is_acronym_letter and len(norm) == 1 and norm in LETTER_NAMES:
        return LETTER_NAMES[norm]
    return norm.replace(" ", "")


def _tokens(line: str) -> list[tuple[str, bool]]:
    """Whitespace tokens of a line with a backing flag (inside parentheses)."""
    out: list[tuple[str, bool]] = []
    depth = 0
    for tok in line.split():
        # a token can open and close parentheses: "(yeah)" or "over?)"
        starts_inside = depth > 0 or tok.startswith("(")
        opens, closes = tok.count("("), tok.count(")")
        backing = starts_inside or opens > 0
        depth = max(0, depth + opens - closes)
        clean = tok.replace("(", "").replace(")", "")
        if clean:
            out.append((clean, backing))
    return out


def parse(text: str) -> Lyrics:
    sections: list[Section] = []
    lines: list[Line] = []
    words: list[Word] = []
    counts: dict[str, int] = {}

    def new_section(tag: str) -> Section:
        label = _split_label(tag) if tag else "Lyrics"
        base = label
        counts[base] = counts.get(base, 0) + 1
        # Number repeats that carry no number of their own: Chorus, Chorus 2, ...
        if counts[base] > 1 and not re.search(r"\d\s*$", base):
            label = f"{base} {counts[base]}"
        s = Section(len(sections), label, kind_for(base) if tag else "other", tag, delivery_for(tag))
        sections.append(s)
        return s

    cur: Section | None = None
    for raw in text.splitlines():
        m = TAG_RE.match(raw)
        if m:
            tag = m.group(1).strip()
            if re.fullmatch(r"end", tag, flags=re.I):
                break
            cur = new_section(tag)
            continue
        body = INLINE_TAG_RE.sub(" ", raw).strip()
        if not body:
            continue
        if cur is None:
            cur = new_section("")
        line = Line(len(lines), body, cur.i)
        toks = _tokens(body)
        norms = [normalize(t) for t, _ in toks]
        for k, ((tok, backing), nm) in enumerate(zip(toks, norms)):
            if not nm:
                continue  # pure punctuation such as "-" or "..."
            # A single letter is spelled out ("A G I") unless it is a word on its own ("I", "a") with no
            # single-letter neighbour.
            neigh = [norms[j] for j in (k - 1, k + 1) if 0 <= j < len(norms)]
            acronym = len(nm) == 1 and (nm not in WORD_LETTERS or any(len(n) == 1 for n in neigh))
            w = Word(len(words), tok, nm, _say(nm, acronym), backing, line.i, cur.i)
            words.append(w)
            line.words.append(w.i)
        if line.words:
            lines.append(line)
            cur.lines.append(line.i)
    return Lyrics(sections, lines, words)


def parse_file(path) -> Lyrics:
    from pathlib import Path

    return parse(Path(path).read_text(encoding="utf-8"))
