# Phase 3: song breakdown

Takes a finished song and turns it into the timing data every later phase is placed on. Run it again
whenever the song changes.

```
aimv breakdown projects/<slug>            # all steps, cached
aimv breakdown projects/<slug> --only align --force
```

## Inputs (`projects/<slug>/song/`)

| File | Required | Notes |
|---|---|---|
| `master.wav` (or `.mp3`, `.m4a`, `.flac`) | yes | The exact take the video uses |
| `lyrics.txt` | for word timing | Suno lyrics field as written: `[Section - directions]` tags on their own lines, `(parenthetical)` backing vocals |
| `suno/` | no | Optional exports from the song generator, if you have them: `clip.json`, `aligned_lyrics.json`, `downbeats.json`, `key.json`, `novelty_sections.json`, optional `stems/*.wav` |

`project.json` at the project root holds `fps` (default 24) used for envelopes.

## Steps

Each step is a module in `aimv/breakdown/`, runs on its own, reads earlier outputs from disk, and is skipped
when its inputs have not changed (hash recorded in `breakdown/.cache.json`).

| Step | Module | Reads | Writes |
|---|---|---|---|
| ingest | `ingest.py` | song audio | `audio/mix.wav` (48 kHz stereo, 24-bit, untouched level), `audio/mix16k.wav` (mono, for models), `meta.json` |
| separate | `separate.py` | `audio/mix.wav` | `stems/{vocals,instrumental,drums,bass,other}.wav` |
| beats | `beats.py` | mix, `stems/drums.wav` | `beats.json` |
| align | `align.py` | `stems/vocals.wav`, `lyrics.txt`, `suno/aligned_lyrics.json` | `words.json`, `lines.json` |
| sections | `sections.py` | beats, lines, stems, suno sections | `sections.json` |
| hits | `hits.py` | stems, beats | `hits.json` |
| envelopes | `envelopes.py` | stems, mix | `envelopes.json` |
| report | `report.py` | all of the above | `song.json`, `TIMELINE.md`, `report.html`, `check.wav` |

## Output contract (`projects/<slug>/breakdown/`)

All times are seconds from the start of `audio/mix.wav`, rounded to 3 decimals (ms). JSON only, no NaN.
Every file has `"version": 1` at the top level.

### `meta.json`
```json
{"version":1, "source":"song/master.wav", "source_sha256":"...", "duration":306.4, "sample_rate":48000,
 "channels":2, "lufs_integrated":-9.8, "true_peak_db":-0.4, "key":{"local":"F minor","suno":"F minor"}}
```

### `beats.json`
```json
{"version":1, "method":"beat_this final0 + kick phase check",
 "meter":4, "bpm_median":132.8,
 "tempo_curve":[{"t":0.22,"bpm":131.3}, ...],          // one point per bar, smoothed
 "beats":[{"i":0,"t":0.220,"bar":0,"beat":0,"bpm":131.3}, ...],  // beat 0 = downbeat
 "bars":[{"i":0,"t0":0.220,"t1":2.060}, ...],
 "phase_check":{"kick_offset_ms":-4.0, "kicks_on_beat_pct":91.0, "shifted_ms":0.0}}
```
`bpm` per beat is the local fitted tempo, not 60/diff (raw beat times are quantised to 20 ms).
Subdivisions (8ths, 16ths) are linear interpolation between neighbouring beats; consumers compute them.

### `words.json`
```json
{"version":1, "method":"wav2vec2 large-lv60k-960h CTC Viterbi on vocals stem with garbage states, ...",
 "suno_agreement":{"median_ms":47, "p90_ms":133}, "suno_offset_ms":190.0,
 "stats":{"words":337, "aligned":335, "flagged":27, "low_conf":24, "disagree":3, "suno_guided":12,
          "lines_not_sung":[0], "whisper_matched":302, "whisper_bias_ms":95.0, "conf_median":0.34},
 "words":[{"i":0,"text":"Eighteen","norm":"eighteen","line":0,"section":2,"t0":12.345,"t1":12.801,
           "conf":0.82,"backing":false,"suno_t0":12.33,"flag":null,"src":"ctc"}, ...]}
```
`text` keeps the lyric's spelling and punctuation; `norm` is what was aligned. `section` is the index into
`sections.json` (not the lyric tag order: instrumental sections count too). `t0` is the word's first sound, `t1`
runs over a held vowel up to the next word. Words of a line that was not sung have `t0`/`t1` `null`, `conf` 0
and `flag` `"low_conf"` (and `section` `null` when their whole lyric section was not sung); consumers skip them.

`flag` is `null`, `"low_conf"` (CTC confidence under 0.15, or under 0.5 with whisper more than 0.5 s away), or
`"disagree"` (local and Suno differ by more than 250 ms). `backing` is true for words that were inside `( )` in
the lyrics. `src` is `"ctc"` (our alignment) or `"suno"`: our CTC was unsure (confidence under 0.5) and Suno's
own alignment put the word more than 250 ms away, so the word was aligned again inside Suno's place and takes
Suno's start. Suno-guided words are never flagged and do not count in `suno_agreement`.

`suno_t0` is Suno's start for the word on our clock: Suno's raw time minus `suno_offset_ms`, the median of
(Suno minus ours) over confident words. The offset is not only decoder padding: on a WAV download (no padding)
Suno's starts still run about 190 ms late, because Suno labels the stressed vowel rather than the first
consonant. `suno_agreement` (median and p90 of |ours - suno_t0| over words not guided by Suno), `suno_offset_ms`
and `suno_t0` are `null` without `song/suno/aligned_lyrics.json`. `stats` summarises coverage and flags;
`lines_not_sung` lists line indices.

### `lines.json`
```json
{"version":1, "lines":[{"i":0,"text":"Eighteen months to escape the permanent underclass","section":2,
  "t0":12.345,"t1":15.9,"words":[0,1,2,3,4,5,6],"bar0":5,"bar1":7,"conf":0.91,"sung":true}, ...],
 "gaps":[{"t0":15.9,"t1":17.2,"dur":1.3}, ...],      // vocal silences >= 0.25 s: cut and window candidates
 "extra":[{"t0":94.02,"t1":95.78,"text":"so over We're so back"}, ...]}
```
`section` indexes `sections.json`. `bar0` is the bar holding the line's first word, `bar1` the bar holding its
last sound (inclusive). `conf` is the mean word confidence. `sung` is false for a line with no evidence in the
audio (Suno skipped it, or a title line above the first tag); its `t0`, `t1`, `bar0` and `bar1` are `null`, and
`section` too when no line of its lyric section was sung. `gaps` are checked against the vocal stem's level, not only word times, and include the silence before
the first word and after the last. `extra` lists vocal phrases whisper heard where no lyric word was placed
(repeats Suno added, ad-libs, vocal chops); they are not in `words.json`.

### `sections.json`
```json
{"version":1, "sections":[{"i":0,"label":"Intro","kind":"intro","tag":"","t0":0.0,"t1":12.1,
  "bar0":0,"bar1":6,"lines":[],"energy":0.31,"vocal":false}, ...],
 "checks":{"suno_tags":{"matched":9,"of":9,"median_s":0.04,"max_s":0.28,"rows":[...]},
           "novelty":{"boundaries":10,"within_2s":8,"median_s":0.89,"labels":[{"section":"Intro","label":"A"}, ...]}}}
```
`kind` is one of `intro, verse, prechorus, chorus, postchorus, bridge, break, drop, build, outro, instrumental, other`.
`tag` is the full lyric tag (`"Verse 1 - spoken, deadpan"`), empty for sections the step added. `label` numbers
repeated tags (`"Chorus 2"`). Sections tile 0..duration with no gaps; inner boundaries are downbeats.
`bar0`/`bar1` are boundary indices into `beats.json` bars: the section covers bars `bar0` to `bar1 - 1`, and
`bar1` equals the next section's `bar0` (exclusive, unlike `lines.json`). A section starts on the downbeat at or
before its first word, or on the next downbeat when the first words are a pickup within half a bar of it.
Stretches of 2 bars or more without lyric vocals become their own section: named by a lyric tag without lines
(`[Drop - instrumental]`) when there is one, else `Intro`, `Outro`, `Instrumental`, or `Break` when quieter
than both neighbours. `lines` are `lines.json` indices; `vocal` is true when the section has sung lyric lines
(an instrumental section can still hold vocal chops, see `lines.json` `extra`). `energy` is mean mix loudness
of the section, 0..1 normalised over the song (5th to 99th percentile of 50 ms frames).

`checks` appears when `song/suno/` has the data; it confirms, it never moves a boundary. `suno_tags` pairs
each vocal section with Suno's tag of the same kind and compares Suno's first sung word after the tag (offset
corrected) with ours. `novelty` counts inner boundaries within 2 s of a Suno novelty peak and gives Suno's
similarity letter for each section (equal letters: similar material).

### `hits.json`
```json
{"version":1, "events":{
  "kick":[{"t":0.221,"s":0.93}, ...], "snare":[...], "hat":[...],
  "vocal_entry":[{"t":12.3,"s":1.0}, ...],    // first word after a gap >= 1 bar
  "impact":[...],                              // big broadband energy steps (drops, hits)
  "riser":[{"t0":40.1,"t1":43.8,"s":0.7}, ...], // rising-energy spans ending on an impact or downbeat
  "stop":[{"t0":55.06,"t1":57.59,"s":0.96,"end_impact":57.607}, ...], // the music drops out >= 1 beat, then returns
  "roll":[{"t0":32.92,"t1":35.66,"s":0.5,"voice":"kick","div":8}, ...] // 8ths-or-faster drum run into a downbeat
 },
 "counts":{"kick":512, "snare":260, ...}}
```
`s` is strength 0..1 within its event type.

**hits.json notes.** Extra key `method`: one line naming how each event type was found (and whether vocal entries
came from `words.json` or from the vocal stem alone). All times are the sound's attack, not a frame or window
centre. Kicks include the kicks of a high-passed break (no sub, only the 120 Hz-1 kHz body) with a lower `s`; a snare
on a kick is only reported when it is as loud as the snares around it. Vocal entries use the lyric words when the
align step has run, so vocal chops or pads in a drop do not hide a gap. The gap is 1 bar by default; a project can
shorten it with `project.json` `{"hits": {"vocal_gap_bars": 0.65}}` (useful at 170+ BPM, where a designed pause
before a drop is shorter than a bar). Risers are rising level or brightness only; a full stop before a drop is a
`stop` and a constant-rate drum roll is a `roll` (the drop itself is an impact).

`stop`: the instrumental (vocals may carry on) falls 24 dB or more under the music around it, in every band that
carries that music, with no drum hit inside, for at least one beat, and something comes back (the song's start, a
quiet intro and a fade-out ending are not stops). `t0` is the cut (before any reverb tail), `t1` the return,
`end_impact` the impact within 150 ms of `t1` or `null`. `s` = depth (40 dB = 1) x sqrt(beats / 8), capped at 1.
`roll`: a kick or snare run at 8ths or faster (one skipped note allowed), at least 4 hits over 2 beats to 4 bars,
whose last hit is within a beat of a downbeat and which does not carry on past it (else it is a groove). `t1` is
that downbeat, `voice` is `kick`, `snare` or `kick+snare`, `div` the subdivision in notes per bar (8, 12, 16, 24,
32). `s` = min(1, beats / 8) x min(1, hits per beat / 4). Both use absolute scales, so a small stop stays small.

### `envelopes.json`
```json
{"version":1, "fps":24, "frames":7354,
 "series":{"mix":[...], "vocals":[...], "drums":[...], "bass":[...], "other":[...],
           "kick":[...], "snare":[...], "hat":[...], "centroid":[...]}}
```
One value per video frame, 0..1, 3 decimals. Loudness series are RMS in dB mapped from -60..0 dB to 0..1 and
smoothed with a fast-attack, slow-release follower (attack 1 frame, release ~150 ms). `kick`/`snare`/`hat` are decaying
pulses from `hits.json`. `centroid` is normalised spectral brightness of the mix.

**envelopes.json notes.** Extra keys: `sampling` says which audio each frame describes (frame `i` covers
`[i/fps, (i+1)/fps)`; a hit pulse peaks in the frame that contains the hit), and `centroid_hz` is the `[low, high]`
centroid in Hz that maps to 0 and 1 for this song (2nd and 98th percentile of audible frames, 150 Hz-20 kHz). The
loudness series are absolute dBFS mapped -60..0 dB, so a mastered mix sits mid-range (a second test song: `mix` median
0.72, `vocals` 0.52) and stems sit lower than the mix; rescale per series (for example by its own 5th..95th
percentile) when a shape needs the full 0..1 range.

### `song.json`
Index for later phases: `meta` summary, file list, counts, and `qa` (list of warnings such as "12 words flagged",
"grid shifted 20 ms").

### `TIMELINE.md`
Human-readable: song facts, then each section with its bars and time range, then each line with time and bar.

### `report.html`
Self-contained page (no network): waveform, beat and bar grid, sections, words, hits; click to play from a point.
For checking by eye.

### `check.wav`
The mix with a click on each beat (accent on downbeats) and a soft tick on each word start. For checking by ear.
