"""Higgsfield pay-per-use API, with our estimate, cap and ledger.

Credentials: HF_KEY (id:secret), or HF_API_KEY_ID + HF_API_KEY_SECRET, in the environment or a .env file
(see keys.hf_credentials).
Every model is an endpoint like "bytedance/seedance-2.5/image-to-video"; arguments are that model's JSON, with the
exact field names of its schema (docs.higgsfield.ai/docs/models/<family>/<workflow>.md).

How a paid run goes (run()):
  1. Audio off by default for video endpoints (the song is our soundtrack) unless the caller set the field.
  2. Price: POST /estimate (free; also validates the arguments). Its number when it gives one, else our formula
     (Seedance 2.5 tokens, GPT Image 2.5 Sunburst/Flare tokens), else the caller's usd, else refuse.
  3. Cap check, then submit ourselves with an Idempotency-Key (uuid4, stored in the ledger), retrying only
     ambiguous failures with the same key so a retry can never create a second paid job.
  4. A "submitted" ledger line as soon as the job is accepted, then poll, download, and a final line.
     Failed, nsfw and canceled jobs are refunded by Higgsfield and recorded as "refunded".
Outputs are kept by Higgsfield for at least 7 days: download immediately. fetch() recovers a job by request id.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..project import Project
from . import keys, ledger

API = "https://api.higgsfield.ai"
TERMINAL = ("completed", "failed", "nsfw", "canceled")
UPLOAD_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".wav", ".mp4"}  # what /files/generate-upload-url takes
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}

# ------------------------------------------------------------------------------------------------ pricing

# Seedance 2.5 (Higgsfield's pricing text, 2026-10-06):
#   billable video tokens = ceil(h * w * (input video s + output s) * 24 / 1024)
#   USD per 1k tokens: 480p/720p 0.0214, 1080p 0.0234; 0.6x that when any video input is present.
#   Image and audio references are free. Video and audio inputs are each normalised to at most 30 s.
SEEDANCE_25 = {f"bytedance/seedance-2.5/{w}" for w in
               ("text-to-video", "image-to-video", "reference-to-video", "video-extend", "video-edit")}
SEEDANCE_DIMS = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}  # 9:16 has the same pixel count
SEEDANCE_RATE = {"480p": 0.0214, "720p": 0.0214, "1080p": 0.0234}
SEEDANCE_VIDEO_INPUT_FACTOR = 0.6
SEEDANCE_MARGIN = 1.05  # list price; account discounts make the real charge lower

# GPT Image 2.5 Sunburst / Flare on Higgsfield: OpenAI's token rates. Output-only cost per image at high etc. for
# OpenAI's sizes (megapixels, USD), interpolated by the area Higgsfield actually renders: "2k" at 16:9 came back
# 2688x1520 (observed 2026-10-06); 1k and 4k are assumed to be a quarter of that and 3840x2160.
IMAGE_ENDPOINTS = {"marketing-studio/image/sunburst", "marketing-studio/image/flare"}
IMAGE_TABLE_MP = [1536 * 864 / 1e6, 2048 * 1152 / 1e6, 2560 * 1440 / 1e6, 3840 * 2160 / 1e6]
IMAGE_TABLE_USD = {"low": [0.004, 0.005, 0.006, 0.011], "medium": [0.008, 0.011, 0.014, 0.026],
                   "high": [0.032, 0.042, 0.055, 0.100], "xhigh": [0.058, 0.075, 0.098, 0.178],
                   "max": [0.129, 0.169, 0.221, 0.400]}
IMAGE_RENDER_MP = {"1k": 2688 * 1520 / 4e6, "2k": 2688 * 1520 / 1e6, "4k": 3840 * 2160 / 1e6}
IMAGE_REF_USD = 0.01  # input tokens per reference image (rough)
IMAGE_MARGIN = 1.10

# Audio off unless the caller says otherwise: the edit uses the song.
AUDIO_OFF = [("bytedance/seedance-2.5/", {"generate_audio": False}),
             ("higgsfield/cinema-studio/", {"generate_audio": False}),
             ("alibaba/wan-3.0/", {"generate_audio": False}),
             ("alibaba/wan-3.0-prime/", {"generate_audio": False}),
             ("kling-video/v3.0/", {"sound": "off"}),
             ("kling-video/o3/first-last-frame", {"sound": "off"}),
             ("kling-video/o3/image-reference", {"sound": "off"}),
             ("kling-video/v3/motion-control/", {"keep_original_sound": "no"})]


class Rejected(RuntimeError):
    """Higgsfield refused the request before accepting it (4xx): nothing was charged."""

    def __init__(self, code: int, detail: str):
        super().__init__(f"HTTP {code}: {detail}")
        self.code, self.detail = code, detail


def apply_defaults(endpoint: str, args: dict) -> dict:
    """A copy of args with audio turned off for video endpoints, unless the caller set the field."""
    out = dict(args)
    for prefix, fields in AUDIO_OFF:
        if endpoint.startswith(prefix):
            for k, v in fields.items():
                out.setdefault(k, v)
    return out


def seedance_usd(endpoint: str, args: dict, input_video_s: float = 0.0) -> float:
    res = args.get("resolution", "720p")
    if res not in SEEDANCE_DIMS:
        raise ValueError(f"Seedance resolution must be one of {list(SEEDANCE_DIMS)}, not {res!r}")
    w, h = SEEDANCE_DIMS[res]
    workflow = endpoint.rsplit("/", 1)[1]
    video_in = workflow in ("video-extend", "video-edit") or bool(args.get("video_urls"))
    if not video_in:
        in_s, out_s = 0.0, float(args.get("duration", 5))
    else:
        if not input_video_s:
            raise ValueError(f"{endpoint} bills the input video's seconds too: pass input_video_s "
                             f"(CLI: --input-video-seconds, or give the video with --file so it can be measured)")
        in_s = min(float(input_video_s), 30.0)
        out_s = max(in_s, 4.0) if workflow == "video-edit" else float(args.get("duration", 5))
    rate = SEEDANCE_RATE[res] * (SEEDANCE_VIDEO_INPUT_FACTOR if video_in else 1.0)
    tokens = math.ceil(w * h * (in_s + out_s) * 24 / 1024)
    return round(tokens / 1000 * rate * SEEDANCE_MARGIN, 4)


def image_usd(args: dict) -> float:
    quality = args.get("quality", "high")
    mp = IMAGE_RENDER_MP[args.get("resolution", "2k")]
    xs, ys = IMAGE_TABLE_MP, IMAGE_TABLE_USD[quality]
    if mp <= xs[0]:
        base = ys[0]
    else:
        i = next((j for j in range(1, len(xs)) if mp <= xs[j]), len(xs) - 1)
        base = ys[i - 1] + (mp - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1])
    return round((base + IMAGE_REF_USD * len(args.get("image_urls", []))) * IMAGE_MARGIN, 4)


def estimate_api(endpoint: str, args: dict) -> dict:
    """POST /estimate (free). Raises Rejected on a validation error; returns {"error": ...} if it is unavailable."""
    try:
        r = _http("POST", f"{API}/estimate/{endpoint}", json=args, headers=_headers(), timeout=60)
    except requests.RequestException as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if r.status_code == 200:
        return r.json()
    if r.status_code in (400, 404, 422):
        raise Rejected(r.status_code, _detail(r))
    return {"error": f"HTTP {r.status_code}: {_detail(r)}"}


def price(endpoint: str, args: dict, input_video_s: float = 0.0) -> dict:
    """{"usd": float | None, "source": str, "higgsfield": <the /estimate answer>}. Pass args after apply_defaults."""
    hf_ans = estimate_api(endpoint, args)
    if hf_ans.get("type") == "estimate" and hf_ans.get("usd") is not None:
        return {"usd": float(hf_ans["usd"]), "source": "Higgsfield /estimate (current discount applied)",
                "higgsfield": hf_ans}
    if endpoint in SEEDANCE_25:
        return {"usd": seedance_usd(endpoint, args, input_video_s),
                "source": "Seedance 2.5 token formula, list price +5%", "higgsfield": hf_ans}
    if endpoint in IMAGE_ENDPOINTS:
        return {"usd": image_usd(args), "source": "GPT Image 2.5 token table at the rendered size, +10%",
                "higgsfield": hf_ans}
    return {"usd": None, "source": "no cost model", "higgsfield": hf_ans}


def estimate_usd(endpoint: str, args: dict, input_video_s: float = 0.0) -> float | None:
    return price(endpoint, apply_defaults(endpoint, args), input_video_s)["usd"]


def describe(hf_ans: dict) -> str:
    """Higgsfield's /estimate answer as one readable line (full text, not truncated)."""
    if hf_ans.get("type") == "estimate":
        d = hf_ans.get("discount") or {}
        off = f", {float(d['percentage']):g}% off (saves ${d['usd']})" if d.get("percentage") else ""
        return f"${hf_ans['usd']} ({hf_ans.get('credits')} credits){off}"
    if hf_ans.get("pricing_description"):
        return hf_ans["pricing_description"]
    return json.dumps(hf_ans)


def describe_price(endpoint: str, args: dict) -> str:
    return describe(estimate_api(endpoint, args))


# ------------------------------------------------------------------------------------------------ HTTP

def _http(method: str, url: str, **kw) -> requests.Response:
    """The one place this module talks HTTP (tests replace it)."""
    return requests.request(method, url, **kw)


def _headers(idempotency_key: str | None = None) -> dict:
    h = {"Authorization": f"Key {keys.hf_credentials()}", "Content-Type": "application/json"}
    if idempotency_key:
        h["Idempotency-Key"] = idempotency_key
    return h


def _detail(r) -> str:
    try:
        d = r.json()
        return str(d.get("detail") or d.get("message") or d.get("error") or d) if isinstance(d, dict) else str(d)
    except ValueError:
        return (r.text or "")[:500]


def _sleep(attempt: int) -> None:
    time.sleep(min(2.0 * 2 ** attempt, 20.0) + random.uniform(0, 0.5))


def submit(endpoint: str, args: dict, idempotency_key: str, tries: int = 4) -> dict:
    """POST the job. Retries network errors, 408, 429 and 5xx with the SAME key (Higgsfield then returns the
    original request instead of a new one). Any other 4xx raises Rejected (not charged)."""
    last = ""
    for attempt in range(tries):
        try:
            r = _http("POST", f"{API}/{endpoint}", json=args, headers=_headers(idempotency_key), timeout=120)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code not in (408, 429) and r.status_code < 500:
                raise Rejected(r.status_code, _detail(r))
            last = f"HTTP {r.status_code}: {_detail(r)}"
        if attempt + 1 < tries:
            _sleep(attempt)
    raise RuntimeError(f"submit failed after {tries} tries ({last}); it may still have been accepted: "
                       f"check console.higgsfield.ai (Idempotency-Key {idempotency_key})")


def status(request_id: str, tries: int = 5) -> dict:
    last = ""
    for attempt in range(tries):
        try:
            r = _http("GET", f"{API}/requests/{request_id}/status", headers=_headers(), timeout=60)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code < 500:
                raise Rejected(r.status_code, _detail(r))
            last = f"HTTP {r.status_code}"
        if attempt + 1 < tries:
            _sleep(attempt)
    raise RuntimeError(f"status of {request_id} unavailable ({last})")


def wait(request_id: str, timeout_s: int = 1800, log=print) -> dict:
    """Poll until a terminal status: 2 s backing off to 10 s, with jitter."""
    t0, delay, last = time.time(), 2.0, None
    while True:
        s = status(request_id)
        st = s.get("status")
        if st != last:
            log(f"  {st}")
            last = st
        if st in TERMINAL:
            return s
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{request_id} still {st} after {timeout_s}s; recover it later with "
                               f"`aimv hf fetch {request_id} --project <p>`")
        time.sleep(delay + random.uniform(0, 0.5))
        delay = min(delay * 1.5, 10.0)


# ------------------------------------------------------------------------------------------------ media

def upload(path: Path) -> str:
    """Upload a local file (frame, reference image, wav, mp4) and return its public URL. Free."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")
    if path.suffix.lower() not in UPLOAD_TYPES:
        raise SystemExit(f"Higgsfield uploads only {sorted(UPLOAD_TYPES)}; convert {path.name} first "
                         f"(ffmpeg: audio to .wav, video to .mp4).")
    import higgsfield_client

    return higgsfield_client.SyncClient(api_key=keys.hf_credentials()).upload_file(path)


def video_seconds(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def apply_files(args: dict, specs: list[str], uploader=upload) -> tuple[dict, float | None]:
    """Apply CLI --file FIELD=PATH specs: upload PATH and set args[FIELD]; FIELD[] appends to an array.
    uploader=None puts placeholder URLs instead (for pricing; /estimate does not fetch media).
    Returns (args, seconds of the local videos given for video fields, or None if none or unmeasurable)."""
    out = json.loads(json.dumps(args))
    vid_s, vid_unknown = 0.0, False
    for spec in specs:
        field, sep, path = spec.partition("=")
        if not sep or not field or not path:
            raise SystemExit(f"--file wants FIELD=PATH (e.g. image_url=plate.png or image_urls[]=ref.png), "
                             f"got {spec!r}")
        path = Path(path).expanduser()
        if not path.is_file():
            raise SystemExit(f"--file {field}: no such file {path}")
        if path.suffix.lower() not in UPLOAD_TYPES:
            raise SystemExit(f"--file {field}: Higgsfield uploads only {sorted(UPLOAD_TYPES)}; convert {path.name}.")
        url = uploader(path) if uploader else f"https://example.com/{path.name}"
        name = field[:-2] if field.endswith("[]") else field
        if field.endswith("[]"):
            cur = out.setdefault(name, [])
            if not isinstance(cur, list):
                raise SystemExit(f"--file {field}: args already has a non-array {name}")
            cur.append(url)
        else:
            out[name] = url
        if name in ("video_url", "video_urls") and path.suffix.lower() in VIDEO_EXTS:
            s = video_seconds(path)
            if s is None:
                vid_unknown = True
            else:
                vid_s += s
    return out, (None if vid_unknown or not vid_s else round(vid_s, 3))


def media_urls(result: dict) -> list[str]:
    """Output URLs of a completed result: video.url, images[].url, audio.url, audios[].url (documented shapes);
    a generic walk only if none of those is present."""
    out: list[str] = []
    for k in ("video", "audio"):
        v = result.get(k)
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            out.append(v["url"])
    for k in ("images", "audios", "videos"):
        for v in result.get(k) or []:
            if isinstance(v, dict) and isinstance(v.get("url"), str):
                out.append(v["url"])
    if not out:
        def walk(v):
            if isinstance(v, dict):
                for kk, x in v.items():
                    if kk == "url" and isinstance(x, str) and x.startswith("http"):
                        out.append(x)
                    elif kk not in ("status_url", "cancel_url", "response_url"):
                        walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
        walk(result)
    return list(dict.fromkeys(out))


def _slug(endpoint: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", endpoint)


def default_stem(p: Project, endpoint: str, request_id: str) -> Path:
    return p.root / "gen" / f"{_slug(endpoint)}-{time.strftime('%Y%m%d-%H%M%S')}-{request_id[:8]}"


def download(urls: list[str], out_stem: Path, request_id: str) -> list[Path]:
    """Download every URL next to out_stem (stem + extension; -0, -1 ... when several). Never overwrites: an existing
    file gets the request id appended instead."""
    files = []
    for i, u in enumerate(urls):
        ext = Path(urlparse(u).path).suffix or ".bin"
        idx = f"-{i}" if len(urls) > 1 else ""
        dst = out_stem.with_name(f"{out_stem.name}{idx}{ext}")
        if dst.exists():
            dst = out_stem.with_name(f"{out_stem.name}-{request_id[:8]}{idx}{ext}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        for attempt in range(3):
            try:
                with _http("GET", u, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                _sleep(attempt)
        tmp.replace(dst)
        files.append(dst)
    return files


def _rel(p: Project, f: Path) -> str:
    return str(f.relative_to(p.root)) if f.is_relative_to(p.root) else str(f)


# ------------------------------------------------------------------------------------------------ run / fetch

def run(p: Project, endpoint: str, args: dict, out_stem: Path | None = None, *, phase: str,
        usd: float | None = None, tag: str = "", timeout_s: int = 1800, log=print,
        input_video_s: float = 0.0) -> list[Path]:
    """Price, check the cap, submit, record, wait, download. Returns the downloaded files.

    out_stem: path without extension; None gives gen/<endpoint>-<time>-<request id>. usd overrides the estimate
    (needed for endpoints without one). input_video_s: seconds of video input (Seedance extend, edit, video refs).
    """
    args = apply_defaults(endpoint, args)
    if out_stem is not None:
        # a relative stem is relative to the working directory (where the files land), not to the project root:
        # make it absolute so the ledger and the returned paths are not joined onto the root a second time
        out_stem = Path(out_stem).expanduser().resolve()
    # Both paths call /estimate, which raises Rejected on invalid arguments before anything is spent or recorded.
    if usd is not None:
        priced = {"usd": usd, "source": "explicit", "higgsfield": estimate_api(endpoint, args)}
    else:
        priced = price(endpoint, args, input_video_s)
    est = priced["usd"]
    if est is None:
        raise SystemExit(f"No price for {endpoint}: pass an explicit estimate (CLI --usd). "
                         f"Higgsfield says: {describe(priced['higgsfield'])}")
    req = {"endpoint": endpoint, "args": args, "tag": tag}
    with ledger.spend(p, phase=phase, provider="higgsfield", model=endpoint, usd_estimate=est, request=req) as rec:
        rec["idempotency_key"] = key = str(uuid.uuid4())
        rec["price_source"] = priced["source"]
        if out_stem is not None:
            rec["out_stem"] = _rel(p, Path(out_stem))
        t0 = time.time()
        try:
            sub = submit(endpoint, args, key)
        except Rejected:
            rec["status"] = "not_charged"
            raise
        rid = sub["request_id"]
        rec.update(request_id=rid, status="submitted")
        ledger.write(p, rec)  # visible to the cap and to `aimv hf fetch` while the job runs
        log(f"submitted {endpoint} ({tag or 'untagged'}), request {rid}, est ${est:.3f}")
        try:
            result = wait(rid, timeout_s, log)
            _finish(p, rec, result, Path(out_stem) if out_stem else default_stem(p, endpoint, rid), t0)
        except TimeoutError:
            rec["status"] = "timeout"
            raise
        except BaseException as e:
            if rec["status"] == "submitted":
                rec["status"] = "error"
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            raise
        log(f"  done in {rec['seconds']}s: {', '.join(Path(f).name for f in rec['files'])}")
        return [p.root / f if not Path(f).is_absolute() else Path(f) for f in rec["files"]]


def _finish(p: Project, rec: dict, result: dict, out_stem: Path, t0: float) -> None:
    """Record the terminal result; raise for refunded outcomes; download completed media."""
    rid = rec["request_id"]
    rec["result"] = result
    rec["seconds"] = round(time.time() - t0, 1)
    st = result.get("status")
    if st != "completed":
        rec["status"] = "refunded"  # failed, nsfw and canceled jobs are not charged on the API
        raise RuntimeError(f"Higgsfield {st} for {rid}: {result.get('error') or json.dumps(result)[:300]}")
    urls = media_urls(result)
    if not urls:
        rec["status"] = "error"
        raise RuntimeError(f"Completed but no media URL in result: {json.dumps(result)[:300]}")
    rec["urls"] = urls
    files = download(urls, out_stem, rid)
    rec["files"] = [_rel(p, f) for f in files]
    rec["status"] = "ok"


def fetch(p: Project, request_id: str, out_stem: Path | None = None, *, timeout_s: int = 1800,
          log=print) -> list[Path]:
    """Recover a job by request id (after a crash or timeout): wait for it, download, and close its ledger record.
    Free (status reads only)."""
    prev = ledger.find(p, request_id) or {}
    endpoint = prev.get("model") or (prev.get("request") or {}).get("endpoint") or "request"
    if out_stem is None:
        out_stem = p.root / prev["out_stem"] if prev.get("out_stem") else default_stem(p, endpoint, request_id)
    else:
        out_stem = Path(out_stem).expanduser().resolve()  # relative to the working directory, as in run()
    rec = {k: v for k, v in prev.items() if k not in ("error", "result", "urls", "files", "seconds")}
    rec.setdefault("id", uuid.uuid4().hex)
    rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec.setdefault("phase", "recovered")
    rec.setdefault("provider", "higgsfield")
    rec.setdefault("model", endpoint)
    rec.setdefault("usd_estimate", 0.0)
    rec.update(request_id=request_id, recovered=True, files=[])
    if not prev:
        log("  no ledger record for this request: recording it with usd_estimate 0 (check console.higgsfield.ai)")
    t0 = time.time()
    try:
        result = wait(request_id, timeout_s, log)
        _finish(p, rec, result, Path(out_stem), t0)
    except TimeoutError:
        rec["status"] = "timeout"
        raise
    except BaseException as e:
        if rec.get("status") not in ("refunded", "error"):
            rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        raise
    finally:
        rec["ts_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        ledger.write(p, rec)
    return [p.root / f if not Path(f).is_absolute() else Path(f) for f in rec["files"]]
