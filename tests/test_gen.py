"""Generation plumbing: ledger, Higgsfield client, CLI file specs. All HTTP is faked; nothing is paid."""

import json
import uuid

import pytest
import requests

from aimv.gen import hf, keys, ledger
from aimv.project import Project

API = hf.API
I2V = "bytedance/seedance-2.5/image-to-video"
EXTEND = "bytedance/seedance-2.5/video-extend"
H3 = "minimax/h3/image-to-video"
SEEDANCE_DESC = {"type": "description", "pricing_description": "For 16:9 video ... $0.4622 at 720p ..."}


class Resp:
    def __init__(self, code=200, body=None, content=b""):
        self.status_code, self._body, self._content = code, body, content
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, n):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeHF:
    """Routes hf._http calls: /estimate, submit, status, and CDN downloads."""

    def __init__(self, estimate=None, submits=None, statuses=None):
        self.estimate = estimate if estimate is not None else SEEDANCE_DESC
        self.submits = list(submits or [Resp(200, {"status": "queued", "request_id": "rid-1234-abcd"})])
        self.statuses = list(statuses or [{"status": "queued"}, {"status": "in_progress"},
                                          {"status": "completed", "request_id": "rid-1234-abcd",
                                           "status_url": f"{API}/requests/rid-1234-abcd/status",
                                           "video": {"url": "https://cdn.example/out/take.mp4"}}])
        self.calls = []

    def __call__(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if url.startswith(f"{API}/estimate/"):
            est = self.estimate
            return est if isinstance(est, Resp) else Resp(200, est)
        if url.startswith(f"{API}/requests/"):
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return Resp(200, st)
        if url.startswith(API) and method == "POST":
            r = self.submits.pop(0) if len(self.submits) > 1 else self.submits[0]
            if isinstance(r, Exception):
                raise r
            return r
        if method == "GET":
            return Resp(200, content=b"MEDIA:" + url.encode())
        raise AssertionError(f"unexpected {method} {url}")

    def submit_calls(self):
        return [c for c in self.calls if c[0] == "POST" and "/estimate/" not in c[1]]


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"version": 1, "slug": "proj", "budget": {"cap_usd": 10}}))
    return Project.open(root)


@pytest.fixture
def fake(monkeypatch):
    f = FakeHF()
    monkeypatch.setattr(hf, "_http", f)
    monkeypatch.setattr(hf.keys, "hf_credentials", lambda: "kid:secret")
    monkeypatch.setattr(hf.time, "sleep", lambda s: None)
    return f


def lines(p):
    return ledger.entries(p)


# ----------------------------------------------------------------------------------------------- ledger

def test_spend_keeps_status_set_by_caller(proj):
    for st in ("refunded", "not_charged"):
        with pytest.raises(RuntimeError):
            with ledger.spend(proj, phase="x", provider="t", model="m", usd_estimate=1.0, request={}) as rec:
                rec["status"] = st
                raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        with ledger.spend(proj, phase="x", provider="t", model="m", usd_estimate=1.0, request={}):
            raise RuntimeError("boom")
    assert [e["status"] for e in lines(proj)] == ["refunded", "not_charged", "error"]
    assert ledger.totals(proj)["total_usd"] == 1.0


def test_totals_count_latest_line_per_request(proj):
    old = {"ts": "t", "phase": "look", "model": "m", "usd_estimate": 0.05, "status": "ok", "request_id": "old-1"}
    ledger.write(proj, old)  # old one-line format (no id)
    ledger.write(proj, {**old, "request_id": "old-2"})
    ledger.write(proj, {"id": "a", "phase": "clips", "usd_estimate": 2.0, "status": "submitted", "request_id": "r1"})
    ledger.write(proj, {"id": "a", "phase": "clips", "usd_estimate": 2.0, "status": "ok", "request_id": "r1"})
    ledger.write(proj, {"id": "b", "phase": "clips", "usd_estimate": 3.0, "status": "submitted", "request_id": "r2"})
    ledger.write(proj, {"id": "b", "phase": "clips", "usd_estimate": 3.0, "status": "refunded", "request_id": "r2"})
    ledger.write(proj, {"id": "c", "phase": "clips", "usd_estimate": 1.5, "status": "submitted", "request_id": "r3"})
    t = ledger.totals(proj)
    assert t["total_usd"] == pytest.approx(0.1 + 2.0 + 1.5)
    assert t["by_phase"] == {"look": 0.1, "clips": 3.5}
    assert [e["request_id"] for e in ledger.pending(proj)] == ["r3"]


def test_legacy_error_records_that_were_refunds_are_free(proj):
    ledger.write(proj, {"usd_estimate": 2.0, "status": "error", "request_id": "x",
                        "error": "RuntimeError: Higgsfield NSFW for x: {}"})
    ledger.write(proj, {"usd_estimate": 0.2, "status": "error", "error": "RuntimeError: OpenAI 400: bad"})
    ledger.write(proj, {"usd_estimate": 1.0, "status": "error", "request_id": "y", "error": "TimeoutError: slow"})
    assert ledger.totals(proj)["total_usd"] == 1.0


def test_ledger_skips_a_torn_line(proj):
    ledger.write(proj, {"usd_estimate": 1.0, "status": "ok", "request_id": "a"})
    with open(proj.root / "ledger.jsonl", "a") as f:
        f.write('{"usd_estimate": 9, "sta\n')
    assert ledger.totals(proj)["total_usd"] == 1.0


# ----------------------------------------------------------------------------------------------- pricing

def test_seedance_formula():
    assert hf.seedance_usd(I2V, {"duration": 5, "resolution": "720p"}) == pytest.approx(108000 / 1000 * 0.0214 * 1.05,
                                                                                         abs=1e-4)
    # video input: input + output seconds at 0.6x
    usd = hf.seedance_usd(EXTEND, {"duration": 5}, input_video_s=5)
    assert usd == pytest.approx(216000 / 1000 * 0.0214 * 0.6 * 1.05, abs=1e-4)
    with pytest.raises(ValueError, match="input video"):
        hf.seedance_usd(EXTEND, {"duration": 5})
    with pytest.raises(ValueError, match="input video"):
        hf.seedance_usd("bytedance/seedance-2.5/reference-to-video", {"video_urls": ["https://x/v.mp4"]})


def test_image_estimate_uses_rendered_size():
    usd = hf.image_usd({"resolution": "2k", "quality": "high"})
    assert 0.06 < usd < 0.07  # 2688x1520 render, between OpenAI's 2560x1440 and 3840x2160 prices, +10%
    assert hf.image_usd({"resolution": "2k", "quality": "high", "image_urls": ["a", "b"]}) == pytest.approx(
        usd + 0.022, abs=1e-3)


def test_price_prefers_numeric_estimate(fake):
    fake.estimate = {"type": "estimate", "credits": "5.200", "usd": "0.325",
                     "discount": {"percentage": "50.00", "credits": "5.200", "usd": "0.325"}}
    info = hf.price(H3, {"prompt": "x", "image_url": "https://e/a.jpg"})
    assert info["usd"] == 0.325 and "Higgsfield" in info["source"]
    assert "50% off" in hf.describe(info["higgsfield"])


def test_price_formula_only_for_seedance_25_exactly(fake):
    assert hf.price(I2V, {"image_url": "u"})["usd"] > 2
    assert hf.price("bytedance/seedance-2.0/image-to-video", {"image_url": "u"})["usd"] is None
    assert hf.estimate_usd("alibaba/wan-3.0/image-to-video", {"prompt": "x", "image_url": "u"}) is None


def test_price_raises_on_invalid_arguments(fake):
    fake.estimate = Resp(400, {"detail": "duration: 3 is less than the minimum of 4"})
    with pytest.raises(hf.Rejected, match="minimum of 4"):
        hf.price(I2V, {"image_url": "u", "duration": 3})


def test_describe_is_not_truncated():
    long = "x" * 2000
    assert hf.describe({"type": "description", "pricing_description": long}) == long


# ----------------------------------------------------------------------------------------------- defaults, results

def test_audio_off_by_default_unless_caller_sets_it():
    assert hf.apply_defaults(I2V, {})["generate_audio"] is False
    assert hf.apply_defaults(I2V, {"generate_audio": True})["generate_audio"] is True
    assert hf.apply_defaults("kling-video/v3.0/pro/image-to-video", {})["sound"] == "off"
    assert hf.apply_defaults("kling-video/o3/first-last-frame", {})["sound"] == "off"
    assert hf.apply_defaults("kling-video/v3/motion-control/pro", {})["keep_original_sound"] == "no"
    assert hf.apply_defaults("alibaba/wan-3.0-prime/image-to-video", {})["generate_audio"] is False
    assert hf.apply_defaults(H3, {"prompt": "x"}) == {"prompt": "x"}
    assert hf.apply_defaults("marketing-studio/image/sunburst", {"prompt": "x"}) == {"prompt": "x"}


def test_media_urls_documented_shapes():
    assert hf.media_urls({"status": "completed", "status_url": "https://api/x", "video": {"url": "https://c/v.mp4"},
                          "input": {"image_url": "https://c/in.png"}}) == ["https://c/v.mp4"]
    assert hf.media_urls({"images": [{"url": "https://c/a.png"}, {"url": "https://c/b.png"}]}) == [
        "https://c/a.png", "https://c/b.png"]
    assert hf.media_urls({"audio": {"url": "https://c/a.mp3"}, "audios": [{"url": "https://c/a.mp3"}]}) == [
        "https://c/a.mp3"]
    assert hf.media_urls({"status_url": "https://api/s", "zip": {"url": "https://c/x.zip"}}) == ["https://c/x.zip"]


# ----------------------------------------------------------------------------------------------- run

def test_run_happy_path(proj, fake):
    files = hf.run(proj, I2V, {"image_url": "https://e/plate.png", "prompt": "p", "duration": 5}, None,
                   phase="clips", tag="s01/t1", log=lambda m: None)
    assert len(files) == 1 and files[0].read_bytes() == b"MEDIA:https://cdn.example/out/take.mp4"
    assert files[0].parent == proj.root / "gen" and "rid-1234" in files[0].name  # unique default name
    (_, _, kw), = fake.submit_calls()
    key = kw["headers"]["Idempotency-Key"]
    assert str(uuid.UUID(key)) == key
    assert kw["json"]["generate_audio"] is False  # default added
    assert kw["headers"]["Authorization"] == "Key kid:secret"
    ls = lines(proj)
    assert [e["status"] for e in ls] == ["submitted", "ok"]
    assert {e["request_id"] for e in ls} == {"rid-1234-abcd"} and {e["idempotency_key"] for e in ls} == {key}
    assert ledger.totals(proj)["total_usd"] == pytest.approx(ls[-1]["usd_estimate"])
    assert ls[-1]["files"] == [str(files[0].relative_to(proj.root))]


def test_run_explicit_stem_never_overwrites(proj, fake):
    stem = proj.root / "clips" / "s01-t1"
    stem.parent.mkdir()
    (stem.parent / "s01-t1.mp4").write_bytes(b"earlier take")
    files = hf.run(proj, I2V, {"image_url": "u"}, stem, phase="clips", log=lambda m: None)
    assert (stem.parent / "s01-t1.mp4").read_bytes() == b"earlier take"
    assert files[0].name == "s01-t1-rid-1234.mp4"


def test_run_moderated_job_is_recorded_as_refunded(proj, fake):
    fake.statuses = [{"status": "nsfw", "request_id": "rid-1234-abcd"}]
    with pytest.raises(RuntimeError, match="nsfw"):
        hf.run(proj, I2V, {"image_url": "u"}, None, phase="clips", log=lambda m: None)
    assert [e["status"] for e in lines(proj)] == ["submitted", "refunded"]
    assert ledger.totals(proj)["total_usd"] == 0


def test_run_rejected_submit_is_not_charged(proj, fake):
    fake.submits = [Resp(400, {"detail": "Maximum number of concurrent requests (2) has been reached"})]
    with pytest.raises(hf.Rejected, match="concurrent"):
        hf.run(proj, I2V, {"image_url": "u"}, None, phase="clips", log=lambda m: None)
    assert [e["status"] for e in lines(proj)] == ["not_charged"]
    assert ledger.totals(proj)["total_usd"] == 0


def test_run_retries_ambiguous_submit_with_same_key(proj, fake):
    fake.submits = [requests.ConnectionError("reset"), Resp(502, {"detail": "bad gateway"}),
                    Resp(200, {"status": "queued", "request_id": "rid-1234-abcd"})]
    hf.run(proj, I2V, {"image_url": "u"}, None, phase="clips", log=lambda m: None)
    keys_sent = {kw["headers"]["Idempotency-Key"] for _, _, kw in fake.submit_calls()}
    assert len(fake.submit_calls()) == 3 and len(keys_sent) == 1


def test_run_refuses_unpriced_endpoint_without_usd(proj, fake):
    fake.estimate = {"type": "description", "pricing_description": "Priced per second"}
    with pytest.raises(SystemExit, match="--usd"):
        hf.run(proj, "alibaba/wan-3.0/image-to-video", {"prompt": "x", "image_url": "u"}, None, phase="clips")
    assert lines(proj) == []
    hf.run(proj, "alibaba/wan-3.0/image-to-video", {"prompt": "x", "image_url": "u"}, None, phase="clips",
           usd=1.0, log=lambda m: None)
    assert ledger.totals(proj)["total_usd"] == 1.0


def test_run_respects_cap(proj, fake):
    with pytest.raises(ledger.BudgetExceeded):
        hf.run(proj, I2V, {"image_url": "u", "duration": 30, "resolution": "1080p"}, None, phase="clips")
    assert fake.submit_calls() == []


def test_timeout_then_fetch_completes_the_record(proj, fake):
    fake.statuses = [{"status": "in_progress"}]
    with pytest.raises(TimeoutError):
        hf.run(proj, I2V, {"image_url": "u"}, proj.root / "clips" / "s02", phase="clips", timeout_s=-1,
               log=lambda m: None)
    assert [e["status"] for e in lines(proj)] == ["submitted", "timeout"]
    assert [e["request_id"] for e in ledger.pending(proj)] == ["rid-1234-abcd"]
    est = lines(proj)[-1]["usd_estimate"]
    fake.statuses = [{"status": "completed", "video": {"url": "https://cdn.example/late.mp4"}}]
    files = hf.fetch(proj, "rid-1234-abcd", log=lambda m: None)
    assert files == [proj.root / "clips" / "s02.mp4"]  # the stem recorded at submit
    assert lines(proj)[-1]["status"] == "ok" and ledger.pending(proj) == []
    assert ledger.totals(proj)["total_usd"] == pytest.approx(est)  # still counted once


# ----------------------------------------------------------------------------------------------- files, keys, upload

def test_apply_files_sets_fields_and_arrays(tmp_path):
    for n in ("plate.png", "ref.png", "v.wav"):
        (tmp_path / n).write_bytes(b"x")
    got, vid = hf.apply_files({"prompt": "p", "image_urls": ["https://e/canon.png"]},
                              [f"image_url={tmp_path / 'plate.png'}", f"image_urls[]={tmp_path / 'ref.png'}",
                               f"audio_urls[]={tmp_path / 'v.wav'}"], uploader=lambda p: f"https://up/{p.name}")
    assert got == {"prompt": "p", "image_url": "https://up/plate.png",
                   "image_urls": ["https://e/canon.png", "https://up/ref.png"], "audio_urls": ["https://up/v.wav"]}
    assert vid is None
    placeholders, _ = hf.apply_files({}, [f"image_url={tmp_path / 'plate.png'}"], uploader=None)
    assert placeholders == {"image_url": "https://example.com/plate.png"}


def test_apply_files_rejects_bad_specs(tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    with pytest.raises(SystemExit, match="FIELD=PATH"):
        hf.apply_files({}, ["image_url"], uploader=None)
    with pytest.raises(SystemExit, match="no such file"):
        hf.apply_files({}, [f"image_url={tmp_path / 'missing.png'}"], uploader=None)
    with pytest.raises(SystemExit, match="convert"):
        hf.apply_files({}, [f"audio_urls[]={tmp_path / 'song.mp3'}"], uploader=None)
    with pytest.raises(SystemExit, match="convert"):
        hf.upload(tmp_path / "song.mp3")


def test_hf_credentials(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr(keys, "ENV_FILE", env)
    for k in ("HF_KEY", "HF_API_KEY_ID", "HF_API_KEY_SECRET"):
        monkeypatch.delenv(k, raising=False)
    env.write_text("HF_API_KEY_ID=id1\nHF_API_KEY_SECRET=sec1\n")
    assert keys.hf_credentials() == "id1:sec1"
    env.write_text("HF_KEY=id2:sec2\nHF_API_KEY_ID=id1\nHF_API_KEY_SECRET=sec1\n")
    assert keys.hf_credentials() == "id2:sec2"
    assert keys.available()["HF_KEY"] is True
    env.write_text("")
    with pytest.raises(SystemExit, match="HF_KEY"):
        keys.hf_credentials()


def test_board_generates_through_higgsfield(proj, fake, monkeypatch):
    import io

    from PIL import Image

    from aimv.gen import board

    buf = io.BytesIO()
    Image.new("RGB", (32, 18), (200, 30, 30)).save(buf, "PNG")
    png = buf.getvalue()
    fake.estimate = {"type": "description", "pricing_description": "Per 1M tokens ..."}
    fake.statuses = [{"status": "completed", "images": [{"url": "https://cdn.example/frame.png"}]}]
    real = fake.__call__

    def route(method, url, **kw):
        if url == "https://cdn.example/frame.png":
            return Resp(200, content=png)
        return real(method, url, **kw)

    monkeypatch.setattr(hf, "_http", route)
    bdir = proj.root / "boards" / "t"
    bdir.mkdir(parents=True)
    (bdir / "board.json").write_text(json.dumps({
        "title": "t", "phase": "look", "defaults": {"provider": "higgsfield", "size": "2048x1152", "quality": "high"},
        "groups": [{"id": "A", "title": "a", "frames": [{"id": "A1", "caption": "c", "prompt": "a red frame"}]}]}))
    board.generate(proj, "t", workers=1, log=lambda m: None)
    assert (bdir / "A1.png").exists() and not list(bdir.glob("A1.raw*"))
    (_, _, kw), = fake.submit_calls()
    assert kw["json"] == {"prompt": "a red frame", "aspect_ratio": "16:9", "resolution": "2k", "quality": "high"}
    assert [e["status"] for e in ledger.entries(proj)] == ["submitted", "ok"]
    assert ledger.totals(proj)["total_usd"] == pytest.approx(hf.image_usd({"resolution": "2k", "quality": "high"}))


def test_run_relative_stem_is_not_joined_to_the_root_twice(proj, fake, monkeypatch):
    # `aimv hf run --project projects/x --out projects/x/clips/s01` from the repo root: the stem is relative to the
    # working directory; the returned path and the ledger must name the real file (no projects/x/projects/x/...)
    monkeypatch.chdir(proj.root.parent)
    files = hf.run(proj, I2V, {"image_url": "u"}, proj.root.name + "/clips/s01", phase="clips", log=lambda m: None)
    assert files[0] == proj.root / "clips" / "s01.mp4" and files[0].exists()
    assert lines(proj)[-1]["files"] == ["clips/s01.mp4"]
