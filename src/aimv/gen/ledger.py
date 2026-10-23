"""Spend ledger and hard caps, per project.

projects/<slug>/ledger.jsonl   append-only, one JSON line per event
project.json "budget": {"cap_usd": 100, "phases": {"look": 10, "bakeoff": 40}}

A request can have several lines: Higgsfield jobs write a "submitted" line as soon as the job is accepted and a
final line when it ends (ok, refunded, error, timeout). Totals count only the latest line per request (keyed by
request_id, else the record id, else the line itself), so in-flight jobs count against the cap and a crash leaves
a record that `aimv hf fetch` can complete. Old one-line records read the same way.

Counted: everything except "refunded" (failed, nsfw, canceled jobs) and "not_charged" (rejected before acceptance).
A request is refused when the estimate would push the project total, or the phase total, over its cap.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..project import Project

DEFAULT_CAP_USD = 25.0  # until the director sets a real budget in project.json
FREE = ("refunded", "not_charged")
PENDING = ("started", "submitted")

# Records written before 2026-10-06 lost "refunded"/"not_charged" to "error" (a bug). Their error text tells.
_LEGACY_FREE = [(re.compile(r"^RuntimeError: Higgsfield (Failed|NSFW|Cancelled) for "), "refunded"),
                (re.compile(r"^RuntimeError: OpenAI (400|401|403|429): "), "not_charged")]

_lock = threading.Lock()


class BudgetExceeded(SystemExit):
    pass


def _path(p: Project) -> Path:
    return p.root / "ledger.jsonl"


def entries(p: Project) -> list[dict]:
    """Every line, in order (a request may appear more than once)."""
    f = _path(p)
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by a crash; the next line for the same request supersedes it
    return out


def _key(e: dict, i: int) -> str:
    return e.get("request_id") or e.get("id") or f"#{i}"


def status_of(e: dict) -> str:
    st = e.get("status", "")
    if st == "error":
        for rx, fixed in _LEGACY_FREE:
            if rx.match(e.get("error", "")):
                return fixed
    return st


def records(p: Project) -> list[dict]:
    """The latest line per request, in order of first appearance."""
    latest: dict[str, dict] = {}
    for i, e in enumerate(entries(p)):
        latest[_key(e, i)] = e  # dicts keep first-insertion order; the value is replaced by later lines
    return list(latest.values())


def find(p: Project, request_id: str) -> dict | None:
    for e in reversed(entries(p)):
        if e.get("request_id") == request_id:
            return e
    return None


def pending(p: Project) -> list[dict]:
    """Requests whose latest line has no outcome yet (in flight, or a process died while waiting)."""
    return [e for e in records(p) if e.get("request_id") and status_of(e) in PENDING + ("timeout", "error")
            and not e.get("files")]


def totals(p: Project) -> dict:
    tot, by_phase = 0.0, {}
    for e in records(p):
        usd = 0.0 if status_of(e) in FREE else float(e.get("usd_actual", e.get("usd_estimate", 0.0)) or 0.0)
        tot += usd
        by_phase[e.get("phase", "")] = by_phase.get(e.get("phase", ""), 0.0) + usd
    return {"total_usd": round(tot, 4), "by_phase": {k: round(v, 4) for k, v in by_phase.items()}}


def caps(p: Project) -> tuple[float, dict]:
    b = p.config.get("budget", {})
    return float(b.get("cap_usd", DEFAULT_CAP_USD)), b.get("phases", {})


def check(p: Project, phase: str, usd: float) -> None:
    cap, phase_caps = caps(p)
    t = totals(p)
    if t["total_usd"] + usd > cap:
        raise BudgetExceeded(f"Budget: ${t['total_usd']:.2f} spent + ${usd:.2f} would pass the project cap ${cap:.2f} "
                             f"(project.json budget.cap_usd).")
    if phase in phase_caps and t["by_phase"].get(phase, 0.0) + usd > float(phase_caps[phase]):
        raise BudgetExceeded(f"Budget: phase '{phase}' at ${t['by_phase'].get(phase, 0):.2f} + ${usd:.2f} would pass "
                             f"its cap ${float(phase_caps[phase]):.2f}.")


def write(p: Project, rec: dict) -> None:
    """Append one line atomically (one write on an O_APPEND file; safe across threads and processes)."""
    data = (json.dumps(rec, ensure_ascii=False) + "\n").encode()
    with _lock:
        fd = os.open(_path(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


@contextmanager
def spend(p: Project, *, phase: str, provider: str, model: str, usd_estimate: float, request: dict):
    """Check the cap, then yield a mutable record; its final state is appended whatever happens.

    The caller may set rec["status"] ("submitted", "refunded", "not_charged", ...) and call write(p, rec) for an
    intermediate line. An exception only turns a record that is still "started" into "error"; a status the caller
    already set is kept.
    """
    check(p, phase, usd_estimate)
    rec = {"id": uuid.uuid4().hex, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase, "provider": provider,
           "model": model, "usd_estimate": round(usd_estimate, 4), "request": request, "status": "started",
           "files": []}
    try:
        yield rec
        if rec["status"] in PENDING:
            rec["status"] = "ok"
    except BaseException as e:
        if rec["status"] == "started":
            rec["status"] = "error"
        rec.setdefault("error", f"{type(e).__name__}: {e}"[:500])
        raise
    finally:
        rec["ts_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write(p, rec)
