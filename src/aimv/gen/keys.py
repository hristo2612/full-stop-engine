"""API keys from the environment, or from a .env file (gitignored). Values are never printed or logged.

The .env file is `$AIMV_ENV_FILE` when set, else `.env` in the current working directory."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(os.environ.get("AIMV_ENV_FILE") or Path.cwd() / ".env")


def _load() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                out[k.strip()] = v
    return out


def get(name: str) -> str | None:
    """Environment first (for CI or one-off overrides), then .env."""
    return os.environ.get(name) or _load().get(name)


def require(name: str, purpose: str) -> str:
    v = get(name)
    if not v:
        raise SystemExit(f"{name} is not set. Add it to {ENV_FILE} ({purpose}).")
    return v


def hf_credentials() -> str:
    """Higgsfield API credentials as "key_id:secret": HF_KEY, else HF_API_KEY_ID + HF_API_KEY_SECRET."""
    key = get("HF_KEY")
    if key and ":" in key:
        return key
    kid, sec = get("HF_API_KEY_ID"), get("HF_API_KEY_SECRET")
    if kid and sec:
        return f"{kid}:{sec}"
    raise SystemExit(f"Higgsfield API key missing: set HF_KEY=id:secret (or HF_API_KEY_ID and HF_API_KEY_SECRET) "
                     f"in {ENV_FILE}.")


def available() -> dict[str, bool]:
    names = ["OPENAI_API_KEY", "HF_KEY", "HF_API_KEY_ID", "HF_API_KEY_SECRET", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]
    return {n: bool(get(n)) for n in names}
