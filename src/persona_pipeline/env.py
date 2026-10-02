"""Tiny .env loader so API keys never have to live in code or config files."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Read KEY=VALUE lines from `path` into os.environ.

    Blank lines and lines starting with # are ignored, surrounding quotes are
    stripped, an inline " # comment" after an unquoted value is dropped, and
    existing environment variables win unless `override` is True.
    Returns the pairs that were found.
    """
    file = Path(path)
    if not file.is_file():
        return {}

    found: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) > 1 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]  # quoted: keep everything inside, including #
        else:
            value = value.split(" #", 1)[0].strip()  # unquoted: drop an inline comment
        if not key:
            continue
        found[key] = value
        if value and (override or key not in os.environ):
            os.environ[key] = value
    return found
