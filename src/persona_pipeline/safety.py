"""Small safety helpers used across the pipeline.

- redact():          hides API keys in any text before it is logged, printed or saved
- RedactFilter:      a logging filter that runs redact() on every log line
- untrusted_text():  cleans model output (drops control characters, caps the length)
- safe_child():      joins a folder and a name, refusing anything that escapes the folder
- check_local_url(): only lets http(s) URLs to allowed hosts through (for local apps)
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

# Environment variables that hold secrets. Their current values are hidden wherever
# they appear, whatever the surrounding text looks like.
SECRET_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
REDACTED = "[redacted]"

_SECRET_PATTERNS = [
    # The shape of a Google API key.
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    # key=... in a URL query string.
    re.compile(r"(?i)([?&](?:key|api_key|apikey|access_token|token)=)[^&\s\"'#]+"),
    # api_key: ..., x-goog-api-key: ..., GEMINI_API_KEY=... in headers, JSON or .env text.
    # The value must not start with "." so hints like "GEMINI_API_KEY=..." stay readable.
    re.compile(
        r"(?i)((?:x-goog-api-key|api[_-]?key|access[_-]?token)[\"']?\s*[:=]\s*[\"']?)"
        r"[^\s\"',}&.][^\s\"',}&]*"
    ),
]


def redact(text: object) -> str:
    """Return `text` as a string with API keys replaced by [redacted]."""
    out = str(text)
    for var in SECRET_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if len(value) >= 8:  # very short values would hide ordinary words
            out = out.replace(value, REDACTED)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + REDACTED,
            out,
        )
    return out


class RedactFilter(logging.Filter):
    """Redacts the message, traceback and stack of every record it sees.

    Attach it to a handler (not a logger) so it also covers log lines from
    third-party libraries such as the HTTP client.
    """

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string: let logging report it as usual
            return True
        record.msg, record.args = redact(message), None
        if record.exc_info and not record.exc_text:
            record.exc_text = self._formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


# C0 and C1 control characters, except tab and newline. Model output that reaches a
# terminal must not carry escape sequences that could rewrite what the user sees.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def untrusted_text(value: object, max_len: int) -> str:
    """Model output as plain text: control characters removed, at most `max_len` chars."""
    text = _CONTROL_CHARS.sub("", str(value))
    return text[:max_len]


class UnsafeValue(ValueError):
    """A path or URL that would reach outside what the user chose."""


def is_inside(path: Path, folder: Path) -> bool:
    """True if `path` resolves to a location inside `folder` (or is the folder)."""
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
    except ValueError:
        return False
    return True


def safe_child(folder: Path, name: str) -> Path:
    """`folder / name`, but only for a plain name that stays inside `folder`."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise UnsafeValue(f"not a safe file or folder name: {name!r}")
    path = Path(folder) / name
    if not is_inside(path, folder):
        raise UnsafeValue(f"not a safe file or folder name: {name!r}")
    return path


DEFAULT_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def check_local_url(url: str, allowed_hosts: Iterable[str] = DEFAULT_LOCAL_HOSTS) -> str:
    """Check a URL for a local app and return it without a trailing slash.

    Only http and https are allowed, the host must be in `allowed_hosts`, and the
    URL may not carry a user name, password, query or fragment.
    """
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeValue("the URL must start with http:// or https://")
    if "@" in parts.netloc:
        raise UnsafeValue("the URL must not contain a user name or password")
    if parts.query or parts.fragment:
        raise UnsafeValue("the URL must not contain ? or #")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeValue("the URL has no host")
    try:
        parts.port  # noqa: B018 (raises ValueError for a bad port)
    except ValueError as exc:
        raise UnsafeValue("the URL has an invalid port") from exc
    allowed = {h.strip().strip("[]").lower() for h in allowed_hosts}
    if host not in allowed:
        raise UnsafeValue(
            f"host '{host}' is not allowed. Allowed: {', '.join(sorted(allowed))}. "
            "To use another computer, add its address to drawthings.allowed_hosts "
            "in the persona file."
        )
    return url.rstrip("/")
