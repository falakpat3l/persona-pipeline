"""Google Gemini backends.

Setup:
    pip install -e ".[gemini]"
    echo "GEMINI_API_KEY=your-key" > .env      # free key: https://aistudio.google.com/apikey

Every call asks Gemini for JSON that matches the stage's schema (structured
output), checks the required keys, and retries rate limits, server errors and
malformed replies with exponential backoff.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from persona_pipeline.backends.base import BackendError
from persona_pipeline.config import PipelineSettings
from persona_pipeline.retry import with_retries

API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class MalformedOutput(BackendError):
    """The model replied, but not with the JSON we asked for. Worth retrying."""


def api_key_from_env() -> str:
    for var in API_KEY_VARS:
        if os.environ.get(var):
            return os.environ[var]
    raise BackendError(
        "No Gemini API key found. Add GEMINI_API_KEY=... to a .env file in the project "
        "folder (free key: https://aistudio.google.com/apikey)."
    )


def parse_json_reply(text: str | None, schema: dict[str, Any]) -> dict:
    """Parse a model reply into a dict and check the schema's required keys."""
    if not text:
        raise MalformedOutput("empty reply")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise MalformedOutput(f"reply is not JSON: {cleaned[:80]!r}") from exc
    if not isinstance(data, dict):
        raise MalformedOutput(f"expected a JSON object, got {type(data).__name__}")
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise MalformedOutput(f"reply is missing keys: {', '.join(missing)}")
    return data


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, MalformedOutput):
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in RETRYABLE_STATUS


class _GeminiBase:
    name = "gemini"

    def __init__(self, settings: PipelineSettings | None = None, client: Any = None):
        self.settings = (settings or PipelineSettings()).gemini
        self._client = client  # injected in tests; created lazily otherwise

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise BackendError(
                    'Gemini support is not installed. Run: pip install -e ".[gemini]"'
                ) from exc
            self._client = genai.Client(
                api_key=api_key_from_env(),
                http_options=types.HttpOptions(timeout=int(self.settings.timeout_s * 1000)),
            )
        return self._client

    def _generate_json(
        self, task: str, model: str, contents: Any, system: str, schema: dict[str, Any]
    ) -> dict:
        config = {
            "system_instruction": system,
            "temperature": self.settings.temperature,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }

        def call() -> dict:
            response = self.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            return parse_json_reply(getattr(response, "text", None), schema)

        return with_retries(
            call,
            attempts=self.settings.max_retries,
            is_retryable=is_retryable,
            label=f"gemini:{task}",
        )


class GeminiText(_GeminiBase):
    """Text role: writes image prompts and captions."""

    def complete_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict:
        return self._generate_json(task, self.settings.text_model, prompt, system, schema)
