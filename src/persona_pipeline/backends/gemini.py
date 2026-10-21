"""Google Gemini backends.

Setup:
    pip install -e ".[gemini]"
    echo "GEMINI_API_KEY=your-key" > .env      # free key: https://aistudio.google.com/apikey

How this file works, top to bottom:
  1. api_key_from_env()  finds your key in .env.
  2. parse_json_reply()  turns Gemini's reply into a dict and checks it has every field we need.
  3. is_retryable()      decides which errors are worth trying again ("busy", "timed out").
  4. _GeminiBase         shared by the three classes below. Its _generate_json() does
                         the actual call: try the main model a few times, and if it
                         stays busy, move on to the next model in fallback_models.
  5. GeminiText          writes prompts and captions.
     GeminiVision        looks at an image and scores it.
     GeminiImage         draws an image (paid model, no free tier).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from persona_pipeline.backends.base import BackendError
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import GeneratedImage, ImagePrompt
from persona_pipeline.retry import with_retries

log = logging.getLogger("persona_pipeline")

API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
# HTTP status codes that mean "try again later": timeout, too many requests,
# and server-side errors. Anything else (like 400 bad request) is our mistake,
# so retrying would only fail again.
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


# Transient network failures from the HTTP layer (httpx), matched by name so this
# module does not need to import httpx.
TRANSIENT_NETWORK_ERRORS = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ReadError",
    "RemoteProtocolError",
}


def is_retryable(exc: Exception) -> bool:
    """Retry rate limits, server errors, dropped connections and malformed replies."""
    if isinstance(exc, MalformedOutput):
        return True
    if type(exc).__name__ in TRANSIENT_NETWORK_ERRORS:
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in RETRYABLE_STATUS


class _GeminiBase:
    """Shared setup for the three Gemini classes: settings, the client, and the call."""

    name = "gemini"

    def __init__(self, settings: PipelineSettings | None = None, client: Any = None):
        self.settings = (settings or PipelineSettings()).gemini
        self._client = client  # injected in tests; created lazily otherwise
        self.last_model: str | None = None  # which model actually answered
        self._sticky_model: str | None = None  # fallback that worked earlier in this run

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
        self,
        task: str,
        model: str,
        contents: Any,
        system: str,
        schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict:
        """Ask Gemini for JSON. Try `model` first; if it stays busy, try the fallbacks.

        Example with fallback_models [B, C]: try A up to max_retries times. Still
        busy? Try B the same way, then C. Once B has worked, later calls in the
        same run start at B instead of waiting on A again ("sticky" fallback).
        """
        config = {
            "system_instruction": system,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            # We never pass tools, so switch off automatic function calling (and its log noise).
            "automatic_function_calling": {"disable": True},
        }
        chain = [model] + [m for m in self.settings.fallback_models if m != model]
        # Sticky fallback: once a model was overloaded in this run, start from the
        # model that worked instead of waiting on the busy one again.
        if self._sticky_model in chain:
            chain = chain[chain.index(self._sticky_model) :]

        for i, current in enumerate(chain):
            # One attempt against one model. with_retries() repeats it on busy errors.
            def call(current: str = current) -> dict:
                response = self.client.models.generate_content(
                    model=current, contents=contents, config=config
                )
                return parse_json_reply(getattr(response, "text", None), schema)

            try:
                result = with_retries(
                    call,
                    attempts=self.settings.max_retries,
                    is_retryable=is_retryable,
                    label=f"gemini:{task}:{current}",
                )
            except Exception as exc:
                is_last = i == len(chain) - 1
                if is_last or not is_retryable(exc):
                    raise
                log.warning("  %s unavailable (%s), falling back to %s", current, exc, chain[i + 1])
                continue
            self.last_model = current
            if current != model:
                self._sticky_model = current
            return result
        raise AssertionError("unreachable")  # pragma: no cover


class GeminiText(_GeminiBase):
    """Text role: writes image prompts and captions."""

    def complete_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict:
        return self._generate_json(task, self.settings.text_model, prompt, system, schema)


MAX_IMAGE_BYTES = 15 * 1024 * 1024  # inline images must stay well under the request limit


def image_part(path: Path) -> dict:
    """An inline image part for a Gemini request, read from disk."""
    data = Path(path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise BackendError(f"{path} is {len(data) // 1_000_000} MB, too large to send inline")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return {"inline_data": {"mime_type": mime, "data": data}}


class GeminiVision(_GeminiBase):
    """Vision role: looks at a generated image and scores it against a rubric."""

    def inspect_json(
        self, task: str, image_path: Path, instruction: str, schema: dict[str, Any]
    ) -> dict:
        system = (
            "You are a strict but fair art director reviewing images for a social media "
            "account. Judge only what you can see. Reply only with JSON."
        )
        contents = [image_part(image_path), instruction]
        # Low temperature: a judge should give the same score to the same image.
        return self._generate_json(
            task, self.settings.vision_model, contents, system, schema, temperature=0.2
        )


def _image_config(aspect_ratio: str):
    """A typed ImageConfig when the SDK is installed (older SDKs reject a plain dict)."""
    try:
        from google.genai import types
    except ImportError:  # tests with a fake client, no SDK needed
        return {"aspect_ratio": aspect_ratio}
    return types.ImageConfig(aspect_ratio=aspect_ratio)


class GeminiImage(_GeminiBase):
    """Image role via Gemini's image model. Note: this model has no free tier."""

    ASPECTS = {"1:1", "4:5", "9:16"}

    def generate(self, prompt: ImagePrompt, out_path: Path) -> GeneratedImage:
        text = prompt.positive
        if prompt.negative:
            text += f"\nDo not include: {prompt.negative}"
        config = {
            "response_modalities": ["IMAGE"],
            "image_config": _image_config(
                prompt.aspect_ratio if prompt.aspect_ratio in self.ASPECTS else "4:5"
            ),
        }
        model = self.settings.image_model

        # One attempt: ask for the image, then dig the picture's bytes out of the reply.
        def call() -> bytes:
            response = self.client.models.generate_content(
                model=model, contents=text, config=config
            )
            for candidate in getattr(response, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        return inline.data
            raise MalformedOutput("the image model returned no image (it may have been filtered)")

        data = with_retries(
            call,
            attempts=self.settings.max_retries,
            is_retryable=is_retryable,
            label=f"gemini:image:{model}",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        try:
            with Image.open(out_path) as img:
                width, height = img.size
        except OSError as exc:
            raise BackendError("Gemini returned data that is not an image") from exc
        return GeneratedImage(path=out_path, width=width, height=height, backend=self.name)
