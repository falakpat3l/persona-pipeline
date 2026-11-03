"""Draw Things backend: free, local Stable Diffusion / FLUX on a Mac.

Setup:
    1. Open Draw Things and pick a model (for example FLUX.1 Schnell or SDXL).
    2. Settings > API Server > enable it (default http://127.0.0.1:7860).
    3. Set `image: drawthings` in the persona file.

Draw Things speaks the AUTOMATIC1111-style API: POST /sdapi/v1/txt2img with a
JSON body, and the reply carries the image as base64. Only the standard library
is used, so no extra install is needed.

Safety: the URL must be http(s) and point at an allowed host (this computer by
default), redirects and proxies are not used, and the reply is read in chunks
with a size cap and an overall deadline, so a broken or hostile server cannot
fill memory or hang the run.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from persona_pipeline.backends.base import BackendError
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import GeneratedImage, ImagePrompt
from persona_pipeline.safety import UnsafeValue, check_local_url

# Multiples of 64 (diffusion models need that) from the standard SDXL size buckets,
# as close as they get to each common feed and story aspect ratio.
SIZES = {
    "1:1": (1024, 1024),
    "4:5": (896, 1152),
    "9:16": (768, 1344),
}


def _snap(value: float) -> int:
    """Round to the nearest multiple of 64, never below 64."""
    return max(64, round(value / 64) * 64)


# A base64 PNG at the largest allowed size is well under 30 MB; anything far
# bigger is not a normal reply.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_CHUNK = 64 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn a redirect into an HTTPError instead of following it to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# No proxies (the app is local) and no redirects.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def open_url(request: urllib.request.Request | str, timeout: float):
    """Open a URL with the safe opener: no proxies, no redirects."""
    return _OPENER.open(request, timeout=timeout)


def read_limited(response, max_bytes: int, deadline: float) -> bytes:
    """Read a reply in chunks, stopping at `max_bytes` or when `deadline` passes."""
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and int(length) > max_bytes:
        raise BackendError(_too_large(max_bytes))
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("the reply took too long")
        chunk = response.read1(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise BackendError(_too_large(max_bytes))
        chunks.append(chunk)


def _too_large(max_bytes: int) -> str:
    return f"Draw Things sent more than {max_bytes // (1024 * 1024)} MB, so it was not read."


def _post_json(url: str, body: dict, timeout: float, max_bytes: int = MAX_RESPONSE_BYTES) -> dict:
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with open_url(request, timeout=timeout) as response:
        data = read_limited(response, max_bytes, deadline)
    return json.loads(data.decode("utf-8"))


class DrawThingsImage:
    name = "drawthings"

    def __init__(self, settings: PipelineSettings | None = None, post=_post_json):
        self.settings = (settings or PipelineSettings()).drawthings
        self._post = post  # injectable so tests never need the app running

    @property
    def base_url(self) -> str:
        """The app's URL, checked: http(s) only, and only to an allowed host."""
        url = os.environ.get("DRAWTHINGS_URL") or self.settings.url
        try:
            return check_local_url(url, self.settings.allowed_hosts)
        except UnsafeValue as exc:
            raise BackendError(f"Draw Things URL refused: {exc}") from exc

    def build_request(self, prompt: ImagePrompt) -> dict[str, Any]:
        base_w, base_h = SIZES.get(prompt.aspect_ratio, SIZES["4:5"])
        width, height = (_snap(v * self.settings.size_scale) for v in (base_w, base_h))
        body: dict[str, Any] = {
            "prompt": prompt.positive,
            "negative_prompt": prompt.negative,
            "width": width,
            "height": height,
            "steps": self.settings.steps,
            "cfg_scale": self.settings.cfg_scale,
            "seed": prompt.seed if prompt.seed is not None else -1,
        }
        if self.settings.model:
            body["model"] = self.settings.model
        return body

    def generate(self, prompt: ImagePrompt, out_path: Path) -> GeneratedImage:
        body = self.build_request(prompt)
        url = f"{self.base_url}/sdapi/v1/txt2img"
        try:
            reply = self._post(url, body, self.settings.timeout_s)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise BackendError(
                    f"Draw Things answered with a redirect (HTTP {exc.code}), which is not "
                    "followed. Check DRAWTHINGS_URL."
                ) from exc
            raise BackendError(
                f"Draw Things answered HTTP {exc.code}. Check that a model is loaded in the app."
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Could not reach Draw Things at {self.base_url} ({exc.reason}). "
                "Open Draw Things and turn on Settings > API Server."
            ) from exc
        except TimeoutError as exc:
            raise BackendError(
                f"Draw Things took longer than {self.settings.timeout_s:.0f}s. Try fewer steps, "
                "a faster model, or a smaller size_scale in the persona file."
            ) from exc
        except ValueError as exc:  # json.JSONDecodeError and UnicodeDecodeError
            raise BackendError(
                f"Something at {self.base_url} answered, but not like Draw Things does. "
                "Check DRAWTHINGS_URL and the port in the app's API Server setting."
            ) from exc

        images = reply.get("images") if isinstance(reply, dict) else None
        if not images:
            raise BackendError(
                "Draw Things finished but returned no image. Some models only return images "
                "while a project is open: open or create a project in Draw Things and retry."
            )
        try:
            data = base64.b64decode(str(images[0]).split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackendError("Draw Things returned an image that is not valid base64") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        try:
            with Image.open(out_path) as img:
                width, height = img.size  # the app may round or change the size
        except (OSError, Image.DecompressionBombError) as exc:
            raise BackendError("Draw Things returned data that is not an image") from exc
        return GeneratedImage(
            path=out_path,
            width=width,
            height=height,
            backend=self.name,
            seed=_seed_from_info(reply.get("info")),
        )


def _seed_from_info(info: Any) -> int | None:
    """A1111-style replies put the used seed in `info`, a JSON string. Best effort."""
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return None
    if isinstance(info, dict) and isinstance(info.get("seed"), int):
        return info["seed"]
    return None
