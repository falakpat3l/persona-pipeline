"""Draw Things backend: free, local Stable Diffusion / FLUX on a Mac.

Setup:
    1. Open Draw Things and pick a model (for example FLUX.1 Schnell or SDXL).
    2. Settings > API Server > enable it (default http://127.0.0.1:7860).
    3. Set `image: drawthings` in the persona file.

Draw Things speaks the AUTOMATIC1111-style API: POST /sdapi/v1/txt2img with a
JSON body, and the reply carries the image as base64. Only the standard library
is used, so no extra install is needed.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from persona_pipeline.backends.base import BackendError
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import GeneratedImage, ImagePrompt

# Multiples of 64 (diffusion models need that) from the standard SDXL size buckets,
# as close as they get to each Instagram aspect ratio.
SIZES = {
    "1:1": (1024, 1024),
    "4:5": (896, 1152),
    "9:16": (768, 1344),
}


def _snap(value: float) -> int:
    """Round to the nearest multiple of 64, never below 64."""
    return max(64, round(value / 64) * 64)


def _post_json(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DrawThingsImage:
    name = "drawthings"

    def __init__(self, settings: PipelineSettings | None = None, post=_post_json):
        self.settings = (settings or PipelineSettings()).drawthings
        self._post = post  # injectable so tests never need the app running

    @property
    def base_url(self) -> str:
        return (os.environ.get("DRAWTHINGS_URL") or self.settings.url).rstrip("/")

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

        images = reply.get("images") or []
        if not images:
            raise BackendError(
                "Draw Things finished but returned no image. Some models only return images "
                "while a project is open: open or create a project in Draw Things and retry."
            )
        try:
            data = base64.b64decode(images[0].split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackendError("Draw Things returned an image that is not valid base64") from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        try:
            with Image.open(out_path) as img:
                width, height = img.size  # the app may round or change the size
        except OSError as exc:
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
