"""Backend interfaces.

Stages never talk to a vendor SDK directly. They talk to one of these three small
interfaces, so any model (Gemini, a local Stable Diffusion server, a mock) can be
swapped in through config without touching pipeline code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from persona_pipeline.models import GeneratedImage, ImagePrompt


@runtime_checkable
class TextBackend(Protocol):
    name: str

    def complete_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict:
        """Return a JSON object that matches `schema`.

        `task` is a short label (e.g. "write_prompt") used for logging and by the
        mock backend to return a sensible fake answer.
        """
        ...


@runtime_checkable
class ImageBackend(Protocol):
    name: str

    def generate(self, prompt: ImagePrompt, out_path: Path) -> GeneratedImage:
        """Render `prompt` to an image file at `out_path`."""
        ...


@runtime_checkable
class VisionBackend(Protocol):
    name: str

    def inspect_json(
        self, task: str, image_path: Path, instruction: str, schema: dict[str, Any]
    ) -> dict:
        """Look at an image and return a JSON object that matches `schema`."""
        ...


class BackendError(RuntimeError):
    """Raised when a backend call fails in a way the pipeline should report."""
