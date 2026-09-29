"""Backend registry: maps a name in the config file to a backend class.

Real backends are imported lazily so their SDKs are only needed when used.
Every backend class takes the pipeline settings as its only constructor argument.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from persona_pipeline.backends.base import BackendError, ImageBackend, TextBackend, VisionBackend

if TYPE_CHECKING:
    from persona_pipeline.config import PipelineSettings

# role -> backend name -> "module:Class"
_REGISTRY: dict[str, dict[str, str]] = {
    "text": {
        "mock": "persona_pipeline.backends.mock:MockText",
        "gemini": "persona_pipeline.backends.gemini:GeminiText",
    },
    "image": {"mock": "persona_pipeline.backends.mock:MockImage"},
    "vision": {"mock": "persona_pipeline.backends.mock:MockVision"},
}


def available(role: str) -> list[str]:
    return sorted(_REGISTRY[role])


def create(role: str, name: str, settings: PipelineSettings | None = None):
    """Instantiate the backend registered as `name` for `role`."""
    from persona_pipeline.config import PipelineSettings

    try:
        target = _REGISTRY[role][name]
    except KeyError as exc:
        raise BackendError(
            f"Unknown {role} backend '{name}'. Available: {', '.join(available(role))}"
        ) from exc
    module_name, class_name = target.split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(settings or PipelineSettings())


__all__ = [
    "BackendError",
    "ImageBackend",
    "TextBackend",
    "VisionBackend",
    "available",
    "create",
]
