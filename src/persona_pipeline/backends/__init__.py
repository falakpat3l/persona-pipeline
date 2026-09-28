"""Backend registry: maps a name in the config file to a backend class.

Real backends are imported lazily so their SDKs are only needed when used.
"""

from __future__ import annotations

from importlib import import_module

from persona_pipeline.backends.base import BackendError, ImageBackend, TextBackend, VisionBackend

# role -> backend name -> "module:Class"
_REGISTRY: dict[str, dict[str, str]] = {
    "text": {"mock": "persona_pipeline.backends.mock:MockText"},
    "image": {"mock": "persona_pipeline.backends.mock:MockImage"},
    "vision": {"mock": "persona_pipeline.backends.mock:MockVision"},
}


def available(role: str) -> list[str]:
    return sorted(_REGISTRY[role])


def create(role: str, name: str):
    """Instantiate the backend registered as `name` for `role`."""
    try:
        target = _REGISTRY[role][name]
    except KeyError as exc:
        raise BackendError(
            f"Unknown {role} backend '{name}'. Available: {', '.join(available(role))}"
        ) from exc
    module_name, class_name = target.split(":")
    return getattr(import_module(module_name), class_name)()


__all__ = [
    "BackendError",
    "ImageBackend",
    "TextBackend",
    "VisionBackend",
    "available",
    "create",
]
