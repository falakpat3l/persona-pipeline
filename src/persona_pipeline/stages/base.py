"""Base class and shared context for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from persona_pipeline.backends import ImageBackend, TextBackend, VisionBackend
from persona_pipeline.config import Config
from persona_pipeline.models import PostJob


@dataclass
class StageContext:
    """Everything a stage may need besides the job itself."""

    config: Config
    text: TextBackend
    image: ImageBackend
    vision: VisionBackend


class Stage(ABC):
    """One step of the pipeline.

    A stage reads what it needs from the job, calls a backend, writes its result
    back onto the job, and returns a short human-readable detail for the trace.
    """

    name: str = "stage"

    @abstractmethod
    def run(self, job: PostJob, ctx: StageContext) -> str: ...
