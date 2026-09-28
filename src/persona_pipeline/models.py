"""Data objects that flow between pipeline stages.

Every stage reads from and writes to a single `PostJob`, so the whole run can be
saved as one JSON manifest and inspected (or resumed) later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Brief(BaseModel):
    """What the post should be about. The only input a human has to write."""

    topic: str
    angle: str | None = Field(default=None, description="Optional hook or point of view")
    format: str = Field(default="portrait", description="portrait | square | story")


class ImagePrompt(BaseModel):
    """A text-to-image prompt written by the prompt-writer stage."""

    positive: str
    negative: str = ""
    aspect_ratio: str = "4:5"
    seed: int | None = None


class GeneratedImage(BaseModel):
    path: Path
    width: int
    height: int
    backend: str
    attempt: int = 1


class Critique(BaseModel):
    """A vision model's judgement of a generated image."""

    score: float = Field(ge=0, le=10)
    passed: bool
    notes: list[str] = Field(default_factory=list)


class Caption(BaseModel):
    text: str
    hashtags: list[str] = Field(default_factory=list)
    alt_text: str = ""

    def render(self) -> str:
        tags = " ".join(f"#{t.lstrip('#')}" for t in self.hashtags)
        return f"{self.text}\n\n{tags}".strip()


class StageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageEvent(BaseModel):
    """One line of the run trace: which stage ran, how long it took, what happened."""

    stage: str
    status: StageStatus
    started_at: datetime
    duration_ms: float
    detail: str = ""


class PostJob(BaseModel):
    """The shared state object that every stage reads and updates."""

    job_id: str
    persona: str
    brief: Brief
    prompt: ImagePrompt | None = None
    image: GeneratedImage | None = None
    critique: Critique | None = None
    caption: Caption | None = None
    output_dir: Path | None = None
    events: list[StageEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def succeeded(self) -> bool:
        return bool(self.events) and all(e.status != StageStatus.FAILED for e in self.events)
