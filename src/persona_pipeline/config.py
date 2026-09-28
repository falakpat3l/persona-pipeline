"""Persona and pipeline configuration, loaded from a single YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class VisualStyle(BaseModel):
    """The look that must stay consistent across every post."""

    subject: str = Field(description="How the persona looks, used in every image prompt")
    signature: list[str] = Field(
        default_factory=list, description="Visual elements that appear in every image"
    )
    lighting: str = "soft natural light"
    camera: str = "35mm, shallow depth of field"
    avoid: list[str] = Field(default_factory=list, description="Goes into the negative prompt")


class Voice(BaseModel):
    """How the persona writes captions."""

    tone: str = "warm, curious, concise"
    language: str = "English"
    max_caption_chars: int = 300
    emoji_level: str = "low"
    base_hashtags: list[str] = Field(default_factory=list)
    max_hashtags: int = 12


class Persona(BaseModel):
    name: str
    tagline: str = ""
    bio: str = ""
    style: VisualStyle
    voice: Voice = Field(default_factory=Voice)


class BackendChoice(BaseModel):
    """Which backend handles each AI role. 'mock' needs no API key."""

    text: str = "mock"
    image: str = "mock"
    vision: str = "mock"


class PipelineSettings(BaseModel):
    backends: BackendChoice = Field(default_factory=BackendChoice)
    output_dir: Path = Path("outputs")
    critic_threshold: float = Field(default=7.0, ge=0, le=10)
    max_attempts: int = Field(default=3, ge=1)


class Config(BaseModel):
    persona: Persona
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


def load_config(path: str | Path) -> Config:
    """Load and validate a persona YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
