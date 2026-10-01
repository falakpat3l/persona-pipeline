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


class GeminiSettings(BaseModel):
    """Options for the Gemini backends. The API key comes from GEMINI_API_KEY."""

    text_model: str = "gemini-3.8-flash"
    vision_model: str = "gemini-3.8-flash"  # must accept image input
    image_model: str = "gemini-3.1-flash-image"  # paid tier only, no free quota
    # Tried in order when the main model stays overloaded (503) or rate limited (429).
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
    )
    temperature: float = Field(default=0.9, ge=0, le=2)
    max_retries: int = Field(default=3, ge=1, description="attempts per model, including the first")
    timeout_s: float = Field(default=60, gt=0)


class DrawThingsSettings(BaseModel):
    """Local image generation with the Draw Things app (its API Server must be on)."""

    url: str = "http://127.0.0.1:7860"  # DRAWTHINGS_URL in .env overrides this
    model: str | None = Field(default=None, description="model file name; None = app's current")
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=1, le=30)
    size_scale: float = Field(default=1.0, gt=0, le=2, description="0.75 = faster, smaller")
    timeout_s: float = Field(default=300, gt=0)


class PipelineSettings(BaseModel):
    backends: BackendChoice = Field(default_factory=BackendChoice)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    drawthings: DrawThingsSettings = Field(default_factory=DrawThingsSettings)
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
