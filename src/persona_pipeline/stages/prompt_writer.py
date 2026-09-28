"""Stage 1: turn a short brief into a detailed, on-style image prompt."""

from __future__ import annotations

from persona_pipeline.models import ImagePrompt, PostJob
from persona_pipeline.stages.base import Stage, StageContext

ASPECT_BY_FORMAT = {"portrait": "4:5", "square": "1:1", "story": "9:16"}

SYSTEM = (
    "You are an art director for a virtual persona's social media. "
    "Write one text-to-image prompt that keeps the persona's look consistent "
    "and tells the topic visually. Reply only with JSON."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "positive": {"type": "string"},
        "negative": {"type": "string"},
    },
    "required": ["positive", "negative"],
}


def build_prompt(job: PostJob, ctx: StageContext, fix: str = "") -> str:
    style = ctx.config.persona.style
    lines = [
        f"Topic: {job.brief.topic}",
        f"Angle: {job.brief.angle or 'none'}",
        f"Subject: {style.subject}",
        f"Signature: {', '.join(style.signature)}",
        f"Lighting: {style.lighting}",
        f"Camera: {style.camera}",
        f"Avoid: {', '.join(style.avoid)}",
    ]
    if fix:
        lines.append(f"Fix: {fix}")
    return "\n".join(lines)


class PromptWriter(Stage):
    name = "prompt_writer"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        data = ctx.text.complete_json("write_prompt", SYSTEM, build_prompt(job, ctx), SCHEMA)
        job.prompt = ImagePrompt(
            positive=data["positive"],
            negative=data.get("negative", ""),
            aspect_ratio=ASPECT_BY_FORMAT.get(job.brief.format, "4:5"),
        )
        return f"{len(job.prompt.positive)} chars via {ctx.text.name}"
