"""Stage 1: turn a short brief into a detailed, on-style image prompt."""

from __future__ import annotations

from persona_pipeline.models import ImagePrompt, PostJob
from persona_pipeline.stages.base import Stage, StageContext

ASPECT_BY_FORMAT = {"portrait": "4:5", "square": "1:1", "story": "9:16"}

SYSTEM = """\
You are the art director for a virtual persona's social media account.
Write ONE text-to-image prompt for a single photo that tells the topic visually.

Rules:
- Start with the persona's subject description, copied exactly, so the face stays consistent.
- Include every signature element, worded exactly as given.
- Then describe setting, action or pose, composition, lighting and camera.
- Show the topic through the scene and props, never through written words in the image.
- 60 to 120 words, comma-separated phrases, no quotation marks.
- If a "Fix" line is given, the previous image had those problems: correct them.
- "negative" lists things to keep out of the image, including everything under Avoid.
Reply only with JSON."""

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
    if fix and job.prompt is not None:
        lines.append(f"Previous prompt: {job.prompt.positive}")
    if fix:
        lines.append(f"Fix: {fix}")
    return "\n".join(lines)


def critic_feedback(job: PostJob) -> str:
    """The critic's notes on the last image, if it did not pass. Empty otherwise."""
    if job.critique is None or job.critique.passed:
        return ""
    return "; ".join(job.critique.notes) or "the image scored below the quality bar"


class PromptWriter(Stage):
    name = "prompt_writer"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        fix = critic_feedback(job)
        data = ctx.text.complete_json("write_prompt", SYSTEM, build_prompt(job, ctx, fix), SCHEMA)
        job.prompt = ImagePrompt(
            positive=data["positive"],
            negative=data.get("negative", ""),
            aspect_ratio=ASPECT_BY_FORMAT.get(job.brief.format, "4:5"),
        )
        mode = "revised with critic notes" if fix else "new"
        return f"{mode}, {len(job.prompt.positive)} chars via {ctx.text.name}"
