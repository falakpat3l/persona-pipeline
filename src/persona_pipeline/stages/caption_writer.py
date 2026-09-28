"""Stage 4: write the caption, hashtags and alt text in the persona's voice."""

from __future__ import annotations

from persona_pipeline.models import Caption, PostJob
from persona_pipeline.stages.base import Stage, StageContext

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "alt_text": {"type": "string"},
    },
    "required": ["text", "hashtags", "alt_text"],
}


def _clean_tag(tag: str) -> str:
    return "".join(ch for ch in tag.lstrip("#") if ch.isalnum() or ch == "_").lower()


class CaptionWriter(Stage):
    name = "caption_writer"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        persona = ctx.config.persona
        voice = persona.voice
        system = (
            f"You write Instagram captions as {persona.name}. {persona.bio} "
            f"Tone: {voice.tone}. Language: {voice.language}. Emoji use: {voice.emoji_level}. "
            f"Keep the caption under {voice.max_caption_chars} characters. Reply only with JSON."
        )
        prompt = "\n".join(
            [
                f"Topic: {job.brief.topic}",
                f"Angle: {job.brief.angle or 'none'}",
                f"Subject: {persona.style.subject}",
                f"Image prompt: {job.prompt.positive if job.prompt else ''}",
            ]
        )
        data = ctx.text.complete_json("write_caption", system, prompt, SCHEMA)

        # Merge model tags with the persona's base tags, dedupe, keep order, cap count.
        tags: list[str] = []
        for tag in [*data.get("hashtags", []), *voice.base_hashtags]:
            clean = _clean_tag(tag)
            if clean and clean not in tags:
                tags.append(clean)

        job.caption = Caption(
            text=data["text"][: voice.max_caption_chars],
            hashtags=tags[: voice.max_hashtags],
            alt_text=data.get("alt_text", ""),
        )
        return f"{len(job.caption.text)} chars, {len(job.caption.hashtags)} hashtags"
