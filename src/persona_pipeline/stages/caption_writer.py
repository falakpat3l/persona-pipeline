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


def _unique(tags) -> list[str]:
    out: list[str] = []
    for tag in tags:
        if tag and tag not in out:
            out.append(tag)
    return out


def _trim(text: str, limit: int) -> str:
    """Cut to `limit` characters without breaking a word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "…"


class CaptionWriter(Stage):
    name = "caption_writer"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        persona = ctx.config.persona
        voice = persona.voice
        system = (
            f"You write Instagram captions as {persona.name}. {persona.bio} "
            f"Tone: {voice.tone}. Language: {voice.language}. Emoji use: {voice.emoji_level}. "
            f"Keep the caption under {voice.max_caption_chars} characters. "
            "Open with a short hook line, give one useful idea about the topic, and end with "
            "a light question to invite comments. "
            f"Suggest up to {voice.max_hashtags} specific hashtags without the # sign. "
            "alt_text describes the image plainly for screen readers in one sentence. "
            "Reply only with JSON."
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

        # The persona's base tags always make it in; model tags fill the remaining slots.
        base = _unique(_clean_tag(t) for t in voice.base_hashtags)[: voice.max_hashtags]
        extra = [
            t for t in _unique(_clean_tag(t) for t in data.get("hashtags", [])) if t not in base
        ]
        tags = extra[: voice.max_hashtags - len(base)] + base

        job.caption = Caption(
            text=_trim(data["text"], voice.max_caption_chars),
            hashtags=tags,
            alt_text=data.get("alt_text", ""),
        )
        return f"{len(job.caption.text)} chars, {len(job.caption.hashtags)} hashtags"
