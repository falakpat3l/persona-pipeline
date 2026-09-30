"""Stage 3: a vision model scores the image against the persona's style."""

from __future__ import annotations

from persona_pipeline.models import Critique, PostJob
from persona_pipeline.stages.base import Stage, StageContext

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "notes"],
}


def build_rubric(job: PostJob, ctx: StageContext) -> str:
    style = ctx.config.persona.style
    return (
        "Score this image from 0 to 10 as a social media post for a virtual persona.\n"
        f"Topic: {job.brief.topic}\n"
        f"Persona look: {style.subject}\n"
        f"Must include: {', '.join(style.signature)}\n"
        f"Must avoid: {', '.join(style.avoid)}\n"
        "Check: persona consistency, anatomy and artifacts (hands, eyes, jewellery), "
        "topic clarity, and scroll-stopping composition.\n"
        "Scoring: 9 to 10 ready to post, 7 to 8 good with minor issues, 5 to 6 visible "
        "problems, below 5 unusable.\n"
        "Notes: at most 4, each a concrete instruction the next prompt can follow "
        "(for example 'show both hands resting on the desk'), not general praise."
    )


class Critic(Stage):
    name = "critic"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        if job.image is None:
            raise ValueError("Critic needs an image")
        data = ctx.vision.inspect_json("critique", job.image.path, build_rubric(job, ctx), SCHEMA)
        try:
            score = float(data["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"critic returned a non-numeric score: {data['score']!r}") from exc
        score = min(10.0, max(0.0, score))  # models sometimes answer 10.5 or -1
        job.critique = Critique(
            score=score,
            passed=score >= ctx.config.pipeline.critic_threshold,
            notes=[str(n) for n in data.get("notes", []) if str(n).strip()],
        )
        verdict = "pass" if job.critique.passed else "below threshold"
        return f"score {score:.1f} ({verdict}) via {ctx.vision.name}"
