"""Stage 2: render the image prompt with the configured image backend."""

from __future__ import annotations

from persona_pipeline.models import PostJob
from persona_pipeline.stages.base import Stage, StageContext


class ImageGenerator(Stage):
    name = "image_generator"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        if job.prompt is None or job.output_dir is None:
            raise ValueError("ImageGenerator needs a prompt and an output_dir")
        attempt = job.image.attempt + 1 if job.image else 1
        out_path = job.output_dir / f"image_{attempt:02d}.png"
        job.image = ctx.image.generate(job.prompt, out_path)
        job.image.attempt = attempt
        return f"{job.image.width}x{job.image.height} via {ctx.image.name} (attempt {attempt})"
