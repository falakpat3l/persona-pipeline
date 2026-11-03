"""Stage 2: render the image prompt with the configured image backend."""

from __future__ import annotations

from persona_pipeline.models import PostJob
from persona_pipeline.safety import UnsafeValue, is_inside
from persona_pipeline.stages.base import Stage, StageContext


class ImageGenerator(Stage):
    name = "image_generator"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        if job.prompt is None or job.output_dir is None:
            raise ValueError("ImageGenerator needs a prompt and an output_dir")
        attempt = job.image.attempt + 1 if job.image else 1
        out_path = job.output_dir / f"image_{attempt:02d}.png"
        image = ctx.image.generate(job.prompt, out_path)
        if not is_inside(image.path, job.output_dir):
            raise UnsafeValue(f"the {ctx.image.name} image backend saved outside the job folder")
        job.image = image
        job.image.attempt = attempt
        return f"{job.image.width}x{job.image.height} via {ctx.image.name} (attempt {attempt})"
