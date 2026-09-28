"""Stage 5: write a ready-to-post folder (final image, caption.txt, manifest.json)."""

from __future__ import annotations

import shutil

from persona_pipeline.models import PostJob
from persona_pipeline.stages.base import Stage, StageContext


class Packager(Stage):
    name = "packager"

    def run(self, job: PostJob, ctx: StageContext) -> str:
        if job.image is None or job.caption is None or job.output_dir is None:
            raise ValueError("Packager needs an image, a caption and an output_dir")
        final = job.output_dir / "post.png"
        shutil.copyfile(job.image.path, final)
        (job.output_dir / "caption.txt").write_text(job.caption.render() + "\n", encoding="utf-8")
        (job.output_dir / "alt_text.txt").write_text(job.caption.alt_text + "\n", encoding="utf-8")
        return f"ready in {job.output_dir}"
