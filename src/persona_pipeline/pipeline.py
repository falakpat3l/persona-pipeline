"""The orchestrator: runs stages in order, times them, and checkpoints the job.

Design notes
- Stages share one `PostJob` object, so the state of a run is always one JSON file.
- After every stage the manifest is rewritten, so a crash leaves a readable trace
  of exactly how far the run got.
- A failed stage stops the run and is recorded, rather than raising into the CLI.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from persona_pipeline import backends
from persona_pipeline.config import Config
from persona_pipeline.models import Brief, PostJob, StageEvent, StageStatus, utcnow
from persona_pipeline.stages import (
    CaptionWriter,
    Critic,
    ImageGenerator,
    Packager,
    PromptWriter,
    Stage,
    StageContext,
)

log = logging.getLogger("persona_pipeline")

DEFAULT_STAGES: tuple[type[Stage], ...] = (
    PromptWriter,
    ImageGenerator,
    Critic,
    CaptionWriter,
    Packager,
)


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "post"


def make_job_id(topic: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{slugify(topic)}"


class Pipeline:
    def __init__(self, config: Config, stages: Sequence[Stage] | None = None):
        self.config = config
        settings = config.pipeline
        choice = settings.backends
        self.ctx = StageContext(
            config=config,
            text=backends.create("text", choice.text, settings),
            image=backends.create("image", choice.image, settings),
            vision=backends.create("vision", choice.vision, settings),
        )
        self.stages: list[Stage] = list(stages) if stages else [cls() for cls in DEFAULT_STAGES]

    def new_job(self, brief: Brief, output_root: Path | None = None) -> PostJob:
        job_id = make_job_id(brief.topic)
        root = output_root or self.config.pipeline.output_dir
        out = Path(root) / job_id
        out.mkdir(parents=True, exist_ok=True)
        return PostJob(job_id=job_id, persona=self.config.persona.name, brief=brief, output_dir=out)

    def run(self, brief: Brief, output_root: Path | None = None) -> PostJob:
        job = self.new_job(brief, output_root)
        log.info("job %s started: %s", job.job_id, brief.topic)
        for stage in self.stages:
            if not self._run_stage(stage, job):
                break
        self.save_manifest(job)
        log.info("job %s %s", job.job_id, "done" if job.succeeded else "failed")
        return job

    def _run_stage(self, stage: Stage, job: PostJob) -> bool:
        started = utcnow()
        t0 = time.perf_counter()
        try:
            detail = stage.run(job, self.ctx)
            status = StageStatus.OK
        except Exception as exc:  # record every failure in the trace
            detail = f"{type(exc).__name__}: {exc}"
            status = StageStatus.FAILED
            if isinstance(exc, backends.BackendError):
                log.error("stage %s failed: %s", stage.name, exc)  # known, readable error
            else:
                log.exception("stage %s failed", stage.name)  # unexpected: show traceback
        duration = (time.perf_counter() - t0) * 1000
        job.events.append(
            StageEvent(
                stage=stage.name,
                status=status,
                started_at=started,
                duration_ms=round(duration, 2),
                detail=detail,
            )
        )
        log.info("  %-16s %-6s %7.1f ms  %s", stage.name, status.value, duration, detail)
        self.save_manifest(job)
        return status == StageStatus.OK

    @staticmethod
    def save_manifest(job: PostJob) -> Path | None:
        if job.output_dir is None:
            return None
        path = job.output_dir / "manifest.json"
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        return path
