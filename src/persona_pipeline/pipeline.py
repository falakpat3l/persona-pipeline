"""The orchestrator: runs stages in order, times them, and checkpoints the job.

Design notes
- Stages share one `PostJob` object, so the state of a run is always one JSON file.
- After every stage the manifest is rewritten, so a crash leaves a readable trace
  of exactly how far the run got.
- A failed stage stops the run and is recorded, rather than raising into the CLI.
- Generation is a feedback loop: a vision critic scores each image, and a weak
  image sends the critic's notes back to the prompt writer for another try. The
  best-scoring attempt is kept, even if none reached the threshold.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from persona_pipeline import backends
from persona_pipeline.config import Config
from persona_pipeline.models import Attempt, Brief, PostJob, StageEvent, StageStatus, utcnow
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


@dataclass
class ReviewLoop:
    """Composite step: generate, critique, and revise until the critic is happy.

    Runs `generate` then `review`. If the critique does not pass and attempts
    remain, runs `revise` (which reads the critic's notes) and tries again.
    `max_attempts=None` means "use the pipeline setting".
    """

    generate: Stage = field(default_factory=ImageGenerator)
    review: Stage = field(default_factory=Critic)
    revise: Stage = field(default_factory=PromptWriter)
    max_attempts: int | None = None
    name: str = "review_loop"


Step = Stage | ReviewLoop


def default_steps() -> list[Step]:
    return [PromptWriter(), ReviewLoop(), CaptionWriter(), Packager()]


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "post"


def make_job_id(topic: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{slugify(topic)}"


class Pipeline:
    def __init__(self, config: Config, stages: Sequence[Step] | None = None):
        self.config = config
        settings = config.pipeline
        choice = settings.backends
        self.ctx = StageContext(
            config=config,
            text=backends.create("text", choice.text, settings),
            image=backends.create("image", choice.image, settings),
            vision=backends.create("vision", choice.vision, settings),
        )
        self.stages: list[Step] = list(stages) if stages else default_steps()

    def new_job(self, brief: Brief, output_root: Path | None = None) -> PostJob:
        root = Path(output_root or self.config.pipeline.output_dir)
        job_id = make_job_id(brief.topic)
        # Two runs of the same topic in the same second must not share a folder.
        base, n = job_id, 2
        while (root / job_id).exists():
            job_id, n = f"{base}-{n}", n + 1
        out = root / job_id
        out.mkdir(parents=True)
        return PostJob(job_id=job_id, persona=self.config.persona.name, brief=brief, output_dir=out)

    def run(self, brief: Brief, output_root: Path | None = None) -> PostJob:
        job = self.new_job(brief, output_root)
        log.info("job %s started: %s", job.job_id, brief.topic)
        for step in self.stages:
            ok = (
                self._run_review_loop(step, job)
                if isinstance(step, ReviewLoop)
                else self._run_stage(step, job)
            )
            if not ok:
                break
        self.save_manifest(job)
        log.info("job %s %s", job.job_id, "done" if job.succeeded else "failed")
        return job

    def _run_review_loop(self, loop: ReviewLoop, job: PostJob) -> bool:
        limit = loop.max_attempts or self.config.pipeline.max_attempts
        started = utcnow()
        t0 = time.perf_counter()

        for number in range(1, limit + 1):
            steps = [loop.generate, loop.review]
            if number > 1:
                steps.insert(0, loop.revise)
            if not all(self._run_stage(step, job) for step in steps):
                if not job.attempts:
                    return False  # nothing usable yet: the run fails
                # A retry failed (for example the API went down): keep the best attempt so far.
                failed = job.events[-1]
                failed.status = StageStatus.SKIPPED
                failed.detail += " (retry abandoned, kept the earlier attempt)"
                log.warning("  retry %d failed, keeping the best earlier attempt", number)
                break
            job.attempts.append(
                Attempt(number=number, prompt=job.prompt, image=job.image, critique=job.critique)
            )
            if job.critique.passed:
                break

        best = max(job.attempts, key=lambda a: a.critique.score)
        job.prompt, job.image, job.critique = best.prompt, best.image, best.critique
        verdict = "passed" if best.critique.passed else "none passed, kept the best"
        detail = (
            f"best {best.critique.score:.1f} from attempt {best.number} "
            f"of {len(job.attempts)} ({verdict})"
        )
        duration = (time.perf_counter() - t0) * 1000
        job.events.append(
            StageEvent(
                stage=loop.name,
                status=StageStatus.OK,
                started_at=started,
                duration_ms=round(duration, 2),
                detail=detail,
            )
        )
        log.info("  %-16s %-6s %7.1f ms  %s", loop.name, "ok", duration, detail)
        self.save_manifest(job)
        return True

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
