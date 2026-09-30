"""The generate -> critique -> revise loop, driven by scripted critics."""

from persona_pipeline.models import Brief, Critique
from persona_pipeline.pipeline import Pipeline, ReviewLoop
from persona_pipeline.stages import CaptionWriter, Packager, PromptWriter, Stage
from persona_pipeline.stages.caption_writer import _trim
from persona_pipeline.stages.critic import Critic


class ScriptedCritic(Stage):
    """Returns the next score from a list, with a note that names the attempt."""

    name = "critic"

    def __init__(self, scores, threshold=7.0):
        self.scores = list(scores)
        self.threshold = threshold

    def run(self, job, ctx):
        score = self.scores.pop(0)
        job.critique = Critique(
            score=score, passed=score >= self.threshold, notes=[f"fix number {score}"]
        )
        return f"score {score}"


def run_with_scores(config, tmp_path, scores, max_attempts=3):
    loop = ReviewLoop(review=ScriptedCritic(scores), max_attempts=max_attempts)
    steps = [PromptWriter(), loop, CaptionWriter(), Packager()]
    return Pipeline(config, stages=steps).run(Brief(topic="Loop test"), output_root=tmp_path)


def test_stops_as_soon_as_an_image_passes(config, tmp_path):
    job = run_with_scores(config, tmp_path, [5.0, 8.0, 9.9])
    assert [a.critique.score for a in job.attempts] == [5.0, 8.0]
    assert job.critique.score == 8.0
    assert job.image.attempt == 2
    assert "attempt 2 of 2 (passed)" in job.events[-3].detail


def test_keeps_the_best_attempt_when_none_pass(config, tmp_path):
    job = run_with_scores(config, tmp_path, [4.0, 6.5, 5.0])
    assert len(job.attempts) == 3
    assert job.critique.score == 6.5
    assert job.image.path.name == "image_02.png"
    # the packaged post is the best image, not the last one
    assert (job.output_dir / "post.png").read_bytes() == job.image.path.read_bytes()
    assert "none passed" in job.events[-3].detail


def test_critic_notes_are_fed_back_into_the_next_prompt(config, tmp_path):
    job = run_with_scores(config, tmp_path, [4.0, 9.0])
    first, second = job.attempts
    assert "fix number 4.0" not in first.prompt.positive
    assert "fix number 4.0" in second.prompt.positive
    revised = [e for e in job.events if e.stage == "prompt_writer"][1]
    assert revised.detail.startswith("revised with critic notes")


def test_every_attempt_is_kept_on_disk(config, tmp_path):
    job = run_with_scores(config, tmp_path, [1.0, 2.0, 3.0])
    names = sorted(p.name for p in job.output_dir.glob("image_*.png"))
    assert names == ["image_01.png", "image_02.png", "image_03.png"]


def test_first_image_passing_means_one_attempt(config, tmp_path):
    job = run_with_scores(config, tmp_path, [9.0])
    assert len(job.attempts) == 1
    assert [e.stage for e in job.events].count("prompt_writer") == 1


class FixedVision:
    name = "fixed"

    def __init__(self, reply):
        self.reply = reply

    def inspect_json(self, task, image_path, instruction, schema):
        return self.reply


def _critic_with(config, tmp_path, reply):
    pipeline = Pipeline(config)
    pipeline.ctx.vision = FixedVision(reply)
    job = pipeline.new_job(Brief(topic="x"), output_root=tmp_path)
    img = tmp_path / "i.png"
    img.write_bytes(b"png")
    from persona_pipeline.models import GeneratedImage

    job.image = GeneratedImage(path=img, width=1, height=1, backend="t")
    Critic().run(job, pipeline.ctx)
    return job.critique


def test_critic_clamps_out_of_range_scores(config, tmp_path):
    assert _critic_with(config, tmp_path, {"score": 11, "notes": []}).score == 10.0
    assert _critic_with(config, tmp_path, {"score": -2, "notes": []}).score == 0.0
    assert _critic_with(config, tmp_path, {"score": "8.5", "notes": ["", "ok"]}).notes == ["ok"]


def test_caption_trim_keeps_whole_words():
    text = "one two three four five"
    assert _trim(text, 100) == text
    trimmed = _trim(text, 12)
    assert len(trimmed) <= 12
    assert trimmed.endswith("…")
    assert trimmed[:-1] in ("one two", "one two three")
