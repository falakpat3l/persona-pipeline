import json

import pytest
from PIL import Image

from persona_pipeline import backends
from persona_pipeline.backends import BackendError
from persona_pipeline.cli import main
from persona_pipeline.models import Brief, StageStatus
from persona_pipeline.pipeline import Pipeline, make_job_id, slugify
from persona_pipeline.stages import Stage


def test_config_loads(config):
    assert config.persona.name == "Mira Vale"
    assert config.pipeline.backends.text == "mock"
    assert "layered gold jewellery" in config.persona.style.signature


def test_full_mock_run_produces_post_folder(config, tmp_path):
    job = Pipeline(config).run(Brief(topic="Focus tips for students"), output_root=tmp_path)

    assert job.succeeded
    stages = [e.stage for e in job.events]
    assert stages[:3] == ["prompt_writer", "image_generator", "critic"]
    assert stages[-3:] == ["review_loop", "caption_writer", "packager"]
    out = job.output_dir
    for name in ("post.png", "caption.txt", "alt_text.txt", "manifest.json"):
        assert (out / name).exists(), name

    with Image.open(out / "post.png") as img:
        assert img.size == (1080, 1350)  # portrait = 4:5

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["brief"]["topic"] == "Focus tips for students"
    assert manifest["critique"]["score"] >= 0


def test_format_controls_aspect_ratio(config, tmp_path):
    job = Pipeline(config).run(Brief(topic="x", format="story"), output_root=tmp_path)
    assert job.prompt.aspect_ratio == "9:16"
    assert (job.image.width, job.image.height) == (1080, 1920)


def test_prompt_carries_persona_signature(config, tmp_path):
    job = Pipeline(config).run(Brief(topic="Desk setup"), output_root=tmp_path)
    assert "layered gold jewellery" in job.prompt.positive
    assert "watermarks" in job.prompt.negative


def test_hashtags_are_clean_deduped_and_capped(config, tmp_path):
    config.persona.voice.max_hashtags = 3
    job = Pipeline(config).run(Brief(topic="Design design habits"), output_root=tmp_path)
    tags = job.caption.hashtags
    assert len(tags) <= 3
    assert len(tags) == len(set(tags))
    assert all(t.isalnum() or "_" in t for t in tags)


def test_base_hashtags_are_never_crowded_out(config, tmp_path):
    """Regression: a model that returns many tags used to push the persona's own tags out."""
    config.persona.voice.max_hashtags = 4
    config.persona.voice.base_hashtags = ["brandtag", "second"]
    job = Pipeline(config).run(
        Brief(topic="alpha bravo charlie delta echo foxtrot"), output_root=tmp_path
    )
    assert len(job.caption.hashtags) == 4
    assert job.caption.hashtags[-2:] == ["brandtag", "second"]


def test_failing_stage_stops_run_and_is_recorded(config, tmp_path):
    class Boom(Stage):
        name = "boom"

        def run(self, job, ctx):
            raise RuntimeError("model timed out")

    class NeverRuns(Stage):
        name = "never"

        def run(self, job, ctx):  # pragma: no cover
            raise AssertionError("should not run")

    job = Pipeline(config, stages=[Boom(), NeverRuns()]).run(Brief(topic="x"), output_root=tmp_path)
    assert not job.succeeded
    assert len(job.events) == 1
    assert job.events[0].status == StageStatus.FAILED
    assert "model timed out" in job.events[0].detail
    assert (job.output_dir / "manifest.json").exists()


def test_unknown_backend_gives_helpful_error():
    with pytest.raises(BackendError, match="Available: mock"):
        backends.create("image", "does-not-exist")


def test_same_topic_twice_in_one_second_gets_separate_folders(config, tmp_path):
    pipeline = Pipeline(config)
    a = pipeline.new_job(Brief(topic="Same"), output_root=tmp_path)
    b = pipeline.new_job(Brief(topic="Same"), output_root=tmp_path)
    assert a.output_dir != b.output_dir


def test_slug_and_job_id():
    assert slugify("Hello, World! 2026") == "hello-world-2026"
    assert slugify("!!!") == "post"
    assert make_job_id("Morning Light").endswith("-morning-light")


def test_cli_run(tmp_path, capsys):
    code = main(
        [
            "run",
            "--persona",
            "personas/example.yaml",
            "--topic",
            "Calm mornings",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "Status:  OK" in capsys.readouterr().out
