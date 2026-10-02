import os

import pytest

from persona_pipeline.env import load_dotenv
from persona_pipeline.retry import with_retries


def test_load_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("PP_A", raising=False)
    monkeypatch.delenv("PP_B", raising=False)
    monkeypatch.setenv("PP_KEEP", "original")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nPP_A=one\nexport PP_B='two words'\nPP_KEEP=changed\nPP_EMPTY=\nbad line\n"
    )

    found = load_dotenv(env)

    assert os.environ["PP_A"] == "one"
    assert os.environ["PP_B"] == "two words"
    assert os.environ["PP_KEEP"] == "original"  # existing env wins
    assert "PP_EMPTY" not in os.environ
    assert found["PP_KEEP"] == "changed"


def test_load_dotenv_inline_comments(tmp_path, monkeypatch):
    monkeypatch.delenv("PP_C", raising=False)
    monkeypatch.delenv("PP_D", raising=False)
    env = tmp_path / ".env"
    env.write_text('PP_C=abc123  # my key\nPP_D="has # inside"\n')
    load_dotenv(env)
    assert os.environ["PP_C"] == "abc123"
    assert os.environ["PP_D"] == "has # inside"


def test_load_dotenv_missing_file(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_with_retries_backoff_grows():
    delays = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise TimeoutError
        return "ok"

    assert with_retries(flaky, attempts=4, base_delay=1, sleep=delays.append) == "ok"
    assert len(delays) == 3
    assert delays[0] < delays[1] < delays[2]  # 1s, 2s, 4s plus jitter


def test_with_retries_stops_on_non_retryable():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retries(boom, is_retryable=lambda e: False, sleep=lambda s: None)
