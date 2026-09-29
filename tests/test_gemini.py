"""Gemini backend tests. They use a fake client, so no API key or network is needed."""

from types import SimpleNamespace

import pytest

from persona_pipeline import backends
from persona_pipeline.backends.base import BackendError
from persona_pipeline.backends.gemini import (
    GeminiText,
    MalformedOutput,
    api_key_from_env,
    is_retryable,
    parse_json_reply,
)
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import Brief
from persona_pipeline.pipeline import Pipeline

SCHEMA = {"type": "object", "required": ["positive", "negative"]}


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeClient:
    """Mimics client.models.generate_content and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("persona_pipeline.retry.time.sleep", lambda s: None)


def make_backend(replies, **gemini):
    settings = PipelineSettings()
    for key, value in gemini.items():
        setattr(settings.gemini, key, value)
    client = FakeClient(replies)
    return GeminiText(settings, client=client), client


# parse_json_reply


def test_parse_plain_json():
    assert parse_json_reply('{"positive": "a", "negative": "b"}', SCHEMA)["positive"] == "a"


def test_parse_strips_code_fences():
    reply = '```json\n{"positive": "a", "negative": ""}\n```'
    assert parse_json_reply(reply, SCHEMA)["negative"] == ""


@pytest.mark.parametrize(
    "reply, message",
    [
        (None, "empty"),
        ("", "empty"),
        ("sure! here you go", "not JSON"),
        ("[1, 2]", "JSON object"),
        ('{"positive": "a"}', "missing keys: negative"),
    ],
)
def test_parse_rejects_bad_replies(reply, message):
    with pytest.raises(MalformedOutput, match=message):
        parse_json_reply(reply, SCHEMA)


# retry policy


@pytest.mark.parametrize("code, expected", [(429, True), (503, True), (400, False), (403, False)])
def test_retryable_status_codes(code, expected):
    assert is_retryable(FakeAPIError(code)) is expected


def test_malformed_output_is_retryable():
    assert is_retryable(MalformedOutput("x"))


# GeminiText


def test_sends_structured_output_request():
    backend, client = make_backend(['{"positive": "p", "negative": "n"}'], text_model="test-model")
    out = backend.complete_json("write_prompt", "SYSTEM", "Topic: x", SCHEMA)

    assert out == {"positive": "p", "negative": "n"}
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "Topic: x"
    assert call["config"]["system_instruction"] == "SYSTEM"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["response_json_schema"] is SCHEMA


def test_retries_rate_limit_then_succeeds():
    backend, client = make_backend(
        [FakeAPIError(429), "not json", '{"positive": "p", "negative": "n"}']
    )
    assert backend.complete_json("t", "s", "p", SCHEMA)["positive"] == "p"
    assert len(client.calls) == 3


def test_gives_up_after_max_retries():
    backend, client = make_backend([FakeAPIError(503)] * 5, max_retries=2)
    with pytest.raises(FakeAPIError):
        backend.complete_json("t", "s", "p", SCHEMA)
    assert len(client.calls) == 2


def test_does_not_retry_bad_request():
    backend, client = make_backend([FakeAPIError(400), "unused"])
    with pytest.raises(FakeAPIError):
        backend.complete_json("t", "s", "p", SCHEMA)
    assert len(client.calls) == 1


def test_missing_api_key_has_helpful_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(BackendError, match="aistudio.google.com"):
        api_key_from_env()


def test_registered_as_text_backend():
    assert "gemini" in backends.available("text")


def test_full_pipeline_with_gemini_text(config, tmp_path):
    """Gemini writes the prompt and caption; image and vision stay on mock."""
    replies = [
        '{"positive": "woman with gold jewellery at a sunlit desk", "negative": "text"}',
        '{"text": "Small habits, big focus. Which one would you try?",'
        ' "hashtags": ["#Focus", "deep work", "focus"], "alt_text": "A woman at a desk."}',
    ]
    pipeline = Pipeline(config)
    pipeline.ctx.text, client = make_backend(replies)

    job = pipeline.run(Brief(topic="Deep focus"), output_root=tmp_path)

    assert job.succeeded
    assert job.prompt.positive.startswith("woman with gold jewellery")
    assert job.caption.text.startswith("Small habits")
    # model tags are cleaned and merged with the persona's base tags, without duplicates
    assert job.caption.hashtags[:2] == ["focus", "deepwork"]
    assert "digitalcreator" in job.caption.hashtags
    # the prompt writer sent the persona's signature to the model
    assert "layered gold jewellery" in client.calls[0]["contents"]
