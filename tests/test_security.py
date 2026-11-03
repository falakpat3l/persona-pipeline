"""Regression tests for the v1.1.1 security fixes: redaction, timeouts and size
limits, URL checks, safe paths, and model output treated as untrusted text."""

import contextlib
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from persona_pipeline.backends.base import BackendError
from persona_pipeline.backends.drawthings import DrawThingsImage, _post_json
from persona_pipeline.cli import main
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import Brief, GeneratedImage, ImagePrompt
from persona_pipeline.pipeline import Pipeline, slugify
from persona_pipeline.safety import (
    RedactFilter,
    UnsafeValue,
    check_local_url,
    redact,
    safe_child,
    untrusted_text,
)
from persona_pipeline.stages import CaptionWriter, ImageGenerator, Packager, PromptWriter, Stage
from persona_pipeline.stages.critic import Critic

FAKE_KEY = "AIzaFAKEfakeFAKEfakeFAKEfakeFAKEfake1234"
PLAIN_KEY = "plain-secret-value-123"  # not shaped like a Google key


@pytest.fixture
def key_in_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", PLAIN_KEY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return PLAIN_KEY


# Redaction


@pytest.mark.parametrize(
    "text",
    [
        f"request failed for key {FAKE_KEY}",
        "GET https://example.test/v1/models?key=abc123secret&alt=json",
        "headers: {'x-goog-api-key': 'abc123secret'}",
        '{"api_key": "abc123secret"}',
        "GEMINI_API_KEY=abc123secret",
    ],
)
def test_redact_hides_key_shapes(text):
    out = redact(text)
    assert "[redacted]" in out
    assert FAKE_KEY not in out and "abc123secret" not in out


def test_redact_hides_the_current_key_value_anywhere(key_in_env):
    assert redact(f"boom: {PLAIN_KEY}!") == "boom: [redacted]!"


def test_redact_keeps_setup_hints_readable():
    hint = "Add GEMINI_API_KEY=... to a .env file in the project folder."
    assert redact(hint) == hint


def test_failed_stage_never_leaks_key_to_log_or_manifest(config, tmp_path, key_in_env, caplog):
    class LeakyBackendError(Stage):
        name = "leaky_known"

        def run(self, job, ctx):
            raise BackendError(f"auth failed for {PLAIN_KEY}")

    class LeakyCrash(Stage):
        name = "leaky_crash"

        def run(self, job, ctx):
            raise RuntimeError(f"crashed with ?key={FAKE_KEY}")

    for stage in (LeakyBackendError(), LeakyCrash()):
        with caplog.at_level(logging.INFO, logger="persona_pipeline"):
            job = Pipeline(config, stages=[stage]).run(Brief(topic="x"), output_root=tmp_path)
        manifest = (job.output_dir / "manifest.json").read_text()
        for secret in (PLAIN_KEY, FAKE_KEY):
            assert secret not in manifest
            assert secret not in job.events[0].detail
            assert secret not in caplog.text
        assert "[redacted]" in job.events[0].detail
    assert "Traceback" in caplog.text  # unexpected errors still show where they came from


def test_cli_error_message_is_redacted(monkeypatch, capsys, key_in_env):
    def boom(self, brief, output_root=None):
        raise BackendError(f"bad key {PLAIN_KEY}")

    monkeypatch.setattr("persona_pipeline.cli.Pipeline.run", boom)
    code = main(["run", "--persona", "personas/example.yaml", "--topic", "x"])
    err = capsys.readouterr().err
    assert code == 2
    assert PLAIN_KEY not in err and "[redacted]" in err


def test_redact_filter_covers_third_party_logs_and_tracebacks(key_in_env):
    record = logging.LogRecord("httpx", logging.DEBUG, __file__, 1, "sent %s", (PLAIN_KEY,), None)
    try:
        raise ValueError(f"inside {PLAIN_KEY}")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()
    assert RedactFilter().filter(record)
    text = logging.Formatter().format(record)
    assert PLAIN_KEY not in text
    assert "sent [redacted]" in text


def test_gemini_fallback_warning_is_redacted(key_in_env, caplog, monkeypatch):
    from persona_pipeline.backends.gemini import GeminiText

    monkeypatch.setattr("persona_pipeline.retry.time.sleep", lambda s: None)

    class Busy(Exception):
        code = 503

    class Client:
        def __init__(self):
            self.models = self
            self.calls = 0

        def generate_content(self, *, model, contents, config):
            self.calls += 1
            if model == "big":
                raise Busy(f"overloaded, key {PLAIN_KEY}")
            return SimpleNamespace(text='{"a": 1}')

    settings = PipelineSettings()
    settings.gemini.text_model = "big"
    settings.gemini.fallback_models = ["small"]
    settings.gemini.max_retries = 1
    backend = GeminiText(settings, client=Client())
    with caplog.at_level(logging.WARNING, logger="persona_pipeline"):
        assert backend.complete_json("t", "s", "p", {"required": ["a"]}) == {"a": 1}
    assert "falling back" in caplog.text
    assert PLAIN_KEY not in caplog.text


def test_gemini_client_is_created_with_a_timeout(monkeypatch, key_in_env):
    genai = pytest.importorskip("google.genai")
    from persona_pipeline.backends.gemini import GeminiText

    seen = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", fake_client)
    settings = PipelineSettings()
    settings.gemini.timeout_s = 12
    GeminiText(settings).client  # noqa: B018 (creates the client)
    assert seen["http_options"].timeout == 12_000


# HTTP timeouts and size limits (Draw Things)


class StreamingServer:
    """A misbehaving server: `handler(self)` writes whatever the test wants."""

    def __init__(self, handler):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers["Content-Length"]))
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    handler(self)

            do_GET = do_POST  # noqa: N815

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(handler):
        servers.append(StreamingServer(handler))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def _endless(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    for _ in range(10_000):  # far more than the test limit, with no Content-Length
        handler.wfile.write(b" " * 65536)


def test_endless_reply_is_cut_off_at_the_size_limit(serve):
    app = serve(_endless)
    with pytest.raises(BackendError, match="more than 1 MB"):
        _post_json(f"{app.url}/sdapi/v1/txt2img", {}, timeout=10, max_bytes=1024 * 1024)


def test_huge_content_length_is_refused_before_reading(serve):
    def huge(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", str(10 * 1024**3))
        handler.end_headers()

    app = serve(huge)
    with pytest.raises(BackendError, match="was not read"):
        _post_json(f"{app.url}/x", {}, timeout=10, max_bytes=1024 * 1024)


def test_slow_trickle_reply_hits_the_overall_timeout(serve, tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)

    def trickle(handler):
        handler.send_response(200)
        handler.end_headers()
        for _ in range(100):  # one byte every 0.3 s: each read is quick, the whole is not
            handler.wfile.write(b" ")
            handler.wfile.flush()
            time.sleep(0.3)

    app = serve(trickle)
    settings = PipelineSettings()
    settings.drawthings.url = app.url
    settings.drawthings.timeout_s = 1
    started = time.monotonic()
    with pytest.raises(BackendError, match="took longer"):
        DrawThingsImage(settings).generate(ImagePrompt(positive="x"), tmp_path / "i.png")
    assert time.monotonic() - started < 5


def test_redirects_are_not_followed(serve, tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)

    def redirect(handler):
        handler.send_response(307)
        handler.send_header("Location", "http://example.com/steal")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    app = serve(redirect)
    settings = PipelineSettings()
    settings.drawthings.url = app.url
    with pytest.raises(BackendError, match="redirect"):
        DrawThingsImage(settings).generate(ImagePrompt(positive="x"), tmp_path / "i.png")


def test_gemini_rejects_oversized_replies(monkeypatch, tmp_path):
    from persona_pipeline.backends import gemini

    with pytest.raises(gemini.MalformedOutput, match="too long"):
        gemini.parse_json_reply("{" + " " * (gemini.MAX_REPLY_CHARS + 1) + "}", {})

    monkeypatch.setattr(gemini, "MAX_REPLY_IMAGE_BYTES", 10)
    part = SimpleNamespace(inline_data=SimpleNamespace(data=b"x" * 11))
    reply = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: reply))
    backend = gemini.GeminiImage(PipelineSettings(), client=client)
    with pytest.raises(BackendError, match="too large"):
        backend.generate(ImagePrompt(positive="x"), tmp_path / "i.png")
    assert not (tmp_path / "i.png").exists()


# Draw Things URL checks


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://127.0.0.1:7860",
        "127.0.0.1:7860",
        "http://example.com:7860",
        "http://127.0.0.1.example.com:7860",
        "http://localhost@example.com:7860",
        "http://user:pw@127.0.0.1:7860",
        "http://127.0.0.1:7860/?next=http://example.com",
        "http://127.0.0.1:99999",
        "http://:7860",
    ],
)
def test_unsafe_urls_are_refused(url):
    with pytest.raises(UnsafeValue):
        check_local_url(url)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:7860/", "http://127.0.0.1:7860"),
        ("http://localhost:7860", "http://localhost:7860"),
        ("https://LOCALHOST:7860", "https://LOCALHOST:7860"),
        ("http://[::1]:7860", "http://[::1]:7860"),
    ],
)
def test_local_urls_are_allowed(url, expected):
    assert check_local_url(url) == expected


def test_extra_hosts_are_opt_in():
    with pytest.raises(UnsafeValue, match="allowed_hosts"):
        check_local_url("http://192.168.1.20:7860")
    assert check_local_url("http://192.168.1.20:7860", ["192.168.1.20"])


def test_refused_url_never_sends_the_request(monkeypatch, tmp_path):
    monkeypatch.setenv("DRAWTHINGS_URL", "http://example.com:7860")
    sent = []
    backend = DrawThingsImage(PipelineSettings(), post=lambda *a: sent.append(a))
    with pytest.raises(BackendError, match="not allowed"):
        backend.generate(ImagePrompt(positive="x"), tmp_path / "i.png")
    assert sent == []


def test_doctor_flags_a_refused_url(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DRAWTHINGS_URL", "file:///etc/passwd")
    text = Path("personas/example.yaml").read_text().replace("image: mock", "image: drawthings")
    persona = tmp_path / "p.yaml"
    persona.write_text(text)
    assert main(["doctor", "--persona", str(persona)]) == 1
    out = capsys.readouterr().out
    assert "http:// or https://" in out
    assert "answering" not in out  # it did not try to connect


# Output paths stay inside the output folder


@pytest.mark.parametrize(
    "topic",
    ["../../escape", "/etc/passwd", "..\\..\\windows", "C:\\Temp\\x", "a\0b", "..", "~/x"],
)
def test_topics_cannot_escape_the_output_folder(config, tmp_path, topic):
    root = tmp_path / "out"
    job = Pipeline(config).new_job(Brief(topic=topic), output_root=root)
    assert job.output_dir.parent.resolve() == root.resolve()
    assert set(job.job_id) <= set("0123456789abcdefghijklmnopqrstuvwxyz-")


def test_slugify_only_keeps_safe_characters():
    for text in ["../../x", "é/ü\\..", "\x1b[31mred", "a" * 500]:
        slug = slugify(text)
        assert set(slug) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")
        assert 0 < len(slug) <= 40


def test_persona_name_and_model_output_never_become_paths(config, tmp_path):
    config.persona.name = "../../evil"
    root = tmp_path / "out"
    job = Pipeline(config).run(Brief(topic="../../../tmp/evil"), output_root=root)
    assert job.succeeded
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.resolve().is_relative_to(root.resolve()) for p in written)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "x\0y"])
def test_safe_child_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(UnsafeValue):
        safe_child(tmp_path, name)


class _OutsideImage:
    name = "outside"

    def __init__(self, path):
        self.path = path

    def generate(self, prompt, out_path):
        self.path.write_bytes(b"png")
        return GeneratedImage(path=self.path, width=1, height=1, backend=self.name)


def test_image_backend_cannot_save_outside_the_job_folder(config, tmp_path):
    pipeline = Pipeline(config)
    pipeline.ctx.image = _OutsideImage(tmp_path / "elsewhere.png")
    job = pipeline.new_job(Brief(topic="x"), output_root=tmp_path / "out")
    job.prompt = ImagePrompt(positive="x")
    with pytest.raises(UnsafeValue, match="outside the job folder"):
        ImageGenerator().run(job, pipeline.ctx)


def test_packager_refuses_an_image_outside_the_job_folder(config, tmp_path):
    from persona_pipeline.models import Caption

    pipeline = Pipeline(config)
    job = pipeline.new_job(Brief(topic="x"), output_root=tmp_path / "out")
    secret = tmp_path / "not-yours.txt"
    secret.write_text("private")
    job.image = GeneratedImage(path=secret, width=1, height=1, backend="t")
    job.caption = Caption(text="hi")
    with pytest.raises(UnsafeValue):
        Packager().run(job, pipeline.ctx)
    assert not (job.output_dir / "post.png").exists()


# Model output is untrusted text


def test_untrusted_text_drops_control_characters_and_caps_length():
    assert untrusted_text("ok\x1b[2J\x07 fine\nnext\tline\r", 100) == "ok[2J fine\nnext\tline"
    assert untrusted_text("x" * 50, 10) == "x" * 10


class _ScriptedText:
    name = "scripted"

    def __init__(self, replies):
        self.replies = replies

    def complete_json(self, task, system, prompt, schema):
        return self.replies[task]


def test_model_output_is_cleaned_before_use(config, tmp_path):
    pipeline = Pipeline(config)
    pipeline.ctx.text = _ScriptedText(
        {
            "write_prompt": {"positive": "desk\x1b]0;owned\x07" + "p" * 10_000, "negative": 5},
            "write_caption": {
                "text": "Hello\x1b[2J world",
                "hashtags": "not a list",
                "alt_text": "a" * 5000,
            },
        }
    )
    job = pipeline.new_job(Brief(topic="x"), output_root=tmp_path)
    PromptWriter().run(job, pipeline.ctx)
    CaptionWriter().run(job, pipeline.ctx)
    assert "\x1b" not in job.prompt.positive and len(job.prompt.positive) <= 4000
    assert job.prompt.negative == "5"
    assert job.caption.text == "Hello[2J world"
    assert len(job.caption.alt_text) <= 1000
    assert job.caption.hashtags == ["digitalcreator", "designthinking"]


class _FixedVision:
    name = "fixed"

    def __init__(self, reply):
        self.reply = reply

    def inspect_json(self, task, image_path, instruction, schema):
        return self.reply


def _critique(config, tmp_path, reply):
    pipeline = Pipeline(config)
    pipeline.ctx.vision = _FixedVision(reply)
    job = pipeline.new_job(Brief(topic="x"), output_root=tmp_path)
    job.image = GeneratedImage(path=job.output_dir / "i.png", width=1, height=1, backend="t")
    Critic().run(job, pipeline.ctx)
    return job.critique


def test_critic_notes_are_capped_and_cleaned(config, tmp_path):
    critique = _critique(config, tmp_path, {"score": 5, "notes": ["n\x1b[0m" + "x" * 900] * 50})
    assert len(critique.notes) <= 8
    assert all(len(n) <= 500 and "\x1b" not in n for n in critique.notes)


def test_critic_rejects_nan_and_huge_scores_safely(config, tmp_path):
    with pytest.raises(ValueError, match="not a number"):
        _critique(config, tmp_path, {"score": "nan", "notes": []})
    with pytest.raises(ValueError) as info:
        _critique(config, tmp_path, {"score": "x" * 10_000, "notes": []})
    assert len(str(info.value)) < 200


def test_manifest_is_plain_json_even_with_hostile_model_output(config, tmp_path):
    pipeline = Pipeline(config)
    pipeline.ctx.text = _ScriptedText(
        {
            "write_prompt": {"positive": '"}, "evil": true, {"', "negative": ""},
            "write_caption": {"text": "__import__('os')", "hashtags": [], "alt_text": ""},
        }
    )
    job = pipeline.run(Brief(topic="x"), output_root=tmp_path)
    manifest = json.loads((job.output_dir / "manifest.json").read_text())
    assert "evil" not in manifest
    assert manifest["caption"]["text"] == "__import__('os')"  # kept as text, never run
