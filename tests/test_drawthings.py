"""Draw Things backend tests: a tiny fake server stands in for the app."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from persona_pipeline import backends
from persona_pipeline.backends.base import BackendError
from persona_pipeline.backends.drawthings import DrawThingsImage, _seed_from_info
from persona_pipeline.config import PipelineSettings
from persona_pipeline.models import Brief, ImagePrompt
from persona_pipeline.pipeline import Pipeline


def png_b64(size=(64, 80)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, "teal").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeDrawThings:
    """Serves POST /sdapi/v1/txt2img like the Draw Things app does."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server naming)
                length = int(self.headers["Content-Length"])
                outer.requests.append((self.path, json.loads(self.rfile.read(length))))
                body = json.dumps(outer.reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_app():
    servers = []

    def start(reply):
        server = FakeDrawThings(reply)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def backend_for(url, **overrides):
    settings = PipelineSettings()
    settings.drawthings.url = url
    for key, value in overrides.items():
        setattr(settings.drawthings, key, value)
    return DrawThingsImage(settings)


def test_generates_image_over_http(fake_app, tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)
    app = fake_app({"images": [png_b64()], "info": json.dumps({"seed": 1234})})
    backend = backend_for(app.url, steps=4, model="flux_1_schnell.ckpt")
    prompt = ImagePrompt(positive="a calm desk", negative="text", aspect_ratio="4:5")

    image = backend.generate(prompt, tmp_path / "img.png")

    path, body = app.requests[0]
    assert path == "/sdapi/v1/txt2img"
    assert body["prompt"] == "a calm desk"
    assert body["negative_prompt"] == "text"
    assert (body["width"], body["height"]) == (896, 1152)
    assert body["steps"] == 4
    assert body["model"] == "flux_1_schnell.ckpt"
    assert body["seed"] == -1
    assert image.seed == 1234
    assert image.backend == "drawthings"
    with Image.open(image.path) as img:
        assert img.size == (64, 80)


@pytest.mark.parametrize(
    "aspect, scale, expected",
    [("1:1", 1.0, (1024, 1024)), ("9:16", 1.0, (768, 1344)), ("4:5", 0.5, (448, 576))],
)
def test_sizes_are_multiples_of_64(aspect, scale, expected):
    backend = backend_for("http://unused", size_scale=scale)
    body = backend.build_request(ImagePrompt(positive="x", aspect_ratio=aspect))
    assert (body["width"], body["height"]) == expected
    assert body["width"] % 64 == 0 and body["height"] % 64 == 0


def test_env_var_overrides_url(monkeypatch):
    monkeypatch.setenv("DRAWTHINGS_URL", "http://10.0.0.5:7860/")
    assert backend_for("http://127.0.0.1:7860").base_url == "http://10.0.0.5:7860"


def test_app_not_running_gives_setup_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)
    backend = backend_for("http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(BackendError, match="API Server"):
        backend.generate(ImagePrompt(positive="x"), tmp_path / "img.png")


def test_empty_reply_explains_the_open_project_workaround(fake_app, tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)
    app = fake_app({"images": []})
    with pytest.raises(BackendError, match="open or create a project"):
        backend_for(app.url).generate(ImagePrompt(positive="x"), tmp_path / "img.png")


def test_seed_parsing_is_best_effort():
    assert _seed_from_info('{"seed": 7}') == 7
    assert _seed_from_info({"seed": 9}) == 9
    assert _seed_from_info("not json") is None
    assert _seed_from_info(None) is None


def test_registered_image_backends():
    assert {"mock", "drawthings", "gemini"} <= set(backends.available("image"))


def test_full_pipeline_with_drawthings(fake_app, config, tmp_path, monkeypatch):
    monkeypatch.delenv("DRAWTHINGS_URL", raising=False)
    app = fake_app({"images": [png_b64((896, 1152))]})
    config.pipeline.backends.image = "drawthings"
    config.pipeline.drawthings.url = app.url
    config.pipeline.critic_threshold = 0  # mock critic always passes
    job = Pipeline(config).run(Brief(topic="Desk reset"), output_root=tmp_path)
    assert job.succeeded
    assert job.image.backend == "drawthings"
    with Image.open(job.output_dir / "post.png") as img:
        assert img.size == (896, 1152)
