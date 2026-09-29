# persona-pipeline

A multi-stage AI orchestration pipeline that turns a one-line topic into a
ready-to-post image, caption, hashtags and alt text for a **virtual persona**,
while keeping the persona's look and voice consistent across every post.

Each step is handled by a different AI role (writer, image model, vision critic),
and every model sits behind a small interface, so vendors can be swapped through
config without touching pipeline code.

```mermaid
flowchart LR
    B[Brief<br/>topic + angle] --> P[Prompt writer<br/>LLM]
    P --> G[Image generator<br/>diffusion model]
    G --> C{Critic<br/>vision model}
    C -- score below threshold --> P
    C -- pass --> W[Caption writer<br/>LLM]
    W --> K[Packager]
    K --> O[(post.png<br/>caption.txt<br/>manifest.json)]
```

> The retry loop from the critic back to the prompt writer lands in milestone 3
> (see [ROADMAP.md](ROADMAP.md)).

## Why this exists

Running an AI persona means making the same kind of decision many times a day:
write a prompt that matches the character, generate, check for broken hands or
off-style results, then write a caption in the right voice. This project turns
that loop into a traceable, testable pipeline.

## Features

- **Persona as config**: look, signature elements, lighting, camera, voice and
  hashtags live in one YAML file ([example](personas/example.yaml)).
- **Pluggable backends**: separate `text`, `image` and `vision` roles, each chosen
  by name in config.
- **Offline mock mode**: the whole pipeline runs with no API key, producing real
  image files, so anyone can clone and try it in seconds.
- **Run trace and checkpoints**: every stage is timed and logged into a
  `manifest.json` that is rewritten after each step, so a crash still leaves a
  readable record of how far the run got.
- **Structured outputs**: every model call asks for JSON against a schema and is
  validated with Pydantic before the next stage sees it.

## Quick start

```bash
git clone https://github.com/falakpat3l/persona-pipeline.git
cd persona-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

persona-pipeline run --persona personas/example.yaml --topic "Three tiny habits for deep focus"
```

Output:

```text
job 20260928-074819-three-tiny-habits-for-deep-focus started: Three tiny habits for deep focus
  prompt_writer    ok         0.4 ms  266 chars via mock
  image_generator  ok        57.0 ms  1080x1350 via mock (attempt 1)
  critic           ok         0.2 ms  score 7.5 (pass) via mock
  caption_writer   ok         0.2 ms  85 chars, 7 hashtags
  packager         ok         0.2 ms  ready in outputs/20260928-074819-three-tiny-habits-for-deep-focus
```

Each run gets its own folder:

```text
outputs/20260928-074819-three-tiny-habits-for-deep-focus/
├── image_01.png     # every attempt is kept
├── post.png         # the chosen image
├── caption.txt      # caption + hashtags, ready to paste
├── alt_text.txt     # accessibility text
└── manifest.json    # full trace: brief, prompt, scores, timings
```

## Backends

| Role   | Available now    | Planned                        |
| ------ | ---------------- | ------------------------------ |
| text   | `mock`, `gemini` |                                |
| image  | `mock`           | `drawthings` (local), `gemini` |
| vision | `mock`           | `gemini`                       |

### Using Gemini

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Install the extra and add your key:

   ```bash
   pip install -e ".[gemini]"
   echo "GEMINI_API_KEY=your-key-here" > .env
   ```

3. Run with Gemini writing the prompt and caption:

   ```bash
   persona-pipeline run --persona personas/example.yaml --topic "Three tiny habits for deep focus" --text gemini
   ```

Every Gemini call uses structured output (a JSON schema per stage), checks the
reply before the next stage sees it, and retries rate limits (429), server errors
(5xx), dropped connections and malformed replies with exponential backoff and jitter. `.env` is
git-ignored, so the key never lands in the repo.

Pick backends in the persona file, or override per run:

```bash
persona-pipeline run --persona personas/example.yaml --topic "..." --image mock
persona-pipeline backends   # list what is installed
```

## Project layout

```text
src/persona_pipeline/
├── pipeline.py          # orchestrator: ordering, timing, checkpoints, failure handling
├── config.py            # persona + pipeline settings (Pydantic)
├── models.py            # PostJob and the data passed between stages
├── retry.py             # exponential backoff with jitter
├── env.py               # tiny .env loader for API keys
├── backends/
│   ├── base.py          # TextBackend, ImageBackend, VisionBackend interfaces
│   ├── mock.py          # offline backends used by default and in tests
│   ├── gemini.py        # Gemini text backend: structured JSON output + retries
│   └── __init__.py      # registry: config name -> backend class
└── stages/
    ├── prompt_writer.py
    ├── image_generator.py
    ├── critic.py
    ├── caption_writer.py
    └── packager.py
```

## Adding a stage

A stage is a class with a `name` and a `run(job, ctx)` method that reads from and
writes to the shared `PostJob`:

```python
from persona_pipeline.stages import Stage

class Watermark(Stage):
    name = "watermark"

    def run(self, job, ctx):
        ...  # edit job.image.path
        return "added corner mark"
```

Pass a custom list to `Pipeline(config, stages=[...])` to change the flow.

## Tests

```bash
pytest
ruff check .
```

## License

MIT
