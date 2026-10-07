# Guide: how this code works

A plain-English tour, so you can find your way around and change things safely.

## What it does

You give it a topic. It gives you back a folder with an image, a caption, hashtags and alt text for one social media post, in the style of the persona described in a YAML file.

## Run it

1. `pip install -e ".[gemini,dev]"`
2. `persona-pipeline run --persona personas/example.yaml --topic "calm mornings"`
3. Open the new folder inside `outputs/`.

With the example file everything runs on fake ("mock") models, so it works offline with no key.

## The big picture

```
topic
  -> 1. prompt_writer    writes the image prompt         (text model)
  -> 2. image_generator  draws the picture               (image model)
  -> 3. critic           scores the picture out of 10    (vision model)
        score too low? send the critic's notes back to step 1 and try again
        (up to max_attempts; the best picture is kept)
  -> 4. caption_writer   writes caption and hashtags     (text model)
  -> 5. packager         saves post.png, caption.txt, alt_text.txt
```

Every step reads from and writes to one shared object, the `PostJob`. After each step the whole job is saved to `manifest.json`, so if something fails you can open that file and see exactly how far it got.

## Where things live

| File | What it does | Open it when you want to... |
|---|---|---|
| `personas/example.yaml` | The persona: looks, voice, hashtags, which models to use | make a new persona or switch models |
| `src/persona_pipeline/cli.py` | The `persona-pipeline` command (`run`, `doctor`, `backends`) | add a command or a flag |
| `src/persona_pipeline/pipeline.py` | Runs the steps in order, the retry loop, saves `manifest.json` | change the order of steps |
| `src/persona_pipeline/stages/` | One file per step (the five above) | change what a step asks the model, or what it does with the answer |
| `src/persona_pipeline/backends/` | The model connectors: `mock`, `gemini`, `drawthings` | add a new AI service |
| `src/persona_pipeline/config.py` | Every setting the YAML file can have, with its default | add a new setting |
| `src/persona_pipeline/models.py` | The shapes of the data passed between steps (prompt, image, critique, caption) | store a new piece of information |
| `src/persona_pipeline/retry.py` | Try again after a short wait when an API is busy | change how patient it is |
| `src/persona_pipeline/env.py` | Reads your `.env` file (where the API key lives) | rarely |
| `tests/` | Checks that everything still works | after any change |

## Common changes

**Change the persona.** Copy `personas/example.yaml`, edit the text, run with `--persona` pointing at your copy. No code needed.

**Use real models.** In the YAML, set `text: gemini`, `vision: gemini`, `image: drawthings`. Put `GEMINI_API_KEY=...` in a `.env` file. Run `persona-pipeline doctor --persona yourfile.yaml` to check everything is ready.

**Change what the critic looks for.** Edit the instruction text in `stages/critic.py`. The pass mark is `critic_threshold` in the YAML.

**Change the caption style.** Edit the instruction text in `stages/caption_writer.py`. Length, emoji level and hashtags come from the `voice` section of the YAML.

**Add a new step.** Make a new file in `stages/` with a class that has a `name` and a `run(job, ctx)` method that returns a short message. Add it to `default_steps()` in `pipeline.py`.

**Add a new AI service.** Make a new file in `backends/` with a class that has the same method as the others for its role (`complete_json` for text, `generate` for images, `inspect_json` for vision). Register its name in `backends/__init__.py`.

## Check you didn't break anything

```
pytest
```

All tests should pass. They use fake models, so they run in a few seconds and never touch the internet.

## Words you will see

- **Backend**: the connector to one AI service. Each has a role: text, image or vision.
- **Mock**: a fake backend that returns sensible made-up answers, for testing.
- **Stage**: one step of the pipeline.
- **Schema**: a description of the JSON we want back from the model, so the answer always has the same fields.
- **Retry with backoff**: if the API says "busy", wait 1 s, then 2 s, then 4 s, and try again.
- **Fallback model**: if the main Gemini model stays busy, use the next one in `fallback_models`.
- **Manifest**: `manifest.json`, the full record of one run.
