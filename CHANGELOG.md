# Changelog

## 1.1.0 (2026-10-21)

Final release. The project is complete and maintained only for fixes.

Fixes:

- An unknown backend name (for example `--text nope`) printed a traceback; it is
  now a one-line error with the list of valid names, and exit code 2
- Passing a folder as `--persona`, or an `--out` path that is a file, printed a
  traceback; both are now short errors with exit code 2
- `doctor` said "All good." for a persona file with a misspelled backend name; it
  now flags it
- An empty `--topic` is rejected instead of creating a folder named `post`
- Ctrl+C stops cleanly with exit code 130
- A non-image reply from the Gemini image model, or a non-JSON reply from
  whatever is listening at `DRAWTHINGS_URL`, now raises a readable `BackendError`
- Persona files are checked for an empty `name`, `max_caption_chars` below 20 and
  a negative `max_hashtags`
- "best of 1 attempts" now reads "best of 1 attempt"
- Removed a stray screenshot that was committed by mistake

Docs and CI:

- README: one-paragraph summary, a text diagram of the stages, a no-key quick
  start with the mock backends, exit codes, and a "Possible extensions" section
  that replaces `ROADMAP.md`
- GitHub Actions now also tests Python 3.13
- 7 new tests (74 in total)

## 1.0.2 (2026-10-07)

Easier to read and work on. No change in behaviour.

- New `GUIDE.md`: a plain-English tour of how the code works and where to make changes
- Plainer comments in the Gemini backend (`backends/gemini.py`)

## 1.0.1 (2026-10-02)

Bug fixes found in a review:

- Mock images crashed on Pillow 10.0; the minimum is now Pillow 10.1
- Gemini image generation failed on google-genai older than 1.40; it now sends a
  typed `ImageConfig` and requires google-genai 1.40 or newer
- A failure on a retry (for example the API going down on attempt 2) failed the
  whole run; the critic loop now keeps the best earlier attempt instead
- `.env` values followed by an inline `# comment` included the comment
- A missing or invalid persona file printed a long traceback; the CLI now shows a
  short, readable error
- The test suite now also runs on the oldest supported versions (Python 3.10)

## 1.0.0 (2026-10-02)

First complete release.

- GitHub Actions runs the test suite and lint on every push
- Design decisions and project status documented in the README

## 0.4.0 (2026-10-01)

- Draw Things image backend (local, free), talking to the app's HTTP API
- Optional Gemini image backend (paid tier)
- `persona-pipeline doctor` checks API keys and apps before a run

## 0.3.0 (2026-09-30)

- Vision critic loop: score, feed notes back to the prompt writer, keep the best attempt
- Gemini vision backend
- Fixes: persona hashtags no longer crowded out, out-of-range scores clamped,
  same-second runs get separate folders, captions trimmed on whole words

## 0.2.x (2026-09-29)

- Gemini text backend with structured JSON output
- Retries with exponential backoff and jitter; fallback to lighter models when overloaded
- `.env` loading for API keys; quieter logs by default

## 0.1.0 (2026-09-28)

- Pipeline skeleton: persona config, stages, orchestrator, offline mock backends, CLI, tests
