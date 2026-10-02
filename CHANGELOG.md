# Changelog

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
