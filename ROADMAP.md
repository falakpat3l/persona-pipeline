# Roadmap

Built in small daily milestones.

## v1.0 (complete)

- [x] **1. Skeleton**: persona config, stage interfaces, orchestrator with timing and
      checkpoints, offline mock backends, CLI, tests
- [x] **2. Gemini text**: real prompt writer and caption writer with JSON schema output,
      retries with backoff for rate limits and malformed replies, .env key loading
- [x] **3. Critic loop**: Gemini vision scoring, retry with the critic's notes fed back
      into the prompt, keep the best attempt; bug fixes (hashtag slots, score
      clamping, folder collisions, word-safe caption trimming)
- [x] **4. Image backends**: Draw Things (free local Stable Diffusion / FLUX on macOS)
      over HTTP, optional Gemini image generation (paid tier), `doctor` setup check
- [x] **5. Release**: GitHub Actions test run on every push, design notes, changelog,
      version 1.0.0

## Ideas beyond v1.0 (not planned)

- Batch mode: a content calendar from CSV, with concurrent jobs and a run summary
- HTML contact sheet comparing every attempt side by side
- Cost and token tracking per run
- Resuming a failed job from its manifest
