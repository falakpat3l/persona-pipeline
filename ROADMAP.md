# Roadmap

Built in small daily milestones.

- [x] **1. Skeleton**: persona config, stage interfaces, orchestrator with timing and
      checkpoints, offline mock backends, CLI, tests
- [x] **2. Gemini text**: real prompt writer and caption writer with JSON schema output,
      retries with backoff for rate limits and malformed replies, .env key loading
- [ ] **3. Critic loop**: Gemini vision scoring, retry with the critic's notes fed back
      into the prompt, keep the best attempt
- [ ] **4. Image backends**: Draw Things (free local Stable Diffusion on macOS) over
      HTTP, plus optional Gemini image generation (paid tier)
- [ ] **5. Batch mode**: content calendar from CSV, concurrent jobs, per-run summary
- [ ] **6. CI and report**: GitHub Actions for tests and lint, HTML contact sheet per batch
- [ ] **7. Extras**: cost and token tracking, resume a failed job from its manifest
