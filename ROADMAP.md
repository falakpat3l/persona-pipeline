# Roadmap

Built in small daily milestones.

- [x] **1. Skeleton**: persona config, stage interfaces, orchestrator with timing and
      checkpoints, offline mock backends, CLI, tests
- [ ] **2. Gemini text**: real prompt writer and caption writer with JSON schema output
- [ ] **3. Critic loop**: Gemini vision scoring, retry with the critic's notes fed back
      into the prompt, keep the best attempt
- [ ] **4. Image backends**: Gemini image generation and Draw Things (local Stable
      Diffusion on macOS) over HTTP
- [ ] **5. Batch mode**: content calendar from CSV, concurrent jobs, per-run summary
- [ ] **6. CI and report**: GitHub Actions for tests and lint, HTML contact sheet per batch
- [ ] **7. Extras**: cost and token tracking, resume a failed job from its manifest
