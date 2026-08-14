# 0002 — Plan, research fleet, scaffold (2026-08-13)

## What happened

- Wrote the overall plan (`docs/plan.md`): 14 steps, M1→M4, each step a lego
  block proven before the next.
- Ran a 4-agent research fleet; results distilled into `docs/research/`:
  our own mlx-engine's ~30 paid-for pitfalls, Qwen3-0.6B's exact 311-tensor
  layout, DeepSeek-V2-Lite's compressed-MLA math, and current crate APIs
  (notably: metal-rs is deprecated → objc2-metal; tokenizers needs
  default-features=false to stay pure-Rust; hf-hub 1.0 broke the old API).
- Chose **mdBook** for the book (`docs/book/`) — the same tool The Rust Book
  uses; installed via brew. Chapter list mirrors the plan steps.
- Scaffolded both crates: folder-per-block with mod.rs barrel + named module
  + README each. Workspace builds; clippy runs; tensor's 3 first tests pass.

## Wrong turn (recorded on purpose)

The scaffold was machine-gunned without explanation — wrong for a learning
project, and the user rightly called it out. Fixed structurally: CLAUDE.md
now has a Teaching-mode HARD RULE (explain what/why/which-concept before
creating anything; per-block rhythm explain → research → discuss → TDD →
review). Also: theory that llm-lab.github.io (the user's interactive
CS336-based lab site) already covers gets linked from chapters, not
re-explained.

## State

Workspace green (3 tests, 4 clippy warnings to address in chapter 1).
Next: chapter 1 deep dive — tensor — in teaching rhythm.
