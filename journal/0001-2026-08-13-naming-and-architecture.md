# 0001 — Naming and architecture (2026-08-13)

The project started as a question: *what's unique about llama.cpp, and why
can't we do that in Rust?* The answer we converged on: nothing C does is
C-magic — the hot loops are GPU shaders regardless of host language, and the
real moat is accumulated kernel engineering, not the language. So: build one,
in Rust, and learn everything on the way.

## Wrong turns and decisions

- **Nearly went half-and-half again.** First instinct was to reuse mlx-rs
  (prior fork work). But mlx-rs wraps Apple's C++ MLX via FFI — the same
  split-brain we were escaping. Decision: no MLX. The GPU is reached through
  the Metal API directly (`objc2-metal`), which needs zero C/C++.
- **Family picks changed twice.** Nemotron was dropped when it turned out the
  interesting ones are Mamba2 hybrids (a whole extra kernel family too early).
  DeepSeek-R1-*distills* were rejected as a second family because they ARE
  Qwen architecture — they'd teach the abstraction nothing. Final:
  **Qwen3 dense** (walking skeleton) then **DeepSeek-V2-Lite** (MLA + MoE,
  the real stress test). Distills run for free via the Qwen family.
- **Naming safari.** quark/gluon/fermion/muon all collide on crates.io;
  gluon doubly bad (a Rust language AND MXNet's ML API). Walked the whole
  Standard Model: the only unclaimed particles turned out to be the **W and Z
  bosons**. But the user proposed **nucleon** — minor collisions only, and
  physically apt: the thing quarks and gluons actually build. Sits next to
  higgs. Backend crates named after the GPU API, not Apple's library:
  `nucleon-metal`, later `nucleon-cuda` (NOT `nucleon-mlx` — we don't use MLX).
- **CpuBackend-first** chosen as the pedagogical spine: the entire engine runs
  on naive single-steppable CPU code before the first shader exists, and every
  Metal kernel is verified against it.
- **Safetensors bf16 before GGUF quant** — tokens flowing early beats writing
  Q4_K dequant kernels before anything works.

## State at end of session

Workspace scaffolded, design doc written, M1 underway.
