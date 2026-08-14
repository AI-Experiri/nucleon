# nucleon design

Date: 2026-08-13. Status: approved.

## Goal

A from-scratch, pure-Rust LLM inference engine for Apple Silicon, optimized
for **readability and learnability first, performance second** — and a
documented journey good enough to become a book.

- Everything in Rust except `.metal` shader files (the GPU's native language).
- No llama.cpp, no MLX, no FFI to anyone's C/C++.
- Standalone project; not part of higgs (higgs may consume it much later).
- Lego-block architecture: each module is one chapter — small, single-purpose,
  swappable, testable alone.

## Non-goals (for now)

- CPU SIMD hand-tuning (CpuBackend is a *reference*, not a race car).
- Windows/Linux/CUDA (`nucleon-cuda` is a later sibling crate, same shape).
- Serving/HTTP (library + CLI only).
- Training.

## Key decisions (each has an ADR)

1. **No MLX, no llama.cpp** — the point is to own every layer.
2. **Backend trait with CpuBackend first** — the whole engine runs end-to-end
   on naive, single-steppable CPU code before any GPU work. Every Metal kernel
   is then verified op-by-op against the CPU reference.
3. **Safetensors bf16/f16 first, GGUF+quant later (M3)** — get tokens flowing
   on the simple path; quantization lands as an isolated layer.
4. **Families: Qwen3 first, DeepSeek-V2-Lite second** — Qwen3 is the clean
   dense transformer; DeepSeek (MLA attention + MoE) stress-tests the
   ModelFamily abstraction so it stays real, not llama-with-renamed-fields.
   R1-Distill-Qwen checkpoints run for free via the Qwen3 family.
5. **objc2-metal for the GPU** — thin Rust bindings straight to the Metal API.

## Architecture

Two crates in one workspace:

- `nucleon` — tensor, Backend trait + CpuBackend, loader, tokenizer, families,
  cache, sampler, generate loop, CLI. Zero GPU code.
- `nucleon-metal` — MetalBackend implementing the same trait; `kernels/*.metal`
  one op per file.

Data flow: `loader → ModelFamily::build → generate loop
(tokenize → prefill → decode… → sample → detokenize/stream)`, with every math
op dispatched through `Backend`.

### The Backend trait (the load-bearing seam)

~15 ops: matmul, rmsnorm, rope, attention (or its primitives), silu/gelu,
softmax, embedding lookup, elementwise add/mul. A model family only speaks
Backend; a backend only implements ops. Adding hardware = one new impl.

### ModelFamily (the custom-family seam)

`trait ModelFamily`: from config.json + tensors, build the forward pass.
Each family is one readable file: embed → N blocks → head. Adding a family
never touches kernels, loader, sampler, or the generate loop.

## Milestones

| Tag | Deliverable |
|-----|-------------|
| m1-cpu-hello | Qwen3-0.6B (safetensors, f32 on CPU) generates coherent text via `nucleon run` |
| m2-metal-parity | MetalBackend passes op-parity tests vs CpuBackend; tokens/s worth reporting |
| m3-quant | GGUF loader + one quant format end-to-end |
| m4-deepseek | DeepSeek-V2-Lite: MLA cache + MoE routing |

## Testing

- Unit tests in sibling `_tests.rs` files per module (same convention as higgs).
- Op-parity tests: every backend op vs CpuBackend within tolerance.
- Golden tests: fixed prompt + greedy sampling → exact expected token ids
  (guards refactors of the whole stack).

## Journey documentation

`journal/NNNN-date-title.md` per session (including failures and dead ends);
`docs/decisions/NNN-*.md` ADRs; module READMEs written when the module is born;
milestone git tags so history is replayable.
