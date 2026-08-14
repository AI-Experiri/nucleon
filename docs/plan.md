# nucleon — overall implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> or superpowers:executing-plans, task-by-task. Steps use `- [ ]` for tracking.

**Goal:** a pure-Rust LLM inference engine that generates real tokens from
Qwen3-0.6B on CPU (M1), reaches GPU speed via our own Metal kernels (M2),
runs quantized GGUF (M3), and lands the flagship: Qwen3.8-27B, family
qwen3_5 — Gated DeltaNet hybrid layers, hybrid recurrent+KV cache (M4).
Facts: docs/research/qwen38-27b.md. DeepSeek MLA+MoE moves to the
later-families list. bf16 Metal compute is a hard M2 requirement
(27.8B does not fit 128 GB at f32).

**Architecture:** two crates — `nucleon` (tensor, Backend trait + CpuBackend,
loader, tokenizer, families, cache, sampler, generate, CLI) and
`nucleon-metal` (MetalBackend + MSL kernels). Every step is a lego block:
built, tested, journaled before the next begins.

**Tech stack:** safetensors 0.8 + memmap2, tokenizers 0.23
(default-features=false, fancy-regex), half 2.7, serde_json, objc2-metal 0.3.
Grounding facts: `docs/research/*.md`. Each step below gets its own detailed
task plan (with code) when we reach it; this document is the map.

---

## The order (and why this order)

Each step produces something runnable/testable on its own. A step never
depends on a later step.

### Step 1 — Tensor: the one data structure `[M1]`
`nucleon/src/tensor.rs`
What: `Tensor { shape: Vec<usize>, data: Vec<f32> }` + `DType` for loader-side
views; row-major, contiguous, f32-only compute in M1. Constructors, shape
helpers, pretty-printer for debugging.
Why first: every other block speaks Tensor; it must exist before anything.
Proof: unit tests on shape math + indexing.

### Step 2 — Backend trait + CpuBackend: the math `[M1]`
`nucleon/src/backend.rs`, `backend_cpu.rs`
What: `trait Backend` with the ~12 ops a transformer needs — matmul (x@W.T,
[out,in] weights), rmsnorm, silu, softmax, rope (NeoX rotate-half with offset),
elementwise add/mul, embedding lookup, argmax. CpuBackend = naive readable
loops (the frozen reference all GPU kernels are later judged against).
Why now: with ops in hand, everything above is plumbing.
Proof: TDD — each op vs tiny hand-computed cases (2x2 matmul by hand, rope at
position 0 = identity for the cos part, softmax sums to 1, …).

### Step 3 — Loader: checkpoint → tensors `[M1]`
`nucleon/src/loader.rs`
What: read config.json (serde, no per-family defaults for required fields),
mmap model.safetensors (+ index.json sharding for multi-file models), build
`name -> (dtype, shape, byte range)` map, materialize bf16→f32 Tensors.
Strict: every expected tensor present + shape-checked, unexpected = error
(tied lm_head exempt). Path-traversal guard on shard names. Layer cap 10k.
Proof: unit tests on a tiny synthetic safetensors file written by the test.

### Step 4 — Tokenizer: text ↔ ids `[M1]`
`nucleon/src/tokenizer.rs`
What: thin wrapper over `tokenizers`: load tokenizer.json, encode (never adds
specials), streaming decode via `DecodeStream::step` (withholds partial
UTF-8), token_to_id for EOS resolution. No BOS for Qwen — the chat template
is the single source of special tokens.
Proof: round-trip tests incl. an emoji split across tokens (the U+FFFD case).

### Step 5 — KV cache `[M1]`
`nucleon/src/cache.rs`
What: `KvCache` per layer: append k/v rows, expose contiguous views +
`offset()` (= RoPE position). Plain growable f32 buffers, layout
[n_kv_heads, seq, head_dim]. Designed as a trait from day one solely so M4's
MlaCache can be a second impl — no other speculation.
Proof: unit tests: offsets advance, views match appended data.

### Step 6 — ModelFamily + Qwen3 forward: the model `[M1]`
`nucleon/src/families/mod.rs`, `families/qwen3.rs`
What: `trait ModelFamily { load(config, tensors) -> Self; forward(tokens,
cache, backend) -> logits }`. qwen3.rs is "a transformer on one page":
embed → 28 × (rmsnorm → attn[q/k/v proj → per-head q/k_norm → rope →
cache → GQA attention → o_proj] → residual → rmsnorm → SwiGLU mlp →
residual) → final norm → tied lm_head. head_dim=128 explicit; facts from
docs/research/qwen3-0.6b.md.
Proof: shape-correctness unit tests with a 2-layer random mini-config; the
real correctness proof is Step 8's golden test.

### Step 7 — Sampler: logits → token `[M1]`
`nucleon/src/sampler.rs`
What: greedy argmax first; then temperature + top-k + top-p in logprob space,
repetition penalty, explicit seeded RNG. Config validated at construction
(NaN temp, min_p range — inherited pitfalls). Padding vocab rows
(151669..151935) never sampled.
Proof: unit tests with hand-built logit vectors; seeded determinism.

### Step 8 — Generate loop: the engine beats `[M1]`
`nucleon/src/generate.rs`
What: prompt ids bounds-checked → prefill (chunked; first decode token sampled
from prefill logits — no extra forward) → decode loop (sample → EOS check
BEFORE detokenize → stream text out via callback) → stop reasons
(EosToken/TokenLimit/Cancelled). Stop on both Qwen EOS ids. max_tokens=0
honored before first sample.
Proof: **the golden test** — fixed prompt, greedy, real Qwen3-0.6B weights →
exact expected token ids (generated once with HF transformers as oracle).
This single test proves steps 1-8 jointly.

### Step 9 — CLI + chat template: usable `[M1 → tag m1-cpu-hello]`
`nucleon/src/bin/nucleon.rs`, `nucleon/src/chat.rs`
What: `nucleon run --model <dir> --prompt "…" [--raw]` — ChatML rendering
(hardcoded Qwen template first; minijinja is a later step if ever), streamed
output, tokens/sec report. Integration test in `tests/` spawns the real CLI.
**Milestone: coherent text from Qwen3-0.6B on CPU.** Slow is fine (~f32 0.6B
≈ a few tokens/s) — correctness is the deliverable; speed is M2's job.

### Step 10 — Metal device layer: GPU plumbing `[M2]`
`nucleon-metal/src/device.rs`
What: objc2-metal wrapper: device, queue, runtime MSL compile (surfacing
NSError text), shared-memory buffers, dispatch helper sized from pipeline
properties (never hardcode 256). One "add two buffers" smoke kernel.
Proof: smoke test (skips gracefully when no GPU, e.g. CI).

### Step 11 — Metal kernels + MetalBackend: fast `[M2 → tag m2-metal-parity]`
`nucleon-metal/kernels/*.metal`, `src/backend_metal.rs`
What: implement Backend op-by-op in MSL (matmul first — it's ~95% of time;
then rmsnorm, rope, softmax, attention). Tensor storage moves behind the
backend (MetalBackend holds MTLBuffers; unified memory keeps copies free).
Proof: **op-parity tests vs CpuBackend** (tolerance), then the same golden
test on GPU, then a benchmark journal entry (tokens/s CPU vs GPU).

### Step 12 — GGUF + quantization `[M3 → tag m3-quant]`
`nucleon/src/loader/gguf.rs`, quant kernels
What: GGUF container parse (we know it from higgs/gguf-rs-lib), Q8_0 first
(simplest: scale+i8 blocks), then Q4_K. Dequant-on-load initially (correct,
memory-hungry), then fused dequant-matmul kernels (fast). Runs the same
files higgs serves.
Proof: parity tests quant vs f32 within tolerance; golden test on a quant model.

### Step 13 — Qwen3.8-27B: the qwen3_5 hybrid family `[M4 → tag m4-qwen38]`
`nucleon/src/families/qwen3_5.rs`, `cache` gains the recurrent+conv state
What: Gated DeltaNet blocks (delta-rule recurrent state, short conv
kernel 4) for 48 of 64 layers; gated GQA attention with partial RoPE
(0.25) at head_dim 256 for the other 16; hybrid per-layer cache; mrope
text path; MTP head and vision tower skipped. One family runs
Qwen3.5/3.6/3.8-27B. Facts: docs/research/qwen38-27b.md.
Proof: golden test vs HF oracle on a fixed prompt; memory ceiling check.
(DeepSeek-V2-Lite MLA+MoE stays a later-family candidate; its research
remains in docs/research/deepseek-v2-lite.md.)

### Step 14 — Engine API polish + higgs adapter `[M4]`
What: the `nucleon` library facade (what higgs/jigglebot embeds):
load → session → stream. Docs pass over every module README; journal
retrospective; decide what's next (server? more families? nucleon-cuda?).
The higgs integration is a thin adapter IN THE HIGGS REPO
(`higgs/src/engine/nucleon/`) implementing higgs's engine trait over this
facade — it cannot live here (it needs higgs's types; Cargo forbids the
cycle). Every public API from step 1 onward is judged by "could higgs call
this cleanly?".

---

## Standing rules (from CLAUDE.md, apply to every step)

- TDD; unit tests in sibling `_tests.rs`; integration tests in `tests/`.
- Codex convergence per step (3 stable rounds) before the step is "done".
- Journal entry per session; ADR per consequential choice; module README at
  module birth.
- fmt + clippy -D warnings + test green before "done"; commit only when asked.
