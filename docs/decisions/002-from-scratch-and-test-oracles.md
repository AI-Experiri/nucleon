# ADR 002 — Build ops from scratch; test against candle; benchmark against llama.cpp/MLX

Date: 2026-08-14. Status: accepted.

## Decision

1. Tensor and the op set are written from scratch, not taken from candle or
   ndarray. The ops a transformer needs are ~12 short loops (~400 lines);
   they are the curriculum, and the expensive work (Metal kernels, quant,
   correctness plumbing) is identical under any option. The Backend trait
   is the seam where a library could still slot in later if this proves
   wrong.
2. Op correctness oracle: candle-core as a DEV-DEPENDENCY only (pure Rust,
   in-process, can fuzz random shapes). Plus hand-computed cases and math
   properties (softmax sums to 1, rope preserves pair norms, rmsnorm unit
   RMS). tch-rs was rejected: LibTorch download + C++ toolchain in every
   dev environment to verify naive loops. Per-op PyTorch fixture files were
   rejected as workflow pain.
3. End-to-end correctness oracle: HF transformers, used once per golden
   test to record exact greedy token ids for a fixed prompt.
4. Speed references (M2+): llama.cpp (llama-bench), MLX (mlx-lm), candle,
   mistral.rs on the same machine/model/quant; roofline (bytes moved /
   bandwidth) per kernel; results recorded in journal per milestone via
   the existing engine-bench harness.

## Consequences

Production nucleon stays pure Rust with zero ML dependencies; candle
appears only in test builds. If fuzz-vs-candle and the golden test
disagree, transformers is the tie-breaker.
