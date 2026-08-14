# 0004 — Metal ops study: all tensor ops on GPU, fusion measured (2026-08-14)

## Plan change (user-driven, the right call)

The written plan said CPU engine first, Metal at M2. The user pushed back
twice: first for per-op CPU+Metal pairs, then sharper — understand Metal
mechanics and fusion BEFORE designing the Backend trait, because kernel
granularity determines what can ever be fused behind the seam. Accepted:
backend trait design is deferred until after this study.

## What was built

- nucleon-metal wired to objc2/objc2-metal/objc2-foundation (0.6/0.3.2).
- kernels/ops.metal: all 11 kernels in one file — add, mul, silu, embed,
  matvec, matmul, rope (NeoX half-split), rmsnorm, softmax, argmax
  (tie -> lowest index), plus fused rmsnorm_matvec.
- device.rs: Gpu (compile-at-startup pipelines, shared buffers, Scalar
  args via setBytes, run = single dispatch + wait, run_many = batched
  encodes + one wait). CoreGraphics link needed for
  MTLCreateSystemDefaultDevice — research note was right.
- ops.rs: MetalOps safe wrappers, slices in / Vec out.
- 15 parity tests vs inline Rust reference loops, 1e-5 relative tolerance,
  graceful skip when no GPU. All passed on first GPU run.

## The fusion numbers (M-series, dim 1024 -> 2048, 500 steps)

- composed, sync per op: 505.4 us/step
- composed, batched (one command buffer): 92.3 us/step  -> sync cost 5.5x
- fused v1 (naive, per-thread redundant sum_sq): 134.1 us/step — fusion
  LOST (0.69x). Wrong-turn recorded on purpose: fusion is a trade, and
  v1's redundant arithmetic exceeded the saved dispatch + memory traffic.
- fused v2 (cooperative sum_sq per threadgroup + factoring inv_rms out of
  the dot product by linearity): 80.7 us/step -> 1.14x over composed.

Design consequence for the backend seam: batching dispatches matters more
than fusion at decode shapes; the trait must allow whole-forward-pass
encoding and fused kernels behind it. This decides the chapter-2 design
discussion's terms.

## Codex convergence (11+ rounds — the module deserved it)

Real defects caught and fixed, best first:
- r1: unsound safe reads (slice from raw ptr, caller-chosen len, no
  capacity check); empty-slice dangling-pointer copy; embed unvalidated
  (bad token id = GPU OOB read); command-buffer status never checked
  (shader fault = silently reading garbage); tests skipped on compile
  errors, which would have turned a broken ops.metal green.
- r2: the same overflow class fixed in the tensor module (r1 there)
  reappeared in my new read-bounds check — len*size wraps before compare.
  Raw dispatch made `unsafe` with a real safety contract (GPU OOB into
  shared memory = process memory corruption).
- r3: rope divided by zero before validating head_dim; missing 32-bit
  guards on map dispatch paths.
- r4: dispatchThreads is UB on GPUs without nonuniform-threadgroup
  support — Gpu::new now requires Apple4+ family.
- r6: soundness hole — a foreign private-storage MTLBuffer's contents()
  is NULL; objc2 models NonNull; safe read would be UB. Storage-mode
  check added.
- r7: caught that my r5 patch had SILENTLY FAILED to apply (python
  str.replace no-ops on non-match after cargo fmt reformatting). Process
  lesson: every scripted patch now asserts the pattern matched.
- r10: my own r9 NaN-policy doc was wrong at the -inf edge (the scan
  sentinel means literal -inf can never win; samplers do use -inf masks).

Recurring dismissals (documented, not fixed — study-grade contracts):
fused kernel finite-range divergence, logical-length buffer wrappers
(deferred to the real MetalBackend), scalar domain checks on
trusted-config values.

## State

Workspace: 47 tests green (21 tensor + 26 metal), fmt/clippy clean.
Fusion numbers after de-biasing: sync 5.8x, fusion 1.15x (v2). Book
chapter 10 carries the measured numbers. Convergence: rounds 11+ running
(stable rounds accumulating; only re-stated dismissals and doc nits).
