# 0003 — Tensor chapter built and converged (2026-08-14)

## Decisions settled before code

- From-scratch ops confirmed over candle/ndarray after real pushback
  ("reinventing tensor arithmetic seems extreme"). The counter that landed:
  we build ~12 fixed loops (~400 lines), not a tensor library; llama2.c is
  700 lines total. ADR 002.
- Test oracle debate: user rejected PyTorch fixture files ("comparing
  python output to rust is pain") and tch-rs was rejected for LibTorch
  weight; settled on candle-core as dev-dependency (pure Rust, in-process).
  transformers stays the end-to-end golden-token oracle; llama.cpp/MLX are
  speed bars and tie-breakers only.
- Writing style reset: user called the book prose "LLM slop." Added
  tropes.fyi guide + plain-technical-register rule to CLAUDE.md as hard
  rules; rewrote chapter 0, intro, README; turned off mdBook smart quotes.

## The build

TDD: 8 failing tests first (at/row/from_fn/Display), then implementation.
Clippy's module_inception lint allowed crate-wide (folder-per-chapter
layout is deliberate).

## Codex convergence: 6 rounds

- r1: shape-product overflow (real, High — release-mode wrap accepts lying
  shapes); PartialEq-on-f32 trap (real); barrel leaking inner module
  (real); test gaps. Fixed; overflow fix proven by revert-run-restore.
- r2: checked helper not reusable by loader (took: pub(crate) split);
  FnMut over Fn (took); Display ambiguity scalar/[1]/[1,1] (took: shape
  header); huge-allocation guard (dismissed to loader, documented).
- r3: caught that r2's pub(crate) helper was UNREACHABLE (private module,
  barrel didn't re-export) — the fix to a review fix was itself broken.
  Reset stability count. Lesson recorded: a reviewer verifying the
  previous round's fix is the protocol working, not overhead.
- r4-r6: stable (only already-assessed trust-boundary items + doc/test
  polish, which we took: # Panics sections, exact Display assertions,
  zero-axis convention tests).

Final: 21 tests, fmt/clippy/test green in debug and release.

## Open items carried forward

- Chapter 3 loader checklist: byte-length arithmetic (count x dtype size,
  checked), zero-dim policy, max-rank/max-dim/max-size guards, try_new
  question at the facade (step 14).
