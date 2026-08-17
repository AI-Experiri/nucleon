# ADR 006: tokenizer via the HF tokenizers crate

Date: 2026-08-16. Status: accepted (user decision).
Amends ADR 002's "zero ML dependencies in production" consequence.

## Decision

nucleon's tokenizer is glue over Hugging Face's `tokenizers` crate
(0.23, `default-features = false`, `features = ["fancy-regex"]`) —
a production dependency, not dev-only. We do not write our own BPE.

User's words: "the learning is making the engine, I don't want to
spend time on tokenizer, I want to spend time on kernels
improvement later."

## Why

A byte-level BPE encoder is small and teachable, which made
write-our-own genuinely defensible for a learning project. The
deciding argument was time allocation: the project's depth budget
goes to the engine and later to custom Metal kernels, and the HF
crate is the reference implementation the labs themselves ship.
The from-scratch option was considered with the crate as test
oracle; rejected to keep the block small.

## Consequences

- `tokenizers` joins the production dependency tree. With
  default features off and `fancy-regex` on, the production tree
  stays free of C (`onig` is what the default features would pull).
  The candle-core DEV-dependency oracle still pulls its own
  tokenizers 0.22 with onig — test builds only.
- ADR 002's consequence line "production nucleon stays pure Rust
  with zero ML dependencies" is superseded on the tokenizer front
  (as ADR 005 already superseded it for compute).
- What we still own and test: the Yamf-to-tokenizer construction
  (NFC pin, qwen2 regex selection by pre id, added-token flags),
  the two hard-wired booleans (never add specials on encode, never
  skip them on decode), the 512 KiB encode bound, the stream
  `finish()` drain, and the stop-set pass-through.
