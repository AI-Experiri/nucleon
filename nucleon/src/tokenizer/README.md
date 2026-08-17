# tokenizer

Text to ids and back. Glue around the HF `tokenizers` crate
(default-features off, `fancy-regex` on — the production dependency
tree stays free of the onig C library; the candle-core DEV-dependency
oracle pulls its own tokenizers 0.22 with onig, test builds only):
the engine's learning time goes to kernels, not to rebuilding BPE
(user decision, book chapter 8).

## What it does

`from_yamf(&Yamf)` assembles an in-memory tokenizer from the
fields the loader already validated — vocab + merges into a BPE
model, the qwen2 regex + byte-level stage as pre-tokenizer,
byte-level decoder, every Control/UserDefined token registered so
specials match whole (one id, never a BPE decomposition).

Runtime surface, all specials-safe by construction (the two HF
booleans are hard-wired false and hidden):

- `encode(&str) -> Vec<u32>` — never adds BOS/EOS; the chat
  template owns specials.
- `decode(&[u32]) -> String` — keeps markers.
- `stream_decoder() -> DecodeStream` — per-generation UTF-8
  buffering; `step(id)` yields `None` until a codepoint completes,
  so a split emoji never prints U+FFFD. End every stream with
  `finish()`: it drains a tail truncated mid-codepoint (lossily,
  visible U+FFFD) that would otherwise vanish.
- `stop_token_ids() -> &[u32]` — the loader's stop set, verbatim.

## How to read it

One concern per file: `build.rs` (construction + the struct),
`encode.rs`, `decode.rs` (+ `DecodeStream`), `error.rs`. Sibling
`_tests.rs` each. The book chapter's 8.11 table maps each part to
the crate item used here.
