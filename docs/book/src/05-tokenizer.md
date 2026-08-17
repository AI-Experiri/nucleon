# The Tokenizer

The border between two representations. Every step of generation
that touches the user speaks strings; every step inside the engine
speaks integer ids. The tokenizer runs both directions:

- **encode**: the user's text (or the ChatML-wrapped prompt the
  chat template built) becomes a `Vec<u32>` of ids for the model.
- **decode**: the ids the sampler emits become text, streamed to
  the user one chunk at a time.

Nothing about the tokenizer is family-generic today: it is the
Qwen3 byte-level BPE tokenizer, reconstructed from what the loader
already validated and put in `Yamf`
([The Loader](04-loader.md), 7.8). When a second family arrives, the
reconstruction chooses a different pre-tokenizer regex; the rest of
this chapter stays the same.

## 8.1 What the loader already validated

By the time we get here, [The Loader](04-loader.md) has already
refused every file that would break this chapter:

- 256 byte-level base tokens present and typed `Normal` (7.7 rule);
- every merge references vocab entries that exist and are `Normal`;
- the tokenizer model is `"gpt2"` (byte-level BPE) and the
  pre-tokenizer id is `"qwen2"` — anything else refused by name;
- `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` all present, typed
  `Control`, and referenced correctly by `eos_token_id`;
- `add_bos_token` / `add_eos_token` are false (or absent) — the
  chat template owns special tokens, the tokenizer never prepends
  or appends them.

So this chapter trusts `Yamf.tokenizer` completely and builds an
in-memory tokenizer from it.

## 8.2 What byte-level BPE is, briefly

BPE (byte-pair encoding) merges the most frequent adjacent pair of
tokens into a new token, over and over, until the vocab is full. At
encode time, the same merges run in the same order, greedily. The
full walkthrough with visualizations and every tokenizer variant
(BPE, WordPiece, SentencePiece, byte-level) sits in the
[LLM Lab tokenizing chapter](https://llm-lab.bicepjai.com/llm-tokenizing/);
read that first if the algorithm is new. This chapter only teaches
the pieces specific to what nucleon actually builds.

Two properties that matter for the code:

1. **Byte-level** means the alphabet is exactly 256 characters —
   one per possible byte — so any input string can encode. Nothing
   is "out of vocabulary." The 256 characters are not the raw bytes:
   printable bytes map to themselves, and the rest get a
   sequential codepoint from `U+0100` upward (see the loader
   chapter's `byte_level_char` for the exact function). A leading
   space in a word becomes the character for byte `0x20`, which is
   the famous `Ġ`.
2. **Merges are ordered.** Rank in the merges list determines which
   pair wins when two are candidates for the same position. The
   loader stores merges as `Vec<(String, String)>` in the same order
   they appeared in the file, so rank is index.

## 8.3 From `Yamf` pieces to an in-memory tokenizer

The construction, one field at a time, using Hugging Face's
`tokenizers` crate (default-features off, `fancy-regex` feature on,
per plan; onig would drag in the C library):

<div class="diagram"><img src="diagrams/tokenizer-build.svg" alt="Yamf tokenizer pieces feed each layer of the HF Tokenizer"></div>

| Yamf field | consumed by |
|---|---|
| `tokens` (Vec<String>, 151936 entries) | `BpeBuilder::vocab_and_merges` — one side of the BPE model |
| `merges` (Vec<(String, String)>) | same builder — the other side |
| `pre` (`"qwen2"`) | selects the pre-tokenizer regex (see 8.4) |
| `token_types` (Control entries) | added-tokens registration so `<|im_start|>` etc encode atomically |
| `stop_token_ids` | passed through to `Tokenizer::stop_token_ids()` for the loop |

The three tokenizer layers we assemble:

1. **Pre-tokenizer** — splits the raw string into candidate pieces
   before BPE runs. Two stages, chained:
   - `Split` with the qwen2 regex (isolated behavior);
   - `ByteLevel` with `add_prefix_space=false`, `trim_offsets=false`,
     and `use_regex=false` (the regex is already applied by Split).
2. **Model** — `BPE` built from `(vocab, merges)`. No unk token, no
   byte-fallback, no ignore-merges: the byte-level alphabet gives
   every input a path.
3. **Decoder** — `ByteLevel` with the same flags, so the inverse
   mapping recovers the original bytes.

Added tokens are covered in 8.5 (the one thing the reconstruction
does that needs its own section).

## 8.4 The pre-tokenizer regex

The qwen2 pre-tokenizer regex was verified against Qwen's real
`tokenizer.json` and llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN2`
during the research pass. The pattern (docs/research/gguf-qwen3.md
section 7 quotes both):

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|
[^\r\n\p{L}\p{N}]?\p{L}+|
\p{N}|
 ?[^\s\p{L}\p{N}]+[\r\n]*|
\s*[\r\n]+|
\s+(?!\S)|
\s+
```

Six alternatives, in order: contractions like `'s` / `'t`; a run of
letters possibly preceded by one non-letter/digit character; a run
of digits; a run of symbols possibly preceded by one space, with
trailing newlines; blank lines; trailing whitespace; any remaining
whitespace.

Two Rust-specific notes:

- Every one of these alternatives needs Unicode property support
  (`\p{L}` for letters, `\p{N}` for digits), and the last one has
  a lookahead `(?!\S)`. The default `regex` crate cannot do
  lookaheads. `fancy-regex` (pure Rust) can, and is our chosen
  backend.
- The tokenizers crate default features are `["progressbar", "onig",
  "esaxx_fast"]`; `onig` binds the C library Oniguruma. Depend on
  it with `default-features = false, features = ["fancy-regex"]`.

## 8.5 Special tokens

A **special token** is a vocabulary entry whose string form is a
marker the model was trained to recognize as structure, not as
literal text. Qwen3 has three that matter now, four more that show
up under tools/thinking modes, and a `token_type` on each so the
tokenizer knows how to treat them:

| id | string | token_type | what it means |
|---|---|---|---|
| 151643 | `<\|endoftext\|>` | Control | end of the whole generation (stop the loop) |
| 151644 | `<\|im_start\|>` | Control | ChatML: a message begins |
| 151645 | `<\|im_end\|>` | Control | ChatML: this message ended (also stops decode of assistant turn) |
| 151657 | `<tool_call>` | UserDefined | tools mode: model requests a tool call |
| 151658 | `</tool_call>` | UserDefined | tools mode: end of the request |
| 151667 | `<think>` | UserDefined | thinking mode: model's private scratch space starts |
| 151668 | `</think>` | UserDefined | thinking mode: end of scratch |

Two things about specials that trip everyone up. Both are shown
side by side in the diagram below.

**They must encode as ONE id.** The string `<|im_start|>` is nine
characters. If we hand it to plain BPE without telling the tokenizer
about it, BPE splits it into whatever byte-level merges cover
`<`, `|`, `im`, `_`, `start`, `|`, `>` — several ids, none of them
151644. The model was trained expecting id 151644, so a decomposed
form is *not the same input* — it's ordinary text that happens to
look like the marker. Registering the string as an "added token"
makes the tokenizer match it whole and emit exactly one id.

<div class="diagram"><img src="diagrams/tokenizer-specials.svg" alt="unregistered specials become BPE decomposition; registered specials encode to one id"></div>

**Encode does not add them for us.** The tokenizers crate has a
convenience where `encode(text, add_special_tokens=true)` sprinkles
BOS at the start / EOS at the end automatically. We turn that OFF:
`encode(text, add_special_tokens=false)`. Why: the **chat template**
already built the specials into the string. If the tokenizer also
inserted them, every one doubles. Qwen3 explicitly does not want a
BOS at all; `add_bos_token=false` in the metadata (the loader gates
on it). Concrete example:

```text
one turn of chat, before the tokenizer sees anything:

  messages = [{"role": "user", "content": "Why is the sky blue?"}]

the chat template renders that to a plain string:

  <|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n

that string is what encode() gets. Every <|im_start|>, <|im_end|>
in it is one id (added-token match); the rest goes through byte-level
BPE like any other prose.
```

If the tokenizer added its OWN BOS on top of this, the model would
see `<|im_start|><|im_start|>user...` — two openers, one from us and
one from the template, and the model was never trained on that.

Where each rule is enforced:

| the rule | enforced where |
|---|---|
| `<\|im_start\|>` encodes as one id, not seven | the reconstruction registers every Control/UserDefined vocab entry as an added token (8.3) |
| encode does not sprinkle BOS/EOS | our `encode()` passes `add_special_tokens=false` (8.6) |
| add_bos_token / add_eos_token are false in the file | the loader refuses any file that sets them true (7.7) |
| the ChatML wrapping is correct | the loader's render smoke test in the gate (7.7) checked both roles and both markers in order on a real render |

## 8.6 Encode: never adds specials

The rule restated in code terms (see 8.5 for why):
`tokenizer.encode(text, /*add_specials=*/ false)`. Our `encode`
wraps that and hides the boolean, so callers cannot accidentally
turn specials back on.

<div class="diagram"><img src="diagrams/tokenizer-encode.svg" alt="text through pre-tokenizer, byte-level mapping, BPE merges, then ids"></div>

Two encoding cases the tests pin explicitly:

1. Raw text that ends inside a UTF-8 codepoint: byte-level BPE
   handles it because every byte has a token, but the boundary
   still needs care on decode (8.7).
2. A ChatML-wrapped prompt containing `<|im_start|>user\n...`: the
   `<|im_start|>` must encode as one id (its added-token id 151644),
   not as the BPE decomposition of the literal string. This is the
   "one id, not seven" case from 8.5, in a test.

## 8.7 Streaming decode without broken UTF-8

Generation emits one token id per step. If we decode each id in
isolation and print, a multi-byte codepoint (an emoji, a CJK
character) whose UTF-8 bytes fall across two token boundaries
prints as `U+FFFD` (the replacement character) followed by the
missing continuation. The user sees garbage.

`tokenizers::DecodeStream` solves this: it buffers bytes that do
not yet form a valid UTF-8 codepoint, waits for the next token,
and emits nothing until the buffer is a clean string.

<div class="diagram"><img src="diagrams/tokenizer-stream-decode.svg" alt="four-byte emoji split across two token decodes; DecodeStream withholds until valid UTF-8"></div>

The API we expose: `Tokenizer::stream_decoder(&self) -> DecodeStream`,
then `stream.step(id) -> Result<Option<String>, _>` — `None` while
buffering, `Some(chunk)` when a valid piece emerges. The generation
loop calls this in the sample-then-print inner loop.

## 8.8 The stop set, passed through

The loader assembled `stop_token_ids` for us (7.3 landmine 2:
`eos_token_id` alone under-reports; `<|endoftext|>` must be added).
The tokenizer chapter does nothing with it beyond passing it
through — `Tokenizer::stop_token_ids() -> &[u32]` returns exactly
what `Yamf.tokenizer.stop_token_ids` contained. Stopping is the
loop's job.

## 8.9 The API

```rust
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,     // HF crate, wrapped
    stops: Vec<u32>,                  // pass-through of Yamf's stop set
}

pub fn from_yamf(y: &Yamf) -> Result<Tokenizer, TokenizerError>;

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
    pub fn stream_decoder(&self) -> DecodeStream;
    pub fn stop_token_ids(&self) -> &[u32];
    pub fn vocab_size(&self) -> usize;
}
```

> **Rust: `String` vs `&str`.** `String` owns its bytes (heap,
> growable); `&str` borrows a stretch of UTF-8 someone else owns.
> Function params take `&str` unless the function needs to keep
> the value; returns hand back `String` when the value was built
> here. Sibling: `Cow<str>` when the return might sometimes borrow
> and sometimes own. [The Book, ch. 8.2](https://doc.rust-lang.org/book/ch08-02-strings.html)

Module layout, one concern per file:

| file | concern |
|---|---|
| tokenizer/mod.rs | export barrel |
| tokenizer/build.rs | from_yamf: Yamf pieces to HF Tokenizer |
| tokenizer/encode.rs | thin encode wrapper (never adds specials) |
| tokenizer/decode.rs | full decode + `stream_decoder` |
| tokenizer/error.rs | `TokenizerError` |

## 8.10 What the tests will pin

All on the real Qwen3 tokenizer data the loader gives us (loaded
from a tiny GGUF fixture, same builder pattern as the loader
tests):

1. round-trip on ASCII: `encode` then `decode` returns the input;
2. round-trip on latin-1 (accented letters): same;
3. round-trip on emoji: a four-byte emoji encodes and decodes back;
4. **emoji split across tokens streams correctly** — the failure
   this whole chapter exists to prevent: feed each id one at a
   time through `stream_decoder`, concatenate emitted chunks,
   must equal the original string with no `U+FFFD`;
5. `<|im_start|>` encodes to exactly one id (added-token, not BPE);
6. encode of a bare prompt does NOT contain the eos or endoftext id
   (never adds specials, no matter what);
7. `stop_token_ids()` returns the exact set the loader assembled;
8. round-trip on the full ChatML-wrapped prompt the chat template
   would produce for `[{role: user, content: "Why is the sky blue?"}]`
   preserves every marker as a single id.

The golden test that proves the whole encode/decode path against HF
transformers lives in the loop chapter, not here — it's an
end-to-end test, and the tokenizer is one thing it checks.

## 8.11 Upcoming tokenizer topics

Visible from here, scheduled later:

1. Tool-call parsing on decode: recognizing when the model emits
   `<tool_call>...</tool_call>` and handing the JSON to the caller
   before the raw text lands in the user's terminal.
2. A second family's tokenizer: different `pre` id → different
   regex, but the same byte-level BPE plumbing. When it lands, the
   pre-tokenizer selection moves from a hardcoded regex constant to
   a pre-id-keyed match.
3. Speculative decoding needs a fast "encode a single token"
   path; a follow-up if we ever adopt it.

Next: [Qwen3](07-qwen3.md), the family that turns ids into logits.
