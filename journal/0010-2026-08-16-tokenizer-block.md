# 0010 — 2026-08-16 — the tokenizer block

## What happened

Chapter 8 written first (design-in-book rule), then rewritten twice
on user feedback before any code existed:

1. First draft explained things in pipeline order with a test list
   at the end. User: "I don't get the specials and the examples are
   not clear" — specials became their own section with the real
   Qwen3 id table, a ChatML render example, and a new diagram
   (unregistered BPE decomposition vs registered one-id).
2. User: "review tokenizer section where things are explained in
   increasing order of complexity. Always present parts then
   explain each of them" — restructured to an anatomy-first shape:
   8.1 presents seven parts (diagram + table), then one section per
   part sorted simplest (stop set) to hardest (specials), then
   assembly, then the runtime paths.
3. User: "why are we discussing tests in the chapter?" — the test
   enumeration was build bookkeeping; deleted. Replaced by the
   closing the user asked for instead: 8.11 maps every part to its
   exact tokenizers-crate item.
4. Section titles briefly read "8.2 Part 7", "8.3 Part 3" (parts
   numbered by pipeline, explained by complexity). User caught it;
   titles went to plain names, numbers stay only in the diagram.

Build-vs-buy: I laid out write-our-own-BPE (with the HF crate as
oracle) against glue-over-the-crate. User picked the crate: "the
learning is making the engine... I want to spend time on kernels
improvement later." ADR 006.

## Mid-block policy change (user)

"3 review rounds coming clean from independent agents using opus
4.7 or codex; along with integration test up to 85% and unit test
up to 90% coverage." Written into CLAUDE.md. Codex turned out to be
out of credits until Aug 20, so all rounds ran as fresh Opus
subagents, one per round, none seeing the previous rounds' context
beyond a factual already-fixed list.

## The code

`nucleon/src/tokenizer/`: build.rs (from_yamf + the struct),
encode.rs, decode.rs (+ DecodeStream wrapper), error.rs, one
sibling _tests.rs each. 512 lines of glue, ~30 tests.

## Review rounds (the block's best material)

Nine rounds so far; every finding validated by reading the crate
source before fixing; every code fix revert-proof-verified
(actually reverted, watched the named test fail, restored).

- r1 (found 6): NFC normalizer MISSING — the reference
  tokenizer.json carries `"normalizer": {"type": "NFC"}`, GGUF has
  no field for it, llama.cpp skips it; without it decomposed input
  (macOS!) encodes to ids the model never saw. Also: everything
  registered special=true where the reference marks <think>/
  <tool_call> special=false; the fixture couldn't detect a missing
  Split stage at all (agent deleted it, all tests stayed green);
  no stream flush (max-token stop mid-emoji silently loses the
  tail); duplicate test.
- r2 (found 5): UserDefined registration completely unguarded
  (deleting it from the filter left every test green); use_regex
  flag unverified (flipping it changed nothing detectable);
  vocab_size() cloned the whole 151k-entry vocab per call;
  normalizer-ordering comment unenforced; "only Control is
  special" comment overclaimed (GGUF types fim/repo markers
  Control, diverging from the reference — accepted, documented).
- r3: CLEAN. Notes: Unknown missing from the filter (llama.cpp
  caches CONTROL|USER_DEFINED|UNKNOWN), stale fixture comments.
- r4 (found 2): fancy-regex's backtrack limit trips at exactly
  999,999 whitespace chars and the tokenizers crate SWALLOWS the
  error (`Some(Err(_)) => None`) — the whole input becomes one
  unsplit piece, wrong ids, Ok returned. Fixed with a 512 KiB
  encode cap (1.9x margin, measured). The Unknown filter arm was
  not revert-proof (no type-2 fixture row existed).
- r5 (found 2): the `pre` field was validated by the loader and
  read by NOBODY — from_yamf hardcoded the qwen2 regex; a second
  family's pre would silently get the wrong split. Now matched,
  refuses by name. README's "no onig" claim was false via the
  candle dev-dep (its own tokenizers 0.22 + onig compiles in test
  builds); narrowed to production-tree-only.
- r6 (found 3): merge order = rank was load-bearing and untested
  (reversing merges changed ".ab" ids, everything green); the
  added-token zip silently truncated on a short token_types; the
  special flag had no through-the-crate pin.
- r7 (found 1): duplicate token strings silently shadow ids
  (later index wins the vocab map, earlier id decodes to "").
- r8: CLEAN.
- r9 (found 6, all doc/edge): finish() existed in code but in no
  documentation (book, README) — the exact failure it prevents
  would have been re-created by a loop written from the chapter;
  #[must_use] added. Byte-alphabet completeness now re-checked at
  from_yamf's border (a missing char is DROPPED by unk-less BPE
  and neighbors merge across the hole). Untruncated echo of the
  pre field. error.rs doc drift. Book drift (Unknown, vocab_len,
  the cap). ADR 002's consequence line was stale; no journal
  entry existed (this one).

Also switched encode → encode_fast (offsets/word-ids computed and
discarded otherwise).

## Fixture archaeology

mini() grew from 265 to 269 vocab entries across the rounds — every
addition a discriminating probe some reviewer proved was missing:
"76" + merge "7 6" (Split observability: digits split before
merges), ".a" + merge ". a" (use_regex observability: GPT-2 regex
would cut the boundary), "<think>" type 4 (UserDefined
registration), "<unk>" type 2 (Unknown registration). Each ripple
meant vocab-count updates in ~14 places; the counts are
fixture-coupled by design (embd shape = [vocab, 32]).

## State at close

103 tests, quality gate green, tokenizer files at 100% line
coverage. Convergence counter: r8 clean, r9 found items → 0 again;
next session resumes at round 10. Loader-wide coverage (84% lines,
error.rs at 45%) remains a follow-up against the new 90% unit gate.
