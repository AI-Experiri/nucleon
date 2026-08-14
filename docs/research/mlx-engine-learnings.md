# Learnings from mlx-engine (our prior engine on mlx-rs)

Extracted 2026-08-13 from `AI-Experiri/mlx-rs/mlx-engine/src`. This is the
distilled experience of building one engine already — the pipeline shapes that
worked and the bugs that bit. nucleon reuses the designs, not the code.

## Pipeline shape that worked

loader (config normalize → dispatch → strict weight apply)
→ models/<family>.rs (generic over a cache trait)
→ cache (enum of cache kinds; engine drives `Vec<Option<Cache>>`)
→ sampler (logprob-space chain: top_p → min_p → top_k; penalty preprocessor)
→ stop (text-level stop-string state machine)
→ detokenizer (full-buffer re-decode, U+FFFD withholding)
→ lm_model facade (chunked prefill + decode loop + callbacks)

Adding a family = one file + one enum arm. That rule held; keep it.

## Designs to carry over

- **First decode token is sampled from the prefill logits** — no extra forward.
- **EOS checked before detokenizing** — EOS text is never emitted.
- **Stop-string finalize flush**: at end of generation, run the detokenizer
  tail through stop-processing again so a stop completed only at flush wins.
- **Strict weight loading**: every model param present + shape-matched; every
  checkpoint tensor consumed or explicitly exempt (e.g. tied `lm_head.weight`).
- **Reject-unsupported-before-normalize**: unsupported model_type errors as
  such, not as a missing llama-family field.
- **Chunked prefill** (~2048 tokens/chunk) with a cancelable progress callback.
- **max_tokens honored before the first sample** (0 = prefill only).
- **Prompt ids bounds-checked against vocab_size** before embedding gather.
- **Stop-string processor** operates on TEXT after detokenization (detokenizer
  guarantees complete UTF-8): earliest full match wins; suffix-prefix overlap
  → withhold; empty stop strings filtered (OpenAI `stop:[""]`).
- **EOT sanitizing**: config EOS ids ∪ known EOT strings that map to real
  single tokens; per-arch override REPLACES defaults; empty set = typed error.
- **Sampler validation at construction** (temp finite ≥0, min_p ∈ [0,1], …);
  explicit RNG state, never a global PRNG.

## Pitfalls we already paid for (do not re-pay)

1. **Streaming detokenizer must never reset its buffer mid-generation** —
   SPM/Metaspace decoders strip a segment-leading space, so isolated re-decode
   drops spaces after newlines. (nucleon: use `tokenizers`' `DecodeStream`,
   which handles this.)
2. **Sample from the last position's logits every step** — letting an axis grow
   per decode step blew up only after many tokens.
3. **A `None` KV-cache slot silently recomputes from scratch** and RoPE offsets
   never advance — fill caches before the layer loop.
4. **`temp == 0.0` check is false for NaN** — NaN temperature collapses
   sampling to a constant token "successfully". Validate with `is_finite()`.
5. **`repetition_context_size == 0` means unbounded** (Python `[-0:]`), not
   disabled — bit twice.
6. **Cancellation starvation**: during a long stop-string withhold the sink
   never fires; a per-decode-token progress callback is the cancel probe.
7. **Deep JSON from clients must be depth-checked** before recursive parsing.
8. **Chat template is the single source of BOS/EOS**; `encode()` never adds
   special tokens.
9. **Never default `tie_word_embeddings` or `rope_theta` per-family** — a
   wrong tie default silently misprojects logits.
10. **Shard names in `model.safetensors.index.json` are untrusted** — allow
    only Normal/CurDir path components.
11. **Unknown enum values from checkpoint JSON return Err**, never panic —
    they arrive from data.
12. **Sliding-window configs on families that don't implement windowing must
    be rejected loudly** — silent divergence past the window otherwise.
13. **Cap `num_hidden_layers`** (10k) before building the layer stack.
14. **Dtype discipline**: every scalar op on bf16/f16 streams must not
    silently promote to f32 (mlx-rs scalar-promotion bugs, 4+ fixes). nucleon
    M1 computes in f32 so this is moot until we add bf16 compute on Metal —
    then it applies to mask fills, rope tables, sampler thresholds.
15. **Custom-freqs RoPE on Metal**: `mx.fast.rope(freqs=)` was broken on GPU —
    generalized lesson: verify every fused/fast GPU path against the naive
    reference on BOTH devices; CPU-vs-GPU divergence is real.
16. **Qwen3 = llama + per-head RMSNorm q_norm/k_norm applied after head split,
    BEFORE RoPE**; RoPE offset comes from `cache.offset()`.
17. **gemma3 ships a bogus EOS id; gpt-oss needs its EOT set REPLACED** —
    per-arch EOS/EOT sanitization is a real table, not a nicety.
