# ADR 004: GGUF is nucleon's only weights format, from day one

Date: 2026-08-14. Status: accepted (user decision). Supersedes ADR
003's two-adapter split.

## Decision

nucleon reads exactly one weights format for its entire life: GGUF.
The safetensors adapter planned for M1 is dropped before being built.
The first model nucleon loads is `Qwen/Qwen3-0.6B-GGUF`
(Qwen3-0.6B-Q8_0.gguf, the official file).

## Context

The user's argument: one format that works everywhere means the
format work is done once and never redone. GGUF qualifies: it is the
local-inference standard (llama.cpp, ollama, LM Studio, higgs), it
has a written versioned spec, quantized and full-precision weights
both fit in it, and official or community GGUFs exist for every model
this project targets (official Q8_0 for the 0.6B mule; community
bf16/4-bit for the Qwen3.8-27B flagship).

The costs, accepted knowingly:

1. Q8_0 dequantization moves from M3 into M1 (dequant-on-load to f32
   Tensors; fused dequant kernels stay a later speed step).
2. The tokenizer must be reconstructed from GGUF metadata (vocab,
   merges, token types are arrays inside the file) instead of read
   from tokenizer.json.
3. The golden test oracle compares dequantized-Q8_0 against
   dequantized-Q8_0 (HF transformers can load GGUF files), not
   byte-identical bf16. Q8_0 dequant is deterministic (scale x int),
   so token-exact matching is still the target; this must be proven,
   not assumed, when the golden test lands.

## Consequences

- loader = GGUF container parse (magic, version check, metadata
  key-values, tensor index, alignment) + Q8_0 dequant. Config numbers
  come from GGUF metadata keys, not config.json.
- tokenizer block gains a reconstruction step: GGUF metadata to an
  in-memory tokenizer.
- Facts needed before building are gathered in
  docs/research/gguf-qwen3.md (container layout, qwen3 metadata keys,
  tensor name mapping, Q8_0 block layout, oracle support), each with
  sources.
- safetensors support, if ever wanted, is a future ADR; the loader's
  adapter/core split keeps it additive.
