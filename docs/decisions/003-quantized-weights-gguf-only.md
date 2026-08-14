# ADR 003: quantized weights come from GGUF only

Date: 2026-08-14. Status: accepted (user decision).

## Decision

nucleon reads quantized weights from exactly one container: GGUF.
Quantized-safetensors repos (GPTQ, AWQ, and similar
transformers-world methods) are out of scope, permanently unless
revisited by ADR.

Full-precision weights keep coming from safetensors (the official
release artifact, the golden-test baseline). So the loader ends with
exactly two format adapters, one per purpose:

| purpose | format | why this one |
|---|---|---|
| full precision (correctness, oracle parity) | safetensors | what labs officially release; identical bytes to what HF transformers runs |
| quantized (speed, memory) | GGUF | the de facto local-inference standard; versioned written spec; one adapter covers every quant type inside |

## Why GGUF for quantized

1. Works everywhere local inference happens (llama.cpp, ollama,
   LM Studio); higgs already serves these exact files.
2. Self-describing and versioned: one spec, refuseable on version
   mismatch.
3. Quant variety lives INSIDE the container as tensor types (Q8_0,
   Q4_K, ...): supporting more quants never means another format
   adapter. Plan: Q8_0 first, then Q4_K.
4. Official Qwen GGUFs exist for the dense family
   (Qwen/Qwen3-0.6B-GGUF); community GGUFs cover Qwen3.8-27B.

## Consequence

The loader's format surface is frozen at two adapters. Anything
shipped only as GPTQ/AWQ is answered with "convert to GGUF", not with
a third adapter.
