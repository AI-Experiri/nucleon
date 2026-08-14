# Qwen3.8-27B — the flagship target (family: qwen3_5)

Verified 2026-08-14 against HF `Qwen/Qwen3.8-27B` (config.json, model
card, HF API). Released in the 2026-08-13/14 rollout. Apache 2.0.
~27.8B params, BF16, 55.6 GB in 18 shards. The same `qwen3_5`
architecture covers Qwen3.5-27B (Feb 2026) and Qwen3.6-27B (Apr 2026):
one family implementation runs all three generations.

## Architecture (model_type "qwen3_5", Qwen3_5ForConditionalGeneration)

- 64 text layers in a 3:1 hybrid pattern (`full_attention_interval: 4`):
  16 repetitions of [3 x Gated DeltaNet -> FFN, 1 x Gated Attention ->
  FFN]. Dense FFN (intermediate 17408), no MoE.
- Full-attention layers (16 of 64): GQA 24 query / 4 KV heads, head_dim
  256, output gate (`attn_output_gate: true`), partial RoPE
  (`partial_rotary_factor: 0.25` — only a quarter of head dims rotate),
  rope_theta 1e7, interleaved mrope sections [11, 11, 10].
- Linear-attention layers (48 of 64): Gated DeltaNet with 16 QK heads /
  48 V heads, head_dim 128, short conv kernel 4. Cache is a recurrent
  state + conv window, NOT a KV cache.
- hidden 5120; vocab 248320; context 262144 native (1M extensible).
- MTP head (`mtp_num_hidden_layers: 1`) — optional fast decode,
  skippable for correctness.
- 27-layer vision encoder (hidden 1152, patch 16) — skippable for
  text-only generation.

## GGUF

No official Qwen GGUF repo. Community: unsloth/Qwen3.8-27B-GGUF
(IQ2_XXS 9.0 GB up to 16-bit), plus many derivatives. ~17 GB at 4-bit.

## Memory on a 128 GB M3 Max

- f32: ~111 GB — not workable. bf16: ~56 GB — comfortable (needs bf16
  compute on Metal). 4-bit GGUF: ~17 GB.

## What nucleon needs that dense qwen3 does not

1. Gated DeltaNet block (delta-rule recurrent state + short conv) and
   its kernels.
2. Hybrid cache: recurrent+conv state for 48 layers, KV for 16 — the
   cache trait's second implementation.
3. Gated attention with partial RoPE at head_dim 256.
4. mrope position handling (text-only path first).
5. bf16 compute to fit the weights at all.

## Sources

https://huggingface.co/Qwen/Qwen3.8-27B
https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
https://huggingface.co/api/models/Qwen/Qwen3.8-27B
https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
https://www.yottalabs.ai/post/qwen-3-8-27b-specs-hardware-requirements-how-to-run-2026
