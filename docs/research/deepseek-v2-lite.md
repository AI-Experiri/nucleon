# DeepSeek-V2-Lite — exact architecture facts (M4 target model)

Verified 2026-08-13 against HF `deepseek-ai/DeepSeek-V2-Lite`: raw config.json,
model.safetensors.index.json, modeling_deepseek.py, shard-1 header.

15.7B total / ~2.4B active params. 4 BF16 shards, 31.4 GB — fits a 64 GB Mac
(weights ~29.3 GiB + ~1 GiB compressed MLA cache @32k).

## Config

hidden 2048, 27 layers, 16 heads, vocab 102400, untied embeddings,
rms_norm_eps 1e-6, silu, no biases, bos 100000, eos 100001, bf16.

**MLA:** kv_lora_rank=512, **q_lora_rank=null** (Lite has NO query
compression — plain q_proj; full V2 differs), qk_nope_head_dim=128,
qk_rope_head_dim=64 (q_head_dim=192), v_head_dim=128.

**MoE:** n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6,
moe_intermediate_size=1408, intermediate_size=10944 (dense layer 0),
first_k_dense_replace=1, routed_scaling_factor=1.0, topk_method=greedy,
scoring_func=softmax, **norm_topk_prob=false**.

Layer 0 = dense SwiGLU; layers 1..26 = MoE.

## MLA forward (inference)

- q = q_proj(h) → [16,192]; split q_nope[128] + q_pe[64].
- c = kv_a_proj_with_mqa(h) → 576; split compressed_kv[512] + k_pe[64].
  **k_pe is a single shared (MQA-style) rope key — rope once, broadcast to
  all heads.**
- kv = kv_b_proj(kv_a_layernorm(compressed_kv)) → [16, 128+128] = k_nope + v.
- RoPE only on q_pe/k_pe (theta 10000, YaRN). K per head =
  concat(k_nope[128], k_pe[64]); softmax fp32; V heads are 128-dim → o_proj.

**Cache the compressed form** (the whole point of MLA): post-layernorm latent
[512] + post-rope k_pe [64] = 576 dims/token/layer (~30.4 KiB/token total).
Decode uses weight absorption: fold kv_b's k_nope block into the query
(q_nope @ W_UK → 512-d latent query) and its v block into o_proj — attention
runs like MQA with K=576, V=512. (HF reference code caches decompressed
K/V ≈ 270 KiB/token — 9× worse; do not port it literally.)

## MoE forward

Router: Linear[64,2048] **in fp32** (weight cast + softmax fp32 — bf16 changes
top-k near ties); top-6 greedy on softmax scores; **weights are the RAW probs
(sum<1) — do NOT renormalize** (norm_topk_prob=false), × scaling 1.0;
y = Σ wᵢ·expertᵢ(x) + shared_experts(x). The 2 shared experts are ONE fused
SwiGLU MLP with intermediate 2816 (`mlp.shared_experts.*`).

## RoPE / YaRN

rope_scaling = {type:yarn, factor:40, original_max_position_embeddings:4096,
beta_fast:32, beta_slow:1, mscale:0.707, mscale_all_dim:0.707}. Because
mscale == mscale_all_dim the cos/sin tables end up UNSCALED (factor 1.0); the
entire YaRN correction lands in the softmax scale:
**softmax_scale = 192^-0.5 × (0.1·0.707·ln 40 + 1)² ≈ 0.114723** (forgetting
the mscale² factor degrades quality silently). Correction range low=10,
high=23 for these params. Config says max_position 163840; the model is
advertised/evaluated at 32k — size caches for 32k.

**Pair-layout trap:** checkpoint q_pe/k_pe weights are in INTERLEAVED
(GPT-J-style) pair layout; reference code de-interleaves
(view(d/2,2).transpose) then applies NeoX rotate_half with cat(freqs,freqs)
tables. Mixing conventions = plausible-looking wrong logits.

## Tensor names (per layer, [out,in])

`self_attn.q_proj` [3072,2048], `kv_a_proj_with_mqa` [576,2048],
`kv_a_layernorm` [512], `kv_b_proj` [4096,512], `o_proj` [2048,2048];
layer 0: `mlp.{gate,up}_proj` [10944,2048], `mlp.down_proj` [2048,10944];
MoE layers: `mlp.gate.weight` [64,2048] (router),
`mlp.experts.{0..63}.{gate,up}_proj` [1408,2048] / `.down_proj` [2048,1408],
`mlp.shared_experts.{gate,up}_proj` [2816,2048] / `.down_proj` [2048,2816].
5291 tensors total. No q_a_proj/q_a_layernorm in Lite.

## Design implications for nucleon

- Cache must be a trait/enum with a second implementation (MlaCache) — this is
  the stress test the family abstraction exists for.
- Loader must handle sharded index.json + per-layer heterogeneity (dense L0).
- num_key_value_heads=16 in config is vestigial — MLA ignores GQA entirely.
