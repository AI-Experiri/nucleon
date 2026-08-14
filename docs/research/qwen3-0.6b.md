# Qwen3-0.6B — exact architecture facts (M1 target model)

Verified 2026-08-13 against HF `Qwen/Qwen3-0.6B` (sha c1899de): raw
config.json, generation_config.json, tokenizer_config.json, and the parsed
safetensors header.

## config.json (verbatim values that matter)

| field | value |
|---|---|
| model_type | "qwen3" |
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (GQA, 2× repeat) |
| **head_dim** | **128 — explicit field, NOT hidden/heads (=64)!** |
| intermediate_size | 3072 |
| vocab_size | 151936 |
| rope_theta | 1000000 |
| rope_scaling | null |
| rms_norm_eps | 1e-06 |
| tie_word_embeddings | true |
| max_position_embeddings | 40960 |
| attention_bias | false |
| hidden_act | "silu" |
| eos_token_id | 151645 |
| torch_dtype | "bfloat16" |

## Tensors (single `model.safetensors`, 311 tensors, ALL BF16, [out,in] order)

- `model.embed_tokens.weight` [151936,1024]; `lm_head.weight` [151936,1024]
  (physically present AND byte-identical to embed — tied; do not treat its
  presence as evidence of untied heads); `model.norm.weight` [1024]
- per layer N in 0..27 (`model.layers.N.`):
  - `self_attn.q_proj.weight` [2048,1024] (16×128), `k_proj` [1024,1024]
    (8×128), `v_proj` [1024,1024], `o_proj` [1024,2048]
  - `self_attn.q_norm.weight` [128], `k_norm.weight` [128] — per-head-dim,
    shared across heads
  - `input_layernorm.weight` [1024], `post_attention_layernorm.weight` [1024]
  - `mlp.gate_proj` [3072,1024], `mlp.up_proj` [3072,1024],
    `mlp.down_proj` [1024,3072]
- **No bias tensors exist anywhere.**

## Forward pass (transformers order, exact)

Per layer: `h = x + attn(rmsnorm_in(x))`; `out = h + mlp(rmsnorm_post(h))`;
mlp = `down(silu(gate(x)) * up(x))`.

Attention: project → reshape to (…, n_heads, 128) → **q_norm/k_norm (RMSNorm
over head_dim, BEFORE RoPE; v gets no norm)** → RoPE (rotate-half/NeoX,
theta 1e6, position offset = tokens already in cache) → cache append →
GQA attention (scale = 128^-0.5) → reshape → o_proj.

Logits = `hidden @ embed_tokens.weight.T` (tied).

## Tokenizer / template / stopping

- `tokenizer.json` (byte-level BPE, Qwen2 class) is sufficient alone.
- **No BOS is prepended, ever** (bos_token null; prepending changes outputs).
- ChatML template: `<|im_start|>{role}\n{content}<|im_end|>\n`, generation
  prompt ends `<|im_start|>assistant\n`. Thinking: `<think>`(151667) /
  `</think>`(151668); enable_thinking=false inserts empty think block.
- **Stop on BOTH** 151645 `<|im_end|>` and 151643 `<|endoftext|>`
  (generation_config eos_token_id is the list [151645, 151643]).
- Default sampling (generation_config): temp 0.6, top_k 20, top_p 0.95.
- Vocab rows 151669..151935 are untrained padding — never sample them.

## Pitfall summary

1. head_dim=128 explicit; projection widths follow n_heads*head_dim.
2. q/k_norm: after head split, before RoPE, weights shape [128].
3. No BOS. 4. Stop on both EOS ids. 5. [out,in] weight order (transpose for
x@W.T-style matvec). 6. eps 1e-6, theta 1e6 (not llama defaults).
