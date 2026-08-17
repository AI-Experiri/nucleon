//! Qwen3 forward pass — ids in, logits out. Book chapter 9.
//!
//! Follows 9.8's shape annotations exactly:
//!   xn = rmsnorm(x, attn_norm)                       [seq, hidden]
//!   q  = xn @ attn_q^T  -> n_heads    x head_dim
//!   k  = xn @ attn_k^T  -> n_kv_heads x head_dim (GQA)
//!   v  = xn @ attn_v^T  -> n_kv_heads x head_dim
//!   q, k = qk_norm(q), qk_norm(k)   (per head, RMSNorm on last axis)
//!   q, k = rope(q, 0), rope(k, 0)   (offset 0: no cache yet; whole prompt each call)
//!   o  = sdpa(q, k, v, causal, scale = 1/sqrt(head_dim))
//!   x  = x + concat(o) @ attn_output^T
//!   (MLP sub-block, then residual again)
//! Repeated per block. Final RMSNorm + lm head at the end — tied
//! for 0.6B (reuses `embed`), untied for the larger sizes (uses
//! `output.weight`); the branch is `unwrap_or(&self.embed)`.
//!
//! Naive-on-purpose: no cache, so every call recomputes K/V for
//! every position. The loop chapter measures this; the cache
//! chapter (Part III) fixes it.

use nucleon_mlx::mlx_rs::fast::ScaledDotProductAttentionMask;
use nucleon_mlx::{fast, ops, silu, Array};

use crate::families::qwen3::build::Qwen3;
use crate::families::qwen3::error::FamilyError;

impl Qwen3 {
    /// Run the forward pass on a sequence of token ids.
    ///
    /// Returns logits of shape [seq, vocab]. Sampling (picking the
    /// next id from the last row) is the sampler chapter's job.
    pub fn forward(&self, ids: &[u32]) -> Result<Array, FamilyError> {
        if ids.is_empty() {
            return Err(FamilyError::Forward {
                reason: "ids is empty; the forward pass needs at least one token".to_string(),
            });
        }

        let cfg = &self.cfg;
        // Bound-check every id here: MLX's gather does no bounds
        // check (it reads src_ptr directly from the CPU backend's
        // inner loop), so an out-of-range id silently returns
        // garbage logits — or worse, reads OOB memory. Today's
        // tokenizer cannot produce >= vocab_size (its vocab_len
        // pin), but a hand-fed loop could; refuse at our door.
        let vocab = cfg.vocab_size;
        if let Some((pos, &bad)) = ids.iter().enumerate().find(|(_, &i)| i >= vocab) {
            return Err(FamilyError::Forward {
                reason: format!("id {bad} at position {pos} is >= vocab_size {vocab}"),
            });
        }
        let seq = ids.len() as i32;
        let head_dim = cfg.head_dim as i32;
        let n_heads = cfg.num_attention_heads as i32;
        let n_kv_heads = cfg.num_key_value_heads as i32;
        let q_rows = n_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let rope_base = cfg.rope_theta;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        // Embedding lookup. ids as an i32 Array along axis 0 of
        // token_embd [vocab, hidden] -> [seq, hidden].
        let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let ids_arr = Array::from_slice(&ids_i32, &[seq]);
        let mut x = self
            .embed
            .take_axis(&ids_arr, 0)
            .map_err(FamilyError::forward)?;

        for (layer, block) in self.blocks.iter().enumerate() {
            // Wrap MLX runtime errors with the block index so a
            // failure inside a 28-layer forward pass reports "layer
            // N: <mlx message>" instead of just the raw string.
            let layer_err = |e: nucleon_mlx::mlx_rs::error::Exception| FamilyError::Forward {
                reason: format!("layer {layer}: {e}"),
            };

            // ----- attention sub-block -----
            let xn = fast::rms_norm(&x, &block.attn_norm, eps).map_err(layer_err)?;

            // Q, K, V projections via x @ W^T (W is stored [out, in]).
            let attn_q_t = block.attn_q.transpose().map_err(layer_err)?;
            let attn_k_t = block.attn_k.transpose().map_err(layer_err)?;
            let attn_v_t = block.attn_v.transpose().map_err(layer_err)?;
            let q = ops::matmul(&xn, &attn_q_t).map_err(layer_err)?;
            let k = ops::matmul(&xn, &attn_k_t).map_err(layer_err)?;
            let v = ops::matmul(&xn, &attn_v_t).map_err(layer_err)?;

            // Split heads: [seq, rows] -> [seq, n_heads, head_dim]
            let q = q.reshape(&[seq, n_heads, head_dim]).map_err(layer_err)?;
            let k = k.reshape(&[seq, n_kv_heads, head_dim]).map_err(layer_err)?;
            let v = v.reshape(&[seq, n_kv_heads, head_dim]).map_err(layer_err)?;

            // QK-norm per head (RMSNorm normalizes on the last axis,
            // broadcasting the [head_dim] weight across seq and heads).
            let q = fast::rms_norm(&q, &block.attn_q_norm, eps).map_err(layer_err)?;
            let k = fast::rms_norm(&k, &block.attn_k_norm, eps).map_err(layer_err)?;

            // RoPE. offset=0 because the naive loop hands us the whole
            // prompt each call; the cache chapter will pass positions.
            let q = fast::rope(&q, head_dim, false, Some(rope_base), 1.0f32, 0, None)
                .map_err(layer_err)?;
            let k = fast::rope(&k, head_dim, false, Some(rope_base), 1.0f32, 0, None)
                .map_err(layer_err)?;

            // SDPA wants rank 4 [batch, n_heads, seq, head_dim]; we
            // run single-batch. Add a leading batch dim while
            // transposing seq and heads.
            let q = q
                .reshape(&[1, seq, n_heads, head_dim])
                .map_err(layer_err)?
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(layer_err)?;
            let k = k
                .reshape(&[1, seq, n_kv_heads, head_dim])
                .map_err(layer_err)?
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(layer_err)?;
            let v = v
                .reshape(&[1, seq, n_kv_heads, head_dim])
                .map_err(layer_err)?
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(layer_err)?;

            // MLX's SDPA handles GQA when n_heads is an integer
            // multiple of n_kv_heads (Qwen3-0.6B: 16/8 = 2; mini
            // fixture: 2/1 = 2).
            let o = fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                attn_scale,
                ScaledDotProductAttentionMask::Causal,
            )
            .map_err(layer_err)?;
            // o: [1, n_heads, seq, head_dim]

            // Drop batch, bring heads next to feature dim, flatten.
            let o = o
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(layer_err)?
                .reshape(&[seq, q_rows])
                .map_err(layer_err)?;

            // Output projection: [seq, q_rows] @ [q_rows, hidden].
            let attn_output_t = block.attn_output.transpose().map_err(layer_err)?;
            let delta = ops::matmul(&o, &attn_output_t).map_err(layer_err)?;

            x = ops::add(&x, &delta).map_err(layer_err)?;

            // ----- MLP sub-block (SwiGLU) -----
            let xn = fast::rms_norm(&x, &block.ffn_norm, eps).map_err(layer_err)?;
            let ffn_gate_t = block.ffn_gate.transpose().map_err(layer_err)?;
            let ffn_up_t = block.ffn_up.transpose().map_err(layer_err)?;
            let ffn_down_t = block.ffn_down.transpose().map_err(layer_err)?;

            let gate = ops::matmul(&xn, &ffn_gate_t).map_err(layer_err)?;
            let up = ops::matmul(&xn, &ffn_up_t).map_err(layer_err)?;
            let activated = silu(&gate).map_err(layer_err)?;
            let hidden_act = ops::multiply(&activated, &up).map_err(layer_err)?;
            let ffn_delta = ops::matmul(&hidden_act, &ffn_down_t).map_err(layer_err)?;

            x = ops::add(&x, &ffn_delta).map_err(layer_err)?;
        }

        // Final RMSNorm + lm head (tied to embedding for 0.6B).
        let x = fast::rms_norm(&x, &self.output_norm, eps).map_err(FamilyError::forward)?;
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        let head_t = head.transpose().map_err(FamilyError::forward)?;
        let logits = ops::matmul(&x, &head_t).map_err(FamilyError::forward)?;

        Ok(logits)
    }
}

#[cfg(test)]
#[path = "forward_tests.rs"]
mod tests;
