// All tensor ops in Metal Shading Language, one file. Each kernel is the
// body of one loop iteration; the thread grid replaces the loop header.
//
// Two kinds of kernel live here:
//  - map kernels (add, mul, silu, embed, matvec, matmul, rope): each thread
//    writes one output element and never talks to another thread.
//  - reduction kernels (rmsnorm, softmax, argmax): threads must cooperate
//    (a sum or max over a whole row), so they share a threadgroup-memory
//    scratch array and synchronize with barriers.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- map ops

kernel void add(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* out     [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

kernel void mul(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* out     [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * b[i];
}

// silu(x) = x * sigmoid(x), the activation inside SwiGLU MLPs
kernel void silu(device const float* x [[buffer(0)]],
                 device float* out     [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = v / (1.0f + exp(-v));
}

// embedding lookup: out = table[id], one thread per copied element
kernel void embed(device const float* table [[buffer(0)]],
                  device const uint* id     [[buffer(1)]],
                  device float* out         [[buffer(2)]],
                  constant uint& dim        [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    out[i] = table[id[0] * dim + i];
}

// y = W @ x with W stored [out_dim, in_dim] (checkpoint order).
// One thread per output row: a dot product. This is THE decode-time op.
kernel void matvec(device const float* w [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y       [[buffer(2)]],
                   constant uint& in_dim [[buffer(3)]],
                   uint row [[thread_position_in_grid]]) {
    float sum = 0.0f;
    for (uint j = 0; j < in_dim; j++) {
        sum += w[row * in_dim + j] * x[j];
    }
    y[row] = sum;
}

// C = A @ B^T with A [m, k], B [n, k] (so B rows are dot-ready like the
// checkpoint's [out, in] weights). One thread per output element, 2-D grid.
// Naive on purpose for the study; the tuned version is chapter 11's job.
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c       [[buffer(2)]],
                   constant uint& k      [[buffer(3)]],
                   constant uint& n      [[buffer(4)]],
                   uint2 pos [[thread_position_in_grid]]) {
    // pos.x = column (0..n), pos.y = row (0..m)
    float sum = 0.0f;
    for (uint j = 0; j < k; j++) {
        sum += a[pos.y * k + j] * b[pos.x * k + j];
    }
    c[pos.y * n + pos.x] = sum;
}

// RoPE, NeoX half-split convention (what Qwen3 uses): for a head vector of
// size head_dim, element i pairs with element i + head_dim/2, rotated by
// angle = pos / theta^(2i/head_dim). One thread per pair per head; the
// buffer holds [n_heads, head_dim] contiguously.
kernel void rope(device float* x          [[buffer(0)]],
                 constant uint& head_dim  [[buffer(1)]],
                 constant uint& pos       [[buffer(2)]],
                 constant float& theta    [[buffer(3)]],
                 uint tid [[thread_position_in_grid]]) {
    uint half_dim = head_dim / 2;
    uint head = tid / half_dim;
    uint i = tid % half_dim;
    float freq = pow(theta, -2.0f * float(i) / float(head_dim));
    float angle = float(pos) * freq;
    float c = cos(angle);
    float s = sin(angle);
    uint base = head * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + i + half_dim];
    x[base + i] = x0 * c - x1 * s;
    x[base + i + half_dim] = x0 * s + x1 * c;
}

// ------------------------------------------------------------- fused ops

// rmsnorm + matvec in ONE kernel: y = W @ (rmsnorm(x) * nw).
// The composed version writes the normalized vector to device memory and
// reads it back in a second dispatch. Here each thread owns one output row
// and recomputes the sum-of-squares itself — redundant arithmetic (every
// thread sums the same x) traded for zero intermediate memory traffic and
// one dispatch instead of two. This trade is the whole fusion lesson.
// v1 of this kernel had every thread recompute sum_sq alone: 2048 threads
// x 1024 redundant reads, and it LOST to the composed pair (134us vs 93us
// measured). v2 exploits two facts: threads in a threadgroup can cooperate
// (sum_sq computed once per group of 256 rows), and matvec is linear, so
// the scalar inv_rms factors out of the dot product entirely.
// Requires: threadgroup size a power of two (the tree reduction).
kernel void rmsnorm_matvec(device const float* x  [[buffer(0)]],
                           device const float* nw [[buffer(1)]],
                           device const float* w  [[buffer(2)]],
                           device float* y        [[buffer(3)]],
                           constant uint& dim     [[buffer(4)]],
                           constant float& eps    [[buffer(5)]],
                           uint row [[thread_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tg_count [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    float sum_sq = 0.0f;
    for (uint j = tid; j < dim; j += tg_count) {
        sum_sq += x[j] * x[j];
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(scratch[0] / float(dim) + eps);

    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        sum += (x[j] * nw[j]) * w[row * dim + j];
    }
    y[row] = sum * inv_rms;
}

// ---------------------------------------------------------- reduction ops
//
// Pattern shared by the three kernels below: one threadgroup handles one
// row. Each thread accumulates a strided slice of the row into a private
// value, parks it in threadgroup memory, then a tree reduction halves the
// active thread count per step. threadgroup_barrier makes writes visible
// across threads between steps. tg_count is the threadgroup size chosen by
// the host (max 256 here = scratch array size).

// rmsnorm: out = x / sqrt(mean(x^2) + eps) * weight
kernel void rmsnorm(device const float* x      [[buffer(0)]],
                    device const float* weight [[buffer(1)]],
                    device float* out          [[buffer(2)]],
                    constant uint& dim         [[buffer(3)]],
                    constant float& eps        [[buffer(4)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tg_count [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_count) {
        sum_sq += x[i] * x[i];
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(scratch[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_count) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// softmax with the max-subtraction trick for numerical stability:
// out = exp(x - max(x)) / sum(exp(x - max(x)))
kernel void softmax(device const float* x [[buffer(0)]],
                    device float* out     [[buffer(1)]],
                    constant uint& dim    [[buffer(2)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tg_count [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];

    // pass 1: global max (guards exp against overflow)
    float local_max = -INFINITY;
    for (uint i = tid; i < dim; i += tg_count) {
        local_max = max(local_max, x[i]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 2: sum of shifted exponentials
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_count) {
        local_sum += exp(x[i] - row_max);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];

    // pass 3: normalize
    for (uint i = tid; i < dim; i += tg_count) {
        out[i] = exp(x[i] - row_max) * inv_sum;
    }
}

// argmax over a row -> single u32 index. Ties resolve to the LOWEST index
// (greedy sampling must be deterministic). Reduces (value, index) pairs.
kernel void argmax(device const float* x   [[buffer(0)]],
                   device uint* out_index  [[buffer(1)]],
                   constant uint& dim      [[buffer(2)]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tg_count [[threads_per_threadgroup]]) {
    threadgroup float best_val[256];
    threadgroup uint best_idx[256];
    float local_best = -INFINITY;
    uint local_idx = 0;
    for (uint i = tid; i < dim; i += tg_count) {
        if (x[i] > local_best) {
            local_best = x[i];
            local_idx = i;
        }
    }
    best_val[tid] = local_best;
    best_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            bool take_right = best_val[tid + stride] > best_val[tid] ||
                (best_val[tid + stride] == best_val[tid] &&
                 best_idx[tid + stride] < best_idx[tid]);
            if (take_right) {
                best_val[tid] = best_val[tid + stride];
                best_idx[tid] = best_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_index[0] = best_idx[0];
    }
}

// Single-position GQA attention, fused: scores, softmax, and the weighted
// value sum in ONE dispatch — the op that composition (matmul kernel,
// softmax kernel, matmul kernel) would spread over three round trips.
// One threadgroup per query head. Study-grade limit: seq <= 4096 so the
// score row fits threadgroup memory (16 KB of the ~32 KB budget).
kernel void attention(device const float* q   [[buffer(0)]],
                      device const float* k   [[buffer(1)]],
                      device const float* v   [[buffer(2)]],
                      device float* out       [[buffer(3)]],
                      constant uint& head_dim [[buffer(4)]],
                      constant uint& seq      [[buffer(5)]],
                      constant uint& group    [[buffer(6)]],
                      constant float& scale   [[buffer(7)]],
                      uint head [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg_count [[threads_per_threadgroup]]) {
    threadgroup float scores[4096];
    threadgroup float scratch[256];
    uint kv_head = head / group;

    // phase 1: this head's score for every cached position
    for (uint t = tid; t < seq; t += tg_count) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q[head * head_dim + d] * k[(kv_head * seq + t) * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 2: softmax over the scores (max, then sum, in-group)
    float local_max = -INFINITY;
    for (uint t = tid; t < seq; t += tg_count) {
        local_max = max(local_max, scores[t]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint t = tid; t < seq; t += tg_count) {
        float e = exp(scores[t] - row_max);
        scores[t] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_count / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];

    // phase 3: weighted sum of value rows
    for (uint d = tid; d < head_dim; d += tg_count) {
        float acc = 0.0f;
        for (uint t = 0; t < seq; t++) {
            acc += scores[t] * v[(kv_head * seq + t) * head_dim + d];
        }
        out[head * head_dim + d] = acc * inv_sum;
    }
}
