use super::*;

// Every test computes the expected result with a plain Rust loop (the
// future CpuBackend's reference math) and requires the GPU to match within
// tolerance. GPU float ops reorder, so exact equality is wrong; 1e-5
// relative is the standard op-parity bar at f32.

/// Skip (not fail) ONLY on machines with no GPU, e.g. CI. Every other
/// initialization error — above all a kernel compile failure — must fail
/// the test, or a broken ops.metal would turn the whole suite green.
fn gpu() -> Option<MetalOps> {
    match MetalOps::new() {
        Ok(ops) => Some(ops),
        Err(e) if e.contains("no Metal device") => {
            eprintln!("skipping Metal test: {e}");
            None
        }
        Err(e) => panic!("Metal init failed (not a missing-GPU skip): {e}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = 1e-5_f32.max(e.abs() * 1e-5);
        assert!(
            (a - e).abs() <= tol,
            "element {i}: gpu={a}, cpu={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Deterministic pseudo-random values in [-1, 1] without an RNG dependency.
fn test_values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            (x % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

#[test]
fn add_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let a = test_values(1000, 1);
    let b = test_values(1000, 2);
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close(&ops.add(&a, &b), &expected);
}

#[test]
fn mul_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let a = test_values(1000, 3);
    let b = test_values(1000, 4);
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    assert_close(&ops.mul(&a, &b), &expected);
}

#[test]
fn silu_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let x = test_values(1000, 5);
    let expected: Vec<f32> = x.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    assert_close(&ops.silu(&x), &expected);
}

#[test]
fn embed_returns_the_id_row() {
    let Some(ops) = gpu() else { return };
    let dim = 8;
    let table = test_values(5 * dim, 6); // vocab of 5
    let expected = &table[3 * dim..4 * dim];
    assert_close(&ops.embed(&table, dim, 3), expected);
}

#[test]
fn matvec_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let (out_dim, in_dim) = (64, 96);
    let w = test_values(out_dim * in_dim, 7);
    let x = test_values(in_dim, 8);
    let mut expected = vec![0.0; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0;
        for j in 0..in_dim {
            sum += w[row * in_dim + j] * x[j];
        }
        expected[row] = sum;
    }
    assert_close(&ops.matvec(&w, &x), &expected);
}

#[test]
fn matvec_hand_case() {
    // [1 2; 3 4] @ [10, 100] = [210, 430] — checked on paper, so a layout
    // mistake (row-major vs column-major) cannot hide behind the loop test
    let Some(ops) = gpu() else { return };
    let w = [1.0, 2.0, 3.0, 4.0];
    let x = [10.0, 100.0];
    assert_close(&ops.matvec(&w, &x), &[210.0, 430.0]);
}

#[test]
fn matmul_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let (m, k, n) = (17, 33, 29); // deliberately not multiples of 16
    let a = test_values(m * k, 9);
    let b = test_values(n * k, 10);
    let mut expected = vec![0.0; m * n];
    for i in 0..m {
        for jj in 0..n {
            let mut sum = 0.0;
            for j in 0..k {
                sum += a[i * k + j] * b[jj * k + j];
            }
            expected[i * n + jj] = sum;
        }
    }
    assert_close(&ops.matmul(&a, &b, m, k, n), &expected);
}

#[test]
fn rope_at_position_zero_is_identity() {
    // angle = 0 for every pair at pos 0: cos=1, sin=0, nothing moves
    let Some(ops) = gpu() else { return };
    let x = test_values(4 * 64, 11); // 4 heads, head_dim 64
    assert_close(&ops.rope(&x, 64, 0, 1e6), &x);
}

#[test]
fn rope_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let (n_heads, head_dim) = (4, 64);
    let x = test_values(n_heads * head_dim, 12);
    let (pos, theta) = (17_u32, 1e6_f32);
    let half = head_dim / 2;
    let mut expected = x.clone();
    for h in 0..n_heads {
        for i in 0..half {
            let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            let base = h * head_dim;
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            expected[base + i] = x0 * c - x1 * s;
            expected[base + i + half] = x0 * s + x1 * c;
        }
    }
    assert_close(&ops.rope(&x, head_dim, pos, theta), &expected);
}

#[test]
fn rmsnorm_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let dim = 1024; // bigger than the 256-thread group: strided loops matter
    let x = test_values(dim, 13);
    let w = test_values(dim, 14);
    let eps = 1e-6;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = x.iter().zip(&w).map(|(v, wt)| v * inv_rms * wt).collect();
    assert_close(&ops.rmsnorm(&x, &w, eps), &expected);
}

#[test]
fn softmax_matches_cpu_and_sums_to_one() {
    let Some(ops) = gpu() else { return };
    let x = test_values(1000, 15);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    let actual = ops.softmax(&x);
    assert_close(&actual, &expected);
    let total: f32 = actual.iter().sum();
    assert!((total - 1.0).abs() < 1e-4, "softmax sums to {total}");
}

#[test]
fn argmax_matches_cpu() {
    let Some(ops) = gpu() else { return };
    let mut x = test_values(5000, 16);
    x[3777] = 9.0; // unambiguous winner
    assert_eq!(ops.argmax(&x), 3777);
}

#[test]
fn fused_rmsnorm_matvec_matches_the_composed_ops() {
    // the fused kernel must match the composed pair within tolerance for
    // normal-range values (its documented contract; it is NOT bit-equal
    // and can diverge for extreme magnitudes, see the wrapper doc)
    let Some(ops) = gpu() else { return };
    let (dim, out_dim) = (256, 128);
    let x = test_values(dim, 20);
    let nw = test_values(dim, 21);
    let w = test_values(out_dim * dim, 22);
    let eps = 1e-6;
    let composed = ops.matvec(&w, &ops.rmsnorm(&x, &nw, eps));
    let fused = ops.rmsnorm_matvec(&x, &nw, &w, eps);
    assert_close(&fused, &composed);
}

#[test]
fn fused_kernel_matches_cpu_reference_directly() {
    // GPU-vs-GPU comparison (the two tests below) can hide a bug shared by
    // both paths; this one recomputes the whole thing in plain Rust
    let Some(ops) = gpu() else { return };
    let (dim, out_dim) = (128, 64);
    let x = test_values(dim, 26);
    let nw = test_values(dim, 27);
    let w = test_values(out_dim * dim, 28);
    let eps = 1e-6;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let mut expected = vec![0.0; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0;
        for j in 0..dim {
            sum += (x[j] * inv_rms * nw[j]) * w[row * dim + j];
        }
        expected[row] = sum;
    }
    assert_close(&ops.rmsnorm_matvec(&x, &nw, &w, eps), &expected);
}

#[test]
fn rope_zero_head_dim_panics_with_a_clear_message() {
    // regression: this used to die on x.len() % 0 before validation.
    // catch_unwind instead of should_panic so a no-GPU machine skips
    // honestly instead of faking the expected panic.
    let Some(ops) = gpu() else { return };
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ops.rope(&[], 0, 0, 1e6)))
        .expect_err("rope must reject head_dim 0");
    // a literal assert! panics with &str, a formatted one with String
    let msg = err
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("head_dim must be nonzero"), "got: {msg}");
}

#[test]
fn empty_and_zero_row_inputs_return_empty() {
    // locked conventions: no zero-thread GPU dispatches, no dangling reads
    let Some(ops) = gpu() else { return };
    assert!(ops.add(&[], &[]).is_empty());
    assert!(ops.silu(&[]).is_empty());
    assert!(ops.softmax(&[]).is_empty());
    assert!(ops.rmsnorm(&[], &[], 1e-6).is_empty());
    assert!(ops.matvec(&[], &[1.0]).is_empty()); // zero output rows
    assert!(ops.rope(&[], 64, 3, 1e6).is_empty());
    assert!(ops.matmul(&[], &[1.0, 2.0], 0, 2, 1).is_empty());
}

#[test]
fn fused_kernel_parity_at_benchmark_shape() {
    // out_dim 2048 = 8 threadgroups: exercises the multi-group path the
    // 128-row test never reaches (each group redoes its own reduction)
    let Some(ops) = gpu() else { return };
    let (dim, out_dim) = (1024, 2048);
    let x = test_values(dim, 23);
    let nw = test_values(dim, 24);
    let w = test_values(out_dim * dim, 25);
    let eps = 1e-6;
    let composed = ops.matvec(&w, &ops.rmsnorm(&x, &nw, eps));
    let fused = ops.rmsnorm_matvec(&x, &nw, &w, eps);
    assert_close(&fused, &composed);
}

#[test]
fn argmax_tie_takes_lowest_index() {
    let Some(ops) = gpu() else { return };
    let mut x = vec![0.0; 600];
    x[100] = 5.0;
    x[500] = 5.0; // same value later — must lose the tie
    assert_eq!(ops.argmax(&x), 100);
}

#[test]
fn reduction_ops_handle_small_and_odd_dims() {
    // dims below the 256-thread group exercise idle threads' identity
    // values (0 for sums, -inf for maxes) inside the tree reduction
    for dim in [1_usize, 3, 255] {
        let Some(ops) = gpu() else { return };
        let x = test_values(dim, 30 + dim as u32);
        let w = test_values(dim, 40 + dim as u32);
        let eps = 1e-6;

        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let expected: Vec<f32> = x.iter().zip(&w).map(|(v, wt)| v * inv_rms * wt).collect();
        assert_close(&ops.rmsnorm(&x, &w, eps), &expected);

        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
        let expected: Vec<f32> = x.iter().map(|v| (v - max).exp() / sum).collect();
        assert_close(&ops.softmax(&x), &expected);

        let mut best = 0;
        for (i, v) in x.iter().enumerate() {
            if *v > x[best] {
                best = i;
            }
        }
        assert_eq!(ops.argmax(&x) as usize, best, "argmax at dim {dim}");
    }
}

#[test]
fn argmax_ignores_nan() {
    // documented policy: NaN never wins (NaN > best is false on GPU too)
    let Some(ops) = gpu() else { return };
    let x = [f32::NAN, -1.0, -5.0];
    assert_eq!(ops.argmax(&x), 1);
}

#[test]
fn rope_preserves_pair_norms() {
    // independent property, no formula duplication: rotation never changes
    // a pair's length, at any position
    let Some(ops) = gpu() else { return };
    let (n_heads, head_dim) = (2, 8);
    let x = test_values(n_heads * head_dim, 50);
    let rotated = ops.rope(&x, head_dim, 12345, 1e6);
    let half = head_dim / 2;
    for h in 0..n_heads {
        for i in 0..half {
            let base = h * head_dim;
            let before = (x[base + i].powi(2) + x[base + i + half].powi(2)).sqrt();
            let after = (rotated[base + i].powi(2) + rotated[base + i + half].powi(2)).sqrt();
            assert!(
                (before - after).abs() < 1e-5,
                "pair ({h},{i}) norm changed: {before} -> {after}"
            );
        }
    }
}

#[test]
fn rope_hand_computed_case() {
    // head_dim 2: one pair, freq = theta^0 = 1, angle = pos. With pos=1,
    // theta arbitrary: [1, 0] must rotate to [cos 1, sin 1]. Worked by
    // hand from the NeoX definition, so a shared sign/layout bug in both
    // the kernel and the loop-reference test cannot hide here.
    let Some(ops) = gpu() else { return };
    let out = ops.rope(&[1.0, 0.0], 2, 1, 1e6);
    assert!((out[0] - 1.0_f32.cos()).abs() < 1e-6, "got {out:?}");
    assert!((out[1] - 1.0_f32.sin()).abs() < 1e-6, "got {out:?}");
}

#[test]
fn argmax_all_nan_or_neg_inf_returns_zero() {
    // documented sentinel policy: -inf entries cannot win either
    let Some(ops) = gpu() else { return };
    assert_eq!(ops.argmax(&[f32::NAN, f32::NEG_INFINITY]), 0);
    assert_eq!(ops.argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    // one real value beats both
    assert_eq!(ops.argmax(&[f32::NAN, f32::NEG_INFINITY, -9.0]), 2);
}

#[test]
fn fused_kernel_rejects_out_of_contract_shapes() {
    // keep the narrow shape contract executable, not just a comment
    let Some(ops) = gpu() else { return };
    let x = test_values(4, 60);
    let nw = test_values(4, 61);
    let w = test_values(129 * 4, 62); // 129: not power-of-two, not mult of 256
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ops.rmsnorm_matvec(&x, &nw, &w, 1e-6)
    }))
    .expect_err("fused must reject out_dim 129");
    let msg = err
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(
        msg.contains("power-of-two or multiple of 256"),
        "got: {msg}"
    );
}

#[test]
#[should_panic(expected = "exceeds u32")]
fn dim_u32_rejects_oversized_dimensions() {
    dim_u32(usize::MAX);
}

#[test]
#[should_panic(expected = "stride wrap")]
fn reduction_dim_u32_rejects_near_cap_dimensions() {
    reduction_dim_u32(u32::MAX as usize);
}
