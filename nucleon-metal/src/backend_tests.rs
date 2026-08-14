use super::*;
use nucleon::backend::CpuBackend;

// Trait-level parity: the SAME calls through the SAME trait, CPU vs GPU,
// must agree within float tolerance. This is the contract that lets the
// engine swap executors without caring which one runs.

fn gpu() -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(b) => Some(b),
        Err(e) if e.contains("no Metal device") => {
            eprintln!("skipping Metal test: {e}");
            None
        }
        Err(e) => panic!("Metal init failed (not a missing-GPU skip): {e}"),
    }
}

fn assert_close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    for (i, (a, e)) in actual.data().iter().zip(expected.data()).enumerate() {
        let tol = 1e-4_f32.max(e.abs() * 1e-4);
        assert!((a - e).abs() <= tol, "element {i}: gpu={a}, cpu={e}");
    }
}

fn values(shape: Vec<usize>, seed: u32) -> Tensor {
    let mut n = seed;
    Tensor::from_fn(shape, |_| {
        n = n.wrapping_mul(2654435761).wrapping_add(12345);
        (n % 2000) as f32 / 1000.0 - 1.0
    })
}

#[test]
fn all_ops_match_cpu_through_the_trait() {
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend;

    let a = values(vec![512], 1);
    let b = values(vec![512], 2);
    assert_close(&gpu.add(&a, &b), &cpu.add(&a, &b));
    assert_close(&gpu.mul(&a, &b), &cpu.mul(&a, &b));
    assert_close(&gpu.silu(&a), &cpu.silu(&a));
    assert_close(&gpu.softmax(&a), &cpu.softmax(&a));

    let table = values(vec![50, 16], 3);
    assert_close(&gpu.embed(&table, 17), &cpu.embed(&table, 17));

    let w = values(vec![96, 64], 4);
    let x = values(vec![64], 5);
    assert_close(&gpu.matvec(&w, &x), &cpu.matvec(&w, &x));

    let ma = values(vec![17, 33], 6);
    let mb = values(vec![29, 33], 7);
    assert_close(&gpu.matmul(&ma, &mb), &cpu.matmul(&ma, &mb));

    let nw = values(vec![512], 8);
    assert_close(&gpu.rmsnorm(&a, &nw, 1e-6), &cpu.rmsnorm(&a, &nw, 1e-6));

    let heads = values(vec![4, 64], 9);
    assert_close(&gpu.rope(&heads, 17, 1e6), &cpu.rope(&heads, 17, 1e6));

    assert_eq!(gpu.argmax(&a), cpu.argmax(&a));
}

#[test]
fn attention_matches_cpu_with_gqa() {
    // Qwen3-like shape scaled down: 4 query heads sharing 2 kv heads
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend;
    let (n_heads, n_kv, seq, head_dim) = (4, 2, 33, 64);
    let q = values(vec![n_heads, head_dim], 10);
    let k = values(vec![n_kv, seq, head_dim], 11);
    let v = values(vec![n_kv, seq, head_dim], 12);
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_close(
        &gpu.attention(&q, &k, &v, scale),
        &cpu.attention(&q, &k, &v, scale),
    );
}

#[test]
fn attention_matches_cpu_at_longer_seq() {
    // seq beyond one 256-thread pass: strided score loops matter
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend;
    let (n_heads, n_kv, seq, head_dim) = (2, 1, 1000, 32);
    let q = values(vec![n_heads, head_dim], 13);
    let k = values(vec![n_kv, seq, head_dim], 14);
    let v = values(vec![n_kv, seq, head_dim], 15);
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_close(
        &gpu.attention(&q, &k, &v, scale),
        &cpu.attention(&q, &k, &v, scale),
    );
}

#[test]
fn attention_parity_at_the_seq_cap_boundary() {
    // exactly 4096: the score row fills threadgroup memory completely
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend;
    let (n_heads, n_kv, seq, head_dim) = (1, 1, 4096, 8);
    let q = values(vec![n_heads, head_dim], 20);
    let k = values(vec![n_kv, seq, head_dim], 21);
    let v = values(vec![n_kv, seq, head_dim], 22);
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_close(
        &gpu.attention(&q, &k, &v, scale),
        &cpu.attention(&q, &k, &v, scale),
    );
}

#[test]
fn attention_rejects_seq_beyond_cap() {
    let Some(gpu) = gpu() else { return };
    let q = Tensor::zeros(vec![1, 2]);
    let kv = Tensor::zeros(vec![1, 4097, 2]);
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0)
    }))
    .expect_err("must reject seq 4097");
    let msg = err
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("caps seq at 4096"), "got: {msg}");
}
