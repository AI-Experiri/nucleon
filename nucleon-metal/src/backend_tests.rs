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

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let err = std::panic::catch_unwind(f).expect_err("expected a panic");
    err.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

#[test]
fn guard_regressions_panic_before_the_gpu() {
    // the wrappers are the only line between safe Rust and unchecked GPU
    // indexing; these pin them so a guard regression cannot become OOB
    let Some(gpu) = gpu() else { return };

    let table = Tensor::zeros(vec![4, 8]);
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.embed(&table, 4); // vocab is 4, id 4 is out of range
    }));
    assert!(msg.contains(">= vocab"), "got: {msg}");

    let q = Tensor::zeros(vec![3, 4]);
    let kv = Tensor::zeros(vec![2, 5, 4]); // 3 heads over 2 kv heads: invalid
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0);
    }));
    assert!(msg.contains("divide evenly"), "got: {msg}");

    let q = Tensor::zeros(vec![1, 0]);
    let kv = Tensor::zeros(vec![1, 1, 0]);
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0);
    }));
    assert!(msg.contains("nonzero"), "got: {msg}");
}

#[test]
fn zero_sized_outer_shapes_agree_across_backends() {
    // the trait allows empty outer shapes; both executors must return
    // the same empty results
    let Some(gpu) = gpu() else { return };
    let cpu = CpuBackend;

    fn same(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.data(), b.data());
    }
    let empty = Tensor::zeros(vec![0]);
    same(&gpu.add(&empty, &empty), &cpu.add(&empty, &empty));
    same(&gpu.mul(&empty, &empty), &cpu.mul(&empty, &empty));
    same(&gpu.silu(&empty), &cpu.silu(&empty));
    same(&gpu.softmax(&empty), &cpu.softmax(&empty));
    same(
        &gpu.rmsnorm(&empty, &empty, 1e-6),
        &cpu.rmsnorm(&empty, &empty, 1e-6),
    );

    let w = Tensor::zeros(vec![0, 3]); // zero output rows
    let x = Tensor::zeros(vec![3]);
    same(&gpu.matvec(&w, &x), &cpu.matvec(&w, &x));

    let a = Tensor::zeros(vec![0, 3]);
    let b = Tensor::zeros(vec![2, 3]);
    same(&gpu.matmul(&a, &b), &cpu.matmul(&a, &b));

    let no_heads = Tensor::zeros(vec![0, 8]);
    same(&gpu.rope(&no_heads, 3, 1e6), &cpu.rope(&no_heads, 3, 1e6));
}

#[test]
fn attention_rejects_every_zero_dim() {
    let Some(gpu) = gpu() else { return };
    // zero seq
    let q = Tensor::zeros(vec![1, 2]);
    let kv = Tensor::zeros(vec![1, 0, 2]);
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0);
    }));
    assert!(msg.contains("nonzero"), "got: {msg}");
    // zero query heads (kv heads present)
    let q = Tensor::zeros(vec![0, 2]);
    let kv = Tensor::zeros(vec![1, 1, 2]);
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0);
    }));
    assert!(
        msg.contains("nonzero") || msg.contains("divide"),
        "got: {msg}"
    );
    // zero kv heads (query heads present)
    let q = Tensor::zeros(vec![2, 2]);
    let kv = Tensor::zeros(vec![0, 1, 2]);
    let msg = panic_message(std::panic::AssertUnwindSafe(|| {
        gpu.attention(&q, &kv, &kv, 1.0);
    }));
    assert!(
        msg.contains("nonzero") || msg.contains("divide"),
        "got: {msg}"
    );
}
