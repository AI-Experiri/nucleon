//! The fusion lesson, measured. Same math three ways at Qwen3's q_proj
//! decode shape (rmsnorm a 1024-vector, then a 2048x1024 matvec):
//!
//!   1. composed, sync per op   — 2 dispatches, 2 waits per step
//!      (isolates dispatch+wait cost; buffers stay resident, so this is
//!      still CHEAPER than the per-call wrappers, which also re-create
//!      buffers and read back per call)
//!   2. composed, batched       — 2 dispatches, 1 wait for ALL steps
//!      (real engines encode a whole forward pass per command buffer)
//!   3. fused, batched          — 1 dispatch per step
//!
//! Run: cargo run --release -p nucleon-metal --example fusion

use nucleon_metal::device::{KernelCall, Scalar};
use nucleon_metal::ops::MetalOps;
use std::time::Instant;

const DIM: usize = 1024;
const OUT_DIM: usize = 2048;
const STEPS: usize = 500;
const EPS: f32 = 1e-6;

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            (x % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn main() {
    let ops = match MetalOps::new() {
        Ok(o) => o,
        Err(e) if e.contains("no Metal device") => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => {
            // a compile or family failure must not exit 0 as if benign
            eprintln!("Metal init failed: {e}");
            std::process::exit(1);
        }
    };
    let gpu = ops.gpu();

    let x = values(DIM, 1);
    let nw = values(DIM, 2);
    let w = values(OUT_DIM * DIM, 3);

    // resident buffers, created once — only dispatches are timed
    let xb = gpu.buffer_from_f32(&x);
    let nb = gpu.buffer_from_f32(&nw);
    let wb = gpu.buffer_from_f32(&w);
    let tmp = gpu.buffer_for_output(DIM);
    let yb = gpu.buffer_for_output(OUT_DIM);

    let rmsnorm = || KernelCall {
        kernel: "rmsnorm",
        buffers: vec![&xb, &nb, &tmp],
        scalars: vec![Scalar::U32(DIM as u32), Scalar::F32(EPS)],
        grid: (256, 1),
        threadgroup: Some((256, 1)),
    };
    let matvec = || KernelCall {
        kernel: "matvec",
        buffers: vec![&wb, &tmp, &yb],
        scalars: vec![Scalar::U32(DIM as u32)],
        grid: (OUT_DIM, 1),
        threadgroup: None,
    };
    let fused = || KernelCall {
        kernel: "rmsnorm_matvec",
        buffers: vec![&xb, &nb, &wb, &yb],
        scalars: vec![Scalar::U32(DIM as u32), Scalar::F32(EPS)],
        grid: (OUT_DIM, 1),
        threadgroup: Some((256, 1)), // power-of-two groups: tree reduction
    };

    // warmup
    unsafe { gpu.run_many(&[rmsnorm(), matvec(), fused()]) };

    // 1: composed, wait after every op (calls prebuilt outside the timed
    // loop so host-side allocation does not bias this case upward)
    let sync_steps: Vec<[KernelCall; 1]> = (0..STEPS * 2)
        .map(|i| if i % 2 == 0 { [rmsnorm()] } else { [matvec()] })
        .collect();
    let t = Instant::now();
    for call in &sync_steps {
        unsafe { gpu.run_many(call) };
    }
    let per_step_sync = t.elapsed().as_micros() as f64 / STEPS as f64;

    // 2: composed, one command buffer for all steps
    let composed: Vec<KernelCall> = (0..STEPS).flat_map(|_| [rmsnorm(), matvec()]).collect();
    let t = Instant::now();
    unsafe { gpu.run_many(&composed) };
    let per_step_batched = t.elapsed().as_micros() as f64 / STEPS as f64;

    // 3: fused, one command buffer for all steps
    let fused_calls: Vec<KernelCall> = (0..STEPS).map(|_| fused()).collect();
    let t = Instant::now();
    unsafe { gpu.run_many(&fused_calls) };
    let per_step_fused = t.elapsed().as_micros() as f64 / STEPS as f64;

    println!("rmsnorm + matvec, dim {DIM} -> {OUT_DIM}, {STEPS} steps");
    println!("1. composed, sync per op : {per_step_sync:8.1} us/step");
    println!("2. composed, batched     : {per_step_batched:8.1} us/step");
    println!("3. fused, batched        : {per_step_fused:8.1} us/step");
    println!(
        "sync overhead {:.1}x, fusion gain {:.2}x",
        per_step_sync / per_step_batched,
        per_step_batched / per_step_fused
    );
}
