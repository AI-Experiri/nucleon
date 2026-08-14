# backend — chapter 2

The math. `ops.rs` defines the `Backend` trait (~12 ops a transformer is
made of); `cpu.rs` is `CpuBackend`, the frozen naive reference every GPU
kernel is verified against.

**The design question of this chapter:** choosing the op list — coarse ops
(a whole `attention`) run faster and match GPU kernel granularity; fine ops
(softmax, matmul) teach more and test easier. The deep dive decides
per-op, against both the Qwen3 and DeepSeek-V2-Lite forward passes, so the
trait survives M1→M4 without churn.

**Read order:** `ops.rs` (the trait is the chapter), then `cpu.rs` loops,
then `cpu_tests.rs` hand-computed cases.
