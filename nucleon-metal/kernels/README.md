# kernels — the only non-Rust files in nucleon

One `.metal` file per Backend op (matmul first — it is ~95% of decode time;
then rmsnorm, rope, softmax, attention). Each kernel is leaf math: a loop a
GPU thread runs. Compiled at runtime by `src/device.rs`; verified op-by-op
against `CpuBackend` within tolerance (chapter 11).
