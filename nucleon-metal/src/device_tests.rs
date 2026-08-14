use super::*;

#[test]
fn gpu_initializes_and_compiles_all_kernels() {
    // Gpu::new compiles every kernel in ops.metal. Only a missing GPU is a
    // skip; a compile failure must FAIL (its message carries the Metal
    // compiler diagnostic).
    match Gpu::new() {
        Ok(_) => {}
        Err(e) if e.contains("no Metal device") => eprintln!("skipping: {e}"),
        Err(e) => panic!("Metal init failed: {e}"),
    }
}

#[test]
fn read_bounds_reject_overlarge_len_and_allow_zero() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) if e.contains("no Metal device") => return,
        Err(e) => panic!("Metal init failed: {e}"),
    };
    let buf = gpu.buffer_for_output(4);
    assert_eq!(gpu.read_f32(&buf, 4).len(), 4); // exact capacity is fine
    assert!(gpu.read_f32(&buf, 0).is_empty());
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.read_f32(&buf, 5)))
        .expect_err("overlarge read must panic");
    let msg = err
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(msg.contains("exceeds buffer capacity"), "got: {msg}");
}
