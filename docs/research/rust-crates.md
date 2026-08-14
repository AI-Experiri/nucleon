# Rust crate facts (verified against docs.rs, 2026-08)

Pinned versions and the API shapes nucleon builds on.

## safetensors 0.8

- `SafeTensors::deserialize(&mmap)` parses header only (cheap); tensor bytes
  are lazy via page faults. Mmap via `memmap2` (separate crate).
- `TensorView`: `dtype()`, `shape() -> &[usize]`, `data() -> &[u8]` zero-copy.
- **Lifetime trap:** `SafeTensors<'data>` borrows the mmap — don't try to
  store (Mmap, SafeTensors) self-referentially. Either re-deserialize on
  access, or `read_metadata` once and keep our own
  `name -> (dtype, shape, absolute byte range)` map (offset = 8 + header_len
  + data_offsets.0). We'll do the map.
- **No index.json support** — parse `model.safetensors.index.json`
  (`weight_map: HashMap<String, String>`) ourselves; mmap each shard once;
  validate shard names (path traversal).
- **Alignment not guaranteed** — use `bytemuck::try_cast_slice` with an
  unaligned-read fallback, never bare `cast_slice`.

## tokenizers 0.23 (HF)

- `Tokenizer::from_file("tokenizer.json")`; `encode(text, add_special)` →
  `Encoding::get_ids() -> &[u32]`; `token_to_id`, `decode`.
- **Streaming decode**: `tokenizer.decode_stream(skip_special)` →
  `step(id) -> Result<Option<String>>`; `None` = withheld (partial UTF-8).
  Never re-decode full buffers or decode single ids ourselves.
- Build lean: `default-features = false, features = ["fancy-regex"]`
  (default pulls onig — a C build, violates pure-Rust).

## half 2.7

- `bf16::to_f32` / `from_f32` (rounds-to-nearest-even; don't hand-roll
  bit-shift truncation). Bulk: `half::slice::HalfFloatSliceExt`
  (`reinterpret_cast`, `convert_to_f32_slice`); `bytemuck` feature for Pod.

## objc2-metal 0.3 (metal-rs 0.33 is officially deprecated — do not use)

- `MTLCreateSystemDefaultDevice()` (needs CoreGraphics linked: dep
  objc2-core-graphics or a `#[link]` block; returns None headless).
- Compile MSL at runtime: `newLibraryWithSource_options_error(&NSString, …)`
  → `newFunctionWithName` → `newComputePipelineStateWithFunction_error`.
  **Always surface the NSError localizedDescription** or shader compile
  errors are invisible.
- Buffers: `newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)`
  (unified memory; contents() readable after waitUntilCompleted — reading
  while the GPU writes is a data race).
- Dispatch: commandBuffer → computeCommandEncoder →
  setComputePipelineState → `unsafe setBuffer_offset_atIndex` →
  `dispatchThreads_threadsPerThreadgroup` (non-uniform grids; fine on all
  Apple7+ GPUs — simpler than manual threadgroup tails) → endEncoding
  (lives on parent trait `MTLCommandEncoder` — must be in scope!) → commit →
  waitUntilCompleted.
- Threadgroup sizing from `pso.maxTotalThreadsPerThreadgroup()` /
  `threadExecutionWidth()` — never hardcode 256.
- Pin objc2 family versions together (0.3.x pre-1.0, breaking minors).

## hf-hub 1.0 (model downloads)

- **1.0 is a full rewrite** (async `HFClient`, `client.model(owner, name)` —
  owner and name SEPARATE args, `.download_file().filename(..).send().await`).
  All old docs/candle examples show the dead 0.4 sync API. For a simple sync
  CLI either pin `hf-hub = "0.4"` or use 1.0's `blocking` feature. Auth via
  HF_TOKEN env; cache at ~/.cache/huggingface.
- M1 alternative: instruct the user to download with `hf` CLI; add hf-hub
  integration in a later step (keeps M1 dependency-light).
