# loader

The engine's gate. Runs once at startup: one GGUF file in, one `Yamf`
out (the engine's in-memory bundle: family config, f32 tensors,
tokenizer data, compiled chat template), or a refusal that names what
was expected, what the file held, and why it cannot be supported.
Nothing GGUF-shaped crosses this border; no other module ever sees
raw metadata.

Why it exists: formats churn (chapters "Reading the Model" and "The
Loader" in docs/book). Converting once at a single border means a new
format is one new adapter here and zero engine changes.

How to read it, in dependency order:

| file | concern |
|---|---|
| error.rs | every refusal, with found-vs-supported in the message |
| container.rs | GGUF v3 bytes: header, metadata KVs, tensor infos, alignment |
| config.rs | metadata keys to `Qwen3Config`, wrapped in `FamilyConfig` |
| dequant.rs | ggml types to f32: F32 passthrough, Q8_0 blocks |
| yamf.rs | the gate itself: expected-tensor contract, checks, `load` |
| gguf_builder.rs | test-only: writes tiny valid GGUF files, reference Q8_0 quantizer |

Every fact about the format and the real file's bytes is sourced in
docs/research/gguf-qwen3.md. Tests run on synthetic files only; the
oracle test cross-checks our parser and dequant against candle-core
(dev-dependency).
