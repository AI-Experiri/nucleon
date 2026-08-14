# Introduction

This book documents building **nucleon**, an LLM inference engine written
from scratch in pure Rust for Apple Silicon, as it actually happened:
decisions, wrong turns, benchmarks, and all.

It teaches two subjects at once, deliberately interleaved. The engine
side: what a tokenizer, KV cache, sampler, transformer block, Metal
kernel, and quantized weight actually are, learned by building each one.
The Rust side: ownership when the tensor is born, traits when the backend
seam appears, `unsafe` when we mmap a checkpoint, GPU programming without
FFI when Metal arrives. Each concept shows up exactly when the engine
needs it, never as an abstract lesson.

Theory that has a better home is linked, not repeated: the interactive
labs at llm-lab.github.io cover tokenization algorithms, positional
encodings, KV caching, scaling laws, and quantization with visualizations
this book cannot match. The chapters here carry the build story.

Each chapter maps to one module of the codebase (one lego block), and each
milestone is a git tag you can check out to see the whole engine at that
stage of its life:

| Tag | The engine can... |
|-----|-------------------|
| `m1-cpu-hello` | generate coherent text from Qwen3-0.6B on CPU |
| `m2-metal-parity` | do the same on the GPU, fast |
| `m3-quant` | run 4-bit GGUF files |
| `m4-deepseek` | run DeepSeek-V2-Lite (MLA attention plus MoE) |

The raw session-by-session log lives in `journal/`; the architecture
decisions in `docs/decisions/`. This book is the curated retelling.

## How to read this book

The main text is the engine build. Rust language constructs are explained
in labeled side boxes at their first appearance, like this:

> **Rust: side boxes.** At most two lines on what a construct is, plus a
> link to the official docs. Later chapters assume earlier boxes.
> [The Rust Book](https://doc.rust-lang.org/book/)

Every chapter maps to one code module, and every module follows the same
test convention, so it's worth learning once, here. Each production file
ends with:

```rust
#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
```

> **Rust: `#[cfg(test)]`.** Compiles the module only for test builds;
> `#[path]` points it at the sibling `_tests.rs` file, and as a child
> module it can see private items.
> [Book ch. 11.3](https://doc.rust-lang.org/book/ch11-03-test-organization.html)

Read the tests as the specification: they encode hand-verified examples,
panic tests for every documented contract violation, and locked-down
conventions for edge cases. When a chapter's prose leaves you unsure what
a function promises, its `_tests.rs` file is the authoritative answer.

Build and read this book locally:

```bash
mdbook serve docs/book --open
```
