# Chapter 0: what an inference engine is

This chapter is the map. Every later chapter builds one box of the diagram
below, so it is worth understanding the whole pipeline before writing any
code.

## 1.1 The pipeline

An inference engine takes text in and produces text out, one token at a
time. Internally the work splits into four stages:

```text
          "Why is the sky blue?"
                   |
        +----------v----------+
        | 1. TOKENIZER        |  text -> ids  [14990, 374, 279, ...]
        +----------+----------+
                   | ids (u32)
        +----------v----------+
        | 2. THE MODEL        |  ids -> a score for every word in the
        |    (per token:)     |  vocabulary ("logits", ~152k f32s)
        |  embed lookup       |
        |  N x transformer    |---- reads WEIGHTS (the .safetensors
        |      block          |     file: billions of frozen floats)
        |  final norm + head  |---- reads/writes KV CACHE (keys and
        +----------+----------+     values of all previous tokens)
                   | logits (f32 x vocab)
        +----------v----------+
        | 3. SAMPLER          |  152k scores -> one chosen id
        +----------+----------+
                   | next id
        +----------v----------+
        | 4. DETOKENIZER      |  id -> text fragment -> stream to user
        +----------+----------+
                   |
              append id, GOTO 2   (until EOS id or token limit)
```

The tokenizer converts text to integer ids from a fixed vocabulary. The
model maps the id sequence to a score for every possible next id. The
sampler picks one id from those scores. The detokenizer converts it back
to text, the id is appended to the sequence, and the process repeats until
the model emits an end-of-sequence id or hits a token limit.

The model itself is a pure function: given the same id sequence it always
produces the same scores. It has no hidden state between calls. Everything
that looks like memory or personality in a chat session comes from feeding
the growing sequence back in on every step.

## 1.2 The KV cache

Computed naively, predicting token 1000 would require re-processing tokens
1 through 999. The standard fix relies on a property of the transformer:
inside each block, the contribution of earlier tokens is a pair of
matrices (keys and values) that never change once computed. Storing them
means each new token costs one token's worth of computation rather than
the whole sequence's.

This store is the KV cache. It is the largest RAM consumer after the
weights, and it carries a position counter that the positional encoding
(RoPE) depends on. Errors in cache handling do not crash; they degrade
output quality in ways that are hard to trace back. Chapter 5 covers the
failure modes we inherited from the previous engine.

## 1.3 Prefill and decode

Processing the prompt ("prefill") runs all prompt tokens through the model
at once. This is matrix-matrix multiplication and the limiting factor is
arithmetic throughput. Generating ("decode") runs one token at a time.
This is matrix-vector multiplication, and the limiting factor is memory
bandwidth: every weight in the model is read once per generated token.

The numbers for Qwen3-0.6B in f32 on CPU: the weights are about 2.4 GB
(0.6B parameters x 4 bytes), read in full for every generated token. At a
realistic 50 GB/s of CPU memory bandwidth, that caps decode at roughly 20
tokens/s before counting any arithmetic. The KV cache adds about 229 KB
per token of context (28 layers x 8 kv-heads x 128 dims x 2 matrices x 4
bytes), so a 4k-token conversation holds about 0.9 GB of cache.
Activations and logits are a few MB, reused each step.

Two consequences follow. Apple Silicon is well suited to inference because
its unified memory has high bandwidth and the GPU can address all of it
(M2 targets the GPU's ~400 GB/s). And quantization (M3) speeds up decode
in addition to saving memory, because fewer bytes per weight means less to
stream per token.

## 1.4 Model families

Qwen3, Llama, and DeepSeek are built from the same small set of math
operations: matmul, rmsnorm, rope, softmax, silu, and a few elementwise
ops. They differ in how those operations are wired: head counts, where the
norms sit, whether attention compresses its cache (DeepSeek's MLA),
whether the MLP is a single block or 64 routed experts (MoE).

nucleon's structure follows from this: the operations live behind a
`Backend` trait, the wiring lives in a `ModelFamily`, and supporting a new
model means writing new wiring against existing operations.

## 1.5 Correctness

The engine's correctness check is exact token matching. Run a fixed prompt
through a reference implementation (HF transformers) with greedy sampling,
record the token ids, and require nucleon to produce the same ids. One
passing run of this "golden test" checks the tokenizer, the loader, every
math op, the cache offsets, and the sampling loop at once. Each milestone
ends by making it pass.

## 1.6 How the code is organized

The module boundaries follow two rules.

First, cut where the data changes representation. Text becomes ids at the
tokenizer, ids become logits in the model, logits become a single id at
the sampler, and the id becomes text at the detokenizer. Modules on
opposite sides of such a boundary exchange only a simple type (ids,
logits), which keeps them independent: either side can be rewritten
without reading the other.

Second, give things that change independently separate homes:

| Module | Changes when... |
|---|---|
| backend | hardware changes (CPU, then Metal, then CUDA) |
| families | a new model comes out (Qwen4, DeepSeek) |
| loader | the file format changes (safetensors, then GGUF) |
| tokenizer | the vocab or algorithm changes |
| sampler | decoding methods change (top-p, min-p, ...) |
| cache | the attention memory scheme changes (KV, then MLA) |
| chat | the prompt convention changes |
| generate | ideally never; it is the fixed loop |

The milestone plan doubles as a test of these boundaries: M2 should change
only `backend`, M3 only `loader` plus kernels, M4 only `families` plus a
cache implementation. A milestone that forces edits across several modules
means the boundaries were drawn wrong, and ADR 001 gets reopened.

Two boundaries deserve a note. Tensor and backend are separate so that the
data type stays stable while the compute implementation varies; merging
them would tie every backend to its own tensor type. Generate and families
are separate because the generation loop is identical for every model,
so a new family gets it for free.

The anti-example that motivates all this is in your own history: llama.cpp's
`llama.cpp` file is ~20k lines where format, families, cache, and sampling
interleave — brilliant code, but you can't rewrite one concern without
reading all of them. nucleon's bet is the opposite: each block small enough
to hold in your head, connected by types too dumb to leak complexity.

## 1.7 The chapters

| Ch | Block | Covers |
|----|-------|--------|
| 2 | gpu | the hardware: cores, memories, vendor comparison |
| 3 | metal | writing GPU software: kernels, composed vs fused |
| 4 | mlx | Apple's array library over Metal; the compute layer we run on |
| 5 | engine plan | what Part II builds, in which order, and why |
| 6 | reading the model | families, packages, formats, and the one-conversion thesis |
| 7 | loader | the gate: GGUF in, Yamf out |
| 8 | tokenizer | encoding, and streaming decode without broken UTF-8 |
| 9 | families/qwen3 | the full transformer forward pass |
| 10 | generate | the loop with no cache: correct first, measured slow |
| 11 | cache | KV storage as the first measured improvement |
| 12 | sampler | greedy, temperature, top-k, top-p |
| 13 | cli/chat | the ChatML template and the command line |
| 14 | fast kernels | tuning matmul and attention toward llama.cpp |
| 15 | gguf/quant | more quant types and fused dequant |
| 16 | qwen3.8 | the qwen3_5 hybrid: Gated DeltaNet and a second cache |
