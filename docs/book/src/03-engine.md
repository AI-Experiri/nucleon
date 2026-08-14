# The Engine

Part I built parts: a tensor, eleven operations, two executors that
agree. Before assembling them into an engine, this chapter builds the
high-level view: what a model release actually is (Qwen as the
example), what is inside it, how the two packaging formats differ,
which engines run all this today, and the order nucleon builds in.

## 5.1 What Part I left us

- `Tensor` ([Tensor 0](01-tensor.md)): shape plus flat f32 data.
- `trait Backend` ([Tensor 0](01-tensor.md), [Metal 0](02-metal.md)):
  add, mul, silu, embed, matvec, matmul, rmsnorm, softmax, rope,
  attention, argmax.
- `CpuBackend`: the naive loops that define correct.
- `MetalBackend`: the same trait on the GPU, held to the CPU by parity
  tests.

## 5.2 Models ship as families

- checkpoint: originally a weights snapshot saved during training so a
  run can resume; the ecosystem now uses the word for any saved
  weights, including the final released ones (a release is the last
  checkpoint that survived). In this book a checkpoint is a released
  model you can download: a folder of weights and configuration. The
  real one: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
  on Hugging Face (HF), the site checkpoints are distributed through.
- family: the architecture checkpoints share. The `model_type` field
  in config.json ("qwen3") names it.

Every lab ships this way — one family, sizes as checkpoints:

| lab | family (`model_type`) | checkpoints |
|---|---|---|
| Qwen | qwen3 (dense) | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| Qwen | qwen3_moe | routed-experts variants (separate family; not building it) |
| Meta | llama | Llama 3.1 at 8B, 70B, 405B |
| Google | gemma2 | Gemma 2 at 2B, 9B, 27B |

<div class="diagram"><img src="diagrams/family-checkpoints.svg" alt="one family implementation runs six checkpoints"></div>

Supporting a family costs two gates, and paying them is what "the
engine supports a model family" means in this book:

1. correctness gate: every block wired exactly right, proven by the
   golden test (map chapter's
   [1.5](00-big-picture.md#15-correctness));
2. speed gate: fast kernels for the family's shapes (Part III).

We pay them for two families:

| | `qwen3` | `qwen3_5` |
|---|---|---|
| checkpoint we run | Qwen3-0.6B | Qwen3.8-27B, the flagship |
| layers | 28, all GQA attention | 64: 48 Gated DeltaNet + 16 attention |
| new ops needed | none (see 5.7) | Gated DeltaNet, hybrid cache |
| built in | this part | [Qwen3.8, the hybrid](13-qwen38.md) |

Part II never touches `qwen3_5`, but it already shaped two designs:
the cache is a trait, and families own every model-specific decision.

`qwen3`'s numbers, from the real
[config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
(also as [raw JSON](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json),
the exact bytes our loader reads):

| config.json field | Qwen3-0.6B |
|---|---|
| num_hidden_layers | 28 |
| hidden_size | 1024 |
| num_attention_heads / num_key_value_heads | 16 / 8 |
| head_dim | 128 (an explicit field, NOT hidden/heads = 64) |
| intermediate_size | 3072 |
| vocab_size | 151936 |
| rope_theta / rms_norm_eps | 1000000 / 1e-6 |
| tie_word_embeddings | true |

## 5.3 The package on disk

A checkpoint downloads as a folder
([browse Qwen3-0.6B's](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)):

| file | role |
|---|---|
| config.json | the family name and every dimension number above |
| model.safetensors | all 311 weight tensors, bf16, about 1.2 GB |
| tokenizer.json | the full tokenizer: vocab and merge rules |
| tokenizer_config.json | the chat template and special token names |
| generation_config.json | stop token ids and default sampling settings |

<div class="diagram"><img src="diagrams/engine-package.svg" alt="each package file feeds one engine block"></div>

<div class="warn">
<p>config.json has no specification. No standards body, no list of legal fields, no stability promise: each lab ships whatever fields its architecture needs, and may add, rename, or repurpose them next release.</p>
<p>This is a large part of why engines are hard. They lag new models not because the math is secret, but because someone must read each new config and modeling code and rewire the engine to match. Every architecture, every time.</p>
</div>

The same story holds for every json in the package. Each is written
by the HF stack when the lab saves its model, and the class that
writes it is the only schema it has:

| file | the class that writes it (= the schema) |
|---|---|
| config.json | the family's config class, picked by `model_type`: [Qwen3Config](https://huggingface.co/docs/transformers/model_doc/qwen3); the few fields every model shares come from the base class, [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration) |
| generation_config.json | [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation) — stop ids, default sampling |
| tokenizer_config.json | the tokenizer class's saved settings; the chat template is a string field in here |
| tokenizer.json | the [HF tokenizers library](https://huggingface.co/docs/tokenizers/index)'s own serialization (the one file with real documentation) |

Two questions this table raises, answered now because the tokenizer
and loop chapters depend on them:

- where token ids come from: the lab assigns them when it freezes the
  vocab before training, and the model learns embeddings for exactly
  those ids. Qwen3 reserves 151643 upward for special tokens: 151643
  `<|endoftext|>`, 151644 `<|im_start|>`, 151645 `<|im_end|>`, and so
  on, recorded in tokenizer.json's added-tokens section. Nothing
  about these numbers is standard; they are this family's
  training-time choices.
- where the meanings live: the files hold values; their semantics are
  defined by the family's modeling code,
  [modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).
  The model was trained with that code, so it outranks papers and
  blog posts when they disagree. Our `families/qwen3.rs` is a Rust
  port of what that file actually does.

So reading the config is the first step of supporting any model, every
time, and it always carries surprises:

| model | config field | the surprise |
|---|---|---|
| Qwen3-0.6B | `head_dim: 128` | explicit, while hidden/heads = 64; compute it instead of reading it and every weight loads, then the model generates garbage |
| Qwen3.8-27B | `full_attention_interval: 4`, `partial_rotary_factor: 0.25` | same lab, one generation later: fields qwen3 never had |
| DeepSeek-V2-Lite | `q_lora_rank: null` | one nullable field switches a whole projection block off; the larger checkpoint of the same `model_type` has it on |

It is also why our loader takes no defaults for required fields: with
no standard to fall back on, a value the file does not state is an
error, not a guess.

## 5.4 The formats, drawn

The same weights ship in several packagings. The full landscape, two
of which are lineages (pytorch .bin was replaced by safetensors;
GGML/GGJT by GGUF):

| format | from | since | shape | role for LLM weights today |
|---|---|---|---|---|
| pytorch .bin | PyTorch | 2016 | folder + sidecars | the old HF default; pickle-based, executes code on load; still on older repos |
| safetensors | Hugging Face | 2022 | folder + sidecars | the HF release default; labs publish in it; nucleon does not read it (ADR 004) |
| GGML / GGJT | llama.cpp | early 2023 | one file | GGUF's predecessors, superseded |
| GGUF | llama.cpp | Aug 2023 | one file | the local/quantized world's default; nucleon's one and only format |
| ONNX | Microsoft + Meta | 2017 | one graph file | cross-framework deployment; not how LLMs release weights |

(Names like GPTQ and AWQ on HF are not containers: they are
quantization methods, and those repos still ship safetensors files
holding the quantized values.)

The two nucleon reads, in detail; the difference is where the
configuration lives:

<div class="diagram"><img src="diagrams/formats-layout.svg" alt="safetensors folder with sidecar jsons vs GGUF single self-describing file"></div>

| | safetensors | GGUF |
|---|---|---|
| layout | 8 bytes of header length, JSON header (tensor name to dtype, shape, byte range), raw bytes | one self-describing file: metadata key-values plus tensors |
| weights held as | bf16 here (full precision) | usually quantized (reduced precision) |
| sits beside it | config.json, tokenizer files | nothing; metadata is inside |
| nucleon reads it | no: ADR 004 chose one format for the engine's whole life | yes, from day one; more quant types in Part III ([Quantization](12-quantization.md)) |

One dtype word: bf16 is a 16-bit float with f32's exponent range and
fewer fraction bits; the weights are stored in it. Part II converts to
f32 at load; computing in bf16 is Part III.

A format is only an envelope. Any container from which the loader can
recover the same three things — every tensor as (name, shape, dtype,
bytes), the config numbers, the tokenizer — feeds the same engine.
Supporting a new format costs one adapter inside the loader (parse the
layout, translate the tensor names); everything past the loader cannot
tell the difference.

Can a loader know it is reading a version it supports? Depends on the
format:

| | version marker | what a loader can do |
|---|---|---|
| GGUF | explicit version field right after the magic bytes (3 today; the spec documents 1 to 3) | refuse unsupported versions with a clear error |
| safetensors | none; the layout has never changed | validate structure strictly instead |
| config.json | none for the schema; only `transformers_version` (4.51.0 in Qwen3-0.6B's), which records the library that wrote it, not what the fields mean | strict required fields: missing or unknown = error, which catches schema drift the file cannot announce |
| ONNX | an IR version plus per-operator-set versions | full version negotiation |

config.json's row is the no-spec warning again: a file with no schema
has nothing to version. Our strict loader contract in 5.6 is the
substitute.

<div class="note">
<p>Where the formats come from: safetensors is Hugging Face's own format, built in 2022 to replace pickle-based PyTorch checkpoint files, which can execute arbitrary code when loaded. Its reference implementation is written in Rust, and our loader uses that exact crate (<a href="https://huggingface.co/docs/safetensors/index">format docs</a>, <a href="https://github.com/huggingface/safetensors">source</a>).</p>
<p>GGUF comes from the llama.cpp project (August 2023, replacing its earlier GGML and GGJT files): one self-describing file carrying weights and all metadata as key-value pairs (dimensions, even the whole tokenizer), so nothing sits beside it. Unlike config.json, GGUF has an actual written <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">specification</a>.</p>
</div>

## 5.5 The engine landscape

The same released checkpoint gets run by very different engines.
They differentiate on three axes: which format they read, which
hardware they bet on, and what they optimize for.

| engine | from | reads | hardware | built for |
|---|---|---|---|---|
| transformers | Hugging Face | safetensors | CPU, CUDA, Apple MPS | the reference: every model, correctness first, speed last |
| llama.cpp | ggml community | GGUF | CPU, Metal, CUDA, Vulkan | runs anywhere, quantized, single binary |
| MLX | Apple | safetensors | Apple silicon only | unified-memory-native research and local use |
| vLLM | UC Berkeley origin | safetensors | server GPUs (CUDA, ROCm) | serving many requests at once |
| nucleon | this book | GGUF | Apple silicon (CPU + Metal) | readable pure Rust, every block a lesson |

Two rows explain nucleon's ancestry: llama.cpp proves a from-scratch
engine can match the labs, and MLX proves Apple silicon rewards an
engine built for unified memory. nucleon takes both bets in Rust.

## 5.6 What loading requires

The loader's contract, strict on purpose:

1. every tensor the config promises is present, with the exact shape;
2. any tensor the config does not explain is an error (one exception:
   the tied lm_head copy, byte-identical to the embedding table);
3. bf16 converts to f32 once, at load;
4. nothing from the file is trusted: byte ranges bounds-checked, sizes
   checked-multiplied, shard names path-checked.

Why strict: a missing or misshapen weight does not crash a
transformer; it generates fluent, wrong tokens. Fail at load, not
mid-generation.

## 5.7 The op inventory

One decode step of Qwen3, as operation calls (the full walk is
[Qwen3](07-qwen3.md)'s chapter):

1. embed: look the token id up in the embedding table.
2. 28 times, once per layer:
   1. rmsnorm, three matvecs (q, k, v projections);
   2. rmsnorm on each q and k head, rope on q and k;
   3. attention, one call: scores, softmax, weighted values;
   4. matvec (output projection), add (residual);
   5. rmsnorm, two matvecs (gate, up), silu, mul, matvec (down), add.
3. final rmsnorm; matvec against the embedding table (tied) for
   logits; argmax picks the token.

Checked against the trait:

- every call above is one of the eleven operations: implemented on
  both executors, parity-tested, zero new kernels needed for Part II;
- softmax: not called directly, runs inside attention;
- matmul: takes over from matvec when prefill (map chapter's
  [1.3](00-big-picture.md#13-prefill-and-decode)) processes many
  tokens at once.

## 5.8 The build order

The rule: every block lands plain and correct first; every
optimization after that shows a before/after number on the same
machine (the Apple M3 Max this book measures on).

<div class="diagram"><img src="diagrams/engine-build-order.svg" alt="build order: package to first tokens to measured improvements"></div>

The steps, each a chapter:

1. fetch Qwen3-0.6B-Q8_0.gguf, the official one-file GGUF;
2. [The Loader](04-loader.md): file in, checked f32 tensors out
   (Q8_0 dequantized once, at load);
3. [The Tokenizer](05-tokenizer.md): text to ids and back, safe to
   stream;
4. the chat template: user text to ChatML, the conversation format
   Qwen3 was trained on; the template, never the tokenizer, inserts
   special tokens ([The CLI](10-cli.md));
5. [Qwen3](07-qwen3.md): the forward pass from the inventory above;
6. [The Loop](09-generate.md), naive on purpose: no cache, each step
   re-runs the whole sequence; greedy pick; stop on ids 151645 and
   151643; tokens/sec recorded as the baseline;
7. the golden test: same prompt through HF transformers, greedy both
   sides, ids match exactly; on mismatch, diff hidden states layer by
   layer and fix the first divergence;
8. improvements, one at a time, each measured:
   [The Cache](06-cache.md) removes step 6's re-work, then GPU
   residency, fused kernels, bf16 compute, quantized weights
   (Part III).

## 5.9 Upcoming engine topics

Visible from here, scheduled after Parts II and III:

1. batching: several generation requests in one forward pass;
2. a server API, so other programs call nucleon without linking it;
3. speculative and multi-token-prediction decoding (Qwen3.8 ships an
   MTP head we skip at first);
4. nucleon-cuda, a third `Backend`.

Next: [The Loader](04-loader.md).
