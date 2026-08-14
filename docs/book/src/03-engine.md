# The Engine

Part I built the parts: a tensor type, eleven operations behind the
`Backend` trait, and two implementations of that trait that agree with
each other. None of it has touched a real model. Part II assembles the
parts into an engine that loads Qwen3-0.6B from disk and generates text
with it. This chapter is the plan for that assembly: the premise, the
package on disk, the operation inventory, the build order, and the rule
every step follows.

## 5.1 What Part I left us

- `Tensor` ([Tensor 0](01-tensor.md)): shape plus flat f32 data.
- `trait Backend` ([Tensor 0](01-tensor.md), [Metal 0](02-metal.md)):
  eleven operations: add, mul, silu, embed, matvec, matmul, rmsnorm,
  softmax, rope, attention, argmax.
- `CpuBackend`: the naive loops that define correct.
- `MetalBackend`: the same trait on the GPU, held to the CPU by parity
  tests.

Everything below is wiring these calls together in the right order with
the right weights.

## 5.2 Models ship as families

Two words this part uses constantly:

- checkpoint: one trained model you can download, a folder of weights
  and configuration. Qwen3-0.6B is a checkpoint; the real one lives on
  Hugging Face (HF), the site checkpoints are distributed through:
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
- family: the architecture those checkpoints share; which blocks exist,
  in which order, with which normalizations. The `model_type` field in
  config.json ("qwen3") names the family.

Qwen3 the family shipped as dense checkpoints at 0.6B, 1.7B, 4B, 8B,
14B, and 32B parameters. All six have the same wiring; they differ only
in configuration numbers: hidden width, layer count, head counts. (The
Qwen3 MoE releases, whose MLP is a set of routed experts instead of
one block, are a different family, `qwen3_moe`; we are not building
it.)

The engine consequence: implement the wiring once per family, read the
numbers from config.json, and every size runs. Supporting a family
costs two things:

1. the correctness gate: every block the family needs exists and is
   wired exactly right, proven by the golden test from the map
   chapter's [1.5](00-big-picture.md#15-correctness);
2. the speed gate: the operations the family spends its time in get
   fast kernels for that family's shapes, which is Part III's job.

This two-gate pattern is what "the engine supports a model family"
means in this book.

We pay the gates for two families. `qwen3` comes first, at its smallest
checkpoint:

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

Every number in that table is readable in the checkpoint's actual
[config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
(or as [raw JSON](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json),
the exact bytes our loader will read); open it once now, because the
loader chapter parses exactly that file.

`qwen3_5` is the flagship: Qwen3.8-27B, a 64-layer hybrid that needs
operations we have not built yet. It gets its own chapter
([Qwen3.8, the hybrid](13-qwen38.md)). Nothing in Part II depends on
it, but two designs are shaped by knowing it is coming: the cache is a
trait so a second cache type can exist, and the family seam keeps every
model-specific decision out of the shared code.

## 5.3 The package on disk

A checkpoint downloads as a folder
([browse Qwen3-0.6B's](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)
to see it). For Qwen3-0.6B:

| file | role |
|---|---|
| config.json | the family name and every dimension number above |
| model.safetensors | all 311 weight tensors, bf16, about 1.2 GB |
| tokenizer.json | the full tokenizer: vocab and merge rules |
| tokenizer_config.json | the chat template and special token names |
| generation_config.json | stop token ids and default sampling settings |

<div class="diagram"><img src="diagrams/engine-package.svg" alt="each package file feeds one engine block"></div>

Each file feeds exactly one of the blocks this part builds.

config.json has no standalone specification. The model's authors do
not write it by hand; the HF transformers library writes it when they
save the trained model. The `model_type` value selects a configuration
class inside that library
([Qwen3Config](https://huggingface.co/docs/transformers/model_doc/qwen3)
for "qwen3"), and that class's fields and defaults are the only schema
there is; the fields every model shares are defined by its base class,
[PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration).
Because the schema is library code rather than a standard, our loader
takes no defaults for required fields: a value the file does not state
is an error, not a guess.

Two format words:

- safetensors: the weights container HF models ship in. Layout: 8 bytes
  holding the header length, a JSON header mapping each tensor name to
  its dtype, shape, and byte range, then the raw bytes. Simple enough
  that our loader parses it directly.
- GGUF: llama.cpp's single-file container, usually holding quantized
  (reduced-precision) weights. nucleon reads it in Part III
  ([Quantization](12-quantization.md)).

One dtype word: bf16 is a 16-bit float with f32's exponent range and
fewer fraction bits; it is what the weights are stored in. In Part II
the loader converts bf16 to f32 once at load time and all math stays
f32. Computing in bf16 directly is a Part III concern.

## 5.4 What loading requires

The loader's contract is strict on purpose:

1. every tensor the config promises must be present, with exactly the
   expected shape;
2. any tensor the config does not explain is an error (one exception:
   the tied lm_head copy, which Qwen3-0.6B ships even though it is
   byte-identical to the embedding table);
3. bf16 converts to f32 once, at load;
4. nothing from the file is trusted: byte ranges are bounds-checked,
   sizes are checked-multiplied, shard file names are path-checked.

The reason for strictness: a missing or misshapen weight does not crash
a transformer. It produces fluent, wrong tokens. Errors at load time
are cheap; errors at generation time cost a debugging session.

## 5.5 The op inventory

One decode step of Qwen3, written as operation calls (the full walk is
[Qwen3](07-qwen3.md)'s chapter):

1. embed: look the token id up in the embedding table.
2. 28 times, once per layer:
   1. rmsnorm, then three matvecs (the q, k, v projections);
   2. rmsnorm on each q and k head, then rope on q and k;
   3. attention, one call: scores, softmax, weighted values;
   4. matvec (the output projection), add (the residual);
   5. rmsnorm, two matvecs (gate, up), silu, mul, matvec (down), add.
3. final rmsnorm; matvec against the embedding table (tied) for the
   logits; argmax picks the token.

Check that list against the trait: embed, rmsnorm, matvec, rope,
attention, silu, mul, add, argmax. Every call is one of the eleven
operations, already implemented on both executors, already
parity-tested. Part II needs zero new kernels; that is why Part I was
built first. The two operations the list skips are covered too: softmax
runs inside the attention call, and matmul replaces matvec when prefill
processes many tokens at once (prefill and decode are defined in the
map chapter's [1.3](00-big-picture.md#13-prefill-and-decode)).

## 5.6 The order, and the rule

The rule: every block lands plain and correct before anything is
optimized, and every optimization after that must show a before/after
measurement on the same machine (the Apple M3 Max this book measures
on). [GPU 0](02-gpu.md) and [Metal 0](02-metal.md) already worked this
way; the engine keeps it.

The build order:

1. Fetch the Qwen3-0.6B package.
2. [The Loader](04-loader.md): folder in, checked f32 tensors out.
3. [The Tokenizer](05-tokenizer.md): text to ids and ids back to text,
   safe to stream.
4. The chat template: user text to ChatML, the conversation format
   Qwen3 was trained on; the template, never the tokenizer, inserts
   special tokens ([The CLI](10-cli.md) covers it).
5. [Qwen3](07-qwen3.md): the forward pass, wired from the inventory
   above.
6. [The Loop](09-generate.md), naive on purpose: no cache, so each
   step re-runs the whole sequence through the model. Greedy pick,
   stop on Qwen3's two stop ids (151645, 151643), tokens/sec recorded.
   Step t redoes the work of all earlier steps, so the recorded speed
   falls as the sequence grows; that falling curve is the baseline the
   improvements are judged against.
7. The golden test: the same prompt through HF transformers once,
   greedy on both sides, token ids must match exactly (the map
   chapter's [1.5](00-big-picture.md#15-correctness)). When they do not
   match, the method is layer-by-layer: dump the reference model's
   hidden states, diff against ours, fix the first layer that
   diverges.
8. Improvements, one at a time, each with its number:
   [The Cache](06-cache.md) removes the re-work from step 6, then GPU
   residency, fused kernels, bf16 compute, and quantized weights
   (Part III).

## 5.7 Upcoming engine topics

Engine work visible from here but scheduled after Parts II and III:

1. batching: serving several generation requests in one forward pass;
2. a server API, so other programs can call nucleon without linking it;
3. speculative and multi-token-prediction decoding (Qwen3.8 ships an
   MTP head we skip at first);
4. nucleon-cuda, a third `Backend`.

Next: [The Loader](04-loader.md).
