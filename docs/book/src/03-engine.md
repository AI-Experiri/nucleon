# The Engine

Part I set the ground: [GPU](02-gpu.md) taught the hardware,
[Metal](02-metal.md) taught what a kernel is and what composed vs
fused costs, [MLX](03-mlx.md) is the library we run on. Part II
assembles an engine out of that: loads Qwen3-0.6B and generates
text. This chapter is only the plan — what gets built, in which
order, under which rule. The world knowledge the plan rests on (how
models ship, what is in their files, the format we chose) is the
next chapter, [Reading the Model](03-reading-the-model.md).

## 5.1 What Part I left us

- **Hardware**: Apple Silicon's unified-memory GPU, one command
  buffer per submission, the launch cost of a chain of ops.
- **Concepts**: kernel, composed vs fused, why lazy graphs beat
  per-op dispatch.
- **The type**: `Array` (from `nucleon_mlx`, re-exported from
  `mlx-rs`) — every tensor value in the engine is one of these.
- **Ops we can call**: matmul, rms_norm, rope, silu, softmax,
  scaled_dot_product_attention, argmax, plus elementwise adds and
  multiplies — all `nucleon_mlx::` wrappers over MLX's built-ins
  (chapter 5.4 of MLX).

## 5.2 The thesis: families and the two gates

Two definitions, in one line each (their full story is the next
chapter's):

- checkpoint: a released model you can download;
- family: the architecture that checkpoints share. Implement the
  family once, and every checkpoint size runs on config numbers
  alone.

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
| new ops needed | none (see 5.3) | Gated DeltaNet, hybrid cache |
| built in | this part | [Qwen3.8, the hybrid](13-qwen38.md) |

Part II never touches `qwen3_5`, but it already shaped two designs:
the cache is a trait, and families own every model-specific decision.

## 5.3 The op inventory

One decode step of Qwen3, as operation calls (the full walk is
[Qwen3](07-qwen3.md)'s chapter):

1. embed: look the token id up in the embedding table.
2. 28 times, once per layer:
   1. rmsnorm, three matvecs (the q, k, v projections);
   2. rmsnorm on each q and k head, then rope on q and k;
   3. attention, one call: scores, softmax, weighted values;
   4. matvec (the output projection), add (the residual);
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

## 5.4 The build order

The rule: every block lands plain and correct first; every
optimization after that shows a before/after number on the same
machine (the Apple M3 Max this book measures on).

The first move is [Reading the Model](03-reading-the-model.md), the
next chapter: read everything about the release before touching it.
Then:

<div class="diagram"><img src="diagrams/engine-build-order.svg" alt="build order: package to first tokens to measured improvements"></div>

1. fetch Qwen3-0.6B-Q8_0.gguf, the official one-file GGUF;
2. [The Loader](04-loader.md): the gate; file in, Yamf out
   (the engine's own in-memory bundle, defined there);
3. [The Tokenizer](05-tokenizer.md): text to ids and ids back to
   text, safe to stream;
4. the chat template: user text to ChatML, the conversation format
   Qwen3 was trained on; the template, never the tokenizer, inserts
   special tokens ([The CLI](10-cli.md));
5. [Qwen3](07-qwen3.md): the forward pass from the inventory above;
6. [The Loop](09-generate.md), naive on purpose: no cache, each step
   re-runs the whole sequence; greedy pick; stop on ids 151645 and
   151643; tokens/sec recorded as the baseline;
7. the golden test: same prompt through HF transformers loading the
   same GGUF file, greedy both sides, ids match exactly; on mismatch,
   diff hidden states layer by layer and fix the first divergence;
8. improvements, one at a time, each measured:
   [The Cache](06-cache.md) removes step 6's re-work, then GPU
   residency, fused kernels, bf16 compute, more quant types
   (Part III).

## 5.5 Upcoming engine topics

Visible from here, scheduled after Parts II and III:

1. batching: several generation requests in one forward pass;
2. a server API, so other programs call nucleon without linking it;
3. speculative and multi-token-prediction decoding (Qwen3.8 ships an
   MTP head we skip at first);
4. nucleon-cuda, a third `Backend`.

Next: [Reading the Model](03-reading-the-model.md).
