# The Loader

The first block of Part II. It runs once, at startup, and crosses the
border drawn in [Reading the Model](03-reading-the-model.md): one
GGUF file on one side, the engine's plain types on the other. This
chapter is the design; the build follows it. Every number below was verified against
the real file's bytes (docs/research/gguf-qwen3.md carries the
sources).

## 7.1 The job

<div class="diagram"><img src="diagrams/loader-border.svg" alt="one gguf file in, the Yamf bundle out; only the loader knows the format"></div>

- in: one file, `Qwen3-0.6B-Q8_0.gguf` (639,446,688 bytes);
- out: the `Yamf` bundle (7.8): the family's config numbers, every
  weight as a checked f32 `Tensor`, the tokenizer data, the compiled
  chat template;
- guarantee: nothing past the loader can tell how weights were
  stored. GGUF is the only format
  ([ADR 004](../decisions/004-gguf-only-engine.md)); the parser is
  nucleon's own code, not a dependency.

Fetch the file once before the build:

```bash
hf download Qwen/Qwen3-0.6B-GGUF --local-dir models/qwen3-0.6b-gguf
```

`models/` stays out of git; unit tests never touch it.

## 7.2 The container, byte by byte

<div class="diagram"><img src="diagrams/gguf-bytes.svg" alt="header, metadata KVs, tensor infos, pad, tensor data; offsets relative to data start"></div>

Everything is little-endian. The file opens with a fixed header:

| offset | field | type | this file |
|---|---|---|---|
| 0 | magic | 4 bytes | "GGUF" |
| 4 | version | u32 | 3 (anything else: refuse) |
| 8 | tensor_count | u64 | 310 |
| 16 | metadata_kv_count | u64 | 28 |

Then `metadata_kv_count` key-value pairs, then `tensor_count` tensor
infos, then zero-padding to the alignment (32 unless the metadata says
otherwise; 28 pad bytes here), then the tensor data region, which
starts at byte 5,951,136 in this file.

Naming collision warning: "KV" here means key-value pair, a named
entry in the file's dictionary. It has nothing to do with the KV
cache from [the map chapter](00-big-picture.md#12-the-kv-cache),
which is runtime memory of attention key and value vectors; that
cache does not exist until generation runs.

The building blocks:

- string: a u64 byte length, then that many UTF-8 bytes, never
  null-terminated. Lengths are bounds-checked against the file before
  any allocation;
- metadata value: a u32 type id, then the value. Type ids 0 to 12:
  u8/i8, u16/i16, u32/i32, f32, bool (a byte that must be 0 or 1),
  string, array (element type id + u64 count + packed values,
  nestable), u64/i64, f64;
- tensor info: name (string, max 64 bytes), n_dimensions (u32, max
  4), the dimensions as u64s, a u32 ggml type id (F32 = 0, Q8_0 = 8),
  and a u64 offset relative to the data region's start, which must be
  a multiple of the alignment.

A version note for the error message: v1 encoded string lengths as
u32, v2 widened them to u64, v3 added big-endian support with
identical structure. A big-endian file has no marker; it reveals
itself by version reading as 50,331,648 instead of 3, and the parser
refuses it by value.

The file is read through mmap: the operating system maps it into our
address space and pages it in as bytes are touched. The mapping is
`unsafe` (the box is in [Metal 0](02-metal.md)) because Rust cannot
prove the file stays unmodified while mapped; our stated invariant is
a local file nobody rewrites mid-run. The memmap2 crate provides it.

## 7.3 The metadata this file carries

You can see all of this without downloading anything: Hugging Face
parses GGUF server-side, so the
[file viewer](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF?show_file_info=Qwen3-0.6B-Q8_0.gguf)
shows every metadata key and the full tensor list in the browser, and
the [model API](https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF)
returns a parsed `gguf` object for scripts. All 28 keys are
enumerated in the research doc; the ones the loader reads for the
config:

| GGUF metadata key | value | Qwen3Config field |
|---|---|---|
| general.architecture | "qwen3" | checked, not stored |
| qwen3.block_count | 28 | num_hidden_layers |
| qwen3.embedding_length | 1024 | hidden_size |
| qwen3.feed_forward_length | 3072 | intermediate_size |
| qwen3.attention.head_count | 16 | num_attention_heads |
| qwen3.attention.head_count_kv | 8 | num_key_value_heads |
| qwen3.attention.key_length | 128 | head_dim (no head_dim key exists; key_length carries it) |
| qwen3.attention.layer_norm_rms_epsilon | 1e-6 (an f32; parse it as f32) | rms_norm_eps |
| qwen3.rope.freq_base | 1000000.0 | rope_theta |
| qwen3.context_length | 40960 | max_position_embeddings |
| tokenizer.ggml.eos_token_id | 151645 | into TokenizerData's stop set (7.8) |

Absent keys get spec defaults or nothing: `general.alignment` absent
means 32; there is no vocab_size key, the tokens array's length
(151,936) is the vocab size. Missing required keys are errors; keys
appear in any order, so the parser matches by name, never by
position.

The rest of the metadata feeds later blocks: the tokenizer arrays
(tokens, merges, token_type — rebuilt into a tokenizer in
[The Tokenizer](05-tokenizer.md)) and a 4100-character ChatML chat
template string ([The CLI](10-cli.md)).

Two landmines flagged now because each costs a debugging day later:

1. `tokenizer.ggml.bos_token_id` exists (151643) while
   `tokenizer.ggml.add_bos_token` is false. Qwen3 never prepends BOS;
   reading the id and "helpfully" using it changes every output.
2. the metadata under-reports stopping: it carries only
   `eos_token_id: 151645`, but generation must also stop on 151643
   (`<|endoftext|>`) — the base release's generation_config lists
   both. The gate's answer: assemble the full stop set at load
   (the metadata's eos, plus `<|endoftext|>` looked up by string in
   the tokens array) and hand it over as `stop_token_ids` (7.8), so
   the loop never consults the file's quirks.

## 7.4 The tensors

310 tensors: 2 global + 28 layers x 11. The 197 matmul weights are
Q8_0; the 113 one-dimensional norm weights stay F32 (the file's own
`general.file_type = 7` means "mostly Q8_0, except 1-d tensors").

| GGUF name | dims (as stored) | type | HF equivalent |
|---|---|---|---|
| token_embd.weight | [1024, 151936] | Q8_0 | model.embed_tokens.weight |
| output_norm.weight | [1024] | F32 | model.norm.weight |
| blk.N.attn_norm.weight | [1024] | F32 | input_layernorm |
| blk.N.attn_q.weight | [1024, 2048] | Q8_0 | q_proj |
| blk.N.attn_k.weight | [1024, 1024] | Q8_0 | k_proj |
| blk.N.attn_v.weight | [1024, 1024] | Q8_0 | v_proj |
| blk.N.attn_q_norm.weight | [128] | F32 | q_norm |
| blk.N.attn_k_norm.weight | [128] | F32 | k_norm |
| blk.N.attn_output.weight | [2048, 1024] | Q8_0 | o_proj |
| blk.N.ffn_norm.weight | [1024] | F32 | post_attention_layernorm |
| blk.N.ffn_gate.weight | [1024, 3072] | Q8_0 | mlp.gate_proj |
| blk.N.ffn_up.weight | [1024, 3072] | Q8_0 | mlp.up_proj |
| blk.N.ffn_down.weight | [3072, 1024] | Q8_0 | mlp.down_proj |

Two traps, both fatal and both silent if missed:

1. dims are REVERSED relative to HF: dims[0] is the contiguous
   dimension (the row length). `token_embd.weight` is stored
   [1024, 151936] where HF says [151936, 1024]. The loader reverses
   into our row-major shape; forgetting this transposes the entire
   model and produces fluent garbage;
2. there is NO `output.weight` in this file. The lm head is tied:
   logits come from `token_embd.weight`. The representation: the
   loader inserts no alias entry, the tensors map holds exactly the
   file's 310, and the family reads `token_embd.weight` for both the
   embedding lookup and the logits matvec. Reference by name, zero
   copies of a 622 MB tensor. llama.cpp does the same fallback.

## 7.5 Q8_0, the first quant type

```text
one block: 34 bytes, 32 values

[ d: f16 ][ q0 | q1 | ... | q31 : i8 each ]

value_j = f32(d) * q_j
```

A row of k elements is k/32 consecutive blocks (every Q8_0 tensor
here has k a multiple of 32). The scale `d` is an IEEE 754 half; the
half crate converts it, which is why that dependency exists. At load,
each block dequantizes to 32 f32 values; computing on packed blocks
directly is Part III's fused-kernel work.

The layout rules can be proven from the file with plain arithmetic:
token_embd at offset 4096 occupies 151936 x 1024 / 32 blocks x 34
bytes = 165,306,368, ending exactly at the next tensor's offset; the
last tensor's end lands exactly at byte 639,446,688, the file size.
The loader's tests pin this arithmetic.

## 7.6 The config struct

`Qwen3Config` is the typed twin of 7.3's metadata table: one field
per key, filled at load, no JSON anywhere. Missing key, wrong type,
or `general.architecture != "qwen3"` each refuse with a named error.

```rust
pub struct Qwen3Config {
    pub num_hidden_layers: u32,       // qwen3.block_count           = 28
    pub hidden_size: u32,             // qwen3.embedding_length      = 1024
    pub intermediate_size: u32,       // qwen3.feed_forward_length   = 3072
    pub num_attention_heads: u32,     // qwen3.attention.head_count  = 16
    pub num_key_value_heads: u32,     // ...head_count_kv            = 8
    pub head_dim: u32,                // ...key_length               = 128
    pub rms_norm_eps: f32,            // ...layer_norm_rms_epsilon   = 1e-6
    pub rope_theta: f32,              // qwen3.rope.freq_base        = 1e6
    pub max_position_embeddings: u32, // qwen3.context_length        = 40960
    pub vocab_size: u32,              // tokens array length         = 151936
}
```

It reaches the engine wrapped as `FamilyConfig::Qwen3` (7.8), so the
qwen3-specific part of the bundle is exactly one enum variant, and
nothing family-flavored leaks anywhere else.

> **Rust: `Result<T, E>` and `?`.** A function that can fail returns
> `Result`: `Ok(value)` or `Err(error)`; `?` unwraps the `Ok` or
> returns the `Err` to the caller early. Sibling: `Option<T>` for
> absence without an error. [The Book, ch. 9.2](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html)

## 7.7 The gate

The loader is the engine's gate. It runs at the very beginning, once,
and exactly two things can come out: a complete `Yamf`, or a refusal
that names the problem. Support is decided here and nowhere deeper:
if the file's family does not match, the gate answers "architecture
llama; nucleon supports: qwen3" and stops. Nothing half-loaded ever
reaches a forward pass.

Every refusal carries three parts: what we expected, what the file
actually held, and, in their difference, why this model cannot be
supported.

The world can fail (files are the world), so loading returns
`Result<_, LoaderError>`; panics stay reserved for caller bugs, the
rule Tensor 0 set.

> **Rust: `enum` and `match`.** An enum value is exactly one of its
> listed variants, each optionally carrying data; `match` forces
> handling every variant. Sibling: `if let` for one-variant checks.
> [The Book, ch. 6](https://doc.rust-lang.org/book/ch06-00-enums.html)

Every variant carries what was found AND what is supported, so the
refusal message doubles as compatibility documentation:

| check | refusal |
|---|---|
| magic | "not a GGUF file" |
| version | "GGUF version {found}; nucleon supports 3" |
| general.architecture | "architecture {found}; nucleon supports: qwen3" |
| tensor type | "{name} is {type}; supported: F32, Q8_0" |
| metadata | "missing key {key}" / "key {key} has type {found}, expected {want}" |
| structure | out-of-bounds string length or tensor offset, misaligned offset, overlapping ranges, bool byte not 0/1 |

The contract on what a loaded model must contain:

1. exactly the expected tensor set, generated from block_count: 2
   global + 28 x 11 = 310 names, each with the exact (reversed,
   checked) shape; missing or extra tensors are errors by name;
2. dims reversed into row-major shapes before any shape check;
3. Q8_0 dequantized to f32 once, during the single pass over the
   mmap; F32 tensors copied straight out;
4. logits weight = the embedding tensor, by reference (7.4 trap 2).

## 7.8 Yamf, what crosses the border

The loader's output struct has a name: `Yamf`, for Yet Another
Model Format. The name is a joke with the irony fully intended: a
YAMF is precisely not a format (it lives only in memory, is never
serialized, and has no version bytes). It commemorates the format zoo of
[Reading the Model](03-reading-the-model.md)'s 6.4 while refusing to
join it.

The rule it enforces: nothing GGUF-shaped crosses the border. The
loader distills the file into exactly what the engine needs, in
plain types; whatever else the file carries is dropped; no other
module ever receives raw metadata. This is how it looks:

```rust
pub struct Yamf {
    pub family: FamilyConfig,
    pub tensors: HashMap<String, Tensor>,
    pub tokenizer: TokenizerData,
    pub chat_template: ChatTemplate,
}

pub enum FamilyConfig {
    Qwen3(Qwen3Config),
    // a second family = a second variant; nothing else in Yamf moves
}

pub struct TokenizerData {
    pub tokens: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub token_types: Vec<TokenType>,
    pub pre: String,
    pub stop_token_ids: Vec<u32>,
}

pub fn load(path: &Path) -> Result<Yamf, LoaderError>
```

Each piece, what it is for, and where the idea was stolen from:

| piece | holds | consumed by | stolen from |
|---|---|---|---|
| `family` | the ONE family-specific corner: a tag (which family) wrapping that family's typed numbers (`Qwen3Config`: u32 layer count, f32 epsilon). Everything else in Yamf is family-agnostic | the family's forward pass | GGUF's architecture tag (a tag namespaces the rest) becomes the enum tag; GGUF's typed metadata becomes compiler-checked fields |
| `tensors` | all 310 weights as f32 `Tensor`s, dims un-reversed, shapes already checked | the forward pass | safetensors' up-front inventory: the full tensor list is verified complete before this struct can exist |
| `tokenizer` | tokens, merges, token types, the pre id, and the stop set the gate assembled (151645 and 151643; 7.3 landmine 2), as plain vectors | The Tokenizer chapter, the stop set by The Loop | GGUF's bundle idea: the tokenizer travels with the weights, nothing external needed |
| `tokenizer.pre` | "qwen2", the tag naming which split regex to build | the tokenizer build | GGUF's architecture-tag pattern: a small tag tells you how to read the rest |
| `chat_template` | the ChatML template, parsed and validated at the gate: `ChatTemplate` wraps a [minijinja](https://docs.rs/minijinja) environment holding the compiled template (minijinja is the established pure-Rust Jinja engine, by Jinja's original author). A broken template fails at load, not at first chat | the chat module (the CLI calls it later) | GGUF's bundle again, plus the gate philosophy: validate everything the moment it enters |
| the struct as a whole | inert: no file handles, no logic; IO is fully over when `load` returns. Nothing from the file ever runs as code; the chat template is parsed into a sandboxed description at the gate, which is validation, not execution | everything downstream | safetensors' dumbness-as-a-virtue, plus two rejections: ONNX's program-carrying (family code is our program) and pickle's executability |

`TokenizerData`'s shape is not qwen3-specific: it is the shape of
every byte-level BPE tokenizer (GGUF's tokenizer model "gpt2"; Llama
3 ships the same shape, different contents). The values are Qwen3's,
and `pre` is the one family-flavored field. A different tokenizer
kind (SentencePiece, GGUF model "llama", scores instead of merges)
would grow a variant here, the same way `FamilyConfig` grows one.

Two properties this struct must keep as it grows:

- extensible: fields are added, never repurposed; the family-specific
  part is already quarantined in `FamilyConfig`, so a second family
  is a new variant and the rest of Yamf never changes;
- versioned by the compiler: an in-memory struct needs no version
  number, because every consumer is type-checked against the current
  definition at build time; drift is impossible. A version field
  becomes necessary the day this struct is ever written to disk,
  which is exactly why we do not have a nucleon disk format.

> **Rust: `HashMap<K, V>`.** The standard hash table: owned keys to
> owned values, `get` returns `Option`. Sibling: `BTreeMap` when
> iteration order matters. [std docs](https://doc.rust-lang.org/std/collections/struct.HashMap.html)

Module layout, one concern per file, each with sibling tests:

| file | concern |
|---|---|
| loader/mod.rs | export barrel only |
| loader/container.rs | 7.2: header, metadata KVs, tensor infos, alignment |
| loader/config.rs | 7.3: metadata keys to Qwen3Config |
| loader/dequant.rs | 7.5: Q8_0 blocks to f32 |
| loader/yamf.rs | 7.7: expected names, checks, the Yamf bundle |

The parser existing also makes `nucleon inspect model.gguf` nearly
free: print version, architecture, dimensions, tensor types, and a
supported/not verdict per check. It lands with the CLI chapter.

## 7.9 What the tests will pin

All on synthetic GGUF files the tests write themselves; no network,
no 640 MB fixture:

1. happy path: a tiny hand-built file (two tensors, one Q8_0, one
   F32) round-trips with exact values;
2. every refusal in 7.7's table, one test each: wrong magic, version
   2 and the big-endian 50,331,648, foreign architecture, Q4_K
   tensor, missing/extra/mistyped metadata keys, lying string
   lengths, misaligned and out-of-bounds offsets, a bool byte of 7;
3. the dims-reversal pin: a [rows, cols] tensor written GGUF-style
   loads with our row-major shape and correct element order;
4. Q8_0 dequant spot values, with fixtures quantized by the reference
   formula (d = amax/127, values rounded), so both directions are
   bit-exact;
5. the epsilon f32 pin: 1e-6 read back must equal the f32 bit
   pattern, not a f64 approximation;
6. the oracle cross-check: the same synthetic files read through
   candle-core (dev-dependency only): header fields, metadata, and
   dequantized values must agree with ours.

The golden test (whole-engine, real file) stays in The Loop's
chapter: HF transformers 4.54.0 or newer loads this exact GGUF via
`gguf_file=`, dequantizes to f32 the same way, and its greedy token
ids become the ids nucleon must reproduce.

## 7.10 Upcoming loader topics

1. Q4_K and friends: more block layouts in dequant.rs (Part III);
2. fused dequant: weights stay packed in the map, kernels read blocks
   directly, dequant-on-load's memory cost disappears (Part III);
3. split GGUF: large models ship as `-00001-of-000NN.gguf` parts; the
   27B needs the multi-file reader;
4. a safetensors adapter, only if a future ADR reopens ADR 004.

Next: the build, test-first, starting with container.rs.
