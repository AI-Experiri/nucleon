# GGUF container + Qwen3-0.6B-Q8_0.gguf — exact facts for the hand parser

Verified 2026-08-14. Sections 2, 3, and parts of 7/8 come from parsing the
first 20 MiB of the actual file `Qwen3-0.6B-Q8_0.gguf` (repo
`Qwen/Qwen3-0.6B-GGUF`, sha `23749fefcc72300e3a2ad315e1317431b06b590a`,
file size 639,446,688 bytes) byte by byte with a throwaway Python script.
llama.cpp line numbers are from `master` on the fetch date.

## 1. GGUF v3 container byte layout

All multi-byte values little-endian by default. Big-endian files exist (for
BE machines) and carry NO in-file marker; "If no additional information is
provided, assume the model is little-endian" (spec). Fields are packed
sequentially with no alignment padding, except the single pad before
tensor data (below).

| offset | field | type | value in our file |
|---|---|---|---|
| 0 | magic | 4 bytes | `0x47 0x47 0x55 0x46` = "GGUF" (as LE u32: `0x46554747`) |
| 4 | version | u32 | 3 (spec: "Must be `3`") |
| 8 | tensor_count | u64 | 310 |
| 16 | metadata_kv_count | u64 | 28 |
| 24 | metadata_kv[kv_count] | | see section 2 |
| ... | tensor_info[tensor_count] | | see section 3 |
| ... | pad with `0x00` to ALIGNMENT | | 28 pad bytes in our file |
| ... | tensor_data | | starts at byte 5,951,136 in our file |

String encoding (`gguf_string_t`): u64 byte length, then that many UTF-8
bytes, NOT null-terminated. Metadata keys are additionally required to be
ASCII, `lower_snake_case` segments joined by `.`, at most 65535 bytes.

Metadata KV entry: key (string), value_type (u32), value. Value types:

| id | type | encoding |
|---|---|---|
| 0 | UINT8 | 1 byte |
| 1 | INT8 | 1 byte |
| 2 | UINT16 | 2 bytes LE |
| 3 | INT16 | 2 bytes LE |
| 4 | UINT32 | 4 bytes LE |
| 5 | INT32 | 4 bytes LE |
| 6 | FLOAT32 | 4 bytes IEEE754 |
| 7 | BOOL | 1 byte, 0=false 1=true, anything else = invalid file |
| 8 | STRING | gguf_string_t |
| 9 | ARRAY | u32 element type id, u64 element count, then packed values; nestable |
| 10 | UINT64 | 8 bytes LE |
| 11 | INT64 | 8 bytes LE |
| 12 | FLOAT64 | 8 bytes IEEE754 |

Tensor-info entry (`gguf_tensor_info_t`), one per tensor, in file order:

| field | type | constraint |
|---|---|---|
| name | string | at most 64 bytes |
| n_dimensions | u32 | currently at most 4 |
| dimensions | u64 x n_dimensions | ggml `ne` order: dims[0] is the contiguous (fastest-varying) dimension, i.e. REVERSED vs HF/PyTorch shape |
| type | u32 | ggml_type id (F32=0, F16=1, Q8_0=8, BF16=30, ...) |
| offset | u64 | relative to start of tensor_data, NOT file start; must satisfy `offset % ALIGNMENT == 0` |

Alignment rule: `general.alignment` (u32, must be a multiple of 8) sets
ALIGNMENT; "If the alignment is not specified, assume it is 32". Our file
does not carry the key, so 32. Tensor data start =
`align_offset(end_of_tensor_infos)` where
`align_offset(o) = o + (ALIGNMENT - o % ALIGNMENT) % ALIGNMENT`; the gap is
zero-padded. Absolute position of a tensor = data_start + info.offset.

Version history (matters for error messages only, we accept v3):
v1 encoded string lengths and array counts as u32 instead of u64; v2
switched to u64; v3 added big-endian support (structure identical to v2).
Candle's reader (section 6) branches exactly this way: v1 reads u32
lengths, v2/v3 read u64.

Dimension-order evidence from the actual file: `token_embd.weight` has
dims `[1024, 151936]` where 1024 = embedding_length (row length) and HF
safetensors stores the same tensor as `[151936, 1024]`.

Sources:
- https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf (bytes 0..20971519 parsed)
- https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/quantized/gguf_file.rs

## 2. Actual metadata of Qwen3-0.6B-Q8_0.gguf (all 28 keys, file order)

| # | key | type | value |
|---|---|---|---|
| 1 | general.architecture | STRING | "qwen3" |
| 2 | general.type | STRING | "model" |
| 3 | general.name | STRING | "Qwen3 0.6B Instruct" |
| 4 | general.finetune | STRING | "Instruct" |
| 5 | general.basename | STRING | "Qwen3" |
| 6 | general.size_label | STRING | "0.6B" |
| 7 | qwen3.block_count | UINT32 | 28 |
| 8 | qwen3.context_length | UINT32 | 40960 |
| 9 | qwen3.embedding_length | UINT32 | 1024 |
| 10 | qwen3.feed_forward_length | UINT32 | 3072 |
| 11 | qwen3.attention.head_count | UINT32 | 16 |
| 12 | qwen3.attention.head_count_kv | UINT32 | 8 |
| 13 | qwen3.rope.freq_base | FLOAT32 | 1000000.0 |
| 14 | qwen3.attention.layer_norm_rms_epsilon | FLOAT32 | 1e-06 (stored f32; reads back as 9.999999974752427e-07 in f64) |
| 15 | qwen3.attention.key_length | UINT32 | 128 |
| 16 | qwen3.attention.value_length | UINT32 | 128 |
| 17 | tokenizer.ggml.model | STRING | "gpt2" |
| 18 | tokenizer.ggml.pre | STRING | "qwen2" |
| 19 | tokenizer.ggml.tokens | ARRAY[STRING] | len 151936; first: "!", "\"", "#", "$", "%"; last 4: "[PAD151932]".."[PAD151935]" |
| 20 | tokenizer.ggml.token_type | ARRAY[INT32] | len 151936; first entries 1 (normal); last 4 entries 5 (unused) |
| 21 | tokenizer.ggml.merges | ARRAY[STRING] | len 151387; first: "Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t", "ĠĠĠĠ ĠĠĠĠ" |
| 22 | tokenizer.ggml.eos_token_id | UINT32 | 151645 |
| 23 | tokenizer.ggml.padding_token_id | UINT32 | 151643 |
| 24 | tokenizer.ggml.bos_token_id | UINT32 | 151643 |
| 25 | tokenizer.ggml.add_bos_token | BOOL | false |
| 26 | tokenizer.chat_template | STRING | present, 4100 chars; first 200: `{%- if tools %}\n    {{- '<|im_start|>system\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\n\n' }}\n    {%- endif %}\n    {{- "# Tools\n\nYou may call one or more functi` |
| 27 | general.quantization_version | UINT32 | 2 |
| 28 | general.file_type | UINT32 | 7 = MOSTLY_Q8_0 ("except 1d tensors") |

Keys ABSENT that a parser must default: `general.alignment` (use 32),
`tokenizer.ggml.add_eos_token`, `tokenizer.ggml.scores`, any
`qwen3.rope.scaling.*`, any explicit head_dim key (use
attention.key_length / value_length = 128).

Chat template facts (from the full 4100-char string): ChatML
(`<|im_start|>{role}\n...<|im_end|>\n`); tools branch emits a `# Tools`
system block with `<tool_call>` JSON instructions; contains
`enable_thinking` logic; ends with: on `add_generation_prompt` emit
`<|im_start|>assistant\n`, and if `enable_thinking is defined and
enable_thinking is false` also emit `<think>\n\n</think>\n\n`.

The HF page `?show_file_info=Qwen3-0.6B-Q8_0.gguf` renders the parse
client-side only (not fetchable as HTML); the API endpoint returns a
server-side `gguf` summary object (architecture "qwen3", context_length
40960, total 596049920 params, full chat_template) that matches the
byte-level parse above.

Sources:
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf (bytes 0..20971519 parsed)
- https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF
- https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF/tree/main
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF?show_file_info=Qwen3-0.6B-Q8_0.gguf
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py (LlamaFileType.MOSTLY_Q8_0 = 7, GGML_QUANT_VERSION = 2)

## 3. Full tensor list (310 tensors)

310 = 2 global + 28 layers x 11. Dims below are GGUF `ne` order
(dims[0] = row length = input dim for matmul weights). Q8_0: 197 tensors,
F32: 113 tensors (all 1-d norms stay F32).

| name | dims | type |
|---|---|---|
| token_embd.weight | [1024, 151936] | Q8_0 |
| output_norm.weight | [1024] | F32 |
| blk.N.attn_norm.weight | [1024] | F32 |
| blk.N.attn_q.weight | [1024, 2048] | Q8_0 |
| blk.N.attn_k.weight | [1024, 1024] | Q8_0 |
| blk.N.attn_v.weight | [1024, 1024] | Q8_0 |
| blk.N.attn_q_norm.weight | [128] | F32 |
| blk.N.attn_k_norm.weight | [128] | F32 |
| blk.N.attn_output.weight | [2048, 1024] | Q8_0 |
| blk.N.ffn_norm.weight | [1024] | F32 |
| blk.N.ffn_gate.weight | [1024, 3072] | Q8_0 |
| blk.N.ffn_up.weight | [1024, 3072] | Q8_0 |
| blk.N.ffn_down.weight | [3072, 1024] | Q8_0 |

`output.weight` DOES NOT EXIST in this file: the lm head is tied.
llama.cpp's qwen3 loader marks output as optional and falls back
(src/models/qwen3.cpp lines 22-26, verbatim):

```c
output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
// if output is NULL, init from the input tok embed
if (output == NULL) {
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
}
```

Whole-file arithmetic check (proves the layout rules): data_start
5,951,136 + last offset 630,153,216 (blk.27.ffn_up) + its size
(1024*3072/32 blocks * 34 B = 3,342,336) = 639,446,688 = exact file size.
Also token_embd at offset 4096 with 151936*1024/32*34 = 165,306,368 bytes
ends at 165,310,464 = next tensor's offset (no padding needed, 34-byte
blocks landed aligned).

Canonical qwen3 tensor set in gguf-py (`MODEL_ARCH.QWEN3` in
constants.py): TOKEN_EMBD, OUTPUT_NORM, OUTPUT, ROPE_FREQS, ATTN_NORM,
ATTN_Q, ATTN_Q_NORM, ATTN_K, ATTN_K_NORM, ATTN_V, ATTN_OUT, FFN_NORM,
FFN_GATE, FFN_DOWN, FFN_UP. OUTPUT and ROPE_FREQS are allowed but absent
here. Name strings (constants.py TENSOR_NAMES): `token_embd`,
`output_norm`, `output`, `blk.{bid}.attn_q`, `blk.{bid}.attn_k`,
`blk.{bid}.attn_v`, `blk.{bid}.attn_output`, `blk.{bid}.attn_q_norm`,
`blk.{bid}.attn_k_norm`, `blk.{bid}.attn_norm`, `blk.{bid}.ffn_norm`,
`blk.{bid}.ffn_gate`, `blk.{bid}.ffn_up`, `blk.{bid}.ffn_down`, each plus
`.weight`.

HF name -> GGUF name (tensor_mapping.py, llama-hf rows):

| HF (Qwen3ForCausalLM) | GGUF |
|---|---|
| model.embed_tokens | token_embd |
| lm_head | output |
| model.norm | output_norm |
| model.layers.N.input_layernorm | blk.N.attn_norm |
| model.layers.N.self_attn.{q,k,v}_proj | blk.N.attn_{q,k,v} |
| model.layers.N.self_attn.{q,k}_norm | blk.N.attn_{q,k}_norm |
| model.layers.N.self_attn.o_proj | blk.N.attn_output |
| model.layers.N.post_attention_layernorm | blk.N.ffn_norm |
| model.layers.N.mlp.{gate,up,down}_proj | blk.N.ffn_{gate,up,down} |

Sources:
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf (bytes 0..20971519 parsed)
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/tensor_mapping.py
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen3.cpp

## 4. Q8_0 block layout

`QK8_0 = 32` elements per block, 34 bytes per block (`GGML_QUANT_SIZES`
gives `(32, 2 + 32)`). Struct from ggml-common.h, verbatim:

```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

Field order: 2-byte IEEE754 half `d` (the scale) FIRST, then 32 signed
int8 values. A row of k elements is k/32 consecutive blocks; k must be a
multiple of 32 (holds here: every Q8_0 tensor's dims[0] is
1024/2048/3072).

Dequant (ggml-quants.c `dequantize_row_q8_0`): `y[i*32+j] =
fp16_to_fp32(block[i].d) * block[i].qs[j]`.

Quant reference, for building bit-exact test fixtures (ggml-quants.c
`quantize_row_q8_0_ref`): per block `d = amax / 127` where amax =
max(|x|); `id = d ? 1/d : 0`; `qs[j] = roundf(x[j] * id)`; `d` stored as
fp16. gguf-py implements the same and states "bit-exact same results as
reference implementation in ggml-quants.c".

Sources:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/quants.py

## 5. HF transformers as golden-test oracle

- Yes: `AutoModelForCausalLM.from_pretrained(repo, gguf_file="...")` and
  `AutoTokenizer.from_pretrained(repo, gguf_file="...")`. Requires
  `pip install gguf`.
- Doc statement (current, docs v5.14.0): "The GGUF checkpoint is
  **dequantized to fp32** where the full model weights are available and
  compatible with PyTorch."
- Internals (modeling_gguf_pytorch_utils.py): `from gguf import
  GGUFReader, dequantize`; per tensor `weights = dequantize(tensor.data,
  tensor.tensor_type)` then `torch.from_numpy(...)`. Quant coverage is
  therefore whatever gguf-py dequantizes; Q8_0 has a dedicated class in
  gguf-py quants.py. Covered.
- qwen3 arch: present in integrations/ggml.py (`GGUF_TO_FAST_CONVERTERS
  ["qwen3"] = GGUFQwen2Converter`, plus a "qwen3" config map). Added by
  PR #38645 (merged 2025-07-15). Version check against tags: v4.53.0 has
  zero `"qwen3"` matches in ggml.py, v4.54.0 has them. Minimum
  transformers version for qwen3 + GGUF: 4.54.0.
- Oracle recipe: load with `dtype=torch.float32`, greedy-generate on a
  fixed prompt, record token ids; nucleon's golden test must reproduce
  them. Since transformers dequantizes ONCE to fp32 and runs fp32 matmuls
  while nucleon computes on Q8_0 blocks, logits differ in the last ulps;
  compare token ids, not logits, or compare per-tensor dequant output
  exactly (both sides implement the same formula).

Sources:
- https://huggingface.co/docs/transformers/gguf
- https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/gguf.md
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_gguf_pytorch_utils.py
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/integrations/ggml.py
- https://github.com/huggingface/transformers/pull/38645
- https://raw.githubusercontent.com/huggingface/transformers/v4.54.0/src/transformers/integrations/ggml.py (and v4.53.0 for the negative check)
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/quants.py

## 6. Rust crates that read GGUF (dev-dependency oracle candidates)

crates.io search "gguf", 2026-08-14 state:

| crate | version | updated | downloads | note |
|---|---|---|---|---|
| candle-core | 0.11.0 | 2026-06-26 | 6.99M | `quantized::gguf_file` reader + dequant |
| gguf-rs-lib | 0.3.0 | 2026-08-03 | 31.8k | read/write GGUF library |
| gguf-rs | 0.1.8 | 2026-06-15 | 19.6k | parser + CLI |
| gguf | 0.1.2 | 2023-09-12 | 12.3k | stale since 2023 |
| gguf-utils | 0.4.1 | 2025-07-22 | 6.5k | utilities |
| rlx-gguf | 0.2.14 | 2026-08-12 | 4.3k | v1/v2/v3 parser + dequant to f32, standalone |

Recommendation: **candle-core** (dev-dependency only). One crate gives
both halves of the oracle: `quantized::gguf_file::Content::read(&mut
reader)` returns `magic`, `metadata: HashMap<String, Value>`,
`tensor_infos: HashMap<String, TensorInfo>`, `tensor_data_offset` to
cross-check our header parser field by field, and
`Content::tensor(reader, name, device)` -> `QTensor` with
`QTensor::dequantize(&Device) -> Tensor` to cross-check our Q8_0 dequant
numerically; it is by far the most exercised implementation (6.99M
downloads, HF-maintained) and handles GGUF v1/v2/v3.

Sources:
- https://crates.io/api/v1/crates?q=gguf&per_page=20
- https://crates.io/api/v1/crates/candle-core
- https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/quantized/gguf_file.rs
- https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/quantized/mod.rs

## 7. Tokenizer reconstruction from GGUF metadata

This file: `tokenizer.ggml.model = "gpt2"` (byte-level BPE),
`tokenizer.ggml.pre = "qwen2"`.

llama.cpp maps pre id "qwen2" to `LLAMA_VOCAB_PRE_TYPE_QWEN2` with
`clean_spaces = false` (llama-vocab.cpp lines 2216-2221), and that pre
type selects exactly one split regex (llama-vocab.cpp lines 373-381,
C string verbatim, shared with STABLELM2/HUNYUAN/SOLAR_OPEN):

```c
// original regex from tokenizer.json
// "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
```

Qwen's original tokenizer.json (Qwen/Qwen3-0.6B, fetched) pre_tokenizer,
verbatim:

```json
"pre_tokenizer": {
  "type": "Sequence",
  "pretokenizers": [
    { "type": "Split",
      "pattern": { "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" },
      "behavior": "Isolated", "invert": false },
    { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false }
  ]
}
```

with decoder and post_processor both `ByteLevel` (same three flags).

Mismatch risk: llama.cpp rewrites the leading `(?i:'s|'t|'re|...)` group
into explicit case classes `(?:'[sS]|'[tT]|'[rR][eE]|...)` because its
regex engine lacks inline case-insensitive groups. The character-class
form accepts the same strings (including mixed case 'rE, 'Re), so the two
regexes are equivalent on the contraction alternatives; the rest of the
pattern is byte-identical to the tokenizer.json original. Use the
ORIGINAL tokenizer.json form in nucleon. Both need lookahead support for
`\s+(?!\S)`: the plain `regex` crate cannot run it; `fancy-regex` or onig
can.

Rebuild with the HF tokenizers Rust crate (in-memory, no tokenizer.json):
- vocab map: `tokenizer.ggml.tokens` index -> id, as `HashMap<String,
  u32>`; merges: each `tokenizer.ggml.merges` entry is one string
  "left right", split ONCE on the single space into `(String, String)`
  (byte-level tokens encode real spaces as `Ġ`, so the separator is
  unambiguous).
- model: `BPE::builder().vocab_and_merges(vocab, merges).build()`
  (BpeBuilder also offers `unk_token`, `byte_fallback`, `ignore_merges`;
  leave all off for this vocab).
- pre-tokenizer: Sequence of Split(original regex, behavior Isolated) +
  `ByteLevel` with `add_prefix_space(false)`, `use_regex(false)`
  (trim_offsets false); decoder: `ByteLevel` (its byte-alphabet mapping
  is the documented `ByteLevel::alphabet()`).
- special tokens: register the 26 non-NORMAL entries (section 8) as added
  tokens so they match before BPE; llama.cpp builds its special-token
  cache from exactly the CONTROL | USER_DEFINED | UNKNOWN attr bits
  (llama-vocab.cpp lines 2951-2955).

PURE-RUST TRAP: the tokenizers crate default features are `["progressbar",
"onig", "esaxx_fast"]` and onig binds the C library Oniguruma. Depend on
it with `default-features = false, features = ["fancy-regex"]`
(fancy-regex is pure Rust); src/utils/mod.rs compiles the fancy backend
when `fancy-regex` is on and `onig` off, and errors if neither is set.

Sources:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-vocab.cpp
- https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json (bytes 0..300000)
- https://docs.rs/tokenizers/latest/tokenizers/models/bpe/struct.BpeBuilder.html
- https://docs.rs/tokenizers/latest/tokenizers/pre_tokenizers/byte_level/struct.ByteLevel.html
- https://raw.githubusercontent.com/huggingface/tokenizers/main/tokenizers/Cargo.toml
- https://raw.githubusercontent.com/huggingface/tokenizers/main/tokenizers/src/utils/mod.rs
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf (bytes 0..20971519 parsed)

## 8. Surprises / pitfalls

1. Dims are REVERSED vs HF: `[in, out]` with dims[0] contiguous. Reading
   `attn_q` as [1024, 2048] and treating it like a PyTorch [out, in]
   matrix silently transposes the whole model.
2. No `output.weight`: logits must reuse `token_embd.weight` (tied).
   llama.cpp duplicates tok_embd when the tensor is missing; nucleon
   should reference, not copy.
3. `tokenizer.ggml.bos_token_id = 151643` EXISTS while
   `tokenizer.ggml.add_bos_token = false`. Reading the bos id and
   prepending it changes every output. Never prepend.
4. rms epsilon is an f32; in f64 it is 9.999999974752427e-07, not 1e-6.
   Exact-compare against the f32 bit pattern or parse into f32.
5. Tensor `offset` is relative to tensor_data, not the file; tensor_data
   itself starts only after zero-padding the header to ALIGNMENT (28 pad
   bytes here). `general.alignment` is absent; default 32.
6. token_type semantics (spec): 1=normal, 2=unknown, 3=control,
   4=user_defined, 5=unused, 6=byte. In this file ids 151643..151656 and
   151659..151664 are CONTROL (`<|endoftext|>`, `<|im_start|>`,
   `<|im_end|>`, vision/FIM tokens); `<tool_call>`, `</tool_call>`,
   `<tool_response>`, `</tool_response>`, `<think>`, `</think>`
   (151657/151658/151665/151666/151667/151668) are USER_DEFINED; rows
   151669..151935 are UNUSED "[PADnnnnn]" fillers. Never sample UNUSED
   rows; match CONTROL and USER_DEFINED strings before BPE.
7. Merges arrive as single strings "A B"; a token can never contain a
   literal space (byte-level uses Ġ), so splitting on the first space is
   safe. vocab len 151936 vs merges len 151387: the 549 difference is
   exactly 256 byte-alphabet base tokens + 26 special tokens
   (151643..151668) + 267 UNUSED pads (151669..151935); do not expect
   len(vocab) == len(merges) + 256.
8. The chat template hardcodes ChatML, tools JSON, and the
   `enable_thinking is false` empty-think-block trick; `enable_thinking`
   is a Jinja render variable, not a metadata key. EOS in metadata is
   only 151645; the base model's generation_config stops on BOTH 151645
   and 151643 (see docs/research/qwen3-0.6b.md), and 151643 is carried
   here as "padding"/"bos" only. Stop on both anyway.
9. Q8_0 scale is fp16: a hand parser needs a half-to-f32 conversion
   before the first multiply (the `half` crate, or a 10-line manual
   conversion).
10. Strings carry u64 lengths; bound-check them against file size before
    allocating (candle's read_string does exactly this). BOOL bytes
    other than 0/1 mean a corrupt file per spec.
11. No endianness marker exists. After reading magic, require
    version == 3; a big-endian file would show version 0x03000000 =
    50331648 and should be rejected with a clear message.
12. `general.file_type = 7` says MOSTLY_Q8_0, "except 1d tensors": all
    113 one-dimensional norm weights are plain F32. A loader that assumes
    one dtype for the whole file breaks on the second tensor.
13. gguf-py's TokenType and file-order metadata make this file
    self-describing, but keys can appear in ANY order; parse by key name,
    never by index (this file happens to interleave general.* keys first
    and last).
14. The HF `?show_file_info=` viewer needs JavaScript; scripted
    verification should use `/api/models/Qwen/Qwen3-0.6B-GGUF` (has a
    server-parsed `gguf` object) or parse the file bytes directly.

Sources: sections 1-7 above (same URLs), plus
- https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md (token_type semantics, BOOL rule, alignment default)
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf (bytes 0..20971519 parsed)
