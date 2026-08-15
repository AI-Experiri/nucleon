use super::*;
use crate::loader::gguf_builder::{f32_bytes, quantize_q8_0_ref, GgufBuilder, GGML_F32, GGML_Q8_0};

// A 1-layer qwen3 mini model: hidden 32, ffn 64, 2 heads / 1 kv
// head, head_dim 16, vocab 10. Every required tensor present;
// ffn_gate is the Q8_0 one — its rows are 32 wide because Q8_0
// blocks never span rows, and the fixture's every block contains
// -127 so the roundtrip is integer-exact.
fn mini() -> GgufBuilder {
    let hidden = 32u64;
    let ffn = 64u64;
    let q_rows = 32u64; // 2 heads x head_dim 16
    let kv_rows = 16u64; // 1 kv head x 16
    let vocab = 10u64;

    let tokens: Vec<String> = (0..8)
        .map(|i| format!("tok{i}"))
        .chain(["<|endoftext|>".to_string(), "<|im_end|>".to_string()])
        .collect();
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let types: Vec<i32> = vec![1, 1, 1, 1, 1, 5, 4, 6, 3, 3];

    // embedding data: value = row * 100 + col, to pin the layout
    let embd: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i / hidden) * 100 + (i % hidden)) as f32)
        .collect();
    // ffn_gate: every 32-value block holds -127, so d = 1.0 exactly
    let gate: Vec<f32> = (0..ffn * hidden)
        .map(|i| ((i % 32) * 4) as f32 - 127.0)
        .collect();

    GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("general.quantization_version", 2)
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", hidden as u32)
        .kv_u32("qwen3.feed_forward_length", ffn as u32)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 16)
        .kv_u32("qwen3.attention.value_length", 16)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "qwen2")
        .kv_u32("tokenizer.ggml.eos_token_id", 9)
        .kv_arr_str("tokenizer.ggml.tokens", &token_refs)
        .kv_arr_i32("tokenizer.ggml.token_type", &types)
        .kv_arr_str("tokenizer.ggml.merges", &["a b", "cc dd"])
        .kv_str("tokenizer.chat_template", "{{ messages }}")
        // ne order: dims[0] = row length
        .tensor(
            "token_embd.weight",
            &[hidden, vocab],
            GGML_F32,
            f32_bytes(&embd),
        )
        .tensor(
            "output_norm.weight",
            &[hidden],
            GGML_F32,
            f32_bytes(&[1.0; 32]),
        )
        .tensor(
            "blk.0.attn_norm.weight",
            &[hidden],
            GGML_F32,
            f32_bytes(&[1.0; 32]),
        )
        .tensor(
            "blk.0.attn_q.weight",
            &[hidden, q_rows],
            GGML_F32,
            f32_bytes(&[0.5; 1024]),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[hidden, kv_rows],
            GGML_F32,
            f32_bytes(&[0.5; 512]),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[hidden, kv_rows],
            GGML_F32,
            f32_bytes(&[0.5; 512]),
        )
        .tensor(
            "blk.0.attn_q_norm.weight",
            &[16],
            GGML_F32,
            f32_bytes(&[1.0; 16]),
        )
        .tensor(
            "blk.0.attn_k_norm.weight",
            &[16],
            GGML_F32,
            f32_bytes(&[1.0; 16]),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[q_rows, hidden],
            GGML_F32,
            f32_bytes(&[0.5; 1024]),
        )
        .tensor(
            "blk.0.ffn_norm.weight",
            &[hidden],
            GGML_F32,
            f32_bytes(&[1.0; 32]),
        )
        .tensor(
            "blk.0.ffn_gate.weight",
            &[hidden, ffn],
            GGML_Q8_0,
            quantize_q8_0_ref(&gate),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[hidden, ffn],
            GGML_F32,
            f32_bytes(&[0.25; 2048]),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[ffn, hidden],
            GGML_F32,
            f32_bytes(&[0.25; 2048]),
        )
}

#[test]
fn loads_a_complete_mini_model() {
    let yamf = load_bytes(&mini().build()).unwrap();
    let FamilyConfig::Qwen3(cfg) = &yamf.family;
    assert_eq!(cfg.num_hidden_layers, 1);
    assert_eq!(cfg.vocab_size, 10);
    assert_eq!(yamf.tensors.len(), 13);
    assert_eq!(yamf.tokenizer.tokens.len(), 10);
    assert_eq!(yamf.tokenizer.pre, "qwen2");
    assert_eq!(yamf.tokenizer.merges[0], ("a".to_string(), "b".to_string()));
    assert_eq!(yamf.tokenizer.token_types[5], TokenType::Unused);
    assert_eq!(yamf.tokenizer.token_types[8], TokenType::Control);
    assert_eq!(yamf.chat_template.source(), "{{ messages }}");
}

#[test]
fn dims_reverse_into_row_major_and_data_stays_put() {
    let yamf = load_bytes(&mini().build()).unwrap();
    let embd = &yamf.tensors["token_embd.weight"];
    // ne [32, 10] becomes ours [10, 32]: 10 vocab rows of 32 values
    assert_eq!(embd.shape(), &[10, 32]);
    // the bytes are NOT moved: row r, col c = r*100 + c
    assert_eq!(embd.at(&[0, 0]), 0.0);
    assert_eq!(embd.at(&[3, 5]), 305.0);
    assert_eq!(embd.at(&[9, 31]), 931.0);
}

#[test]
fn q8_0_tensor_dequantizes_exactly() {
    let yamf = load_bytes(&mini().build()).unwrap();
    let gate = &yamf.tensors["blk.0.ffn_gate.weight"];
    assert_eq!(gate.shape(), &[64, 32]);
    // every fixture block has amax 127 (d = 1.0): exact integers back
    assert_eq!(gate.at(&[0, 0]), -127.0);
    assert_eq!(gate.at(&[0, 1]), -123.0);
}

#[test]
fn stop_set_is_eos_plus_endoftext() {
    let yamf = load_bytes(&mini().build()).unwrap();
    // eos = 9 (<|im_end|>), <|endoftext|> sits at index 8
    assert_eq!(yamf.tokenizer.stop_token_ids, vec![9, 8]);
}

#[test]
fn tied_file_has_no_output_weight_and_untied_is_accepted() {
    let yamf = load_bytes(&mini().build()).unwrap();
    assert!(!yamf.tensors.contains_key("output.weight"));

    // an untied size ships output.weight [vocab, hidden] (ne [32, 10])
    let untied: Vec<f32> = vec![0.0; 320];
    let b = mini().tensor("output.weight", &[32, 10], GGML_F32, f32_bytes(&untied));
    let yamf = load_bytes(&b.build()).unwrap();
    assert!(yamf.tensors.contains_key("output.weight"));
}

#[test]
fn missing_and_unexpected_and_misshapen_tensors_are_named() {
    // missing: a minimal build that stops after two tensors
    let mut without = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "qwen2")
        .kv_u32("tokenizer.ggml.eos_token_id", 1)
        .kv_arr_str("tokenizer.ggml.tokens", &["a", "b"])
        .kv_arr_i32("tokenizer.ggml.token_type", &[1, 1])
        .kv_arr_str("tokenizer.ggml.merges", &[])
        .kv_str("tokenizer.chat_template", "x");
    without = without.tensor(
        "token_embd.weight",
        &[8, 2],
        GGML_F32,
        f32_bytes(&[0.0; 16]),
    );
    match load_bytes(&without.build()) {
        Err(LoaderError::MissingTensor { .. }) => {}
        other => panic!("{:?}", other.err()),
    }

    // unexpected tensor
    let b = mini().tensor("mystery.weight", &[32], GGML_F32, f32_bytes(&[0.0; 32]));
    match load_bytes(&b.build()) {
        Err(LoaderError::UnexpectedTensor { name }) => assert_eq!(name, "mystery.weight"),
        other => panic!("{:?}", other.err()),
    }

    // wrong shape: attn_q with swapped dims — build mini by hand is
    // costly, so mutate via a fresh builder is skipped; instead ship
    // output.weight with a wrong shape, the cheap misshape probe.
    let b = mini().tensor("output.weight", &[10, 32], GGML_F32, f32_bytes(&[0.0; 320]));
    match load_bytes(&b.build()) {
        Err(LoaderError::WrongShape { name, want, found }) => {
            assert_eq!(name, "output.weight");
            assert_eq!(want, vec![10, 32]); // ours: [vocab, hidden]
            assert_eq!(found, vec![32, 10]); // ne [10, 32] reversed
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn unsupported_quant_type_is_refused_by_name() {
    let b = mini().tensor("output.weight", &[32, 10], 2 /* Q4_0 */, vec![0; 180]);
    match load_bytes(&b.build()) {
        Err(LoaderError::UnsupportedTensorType { type_id: 2, .. }) => {}
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn broken_chat_template_fails_at_the_gate() {
    // an unclosed block is a parse error
    let tokens: Vec<&str> = vec!["a"];
    let b = base_kvs("gpt2", "qwen2", 0, &tokens, &[], "{% if x %}no end");
    match load_bytes(&b.build()) {
        Err(LoaderError::Template { .. }) => {}
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn load_from_a_real_file_on_disk_works() {
    let path = std::env::temp_dir().join("nucleon-yamf-test.gguf");
    std::fs::write(&path, mini().build()).unwrap();
    let yamf = load(&path).unwrap();
    assert_eq!(yamf.tensors.len(), 13);
    std::fs::remove_file(&path).ok();
}

// ---- the oracle: the same synthetic bytes read through candle-core
// (dev-dependency only) must agree with our parse and our dequant.

#[test]
fn oracle_candle_agrees_on_metadata_infos_and_dequant() {
    use candle_core::quantized::gguf_file;

    let bytes = mini().build();
    let mut cursor = std::io::Cursor::new(&bytes);
    let content = gguf_file::Content::read(&mut cursor).unwrap();

    // metadata
    match content.metadata.get("general.architecture").unwrap() {
        gguf_file::Value::String(s) => assert_eq!(s, "qwen3"),
        other => panic!("{other:?}"),
    }
    // tensor inventory
    assert_eq!(content.tensor_infos.len(), 13);

    // our container agrees on the data offset
    let ours = crate::loader::container::parse(&bytes).unwrap();
    assert_eq!(ours.data_start as u64, content.tensor_data_offset);

    // dequantized values agree on the Q8_0 tensor
    let mut cursor = std::io::Cursor::new(&bytes);
    let qt = content
        .tensor(
            &mut cursor,
            "blk.0.ffn_gate.weight",
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let theirs = qt
        .dequantize(&candle_core::Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let yamf = load_bytes(&bytes).unwrap();
    let ours = yamf.tensors["blk.0.ffn_gate.weight"].data();
    assert_eq!(ours, theirs.as_slice());
}

// Full valid metadata (no tensors) with the tokenizer-facing pieces
// parameterized, for probing the gate's cheap first pass.
fn base_kvs(
    model: &str,
    pre: &str,
    eos: u32,
    tokens: &[&str],
    merges: &[&str],
    template: &str,
) -> GgufBuilder {
    let types: Vec<i32> = tokens.iter().map(|_| 1).collect();
    GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_str("tokenizer.ggml.model", model)
        .kv_str("tokenizer.ggml.pre", pre)
        .kv_u32("tokenizer.ggml.eos_token_id", eos)
        .kv_arr_str("tokenizer.ggml.tokens", tokens)
        .kv_arr_i32("tokenizer.ggml.token_type", &types)
        .kv_arr_str("tokenizer.ggml.merges", merges)
        .kv_str("tokenizer.chat_template", template)
}

#[test]
fn foreign_tokenizer_model_and_pre_are_refused_by_name() {
    let b = base_kvs("llama", "qwen2", 0, &["a"], &[], "x");
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(
                reason.contains("llama") && reason.contains("gpt2"),
                "{reason}"
            )
        }
        other => panic!("{:?}", other.err()),
    }

    let b = base_kvs("gpt2", "deepseek", 0, &["a"], &[], "x");
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(
                reason.contains("deepseek") && reason.contains("qwen2"),
                "{reason}"
            )
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn eos_outside_the_vocab_is_refused() {
    let b = base_kvs("gpt2", "qwen2", 5, &["a", "b"], &[], "x");
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains('5') && reason.contains('2'), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn malformed_merges_are_refused() {
    for bad in ["a b c", " b", "a ", "ab"] {
        let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[bad], "x");
        match load_bytes(&b.build()) {
            Err(LoaderError::Structure { reason }) => {
                assert!(reason.contains("merge"), "{bad}: {reason}")
            }
            other => panic!("{bad}: {:?}", other.err()),
        }
    }
}

#[test]
fn empty_arrays_keep_their_declared_element_type() {
    // empty string-typed merges: fine
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x");
    // fails later (missing tensors), but NOT on the merges
    match load_bytes(&b.build()) {
        Err(LoaderError::MissingTensor { .. }) => {}
        other => panic!("{:?}", other.err()),
    }

    // an i32-typed empty array where strings are declared: refused
    let mut c = crate::loader::container::parse(
        &GgufBuilder::new()
            .kv_arr_i32("tokenizer.ggml.merges", &[])
            .build(),
    )
    .unwrap();
    match take_str_array(&mut c, "tokenizer.ggml.merges") {
        Err(LoaderError::WrongType { key, .. }) => assert_eq!(key, "tokenizer.ggml.merges"),
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn overflowing_config_arithmetic_is_refused_not_panicking() {
    // head_count x key_length overflows u32
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 1_000_000)
        .kv_u32("qwen3.attention.head_count_kv", 1_000_000)
        .kv_u32("qwen3.attention.key_length", 1_000_000)
        .kv_u32("qwen3.attention.value_length", 1_000_000)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_arr_str("tokenizer.ggml.tokens", &["a"]);
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("overflow"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn value_length_must_equal_key_length() {
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 8)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_arr_str("tokenizer.ggml.tokens", &["a"]);
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(
                reason.contains("value_length 8") && reason.contains('4'),
                "{reason}"
            )
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn quantization_version_is_required_and_gated_when_quantized() {
    // the mini model carries a Q8_0 tensor; strip the version key by
    // rebuilding without it: reuse base_kvs (no tensors, so the key
    // is not required) plus one quantized tensor
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x").tensor(
        "token_embd.weight",
        &[32, 1],
        GGML_Q8_0,
        quantize_q8_0_ref(&[0.0; 32]),
    );
    match load_bytes(&b.build()) {
        Err(LoaderError::MissingKey { key }) => {
            assert_eq!(key, "general.quantization_version")
        }
        other => panic!("{:?}", other.err()),
    }

    // wrong version: refused by value
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x")
        .kv_u32("general.quantization_version", 3)
        .tensor(
            "token_embd.weight",
            &[32, 1],
            GGML_Q8_0,
            quantize_q8_0_ref(&[0.0; 32]),
        );
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("quantization_version 3"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn u64_written_dimension_keys_are_accepted() {
    // same mini metadata but block_count as a u64 value
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u64("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 64)
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "qwen2")
        .kv_u32("tokenizer.ggml.eos_token_id", 0)
        .kv_arr_str("tokenizer.ggml.tokens", &["a"])
        .kv_arr_i32("tokenizer.ggml.token_type", &[1])
        .kv_arr_str("tokenizer.ggml.merges", &[])
        .kv_str("tokenizer.chat_template", "x");
    // fails on missing tensors, which means the u64 key was read fine
    match load_bytes(&b.build()) {
        Err(LoaderError::MissingTensor { .. }) => {}
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn zero_context_length_is_refused() {
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 1)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 0)
        .kv_arr_str("tokenizer.ggml.tokens", &["a"]);
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("context_length"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn out_of_bounds_overlapping_and_misaligned_tensor_ranges_are_refused() {
    // out of bounds: truncate the built file so the last tensor's
    // range sticks out past the data region
    let full = mini().build();
    let cut = &full[..full.len() - 8];
    match load_bytes(cut) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("ends at data byte"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }

    // overlap: two tensors forced onto the same data offset
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x")
        .kv_u32("qwen3.some", 1) // keep keys unique from base
        .tensor_at(
            "token_embd.weight",
            &[8, 1],
            GGML_F32,
            f32_bytes(&[0.0; 8]),
            0,
        )
        .tensor_at(
            "output_norm.weight",
            &[8],
            GGML_F32,
            f32_bytes(&[1.0; 8]),
            0,
        );
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("overlap"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }

    // misalignment: an offset that is not a multiple of 32 refuses at
    // the container layer, before any shape logic runs
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x").tensor_at(
        "token_embd.weight",
        &[8, 1],
        GGML_F32,
        f32_bytes(&[0.0; 8]),
        3,
    );
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("aligned"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn non_finite_or_non_positive_float_knobs_are_refused() {
    for (eps, theta) in [
        (f32::NAN, 1e6),
        (1e-6, 0.0),
        (-1e-6, 1e6),
        (1e-6, f32::INFINITY),
    ] {
        let b = GgufBuilder::new()
            .kv_str("general.architecture", "qwen3")
            .kv_u32("qwen3.block_count", 1)
            .kv_u32("qwen3.embedding_length", 8)
            .kv_u32("qwen3.feed_forward_length", 16)
            .kv_u32("qwen3.attention.head_count", 2)
            .kv_u32("qwen3.attention.head_count_kv", 1)
            .kv_u32("qwen3.attention.key_length", 4)
            .kv_u32("qwen3.attention.value_length", 4)
            .kv_f32("qwen3.attention.layer_norm_rms_epsilon", eps)
            .kv_f32("qwen3.rope.freq_base", theta)
            .kv_u32("qwen3.context_length", 64)
            .kv_arr_str("tokenizer.ggml.tokens", &["a"]);
        match load_bytes(&b.build()) {
            Err(LoaderError::Structure { reason }) => {
                assert!(reason.contains("finite"), "{reason}")
            }
            other => panic!("eps {eps} theta {theta}: {:?}", other.err()),
        }
    }
}

#[test]
fn f16_tensor_refuses_as_unsupported_type_not_missing_quant_version() {
    // F16 (type id 1) is not quantized: no quantization_version is
    // demanded, and the refusal names the type
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x").tensor(
        "token_embd.weight",
        &[8, 1],
        1, // GGML F16
        vec![0; 16],
    );
    match load_bytes(&b.build()) {
        Err(LoaderError::UnsupportedTensorType { type_id: 1, .. }) => {}
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn add_bos_true_is_refused_for_qwen3() {
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x")
        .kv_bool("tokenizer.ggml.add_bos_token", true);
    match load_bytes(&b.build()) {
        Err(LoaderError::Structure { reason }) => {
            assert!(reason.contains("add_bos"), "{reason}")
        }
        other => panic!("{:?}", other.err()),
    }
    // false is the documented reality and passes this gate
    let b = base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x")
        .kv_bool("tokenizer.ggml.add_bos_token", false);
    match load_bytes(&b.build()) {
        Err(LoaderError::MissingTensor { .. }) => {}
        other => panic!("{:?}", other.err()),
    }
}

#[test]
fn non_bool_add_bos_token_is_a_type_error() {
    let b =
        base_kvs("gpt2", "qwen2", 0, &["a"], &[], "x").kv_u32("tokenizer.ggml.add_bos_token", 1);
    match load_bytes(&b.build()) {
        Err(LoaderError::WrongType { key, want, .. }) => {
            assert_eq!(key, "tokenizer.ggml.add_bos_token");
            assert_eq!(want, "bool");
        }
        other => panic!("{:?}", other.err()),
    }
}
