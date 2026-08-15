use super::*;
use crate::loader::container::parse;
use crate::loader::gguf_builder::GgufBuilder;

fn qwen3_kvs() -> GgufBuilder {
    GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.feed_forward_length", 16)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_u32("qwen3.attention.value_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 512)
        .kv_arr_str("tokenizer.ggml.tokens", &["a", "b", "c", "d", "e"])
}

#[test]
fn reads_every_field_and_derives_vocab() {
    let c = parse(&qwen3_kvs().build()).unwrap();
    let FamilyConfig::Qwen3(cfg) = family_config(&c).unwrap();
    assert_eq!(cfg.num_hidden_layers, 2);
    assert_eq!(cfg.hidden_size, 8);
    assert_eq!(cfg.intermediate_size, 16);
    assert_eq!(cfg.num_attention_heads, 2);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.head_dim, 4);
    assert_eq!(cfg.max_position_embeddings, 512);
    assert_eq!(cfg.vocab_size, 5); // tokens array length, no key exists
    assert_eq!(cfg.rope_theta, 1e6);
}

#[test]
fn epsilon_is_the_exact_f32_bit_pattern() {
    let c = parse(&qwen3_kvs().build()).unwrap();
    let FamilyConfig::Qwen3(cfg) = family_config(&c).unwrap();
    assert_eq!(cfg.rms_norm_eps.to_bits(), 1e-6f32.to_bits());
}

#[test]
fn foreign_architecture_is_refused_by_name() {
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "llama")
        .build();
    let c = parse(&b).unwrap();
    match family_config(&c) {
        Err(LoaderError::UnsupportedArchitecture { found }) => assert_eq!(found, "llama"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn missing_key_is_named() {
    // full set minus feed_forward_length
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.embedding_length", 8)
        .kv_u32("qwen3.attention.head_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 1)
        .kv_u32("qwen3.attention.key_length", 4)
        .kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_f32("qwen3.rope.freq_base", 1e6)
        .kv_u32("qwen3.context_length", 512)
        .kv_arr_str("tokenizer.ggml.tokens", &["a"])
        .build();
    let c = parse(&b).unwrap();
    match family_config(&c) {
        Err(LoaderError::MissingKey { key }) => {
            assert_eq!(key, "qwen3.feed_forward_length")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn wrong_value_type_is_named_both_ways() {
    let b = qwen3_kvs()
        .kv_str("qwen3.some_future_key", "irrelevant extra key is fine")
        .build();
    // sanity: extra unknown keys do not bother the config reader
    let c = parse(&b).unwrap();
    assert!(family_config(&c).is_ok());

    // now block_count as a string
    let b = GgufBuilder::new()
        .kv_str("general.architecture", "qwen3")
        .kv_str("qwen3.block_count", "two")
        .kv_arr_str("tokenizer.ggml.tokens", &["a"])
        .build();
    let c = parse(&b).unwrap();
    match family_config(&c) {
        Err(LoaderError::WrongType { key, want, found }) => {
            assert_eq!(key, "qwen3.block_count");
            assert_eq!(want, "u32");
            assert_eq!(found, "string");
        }
        other => panic!("{other:?}"),
    }
}
