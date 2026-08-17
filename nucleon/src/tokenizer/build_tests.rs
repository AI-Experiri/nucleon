use super::*;

use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;

fn tok() -> Tokenizer {
    from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap()
}

#[test]
fn builds_from_the_mini_model() {
    // the same fixture the loader tests use: 15 named tokens plus
    // the 254 alphabet extensions, four merges, three specials
    let t = tok();
    assert_eq!(t.vocab_size(), 269);
}

#[test]
fn pre_tokenizer_isolates_digits_before_merges() {
    // the discriminating test for the qwen2 Split stage: the fixture
    // carries the merge "7"+"6" -> "76", but \p{N} in the regex puts
    // each digit in its own piece and merges never cross pieces. A
    // build that lost the Split (or its behavior flags) collapses
    // this to one id and fails here.
    let t = tok();
    let ids = t.encode("76").unwrap();
    assert_eq!(
        ids.len(),
        2,
        "digits must not merge across the split: {ids:?}"
    );
    assert_eq!(t.decode(&ids).unwrap(), "76");
}

#[test]
fn user_defined_tokens_match_whole() {
    // <think> is UserDefined (type 4) in the fixture: it must be
    // registered and match as one id (12), never BPE-decompose. A
    // filter that drops UserDefined from registration fails here.
    let t = tok();
    assert_eq!(t.encode("<think>").unwrap(), vec![12]);
    let ids = t.encode("ab<think>ab").unwrap();
    assert_eq!(
        ids.iter().filter(|&&i| i == 12).count(),
        1,
        "one whole <think> match: {ids:?}"
    );
}

#[test]
fn unknown_tokens_match_whole() {
    // <unk> is Unknown (type 2) in the fixture: llama.cpp's special
    // cache includes UNKNOWN, so it must register and match as one
    // id (14). A filter that drops Unknown BPE-decomposes it into
    // "<unk" + ">" and fails here.
    assert_eq!(tok().encode("<unk>").unwrap(), vec![14]);
}

#[test]
fn byte_level_regex_stays_off() {
    // the fixture merge "."+"a" -> ".a" spans a boundary the GPT-2
    // regex would cut (punct | letters) but the qwen2 alternative
    // [^\r\n\p{L}\p{N}]?\p{L}+ keeps whole. use_regex=true on the
    // ByteLevel stage re-splits the piece and the merge cannot fire.
    let t = tok();
    let ids = t.encode(".a").unwrap();
    assert_eq!(ids.len(), 1, "merge must fire inside one piece: {ids:?}");
    assert_eq!(t.decode(&ids).unwrap(), ".a");
}

#[test]
fn merge_rank_is_file_order() {
    // index = rank is load-bearing: on ".ab" both "a b" (rank 0)
    // and ". a" (rank 3) are candidates, and the lower rank must
    // win — "." + "ab" (the "." is alphabet id 61), never ".a" + "b".
    // A loader refactor that reorders merges (say through a set)
    // silently changes ids everywhere; this pins the order.
    assert_eq!(tok().encode(".ab").unwrap(), vec![61, 2]);
}

#[test]
fn mismatched_type_array_refuses() {
    // Yamf's fields are public; a hand-built bundle with a short
    // token_types must refuse, not silently truncate the added-token
    // zip and leave trailing specials unregistered
    let mut yamf = load_bytes(&mini().build()).unwrap();
    yamf.tokenizer.token_types.pop();
    let Err(err) = from_yamf(&yamf) else {
        panic!("short token_types must refuse")
    };
    let msg = err.to_string();
    assert!(msg.contains("parallel"), "{msg}");
}

#[test]
fn duplicate_token_refuses() {
    // same public-fields hazard as the short-type-array case: a
    // duplicate string silently shadows the earlier id in the vocab
    // map (decodes to "") — refuse instead
    let mut yamf = load_bytes(&mini().build()).unwrap();
    let first = yamf.tokenizer.tokens[0].clone();
    yamf.tokenizer.tokens.push(first);
    yamf.tokenizer
        .token_types
        .push(crate::loader::yamf::TokenType::Normal);
    let Err(err) = from_yamf(&yamf) else {
        panic!("duplicate token must refuse")
    };
    let msg = err.to_string();
    assert!(msg.contains("distinct"), "{msg}");
}

#[test]
fn special_flag_survives_the_crate_registration() {
    // the observable consequence of added_token()'s mapping, pinned
    // through the crate itself: skipping specials eats the Control
    // marker but must keep the UserDefined/Unknown ones. Guards
    // against a tokenizers upgrade forcing special=true on
    // everything passed to add_special_tokens.
    let t = tok();
    assert_eq!(t.inner.decode(&[9], true).unwrap(), "");
    assert_eq!(t.inner.decode(&[12], true).unwrap(), "<think>");
    assert_eq!(t.inner.decode(&[14], true).unwrap(), "<unk>");
}

#[test]
fn only_control_tokens_are_special() {
    // the reference tokenizer.json: ChatML/eos markers special=true,
    // <think>/<tool_call> added but special=false (so a future
    // skip_special_tokens caller cannot eat them); normalized false
    // for both kinds
    use crate::loader::yamf::TokenType;
    let control = added_token("<|im_end|>", TokenType::Control);
    assert!(control.special);
    assert!(!control.normalized);
    let user_defined = added_token("<think>", TokenType::UserDefined);
    assert!(!user_defined.special);
    assert!(!user_defined.normalized);
    // Unknown rows register too (matched whole) but are not special
    let unknown = added_token("<unk>", TokenType::Unknown);
    assert!(!unknown.special);
    assert!(!unknown.normalized);
}

#[test]
fn nfc_normalizes_decomposed_input() {
    // the reference tokenizer.json carries an NFC normalizer; GGUF
    // has no field for it, so from_yamf pins it. Decomposed input
    // (e + combining acute, routine on macOS) must encode to the
    // same ids as the precomposed form.
    let t = tok();
    let decomposed = t.encode("cafe\u{0301}").unwrap();
    let precomposed = t.encode("caf\u{00e9}").unwrap();
    assert_eq!(decomposed, precomposed);
}

#[test]
fn stop_ids_pass_through_unchanged() {
    // loader assembled [eos=9, endoftext=8]; part 7 is a verbatim copy
    assert_eq!(tok().stop_token_ids(), &[9, 8]);
}

#[test]
fn unknown_pre_id_refuses_by_name() {
    // the pre field selects the regex; a value this build has no
    // regex for must refuse, never silently use the qwen2 pattern
    let mut yamf = load_bytes(&mini().build()).unwrap();
    yamf.tokenizer.pre = "llama3".to_string();
    let Err(err) = from_yamf(&yamf) else {
        panic!("unknown pre id must refuse")
    };
    let msg = err.to_string();
    assert!(msg.contains("llama3") && msg.contains("qwen2"), "{msg}");
}

#[test]
fn broken_merge_refuses_with_build_error() {
    // from_yamf's error contract: a merge referencing a token that
    // is not in the vocab fails BPE construction. The loader gates
    // this for real files; this pins what happens if a caller hands
    // us a hand-built Yamf that skipped the gate.
    let mut yamf = load_bytes(&mini().build()).unwrap();
    yamf.tokenizer
        .merges
        .push(("nosuch".to_string(), "token".to_string()));
    let Err(err) = from_yamf(&yamf) else {
        panic!("broken merge must refuse")
    };
    assert!(matches!(err, TokenizerError::Build { .. }), "{err}");
}

#[test]
fn specials_reuse_their_vocab_ids() {
    // registration must not mint new ids: the Control tokens sit at
    // 8, 9, 10 in the fixture vocab and must encode to exactly those
    let t = tok();
    assert_eq!(t.encode("<|endoftext|>").unwrap(), vec![8]);
    assert_eq!(t.encode("<|im_end|>").unwrap(), vec![9]);
    assert_eq!(t.encode("<|im_start|>").unwrap(), vec![10]);
}
