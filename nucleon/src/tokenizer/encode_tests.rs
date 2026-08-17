use super::*;

use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;
use crate::tokenizer::from_yamf;

fn tok() -> Tokenizer {
    from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap()
}

#[test]
fn ascii_round_trip() {
    let t = tok();
    let ids = t.encode("ab").unwrap();
    // "a" + "b" merge (rank 0) into "ab", vocab id 2
    assert_eq!(ids, vec![2]);
    assert_eq!(t.decode(&ids).unwrap(), "ab");
}

#[test]
fn latin1_round_trip() {
    // é is two UTF-8 bytes; both alphabet characters are in the
    // fixture vocab, so the word encodes byte-level and comes back
    let t = tok();
    let ids = t.encode("café").unwrap();
    assert_eq!(t.decode(&ids).unwrap(), "café");
}

#[test]
fn encode_never_adds_specials() {
    // add_special_tokens=false is hard-wired. Today no
    // post-processor is installed so the flag is inert either way;
    // this test is the tripwire for the day one is added (the
    // reference has a ByteLevel post-processor we deliberately omit
    // — it only computes offsets, which encode() discards).
    let ids = tok().encode("ab").unwrap();
    for special in [8u32, 9, 10] {
        assert!(
            !ids.contains(&special),
            "unexpected special {special} in {ids:?}"
        );
    }
}

#[test]
fn oversized_input_is_refused() {
    // past ~1M whitespace chars in one segment, fancy-regex's
    // backtrack limit trips and the tokenizers crate swallows the
    // error — the whole input becomes ONE unsplit piece and encode
    // returns wrong ids with no failure. The byte cap refuses long
    // before the trip point, loudly.
    let t = tok();
    let big = " ".repeat(512 * 1024 + 1);
    let Err(err) = t.encode(&big) else {
        panic!("oversized input must refuse")
    };
    let msg = err.to_string();
    assert!(msg.contains("524288"), "cap named in the refusal: {msg}");
}

#[test]
fn chatml_prompt_keeps_every_marker_single() {
    // the string the chat template would render for one user turn:
    // each marker must survive as its single added-token id
    let t = tok();
    let ids = t
        .encode("<|im_start|>user\nab<|im_end|>\n<|im_start|>assistant\n")
        .unwrap();
    let im_starts = ids.iter().filter(|&&i| i == 10).count();
    let im_ends = ids.iter().filter(|&&i| i == 9).count();
    assert_eq!(im_starts, 2, "both <|im_start|> markers as id 10: {ids:?}");
    assert_eq!(im_ends, 1, "the <|im_end|> marker as id 9: {ids:?}");
}
