use super::*;

use crate::loader::formats::gguf::builder::mini;
use crate::loader::load_bytes;
use crate::tokenizer::from_yamf;

fn tok() -> Tokenizer {
    from_yamf(&load_bytes(&mini().build()).unwrap()).unwrap()
}

#[test]
fn emoji_round_trip() {
    // four UTF-8 bytes, four byte-level tokens in the fixture (no
    // merges cover them), and back to one emoji on decode
    let t = tok();
    let ids = t.encode("🎯").unwrap();
    assert_eq!(ids.len(), 4);
    assert_eq!(t.decode(&ids).unwrap(), "🎯");
}

#[test]
fn split_emoji_streams_without_replacement_char() {
    // the failure chapter 8.9 exists to prevent: fed one id at a
    // time, the stream must withhold until the codepoint completes,
    // and the concatenated output must be the emoji, never U+FFFD
    let t = tok();
    let ids = t.encode("🎯").unwrap();
    let mut stream = t.stream_decoder();
    let mut out = String::new();
    let mut none_count = 0;
    for id in ids {
        match stream.step(id).unwrap() {
            Some(chunk) => out.push_str(&chunk),
            None => none_count += 1,
        }
    }
    assert_eq!(out, "🎯");
    assert!(
        !out.contains('\u{FFFD}'),
        "replacement char leaked: {out:?}"
    );
    // the first three byte-tokens cannot form a codepoint alone
    assert_eq!(none_count, 3, "stream should buffer the partial bytes");
    // everything emitted: finish has nothing buffered
    assert_eq!(stream.finish().unwrap(), None);
}

#[test]
fn truncated_stream_finish_surfaces_the_partial_codepoint() {
    // generation stopped by max-tokens mid-emoji: the HF stream has
    // no flush, so without finish() the tail bytes vanish silently.
    // finish decodes them lossily — visible U+FFFD, never silent loss.
    let t = tok();
    let ids = t.encode("🎯").unwrap();
    let mut stream = t.stream_decoder();
    for id in &ids[..3] {
        assert_eq!(stream.step(*id).unwrap(), None);
    }
    let tail = stream.finish().unwrap().expect("buffered bytes must drain");
    assert!(
        tail.contains('\u{FFFD}'),
        "truncation must be visible: {tail:?}"
    );
}

#[test]
fn decode_drops_unknown_ids_silently() {
    // pinning the HF crate's semantic so a future version bump that
    // changes it (to an error, say) fails loudly here instead of
    // deep inside the generation loop
    let t = tok();
    assert_eq!(t.decode(&[999_999]).unwrap(), "");
}

#[test]
fn decode_keeps_markers() {
    // skip_special_tokens is hard-wired false: decoding the ChatML
    // marker id gives the marker text back, not an empty string
    assert_eq!(tok().decode(&[10]).unwrap(), "<|im_start|>");
}
