//! Test support: build tiny, valid GGUF files in memory, plus the
//! reference Q8_0 quantizer for bit-exact fixtures. Test-only code;
//! production never writes GGUF.

use half::f16;

/// Metadata value type ids from the spec.
pub(crate) const T_U32: u32 = 4;
pub(crate) const T_I32: u32 = 5;
pub(crate) const T_F32: u32 = 6;
pub(crate) const T_BOOL: u32 = 7;
pub(crate) const T_STR: u32 = 8;
pub(crate) const T_ARR: u32 = 9;

pub(crate) const GGML_F32: u32 = 0;
pub(crate) const GGML_Q8_0: u32 = 8;

enum Kv {
    U32(String, u32),
    U64(String, u64),
    F32(String, f32),
    Bool(String, bool),
    Str(String, String),
    ArrStr(String, Vec<String>),
    ArrI32(String, Vec<i32>),
}

struct TensorSpec {
    name: String,
    ne_dims: Vec<u64>,
    type_id: u32,
    bytes: Vec<u8>,
    forced_offset: Option<u64>,
}

/// Builds a spec-conforming GGUF v3 byte blob.
pub(crate) struct GgufBuilder {
    version: u32,
    kvs: Vec<Kv>,
    tensors: Vec<TensorSpec>,
    alignment: u64,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl GgufBuilder {
    pub(crate) fn new() -> Self {
        Self {
            version: 3,
            kvs: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    pub(crate) fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Declare a custom alignment: writes the metadata key AND lays
    /// tensors out with it, so fixtures cannot drift.
    #[allow(dead_code)]
    pub(crate) fn alignment(mut self, a: u32) -> Self {
        self.alignment = u64::from(a);
        self.kvs.push(Kv::U32("general.alignment".into(), a));
        self
    }

    pub(crate) fn kv_u32(mut self, k: &str, v: u32) -> Self {
        self.kvs.push(Kv::U32(k.into(), v));
        self
    }

    pub(crate) fn kv_u64(mut self, k: &str, v: u64) -> Self {
        self.kvs.push(Kv::U64(k.into(), v));
        self
    }

    pub(crate) fn kv_f32(mut self, k: &str, v: f32) -> Self {
        self.kvs.push(Kv::F32(k.into(), v));
        self
    }

    pub(crate) fn kv_bool(mut self, k: &str, v: bool) -> Self {
        self.kvs.push(Kv::Bool(k.into(), v));
        self
    }

    pub(crate) fn kv_str(mut self, k: &str, v: &str) -> Self {
        self.kvs.push(Kv::Str(k.into(), v.into()));
        self
    }

    pub(crate) fn kv_arr_str(mut self, k: &str, v: &[&str]) -> Self {
        self.kvs.push(Kv::ArrStr(
            k.into(),
            v.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub(crate) fn kv_arr_i32(mut self, k: &str, v: &[i32]) -> Self {
        self.kvs.push(Kv::ArrI32(k.into(), v.to_vec()));
        self
    }

    /// Add a tensor. `ne_dims` is GGUF order: dims[0] = row length.
    pub(crate) fn tensor(
        mut self,
        name: &str,
        ne_dims: &[u64],
        type_id: u32,
        bytes: Vec<u8>,
    ) -> Self {
        self.tensors.push(TensorSpec {
            name: name.into(),
            ne_dims: ne_dims.to_vec(),
            type_id,
            bytes,
            forced_offset: None,
        });
        self
    }

    /// Add a tensor at an explicit data offset (for hostile-layout
    /// fixtures: overlaps, misalignment).
    pub(crate) fn tensor_at(
        mut self,
        name: &str,
        ne_dims: &[u64],
        type_id: u32,
        bytes: Vec<u8>,
        offset: u64,
    ) -> Self {
        self.tensors.push(TensorSpec {
            name: name.into(),
            ne_dims: ne_dims.to_vec(),
            type_id,
            bytes,
            forced_offset: Some(offset),
        });
        self
    }

    pub(crate) fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for kv in &self.kvs {
            match kv {
                Kv::U32(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_U32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::U64(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&10u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_F32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Bool(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_BOOL.to_le_bytes());
                    out.push(u8::from(*v));
                }
                Kv::Str(k, v) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_STR.to_le_bytes());
                    put_str(&mut out, v);
                }
                Kv::ArrStr(k, items) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_ARR.to_le_bytes());
                    out.extend_from_slice(&T_STR.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items {
                        put_str(&mut out, s);
                    }
                }
                Kv::ArrI32(k, items) => {
                    put_str(&mut out, k);
                    out.extend_from_slice(&T_ARR.to_le_bytes());
                    out.extend_from_slice(&T_I32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for v in items {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        // Tensor infos with offsets assigned in order, each aligned.
        let mut offset: u64 = 0;
        let mut blobs: Vec<(u64, &[u8])> = Vec::new();
        for t in &self.tensors {
            let rem = offset % self.alignment;
            if rem != 0 {
                offset += self.alignment - rem;
            }
            if let Some(forced) = t.forced_offset {
                offset = forced;
            }
            put_str(&mut out, &t.name);
            out.extend_from_slice(&(t.ne_dims.len() as u32).to_le_bytes());
            for d in &t.ne_dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.type_id.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            blobs.push((offset, &t.bytes));
            offset += t.bytes.len() as u64;
        }

        // Pad the header to the alignment, then lay the data region.
        let rem = out.len() % (self.alignment as usize);
        if rem != 0 {
            out.resize(out.len() + (self.alignment as usize - rem), 0);
        }
        let data_start = out.len();
        for (off, bytes) in blobs {
            let at = data_start + off as usize;
            let end = at + bytes.len();
            if out.len() < end {
                out.resize(end, 0);
            }
            // write in place at the declared offset; overlapping or
            // backward-offset fixtures behave the way the header says
            out[at..end].copy_from_slice(bytes);
        }
        out
    }
}

/// The reference Q8_0 quantizer (ggml's quantize_row_q8_0_ref):
/// per 32-value block, d = amax/127 stored as f16, q = round(x/d).
pub(crate) fn quantize_q8_0_ref(vals: &[f32]) -> Vec<u8> {
    assert!(vals.len().is_multiple_of(32), "Q8_0 needs multiples of 32");
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for block in vals.chunks(32) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in block {
            out.push(((v * id).round() as i32).clamp(-128, 127) as u8);
        }
    }
    out
}

/// f32 values as raw little-endian bytes (a GGUF F32 tensor's data).
pub(crate) fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
