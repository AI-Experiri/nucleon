//! Weights-file format adapters. Every format that nucleon reads
//! lives in its own submodule here and converts to the same `Yamf`
//! (the border struct in `loader::yamf`). Today: gguf only. If a
//! second format ever lands (safetensors, ONNX, etc.), it becomes a
//! sibling module and the top-level `load` dispatches to it.

pub mod gguf;
