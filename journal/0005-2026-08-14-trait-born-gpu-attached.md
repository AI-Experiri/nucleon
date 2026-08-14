# 0005 — Backend trait born, GPU attached, book goes POV (2026-08-14)

## Narrative pivot (user-driven)

The book restructured as Game-of-Thrones POV chapters: each component is
a recurring character ("Tensor 0" = birthplace; "Tensor 1, 2..." return
when the story demands views, map, ...). Phases replace parts: Substance
/ The Awakening / Power and Strangers. Architecture narrative changed
with it: the Backend trait is BORN in Tensor 0 with CpuBackend attached
as the default; The GPU chapter ends by attaching MetalBackend to the
same trait. The separate "backend chapter" concept dissolved into the
two character chapters.

## Code

- `backend/ops.rs`: trait Backend, 11 ops. Attention is ONE method
  (decision A: fusable behind the seam; ADR-worthy, folded here).
- `backend/cpu.rs`: CpuBackend reference loops, frozen policy. 17 tests:
  hand cases, properties, GQA attention behavior contracts.
- New Metal kernel `attention`: fused single-position GQA attention
  (scores + softmax + weighted sum, one dispatch, one threadgroup per
  head, seq capped 4096 by threadgroup memory). Passed trait-level
  parity vs CPU first run, including seq=1000 and 4:2 GQA.
- `nucleon-metal/src/backend.rs`: MetalBackend implements
  nucleon::backend::Backend by delegating to the kernel wrappers.
- tensor/ split into one-concern files (user hard rule): core.rs /
  access.rs / display.rs, each with its own sibling tests. tensor.rs
  monolith deleted.

## Book

- Tensor 0 gained "The trait: how Tensor gets its math".
- The GPU chapter opens with the bridge ("we have a CPU tensor — how do
  we make it work on the GPU?") and closes with "Attaching the GPU
  trait" + the fused-attention payoff.
- New conventions landed this session (all in CLAUDE.md): Rust side
  boxes (2 lines + official doc link + sibling constructs), diagram
  cards (SVG files in diagrams/, never inline svg), lists over
  paragraphs, verify-library-claims-before-writing, no mdbook build
  while serve runs, POV structure.

## State

Workspace 67 tests green (38 nucleon + 29 nucleon-metal), fmt/clippy
clean. Codex convergence on the backend seam: pending (next session
task). Metal module convergence from 0004 stands at stable rounds
r8/r9/r11 with r10's doc fix; formally needs one more clean round.

## Addendum: chapter 2 rebuilt on sourced hardware facts

- All ch2 ASCII diagrams became SVGs (timeline, thread grid, matvec,
  tree reduction, fusion round-trip, physical package view, and a
  3-way NVIDIA/AMD/Apple architecture comparison). Convention: real
  code in code blocks, figures as SVG diagram cards, results as tables.
- 4-agent-style research pass (single deep agent, 57 tool calls) read
  the Ada/Blackwell whitepapers, CUDA guide, RDNA3 ISA, GPUOpen, Metal
  Feature Set Tables, WWDC22 10159, TBDR docs, Apple newsroom. Findings
  distilled into a sourced "How Apple's GPU differs" section with a
  terminology mapping table.
- Corrections it forced: GPU cores top out at 80 (M3 Ultra), not 76;
  32 KB threadgroup memory stated exactly (feature tables), not "~";
  per-core ALU counts marked as community-only (Apple publishes none);
  "no matrix ops until M5" nuanced (simdgroup_matrix intrinsics since
  M1; dedicated hardware only from M5).
- Verified survivals: PCIe Gen4 x16 = 32 GB/s/direction (NVIDIA's own
  figure), RTX 4090 1008 GB/s, M2 Ultra 800 GB/s / 192 GB, simdgroup
  width 32 via threadExecutionWidth.
- New user conventions this stretch (all in CLAUDE.md): metaphors
  banned (ask first); hardware words get a referenced glossary at first
  use; POV chapters recur (Tensor 0/1/2); one concern per file in
  module folders; backend stays a sibling of tensor (dependency
  direction, ADR-001 axes).
