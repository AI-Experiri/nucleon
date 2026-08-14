# ADR 001 — Module boundaries: representation changes and axes of variation

Date: 2026-08-13. Status: accepted.

## Decision

nucleon is split into tensor / backend / loader / tokenizer / cache /
families / sampler / generate / chat, cut by two rules:

1. **Cut where the data changes representation** (text→ids→logits→id→text).
   Boundaries carry narrow dumb types; modules on either side cannot
   entangle.
2. **Cut along independent axes of variation**: hardware (backend), model
   architecture (families), file format (loader), decoding strategy
   (sampler), memory scheme (cache), prompt convention (chat). One
   real-world change = one module rewritten.

Corollaries: tensor (data) is separate from backend (math) so backends can
swap under a stable data type; generate (choreography, model-agnostic) is
separate from families (wiring, per-model) so new families inherit the loop.

## Verification

The milestone plan tests the split: M2 must land touching only backend, M3
only loader(+kernels), M4 only families(+a cache impl). A milestone forcing
cross-module edits falsifies the architecture and reopens this ADR.

## Context

llama.cpp concentrates these concerns in one ~20k-line file (fast to evolve
for experts, opaque for learners). mlx-engine (our prior work) validated
the family/loop split: its generation loop never changed as families were
added. nucleon is a learning project; isolation and readability outrank
integration speed.
