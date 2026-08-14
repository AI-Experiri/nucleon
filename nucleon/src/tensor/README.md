# tensor — chapter 1

The one data structure the whole engine speaks: `Tensor`, an n-dimensional
block of `f32`s, row-major, contiguous.

**Why so dumb?** No strides, no views, no dtype generics — every design
review of tensor libraries finds the stride machinery is where readability
dies. We accept occasional copies; at M1 scale (0.6B on CPU) they are free.
When M2 moves storage to Metal buffers, this type remains the CPU/reference
representation, so the cleverness would have been thrown away anyway (ADR
to be written when M2 lands).

**Read order:** `tensor.rs` top to bottom (~60 lines), then
`tensor_tests.rs`.
