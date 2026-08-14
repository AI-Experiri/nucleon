# 0007 — 2026-08-14 — the book goes public, Part II gets its plan

## Book on GitHub Pages

- First attempt to enable Pages failed: the AI-Experiri org is on the
  free plan and the repo was private; the Pages API returns 422 ("Your
  current plan does not support GitHub Pages for this repository").
  Options were a separate public book repo, going public, or paying
  for Team. User made the repo public and that resolved it.
- Pages enabled with `build_type=workflow`; `.github/workflows/book.yml`
  (committed on main) builds docs/book with mdBook pinned to 0.5.4, the
  same version used locally, and deploys with actions/deploy-pages.
  First run green in 21s. Live: https://ai-experiri.github.io/nucleon/
- The public book tracks main. develop was fast-forward merged into
  main so the first published version is current; from now on the site
  updates when develop merges to main.

## Part II ordering decisions (user)

- Teaching order is progressive enhancement: load the model and
  generate with the existing generic ops only — no KV cache, no fused
  kernels — record tokens/sec, then land the cache and the kernels as
  measured improvements. Plan step 5 annotated; SUMMARY reordered
  (Loop before Cache).
- Loader comes before tokenizer in the narrative: load the model
  first, then meet the ids problem that motivates the tokenizer. The
  Qwen tokenizer ships in the package; ours wraps it thinly.
- Gaps the user asked to have named in the plan, now threaded into the
  chapter: golden test as the truth check, layer-by-layer diff as the
  debugging method, stop on both EOS ids, chat template as the owner
  of special tokens, padded-vocab guard in sampling, downloading the
  files, and recording the baseline before improving anything.

## The Engine chapter (5.x)

- New Part II opener docs/book/src/03-engine.md: what Part I left,
  families and the two gates (correctness, speed), the package on
  disk (with diagrams/engine-package.svg mapping each file to its
  consuming block), the strict-loader contract, the op inventory audit
  (every op a Qwen3 decode step needs is already in the trait with
  CPU/GPU parity), the build order with the plain-first rule, and
  upcoming engine topics (batching, server API, MTP, cuda).
- Stale chapter table in 00-big-picture.md 1.7 refreshed: backend row
  removed (its content lives in Tensor 0/Metal 0), deepseek2 row
  replaced by the qwen3_5 flagship, numbering matched to the sidebar.
- One near-miss recorded: the chapter briefly claimed Qwen3 MoE has
  128 experts, a number nobody verified this session; replaced with
  "a set of routed experts" before commit. Verify-before-claiming
  applies to throwaway parentheticals too.

## Branch

- Work branch for Part II created off develop: feature/engine (named
  the-engine for about a minute until the user picked the final name).
