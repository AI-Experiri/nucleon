# CLAUDE.md — nucleon working agreement

Hard requirements for any agent changing this repo. Distilled from the higgs
and jigglebot working agreements, adapted to nucleon's purpose.

## What nucleon is

A **learning project and future book**: a pure-Rust LLM inference engine for
Apple Silicon, built from scratch. Readability and learnability outrank
performance. The journey is a first-class artifact.

**Prime directives:**
- **Pure Rust.** No C/C++, no FFI to llama.cpp/MLX/anything. The ONLY non-Rust
  files allowed are `nucleon-metal/kernels/*.metal` shaders.
- **Lego blocks.** Each module is small, single-purpose, depends only on the
  layer below, and is testable in isolation. A block must be rewritable without
  touching its neighbors. If a file grows past ~300 lines of logic, that's a
  smell — split it.
- **One concern per file inside a module folder** (user hard rule): a module
  is a folder of small files (tensor/ = core.rs, access.rs, display.rs, later
  view.rs...), each with its own sibling _tests.rs. Never grow a monolith
  file named after the module.
- **CpuBackend is the reference.** Every backend op is verified against it.
  Keep CpuBackend naive and single-steppable — never "optimize" it.
- **Readable beats clever.** Code a book reader can follow: explicit loops over
  iterator-chain golf, named intermediate values, no macro magic in the core.

## Teaching mode (the point of the project — HARD RULE)

This is a LEARNING project. The user is learning Rust and engine-building by
watching it happen. Therefore:

- **Never batch-create silently.** Before writing any file or code block,
  explain WHAT it is, WHY it exists, and WHAT Rust/engine concept it
  introduces — at teaching level, not changelog level.
- Work rhythm per block: explain the concept → deep research → discuss the
  design with the user → build via TDD with the code explained as it's
  written → user reviews before the next block.
- Theory that llm-lab.github.io already teaches interactively (tokenization,
  RoPE, KV cache, scaling, quantization) gets LINKED, not re-explained —
  the book chapters carry the build story, the labs carry the theory.

## Book rules (docs/book/)

- POV STRUCTURE (user decision): like Game-of-Thrones POV chapters, each
  chapter is one component telling its story — its birth, its job, its
  relationships to the other components, what it refuses to be. Chapter
  titles are the character's name, and characters RECUR with numbered
  chapters like GoT characters get multiple chapters: "Tensor 0" is the
  birthplace, "Tensor 1", "Tensor 2" return when the story demands more
  of it (views for the cache, map for the sampler...). Deferred material always uses the SAME
  section title pattern, placed LAST in the chapter: "Upcoming
  <character> topics" (Upcoming tensor topics, Upcoming GPU topics, ...)
  — the menu future chapters draw from. Never invent a new heading for
  the same concept. Parts group chapters into phases of the larger
  story; grouping evolves as the book grows. The story frame carries the
  chapter, but the content inside stays plain technical register (no
  purple prose).

- The book teaches this project FROM SCRATCH to a reader who knows neither
  Rust nor engines. Every Rust construct gets a "Rust:" side box at its
  FIRST appearance (struct, Vec, ::, ! macros, &/&mut, impl, #[derive],
  closures, ?, match, trait, unsafe, lifetimes...). HARD LIMIT: a box is
  at most TWO lines of explanation plus one link to the official Rust
  docs (doc.rust-lang.org — the Book, std, or reference). Where sibling
  constructs exist, name them in the box (pub -> private default,
  pub(crate); Vec -> array [T; N], slice &[T]; match -> if let; ...) so
  the reader learns the family, not just the instance. Later chapters
  build on earlier boxes instead of repeating them.
- The book is NOT a transcript: chat discussions, review-process
  bookkeeping, and session mechanics stay in journal/. Only what teaches
  goes in a chapter, written for the from-scratch reader.
- Section headings are numbered chapter.section (## 3.4 ...), where the
  chapter number matches the sidebar numbering, so sections are easy to
  reference. Renumber a chapter's sections whenever one is added or
  removed.
- Meta-commentary about the book's own conventions ("new constructs get
  a side box", "Rust notes continue from chapter 1") lives ONLY in the
  introduction's "How to read this book" section — never in chapters.
- NEVER run `mdbook build` while `mdbook serve` is running (check
  lsof -ti:3000): a manual build overwrites the served pages WITHOUT the
  live-reload script and kills auto-refresh. serve rebuilds by itself on
  any change under docs/book/src; just edit and save.
- Diagrams as SVG: standalone files in docs/book/src/diagrams/ embedded
  via <div class="diagram"><img src="diagrams/x.svg"></div>. NEVER
  inline <svg> in markdown: blank lines break the raw-HTML block and
  the HTML pass lowercases case-sensitive SVG attributes (viewBox).
- NO UNINTRODUCED TERMS: a chapter may only use ops, models, and jargon
  already defined in that chapter or an earlier one. Before using a term
  in an explanation (rmsnorm, Qwen3, "composed"), define it in place or
  point at where it was defined. A lesson that name-drops is a failed
  lesson.
- HARDWARE WORDS get the same treatment as Rust constructs: at first use,
  a short "hardware words, with references" list defines each term
  (package, die, ALU lane, SRAM, LPDDR, SLC, ANE...) in one line with a
  reference link. Never assume a reader knows silicon vocabulary.
- NEVER a wall of text where a diagram can carry the concept — this
  applies to the book, replies, and docs alike. Memory layouts, data
  flows, hardware structure, execution timelines, thread grids: draw
  them (ASCII). Prose supports the diagram, not the other way around.
- COMPARISONS ARE TABLES (user hard rule): whenever two or more
  things are being compared or contrasted — models, families,
  formats, options, measurements — render a table or diagram, never
  prose. Same for model/family fact sheets (sizes, dims, layer maps).
- Book rendering conventions (theme/rust-notes.css): blockquotes are the
  Rust boxes (crab label, orange); diagrams render as labeled steel-blue
  cards, either ```text fences (ASCII) or inline SVG wrapped in
  <div class="diagram"> (preferred for figures worth the effort; use
  currentColor for strokes/text so both themes work, rgba fills for
  tints); real code keeps its language tag (rust, bash, metal) and
  normal syntax highlighting.

## Journey documentation (required, per session)

- Every working session appends a `journal/NNNN-YYYY-MM-DD-title.md` entry:
  what was tried, what broke, wrong turns included. Written as you go, never
  retro-fitted. Failures are the book's best material — record them.
- Every consequential choice gets an ADR in `docs/decisions/NNN-title.md`.
- Every module gets a `README.md` written **when the module is born**
  (what it does, why it exists, how to read it).
- Milestones get git tags (`m1-cpu-hello`, `m2-metal-parity`, …).

## Review convergence (required before any change is "done")

Same protocol as higgs — the canonical rules live in
`higgs/CLAUDE.md`; summary:

1. Review every non-trivial change with INDEPENDENT review agents
   in a loop (user policy). Two reviewer kinds, either or mixed:
   - `codex exec --skip-git-repo-check '<scoped prompt>'`
     (`-m gpt-5.5 -c model_reasoning_effort="xhigh"`);
   - Claude Opus subagents (Agent tool, model opus), each round a
     FRESH agent with no prior conclusions preloaded.
2. **Validate every finding yourself by reading the code.** Fix real ones;
   dismiss false positives with a one-line file:line evidence note.
3. **Converged = 3 consecutive stable rounds** (clean, or only
   already-assessed items). A new real bug resets the count.
4. **Reviews are neutral.** Prompts give facts only — never "do not re-flag",
   never pre-load conclusions. Invite codex to CHALLENGE the decisions you are
   least sure of. Scope reviews to the TASK DIFF, never the whole repo;
   out-of-diff findings become follow-up candidates.
5. Every real fix ships a test **proven to fail if the fix is reverted**
   (actually revert, run, see it fail, restore). Testable seams, never
   test-only knobs in production code.

## Test layout (higgs convention, verbatim)

- Unit tests in a SEPARATE sibling file, never inline:
  `src/<name>.rs` ends with the only test line allowed in a prod file:

  ```rust
  #[cfg(test)]
  #[path = "<name>_tests.rs"]
  mod tests;
  ```

  `<name>_tests.rs` starts with `use super::*;`. No inline
  `#[cfg(test)] mod tests { … }` blocks.
- `mod.rs` files are export barrels only — no logic, no tests.
- Integration tests live in `tests/` — end-to-end: load a real (tiny) model,
  generate, assert tokens.
- **Op-parity tests**: every non-CPU backend op is tested against CpuBackend
  within tolerance. **Golden tests**: fixed prompt + greedy sampling → exact
  token ids, guarding whole-stack refactors.

## Quality gate

Before calling any change done: `./scripts/quality.sh`
(fmt apply+verify → clippy --all-targets -D warnings → test --workspace).
Coverage gates (llvm-cov, user policy): unit tests >= 90%, integration
tests >= 85%, following the higgs model — the unit gate measures
production lines only. Integration coverage applies once `tests/`
end-to-end tests exist (the loop chapter's golden test onward).

## Workflow rules (from jigglebot, they apply here too)

- **Terse replies.** Announce → execute → summarize. No walls of text.
  HARD CAP: a chat answer to a question is a few sentences, or a
  diagram/table plus a few sentences. If more seems needed, give the
  short answer and ask which part to expand.
- **Lists over paragraphs.** Multiple parallel points = a list, never
  prose. Applies to replies, book, and docs.
- **Verify before claiming.** Any factual claim about an external
  library/tool (PyTorch, candle, NumPy, Metal...) is checked against its
  official docs (subagent research) before it lands in the book or an
  answer.
- **Read before answering.** Open the file; never answer from memory or names.
- **Diagnose before fixing.** Prove the root cause (repro test, log line, or
  read-trace) before touching code. Speculative fix = bug.
- **One home per piece of data.** A field lives in exactly one struct; others
  wrap or reference it. No parallel shapes.
- **Ruthless naming consistency.** Use existing names verbatim; distinct
  concepts get distinct names; renames need explicit user permission.
- **Context preservation.** Research, exploration, and web searches go through
  subagents; the main context is for decisions.
- **No magic ints.** Model intent with enums/Option/newtypes; raw ints only at
  the wire/GPU-buffer edge.

## Commit discipline

- Branching: day-to-day work happens on `develop`; `main` receives
  merges (user decision). Remote: git@github.com:AI-Experiri/nucleon.

- One logical unit per commit; conventional-commit subject.
- End commit messages with the `Co-Authored-By` trailer.
- Commit/push only when asked.

## Writing style (HARD RULE — applies to all prose: book, docs, journal, replies)

Follow the tropes guide below for every piece of writing in this project.
Any single pattern used once might be fine; multiple tropes together or one
trope repeated is a failure. Write like a human: varied, imperfect, specific.

Beyond the specific tropes, keep a PLAIN TECHNICAL REGISTER: no aphoristic
punchlines ("That is the whole thing."), no cute invented metaphors ("the
data changes costume"), no gimmick headings ("The five ideas that make it
make sense"), no manufactured drama. State facts, give numbers, explain
mechanisms. The voice to imitate is a good engineering book, not a viral
blog post. mdBook smart punctuation stays OFF (curly quotes are a tell).

METAPHORS ARE BANNED everywhere (book, diagrams, replies) — user hard
rule after "the pen" for encoder. If a metaphor ever seems genuinely
worth it, ASK the user first; never ship one unasked.

# AI Writing Tropes to Avoid

Source: [tropes.fyi](https://tropes.fyi) by [ossama.is](https://ossama.is)

## Word Choice

### "Quietly" and Other Magic Adverbs

Overuse of "quietly" and similar adverbs to convey subtle importance or understated power. AI reaches for these adverbs to make mundane descriptions feel significant. Also includes: "deeply", "fundamentally", "remarkably", "arguably".

**Avoid patterns like:**
- "quietly orchestrating workflows, decisions, and interactions"
- "the one that quietly suffocates everything else"
- "a quiet intelligence behind it"

### "Delve" and Friends

Used to be the most infamous AI tell. "Delve" went from an uncommon English word to appearing in a staggering percentage of AI-generated text. Part of a family of overused AI vocabulary including "certainly", "utilize", "leverage" (as a verb), "robust", "streamline", and "harness".

**Avoid patterns like:**
- "Let's delve into the details..."
- "Delving deeper into this topic..."
- "We certainly need to leverage these robust frameworks..."

### "Tapestry" and "Landscape"

Overuse of ornate or grandiose nouns where simpler words would do. "Tapestry" is used to describe anything interconnected. "Landscape" is used to describe any field or domain. Other offenders: "paradigm", "synergy", "ecosystem", "framework".

**Avoid patterns like:**
- "The rich tapestry of human experience..."
- "Navigating the complex landscape of modern AI..."
- "The ever-evolving landscape of technology..."

### The "Serves As" Dodge

Replacing simple "is" or "are" with pompous alternatives like "serves as", "stands as", "marks", or "represents". AI avoids basic copulas because its repetition penalty pushes it toward fancier constructions.

**Avoid patterns like:**
- "The building serves as a reminder of the city's heritage."
- "Gallery 825 serves as LAAA's exhibition space for contemporary art."
- "The station marks a pivotal moment in the evolution of regional transit."

## Sentence Structure

### Negative Parallelism

The "It's not X -- it's Y" pattern, often with an em dash. The single most commonly identified AI writing tell. AI uses this to create false profundity by framing everything as a surprising reframe. One in a piece can be effective; ten in a blog post is a genuine insult to the reader. Includes the causal variant "not because X, but because Y", the em-dash dismissal "X -- not Y", and the cross-sentence reframe: "The question isn't X. The question is Y."

**Avoid patterns like:**
- "It's not bold. It's backwards."
- "Feeding isn't nutrition. It's dialysis."
- "Half the bugs you chase aren't in your code. They're in your head."

### "Not X. Not Y. Just Z."

The dramatic countdown pattern. AI builds tension by negating two or more things before revealing the actual point. Creates a false sense of narrowing down to the truth.

**Avoid patterns like:**
- "Not a bug. Not a feature. A fundamental design flaw."
- "Not ten. Not fifty. Five hundred and twenty-three lint violations across 67 files."
- "not recklessly, not completely, but enough"

### "The X? A Y."

Self-posed rhetorical questions answered immediately in the next sentence or clause. The model asks a question nobody was asking, then answers it for dramatic effect.

**Avoid patterns like:**
- "The result? Devastating."
- "The worst part? Nobody saw it coming."
- "The scary part? This attack vector is perfect for developers."

### Anaphora Abuse

Repeating the same sentence opening multiple times in quick succession.

**Avoid patterns like:**
- "They assume that users will pay... They assume that developers will build... They assume that..."
- "They could expose... They could offer... They could provide... They could create..."
- "They have built engines, but not vehicles. They have built power, but not leverage."

### Tricolon Abuse

Overuse of the rule-of-three pattern, often extended to four or five. A single tricolon is elegant; three back-to-back tricolons are a pattern recognition failure.

**Avoid patterns like:**
- "Products impress people; platforms empower them. Products solve problems; platforms create worlds. Products scale linearly; platforms scale exponentially."
- "identity, payments, compute, distribution"
- "workflows, decisions, and interactions"

### "It's Worth Noting"

Filler transitions that signal nothing. Also includes: "It bears mentioning", "Importantly", "Interestingly", "Notably".

**Avoid patterns like:**
- "It's worth noting that this approach has limitations."
- "Importantly, we must consider the broader implications."
- "Interestingly, this pattern repeats across industries."

### Superficial Analyses

Tacking a present participle ("-ing") phrase onto the end of a sentence to inject shallow analysis that says nothing: "highlighting its importance", "reflecting broader trends", "contributing to the development of...".

**Avoid patterns like:**
- "contributing to the region's rich cultural heritage"
- "This etymology highlights the enduring legacy of the community's resistance..."
- "underscoring its role as a dynamic hub of activity and culture"

### False Ranges

Using "from X to Y" constructions where X and Y aren't on any real scale. In legitimate use, "from X to Y" implies a spectrum with a meaningful middle. AI uses it as a fancy way to list two loosely related things.

**Avoid patterns like:**
- "From innovation to implementation to cultural transformation."
- "From the singularity of the Big Bang to the grand cosmic web."
- "From problem-solving and tool-making to scientific discovery, artistic expression, and technological innovation."

## Paragraph Structure

### Short Punchy Fragments

Excessive use of very short sentences or sentence fragments as standalone paragraphs for manufactured emphasis. It's an inhuman style; no real person writes first drafts this way.

**Avoid patterns like:**
- "He published this. Openly. In a book. As a priest."
- "These weren't just products. And the software side matched. Then it professionalised. But I adapted."
- "Platforms do."

### Listicle in a Trench Coat

Numbered or labeled points dressed up as continuous prose: "The first... The second... The third..." wrapped in paragraphs to disguise the list format.

**Avoid patterns like:**
- "The first wall is the absence of a free, scoped API... The second wall is the lack of delegated access... The third wall is..."
- "The second takeaway is that... The third takeaway is that... The fourth takeaway is that..."

## Tone

### "Here's the Kicker"

False suspense transitions that promise a revelation but deliver a point that did not need the buildup. Also includes: "Here's the thing", "Here's where it gets interesting", "Here's what most people miss", "Here's the deal".

**Avoid patterns like:**
- "Here's the kicker."
- "Here's the thing about AI adoption."
- "Here's where it gets interesting."

### "Think of It As..."

The patronizing analogy. The model defaults to teacher mode and assumes the reader needs a metaphor to understand anything. Often produces analogies less clear than the original concept.

**Avoid patterns like:**
- "Think of it like a highway system for data."
- "Think of it as a Swiss Army knife for your workflow."
- "It's like asking someone to buy a car they're only allowed to sit in while it's parked."

### "Imagine a World Where..."

The classic AI invitation to futurism: "Imagine" followed by a list of wonderful things that will happen if the reader agrees with the premise.

**Avoid patterns like:**
- "Imagine a world where every tool you use has a quiet intelligence behind it..."
- "In that world, workflows stop being collections of manual steps and start becoming orchestrations."

### False Vulnerability

Simulated self-awareness or honesty that reads as performative. Real vulnerability is specific and uncomfortable; AI vulnerability is polished and risk-free.

**Avoid patterns like:**
- "And yes, I'm openly in love with the platform model"
- "And yes, since we're being honest: I'm looking at you, OpenAI, Google, Anthropic, Meta"
- "This is not a rant; it's a diagnosis"

### "The Truth Is Simple"

Asserting that something is obvious, clear or simple instead of actually proving it. If you have to tell the reader your point is clear, it very likely isn't. Includes the dramatic reveal variant: "but none of them is the real story. The real story is..."

**Avoid patterns like:**
- "The reality is simpler and less flattering"
- "History is unambiguous on this point"
- "History is clear, the metrics are clear, the examples are clear"

### Grandiose Stakes Inflation

Everything is the most important thing ever. A blog post about API pricing becomes a meditation on the fate of civilization.

**Avoid patterns like:**
- "This will fundamentally reshape how we think about everything."
- "will define the next era of computing"
- "something entirely new"

### "Let's Break This Down"

The pedagogical voice that assumes the reader needs hand-holding, even for expert audiences. Also includes: "Let's unpack this", "Let's explore", "Let's dive in".

**Avoid patterns like:**
- "Let's break this down step by step."
- "Let's unpack what this really means."
- "Let's explore this idea further."

### Vague Attributions

Attributing claims to unnamed authorities: "experts", "observers", "industry reports", "several publications". If you can't name the expert, you don't have a source.

**Avoid patterns like:**
- "Experts argue that this approach has significant drawbacks."
- "Industry reports suggest that adoption is accelerating."
- "Observers have cited the initiative as a turning point."

### Invented Concept Labels

Compound labels that sound analytical without being grounded: abstract problem-nouns (paradox, trap, creep, divide, vacuum, inversion) appended to domain words and used as if they're established terms. They name a thing and skip the argument.

**Avoid patterns like:**
- "the supervision paradox"
- "the acceleration trap"
- "workload creep"

## Formatting

### Em-Dash Addiction

Compulsive overuse of em dashes for dramatic pauses, parenthetical asides and pivot points. A human writer might use 2-3 per piece; AI will use 20+.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "The tinkerer spirit didn't die of natural causes -- it was bought out."
- "Not recklessly, not completely -- but enough -- enough to matter."

### Double-Hyphen Dash

The em dash wearing a false moustache. The compulsive mid-sentence pivot survives the character substitution, which is what actually gives it away. Flagged at five or more per thousand words.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "It's not a rewrite -- it's a reckoning."
- "We shipped it fast -- maybe too fast -- and paid for it later."

### Bold-First Bullets

Every bullet point or list item starts with a bolded phrase. Almost nobody formats lists this way when writing by hand.

**Avoid patterns like:**
- "Every single bullet point begins with a bold keyword."
- "**Security**: Environment-based configuration with..."
- "**Performance**: Lazy loading of expensive resources..."

### Unicode Decoration

Unicode arrows, smart/curly quotes, and other characters that can't be easily typed on a standard keyboard. Real writers produce straight quotes and -> or =>.

**Avoid patterns like:**
- "Input → Processing → Output"
- "This leads to better outcomes → which means higher engagement"
- "Smart quotes instead of the straight quotes you'd actually type"

## Composition

### Fractal Summaries

"What I'm going to tell you; what I'm telling you; what I just told you" applied at every level. Every subsection gets a summary, every section gets a summary, the document gets a summary.

**Avoid patterns like:**
- "In this section, we'll explore... [3000 words later] ...as we've seen in this section."
- "A conclusion that restates every point already made"
- "And so we return to where we began."

### The Dead Metaphor

Latching onto a single metaphor and beating it into the ground. A human introduces a metaphor, uses it, moves on. AI repeats the same metaphor 5-10 times.

**Avoid patterns like:**
- "The ecosystem needs ecosystems to build ecosystem value."
- "Walls and doors used 30+ times in the same article"
- "Every paragraph finds a way to say 'primitives' again"

### Historical Analogy Stacking

Rapid-fire listing of historical companies or tech revolutions to build false authority. Especially common in technical writing.

**Avoid patterns like:**
- "Apple didn't build Uber. Facebook didn't build Spotify. Stripe didn't build Shopify."
- "Every major technological shift -- the web, mobile, social, cloud -- followed the same pattern."
- "Take Spotify... Or consider Uber... Airbnb followed a similar path... Even Discord..."

### One-Point Dilution

Making a single argument and restating it in 10 different ways across thousands of words. An 800-word argument becomes 4000 words of circular repetition.

**Avoid patterns like:**
- "The same point, restated eight ways across 4000 words."
- "Each section rephrases the thesis with a different metaphor but adds nothing new"

### Content Duplication

Repeating entire sections or paragraphs verbatim within the same piece.

**Avoid patterns like:**
- "The same section appeared twice, word-for-word identical."
- "Paragraph 3 and paragraph 17 are the same sentence reworded"

### The Signposted Conclusion

Explicitly announcing the conclusion with "In conclusion", "To sum up", or "In summary". Competent writing doesn't need to tell you it's concluding.

**Avoid patterns like:**
- "In conclusion, the future of AI depends on..."
- "To sum up, we've explored three key themes..."
- "In summary, the evidence suggests..."

### "Despite Its Challenges..."

The rigid formula where AI acknowledges problems only to immediately dismiss them: "Despite its [positives], [subject] faces challenges..." then "Despite these challenges, [optimistic conclusion]."

**Avoid patterns like:**
- "Despite these challenges, the initiative continues to thrive."
- "Despite its industrial and residential prosperity, Korattur faces challenges typical of urban areas."
- "Despite their promising applications, pyroelectric materials face several challenges..."

Remember: any of these patterns used once might be fine. The problem is when
multiple tropes appear together or when a single trope is used repeatedly.
Write like a human: varied, imperfect, specific.
