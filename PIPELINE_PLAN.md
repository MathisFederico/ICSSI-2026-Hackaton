# Altendor → DeltaBay Endorsement Extraction Pipeline

> Master plan for the **real-endorsement pipeline** powering DeltaBay during
> the ICSSI 2026 hackathon (Sunday 28 June 2026, CU Boulder).
> **Living document** — tick boxes in §3 as stages land. Edit ticket
> sections when constraints change.

---

## 1. Context

DeltaBay currently renders endorsements/flags produced by a seeded RNG
simulator in `DeltaBay/frontend/app/src/data/mira-graph-loader.ts`. We want
**real** endorsements derived from social-media discourse about real papers,
populating the existing skeletal debate **"The optimal scientific system"**
in `DeltaBay/frontend/app/src/data/funding-debate.ts` (3 Questions, no
Answers yet). High-level flow:

```
OpenAlex keyword search ∪ pinned DOIs
  └─→ Altmetric on GBQ (papers + posts)
        └─→ Enrich post text via free APIs (Bluesky AT Protocol, Reddit PRAW)
              └─→ Claude classifies each post:
                    • Endorsement (claim_text + signed magnitude in decibans)
                    • Flag (DeltaBay category)
                    • Irrelevant (dropped)
              └─→ Depth-1 reply traversal on endorsements (Bluesky + Reddit)
              └─→ Cluster claim_texts → canonical claims per paper
              └─→ Route each paper to one of 3 Questions
              └─→ Assemble neutral JSON
                    └─→ Thin DeltaBay loader builds JSON-LD
```

**Scope**: ~10 papers, reply traversal on.
**Output**: altendor JSON + thin DeltaBay-side loader.
**Auth**: Bluesky public read; Reddit script app (`REDDIT_CLIENT_ID/SECRET` in `.env`).
**Magnitude**: signed integer in `[-30, +30]` decibans, calibrated from a
small hand-labeled set.

---

## 2. How we'll work

- **This file is the source of truth** for plan state. Tick boxes in §3 as
  stages land.
- Each stage in §6 is a **self-contained ticket** with everything a
  contributor needs: goal, dependencies, inputs, outputs, file paths,
  reusable utilities, tests, and a Done-when checklist. Pick a stage with
  satisfied dependencies, do the work, tick the box.
- **Schemas are derived from usage, not defined upfront.** Each producing
  stage owns its own local dataclass / DataFrame column list. The canonical
  shared schema (`IntermediateDebate`) is defined at the **integration
  boundary** (S16) once upstream producers and the DeltaBay consumer
  contract are both visible. §5.1 records the consumer's *target* shape but
  fields stay flexible until S16.
- Cross-cutting concerns (deciban rubric, testing strategy, path conventions,
  reuse map) live in §5 and are *referenced* by stage tickets, never
  duplicated.
- If a stage uncovers a constraint that affects later stages, edit those
  tickets *immediately* — keep the plan honest.

---

## 3. Master checklist

### Foundation
- [ ] **S0.** Bootstrap altendor packaging, lint/type/test, output paths, gitignore
- [ ] **S2.** Deciban rubric + calibration JSONL skeleton

### Data acquisition (each stage owns its local shape)
- [ ] **S3.** OpenAlex keyword/topic source → DOI list
- [ ] **S4.** BigQuery query builders + preflight wiring
- [ ] **S5.** Altmetric source: DOIs → ranked papers → posts
- [ ] **S6.** Enrichment: Bluesky text + thread resolution
- [ ] **S7.** Enrichment: Reddit text + reply resolution
- [ ] **S8.** Text resolver dispatcher (uses S6, S7; falls back to `posts.title`)

### Classification
- [ ] **S9.** Classifier prompt (system + exemplars), cached
- [ ] **S10.** Single-post classifier (Claude tool-use → local discriminated union)
- [ ] **S11.** Batch classifier (Anthropic Batches API)
- [ ] **S12.** Calibration gate (notebook 4 + pytest live test)
- [ ] **S13.** Reply traversal (depth 1, Bluesky + Reddit)

### Assembly (integration boundary — canonical schema crystallizes HERE)
- [ ] **S14.** Claim clustering (Haiku one-shot, 3..7 clusters/paper)
- [ ] **S15.** Paper → Question routing (Haiku, diversification)
- [ ] **S16.** Define `IntermediateDebate` schema (from upstream outputs + DeltaBay loader needs) + build it
- [ ] **S17.** Write neutral `debate.json`

### Notebooks (linear orchestrators)
- [ ] **S18.** Notebook `1_select_papers.ipynb` (dry-run + run, separate cells)
- [ ] **S19.** Notebook `2_gather_posts.ipynb`
- [ ] **S20.** Notebook `3_enrich_text.ipynb`
- [ ] **S21.** Notebook `4_calibrate_classifier.ipynb` (gate)
- [ ] **S22.** Notebook `5_classify_batch.ipynb` (+ fallback toggle)
- [ ] **S23.** Notebook `6_traverse_replies.ipynb`
- [ ] **S24.** Notebook `7_cluster_claims.ipynb`
- [ ] **S25.** Notebook `8_route_and_assemble.ipynb`

### Integration & verification
- [ ] **S26.** DeltaBay-side `altendor-loader.ts` + `altendor/debate.json` import
- [ ] **S27.** Merge loader output into `funding-debate.ts` (skeleton fallback preserved)
- [ ] **S28.** Vitest `funding-debate.altendor.test.ts`
- [ ] **S29.** End-to-end smoke (`n=2`, `budget_usd=0.50`)
- [ ] **S30.** Notebook `9_publish_to_deltabay.ipynb` + visual click-through

### Demo polish
- [ ] **S31.** Notebook 8 final markdown summary cell (papers × Q × endorsements × flags × participants)
- [ ] **S32.** README / runbook in `altendor/`

---

## 4. Cost & budget

| Stage | Calls | Est. cost |
|------|------|-----------|
| Classification (Sonnet, cached prompt, Batches) | ~360 | $1.10 |
| Paper → Question routing (Haiku) | 10 | $0.02 |
| Claim clustering (Haiku) | 10 | $0.03 |
| Calibration iteration (Sonnet) | ~30 | $0.30 |

**Total ≈ $1.50**, conservative cap **$2.50**. Hard cap via
`CostTracker(budget_usd=5.0)`.

---

## 5. Cross-cutting references

These sections are referenced by individual stage tickets in §6. Stages
**link** here rather than restate.

### 5.1 DeltaBay consumer contract (target shape — finalised at S16)

This is the shape the DeltaBay loader needs to *consume* — derived from
`funding-debate.ts`, `mira-graph-loader.ts`, and `types.ts`. It is the
**constraint** at the integration boundary, not a pre-built foundation type.
The actual `IntermediateDebate` pydantic model is defined in S16 once
upstream producer shapes are visible; this section records "what the
consumer expects to see."

Pipeline writes `altendor/output/<run_id>/debate.json`. **Neutral** — no
JSON-LD `@type` or `@context`; those are added by the DeltaBay loader.

```json
{
  "run_id": "2026-06-28-001",
  "debate_id": "debate:optimal-funding",
  "generated_at": "2026-06-28T12:00:00Z",
  "questions": [{"id":"question:research-integrity","title":"...","shortTitle":"..."}],
  "participants": [
    {"id":"agent:bsky:did-plc-abc","name":"Jane R","handle":"jane.bsky.social","platform":"bluesky"}
  ],
  "papers": [{
    "doi":"10.1234/x", "ro_id":"alt:123", "title":"...", "abstract":"...",
    "altmetric_score": 412, "routed_question_id":"question:peer-review",
    "answer": {
      "id":"answer:alt-123", "title":"...", "shortTitle":"...",
      "evidence": {
        "id":"evidence:alt-123",
        "sourceDocument": {"id":"src:doi:10.1234/x", "doi":"10.1234/x", "url":"https://doi.org/..."},
        "endorsements":[...], "flags":[...]
      },
      "subclaims": [{
        "id":"claim:alt-123:c1","title":"<canonical claim>","memberPostIds":["..."],
        "endorsements":[{
          "id":"end:bsky:<postid>","participantId":"agent:bsky:did-plc-abc",
          "magnitude":12,"criterion":"Support","createdAt":"2026-06-26T...",
          "sourcePostUrl":"...","sourceText":"..."
        }],
        "flags":[...]
      }]
    }
  }]
}
```

All generated `@id`s use the **`altendor:`** namespace fragment
(e.g. `endorsement:altendor:bsky:<postid>`) to avoid colliding with
MIRA-seeded IDs.

### 5.2 DeltaBay JSON-LD shape (consumer side)

Produced by the loader, not by the pipeline directly. Sample shape lives in
`DeltaBay/frontend/app/src/data/funding-debate.ts`. The loader stamps
`@type`, applies `MIRA_CONTEXT` from `jsonld/context.ts`, dedupes
participants debate-wide, and calls `enrichDebateViews()`.

### 5.3 Deciban rubric

Anchors baked into the cached classifier system prompt as few-shot exemplars:

| Magnitude | Meaning |
|----------|---------|
| **+30** | Explicit strong endorsement of a specific claim, strong language |
| **+20** | Confident positive paraphrase of a claim with reasoning |
| **+10** | Mild positive — agrees but doesn't reason |
| **0**   | (excluded; zero-magnitude carries no information — drop) |
| **−10** | Mild critique or hedge |
| **−20** | Sharp critique with reasoning |
| **−30** | Explicit refutation |

`criterion ∈ {"Support", "Prior"}`. Default to `"Support"`. Use `"Prior"`
when the post endorses the *broader hypothesis* rather than the paper's
specific claim.

**Calibration gate (S12):** MAE ≤ 8 dB on `magnitude_dB` and kind-F1 ≥ 0.8
on `altendor/data/calibration/labeled.jsonl` (15–20 hand-labeled rows).

### 5.4 Path conventions

- Pipeline outputs: `altendor/output/<run_id>/{papers,posts,resolved_posts,classified,clusters,debate}.{parquet,json}` plus `manifest.json` (committable).
- Calibration data: `altendor/data/calibration/labeled.jsonl` (committable).
- Tests: `altendor/tests/{unit,fixtures,live}/`.
- Recorded fixtures: small (< 50 KB), hand-trimmed, committable.
- DeltaBay landing pad: `DeltaBay/frontend/app/src/data/altendor/debate.json`.
- `.gitignore` adds `altendor/output/*` except `**/manifest.json`.

### 5.5 Testing strategy (pragmatic layers)

Per-package layout, **no live network in default test run**.

- **Python** (`altendor/tests/`, pytest under `uv run pytest --durations=10`):
  - **Contract tests** — pydantic round-trip + field bounds on every schema.
  - **Recorded-fixture replays** — Bluesky thread, Reddit submission, GBQ row, OpenAlex page; parsers run on frozen JSON.
  - **Golden file** — `assemble/builder.py`, `cluster/claims.py` (mocked LLM), `route/question_router.py` (mocked LLM): exact JSON out.
  - **Classifier mocking** — `classify_post()` against a recorded Claude payload.
  - **Calibration gate** — runs the **real** classifier (live). `@pytest.mark.live`. Notebook 4 runs the same assertion.
  - **Smoke (live)** — `n=2` papers, `budget_usd=0.50`, full pipeline into a temp run_id. `@pytest.mark.live`.
- **TypeScript** (`DeltaBay/frontend/app/`, vitest):
  - `funding-debate.altendor.test.ts` — canned `altendor/debate.json` fixture; asserts every Question has ≥1 Answer, every Answer ≥1 Subclaim with ≥1 Endorsement, Evidence has non-null `sourceDocument.doi`, participantIds resolve, magnitudes in `[-30,30]`, `@type` correctly stamped.
- **Capture-once helper** — `python -m altendor.tests._capture {bluesky|reddit|gbq|openalex} <id-or-url>` writes trimmed JSON into `fixtures/`. Manual; used when external shapes drift.
- **Iteration discipline** — the calibration JSONL + MAE/F1 gate is the
  *only* test that constrains the classifier prompt wording. Do not lock
  prompt phrasing in unit tests. New stages start with their **contract
  test**; that's the only required "TDD". Then implementation +
  recorded-fixture replay + golden if pure.

### 5.6 Tool/utility reuse map

| Concern | Reuse |
|---------|-------|
| GBQ preflight | `altendor.bigquery.preflight` (existing) |
| GBQ schemas | `.claude/skills/sql/gbq_schema/altmetric/*.yaml` (existing) |
| Schema dump | `altendor.schema` (existing) |
| Claude client | `claude_kit.ClaudeClient` (from the hackathon submodule) |
| Cost tracking | `claude_kit.CostTracker` |
| Prompt caching | `claude_kit` system-prompt cache (5-min TTL — repeated calls win) |
| Conversation/agent | `claude_kit.Agent`, `Conversation` — not needed; classifier is one-shot |

---

## 6. Stage tickets

Each ticket is **self-contained**: it should be possible to do the work
having read only that section + any cross-cutting references it links to.

**Schema policy.** Each producer stage carries its **own local dataclass /
DataFrame column list** in its "Outputs" — no upfront foundation schema.
The canonical `IntermediateDebate` pydantic model is defined in **S16**
(the integration boundary) once upstream shapes and the DeltaBay consumer
contract (§5.1) are both visible.

---

### S0 — Bootstrap altendor packaging, lint/type/test, output paths, gitignore

**Goal.** Confirm the `altendor` package builds, lints, types, and tests
under uv; add the directory skeleton for new submodules and tests; set up
`output/` gitignore.

**Dependencies.** None.

**Inputs.**
- Existing `altendor/` package layout.
- Existing `.gitignore`.

**Outputs.**
- Empty `__init__.py` files for new submodules listed under "Implementation hints" below.
- `altendor/tests/__init__.py`, `altendor/tests/conftest.py` (basic markers: `live`).
- `altendor/tests/fixtures/.gitkeep`.
- `altendor/data/calibration/.gitkeep`.
- `altendor/output/.gitignore` (`*\n!.gitignore\n!**/manifest.json`).
- `pyproject.toml` test markers (`live`) added.

**Implementation hints.**
- New submodules to create as empty packages:
  `altendor/bigquery/queries.py`,
  `altendor/sources/{openalex.py,altmetric.py}`,
  `altendor/enrich/{bluesky.py,reddit.py,text_resolver.py}`,
  `altendor/classify/{schema.py,prompts.py,classifier.py,batch.py}`,
  `altendor/traverse/replies.py`,
  `altendor/cluster/claims.py`,
  `altendor/route/question_router.py`,
  `altendor/assemble/{intermediate.py,builder.py,deltabay_writer.py}`,
  `altendor/io/{paths.py,manifest.py}`.

**Tests.** None at this stage — verify `uv run pytest` runs cleanly,
`uv run ruff check altendor` passes, `uv run ty check altendor` passes.

**Done-when.**
- `uv run pytest altendor/` exits 0 (no tests collected ≠ failure).
- `uv run ruff check altendor` clean.
- `uv run ty check altendor` clean.
- `git status` shows the skeleton files staged.

---

### S2 — Deciban rubric + calibration JSONL skeleton

**Goal.** Lay down `altendor/data/calibration/labeled.jsonl` with the
**rubric anchors** (≥10 rows: 5 across the magnitude scale, 3 flag
categories, 2 irrelevant) and document the rubric.

**Dependencies.** S0.

**Reference.** §5.3.

**Outputs.**
- `altendor/data/calibration/labeled.jsonl` (~10 seed rows).
- `altendor/data/calibration/README.md` — rubric, how to add rows, anchor
  policy. The user hand-labels more rows during S12.

**Implementation hints.**
- JSONL row shape:
  ```json
  {"post_text":"...","paper_title":"...","paper_abstract":"...","gold":{"kind":"endorsement","claim_text":"...","magnitude_dB":20,"criterion":"Support"}}
  ```
- Pick 10 sci-of-sci anchors (peer review, replication, gender in grants) —
  fabricate plausible posts at clear magnitude levels for the seed set;
  user replaces with real captures during S12.

**Done-when.**
- File exists; loadable with `jsonlines`; each row passes a schema check.

---

### S3 — OpenAlex keyword/topic source

**Goal.** Implement `altendor/sources/openalex.py` returning a DOI list from
a keyword search (default `"science of science"`), with optional `topic_id`
or `concept_id` overrides.

**Dependencies.** S0.

**Reference.** §5.6.

**Outputs.**
- `altendor/sources/openalex.py` with:
  ```python
  def search_dois(query: str = "science of science", *,
                  topic_id: str | None = None,
                  concept_id: str | None = None,
                  n: int = 200) -> list[str]: ...
  def resolve_topic_by_name(name: str) -> dict | None: ...
  ```
  Defaults to `filter=title_and_abstract.search:<query>`; switches to
  `filter=topics.id:<id>` if `topic_id` set; `filter=concepts.id:<id>` if
  `concept_id` set. Polite pool: include `mailto=mathis.federico@bycelium.com`
  in URL params.
- `altendor/tests/unit/test_openalex_parse.py` — uses a recorded fixture
  `altendor/tests/fixtures/openalex_works_page.json` (use `_capture.py`
  once; hand-trim to ~5 works), asserts DOI extraction, polite-pool param.

**Implementation hints.**
- Use `httpx` sync client; paginate via `cursor=*`; cap at 200 results.
- DOI is at `result.doi` (URL-prefixed `https://doi.org/...`); strip prefix.
- Skip results missing DOI.

**Done-when.**
- `search_dois("science of science", n=10)` returns ≥5 DOIs live
  (manual run).
- Fixture-replay test passes.

---

### S4 — BigQuery query builders + preflight wiring

**Goal.** Implement `altendor/bigquery/queries.py` with SQL builders for
research-output lookup and posts fetch, plus integration with `preflight()`.

**Dependencies.** S0.

**Reference.** §5.6, `notebooks/0_altmetrics_gbq_setup.ipynb`,
`.claude/skills/sql/gbq_schema/altmetric/`.

**Outputs.**
- `altendor/bigquery/queries.py` with:
  ```python
  def research_outputs_by_dois(dois: list[str]) -> str: ...
  def posts_by_research_output_ids(ro_ids: list[str]) -> str: ...
  ```
- `altendor/tests/unit/test_queries.py` — golden SQL tests
  (parameter-substitution stability, no SQL injection on `dois`).

**Implementation hints.**
- Always use `UNNEST(@dois)`-style parameter binding via BigQuery scripting
  parameters — do NOT inline strings into SQL. Returns a query + params dict.
- Posts query: `UNNEST(research_outputs_ids) AS ro_id` join filter; project
  `id, type, subtype, date, url, title, attention_source, retweet, ro_id`.
- Always filter via `research_outputs_ids` early to satisfy the
  preflight 50 GiB cap.

**Done-when.**
- Golden tests pass.
- `preflight()` returns under 1 GiB for a 10-DOI posts query (verified in
  notebook 2, not in CI).

---

### S5 — Altmetric source: DOIs → ranked papers → posts

**Goal.** Implement `altendor/sources/altmetric.py` to take DOIs through
`research_outputs` to a ranked paper list, then through `posts` to the
post DataFrame.

**Dependencies.** S0, S4.

**Reference.** §5.4 paths.

**Local shape (this stage owns it).**
- `papers` DataFrame columns: `doi: str, ro_id: str, title: str, abstract: str | None, altmetric_score: float, last_mentioned_at: datetime`.
- `posts` DataFrame columns: `post_id: str, ro_id: str, type: str, subtype: str | None, date: datetime, url: str, title: str, attention_source: dict, retweet: bool`.

**Outputs.**
- `altendor/sources/altmetric.py`:
  ```python
  def join_dois_to_attention(client, dois: list[str]) -> pd.DataFrame: ...
  # columns: doi, ro_id, title, abstract, altmetric_score, last_mentioned_at
  def top_papers(df, k: int = 10, pinned_dois: list[str] | None = None) -> pd.DataFrame: ...
  def fetch_posts_for_papers(client, ro_ids: list[str]) -> pd.DataFrame: ...
  ```
- `altendor/tests/unit/test_altmetric_join.py` — fixture replay against a
  recorded `BigQuery` result row.

**Implementation hints.**
- `top_papers` always *includes* pinned DOIs even if below the score cutoff.
- Drop rows with `altmetric_score = 0` from the ranked set.
- `fetch_posts_for_papers` returns one row per `(ro_id, post_id)`; explode
  `research_outputs_ids` server-side.

**Done-when.**
- Fixture-replay tests pass.
- Live smoke (manual): 10-DOI run returns ≥30 posts.

---

### S6 — Enrichment: Bluesky text + thread resolution

**Goal.** Implement `altendor/enrich/bluesky.py` (async) to resolve a post
URL to its full text and (later, S13) its depth-1 replies/quotes.

**Dependencies.** S0.

**Outputs.**
- `altendor/enrich/bluesky.py`:
  ```python
  @dataclass
  class BskyPost: at_uri: str; cid: str; text: str; author_did: str; author_handle: str; created_at: str
  async def resolve_post(url_or_uri: str) -> BskyPost | None: ...
  async def get_thread(at_uri: str, depth: int = 1, parent_height: int = 0) -> list[BskyPost]: ...
  ```
- `altendor/tests/unit/test_bluesky_parse.py` — fixture replay against
  `altendor/tests/fixtures/bluesky_thread.json`.

**Implementation hints.**
- Use `https://public.api.bsky.app/xrpc/` base; no auth.
- `app.bsky.feed.getPostThread?uri=<at-uri>&depth=1&parentHeight=0`.
- Resolve `bsky.app/profile/<handle>/post/<rkey>` → AT URI via
  `com.atproto.identity.resolveHandle` + AT URI construction.
- Wrap network calls in `try/except`; return `None` on failure; log to a
  module-level logger.
- `aiohttp.ClientSession` + `asyncio.Semaphore(8)`.

**Done-when.**
- Fixture-replay tests pass.
- Live smoke (manual): resolves a known public Bluesky URL to non-empty text.

---

### S7 — Enrichment: Reddit text + reply resolution

**Goal.** Implement `altendor/enrich/reddit.py` using PRAW with a
script-type app. Resolve a submission or comment URL to its full text;
fetch depth-1 replies.

**Dependencies.** S0.

**Env.** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
in `.env` (added to `.env.example`).

**Outputs.**
- `altendor/enrich/reddit.py`:
  ```python
  @dataclass
  class RedditNode: id: str; kind: Literal["submission","comment"]; author: str; body: str; created_utc: float; permalink: str
  def resolve_node(url: str) -> RedditNode | None: ...
  def get_replies(node: RedditNode, depth: int = 1, max_per_parent: int = 20) -> list[RedditNode]: ...
  ```
- `altendor/tests/unit/test_reddit_parse.py` — replay against
  `altendor/tests/fixtures/reddit_submission.json`.

**Implementation hints.**
- PRAW: `praw.Reddit(client_id=..., client_secret=..., user_agent=...)`.
  Read-only by default.
- For submissions: `body = submission.selftext or submission.title`.
- For comments: `body = comment.body`.
- `submission.comments.replace_more(limit=0)`; take top-level replies only.
- Author handling for deleted accounts: `author = "[deleted]"`.
- Rate-limited: semaphore=4 + retry-on-429 with backoff.

**Done-when.**
- Fixture-replay tests pass.
- Live smoke (manual): resolves a known public Reddit URL.

---

### S8 — Text resolver dispatcher

**Goal.** Implement `altendor/enrich/text_resolver.py` — dispatch by
`posts.type/subtype` to S6, S7, or pass-through.

**Dependencies.** S6, S7.

**Outputs.**
- `altendor/enrich/text_resolver.py`:
  ```python
  @dataclass
  class ResolvedPost: post_id: str; platform: str; text: str; author_handle: str | None; author_id: str | None; url: str; created_at: str; raw_title: str
  async def resolve_full_text(post_row: dict, *, bsky_session, reddit_client) -> ResolvedPost: ...
  ```
- `altendor/tests/unit/test_text_resolver.py` — three fixtures (bsky,
  reddit, blog) → dispatch behaviour.

**Implementation hints.**
- Dispatch table on `post.type`: `"bsky"` → S6; `"rdt"` → S7; others →
  pass-through (`text = post.title`).
- Catch resolver exceptions per-post; degrade to `posts.title`; log.
- Truncation flag: if `text == raw_title and len(text) < 140`, mark
  `text_confidence="low"` (used by classifier to be conservative).

**Done-when.**
- Dispatch tests pass; failure mode (raise → fallback) verified.

---

### S9 — Classifier prompt (system + exemplars), cached

**Goal.** Lay down `altendor/classify/prompts.py` with the system prompt
text and the few-shot exemplar block built from `labeled.jsonl`.

**Dependencies.** S0, S2.

**Reference.** §5.3.

**Outputs.**
- `altendor/classify/prompts.py`:
  ```python
  CLASSIFIER_SYSTEM_PROMPT: str  # cached system block
  def build_user_message(post: ResolvedPost, paper: Paper) -> str: ...
  def build_exemplars_block() -> str: ...  # loads from labeled.jsonl
  ```
- `altendor/tests/unit/test_prompts.py` — formatting tests (no Claude calls):
  exemplars block contains all rubric anchors, magnitudes appear, no extra
  whitespace, paper context not duplicated.

**Implementation hints.**
- System prompt includes: role, rubric anchor table verbatim, exemplars
  block, JSON-output spec, refusal conditions (off-topic post → irrelevant).
- Mark the system block for caching via `claude_kit` cache control
  (5-min TTL; reused across all 300+ posts).

**Done-when.**
- Formatting tests pass.

---

### S10 — Single-post classifier

**Goal.** `classify_post(client, post, paper)` returning a pydantic
discriminated-union result.

**Dependencies.** S9.

**Local shape (this stage owns it).**
- `ClassifyResult = Endorsement | Flag | Irrelevant` discriminated union (`kind` field).
- `Endorsement(claim_text: str, magnitude_dB: int ∈ [-30, 30], criterion: Literal["Support","Prior"], reasoning: str)`.
- `Flag(category: Literal["methodological","source","data","bias","other"], rationale: str)`.
- `Irrelevant(reason: str)`.
- Lives in `altendor/classify/schema.py` — local to the classifier; consumed by S13, S14, S16 directly.

**Outputs.**
- `altendor/classify/schema.py` — the local discriminated union above.
- `altendor/classify/classifier.py`:
  ```python
  def classify_post(client: ClaudeClient, post: ResolvedPost, paper: Paper) -> ClassifyResult: ...
  ```
- `altendor/tests/unit/test_classifier_mocked.py` — three fixtures of
  recorded Claude responses (endorsement / flag / irrelevant); verify
  result type discrimination and field extraction.

**Implementation hints.**
- Use Claude tool-use with a tool whose input schema is the pydantic schema
  of `ClassifyResult` for guaranteed structure.
- Default model: `claude-sonnet-4-6`.
- Drop zero-magnitude endorsements (per DeltaBay convention).

**Done-when.**
- Mocked tests pass.
- Manual smoke: classify a known endorsement post returns kind=endorsement
  with positive magnitude.

---

### S11 — Batch classifier

**Goal.** `altendor/classify/batch.py` — submit Batch requests, poll, parse.

**Dependencies.** S10.

**Outputs.**
- `altendor/classify/batch.py`:
  ```python
  def submit_batch(client, posts: list[ResolvedPost], papers: dict[str, Paper]) -> str: ...
  def poll_until_done(client, batch_id: str, *, interval_s: int = 30) -> None: ...
  def parse_batch_results(client, batch_id: str) -> dict[str, ClassifyResult]: ...
  ```
- `altendor/tests/unit/test_batch_parse.py` — fixture-replay against a
  recorded Batches result file.

**Implementation hints.**
- Anthropic Batches API: 50% discount; up to 24h SLA, usually <1h.
- Each request: `custom_id = post_id`; reuses the cached system prompt.
- Poll cadence: 30s; fallback to a notebook-friendly polling loop with a
  manual cancel button.

**Done-when.**
- Parse tests pass.
- Manual smoke: 5-request batch returns parsed results.

---

### S12 — Calibration gate

**Goal.** Notebook 4 + pytest live test that gate the run: classifier
MAE ≤ 8 dB and kind-F1 ≥ 0.8 on `labeled.jsonl`.

**Dependencies.** S2, S9, S10.

**Reference.** §5.3, §5.5.

**Outputs.**
- `altendor/tests/live/test_calibration_gate.py` (`@pytest.mark.live`):
  loads labeled JSONL, runs `classify_post` on each, computes metrics,
  asserts thresholds.
- `notebooks/4_calibrate_classifier.ipynb` — interactive: load labels,
  classify, print confusion matrix + magnitude scatter, halt if thresholds
  fail.

**Implementation hints.**
- Optional leave-one-out: per-row, build exemplars from the other rows;
  more faithful but ~14× slower. v1: use the full exemplar set, accept
  slight optimism in metric.
- Metrics: MAE on `magnitude_dB` over rows where both predicted and gold
  are endorsements; F1 on `kind ∈ {endorsement, flag, irrelevant}`.

**Done-when.**
- Notebook runs end-to-end with seed data.
- Live test green (after user expands `labeled.jsonl` to ~20 rows).

---

### S13 — Reply traversal (depth 1, Bluesky + Reddit)

**Goal.** `altendor/traverse/replies.py` — for each endorsement post on
Bluesky/Reddit, fetch depth-1 replies/quotes and re-classify.

**Dependencies.** S6, S7, S10.

**Outputs.**
- `altendor/traverse/replies.py`:
  ```python
  async def traverse_depth1(seed_endorsements: list[ClassifyResult],
                            posts: dict[str, ResolvedPost],
                            *, bsky_session, reddit_client) -> list[tuple[ResolvedPost, ClassifyResult]]: ...
  ```
- `altendor/tests/unit/test_traverse_dedup.py` — golden test verifying
  dedup by `(platform, post_id)` and skip-self-author rule.

**Implementation hints.**
- Hard cap 200 reply nodes per run.
- Skip reply if `author == original.author`.
- Re-classify each reply via `classify_post` (non-batch — small N, low
  latency more useful here).

**Done-when.**
- Dedup tests pass.
- Manual smoke: a Bluesky endorsement with known replies yields ≥1 reply
  classification.

---

### S14 — Claim clustering

**Goal.** `altendor/cluster/claims.py` — Haiku one-shot clustering of
`claim_text` strings per paper into 3..7 canonical clusters.

**Dependencies.** S10.

**Local shape (this stage owns it).**
- `ClaimCluster(canonical_text: str, member_post_ids: list[str])`.

**Outputs.**
- `altendor/cluster/claims.py`:
  ```python
  @dataclass
  class ClaimCluster: canonical_text: str; member_post_ids: list[str]
  def cluster_claims(client, claim_texts: dict[str, str], k_hint: int = 5) -> list[ClaimCluster]: ...
  ```
- `altendor/tests/unit/test_clustering_golden.py` — mocked Claude response;
  verifies clamp to `[3, 7]` and mega-claim fallback on error.

**Implementation hints.**
- Model: `claude-haiku-4-5`.
- Prompt constrains output to JSON tool-use with
  `clusters: [{canonical_text, member_post_ids}]`.
- Post-hoc clamp: if Haiku returns <3, treat as a single mega-claim covering
  all members; if >7, merge by Jaccard on member sets.

**Done-when.**
- Golden tests pass.

---

### S15 — Paper → Question routing

**Goal.** `altendor/route/question_router.py` — assign each paper to one
of the three existing Questions, with post-hoc diversification.

**Dependencies.** S5 (only needs the local `papers` columns).

**Local shape (this stage owns it).**
- Returns `dict[paper_doi, (question_id: str, confidence: float)]`; after
  diversification, a flat `dict[paper_doi, question_id]`.

**Outputs.**
- `altendor/route/question_router.py`:
  ```python
  def route_paper_to_question(client, paper: Paper, questions: list[QuestionStub]) -> tuple[str, float]: ...
  def diversify_routes(routes: dict[str, tuple[str, float]]) -> dict[str, str]: ...
  ```
- `altendor/tests/unit/test_routing_golden.py` — mocked Claude responses;
  verifies the diversification re-routes lowest-confidence paper to an
  empty Question.

**Implementation hints.**
- Three Questions hard-coded with their IDs and titles (Q1 integrity,
  Q2 progress, Q3 peer-review).
- Confidence: model returns a softmax-like score per Question; pick argmax.
- Diversification: if any Question has 0 papers, re-route the
  lowest-confidence assigned paper.

**Done-when.**
- Golden tests pass.

---

### S16 — Define `IntermediateDebate` schema (integration boundary) + build it

**Goal.** Crystallize the shared schema **at this point**, having seen all
upstream producer shapes (S5 `papers/posts`, S8 `ResolvedPost`, S10
`ClassifyResult`, S14 `ClaimCluster`, S15 routes) and the DeltaBay consumer
contract (§5.1). Then implement the pure-function assembly.

**Dependencies.** S5, S8, S10, S14, S15. (Note: schema is *defined here*
because dependencies are now visible — do not anticipate it earlier.)

**Reference.** §5.1 (consumer contract — the constraint), §5.4 (paths).

**Outputs.**
- `altendor/assemble/intermediate.py` — pydantic models for the final shape,
  derived from what upstream producers actually emit + what §5.1 demands.
  At minimum: `IntermediateDebate`, `QuestionStub`, `Participant`, `Paper`,
  `AnswerNode`, `EvidenceNode`, `SourceDocument`, `SubclaimNode`,
  `EndorsementRow`, `FlagRow`. Adjust fields based on observed upstream
  data during implementation — do not lock fields without checking real
  outputs.
- `altendor/assemble/builder.py`:
  ```python
  def build_intermediate(papers: pd.DataFrame,
                         resolved_posts: dict[str, ResolvedPost],
                         classified: dict[str, ClassifyResult],
                         clusters: dict[str, list[ClaimCluster]],
                         routes: dict[str, str]) -> IntermediateDebate: ...
  ```
- `altendor/tests/unit/test_intermediate_schema.py` — contract tests on the
  freshly defined schema: pydantic round-trip on a fixture; bounds
  (`magnitude ∈ [-30, 30]`, `criterion`/`category` literals); `@id`s start
  with `altendor:`.
- `altendor/tests/unit/test_builder_golden.py` — fixture input (real
  upstream samples) → exact JSON output.

**Implementation hints.**
- **Build the schema by reading real samples first.** Run S5/S8/S10/S14/S15
  on a tiny live slice (1–2 papers, ~10 posts) and inspect their outputs;
  let the field set of `IntermediateDebate` follow what's actually there.
  Do not add speculative fields.
- ID conventions: `altendor:` namespace prefix on every generated `@id` (§5.1).
- Participants deduped at debate level by `(platform, author_id)`.
- Flag rows attached to paper-level Evidence; endorsements attached to
  Subclaims (clustered) or directly to Evidence if no cluster match (rare).
- Drop endorsements with magnitude 0 (DeltaBay convention).
- Use `pydantic.BaseModel` with `model_config = ConfigDict(extra="forbid")`
  once the field set has stabilised.

**Done-when.**
- Contract + golden tests pass.
- Schema fits the real outputs from S5/S8/S10/S14/S15 (no contortion).
- `funding-debate.altendor.test.ts` (S28) green on the produced JSON.

---

### S17 — Write neutral debate.json

**Goal.** `altendor/assemble/deltabay_writer.py` — write
`IntermediateDebate` to `output/<run_id>/debate.json`.

**Dependencies.** S16.

**Outputs.**
- `altendor/assemble/deltabay_writer.py`:
  ```python
  def write_debate_json(idb: IntermediateDebate, out_path: Path) -> None: ...
  ```
- `altendor/tests/unit/test_writer_roundtrip.py` — write + read-back round trip.

**Done-when.**
- Round-trip test passes.

---

### S18–S25 — Notebook orchestrators

Each notebook is linear, top-to-bottom executable. Heavy logic lives in
`altendor/*` modules. **BigQuery queries always have a dry-run cell first.**

| Stage | Notebook | Calls | Output |
|------|----------|------|--------|
| S18 | `1_select_papers.ipynb` | S3, S4 (preflight + run), S5.`join_dois_to_attention`, S5.`top_papers` | `output/<run_id>/papers.parquet`, `manifest.json` |
| S19 | `2_gather_posts.ipynb` | S4 (preflight + run), S5.`fetch_posts_for_papers` | `posts.parquet` |
| S20 | `3_enrich_text.ipynb` | S8.`resolve_full_text` async fan-out | `resolved_posts.parquet` |
| S21 | `4_calibrate_classifier.ipynb` | S12 logic | gate; prints metrics |
| S22 | `5_classify_batch.ipynb` | S11 submit/poll/parse; **fallback toggle** to S10 single-call | `classified.parquet` |
| S23 | `6_traverse_replies.ipynb` | S13 | append to `classified.parquet` |
| S24 | `7_cluster_claims.ipynb` | S14 per paper | `clusters.json` |
| S25 | `8_route_and_assemble.ipynb` | S15, S16, S17 | `debate.json` |

**Dependencies.** All upstream `altendor/` modules done.

**Tests.** None directly — notebooks are orchestrators. Smoke verifies
them indirectly (S29).

**Done-when.**
- Each notebook runs top-to-bottom on a fresh kernel without error.
- Dry-run cell always precedes its BigQuery run cell.

---

### S26 — DeltaBay-side loader: `altendor-loader.ts`

**Goal.** New `DeltaBay/frontend/app/src/data/altendor-loader.ts` that reads
`altendor/debate.json` and produces a `DeltabayDebate`.

**Dependencies.** S17.

**Outputs.**
- `DeltaBay/frontend/app/src/data/altendor-loader.ts`:
  ```ts
  export function loadAltendorDebate(): DeltabayDebate | null
  ```
- `DeltaBay/frontend/app/src/data/altendor/debate.json` — landing pad
  (committed empty stub; pipeline overwrites).
- Vite import alias if needed.

**Implementation hints.**
- Mirror `mira-graph-loader.ts` for shape and `enrichDebateViews()` call.
- Stamp `@type` strings (`mira:Endorsement`, `mira:Evidence`, etc.) from
  `MIRA_CONTEXT` in `jsonld/context.ts`.
- Dedupe participants by `id` at debate level.

**Tests.** Covered by S28.

**Done-when.**
- Type-checks under `pnpm -C frontend/app tsc --noEmit`.
- A fixture `debate.json` round-trips into a valid `DeltabayDebate`.

---

### S27 — Merge loader output into `funding-debate.ts`

**Goal.** Modify `DeltaBay/frontend/app/src/data/funding-debate.ts` to merge
loader output into the existing skeleton. Skeleton stays as a fallback.

**Dependencies.** S26.

**Outputs.**
- Edits to `funding-debate.ts`:
  - After building `RAW_FUNDING_DEBATE`, call `loadAltendorDebate()`.
  - If non-null: merge `participants` (union by id) and replace
    `questions[i].answers` for routed questions; keep unrouted Questions as
    their skeleton.

**Tests.** Covered by S28.

**Done-when.**
- With an empty stub `debate.json`, the existing debate renders unchanged.
- With a populated `debate.json`, Answers appear under the routed Questions.

---

### S28 — Vitest `funding-debate.altendor.test.ts`

**Goal.** Mirror `funding-debate.test.ts` for the populated-debate path.

**Dependencies.** S26, S27.

**Outputs.**
- `DeltaBay/frontend/app/src/data/funding-debate.altendor.test.ts`:
  loads a canned fixture; asserts every Question has ≥1 Answer; every Answer
  has ≥1 Subclaim with ≥1 Endorsement; every Evidence has non-null
  `sourceDocument.doi`; all `participantId`s in endorsements/flags resolve
  to debate-level participants; magnitudes in `[-30, 30]`; `@type` tags
  correctly stamped.

**Done-when.**
- `pnpm -C frontend/app vitest run` green.

---

### S29 — End-to-end smoke test

**Goal.** `altendor/tests/live/test_pipeline_smoke.py` — runs the full
pipeline with `n=2` papers and `budget_usd=0.50` into a temp run_id.

**Dependencies.** S0–S25.

**Outputs.**
- Smoke test asserts file presence, non-empty Answer arrays, and
  `IntermediateDebate.model_validate_json(debate.json)` succeeds.
- `@pytest.mark.live`.

**Done-when.**
- One green smoke run with budget under cap.

---

### S30 — Notebook `9_publish_to_deltabay.ipynb`

**Goal.** Copy/symlink the produced `debate.json` into DeltaBay's data
directory and run `pnpm -C frontend/app vitest run funding-debate.altendor`.

**Dependencies.** S28, S29.

**Outputs.**
- `notebooks/9_publish_to_deltabay.ipynb`.

**Done-when.**
- Notebook runs top-to-bottom; vitest passes; user can `pnpm dev` and
  click through the populated debate.

---

### S31 — Notebook 8 final markdown summary cell

**Goal.** End notebook `8_route_and_assemble.ipynb` with a markdown table:
papers × routed Question × #endorsements × #flags × #participants.

**Dependencies.** S25.

**Outputs.**
- One final cell in notebook 8.

**Done-when.**
- Visual table renders cleanly.

---

### S32 — Runbook in `altendor/`

**Goal.** Short `altendor/README.md` describing how to re-run, where outputs
live, and pointing to this plan file.

**Dependencies.** All.

**Outputs.**
- `altendor/README.md` — sections: prereqs (env vars), run order
  (notebooks 1→9), where to look (`output/<run_id>/`), how to refresh
  fixtures, how to extend calibration.

**Done-when.**
- README exists; user can hand it to a teammate.

---

## 7. Risks and mitigations

| Risk | Mitigation | Affects |
|------|-----------|---------|
| Altmetric `posts.title` truncated | Enrichment via Bluesky/Reddit APIs; mark untrusted if < 140 chars | S5, S8, S10 |
| Bluesky URL→AT-URI resolution failures (deleted/blocked posts) | try/except, drop post, log; never crash batch | S6, S13 |
| Reddit 429 rate limits | semaphore=4 + retry-with-backoff | S7, S13 |
| Anthropic Batches latency unpredictable | Fallback toggle in notebook 5 to single-call S10 path | S11, S22 |
| Routing skew (all 10 papers → peer-review) | Diversification in S15 | S15 |
| Cluster cardinality drift | Post-hoc clamp `[3,7]` in S14 | S14 |
| Cross-platform identity (same user across platforms = 2 participants) | Accepted v1; document in `manifest.json` | S16 |
| GBQ scan caps | Filter via `UNNEST(research_outputs_ids)` early; preflight rejects > 50 GiB | S4, S5 |
| JSON-LD `@id` collisions with MIRA-seeded IDs | `altendor:` namespace fragment on every generated ID | S16, S26 |
| Calibration labels too sparse to gate (<10 rows) | S2 seeds 10 rows; user expands during S12 before gate flips on | S2, S12 |
| Demo-day budget overrun | `CostTracker(budget_usd=5.0)` hard cap; smoke test enforces `budget_usd=0.50` | S29, all live calls |
