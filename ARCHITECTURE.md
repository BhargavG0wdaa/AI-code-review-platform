# Architecture

An AI agent that reviews GitHub pull requests. It combines **deterministic tools**, **retrieval over the repository**, a **team of specialist LLM agents**, and an **adversarial verifier** to produce high-signal reviews and post them as PR comments — automatically.

The guiding principle behind every design decision: **an LLM is powerful but unreliable — it both hallucinates and over-hesitates — so each layer checks the model against something more trustworthy** (a deterministic tool, real retrieved code, specialist consensus, or a second adversarial model).

---

## The pipeline

```
A PR is opened/updated
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. STATIC ANALYSIS        ruff + bandit on changed files       │  deterministic facts
│ 2. RAG CONTEXT            index repo → retrieve related code    │  cross-file understanding
│ 3. PLANNER                pick which specialists are relevant   │  save wasted calls
│ 4. SPECIALIST AGENTS      security ‖ perf ‖ arch ‖ test ‖ docs  │  focused expertise (parallel)
│ 5. DEDUP / CONSENSUS      merge overlapping findings            │  multiple agents agreeing = signal
│ 6. CONFIDENCE FILTER      drop low-confidence findings          │  precision
│ 7. ADVERSARIAL VERIFIER   try to REFUTE each finding            │  kills hallucinations
│        static findings bypass the verifier (tools don't lie)    │
│        + a "non-excusable" floor for serious security issues    │
└──────────────────────────────────────────────────────────────┘
        │
        ├──► post a Markdown comment on the PR
        └──► write a trace → /dashboard (timings, tokens, refute rate)
```

---

## Why each layer exists

Every layer was added to fix a concrete failure observed in the previous version.

| Layer | The problem it solves |
|-------|------------------------|
| **Adversarial verifier** | A single LLM confidently reports bugs that aren't real. A second model told to *refute* each finding kills these hallucinations. |
| **Static analysis** | The verifier was too cautious and dropped a *real* `eval()` vulnerability. Deterministic tools (bandit `B307`) flag it every time and can't be argued down. |
| **Specialist agents** | One generalist prompt juggling five concerns produces shallow findings. Five focused agents each go deeper in their lane. |
| **Dedup / consensus** | Fanning out to many agents creates duplicate findings. Merging them — and recording which agents agreed — turns overlap into a confidence signal. |
| **Planner** | Running all 5 agents on every PR wastes calls. The planner skips irrelevant ones (e.g. no `performance` agent on a diff with no loops). |
| **RAG context** | The reviewer only sees the diff, so the verifier kept refusing bugs for "lack of caller context." RAG retrieves related repo code so it can reason about cross-file usage. |
| **Security floor** | More context made the verifier *rationalize away* hardcoded secrets ("it's probably a test file"). A non-excusable carve-out confirms serious security issues on sight. |
| **Observability** | You can't improve what you can't measure. Every review records timings, token usage, and refute rate. |

---

## Components

| File | Responsibility |
|------|----------------|
| `pr_reviewer.py` | Core engine: fetch diff, run the full pipeline (`review_pr`), specialists, planner, dedup, verifier, CLI. |
| `static_analysis.py` | Run ruff + bandit on changed files; normalize their output to the standard finding shape. |
| `rag.py` | Index repo code into Qdrant (chunked by function/class), embed with fastembed, retrieve code relevant to a diff. |
| `app.py` | FastAPI webhook server: verify signature, filter events, review in the background, post the comment; `/stats` + `/dashboard`. |
| `observability.py` | Append-only JSONL trace store + aggregate stats. |

### Key design decisions

- **One strict JSON schema everywhere.** Every agent returns findings in the same shape, so the pipeline can filter, merge, verify, and render them uniformly.
- **`review_pr()` is the single orchestrator.** Both the CLI and the webhook call it, so the whole pipeline — and all instrumentation — lives in exactly one place.
- **Library code raises, entry points exit.** Functions raise `ReviewError`; only `main()` calls `sys.exit()`. This is why the webhook can catch and log failures instead of dying silently.
- **Deterministic before probabilistic.** Cheap, certain tools run first and their results are treated as facts the LLM can't override.
- **Best-effort RAG.** If Qdrant is down, reviews still run — just without retrieved context.
- **Fail loud, fail cheap.** Confidence filtering happens *before* the expensive verifier; rate limits trigger retry-with-backoff.

---

## Tech stack

- **LLM:** Groq — `llama-3.3-70b-versatile` (fast, free tier, JSON output)
- **Static analysis:** ruff (bugs) + bandit (security)
- **Vector DB:** Qdrant (local, via Docker)
- **Embeddings:** fastembed (`bge-small`, ONNX — no PyTorch)
- **Web:** FastAPI + uvicorn
- **Tunnel (dev):** cloudflared

---

## Running it

See [README.md](README.md) for setup. In short:

```bash
# CLI — review one PR
.venv/bin/python pr_reviewer.py https://github.com/owner/repo/pull/123

# Webhook bot — auto-review PRs (needs Qdrant + a tunnel)
docker start qdrant
.venv/bin/python -m uvicorn app:app --reload --port 8000
cloudflared tunnel --url http://localhost:8000   # in another terminal
# then visit http://localhost:8000/dashboard
```

---

## Built in phases

| Phase | What it added |
|-------|----------------|
| 1 | Single-file reviewer + adversarial verifier |
| 2 | GitHub webhook server, auto-posts PR comments |
| 3 | Deterministic static analysis (ruff + bandit) |
| 4 | Parallel specialist agents + dedup/consensus |
| 4b | Planner agent (picks which specialists to run) |
| 5 | Repository RAG (Qdrant + embeddings) + hardened verifier |
| 6 | Observability — traces, stats, dashboard |
