# AI Code Review Platform

An AI agent that reviews GitHub pull requests — combining deterministic tools, repository retrieval (RAG), a team of specialist LLM agents, and an adversarial verifier — and posts the review as a PR comment automatically.

📐 See [ARCHITECTURE.md](ARCHITECTURE.md) for how it all fits together and why.

## Status: all 6 phases complete ✅

## Setup

```bash
# 1. create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add your keys
cp .env.example .env
#   GROQ_API_KEY            (free: https://console.groq.com/keys)
#   GITHUB_TOKEN            (repo scope — for the webhook bot to post comments)
#   GITHUB_WEBHOOK_SECRET   (any random string: openssl rand -hex 20)

# 4. (for RAG) start Qdrant via Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Run

**CLI — review one PR:**
```bash
.venv/bin/python pr_reviewer.py https://github.com/owner/repo/pull/123
```

**Webhook bot — auto-review PRs** (needs Qdrant running + a public tunnel):
```bash
docker start qdrant
.venv/bin/python -m uvicorn app:app --reload --port 8000
cloudflared tunnel --url http://localhost:8000     # in another terminal
# point your repo's webhook (Settings → Webhooks) at <tunnel-url>/webhook
# then visit http://localhost:8000/dashboard
```

## Roadmap

- [x] **Phase 1** — Single script: PR diff → LLM → findings, with an adversarial verifier
- [x] **Phase 2** — FastAPI + GitHub webhook; auto-post review as a PR comment
- [x] **Phase 3** — Run ruff + bandit first, feed results into the prompt as facts
- [x] **Phase 4** — Parallel specialist agents (security, performance, architecture, testing, docs) + planner + dedup
- [x] **Phase 5** — Repository RAG (Qdrant + embeddings) + hardened verifier
- [x] **Phase 6** — Observability (traces, stats, dashboard)
