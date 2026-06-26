# AI Code Review Platform

An AI agent that reviews GitHub pull requests. Built in phases — see the roadmap below.

## Current status: Phase 1 ✅

A single script that fetches a real PR diff, sends it to an LLM, and prints structured review findings.

## Setup

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add your keys
cp .env.example .env
#   then edit .env and paste your GROQ_API_KEY (free: https://console.groq.com/keys)
```

## Run

```bash
python pr_reviewer.py https://github.com/owner/repo/pull/123
```

Try it on any public PR. A good first test is a small PR from a project you know.

## Roadmap

- [x] **Phase 1** — Single script: PR diff → LLM → findings
- [ ] **Phase 2** — Wrap in FastAPI + GitHub webhook; auto-post review as a PR comment
- [ ] **Phase 3** — Run Semgrep / Bandit / Ruff first, feed results into the prompt
- [ ] **Phase 4** — Split into parallel specialist agents (security, performance, architecture, testing, docs) + orchestrator
- [ ] **Phase 5** — Memory + RAG over the repo (Qdrant, embeddings, call graph)
- [ ] **Phase 6** — Observability (tracing, Grafana) + developer dashboard
