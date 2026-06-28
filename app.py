"""
Phase 2: GitHub webhook server.

When a PR is opened or updated, GitHub sends an HTTP POST to /webhook. We:
  1. verify the request really came from GitHub (HMAC signature)
  2. check it's a PR event we care about
  3. respond to GitHub IMMEDIATELY, and do the slow review in the background
  4. fetch the diff, run the Phase 1 pipeline, and post the review as a PR comment

Run locally:  uvicorn app:app --reload --port 8000
"""

import hashlib
import hmac
import os
import traceback

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse

# Reuse everything we built in earlier phases — this is the payoff of review_pr().
from observability import compute_stats, load_traces
from pr_reviewer import review_pr

load_dotenv()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

app = FastAPI(title="AI Code Review Bot")


# ---------------------------------------------------------------------------
# Health check — lets you confirm the server is up with a browser or curl.
# ---------------------------------------------------------------------------
@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "ai-code-review-bot"}


# ---------------------------------------------------------------------------
# Phase 6: observability endpoints.
# ---------------------------------------------------------------------------
@app.get("/stats")
def stats() -> dict:
    """Aggregate metrics across all reviews (JSON)."""
    return compute_stats(load_traces())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """A tiny self-contained HTML dashboard of recent reviews."""
    traces = load_traces()
    s = compute_stats(traces)
    recent = list(reversed(traces[-25:]))  # newest first

    if not traces:
        return "<h2>🤖 AI Code Review — Dashboard</h2><p>No reviews recorded yet.</p>"

    cards = "".join(
        f"<div class='card'><div class='num'>{v}</div><div class='lbl'>{k.replace('_', ' ')}</div></div>"
        for k, v in s.items()
    )
    rows = "".join(
        f"<tr><td>{t.get('pr', '?')}</td>"
        f"<td>{t.get('counts', {}).get('confirmed', 0)}</td>"
        f"<td>{t.get('counts', {}).get('refuted', 0)}</td>"
        f"<td>{', '.join(t.get('planned_agents', []))}</td>"
        f"<td>{t.get('timings_ms', {}).get('total', 0)} ms</td>"
        f"<td>{t.get('tokens', {}).get('total', 0)}</td></tr>"
        for t in recent
    )
    return f"""
    <html><head><title>AI Code Review Dashboard</title><style>
      body{{font-family:system-ui,sans-serif;margin:2rem;background:#0d1117;color:#e6edf3}}
      h2{{margin-bottom:1rem}}
      .cards{{display:flex;flex-wrap:wrap;gap:1rem;margin-bottom:2rem}}
      .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1rem 1.4rem;min-width:120px}}
      .num{{font-size:1.8rem;font-weight:700;color:#58a6ff}}
      .lbl{{font-size:.8rem;color:#8b949e;text-transform:capitalize}}
      table{{width:100%;border-collapse:collapse}}
      th,td{{text-align:left;padding:.5rem .6rem;border-bottom:1px solid #30363d;font-size:.9rem}}
      th{{color:#8b949e}}
    </style></head><body>
      <h2>🤖 AI Code Review — Dashboard</h2>
      <div class='cards'>{cards}</div>
      <h3>Recent reviews</h3>
      <table><tr><th>PR</th><th>Confirmed</th><th>Refuted</th><th>Agents</th><th>Latency</th><th>Tokens</th></tr>
      {rows}</table>
    </body></html>"""


# ---------------------------------------------------------------------------
# Security: verify the webhook actually came from GitHub.
# ---------------------------------------------------------------------------
def verify_signature(body: bytes, signature: str) -> bool:
    """GitHub signs every webhook payload with HMAC-SHA256 using the secret you
    configured. We recompute that signature over the RAW body and compare. If it
    doesn't match, someone other than GitHub is hitting your endpoint — reject.

    hmac.compare_digest is used (not ==) to avoid timing attacks.
    """
    if not WEBHOOK_SECRET:
        # Dev mode: no secret set, so we can't verify. Fine for local curl tests,
        # but ALWAYS set a secret once the endpoint is reachable from the internet.
        return True
    if not signature:
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# The webhook endpoint.
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
) -> dict:
    # Read the RAW body first — signature is computed over these exact bytes,
    # so we must verify before parsing JSON.
    body = await request.body()
    if not verify_signature(body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # GitHub sends many event types; we only review pull requests.
    if x_github_event != "pull_request":
        return {"skipped": f"event '{x_github_event}' is not pull_request"}

    payload = await request.json()
    action = payload.get("action")
    # "opened" = new PR, "synchronize" = new commits pushed, "reopened" = reopened.
    if action not in ("opened", "synchronize", "reopened"):
        return {"skipped": f"action '{action}'"}

    pr = payload["pull_request"]
    repo = payload["repository"]
    owner = repo["owner"]["login"]
    repo_name = repo["name"]
    number = pr["number"]

    # KEY MOVE: GitHub times out webhooks in ~10s, but a review takes longer.
    # So we hand the slow work to a background task and return 200 right away.
    background_tasks.add_task(review_and_comment, owner, repo_name, number)
    return {"status": "accepted", "pr": f"{owner}/{repo_name}#{number}"}


# ---------------------------------------------------------------------------
# The background job: review the PR and post a comment.
# ---------------------------------------------------------------------------
def review_and_comment(owner: str, repo: str, number: int) -> None:
    try:
        result = review_pr(owner, repo, number)
        comment = format_comment(result)
        post_comment(owner, repo, number, comment)
        print(f"[bot] Posted review on {owner}/{repo}#{number}")
    except Exception:
        # Never let a background failure crash silently — log it.
        print(f"[bot] FAILED reviewing {owner}/{repo}#{number}")
        traceback.print_exc()


SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}


def format_comment(result: dict) -> str:
    """Turn the review dict into GitHub-flavored markdown for the PR comment."""
    lines = ["## 🤖 AI Code Review", "", result.get("summary") or "_No summary._", ""]

    confirmed = result.get("confirmed", [])
    if not confirmed:
        lines.append("✅ **No issues found that survived verification.**")
    else:
        for f in confirmed:
            sev = (f.get("severity") or "low").lower()
            emoji = SEVERITY_EMOJI.get(sev, "⚪")
            if f.get("source") == "static":
                src = " · 🔧 _ruff/bandit_"
            elif f.get("agents"):
                src = f" · 🤖 _{', '.join(f['agents'])}_"
            else:
                src = ""
            lines.append(f"### {emoji} {f.get('title', 'Issue')} — `{sev}`")
            lines.append(
                f"**`{f.get('file', '?')}`** line {f.get('line', '?')} · _{f.get('category', '')}_{src}"
            )
            lines.append("")
            lines.append(f.get("description", ""))
            if f.get("evidence"):
                lines.append(f"\n```\n{f['evidence'].strip()}\n```")
            if f.get("suggestion"):
                lines.append(f"\n**Suggested fix:** {f.get('suggestion')}")
            lines.append("")

    # Transparency footer: show what we filtered out so the dev trusts the bot.
    sup = result.get("suppressed", 0)
    ref = len(result.get("refuted", []))
    notes = []
    planned = result.get("planned_agents")
    if planned:
        notes.append(f"agents: {', '.join(planned)}")
    if sup:
        notes.append(f"{sup} low-confidence finding(s) suppressed")
    if ref:
        notes.append(f"{ref} finding(s) refuted by the verifier")
    if notes:
        lines.append("\n---")
        lines.append(f"<sub>🔍 {' · '.join(notes)}</sub>")

    return "\n".join(lines)


def post_comment(owner: str, repo: str, number: int, body: str) -> None:
    """Post a comment on the PR. PR conversation comments are 'issue comments'
    in the GitHub API, hence the /issues/ path."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required to post comments.")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = httpx.post(url, headers=headers, json={"body": body}, timeout=30)
    resp.raise_for_status()
