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

# Reuse everything we built in earlier phases — this is the payoff of run_review().
from pr_reviewer import fetch_diff, gather_static_findings, run_review

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
        diff = fetch_diff(owner, repo, number)
        if not diff.strip():
            return
        static_findings = gather_static_findings(owner, repo, number)
        result = run_review(diff, static_findings)
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
            src = " · 🔧 _ruff/bandit_" if f.get("source") == "static" else ""
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
