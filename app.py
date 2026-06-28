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
import json
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
    """Professional light-theme dashboard with charts (Chart.js via CDN)."""
    traces = load_traces()
    s = compute_stats(traces)

    def c(t, key):
        return t.get("counts", {}).get(key, 0)

    # Chronological series for the charts.
    labels = [t.get("pr", "?").split("/")[-1] for t in traces]
    data = {
        "kpis": [
            {"label": "Reviews", "value": s.get("reviews", 0)},
            {"label": "Findings confirmed", "value": s.get("total_confirmed", 0)},
            {"label": "Findings refuted", "value": s.get("total_refuted", 0)},
            {"label": "Refute rate", "value": f"{round(s.get('refute_rate', 0) * 100)}%"},
            {"label": "Avg tokens / review", "value": f"{s.get('avg_tokens_per_review', 0):,}"},
            {"label": "Avg latency", "value": f"{s.get('avg_latency_ms', 0) / 1000:.1f}s"},
        ],
        "labels": labels,
        "confirmed": [c(t, "confirmed") for t in traces],
        "refuted": [c(t, "refuted") for t in traces],
        "totals": {
            "confirmed": s.get("total_confirmed", 0),
            "refuted": s.get("total_refuted", 0),
            "suppressed": sum(c(t, "suppressed") for t in traces),
        },
        "latency": [round(t.get("timings_ms", {}).get("total", 0) / 1000, 1) for t in traces],
        "tokens": [t.get("tokens", {}).get("total", 0) for t in traces],
        "rows": [
            {
                "pr": t.get("pr", "?"),
                "confirmed": c(t, "confirmed"),
                "refuted": c(t, "refuted"),
                "agents": ", ".join(t.get("planned_agents", [])),
                "latency": t.get("timings_ms", {}).get("total", 0),
                "tokens": t.get("tokens", {}).get("total", 0),
            }
            for t in reversed(traces[-25:])
        ],
    }
    return _DASHBOARD_TEMPLATE.replace("__DATA__", json.dumps(data))


_DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Code Review — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root{
    --bg:#f5f7fa; --card:#ffffff; --border:#e6e8ec; --text:#0f172a; --muted:#64748b;
    --indigo:#6366f1; --green:#10b981; --amber:#f59e0b; --slate:#94a3b8;
    --shadow:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  }
  *{box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       margin:0;background:var(--bg);color:var(--text);padding:2.5rem 2rem 4rem}
  .wrap{max-width:1100px;margin:0 auto}
  header{display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem}
  header h1{font-size:1.5rem;font-weight:700;margin:0}
  .sub{color:var(--muted);font-size:.9rem;margin-bottom:2rem}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-bottom:2rem}
  .kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;box-shadow:var(--shadow)}
  .kpi .v{font-size:1.7rem;font-weight:750;letter-spacing:-.02em}
  .kpi .l{font-size:.8rem;color:var(--muted);margin-top:.2rem}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1.25rem;margin-bottom:2rem}
  .panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem 1.4rem;box-shadow:var(--shadow)}
  .panel h3{margin:0 0 1rem;font-size:.95rem;font-weight:650}
  .chart-box{position:relative;height:240px}
  table{width:100%;border-collapse:collapse;font-size:.88rem}
  th,td{text-align:left;padding:.6rem .7rem;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
  td.num{font-variant-numeric:tabular-nums}
  .empty{color:var(--muted);padding:3rem;text-align:center}
</style></head>
<body><div class="wrap">
  <header><span style="font-size:1.6rem">🤖</span><h1>AI Code Review</h1></header>
  <div class="sub">Observability dashboard · static analysis · specialist agents · adversarial verifier</div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="panel"><h3>Findings per review</h3><div class="chart-box"><canvas id="findings"></canvas></div></div>
    <div class="panel"><h3>Verifier outcomes (overall)</h3><div class="chart-box"><canvas id="outcomes"></canvas></div></div>
    <div class="panel"><h3>Latency per review (seconds)</h3><div class="chart-box"><canvas id="latency"></canvas></div></div>
    <div class="panel"><h3>Tokens per review</h3><div class="chart-box"><canvas id="tokens"></canvas></div></div>
  </div>
  <div class="panel"><h3>Recent reviews</h3><div id="table"></div></div>
</div>
<script>
const D = __DATA__;
const C = {indigo:'#6366f1', green:'#10b981', amber:'#f59e0b', slate:'#94a3b8', muted:'#64748b'};
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif";
Chart.defaults.color = C.muted;
Chart.defaults.plugins.legend.labels.boxWidth = 12;

// KPI cards
document.getElementById('kpis').innerHTML = D.kpis.map(k =>
  `<div class="kpi"><div class="v">${k.value}</div><div class="l">${k.label}</div></div>`).join('');

if (!D.labels.length) {
  document.querySelector('.grid').innerHTML = '<div class="empty">No reviews recorded yet — run a review to populate the dashboard.</div>';
} else {
  const grid = {scales:{x:{grid:{display:false}},y:{beginAtZero:true,grid:{color:'#eef1f5'}}},
                plugins:{legend:{position:'bottom'}},maintainAspectRatio:false};

  new Chart(findings, {type:'bar', data:{labels:D.labels, datasets:[
    {label:'Confirmed', data:D.confirmed, backgroundColor:C.indigo, borderRadius:5},
    {label:'Refuted', data:D.refuted, backgroundColor:C.slate, borderRadius:5}]},
    options:grid});

  new Chart(outcomes, {type:'doughnut', data:{labels:['Confirmed','Refuted','Suppressed'],
    datasets:[{data:[D.totals.confirmed,D.totals.refuted,D.totals.suppressed],
    backgroundColor:[C.indigo,C.slate,C.amber],borderWidth:0}]},
    options:{maintainAspectRatio:false,cutout:'62%',plugins:{legend:{position:'bottom'}}}});

  new Chart(latency, {type:'line', data:{labels:D.labels, datasets:[
    {label:'Latency (s)', data:D.latency, borderColor:C.green, backgroundColor:'rgba(16,185,129,.12)',
     fill:true, tension:.35, pointRadius:3}]},
    options:{...grid, plugins:{legend:{display:false}}}});

  new Chart(tokens, {type:'bar', data:{labels:D.labels, datasets:[
    {label:'Tokens', data:D.tokens, backgroundColor:C.indigo, borderRadius:5}]},
    options:{...grid, plugins:{legend:{display:false}}}});
}

// Recent reviews table
document.getElementById('table').innerHTML = D.rows.length ?
  `<table><thead><tr><th>PR</th><th>Confirmed</th><th>Refuted</th><th>Agents</th><th>Latency</th><th>Tokens</th></tr></thead>
   <tbody>${D.rows.map(r => `<tr><td>${r.pr}</td><td class="num">${r.confirmed}</td>
   <td class="num">${r.refuted}</td><td>${r.agents}</td><td class="num">${r.latency} ms</td>
   <td class="num">${r.tokens.toLocaleString()}</td></tr>`).join('')}</tbody></table>`
  : '<div class="empty">No reviews yet.</div>';
</script>
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
