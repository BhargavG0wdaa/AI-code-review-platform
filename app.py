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
from observability import (
    agent_performance,
    compute_stats,
    get_trace,
    load_traces,
    severity_distribution,
)
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
    """Data-driven dashboard: filters, trends, daily analytics (light/dark)."""
    traces = load_traces()

    def c(t, key):
        return t.get("counts", {}).get(key, 0)

    def sev_counts(t):
        out = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in t.get("findings", []):
            if f.get("status") == "confirmed":
                out[f.get("severity", "low")] = out.get(f.get("severity", "low"), 0) + 1
        return out

    reviews = []
    for i, t in enumerate(traces):
        pr = t.get("pr", "?")
        reviews.append({
            "idx": i,
            "pr": pr,
            "repo": pr.split("#")[0],
            "ts": t.get("timestamp", 0),
            "confirmed": c(t, "confirmed"),
            "refuted": c(t, "refuted"),
            "suppressed": c(t, "suppressed"),
            "latency": round(t.get("timings_ms", {}).get("total", 0) / 1000, 1),
            "tokens": t.get("tokens", {}).get("total", 0),
            "cost": t.get("cost_usd", 0),
            "sev": sev_counts(t),
            "findings": [{"agents": f.get("agents", ["?"]), "status": f.get("status")}
                         for f in t.get("findings", [])],
        })
    return (_DASHBOARD_TEMPLATE
            .replace("__CSS__", _BASE_CSS)
            .replace("__THEMEHEAD__", _THEME_HEAD)
            .replace("__THEMEJS__", _THEME_JS)
            .replace("__DATA__", json.dumps({"reviews": reviews})))


@app.get("/review/{idx}", response_class=HTMLResponse)
def review_detail(idx: int) -> str:
    """Per-review detail: timeline + every finding with status, reason, fix."""
    t = get_trace(idx)
    if not t:
        return "<p style='font-family:sans-serif;padding:2rem'>Review not found.</p>"
    return (_DETAIL_TEMPLATE
            .replace("__CSS__", _BASE_CSS)
            .replace("__THEMEHEAD__", _THEME_HEAD)
            .replace("__THEMEJS__", _THEME_JS)
            .replace("__DATA__", json.dumps(t)))


_BASE_CSS = """
  :root{--bg:#f5f7fa;--card:#fff;--border:#e6e8ec;--text:#0f172a;--muted:#64748b;
        --grid:#eef1f5;--indigo:#6366f1;--green:#10b981;--amber:#f59e0b;--red:#ef4444;--slate:#94a3b8;
        --shadow:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04)}
  [data-theme="dark"]{--bg:#0b1220;--card:#161e2e;--border:#27324a;--text:#e5e9f0;--muted:#94a3b8;
        --grid:#1f2a3d;--shadow:0 1px 3px rgba(0,0,0,.4)}
  *{box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;
       background:var(--bg);color:var(--text);padding:2.5rem 2rem 4rem;transition:background .2s,color .2s}
  .wrap{max-width:1140px;margin:0 auto}
  a{color:var(--indigo);text-decoration:none}
  header{display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem}
  header h1{font-size:1.4rem;font-weight:700;margin:0}
  .spacer{flex:1}
  .toggle{background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);
          padding:.45rem .7rem;cursor:pointer;font-size:.85rem}
  .sub{color:var(--muted);font-size:.9rem;margin-bottom:1.5rem}
  .filters{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem;align-items:center}
  .filters label{font-size:.78rem;color:var(--muted);margin-right:.2rem}
  select{background:var(--card);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:.45rem .6rem;font-size:.85rem}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-bottom:2rem}
  .kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;box-shadow:var(--shadow)}
  .kpi .v{font-size:1.6rem;font-weight:750;letter-spacing:-.02em}
  .kpi .l{font-size:.8rem;color:var(--muted);margin-top:.2rem}
  .kpi .t{font-size:.78rem;margin-top:.2rem;font-weight:650}
  .up{color:var(--green)} .down{color:var(--red)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1.25rem;margin-bottom:1.25rem}
  .panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem 1.4rem;box-shadow:var(--shadow);margin-bottom:1.25rem}
  .panel h3{margin:0 0 1rem;font-size:.95rem;font-weight:650}
  .chart-box{position:relative;height:240px}
  table{width:100%;border-collapse:collapse;font-size:.88rem}
  th,td{text-align:left;padding:.6rem .7rem;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.03em}
  td.num{font-variant-numeric:tabular-nums}
  tr.clk{cursor:pointer} tr.clk:hover{background:var(--grid)}
  .empty{color:var(--muted);padding:3rem;text-align:center}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.72rem;font-weight:650;color:#fff}
  .pill{display:inline-block;padding:.12rem .5rem;border-radius:999px;font-size:.78rem;font-weight:600}
  .pill.ok{background:rgba(16,185,129,.15);color:#059669}
  .pill.no{background:rgba(148,163,184,.2);color:#64748b}
  .chip{display:inline-block;background:var(--grid);color:var(--muted);border-radius:8px;padding:.3rem .6rem;font-size:.8rem;margin-right:.5rem}
  .bar{height:7px;border-radius:99px;background:var(--grid);overflow:hidden;margin-top:.3rem}
  .bar>span{display:block;height:100%;background:var(--indigo)}
  .timeline{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
  .stage{background:var(--grid);border-radius:10px;padding:.5rem .8rem;font-size:.82rem}
  .stage b{display:block;font-size:.95rem;font-variant-numeric:tabular-nums}
  .arrow{color:var(--muted)}
  .finding{border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;margin-bottom:.9rem}
  .finding h4{margin:.1rem 0 .4rem;font-size:1rem}
  .finding .meta{font-size:.8rem;color:var(--muted);margin-bottom:.5rem}
  pre{background:var(--grid);border-radius:8px;padding:.6rem .8rem;overflow:auto;font-size:.82rem;margin:.5rem 0}
  .agent-h{font-size:.95rem;font-weight:700;margin:1.4rem 0 .6rem}
"""

_THEME_HEAD = """<script>(function(){var t=localStorage.getItem('theme')||'light';
document.documentElement.setAttribute('data-theme',t);})();</script>"""

_THEME_JS = """
function toggleTheme(){var c=document.documentElement.getAttribute('data-theme');
localStorage.setItem('theme',c==='dark'?'light':'dark');location.reload();}
var THEME=document.documentElement.getAttribute('data-theme');
var AGENT_NAMES={security:'Security Agent',performance:'Performance Agent',
architecture:'Code Quality Agent',testing:'Testing Agent',docs:'Documentation Agent',
tools:'Static Analysis','?':'Unknown'};
function agentName(a){return AGENT_NAMES[a]||(a.charAt(0).toUpperCase()+a.slice(1)+' Agent');}
"""

_DASHBOARD_TEMPLATE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Multi-Agent Code Review — Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
__THEMEHEAD__<style>__CSS__</style></head><body><div class="wrap">
  <header><span style="font-size:1.5rem">🤖</span><h1>AI Multi-Agent Code Review — Analytics</h1>
    <span class="spacer"></span>
    <button class="toggle" onclick="toggleTheme()">☀️ / 🌙</button></header>
  <div class="sub">Static analysis · specialist agents · adversarial verifier · observability</div>
  <div class="filters">
    <span><label>Repository</label><select id="repo"></select></span>
    <span><label>Date range</label><select id="range">
      <option value="all">All time</option><option value="30">Last 30 days</option>
      <option value="7">Last 7 days</option><option value="1">Last 24 hours</option></select></span>
  </div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="panel"><h3>Findings per review</h3><div class="chart-box"><canvas id="findings"></canvas></div></div>
    <div class="panel"><h3>Verifier outcomes</h3><div class="chart-box"><canvas id="outcomes"></canvas></div></div>
    <div class="panel"><h3>Severity distribution (confirmed)</h3><div class="chart-box"><canvas id="severity"></canvas></div></div>
    <div class="panel"><h3>Reviews per day</h3><div class="chart-box"><canvas id="daily"></canvas></div></div>
  </div>
  <div class="panel"><h3>Agent performance</h3><div id="agents"></div></div>
  <div class="panel"><h3>Recent reviews <span style="font-weight:400;color:var(--muted);font-size:.8rem">— click a row for details</span></h3><div id="table"></div></div>
</div>
<script>
const REVIEWS = (__DATA__).reviews;
__THEMEJS__
const C={indigo:'#6366f1',green:'#10b981',amber:'#f59e0b',red:'#ef4444',slate:'#94a3b8'};
const GRID=THEME==='dark'?'#1f2a3d':'#eef1f5';
Chart.defaults.font.family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif";
Chart.defaults.color=THEME==='dark'?'#94a3b8':'#64748b';
Chart.defaults.plugins.legend.labels.boxWidth=12;
const g={scales:{x:{grid:{display:false}},y:{beginAtZero:true,grid:{color:GRID}}},plugins:{legend:{position:'bottom'}},maintainAspectRatio:false};

// repo filter options
const repos=[...new Set(REVIEWS.map(r=>r.repo))];
const repoSel=document.getElementById('repo'), rangeSel=document.getElementById('range');
repoSel.innerHTML='<option value="all">All repositories</option>'+repos.map(r=>`<option>${r}</option>`).join('');
repoSel.onchange=render; rangeSel.onchange=render;

const sum=(a,k)=>a.reduce((s,r)=>s+(r[k]||0),0);
const mean=(a,k)=>a.length?sum(a,k)/a.length:0;
function trend(cur,prev,goodUp){ if(!prev) return ''; const p=Math.round((cur-prev)/prev*100);
  if(!p) return ''; const up=p>0, good=goodUp?up:!up;
  return `<div class="t ${good?'up':'down'}">${up?'↑':'↓'}${Math.abs(p)}%</div>`; }

let charts={};
function render(){
  const repo=repoSel.value, range=rangeSel.value, now=Date.now()/1000;
  let R=REVIEWS.filter(r=>(repo==='all'||r.repo===repo) &&
        (range==='all'|| now-r.ts <= (+range)*86400));
  const half=Math.floor(R.length/2), older=R.slice(0,half), newer=R.slice(half);
  const conf=sum(R,'confirmed'), ref=sum(R,'refuted'), judged=conf+ref, cost=R.reduce((s,r)=>s+r.cost,0);
  const kpis=[
    {v:R.length,l:'Reviews',t:trend(newer.length,older.length,true)},
    {v:conf,l:'Findings confirmed',t:trend(sum(newer,'confirmed'),sum(older,'confirmed'),true)},
    {v:ref,l:'Findings refuted',t:''},
    {v:(judged?Math.round(ref/judged*100):0)+'%',l:'Refute rate',t:''},
    {v:'$'+cost.toFixed(4),l:'Est. cost',t:trend(sum(newer,'cost'),sum(older,'cost'),false)},
    {v:mean(R,'latency').toFixed(1)+'s',l:'Avg latency',t:trend(mean(newer,'latency'),mean(older,'latency'),false)},
  ];
  document.getElementById('kpis').innerHTML=kpis.map(k=>`<div class="kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div>${k.t}</div>`).join('');

  Object.values(charts).forEach(c=>c&&c.destroy()); charts={};
  if(!R.length){document.querySelector('.grid').style.opacity=.4;}
  else document.querySelector('.grid').style.opacity=1;
  const labels=R.map(r=>r.pr.split('/').pop());
  charts.f=new Chart(findings,{type:'bar',data:{labels,datasets:[
    {label:'Confirmed',data:R.map(r=>r.confirmed),backgroundColor:C.indigo,borderRadius:5},
    {label:'Refuted',data:R.map(r=>r.refuted),backgroundColor:C.slate,borderRadius:5}]},options:g});
  charts.o=new Chart(outcomes,{type:'doughnut',data:{labels:['Confirmed','Refuted','Suppressed'],
    datasets:[{data:[conf,ref,sum(R,'suppressed')],backgroundColor:[C.indigo,C.slate,C.amber],borderWidth:0}]},
    options:{maintainAspectRatio:false,cutout:'62%',plugins:{legend:{position:'bottom'}}}});
  const sev={critical:0,high:0,medium:0,low:0}; R.forEach(r=>Object.keys(sev).forEach(k=>sev[k]+=r.sev[k]||0));
  charts.s=new Chart(severity,{type:'bar',indexAxis:'y',data:{labels:['Critical','High','Medium','Low'],
    datasets:[{data:[sev.critical,sev.high,sev.medium,sev.low],backgroundColor:[C.red,C.amber,C.indigo,C.slate],borderRadius:5}]},
    options:{...g,plugins:{legend:{display:false}}}});
  const byDay={}; R.forEach(r=>{const d=new Date(r.ts*1000).toISOString().slice(5,10);byDay[d]=(byDay[d]||0)+1;});
  const days=Object.keys(byDay).sort();
  charts.d=new Chart(daily,{type:'bar',data:{labels:days,datasets:[{data:days.map(d=>byDay[d]),backgroundColor:C.green,borderRadius:5}]},
    options:{...g,plugins:{legend:{display:false}}}});

  // agent performance
  const agg={}; R.forEach(r=>r.findings.forEach(f=>f.agents.forEach(a=>{
    agg[a]=agg[a]||{findings:0,confirmed:0,refuted:0};
    agg[a].findings++; agg[a][f.status==='confirmed'?'confirmed':'refuted']++; })));
  const arows=Object.entries(agg).map(([a,v])=>({a,...v,p:v.findings?Math.round(v.confirmed/v.findings*100):0}))
    .sort((x,y)=>y.findings-x.findings);
  document.getElementById('agents').innerHTML=arows.length?`<table><thead><tr><th>Agent</th><th>Findings</th><th>Confirmed</th><th>Refuted</th><th>Precision</th></tr></thead><tbody>${arows.map(r=>`<tr><td>${agentName(r.a)}</td><td class="num">${r.findings}</td><td class="num">${r.confirmed}</td><td class="num">${r.refuted}</td><td class="num">${r.p}%</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No agent data.</div>';

  // recent table
  const rows=[...R].reverse().slice(0,25);
  document.getElementById('table').innerHTML=rows.length?`<table><thead><tr><th>PR</th><th>Confirmed</th><th>Refuted</th><th>Latency</th><th>Cost</th></tr></thead><tbody>${rows.map(r=>`<tr class="clk" onclick="location.href='/review/${r.idx}'"><td>${r.pr}</td><td><span class="pill ok">🟢 ${r.confirmed}</span></td><td><span class="pill no">🔴 ${r.refuted}</span></td><td class="num">${r.latency}s</td><td class="num">$${r.cost.toFixed(4)}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No reviews match these filters.</div>';
}
render();
</script></body></html>"""


_DETAIL_TEMPLATE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review detail</title>__THEMEHEAD__<style>__CSS__</style></head><body><div class="wrap">
  <header><span style="font-size:1.4rem">🤖</span><h1 id="title">Review</h1>
    <span class="spacer"></span>
    <a href="/dashboard" class="toggle">← Dashboard</a>
    <button class="toggle" onclick="toggleTheme()" style="margin-left:.5rem">☀️ / 🌙</button></header>
  <div class="sub" id="meta"></div>
  <div class="panel"><h3>Pipeline timeline</h3><div class="timeline" id="timeline"></div></div>
  <div id="findings"></div>
</div>
<script>
const T = __DATA__;
__THEMEJS__
document.getElementById('title').textContent=T.pr||'Review';
const tk=T.tokens||{},tm=T.timings_ms||{};
document.getElementById('meta').innerHTML=
  `<span class="chip">💲 $${(T.cost_usd||0).toFixed(4)}</span>`+
  `<span class="chip">🔢 ${(tk.total||0).toLocaleString()} tokens</span>`+
  `<span class="chip">⏱ ${tm.total||0} ms</span>`+
  `<span class="chip">🤖 ${(T.planned_agents||[]).join(', ')}</span>`;
const stages=[['Diff',tm.diff],['Static',tm.static],['RAG',tm.rag],['Agents+Verify',tm.review]];
document.getElementById('timeline').innerHTML=stages.map((s,i)=>
  `<div class="stage">${s[0]}<b>${s[1]||0} ms</b></div>`+(i<stages.length-1?'<span class="arrow">→</span>':'')).join('');
const SEV={critical:'#ef4444',high:'#f59e0b',medium:'#6366f1',low:'#94a3b8'};
const CONF={high:95,medium:70,low:40,tool:100};
const byAgent={};
(T.findings||[]).forEach(f=>(f.agents||['?']).forEach(a=>{(byAgent[a]=byAgent[a]||[]).push(f);}));
let html='';
if(!(T.findings||[]).length){html='<div class="panel empty">No findings recorded for this review.</div>';}
Object.keys(byAgent).forEach(agent=>{
  html+=`<div class="agent-h">${agentName(agent)}</div><div class="panel">`;
  byAgent[agent].forEach(f=>{
    const ok=f.status==='confirmed';
    const conf=CONF[f.confidence]!==undefined?CONF[f.confidence]:60;
    html+=`<div class="finding">
      <span class="badge" style="background:${ok?'#10b981':'#94a3b8'}">${ok?'✓ Confirmed':'✗ Refuted'}</span>
      <span class="badge" style="background:${SEV[f.severity]||'#94a3b8'}">${f.severity}</span>
      <h4>${f.title||'Finding'}</h4>
      <div class="meta">${f.file||'?'} : ${f.line||'?'} · ${f.category||''} · confidence: ${f.confidence||'n/a'}</div>
      <div class="bar"><span style="width:${conf}%"></span></div>
      <p>${f.description||''}</p>
      ${f.reason?`<p><b>Verifier reasoning:</b> ${f.reason}</p>`:''}
      ${f.evidence?`<pre>${f.evidence.replace(/</g,'&lt;')}</pre>`:''}
      ${f.suggestion?`<p><b>Suggested fix:</b> ${f.suggestion.replace(/</g,'&lt;')}</p>`:''}
    </div>`;
  });
  html+='</div>';
});
document.getElementById('findings').innerHTML=html;
</script></body></html>"""



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
