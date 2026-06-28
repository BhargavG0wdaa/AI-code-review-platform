"""
Phase 6: observability.

Every review appends one JSON line to traces.jsonl with timings, token usage,
and finding counts. The webhook server reads these back for /stats and a
/dashboard page. JSONL (one JSON object per line) is the simplest possible
"trace store" — append-only, no DB, trivially greppable.
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

TRACE_FILE = Path(__file__).parent / "traces.jsonl"
SEVERITIES = ["critical", "high", "medium", "low"]


def record_trace(trace: dict) -> None:
    """Append a trace record (a timestamp is added automatically)."""
    row = {"timestamp": time.time(), **trace}
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_traces(limit: int | None = None) -> list:
    """Load trace records, newest last. `limit` returns only the most recent N."""
    if not TRACE_FILE.exists():
        return []
    traces = [json.loads(line) for line in TRACE_FILE.read_text().splitlines() if line.strip()]
    return traces[-limit:] if limit else traces


def compute_stats(traces: list) -> dict:
    """Aggregate metrics across all reviews."""
    n = len(traces)
    if n == 0:
        return {"reviews": 0}

    confirmed = sum(t.get("counts", {}).get("confirmed", 0) for t in traces)
    refuted = sum(t.get("counts", {}).get("refuted", 0) for t in traces)
    tokens = sum(t.get("tokens", {}).get("total", 0) for t in traces)
    latency = sum(t.get("timings_ms", {}).get("total", 0) for t in traces)
    judged = confirmed + refuted

    return {
        "reviews": n,
        "total_confirmed": confirmed,
        "total_refuted": refuted,
        # How often the verifier kills a proposed finding — a key quality signal.
        "refute_rate": round(refuted / judged, 2) if judged else 0,
        "avg_findings_per_review": round(confirmed / n, 1),
        "total_tokens": tokens,
        "avg_tokens_per_review": round(tokens / n),
        "avg_latency_ms": round(latency / n),
        "total_cost_usd": round(sum(t.get("cost_usd", 0) for t in traces), 4),
    }


def agent_performance(traces: list) -> list:
    """Per-agent scorecard: how many findings each agent raised, how many the
    verifier confirmed vs refuted, and the resulting precision. A consensus
    finding (multiple agents) is credited to each of them."""
    agg = defaultdict(lambda: {"findings": 0, "confirmed": 0, "refuted": 0})
    for t in traces:
        for f in t.get("findings", []):
            for agent in f.get("agents", ["?"]):
                a = agg[agent]
                a["findings"] += 1
                a["confirmed" if f.get("status") == "confirmed" else "refuted"] += 1
    rows = []
    for agent, v in agg.items():
        precision = round(v["confirmed"] / v["findings"] * 100) if v["findings"] else 0
        rows.append({"agent": agent, **v, "precision": precision})
    return sorted(rows, key=lambda r: -r["findings"])


def severity_distribution(traces: list) -> dict:
    """Count of CONFIRMED findings by severity (what actually shipped to devs)."""
    counts = Counter()
    for t in traces:
        for f in t.get("findings", []):
            if f.get("status") == "confirmed":
                counts[f.get("severity", "low")] += 1
    return {s: counts.get(s, 0) for s in SEVERITIES}


def get_trace(index: int) -> dict | None:
    """Return one trace by its position in the log (used by the detail page)."""
    traces = load_traces()
    if 0 <= index < len(traces):
        return traces[index]
    return None
