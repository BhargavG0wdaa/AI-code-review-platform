"""
Phase 1: A single-file AI code reviewer.

Flow:  PR URL --> fetch diff (GitHub API) --> prompt --> Groq LLM --> JSON findings --> print

Run:   python pr_reviewer.py https://github.com/owner/repo/pull/123

This is intentionally ONE file with no web server, DB, or agents. It's the
core loop everything else in the architecture wraps around. Get this solid,
then we layer on top of it in later phases.
"""

import argparse
import json
import os
import re
import sys

import httpx
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from static_analysis import analyze_files

load_dotenv()
console = Console()

# Llama 3.3 70B on Groq: fast, free tier, good at code. This is the "LLM
# INFERENCE" box in your diagram.
MODEL = "llama-3.3-70b-versatile"

# We cap the diff we send to the model. Huge PRs would blow the context
# window and cost. Real systems chunk per-file; for Phase 1 we just truncate
# and tell the user. (We'll do proper chunking in a later phase.)
MAX_DIFF_CHARS = 30_000

# We drop findings below this confidence. The model is told to self-rate, and
# we trust that signal to suppress hunches. Tighten to "high" for very strict
# reviews, loosen to "low" to see everything the model considered.
MIN_CONFIDENCE = "high"
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


class ReviewError(Exception):
    """Raised when the review can't proceed (bad URL, fetch failure, bad model
    output). Reusable callers (CLI, web server) catch this and decide what to do.
    We raise instead of calling sys.exit() so library code never kills its host
    process — that distinction matters once a web server imports these functions.
    """


# ---------------------------------------------------------------------------
# 1. Fetch the PR diff from GitHub
# ---------------------------------------------------------------------------
def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Pull (owner, repo, pr_number) out of a GitHub PR URL."""
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not match:
        raise ReviewError(f"Not a valid GitHub PR URL: {url}")
    owner, repo, number = match.groups()
    return owner, repo, int(number)


def fetch_diff(owner: str, repo: str, number: int) -> str:
    """
    GitHub serves the raw diff if you ask for the right media type.
    The 'Accept: application/vnd.github.v3.diff' header is the trick —
    same endpoint, but you get a unified diff back instead of JSON.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    headers = {"Accept": "application/vnd.github.v3.diff"}

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    if resp.status_code == 404:
        raise ReviewError("PR not found (private repo without a token, or bad URL).")
    if resp.status_code == 403:
        raise ReviewError("GitHub rate-limited you or denied access. Add/refresh GITHUB_TOKEN in .env.")
    resp.raise_for_status()
    return resp.text


def added_line_numbers(patch: str) -> set:
    """Parse a unified-diff patch and return the set of line numbers ADDED in
    the new version of the file. Used to filter static findings down to lines
    this PR actually touched (so we don't flag pre-existing issues)."""
    added: set = set()
    new_line = 0
    for line in (patch or "").splitlines():
        if line.startswith("@@"):
            # Hunk header: @@ -old,n +new,n @@ — grab the new-file start line.
            m = re.search(r"\+(\d+)", line)
            new_line = int(m.group(1)) if m else 0
        elif line.startswith("+"):
            added.add(new_line)
            new_line += 1
        elif line.startswith("-"):
            pass  # removed line — doesn't advance the new-file counter
        else:
            new_line += 1  # context line
    return added


def fetch_pr_files(owner: str, repo: str, number: int) -> dict:
    """Return {path: {"content": <full file at PR head>, "added": {line nums}}}
    for the Python files changed in the PR. Static tools need whole files (they
    parse the AST), so we fetch each file's content, not just the diff hunk."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = httpx.get(url, headers=headers, params={"per_page": 100},
                     timeout=30, follow_redirects=True)
    if resp.status_code in (403, 404):
        raise ReviewError(f"Could not list PR files (HTTP {resp.status_code}).")
    resp.raise_for_status()

    files: dict = {}
    for item in resp.json():
        path = item.get("filename", "")
        if item.get("status") == "removed" or not path.endswith(".py"):
            continue
        added = added_line_numbers(item.get("patch", ""))
        content = ""
        raw_url = item.get("raw_url")
        if raw_url:
            c = httpx.get(raw_url, headers=headers, timeout=30, follow_redirects=True)
            if c.status_code == 200:
                content = c.text
        files[path] = {"content": content, "added": added}
    return files


def _static_to_finding(s: dict) -> dict:
    """Convert a raw static-tool finding into our standard finding shape, tagged
    as deterministic so the rest of the pipeline treats it as ground truth."""
    return {
        "severity": s.get("severity", "medium"),
        "category": s.get("category", "bug"),
        "confidence": "high",
        "file": s.get("file"),
        "line": s.get("line"),
        "evidence": "",
        "title": f"{s.get('tool', '').upper()} {s.get('code', '')}".strip(),
        "description": s.get("message", ""),
        "suggestion": "",
        "source": "static",
    }


def gather_static_findings(owner: str, repo: str, number: int) -> list:
    """Run ruff + bandit on the PR's changed Python files and return findings
    (in standard shape) for lines this PR actually added/changed."""
    files = fetch_pr_files(owner, repo, number)
    contents = {p: f["content"] for p, f in files.items()}
    out = []
    for r in analyze_files(contents):
        added = files.get(r["file"], {}).get("added", set())
        if r.get("line") in added:
            out.append(_static_to_finding(r))
    return out


# ---------------------------------------------------------------------------
# 2. Build the prompt
# ---------------------------------------------------------------------------
# The system prompt defines the reviewer's job AND the exact output shape.
# Forcing a strict JSON schema is what makes the output usable by code later
# (posting comments, scoring, storing). "Vibes" text output is a dead end.
SYSTEM_PROMPT = """You are a staff-level software engineer doing a high-signal code review on a pull request diff. Your reviews are trusted because every comment is concrete, correct, and worth the developer's time.

## What to review
- Review ONLY lines the diff ADDS or MODIFIES (lines starting with `+`). Use surrounding context only to understand them.
- NEVER comment on code you cannot see in the diff. If judging an issue would require code outside the diff (other files, the rest of a function, imports at the top of the file), DO NOT raise it — you'd be guessing.

## What makes a finding worth raising
Only raise a finding if ALL of these are true:
- It points at a SPECIFIC added/changed line, and you can quote that exact line as evidence.
- A competent developer would agree it's a real problem, not a matter of taste.
- It has real impact: a bug, a security hole, a performance trap, a correctness or maintainability risk.

## Do NOT raise (these are noise that erodes trust)
- Vague suggestions: "consider adding a comment", "add a docstring", "this could be clearer".
- Pure style/formatting/import nitpicks — a linter handles those, not you.
- Speculation hedged with "might", "could potentially", "it's not clear" — if you're not confident, omit it.
- Praise or restating what the code does.

Prefer ZERO findings over weak ones. An empty findings array on clean code is a correct, valuable answer — do not invent problems to look useful.

## For each finding, calibrate confidence honestly
- "high": you are certain this is a real issue and your evidence proves it.
- "medium": likely an issue, but depends on context you can partially see.
- "low": a hunch. (You will rarely report these — usually just omit them.)

## MANDATORY self-critique before you answer
First brainstorm every candidate finding. Then, for EACH candidate, apply this test before keeping it:
1. Is the bug PROVABLE from code visible in the diff alone?
2. Or does proving it require assuming how some code you CANNOT fully see behaves
   (a called function's body, the rest of a function, another file, runtime values)?

If proving the finding depends on an assumption about unseen code, you are GUESSING.
DROP it from "findings" and instead record it in "rejected" with the assumption you'd have had to make.
A confidently-worded guess about unseen code is the worst kind of review comment — it destroys trust.

When unsure, reject. It is far better to miss a real bug than to report a fake one.

Respond with ONLY valid JSON in exactly this shape:
{
  "summary": "1-2 sentence overall assessment of the change",
  "findings": [
    {
      "severity": "critical | high | medium | low",
      "category": "security | performance | bug | architecture | testing | style | docs",
      "confidence": "high | medium | low",
      "file": "path/to/file.py",
      "line": "the line number or hunk reference from the diff",
      "evidence": "the exact added/changed line(s) you are flagging, quoted verbatim from the diff",
      "title": "short, specific title",
      "description": "what's wrong and the concrete impact if left unfixed",
      "suggestion": "a specific fix — ideally the corrected code"
    }
  ],
  "rejected": [
    {
      "title": "candidate finding you considered but dropped",
      "reason": "the unseen-code assumption it depended on, so it couldn't be proven"
    }
  ]
}"""


def build_user_prompt(diff: str, static_findings: list | None = None) -> str:
    truncated = diff[:MAX_DIFF_CHARS]
    note = ""
    if len(diff) > MAX_DIFF_CHARS:
        note = "\n\n[NOTE: diff was truncated for length — review what is shown.]"

    static_block = ""
    if static_findings:
        rows = "\n".join(
            f"- {s.get('file')}:{s.get('line')} [{s.get('title')}] {s.get('description')}"
            for s in static_findings
        )
        # These are deterministic tool outputs. Telling the model they're already
        # caught stops it from re-reporting them, and frames them as facts it can
        # build on rather than second-guess.
        static_block = (
            "\n\nStatic-analysis tools (ruff, bandit) already flagged the issues "
            "below. Treat them as ESTABLISHED FACTS — do NOT re-report them, and "
            "do NOT contradict them. You may add deeper context the tools miss:\n"
            f"{rows}"
        )
    return f"Here is the pull request diff:\n\n```diff\n{truncated}\n```{note}{static_block}"


# ---------------------------------------------------------------------------
# 3. Call the LLM
# ---------------------------------------------------------------------------
_client: Groq | None = None


def get_client() -> Groq:
    """One reused client for every LLM call (review + verification)."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _chat_json(system: str, user: str) -> dict:
    """Send one system+user turn, force JSON back, parse it."""
    response = get_client().chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ReviewError("Model returned an empty response.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReviewError(f"Model did not return valid JSON: {e}\n{raw}")


def review_diff(diff: str, static_findings: list | None = None) -> dict:
    """First pass: the reviewer proposes findings (grounded by static findings)."""
    return _chat_json(SYSTEM_PROMPT, build_user_prompt(diff, static_findings))


# ---------------------------------------------------------------------------
# 3b. The adversarial verifier — the highest-value box in the whole system.
# ---------------------------------------------------------------------------
# This is a SEPARATE LLM call whose only job is to REFUTE a finding. It starts
# from "this is wrong" and only flips to confirmed if the diff proves the bug.
# Independence is the point: a model re-checking its own reasoning just agrees
# with itself. A fresh call told to attack the claim catches confident errors.
VERIFIER_SYSTEM_PROMPT = """You are a skeptical staff engineer. Your ONLY job is to DISPROVE a claimed code-review finding.

Assume the finding is WRONG. Your default verdict is "refuted". You flip to "confirmed" ONLY if the diff proves the bug is real.

You are given the full PR diff and one claimed finding. Trace the ACTUAL control flow and data flow in the diff:
- Does the flagged line really do what the finding claims?
- Do OTHER branches or lines in the diff already handle the case the finding worries about? (e.g. a later `elif` that catches the value.) If so, the finding is refuted.
- Does confirming the finding require assuming behavior of code NOT shown in the diff? If so it is unproven — refuted.

Be rigorous, not generous. A finding that "could be an issue" but isn't provably one is refuted. When in doubt, refute.

Respond with ONLY valid JSON:
{
  "verdict": "confirmed | refuted",
  "reasoning": "your control-flow trace, citing specific lines, that justifies the verdict"
}"""


def verify_finding(diff: str, finding: dict) -> dict:
    """Run one finding through the adversarial verifier. Returns {verdict, reasoning}."""
    claim = (
        f"Title: {finding.get('title')}\n"
        f"File: {finding.get('file')}  Line: {finding.get('line')}\n"
        f"Evidence line: {finding.get('evidence')}\n"
        f"Claimed problem: {finding.get('description')}"
    )
    user = (
        f"Full PR diff:\n```diff\n{diff[:MAX_DIFF_CHARS]}\n```\n\n"
        f"Claimed finding to disprove:\n{claim}"
    )
    return _chat_json(VERIFIER_SYSTEM_PROMPT, user)


def _norm(value: str | None) -> str:
    """Normalize an LLM-provided label (severity/confidence) so 'High' == 'high'."""
    return (value or "").strip().lower()


def select_candidates(review: dict) -> tuple[list[dict], int]:
    """Apply the confidence filter. Returns (candidates, suppressed_count)."""
    all_findings = review.get("findings", [])
    cutoff = _CONFIDENCE_RANK[MIN_CONFIDENCE]
    candidates = [f for f in all_findings
                  if _CONFIDENCE_RANK.get(_norm(f.get("confidence")), 9) <= cutoff]
    return candidates, len(all_findings) - len(candidates)


def run_review(diff: str, static_findings: list | None = None) -> dict:
    """The full pipeline as one callable: static facts + LLM review -> verify.

    Returns a plain dict so ANY caller (CLI, web server, tests) can use it.
    Static findings (ruff/bandit) are deterministic, so they skip the verifier
    and are auto-confirmed; only the LLM's probabilistic findings get verified.
    """
    static_findings = static_findings or []
    review = review_diff(diff, static_findings)
    candidates, suppressed = select_candidates(review)

    confirmed: list[dict] = []
    refuted: list[dict] = []
    for f in candidates:
        verdict = verify_finding(diff, f)
        f["_verdict"] = verdict
        if verdict.get("verdict") == "confirmed":
            confirmed.append(f)
        else:
            refuted.append(f)

    return {
        "summary": review.get("summary", ""),
        # Tools first (ground truth), then the verified LLM findings.
        "confirmed": static_findings + confirmed,
        "refuted": refuted,
        "suppressed": suppressed,
        "rejected": review.get("rejected", []),
        "static_count": len(static_findings),
    }


# ---------------------------------------------------------------------------
# 4. Print the results nicely
# ---------------------------------------------------------------------------
SEVERITY_COLOR = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "cyan",
}


def print_review(result: dict, pr_label: str) -> None:
    confirmed = result.get("confirmed", [])
    refuted = result.get("refuted", [])
    suppressed = result.get("suppressed", 0)

    console.print(Panel(result.get("summary") or "(no summary)",
                        title=f"Review of {pr_label}", border_style="blue"))

    if not confirmed:
        console.print("[green]No findings survived verification. Looks clean.[/green]")
    else:
        # Sort most severe first.
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        confirmed.sort(key=lambda f: order.get(_norm(f.get("severity")), 9))

        table = Table(show_lines=True)
        table.add_column("Severity")
        table.add_column("Category")
        table.add_column("Location")
        table.add_column("Issue, Evidence & Fix", max_width=80)

        for f in confirmed:
            sev = _norm(f.get("severity")) or "low"
            color = SEVERITY_COLOR.get(sev, "white")
            conf = f.get("confidence", "?")
            location = f"{f.get('file', '?')}\n:{f.get('line', '?')}"
            evidence = f.get("evidence", "")
            tag = " [magenta](static)[/magenta]" if f.get("source") == "static" else f"  [dim]({conf} confidence)[/dim]"
            body = f"[bold]{f.get('title', '')}[/bold]{tag}\n{f.get('description', '')}"
            if evidence:
                body += f"\n\n[dim]Evidence:[/dim] [italic]{evidence.strip()}[/italic]"
            if f.get("suggestion"):
                body += f"\n\n[dim]Fix:[/dim] {f.get('suggestion', '')}"
            table.add_row(f"[{color}]{sev}[/{color}]", f.get("category", ""), location, body)

        console.print(table)
        console.print(f"\n[bold]{len(confirmed)}[/bold] verified finding(s).")

    # Show everything the pipeline filtered out, and why — full transparency.
    if suppressed:
        console.print(f"[dim]{suppressed} low-confidence finding(s) suppressed before verification.[/dim]")
    if refuted:
        console.print(f"\n[bold red]Verifier refuted {len(refuted)} finding(s)[/bold red] "
                      f"[dim](claimed but disproven by tracing the diff):[/dim]")
        for f in refuted:
            reason = f.get("_verdict", {}).get("reasoning", "")
            console.print(f"  [red]✗[/red] [bold]{f.get('title', '')}[/bold]")
            console.print(f"    [dim]{reason}[/dim]")
    _print_rejected(result.get("rejected", []))


def _print_rejected(rejected: list[dict]) -> None:
    """Show what the model's self-critique threw out. This is the hallucination
    guard working in the open — each one is a fake bug that didn't get shipped."""
    if not rejected:
        return
    console.print(f"\n[dim]Self-critique dropped {len(rejected)} candidate(s) "
                  f"(couldn't be proven from the diff alone):[/dim]")
    for r in rejected:
        console.print(f"  [dim]• {r.get('title', '')} — {r.get('reason', '')}[/dim]")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AI code reviewer for a GitHub PR.")
    parser.add_argument("pr_url", help="GitHub PR URL, e.g. https://github.com/owner/repo/pull/123")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Missing GROQ_API_KEY. Copy .env.example to .env and add your key.")

    try:
        owner, repo, number = parse_pr_url(args.pr_url)
        pr_label = f"{owner}/{repo}#{number}"

        with console.status(f"Fetching diff for {pr_label}..."):
            diff = fetch_diff(owner, repo, number)
        if not diff.strip():
            raise ReviewError("That PR has an empty diff — nothing to review.")

        with console.status("Running static analysis (ruff, bandit)..."):
            static_findings = gather_static_findings(owner, repo, number)

        with console.status(f"Reviewing + verifying with {MODEL}..."):
            result = run_review(diff, static_findings)
    except ReviewError as e:
        sys.exit(f"Error: {e}")

    print_review(result, pr_label)


if __name__ == "__main__":
    main()
