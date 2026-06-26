"""
Phase 3: deterministic static analysis.

Runs real tools (ruff, bandit) on the changed files BEFORE the LLM. Unlike the
model, these tools never hallucinate and never get talked out of a finding —
bandit flags eval() as B307 every single time, no reasoning required. We feed
their output into the review as ground-truth facts.

ruff   -> F (pyflakes: real bugs) + B (bugbear: footguns)   [we skip style noise]
bandit -> Python security issues (eval, subprocess, hardcoded secrets, ...)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Find the tool binaries next to the running Python (i.e. inside our .venv), so
# this works even when PATH doesn't include the venv — e.g. under the webhook
# server. Falls back to the bare name if not found there.
_BIN = Path(sys.executable).parent


def _tool(name: str) -> str:
    candidate = _BIN / name
    return str(candidate) if candidate.exists() else name


def _rel(path: str) -> str:
    """Strip the leading './' that the tools print for cwd-relative paths."""
    return path[2:] if path.startswith("./") else path


def _relativize(filename: str, base: Path) -> str:
    """Normalize a tool's reported path to one relative to the temp dir.

    The tools disagree: bandit prints './demo.py' (relative), ruff prints the
    absolute path. We resolve both against the real temp dir so findings match
    the relative keys we wrote the files under.
    """
    try:
        return str(Path(filename).resolve().relative_to(base))
    except ValueError:
        return _rel(filename)


def analyze_files(files: dict) -> list:
    """files: {relative_path: full_file_content}.

    Returns a normalized list of findings:
      {tool, code, severity, category, file, line, message}
    Only Python files are analyzed (these tools are Python-only).
    """
    py_files = {p: c for p, c in files.items() if p.endswith(".py") and c}
    if not py_files:
        return []

    findings = []
    # Tools need real files on disk (they parse the AST), so we materialize the
    # changed files into a throwaway temp dir, run there, then it's cleaned up.
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        base = tmpdir.resolve()
        for path, content in py_files.items():
            dest = tmpdir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        findings += _run_ruff(tmpdir, base)
        findings += _run_bandit(tmpdir, base)
    return findings


def _run_ruff(tmpdir: Path, base: Path) -> list:
    try:
        proc = subprocess.run(
            [_tool("ruff"), "check", ".", "--output-format", "json", "--select", "F,B"],
            cwd=tmpdir, capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    try:
        items = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        code = it.get("code") or ""
        out.append({
            "tool": "ruff",
            "code": code,
            "severity": "medium",          # F/B are real bugs, not style nits
            "category": "bug",
            "file": _relativize(it.get("filename", ""), base),
            "line": (it.get("location") or {}).get("row"),
            "message": it.get("message", ""),
        })
    return out


def _run_bandit(tmpdir: Path, base: Path) -> list:
    try:
        # bandit exits 1 when it finds issues; that's not an error for us, so we
        # don't pass check=True. JSON goes to stdout.
        proc = subprocess.run(
            [_tool("bandit"), "-r", ".", "-f", "json", "-q"],
            cwd=tmpdir, capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return []
    sev_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
    out = []
    for r in data.get("results", []):
        out.append({
            "tool": "bandit",
            "code": r.get("test_id", ""),
            "severity": sev_map.get(r.get("issue_severity", ""), "medium"),
            "category": "security",
            "file": _relativize(r.get("filename", ""), base),
            "line": r.get("line_number"),
            "message": r.get("issue_text", ""),
        })
    return out


if __name__ == "__main__":
    # Standalone smoke test: analyze a deliberately bad snippet.
    sample = {
        "demo.py": (
            "import os\n"
            "def run(cmd):\n"
            "    return eval(cmd)\n"
            "API_KEY = 'sk_live_hardcoded'\n"
            "def f(items=[]):\n"   # B006 mutable default arg (bugbear)
            "    return items\n"
        )
    }
    for f in analyze_files(sample):
        print(f"[{f['tool']} {f['code']}] {f['file']}:{f['line']} ({f['severity']}/{f['category']}) — {f['message']}")
