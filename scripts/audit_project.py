"""Read-only project audit. Writes audit_summary.txt. Modifies nothing."""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SKIP = {".venv", ".git", "__pycache__", ".ipynb_checkpoints"}
KEYWORDS = re.compile(
    r"(f1|auc|recall|precision|fidelity|shape|pair trend|"
    r"missing|accuracy|score)",
    re.I,
)
DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")
out = []


def log(s=""):
    out.append(str(s))


def read_text_safe(path):
    """Read a text file, handling UTF-8, UTF-8-BOM and UTF-16 (PowerShell)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16") + "\n[NOTE: file is UTF-16]"
    return raw.decode("utf-8-sig", errors="replace")


def files(folder):
    base = ROOT / folder
    if not base.exists():
        return []
    return [
        p for p in base.rglob("*")
        if p.is_file() and not (set(p.parts) & SKIP)
    ]


# 1. Folder contents
log("== FOLDER CONTENTS ==")
for d in ["artifacts", "reports", "results", "data/synthetic", "app", "tests"]:
    fs = files(d)
    log(f"{d}: {len(fs)} files")
    for p in fs[:25]:
        size = p.stat().st_size
        log(f"   {p.relative_to(ROOT)}  ({size:,} B)")

# 2. Git
log("\n== GIT ==")
for cmd in (["git", "status", "--short"], ["git", "log", "--oneline", "-5"]):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log(r.stdout.strip() or r.stderr.strip() or "(empty)")
    except FileNotFoundError:
        log("git not installed")

# 3. Processed data
log("\n== PROCESSED DATA ==")
for f in sorted((ROOT / "data" / "processed").glob("*.csv")):
    try:
        df = pd.read_csv(f)
    except Exception as exc:
        log(f"{f.name}: UNREADABLE ({type(exc).__name__}: {exc})")
        continue
    nan = (df.isna().mean() * 100).round(2).to_dict()
    log(f"{f.name}: shape={df.shape}")
    log(f"   cols={list(df.columns)}")
    log(f"   NaN%={nan}")
    for c in df.columns:
        if "target" in c.lower() or c.lower() in {"engagement", "label"}:
            dist = df[c].value_counts(normalize=True).round(4).to_dict()
            log(f"   {c} dist={dist}")

# 4. Notebooks
log("\n== NOTEBOOKS ==")
for nb in sorted(ROOT.rglob("*.ipynb")):
    if set(nb.parts) & SKIP:
        continue
    size = nb.stat().st_size
    try:
        data = json.loads(read_text_safe(nb))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log(f"{nb.name}: UNREADABLE size={size} B ({type(exc).__name__})")
        continue
    code = [c for c in data.get("cells", []) if c["cell_type"] == "code"]
    executed = [c for c in code if c.get("execution_count")]
    errors = sum(
        1 for c in code for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    src = "\n".join("".join(c["source"]) for c in code)
    has_drive_path = bool(DRIVE_PATH.search(src))
    n_code = len(code)
    n_exec = len(executed)
    log(
        f"{nb.name}: size={size} B code_cells={n_code} executed={n_exec} "
        f"errors={errors} hardcoded_drive_path={has_drive_path}"
    )
    hits = []
    for c in code:
        for o in c.get("outputs", []):
            text = "".join(o.get("text", [])) if "text" in o else ""
            for line in text.splitlines():
                if KEYWORDS.search(line) and len(line) < 200:
                    hits.append(line.strip())
    for h in hits[:40]:
        log(f"   | {h}")

# 5. Small docs and config (truncated)
for name in [
    "configs/base.yaml",
    "requirements-base.txt",
    ".gitignore",
    "results/data_audit.json",
]:
    p = ROOT / name
    log(f"\n== {name} ==")
    log(read_text_safe(p)[:2500] if p.exists() else "MISSING")

p = ROOT / "PROJECT_LOG.md"
log("\n== PROJECT_LOG.md (last 60 lines) ==")
if p.exists():
    log("\n".join(read_text_safe(p).splitlines()[-60:]))
else:
    log("MISSING")

(ROOT / "audit_summary.txt").write_text("\n".join(out), encoding="utf-8")
print("Wrote audit_summary.txt")
