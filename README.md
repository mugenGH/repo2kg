# repo2kg

**Turn any Python repository into a Knowledge Graph that AI agents can query instead of reading every file.**

Instead of feeding entire codebases to Claude Code, Codex, or Copilot, `repo2kg` builds a searchable graph of your codebase — functions, classes, call edges, and semantic embeddings — so agents query exactly what they need.

```
Without repo2kg                      With repo2kg
────────────────────                 ────────────────────
Agent reads 64 files                 Agent reads CODEBASE.md
→ 43,000 tokens                      → 8,900 tokens (80% savings)
→ Slow, hits context limits          → Fast, precise, relevant
→ New session = start over           → KG persists forever
```

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Python AST  │────▶│  Knowledge Graph  │────▶│  FAISS + Embed  │
│   Parsing    │     │  Nodes + Edges   │     │  Semantic Index  │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
  Layer 1: Graph                          Layer 2: RAG
  ─ Functions, Classes, Methods           ─ Sentence embeddings
  ─ CALLS / CALLED_BY edges               ─ FAISS vector search
  ─ File locations, signatures            ─ Semantic entry points
  ─ 8-line body previews
```

1. **Parse** — AST-walks every `.py` file: functions, classes, methods, signatures, docstrings, 8-line previews
2. **Link** — Resolves call edges between nodes (who calls whom)
3. **Embed** — Encodes each node into a vector with `all-MiniLM-L6-v2`, indexes with FAISS
4. **Query** — Semantic search finds entry points → graph traversal expands related code → token-minimal output

## Installation

```bash
pip install repo2kg
```

Or from source:

```bash
git clone https://github.com/shreeramktm2004-dev/repo2kg.git
cd repo2kg
pip install -e .
```

**Requirements:** Python 3.10+, deps auto-installed: `numpy`, `faiss-cpu`, `sentence-transformers`

---

## Complete Workflow

### Step 1 — One-time global setup

```bash
repo2kg user-setup
```

Installs agent instructions globally:
- `~/.claude/CLAUDE.md` — Claude Code reads this at the start of **every session**
- `~/.codex/AGENTS.md` — Codex equivalent
- `~/.repo2kg/GLOBAL_AGENTS.md` — reference doc

### Step 2 — Register all your projects at once

```bash
repo2kg scan                          # scans your entire home directory
repo2kg scan --root ~/projects        # or scope to a specific folder
```

Finds every `kg.json` file (by detecting companion `.faiss` files) and registers them all in `~/.repo2kg/registry.json`. Re-run anytime to pick up new projects.

### Step 3 — Build + set up a project

```bash
cd /your/project
repo2kg build --repo . --out kg.json
repo2kg agent-setup --kg kg.json --dir .
git add CLAUDE.md CODEBASE.md AGENTS.md kg.json
git commit -m "Add repo2kg knowledge graph"
```

`agent-setup` generates five files and auto-registers the project:

| File | Who reads it | Purpose |
|------|-------------|---------|
| `CODEBASE.md` | Any agent / human | Full overview: file map, classes, call graph, all signatures |
| `CLAUDE.md` | Claude Code (auto-detected) | Instructions to use KG instead of reading files |
| `.copilot-instructions.md` | GitHub Copilot | Same, Copilot format |
| `.github/copilot-instructions.md` | Copilot agent mode | Same, agent mode format |
| `AGENTS.md` | Multi-agent systems | Node schema + query strategies |

### Step 4 — Just ask the agent

Open any session in a registered project and ask naturally:

```
"Add a password reset endpoint"
"Fix the double-charge bug in payment flow"
"Where is rate limiting applied?"
"Refactor the database connection pooling"
```

The agent finds the KG automatically via `~/.claude/CLAUDE.md`, reads `CODEBASE.md`, queries `kg.json` with stdlib Python, then reads only the 2-3 source files it actually needs.

---

## CLI Reference

```
# Build
repo2kg build [--repo PATH] [--out FILE] [--exclude PATTERN ...]

# Query (requires FAISS + sentence-transformers)
repo2kg query QUESTION [--kg FILE] [--k INT] [--depth INT] [--format text|json]

# Query (zero dependencies — keyword matching)
repo2kg query-lite KEYWORDS [--kg FILE] [--k INT] [--depth INT] [--format text|json]

# Export standalone markdown for agents
repo2kg export [--kg FILE] [--out FILE]

# Generate all agent instruction files for a project
repo2kg agent-setup [--kg FILE] [--dir PATH] [--no-register]

# Global setup (run once ever)
repo2kg user-setup

# Register all KG files under a directory tree
repo2kg scan [--root PATH]

# Register one project manually
repo2kg register [--kg FILE] [--project PATH]

# List registered projects
repo2kg list

# Show KG statistics
repo2kg stats [--kg FILE]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | `.` | Repository root to scan |
| `--out` | `kg.json` | Output path for the KG |
| `--kg` | `kg.json` | Path to a saved KG |
| `--k` | `5` | Number of search results |
| `--depth` | `1` | Graph traversal depth (1 = direct calls, 2 = calls-of-calls) |
| `--format` | `text` | Output format: `text` or `json` |
| `--dir` | `.` | Target directory for generated agent files |
| `--root` | `~` | Root directory for `scan` |
| `-v` | off | Verbose/debug logging |

### Default Exclusions

Automatically skipped during `build`:

```
__pycache__  .git  node_modules  .tox  .venv  venv  env
.mypy_cache  .pytest_cache  dist  build  site-packages
```

Add more: `repo2kg build --repo . --exclude migrations fixtures`

---

## How Agents Use It (Zero Dependencies)

The global `~/.claude/CLAUDE.md` tells the agent to run this at the start of every task:

```python
import json
from pathlib import Path

# Walk up from cwd to find the closest registered project
registry = json.load(open(Path.home() / ".repo2kg" / "registry.json"))
check = Path.cwd()
while check != check.parent:
    if str(check) in registry["projects"]:
        kg = json.load(open(registry["projects"][str(check)]["kg"]))
        break
    check = check.parent

# Now search — no FAISS, no embeddings, pure stdlib
matches = [n for n in kg.values() if "auth" in n["name"].lower()]
for n in matches[:10]:
    print(n["signature"], "—", n.get("docstring", "")[:100])
    print("calls:", [c.split("::")[-1] for c in n.get("calls", [])])
```

The agent only reads actual source files when the 8-line `body_preview` is not enough.

---

## Commands In Detail

### `repo2kg query` — Semantic search

```bash
repo2kg query "how does authentication work" --kg kg.json
repo2kg query "database connection pooling" --kg kg.json --depth 2
repo2kg query "auth flow" --kg kg.json --format json
```

Output:
```
# Query: how does authentication work
# Nodes returned: 12  |  ~340 tokens
# Entry points: verify_token, authenticate_user, JWTMiddleware
# ─────────────────────────────────────────────

# METHOD: verify_token
# file: auth/jwt.py  class: JWTService
def JWTService.verify_token(self, token: str) -> dict:
"""Verify and decode a JWT token."""
    decoded = jwt.decode(token, self.secret, algorithms=["HS256"])
# calls: decode, get_user_from_cache
# called_by: authenticate_user, AuthMiddleware.process_request
```

### `repo2kg query-lite` — Keyword search, zero heavy deps

No FAISS, no embeddings, instant startup. Uses keyword + graph expansion.

```bash
repo2kg query-lite "auth" --kg kg.json
repo2kg query-lite "database connection" --kg kg.json --format json
```

### `repo2kg export` — Standalone CODEBASE.md

Generates a single markdown file any agent can read without any tooling:

```bash
repo2kg export --kg kg.json --out CODEBASE.md
```

Contains: overview table, file map, architecture grouping, class/method tables, call graph, all signatures.

### `repo2kg scan` — Register everything at once

```bash
repo2kg scan                       # scans ~ (home directory)
repo2kg scan --root ~/work         # scans a specific root
```

Walks the directory tree, finds every KG file (by detecting companion `.faiss` files), and registers them all in `~/.repo2kg/registry.json`. Safe to re-run — existing entries are updated, not duplicated.

```
Scanning /home/user for KG files...
  + /home/user/myproject
      KG: /home/user/myproject/kg.json
  ~ /home/user/other-project (updated)

Done: 1 new, 1 updated — 2 total registered
```

### `repo2kg list`

```
Registered projects (4):
  Project                            Registered
  ──────────────────────────────     ─────────────────────
  /home/user/myproject           ✓   2026-04-16T15:37:13
  /home/user/other-project       ✓   2026-04-16T15:46:34
```

---

## Python API

```python
# Full mode (requires FAISS + sentence-transformers)
from repo2kg import RepoKG

kg = RepoKG().build("./my_project")
kg.save("kg.json")

kg = RepoKG.load("kg.json")
print(kg.query("payment processing", k=5, depth=2))        # text
print(kg.query_json("auth", k=3))                          # structured dict

# Lightweight mode (stdlib only — safe for agent environments)
from repo2kg import RepoKGLite

kg = RepoKGLite("kg.json")
print(kg.query("payment processing", k=5, depth=1))        # text
result = kg.query_json("auth", k=3)                        # structured dict
callers = kg.get_callers("auth/service.py::AuthService.login")
```

---

## Node Schema

Every node in `kg.json`:

```json
{
  "auth/jwt.py::JWTService.verify_token": {
    "id": "auth/jwt.py::JWTService.verify_token",
    "name": "verify_token",
    "kind": "method",
    "file": "auth/jwt.py",
    "parent_class": "JWTService",
    "signature": "def JWTService.verify_token(self, token: str) -> dict:",
    "docstring": "Verify and decode a JWT token.",
    "body_preview": "    decoded = jwt.decode(token, self.secret, ...)\n    ...",
    "calls": ["auth/jwt.py::JWTService.decode", "cache/redis.py::get_user_from_cache"],
    "callers": ["auth/service.py::authenticate_user"],
    "imports": ["jwt", "datetime"]
  }
}
```

---

## When to Use repo2kg

| Project Size | Files | Token Savings | Recommendation |
|---|---|---|---|
| Small | < 20 | 10–20% | Optional |
| Medium | 20–100 | 40–70% | Recommended |
| Large | 100–500 | 70–90% | Strongly recommended |
| Monorepo | 500+ | 85–95% | Essential |

Value scales with **interconnectedness** — more call edges = more graph traversal advantage.

---

## Limitations

- **Python only** — AST parsing is Python-specific. TypeScript/Go/Rust support planned.
- **Static analysis** — Captures AST calls, not dynamic dispatch or runtime-generated calls.
- **Name-based resolution** — Call edges resolved by function name (same-file preferred). No full type inference.
- **No incremental updates** — Rebuilds the full graph each time. Large repos take 30–60 seconds.

## Contributing

Contributions welcome. Open an issue first for major changes.

```bash
git clone https://github.com/shreeramktm2004-dev/repo2kg.git
cd repo2kg
pip install -e .
```

## License

[MIT](LICENSE)
