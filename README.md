<div align="center">

```
  ██████╗ ███████╗██████╗  ██████╗ ██████╗ ██╗  ██╗ ██████╗
  ██╔══██╗██╔════╝██╔══██╗██╔═══██╗╚════██╗██║ ██╔╝██╔════╝
  ██████╔╝█████╗  ██████╔╝██║   ██║ █████╔╝█████╔╝ ██║  ███╗
  ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔═══╝ ██╔═██╗ ██║   ██║
  ██║  ██║███████╗██║      ╚██████╔╝███████╗██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝╚══════╝╚═╝       ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝
```

**Turn any repository into a Knowledge Graph for token-efficient AI agent context.**

[![PyPI version](https://img.shields.io/pypi/v/repo2kg)](https://pypi.org/project/repo2kg/)
[![Python](https://img.shields.io/pypi/pyversions/repo2kg)](https://pypi.org/project/repo2kg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## The Problem

Every time you ask Claude Code, Copilot, or Codex to work on your project, it reads hundreds of files — burning through your context window before it even starts.

```
Without repo2kg                      With repo2kg
────────────────────                 ─────────────────────
Agent reads 229 files                Agent reads CODEBASE.md
→ ~53,000+ tokens used               → ~4,000 tokens (85%-95% saved)
→ Slow, hits context limits          → Fast, precise, relevant
→ New session = start over           → KG persists forever
```

`repo2kg` builds a searchable graph of your entire codebase — functions, classes, call edges — so agents query exactly what they need instead of everything.

---

## Token Efficiency: Real Numbers

Tested on a project with **229 source files**, **2,909 nodes**, and **4,615 call edges**. The agent understood the full architecture without opening a single raw source file.

### With repo2kg

The agent ran four commands:

```bash
repo2kg list
repo2kg stats --kg kg.json
repo2kg query-lite "main app entry" --kg kg.json
repo2kg query-lite "authentication login jwt token" --kg kg.json
repo2kg export --kg kg.json
```

| Step | Tokens |
|---|---|
| stats output | ~100 |
| 4 × query-lite results | ~2,400 |
| CODEBASE.md overview | ~1,500 |
| **Total** | **~4,000 tokens** |

### Without repo2kg

```bash
# Agent must search and read files manually
glob file listing → grep functions/classes → read source files
```

| Step | Tokens |
|---|---|
| File listing | ~3,000 |
| Reading 10–20 source files | ~15,000–40,000 |
| Grep results | ~5,000–10,000 |
| **Total** | **~23,000–53,000 tokens** |

### Result

| Method | Tokens | Quality |
|---|---|---|
| repo2kg | ~4,000 | High — structured graph, signatures, call edges |
| Normal search | ~23,000–53,000 | Medium — raw code scanning |

```
Reduction:  83–92% fewer tokens during codebase discovery
Full session savings (including messages):  ~40–60%
```

The larger and more interconnected your repo, the bigger the gain:

| Project Size | Files | Token Savings |
|---|---|---|
| Small | < 20 | 10–20% |
| Medium | 20–100 | 40–70% |
| Large | 100–500 | 70–90% |
| Monorepo | 500+ | 85–95% |

---

## Quick Start

```bash
pip install repo2kg

# 1. One-time global setup (runs in seconds, no heavy deps)
repo2kg user-setup

# 2. Build a KG for your project
cd /your/project
repo2kg build --repo . --out kg.json

# 3. Generate agent instruction files
repo2kg agent-setup --kg kg.json --dir .

# 4. Query it
repo2kg query "how does authentication work" --kg kg.json

# 5. Generate an interactive visual graph
repo2kg visualize --kg kg.json --out kg_graph.html
```

---

## Installation

```bash
pip install repo2kg
```

**Requirements:** Python 3.10+

**Dependencies auto-installed:**
| Package | Size | When Used |
|---|---|---|
| `numpy` | ~20 MB | build, query |
| `faiss-cpu` | ~60 MB | build, query (CPU only, no GPU/CUDA needed) |
| `sentence-transformers` | ~2 GB (PyTorch) | build, query |
| `tree-sitter` + 9 grammars | ~10 MB | build (AST parsing) |

> **Note:** `query-lite`, `user-setup`, `export`, `agent-setup`, `list`, and `register` all start **instantly** — they never load PyTorch or FAISS. Only `build` and `query` load the heavy dependencies.

---

## How It Works

```
Your Repository
       │
       ▼
┌──────────────────────────────────────┐
│  Parse all source files              │  ← tree-sitter (accurate) or regex (fallback)
│  Python/JS/TS/Java/Go/Rust/C/C++    │
│  Ruby/C#  (10 languages)            │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Build Knowledge Graph               │
│  • Nodes: functions, classes, methods│
│  • Edges: who calls whom             │
│  • Data: signatures, docstrings,     │
│    8-line body previews              │
└──────────────────────────────────────┘
       │                    │
       ▼                    ▼
  kg.json / kg.toon     FAISS index
  (structured data)     (semantic vectors)
       │                    │
       └────────┬───────────┘
                ▼
         query / query-lite
         → token-minimal context for your agent
```

### Two query modes

| Command | How it searches | Deps needed | Speed |
|---|---|---|---|
| `query` | Semantic similarity (FAISS + embeddings) | PyTorch + FAISS | Slower start |
| `query-lite` | Keyword matching + graph expansion | **None** (stdlib only) | Instant |

Both do **graph expansion** — they find entry-point nodes, then follow call edges to return related code automatically.

### Two output formats

| Format | Extension | Best for |
|---|---|---|
| JSON | `.json` | Universal, default |
| TOON | `.toon` | LLM context windows (~40% fewer tokens) |

---

## Complete Workflow

### Step 1 — Global setup (once ever)

```bash
repo2kg user-setup
```

Writes agent instructions to:
- `~/.claude/CLAUDE.md` — Claude Code reads this at the start of **every** session
- `~/.codex/AGENTS.md` — Codex equivalent
- `~/.repo2kg/GLOBAL_AGENTS.md` — reference doc

From this point on, any registered project is automatically discovered by your agent.

---

### Step 2 — Discover & register existing projects

```bash
repo2kg scan                     # scans your entire home directory
repo2kg scan --root ~/projects   # or a specific folder
```

Finds every KG file (by detecting companion `.faiss` files) and registers them in `~/.repo2kg/registry.json`. Re-run any time.

---

### Step 3 — Build a KG for your project

```bash
cd /your/project

repo2kg build --repo . --out kg.json     # JSON (default)
repo2kg build --repo . --out kg.toon     # TOON (40% fewer tokens for LLMs)
```

Output:
```
KG ready: 421 nodes, 892 call edges from 63 files
Languages: javascript: 38, python: 25
Saved to kg.json [JSON] (421 nodes)
```

Three files are created: `kg.json`, `kg.json.faiss`, `kg.json.idx`

---

### Step 4 — Generate agent files

```bash
repo2kg agent-setup --kg kg.json --dir .
```

Creates five files:

| File | Purpose |
|---|---|
| `CODEBASE.md` | Full overview — file map, classes, call graph, all signatures |
| `CLAUDE.md` | Tells Claude Code to use the KG instead of reading files |
| `.copilot-instructions.md` | Same, for GitHub Copilot |
| `.github/copilot-instructions.md` | Same, for Copilot agent mode |
| `AGENTS.md` | Node schema + query strategies for multi-agent systems |

```bash
# Commit everything
git add CLAUDE.md CODEBASE.md AGENTS.md kg.json kg.json.faiss kg.json.idx
git commit -m "Add repo2kg knowledge graph"
```

---

### Step 5 — Query it

```bash
# Keyword search — zero deps, instant
repo2kg query-lite "authentication" --kg kg.json
repo2kg query-lite "database connection" --kg kg.json --k 10 --depth 2

# Semantic search — more accurate, needs FAISS
repo2kg query "how does auth work" --kg kg.json
repo2kg query "payment processing flow" --kg kg.json --depth 2

# Structured JSON output (for scripts/agents)
repo2kg query-lite "auth" --kg kg.json --format json
```

Sample output:
```
# Query: authentication
# Nodes returned: 8  |  ~210 tokens
# Entry points: verify_token, JWTService, authenticate_user
# ─────────────────────────────────────────────

# METHOD: verify_token
# file: auth/jwt.py  class: JWTService
def JWTService.verify_token(self, token: str) -> dict:
"""Verify and decode a JWT token."""
    decoded = jwt.decode(token, self.secret, algorithms=["HS256"])
    return self.get_user_from_cache(decoded["sub"])
# calls: decode, get_user_from_cache
# called_by: authenticate_user, AuthMiddleware.process_request
```

---

## Tree-Sitter Parsing

repo2kg uses **tree-sitter** for all non-Python languages — installed automatically with `pip install repo2kg`. No extra steps needed.

| Advantage over regex | Detail |
|---|---|
| Multi-line signatures | Captured correctly |
| Nested classes | Exact parent class |
| Code in strings/comments | Ignored (no false matches) |
| Generic types `List<T>` | Handled correctly |
| Arrow functions in classes | Correct method detection |

Python always uses Python's own `ast` module. If tree-sitter fails for any file, repo2kg silently falls back to regex.

---

## TOON Format

[TOON (Token-Oriented Object Notation)](https://github.com/toon-format/toon) is a compact line-oriented format optimised for LLM context windows. It uses ~40% fewer tokens than JSON.

```bash
repo2kg build --repo . --out kg.toon     # Save as TOON
repo2kg query-lite "auth" --kg kg.toon   # Query TOON directly
```

Example TOON output for a single node:
```
nodes[788]:
  -
    id: "src/auth.ts::AuthService"
    name: AuthService
    kind: class
    file: src/auth.ts
    signature: export class AuthService
    docstring: Handles JWT authentication
    calls[2]: verifyToken,refreshToken
    callers[0]:
```

All commands (`build`, `query`, `query-lite`, `export`, `agent-setup`, `stats`) work with `.toon` files identically to `.json`.

---

## How Agents Use It (Zero Dependencies)

The global `~/.claude/CLAUDE.md` (installed by `user-setup`) tells the agent to run this at the start of every task:

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

# Search — no FAISS, no embeddings, pure stdlib
matches = [n for n in kg.values() if "auth" in n["name"].lower()]
for n in matches[:10]:
    print(n["signature"], "—", n.get("docstring", "")[:100])
    print("calls:", [c.split("::")[-1] for c in n.get("calls", [])])
```

The agent only reads actual source files when the 8-line `body_preview` is not enough.

---

## CLI Reference

```
repo2kg build        Build KG from a repository
repo2kg query        Semantic search (requires FAISS)
repo2kg query-lite   Keyword search (zero dependencies)
repo2kg export       Export as CODEBASE.md
repo2kg visualize    Generate interactive HTML graph (no browser server needed)
repo2kg agent-setup  Generate all agent instruction files
repo2kg user-setup   Install global agent instructions (run once)
repo2kg scan         Auto-discover and register all KGs under a directory
repo2kg register     Register a single project manually
repo2kg list         Show all registered projects
repo2kg stats        Show KG node/edge statistics
repo2kg info         Print machine-readable tool info (for agents)
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--repo` | `.` | Repository root to scan |
| `--out` (visualize) | `kg_graph.html` | Output HTML path for `visualize` |
| `--max-nodes` | `800` | Max symbol nodes in the HTML graph |
| `--out` | `kg.json` | Output path (`.json` or `.toon`) |
| `--kg` | `kg.json` | Path to a saved KG file |
| `--k` | `5` | Number of top results |
| `--depth` | `1` | Graph traversal depth (1 = direct calls, 2 = calls of calls) |
| `--format` | `text` | Output format: `text` or `json` |
| `--dir` | `.` | Target directory for `agent-setup` |
| `--root` | `~` | Root directory for `scan` |
| `-v` | off | Verbose/debug logging |

### Auto-excluded directories

```
__pycache__  .git  .hg  .svn  .tox  .nox  .mypy_cache  .pytest_cache
node_modules  vendor  bower_components  site-packages
dist  build  out  coverage  .next  .nuxt  .output  .vite  __sapper__
venv  .venv  env  .env
```

### Auto-excluded file patterns

Files matching these patterns are skipped even inside user source directories:

```
*.min.js   *.min.css   *.bundle.js   *.chunk.js
*.generated.*  chunk-*.js   *_pb2.py   *.pb.go
*.g.dart   *.freezed.dart
```

Add more: `repo2kg build --repo . --exclude migrations fixtures`

---

## Python API

```python
# Full mode (FAISS + sentence-transformers)
from repo2kg import RepoKG

kg = RepoKG().build("./my_project")
kg.save("kg.json")                                # JSON
kg.save("kg.toon")                               # TOON (fewer tokens)

kg = RepoKG.load("kg.json")
print(kg.query("auth flow", k=5, depth=2))        # text output
data = kg.query_json("payment", k=3)              # structured dict
```

```python
# Lightweight mode — stdlib only, safe for agents
from repo2kg import RepoKGLite

kg = RepoKGLite("kg.json")
print(kg.query("auth", k=5, depth=1))             # text output
result = kg.query_json("database", k=3)           # structured dict
callers = kg.get_callers("auth/service.py::AuthService.login")
callees = kg.get_callees("auth/service.py::AuthService.login")
```

---

## Node Schema

Every node in `kg.json` / `kg.toon`:

```json
{
  "auth/jwt.py::JWTService.verify_token": {
    "id":           "auth/jwt.py::JWTService.verify_token",
    "name":         "verify_token",
    "kind":         "method",
    "file":         "auth/jwt.py",
    "parent_class": "JWTService",
    "signature":    "def JWTService.verify_token(self, token: str) -> dict:",
    "docstring":    "Verify and decode a JWT token.",
    "body_preview": "    decoded = jwt.decode(token, self.secret, ...)\n    ...",
    "calls":        ["auth/jwt.py::JWTService.decode", "cache/redis.py::get_user"],
    "callers":      ["auth/service.py::authenticate_user"],
    "imports":      ["jwt", "datetime"]
  }
}
```

---

## Supported Languages

| Language | Extensions | Parser |
|---|---|---|
| Python | `.py` | `ast` module (always accurate) |
| JavaScript | `.js` `.jsx` `.mjs` | tree-sitter / regex |
| TypeScript | `.ts` `.tsx` `.mts` | tree-sitter / regex |
| Java | `.java` | tree-sitter / regex |
| Go | `.go` | tree-sitter / regex |
| Rust | `.rs` | tree-sitter / regex |
| C | `.c` `.h` | tree-sitter / regex |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` | tree-sitter / regex |
| Ruby | `.rb` | tree-sitter / regex |
| C# | `.cs` | tree-sitter / regex |

---

## Interactive Graph Visualization

Generate a fully self-contained HTML file — no server, no CDN, works offline:

```bash
repo2kg visualize --kg kg.json --out kg_graph.html
# open kg_graph.html in any browser
```

### What you see

The graph uses a **file-hub layout**:

- **Large indigo circles** = your source files, sized by how many symbols they contain
- **Small pill nodes** = functions, classes, and methods inside each file
- **Dashed lines** = containment (file → its symbols)
- **Solid arrows** = call edges between symbols

### What's included

Only **user-written code** is rendered. Third-party and bundled files are automatically excluded from the HTML graph (but remain in the KG file for queries):

| Excluded path pattern | Example |
|---|---|
| `.vite/deps/` | Vite-bundled npm packages |
| `node_modules/` | Raw npm packages |
| `site-packages/` | Python pip packages |
| `/dist/` | Build output |
| `/.cache/` | Build caches |

This typically reduces a 1500-node cluttered graph down to just your 100–300 meaningful nodes.

### Interactions

| Action | Result |
|---|---|
| Click a **file node** | Sidebar lists all its classes, functions, methods |
| Click a **symbol node** | Sidebar shows signature, docstring, calls, callers |
| Toggle **Classes / Functions / Methods** | Show/hide node kinds |
| Search box | Highlights matching nodes across files |
| **Fit** button | Auto-zooms to show the full graph |
| Drag any node | Pin it in place |

---

## Limitations

- **Static analysis only** — captures defined calls, not dynamic dispatch or runtime-generated calls
- **Name-based call resolution** — same-file preferred; no full type inference
- **No incremental updates** — rebuilds the full graph each time (30–60s for large repos)
- **Visualization shows user code only** — third-party files (`.vite/deps/`, `node_modules/`, `site-packages/`) are kept in the KG file but excluded from the HTML graph

---

## Contributing

Contributions welcome! Open an issue first for major changes.

```bash
git clone https://github.com/mugenGH/repo2kg.git
cd repo2kg
pip install -e .
```

---

## License

[MIT](LICENSE)
