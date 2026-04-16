"""
repo2kg — Repository Knowledge Graph for token-efficient agent context.

Architecture:
  Layer 1 (KG)  — nodes=functions/classes, edges=CALLS/IMPORTS/DEFINED_IN
  Layer 2 (RAG) — FAISS over rich node summaries for semantic entry-point search
  Output        — token-minimal structured context for LLM agents

Usage:
  repo2kg build --repo ./myrepo --out kg.json
  repo2kg query "how does authentication work" --kg kg.json --depth 2
  repo2kg stats --kg kg.json

Agent-friendly commands (no FAISS/embeddings needed):
  repo2kg export --kg kg.json --out CODEBASE.md      # standalone markdown for agents
  repo2kg agent-setup --kg kg.json --dir ./myrepo     # generate CLAUDE.md + .copilot-instructions.md
  repo2kg query-lite "auth" --kg kg.json              # keyword search, zero heavy deps

User-level (global, across all projects):
  repo2kg user-setup                                  # install global agent instructions once
  repo2kg scan                                        # auto-discover ALL KGs under home dir
  repo2kg scan --root /path/to/projects               # scan a specific root
  repo2kg register --kg kg.json --project .           # register one project manually
  repo2kg list                                        # show all registered projects
"""

__version__ = "0.3.0"

import os
import sys
import ast
import json
import argparse
import logging
import textwrap
import fnmatch
from pathlib import Path

import numpy as np
import faiss
from dataclasses import dataclass, field, asdict
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("repo2kg")

REPO2KG_HOME = Path.home() / ".repo2kg"
REGISTRY_PATH = REPO2KG_HOME / "registry.json"

DEFAULT_EXCLUDE = [
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.egg-info",
    "venv", ".venv", "env", ".env",
    "dist", "build", "site-packages",
]


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class CodeNode:
    id: str                          # unique: "file::ClassName.method" or "file::func"
    name: str
    kind: str                        # "function" | "method" | "class"
    file: str
    parent_class: Optional[str]
    signature: str
    docstring: str
    body_preview: str                # first 8 lines — enough context, minimal tokens
    calls: list[str] = field(default_factory=list)   # node ids this node calls
    callers: list[str] = field(default_factory=list) # node ids that call this node
    imports: list[str] = field(default_factory=list) # raw import strings in this file

    def summary(self) -> str:
        """Rich text fed to the embedding model."""
        parts = [f"{self.kind} {self.name}"]
        if self.parent_class:
            parts.append(f"in class {self.parent_class}")
        parts.append(f"in file {self.file}")
        if self.docstring:
            parts.append(self.docstring)
        parts.append(self.signature)
        return " | ".join(parts)

    def render(self, include_body: bool = True) -> str:
        """Token-minimal text returned to the agent."""
        lines = [
            f"# {self.kind.upper()}: {self.name}",
            f"# file: {self.file}" + (f"  class: {self.parent_class}" if self.parent_class else ""),
            self.signature,
        ]
        if self.docstring:
            lines.append(f'"""{self.docstring}"""')
        if include_body and self.body_preview:
            lines.append(self.body_preview)
        if self.calls:
            lines.append(f"# calls: {', '.join(c.split('::')[-1] for c in self.calls)}")
        if self.callers:
            lines.append(f"# called_by: {', '.join(c.split('::')[-1] for c in self.callers)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# AST PARSING
# ─────────────────────────────────────────────

def _get_source_lines(source_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = min(start + 8, len(source_lines))
    return textwrap.dedent("".join(source_lines[start:end])).rstrip()


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: Optional[str] = None) -> str:
    args = []
    for arg in node.args.args:
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        args.append(f"{arg.arg}{annotation}")
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    qualified = f"{class_name}.{node.name}" if class_name else node.name
    return f"{prefix} {qualified}({', '.join(args)}){ret}:"


def _collect_calls(func_node: ast.AST) -> list[str]:
    """Collect bare function/method names called inside this node."""
    calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return list(set(calls))


def _collect_imports(tree: ast.Module) -> list[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def parse_file(file_path: str, repo_root: str = "") -> list[CodeNode]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("Syntax error, skipping: %s", file_path)
        return []

    # Store relative path for portability
    if repo_root:
        file_rel = os.path.relpath(file_path, repo_root)
    else:
        file_rel = file_path

    source_lines = source.splitlines(keepends=True)
    imports = _collect_imports(tree)
    nodes: list[CodeNode] = []

    # Tag parents BEFORE walking so we can distinguish top-level functions
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]

    def make_node(func_node, class_name=None):
        node_id = f"{file_rel}::{class_name + '.' if class_name else ''}{func_node.name}"
        docstring = ast.get_docstring(func_node) or ""
        signature = _format_signature(func_node, class_name)
        preview = _get_source_lines(source_lines, func_node)
        raw_calls = _collect_calls(func_node)
        return CodeNode(
            id=node_id,
            name=func_node.name,
            kind="method" if class_name else "function",
            file=file_rel,
            parent_class=class_name,
            signature=signature,
            docstring=docstring,
            body_preview=preview,
            calls=raw_calls,   # resolved to ids in RepoKG.build()
            imports=imports,
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_node_id = f"{file_rel}::{node.name}"
            docstring = ast.get_docstring(node) or ""
            nodes.append(CodeNode(
                id=class_node_id,
                name=node.name,
                kind="class",
                file=file_rel,
                parent_class=None,
                signature=f"class {node.name}:",
                docstring=docstring,
                body_preview=_get_source_lines(source_lines, node),
                imports=imports,
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes.append(make_node(item, class_name=node.name))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # top-level only — skip methods already caught inside ClassDef
            parent = getattr(node, "_parent", None)
            if not isinstance(parent, ast.ClassDef):
                nodes.append(make_node(node))

    return nodes


# ─────────────────────────────────────────────
# KNOWLEDGE GRAPH
# ─────────────────────────────────────────────

class RepoKG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.nodes: dict[str, CodeNode] = {}
        self.model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatL2 | None = None
        self._index_ids: list[str] = []   # maps FAISS row → node id

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, repo_path: str, exclude: list[str] | None = None) -> "RepoKG":
        """Parse repo, build nodes, resolve call edges, build FAISS index."""
        repo_path = os.path.abspath(repo_path)
        exclude_patterns = exclude or DEFAULT_EXCLUDE
        logger.info("Scanning %s ...", repo_path)

        py_files = []
        for root, dirs, filenames in os.walk(repo_path):
            # Prune excluded directories in-place
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, pat) for pat in exclude_patterns)
            ]
            for f in filenames:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        if not py_files:
            logger.warning("No Python files found in %s", repo_path)
            print(f"Warning: No Python files found in {repo_path}")
            return self

        for fpath in py_files:
            for node in parse_file(fpath, repo_root=repo_path):
                self.nodes[node.id] = node

        if not self.nodes:
            logger.warning("No parseable functions/classes found")
            print("Warning: No parseable functions/classes found")
            return self

        self._resolve_edges()
        self._build_faiss()
        edge_count = sum(len(n.calls) for n in self.nodes.values())
        logger.info("KG ready: %d nodes, %d call edges from %d files",
                     len(self.nodes), edge_count, len(py_files))
        print(f"KG ready: {len(self.nodes)} nodes, {edge_count} call edges from {len(py_files)} files")
        return self

    def _resolve_edges(self):
        """Convert raw call names → node ids. Also populate callers."""
        name_index: dict[str, list[str]] = {}
        for node_id, node in self.nodes.items():
            name_index.setdefault(node.name, []).append(node_id)

        for node in self.nodes.values():
            resolved = []
            for raw_call in node.calls:
                if raw_call in name_index:
                    # prefer same-file match, else first match
                    matches = name_index[raw_call]
                    same_file = [m for m in matches if self.nodes[m].file == node.file]
                    target = same_file[0] if same_file else matches[0]
                    resolved.append(target)
            node.calls = resolved

        # populate callers (reverse edges)
        for node in self.nodes.values():
            for callee_id in node.calls:
                if callee_id in self.nodes:
                    self.nodes[callee_id].callers.append(node.id)

    def _build_faiss(self):
        ids = list(self.nodes.keys())
        summaries = [self.nodes[i].summary() for i in ids]
        embeddings = self.model.encode(summaries, show_progress_bar=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(np.array(embeddings))
        self._index_ids = ids

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str):
        if not self.nodes:
            logger.error("Nothing to save — KG is empty")
            print("Error: KG is empty, nothing to save")
            return
        data = {nid: asdict(n) for nid, n in self.nodes.items()}
        with open(path, "w") as f:
            json.dump(data, f)
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".idx", "w") as f:
            json.dump(self._index_ids, f)
        print(f"Saved to {path} ({len(self.nodes)} nodes)")

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "RepoKG":
        for required in [path, path + ".faiss", path + ".idx"]:
            if not os.path.exists(required):
                raise FileNotFoundError(f"Missing KG file: {required}")
        kg = cls(model_name)
        with open(path) as f:
            data = json.load(f)
        for nid, ndata in data.items():
            kg.nodes[nid] = CodeNode(**ndata)
        kg._index = faiss.read_index(path + ".faiss")
        with open(path + ".idx") as f:
            kg._index_ids = json.load(f)
        print(f"Loaded KG: {len(kg.nodes)} nodes")
        return kg

    # ── Query ──────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 5) -> list[CodeNode]:
        """Layer 2: FAISS semantic search → entry point nodes."""
        q_emb = self.model.encode([query])
        _, indices = self._index.search(np.array(q_emb), k)
        return [self.nodes[self._index_ids[i]] for i in indices[0] if i < len(self._index_ids)]

    def expand(self, node: CodeNode, depth: int = 1) -> set[str]:
        """Layer 1: KG traversal — follow CALLS + CALLERS edges up to depth."""
        visited: set[str] = set()
        frontier = {node.id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited or nid not in self.nodes:
                    continue
                visited.add(nid)
                n = self.nodes[nid]
                next_frontier.update(n.calls)
                next_frontier.update(n.callers)
            frontier = next_frontier - visited
        return visited

    def query(self, question: str, k: int = 5, depth: int = 1) -> str:
        """
        Full pipeline: semantic search → graph expansion → token-minimal output.
        This is what you pass to the agent instead of raw file contents.
        """
        entry_nodes = self.semantic_search(question, k=k)
        all_ids: set[str] = set()
        for node in entry_nodes:
            all_ids.add(node.id)
            all_ids.update(self.expand(node, depth=depth))

        # Render: entry nodes get body preview, expanded nodes get signatures only
        entry_ids = {n.id for n in entry_nodes}
        sections = []
        for nid in sorted(all_ids):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            sections.append(node.render(include_body=(nid in entry_ids)))

        token_estimate = sum(len(s.split()) for s in sections)
        header = (
            f"# Query: {question}\n"
            f"# Nodes returned: {len(all_ids)}  |  ~{token_estimate} tokens\n"
            f"# Entry points: {', '.join(n.name for n in entry_nodes)}\n"
            "# ─────────────────────────────────────────────\n"
        )
        return header + "\n\n".join(sections)

    def query_json(self, question: str, k: int = 5, depth: int = 1) -> dict:
        """
        Same as query() but returns structured JSON for programmatic agent consumption.
        """
        entry_nodes = self.semantic_search(question, k=k)
        all_ids: set[str] = set()
        for node in entry_nodes:
            all_ids.add(node.id)
            all_ids.update(self.expand(node, depth=depth))

        entry_ids = {n.id for n in entry_nodes}
        nodes_out = []
        for nid in sorted(all_ids):
            if nid not in self.nodes:
                continue
            n = self.nodes[nid]
            nodes_out.append({
                "id": n.id,
                "name": n.name,
                "kind": n.kind,
                "file": n.file,
                "parent_class": n.parent_class,
                "signature": n.signature,
                "docstring": n.docstring,
                "body_preview": n.body_preview if nid in entry_ids else None,
                "calls": [c.split("::")[-1] for c in n.calls],
                "called_by": [c.split("::")[-1] for c in n.callers],
                "is_entry_point": nid in entry_ids,
            })

        return {
            "query": question,
            "node_count": len(nodes_out),
            "token_estimate": sum(len(n["signature"].split()) for n in nodes_out),
            "entry_points": [n.name for n in entry_nodes],
            "nodes": nodes_out,
        }


# ─────────────────────────────────────────────
# LIGHTWEIGHT KG (no FAISS / no embeddings)
# ─────────────────────────────────────────────

class RepoKGLite:
    """
    Load a repo2kg JSON file and query it with keyword matching.
    Zero heavy dependencies — only stdlib json + re.
    This is what agents like Codex / Claude Code should use.
    """

    def __init__(self, path: str):
        import re
        self._re = re
        with open(path) as f:
            data = json.load(f)
        self.nodes: dict[str, CodeNode] = {}
        for nid, ndata in data.items():
            self.nodes[nid] = CodeNode(**ndata)
        self.path = path

    def search(self, keywords: str, k: int = 10) -> list[CodeNode]:
        """Keyword search over node names, docstrings, signatures, and file paths."""
        terms = [t.lower() for t in self._re.split(r"[\s,_]+", keywords) if len(t) > 1]
        scored: list[tuple[float, str]] = []
        for nid, node in self.nodes.items():
            text = f"{node.name} {node.docstring} {node.signature} {node.file} {node.parent_class or ''}".lower()
            score = sum(3 if t in node.name.lower() else 1 for t in terms if t in text)
            if score > 0:
                scored.append((score, nid))
        scored.sort(key=lambda x: -x[0])
        return [self.nodes[nid] for _, nid in scored[:k]]

    def get_callers(self, node_id: str) -> list[CodeNode]:
        if node_id in self.nodes:
            return [self.nodes[c] for c in self.nodes[node_id].callers if c in self.nodes]
        return []

    def get_callees(self, node_id: str) -> list[CodeNode]:
        if node_id in self.nodes:
            return [self.nodes[c] for c in self.nodes[node_id].calls if c in self.nodes]
        return []

    def expand(self, node_id: str, depth: int = 1) -> list[CodeNode]:
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_f: set[str] = set()
            for nid in frontier:
                if nid in visited or nid not in self.nodes:
                    continue
                visited.add(nid)
                n = self.nodes[nid]
                next_f.update(n.calls)
                next_f.update(n.callers)
            frontier = next_f - visited
        return [self.nodes[nid] for nid in sorted(visited) if nid in self.nodes]

    def query(self, keywords: str, k: int = 5, depth: int = 1) -> str:
        """Keyword search → graph expansion → agent-ready output (like RepoKG.query)."""
        entry_nodes = self.search(keywords, k=k)
        all_ids: set[str] = set()
        for node in entry_nodes:
            all_ids.add(node.id)
            expanded = self.expand(node.id, depth=depth)
            all_ids.update(n.id for n in expanded)

        entry_ids = {n.id for n in entry_nodes}
        sections = []
        for nid in sorted(all_ids):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            sections.append(node.render(include_body=(nid in entry_ids)))

        token_estimate = sum(len(s.split()) for s in sections)
        header = (
            f"# Query: {keywords}\n"
            f"# Nodes returned: {len(all_ids)}  |  ~{token_estimate} tokens\n"
            f"# Entry points: {', '.join(n.name for n in entry_nodes)}\n"
            "# ─────────────────────────────────────────────\n"
        )
        return header + "\n\n".join(sections)

    def query_json(self, keywords: str, k: int = 5, depth: int = 1) -> dict:
        """Keyword search → graph expansion → structured JSON for agents."""
        entry_nodes = self.search(keywords, k=k)
        all_ids: set[str] = set()
        for node in entry_nodes:
            all_ids.add(node.id)
            expanded = self.expand(node.id, depth=depth)
            all_ids.update(n.id for n in expanded)

        entry_ids = {n.id for n in entry_nodes}
        nodes_out = []
        for nid in sorted(all_ids):
            if nid not in self.nodes:
                continue
            n = self.nodes[nid]
            nodes_out.append({
                "id": n.id, "name": n.name, "kind": n.kind, "file": n.file,
                "parent_class": n.parent_class, "signature": n.signature,
                "docstring": n.docstring,
                "body_preview": n.body_preview if nid in entry_ids else None,
                "calls": [c.split("::")[-1] for c in n.calls],
                "called_by": [c.split("::")[-1] for c in n.callers],
                "is_entry_point": nid in entry_ids,
            })
        return {
            "query": keywords, "node_count": len(nodes_out),
            "entry_points": [n.name for n in entry_nodes], "nodes": nodes_out,
        }


# ─────────────────────────────────────────────
# EXPORT: Generate agent-readable files
# ─────────────────────────────────────────────

def export_codebase_md(kg_path: str, out_path: str):
    """Generate a standalone CODEBASE.md that any agent can read — no tools needed."""
    with open(kg_path) as f:
        data = json.load(f)
    nodes = {nid: CodeNode(**d) for nid, d in data.items()}

    files: dict[str, list[CodeNode]] = {}
    for n in nodes.values():
        files.setdefault(n.file, []).append(n)

    classes = [n for n in nodes.values() if n.kind == "class"]
    functions = [n for n in nodes.values() if n.kind == "function"]
    methods = [n for n in nodes.values() if n.kind == "method"]
    edge_count = sum(len(n.calls) for n in nodes.values())

    lines = [
        "# Codebase Knowledge Graph",
        "",
        "> Auto-generated by repo2kg. Read this file to understand the codebase",
        "> without reading every source file. Query the companion kg.json for deeper dives.",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Files | {len(files)} |",
        f"| Classes | {len(classes)} |",
        f"| Functions | {len(functions)} |",
        f"| Methods | {len(methods)} |",
        f"| Call Edges | {edge_count} |",
        f"| Total Nodes | {len(nodes)} |",
        "",
        "## File Map",
        "",
    ]

    for fpath in sorted(files.keys()):
        fnodes = files[fpath]
        node_names = ", ".join(
            f"`{n.name}`" + (f" ({n.kind})" if n.kind == "class" else "")
            for n in sorted(fnodes, key=lambda x: (x.kind != "class", x.name))
        )
        lines.append(f"- **{fpath}** — {node_names}")
    lines.append("")

    # Architecture: group by top-level directory
    dirs: dict[str, set[str]] = {}
    for fpath in files:
        top = fpath.split("/")[0] if "/" in fpath else "."
        dirs.setdefault(top, set()).add(fpath)

    if len(dirs) > 1:
        lines.extend(["## Architecture", ""])
        lines.append("```")
        for d in sorted(dirs.keys()):
            dfiles = sorted(dirs[d])
            lines.append(f"{d}/")
            for fp in dfiles:
                count = len(files[fp])
                lines.append(f"  {fp} ({count} nodes)")
        lines.append("```")
        lines.append("")

    # Key classes
    if classes:
        lines.extend(["## Classes", ""])
        for cls in sorted(classes, key=lambda c: c.name):
            lines.append(f"### `{cls.name}` ({cls.file})")
            if cls.docstring:
                lines.append(f"> {cls.docstring[:200]}")
            # Find methods belonging to this class
            cls_methods = [n for n in nodes.values() if n.parent_class == cls.name and n.file == cls.file]
            if cls_methods:
                lines.append("")
                lines.append("| Method | Signature |")
                lines.append("|--------|-----------|")
                for m in sorted(cls_methods, key=lambda x: x.name):
                    sig = m.signature.replace("|", "\\|")
                    lines.append(f"| `{m.name}` | `{sig}` |")
            lines.append("")

    # Top-level functions
    if functions:
        lines.extend(["## Functions", ""])
        lines.append("| Function | File | Description |")
        lines.append("|----------|------|-------------|")
        for fn in sorted(functions, key=lambda f: f.name):
            doc = (fn.docstring or "").split("\n")[0][:80].replace("|", "\\|")
            lines.append(f"| `{fn.name}` | {fn.file} | {doc} |")
        lines.append("")

    # Call graph (top connections)
    nodes_with_calls = [(n, len(n.calls) + len(n.callers)) for n in nodes.values() if n.calls or n.callers]
    nodes_with_calls.sort(key=lambda x: -x[1])
    if nodes_with_calls:
        lines.extend(["## Key Call Relationships", ""])
        lines.append("```")
        for n, _ in nodes_with_calls[:30]:
            if n.calls:
                callees = ", ".join(c.split("::")[-1] for c in n.calls[:5])
                lines.append(f"{n.name} → {callees}")
            if n.callers:
                callers = ", ".join(c.split("::")[-1] for c in n.callers[:5])
                lines.append(f"{n.name} ← {callers}")
        lines.append("```")
        lines.append("")

    # Entry points (nodes with many callers or no callers but many calls)
    entry_candidates = [n for n in nodes.values() if n.kind != "class" and not n.callers and n.calls]
    if entry_candidates:
        lines.extend(["## Likely Entry Points", ""])
        for n in sorted(entry_candidates, key=lambda x: -len(x.calls))[:15]:
            lines.append(f"- `{n.name}` in {n.file} — calls {len(n.calls)} functions")
        lines.append("")

    # Signatures reference (compact)
    lines.extend(["## All Signatures (compact reference)", ""])
    lines.append("```python")
    current_file = None
    for nid in sorted(nodes.keys()):
        n = nodes[nid]
        if n.file != current_file:
            current_file = n.file
            lines.append(f"\n# ── {current_file} ──")
        lines.append(n.signature)
    lines.append("```")
    lines.append("")

    content = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(content)
    token_estimate = len(content.split())
    print(f"Exported {out_path} ({len(nodes)} nodes, ~{token_estimate} tokens)")


def generate_agent_instructions(kg_path: str, target_dir: str):
    """Generate CLAUDE.md and .copilot-instructions.md so agents auto-discover the KG."""
    with open(kg_path) as f:
        data = json.load(f)
    node_count = len(data)
    files = set()
    for ndata in data.values():
        files.add(ndata.get("file", ""))
    edge_count = sum(len(ndata.get("calls", [])) for ndata in data.values())

    kg_basename = os.path.basename(kg_path)
    kg_rel = os.path.relpath(kg_path, target_dir) if os.path.isabs(kg_path) else kg_path

    # ── CLAUDE.md ──────────────────────────────────────────────────────
    claude_md = f"""# CLAUDE.md — Agent Instructions for this Repository

## Knowledge Graph Available

This repository has a pre-built code knowledge graph at `{kg_rel}`.
**Use it instead of reading source files directly.** It contains {node_count} nodes
and {edge_count} call edges across {len(files)} files.

## How to Use the Knowledge Graph

### Option 1: Read the exported summary (fastest, zero deps)
```bash
cat CODEBASE.md
```
This file contains the full architecture overview, file map, class/function index,
call graph, and all signatures. It's usually enough to answer most questions.

### Option 2: Lightweight keyword query (Python stdlib only)
```bash
python3 -c "
import json
data = json.load(open('{kg_rel}'))
# Search by keyword
matches = {{k: v for k, v in data.items() if 'YOUR_KEYWORD' in k.lower() or 'YOUR_KEYWORD' in v.get('docstring','').lower() or 'YOUR_KEYWORD' in v.get('name','').lower()}}
for node_id, node in list(matches.items())[:10]:
    print(f\\"\\n## {{node['kind'].upper()}}: {{node['name']}}\\")
    print(f\\"File: {{node['file']}}\\")
    print(node['signature'])
    if node.get('docstring'): print(node['docstring'][:200])
    if node.get('calls'): print(f\\"Calls: {{', '.join(c.split(\\\\\"::\\\\\")[-1] for c in node['calls'][:8])}}\\")
"
```

### Option 3: repo2kg CLI (if installed)
```bash
repo2kg query "how does authentication work" --kg {kg_rel}
repo2kg query-lite "auth" --kg {kg_rel}    # no FAISS needed
```

### Option 4: Structured JSON output (for programmatic use)
```bash
repo2kg query "auth" --kg {kg_rel} --format json
```

## Rules

1. **Before reading any source file**, check if the KG already has what you need
2. Only read actual source files when you need the full implementation (>8 lines)
3. The KG includes 8-line body previews — often sufficient for understanding logic
4. Use call edges to trace dependencies instead of grepping across files
5. The `CODEBASE.md` file is the fastest way to get a project overview

## Project Structure

- `{kg_rel}` — Knowledge graph (JSON, {node_count} nodes)
- `CODEBASE.md` — Agent-readable project overview (generated from KG)
"""

    # ── .copilot-instructions.md ──────────────────────────────────────
    copilot_md = f"""# Copilot Instructions

## Code Knowledge Graph

A pre-built knowledge graph is available at `{kg_rel}` with {node_count} code nodes
({len(files)} files, {edge_count} call edges).

### Quick lookup pattern (Python, zero deps):
```python
import json
kg = json.load(open("{kg_rel}"))
# Find nodes by name/keyword:
for nid, n in kg.items():
    if "KEYWORD" in nid.lower() or "KEYWORD" in n.get("name","").lower():
        print(n["signature"], "—", n.get("docstring","")[:100])
        print("calls:", [c.split("::")[-1] for c in n.get("calls",[])])
```

### Full overview: Read `CODEBASE.md` for architecture, file map, and all signatures.

### Best practices:
- Use the KG to find relevant code before reading source files
- Follow call edges (`calls` / `callers` fields) to trace execution flow
- Body previews (first 8 lines) are in `body_preview` field
"""

    # ── .github/copilot-instructions.md  (for Copilot agent mode) ─────
    copilot_agent_md = f"""# Copilot Agent Instructions

This project has a code knowledge graph. Before exploring source files:

1. Read `CODEBASE.md` for a full project overview
2. Query `{kg_rel}` for specific code details:
   ```python
   import json; kg = json.load(open("{kg_rel}"))
   matches = [n for n in kg.values() if "KEYWORD" in n["name"].lower()]
   ```
3. Only read source files when you need implementation details beyond the 8-line preview
"""

    # Write files
    os.makedirs(target_dir, exist_ok=True)

    claude_path = os.path.join(target_dir, "CLAUDE.md")
    with open(claude_path, "w") as f:
        f.write(claude_md)
    print(f"Created {claude_path}")

    copilot_path = os.path.join(target_dir, ".copilot-instructions.md")
    with open(copilot_path, "w") as f:
        f.write(copilot_md)
    print(f"Created {copilot_path}")

    github_dir = os.path.join(target_dir, ".github")
    os.makedirs(github_dir, exist_ok=True)
    copilot_agent_path = os.path.join(github_dir, "copilot-instructions.md")
    with open(copilot_agent_path, "w") as f:
        f.write(copilot_agent_md)
    print(f"Created {copilot_agent_path}")

    # ── .agents.md (for multi-agent systems) ─────────────────────────
    agents_md = f"""# Agent Configuration

## Available Tools

### Knowledge Graph Query
- **File**: `{kg_rel}`
- **Format**: JSON dictionary of code nodes
- **Nodes**: {node_count} (classes, functions, methods)
- **Edges**: {edge_count} call relationships
- **Coverage**: {len(files)} source files

### Node Schema
```json
{{
  "file_path::ClassName.method_name": {{
    "id": "file_path::ClassName.method_name",
    "name": "method_name",
    "kind": "method|function|class",
    "file": "relative/path.py",
    "parent_class": "ClassName or null",
    "signature": "def method_name(self, arg: type) -> type:",
    "docstring": "Method description",
    "body_preview": "First 8 lines of implementation",
    "calls": ["other_node_id", "..."],
    "callers": ["calling_node_id", "..."],
    "imports": ["module.name", "..."]
  }}
}}
```

### Query Strategies
1. **By name**: Filter nodes where `name` matches your target
2. **By file**: Filter nodes where `file` matches a path
3. **By relationship**: Follow `calls`/`callers` to understand execution flow
4. **By kind**: Filter `kind == "class"` for architecture overview
5. **By docstring**: Search `docstring` field for domain concepts
"""
    agents_path = os.path.join(target_dir, "AGENTS.md")
    with open(agents_path, "w") as f:
        f.write(agents_md)
    print(f"Created {agents_path}")


# ─────────────────────────────────────────────
# GLOBAL REGISTRY
# ─────────────────────────────────────────────

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"version": 1, "projects": {}}


def _save_registry(reg: dict):
    REPO2KG_HOME.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def register_project(kg_path: str, project_dir: str, *, silent: bool = False):
    """Register a project + KG path in the global ~/.repo2kg/registry.json."""
    import datetime
    project_abs = str(Path(project_dir).resolve())
    kg_abs = str(Path(kg_path).resolve())
    reg = _load_registry()
    reg["projects"][project_abs] = {
        "kg": kg_abs,
        "name": Path(project_abs).name,
        "registered_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _save_registry(reg)
    if not silent:
        print(f"Registered: {project_abs}")
        print(f"         KG: {kg_abs}")


def scan_and_register(root: str, kg_patterns: list[str] | None = None):
    """
    Walk `root`, find every kg.json / *_kg.json file, infer the project dir,
    and register them all in the global registry in one shot.
    """
    import datetime
    patterns = kg_patterns or ["kg.json", "*_kg.json"]
    root_path = Path(root).resolve()

    # Directories to never descend into
    skip_dirs = {
        "__pycache__", ".git", ".hg", ".svn", "node_modules",
        ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "venv", ".venv", "env", ".env", "dist", "build", "site-packages",
        ".repo2kg",
    }

    found: list[tuple[Path, Path]] = []  # (project_dir, kg_path)

    for dirpath, dirs, filenames in os.walk(root_path):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            # Skip FAISS / index sidecars
            fpath = Path(dirpath) / fname
            if fpath.suffix != ".json":
                continue
            if (fpath.parent / (fname + ".faiss")).exists():
                # This is a KG file (has a companion .faiss)
                project_dir = fpath.parent
                found.append((project_dir, fpath))

    if not found:
        print(f"No KG files found under {root_path}")
        print("Tip: run 'repo2kg build --repo . --out kg.json' first in your project")
        return

    reg = _load_registry()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    new_count = 0
    update_count = 0

    for project_dir, kg_path in found:
        project_abs = str(project_dir)
        kg_abs = str(kg_path)
        is_new = project_abs not in reg["projects"]
        reg["projects"][project_abs] = {
            "kg": kg_abs,
            "name": project_dir.name,
            "registered_at": now,
        }
        if is_new:
            new_count += 1
            print(f"  + {project_abs}")
            print(f"      KG: {kg_abs}")
        else:
            update_count += 1
            print(f"  ~ {project_abs} (updated)")

    _save_registry(reg)
    print(f"\nDone: {new_count} new, {update_count} updated — {len(reg['projects'])} total registered")


def generate_user_instructions():
    """
    Write global agent instruction files:
      ~/.claude/CLAUDE.md       — Claude Code global instructions
      ~/.codex/AGENTS.md        — Codex global instructions
      ~/.config/gh-copilot/instructions.md — Copilot global (if supported)
      ~/.repo2kg/GLOBAL_AGENTS.md — human-readable reference
    """
    registry_path_str = str(REGISTRY_PATH)

    # Shared lookup snippet both agents can copy-paste
    lookup_snippet = '''\
import json, os
from pathlib import Path

# Find KG for the current project
registry_path = Path.home() / ".repo2kg" / "registry.json"
if registry_path.exists():
    registry = json.load(open(registry_path))
    projects = registry.get("projects", {})
    cwd = str(Path.cwd())
    # Walk up directory tree to find closest registered project
    check = Path(cwd)
    kg_path = None
    while check != check.parent:
        if str(check) in projects:
            kg_path = projects[str(check)]["kg"]
            break
        check = check.parent
    if kg_path and Path(kg_path).exists():
        kg = json.load(open(kg_path))
        print(f"KG loaded: {len(kg)} nodes from {kg_path}")
        # Now search: replace KEYWORD with what you need
        matches = [n for n in kg.values() if "KEYWORD" in n["name"].lower()]
        for n in matches[:10]:
            print(n["signature"], "—", n.get("docstring","")[:100])
'''

    # ── ~/.claude/CLAUDE.md ─────────────────────────────────────────
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    claude_global = Path(claude_dir / "CLAUDE.md")

    existing_content = claude_global.read_text() if claude_global.exists() else ""
    marker = "<!-- repo2kg-start -->"
    end_marker = "<!-- repo2kg-end -->"
    repo2kg_block = f"""{marker}
## repo2kg — Code Knowledge Graph

A global project registry is at `{registry_path_str}`.
For any Python project you open, check if it has a registered knowledge graph:

```python
{lookup_snippet}
```

### Workflow for every task:
1. Run the lookup above to find the project's KG
2. If a KG exists, read `CODEBASE.md` (project dir) for architecture overview
3. Query `kg.json` with the snippet above instead of reading source files
4. Only read actual source files when the 8-line body preview is not enough
5. Follow `calls`/`callers` fields to trace dependencies

### Node schema (what's in kg.json)
```
key: "relative/file.py::ClassName.method"
fields: id, name, kind, file, parent_class, signature,
        docstring, body_preview (8 lines), calls[], callers[], imports[]
```
{end_marker}
"""

    if marker in existing_content:
        # Replace existing block
        import re
        new_content = re.sub(
            rf"{re.escape(marker)}.*?{re.escape(end_marker)}",
            repo2kg_block.strip(),
            existing_content,
            flags=re.DOTALL,
        )
    else:
        new_content = existing_content.rstrip() + ("\n\n" if existing_content else "") + repo2kg_block

    claude_global.write_text(new_content)
    print(f"Updated {claude_global}")

    # ── ~/.codex/AGENTS.md ──────────────────────────────────────────
    codex_dir = Path.home() / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    codex_agents = codex_dir / "AGENTS.md"
    existing_codex = codex_agents.read_text() if codex_agents.exists() else ""
    codex_block = f"""{marker}
# repo2kg — Code Knowledge Graph

Registry: `{registry_path_str}`

For any task on a Python project, start with:
```python
{lookup_snippet}
```

Then query the returned `kg` dict instead of reading source files.
Node fields: `name`, `kind`, `file`, `signature`, `docstring`, `body_preview`, `calls`, `callers`.
{end_marker}
"""
    if marker in existing_codex:
        import re
        new_codex = re.sub(
            rf"{re.escape(marker)}.*?{re.escape(end_marker)}",
            codex_block.strip(),
            existing_codex,
            flags=re.DOTALL,
        )
    else:
        new_codex = existing_codex.rstrip() + ("\n\n" if existing_codex else "") + codex_block
    codex_agents.write_text(new_codex)
    print(f"Updated {codex_agents}")

    # ── ~/.repo2kg/GLOBAL_AGENTS.md ─────────────────────────────────
    REPO2KG_HOME.mkdir(parents=True, exist_ok=True)
    global_ref = REPO2KG_HOME / "GLOBAL_AGENTS.md"
    global_ref.write_text(f"""# repo2kg Global Agent Reference

## Registry
`{registry_path_str}` — maps project paths to their KG files.

## Commands
```bash
# Register a project (run in the project dir after building the KG)
repo2kg register --kg kg.json --project .

# Re-run user-setup after upgrading repo2kg
repo2kg user-setup

# See all registered projects
repo2kg list

# Full per-project setup (also auto-registers)
repo2kg agent-setup --kg kg.json --dir .
```

## Lookup snippet (Python stdlib)
```python
{lookup_snippet}
```

## Registry format
```json
{{
  "version": 1,
  "projects": {{
    "/absolute/path/to/project": {{
      "kg": "/absolute/path/to/kg.json",
      "name": "project-name",
      "registered_at": "2026-04-16T12:00:00"
    }}
  }}
}}
```
""")
    print(f"Updated {global_ref}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="repo2kg — Build a code Knowledge Graph for token-efficient AI agent context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"repo2kg {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="cmd")

    # ── build ──
    build_p = sub.add_parser("build", help="Build KG from a repository")
    build_p.add_argument("--repo", default=".", help="Path to repo root (default: current dir)")
    build_p.add_argument("--out", default="kg.json", help="Output path for KG (default: kg.json)")
    build_p.add_argument("--exclude", nargs="*", help="Additional directory patterns to exclude")

    # ── query (semantic, requires FAISS) ──
    query_p = sub.add_parser("query", help="Query the KG with natural language (requires FAISS)")
    query_p.add_argument("question", help="Natural language question")
    query_p.add_argument("--kg", default="kg.json", help="Path to saved KG (default: kg.json)")
    query_p.add_argument("--k", type=int, default=5, help="Semantic search top-k (default: 5)")
    query_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    query_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json (structured)")

    # ── query-lite (keyword, no heavy deps) ──
    qlite_p = sub.add_parser("query-lite",
                             help="Query KG with keyword matching (no FAISS/embeddings needed)")
    qlite_p.add_argument("keywords", help="Keywords to search for")
    qlite_p.add_argument("--kg", default="kg.json", help="Path to saved KG JSON")
    qlite_p.add_argument("--k", type=int, default=5, help="Top-k results (default: 5)")
    qlite_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    qlite_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json")

    # ── export (generate CODEBASE.md) ──
    export_p = sub.add_parser("export", help="Export KG as agent-readable CODEBASE.md")
    export_p.add_argument("--kg", default="kg.json", help="Path to saved KG JSON")
    export_p.add_argument("--out", default="CODEBASE.md", help="Output markdown file (default: CODEBASE.md)")

    # ── agent-setup (generate CLAUDE.md, .copilot-instructions.md, etc.) ──
    agent_p = sub.add_parser("agent-setup",
                             help="Generate agent instruction files (CLAUDE.md, .copilot-instructions.md, etc.)")
    agent_p.add_argument("--kg", default="kg.json", help="Path to saved KG JSON")
    agent_p.add_argument("--dir", default=".", help="Target directory for agent files (default: current dir)")
    agent_p.add_argument("--no-register", action="store_true",
                         help="Skip registering in global ~/.repo2kg/registry.json")

    # ── user-setup (global agent instructions) ──
    sub.add_parser("user-setup",
                   help="Install global agent instructions (~/.claude/CLAUDE.md, ~/.codex/AGENTS.md, etc.)")

    # ── register (add project to global registry) ──
    reg_p = sub.add_parser("register",
                           help="Register a project KG in the global ~/.repo2kg/registry.json")
    reg_p.add_argument("--kg", default="kg.json", help="Path to saved KG JSON")
    reg_p.add_argument("--project", default=".",
                       help="Project root directory to register (default: current dir)")

    # ── scan (auto-discover and register all KGs under a root) ──
    scan_p = sub.add_parser("scan",
                            help="Scan a directory tree, find all KG files, and register them globally")
    scan_p.add_argument("--root", default=str(Path.home()),
                        help="Root directory to scan (default: home directory)")

    # ── list (show registered projects) ──
    sub.add_parser("list", help="List all projects registered in the global registry")

    # ── stats ──
    stats_p = sub.add_parser("stats", help="Show KG statistics")
    stats_p.add_argument("--kg", default="kg.json", help="Path to saved KG (default: kg.json)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.cmd == "build":
        exclude = DEFAULT_EXCLUDE + (args.exclude or [])
        kg = RepoKG().build(args.repo, exclude=exclude)
        kg.save(args.out)

    elif args.cmd == "query":
        kg = RepoKG.load(args.kg)
        if args.format == "json":
            result = kg.query_json(args.question, k=args.k, depth=args.depth)
            print(json.dumps(result, indent=2))
        else:
            result = kg.query(args.question, k=args.k, depth=args.depth)
            print(result)

    elif args.cmd == "query-lite":
        kg_lite = RepoKGLite(args.kg)
        if args.format == "json":
            result = kg_lite.query_json(args.keywords, k=args.k, depth=args.depth)
            print(json.dumps(result, indent=2))
        else:
            result = kg_lite.query(args.keywords, k=args.k, depth=args.depth)
            print(result)

    elif args.cmd == "export":
        export_codebase_md(args.kg, args.out)

    elif args.cmd == "agent-setup":
        generate_agent_instructions(args.kg, args.dir)
        codebase_path = os.path.join(args.dir, "CODEBASE.md")
        export_codebase_md(args.kg, codebase_path)
        if not args.no_register:
            register_project(args.kg, args.dir)
        print(f"\nAgent setup complete. Generated files in {os.path.abspath(args.dir)}:")
        print(f"  CLAUDE.md               — Claude Code / Claude agent instructions")
        print(f"  .copilot-instructions.md — GitHub Copilot instructions")
        print(f"  .github/copilot-instructions.md — Copilot agent mode instructions")
        print(f"  AGENTS.md               — Multi-agent system configuration")
        print(f"  CODEBASE.md             — Full project overview (agent-readable)")
        if not args.no_register:
            print(f"  ~/.repo2kg/registry.json — Project registered globally")

    elif args.cmd == "user-setup":
        generate_user_instructions()
        print("\nGlobal agent setup complete:")
        print("  ~/.claude/CLAUDE.md          — Claude Code global instructions")
        print("  ~/.codex/AGENTS.md           — Codex global instructions")
        print("  ~/.repo2kg/GLOBAL_AGENTS.md  — Reference for all agents")
        print("\nNext: register your projects with:")
        print("  cd /your/project && repo2kg register --kg kg.json")

    elif args.cmd == "register":
        register_project(args.kg, args.project)

    elif args.cmd == "scan":
        print(f"Scanning {args.root} for KG files...")
        scan_and_register(args.root)

    elif args.cmd == "list":
        reg = _load_registry()
        projects = reg.get("projects", {})
        if not projects:
            print("No projects registered. Run: repo2kg register --kg kg.json --project .")
        else:
            print(f"Registered projects ({len(projects)}):")
            print(f"  {'Project':<50} {'KG':<40} {'Registered'}")
            print(f"  {'-'*50} {'-'*40} {'-'*19}")
            for path, info in sorted(projects.items()):
                kg_exists = "✓" if Path(info["kg"]).exists() else "✗ missing"
                print(f"  {path:<50} {kg_exists} {info['registered_at']}")

    elif args.cmd == "stats":
        kg = RepoKG.load(args.kg)
        files = set(n.file for n in kg.nodes.values())
        classes = sum(1 for n in kg.nodes.values() if n.kind == "class")
        functions = sum(1 for n in kg.nodes.values() if n.kind == "function")
        methods = sum(1 for n in kg.nodes.values() if n.kind == "method")
        edges = sum(len(n.calls) for n in kg.nodes.values())
        print(f"Knowledge Graph Statistics:")
        print(f"  Files:     {len(files)}")
        print(f"  Nodes:     {len(kg.nodes)} ({classes} classes, {functions} functions, {methods} methods)")
        print(f"  Edges:     {edges} call edges")
        print(f"  Avg edges: {edges / max(len(kg.nodes), 1):.1f} per node")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
