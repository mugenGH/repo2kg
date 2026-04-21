"""
repo2kg — Repository Knowledge Graph for token-efficient agent context.

Architecture:
  Layer 1 (KG)  — nodes=functions/classes, edges=CALLS/IMPORTS/DEFINED_IN
  Layer 2 (RAG) — FAISS over rich node summaries for semantic entry-point search
  Output        — token-minimal structured context for LLM agents
  Formats       — JSON (default) or TOON (token-optimized)

Usage:
  repo2kg build --repo ./myrepo --out kg.json     # JSON output
  repo2kg build --repo ./myrepo --out kg.toon     # TOON output (40% fewer tokens)
  repo2kg query "how does authentication work" --kg kg.json --depth 2
  repo2kg stats --kg kg.json
  repo2kg info                                    # Machine-readable tool info (JSON)

Agent-friendly commands (no FAISS/embeddings needed):
  repo2kg export --kg kg.json --out CODEBASE.md      # standalone markdown for agents
  repo2kg agent-setup --kg kg.json --dir ./myrepo     # generate CLAUDE.md + .copilot-instructions.md
  repo2kg query-lite "auth" --kg kg.json              # keyword search, zero heavy deps
  repo2kg info                                        # JSON tool description for agent discovery

User-level (global, across all projects):
  repo2kg user-setup                                  # install global agent instructions once
  repo2kg scan                                        # auto-discover ALL KGs under home dir
  repo2kg scan --root /path/to/projects               # scan a specific root
  repo2kg register --kg kg.json --project .           # register one project manually
  repo2kg list                                        # show all registered projects

Format auto-detection:
  Format is determined by file extension: .json → JSON, .toon → TOON
  FAISS sidecar files (.faiss, .idx) use the base name regardless of format
"""

__version__ = "0.5.4"

import os
import sys
import ast
import json
import argparse
import logging
import textwrap
import fnmatch
import re as _re
from pathlib import Path

from dataclasses import dataclass, field, asdict
from typing import Optional

# Heavy imports are lazy-loaded to keep --help and info fast
_numpy = None
_faiss = None
_SentenceTransformer = None


def _load_heavy_deps():
    """Lazy-load numpy, faiss, and sentence-transformers."""
    global _numpy, _faiss, _SentenceTransformer
    if _numpy is None:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        _numpy = np
        _faiss = faiss
        _SentenceTransformer = SentenceTransformer

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

# Marker pair used to identify repo2kg-managed sections in shared files.
# Content between these markers is replaced on re-run; content outside is preserved.
_REPO2KG_MARKER = "<!-- repo2kg-start -->"
_REPO2KG_END_MARKER = "<!-- repo2kg-end -->"

CONFIG_PATH = REPO2KG_HOME / "config.json"


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
def _detect_format(path: str) -> str:
    """Detect KG file format from extension."""
    if path.endswith(".toon"):
        return "toon"
    return "json"


# ─────────────────────────────────────────────
# TOON SERIALIZATION (Token-Oriented Object Notation)
# Spec: https://github.com/toon-format/spec v3.0
# ~40% fewer tokens than JSON for LLM context
# ─────────────────────────────────────────────

def _toon_needs_quoting(s: str) -> bool:
    """Check if a TOON string value needs quoting."""
    if not s:
        return False
    if s[0] in (' ', '\t', '"') or s[-1] in (' ', '\t', '"'):
        return True
    return any(c in s for c in (',', ':', '[', ']', '{', '}', '"', '\\', '\n', '\r', '\t'))


def _toon_quote(s: str) -> str:
    """Quote and escape a string for TOON format."""
    if not s:
        return ''
    if not _toon_needs_quoting(s):
        return s
    escaped = (s.replace('\\', '\\\\')
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('\t', '\\t'))
    return f'"{escaped}"'


def _toon_unquote(s: str) -> str:
    """Unquote and unescape a TOON string value."""
    s = s.strip()
    if not s:
        return ''
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
        s = (s.replace('\\n', '\n')
              .replace('\\r', '\r')
              .replace('\\t', '\t')
              .replace('\\"', '"')
              .replace('\\\\', '\\'))
    return s


def _split_toon_array(s: str) -> list[str]:
    """Split a TOON inline array respecting quoted strings."""
    items = []
    current: list[str] = []
    in_quote = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_quote and i + 1 < len(s):
            current.append(c)
            current.append(s[i + 1])
            i += 2
            continue
        if c == '"':
            in_quote = not in_quote
            current.append(c)
        elif c == ',' and not in_quote:
            items.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        rest = ''.join(current).strip()
        if rest:
            items.append(rest)
    return items


def serialize_toon(data: dict[str, dict]) -> str:
    """Serialize KG node dict to TOON format (Token-Oriented Object Notation)."""
    lines = [
        "# repo2kg Knowledge Graph",
        f"# Format: TOON v3.0 (Token-Oriented Object Notation)",
        f"# Generated by repo2kg v{__version__}",
        f"# Nodes: {len(data)}",
        "",
        f"nodes[{len(data)}]:",
    ]
    for node_data in data.values():
        lines.append("  -")
        for key, val in node_data.items():
            if val is None:
                lines.append(f"    {key}:")
            elif isinstance(val, list):
                if val:
                    items = ",".join(_toon_quote(str(v)) for v in val)
                    lines.append(f"    {key}[{len(val)}]: {items}")
                else:
                    lines.append(f"    {key}[0]:")
            else:
                lines.append(f"    {key}: {_toon_quote(str(val))}")
    lines.append("")
    return "\n".join(lines)


def deserialize_toon(text: str) -> dict[str, dict]:
    """Deserialize a TOON KG file back to dict-of-dicts keyed by node id."""
    result: dict[str, dict] = {}
    current_node: dict | None = None

    for line in text.splitlines():
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith('#'):
            continue

        # Skip array header
        if stripped.startswith('nodes['):
            continue

        # New node delimiter
        if stripped == '-':
            if current_node and 'id' in current_node:
                result[current_node['id']] = current_node
            current_node = {}
            continue

        if current_node is None:
            continue

        # Array field: key[N]: values
        array_match = _re.match(r'(\w+)\[(\d+)\]:\s*(.*)', stripped)
        if array_match:
            key = array_match.group(1)
            size = int(array_match.group(2))
            val_str = array_match.group(3).strip()
            if size == 0 or not val_str:
                current_node[key] = []
            else:
                items = _split_toon_array(val_str)
                current_node[key] = [_toon_unquote(item) for item in items]
            continue

        # Scalar field: key: value
        colon_idx = stripped.find(':')
        if colon_idx > 0:
            key = stripped[:colon_idx].strip()
            val_str = stripped[colon_idx + 1:].strip()
            if not val_str:
                current_node[key] = None
            else:
                current_node[key] = _toon_unquote(val_str)

    # Don't forget the last node
    if current_node and 'id' in current_node:
        result[current_node['id']] = current_node

    return result


# ─────────────────────────────────────────────
# MULTI-LANGUAGE SUPPORT
# ─────────────────────────────────────────────

LANG_EXTENSIONS: dict[str, str] = {
    '.py': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript', '.mts': 'typescript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c', '.h': 'c',
    '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp',
    '.rb': 'ruby',
    '.cs': 'csharp',
}


def _read_source(file_path: str) -> str:
    """Read source file with fallback encoding."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _rel_path(file_path: str, repo_root: str) -> str:
    """Get relative path for node IDs."""
    if repo_root:
        return os.path.relpath(file_path, repo_root)
    return file_path


def _source_preview(source_lines: list[str], start_line: int, max_lines: int = 8) -> str:
    """Get up to max_lines source starting at start_line (0-indexed)."""
    end = min(start_line + max_lines, len(source_lines))
    return textwrap.dedent("".join(source_lines[start_line:end])).rstrip()


def _preceding_doc(lines: list[str], line_idx: int, style: str = 'c') -> str:
    """Extract doc comment immediately preceding line_idx.

    style: 'c' for /** */ and //, 'hash' for #, 'triple_slash' for ///
    """
    doc_lines: list[str] = []
    i = line_idx - 1

    if style == 'c':
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith('//'):
                doc_lines.insert(0, stripped.lstrip('/ '))
                i -= 1
            elif stripped.endswith('*/'):
                # Collect multi-line comment block
                while i >= 0:
                    stripped = lines[i].strip()
                    if stripped.startswith('/*') or stripped.startswith('/**'):
                        content = stripped.lstrip('/* ').rstrip('*/ ')
                        if content:
                            doc_lines.insert(0, content)
                        break
                    else:
                        doc_lines.insert(0, stripped.lstrip('* '))
                    i -= 1
                break
            elif stripped.startswith('@'):
                # Skip annotations/decorators between doc and definition
                i -= 1
                continue
            elif stripped == '':
                break
            else:
                break

    elif style == 'hash':
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith('#'):
                doc_lines.insert(0, stripped.lstrip('# '))
                i -= 1
            elif stripped == '':
                break
            else:
                break

    elif style == 'triple_slash':
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith('///'):
                content = stripped[3:].strip()
                content = _re.sub(r'</?summary>|</?param[^>]*>|</?returns>', '', content).strip()
                if content:
                    doc_lines.insert(0, content)
                i -= 1
            elif stripped == '':
                break
            else:
                break

    return ' '.join(doc_lines).strip()[:500]


# Keyword set to exclude from call extraction
_CALL_SKIP = frozenset({
    'if', 'else', 'elif', 'while', 'for', 'switch', 'case', 'catch',
    'typeof', 'instanceof', 'sizeof', 'new', 'return', 'throw',
    'await', 'yield', 'assert', 'class', 'interface', 'struct',
    'enum', 'fn', 'func', 'function', 'def', 'lambda', 'require',
    'import', 'from', 'package', 'use', 'pub', 'super', 'self',
    'this', 'static', 'const', 'let', 'var', 'type', 'export',
    'extends', 'implements', 'try', 'finally', 'do', 'goto',
    'break', 'continue', 'in', 'is', 'not', 'and', 'or', 'as',
    'with', 'del', 'pass', 'raise', 'global', 'nonlocal',
    'select', 'chan', 'go', 'defer', 'range', 'make', 'delete',
    'void', 'int', 'float', 'double', 'char', 'bool', 'string',
    'auto', 'register', 'volatile', 'nil', 'null', 'undefined',
    'true', 'false', 'True', 'False', 'None',
})


def _generic_calls(body_text: str) -> list[str]:
    """Extract function/method call names from source body."""
    calls: set[str] = set()
    for m in _re.finditer(r'\b([a-zA-Z_]\w*)\s*\(', body_text):
        name = m.group(1)
        if name not in _CALL_SKIP:
            calls.add(name)
    return list(calls)


def _build_brace_scope_map(source_lines: list[str], class_defs: list[tuple[int, str]]) -> dict[int, str]:
    """Map line numbers to their containing class name using brace counting.

    class_defs: list of (0-indexed line number, class_name).
    Returns: dict mapping line_number -> class_name.
    """
    scope_map: dict[int, str] = {}
    for class_line, class_name in class_defs:
        depth = 0
        found_open = False
        for i in range(class_line, len(source_lines)):
            line = source_lines[i]
            for ch in line:
                if ch == '{':
                    depth += 1
                    found_open = True
                elif ch == '}':
                    depth -= 1
            if found_open and depth <= 0:
                for j in range(class_line + 1, i + 1):
                    if j not in scope_map:
                        scope_map[j] = class_name
                break
    return scope_map


# ── JavaScript / TypeScript ──────────────────────────────────────────

def _parse_js_ts_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse JavaScript/TypeScript file for classes, functions, methods, imports."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    # Imports
    imports: list[str] = []
    for m in _re.finditer(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', source):
        imports.append(m.group(1))
    for m in _re.finditer(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', source):
        imports.append(m.group(1))

    # Classes
    class_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+(\w+)', line)
        if m:
            class_name = m.group(4)
            class_defs.append((i, class_name))
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            nodes.append(CodeNode(
                id=f"{file_rel}::{class_name}", name=class_name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    # Interfaces (TypeScript)
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(export\s+)?interface\s+(\w+)', line)
        if m:
            name = m.group(2)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    scope_map = _build_brace_scope_map(lines, class_defs)

    # Function declarations
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)\s*\(', line)
        if m:
            name = m.group(4)
            parent = scope_map.get(i)
            kind = "method" if parent else "function"
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{parent + '.' if parent else ''}{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind=kind, file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    # Class methods (indented, no function keyword)
    for i, line in enumerate(lines):
        parent = scope_map.get(i)
        if not parent:
            continue
        m = _re.match(r'^\s+(async\s+)?(static\s+)?(get\s+|set\s+)?(\w+)\s*\([^)]*\)\s*[:{]', line)
        if m:
            name = m.group(4)
            if name in ('if', 'while', 'for', 'switch', 'catch', 'return', 'new'):
                continue
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{parent}.{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="method", file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    # Arrow functions / function expressions (top-level only)
    for i, line in enumerate(lines):
        if scope_map.get(i):
            continue
        m = _re.match(
            r'^\s*(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?(function|\([^)]*\)\s*=>|\w+\s*=>)',
            line
        )
        if m:
            name = m.group(3)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="function", file=file_rel,
                parent_class=None, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── Java ─────────────────────────────────────────────────────────────

def _parse_java_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse Java file for classes, interfaces, enums, methods, imports."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    imports = [m.group(1) for m in _re.finditer(r'import\s+([\w.*]+)\s*;', source)]

    # Classes, interfaces, enums
    class_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _re.match(
            r'^\s*(public\s+|private\s+|protected\s+)?(abstract\s+|static\s+|final\s+)*'
            r'(class|interface|enum)\s+(\w+)', line
        )
        if m:
            name = m.group(4)
            class_defs.append((i, name))
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    scope_map = _build_brace_scope_map(lines, class_defs)

    # Methods
    _java_method_re = _re.compile(
        r'^\s*(public\s+|private\s+|protected\s+)?'
        r'(abstract\s+|static\s+|final\s+|synchronized\s+|native\s+|default\s+)*'
        r'([\w<>\[\]?,\s]+?)\s+(\w+)\s*\('
    )
    _java_skip = {'if', 'while', 'for', 'switch', 'catch', 'return', 'new',
                  'class', 'interface', 'enum', 'import', 'package'}
    for i, line in enumerate(lines):
        m = _java_method_re.match(line)
        if m:
            name = m.group(4)
            if name in _java_skip:
                continue
            parent = scope_map.get(i)
            if not parent:
                continue  # Java methods must be in a class
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{parent}.{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="method", file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── Go ───────────────────────────────────────────────────────────────

def _parse_go_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse Go file for functions, methods, types, imports."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    # Imports
    imports: list[str] = []
    in_import_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ('):
            in_import_block = True
            continue
        if in_import_block:
            if stripped == ')':
                in_import_block = False
                continue
            m = _re.match(r'.*"([^"]+)"', stripped)
            if m:
                imports.append(m.group(1))
        elif stripped.startswith('import '):
            m = _re.match(r'import\s+"([^"]+)"', stripped)
            if m:
                imports.append(m.group(1))

    # Types (struct, interface)
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*type\s+(\w+)\s+(struct|interface)\s*\{', line)
        if m:
            name = m.group(1)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    # Functions and methods
    for i, line in enumerate(lines):
        # Method with receiver: func (r *Receiver) Name(...)
        m = _re.match(r'^func\s+\(\s*\w+\s+\*?(\w+)\s*\)\s+(\w+)\s*\(', line)
        if m:
            parent = m.group(1)
            name = m.group(2)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{parent}.{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="method", file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))
            continue

        # Top-level function: func Name(...)
        m = _re.match(r'^func\s+(\w+)\s*\(', line)
        if m:
            name = m.group(1)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="function", file=file_rel,
                parent_class=None, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── Rust ─────────────────────────────────────────────────────────────

def _parse_rust_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse Rust file for functions, structs, enums, traits, impls, imports."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    imports = [m.group(1) for m in _re.finditer(r'use\s+([^;]+);', source)]

    # Structs, enums, traits
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(pub\s+)?(struct|enum|trait)\s+(\w+)', line)
        if m:
            name = m.group(3)
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'triple_slash')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    # Find impl blocks to track parent
    impl_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*impl\s+(?:\w+\s+for\s+)?(\w+)', line)
        if m:
            impl_defs.append((i, m.group(1)))

    scope_map = _build_brace_scope_map(lines, impl_defs)

    # Functions
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)\s*[(<]', line)
        if m:
            name = m.group(3)
            parent = scope_map.get(i)
            kind = "method" if parent else "function"
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'triple_slash')
            node_id = f"{file_rel}::{parent + '.' if parent else ''}{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind=kind, file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── C / C++ ──────────────────────────────────────────────────────────

def _parse_c_cpp_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse C/C++ file for functions, classes, structs, includes."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    imports = [m.group(1) for m in _re.finditer(r'#include\s+[<"]([^>"]+)[>"]', source)]

    # Classes and structs
    class_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _re.match(r'^\s*(class|struct)\s+(\w+)(?:\s*:\s*.*?)?\s*\{', line)
        if m:
            name = m.group(2)
            class_defs.append((i, name))
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    scope_map = _build_brace_scope_map(lines, class_defs)

    # Functions/methods
    _cpp_skip = {'if', 'while', 'for', 'switch', 'catch', 'return', 'else',
                 'define', 'ifdef', 'ifndef', 'include', 'class', 'struct',
                 'enum', 'namespace', 'using', 'typedef'}
    _cpp_type_skip = {'class', 'struct', 'enum', 'namespace', 'using', 'typedef',
                      'return', 'if', 'while', 'for', 'switch', 'else'}
    func_re = _re.compile(
        r'^\s*([\w:*&<>\[\]]+(?:\s+[\w:*&<>\[\]]+)*)\s+(\w+)\s*\([^)]*\)\s*'
        r'(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?\{'
    )
    for i, line in enumerate(lines):
        m = func_re.match(line)
        if m:
            ret_type = m.group(1).strip()
            name = m.group(2)
            if name in _cpp_skip or ret_type in _cpp_type_skip:
                continue
            parent = scope_map.get(i)
            kind = "method" if parent else "function"
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'c')
            node_id = f"{file_rel}::{parent + '.' if parent else ''}{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind=kind, file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── Ruby ─────────────────────────────────────────────────────────────

def _parse_ruby_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse Ruby file for classes, modules, methods, requires."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    imports: list[str] = []
    for m in _re.finditer(r"require(?:_relative)?\s+['\"]([^'\"]+)['\"]", source):
        imports.append(m.group(1))

    # Track class/module nesting using indentation
    class_stack: list[tuple[int, str]] = []  # (indent_level, name)

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # Pop class stack based on 'end' keyword at matching indentation
        if stripped == 'end' and class_stack:
            if indent <= class_stack[-1][0]:
                class_stack.pop()
            continue

        # Class or module
        m = _re.match(r'^\s*(class|module)\s+(\w+)(?:\s*<\s*(\w+))?', line)
        if m:
            name = m.group(2)
            sig = stripped
            docstring = _preceding_doc(lines, i, 'hash')
            parent = class_stack[-1][1] if class_stack else None
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=parent, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))
            class_stack.append((indent, name))
            continue

        # Method
        m = _re.match(r'^\s*def\s+(self\.)?(\w+[?!=]?)', line)
        if m:
            name = m.group(2)
            parent = class_stack[-1][1] if class_stack else None
            kind = "method" if parent else "function"
            sig = stripped
            docstring = _preceding_doc(lines, i, 'hash')
            node_id = f"{file_rel}::{parent + '.' if parent else ''}{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind=kind, file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# ── C# ───────────────────────────────────────────────────────────────

def _parse_csharp_file(file_path: str, repo_root: str = '') -> list[CodeNode]:
    """Parse C# file for classes, interfaces, methods, using statements."""
    source = _read_source(file_path)
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    nodes: list[CodeNode] = []

    imports = [m.group(1) for m in _re.finditer(r'using\s+([\w.]+)\s*;', source)]

    # Classes, interfaces, structs, enums, records
    class_defs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _re.match(
            r'^\s*(public\s+|private\s+|protected\s+|internal\s+)?'
            r'(partial\s+)?(static\s+)?(abstract\s+|sealed\s+)?'
            r'(class|interface|struct|enum|record)\s+(\w+)',
            line
        )
        if m:
            name = m.group(6)
            class_defs.append((i, name))
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'triple_slash')
            nodes.append(CodeNode(
                id=f"{file_rel}::{name}", name=name, kind="class",
                file=file_rel, parent_class=None, signature=sig, docstring=docstring,
                body_preview=_source_preview(lines, i), imports=imports,
            ))

    scope_map = _build_brace_scope_map(lines, class_defs)

    # Methods
    _cs_method_re = _re.compile(
        r'^\s*(public\s+|private\s+|protected\s+|internal\s+)?'
        r'(static\s+|virtual\s+|override\s+|abstract\s+|async\s+|new\s+|sealed\s+)*'
        r'([\w<>\[\]?,\s]+?)\s+(\w+)\s*\('
    )
    _cs_skip = {'if', 'while', 'for', 'foreach', 'switch', 'catch', 'return',
                'new', 'class', 'interface', 'struct', 'enum', 'using',
                'namespace', 'lock', 'checked', 'unchecked', 'fixed'}
    for i, line in enumerate(lines):
        parent = scope_map.get(i)
        if not parent:
            continue
        m = _cs_method_re.match(line)
        if m:
            name = m.group(4)
            if name in _cs_skip:
                continue
            sig = line.strip().rstrip('{').strip()
            docstring = _preceding_doc(lines, i, 'triple_slash')
            node_id = f"{file_rel}::{parent}.{name}"
            body = _source_preview(lines, i)
            nodes.append(CodeNode(
                id=node_id, name=name, kind="method", file=file_rel,
                parent_class=parent, signature=sig, docstring=docstring,
                body_preview=body, calls=_generic_calls(body), imports=imports,
            ))

    return nodes


# Language parser dispatch table
_LANG_PARSERS: dict[str, object] = {
    'javascript': _parse_js_ts_file,
    'typescript': _parse_js_ts_file,
    'java': _parse_java_file,
    'go': _parse_go_file,
    'rust': _parse_rust_file,
    'c': _parse_c_cpp_file,
    'cpp': _parse_c_cpp_file,
    'ruby': _parse_ruby_file,
    'csharp': _parse_csharp_file,
}


# ─────────────────────────────────────────────
# AST PARSING (Python — primary, AST-based)
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


def _parse_python_file(file_path: str, repo_root: str = "") -> list[CodeNode]:
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
# TREE-SITTER SUPPORT (optional, more accurate)
# Install extras: pip install "repo2kg[treesitter]"
# ─────────────────────────────────────────────

_TS_LANG_CACHE: dict[str, "object | None"] = {}


def _ts_load_lang(lang: str):
    """
    Lazy-load a tree-sitter Language object.
    Returns None silently if the package is not installed — regex parser is used instead.
    """
    if lang in _TS_LANG_CACHE:
        return _TS_LANG_CACHE[lang]
    result = None
    try:
        import importlib
        from tree_sitter import Language as _TSLang
        _pkg_map = {
            'javascript': ('tree_sitter_javascript', 'language'),
            'typescript': ('tree_sitter_typescript', 'language_typescript'),
            'java':       ('tree_sitter_java',        'language'),
            'go':         ('tree_sitter_go',          'language'),
            'rust':       ('tree_sitter_rust',        'language'),
            'c':          ('tree_sitter_c',           'language'),
            'cpp':        ('tree_sitter_cpp',         'language'),
            'ruby':       ('tree_sitter_ruby',        'language'),
            'csharp':     ('tree_sitter_c_sharp',     'language'),
        }
        if lang in _pkg_map:
            mod_name, fn_name = _pkg_map[lang]
            mod = importlib.import_module(mod_name)
            raw = getattr(mod, fn_name)()
            result = raw if isinstance(raw, _TSLang) else _TSLang(raw)
    except Exception:
        pass
    _TS_LANG_CACHE[lang] = result
    return result


def _ts_parser(ts_language):
    """Create a tree-sitter Parser — compatible with API v0.20 and v0.21+."""
    from tree_sitter import Parser as _TSP
    try:
        return _TSP(ts_language)       # v0.21+
    except TypeError:
        p = _TSP()
        p.set_language(ts_language)    # v0.20
        return p


def _ts_txt(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _ts_all(root, types: set) -> list:
    """Depth-first collect all descendant nodes whose type is in `types`."""
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.type in types:
            out.append(n)
        stack.extend(reversed(n.children))
    return out


def _ts_ancestor_name(node, src: bytes, ancestor_types: tuple) -> "str | None":
    """Walk up the AST and return the `name` field of the first matching ancestor."""
    cur = node.parent
    while cur is not None:
        if cur.type in ancestor_types:
            nn = cur.child_by_field_name('name')
            if nn:
                return _ts_txt(nn, src)
        cur = cur.parent
    return None


def _ts_calls(node, src: bytes) -> list[str]:
    """
    Extract called function/method names from a subtree.
    Covers JS call_expression, Java method_invocation, Ruby call,
    Go/Rust call_expression + method_call_expression, C# invocation_expression.
    """
    calls: set[str] = set()
    for call_type, func_field in (
        ('call_expression',        'function'),   # JS, Go, Rust, C/C++
        ('method_invocation',      'name'),        # Java
        ('call',                   'method'),      # Ruby
        ('method_call_expression', 'method'),      # Rust
        ('invocation_expression',  'expression'),  # C#
    ):
        for call_node in _ts_all(node, {call_type}):
            fn = call_node.child_by_field_name(func_field)
            if fn is None:
                continue
            name = ''
            if fn.type == 'identifier':
                name = _ts_txt(fn, src)
            elif fn.type == 'member_expression':          # JS: obj.method
                prop = fn.child_by_field_name('property')
                name = _ts_txt(prop, src) if prop else ''
            elif fn.type == 'selector_expression':        # Go: pkg.Func
                field = fn.child_by_field_name('field')
                name = _ts_txt(field, src) if field else ''
            elif fn.type == 'member_access_expression':   # C#: obj.Method
                mn = fn.child_by_field_name('name')
                name = _ts_txt(mn, src) if mn else ''
            elif fn.type == 'scoped_identifier':          # Rust: mod::func
                mn = fn.child_by_field_name('name')
                name = _ts_txt(mn, src) if mn else ''
            elif fn.type == 'field_expression':           # Rust: val.method
                fd = fn.child_by_field_name('field')
                name = _ts_txt(fd, src) if fd else ''
            if name and name not in _CALL_SKIP:
                calls.add(name)
    return list(calls)


def _ts_make_node(defnode, src: bytes, source_lines: list, file_rel: str,
                  imports: list, name: str, kind: str, parent: "str | None",
                  doc_style: str = 'c') -> CodeNode:
    """Build a CodeNode from a tree-sitter AST node."""
    row = defnode.start_point[0]
    sig = source_lines[row].rstrip() if row < len(source_lines) else ''
    preview = _source_preview(source_lines, row)
    doc = _preceding_doc(source_lines, row, doc_style)
    node_id = f"{file_rel}::{(parent + '.') if parent else ''}{name}"
    return CodeNode(
        id=node_id, name=name, kind=kind, file=file_rel,
        parent_class=parent, signature=sig, docstring=doc,
        body_preview=preview, calls=_ts_calls(defnode, src),
        callers=[], imports=imports,
    )


# ── JavaScript / TypeScript (tree-sitter) ──────────────────────────

def _ts_parse_js_ts(file_path: str, lang: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang(lang)
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports: list[str] = []
    for n in _ts_all(root, {'import_statement'}):
        s = n.child_by_field_name('source')
        if s:
            imports.append(_ts_txt(s, src).strip("'\"`"))
    for n in _ts_all(root, {'call_expression'}):
        fn = n.child_by_field_name('function')
        if fn and _ts_txt(fn, src) == 'require':
            args = n.child_by_field_name('arguments')
            if args:
                for ch in args.children:
                    if ch.type == 'string':
                        imports.append(_ts_txt(ch, src).strip("'\"`"))

    CLASS_TYPES = ('class_declaration', 'abstract_class_declaration',
                   'class', 'interface_declaration')

    def _par(node):
        return _ts_ancestor_name(node, src, CLASS_TYPES)

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'c')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(CLASS_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'function_declaration', 'generator_function_declaration'}):
        nn = n.child_by_field_name('name')
        if nn:
            p = _par(n)
            nodes.append(_mk(n, _ts_txt(nn, src), 'method' if p else 'function', p))

    for n in _ts_all(root, {'method_definition', 'method_signature'}):
        nn = n.child_by_field_name('name')
        if nn:
            p = _par(n)
            nodes.append(_mk(n, _ts_txt(nn, src), 'method' if p else 'function', p))

    for n in _ts_all(root, {'variable_declarator'}):
        nn = n.child_by_field_name('name')
        vn = n.child_by_field_name('value')
        if nn and vn and vn.type in ('arrow_function', 'function_expression',
                                     'generator_function_expression'):
            p = _par(n)
            nodes.append(_mk(vn, _ts_txt(nn, src), 'method' if p else 'function', p))

    return nodes


# ── Java (tree-sitter) ──────────────────────────────────────────────

def _ts_parse_java(file_path: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang('java')
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports = [_ts_txt(n.child_by_field_name('name'), src)
               for n in _ts_all(root, {'import_declaration'})
               if n.child_by_field_name('name')]

    CLASS_TYPES = ('class_declaration', 'interface_declaration',
                   'enum_declaration', 'record_declaration',
                   'annotation_type_declaration')

    def _par(node):
        return _ts_ancestor_name(node, src, CLASS_TYPES)

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'c')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(CLASS_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'method_declaration', 'constructor_declaration'}):
        nn = n.child_by_field_name('name')
        if not nn:
            continue
        p = _par(n)
        if p:  # Java methods must be inside a class
            nodes.append(_mk(n, _ts_txt(nn, src), 'method', p))

    return nodes


# ── Go (tree-sitter) ────────────────────────────────────────────────

def _ts_parse_go(file_path: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang('go')
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports: list[str] = []
    for n in _ts_all(root, {'import_spec'}):
        pn = n.child_by_field_name('path')
        if pn:
            imports.append(_ts_txt(pn, src).strip('"'))

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'c')

    nodes: list[CodeNode] = []

    for td in _ts_all(root, {'type_declaration'}):
        for spec in _ts_all(td, {'type_spec'}):
            nn = spec.child_by_field_name('name')
            if nn:
                nodes.append(_mk(spec, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'function_declaration'}):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'function', None))

    for n in _ts_all(root, {'method_declaration'}):
        nn = n.child_by_field_name('name')
        if not nn:
            continue
        recv = n.child_by_field_name('receiver')
        parent = None
        if recv:
            for param in _ts_all(recv, {'parameter_declaration'}):
                tnode = param.child_by_field_name('type')
                if tnode:
                    if tnode.type == 'type_identifier':
                        parent = _ts_txt(tnode, src)
                    elif tnode.type == 'pointer_type':
                        for ch in tnode.children:
                            if ch.type == 'type_identifier':
                                parent = _ts_txt(ch, src)
                                break
                    if parent:
                        break
        nodes.append(_mk(n, _ts_txt(nn, src), 'method' if parent else 'function', parent))

    return nodes


# ── Rust (tree-sitter) ──────────────────────────────────────────────

def _ts_parse_rust(file_path: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang('rust')
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports = [_ts_txt(n, src) for n in _ts_all(root, {'use_declaration'})]

    STRUCT_TYPES = ('struct_item', 'enum_item', 'trait_item', 'type_item')

    def _impl_parent(node) -> "str | None":
        cur = node.parent
        while cur is not None:
            if cur.type == 'impl_item':
                tnode = cur.child_by_field_name('type')
                if tnode:
                    for ch in tnode.children:
                        if ch.type == 'type_identifier':
                            return _ts_txt(ch, src)
                    return _ts_txt(tnode, src).split('<')[0]
            cur = cur.parent
        return None

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'triple_slash')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(STRUCT_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'function_item'}):
        nn = n.child_by_field_name('name')
        if nn:
            p = _impl_parent(n)
            nodes.append(_mk(n, _ts_txt(nn, src), 'method' if p else 'function', p))

    return nodes


# ── C / C++ (tree-sitter) ───────────────────────────────────────────

def _ts_parse_c_cpp(file_path: str, lang: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang(lang)
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports = [_ts_txt(n.child_by_field_name('path'), src).strip('<>"')
               for n in _ts_all(root, {'preproc_include'})
               if n.child_by_field_name('path')]

    CLASS_TYPES = ('class_specifier', 'struct_specifier')
    _SKIP = frozenset({'if', 'while', 'for', 'switch', 'return', 'catch', 'else'})

    def _par(node):
        return _ts_ancestor_name(node, src, CLASS_TYPES)

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'c')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(CLASS_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'function_definition'}):
        decl = n.child_by_field_name('declarator')
        while decl and decl.type in ('pointer_declarator', 'reference_declarator'):
            decl = decl.child_by_field_name('declarator')
        if not decl or decl.type != 'function_declarator':
            continue
        inner = decl.child_by_field_name('declarator')
        if not inner:
            continue
        if inner.type == 'qualified_identifier':     # C++: Class::method
            nn = inner.child_by_field_name('name')
            scope = inner.child_by_field_name('scope')
            parent = _ts_txt(scope, src).rstrip(':') if scope else _par(n)
        elif inner.type == 'identifier':
            nn = inner
            parent = _par(n)
        else:
            nn = None
            parent = _par(n)
        if not nn:
            continue
        name = _ts_txt(nn, src)
        if name in _SKIP:
            continue
        nodes.append(_mk(n, name, 'method' if parent else 'function', parent))

    return nodes


# ── Ruby (tree-sitter) ──────────────────────────────────────────────

def _ts_parse_ruby(file_path: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang('ruby')
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports: list[str] = []
    for n in _ts_all(root, {'call'}):
        meth = n.child_by_field_name('method')
        if meth and _ts_txt(meth, src) in ('require', 'require_relative'):
            args = n.child_by_field_name('arguments')
            if args:
                for ch in args.children:
                    if ch.type == 'string':
                        content = ch.child_by_field_name('content')
                        imports.append(_ts_txt(content if content else ch, src).strip("'\""))

    CLASS_TYPES = ('class', 'module')

    def _par(node):
        return _ts_ancestor_name(node, src, CLASS_TYPES)

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'hash')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(CLASS_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'method', 'singleton_method'}):
        nn = n.child_by_field_name('name')
        if nn:
            p = _par(n)
            nodes.append(_mk(n, _ts_txt(nn, src), 'method' if p else 'function', p))

    return nodes


# ── C# (tree-sitter) ────────────────────────────────────────────────

def _ts_parse_csharp(file_path: str, repo_root: str = '') -> list[CodeNode]:
    ts_language = _ts_load_lang('csharp')
    if ts_language is None:
        return []
    source = _read_source(file_path)
    src = source.encode('utf-8', errors='replace')
    file_rel = _rel_path(file_path, repo_root)
    lines = source.splitlines(keepends=True)
    root = _ts_parser(ts_language).parse(src).root_node

    imports = [_ts_txt(n.child_by_field_name('name'), src)
               for n in _ts_all(root, {'using_directive'})
               if n.child_by_field_name('name')]

    CLASS_TYPES = ('class_declaration', 'interface_declaration', 'struct_declaration',
                   'record_declaration', 'enum_declaration')

    def _par(node):
        return _ts_ancestor_name(node, src, CLASS_TYPES)

    def _mk(n, name, kind, parent):
        return _ts_make_node(n, src, lines, file_rel, imports, name, kind, parent, 'triple_slash')

    nodes: list[CodeNode] = []

    for n in _ts_all(root, set(CLASS_TYPES)):
        nn = n.child_by_field_name('name')
        if nn:
            nodes.append(_mk(n, _ts_txt(nn, src), 'class', None))

    for n in _ts_all(root, {'method_declaration', 'constructor_declaration',
                             'property_declaration'}):
        nn = n.child_by_field_name('name')
        if nn:
            p = _par(n)
            nodes.append(_mk(n, _ts_txt(nn, src), 'method' if p else 'function', p))

    return nodes


# Tree-sitter lang key → parser function
_TS_PARSERS: dict[str, object] = {
    'javascript': lambda fp, r: _ts_parse_js_ts(fp, 'javascript', r),
    'typescript': lambda fp, r: _ts_parse_js_ts(fp, 'typescript', r),
    'java':       _ts_parse_java,
    'go':         _ts_parse_go,
    'rust':       _ts_parse_rust,
    'c':          lambda fp, r: _ts_parse_c_cpp(fp, 'c', r),
    'cpp':        lambda fp, r: _ts_parse_c_cpp(fp, 'cpp', r),
    'ruby':       _ts_parse_ruby,
    'csharp':     _ts_parse_csharp,
}


# ─────────────────────────────────────────────
# UNIFIED PARSE DISPATCHER
# ─────────────────────────────────────────────

def parse_file(file_path: str, repo_root: str = "") -> list[CodeNode]:
    """
    Parse a source file and extract code nodes.
    Uses tree-sitter (accurate AST) when the relevant package is installed,
    falls back to regex-based parsers otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    lang = LANG_EXTENSIONS.get(ext)

    if lang is None:
        return []

    if lang == 'python':
        return _parse_python_file(file_path, repo_root)

    # Try tree-sitter first — more accurate, optional dependency
    ts_fn = _TS_PARSERS.get(lang)
    if ts_fn and _ts_load_lang(lang) is not None:
        try:
            return ts_fn(file_path, repo_root)
        except Exception as e:
            logger.warning("tree-sitter error in %s (%s): %s — falling back to regex",
                           file_path, lang, e)

    # Regex fallback
    parser = _LANG_PARSERS.get(lang)
    if parser:
        try:
            return parser(file_path, repo_root)
        except Exception as e:
            logger.warning("Parse error in %s: %s", file_path, e)
            return []

    return []


# ─────────────────────────────────────────────
# KNOWLEDGE GRAPH
# ─────────────────────────────────────────────

class RepoKG:
    def __init__(self, model_name: str | None = None):
        _load_heavy_deps()
        model_name = model_name or _default_model()
        self.nodes: dict[str, CodeNode] = {}
        self.model = _SentenceTransformer(model_name)
        self._index = None   # faiss.IndexFlatL2
        self._index_ids: list[str] = []   # maps FAISS row → node id

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, repo_path: str, exclude: list[str] | None = None) -> "RepoKG":
        """Parse repo, build nodes, resolve call edges, build FAISS index."""
        repo_path = os.path.abspath(repo_path)
        exclude_patterns = exclude or DEFAULT_EXCLUDE
        logger.info("Scanning %s ...", repo_path)

        supported_exts = set(LANG_EXTENSIONS.keys())
        source_files: list[str] = []
        lang_counts: dict[str, int] = {}

        for root, dirs, filenames in os.walk(repo_path):
            # Prune excluded directories in-place
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, pat) for pat in exclude_patterns)
            ]
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    fpath = os.path.join(root, f)
                    source_files.append(fpath)
                    lang = LANG_EXTENSIONS[ext]
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if not source_files:
            logger.warning("No supported source files found in %s", repo_path)
            print(f"Warning: No supported source files found in {repo_path}")
            print(f"Supported extensions: {', '.join(sorted(supported_exts))}")
            return self

        for fpath in source_files:
            for node in parse_file(fpath, repo_root=repo_path):
                self.nodes[node.id] = node

        if not self.nodes:
            logger.warning("No parseable functions/classes found")
            print("Warning: No parseable functions/classes found")
            return self

        self._resolve_edges()
        self._build_faiss()
        edge_count = sum(len(n.calls) for n in self.nodes.values())
        lang_summary = ", ".join(f"{lang}: {count}" for lang, count in sorted(lang_counts.items()))
        logger.info("KG ready: %d nodes, %d call edges from %d files (%s)",
                     len(self.nodes), edge_count, len(source_files), lang_summary)
        print(f"KG ready: {len(self.nodes)} nodes, {edge_count} call edges from {len(source_files)} files")
        print(f"Languages: {lang_summary}")
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
        self._index = _faiss.IndexFlatL2(dim)
        self._index.add(_numpy.array(embeddings))
        self._index_ids = ids

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str, fmt: str | None = None):
        if not self.nodes:
            logger.error("Nothing to save — KG is empty")
            print("Error: KG is empty, nothing to save")
            return
        data = {nid: asdict(n) for nid, n in self.nodes.items()}
        fmt = fmt or _detect_format(path)
        if fmt == "toon":
            with open(path, "w") as f:
                f.write(serialize_toon(data))
        else:
            with open(path, "w") as f:
                json.dump(data, f)
        # FAISS sidecar uses the base path (strip .json/.toon, add .faiss)
        base = _re.sub(r"\.(json|toon)$", "", path)
        _faiss.write_index(self._index, base + ".faiss")
        with open(base + ".idx", "w") as f:
            json.dump(self._index_ids, f)
        print(f"Saved to {path} [{fmt.upper()}] ({len(self.nodes)} nodes)")

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "RepoKG":
        base = _re.sub(r"\.(json|toon)$", "", path)
        for required in [path, base + ".faiss", base + ".idx"]:
            if not os.path.exists(required):
                raise FileNotFoundError(f"Missing KG file: {required}")
        kg = cls(model_name)
        fmt = _detect_format(path)
        if fmt == "toon":
            with open(path) as f:
                data = deserialize_toon(f.read())
        else:
            with open(path) as f:
                data = json.load(f)
        for nid, ndata in data.items():
            kg.nodes[nid] = CodeNode(**ndata)
        kg._index = _faiss.read_index(base + ".faiss")
        with open(base + ".idx") as f:
            kg._index_ids = json.load(f)
        print(f"Loaded KG: {len(kg.nodes)} nodes [{fmt.upper()}]")
        return kg

    # ── Query ──────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 5) -> list[CodeNode]:
        """Layer 2: FAISS semantic search → entry point nodes."""
        q_emb = self.model.encode([query])
        _, indices = self._index.search(_numpy.array(q_emb), k)
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
        fmt = _detect_format(path)
        with open(path) as f:
            if fmt == "toon":
                data = deserialize_toon(f.read())
            else:
                data = json.load(f)
        self.nodes: dict[str, CodeNode] = {}
        for nid, ndata in data.items():
            self.nodes[nid] = CodeNode(**ndata)
        self.path = path
        self.format = fmt

    def search(self, keywords: str, k: int = 10) -> list[CodeNode]:
        """Keyword search over node names, docstrings, signatures, and file paths."""
        terms = [t.lower() for t in _re.split(r"[\\s,_]+", keywords) if len(t) > 1]
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
    fmt = _detect_format(kg_path)
    with open(kg_path) as f:
        if fmt == "toon":
            data = deserialize_toon(f.read())
        else:
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


# ─────────────────────────────────────────────
# VISUAL GRAPH (interactive HTML for humans)
# ─────────────────────────────────────────────

def generate_visual_graph(kg_path: str, out_path: str, max_nodes: int = 800) -> None:
    """
    Generate a self-contained interactive HTML knowledge-graph visualization.

    Opens in any browser — no server needed.  Uses D3 v7 (CDN) for a
    force-directed layout with:
      • Node colours by kind  (class=blue / function=green / method=orange)
      • Directed call edges with arrow heads
      • Hover tooltip  — name, kind, file, docstring, signature, calls
      • Click to highlight a node and its direct neighbours
      • Keyword search box — filters visible nodes in real time
      • Filter buttons  — toggle classes / functions / methods
      • Drag-to-pin, zoom, pan

    Large KGs (>max_nodes nodes) are capped: the most-connected nodes are kept.
    """
    fmt = _detect_format(kg_path)
    with open(kg_path) as f:
        data = deserialize_toon(f.read()) if fmt == "toon" else json.load(f)

    nodes_dict = {nid: CodeNode(**nd) for nid, nd in data.items()}
    title = os.path.splitext(os.path.basename(kg_path))[0]

    # If the KG is very large, keep the top max_nodes most-connected nodes.
    if len(nodes_dict) > max_nodes:
        print(f"KG has {len(nodes_dict)} nodes — limiting to {max_nodes} most-connected for rendering.")
        ranked = sorted(
            nodes_dict.values(),
            key=lambda n: len(n.calls) + len(n.callers),
            reverse=True,
        )
        keep_ids = {n.id for n in ranked[:max_nodes]}
        nodes_dict = {nid: n for nid, n in nodes_dict.items() if nid in keep_ids}

    keep_ids = set(nodes_dict.keys())

    nodes_json: list[dict] = []
    for n in nodes_dict.values():
        nodes_json.append({
            "id": n.id,
            "name": n.name,
            "kind": n.kind,
            "file": n.file,
            "parent_class": n.parent_class or "",
            "signature": n.signature[:150],
            "docstring": (n.docstring or "")[:250],
            "calls": [c.split("::")[-1] for c in n.calls[:8]],
            "callers": [c.split("::")[-1] for c in n.callers[:8]],
        })

    links_json: list[dict] = []
    seen_links: set[tuple] = set()
    for n in nodes_dict.values():
        for callee_id in n.calls:
            if callee_id in keep_ids:
                key = (n.id, callee_id)
                if key not in seen_links:
                    seen_links.add(key)
                    links_json.append({"source": n.id, "target": callee_id})

    graph_data = json.dumps({"nodes": nodes_json, "links": links_json}, ensure_ascii=False)
    node_count = len(nodes_json)
    link_count = len(links_json)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>repo2kg — {title}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;overflow:hidden}}
  #controls{{position:fixed;top:0;left:0;right:0;z-index:10;background:#161b22;border-bottom:1px solid #30363d;
    padding:8px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
  #search{{background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;
    padding:4px 10px;font-size:13px;width:200px}}
  #search:focus{{outline:none;border-color:#58a6ff}}
  .fbtn{{background:none;border:1px solid #30363d;border-radius:4px;color:#8b949e;
    padding:3px 10px;cursor:pointer;font-size:12px;transition:all .15s}}
  .fbtn.on{{border-color:#58a6ff;color:#c9d1d9;background:#21262d}}
  #legend{{display:flex;gap:10px;align-items:center;margin-left:auto;font-size:12px}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:3px}}
  #info{{font-size:11px;color:#6e7681}}
  #tip{{position:fixed;background:#1c2128;border:1px solid #373e47;border-radius:8px;
    padding:10px 14px;font-size:12px;max-width:320px;pointer-events:none;opacity:0;
    transition:opacity .15s;z-index:100;line-height:1.5}}
  #tip h4{{margin:0 0 4px;color:#58a6ff;font-size:13px}}
  #tip .m{{color:#8b949e;margin:2px 0}}
  #tip .sig{{font-family:monospace;font-size:11px;color:#d2a8ff;margin:4px 0;
    white-space:pre-wrap;word-break:break-all;max-height:80px;overflow:hidden}}
  svg{{width:100vw;height:100vh;cursor:grab}}
  svg.dragging{{cursor:grabbing}}
  .link{{stroke:#30363d;stroke-opacity:.7;fill:none;marker-end:url(#arr)}}
  .link.hl{{stroke:#f78166;stroke-opacity:1;stroke-width:2}}
  .node circle{{stroke-width:1.5;cursor:pointer}}
  .node text{{font-size:9px;fill:#6e7681;pointer-events:none;dominant-baseline:middle}}
  .node.sel circle{{stroke:#f0e68c!important;stroke-width:3}}
  .node.dim{{opacity:.12}}
</style>
</head>
<body>
<div id="controls">
  <input id="search" type="text" placeholder="Search nodes…" autocomplete="off">
  <button class="fbtn on" data-k="class">Classes</button>
  <button class="fbtn on" data-k="function">Functions</button>
  <button class="fbtn on" data-k="method">Methods</button>
  <div id="legend">
    <span><span class="dot" style="background:#79c0ff"></span>Class</span>
    <span><span class="dot" style="background:#7ee787"></span>Function</span>
    <span><span class="dot" style="background:#ffa657"></span>Method</span>
  </div>
  <span id="info">{node_count} nodes · {link_count} edges</span>
</div>
<div id="tip"></div>
<svg>
  <defs>
    <marker id="arr" viewBox="0 -3 6 6" refX="15" refY="0"
            markerWidth="4" markerHeight="4" orient="auto">
      <path d="M0,-3L6,0L0,3" fill="#484f58"/>
    </marker>
  </defs>
  <g id="zg">
    <g id="lg"></g>
    <g id="ng"></g>
  </g>
</svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = {graph_data};
const KC = {{class:"#79c0ff",function:"#7ee787",method:"#ffa657"}};
const KR = {{class:9,function:6,method:5}};
let active = new Set(["class","function","method"]);
let sel = null;

const svg = d3.select("svg");
const g   = svg.select("#zg");
const tip = document.getElementById("tip");

const zoom = d3.zoom().scaleExtent([0.03,8]).on("zoom", e => {{
  g.attr("transform", e.transform);
  svg.classed("dragging", e.sourceEvent && e.sourceEvent.buttons === 1);
}});
svg.call(zoom).on("dblclick.zoom", null);

const W = window.innerWidth, H = window.innerHeight;
const sim = d3.forceSimulation()
  .force("link",  d3.forceLink().id(d=>d.id).distance(60).strength(0.5))
  .force("charge",d3.forceManyBody().strength(-150))
  .force("center",d3.forceCenter(W/2, H/2))
  .force("x",     d3.forceX(W/2).strength(0.03))
  .force("y",     d3.forceY(H/2).strength(0.03))
  .force("coll",  d3.forceCollide(14));

let linkSel, nodeSel;

function visible() {{
  const q = document.getElementById("search").value.toLowerCase().trim();
  return DATA.nodes.filter(n => {{
    if (!active.has(n.kind)) return false;
    if (q && !n.name.toLowerCase().includes(q) && !n.file.toLowerCase().includes(q)) return false;
    return true;
  }});
}}

function render() {{
  const vis    = visible();
  const visSet = new Set(vis.map(n=>n.id));
  const visLinks = DATA.links.filter(l =>
    visSet.has(typeof l.source==="object"?l.source.id:l.source) &&
    visSet.has(typeof l.target==="object"?l.target.id:l.target));

  document.getElementById("info").textContent =
    vis.length + " nodes · " + visLinks.length + " edges";

  // Links
  linkSel = g.select("#lg").selectAll(".link")
    .data(visLinks, d=>`${{d.source.id||d.source}}-${{d.target.id||d.target}}`);
  linkSel.enter().append("line").attr("class","link").merge(linkSel);
  linkSel.exit().remove();
  linkSel = g.select("#lg").selectAll(".link");

  // Nodes
  const nd = g.select("#ng").selectAll(".node").data(vis, d=>d.id);
  const ndE = nd.enter().append("g").attr("class","node")
    .call(d3.drag()
      .on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}})
      .on("drag", (e,d)=>{{d.fx=e.x;d.fy=e.y}})
      .on("end",  (e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}))
    .on("click", (_,d)=>{{sel=sel===d.id?null:d.id;paint()}})
    .on("mouseover",(e,d)=>showTip(e,d))
    .on("mousemove",(e)=>moveTip(e))
    .on("mouseout", ()=>{{tip.style.opacity=0}});
  ndE.append("circle")
    .attr("r",d=>KR[d.kind]||5)
    .attr("fill",d=>KC[d.kind]||"#8b949e")
    .attr("stroke",d=>d3.color(KC[d.kind]||"#8b949e").darker(0.6));
  ndE.append("text").attr("dx",d=>(KR[d.kind]||5)+4).attr("dy","0.35em").text(d=>d.name);
  nd.exit().remove();
  nodeSel = g.select("#ng").selectAll(".node");

  sim.nodes(vis).on("tick", ()=>{{
    linkSel
      .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
      .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
    nodeSel.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
  }});
  sim.force("link").links(visLinks);
  sim.alpha(0.5).restart();
  paint();
}}

function connectedIds(id) {{
  const ids = new Set([id]);
  (linkSel||d3.selectAll(".link")).each(d=>{{
    const s=d.source.id||d.source, t=d.target.id||d.target;
    if(s===id) ids.add(t);
    if(t===id) ids.add(s);
  }});
  return ids;
}}

function paint() {{
  if (!nodeSel) return;
  if (!sel) {{
    nodeSel.classed("sel",false).classed("dim",false);
    if(linkSel) linkSel.classed("hl",false);
    return;
  }}
  const nb = connectedIds(sel);
  nodeSel.classed("sel",d=>d.id===sel).classed("dim",d=>!nb.has(d.id));
  if(linkSel) linkSel.classed("hl",d=>{{
    const s=d.source.id||d.source,t=d.target.id||d.target;
    return (s===sel||t===sel);
  }});
}}

function showTip(e, d) {{
  const calls   = (d.calls||[]).slice(0,6).join(", ");
  const callers = (d.callers||[]).slice(0,6).join(", ");
  tip.innerHTML = `
    <h4>${{d.name}}</h4>
    <div class="m"><b>${{d.kind}}</b> · ${{d.file}}</div>
    ${{d.docstring?`<div class="m">${{d.docstring.slice(0,200)}}</div>`:""}}
    <div class="sig">${{(d.signature||"").slice(0,130)}}</div>
    ${{calls?`<div class="m">→ calls: ${{calls}}</div>`:""}}
    ${{callers?`<div class="m">← by: ${{callers}}</div>`:""}}
  `;
  tip.style.opacity = 1;
  moveTip(e);
}}
function moveTip(e) {{
  const x = Math.min(e.clientX+18, window.innerWidth-340);
  const y = Math.min(e.clientY+12, window.innerHeight-200);
  tip.style.left = x+"px"; tip.style.top = y+"px";
}}

document.querySelectorAll(".fbtn").forEach(b=>b.addEventListener("click",()=>{{
  const k=b.dataset.k;
  if(active.has(k)){{active.delete(k);b.classList.remove("on")}}
  else{{active.add(k);b.classList.add("on")}}
  render();
}}));

let st;
document.getElementById("search").addEventListener("input",()=>{{
  clearTimeout(st); st=setTimeout(render,250);
}});

svg.on("click",(e)=>{{
  if(e.target===svg.node()||e.target===g.node()){{sel=null;paint()}}
}});

render();

// Auto-fit on first render
setTimeout(()=>{{
  const bounds = g.node().getBBox();
  if(bounds.width>0){{
    const scale = Math.min(0.9, Math.min(W/bounds.width, (H-48)/bounds.height));
    const tx = (W - bounds.width*scale)/2 - bounds.x*scale;
    const ty = (H - bounds.height*scale)/2 - bounds.y*scale + 24;
    svg.call(zoom.transform, d3.zoomIdentity.translate(tx,ty).scale(scale));
  }}
}}, 2500);
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Visual graph saved to {out_path} ({node_count} nodes, {link_count} edges)")
    print(f"Open in browser: file://{os.path.abspath(out_path)}")


def generate_agent_instructions(kg_path: str, target_dir: str):
    """Generate CLAUDE.md and .copilot-instructions.md so agents auto-discover the KG."""
    fmt = _detect_format(kg_path)
    with open(kg_path) as f:
        if fmt == "toon":
            data = deserialize_toon(f.read())
        else:
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

    # ── Write / merge files ────────────────────────────────────────────
    # All files use marker-based merge: existing user content outside the
    # <!-- repo2kg-start/end --> markers is preserved on re-runs.
    os.makedirs(target_dir, exist_ok=True)

    def _write_merged(rel_path: str, content: str, verb: str = "Updated") -> None:
        full_path = Path(os.path.join(target_dir, rel_path))
        full_path.parent.mkdir(parents=True, exist_ok=True)
        block = f"{_REPO2KG_MARKER}\n{content.strip()}\n{_REPO2KG_END_MARKER}\n"
        existed = full_path.exists()
        _merge_file_with_block(full_path, block)
        action = "Merged →" if existed else "Created  "
        print(f"{action} {full_path}")

    _write_merged("CLAUDE.md", claude_md)
    _write_merged(".copilot-instructions.md", copilot_md)
    _write_merged(".github/copilot-instructions.md", copilot_agent_md)
    _write_merged("AGENTS.md", agents_md)


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


# ─────────────────────────────────────────────
# CONFIGURATION (embedding model, etc.)
# ─────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(config: dict):
    REPO2KG_HOME.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _default_model() -> str:
    """Return the configured embedding model name (or the built-in default)."""
    return _load_config().get("model", "all-MiniLM-L6-v2")


def _merge_file_with_block(file_path: Path, new_block: str) -> None:
    """
    Write new_block into file_path using marker-based merge.
    If markers already exist the section between them is replaced.
    Otherwise the block is appended, preserving all existing content.
    """
    existing = file_path.read_text() if file_path.exists() else ""
    if _REPO2KG_MARKER in existing:
        new_content = _re.sub(
            rf"{_re.escape(_REPO2KG_MARKER)}.*?{_re.escape(_REPO2KG_END_MARKER)}",
            new_block.strip(),
            existing,
            flags=_re.DOTALL,
        )
    else:
        new_content = existing.rstrip() + ("\n\n" if existing else "") + new_block
    file_path.write_text(new_content)


def _delete_cached_model(model_name: str) -> None:
    """
    Delete a cached SentenceTransformer / HuggingFace model from disk.
    Checks both the modern HF Hub cache and the legacy torch cache.
    """
    import shutil

    # Modern HuggingFace Hub cache layout:
    # ~/.cache/huggingface/hub/models--<org>--<name>/
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if "/" in model_name:
        org, name = model_name.split("/", 1)
        hf_dir_name = f"models--{org}--{name}"
    else:
        hf_dir_name = f"models--sentence-transformers--{model_name}"
    hf_model_path = hf_cache / hf_dir_name

    # Legacy SentenceTransformer cache:
    # ~/.cache/torch/sentence_transformers/<model_name>/
    st_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
    st_model_name = model_name.replace("/", "_")
    st_model_path = st_cache / st_model_name

    deleted: list[str] = []
    for path in [hf_model_path, st_model_path]:
        if path.exists():
            shutil.rmtree(path)
            deleted.append(str(path))

    if deleted:
        for p in deleted:
            print(f"Deleted cached model: {p}")
    else:
        print(f"No cached model found for '{model_name}' (cache already clean)")


def cmd_set_model(model_name: str, delete_old: bool = False) -> None:
    """
    Change the default embedding model used by 'build' and 'query'.
    Optionally delete the previously cached model to free disk space.
    The new model is downloaded immediately so the next 'build' is fast.
    """
    config = _load_config()
    old_model = config.get("model", "all-MiniLM-L6-v2")

    if delete_old and old_model != model_name:
        print(f"Deleting old model '{old_model}' from cache…")
        _delete_cached_model(old_model)

    config["model"] = model_name
    _save_config(config)
    print(f"Default embedding model set to: {model_name}")

    # Pre-download so the first 'build' doesn't have to wait
    print(f"Downloading '{model_name}' (this may take a moment)…")
    _load_heavy_deps()
    _SentenceTransformer(model_name)
    print(f"Model '{model_name}' is ready.")


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
            if not (fname.endswith(".json") or fname.endswith(".toon")):
                continue
            fpath = Path(dirpath) / fname
            # Check for FAISS sidecar — strip extension, look for .faiss
            base = _re.sub(r"\.(json|toon)$", "", str(fpath))
            if Path(base + ".faiss").exists():
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
    global_block = f"""{_REPO2KG_MARKER}
# repo2kg Global Agent Reference

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
{_REPO2KG_END_MARKER}
"""
    _merge_file_with_block(global_ref, global_block)
    print(f"Updated {global_ref}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# AGENT DISCOVERY: info command
# ─────────────────────────────────────────────

def print_tool_info():
    """Machine-readable tool info for agents that run `repo2kg --help` or `repo2kg info`."""
    info = {
        "tool": "repo2kg",
        "version": __version__,
        "description": "Build a code Knowledge Graph from any repository for token-efficient AI agent context",
        "languages": sorted(set(LANG_EXTENSIONS.values())),
        "formats": ["json", "toon"],
        "commands": {
            "build": {
                "usage": "repo2kg build --repo <path> --out <kg.json|kg.toon>",
                "description": "Parse source files (Python/JS/TS/Java/Go/Rust/C/C++/Ruby/C#), build KG with nodes + call edges + FAISS index",
                "outputs": ["<out> (KG data)", "<out>.faiss (vector index)", "<out>.idx (FAISS id map)"],
            },
            "query": {
                "usage": "repo2kg query 'question' --kg <kg.json|kg.toon>",
                "description": "Semantic search (FAISS) + graph expansion. Requires FAISS+embeddings",
                "flags": ["--k N (top-k, default 5)", "--depth N (expansion, default 1)", "--format text|json"],
            },
            "query-lite": {
                "usage": "repo2kg query-lite 'keywords' --kg <kg.json|kg.toon>",
                "description": "Keyword search + graph expansion. Zero heavy deps — agents should prefer this",
                "flags": ["--k N", "--depth N", "--format text|json"],
            },
            "export": {
                "usage": "repo2kg export --kg <kg.json|kg.toon> --out CODEBASE.md",
                "description": "Generate standalone markdown overview for agents",
            },
            "agent-setup": {
                "usage": "repo2kg agent-setup --kg <kg.json|kg.toon> --dir .",
                "description": "Generate CLAUDE.md, .copilot-instructions.md, AGENTS.md, CODEBASE.md",
            },
            "info": {
                "usage": "repo2kg info",
                "description": "Print this machine-readable tool description (JSON)",
            },
            "stats": {
                "usage": "repo2kg stats --kg <kg.json|kg.toon>",
                "description": "Show node/edge/file counts",
            },
            "scan": {
                "usage": "repo2kg scan --root <dir>",
                "description": "Auto-discover all KG files under a directory and register them",
            },
            "list": {
                "usage": "repo2kg list",
                "description": "Show all registered projects",
            },
            "user-setup": {
                "usage": "repo2kg user-setup",
                "description": "Install global agent instructions (~/.claude/CLAUDE.md, ~/.codex/AGENTS.md)",
            },
            "register": {
                "usage": "repo2kg register --kg <kg.json|kg.toon> --project .",
                "description": "Register a project in ~/.repo2kg/registry.json",
            },
        },
        "quick_start": [
            "repo2kg build --repo . --out kg.json       # Build KG (JSON)",
            "repo2kg build --repo . --out kg.toon       # Build KG (TOON, 40% fewer tokens)",
            "repo2kg query-lite 'auth' --kg kg.json     # Search (no heavy deps)",
            "repo2kg export --kg kg.json                # Generate CODEBASE.md",
            "repo2kg agent-setup --kg kg.json --dir .   # Full agent setup",
        ],
        "agent_tips": [
            "Prefer query-lite over query — it needs zero heavy dependencies",
            "Read CODEBASE.md for a full project overview (no tool calls needed)",
            "Use --format json for structured output you can parse programmatically",
            "KG supports .json and .toon formats — auto-detected from extension",
            "TOON format uses ~40% fewer tokens than JSON — best for LLM context windows",
            "Supports Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, C#",
        ],
    }
    print(json.dumps(info, indent=2))


def _print_banner():
    """Print a colourful ASCII banner for repo2kg user-setup."""
    # ANSI colour codes (gracefully degrade on non-colour terminals)
    try:
        import shutil
        _w = shutil.get_terminal_size().columns
    except Exception:
        _w = 80

    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    logo = f"""{CYAN}{BOLD}
  ██████╗ ███████╗██████╗  ██████╗ ██████╗ ██╗  ██╗ ██████╗
  ██╔══██╗██╔════╝██╔══██╗██╔═══██╗╚════██╗██║ ██╔╝██╔════╝
  ██████╔╝█████╗  ██████╔╝██║   ██║ █████╔╝█████╔╝ ██║  ███
  ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔═══╝ ██╔═██╗ ██║   ██║
  ██║  ██║███████╗██║      ╚██████╔╝███████╗██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝╚══════╝╚═╝       ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝{RESET}"""

    tagline  = f"  {GREEN}Code Knowledge Graph for token-efficient AI agent context{RESET}"
    version  = f"  {DIM}v{__version__}  •  JSON & TOON  •  Python/JS/TS/Java/Go/Rust/C/C++/Ruby/C#{RESET}"
    divider  = f"  {YELLOW}{'─' * min(62, _w - 4)}{RESET}"
    headline = f"  {BOLD}Setting up global agent instructions...{RESET}"

    print(logo)
    print(tagline)
    print(version)
    print(divider)
    print(headline)
    print()


_CLI_EPILOG = """\
──────────────────────────────────────────────────────────────────────
Quick Start:
  repo2kg build --repo . --out kg.json       # Build from current dir (JSON)
  repo2kg build --repo . --out kg.toon       # Build from current dir (TOON, 40% fewer tokens)
  repo2kg query-lite "auth" --kg kg.json     # Keyword search (no heavy deps)
  repo2kg query "how does auth work" --kg kg.json  # Semantic search (FAISS)
  repo2kg export --kg kg.json                # Generate CODEBASE.md
  repo2kg agent-setup --kg kg.json --dir .   # Full agent file setup
  repo2kg info                               # Machine-readable tool info (JSON)

Formats:
  .json  — Default. Compact, fast, universal.
  .toon  — Token-Oriented Object Notation. ~40% fewer tokens than JSON.
           Best for LLM context windows. Spec: github.com/toon-format

Languages:
  Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, C#

Agent Integration:
  repo2kg user-setup                         # Install global agent instructions
  repo2kg scan --root ~                      # Auto-discover all KGs
  repo2kg list                               # Show registered projects
──────────────────────────────────────────────────────────────────────
"""


def main():
    parser = argparse.ArgumentParser(
        prog="repo2kg",
        description="repo2kg — Build a code Knowledge Graph for token-efficient AI agent context.\n"
                    "Supports JSON and TOON output formats (auto-detected from file extension).\n"
                    "Languages: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, C#.",
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"repo2kg {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="cmd")

    # ── build ──
    build_p = sub.add_parser("build", help="Build KG from a repository (all supported languages)",
                             description="Parse source files (Python/JS/TS/Java/Go/Rust/C/C++/Ruby/C#), "
                                         "extract functions/classes/methods, "
                                         "resolve call edges, build FAISS index, and save to JSON or TOON.")
    build_p.add_argument("--repo", default=".", help="Path to repo root (default: current dir)")
    build_p.add_argument("--out", default="kg.json",
                         help="Output path for KG. Use .json or .toon extension (default: kg.json)")
    build_p.add_argument("--exclude", nargs="*", help="Additional directory patterns to exclude")

    # ── query (semantic, requires FAISS) ──
    query_p = sub.add_parser("query", help="Query the KG with natural language (requires FAISS)")
    query_p.add_argument("question", help="Natural language question")
    query_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    query_p.add_argument("--k", type=int, default=5, help="Top-k results (default: 5)")
    query_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    query_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json (structured)")

    # ── query-lite (keyword, no heavy deps) ──
    qlite_p = sub.add_parser("query-lite",
                             help="Query KG with keyword matching (no FAISS/embeddings needed)")
    qlite_p.add_argument("keywords", help="Keywords to search for")
    qlite_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    qlite_p.add_argument("--k", type=int, default=5, help="Top-k results (default: 5)")
    qlite_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    qlite_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json")

    # ── export (generate CODEBASE.md) ──
    export_p = sub.add_parser("export", help="Export KG as agent-readable CODEBASE.md")
    export_p.add_argument("--kg", default="kg.json",
                          help="Path to saved KG .json/.toon (default: kg.json)")
    export_p.add_argument("--out", default="CODEBASE.md",
                          help="Output markdown file (default: CODEBASE.md)")

    # ── agent-setup ──
    agent_p = sub.add_parser("agent-setup",
                             help="Generate agent instruction files (CLAUDE.md, .copilot-instructions.md, etc.)")
    agent_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    agent_p.add_argument("--dir", default=".",
                         help="Target directory for agent files (default: current dir)")
    agent_p.add_argument("--no-register", action="store_true",
                         help="Skip registering in global ~/.repo2kg/registry.json")

    # ── user-setup (global agent instructions) ──
    sub.add_parser("user-setup",
                   help="Install global agent instructions (~/.claude/CLAUDE.md, ~/.codex/AGENTS.md, etc.)")

    # ── register ──
    reg_p = sub.add_parser("register",
                           help="Register a project KG in the global ~/.repo2kg/registry.json")
    reg_p.add_argument("--kg", default="kg.json",
                       help="Path to saved KG .json/.toon (default: kg.json)")
    reg_p.add_argument("--project", default=".",
                       help="Project root directory to register (default: current dir)")

    # ── scan ──
    scan_p = sub.add_parser("scan",
                            help="Scan a directory tree, find all KG files, and register them globally")
    scan_p.add_argument("--root", default=str(Path.home()),
                        help="Root directory to scan (default: home directory)")

    # ── list (show registered projects) ──
    sub.add_parser("list", help="List all projects registered in the global registry")

    # ── info (machine-readable tool discovery for agents) ──
    sub.add_parser("info",
                   help="Print machine-readable tool info (JSON) — for agent discovery")

    # ── stats ──
    stats_p = sub.add_parser("stats", help="Show KG statistics")
    stats_p.add_argument("--kg", default="kg.json", help="Path to saved KG (default: kg.json)")

    # ── set-model (change embedding model) ──
    setm_p = sub.add_parser(
        "set-model",
        help="Change the default embedding model (HuggingFace model name)",
        description=(
            "Set the SentenceTransformer model used by 'build' and 'query'. "
            "The model is downloaded immediately. Note: 'query-lite' uses keyword search "
            "and never loads embeddings — use 'query' for semantic accuracy."
        ),
    )
    setm_p.add_argument("model", help="HuggingFace model name, e.g. 'sentence-transformers/all-mpnet-base-v2'")
    setm_p.add_argument(
        "--delete-old", action="store_true",
        help="Delete the previously cached model from disk to free space",
    )

    # ── visualize (interactive HTML graph for humans) ──
    viz_p = sub.add_parser(
        "visualize",
        help="Generate an interactive HTML knowledge-graph visualization (for humans)",
        description=(
            "Creates a self-contained HTML file with a D3.js force-directed graph. "
            "Open in any browser — no server needed. "
            "Nodes are coloured by kind; edges show call relationships. "
            "Supports search, kind filters, hover tooltips, and zoom/pan."
        ),
    )
    viz_p.add_argument("--kg", default="kg.json",
                       help="Path to saved KG .json/.toon (default: kg.json)")
    viz_p.add_argument("--out", default="kg_graph.html",
                       help="Output HTML file (default: kg_graph.html)")
    viz_p.add_argument("--max-nodes", type=int, default=800,
                       help="Cap on nodes rendered (most-connected kept, default: 800)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.cmd == "info":
        print_tool_info()

    elif args.cmd == "build":
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
        _print_banner()
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
        model_in_use = _default_model()
        print(f"Knowledge Graph Statistics:")
        print(f"  Files:         {len(files)}")
        print(f"  Nodes:         {len(kg.nodes)} ({classes} classes, {functions} functions, {methods} methods)")
        print(f"  Edges:         {edges} call edges")
        print(f"  Avg edges:     {edges / max(len(kg.nodes), 1):.1f} per node")
        print(f"  Embedding model: {model_in_use}")
        print(f"  Note: 'query-lite' uses keyword search (no embeddings).")
        print(f"        Use 'repo2kg query' for semantic (embedding-based) search accuracy.")

    elif args.cmd == "set-model":
        cmd_set_model(args.model, delete_old=args.delete_old)

    elif args.cmd == "visualize":
        generate_visual_graph(args.kg, args.out, max_nodes=args.max_nodes)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
