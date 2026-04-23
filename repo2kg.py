"""
repo2kg — Repository Knowledge Graph for token-efficient agent context.

Architecture:
  Layer 1 (KG)  — nodes=functions/classes, edges=CALLS/IMPORTS/DEFINED_IN
  Layer 2 (RAG) — FAISS over rich node summaries for semantic entry-point search
  Output        — token-minimal structured context for LLM agents
  Formats       — JSON (default) or TOON (token-optimized)

Recommended agent workflow (minimal tokens):
  repo2kg build --repo . --out kg.toon            # 1. Build KG (TOON = 40% fewer tokens)
  repo2kg summary --kg kg.toon                    # 2. Compact map (~10k tokens) — orientation
  repo2kg query "how does auth work" --kg kg.toon # 3. Targeted semantic search (~400 tokens/query)
  repo2kg query-lite "login" --kg kg.toon         # 3b. Keyword search (no deps, instant)

Agent setup commands:
  repo2kg agent-setup --kg kg.toon --dir .        # generate CLAUDE.md + .copilot-instructions.md
  repo2kg info                                    # JSON tool description for agent discovery

Full export (use only when you need everything):
  repo2kg export --kg kg.toon --out CODEBASE.md   # full dump — ~60k tokens for large projects
  repo2kg summary --kg kg.toon --out SUMMARY.md   # compact alternative — ~10k tokens

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

__version__ = "0.6.0"

import os
import sys
import ast
import json
import argparse
import logging
import textwrap
import fnmatch
import hashlib
import io
import threading
import re as _re
from collections import deque
from pathlib import Path

from dataclasses import dataclass, field, asdict, fields as _dc_fields
from typing import Optional

try:
    import tomllib as _tomllib
except ImportError:
    try:
        import tomli as _tomllib  # type: ignore
    except ImportError:
        _tomllib = None  # type: ignore

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
    # VCS / tooling
    "__pycache__", ".git", ".hg", ".svn",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "*.egg-info",
    # Virtual environments
    "venv", ".venv", "env", ".env",
    # Build / bundler outputs  ← these produce generated/minified code
    "dist", "build", "out", "coverage",
    ".next", ".nuxt", ".output",
    ".vite", "__sapper__",
    # Dependency folders
    "node_modules", "vendor", "site-packages", "bower_components",
]

# File-level patterns filtered during build (checked against the full relative path)
_EXCLUDE_FILE_PATTERNS = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    ".generated.",
    ".pb.go",        # protobuf generated
    "_pb2.py",       # protobuf generated (Python)
    ".g.dart",       # generated Dart
    ".freezed.dart", # generated Dart
    "chunk-",        # Vite chunk files like chunk-ABCD1234.js
)

def _is_generated_file(fpath: str) -> bool:
    """Return True if the file looks like a build artifact or generated file."""
    name = os.path.basename(fpath).lower()
    return any(pat in name for pat in _EXCLUDE_FILE_PATTERNS)

# Marker pair used to identify repo2kg-managed sections in shared files.
# Content between these markers is replaced on re-run; content outside is preserved.
_REPO2KG_MARKER = "<!-- repo2kg-start -->"
_REPO2KG_END_MARKER = "<!-- repo2kg-end -->"

CONFIG_PATH = REPO2KG_HOME / "config.json"

BODY_PREVIEW_LINES = 12        # default preview lines (configurable via repo2kg.toml)
MAX_TOON_FIELD_LEN = 1_000_000 # guard against pathological TOON inputs


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
    in_cycle: bool = False                            # True if this node is in a call cycle

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
    if len(s) > MAX_TOON_FIELD_LEN:
        raise ValueError(f"TOON field exceeds MAX_TOON_FIELD_LEN ({MAX_TOON_FIELD_LEN})")
    items = []
    buf = io.StringIO()
    in_quote = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_quote and i + 1 < len(s):
            buf.write(c)
            buf.write(s[i + 1])
            i += 2
            continue
        if c == '"':
            in_quote = not in_quote
            buf.write(c)
        elif c == ',' and not in_quote:
            items.append(buf.getvalue().strip())
            buf = io.StringIO()
        else:
            buf.write(c)
        i += 1
    rest = buf.getvalue().strip()
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
    if not text.lstrip().startswith("# repo2kg") and not text.lstrip().startswith("# TOON"):
        raise ValueError("File does not appear to be a valid TOON KG (missing header)")
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
# PROJECT CONFIG (repo2kg.toml)
# ─────────────────────────────────────────────

def _load_project_config(repo_path: str) -> dict:
    """Load repo2kg.toml from repo root, merging over built-in defaults."""
    defaults: dict = {
        "build": {
            "exclude_dirs": [],
            "exclude_file_patterns": [],
            "body_preview_lines": BODY_PREVIEW_LINES,
            "max_file_size_kb": 500,
            "strict_calls": False,
        },
        "embeddings": {
            "model": None,
            "batch_size": 64,
        },
        "output": {
            "format": None,
        },
    }
    config_path = Path(repo_path) / "repo2kg.toml"
    if not config_path.exists():
        return defaults
    if _tomllib is None:
        logger.warning("repo2kg.toml found but tomllib/tomli not available — ignoring config")
        return defaults
    try:
        with open(config_path, "rb") as f:
            user_cfg = _tomllib.load(f)
        for section, values in user_cfg.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
            else:
                defaults[section] = values
        logger.info("Loaded config from %s", config_path)
    except Exception as e:
        logger.warning("Failed to load repo2kg.toml: %s", e)
    return defaults


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


def _source_preview(source_lines: list[str], start_line: int,
                    max_lines: int = BODY_PREVIEW_LINES) -> str:
    """Get up to max_lines source starting at start_line (0-indexed).

    Never cuts mid-expression: if max_lines leaves unclosed brackets/parens/braces,
    extends up to 3 additional lines to close them.
    """
    end = min(start_line + max_lines, len(source_lines))
    if end < len(source_lines):
        depth = 0
        for i in range(start_line, end):
            for ch in source_lines[i]:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
        if depth > 0:
            for i in range(end, min(end + 3, len(source_lines))):
                end = i + 1
                for ch in source_lines[i]:
                    if ch in '([{':
                        depth += 1
                    elif ch in ')]}':
                        depth -= 1
                if depth <= 0:
                    break
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
    return _source_preview(source_lines, node.lineno - 1)


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
        self.parse_errors: list[str] = []
        self.cycles: list[list[str]] = []
        self._repo_path: str = ""

    # ── Build ──────────────────────────────────────────────────────────────

    # ── Build cache helpers ────────────────────────────────────────────

    def _file_cache_key(self, fpath: str) -> str:
        """Stable per-file cache key based on mtime + size."""
        s = os.stat(fpath)
        return f"{s.st_mtime:.0f}:{s.st_size}"

    def _load_build_cache(self, cache_file: str) -> dict:
        """Load file-level parse cache; returns {} on any error or version mismatch."""
        if not os.path.exists(cache_file):
            return {}
        try:
            with open(cache_file) as f:
                data = json.load(f)
            if data.get("version") != __version__:
                return {}
            return data.get("entries", {})
        except Exception:
            return {}

    def _save_build_cache(self, cache_file: str, entries: dict):
        """Persist per-file parse cache; silently skips on error."""
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"version": __version__, "entries": entries}, f)
        except Exception as e:
            logger.debug("Could not save build cache: %s", e)

    # ── Main build ─────────────────────────────────────────────────────

    def build(self, repo_path: str, exclude: list[str] | None = None,
              config: dict | None = None, strict_calls: bool = False,
              batch_size: int = 64) -> "RepoKG":
        """Parse repo, build nodes, resolve call edges, build FAISS index.

        Supports incremental caching: unchanged files are loaded from
        .repo2kg_cache/ instead of re-parsed, cutting rebuild time by 90%+.
        """
        repo_path = os.path.abspath(repo_path)
        self._repo_path = repo_path

        if config is None:
            config = _load_project_config(repo_path)

        build_cfg = config.get("build", {})
        embed_cfg = config.get("embeddings", {})

        exclude_patterns = list(DEFAULT_EXCLUDE)
        if exclude:
            exclude_patterns.extend(exclude)
        exclude_patterns.extend(build_cfg.get("exclude_dirs", []))

        max_file_bytes = build_cfg.get("max_file_size_kb", 500) * 1024
        _strict = strict_calls or build_cfg.get("strict_calls", False)
        _batch  = embed_cfg.get("batch_size", batch_size)

        logger.info("Scanning %s ...", repo_path)

        # Incremental build cache lives at {repo}/.repo2kg_cache/build_cache.json
        cache_dir  = os.path.join(repo_path, ".repo2kg_cache")
        cache_file = os.path.join(cache_dir, "build_cache.json")
        file_cache = self._load_build_cache(cache_file)
        new_entries: dict = {}

        supported_exts = set(LANG_EXTENSIONS.keys())
        source_files: list[str] = []
        lang_counts: dict[str, int] = {}

        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, pat) for pat in exclude_patterns)
            ]
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext not in supported_exts:
                    continue
                fpath = os.path.join(root, f)
                if _is_generated_file(fpath):
                    continue
                try:
                    if os.path.getsize(fpath) > max_file_bytes:
                        logger.debug("Skipping oversized file: %s", fpath)
                        continue
                except OSError:
                    continue
                source_files.append(fpath)
                lang_counts[LANG_EXTENSIONS[ext]] = lang_counts.get(LANG_EXTENSIONS[ext], 0) + 1

        if not source_files:
            logger.warning("No supported source files found in %s", repo_path)
            print(f"Warning: No supported source files found in {repo_path}")
            print(f"Supported extensions: {', '.join(sorted(supported_exts))}")
            return self

        cache_hits = 0
        _node_fields = {f.name for f in _dc_fields(CodeNode)}

        for fpath in source_files:
            rel = os.path.relpath(fpath, repo_path)
            key = self._file_cache_key(fpath)
            cached = file_cache.get(rel)

            if cached and cached.get("key") == key:
                try:
                    for nd in cached["nodes"]:
                        filtered = {k: v for k, v in nd.items() if k in _node_fields}
                        node = CodeNode(**filtered)
                        self.nodes[node.id] = node
                    new_entries[rel] = cached
                    cache_hits += 1
                    continue
                except Exception:
                    pass  # fall through to re-parse

            try:
                nodes = parse_file(fpath, repo_root=repo_path)
                for node in nodes:
                    self.nodes[node.id] = node
                new_entries[rel] = {"key": key, "nodes": [asdict(n) for n in nodes]}
            except Exception as e:
                err = f"{rel}: {type(e).__name__}: {e}"
                self.parse_errors.append(err)
                logger.error("Parse error in %s: %s", rel, e)

        if cache_hits:
            logger.info("Cache: %d/%d files reused", cache_hits, len(source_files))

        if not self.nodes:
            logger.warning("No parseable functions/classes found")
            print("Warning: No parseable functions/classes found")
            return self

        self._resolve_edges(strict_calls=_strict)
        self._detect_cycles()
        self._build_faiss(batch_size=_batch)
        self._save_build_cache(cache_file, new_entries)

        edge_count  = sum(len(n.calls) for n in self.nodes.values())
        lang_summary = ", ".join(f"{lang}: {c}" for lang, c in sorted(lang_counts.items()))
        logger.info("KG ready: %d nodes, %d edges from %d files (%s)",
                    len(self.nodes), edge_count, len(source_files), lang_summary)
        print(f"KG ready: {len(self.nodes)} nodes, {edge_count} call edges from {len(source_files)} files")
        if cache_hits:
            print(f"Cache: {cache_hits}/{len(source_files)} files reused from .repo2kg_cache/")
        print(f"Languages: {lang_summary}")
        if self.parse_errors:
            print(f"Parse errors: {len(self.parse_errors)} file(s) skipped (see .parse_errors.log)")
        if self.cycles:
            print(f"Cycles detected: {len(self.cycles)} strongly-connected component(s)")
        return self

    def _resolve_edges(self, strict_calls: bool = False):
        """Convert raw call names → node ids with import-aware disambiguation.

        Resolution priority:
          1. Same-file match (definitive)
          2. Node in a file explicitly imported by the caller's file
          3. First match (ambiguous) — dropped when strict_calls=True
        """
        name_index: dict[str, list[str]] = {}
        for node_id in self.nodes:
            name_index.setdefault(self.nodes[node_id].name, []).append(node_id)

        # Per-file set of imported module basenames, e.g. {"psycopg2", "socket", "utils"}
        file_import_bases: dict[str, set[str]] = {}
        for node in self.nodes.values():
            if node.file in file_import_bases:
                continue
            bases: set[str] = set()
            for imp in node.imports:
                # "psycopg2.connect" → "psycopg2"
                # "./utils" → "utils"
                part = imp.replace("\\", "/").split("/")[-1].split(".")[0].lstrip("_.")
                if part:
                    bases.add(part)
            file_import_bases[node.file] = bases

        # File basename → list of file paths in the KG
        base_to_files: dict[str, list[str]] = {}
        for node in self.nodes.values():
            b = os.path.splitext(os.path.basename(node.file))[0]
            if node.file not in base_to_files.get(b, []):
                base_to_files.setdefault(b, []).append(node.file)

        for node in self.nodes.values():
            caller_imports = file_import_bases.get(node.file, set())
            resolved: list[str] = []

            for raw_call in node.calls:
                matches = name_index.get(raw_call)
                if not matches:
                    continue
                if len(matches) == 1:
                    resolved.append(matches[0])
                    continue

                # 1. Prefer same-file match
                same_file = [m for m in matches if self.nodes[m].file == node.file]
                if same_file:
                    resolved.append(same_file[0])
                    continue

                # 2. Prefer nodes in explicitly imported files
                imported = [
                    m for m in matches
                    if os.path.splitext(os.path.basename(self.nodes[m].file))[0] in caller_imports
                ]
                if imported:
                    resolved.append(imported[0])
                    continue

                # 3. Ambiguous — skip under strict_calls, else take first
                if not strict_calls:
                    resolved.append(matches[0])

            node.calls = resolved

        # Populate callers (reverse edges), deduplicated
        for node in self.nodes.values():
            for callee_id in node.calls:
                if callee_id in self.nodes:
                    callers = self.nodes[callee_id].callers
                    if node.id not in callers:
                        callers.append(node.id)

    def _build_faiss(self, batch_size: int = 64):
        """Build FAISS index using batched encoding to avoid OOM on large repos."""
        ids = list(self.nodes.keys())
        summaries = [self.nodes[i].summary() for i in ids]

        batches = []
        for i in range(0, len(summaries), batch_size):
            batch_emb = self.model.encode(summaries[i:i + batch_size], show_progress_bar=False)
            batches.append(batch_emb)

        embeddings = _numpy.vstack(batches)
        dim = embeddings.shape[1]
        self._index = _faiss.IndexFlatL2(dim)
        self._index.add(_numpy.array(embeddings, dtype=_numpy.float32))
        self._index_ids = ids

    def _detect_cycles(self):
        """Kahn's algorithm: nodes that survive topological sort are cycle-free.
        Remaining nodes belong to SCCs (strongly-connected components = cycles).
        Annotates node.in_cycle and populates self.cycles.
        """
        node_ids = list(self.nodes.keys())
        adj: dict[str, list[str]] = {
            nid: [c for c in self.nodes[nid].calls if c in self.nodes]
            for nid in node_ids
        }
        in_deg: dict[str, int] = {nid: 0 for nid in node_ids}
        for nid in node_ids:
            for c in adj[nid]:
                in_deg[c] += 1

        queue: deque[str] = deque(nid for nid in node_ids if in_deg[nid] == 0)
        visited = 0
        while queue:
            v = queue.popleft()
            visited += 1
            for w in adj[v]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)

        cycle_members = {nid for nid in node_ids if in_deg[nid] > 0}
        for nid in cycle_members:
            self.nodes[nid].in_cycle = True

        # Extract connected components among cycle nodes
        cycles: list[list[str]] = []
        remaining = set(cycle_members)
        while remaining:
            start = next(iter(remaining))
            component: list[str] = []
            stack = [start]
            seen: set[str] = set()
            while stack:
                v = stack.pop()
                if v in seen or v not in remaining:
                    continue
                seen.add(v)
                component.append(v)
                for w in adj.get(v, []):
                    if w in remaining and w not in seen:
                        stack.append(w)
            cycles.append(component)
            remaining -= seen
        self.cycles = cycles

    def export_dependency_graph(self, out_path: str | None = None,
                                kg_path: str | None = None) -> dict:
        """Export module-level import dependency graph.

        Returns a dict with 'modules', 'edges', and 'top_imported' keys.
        Saves to {kg_path}.deps.json when kg_path is provided.
        """
        if out_path is None and kg_path:
            base = _re.sub(r"\.(json|toon)$", "", kg_path)
            out_path = base + ".deps.json"

        all_files = sorted({n.file for n in self.nodes.values()})
        file_imports: dict[str, set[str]] = {}
        for node in self.nodes.values():
            file_imports.setdefault(node.file, set()).update(node.imports)

        edges: list[dict] = []
        import_counts: dict[str, int] = {}
        file_set = set(all_files)

        for src, imports in file_imports.items():
            for imp in imports:
                # Match import string to a KG file via normalized basename
                imp_base = imp.replace(".", "/")
                for ext in (".py", ".js", ".ts", ".tsx", ".jsx"):
                    candidate = imp_base + ext
                    if candidate in file_set:
                        edges.append({"from": src, "to": candidate, "via": imp})
                        import_counts[candidate] = import_counts.get(candidate, 0) + 1
                        break

        fan_in:  dict[str, int] = {}
        fan_out: dict[str, int] = {}
        for e in edges:
            fan_out[e["from"]] = fan_out.get(e["from"], 0) + 1
            fan_in[e["to"]]    = fan_in.get(e["to"],    0) + 1

        modules = [
            {"file": f, "fan_in": fan_in.get(f, 0), "fan_out": fan_out.get(f, 0)}
            for f in all_files
        ]
        top_imported = sorted(import_counts, key=lambda k: -import_counts[k])[:10]
        result = {"modules": modules, "edges": edges, "top_imported": top_imported}

        if out_path:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Dependency graph saved: {out_path}")

        return result

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
        base = _re.sub(r"\.(json|toon)$", "", path)
        _faiss.write_index(self._index, base + ".faiss")
        with open(base + ".idx", "w") as f:
            json.dump(self._index_ids, f)
        print(f"Saved to {path} [{fmt.upper()}] ({len(self.nodes)} nodes)")

        # Write parse errors log (clear stale log if no errors this run)
        errors_log = path + ".parse_errors.log"
        if self.parse_errors:
            with open(errors_log, "w") as f:
                f.write(f"# repo2kg parse errors — {len(self.parse_errors)} file(s) skipped\n")
                for err in self.parse_errors:
                    f.write(err + "\n")
        elif os.path.exists(errors_log):
            os.remove(errors_log)

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "RepoKG":
        """Load a saved KG. Results are cached in-memory keyed by path+mtime."""
        path = os.path.abspath(path)

        # In-memory cache: skip full deserialization if file unchanged
        try:
            mtime = os.path.getmtime(path)
            cache_key = f"{path}:{mtime}"
        except OSError:
            cache_key = None

        if cache_key:
            with _KG_CACHE_LOCK:
                cached = _KG_CACHE.get(cache_key)
            if cached is not None:
                return cached

        base = _re.sub(r"\.(json|toon)$", "", path)
        for required in [path, base + ".faiss", base + ".idx"]:
            if not os.path.exists(required):
                raise FileNotFoundError(f"Missing KG file: {required}")
        kg = cls(model_name)
        fmt = _detect_format(path)
        _node_fields = {f.name for f in _dc_fields(CodeNode)}
        if fmt == "toon":
            with open(path) as f:
                data = deserialize_toon(f.read())
        else:
            with open(path) as f:
                data = json.load(f)
        for nid, ndata in data.items():
            filtered = {k: v for k, v in ndata.items() if k in _node_fields}
            kg.nodes[nid] = CodeNode(**filtered)
        kg._index = _faiss.read_index(base + ".faiss")
        with open(base + ".idx") as f:
            kg._index_ids = json.load(f)
        print(f"Loaded KG: {len(kg.nodes)} nodes [{fmt.upper()}]")

        if cache_key:
            with _KG_CACHE_LOCK:
                _KG_CACHE[cache_key] = kg
        return kg

    @classmethod
    def clear_cache(cls):
        """Invalidate the in-memory KG load cache."""
        with _KG_CACHE_LOCK:
            _KG_CACHE.clear()

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


# Module-level in-memory cache for RepoKG.load()
_KG_CACHE: dict[str, "RepoKG"] = {}
_KG_CACHE_LOCK = threading.Lock()


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
        _node_fields = {f.name for f in _dc_fields(CodeNode)}
        for nid, ndata in data.items():
            filtered = {k: v for k, v in ndata.items() if k in _node_fields}
            self.nodes[nid] = CodeNode(**filtered)
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
    _node_fields = {f.name for f in _dc_fields(CodeNode)}
    nodes = {nid: CodeNode(**{k: v for k, v in d.items() if k in _node_fields})
             for nid, d in data.items()}

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
        lines.append(f"- **{fpath}** -- {node_names}")
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
            lines.append(f"- `{n.name}` in {n.file} -- calls {len(n.calls)} functions")
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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    token_estimate = len(content.split())
    print(f"Exported {out_path} ({len(nodes)} nodes, ~{token_estimate} tokens)")


def generate_summary_md(kg_path: str, out_path: str | None = None) -> str:
    """Compact project summary: directory tree + top-level classes & functions only.

    Omits methods entirely. Produces ~10× fewer tokens than CODEBASE.md while still
    giving an agent the full structural map (what files exist, what lives in each).
    Follow up with ``repo2kg query`` for deep dives into any symbol.
    """
    fmt = _detect_format(kg_path)
    with open(kg_path) as f:
        if fmt == "toon":
            data = deserialize_toon(f.read())
        else:
            data = json.load(f)
    _node_fields = {f.name for f in _dc_fields(CodeNode)}
    nodes = {nid: CodeNode(**{k: v for k, v in d.items() if k in _node_fields})
             for nid, d in data.items()}

    # Per-file symbol list: classes first, then standalone functions — no methods
    files: dict[str, list[str]] = {}
    for n in sorted(nodes.values(), key=lambda x: x.name):
        if n.kind == "method":
            continue
        files.setdefault(n.file, []).append(
            n.name + (" (class)" if n.kind == "class" else "")
        )

    all_files = sorted(files.keys())
    class_count  = sum(1 for n in nodes.values() if n.kind == "class")
    func_count   = sum(1 for n in nodes.values() if n.kind == "function")
    method_count = sum(1 for n in nodes.values() if n.kind == "method")
    edge_count   = sum(len(n.calls) for n in nodes.values())

    lines = [
        "# Project Summary",
        "",
        "> Generated by repo2kg. Use `repo2kg query` for deep dives into any symbol.",
        "",
        "## Stats",
        "",
        f"{len(all_files)} files · {class_count} classes · {func_count} functions · "
        f"{method_count} methods · {edge_count} call edges",
        "",
        "## File Map",
        "> Classes and top-level functions only. Methods omitted — use `repo2kg query` to explore them.",
        "",
    ]

    # Group by directory so paths aren't repeated on every line
    dir_files: dict[str, list[str]] = {}
    for fpath in all_files:
        d = os.path.dirname(fpath) or "."
        dir_files.setdefault(d, []).append(fpath)

    for dir_path in sorted(dir_files.keys()):
        lines.append(f"**{dir_path}/**")
        for fpath in sorted(dir_files[dir_path]):
            basename = os.path.basename(fpath)
            syms = files.get(fpath, [])
            if syms:
                lines.append(f"  {basename:<32}→ {', '.join(syms)}")
            else:
                lines.append(f"  {basename}")
        lines.append("")

    # Entry points: nodes that call others but are called by no-one
    entry_candidates = [
        n for n in nodes.values()
        if n.kind != "method" and not n.callers and n.calls
    ]
    entry_candidates.sort(key=lambda x: -len(x.calls))
    if entry_candidates:
        lines.extend(["## Likely Entry Points", ""])
        for n in entry_candidates[:12]:
            lines.append(f"- `{n.name}` ({n.file}) — calls {len(n.calls)} symbols")
        lines.append("")

    # Circular dependencies
    cycle_nodes = [n for n in nodes.values() if getattr(n, "in_cycle", False)]
    if cycle_nodes:
        cycle_files = sorted({n.file for n in cycle_nodes})
        lines.extend(["## Circular Dependencies", ""])
        lines.append(f"{len(cycle_nodes)} nodes in cycles across {len(cycle_files)} file(s):")
        for f in cycle_files[:10]:
            lines.append(f"  - {f}")
        if len(cycle_files) > 10:
            lines.append(f"  - … and {len(cycle_files) - 10} more")
        lines.append("")

    # Parse errors (read from .parse_errors.log if present)
    errors_log = kg_path + ".parse_errors.log"
    if os.path.exists(errors_log):
        with open(errors_log) as ef:
            error_lines = [l.strip() for l in ef if l.strip() and not l.startswith("#")]
        if error_lines:
            lines.extend(["## Parse Errors (Blind Spots)", ""])
            lines.append(f"> {len(error_lines)} file(s) failed to parse — KG may be incomplete.")
            for e in error_lines[:10]:
                lines.append(f"  - `{e}`")
            if len(error_lines) > 10:
                lines.append(f"  - … and {len(error_lines) - 10} more (see {errors_log})")
            lines.append("")

    content = "\n".join(lines)
    token_estimate = len(content) // 4  # chars/4 ≈ tokens

    lines.extend([
        "---",
        f"> *repo2kg summary — {len(all_files)} files, ~{token_estimate} tokens.*",
        f"> *`repo2kg query \"<topic>\" --kg <kg_file>` for targeted semantic search.*",
    ])

    content = "\n".join(lines)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Exported {out_path} ({len(all_files)} files, ~{token_estimate} tokens)")
    else:
        print(content)
        print(f"\n# ~{token_estimate} tokens | {len(all_files)} files", file=sys.stderr)

    return content


# ─────────────────────────────────────────────
# VISUAL GRAPH (interactive HTML for humans)
# ─────────────────────────────────────────────

_D3_CDN = "https://d3js.org/d3.v7.min.js"
_D3_CACHE_PATH = REPO2KG_HOME / "d3.v7.min.js"


def _get_d3_script() -> str:
    """Return D3 v7 as an inline <script> block. Caches locally after first download."""
    # Use cached copy if available
    if _D3_CACHE_PATH.exists():
        return f"<script>\n{_D3_CACHE_PATH.read_text(encoding='utf-8')}\n</script>"
    # Try to download and cache
    try:
        import urllib.request
        print("Downloading D3.js for offline use (cached to ~/.repo2kg/d3.v7.min.js)…")
        with urllib.request.urlopen(_D3_CDN, timeout=15) as resp:  # noqa: S310
            js = resp.read().decode("utf-8")
        _D3_CACHE_PATH.write_text(js, encoding="utf-8")
        return f"<script>\n{js}\n</script>"
    except Exception:
        # Fall back to CDN link — graph will need internet to render
        print("Warning: could not download D3.js; falling back to CDN (requires internet).")
        return f'<script src="{_D3_CDN}"></script>'


def generate_visual_graph(kg_path: str, out_path: str, max_nodes: int = 800) -> None:
    """
    Generate a self-contained interactive HTML knowledge-graph visualization.

    Opens in any browser — no server needed.  Uses D3 v7 (inlined, no CDN required).
    """
    fmt = _detect_format(kg_path)
    with open(kg_path) as f:
        data = deserialize_toon(f.read()) if fmt == "toon" else json.load(f)

    nodes_dict = {nid: CodeNode(**nd) for nid, nd in data.items()}
    title = os.path.splitext(os.path.basename(kg_path))[0]

    # ── Filter to user-written code only (drop 3rd-party / bundled files) ──
    _THIRD_PARTY_PATTERNS = (
        ".vite/", "/.vite/",
        "node_modules/",
        "site-packages/",
        "/dist/", "\\dist\\",
        "/build/", "\\build\\",
        "/.next/", "/.nuxt/",
        "/out/", "/coverage/",
        "/vendor/",
        "/.cache/",
        "<builtin",
        "<frozen",
        "<string>",
    )
    def _is_user_file(filepath: str) -> bool:
        if not filepath:
            return False
        fp = filepath.replace("\\", "/")
        return not any(pat in fp for pat in _THIRD_PARTY_PATTERNS)

    nodes_dict = {nid: n for nid, n in nodes_dict.items() if _is_user_file(n.file)}

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

    # ── Build file hub nodes (one big node per unique source file) ──────────
    file_map: dict[str, str] = {}   # file_path → synthetic file-node id
    for n in nodes_dict.values():
        if n.file not in file_map:
            file_map[n.file] = f"__file__{n.file}"

    nodes_json: list[dict] = []
    # File hub nodes first
    for fpath, fid in file_map.items():
        sym_count = sum(1 for n in nodes_dict.values() if n.file == fpath)
        nodes_json.append({
            "id": fid,
            "name": os.path.basename(fpath),
            "kind": "file",
            "file": fpath,
            "parent_class": "",
            "signature": fpath,
            "docstring": f"{sym_count} symbols",
            "calls": [],
            "callers": [],
            "sym_count": sym_count,
        })
    # Symbol nodes (class / function / method)
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
            "sym_count": 0,
        })

    links_json: list[dict] = []
    # Containment links: file hub → each of its symbols  (ltype "c")
    for n in nodes_dict.values():
        fid = file_map.get(n.file)
        if fid:
            links_json.append({"source": fid, "target": n.id, "ltype": "c"})
    # Call links: symbol → symbol  (ltype "e")
    seen_links: set[tuple] = set()
    for n in nodes_dict.values():
        for callee_id in n.calls:
            if callee_id in keep_ids:
                key = (n.id, callee_id)
                if key not in seen_links:
                    seen_links.add(key)
                    links_json.append({"source": n.id, "target": callee_id, "ltype": "e"})

    file_count  = len(file_map)
    sym_count   = len(nodes_dict)
    graph_data  = json.dumps({"nodes": nodes_json, "links": links_json}, ensure_ascii=False)
    node_count  = len(nodes_json)
    link_count  = sum(1 for lk in links_json if lk["ltype"] == "e")   # display only call edges

    # Simulation tuning
    alpha_decay = 0.025
    contain_dist = 70   if sym_count > 400 else 90     # file → symbol distance
    call_dist    = 280  if sym_count > 400 else 380    # symbol → symbol call distance

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>repo2kg — {title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background: #080c15; color: #f1f5f9; overflow: hidden; }}

/* ── TOPBAR ── */
#topbar {{ position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 30;
  background: rgba(8,12,21,0.85); backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255,255,255,0.07);
  display: flex; align-items: center; gap: 14px; padding: 0 22px; }}
#logo {{ font-size: 14px; font-weight: 600; color: #f8fafc; display: flex; align-items: center; gap: 7px; flex-shrink: 0; }}
#logo svg {{ width: 16px; height: 16px; color: #6366f1; }}
#search-wrap {{ position: relative; flex: 1; max-width: 300px; margin-left: 10px; }}
#search-wrap svg {{ position: absolute; left: 10px; top: 50%; transform: translateY(-50%); width: 13px; height: 13px; color: #64748b; pointer-events: none; }}
#search {{ width: 100%; background: #111827; border: 1px solid rgba(255,255,255,0.07); border-radius: 6px;
  color: #f1f5f9; padding: 5px 12px 5px 30px; font-size: 12px; font-family: 'Inter'; outline: none; transition: all 0.2s; }}
#search:focus {{ border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,0.15); }}
#search::placeholder {{ color: #4b5563; }}

.kbtn {{ display: flex; align-items: center; gap: 5px; background: transparent;
  border: 1px solid rgba(255,255,255,0.08); border-radius: 5px;
  color: #64748b; padding: 4px 10px; cursor: pointer; font-size: 11.5px; font-weight: 500;
  transition: all 0.15s; flex-shrink: 0; }}
.kbtn .dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
.kbtn[data-k="class"] .dot {{ background: #60a5fa; }}
.kbtn[data-k="function"] .dot {{ background: #34d399; }}
.kbtn[data-k="method"] .dot {{ background: #fb923c; }}
.kbtn.on {{ background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.15); color: #e2e8f0; }}
#stats {{ margin-left: auto; font-size: 11px; color: #374151; flex-shrink: 0; }}
#fit-btn {{ background: transparent; border: 1px solid rgba(255,255,255,0.08); border-radius: 5px;
  color: #64748b; padding: 4px 10px; cursor: pointer; font-size: 11.5px; font-weight: 500;
  transition: all 0.15s; display: flex; align-items: center; gap: 5px; flex-shrink: 0; }}
#fit-btn:hover {{ background: rgba(255,255,255,0.06); color: #e2e8f0; }}
#fit-btn svg {{ width: 13px; height: 13px; }}

/* ── PANEL ── */
#panel {{ position: fixed; right: 0; top: 52px; bottom: 0; width: 320px; z-index: 20;
  background: #0a0e1a; border-left: 1px solid rgba(255,255,255,0.07);
  box-shadow: -8px 0 24px rgba(0,0,0,0.4); overflow-y: auto;
  transform: translateX(100%); transition: transform 0.28s cubic-bezier(0.16,1,0.3,1); }}
#panel.open {{ transform: translateX(0); }}
#panel-hdr {{ position: sticky; top: 0; background: #0a0e1a; border-bottom: 1px solid rgba(255,255,255,0.07);
  padding: 18px 20px 14px; z-index: 2; display: flex; justify-content: space-between; align-items: flex-start; }}
#close-panel {{ background: none; border: none; color: #4b5563; cursor: pointer; padding: 3px; border-radius: 4px; transition: all 0.15s; }}
#close-panel:hover {{ background: rgba(255,255,255,0.08); color: #e2e8f0; }}
#close-panel svg {{ width: 16px; height: 16px; }}
#panel-body {{ padding: 0 20px 40px; }}
.p-name {{ font-size: 15px; font-weight: 600; color: #f8fafc; margin-bottom: 6px; line-height: 1.3; word-break: break-word; padding-right: 20px; }}
.badge {{ display: inline-flex; align-items: center; border-radius: 3px; padding: 2px 7px; font-size: 10px; font-weight: 700; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.6px; }}
.badge.class {{ background: rgba(96,165,250,0.1); color: #93c5fd; border: 1px solid rgba(96,165,250,0.2); }}
.badge.function {{ background: rgba(52,211,153,0.1); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.2); }}
.badge.method {{ background: rgba(251,146,60,0.1); color: #fdba74; border: 1px solid rgba(251,146,60,0.2); }}
.badge.file {{ background: rgba(99,102,241,0.1); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }}
.psec {{ margin-top: 18px; }}
.psec-title {{ font-size: 10px; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; display: flex; align-items: center; gap: 5px; }}
.psec-title svg {{ width: 11px; height: 11px; }}
.code-block {{ font-family: 'SFMono-Regular', Consolas, monospace; font-size: 11px; color: #c4b5fd;
  background: #111827; border: 1px solid rgba(99,102,241,0.12); border-radius: 5px; padding: 10px; white-space: pre-wrap; word-break: break-all; line-height: 1.5; }}
.path-block {{ font-family: monospace; font-size: 11px; color: #6b7280; word-break: break-all; line-height: 1.5; }}
.doc-block {{ font-size: 12px; color: #9ca3af; line-height: 1.6; }}
.sym-list {{ list-style: none; display: flex; flex-direction: column; gap: 4px; }}
.sym-list li {{ font-size: 11px; color: #94a3b8; padding: 6px 10px; border-radius: 5px;
  background: #111827; border: 1px solid rgba(255,255,255,0.04); font-family: monospace;
  cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 7px; word-break: break-all; }}
.sym-list li:hover {{ border-color: rgba(99,102,241,0.35); background: #1c2640; color: #f1f5f9; }}
.sym-list li svg {{ width: 12px; height: 12px; flex-shrink: 0; opacity: 0.7; }}
.sym-list li.ck svg {{ color: #60a5fa; }}
.sym-list li.fk svg {{ color: #34d399; }}
.sym-list li.mk svg {{ color: #fb923c; }}

/* ── SVG ── */
#kg-svg {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; cursor: grab; }}
#kg-svg.dragging {{ cursor: grabbing; }}

/* Grid background */
.bg-grid {{ fill: url(#gridPat); }}
.bg-dots {{ fill: url(#dotPat); }}

/* Links */
.link-c {{ fill: none; stroke: rgba(99,102,241,0.08); stroke-width: 1; stroke-dasharray: 3,5; }}
.link-e {{ fill: none; stroke: rgba(148,163,184,0.14); stroke-width: 1.4; }}
.link.hl {{ stroke: rgba(99,102,241,0.65) !important; stroke-width: 1.8 !important; stroke-dasharray: none !important; }}
.link.dim {{ opacity: 0.04 !important; }}

/* File hub nodes */
.file-aura {{ pointer-events: none; }}
.file-circle {{ transition: all 0.2s; }}
.node-g[data-kind="file"]:hover .file-circle {{ filter: brightness(1.3); }}
.node-g.sel[data-kind="file"] .file-circle {{ stroke: #818cf8 !important; stroke-width: 2.5px; filter: drop-shadow(0 0 12px rgba(99,102,241,0.6)); }}
.file-label {{ font-size: 10.5px; fill: #94a3b8; font-family: 'Inter'; font-weight: 500; text-anchor: middle; pointer-events: none; transition: fill 0.15s; }}
.node-g[data-kind="file"]:hover .file-label, .node-g.sel[data-kind="file"] .file-label {{ fill: #f8fafc; }}
.file-count {{ font-size: 11px; fill: #a5b4fc; font-family: 'Inter'; font-weight: 600; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
.node-g[data-kind="file"].match .file-circle {{ stroke: #fde047 !important; stroke-width: 2px !important; }}

/* Symbol nodes (pill) */
.node-g {{ cursor: pointer; }}
.node-g.dim {{ opacity: 0.12; }}
.node-rect {{ fill: #111827; stroke-width: 1px; transition: fill 0.15s, filter 0.15s; }}
.node-g[data-kind="class"]    .node-rect {{ stroke: rgba(96,165,250,0.4); }}
.node-g[data-kind="function"] .node-rect {{ stroke: rgba(52,211,153,0.4); }}
.node-g[data-kind="method"]   .node-rect {{ stroke: rgba(251,146,60,0.4); }}
.node-g:hover                 .node-rect {{ fill: #1c2640; }}
.node-g.sel                   .node-rect {{ fill: #1e293b; stroke-width: 2px; }}
.node-g.sel[data-kind="class"]    .node-rect {{ stroke: #60a5fa; filter: drop-shadow(0 0 5px rgba(96,165,250,0.4)); }}
.node-g.sel[data-kind="function"] .node-rect {{ stroke: #34d399; filter: drop-shadow(0 0 5px rgba(52,211,153,0.4)); }}
.node-g.sel[data-kind="method"]   .node-rect {{ stroke: #fb923c; filter: drop-shadow(0 0 5px rgba(251,146,60,0.4)); }}
.node-g.match .node-rect {{ stroke: #fde047 !important; stroke-width: 2px !important; }}

.node-icon-bg {{ opacity: 0.12; }}
.node-g[data-kind="class"]    .node-icon-bg {{ fill: #60a5fa; }}
.node-g[data-kind="function"] .node-icon-bg {{ fill: #34d399; }}
.node-g[data-kind="method"]   .node-icon-bg {{ fill: #fb923c; }}

.node-icon {{ font-size: 9.5px; font-weight: 700; font-family: 'Inter'; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
.node-g[data-kind="class"]    .node-icon {{ fill: #93c5fd; }}
.node-g[data-kind="function"] .node-icon {{ fill: #6ee7b7; }}
.node-g[data-kind="method"]   .node-icon {{ fill: #fdba74; }}

.node-text {{ fill: #94a3b8; font-size: 10.5px; font-family: 'Inter'; font-weight: 500; dominant-baseline: central; pointer-events: none; }}
.node-g:hover .node-text, .node-g.sel .node-text {{ fill: #f1f5f9; }}
</style>
</head>
<body>

<svg style="display:none">
  <defs>
    <symbol id="i-repo"   viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></symbol>
    <symbol id="i-search" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></symbol>
    <symbol id="i-fit"    viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></symbol>
    <symbol id="i-x"      viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></symbol>
    <symbol id="i-file"   viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></symbol>
    <symbol id="i-code"   viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></symbol>
    <symbol id="i-doc"    viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="21" y1="6" x2="3" y2="6"/><line x1="15" y1="12" x2="3" y2="12"/><line x1="17" y1="18" x2="3" y2="18"/></symbol>
    <symbol id="i-out"    viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></symbol>
    <symbol id="i-in"     viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></symbol>
  </defs>
</svg>

<div id="topbar">
  <div id="logo"><svg><use href="#i-repo"/></svg>repo2kg</div>
  <div id="search-wrap">
    <svg><use href="#i-search"/></svg>
    <input id="search" type="text" placeholder="Search..." autocomplete="off" spellcheck="false">
  </div>
  <button class="kbtn on" data-k="class"><span class="dot"></span>Classes</button>
  <button class="kbtn on" data-k="function"><span class="dot"></span>Functions</button>
  <button class="kbtn on" data-k="method"><span class="dot"></span>Methods</button>
  <span id="stats">{file_count} files · {sym_count} symbols · {link_count} calls</span>
  <button id="fit-btn"><svg><use href="#i-fit"/></svg>Fit</button>
</div>

<div id="panel">
  <div id="panel-hdr">
    <div>
      <div id="p-name" class="p-name"></div>
      <span id="p-badge" class="badge"></span>
    </div>
    <button id="close-panel"><svg><use href="#i-x"/></svg></button>
  </div>
  <div id="panel-body"></div>
</div>

<svg id="kg-svg">
  <defs>
    <pattern id="gridPat" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M40 0L0 0 0 40" fill="none" stroke="rgba(255,255,255,0.025)" stroke-width="1"/>
    </pattern>
    <pattern id="dotPat" width="20" height="20" patternUnits="userSpaceOnUse">
      <circle cx="10" cy="10" r="1.2" fill="rgba(255,255,255,0.04)"/>
    </pattern>
    <radialGradient id="file-grad" cx="40%" cy="35%" r="65%">
      <stop offset="0%"   stop-color="#1e1b4b"/>
      <stop offset="100%" stop-color="#080c15"/>
    </radialGradient>
    <marker id="arr" viewBox="0 -4 8 8" refX="22" refY="0" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,-3L6,0L0,3" fill="rgba(148,163,184,0.3)"/>
    </marker>
    <marker id="arr-hl" viewBox="0 -4 8 8" refX="22" refY="0" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,-3L6,0L0,3" fill="rgba(99,102,241,0.8)"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" class="bg-grid"/>
  <rect width="100%" height="100%" class="bg-dots"/>
  <g id="zg">
    <g id="lg"></g>
    <g id="ng"></g>
  </g>
</svg>

{_get_d3_script()}
<script>
const DATA = {graph_data};
const CONTAIN_DIST = {contain_dist};
const CALL_DIST    = {call_dist};
const ALPHA_DECAY  = {alpha_decay};

const NODE_H = 26, ICON_R = 13, PAD_R = 10;

let active = new Set(["class","function","method"]);
let sel = null, searchQ = "";

const W = window.innerWidth, H = window.innerHeight;
const svg   = d3.select("#kg-svg");
const gRoot = svg.select("#zg");

/* ── Zoom ── */
const zoom = d3.zoom().scaleExtent([0.05, 8]).on("zoom", e => {{
  gRoot.attr("transform", e.transform);
  svg.classed("dragging", e.sourceEvent && e.sourceEvent.buttons === 1);
}});
svg.call(zoom).on("dblclick.zoom", null);
svg.on("click", e => {{
  if (e.target === svg.node() || e.target.tagName === "rect" || e.target.tagName === "circle")
    {{ sel = null; closePanel(); paint(); }}
}});

/* ── File node radius (scales with symbol count) ── */
function fileR(d) {{
  return Math.max(24, Math.min(48, 18 + (d.sym_count || 1) * 1.2));
}}

/* ── Symbol pill width ── */
function pillW(name) {{
  const n = Math.min(name.length, 28);
  return ICON_R * 2 + n * 6.2 + PAD_R;
}}

/* ── Simulation ── */
const sim = d3.forceSimulation()
  .alphaDecay(ALPHA_DECAY)
  .velocityDecay(0.45)
  .force("link", d3.forceLink().id(d => d.id)
    .distance(d => d.ltype === "c" ? CONTAIN_DIST : CALL_DIST)
    .strength(d => d.ltype === "c" ? 0.9 : 0.07))
  .force("charge", d3.forceManyBody()
    .strength(d => d.kind === "file" ? -1800 : -60)
    .distanceMax(1200))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("coll", d3.forceCollide(d => d.kind === "file" ? fileR(d) + 18 : pillW(d.name) / 2 + 8))
  .force("x", d3.forceX(W / 2).strength(0.018))
  .force("y", d3.forceY(H / 2).strength(0.018));

let linkSel = d3.select(null), nodeSel = d3.select(null);

/* ── Visible nodes (file nodes always shown) ── */
function visible() {{
  return DATA.nodes.filter(n => n.kind === "file" || active.has(n.kind));
}}

/* ── Render ── */
function render() {{
  const vis    = visible();
  const visSet = new Set(vis.map(n => n.id));
  const visLinks = DATA.links.filter(l => {{
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return visSet.has(s) && visSet.has(t);
  }});

  /* Links */
  const lJoin = gRoot.select("#lg").selectAll("path.link")
    .data(visLinks, d => `${{(d.source.id||d.source)}}→${{(d.target.id||d.target)}}`);
  lJoin.enter().append("path")
    .attr("class", d => `link link-${{d.ltype}}`)
    .attr("marker-end", d => d.ltype === "e" ? "url(#arr)" : null);
  lJoin.exit().remove();
  linkSel = gRoot.select("#lg").selectAll("path.link");

  /* Nodes */
  const nJoin = gRoot.select("#ng").selectAll("g.node-g").data(vis, d => d.id);
  const nEnter = nJoin.enter().append("g")
    .attr("class", "node-g")
    .attr("data-kind", d => d.kind)
    .call(d3.drag()
      .on("start", (e, d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
      .on("drag",  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
      .on("end",   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}))
    .on("click", (e, d) => {{
      e.stopPropagation();
      sel = sel === d.id ? null : d.id;
      if (sel) openPanel(d); else closePanel();
      paint();
    }});

  /* Build per-node shapes */
  nEnter.each(function(d) {{
    const g = d3.select(this);
    if (d.kind === "file") {{
      const r = fileR(d);
      /* Aura ring */
      g.append("circle").attr("class", "file-aura")
        .attr("r", r + 14)
        .attr("fill", "rgba(99,102,241,0.03)")
        .attr("stroke", "rgba(99,102,241,0.1)")
        .attr("stroke-width", 1);
      /* Main circle */
      g.append("circle").attr("class", "file-circle")
        .attr("r", r)
        .attr("fill", "url(#file-grad)")
        .attr("stroke", "#4338ca")
        .attr("stroke-width", 1.5);
      /* Symbol count */
      g.append("text").attr("class", "file-count")
        .attr("y", 0)
        .text(d.sym_count || "");
      /* Filename label below */
      g.append("text").attr("class", "file-label")
        .attr("y", r + 13)
        .text(d.name.length > 24 ? d.name.slice(0, 22) + "…" : d.name);
    }} else {{
      const w = pillW(d.name);
      g.append("rect").attr("class", "node-rect")
        .attr("rx", NODE_H / 2).attr("ry", NODE_H / 2)
        .attr("height", NODE_H).attr("y", -NODE_H / 2)
        .attr("width", w).attr("x", -ICON_R);
      g.append("circle").attr("class", "node-icon-bg").attr("r", ICON_R);
      g.append("text").attr("class", "node-icon").text(d.kind[0].toUpperCase());
      g.append("text").attr("class", "node-text")
        .attr("x", ICON_R + 5)
        .text(d.name.length > 28 ? d.name.slice(0, 26) + "…" : d.name);
    }}
  }});

  nJoin.exit().remove();
  nodeSel = gRoot.select("#ng").selectAll("g.node-g");

  sim.nodes(vis).on("tick", () => {{
    linkSel.attr("d", d => {{
      const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
      const mx = (sx + tx) / 2, my = (sy + ty) / 2;
      return `M${{sx}},${{sy}} Q${{mx}},${{my}} ${{tx}},${{ty}}`;
    }});
    nodeSel.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
  }});
  sim.force("link").links(visLinks);
  sim.alpha(0.7).restart();
  paint();
}}

/* ── Connected IDs ── */
function connectedIds(id) {{
  const ids = new Set([id]);
  linkSel.each(d => {{
    const s = d.source.id || d.source, t = d.target.id || d.target;
    if (s === id) ids.add(t);
    if (t === id) ids.add(s);
  }});
  return ids;
}}

/* ── Paint ── */
function paint() {{
  const q = searchQ.toLowerCase().trim();

  if (!sel && !q) {{
    nodeSel.classed("sel", false).classed("dim", false).classed("match", false);
    linkSel.classed("hl", false).classed("dim", false).attr("marker-end", d => d.ltype === "e" ? "url(#arr)" : null);
    return;
  }}

  if (q && !sel) {{
    const hits = new Set(DATA.nodes.filter(n =>
      n.name.toLowerCase().includes(q) || n.file.toLowerCase().includes(q)).map(n => n.id));
    // Also highlight file nodes that contain a hit symbol
    DATA.nodes.filter(n => n.kind === "file").forEach(fn => {{
      if (DATA.nodes.some(sn => sn.kind !== "file" && sn.file === fn.file && hits.has(sn.id)))
        hits.add(fn.id);
    }});
    nodeSel.classed("match", d => hits.has(d.id))
           .classed("dim",   d => d.kind !== "file" && !hits.has(d.id))
           .classed("sel", false);
    linkSel.classed("hl", false).classed("dim", true).attr("marker-end", d => d.ltype === "e" ? "url(#arr)" : null);
    return;
  }}

  const nb = connectedIds(sel);
  nodeSel.classed("sel",   d => d.id === sel)
         .classed("dim",   d => d.kind !== "file" && !nb.has(d.id))
         .classed("match", false);
  linkSel.classed("hl",  d => {{ const s = d.source.id||d.source, t = d.target.id||d.target; return s === sel || t === sel; }})
         .classed("dim", d => {{ const s = d.source.id||d.source, t = d.target.id||d.target; return !(s === sel || t === sel); }})
         .attr("marker-end", d => {{
           const s = d.source.id||d.source, t = d.target.id||d.target;
           return (s === sel || t === sel) ? "url(#arr-hl)" : (d.ltype === "e" ? "url(#arr)" : null);
         }});
}}

/* ── Panel ── */
function esc(s) {{ const d = document.createElement("div"); d.textContent = String(s||""); return d.innerHTML; }}

function openPanel(d) {{
  document.getElementById("p-name").textContent = d.name;
  const badge = document.getElementById("p-badge");
  badge.className = "badge " + d.kind;
  badge.textContent = d.kind;

  let html = "";

  if (d.kind === "file") {{
    const symbols = DATA.nodes.filter(n => n.file === d.file && n.kind !== "file");
    const classes  = symbols.filter(n => n.kind === "class");
    const funcs    = symbols.filter(n => n.kind === "function");
    const methods  = symbols.filter(n => n.kind === "method");

    html += `<div class="psec"><div class="psec-title"><svg><use href="#i-file"/></svg>File</div>
      <div class="path-block">${{esc(d.file)}}</div></div>`;

    for (const [label, items, cls, icon] of [
      ["Classes",   classes, "ck", "i-code"],
      ["Functions", funcs,   "fk", "i-code"],
      ["Methods",   methods, "mk", "i-code"],
    ]) {{
      if (!items.length) continue;
      html += `<div class="psec"><div class="psec-title"><svg><use href="#${{icon}}"/></svg>${{label}} (${{items.length}})</div>
        <ul class="sym-list">${{items.slice(0, 25).map(n =>
          `<li class="${{cls}}" onclick="clickSym('${{esc(n.id)}}')"><svg><use href="#i-code"/></svg>${{esc(n.name)}}</li>`
        ).join("")}}${{items.length > 25 ? `<li style="color:#4b5563;cursor:default">... ${{items.length-25}} more</li>` : ""}}</ul></div>`;
    }}
  }} else {{
    html += `<div class="psec"><div class="psec-title"><svg><use href="#i-file"/></svg>File</div>
      <div class="path-block">${{esc(d.file)}}</div></div>`;

    if (d.signature) html += `<div class="psec"><div class="psec-title"><svg><use href="#i-code"/></svg>Signature</div>
      <div class="code-block">${{esc((d.signature||"").slice(0, 250))}}</div></div>`;

    if (d.docstring) html += `<div class="psec"><div class="psec-title"><svg><use href="#i-doc"/></svg>Description</div>
      <div class="doc-block">${{esc((d.docstring||"").slice(0, 400))}}</div></div>`;

    const calls   = (d.calls||[]).slice(0, 12);
    const callers = (d.callers||[]).slice(0, 12);

    if (calls.length) html += `<div class="psec"><div class="psec-title"><svg><use href="#i-out"/></svg>Calls (${{d.calls.length}})</div>
      <ul class="sym-list">${{calls.map(c => `<li class="fk" onclick="selectByName('${{esc(c)}}')"><svg><use href="#i-out"/></svg>${{esc(c)}}</li>`).join("")}}</ul></div>`;

    if (callers.length) html += `<div class="psec"><div class="psec-title"><svg><use href="#i-in"/></svg>Called by (${{d.callers.length}})</div>
      <ul class="sym-list">${{callers.map(c => `<li class="mk" onclick="selectByName('${{esc(c)}}')"><svg><use href="#i-in"/></svg>${{esc(c)}}</li>`).join("")}}</ul></div>`;
  }}

  document.getElementById("panel-body").innerHTML = html;
  document.getElementById("panel").classList.add("open");
}}

function closePanel() {{ document.getElementById("panel").classList.remove("open"); }}

window.clickSym = function(nodeId) {{
  const node = DATA.nodes.find(n => n.id === nodeId);
  if (!node) return;
  if (!active.has(node.kind)) {{ active.add(node.kind); document.querySelector(`.kbtn[data-k="${{node.kind}}"]`)?.classList.add("on"); render(); }}
  sel = node.id;
  openPanel(node);
  paint();
  panTo(node);
}};

window.selectByName = function(name) {{
  const node = DATA.nodes.find(n => n.name === name || n.id.endsWith("::" + name));
  if (!node) return;
  window.clickSym(node.id);
}};

function panTo(node) {{
  const sc = d3.zoomTransform(svg.node()).k;
  const pW = document.getElementById("panel").classList.contains("open") ? 320 : 0;
  svg.transition().duration(400)
     .call(zoom.transform, d3.zoomIdentity.translate((W - pW) / 2 - node.x * sc, H / 2 - node.y * sc).scale(sc));
}}

/* ── Controls ── */
document.querySelectorAll(".kbtn").forEach(b => b.addEventListener("click", () => {{
  const k = b.dataset.k;
  active.has(k) ? (active.delete(k), b.classList.remove("on")) : (active.add(k), b.classList.add("on"));
  render();
}}));

let _st;
document.getElementById("search").addEventListener("input", e => {{
  searchQ = e.target.value;
  clearTimeout(_st); _st = setTimeout(paint, 140);
}});
document.getElementById("close-panel").addEventListener("click", () => {{ sel = null; closePanel(); paint(); }});
document.getElementById("fit-btn").addEventListener("click", fitView);

function fitView() {{
  const b = gRoot.node().getBBox();
  if (!b.width) return;
  const pW = document.getElementById("panel").classList.contains("open") ? 320 : 0;
  const sc = Math.min(0.88, Math.min((W - pW - 40) / b.width, (H - 72) / b.height));
  svg.transition().duration(600)
     .call(zoom.transform, d3.zoomIdentity
       .translate((W - pW - b.width * sc) / 2 - b.x * sc, (H - b.height * sc) / 2 - b.y * sc + 52)
       .scale(sc));
}}

render();
setTimeout(fitView, 2800);
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

### Option 1: Compact project map — START HERE (~10k tokens)
```bash
repo2kg summary --kg {kg_rel}
```
Prints a directory-grouped file map of all classes and top-level functions.
Covers the whole codebase in ~10k tokens. Use this for orientation before
making any changes. Save to file with `--out SUMMARY.md`.

### Option 2: Targeted semantic search (~400 tokens/query)
```bash
repo2kg query "how does authentication work" --kg {kg_rel}
repo2kg query-lite "auth" --kg {kg_rel}    # instant keyword search, no FAISS
```
Run 2-5 targeted queries to find the specific symbols relevant to your task.

### Option 3: Lightweight keyword query (Python stdlib only, zero deps)
```bash
python3 -c "
import json
data = json.load(open('{kg_rel}'))
matches = {{k: v for k, v in data.items() if 'YOUR_KEYWORD' in k.lower() or 'YOUR_KEYWORD' in v.get('name','').lower()}}
for node_id, node in list(matches.items())[:10]:
    print(f\\"\\n## {{node['kind'].upper()}}: {{node['name']}}\\")
    print(f\\"File: {{node['file']}}\\")
    print(node['signature'])
    if node.get('docstring'): print(node['docstring'][:200])
    if node.get('calls'): print(f\\"Calls: {{', '.join(c.split(\\\\\"::\\\\\")[-1] for c in node['calls'][:8])}}\\")
"
```

### Option 4: Structured JSON output (for programmatic use)
```bash
repo2kg query "auth" --kg {kg_rel} --format json
```

## Recommended Workflow (minimal token cost)

1. `repo2kg summary --kg {kg_rel}` — orientation map (~10k tokens)
2. `repo2kg query "<topic>" --kg {kg_rel}` — drill into relevant area (~400 tokens/query)
3. `read_file` only the specific file you need to edit

**Total: ~11k tokens** vs ~60k for a full CODEBASE.md export — 83% savings.

## Rules

1. **Before reading any source file**, check if the KG already has what you need
2. Start every session with `repo2kg summary` — not `cat CODEBASE.md`
3. Only read actual source files when you need the full implementation (>8 lines)
4. The KG includes 8-line body previews — often sufficient for understanding logic
5. Use call edges to trace dependencies instead of grepping across files

## Project Structure

- `{kg_rel}` — Knowledge graph ({node_count} nodes)
- Run `repo2kg summary --kg {kg_rel}` for the project overview
"""

    # ── .copilot-instructions.md ──────────────────────────────────────
    copilot_md = f"""# Copilot Instructions

## Code Knowledge Graph

A pre-built knowledge graph is available at `{kg_rel}` with {node_count} code nodes
({len(files)} files, {edge_count} call edges).

### Start here — compact project map (~10k tokens):
```bash
repo2kg summary --kg {kg_rel}
```
Shows all files, classes and functions grouped by directory. Much cheaper than
reading CODEBASE.md (~60k tokens). Follow up with targeted queries.

### Targeted search:
```bash
repo2kg query "<topic>" --kg {kg_rel}      # semantic (FAISS)
repo2kg query-lite "<keyword>" --kg {kg_rel}  # instant keyword, no deps
```

### Quick inline lookup (Python, zero deps):
```python
import json
kg = json.load(open("{kg_rel}"))
for nid, n in kg.items():
    if "KEYWORD" in nid.lower() or "KEYWORD" in n.get("name","").lower():
        print(n["signature"], "—", n.get("docstring","")[:100])
        print("calls:", [c.split("::")[-1] for c in n.get("calls",[])])
```

### Best practices:
- Run `repo2kg summary` first, then `repo2kg query` for specifics
- Follow call edges (`calls` / `callers` fields) to trace execution flow
- Body previews (first 8 lines) are in `body_preview` field
"""

    # ── .github/copilot-instructions.md  (for Copilot agent mode) ─────
    copilot_agent_md = f"""# Copilot Agent Instructions

This project has a code knowledge graph. Before exploring source files:

1. Run `repo2kg summary --kg {kg_rel}` for a compact project map (~10k tokens)
2. Run `repo2kg query "<topic>" --kg {kg_rel}` for targeted deep dives (~400 tokens)
3. Only read source files when you need implementation details beyond the 8-line preview

Do NOT use `cat CODEBASE.md` — it's ~60k tokens. Use `summary` + `query` instead.
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
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
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
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
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
    existing = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    if _REPO2KG_MARKER in existing:
        new_content = _re.sub(
            rf"{_re.escape(_REPO2KG_MARKER)}.*?{_re.escape(_REPO2KG_END_MARKER)}",
            new_block.strip(),
            existing,
            flags=_re.DOTALL,
        )
    else:
        new_content = existing.rstrip() + ("\n\n" if existing else "") + new_block
    file_path.write_text(new_content, encoding="utf-8")


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
            print(n["signature"], "--", n.get("docstring","")[:100])
'''

    # ── ~/.claude/CLAUDE.md ─────────────────────────────────────────
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    claude_global = claude_dir / "CLAUDE.md"

    repo2kg_block = f"""{_REPO2KG_MARKER}
## repo2kg — Code Knowledge Graph

A global project registry is at `{registry_path_str}`.
For any Python project you open, check if it has a registered knowledge graph:

```python
{lookup_snippet}
```

### Workflow for every task:
1. Run the lookup above to find the project's KG
2. If a KG exists, run `repo2kg summary --kg <kg_path>` for a compact project map (~10k tokens)
3. Run `repo2kg query "<topic>" --kg <kg_path>` for targeted deep dives (~400 tokens/query)
4. Only read actual source files when the 8-line body preview is not enough
5. Follow `calls`/`callers` fields to trace dependencies

Do NOT load CODEBASE.md for every task — it can exceed 60k tokens. Use summary + query instead.

### Node schema (what's in kg.json)
```
key: "relative/file.py::ClassName.method"
fields: id, name, kind, file, parent_class, signature,
        docstring, body_preview (8 lines), calls[], callers[], imports[]
```
{_REPO2KG_END_MARKER}
"""

    _merge_file_with_block(claude_global, repo2kg_block)
    print(f"Updated {claude_global}")

    # ── ~/.codex/AGENTS.md ──────────────────────────────────────────
    codex_dir = Path.home() / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    codex_agents = codex_dir / "AGENTS.md"
    codex_block = f"""{_REPO2KG_MARKER}
# repo2kg -- Code Knowledge Graph

Registry: `{registry_path_str}`

For any task on a Python project, start with:
```python
{lookup_snippet}
```

Then query the returned `kg` dict instead of reading source files.
Node fields: `name`, `kind`, `file`, `signature`, `docstring`, `body_preview`, `calls`, `callers`.
{_REPO2KG_END_MARKER}
"""
    _merge_file_with_block(codex_agents, codex_block)
    print(f"Updated {codex_agents}")

    # ── ~/.repo2kg/GLOBAL_AGENTS.md ─────────────────────────────────
    REPO2KG_HOME.mkdir(parents=True, exist_ok=True)
    global_ref = REPO2KG_HOME / "GLOBAL_AGENTS.md"
    global_block = f"""{_REPO2KG_MARKER}
# repo2kg Global Agent Reference

## Registry
`{registry_path_str}` -- maps project paths to their KG files.

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
            "repo2kg build --repo . --out kg.toon       # Build KG (TOON, 40% fewer tokens)",
            "repo2kg summary --kg kg.toon               # Compact project map (~10k tokens) — start here",
            "repo2kg query 'auth flow' --kg kg.toon     # Semantic deep dive (~400 tokens/query)",
            "repo2kg query-lite 'auth' --kg kg.toon     # Instant keyword search (no FAISS)",
            "repo2kg agent-setup --kg kg.toon --dir .   # Generate CLAUDE.md + .copilot-instructions.md",
        ],
        "agent_tips": [
            "Run 'repo2kg summary' first — it covers the whole codebase in ~10k tokens vs ~60k for export",
            "Then use 'repo2kg query' for targeted deep dives — ~400 tokens per query",
            "Total context: ~11k tokens (summary + 3 queries) vs ~60k for CODEBASE.md — 83% savings",
            "Prefer query-lite over query when you know the symbol name — zero heavy dependencies",
            "Use --format json for structured output you can parse programmatically",
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
Recommended Agent Workflow (minimal token cost):
  repo2kg build --repo . --out kg.toon       # 1. Build KG
  repo2kg summary --kg kg.toon               # 2. Orientation map (~10k tokens)
  repo2kg query "auth flow" --kg kg.toon     # 3. Deep dive (~400 tokens/query)
  repo2kg query-lite "login" --kg kg.toon    # 3b. Instant keyword search (no ML)

Agent Setup:
  repo2kg agent-setup --kg kg.toon --dir .   # CLAUDE.md + .copilot-instructions.md
  repo2kg user-setup                         # Global agent instructions (run once)

Discovery & Registry:
  repo2kg scan --root ~                      # Auto-discover all KGs under home
  repo2kg register --kg kg.toon --name myproject  # Register a KG manually
  repo2kg list                               # Show registered projects
  repo2kg stats --kg kg.toon                 # Show KG statistics (nodes, edges, files)
  repo2kg info                               # Machine-readable tool info (JSON)

Visualization:
  repo2kg visualize --kg kg.toon --out graph.html  # Interactive D3.js HTML graph

Export (full snapshot — use sparingly):
  repo2kg summary --kg kg.toon --out SUMMARY.md    # ~10k tokens — preferred
  repo2kg export --kg kg.toon --out CODEBASE.md    # ~60k tokens — use sparingly

Advanced:
  repo2kg set-model all-mpnet-base-v2    # Change embedding model (used by build + query)
  repo2kg set-model all-mpnet-base-v2 --delete-old  # Also remove old cached model

Formats:
  .json  — Default. Compact, fast, universal.
  .toon  — Token-Oriented Object Notation. ~40% fewer tokens than JSON.
           Best for LLM context windows.

Languages:
  Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, C#
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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity (overrides -v)",
    )
    sub = parser.add_subparsers(dest="cmd")

    # ── build ──
    build_p = sub.add_parser("build", help="Build KG from a repository",
                             description="Parse source files (Python/JS/TS/Java/Go/Rust/C/C++/Ruby/C#), "
                                         "extract functions/classes/methods, "
                                         "resolve call edges, build FAISS index, and save to JSON or TOON.")
    build_p.add_argument("--repo", default=".", help="Path to repo root (default: current dir)")
    build_p.add_argument("--out", default="kg.json",
                         help="Output path for KG. Use .json or .toon extension (default: kg.json)")
    build_p.add_argument("--exclude", nargs="*", help="Additional directory patterns to exclude")
    build_p.add_argument("--config", default=None,
                         help="Path to repo2kg.toml config file (default: {repo}/repo2kg.toml)")
    build_p.add_argument("--strict-calls", action="store_true",
                         help="Drop ambiguous call edges instead of guessing (reduces false edges)")
    build_p.add_argument("--batch-size", type=int, default=64,
                         help="Embedding batch size for FAISS (default: 64)")
    build_p.add_argument("--deps", action="store_true",
                         help="Also export module dependency graph ({out}.deps.json)")

    # ── query (semantic, requires FAISS) ──
    query_p = sub.add_parser("query", help="Semantic search (requires FAISS)")
    query_p.add_argument("question", help="Natural language question")
    query_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    query_p.add_argument("--k", type=int, default=5, help="Top-k results (default: 5)")
    query_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    query_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json (structured)")

    # ── query-lite (keyword, no heavy deps) ──
    qlite_p = sub.add_parser("query-lite",
                             help="Keyword search (zero dependencies)")
    qlite_p.add_argument("keywords", help="Keywords to search for")
    qlite_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    qlite_p.add_argument("--k", type=int, default=5, help="Top-k results (default: 5)")
    qlite_p.add_argument("--depth", type=int, default=1, help="Graph traversal depth (default: 1)")
    qlite_p.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format: text (default) or json")

    # ── export (generate CODEBASE.md) ──
    export_p = sub.add_parser(
        "export",
        help="Export as CODEBASE.md (~60k tokens for large projects — prefer 'summary' for agents)",
        description=(
            "Generates a complete CODEBASE.md with every symbol, signature, docstring, "
            "call graph, and entry points.\n\n"
            "WARNING: For large projects (100+ files) this can exceed 60,000 tokens — "
            "larger than many LLM context budgets. Prefer 'repo2kg summary' (~10k tokens) "
            "for agent orientation, then use 'repo2kg query' for targeted deep dives."
        ),
    )
    export_p.add_argument("--kg", default="kg.json",
                          help="Path to saved KG .json/.toon (default: kg.json)")
    export_p.add_argument("--out", default="CODEBASE.md",
                          help="Output markdown file (default: CODEBASE.md)")

    # ── summary (compact map, ~10× fewer tokens than CODEBASE.md) ──
    summary_p = sub.add_parser(
        "summary",
        help="Compact project map (~10k tokens) — classes + functions only, no methods",
        description=(
            "Outputs a directory-grouped file map showing only top-level classes and "
            "standalone functions. Methods are omitted. Ideal as the first context "
            "block for an agent before it starts querying.\n\n"
            "Token cost: ~4–8k for a 200-file project (vs ~60k for 'export')."
        ),
    )
    summary_p.add_argument("--kg", default="kg.json",
                            help="Path to saved KG .json/.toon (default: kg.json)")
    summary_p.add_argument("--out", default=None,
                            help="Write to file instead of stdout")

    # ── agent-setup ──
    agent_p = sub.add_parser("agent-setup",
                             help="Generate all agent instruction files (CLAUDE.md, .copilot-instructions.md, AGENTS.md)")
    agent_p.add_argument("--kg", default="kg.json",
                         help="Path to saved KG .json/.toon (default: kg.json)")
    agent_p.add_argument("--dir", default=".",
                         help="Target directory for agent files (default: current dir)")
    agent_p.add_argument("--no-register", action="store_true",
                         help="Skip registering in global ~/.repo2kg/registry.json")

    # ── user-setup (global agent instructions) ──
    sub.add_parser("user-setup",
                   help="Install global agent instructions (run once — ~/.claude/CLAUDE.md, ~/.codex/AGENTS.md, etc.)")

    # ── register ──
    reg_p = sub.add_parser("register",
                           help="Register a single project KG manually in the global registry")
    reg_p.add_argument("--kg", default="kg.json",
                       help="Path to saved KG .json/.toon (default: kg.json)")
    reg_p.add_argument("--project", default=".",
                       help="Project root directory to register (default: current dir)")

    # ── scan ──
    scan_p = sub.add_parser("scan",
                            help="Auto-discover and register all KGs under a directory")
    scan_p.add_argument("--root", default=str(Path.home()),
                        help="Root directory to scan (default: home directory)")

    # ── list (show registered projects) ──
    sub.add_parser("list", help="Show all registered projects")

    # ── info (machine-readable tool discovery for agents) ──
    sub.add_parser("info",
                   help="Print machine-readable tool info (JSON) — for agents")

    # ── deps (module dependency graph) ──
    deps_p = sub.add_parser("deps",
                            help="Export module-level import dependency graph")
    deps_p.add_argument("--kg", default="kg.json",
                        help="Path to saved KG .json/.toon (default: kg.json)")
    deps_p.add_argument("--out", default=None,
                        help="Output JSON file (default: {kg}.deps.json)")

    # ── stats ──
    stats_p = sub.add_parser("stats", help="Show KG node/edge/file statistics")
    stats_p.add_argument("--kg", default="kg.json", help="Path to saved KG (default: kg.json)")

    # ── set-model (change embedding model) ──
    setm_p = sub.add_parser(
        "set-model",
        help="Change the embedding model used by 'build' and 'query' (HuggingFace model name)",
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
        help="Generate an interactive HTML graph — open in any browser, no server needed",
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

    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s: %(message)s")

    if args.cmd == "info":
        print_tool_info()

    elif args.cmd == "build":
        exclude = DEFAULT_EXCLUDE + (args.exclude or [])
        cfg = None
        if args.config:
            cfg = _load_project_config(os.path.dirname(os.path.abspath(args.config)))
        kg = RepoKG().build(
            args.repo,
            exclude=exclude,
            config=cfg,
            strict_calls=args.strict_calls,
            batch_size=args.batch_size,
        )
        kg.save(args.out)
        if args.deps:
            kg.export_dependency_graph(kg_path=args.out)

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

    elif args.cmd == "summary":
        generate_summary_md(args.kg, args.out)

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

    elif args.cmd == "deps":
        kg = RepoKG.load(args.kg)
        kg.export_dependency_graph(out_path=args.out, kg_path=args.kg)

    elif args.cmd == "stats":
        kg = RepoKG.load(args.kg)
        files = set(n.file for n in kg.nodes.values())
        classes = sum(1 for n in kg.nodes.values() if n.kind == "class")
        functions = sum(1 for n in kg.nodes.values() if n.kind == "function")
        methods = sum(1 for n in kg.nodes.values() if n.kind == "method")
        edges = sum(len(n.calls) for n in kg.nodes.values())
        cycles = sum(1 for n in kg.nodes.values() if getattr(n, "in_cycle", False))
        model_in_use = _default_model()
        print(f"Knowledge Graph Statistics:")
        print(f"  Files:         {len(files)}")
        print(f"  Nodes:         {len(kg.nodes)} ({classes} classes, {functions} functions, {methods} methods)")
        print(f"  Edges:         {edges} call edges")
        print(f"  Avg edges:     {edges / max(len(kg.nodes), 1):.1f} per node")
        if cycles:
            print(f"  Cycle nodes:   {cycles}")
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
