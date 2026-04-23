"""Test parsing of fixture files — correct nodes, edges, and imports."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from repo2kg import _parse_python_file, parse_file, CodeNode

FIXTURES = Path(__file__).parent / "fixtures"


class TestPythonParsing:
    def setup_method(self):
        self.nodes = {n.id: n for n in _parse_python_file(str(FIXTURES / "sample.py"))}

    def test_finds_standalone_function(self):
        assert any(n.name == "standalone_function" for n in self.nodes.values())

    def test_finds_class(self):
        assert any(n.name == "SampleClass" and n.kind == "class" for n in self.nodes.values())

    def test_finds_methods(self):
        assert any(n.name == "greet" and n.kind == "method" for n in self.nodes.values())
        assert any(n.name == "process" and n.kind == "method" for n in self.nodes.values())

    def test_method_has_parent_class(self):
        greet = next(n for n in self.nodes.values() if n.name == "greet")
        assert greet.parent_class == "SampleClass"

    def test_docstring_extracted(self):
        func = next(n for n in self.nodes.values() if n.name == "standalone_function")
        assert "Add two numbers" in func.docstring

    def test_imports_collected(self):
        func = next(n for n in self.nodes.values() if n.name == "standalone_function")
        assert any("os" in imp or "json" in imp for imp in func.imports)

    def test_raw_calls_collected(self):
        caller = next(n for n in self.nodes.values() if n.name == "calls_other")
        assert "helper_func" in caller.calls

    def test_body_preview_not_empty(self):
        func = next(n for n in self.nodes.values() if n.name == "standalone_function")
        assert func.body_preview.strip()

    def test_body_preview_min_lines(self):
        func = next(n for n in self.nodes.values() if n.name == "standalone_function")
        assert len(func.body_preview.splitlines()) >= 1

    def test_file_field_is_relative_when_repo_root_given(self):
        nodes = _parse_python_file(str(FIXTURES / "sample.py"), repo_root=str(FIXTURES))
        for n in nodes:
            assert not os.path.isabs(n.file)


class TestParseFallback:
    def test_parse_file_returns_nodes_for_python(self):
        nodes = parse_file(str(FIXTURES / "sample.py"))
        assert len(nodes) > 0

    def test_parse_file_returns_list_for_unsupported_ext(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("hello")
        assert parse_file(str(f)) == []

    def test_parse_file_handles_syntax_error_gracefully(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(:\n    pass\n")
        result = parse_file(str(f))
        assert result == []


class TestEdgeResolution:
    def test_call_edges_resolved_to_node_ids(self):
        from repo2kg import RepoKG
        import unittest.mock as mock
        # Build a minimal KG from fixtures
        with mock.patch.object(RepoKG, '_build_faiss'):
            kg = RepoKG.__new__(RepoKG)
            kg.nodes = {}
            kg.parse_errors = []
            kg.cycles = []
            kg._repo_path = ""
            from repo2kg import _parse_python_file
            for n in _parse_python_file(str(FIXTURES / "sample.py"),
                                         repo_root=str(FIXTURES)):
                kg.nodes[n.id] = n
            kg._resolve_edges()

        # calls_other calls helper_func — should be resolved to a node id
        caller = next(n for n in kg.nodes.values() if n.name == "calls_other")
        for cid in caller.calls:
            assert "::" in cid, f"Expected node id, got raw name: {cid}"
