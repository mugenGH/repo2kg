"""Test TOON serialization round-trips and edge cases."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from repo2kg import serialize_toon, deserialize_toon, _split_toon_array, MAX_TOON_FIELD_LEN


def _make_node(**kwargs):
    defaults = {
        "id": "file.py::func",
        "name": "func",
        "kind": "function",
        "file": "file.py",
        "parent_class": None,
        "signature": "def func():",
        "docstring": "",
        "body_preview": "    pass",
        "calls": [],
        "callers": [],
        "imports": [],
        "in_cycle": False,
    }
    defaults.update(kwargs)
    return defaults


class TestToonRoundTrip:
    def test_empty_graph(self):
        data = {}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert result == {}

    def test_single_node(self):
        node = _make_node()
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert node["id"] in result
        assert result[node["id"]]["name"] == "func"

    def test_node_with_calls(self):
        node = _make_node(calls=["file.py::helper", "file.py::other"])
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert result[node["id"]]["calls"] == ["file.py::helper", "file.py::other"]

    def test_node_with_docstring_containing_special_chars(self):
        node = _make_node(docstring='Has "quotes", commas, and [brackets].')
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert result[node["id"]]["docstring"] == node["docstring"]

    def test_node_with_newlines_in_body(self):
        node = _make_node(body_preview="    x = 1\n    y = 2\n    return x + y")
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert "\n" in result[node["id"]]["body_preview"]

    def test_multiple_nodes(self):
        nodes = {f"file.py::func{i}": _make_node(id=f"file.py::func{i}", name=f"func{i}")
                 for i in range(10)}
        text = serialize_toon(nodes)
        result = deserialize_toon(text)
        assert len(result) == 10

    def test_empty_arrays_roundtrip(self):
        node = _make_node(calls=[], callers=[], imports=[])
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert result[node["id"]]["calls"] == []

    def test_unicode_content(self):
        node = _make_node(docstring="Returns à résumé: données français")
        data = {node["id"]: node}
        text = serialize_toon(data)
        result = deserialize_toon(text)
        assert result[node["id"]]["docstring"] == node["docstring"]


class TestToonHardening:
    def test_magic_number_validated(self):
        with pytest.raises(ValueError, match="TOON"):
            deserialize_toon("id: foo\nname: bar\n")

    def test_max_field_len_enforced(self):
        huge = "a," * (MAX_TOON_FIELD_LEN // 2 + 1)
        with pytest.raises(ValueError, match="MAX_TOON_FIELD_LEN"):
            _split_toon_array(huge)

    def test_quoted_commas_not_split(self):
        items = _split_toon_array('"a,b","c,d"')
        assert items == ['"a,b"', '"c,d"']

    def test_escaped_quotes_handled(self):
        items = _split_toon_array('"he said \\"hello\\""')
        assert len(items) == 1
