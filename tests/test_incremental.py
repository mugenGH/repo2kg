"""Test incremental build cache: hit/miss behavior on file modification."""
import sys
import os
import json
import time
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from repo2kg import RepoKG, __version__


@pytest.fixture
def repo_dir(tmp_path):
    src = tmp_path / "src.py"
    src.write_text(
        "def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n"
    )
    return tmp_path


@pytest.fixture
def mock_kg(repo_dir):
    """Return a RepoKG with FAISS mocked out (no heavy deps needed)."""
    with patch.object(RepoKG, '_build_faiss'), \
         patch('repo2kg._SentenceTransformer'), \
         patch('repo2kg._load_heavy_deps'):
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        kg.parse_errors = []
        kg.cycles = []
        kg._repo_path = ""
        kg.model = MagicMock()
        kg._index = None
        kg._index_ids = []
    return kg


class TestCacheKeyStability:
    def test_cache_key_changes_when_content_changes(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        key1 = kg._file_cache_key(str(f))
        # Write more content so size changes even if mtime resolution is coarse
        f.write_text("x = 1\ny = 2\nz = 3\n")
        key2 = kg._file_cache_key(str(f))
        assert key1 != key2

    def test_cache_key_stable_when_unchanged(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        key1 = kg._file_cache_key(str(f))
        key2 = kg._file_cache_key(str(f))
        assert key1 == key2


class TestCacheLoadSave:
    def test_empty_cache_on_missing_file(self, tmp_path):
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        result = kg._load_build_cache(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_cache_invalidated_on_version_mismatch(self, tmp_path):
        cache_file = tmp_path / "build_cache.json"
        cache_file.write_text(json.dumps({"version": "0.0.0", "entries": {"a": {"key": "x", "nodes": []}}}))
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        result = kg._load_build_cache(str(cache_file))
        assert result == {}

    def test_cache_loaded_with_current_version(self, tmp_path):
        cache_file = tmp_path / "build_cache.json"
        entries = {"src.py": {"key": "12345:100", "nodes": []}}
        cache_file.write_text(json.dumps({"version": __version__, "entries": entries}))
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        result = kg._load_build_cache(str(cache_file))
        assert "src.py" in result

    def test_cache_saved_and_reloaded(self, tmp_path):
        cache_file = tmp_path / ".repo2kg_cache" / "build_cache.json"
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        entries = {"file.py": {"key": "1:100", "nodes": []}}
        kg._save_build_cache(str(cache_file), entries)
        assert cache_file.exists()
        loaded = kg._load_build_cache(str(cache_file))
        assert loaded == entries

    def test_cache_save_does_not_crash_on_bad_path(self):
        kg = RepoKG.__new__(RepoKG)
        kg.nodes = {}
        # Should not raise even if path is invalid
        kg._save_build_cache("/nonexistent/deeply/nested/path/cache.json", {})
