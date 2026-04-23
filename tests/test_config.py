"""Test project config loading from repo2kg.toml."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from repo2kg import _load_project_config, BODY_PREVIEW_LINES


class TestLoadProjectConfig:
    def test_returns_defaults_when_no_toml(self, tmp_path):
        cfg = _load_project_config(str(tmp_path))
        assert cfg["build"]["body_preview_lines"] == BODY_PREVIEW_LINES
        assert cfg["build"]["strict_calls"] is False
        assert cfg["build"]["max_file_size_kb"] == 500
        assert cfg["embeddings"]["batch_size"] == 64

    def test_toml_overrides_defaults(self, tmp_path):
        toml = tmp_path / "repo2kg.toml"
        toml.write_text('[build]\nbody_preview_lines = 20\nstrict_calls = true\n')
        cfg = _load_project_config(str(tmp_path))
        assert cfg["build"]["body_preview_lines"] == 20
        assert cfg["build"]["strict_calls"] is True
        # Unspecified keys still have defaults
        assert cfg["build"]["max_file_size_kb"] == 500

    def test_invalid_toml_falls_back_to_defaults(self, tmp_path):
        toml = tmp_path / "repo2kg.toml"
        toml.write_text("this is not valid toml ][")
        cfg = _load_project_config(str(tmp_path))
        # Should not raise; falls back to defaults
        assert cfg["build"]["body_preview_lines"] == BODY_PREVIEW_LINES

    def test_embeddings_section_configurable(self, tmp_path):
        toml = tmp_path / "repo2kg.toml"
        toml.write_text('[embeddings]\nbatch_size = 128\n')
        cfg = _load_project_config(str(tmp_path))
        assert cfg["embeddings"]["batch_size"] == 128
