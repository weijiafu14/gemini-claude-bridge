"""
Tests for thoughtSignature persistence manager.
"""

import os
import json
import tempfile
import pytest

from gemini_claude_bridge.signature_manager import GeminiSignatureManager


class TestSignatureManager:

    def test_store_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("sess1", "/tmp/proj", "tool_1", "sig_abc")

            result = mgr.get("sess1", "/tmp/proj", "tool_1")
            assert result == "sig_abc"

    def test_get_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            result = mgr.get("sess1", "/tmp/proj", "tool_nonexistent")
            assert result is None

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = GeminiSignatureManager(base_dir=tmpdir)
            mgr1.store("sess1", "/tmp/proj", "tool_1", "sig_persisted")

            # New instance should load from file
            mgr2 = GeminiSignatureManager(base_dir=tmpdir)
            result = mgr2.get("sess1", "/tmp/proj", "tool_1")
            assert result == "sig_persisted"

    def test_multiple_signatures_same_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("sess1", "/tmp/proj", "tool_1", "sig_1")
            mgr.store("sess1", "/tmp/proj", "tool_2", "sig_2")
            mgr.store("sess1", "/tmp/proj", "tool_3", "sig_3")

            assert mgr.get("sess1", "/tmp/proj", "tool_1") == "sig_1"
            assert mgr.get("sess1", "/tmp/proj", "tool_2") == "sig_2"
            assert mgr.get("sess1", "/tmp/proj", "tool_3") == "sig_3"

    def test_different_sessions_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("sess1", "/tmp/proj", "tool_1", "sig_from_sess1")
            mgr.store("sess2", "/tmp/proj", "tool_1", "sig_from_sess2")

            assert mgr.get("sess1", "/tmp/proj", "tool_1") == "sig_from_sess1"
            assert mgr.get("sess2", "/tmp/proj", "tool_1") == "sig_from_sess2"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("sess1", "/proj1", "tool_1", "sig1")
            mgr.store("sess1", "/proj1", "tool_2", "sig2")
            mgr.store("sess2", "/proj2", "tool_3", "sig3")

            stats = mgr.get_stats()
            assert stats["total_sessions"] == 2
            assert stats["total_signatures"] == 3

    def test_invalid_params_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("", "/proj", "tool_1", "sig")  # empty session
            mgr.store("sess", "/proj", "", "sig")  # empty tool_id
            mgr.store("sess", "/proj", "tool_1", "")  # empty signature

            assert mgr.get("", "/proj", "tool_1") is None
            assert mgr.get("sess", "/proj", "") is None

    def test_file_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = GeminiSignatureManager(base_dir=tmpdir)
            mgr.store("sess1", "/tmp/proj", "tool_1", "sig_abc")

            # Find the written file
            sig_file = None
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".signatures.json"):
                        sig_file = os.path.join(root, f)

            assert sig_file is not None
            with open(sig_file) as f:
                data = json.load(f)

            assert data["version"] == 1
            assert "signatures" in data
            assert data["signatures"]["tool_1"] == "sig_abc"
            assert "updated_at" in data
