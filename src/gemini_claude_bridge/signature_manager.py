"""
Gemini Thought Signature persistence manager.

Stores tool_use_id -> thoughtSignature mappings to disk so they survive
across multi-turn conversations. Gemini requires signatures to be echoed
back when replaying tool calls in thinking mode.

File format:
{
    "version": 1,
    "signatures": {
        "toolu_abc123": "CvkmAY89a1/IL/rH...",
        "toolu_def456": "CvgKAY89a1+cHcQP..."
    },
    "updated_at": "2025-12-22T14:30:00Z"
}
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict
from threading import Lock
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class GeminiSignatureManager:
    """Manages reading/writing of Gemini thoughtSignature files.

    Args:
        base_dir: Root directory for signature files.
            Defaults to ~/.gemini-claude-bridge/signatures/
            Signatures are stored at: {base_dir}/{formatted_path}/{session_id}.signatures.json
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self._base_dir = Path(base_dir)
        else:
            home = os.environ.get("HOME", os.path.expanduser("~"))
            self._base_dir = Path(home) / ".gemini-claude-bridge" / "signatures"

        # In-memory cache: cache_key -> {tool_use_id: signature}
        self._cache: Dict[str, Dict[str, str]] = {}
        self._lock = Lock()

    def _format_path(self, project_path: str) -> str:
        """Format project path for use as directory name (replace / and . with -)."""
        return project_path.replace("/", "-").replace(".", "-")

    def _get_signature_file_path(self, session_id: str, project_path: str) -> Path:
        formatted_path = self._format_path(project_path)
        return self._base_dir / formatted_path / f"{session_id}.signatures.json"

    def _get_cache_key(self, session_id: str, project_path: str) -> str:
        return f"{session_id}:{project_path}"

    def store(
        self,
        session_id: str,
        project_path: str,
        tool_use_id: str,
        signature: str,
    ) -> None:
        """
        Store a signature to memory cache and persist to disk.

        Args:
            session_id: Session identifier.
            project_path: Project path (used for file organization).
            tool_use_id: The tool_use ID (e.g. toolu_xxx or call_xxx).
            signature: The Gemini thoughtSignature string.
        """
        if not session_id or not project_path or not tool_use_id or not signature:
            return

        with self._lock:
            cache_key = self._get_cache_key(session_id, project_path)
            if cache_key not in self._cache:
                self._load_from_file_unlocked(session_id, project_path)
                if cache_key not in self._cache:
                    self._cache[cache_key] = {}

            self._cache[cache_key][tool_use_id] = signature
            self._write_to_file_unlocked(session_id, project_path)

            logger.debug(f"Stored signature: session={session_id[:8]}..., tool_use_id={tool_use_id}")

    def get(
        self,
        session_id: str,
        project_path: str,
        tool_use_id: str,
    ) -> Optional[str]:
        """
        Retrieve a signature from cache or disk.

        Args:
            session_id: Session identifier.
            project_path: Project path.
            tool_use_id: The tool_use ID.

        Returns:
            The thoughtSignature string, or None if not found.
        """
        if not session_id or not project_path or not tool_use_id:
            return None

        with self._lock:
            cache_key = self._get_cache_key(session_id, project_path)

            if cache_key in self._cache and tool_use_id in self._cache[cache_key]:
                return self._cache[cache_key][tool_use_id]

            self._load_from_file_unlocked(session_id, project_path)

            if cache_key in self._cache and tool_use_id in self._cache[cache_key]:
                return self._cache[cache_key][tool_use_id]

            return None

    def _write_to_file_unlocked(self, session_id: str, project_path: str) -> None:
        cache_key = self._get_cache_key(session_id, project_path)
        if cache_key not in self._cache:
            return

        file_path = self._get_signature_file_path(session_id, project_path)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "signatures": self._cache[cache_key],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write signature file: {e}")

    def _load_from_file_unlocked(self, session_id: str, project_path: str) -> None:
        file_path = self._get_signature_file_path(session_id, project_path)

        if not file_path.exists():
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("version") == 1 and "signatures" in data:
                cache_key = self._get_cache_key(session_id, project_path)
                self._cache[cache_key] = data["signatures"]
                logger.debug(f"Loaded {len(data['signatures'])} signatures from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load signature file: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            total_sessions = len(self._cache)
            total_signatures = sum(len(sigs) for sigs in self._cache.values())
            return {
                "total_sessions": total_sessions,
                "total_signatures": total_signatures,
            }
