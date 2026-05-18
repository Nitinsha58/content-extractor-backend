"""
Global ML model state (singleton instances).

Replaces FastAPI's global variables with Django-compatible module-level singletons.
"""

import time

# ExamLayoutParser instance (loaded at startup via AppConfig.ready())
layout_parser = None

# Debug session cache: session_id -> { "image_path": Path, "image_w": int, "image_h": int, "_ts": float }
# Capped at _SESSION_MAX_SIZE entries; oldest by access time are evicted first.
_debug_sessions = {}
_SESSION_MAX_SIZE = 200   # keeps ~130 MB of in-memory metadata at most


def get_layout_parser():
    """Get the preloaded layout parser, or raise if not initialized."""
    if layout_parser is None:
        raise RuntimeError("Layout parser not initialized. Check logs above.")
    return layout_parser


def _evict_sessions_if_needed():
    """Drop oldest sessions when the cache exceeds _SESSION_MAX_SIZE."""
    if len(_debug_sessions) >= _SESSION_MAX_SIZE:
        sorted_keys = sorted(_debug_sessions, key=lambda k: _debug_sessions[k].get("_ts", 0))
        for key in sorted_keys[:len(_debug_sessions) - _SESSION_MAX_SIZE + 1]:
            _debug_sessions.pop(key, None)


def register_session(session_id: str, data: dict):
    """Add or refresh a session, evicting oldest entries if over capacity."""
    data["_ts"] = time.monotonic()
    _debug_sessions[session_id] = data
    _evict_sessions_if_needed()


def touch_session(session_id: str):
    """Update the access timestamp so this session survives future evictions."""
    if session_id in _debug_sessions:
        _debug_sessions[session_id]["_ts"] = time.monotonic()


def clear_session(session_id):
    """Clear a debug session from memory."""
    _debug_sessions.pop(session_id, None)


def clear_all_sessions():
    """Clear all debug sessions (useful for cleanup)."""
    _debug_sessions.clear()
