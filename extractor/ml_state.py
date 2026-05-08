"""
Global ML model state (singleton instances).

Replaces FastAPI's global variables with Django-compatible module-level singletons.
"""

# ExamLayoutParser instance (loaded at startup via AppConfig.ready())
layout_parser = None

# Debug session cache: session_id -> { "image_path": Path, "image_w": int, "image_h": int }
_debug_sessions = {}


def get_layout_parser():
    """Get the preloaded layout parser, or raise if not initialized."""
    if layout_parser is None:
        raise RuntimeError("Layout parser not initialized. Check logs above.")
    return layout_parser


def clear_session(session_id):
    """Clear a debug session from memory."""
    _debug_sessions.pop(session_id, None)


def clear_all_sessions():
    """Clear all debug sessions (useful for cleanup)."""
    _debug_sessions.clear()
