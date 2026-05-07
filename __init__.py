"""
ComfyUI Audio SRT Aligner — Plugin entry point.

Registers nodes via importlib.util to avoid namespace conflicts with
ComfyUI's built-in `nodes` module.  On first load it also attempts to
download the Noto Sans SC font for subtitle rendering.
"""

import importlib.util
import os
import subprocess
import sys

plugin_dir = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, path: str):
    """Load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pre-load internal packages into sys.modules with UNIQUE names.
#
# This avoids name collisions with ComfyUI's own 'utils' and 'nodes' modules.
# ComfyUI may have ~/App/ComfyUI/utils on sys.path, so a bare 'import utils'
# would resolve to the wrong package.
#
# We register our packages under unique prefixed names so that node files
# can import them reliably.
# ---------------------------------------------------------------------------

# Load aligner package
_aligner_pkg = _load_module(
    "audio_srt_aligner_aligner",
    os.path.join(plugin_dir, "aligner", "__init__.py"),
)

# Load aligner.engine module
_load_module(
    "audio_srt_aligner_engine",
    os.path.join(plugin_dir, "aligner", "engine.py"),
)

# Load utils package
_load_module(
    "audio_srt_aligner_utils",
    os.path.join(plugin_dir, "utils", "__init__.py"),
)

# Load utils.srt_parser module
_load_module(
    "audio_srt_aligner_utils_srt_parser",
    os.path.join(plugin_dir, "utils", "srt_parser.py"),
)


# ---------------------------------------------------------------------------
# Load aligner node (Phase 1 — always present)
# ---------------------------------------------------------------------------
_aligner_node = _load_module(
    "audio_srt_aligner_node",
    os.path.join(plugin_dir, "nodes", "aligner_node.py"),
)

# ---------------------------------------------------------------------------
# Load video overlay node (Phase 2 — may not exist yet)
# ---------------------------------------------------------------------------
_video_srt_node = None
_video_node_path = os.path.join(plugin_dir, "nodes", "video_srt_overlay_node.py")
if os.path.exists(_video_node_path):
    _video_srt_node = _load_module(
        "video_srt_overlay_node",
        _video_node_path,
    )

# ---------------------------------------------------------------------------
# Node registrations
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AudioSrtAligner": _aligner_node.AudioSrtAligner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSrtAligner": "Audio SRT Aligner (文稿校对字幕)",
}

if _video_srt_node is not None and hasattr(_video_srt_node, "VideoSrtOverlay"):
    NODE_CLASS_MAPPINGS["VideoSrtOverlay"] = _video_srt_node.VideoSrtOverlay
    NODE_DISPLAY_NAME_MAPPINGS["VideoSrtOverlay"] = "Video SRT Overlay (字幕合成)"

# ---------------------------------------------------------------------------
# Font auto-download on first run (non-critical, must not block loading)
# Downloads Noto Sans SC into the plugin's own fonts/ directory.
# ---------------------------------------------------------------------------
try:
    _font_script = os.path.join(plugin_dir, "scripts", "download_fonts.py")
    if os.path.exists(_font_script):
        subprocess.run(
            [sys.executable, _font_script],
            capture_output=True,
            timeout=120,
        )
except Exception:
    pass  # Non-critical — don't block plugin loading

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
