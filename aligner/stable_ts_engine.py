"""Alignment engine powered by stable-ts for the proofreading (with-reference) scenario.

Uses ``stable_whisper.load_faster_whisper`` which reuses the same faster-whisper
backend the existing engine depends on, but adds stable-ts's ``model.align()``
API for direct audio↔text alignment — no manual token-matching required.

Keep ``engine.py`` for the no-reference-text (pure transcription) scenario.
This module is for the proofreading scenario where the user already has an
accurate reference text that just needs timestamp alignment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Re-export AlignmentConfig so callers only need one import.
# We intentionally do NOT import faster_whisper at module level — the
# existing engine.py already handles that dependency gate, and we want
# lazy imports so ``stable-ts`` is only required when this engine is used.
# ---------------------------------------------------------------------------

from aligner.engine import AlignmentConfig  # noqa: F401 — public re-export


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_model_path(model_name: str) -> str:
    """Resolve *model_name* to a concrete path.

    Priority:
    1. Local ComfyUI model directory (``<models_dir>/stt/whisper/<model_name>``).
    2. Bare model name — let faster-whisper use its own cache or download.
    """
    # Check ComfyUI's local model directory
    try:
        import folder_paths  # type: ignore[import-untyped]

        local_dir = os.path.join(folder_paths.models_dir, "stt", "whisper")
        local_path = os.path.join(local_dir, model_name)
        if os.path.isdir(local_path):
            return local_path
    except Exception:
        pass  # Running outside ComfyUI — fall through

    # Let faster-whisper handle caching and downloading via its own mechanism
    # (it respects HF_ENDPOINT for mirror, and uses HuggingFace hub cache)
    return model_name

    # 3. Fallback: let faster-whisper handle it
    return model_name


def _count_meaningful_chars(text: str) -> int:
    """Count characters excluding whitespace and punctuation."""
    import unicodedata
    return sum(
        1 for ch in text
        if not ch.isspace()
        and unicodedata.category(ch)[0] != "P"
    )


def _format_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments) -> str:
    """Build an SRT string from a list of stable-ts Segment objects."""
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        text = getattr(seg, "text", "").strip()
        if not text:
            continue
        start = _format_timestamp(getattr(seg, "start", 0.0))
        end = _format_timestamp(getattr(seg, "end", 0.0))
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank separator
    return "\n".join(lines).strip() + "\n" if lines else ""


# ---------------------------------------------------------------------------
# Lazy singleton model cache
# ---------------------------------------------------------------------------

_model_cache: dict[tuple[str, str, str], object] = {}


def _get_model(model_name: str, device: str, compute_type: str):
    """Return a cached ``stable_whisper.WhisperModel`` instance.

    The cache key is ``(model_name, device, compute_type)`` — if any of
    these change, a new model is loaded (the old one is released for GC).
    """
    cache_key = (model_name, device, compute_type)

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Lazy import — stable-ts is only required when this engine is used
    try:
        import stable_whisper  # type: ignore[import-untyped]
    except ImportError as exc:
        raise Exception(
            "stable-ts not installed. Run: pip install stable-ts[fw]"
        ) from exc

    resolved_path = _resolve_model_path(model_name)

    if cache_key in _model_cache:
        # Another thread may have loaded while we were resolving — check again
        return _model_cache[cache_key]

    model = stable_whisper.load_faster_whisper(
        model_size_or_path=resolved_path,
        device=device,
        compute_type=compute_type,
    )
    _model_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_srt_string(
    audio_path: Path,
    reference_text: str,
    config: AlignmentConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str, int, float]:
    """Align *reference_text* to *audio_path* using stable-ts and return SRT.

    This is a drop-in replacement for :func:`aligner.engine.generate_srt_string`
    that leverages ``stable_whisper``'s ``model.align()`` for direct
    audio↔text alignment.  Use it when a high-quality reference transcript
    is available (proofreading scenario).

    Args:
        audio_path: Path to the input audio file.
        reference_text: The reference transcript to align.
        config: Alignment configuration (only ``model_name``, ``device``,
            ``compute_type``, and ``language`` fields are used).
        progress: Optional callback for progress messages.

    Returns:
        Tuple of ``(srt_string, detected_language, srt_entry_count, coverage)``.

    Raises:
        Exception: If stable-ts is not installed or alignment fails.
    """
    # --- Validation ---------------------------------------------------------
    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    if not reference_text or not reference_text.strip():
        raise ValueError("Reference text must not be empty.")

    # --- Lazy import gate ---------------------------------------------------
    try:
        import stable_whisper  # type: ignore[import-untyped]  # noqa: F811
    except ImportError as exc:
        raise Exception(
            "stable-ts not installed. Run: pip install stable-ts[fw]"
        ) from exc

    # --- Model loading ------------------------------------------------------
    if progress:
        progress("Loading stable-ts model…")

    model = _get_model(config.model_name, config.device, config.compute_type)

    # --- Alignment ----------------------------------------------------------
    lang = config.language or "zh"

    if progress:
        progress(f"Aligning audio with reference text (lang={lang})…")

    try:
        result = model.align(
            audio=str(audio_path),
            text=reference_text.strip(),
            language=lang,
        )
    except Exception as exc:
        raise Exception(
            f"stable-ts alignment failed: {exc}"
        ) from exc

    # --- Build SRT string ---------------------------------------------------
    segments = result.segments if hasattr(result, "segments") else []

    srt_string = _segments_to_srt(segments)
    srt_entry_count = sum(
        1 for seg in segments if getattr(seg, "text", "").strip()
    )

    # --- Coverage -----------------------------------------------------------
    ref_chars = _count_meaningful_chars(reference_text)
    aligned_text = " ".join(
        getattr(seg, "text", "") for seg in segments
    )
    aligned_chars = _count_meaningful_chars(aligned_text)

    coverage = (aligned_chars / ref_chars * 100.0) if ref_chars > 0 else 0.0

    # --- Language -----------------------------------------------------------
    detected_language = getattr(result, "language", None) or lang

    if progress:
        progress(
            f"Alignment complete: {srt_entry_count} entries, "
            f"{coverage:.1f}% coverage, lang={detected_language}"
        )

    return srt_string, detected_language, srt_entry_count, coverage
