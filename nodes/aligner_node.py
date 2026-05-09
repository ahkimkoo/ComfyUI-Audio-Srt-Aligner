"""
ComfyUI AudioSrtAligner node.

Takes an AUDIO tensor (from LoadAudio, VAE decode, etc.) and reference text,
runs the alignment engine to produce an SRT subtitle string.
"""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Import engine via sys.modules (pre-loaded by __init__.py with unique name)
# ---------------------------------------------------------------------------
_engine_module = sys.modules.get("audio_srt_aligner_engine")
if _engine_module is not None:
    AlignmentConfig = _engine_module.AlignmentConfig
    create_alignment_config = _engine_module.create_alignment_config
    generate_srt_string = _engine_module.generate_srt_string
else:
    # Fallback for standalone testing outside ComfyUI
    _plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from aligner.engine import AlignmentConfig, create_alignment_config, generate_srt_string

_srt_parser = sys.modules.get("audio_srt_aligner_utils_srt_parser")
if _srt_parser is not None:
    SrtEntry = _srt_parser.SrtEntry
    parse_srt = _srt_parser.parse_srt
    format_srt = _srt_parser.format_srt
else:
    # Fallback for standalone testing
    from utils.srt_parser import SrtEntry, parse_srt, format_srt

_uvr5_module = sys.modules.get("audio_srt_aligner_aligner_uvr5_separator")
if _uvr5_module is not None:
    UVR5_MODE_OPTIONS = _uvr5_module.UVR5_MODE_OPTIONS
else:
    _plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from aligner.uvr5_separator import UVR5_MODE_OPTIONS

# ---------------------------------------------------------------------------
# Whisper model directory — tell faster-whisper where to find local models
# ---------------------------------------------------------------------------
try:
    import folder_paths

    WHISPER_MODEL_DIR = os.path.join(folder_paths.models_dir, "stt", "whisper")
    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    os.environ["FASTER_WHISPER_MODEL_DIR"] = WHISPER_MODEL_DIR
except Exception:
    # Running outside ComfyUI — let the engine fall back to default resolution
    pass

# China mirror for HuggingFace model downloads
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _count_chars(text: str) -> int:
    """Count subtitle characters using the user-defined rules.

    - Chinese character / CJK punctuation / any non-ASCII: 1 each
    - ASCII space: 1
    - ASCII punctuation: 1
    - English letters: grouped in pairs, ceil(n/2)
    """
    count = 0
    i = 0
    while i < len(text):
        c = text[i]
        if ord(c) > 127:
            # CJK or other non-ASCII — always 1
            count += 1
            i += 1
        elif c.isspace():
            count += 1
            i += 1
        elif not c.isalpha():
            # ASCII punctuation or digit — 1 each
            count += 1
            i += 1
        else:
            # English letters — count consecutive group, ceil(n/2)
            j = i + 1
            while j < len(text) and text[j].isascii() and text[j].isalpha():
                j += 1
            letter_count = j - i
            count += math.ceil(letter_count / 2)
            i = j
    return count


# Punctuation & newline pattern used for splitting and cleaning.
_PUNCT_PATTERN = re.compile(
    r'[\n\r，。！？；：、…—""''（）【】《》「」'
    r'\[\]{}(),.!?;:\-\u3000]'
)


def _split_by_punctuation(text: str) -> List[str]:
    """Split *text* by punctuation marks and newlines into segments."""
    parts = _PUNCT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def _clean_text(text: str) -> str:
    """Remove punctuation and extra whitespace, returning only readable chars."""
    return _PUNCT_PATTERN.sub("", text).replace(" ", "")


# ---------------------------------------------------------------------------
# Chinese word segmentation via jieba
# ---------------------------------------------------------------------------

def _segment_chinese(text: str) -> List[str]:
    """Segment Chinese text into words using jieba.

    Returns a list of tokens.  Non-CJK runs (ASCII letters, digits,
    punctuation, spaces) are kept as-is.
    """
    import jieba
    # jieba.cut produces generator; filter empty strings
    return [t for t in jieba.cut(text) if t.strip()]


def _split_text_at_limit(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks whose char-count ≤ *max_chars*.

    Uses word-level segmentation so that splits always occur at word
    boundaries, never in the middle of a word.  Each chunk is at least
    one complete word (never a single trailing character cut from a word).
    """
    if _count_chars(text) <= max_chars:
        return [text]

    tokens = _segment_chinese(text)

    chunks: List[str] = []
    cur_tokens: List[str] = []
    cur_count = 0

    for token in tokens:
        tc = _count_chars(token)
        # +1 for joining space between tokens in same chunk
        space = 1 if cur_tokens else 0
        if cur_count + space + tc > max_chars and cur_tokens:
            # Flush current chunk
            chunks.append("".join(cur_tokens))
            cur_tokens = [token]
            cur_count = tc
        else:
            cur_tokens.append(token)
            cur_count += space + tc

    if cur_tokens:
        chunks.append("".join(cur_tokens))

    return chunks if chunks else [text]


def _adjust_srt_by_char_limit(
    reference_text: str,
    srt_string: str,
    max_chars: int,
) -> Tuple[str, int]:
    """Re-segment SRT entries using the original reference text's punctuation.

    1. Split ``reference_text`` by punctuation / newlines → clauses.
    2. Build a character→time mapping from the engine-produced SRT entries.
    3. Greedily group consecutive clauses into subtitle lines (each ≤
       *max_chars*).  Joined clauses are separated by a single space
       (the original punctuation is discarded).
    4. Any single clause that still exceeds *max_chars* is hard-split.
    5. Timing for each output line is derived from the char→time map.

    Returns (adjusted_srt_string, new_entry_count).
    """
    if max_chars <= 0:
        return srt_string, -1

    entries = parse_srt(srt_string)
    if not entries:
        return srt_string, 0

    # ---- Build char→time arrays from SRT entries ----
    MAX_CHARS_PER_SEC = 3.0
    char_starts: List[float] = []
    char_ends: List[float] = []
    all_clean = ""

    for entry in entries:
        clean = entry.text.replace(" ", "")
        if not clean:
            continue
        # Skip likely hallucination entries
        dur = entry.end - entry.start
        if dur > 10.0:
            density = len(clean) / dur
            if density < MAX_CHARS_PER_SEC:
                continue
            unique_chars = len(set(clean))
            if len(clean) > 3 and unique_chars / len(clean) < 0.4:
                continue
        n = len(clean)
        for j in range(n):
            frac_start = j / n
            frac_end = (j + 1) / n
            char_starts.append(entry.start + frac_start * dur)
            char_ends.append(entry.start + frac_end * dur)
        all_clean += clean

    if not all_clean:
        return srt_string, len(entries)

    # ---- Split reference text into clauses ----
    clauses = _split_by_punctuation(reference_text)
    if not clauses:
        return srt_string, len(entries)

    # ---- Locate each clause in the clean text ----
    clause_spans: List[Tuple[int, int, str]] = []  # (clean_start, clean_end, text)
    search_pos = 0
    for clause in clauses:
        clause_clean = _clean_text(clause)
        if not clause_clean:
            continue
        idx = all_clean.find(clause_clean, search_pos)
        if idx >= 0:
            clause_spans.append((idx, idx + len(clause_clean), clause))
            search_pos = idx + len(clause_clean)
        # else: skip (shouldn't happen with a well-aligned engine)

    if not clause_spans:
        return srt_string, len(entries)

    n_chars = len(char_starts)

    # ---- Greedy group by max_chars AND max duration ----
    MAX_ENTRY_DURATION = 8.0  # seconds — hard cap per subtitle entry
    groups: List[List[Tuple[int, int, str]]] = []
    cur_group: List[Tuple[int, int, str]] = []
    cur_count = 0
    cur_start_time = 0.0

    for span in clause_spans:
        _, _, clause_text = span
        cc = _count_chars(clause_text)
        # Estimate this clause's timing
        span_cs = max(0, min(span[0], n_chars - 1))
        span_ce = max(0, min(span[1] - 1, n_chars - 1))
        clause_start = char_starts[span_cs] if span_cs < len(char_starts) else 0.0
        clause_end = char_ends[span_ce] if span_ce < len(char_ends) else 0.0

        if cur_group:
            combined = cur_count + 1 + cc  # +1 for joining space
            combined_dur = clause_end - cur_start_time
            if combined <= max_chars and combined_dur <= MAX_ENTRY_DURATION:
                cur_group.append(span)
                cur_count = combined
            else:
                groups.append(cur_group)
                cur_group = [span]
                cur_count = cc
                cur_start_time = clause_start
        else:
            cur_group = [span]
            cur_count = cc
            cur_start_time = clause_start

    if cur_group:
        groups.append(cur_group)

    # ---- Build final SRT entries with timing ----
    final: List[SrtEntry] = []

    for group in groups:
        first_cs = group[0][0]
        last_ce = group[-1][1]  # exclusive end in clean text

        # Clamp indices into valid range
        si = max(0, min(first_cs, n_chars - 1))
        ei = max(0, min(last_ce - 1, n_chars - 1))

        start_time = char_starts[si]
        end_time = char_ends[ei]

        display_text = " ".join(s[2] for s in group)

        # Hard-split if this line still exceeds limit (single long clause)
        if _count_chars(display_text) > max_chars:
            parts = _split_text_at_limit(display_text, max_chars)
            total_c = sum(_count_chars(p) for p in parts) or 1
            dur = end_time - start_time
            elapsed = 0.0
            for part in parts:
                pc = _count_chars(part)
                pd = (pc / total_c) * dur
                final.append(SrtEntry(
                    index=0,
                    start=start_time + elapsed,
                    end=start_time + elapsed + pd,
                    text=part,
                ))
                elapsed += pd
        else:
            final.append(SrtEntry(
                index=0, start=start_time, end=end_time, text=display_text,
            ))

    # Renumber
    for idx, e in enumerate(final):
        e.index = idx + 1

    return format_srt(final), len(final)


def _process_srt_entries(
    entries: List[SrtEntry],
    max_chars: int,
) -> Tuple[str, int]:
    """Process SRT entries: split by punctuation, clean, and apply max_chars limit.

    This function is used when there is no reference text (raw Whisper transcription).
    It applies the same punctuation-based splitting and max_chars logic.

    1. Split each entry's text by punctuation into clauses.
    2. Remove punctuation from clauses (keep only readable chars).
    3. Filter out hallucinated segments (very few chars spanning very long time).
    4. Rebuild SRT entries grouping clauses up to max_chars with duration cap.
    5. Join multiple clauses with space; single clause exceeding max_chars is word-split.
    6. Redistribute timing proportionally.

    Returns (processed_srt_string, entry_count).
    """
    if not entries:
        return "", 0

    MAX_ENTRY_DURATION = 8.0  # seconds — hard cap per subtitle entry
    MAX_CHARS_PER_SEC = 3.0   # chars/sec threshold — below this is likely hallucination

    # Collect all clauses with their original timing info
    # Each item: (start_time, end_time, clause_text)
    all_clauses: List[Tuple[float, float, str]] = []

    for entry in entries:
        text = entry.text
        if not text or not text.strip():
            continue

        entry_dur = entry.end - entry.start

        # Skip entries that are likely hallucinations:
        # 1. very few characters spanning a very long time
        # 2. text has very low unique character ratio (repeating gibberish like "A A a a sue")
        if entry_dur > 10.0:
            chars = len(text.strip())
            density = chars / entry_dur  # chars per second
            if density < MAX_CHARS_PER_SEC:
                continue
            # Check for repetitive gibberish: if unique chars < 40% of total, it's likely noise
            unique_chars = len(set(text.strip()))
            if chars > 3 and unique_chars / chars < 0.4:
                continue

        # Split by punctuation
        clauses = _split_by_punctuation(text)
        if not clauses:
            continue

        # Calculate time per character for this entry
        entry_chars = sum(len(c) for c in clauses)
        if entry_chars == 0:
            entry_chars = 1

        current_time = entry.start
        for clause in clauses:
            clause_clean = _clean_text(clause)
            if not clause_clean:
                continue
            # Estimate duration based on character count
            clause_chars = len(clause_clean)
            clause_dur = (clause_chars / entry_chars) * entry_dur
            clause_end = min(current_time + clause_dur, entry.end)

            # If a single clause spans too long, split by character
            if clause_dur > MAX_ENTRY_DURATION:
                n = len(clause_clean)
                chars_per_sec = n / clause_dur
                chunk_chars = max(1, int(MAX_ENTRY_DURATION * chars_per_sec))
                if chunk_chars < 2 and n > 2:
                    chunk_chars = 2
                for ci in range(0, n, chunk_chars):
                    sub_text = clause_clean[ci:ci + chunk_chars]
                    if not sub_text:
                        continue
                    frac_s = ci / n
                    frac_e = min((ci + chunk_chars) / n, 1.0)
                    sub_start = current_time + clause_dur * frac_s
                    sub_end = current_time + clause_dur * frac_e
                    all_clauses.append((sub_start, sub_end, sub_text))
            else:
                all_clauses.append((current_time, clause_end, clause_clean))

            current_time = clause_end

    if not all_clauses:
        return format_srt(entries), len(entries)

    # Group clauses by max_chars AND max duration
    groups: List[List[Tuple[float, float, str]]] = []
    cur_group: List[Tuple[float, float, str]] = []
    cur_count = 0
    cur_dur = 0.0

    for clause_info in all_clauses:
        start, end, clause_text = clause_info
        cc = _count_chars(clause_text)
        clause_dur = end - start
        if cur_group:
            combined_count = cur_count + 1 + cc  # +1 for space between clauses
            combined_dur = cur_dur + clause_dur
            if combined_count <= max_chars and combined_dur <= MAX_ENTRY_DURATION:
                cur_group.append(clause_info)
                cur_count = combined_count
                cur_dur = combined_dur
            else:
                groups.append(cur_group)
                cur_group = [clause_info]
                cur_count = cc
                cur_dur = clause_dur
        else:
            cur_group = [clause_info]
            cur_count = cc
            cur_dur = clause_dur

    if cur_group:
        groups.append(cur_group)

    # Build final SRT entries
    final: List[SrtEntry] = []

    for group in groups:
        first_start = group[0][0]
        last_end = group[-1][1]
        # Join clauses with space
        display_text = " ".join(c[2] for c in group)

        # Check if single clause exceeds max_chars and needs word-splitting
        if len(group) == 1 and _count_chars(display_text) > max_chars:
            parts = _split_text_at_limit(display_text, max_chars)
            total_chars = sum(_count_chars(p) for p in parts) or 1
            dur = last_end - first_start
            elapsed = 0.0
            for part in parts:
                pc = _count_chars(part)
                pd = (pc / total_chars) * dur
                final.append(SrtEntry(
                    index=0,
                    start=first_start + elapsed,
                    end=first_start + elapsed + pd,
                    text=part,
                ))
                elapsed += pd
        else:
            final.append(SrtEntry(
                index=0, start=first_start, end=last_end, text=display_text,
            ))

    # Renumber
    for idx, e in enumerate(final):
        e.index = idx + 1

    return format_srt(final), len(final)


def _audio_to_wav_file(audio: dict) -> str:
    """Convert ComfyUI AUDIO dict to a temporary WAV file.

    ComfyUI AUDIO is ``{"waveform": tensor(batch, channels, samples),
    "sample_rate": int}``.

    Returns the temporary file path (caller is responsible for cleanup).
    """
    waveform = audio["waveform"].cpu()     # (batch, channels, samples) float32
    sample_rate = int(audio["sample_rate"])

    # Take first item from batch; convert stereo to mono
    waveform = waveform[0]                  # (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Create temp WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_path = tmp.name

    try:
        import av
        layout = "mono"
        container = av.open(wav_path, mode="w", format="wav")
        stream = container.add_stream("pcm_s16le", rate=sample_rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(
            waveform.numpy(), format="flt", layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()
    except ImportError:
        # Fallback: use scipy or wave module
        import struct
        import numpy as np
        audio_data = (waveform[0].numpy() * 32767).clip(-32768, 32767).astype(np.int16)
        with open(wav_path, "wb") as f:
            f.write(b"RIFF")
            data_len = len(audio_data) * 2
            f.write(struct.pack("<I", 36 + data_len))
            f.write(b"WAVEfmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))   # PCM
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))
            f.write(struct.pack("<H", 2))   # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_len))
            f.write(audio_data.tobytes())

    return wav_path


class AudioSrtAligner:
    """ComfyUI node: align reference text to audio and produce SRT subtitle."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "audio": ("AUDIO",),
                "reference_text": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "在这里输入参考文本/台词文稿...",
                        "multiline": True,
                    },
                ),
                "model_size": (
                    ["tiny", "base", "small", "medium", "large-v3"],
                    {"default": "small"},
                ),
                "language": (
                    "STRING",
                    {
                        "default": "zh",
                        "placeholder": "zh, en, ja... 留空自动检测",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "max_chars": ("INT", {"default": 12, "min": 1, "max": 100, "step": 1}),
                "compute_type": (
                    ["int8", "int8_float16", "float16", "float32"],
                    {"default": "int8"},
                ),
                "uvr5_mode": (UVR5_MODE_OPTIONS, {"default": "roformer"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("srt_string", "detected_language", "srt_entries", "coverage")
    FUNCTION = "process"
    CATEGORY = "audio/srt"
    DESCRIPTION = "Align reference text to audio and generate SRT subtitles"

    def process(
        self,
        audio: dict,
        reference_text: str,
        model_size: str = "small",
        language: str = "zh",
        beam_size: int = 5,
        max_chars: int = 12,
        compute_type: str = "int8",
        uvr5_mode: str = "roformer",
    ) -> Tuple[str, str, int, float]:
        """Run alignment pipeline and return SRT content."""
        # --- Validate inputs ---
        # reference_text can be empty - in that case we skip alignment and use raw Whisper transcription
        has_reference = bool(reference_text and reference_text.strip())

        # --- Convert AUDIO dict to temp WAV file ---
        wav_path: Optional[str] = None
        try:
            wav_path = _audio_to_wav_file(audio)
        except Exception as exc:
            raise Exception(f"[AudioSrtAligner] Failed to convert audio to WAV: {exc}") from exc

        # Resolve language — empty string means auto-detect
        resolved_language: Optional[str] = language.strip() if language and language.strip() else None

        # --- Build config ---
        config: AlignmentConfig = create_alignment_config(
            model_name=model_size,
            device="auto",
            compute_type=compute_type,
            language=resolved_language,
            beam_size=beam_size,
            uvr5_mode=uvr5_mode,
        )

        # --- Run alignment or raw transcription ---
        try:
            if has_reference:
                # With reference text: run alignment (校对模式)
                srt_string, detected_language, srt_entries, coverage = generate_srt_string(
                    audio_path=Path(wav_path),
                    reference_text=reference_text.strip(),
                    config=config,
                )
            else:
                # Without reference text: use raw Whisper transcription (无校对模式)
                # Import the transcribe function from engine
                _engine_module = sys.modules.get("audio_srt_aligner_engine")
                if _engine_module is None:
                    raise Exception("Engine module not loaded")
                
                timed_entries, audio_end, detected_language, segment_count = _engine_module.transcribe_to_timed_subtitles(
                    audio_path=Path(wav_path),
                    model_name=config.model_name,
                    device=config.device,
                    compute_type=config.compute_type,
                    language=config.language,
                    beam_size=config.beam_size,
                )
                
                # Convert TimedSubtitleEntry to SrtEntry and format
                srt_entries_list = [
                    SrtEntry(
                        index=i + 1,
                        start=e.start,
                        end=e.end,
                        text=e.text,
                    )
                    for i, e in enumerate(timed_entries)
                ]
                srt_string = format_srt(srt_entries_list)
                srt_entries = len(srt_entries_list)
                coverage = 0.0  # No reference text means no coverage metric
        except ValueError as exc:
            raise Exception(f"[AudioSrtAligner] {exc}") from exc
        except ImportError as exc:
            raise Exception(
                f"[AudioSrtAligner] Missing dependency: {exc}. "
                "Install with: pip install faster-whisper"
            ) from exc
        except Exception as exc:
            raise Exception(f"[AudioSrtAligner] Processing failed: {exc}") from exc
        finally:
            # Always clean up temp WAV file
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        # --- Post-process: apply max_chars subtitle length limit ---
        if max_chars and max_chars > 0:
            if has_reference:
                # With reference text: use reference-based alignment adjustment
                srt_string, srt_entries = _adjust_srt_by_char_limit(
                    reference_text.strip(), srt_string, max_chars,
                )
            else:
                # Without reference text: process raw Whisper transcription
                # Parse current SRT and apply punctuation-based splitting + max_chars
                entries = parse_srt(srt_string)
                srt_string, srt_entries = _process_srt_entries(entries, max_chars)

        return (
            srt_string,
            detected_language or "unknown",
            srt_entries,
            coverage,
        )
