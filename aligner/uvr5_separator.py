"""UVR5 vocal separation using audio-separator library."""

import logging
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Model file names for each mode
UVR5_MODEL_MAP = {
    "roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "mdxnet": "UVR_MDXNET_KARA_2.onnx",
}

# UI dropdown options
UVR5_MODE_OPTIONS = ["roformer", "mdxnet", "none"]

# Hardcoded model directory (relative to ComfyUI root)
UVR5_MODEL_DIR = "models/uvr5"


def resolve_uvr5_model(uvr5_mode: str) -> Optional[str]:
    """Resolve a uvr5_mode string to a model filename.

    Returns None when uvr5_mode is "none" (skip separation).
    """
    if uvr5_mode == "none" or not uvr5_mode:
        return None
    return UVR5_MODEL_MAP.get(uvr5_mode)


def separate_vocals(
    audio_path: str,
    model_filename: str,
    model_dir: str = UVR5_MODEL_DIR,
    output_dir: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Separate vocals from audio file using UVR5 models.

    Args:
        audio_path: Path to input audio file.
        model_filename: Name of the model file.
        model_dir: Directory where models are stored/downloaded.
        output_dir: Directory for output files. If None, uses temp directory.
        progress: Optional progress callback.

    Returns:
        Path to the separated vocals WAV file.

    Raises:
        ImportError: If audio-separator is not installed.
        FileNotFoundError: If audio file doesn't exist.
        Exception: If separation fails.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise ImportError(
            "audio-separator is required for vocal separation. "
            "Install it with: pip install 'audio-separator[gpu]'"
        )

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Ensure model directory exists
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"[AudioSrtAligner] Loading UVR5 model: {model_filename}")

    # Create separator instance
    separator = Separator(
        model_file_dir=str(model_path),
        output_dir=output_dir,
        output_single_stem="Vocals",
        sample_rate=44100,
    )

    if progress:
        progress(f"[AudioSrtAligner] Running vocal separation...")

    # Load model and separate
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(str(audio_file))

    if not output_files:
        raise Exception("UVR5 separation produced no output files.")

    # Find the vocals file
    vocals_path = None
    for f in output_files:
        if "Vocals" in f or "vocals" in f:
            vocals_path = f
            break

    if vocals_path is None:
        # Use first output file if no vocals-specific file found
        vocals_path = output_files[0]

    if progress:
        progress(f"[AudioSrtAligner] Vocal separation complete: {vocals_path}")

    return vocals_path
