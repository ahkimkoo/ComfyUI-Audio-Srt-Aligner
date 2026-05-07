"""
VideoSrtOverlay — Render SRT subtitles onto image frames in ComfyUI.

Strategy: PIL pre-renders each unique subtitle text ONCE (shadow, border,
blur, etc.), stores the result as a cropped RGBA tensor.  Per-frame
compositing is a single torch alpha-blend — no PIL overhead per frame,
no numpy round-trip, correct on CPU / CUDA / MPS.
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Import SRT parser via sys.modules (pre-loaded by __init__.py with unique name)
# ---------------------------------------------------------------------------
_srt_parser_module = sys.modules.get("audio_srt_aligner_utils_srt_parser")
if _srt_parser_module is not None:
    parse_srt = _srt_parser_module.parse_srt
    SrtEntry = _srt_parser_module.SrtEntry
else:
    _plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _plugin_dir not in sys.path:
        sys.path.insert(0, _plugin_dir)
    from utils.srt_parser import parse_srt, SrtEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    raise ValueError(f"Invalid hex color: {hex_color}")


def _text_position(
    img_w: int, img_h: int, text_w: int,
    y_position: float, x_margin: float,
) -> Tuple[float, float]:
    """Calculate subtitle anchor position.

    *y_position* (0.0–1.0): fraction from top where the subtitle sits.
      0.3 means the subtitle text top is at 30% of image height.
    *x_margin* (0.0–0.5): total horizontal margin as fraction of image width,
      split equally on both sides.  0.30 means left 15% + right 15%.
      Text is centered within the remaining (1 - x_margin) width.
    """
    usable_w = img_w * (1.0 - x_margin)
    margin_left = img_w * (x_margin / 2.0)
    x = margin_left + (usable_w - text_w) / 2.0
    y = img_h * y_position
    return x, y


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    """Wrap *text* into multiple lines so each line fits within *max_width* pixels.

    Splits at character boundaries (safe for CJK).  Returns the text with
    ``\\n`` line breaks, or the original text if it fits in one line.
    """
    # Check if single-line fits
    bbox = font.getbbox(text)  # (left, top, right, bottom)
    if bbox[2] - bbox[0] <= max_width:
        return text

    lines: List[str] = []
    current_line = ""
    for ch in text:
        test = current_line + ch
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w > max_width and current_line:
            lines.append(current_line)
            current_line = ch
        else:
            current_line = test
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
# Pre-render: PIL → cropped RGBA torch tensor (called once per unique text)
# ---------------------------------------------------------------------------

def _pre_render_subtitle(
    text: str,
    img_w: int,
    img_h: int,
    font: ImageFont.FreeTypeFont,
    font_color: str,
    border_color: str,
    border_size: int,
    shadow_color: str,
    shadow_size: int,
    shadow_offset_x: int,
    shadow_offset_y: int,
    y_position: float,
    x_margin: float,
) -> Tuple[torch.Tensor, int, int]:
    """Render *text* with all visual effects onto a full-size RGBA canvas,
    then crop to the tight bounding box of visible pixels.

    Long text is automatically wrapped to stay within the margin area.
    Each line is horizontally centered within the (1 - x_margin) width.

    Returns ``(rgba_tensor, crop_x, crop_y)`` where:
      - *rgba_tensor* has shape ``(crop_h, crop_w, 4)``, dtype float32 [0,1]
      - *(crop_x, crop_y)* is where the top-left of the crop sits on the
        original image so the composite lands at the right spot.
    """
    usable_w = int(img_w * (1.0 - x_margin))
    margin_left = img_w * (x_margin / 2.0)

    # --- Wrap text if it exceeds usable width ---
    wrapped_text = _wrap_text(text, font, usable_w)
    text_lines = wrapped_text.split("\n")

    # --- Canvas ---
    canvas = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # --- Measure total height for vertical centering within the y slot ---
    line_spacing = 8  # pixels between lines
    line_heights = []
    for line in text_lines:
        bbox = font.getbbox(line)
        line_heights.append(bbox[3] - bbox[1])
    total_text_h = sum(line_heights) + line_spacing * max(0, len(text_lines) - 1)

    y = img_h * y_position

    # --- Shadow layer (render all lines) ---
    if shadow_size > 0:
        shadow_rgb = _hex_to_rgb(shadow_color)
        shadow_layer = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_layer)
        line_y = y
        for i, line in enumerate(text_lines):
            bbox = font.getbbox(line)
            lw = bbox[2] - bbox[0]
            lx = margin_left + (usable_w - lw) / 2.0
            sdraw.text(
                (lx + shadow_offset_x, line_y + shadow_offset_y),
                line, fill=(*shadow_rgb, 255), font=font,
            )
            line_y += line_heights[i] + line_spacing
        if shadow_size > 1:
            shadow_layer = shadow_layer.filter(
                ImageFilter.GaussianBlur(radius=shadow_size),
            )
        canvas = Image.alpha_composite(canvas, shadow_layer)
        draw = ImageDraw.Draw(canvas)

    # --- Border + Main text per line ---
    border_rgb = _hex_to_rgb(border_color)
    font_rgb = _hex_to_rgb(font_color)

    line_y = y
    for i, line in enumerate(text_lines):
        bbox = font.getbbox(line)
        lw = bbox[2] - bbox[0]
        lx = margin_left + (usable_w - lw) / 2.0

        if border_size > 0:
            draw.text(
                (lx, line_y), line,
                fill=(*border_rgb, 255), font=font,
                stroke_width=border_size, stroke_fill=(*border_rgb, 255),
            )
        draw.text((lx, line_y), line, fill=(*font_rgb, 255), font=font)
        line_y += line_heights[i] + line_spacing

    # --- Crop to visible bounding box ---
    rgba_np = np.array(canvas)                       # (H, W, 4) uint8
    alpha_ch = rgba_np[:, :, 3]
    ys, xs = np.nonzero(alpha_ch > 0)
    if len(ys) == 0:
        return torch.zeros(1, 1, 4, dtype=torch.float32), 0, 0

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    # 1-px pad for anti-aliased edges
    y1, x1 = max(0, y1 - 1), max(0, x1 - 1)
    y2, x2 = min(img_h, y2 + 1), min(img_w, x2 + 1)

    cropped = rgba_np[y1:y2, x1:x2].copy()           # (crop_h, crop_w, 4)
    rgba_tensor = torch.from_numpy(cropped).float() / 255.0
    return rgba_tensor, x1, y1


# ---------------------------------------------------------------------------
# Per-frame composite (pure torch, works on CPU / CUDA / MPS)
# ---------------------------------------------------------------------------

def _composite_subtitle(
    frame: torch.Tensor,
    rgba: torch.Tensor,
    x: int,
    y: int,
    opacity: float,
) -> None:
    """Alpha-blend *rgba* onto *frame* **in-place**.

    Only touches the pixels covered by the subtitle crop.
    Handles boundary clipping when the subtitle extends past image edges.
    """
    sh, sw = rgba.shape[0], rgba.shape[1]
    fh, fw = frame.shape[0], frame.shape[1]

    # Source region inside rgba
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = sw + min(0, fw - x - sw)
    sy2 = sh + min(0, fh - y - sh)

    # Destination region on frame
    dx1 = max(0, x)
    dy1 = max(0, y)
    dx2 = min(fw, x + sw)
    dy2 = min(fh, y + sh)

    if dx1 >= dx2 or dy1 >= dy2:
        return  # no overlap

    sub = rgba[sy1:sy2, sx1:sx2]              # (rh, rw, 4)
    alpha = sub[:, :, 3:4] * opacity          # (rh, rw, 1)
    rgb = sub[:, :, :3]                        # (rh, rw, 3)

    # alpha blend:  out = rgb * a + dst * (1 - a)
    dst = frame[dy1:dy2, dx1:dx2]             # view, NOT a copy
    frame[dy1:dy2, dx1:dx2] = rgb * alpha + dst * (1.0 - alpha)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class VideoSrtOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = cls._scan_fonts()
        return {
            "required": {
                "images": ("IMAGE",),
                "srt_string": ("STRING", {"multiline": True, "default": ""}),
                "font_family": (font_list if font_list else [""], {"default": font_list[0] if font_list else ""}),
                "font_size": ("INT", {"default": 120, "min": 12, "max": 200, "step": 1}),
                "font_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "border_color": ("STRING", {"default": "#000000", "multiline": False}),
                "border_size": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "shadow_color": ("STRING", {"default": "#000000", "multiline": False}),
                "shadow_size": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "shadow_offset_x": ("INT", {"default": 2, "min": -20, "max": 20, "step": 1}),
                "shadow_offset_y": ("INT", {"default": 2, "min": -20, "max": 20, "step": 1}),
                "effect": (["fade", "none"], {"default": "fade"}),
                "fade_in_duration": ("INT", {"default": 300, "min": 0, "max": 2000, "step": 50}),
                "fade_out_duration": ("INT", {"default": 300, "min": 0, "max": 2000, "step": 50}),
            },
            "optional": {
                "subtitle_y_position": (
                    "FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"},
                ),
                "subtitle_x_margin": (
                    "FLOAT", {"default": 0.20, "min": 0.0, "max": 0.5, "step": 0.01, "display": "number"},
                ),
                "fps": (
                    "FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1, "display": "number"},
                ),
            }
        }

    @staticmethod
    def _scan_fonts() -> List[str]:
        """Scan font directories and return a list of font filenames.

        Aggregates files from:
          1. ``<comfyui>/models/fonts/`` (if inside ComfyUI)
          2. Plugin's own ``fonts/`` directory

        Returns sorted list of filenames (display names in the dropdown).
        """
        FONT_EXTS = {'.ttf', '.otf', '.ttc', '.TTF', '.OTF', '.TTC'}

        dirs: List[str] = []

        # 1. Plugin bundled fonts (always available)
        _plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs.append(os.path.join(_plugin_dir, "fonts"))

        # 2. ComfyUI models/fonts/ (if inside ComfyUI)
        try:
            import folder_paths
            comfyui_fonts = os.path.join(folder_paths.models_dir, "fonts")
            if os.path.isdir(comfyui_fonts):
                dirs.append(comfyui_fonts)
        except Exception:
            pass

        seen: set = set()
        results: List[str] = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if any(fname.endswith(ext) for ext in FONT_EXTS):
                    if fname not in seen:
                        seen.add(fname)
                        results.append(fname)
        return results

    @staticmethod
    def _font_filename_to_path(filename: str) -> str:
        """Resolve a dropdown filename back to a full file path.

        Checks plugin ``fonts/`` first, then ``models/fonts/``.
        Returns empty string if not found.
        """
        FONT_EXTS = ('.ttf', '.otf', '.ttc')
        if not filename:
            return ""

        _plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = [os.path.join(_plugin_dir, "fonts")]

        try:
            import folder_paths
            comfyui_fonts = os.path.join(folder_paths.models_dir, "fonts")
            if os.path.isdir(comfyui_fonts):
                dirs.append(comfyui_fonts)
        except Exception:
            pass

        for d in dirs:
            candidate = os.path.join(d, filename)
            if os.path.isfile(candidate):
                return candidate
        # Filename without extension — try extensions
        for d in dirs:
            for ext in FONT_EXTS:
                candidate = os.path.join(d, filename + ext)
                if os.path.isfile(candidate):
                    return candidate
        return ""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_with_subtitle",)
    FUNCTION = "process"
    CATEGORY = "video/srt"

    def process(
        self,
        images: torch.Tensor,
        srt_string: str,
        font_family: str,
        font_size: int,
        font_color: str,
        border_color: str,
        border_size: int,
        shadow_color: str,
        shadow_size: int,
        shadow_offset_x: int,
        shadow_offset_y: int,
        effect: str,
        fade_in_duration: int,
        fade_out_duration: int,
        subtitle_y_position: float = 0.3,
        subtitle_x_margin: float = 0.3,
        fps: float = 24.0,
    ) -> Tuple[torch.Tensor, ...]:
        """Render SRT subtitles onto image batch.

        images: (B, H, W, C) float32 [0, 1]
        Returns: same-shape tensor with subtitles composited.
        """
        if images is None or images.numel() == 0:
            raise Exception("[VideoSrtOverlay] Empty images input")

        # --- Parse SRT ---
        try:
            entries = parse_srt(srt_string)
        except ValueError as e:
            raise Exception(f"[VideoSrtOverlay] SRT parse error: {e}")

        if not entries:
            return (images,)

        # --- Resolve font ---
        font_path = self._font_filename_to_path(font_family)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except (IOError, OSError):
            font = ImageFont.load_default()

        batch_size, height, width, channels = images.shape

        # --- Use SRT entries as-is: no merging, no splitting ---
        # Each entry: (start_sec, end_sec, original_text)
        groups: List[Tuple[float, float, str]] = [
            (e.start, e.end, e.text) for e in entries
        ]

        # --- Pre-render each unique subtitle text (PIL, once) ---
        pre_rendered: Dict[str, Tuple[torch.Tensor, int, int]] = {}
        for _, _, text in groups:
            if text not in pre_rendered:
                rgba, px, py = _pre_render_subtitle(
                    text, width, height, font,
                    font_color, border_color, border_size,
                    shadow_color, shadow_size,
                    shadow_offset_x, shadow_offset_y,
                    subtitle_y_position, subtitle_x_margin,
                )
                # Move to same device as images (CPU / CUDA / MPS)
                pre_rendered[text] = (rgba.to(images.device), px, py)

        # --- Build frame → group lookup (O(B·G), tiny for typical SRT) ---
        has_fade = effect == "fade" and (fade_in_duration > 0 or fade_out_duration > 0)
        fade_in_sec = fade_in_duration / 1000.0
        fade_out_sec = fade_out_duration / 1000.0

        frame_plan: List[Tuple[int, float, float, str]] = []  # (fidx, g_start, g_end, text)
        for fidx in range(batch_size):
            t = fidx / fps
            for g_start, g_end, g_text in groups:
                if g_start <= t <= g_end:
                    frame_plan.append((fidx, g_start, g_end, g_text))
                    break  # first matching group wins

        if not frame_plan:
            return (images,)

        # --- Composite: one torch alpha-blend per subtitle frame ---
        result = images.clone()

        for fidx, g_start, g_end, g_text in frame_plan:
            rgba, px, py = pre_rendered[g_text]
            t = fidx / fps

            # Opacity (fade in / fade out)
            if has_fade:
                if t < g_start + fade_in_sec:
                    opacity = (
                        max(0.0, (t - g_start) / fade_in_sec)
                        if fade_in_sec > 0 else 1.0
                    )
                elif t > g_end - fade_out_sec:
                    opacity = (
                        max(0.0, (g_end - t) / fade_out_sec)
                        if fade_out_sec > 0 else 1.0
                    )
                else:
                    opacity = 1.0
            else:
                opacity = 1.0

            if opacity > 0.0:
                _composite_subtitle(result[fidx], rgba, px, py, opacity)

        return (result,)
