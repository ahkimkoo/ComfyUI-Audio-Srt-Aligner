"""
Download Alibaba PuHuiTi Medium font for ComfyUI Audio SRT Aligner.

On first plugin load the font is downloaded into the plugin's own
``fonts/`` directory so it travels with the repo.  Medium weight is
optimal for video subtitles — readable on small screens without
looking overly heavy.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error

FONT_FILENAME = "Alibaba-PuHuiTi-Medium.otf"
FONT_URL = (
    "https://cdn.jsdelivr.net/gh/liruifengv/alibaba-puhuiti@master/"
    "alibabaFont/zh-cn/Alibaba-PuHuiTi-Medium.otf"
)


def _get_bundled_fonts_dir() -> str:
    """Return the plugin's own ``fonts/`` directory (lives in the repo)."""
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(plugin_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    return fonts_dir


def download_font() -> bool:
    """Download Alibaba PuHuiTi Medium to the plugin's ``fonts/`` directory.

    Returns True if the font is already present or was downloaded.
    """
    fonts_dir = _get_bundled_fonts_dir()
    target_path = os.path.join(fonts_dir, FONT_FILENAME)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return True

    print(f"[AudioSrtAligner] Downloading {FONT_FILENAME}...", flush=True)
    try:
        proxy_handler = urllib.request.ProxyHandler({"http": "http://127.0.0.1:8118",
                                                      "https": "http://127.0.0.1:8118"})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(FONT_URL, target_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"[AudioSrtAligner] Font saved to {target_path} ({size_mb:.1f} MB)")
            return True
        print("[AudioSrtAligner] Font download failed: empty file.")
        return False
    except urllib.error.URLError as exc:
        print(f"[AudioSrtAligner] Font download failed (network): {exc}")
        return False
    except Exception as exc:
        print(f"[AudioSrtAligner] Font download failed: {exc}")
        return False


if __name__ == "__main__":
    success = download_font()
    sys.exit(0 if success else 1)
