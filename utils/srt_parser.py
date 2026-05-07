"""
SRT subtitle format parser for ComfyUI nodes.
"""
import re
from typing import List
from dataclasses import dataclass


@dataclass
class SrtEntry:
    """A single SRT subtitle entry."""
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


TIME_PATTERN = re.compile(
    r'(\d{1,2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2}),(\d{3})'
)


def _parse_time_match(match: re.Match) -> tuple:
    """Convert time regex match to (start_seconds, end_seconds)."""
    h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
    start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
    end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
    return start, end


def parse_srt(srt_string: str) -> List[SrtEntry]:
    """
    Parse an SRT format string into a list of SrtEntry objects.
    
    Handles:
    - Standard SRT time format: HH:MM:SS,mmm --> HH:MM:SS,mmm
    - Empty lines between entries
    - Multi-line text (joined with space)
    - BOM markers
    - Various line endings (\\n, \\r\\n)
    
    Raises ValueError if format is invalid, with line number info.
    """
    # 1. Strip BOM if present
    if srt_string.startswith('\ufeff'):
        srt_string = srt_string[1:]
    
    # 2. Normalize line endings
    srt_string = srt_string.replace('\r\n', '\n').replace('\r', '\n')
    
    # 3. Split by double newlines to get blocks
    blocks = srt_string.strip().split('\n\n')
    
    entries = []
    line_num = 1
    
    for block in blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        
        if len(lines) < 2:
            raise ValueError(f"Invalid SRT block at line ~{line_num}: missing time line")
        
        # Parse index
        try:
            index = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"Invalid SRT index at line {line_num}: '{lines[0]}'")
        
        # Parse time line
        time_match = TIME_PATTERN.search(lines[1])
        if not time_match:
            raise ValueError(f"Invalid SRT time format at line {line_num + 1}: '{lines[1]}'")
        
        start, end = _parse_time_match(time_match)
        
        # Parse text (join multi-line with space)
        if len(lines) < 3:
            raise ValueError(f"Invalid SRT block at line ~{line_num}: missing text")
        
        text = ' '.join(line.strip() for line in lines[2:])
        
        entries.append(SrtEntry(index=index, start=start, end=end, text=text))
        line_num += len(lines) + 1  # +1 for the separator newline
    
    return entries


def format_srt(entries: List[SrtEntry]) -> str:
    """Serialize a list of SrtEntry objects back to standard SRT string."""
    if not entries:
        return ""
    lines: List[str] = []
    for entry in entries:
        h = int(entry.start // 3600)
        m = int((entry.start % 3600) // 60)
        s = int(entry.start % 60)
        ms = int(round((entry.start % 1) * 1000))
        eh = int(entry.end // 3600)
        em = int((entry.end % 3600) // 60)
        es = int(entry.end % 60)
        ems = int(round((entry.end % 1) * 1000))
        lines.append(f"{entry.index}")
        lines.append(f"{h:02}:{m:02}:{s:02},{ms:03} --> {eh:02}:{em:02}:{es:02},{ems:03}")
        lines.append(f"{entry.text}")
        lines.append("")
    return "\n".join(lines)