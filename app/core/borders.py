"""Black-border (letterbox / pillarbox) detection and removal.

Videos are sampled at several points across their whole duration and the
content bounding boxes are unioned, so dark scenes, fades, or black intro
frames never cause overcropping. If no borders exist, detection returns None
and nothing is cropped.
"""

import subprocess
from pathlib import Path

import numpy as np

from .video_io import _CREATIONFLAGS, get_ffmpeg_exe, probe_video

_THRESH = 24     # a pixel counts as content above this (compression noise floor)
_MIN_BORDER = 4  # ignore borders thinner than this many pixels
_SAMPLES = 12


def _content_bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of non-black content in one frame; None if all black."""
    gray = arr.max(axis=2) if arr.ndim == 3 else arr
    rows = np.flatnonzero((gray > _THRESH).any(axis=1))
    cols = np.flatnonzero((gray > _THRESH).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _finalize(bbox, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Ignore borders too thin to matter, keep the size even (codec-friendly);
    None when there is nothing worth cropping."""
    x0, y0, x1, y1 = bbox
    if x0 < _MIN_BORDER:
        x0 = 0
    if y0 < _MIN_BORDER:
        y0 = 0
    if w - x1 < _MIN_BORDER:
        x1 = w
    if h - y1 < _MIN_BORDER:
        y1 = h
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    if (x0, y0, x1, y1) == (0, 0, w, h) or x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def image_borders(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Crop box for one image, or None if it has no black borders."""
    h, w = arr.shape[:2]
    bbox = _content_bbox(arr)
    return _finalize(bbox, w, h) if bbox else None


def video_borders(path: str | Path) -> tuple[int, int, int, int] | None:
    """Crop box for a video, or None if it has no constant black borders."""
    info = probe_video(path)
    w, h = info.width, info.height
    union = None
    for i in range(_SAMPLES):
        t = info.duration * (i + 0.5) / _SAMPLES if info.duration > 0 else 0.0
        out = subprocess.run(
            [get_ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, creationflags=_CREATIONFLAGS).stdout
        if len(out) >= w * h:
            bbox = _content_bbox(np.frombuffer(out[:w * h], np.uint8).reshape(h, w))
            if bbox is not None:
                union = bbox if union is None else (
                    min(union[0], bbox[0]), min(union[1], bbox[1]),
                    max(union[2], bbox[2]), max(union[3], bbox[3]))
                if union == (0, 0, w, h):
                    break  # full-frame content somewhere - nothing to crop
        if info.duration <= 0:
            break  # unknown duration: only the first frame is reachable
    return _finalize(union, w, h) if union else None


def scale_box(box, src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Map a crop box onto content of a different resolution (e.g. a depth
    video that does not exactly match the original's dimensions)."""
    if (src_w, src_h) == (dst_w, dst_h):
        return box
    sx, sy = dst_w / src_w, dst_h / src_h
    x0, y0 = int(round(box[0] * sx)), int(round(box[1] * sy))
    x1, y1 = int(round(box[2] * sx)), int(round(box[3] * sy))
    return max(x0, 0), max(y0, 0), min(x1, dst_w), min(y1, dst_h)


def crop(arr: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]
