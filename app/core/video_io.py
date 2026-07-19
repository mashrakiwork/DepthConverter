"""Streaming video I/O over ffmpeg pipes.

Frames are never accumulated in RAM: the reader yields one decoded frame at a
time from an ffmpeg rawvideo pipe, and the writer streams raw frames straight
into the encoder process.
"""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CREATIONFLAGS = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


def get_ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_str: str  # exact rate, possibly a fraction like "30000/1001"
    n_frames: int  # 0 if unknown
    duration: float
    has_audio: bool


def _fraction_to_float(rate: str) -> float:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str | Path) -> VideoInfo:
    ffprobe = _get_ffprobe_exe()
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
             "-show_format", str(path)],
            capture_output=True, text=True, creationflags=_CREATIONFLAGS,
        )
        info = json.loads(out.stdout or "{}")
        vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if vstream is None:
            raise ValueError(f"No video stream found in {path}")
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        rate = vstream.get("avg_frame_rate") or "0/0"
        if _fraction_to_float(rate) <= 0:
            rate = vstream.get("r_frame_rate") or "30"
        fps = _fraction_to_float(rate) or 30.0
        duration = float(vstream.get("duration") or info.get("format", {}).get("duration") or 0)
        nb = vstream.get("nb_frames") or ""
        n_frames = int(nb) if nb.isdigit() else int(round(duration * fps))
        return VideoInfo(int(vstream["width"]), int(vstream["height"]), fps, rate,
                         n_frames, duration, has_audio)

    # No ffprobe (e.g. only the bundled imageio-ffmpeg binary): parse `ffmpeg -i` output.
    proc = subprocess.run(
        [get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, creationflags=_CREATIONFLAGS,
    )
    err = proc.stderr
    m = re.search(r"Video:.*?\s(\d{2,5})x(\d{2,5})", err)
    if not m:
        raise ValueError(f"Could not read video info from {path}")
    width, height = int(m.group(1)), int(m.group(2))
    m = re.search(r"([\d.]+)\s*fps", err)
    fps = float(m.group(1)) if m else 30.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    duration = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0.0
    return VideoInfo(width, height, fps, f"{fps}", int(round(duration * fps)),
                     duration, "Audio:" in err)


class VideoReader:
    """Yields decoded frames as numpy arrays (HxWx3 rgb24 or HxW gray)."""

    def __init__(self, path: str | Path, pix_fmt: str = "rgb24"):
        self.info = probe_video(path)
        self.width, self.height = self.info.width, self.info.height
        self.channels = 3 if pix_fmt == "rgb24" else 1
        self._frame_bytes = self.width * self.height * self.channels
        self.proc = subprocess.Popen(
            [get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0",
             "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 4, creationflags=_CREATIONFLAGS,
        )

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def frames(self):
        try:
            while True:
                buf = self._read_exact(self._frame_bytes)
                if len(buf) < self._frame_bytes:
                    break
                arr = np.frombuffer(buf, np.uint8)
                if self.channels == 3:
                    yield arr.reshape(self.height, self.width, 3)
                else:
                    yield arr.reshape(self.height, self.width)
        finally:
            self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.wait()


class VideoWriter:
    """Streams raw frames into an ffmpeg encoder process.

    audio_from: optional source file whose audio track is copied into the output.
    in_rate / out_rate let images-as-video use a slow input rate (e.g. one image
    every 2 s) while still producing a normal-fps, player-friendly file.
    """

    def __init__(self, path: str | Path, width: int, height: int, in_rate: str,
                 codec_args: list[str], pix_fmt_in: str = "rgb24",
                 audio_from: str | Path | None = None, out_rate: str | None = None):
        self.width, self.height = width, height
        self.channels = 3 if pix_fmt_in == "rgb24" else 1
        cmd = [get_ffmpeg_exe(), "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", pix_fmt_in,
               "-s", f"{width}x{height}", "-framerate", in_rate, "-i", "-"]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "copy", "-shortest"]
        if width % 2 or height % 2:  # yuv420p requires even dimensions
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        if out_rate:
            cmd += ["-r", out_rate, "-fps_mode", "cfr"]
        cmd += codec_args + [str(path)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=width * height * self.channels * 4, creationflags=_CREATIONFLAGS,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(f"Frame size {frame.shape[1]}x{frame.shape[0]} != "
                             f"writer size {self.width}x{self.height}")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self._raise_encoder_error()

    def _raise_encoder_error(self):
        err = b""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.stderr:
            err = self.proc.stderr.read()
        self.proc.wait()
        raise RuntimeError(f"ffmpeg encoder failed: {err.decode(errors='replace').strip()}")

    def close(self, abort: bool = False) -> None:
        if abort:
            self.proc.kill()
            self.proc.wait()
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        err = self.proc.stderr.read() if self.proc.stderr else b""
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}: "
                               f"{err.decode(errors='replace').strip()}")


_ENCODER_LABELS = {
    "libx265": "H.265 / HEVC (CPU, libx265)",
    "hevc_nvenc": "H.265 / HEVC (NVIDIA NVENC)",
    "libx264": "H.264 (CPU, libx264)",
    "h264_nvenc": "H.264 (NVIDIA NVENC)",
}

_encoders_cache: list[tuple[str, str]] | None = None


def available_encoders() -> list[tuple[str, str]]:
    """Return [(encoder_key, label), ...] actually supported by the local ffmpeg."""
    global _encoders_cache
    if _encoders_cache is not None:
        return _encoders_cache
    try:
        out = subprocess.run(
            [get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, creationflags=_CREATIONFLAGS,
        ).stdout
    except OSError:
        out = ""
    found = [(k, v) for k, v in _ENCODER_LABELS.items()
             if re.search(rf"^\s*V[^\s]*\s+{k}\s", out, re.MULTILINE)]
    _encoders_cache = found or [("libx264", _ENCODER_LABELS["libx264"])]
    return _encoders_cache


# preset knob -> (x264/x265 preset, nvenc preset)
_PRESETS = {"quality": ("slow", "p7"), "balanced": ("medium", "p5"), "fast": ("fast", "p3")}


def encoder_args(encoder: str, quality: int, preset: str = "balanced") -> list[str]:
    """Build ffmpeg output args. quality = CRF/CQ (lower is better)."""
    cpu_preset, nv_preset = _PRESETS.get(preset, _PRESETS["balanced"])
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", cpu_preset, "-crf", str(quality),
                "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]
    if encoder == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", nv_preset, "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", cpu_preset, "-crf", str(quality),
                "-pix_fmt", "yuv420p"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", nv_preset, "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unknown encoder: {encoder}")
