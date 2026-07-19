# DepthConverter — local 2D → 3D VR converter

Convert regular images and videos into side-by-side (SBS) 3D content for VR
headsets. **Everything runs locally** — nothing is uploaded anywhere; the only
network traffic is the one-time download of the depth model you select from
HuggingFace (telemetry is disabled).

## How it works

Two steps, matching the two tabs in the app:

1. **Depth** — pick an input folder (images, or a single video). A monocular
   depth model (Depth Anything V2 etc.) runs on your GPU and produces a depth
   map per frame: a grayscale video for video input, 16-bit depth PNGs for
   images (optionally also a slideshow video + depth video at *N* seconds per
   image).
2. **Converter (SBS)** — pick the original content + its depth content. Each
   frame is warped per-pixel on the GPU into a left/right eye pair and encoded
   as a full-SBS (2× width) or half-SBS video/image. Audio is copied from the
   original. The per-eye aspect ratio is always exactly the source aspect ratio.

## Quick start (new PC)

**Windows**: double-click **`DepthConverter.bat`**.
**Linux / macOS**: run **`./depthconverter.sh`**.

On first run the launcher sets up everything automatically:
[uv](https://docs.astral.sh/uv/), a managed Python 3.12, all dependencies —
including CUDA-enabled PyTorch — and a **full FFmpeg build (x265 + NVENC)**
into `tools/ffmpeg`. Later runs start the app instantly, with no console
window on Windows.

You can also run the setup and app manually:

```powershell
.\setup.ps1            # or ./setup.sh on Linux/macOS
uv run depthconverter
```

### ffmpeg

Setup installs a full FFmpeg into `tools/ffmpeg` (Windows and Linux x86_64),
which the app prefers automatically — this enables **NVENC GPU encoding**
(`hevc_nvenc` / `h264_nvenc`) on NVIDIA cards. If it's missing, the app falls
back to an ffmpeg on PATH, then to the basic bundled `imageio-ffmpeg` binary.
The encoder dropdown only ever lists what your ffmpeg actually supports.

## Usage tips

- **Models**: presets include Depth Anything V2 Small/Base/Large and
  MiDaS/DPT. The dropdown is editable — paste any HuggingFace
  depth-estimation repo id (e.g. a Depth Anything V3 transformers port) and it
  will be downloaded and cached locally on first use.
- **Invert**: depth output convention is *white = near*. If near/far look
  swapped (in the depth output, or 3D looks "inside out" in VR), toggle the
  invert checkbox in the corresponding tab.
- **3D strength (divergence)**: 1.5–3 % is comfortable; higher pops more but
  can strain the eyes. **Convergence** sets which depth sits on the screen
  plane.
- **VR playback**: play the `*_SBS.mp4` in Skybox / DeoVR / Pigasus etc. and
  set the layout to *Side-by-Side, full* (or *half* for `_HSBS`).
- All folder/file choices and options are saved automatically
  (`~/.depthconverter/config.json`) and restored on the next launch.

## Performance notes

- Frames stream through ffmpeg pipes — RAM use is constant regardless of video
  length; nothing is ever fully loaded into memory.
- Inference runs in FP16 with a batch size chosen from your free VRAM, and
  automatically halves on out-of-memory instead of crashing.
- Depth videos use temporally smoothed normalization to avoid depth flicker.
- Encoding streams directly into x265/x264 (or NVENC when available).

## Project layout

```
app/
  main.py            entry point (DPI-aware Qt app)
  config.py          persistent settings (JSON)
  core/
    hardware.py      CUDA detection, VRAM-based batch sizing
    models.py        model registry + HF download/cache
    depth_engine.py  batched FP16 inference, OOM backoff, normalization
    video_io.py      streaming ffmpeg reader/writer, encoder detection
    sbs.py           GPU stereo warping (DIBR)
    pipeline.py      the two jobs (depth generation, SBS conversion)
  ui/                PySide6 tabs (Depth, Converter)
```
