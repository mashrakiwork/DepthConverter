# DepthConverter — local 2D → 3D VR converter

Convert regular images and videos into side-by-side (SBS) 3D content for VR
headsets. **Everything runs locally** — nothing is uploaded anywhere; the only
network traffic is the one-time download of the depth model you select from
HuggingFace (telemetry is disabled).

## How it works

Three tabs — use what you need, in order:

1. **Upscale** *(optional)* — AI-upscale images or a video 2x/3x/4x with
   Real-ESRGAN-family models (UltraSharp, AnimeSharp, or any HuggingFace-hosted
   `.pth` typed as `repo_id :: filename.pth`). Frames are processed in
   overlapping GPU tiles, so even 4K input fits in limited VRAM; the tile size
   halves itself on out-of-memory. Upscaling *before* depth + SBS gives the
   best 3D quality.
2. **Depth** — pick an input folder (images, or a single video). A monocular
   depth model (Depth Anything V2 etc.) runs on your GPU and produces a depth
   map per frame: a grayscale video for video input, 16-bit depth PNGs for
   images (optionally also a slideshow video + depth video at *N* seconds per
   image and a chosen frame rate).
3. **Converter (SBS)** — pick the original content + its depth content. Each
   frame is warped per-pixel on the GPU into a left/right eye pair and encoded
   as a full-SBS (2× width) or half-SBS video/image. Audio is copied from the
   original (optional). The per-eye aspect ratio is always exactly the source
   aspect ratio.

**▶ Run all** — one Start button for the whole chain. Tick the stages you want
(e.g. untick Upscale if your content is already high-res), pick one input and
one output folder, and each stage's output feeds the next automatically. Every
stage takes its settings live from its own tab.

## Quick start (new PC)

**Windows**: double-click **`DepthConverter.bat`**. It hands the app to
`pythonw.exe` and exits, so **no console window stays open** — only a brief
flash while the batch file itself runs, which is a Windows limitation of any
`.bat`. The first run shows a console on purpose, so you can watch the setup
downloads, and it creates **`DepthConverter.lnk`** — use or pin that shortcut
to launch with the app icon and no flash at all.
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

- **Models**: presets include Depth Anything **V3** (DA3 Small/Base/Large and
  Mono-Large — Mono-Large is tuned for monocular depth, ideal for 2D→3D),
  Depth Anything V2, and MiDaS/DPT. The dropdown is editable — paste any
  HuggingFace depth-estimation repo id. Models download once and are cached
  locally.
- **Depth convention**: *white = near, black = far*. Only use the invert
  checkbox if your depth output visibly shows the opposite (or 3D looks
  "inside out" in VR).
- **Input**: the Depth tab accepts a single video file, a single image file,
  or a folder of images.
- **Image order**: image folders are processed in natural numeric order
  (`1 → 2 → … → 11`, also `00/01` styles and `page_2` before `page_11`), so
  book/manga slideshows always come out in reading order.
- **3D strength (disparity)**: this is the *total* disparity budget between
  the eyes as % of width. 1–1.5 % is comfortable; more pops harder but risks
  eye strain and double vision. **Auto convergence** (default) tracks the
  subject depth so the main subject stays on the screen plane — keep it on
  unless you have a reason not to.
- **Black borders**: the Upscale and Converter tabs remove constant black bars
  (letterbox/pillarbox) by default, so bars are never upscaled and never eat
  VR resolution. Videos are sampled across their whole duration, so dark
  scenes or fade-ins don't cause overcropping. Untick the checkbox to keep
  the bars.
- Every job logs the completion time per output (HH:MM:SS).
- All folder/file choices and options are saved automatically
  (`~/.depthconverter/config.json`) and restored on the next launch.

## VR playback (important)

Output files are tagged so players auto-detect the 3D layout:

- Full SBS (2× width): `name_Full_SBS_LRF.mp4`
- Half SBS (same width): `name_Half_SBS_LR.mp4`

If a video looks **squeezed** (too wide / too short) or you see **double
images** in the headset, the player is interpreting a *full* SBS file as
*half* SBS. In Skybox open the format menu and select **Full SBS**
(or "3D → Side-by-side → full"). VLC on a monitor always shows the raw
double-wide frame — that is normal and correct for a full SBS file.

Note: Skybox **remembers the format you chose per file name**. If you
re-generate a video under a name you used before, it reapplies the old
(possibly wrong) setting — give new versions a fresh name, or delete the old
entry from Skybox's library/history first.

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
    upscale.py       AI upscaling (spandrel models, tiled VRAM-safe inference)
    pipeline.py      the jobs (upscale, depth, SBS) + the run-all chain
  ui/                PySide6 tabs (Upscale, Depth, Converter, Run all)
```
