"""Job orchestration for the three windows.

run_upscale_job:  input (folder, single video, or single image) -> upscaled output
run_depth_job:    input (folder, single video, or single image) -> depth output
run_sbs_job:      original + depth content -> side-by-side 3D output
run_pipeline_job: chains any selected stages of the above; each stage's output
                  feeds the next stage's input automatically

All jobs stream frame-by-frame (constant RAM), batch to the GPU, and report
through progress(fraction, message) / log(message) callbacks. cancel is a
threading.Event; JobCancelled is raised when it fires.

Output naming uses the tags VR players (Skybox, DeoVR, Pigasus...) auto-detect:
full SBS -> `_Full_SBS_LRF`, half SBS -> `_Half_SBS_LR`. This matters: a full
SBS file detected as *half* SBS looks horizontally stretched / vertically
squeezed in the headset and its doubled disparity causes double vision.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .video_io import VideoReader, VideoWriter, encoder_args, probe_video

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".ts"}

SBS_WRITE_BATCH = 2  # supersampled warping doubles per-frame VRAM; keep bounded


class JobCancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise JobCancelled()


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def scan_input_folder(folder: Path) -> tuple[str, list[Path]]:
    """Classify an input folder as ('video', [file]) or ('images', files)."""
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VID_EXTS)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS
                    and not p.stem.endswith("_depth"))
    if videos and images:
        raise ValueError("Input folder contains both videos and images - use one kind "
                         "only, or pick a single file instead.")
    if len(videos) > 1:
        raise ValueError(f"Input folder contains {len(videos)} videos - pick a single "
                         f"video file instead.")
    if videos:
        return "video", videos
    if images:
        return "images", images
    raise ValueError("No supported video or image files found in the input folder.")


def classify_input(path: Path) -> tuple[str, list[Path]]:
    """Accept a folder, a single video file, or a single image file."""
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VID_EXTS:
            return "video", [path]
        if suffix in IMG_EXTS:
            return "images", [path]
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.is_dir():
        return scan_input_folder(path)
    raise ValueError("Input path does not exist.")


def _load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGB"))


def _load_depth_gray(path: Path) -> np.ndarray:
    """Load a depth image as uint8 or uint16 grayscale."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I"):
        return np.asarray(img, dtype=np.uint16)
    return np.asarray(img.convert("L"))


def _letterbox(arr: np.ndarray, tw: int, th: int, resample=Image.LANCZOS) -> np.ndarray:
    """Fit into (tw, th) preserving aspect ratio, centered on black."""
    img = Image.fromarray(arr)
    scale = min(tw / img.width, th / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), resample)
    canvas = Image.new(img.mode, (tw, th), 0)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return np.asarray(canvas)


# --------------------------------------------------------------------------
# Upscale job
# --------------------------------------------------------------------------

def run_upscale_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    from .hardware import resolve_device
    from .upscale import Upscaler

    t0 = time.monotonic()
    in_path = Path(opts["input_path"])
    out_dir = Path(opts["output_dir"])
    scale = int(opts["scale"])
    kind, files = classify_input(in_path)
    base_name = in_path.stem if in_path.is_file() else (in_path.name or "images")
    if kind == "images" and len(files) > 1:
        out_dir = out_dir / f"{base_name}_x{scale}"
        log(f"Batch of {len(files)} images -> subfolder {out_dir.name}\\")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(opts.get("device", "auto"))
    log(f"Input: {kind} ({len(files)} file(s)). Device: {device.upper()}. "
        f"Target scale: {scale}x.")
    up = Upscaler(opts["repo_id"], opts["filename"], device=device,
                  fp16=opts.get("fp16", True), log=log)

    deborder = bool(opts.get("remove_borders", True))
    if kind == "video":
        codec = encoder_args(opts["encoder"], opts["quality"],
                             opts.get("preset", "balanced"))
        result = _upscale_video(files[0], out_dir, up, scale, codec, deborder,
                                progress, log, cancel)
    else:
        result = _upscale_images(files, out_dir, up, scale, deborder,
                                 progress, log, cancel)
    log(f"Upscale job finished. Total time: {_fmt_hms(time.monotonic() - t0)}")
    return result


def _upscale_video(src: Path, out_dir: Path, up, scale: int, codec, deborder,
                   progress, log, cancel):
    from .borders import crop, video_borders

    t0 = time.monotonic()
    info = probe_video(src)
    box = video_borders(src) if deborder else None
    cw, ch = ((box[2] - box[0], box[3] - box[1]) if box
              else (info.width, info.height))
    if box:
        log(f"Black borders removed: {info.width}x{info.height} -> {cw}x{ch}")
    out_w, out_h = cw * scale, ch * scale
    out_path = out_dir / f"{src.stem}_x{scale}.mp4"
    log(f"Video: {info.width}x{info.height} @ {info.fps:.3f} fps -> "
        f"{out_w}x{out_h} -> {out_path.name}"
        + (" [audio copied]" if info.has_audio else ""))
    reader = VideoReader(src, "rgb24")
    writer = VideoWriter(out_path, out_w, out_h, info.fps_str, codec,
                         pix_fmt_in="rgb24", audio_from=src)
    total = max(info.n_frames, 1)
    done = 0
    try:
        for frame in reader.frames():
            _check(cancel)
            if box:
                frame = crop(frame, box)
            writer.write(up.upscale(frame, scale))
            done += 1
            progress(min(done / total, 1.0), f"frame {done}/{info.n_frames or '?'}")
        writer.close()
    except BaseException:
        reader.close()
        writer.close(abort=True)
        raise
    log(f"Output {out_path.name} completed in {_fmt_hms(time.monotonic() - t0)}")
    return out_path


def _upscale_images(files, out_dir: Path, up, scale: int, deborder,
                    progress, log, cancel):
    from .borders import crop, image_borders

    t0 = time.monotonic()
    for i, path in enumerate(files):
        _check(cancel)
        rgb = _load_image_rgb(path)
        if deborder:
            box = image_borders(rgb)
            if box:
                log(f"{path.name}: black borders removed "
                    f"({rgb.shape[1]}x{rgb.shape[0]} -> "
                    f"{box[2] - box[0]}x{box[3] - box[1]})")
                rgb = crop(rgb, box)
        out = up.upscale(rgb, scale)
        Image.fromarray(out).save(out_dir / f"{path.stem}_x{scale}.png")
        progress((i + 1) / len(files), f"image {i + 1}/{len(files)}: {path.name}")
    log(f"Wrote {len(files)} upscaled PNGs to {out_dir} "
        f"in {_fmt_hms(time.monotonic() - t0)}")
    return out_dir if len(files) > 1 else out_dir / f"{files[0].stem}_x{scale}.png"


# --------------------------------------------------------------------------
# Depth job
# --------------------------------------------------------------------------

def run_depth_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    from .depth_engine import DepthEstimator
    from .hardware import resolve_device

    t0 = time.monotonic()
    in_path = Path(opts["input_path"])
    out_dir = Path(opts["output_dir"])
    kind, files = classify_input(in_path)
    base_name = in_path.stem if in_path.is_file() else (in_path.name or "images")
    # Multi-file jobs get their own indicative subfolder under the output dir.
    if kind == "images" and len(files) > 1:
        out_dir = out_dir / f"{base_name}_depth"
        log(f"Batch of {len(files)} images -> subfolder {out_dir.name}\\")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(opts.get("device", "auto"))
    log(f"Input: {kind} ({len(files)} file(s)). Device: {device.upper()}.")
    est = DepthEstimator(opts["model_id"], device=device, fp16=opts.get("fp16", True),
                         invert=opts.get("invert", False), log=log)
    codec = encoder_args(opts["encoder"], opts["quality"], opts.get("preset", "balanced"))

    if kind == "video":
        result = (files[0], _depth_video(files[0], out_dir, est, codec,
                                         progress, log, cancel))
    else:
        orig, depth = _depth_images(files, base_name, out_dir, est, codec, opts,
                                    progress, log, cancel)
        result = (orig if orig is not None else in_path, depth)
    log(f"Depth job finished. Total time: {_fmt_hms(time.monotonic() - t0)}")
    # (matching original content, depth content) - what the SBS stage needs.
    return result


def _depth_video(src: Path, out_dir: Path, est, codec, progress, log, cancel):
    t0 = time.monotonic()
    info = probe_video(src)
    out_path = out_dir / f"{src.stem}_depth.mp4"
    log(f"Video: {info.width}x{info.height} @ {info.fps:.3f} fps, "
        f"~{info.n_frames or '?'} frames -> {out_path.name}")
    reader = VideoReader(src, "rgb24")
    writer = VideoWriter(out_path, info.width, info.height, info.fps_str, codec,
                         pix_fmt_in="gray")
    est.reset_temporal()
    total = max(info.n_frames, 1)
    done = 0
    batch: list[np.ndarray] = []

    def flush():
        nonlocal done
        if not batch:
            return
        preds = est.infer(batch)
        for gray in est.normalize(preds, temporal=True):
            writer.write(gray)
        done += len(batch)
        batch.clear()
        progress(min(done / total, 1.0), f"frame {done}/{info.n_frames or '?'}")

    try:
        for frame in reader.frames():
            _check(cancel)
            batch.append(frame)
            if len(batch) >= est.batch_size:
                flush()
        flush()
        writer.close()
    except BaseException:
        reader.close()
        writer.close(abort=True)
        raise
    log(f"Output {out_path.name} completed in {_fmt_hms(time.monotonic() - t0)}")
    return out_path


def _depth_images(files, base_name: str, out_dir: Path, est, codec, opts,
                  progress, log, cancel):
    t0 = time.monotonic()
    make_video = opts.get("images_to_video", False) and len(files) > 0
    spf = float(opts.get("seconds_per_image", 2.0))
    depth_frames_8bit: list[np.ndarray] = []  # only kept when building a video
    rgb_paths: list[Path] = []

    for i, path in enumerate(files):
        _check(cancel)
        rgb = _load_image_rgb(path)
        pred = est.infer([rgb])
        depth16 = est.normalize16(pred)[0]
        out_path = out_dir / f"{path.stem}_depth.png"
        Image.fromarray(depth16, mode="I;16").save(out_path)
        rgb_paths.append(path)
        if make_video:
            depth_frames_8bit.append((depth16 // 257).astype(np.uint8))
        progress((i + 1) / len(files) * (0.7 if make_video else 1.0),
                 f"image {i + 1}/{len(files)}: {path.name}")
    log(f"Wrote {len(files)} 16-bit depth PNGs to {out_dir} "
        f"in {_fmt_hms(time.monotonic() - t0)}")

    if not make_video:
        # Multi-image: caller substitutes the input folder as the original.
        if len(files) > 1:
            return None, out_dir
        return files[0], out_dir / f"{files[0].stem}_depth.png"

    # Build original + depth videos with every image letterboxed (aspect kept)
    # onto a common even-sized canvas.
    _check(cancel)
    t1 = time.monotonic()
    sizes = [Image.open(p).size for p in rgb_paths]
    tw = max(w for w, _ in sizes)
    th = max(h for _, h in sizes)
    tw += tw % 2
    th += th % 2
    in_rate = f"{1.0 / spf:.6f}"
    # Modest CFR output: slideshow frames are static duplicates, so a high fps
    # only bloats the frame count and slows the later SBS conversion. Clamped
    # so every image still gets at least one frame.
    fps = int(opts.get("slideshow_fps", 10))
    out_fps = str(max(fps, int(np.ceil(1.0 / spf))))
    orig_out = out_dir / f"{base_name}.mp4"
    depth_out = out_dir / f"{base_name}_depth.mp4"
    log(f"Building videos at {spf:g}s per image ({out_fps} fps), canvas {tw}x{th}: "
        f"{orig_out.name} + {depth_out.name}")

    w1 = VideoWriter(orig_out, tw, th, in_rate, codec, pix_fmt_in="rgb24", out_rate=out_fps)
    w2 = VideoWriter(depth_out, tw, th, in_rate, codec, pix_fmt_in="gray", out_rate=out_fps)
    try:
        for i, (path, dframe) in enumerate(zip(rgb_paths, depth_frames_8bit)):
            _check(cancel)
            w1.write(_letterbox(_load_image_rgb(path), tw, th))
            w2.write(_letterbox(dframe, tw, th, resample=Image.BILINEAR))
            progress(0.7 + 0.3 * (i + 1) / len(rgb_paths),
                     f"video frame {i + 1}/{len(rgb_paths)}")
        w1.close()
        w2.close()
    except BaseException:
        w1.close(abort=True)
        w2.close(abort=True)
        raise
    log(f"Outputs {orig_out.name} + {depth_out.name} completed in "
        f"{_fmt_hms(time.monotonic() - t1)}")
    return orig_out, depth_out


# --------------------------------------------------------------------------
# SBS job
# --------------------------------------------------------------------------

def _sbs_tag(half: bool) -> str:
    # Tags VR players auto-detect: LRF/Full_SBS = full side-by-side.
    return "Half_SBS_LR" if half else "Full_SBS_LRF"


def run_sbs_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    from .hardware import resolve_device

    t0 = time.monotonic()
    orig = Path(opts["original"])
    dep = Path(opts["depth"])
    out_dir = Path(opts["output_dir"])
    if not orig.exists() or not dep.exists():
        raise ValueError("Original or depth path does not exist.")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(opts.get("device", "auto"))
    auto_conv = bool(opts.get("auto_convergence", True))
    conv_desc = "AUTO (subject-tracked)" if auto_conv else f"{opts['convergence']:g}"
    aa = int(opts.get("aa_supersample", 2))
    log(f"SBS conversion on {device.upper()}: total disparity {opts['divergence']:g}% "
        f"of width, convergence {conv_desc}, "
        f"{'half' if opts.get('half_sbs') else 'full'} SBS, "
        f"anti-aliasing {'off' if aa <= 1 else f'{aa}x'}")

    if orig.is_file() and dep.is_file():
        if orig.suffix.lower() in IMG_EXTS:
            _sbs_image_pair(orig, dep, out_dir, device, opts, progress, log, cancel)
        else:
            _sbs_video(orig, dep, out_dir, device, opts, progress, log, cancel)
    elif orig.is_dir() and dep.is_dir():
        _sbs_images(orig, dep, out_dir, device, opts, progress, log, cancel)
    else:
        raise ValueError("Original and depth must both be video files, both be "
                         "image files, or both be image folders.")
    log(f"SBS job finished. Total time: {_fmt_hms(time.monotonic() - t0)}")


def _sbs_video(orig: Path, dep: Path, out_dir: Path, device, opts,
               progress, log, cancel):
    import torch

    from .borders import crop, scale_box, video_borders
    from .sbs import ConvergenceTracker, make_sbs

    t0 = time.monotonic()
    half = bool(opts.get("half_sbs"))
    invert = bool(opts.get("invert_depth"))
    auto_conv = bool(opts.get("auto_convergence", True))
    smooth = bool(opts.get("smooth_depth", True))
    supersample = int(opts.get("aa_supersample", 2))
    codec = encoder_args(opts["encoder"], opts["quality"], opts.get("preset", "balanced"))

    r_orig = VideoReader(orig, "rgb24")
    r_dep = VideoReader(dep, "gray")
    info = r_orig.info
    if abs(info.n_frames - r_dep.info.n_frames) > 2 and info.n_frames and r_dep.info.n_frames:
        log(f"WARNING: frame counts differ (original ~{info.n_frames}, "
            f"depth ~{r_dep.info.n_frames}); output stops at the shorter one.")

    box = video_borders(orig) if opts.get("remove_borders", True) else None
    dep_box = None
    cw, ch = info.width, info.height
    if box:
        cw, ch = box[2] - box[0], box[3] - box[1]
        dep_box = scale_box(box, info.width, info.height,
                            r_dep.info.width, r_dep.info.height)
        log(f"Black borders removed: {info.width}x{info.height} -> {cw}x{ch}")

    keep_audio = bool(opts.get("keep_audio", True)) and info.has_audio
    out_w = cw if half else cw * 2
    out_path = out_dir / f"{orig.stem}_{_sbs_tag(half)}.mp4"
    log(f"{info.width}x{info.height} -> {out_w}x{ch} -> {out_path.name}"
        + (" [audio copied]" if keep_audio else ""))
    writer = VideoWriter(out_path, out_w, ch, info.fps_str, codec,
                         pix_fmt_in="rgb24", audio_from=orig if keep_audio else None)

    tracker = ConvergenceTracker()
    total = max(info.n_frames, 1)
    done = 0
    imgs: list[np.ndarray] = []
    deps: list[np.ndarray] = []

    def flush():
        nonlocal done, supersample
        if not imgs:
            return
        img_t = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        dep_t = torch.from_numpy(np.stack(deps)).to(device).float().div_(255.0)
        if invert:
            dep_t = 1.0 - dep_t
        if auto_conv:
            convergences = [tracker.update(dep_t[i]) for i in range(dep_t.shape[0])]
        while True:
            try:
                if auto_conv:
                    sbs = make_sbs(img_t, dep_t, opts["divergence"], 0.5, half,
                                   smooth_depth=smooth, convergences=convergences,
                                   supersample=supersample)
                else:
                    sbs = make_sbs(img_t, dep_t, opts["divergence"],
                                   opts["convergence"], half, smooth_depth=smooth,
                                   supersample=supersample)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if supersample <= 1:
                    raise
                supersample -= 1
                log(f"Low VRAM: reducing anti-aliasing to "
                    f"{'off' if supersample <= 1 else f'{supersample}x'}")
        sbs_u8 = (sbs.clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for frame in sbs_u8:
            writer.write(frame)
        done += len(imgs)
        imgs.clear()
        deps.clear()
        progress(min(done / total, 1.0), f"frame {done}/{info.n_frames or '?'}")

    try:
        for frame, dframe in zip(r_orig.frames(), r_dep.frames()):
            _check(cancel)
            if box:
                frame = crop(frame, box)
                dframe = crop(dframe, dep_box)
            imgs.append(frame)
            deps.append(dframe)
            if len(imgs) >= SBS_WRITE_BATCH:
                flush()
        flush()
        writer.close()
    except BaseException:
        r_orig.close()
        r_dep.close()
        writer.close(abort=True)
        raise
    log(f"Output {out_path.name} completed in {_fmt_hms(time.monotonic() - t0)}")


def _deborder_pair(rgb, depth, log, name: str):
    """Crop the same black borders (if any) off an image + its depth map."""
    from .borders import crop, image_borders, scale_box

    box = image_borders(rgb)
    if box is None:
        return rgb, depth
    log(f"{name}: black borders removed ({rgb.shape[1]}x{rgb.shape[0]} -> "
        f"{box[2] - box[0]}x{box[3] - box[1]})")
    dep_box = scale_box(box, rgb.shape[1], rgb.shape[0],
                        depth.shape[1], depth.shape[0])
    return crop(rgb, box), crop(depth, dep_box)


def _sbs_image_pair(orig: Path, dep: Path, out_dir: Path, device, opts,
                    progress, log, cancel):
    """One original image + one depth image -> one SBS image."""
    from PIL import Image as PILImage

    from .sbs import sbs_frame_uint8

    t0 = time.monotonic()
    _check(cancel)
    half = bool(opts.get("half_sbs"))
    auto_conv = bool(opts.get("auto_convergence", True))
    rgb, depth = _load_image_rgb(orig), _load_depth_gray(dep)
    if opts.get("remove_borders", True):
        rgb, depth = _deborder_pair(rgb, depth, log, orig.name)
    out = sbs_frame_uint8(rgb, depth, device,
                          opts["divergence"],
                          None if auto_conv else opts["convergence"], half,
                          invert_depth=bool(opts.get("invert_depth")),
                          smooth_depth=bool(opts.get("smooth_depth", True)),
                          supersample=int(opts.get("aa_supersample", 2)))
    out_path = out_dir / f"{orig.stem}_{_sbs_tag(half)}.png"
    PILImage.fromarray(out).save(out_path)
    progress(1.0, orig.name)
    log(f"Output {out_path.name} completed in {_fmt_hms(time.monotonic() - t0)}")


def _find_depth_match(orig_stem: str, dep_dir: Path) -> Path | None:
    for suffix in ("_depth", ""):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"):
            p = dep_dir / f"{orig_stem}{suffix}{ext}"
            if p.exists():
                return p
    return None


def _sbs_images(orig_dir: Path, dep_dir: Path, out_dir: Path, device, opts,
                progress, log, cancel):
    from PIL import Image as PILImage

    from .sbs import sbs_frame_uint8

    t0 = time.monotonic()
    files = sorted(p for p in orig_dir.iterdir() if p.suffix.lower() in IMG_EXTS
                   and not p.stem.endswith(("_depth", "_SBS", "_HSBS"))
                   and "_SBS_" not in p.stem)
    if not files:
        raise ValueError("No images found in the original folder.")
    if len(files) > 1:
        out_dir = out_dir / f"{orig_dir.name}_3D"
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"Batch of {len(files)} images -> subfolder {out_dir.name}\\")
    half = bool(opts.get("half_sbs"))
    auto_conv = bool(opts.get("auto_convergence", True))
    tag = _sbs_tag(half)
    skipped = 0
    for i, path in enumerate(files):
        _check(cancel)
        dpath = _find_depth_match(path.stem, dep_dir)
        if dpath is None:
            log(f"WARNING: no depth match for {path.name}, skipped.")
            skipped += 1
            continue
        rgb = _load_image_rgb(path)
        depth = _load_depth_gray(dpath)
        if opts.get("remove_borders", True):
            rgb, depth = _deborder_pair(rgb, depth, log, path.name)
        out = sbs_frame_uint8(rgb, depth, device, opts["divergence"],
                              None if auto_conv else opts["convergence"], half,
                              invert_depth=bool(opts.get("invert_depth")),
                              smooth_depth=bool(opts.get("smooth_depth", True)),
                              supersample=int(opts.get("aa_supersample", 2)))
        PILImage.fromarray(out).save(out_dir / f"{path.stem}_{tag}.png")
        progress((i + 1) / len(files), f"image {i + 1}/{len(files)}: {path.name}")
    log(f"Wrote {len(files) - skipped} SBS images to {out_dir} "
        f"in {_fmt_hms(time.monotonic() - t0)}"
        + (f" ({skipped} skipped)" if skipped else ""))


# --------------------------------------------------------------------------
# Run-all pipeline: chain the selected stages
# --------------------------------------------------------------------------

def run_pipeline_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    """opts: input_path, output_dir, do_upscale/do_depth/do_sbs flags, and one
    sub-dict per selected stage ('upscale', 'depth', 'sbs') holding that
    stage's settings (collected from its tab)."""
    t0 = time.monotonic()
    stages = [name for name, key in
              (("Upscale", "upscale"), ("Depth", "depth"), ("Convert", "sbs"))
              if opts.get(f"do_{key}")]
    if not stages:
        raise ValueError("No stages selected.")
    n = len(stages)
    out_dir = str(opts["output_dir"])
    log(f"Pipeline: {' -> '.join(stages)}")

    def stage_progress(i, name):
        return lambda p, m: progress((i + p) / n, f"[{i + 1}/{n} {name}] {m}")

    i = 0
    current = Path(opts["input_path"])  # content flowing through the stages
    depth_out = None
    if opts.get("do_upscale"):
        log(f"=== Stage {i + 1}/{n}: Upscale ===")
        current = run_upscale_job(
            {**opts["upscale"], "input_path": str(current), "output_dir": out_dir},
            stage_progress(i, "Upscale"), log, cancel)
        i += 1
    orig_for_sbs = current
    if opts.get("do_depth"):
        log(f"=== Stage {i + 1}/{n}: Depth ===")
        orig_for_sbs, depth_out = run_depth_job(
            {**opts["depth"], "input_path": str(current), "output_dir": out_dir},
            stage_progress(i, "Depth"), log, cancel)
        i += 1
    if opts.get("do_sbs"):
        log(f"=== Stage {i + 1}/{n}: Convert (SBS) ===")
        run_sbs_job(
            {**opts["sbs"], "original": str(orig_for_sbs),
             "depth": str(depth_out), "output_dir": out_dir},
            stage_progress(i, "Convert"), log, cancel)
    log(f"Pipeline finished ({n} stage(s)). "
        f"Total time: {_fmt_hms(time.monotonic() - t0)}")
