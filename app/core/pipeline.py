"""Job orchestration for the two windows.

run_depth_job:  input folder (images or one video) -> depth maps / depth video
run_sbs_job:    original + depth content -> side-by-side 3D output

All jobs stream frame-by-frame (constant RAM), batch to the GPU, and report
through progress(fraction, message) / log(message) callbacks. cancel is a
threading.Event; JobCancelled is raised when it fires.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .video_io import VideoReader, VideoWriter, encoder_args, probe_video

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".ts"}

SBS_WRITE_BATCH = 4


class JobCancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise JobCancelled()


def scan_input_folder(folder: Path) -> tuple[str, list[Path]]:
    """Classify an input folder as ('video', [file]) or ('images', files)."""
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VID_EXTS)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS
                    and not p.stem.endswith("_depth"))
    if videos and images:
        raise ValueError("Input folder contains both videos and images - use one kind only.")
    if len(videos) > 1:
        raise ValueError(f"Input folder contains {len(videos)} videos - only 1 is supported.")
    if videos:
        return "video", videos
    if images:
        return "images", images
    raise ValueError("No supported video or image files found in the input folder.")


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
# Depth job
# --------------------------------------------------------------------------

def run_depth_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    from .depth_engine import DepthEstimator
    from .hardware import resolve_device

    in_dir = Path(opts["input_dir"])
    out_dir = Path(opts["output_dir"])
    if not in_dir.is_dir():
        raise ValueError("Input folder does not exist.")
    out_dir.mkdir(parents=True, exist_ok=True)

    kind, files = scan_input_folder(in_dir)
    device = resolve_device(opts.get("device", "auto"))
    log(f"Input: {kind} ({len(files)} file(s)). Device: {device.upper()}.")
    est = DepthEstimator(opts["model_id"], device=device, fp16=opts.get("fp16", True),
                         invert=opts.get("invert", False), log=log)
    codec = encoder_args(opts["encoder"], opts["quality"], opts.get("preset", "balanced"))

    if kind == "video":
        _depth_video(files[0], out_dir, est, codec, progress, log, cancel)
    else:
        _depth_images(files, in_dir, out_dir, est, codec, opts, progress, log, cancel)
    log("Depth job finished.")


def _depth_video(src: Path, out_dir: Path, est, codec, progress, log, cancel):
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


def _depth_images(files, in_dir: Path, out_dir: Path, est, codec, opts,
                  progress, log, cancel):
    make_video = opts.get("images_to_video", False)
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
    log(f"Wrote {len(files)} 16-bit depth PNGs to {out_dir}")

    if not make_video:
        return

    # Build original + depth videos with every image letterboxed (aspect kept)
    # onto a common even-sized canvas.
    _check(cancel)
    sizes = [Image.open(p).size for p in rgb_paths]
    tw = max(w for w, _ in sizes)
    th = max(h for _, h in sizes)
    tw += tw % 2
    th += th % 2
    in_rate = f"{1.0 / spf:.6f}"
    out_fps = "30"
    name = in_dir.name or "images"
    orig_out = out_dir / f"{name}.mp4"
    depth_out = out_dir / f"{name}_depth.mp4"
    log(f"Building videos at {spf:g}s per image, canvas {tw}x{th}: "
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


# --------------------------------------------------------------------------
# SBS job
# --------------------------------------------------------------------------

def run_sbs_job(opts: dict, progress=lambda p, m: None, log=print, cancel=None):
    from .hardware import resolve_device

    orig = Path(opts["original"])
    dep = Path(opts["depth"])
    out_dir = Path(opts["output_dir"])
    if not orig.exists() or not dep.exists():
        raise ValueError("Original or depth path does not exist.")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(opts.get("device", "auto"))
    log(f"SBS conversion on {device.upper()}: divergence {opts['divergence']:g}%, "
        f"convergence {opts['convergence']:g}, "
        f"{'half' if opts.get('half_sbs') else 'full'} SBS")

    if orig.is_file() and dep.is_file():
        _sbs_video(orig, dep, out_dir, device, opts, progress, log, cancel)
    elif orig.is_dir() and dep.is_dir():
        _sbs_images(orig, dep, out_dir, device, opts, progress, log, cancel)
    else:
        raise ValueError("Original and depth must both be video files, or both be "
                         "image folders.")
    log("SBS job finished.")


def _sbs_video(orig: Path, dep: Path, out_dir: Path, device, opts,
               progress, log, cancel):
    import torch

    from .sbs import make_sbs

    half = bool(opts.get("half_sbs"))
    invert = bool(opts.get("invert_depth"))
    codec = encoder_args(opts["encoder"], opts["quality"], opts.get("preset", "balanced"))

    r_orig = VideoReader(orig, "rgb24")
    r_dep = VideoReader(dep, "gray")
    info = r_orig.info
    if abs(info.n_frames - r_dep.info.n_frames) > 2 and info.n_frames and r_dep.info.n_frames:
        log(f"WARNING: frame counts differ (original ~{info.n_frames}, "
            f"depth ~{r_dep.info.n_frames}); output stops at the shorter one.")

    out_w = info.width if half else info.width * 2
    tag = "HSBS" if half else "SBS"
    out_path = out_dir / f"{orig.stem}_{tag}.mp4"
    log(f"{info.width}x{info.height} -> {out_w}x{info.height} ({tag}) -> {out_path.name}"
        + (" [audio copied]" if info.has_audio else ""))
    writer = VideoWriter(out_path, out_w, info.height, info.fps_str, codec,
                         pix_fmt_in="rgb24", audio_from=orig)

    total = max(info.n_frames, 1)
    done = 0
    imgs: list[np.ndarray] = []
    deps: list[np.ndarray] = []

    def flush():
        nonlocal done
        if not imgs:
            return
        img_t = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        dep_t = torch.from_numpy(np.stack(deps)).to(device).float().div_(255.0)
        if invert:
            dep_t = 1.0 - dep_t
        sbs = make_sbs(img_t, dep_t, opts["divergence"], opts["convergence"], half)
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

    files = sorted(p for p in orig_dir.iterdir() if p.suffix.lower() in IMG_EXTS
                   and not p.stem.endswith(("_depth", "_SBS", "_HSBS")))
    if not files:
        raise ValueError("No images found in the original folder.")
    half = bool(opts.get("half_sbs"))
    tag = "HSBS" if half else "SBS"
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
        out = sbs_frame_uint8(rgb, depth, device, opts["divergence"],
                              opts["convergence"], half,
                              invert_depth=bool(opts.get("invert_depth")))
        PILImage.fromarray(out).save(out_dir / f"{path.stem}_{tag}.png")
        progress((i + 1) / len(files), f"image {i + 1}/{len(files)}: {path.name}")
    log(f"Wrote {len(files) - skipped} SBS images to {out_dir}"
        + (f" ({skipped} skipped)" if skipped else ""))
