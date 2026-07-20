"""AI upscaling engine - Real-ESRGAN-family models loaded via spandrel.

- Curated registry of popular HuggingFace-hosted .pth weights; the UI combo
  is editable, so any "repo_id :: filename.pth" pair can be typed too.
- Tiled inference keeps VRAM bounded regardless of frame size; the tile size
  halves itself on out-of-memory (same spirit as the depth engine's batch
  backoff). Tiles overlap and only their inner region is kept, so seams are
  invisible.
- FP16 on CUDA when the architecture supports it.
- The model's native scale (2x/4x) is chained and/or downscaled to reach the
  requested 2x/3x/4x exactly, preserving the aspect ratio.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .hardware import free_vram_gb

# (label, HF repo id, filename). All are well-known community upscalers.
UPSCALE_MODELS: list[tuple[str, str, str]] = [
    ("Real-ESRGAN x4plus (general, recommended)",
     "lllyasviel/Annotators", "RealESRGAN_x4plus.pth"),
    ("Real-ESRGAN x2 (faster, lighter)",
     "ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth"),
    ("Real-ESRGAN x4",
     "ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth"),
    ("4x UltraSharp (crisp fine detail)",
     "Kim2091/UltraSharp", "4x-UltraSharp.pth"),
    ("4x AnimeSharp (anime / cartoons)",
     "Kim2091/AnimeSharp", "4x-AnimeSharp.pth"),
]

_TILE_OVERLAP = 16
_MIN_TILE = 128


def parse_model_spec(text: str) -> tuple[str, str]:
    """'repo_id :: filename.pth' -> (repo_id, filename)."""
    repo, sep, filename = text.partition("::")
    repo, filename = repo.strip(), filename.strip()
    if not sep or not repo or not filename:
        raise ValueError(
            "Custom upscale models must be written as 'repo_id :: filename.pth' "
            "(a HuggingFace repo id and the weights file inside it).")
    return repo, filename


def download_weights(repo_id: str, filename: str, log=print) -> str:
    """Fetch one weights file into the local HF cache; download only if missing."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except Exception:
        log(f"Model '{repo_id} :: {filename}' not cached yet - downloading from "
            f"Hugging Face (one-time, this can take a while)...")
        path = hf_hub_download(repo_id, filename)
        log("Model download complete.")
        return path


def _initial_tile(device: str) -> int:
    if device != "cuda":
        return 512  # bounds CPU RAM / keeps progress responsive
    free = free_vram_gb()
    if free >= 10:
        return 1024
    if free >= 5:
        return 512
    return 256


class Upscaler:
    def __init__(self, repo_id: str, filename: str, device: str = "cuda",
                 fp16: bool = True, log=print):
        from spandrel import ImageModelDescriptor, ModelLoader

        path = download_weights(repo_id, filename, log=log)
        log(f"Loading model '{filename}' on {device.upper()}...")
        desc = ModelLoader().load_from_file(path)
        if not isinstance(desc, ImageModelDescriptor):
            raise ValueError(f"'{filename}' is not a single-image model "
                             f"(got {type(desc).__name__}).")
        self.device = device
        self.scale = int(desc.scale)
        self.fp16 = fp16 and device == "cuda" and bool(desc.supports_half)
        desc.to(device).eval()
        if self.fp16:
            desc.half()
        self.model = desc.model
        self.tile = _initial_tile(device)
        self._log = log
        log(f"Model ready: {desc.architecture.name}, native {self.scale}x, "
            f"{'FP16' if self.fp16 else 'FP32'}, tile {self.tile}px.")

    @torch.inference_mode()
    def _infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.half() if self.fp16 else x).float()

    def _tiled(self, x: torch.Tensor) -> torch.Tensor:
        """Overlap-and-crop tiling: each tile is run with a margin of context
        and only its inner region is pasted, so tile seams are invisible."""
        _, _, h, w = x.shape
        s, t, ov = self.scale, self.tile, _TILE_OVERLAP
        out = torch.zeros((1, x.shape[1], h * s, w * s), device=x.device,
                          dtype=torch.float32)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                ey0, ex0 = max(y0 - ov, 0), max(x0 - ov, 0)
                ey1, ex1 = min(y1 + ov, h), min(x1 + ov, w)
                pred = self._infer(x[:, :, ey0:ey1, ex0:ex1])
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = pred[
                    :, :, (y0 - ey0) * s:(y1 - ey0) * s,
                    (x0 - ex0) * s:(x1 - ex0) * s]
        return out

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """One model pass over a full frame, tile-halving on out-of-memory."""
        while True:
            try:
                _, _, h, w = x.shape
                if max(h, w) <= self.tile:
                    return self._infer(x)
                return self._tiled(x)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.tile <= _MIN_TILE:
                    raise
                self.tile //= 2
                self._log(f"Low VRAM: reducing tile size to {self.tile}px")

    @torch.inference_mode()
    def upscale(self, rgb: np.ndarray, target_scale: int) -> np.ndarray:
        """uint8 (H, W, 3) -> uint8 (H*target, W*target, 3)."""
        h, w = rgb.shape[:2]
        # copy: video frames arrive as read-only buffer views
        x = (torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
             .permute(2, 0, 1).unsqueeze(0).float().div_(255.0))
        x = self._run(x)
        done = self.scale
        while 1 < done < target_scale:  # chain passes (e.g. 2x model -> 4x target)
            x = self._run(x)
            done *= self.scale
        if done != target_scale:  # e.g. 4x model -> 3x target
            x = F.interpolate(x, size=(h * target_scale, w * target_scale),
                              mode="bicubic", antialias=True)
        return (x.clamp_(0, 1).mul_(255.0).round_().byte()
                .squeeze(0).permute(1, 2, 0).cpu().numpy())
