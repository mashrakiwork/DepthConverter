"""Depth estimation engine.

- Depth Anything V2 / DPT via transformers; Depth Anything V3 (DA3) via the
  official depth_anything_3 package (optional extra).
- FP16 on CUDA, adaptive batch size that halves itself on out-of-memory.
- Robust percentile normalization (2%/98%) with temporal EMA for videos so
  the depth range doesn't flicker; per-image range for stills.
- Output convention: WHITE = NEAR, black = far (the `invert` flag flips it).
  V2/DPT predict disparity (higher = closer) directly; DA3 predicts scene
  distance, which is converted to disparity here (1/d).
"""

import numpy as np
import torch

from .hardware import suggest_batch_size
from .models import ensure_downloaded, is_da3

_EMA_ALPHA = 0.05
_DA3_PROCESS_RES = 504  # DA3 native processing resolution
_MAX_QUANTILE_SAMPLES = 1_000_000


class DepthEstimator:
    def __init__(self, model_id: str, device: str = "cuda", fp16: bool = True,
                 invert: bool = False, log=print):
        self.device = device
        self.fp16 = fp16 and device == "cuda"
        self.invert = invert
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self._is_da3 = is_da3(model_id)

        ensure_downloaded(model_id, log=log)
        log(f"Loading model '{model_id}' on {device.upper()}...")
        if self._is_da3:
            # Silence the DA3 package's console chatter (per-batch INFO timing
            # lines and the gsplat warning - gsplat is only for 3D Gaussian
            # rendering, which we don't use). Must be set before import.
            import logging
            import os
            os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
            # Triton doesn't exist on Windows; stop xformers/torch from
            # probing for it and printing a harmless traceback.
            os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
            logging.getLogger("xformers").setLevel(logging.ERROR)
            logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError:
                raise RuntimeError(
                    "Depth Anything V3 support is not installed. Run the setup "
                    "script again (it installs the 'da3' extra), or run: "
                    "uv sync --extra da3") from None
            self.model = DepthAnything3.from_pretrained(model_id)
            self.model.to(device)
            self.model.eval()
            self.processor = None
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id, torch_dtype=self.dtype)
            self.model.to(device).eval()

        self.batch_size = suggest_batch_size(model_id, device)
        if self._is_da3:
            # DA3 attends across all images in a batch (multi-view); keep
            # chunks small so VRAM stays bounded.
            self.batch_size = min(self.batch_size, 4)
        log(f"Model ready (batch size {self.batch_size}, "
            f"{'FP16' if self.fp16 else 'FP32'}).")
        self._ema_lo: float | None = None
        self._ema_hi: float | None = None

    @torch.inference_mode()
    def _forward(self, frames: list[np.ndarray]) -> torch.Tensor:
        h, w = frames[0].shape[:2]
        if self._is_da3:
            from contextlib import nullcontext

            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if self.fp16 else nullcontext())
            with ctx:
                pred = self.model.inference(list(frames),
                                            process_res=_DA3_PROCESS_RES)
            depth = pred.depth
            if not torch.is_tensor(depth):
                depth = torch.from_numpy(np.asarray(depth))
            depth = depth.float().to(self.device)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            # DA3 outputs scene distance (larger = farther) -> disparity.
            pred_t = 1.0 / depth.clamp_min(1e-4)
        else:
            inputs = self.processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            pred_t = self.model(pixel_values=pixel_values).predicted_depth.float()
        pred_t = torch.nn.functional.interpolate(
            pred_t.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(1)
        return pred_t

    def infer(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Disparity-style depth (B, H, W) float32, higher = closer. OOM-safe."""
        while True:
            try:
                outs = [self._forward(frames[i:i + self.batch_size])
                        for i in range(0, len(frames), self.batch_size)]
                return outs[0] if len(outs) == 1 else torch.cat(outs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)

    def reset_temporal(self) -> None:
        """Call between independent items (each image, each new video)."""
        self._ema_lo = self._ema_hi = None

    @staticmethod
    def _robust_range(d: torch.Tensor) -> tuple[float, float]:
        """2%/98% percentiles (subsampled) - robust to sky/speckle outliers."""
        flat = d.reshape(-1)
        if flat.numel() > _MAX_QUANTILE_SAMPLES:
            step = flat.numel() // _MAX_QUANTILE_SAMPLES + 1
            flat = flat[::step]
        q = torch.quantile(flat.float(), torch.tensor([0.02, 0.98], device=flat.device))
        return float(q[0]), float(q[1])

    def _normalized(self, pred: torch.Tensor, temporal: bool):
        out = torch.empty_like(pred)
        for i in range(pred.shape[0]):
            lo, hi = self._robust_range(pred[i])
            if temporal:
                if self._ema_lo is None:
                    self._ema_lo, self._ema_hi = lo, hi
                else:
                    self._ema_lo += _EMA_ALPHA * (lo - self._ema_lo)
                    self._ema_hi += _EMA_ALPHA * (hi - self._ema_hi)
                lo, hi = self._ema_lo, self._ema_hi
            out[i] = (pred[i] - lo) / max(hi - lo, 1e-6)
        out = out.clamp(0, 1)
        if self.invert:
            out = 1.0 - out
        return out

    def normalize(self, pred: torch.Tensor, temporal: bool = True) -> np.ndarray:
        """(B, H, W) raw depth -> (B, H, W) uint8, white = near (unless invert)."""
        out = self._normalized(pred, temporal)
        return (out * 255.0).round().byte().cpu().numpy()

    def normalize16(self, pred: torch.Tensor) -> np.ndarray:
        """Per-frame 16-bit variant, for high-precision depth PNGs."""
        out = self._normalized(pred, temporal=False)
        return (out * 65535.0).round().to(torch.int32).cpu().numpy().astype(np.uint16)
