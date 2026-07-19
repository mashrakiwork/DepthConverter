"""Depth estimation engine.

- FP16 on CUDA, adaptive batch size that halves itself on out-of-memory.
- Temporal EMA normalization for videos so the depth range doesn't flicker
  frame to frame; per-image min/max for stills.
- Models output inverse-depth style maps: higher = closer, so the written
  grayscale convention is white = near (the `invert` flag flips it).
"""

import numpy as np
import torch

from .hardware import suggest_batch_size
from .models import ensure_downloaded

_EMA_ALPHA = 0.05


class DepthEstimator:
    def __init__(self, model_id: str, device: str = "cuda", fp16: bool = True,
                 invert: bool = False, log=print):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        ensure_downloaded(model_id, log=log)
        log(f"Loading model '{model_id}' on {device.upper()}...")
        self.device = device
        self.fp16 = fp16 and device == "cuda"
        self.invert = invert
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(device).eval()
        self.batch_size = suggest_batch_size(model_id, device)
        log(f"Model ready (batch size {self.batch_size}, "
            f"{'FP16' if self.fp16 else 'FP32'}).")
        self._ema_lo: float | None = None
        self._ema_hi: float | None = None

    @torch.inference_mode()
    def _forward(self, frames: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        pred = self.model(pixel_values=pixel_values).predicted_depth  # (B, h, w)
        h, w = frames[0].shape[:2]
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(1)
        return pred

    def infer(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Raw relative depth (B, H, W) float32 on self.device. OOM-safe."""
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

    def normalize(self, pred: torch.Tensor, temporal: bool = True) -> np.ndarray:
        """(B, H, W) raw depth -> (B, H, W) uint8, white = near (unless invert)."""
        out = torch.empty_like(pred)
        for i in range(pred.shape[0]):
            lo = float(pred[i].amin())
            hi = float(pred[i].amax())
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
        return (out * 255.0).round().byte().cpu().numpy()

    def normalize16(self, pred: torch.Tensor) -> np.ndarray:
        """Like normalize() but per-frame 16-bit, for high-precision depth PNGs."""
        outs = []
        for i in range(pred.shape[0]):
            lo = float(pred[i].amin())
            hi = float(pred[i].amax())
            d = ((pred[i] - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
            if self.invert:
                d = 1.0 - d
            outs.append((d * 65535.0).round().to(torch.int32).cpu().numpy().astype(np.uint16))
        return np.stack(outs)
