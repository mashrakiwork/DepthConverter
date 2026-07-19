"""Stereo (SBS) synthesis from image + depth via per-pixel horizontal warping.

Depth-image-based rendering with inverse warping on the GPU (grid_sample).
Conventions and comfort math (what makes it fuse properly in a headset):

- depth in [0, 1], 1 = near (white).
- `divergence` is the TOTAL disparity budget between the two eyes as a percent
  of frame width; each eye is warped by half of it. ~1-1.5% total is the
  comfortable range for VR viewing.
- pixels at `convergence` depth have zero disparity (they sit exactly on the
  screen plane); nearer pixels get crossed disparity (pop out), farther pixels
  uncrossed (go in). Auto-convergence tracks the subject depth per frame so
  the main subject stays on the screen plane - this is what keeps the image
  easy to fuse and avoids "double vision".
- per-eye shift is hard-clamped to MAX_EYE_SHIFT_PCT of width as a safety net.
- the depth map is lightly gaussian-smoothed before warping to avoid tearing
  and edge shimmer at depth discontinuities.

Left eye samples at x - s, so near content moves RIGHT in the left eye and
LEFT in the right eye (crossed disparity) - correct for parallel viewing on
VR headsets (left half -> left eye).
"""

import torch
import torch.nn.functional as F

MAX_EYE_SHIFT_PCT = 1.0  # max per-eye shift, % of width (2% total disparity)


def gaussian_blur_depth(depths: torch.Tensor, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Separable gaussian blur on (B, H, W) depth maps."""
    x = torch.arange(ksize, dtype=torch.float32, device=depths.device) - (ksize - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    d = depths.unsqueeze(1)
    pad = ksize // 2
    d = F.conv2d(F.pad(d, (pad, pad, 0, 0), mode="replicate"),
                 kernel.view(1, 1, 1, -1))
    d = F.conv2d(F.pad(d, (0, 0, pad, pad), mode="replicate"),
                 kernel.view(1, 1, -1, 1))
    return d.squeeze(1)


def estimate_subject_depth(depth: torch.Tensor) -> float:
    """Robust subject depth for one frame (H, W): median of the center region,
    where the subject usually is. Used by auto-convergence."""
    h, w = depth.shape[-2:]
    center = depth[..., h // 4: h - h // 4, w // 4: w - w // 4]
    return float(center.median())


class ConvergenceTracker:
    """EMA-smoothed subject depth across video frames, so the screen plane
    follows the subject without jumping frame to frame."""

    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, depth_frame: torch.Tensor) -> float:
        subject = estimate_subject_depth(depth_frame)
        if self.value is None:
            self.value = subject
        else:
            self.value += self.alpha * (subject - self.value)
        return self.value


def make_sbs(images: torch.Tensor, depths: torch.Tensor, divergence: float = 1.2,
             convergence: float = 0.5, half: bool = False,
             smooth_depth: bool = True,
             convergences: list[float] | None = None) -> torch.Tensor:
    """
    images: (B, 3, H, W) float in [0, 1]
    depths: (B, H, W) float in [0, 1], 1 = near
    divergence: TOTAL disparity budget between eyes, percent of frame width
    convergence: depth that sits on the screen plane (per-batch scalar), or
                 per-frame values via `convergences`
    Returns (B, 3, H, 2W) full SBS, or (B, 3, H, W) squeezed half SBS.
    """
    b, _, h, w = images.shape
    device = images.device
    if depths.shape[-2:] != (h, w):
        depths = F.interpolate(depths.unsqueeze(1), size=(h, w), mode="bilinear",
                               align_corners=False).squeeze(1)
    if smooth_depth:
        depths = gaussian_blur_depth(depths)

    if convergences is not None:
        conv = torch.tensor(convergences, device=device,
                            dtype=torch.float32).view(b, 1, 1)
    else:
        conv = torch.full((b, 1, 1), float(convergence), device=device)

    # per-eye shift in pixels-as-fraction-of-width, then clamped for comfort
    eye_shift = (depths - conv) * (divergence / 100.0) / 2.0
    eye_shift = eye_shift.clamp(-MAX_EYE_SHIFT_PCT / 100.0, MAX_EYE_SHIFT_PCT / 100.0)
    # to normalized grid units ([-1, 1] spans the width -> factor 2)
    shift = eye_shift * 2.0

    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * 2.0 / h - 1.0
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * 2.0 / w - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.expand(b, h, w)
    grid_x = grid_x.expand(b, h, w)

    left = F.grid_sample(images, torch.stack((grid_x - shift, grid_y), dim=-1),
                         mode="bilinear", padding_mode="border", align_corners=False)
    right = F.grid_sample(images, torch.stack((grid_x + shift, grid_y), dim=-1),
                          mode="bilinear", padding_mode="border", align_corners=False)
    sbs = torch.cat([left, right], dim=3)
    if half:
        sbs = F.interpolate(sbs, size=(h, w), mode="bilinear", align_corners=False,
                            antialias=True)
    return sbs


def sbs_frame_uint8(image_u8, depth_arr, device: str, divergence: float,
                    convergence: float | None, half: bool,
                    invert_depth: bool = False, smooth_depth: bool = True):
    """numpy uint8 image + uint8/uint16 depth -> uint8 SBS frame.
    convergence=None enables auto-convergence for this image."""
    img = torch.from_numpy(image_u8.copy()).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    depth_max = 65535.0 if depth_arr.dtype.itemsize == 2 else 255.0
    dep = torch.from_numpy(depth_arr.astype("float32")).to(device).div_(depth_max).unsqueeze(0)
    if invert_depth:
        dep = 1.0 - dep
    if convergence is None:
        convergence = estimate_subject_depth(dep[0])
    out = make_sbs(img, dep, divergence, convergence, half, smooth_depth=smooth_depth)
    return (out[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
