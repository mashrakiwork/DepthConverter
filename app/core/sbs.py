"""Stereo (SBS) synthesis from image + depth via per-pixel horizontal warping.

Depth-image-based rendering with inverse warping on the GPU (grid_sample):
near pixels get negative parallax (pop out of the screen), pixels at the
convergence depth sit on the screen plane. The per-eye aspect ratio is always
identical to the source: full SBS is exactly 2x the source width.
"""

import torch
import torch.nn.functional as F


def make_sbs(images: torch.Tensor, depths: torch.Tensor, divergence: float = 2.0,
             convergence: float = 0.5, half: bool = False) -> torch.Tensor:
    """
    images: (B, 3, H, W) float in [0, 1]
    depths: (B, H, W) float in [0, 1], 1 = near
    divergence: max per-eye shift as percent of frame width
    Returns (B, 3, H, 2W) full SBS, or (B, 3, H, W) squeezed half SBS.
    """
    b, _, h, w = images.shape
    device = images.device
    if depths.shape[-2:] != (h, w):
        depths = F.interpolate(depths.unsqueeze(1), size=(h, w), mode="bilinear",
                               align_corners=False).squeeze(1)

    # normalized grid coords ([-1, 1], pixel centers)
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * 2.0 / h - 1.0
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * 2.0 / w - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.expand(b, h, w)
    grid_x = grid_x.expand(b, h, w)

    # shift in normalized units: shift_px = (depth - conv) * div% * W  ->  * 2/W
    shift = (depths - convergence) * (divergence / 100.0) * 2.0

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
                    convergence: float, half: bool, invert_depth: bool = False):
    """Convenience wrapper for numpy uint8 image + uint8/uint16 depth -> uint8 SBS."""
    img = torch.from_numpy(image_u8.copy()).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    depth_max = 65535.0 if depth_arr.dtype.itemsize == 2 else 255.0
    dep = torch.from_numpy(depth_arr.astype("float32")).to(device).div_(depth_max).unsqueeze(0)
    if invert_depth:
        dep = 1.0 - dep
    out = make_sbs(img, dep, divergence, convergence, half)
    return (out[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
