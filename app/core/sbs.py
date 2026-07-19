"""Stereo (SBS) synthesis from image + depth - proper DIBR on the GPU.

Method: forward warping with depth ordering. Every source pixel is projected
to its new horizontal position for each eye; where several pixels land on the
same spot the NEAREST one wins (z-buffer), so foreground genuinely occludes
background - no stretched/repeated edge artifacts like naive inverse warping
produces. Disoccluded holes (background revealed next to a near object) are
filled from the background side and lightly blurred, which is how dedicated
2D->3D tools handle them.

Comfort math:
- depth in [0, 1], 1 = near (white).
- `divergence` = TOTAL disparity budget between the eyes as % of frame width;
  each eye gets half. ~1-1.5% is comfortable in VR.
- pixels at `convergence` depth have zero disparity (screen plane); auto
  convergence tracks the subject so it stays there.
- per-eye shift is hard-clamped to MAX_EYE_SHIFT_PCT of width.

Eye geometry: near content moves RIGHT in the left eye and LEFT in the right
eye (crossed disparity) - correct for VR headsets (left half -> left eye).
"""

import torch
import torch.nn.functional as F

MAX_EYE_SHIFT_PCT = 1.0  # max per-eye shift, % of width (2% total disparity)


def gaussian_blur_depth(depths: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
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


def _smooth_for_warp(depths: torch.Tensor) -> torch.Tensor:
    """Light, resolution-scaled smoothing: removes compression stair-steps in
    the depth map (which cause zig-zag edges) without big depth halos."""
    h, w = depths.shape[-2:]
    ksize = max(5, (min(h, w) // 150) | 1)
    return gaussian_blur_depth(depths, ksize=ksize, sigma=ksize / 3.0)


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


def _forward_warp_eye(images: torch.Tensor, depths: torch.Tensor,
                      shift_px: torch.Tensor) -> torch.Tensor:
    """Forward-warp (B,3,H,W) images horizontally by per-pixel shift_px
    (positive = move right). Depth-ordered: nearer pixels win collisions.
    Holes are filled from the background side, then softened."""
    b, c, h, w = images.shape
    device = images.device

    xs = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
    ys = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
    bs = torch.arange(b, device=device).view(b, 1, 1).expand(b, h, w)

    xt = (xs + shift_px).round().long()
    valid = (xt >= 0) & (xt < w)
    tgt_flat = ((bs * h + ys) * w + xt.clamp(0, w - 1)).reshape(-1)
    src_flat = ((bs * h + ys) * w + xs).reshape(-1)
    valid_flat = valid.reshape(-1)
    depth_flat = depths.reshape(-1)

    # z-buffer: nearest depth per target pixel
    zbuf = torch.full((b * h * w,), -1.0, device=device)
    zbuf.scatter_reduce_(0, tgt_flat[valid_flat], depth_flat[valid_flat],
                         reduce="amax", include_self=True)

    # winners: pixels whose depth matches the z-buffer at their target
    winner = valid_flat & (depth_flat >= zbuf[tgt_flat] - 1e-4)
    img_flat = images.permute(1, 0, 2, 3).reshape(c, -1)
    out_flat = torch.zeros_like(img_flat)
    out_flat[:, tgt_flat[winner]] = img_flat[:, src_flat[winner]]
    out = out_flat.reshape(c, b, h, w).permute(1, 0, 2, 3)

    zb = zbuf.reshape(b, h, w)
    hole = zb < 0
    if hole.any():
        # nearest valid pixel to the left / right of every position (per row)
        idx = xs
        li = torch.where(hole, torch.full_like(idx, -1), idx)
        left_near = li.cummax(dim=-1).values                       # -1 = none
        ri = torch.where(hole, torch.full_like(idx, w), idx)
        right_near = torch.flip(torch.flip(ri, dims=[-1]).cummin(dim=-1).values,
                                dims=[-1])                          # w = none
        zl = zb.gather(-1, left_near.clamp(min=0))
        zr = zb.gather(-1, right_near.clamp(max=w - 1))
        # fill from the BACKGROUND side (smaller depth), so revealed areas get
        # background content, never a smeared copy of the foreground edge
        use_left = (zl <= zr) | (right_near >= w)
        use_left &= left_near >= 0
        fill_idx = torch.where(use_left, left_near.clamp(min=0),
                               right_near.clamp(max=w - 1))
        filled = out.gather(3, fill_idx.unsqueeze(1).expand(b, c, h, w))
        out = torch.where(hole.unsqueeze(1), filled, out)
        # soften the filled streaks
        blur = F.avg_pool2d(F.pad(out, (2, 2, 0, 0), mode="replicate"),
                            kernel_size=(1, 5), stride=1)
        out = torch.where(hole.unsqueeze(1), blur, out)
    return out


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
        depths = _smooth_for_warp(depths)

    if convergences is not None:
        conv = torch.tensor(convergences, device=device,
                            dtype=torch.float32).view(b, 1, 1)
    else:
        conv = torch.full((b, 1, 1), float(convergence), device=device)

    # per-eye shift as fraction of width, clamped for comfort, then to pixels
    eye_shift = (depths - conv) * (divergence / 100.0) / 2.0
    eye_shift = eye_shift.clamp(-MAX_EYE_SHIFT_PCT / 100.0, MAX_EYE_SHIFT_PCT / 100.0)
    shift_px = eye_shift * w

    left = _forward_warp_eye(images, depths, shift_px)    # near moves right
    right = _forward_warp_eye(images, depths, -shift_px)  # near moves left
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
