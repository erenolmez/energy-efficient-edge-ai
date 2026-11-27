"""Batched torch ports of the hand-made router features (run on the A40, or on CPU for checks).

  compact_texture(x)   -> (B, 46), (B, 450): jetson_tests/scripts/router_features.py
  taylor29(x, kp_thr)  -> (B, 29): hpc/taylor_premodel.raw_features
x is a uint8 (B, 32, 32, 3) RGB tensor already on the device. Everything is float32 (the CPU
versions use float32 for compact/texture and float64 for Taylor), so values agree to rounding;
features that threshold or bin a value can flip when it sits on a boundary. hpc.gpu_feature_routers
measures the agreement on every fit and calibration image.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

GRAY = (0.2989, 0.5870, 0.1140)       # router_features.py
GRAY_T = (0.299, 0.587, 0.114)        # taylor_premodel.py
HESSIAN_SCALES = (1.2, 1.6, 2.0, 2.8)  # taylor_premodel.hessian_maxima


def np_gradient(g):
    """numpy.gradient of (B, H, W): returns (dy, dx), central inside, one-sided at the edges."""
    def along(t, dim):
        n = t.shape[dim]
        inner = (t.narrow(dim, 2, n - 2) - t.narrow(dim, 0, n - 2)) / 2
        first = t.narrow(dim, 1, 1) - t.narrow(dim, 0, 1)
        last = t.narrow(dim, n - 1, 1) - t.narrow(dim, n - 2, 1)
        return torch.cat([first, inner, last], dim)
    return along(g, 1), along(g, 2)


def bin_hist(values, bins, mask=None, weights=None, norm="size"):
    """router_features._histogram: index = min(floor(v * bins), bins - 1), per image (B, N)."""
    idx = (values * bins).to(torch.int64).clamp_(0, bins - 1)
    w = torch.ones_like(values) if weights is None else weights
    if mask is not None:
        w = w * mask
    out = torch.zeros(values.shape[0], bins, device=values.device, dtype=values.dtype)
    out.scatter_add_(1, idx, w)
    if norm == "size":
        return out / values.shape[1]
    total = out.sum(1, keepdim=True)
    return out / total.clamp_min(1)  # numpy-histogram counts / max(count, 1)


def spatial_index(h, w, d, device):
    rows = torch.clamp(torch.arange(h, device=device) * d // h, max=d - 1)
    cols = torch.clamp(torch.arange(w, device=device) * d // w, max=d - 1)
    return (rows[:, None] * d + cols[None, :]).reshape(-1)


def compact_texture(x, texture=True):
    B, H, W, _ = x.shape
    dev = x.device
    rgb = x.float() / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = r * GRAY[0] + g * GRAY[1] + b * GRAY[2]
    flat = gray.reshape(B, -1)
    gh = bin_hist(flat, 32)
    entropy = -torch.where(gh > 0, gh * torch.log2(gh.clamp_min(1e-30)), torch.zeros_like(gh)).sum(1)
    q = torch.quantile(flat, torch.tensor([0.1, 0.9], device=dev), dim=1)
    dy, dx = np_gradient(gray)
    mag = torch.hypot(dx, dy)
    pad = F.pad(gray[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    lap = pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:] - 4 * gray
    sat = rgb.amax(3) - rgb.amin(3)
    quads = rgb.reshape(B, 2, H // 2, 2, W // 2, 3)
    qm = quads.mean((2, 4))
    qs = quads.std((2, 4), unbiased=False)
    spatial = torch.stack([qm, qs], 3).reshape(B, -1)  # [qy, qx, stat, channel]
    center = gray[:, H // 4:3 * H // 4, W // 4:3 * W // 4]
    cm = center.mean((1, 2))
    bm = (gray.sum((1, 2)) - center.sum((1, 2))) / (H * W - center[0].numel())
    std = lambda t: t.reshape(B, -1).std(1, unbiased=False)
    compact = torch.cat([
        rgb.mean((1, 2)), rgb.std((1, 2), unbiased=False),
        torch.stack([flat.mean(1), std(gray), entropy, q[0], q[1], q[1] - q[0], mag.mean((1, 2)), std(mag),
                     (mag > 0.1).float().mean((1, 2)), lap.abs().mean((1, 2)), lap.reshape(B, -1).var(1, unbiased=False),
                     sat.mean((1, 2)), std(sat)], 1),
        spatial, torch.stack([cm, bm, cm - bm], 1)], 1)
    if not texture:
        return compact, None

    hists = [bin_hist(rgb[..., i].reshape(B, -1), 8) for i in range(3)] + [bin_hist(flat, 16)]
    # HOG (router_features._hog)
    orient = torch.remainder(torch.atan2(dy, dx), math.pi).reshape(B, -1) * (9 / math.pi)
    lower = torch.floor(orient)
    frac = orient - lower
    lower = lower.to(torch.int64)
    cells = (spatial_index(H, W, 4, dev) * 9)[None]
    m = mag.reshape(B, -1)
    hog = torch.zeros(B, 144, device=dev)
    hog.scatter_add_(1, cells + lower % 9, m * (1 - frac))
    hog.scatter_add_(1, cells + (lower + 1) % 9, m * frac)
    h4 = hog.reshape(B, 4, 4, 9)
    blocks = torch.stack([h4[:, :-1, :-1], h4[:, :-1, 1:], h4[:, 1:, :-1], h4[:, 1:, 1:]], 3).reshape(B, 3, 3, 36)
    blocks = blocks / torch.sqrt((blocks * blocks).sum(3, keepdim=True) + 1e-10)
    blocks = blocks.clamp(max=0.2)
    blocks = blocks / torch.sqrt((blocks * blocks).sum(3, keepdim=True) + 1e-10)
    # LBP (router_features._lbp)
    c = gray[:, 1:-1, 1:-1]
    nb = torch.stack([gray[:, :-2, :-2], gray[:, :-2, 1:-1], gray[:, :-2, 2:], gray[:, 1:-1, 2:],
                      gray[:, 2:, 2:], gray[:, 2:, 1:-1], gray[:, 2:, :-2], gray[:, 1:-1, :-2]], 1)
    bits = nb >= c[:, None]
    trans = (bits != torch.roll(bits, 1, dims=1)).sum(1)
    codes = torch.where(trans <= 2, bits.sum(1), torch.full_like(trans, 9)).reshape(B, -1)
    quad = spatial_index(H - 2, W - 2, 2, dev)[None]
    lbp = torch.zeros(B, 40, device=dev)
    lbp.scatter_add_(1, quad * 10 + codes, torch.ones_like(codes, dtype=torch.float32))
    lbp = (lbp.reshape(B, 4, 10) / lbp.reshape(B, 4, 10).sum(2, keepdim=True)).reshape(B, 40)
    texture = torch.cat([compact, *hists, blocks.reshape(B, -1), lbp], 1)
    return compact, texture


# ------------------------------------------------------------------ Taylor et al. (29 features)
def _gauss_weights(sigma, order):
    """Correlation weights scipy.ndimage.gaussian_filter1d applies (truncate 4.0)."""
    from scipy import ndimage
    radius = int(4.0 * sigma + 0.5)
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1.0
    return ndimage.gaussian_filter1d(impulse, sigma, order=order, mode="constant")[::-1].copy(), radius


class Taylor29:
    """Holds the Gaussian derivative kernels (computed once with scipy) for a device."""

    def __init__(self, device, kp_threshold):
        self.device, self.kp_threshold = device, kp_threshold
        self.k = {}
        for s in HESSIAN_SCALES:
            for order in (0, 1, 2):
                w, r = _gauss_weights(s, order)
                self.k[(s, order)] = (torch.tensor(w, dtype=torch.float32, device=device), r)

    def _sym_index(self, n, r):
        return torch.as_tensor(np.pad(np.arange(n), r, mode="symmetric"), device=self.device)

    def _filt(self, img, s, order_y, order_x):
        """scipy gaussian_filter(img, s, order=(order_y, order_x)), mode 'reflect' (symmetric)."""
        B, H, W = img.shape
        ky, ry = self.k[(s, order_y)]
        kx, rx = self.k[(s, order_x)]
        t = img[:, self._sym_index(H, ry)]
        t = F.conv2d(t[:, None], ky.view(1, 1, -1, 1))[:, 0]
        t = t[:, :, self._sym_index(W, rx)]
        return F.conv2d(t[:, None], kx.view(1, 1, 1, -1))[:, 0]

    def hessian_peaks(self, gray):
        stack = []
        for s in HESSIAN_SCALES:
            dxx = self._filt(gray, s, 0, 2)
            dyy = self._filt(gray, s, 2, 0)
            dxy = self._filt(gray, s, 1, 1)
            stack.append((dxx * dyy - dxy ** 2) * s ** 4)
        r = torch.stack(stack, 1)  # B, S, H, W
        mx = F.max_pool3d(F.pad(r[:, None], (1, 1, 1, 1, 1, 1), mode="replicate"), 3, 1)[:, 0]
        peak = (r == mx) & (r > 0)
        peak[:, :, :2], peak[:, :, -2:], peak[:, :, :, :2], peak[:, :, :, -2:] = False, False, False, False
        return r, peak

    @staticmethod
    def components(mask, max_iter=1100):
        """8-connected labels by max-label propagation; label = 1 + max flat index in the component."""
        B, H, W = mask.shape
        ids = torch.arange(1, H * W + 1, device=mask.device, dtype=torch.float32).view(1, H, W)
        lab = torch.where(mask, ids.expand(B, H, W), torch.zeros(1, device=mask.device))
        for i in range(max_iter):
            new = torch.where(mask, F.max_pool2d(lab[:, None], 3, 1, 1)[:, 0], lab)
            if i % 8 == 7 and torch.equal(new, lab):
                break
            lab = new
        return lab.to(torch.int64)

    def __call__(self, x):
        B, H, W, _ = x.shape
        dev = x.device
        rgb = x.float() / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        gray = GRAY_T[0] * r + GRAY_T[1] * g + GRAY_T[2] * b
        plain = rgb.mean(3)
        perceived = torch.sqrt(0.241 * r ** 2 + 0.691 * g ** 2 + 0.068 * b ** 2)
        resp, peak = self.hessian_peaks(gray)
        kp = ((resp > self.kp_threshold) & peak).sum((1, 2, 3)).float() if self.kp_threshold is not None \
            else torch.zeros(B, device=dev)

        dy, dx = np_gradient(gray)
        mag = torch.hypot(dx, dy)
        edges = mag > 0.1
        lab = self.components(edges).reshape(B, -1)
        sizes = torch.zeros(B, H * W + 1, device=dev)
        sizes.scatter_add_(1, lab, torch.ones_like(lab, dtype=torch.float32))
        sizes = sizes[:, 1:]
        n_seg = (sizes > 0).sum(1).float()
        bins = torch.tensor([1, 2, 3, 5, 8, 13, 21], device=dev, dtype=torch.float32)
        len_bin = (torch.bucketize(sizes.contiguous(), bins, right=True) - 1).clamp_(0, 6)
        len_hist = torch.zeros(B, 7, device=dev).scatter_add_(1, len_bin, (sizes > 0).float())
        len_hist = len_hist / n_seg.clamp_min(1)[:, None]
        ang = torch.remainder(torch.atan2(dy, dx), math.pi).reshape(B, -1)
        ang_hist = bin_hist(ang / math.pi, 7, mask=edges.reshape(B, -1).float(), norm="count")

        # Otsu (64 bins on [0, 1]) and the main object
        hist = bin_hist(gray.reshape(B, -1), 64, norm="count")
        p = hist / hist.sum(1, keepdim=True)
        w = p.cumsum(1)
        centers = (torch.arange(64, device=dev, dtype=torch.float32) + 0.5) / 64
        mu = (p * centers).cumsum(1)
        between = (mu[:, -1:] * w - mu) ** 2 / torch.clamp(w * (1 - w), min=1e-12)
        thr = (between.argmax(1).float() + 1) / 64
        fg = gray > thr[:, None, None]
        border = torch.ones(H, W, dtype=torch.bool, device=dev)
        border[1:-1, 1:-1] = False
        flip = fg[:, border].float().mean(1) > 0.5
        fg = torch.where(flip[:, None, None], ~fg, fg)
        lab = self.components(fg).reshape(B, -1)
        sz = torch.zeros(B, H * W + 1, device=dev).scatter_add_(1, lab, torch.ones_like(lab, dtype=torch.float32))
        sz[:, 0] = -1
        has = sz.amax(1) > 0
        # scipy labels in raster order and np.argmax keeps the first: among equal sizes take the
        # component whose first pixel comes earliest
        flat_idx = torch.arange(H * W, device=dev).expand(B, -1)
        first = torch.full((B, H * W + 1), H * W, device=dev, dtype=torch.int64)
        first.scatter_reduce_(1, lab, flat_idx, reduce="amin")
        key = sz * (H * W + 1) - first.float()
        main = (lab == key.argmax(1, keepdim=True)) & has[:, None]
        main = main.reshape(B, H, W)
        area = main.sum((1, 2)).float()
        mp = F.pad(main[:, None].float(), (1, 1, 1, 1))[:, 0]
        inner = (mp[:, 1:-1, 1:-1] * mp[:, :-2, 1:-1] * mp[:, 2:, 1:-1] * mp[:, 1:-1, :-2] * mp[:, 1:-1, 2:]) > 0
        perim = (main & ~inner).sum((1, 2)).float().clamp_min(1)
        ys = torch.arange(H, device=dev).view(1, H, 1).expand(B, H, W)
        xs = torch.arange(W, device=dev).view(1, 1, W).expand(B, H, W)
        big = torch.full_like(ys, 10 ** 6)
        width = torch.where(main, xs, -big).amax((1, 2)) - torch.where(main, xs, big).amin((1, 2)) + 1
        height = torch.where(main, ys, -big).amax((1, 2)) - torch.where(main, ys, big).amin((1, 2)) + 1
        area_by_perim = torch.where(has, area / perim, torch.zeros_like(area))
        aspect = torch.where(has, width.float() / height.float().clamp_min(1), torch.ones_like(area))

        # hue in float64 like the CPU version: its bin edges sit on exact ratios (e.g. pure hues)
        rgb64 = x.double() / 255.0
        r, g, b = rgb64[..., 0], rgb64[..., 1], rgb64[..., 2]
        mx, mn = rgb64.amax(3), rgb64.amin(3)
        sat = torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-12), torch.zeros_like(mx))
        delta = (mx - mn).clamp_min(1e-12)
        hue = torch.where(mx == r, torch.remainder((g - b) / delta, 6),
                          torch.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) / 6
        hue_hist = bin_hist(hue.reshape(B, -1), 7, mask=(sat > 0.1).reshape(B, -1).double(), norm="count").float()

        std = lambda t: t.reshape(B, -1).std(1, unbiased=False)
        return torch.cat([torch.stack([kp, plain.mean((1, 2)), torch.sqrt((plain ** 2).mean((1, 2))),
                                       perceived.mean((1, 2)), torch.sqrt((perceived ** 2).mean((1, 2))),
                                       std(gray)], 1),
                          len_hist, ang_hist, torch.stack([area_by_perim, aspect], 1), hue_hist], 1)
