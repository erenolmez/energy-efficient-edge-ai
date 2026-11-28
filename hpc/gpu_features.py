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


def np_percentiles(flat, qs):
    """numpy.percentile(..., method='linear') along dim 1, without torch.quantile's host sync."""
    v = flat.sort(1).values
    n = v.shape[1]
    out = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        t = pos - lo
        a, b = v[:, lo], v[:, min(lo + 1, n - 1)]
        out.append(a + (b - a) * t if t < 0.5 else b - (b - a) * (1 - t))
    return out


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
    rgb = x.float() / torch.full((), 255.0, device=dev)  # true division, as numpy (not x * 1/255)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = r * GRAY[0] + g * GRAY[1] + b * GRAY[2]
    flat = gray.reshape(B, -1)
    gh = bin_hist(flat, 32)
    entropy = -torch.where(gh > 0, gh * torch.log2(gh.clamp_min(1e-30)), torch.zeros_like(gh)).sum(1)
    q = np_percentiles(flat, (0.1, 0.9))
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


def np_mod(a, m):
    """numpy.mod for floats (fmod, then shift into the divisor's sign)."""
    r = torch.fmod(a, m)
    return torch.where((r != 0) & ((r < 0) != (m < 0)), r + m, r)


class NpHist:
    """numpy.histogram(a, bins, range=(lo, hi)) counts per image, numpy's exact binning rules
    (float64, true division, edge corrections), optionally restricted to mask."""

    def __init__(self, bins, lo, hi, device):
        self.bins, self.lo = bins, lo
        self.edges = torch.as_tensor(np.linspace(lo, hi, bins + 1), dtype=torch.float64, device=device)
        self.denom = torch.tensor(hi - lo, dtype=torch.float64, device=device)
        self.numer = torch.tensor(float(bins), dtype=torch.float64, device=device)
        self.lo_t = torch.tensor(lo, dtype=torch.float64, device=device)

    def __call__(self, a, mask=None):
        f = (a - self.lo_t) / self.denom * self.numer
        idx = f.to(torch.int64).clamp_(0, self.bins)
        idx = torch.where(idx == self.bins, idx - 1, idx)
        idx = idx - (a < self.edges[idx]).to(torch.int64)
        idx = idx + ((a >= self.edges[idx + 1]) & (idx != self.bins - 1)).to(torch.int64)
        w = torch.ones_like(a) if mask is None else mask.to(a.dtype)
        return torch.zeros(a.shape[0], self.bins, dtype=a.dtype, device=a.device).scatter_add_(1, idx, w)


class Taylor29:
    """Taylor et al. 29 features on (B, 32, 32, 3) uint8, float64 throughout like the CPU version.

    Every division is tensor / tensor (CUDA divides by a Python scalar via its reciprocal, which
    moves values that sit exactly on a bin edge), histograms follow numpy's binning rules, and
    numpy.mod is reproduced with fmod. Connected components use max-label propagation with
    pointer jumping; cc_iters=None iterates to convergence (host sync, for checks), an integer
    runs a fixed count (no sync, CUDA-graph capturable; hpc.fast_feature_routers verifies it
    converges on every fit and score image).
    """

    def __init__(self, device, kp_threshold, cc_iters=None, H=32, W=32):
        self.device, self.kp_threshold, self.cc_iters = device, kp_threshold, cc_iters
        d64 = dict(dtype=torch.float64, device=device)
        self.k = {}
        for s in HESSIAN_SCALES:
            for order in (0, 1, 2):
                w, r = _gauss_weights(s, order)
                self.k[(s, order)] = (torch.tensor(w, **d64), r)
        self.sym = {r: (torch.as_tensor(np.pad(np.arange(H), r, mode="symmetric"), device=device),
                        torch.as_tensor(np.pad(np.arange(W), r, mode="symmetric"), device=device))
                    for r in {r for _, r in self.k.values()}}
        self.c255 = torch.tensor(255.0, **d64)
        self.c2, self.c3, self.c6 = (torch.tensor(v, **d64) for v in (2.0, 3.0, 6.0))
        self.pi = torch.tensor(math.pi, **d64)
        self.tiny = torch.tensor(1e-12, **d64)
        self.hist7 = NpHist(7, 0.0, 1.0, device)
        self.hist_ang = NpHist(7, 0.0, math.pi, device)
        self.hist64 = NpHist(64, 0.0, 1.0, device)
        e = np.linspace(0, 1, 65)
        self.centers = torch.as_tensor((e[:-1] + e[1:]) / 2, **d64)
        self.otsu_edges = torch.as_tensor(e[1:], **d64)
        self.len_bins = torch.as_tensor(np.array([1, 2, 3, 5, 8, 13, 21], np.float64), **d64)
        border = torch.ones(H, W, dtype=torch.bool, device=device)
        border[1:-1, 1:-1] = False
        self.border = border
        self.n_border = torch.tensor(float(border.sum()), **d64)
        self.kp_t = None if kp_threshold is None else torch.tensor(kp_threshold, **d64)

    def _filt(self, img, s, order_y, order_x):
        """scipy gaussian_filter(img, s, order=(order_y, order_x)), mode 'reflect' (symmetric)."""
        ky, ry = self.k[(s, order_y)]
        kx, rx = self.k[(s, order_x)]
        t = img[:, self.sym[ry][0]]
        t = F.conv2d(t[:, None], ky.view(1, 1, -1, 1))[:, 0]
        t = t[:, :, self.sym[rx][1]]
        return F.conv2d(t[:, None], kx.view(1, 1, 1, -1))[:, 0]

    def hessian_peaks(self, gray):
        stack = []
        for s in HESSIAN_SCALES:
            dxx = self._filt(gray, s, 0, 2)
            dyy = self._filt(gray, s, 2, 0)
            dxy = self._filt(gray, s, 1, 1)
            stack.append((dxx * dyy - dxy * dxy) * (s ** 4))
        r = torch.stack(stack, 1)  # B, S, H, W
        mx = F.max_pool3d(F.pad(r[:, None], (1, 1, 1, 1, 1, 1), mode="replicate"), 3, 1)[:, 0]
        peak = (r == mx) & (r > 0)
        peak[:, :, :2], peak[:, :, -2:], peak[:, :, :, :2], peak[:, :, :, -2:] = False, False, False, False
        return r, peak

    def peak_responses(self, x):
        """Per image, Hessian responses at peaks (others -inf), flattened: for fold-wise thresholds."""
        resp, peak = self.hessian_peaks(self._gray(x)[0])
        return torch.where(peak, resp, torch.full_like(resp, -math.inf)).reshape(x.shape[0], -1)

    def components(self, mask, iters=None):
        """8-connected labels (label = 1 + max flat index in the component) and iterations used."""
        B, H, W = mask.shape
        m = mask.reshape(B, -1)
        ids = torch.arange(1, H * W + 1, device=mask.device).expand(B, -1)
        lab = torch.where(m, ids, torch.zeros_like(ids))
        zero = torch.zeros_like(lab)
        n = 0
        while True:
            prop = F.max_pool2d(lab.view(B, 1, H, W).float(), 3, 1, 1).view(B, -1).long()
            new = torch.where(m, prop, zero)
            jump = new.gather(1, (new - 1).clamp_min(0))
            new = torch.where(m, torch.maximum(new, jump), zero)
            n += 1
            if iters is None:
                if torch.equal(new, lab):
                    return new, n
            elif n >= iters:
                return new, n
            lab = new

    def _gray(self, x):
        rgb = x.double() / self.c255
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return GRAY_T[0] * r + GRAY_T[1] * g + GRAY_T[2] * b, rgb

    def __call__(self, x, return_iters=False):
        B, H, W, _ = x.shape
        dev = x.device
        gray, rgb = self._gray(x)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        plain = (r + g + b) / self.c3
        perceived = torch.sqrt(0.241 * (r * r) + 0.691 * (g * g) + 0.068 * (b * b))
        resp, peak = self.hessian_peaks(gray)
        kp = ((resp > self.kp_t) & peak).sum((1, 2, 3)).double() if self.kp_t is not None \
            else torch.zeros(B, dtype=torch.float64, device=dev)

        dy, dx = np_gradient(gray)
        mag = torch.hypot(dx, dy)
        edges = mag > 0.1
        lab, it1 = self.components(edges, self.cc_iters)
        ones = torch.ones_like(lab, dtype=torch.float64)
        sizes = torch.zeros(B, H * W + 1, dtype=torch.float64, device=dev).scatter_add_(1, lab, ones)[:, 1:]
        n_seg = (sizes > 0).sum(1).double()
        len_bin = (torch.bucketize(sizes.contiguous(), self.len_bins, right=True) - 1).clamp_(0, 6)
        len_hist = torch.zeros(B, 7, dtype=torch.float64, device=dev).scatter_add_(1, len_bin, (sizes > 0).double())
        len_hist = len_hist / n_seg.clamp_min(1)[:, None]
        ang = np_mod(torch.atan2(dy, dx), self.pi).reshape(B, -1)
        em = edges.reshape(B, -1)
        ang_hist = self.hist_ang(ang, em) / em.sum(1, keepdim=True).double().clamp_min(1)

        # Otsu (64 bins on [0, 1]) and the main object
        hist = self.hist64(gray.reshape(B, -1))
        p = hist / hist.sum(1, keepdim=True).clamp_min(1)
        w = p.cumsum(1)
        mu = (p * self.centers).cumsum(1)
        between = (mu[:, -1:] * w - mu) ** 2 / torch.maximum(w * (1 - w), self.tiny)
        thr = self.otsu_edges[between.argmax(1)]
        fg = gray > thr[:, None, None]
        flip = (fg & self.border).sum((1, 2)).double() / self.n_border > 0.5  # no boolean indexing (host sync)
        fg = torch.where(flip[:, None, None], ~fg, fg)
        lab, it2 = self.components(fg, self.cc_iters)
        sz = torch.zeros(B, H * W + 1, dtype=torch.float64, device=dev).scatter_add_(1, lab, ones)
        sz[:, 0] = -1
        has = sz.amax(1) > 0
        # scipy labels in raster order and np.argmax keeps the first: among equal sizes take the
        # component whose first pixel comes earliest
        flat_idx = torch.arange(H * W, device=dev).expand(B, -1)
        first = torch.full((B, H * W + 1), H * W, device=dev, dtype=torch.int64)
        first.scatter_reduce_(1, lab, flat_idx, reduce="amin")
        key = sz * (H * W + 1) - first.double()
        main = ((lab == key.argmax(1, keepdim=True)) & has[:, None]).reshape(B, H, W)
        area = main.sum((1, 2)).double()
        mp = F.pad(main[:, None].float(), (1, 1, 1, 1))[:, 0]
        inner = (mp[:, 1:-1, 1:-1] * mp[:, :-2, 1:-1] * mp[:, 2:, 1:-1] * mp[:, 1:-1, :-2] * mp[:, 1:-1, 2:]) > 0
        perim = (main & ~inner).sum((1, 2)).double().clamp_min(1)
        ys = torch.arange(H, device=dev).view(1, H, 1).expand(B, H, W)
        xs = torch.arange(W, device=dev).view(1, 1, W).expand(B, H, W)
        big = torch.full_like(ys, 10 ** 6)
        width = torch.where(main, xs, -big).amax((1, 2)) - torch.where(main, xs, big).amin((1, 2)) + 1
        height = torch.where(main, ys, -big).amax((1, 2)) - torch.where(main, ys, big).amin((1, 2)) + 1
        area_by_perim = torch.where(has, area / perim, torch.zeros_like(area))
        aspect = torch.where(has, width.double() / height.double().clamp_min(1), torch.ones_like(area))

        mx, mn = rgb.amax(3), rgb.amin(3)
        sat = torch.where(mx > 0, (mx - mn) / torch.maximum(mx, self.tiny), torch.zeros_like(mx))
        delta = torch.maximum(mx - mn, self.tiny)
        hue = torch.where(mx == r, np_mod((g - b) / delta, self.c6),
                          torch.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) / self.c6
        sm = (sat > 0.1).reshape(B, -1)
        hue_hist = self.hist7(hue.reshape(B, -1), sm) / sm.sum(1, keepdim=True).double().clamp_min(1)

        std = lambda t: t.reshape(B, -1).std(1, unbiased=False)
        out = torch.cat([torch.stack([kp, plain.mean((1, 2)), torch.sqrt((plain * plain).mean((1, 2))),
                                      perceived.mean((1, 2)), torch.sqrt((perceived * perceived).mean((1, 2))),
                                      std(gray)], 1),
                         len_hist, ang_hist, torch.stack([area_by_perim, aspect], 1), hue_hist], 1)
        return (out, max(it1, it2)) if return_iters else out
