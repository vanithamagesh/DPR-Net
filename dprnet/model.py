"""DPR-Net: dual-domain encoder, pinwheel-edge attention, selective fusion
bottleneck, gated decoder, line branch and uncertainty-driven refinement.

Setting ``freq=False, pea=False, sfb=False`` gives the baseline network used
in the paper, which shares every other component with DPR-Net.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _zero(module: nn.Module) -> nn.Module:
    """Initialise the last layer of a correction branch at zero."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    """Two 3x3 convolutions with batch normalisation and GELU, plus a 1x1 shortcut."""

    def __init__(self, ci: int, co: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
        )
        self.short = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()

    def forward(self, x):
        return F.gelu(self.body(x) + self.short(x))


class FrequencyPath(nn.Module):
    """Corrects the log-amplitude and phase of the 2-D Fourier spectrum (Eqs. 1-2)."""

    def __init__(self, ci: int, co: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(ci, co, 1), nn.BatchNorm2d(co))
        h = max(co // 2, 1)
        self.phi_a = nn.Sequential(nn.Conv2d(co, h, 1, bias=False), nn.GELU(),
                                   _zero(nn.Conv2d(h, co, 1, bias=False)))
        self.phi_p = nn.Sequential(nn.Conv2d(2 * co, co, 1, bias=False), nn.GELU(),
                                   _zero(nn.Conv2d(co, 2 * co, 1, bias=False)))
        self.post = nn.Conv2d(co, co, 1)

    def forward(self, x):
        u = self.proj(x)
        h, w = u.shape[-2:]
        spec = torch.fft.rfft2(u.float(), norm="ortho")
        mag = torch.log1p(spec.abs())
        ang = torch.angle(spec)
        cs = torch.cat([torch.cos(ang), torch.sin(ang)], 1)
        mag = mag + self.phi_a(mag)
        cs = cs + self.phi_p(cs)
        c, s = cs.chunk(2, 1)
        norm = torch.sqrt(c * c + s * s).clamp_min(1e-6)
        amp = torch.expm1(mag)
        spec = torch.complex(amp * c / norm, amp * s / norm)
        out = torch.fft.irfft2(spec, s=(h, w), norm="ortho").to(u.dtype)
        return self.post(out)


class DualDomainBlock(nn.Module):
    """Spatial residual block A with an optional frequency path, Y = A + W_f[A, F^-1(F')]."""

    def __init__(self, ci: int, co: int, freq: bool = True):
        super().__init__()
        self.spatial = ResBlock(ci, co)
        self.freq = FrequencyPath(ci, co) if freq else None
        if freq:
            self.fuse = nn.Sequential(nn.Conv2d(2 * co, co, 1), nn.BatchNorm2d(co))
            nn.init.zeros_(self.fuse[1].weight)  # block starts as a plain residual block

    def forward(self, x):
        a = self.spatial(x)
        if self.freq is None:
            return a
        return a + self.fuse(torch.cat([a, self.freq(x)], 1))


class PinwheelConv(nn.Module):
    """Four asymmetrically padded 1xk / kx1 convolutions merged by a 2x2 convolution."""

    def __init__(self, c: int, k: int = 5):
        super().__init__()
        o = max(c // 4, 1)
        # F.pad order is (left, right, top, bottom); every branch returns (H+1) x (W+1)
        self.pads = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.branches = nn.ModuleList([
            nn.Conv2d(c, o, (1, k), bias=False), nn.Conv2d(c, o, (1, k), bias=False),
            nn.Conv2d(c, o, (k, 1), bias=False), nn.Conv2d(c, o, (k, 1), bias=False)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(o) for _ in range(4)])
        self.merge = nn.Sequential(nn.Conv2d(4 * o, c, 2, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        ys = [F.silu(bn(conv(F.pad(x, pad)))) for pad, conv, bn in zip(self.pads, self.branches, self.bn)]
        return self.merge(torch.cat(ys, 1))  # 2x2 valid conv brings the size back to H x W


class SobelMagnitude(nn.Module):
    """Magnitude of depthwise Sobel responses at 0, 45, 90 and 135 degrees (fixed weights)."""

    def __init__(self, c: int):
        super().__init__()
        k0 = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        k45 = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
        kernels = torch.stack([k0, k45, k0.t(), k45.flip(1)])  # 0, 45, 90, 135
        self.register_buffer("weight", kernels[:, None].repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        r = F.conv2d(x, self.weight, padding=1, groups=self.c)
        r = r.view(x.shape[0], self.c, 4, *x.shape[-2:])
        return torch.sqrt((r * r).sum(2) + 1e-6)


class SqueezeExcite(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(h, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class PinwheelEdgeAttention(nn.Module):
    """G = sigma(BN(W1 relu(PW(S)))) + sigma(BN(W2 relu(E(S)))), S' = SE(S*G) + S*G (Eqs. 3-4)."""

    def __init__(self, c: int):
        super().__init__()
        self.pw = PinwheelConv(c)
        self.edge = SobelMagnitude(c)
        self.w1 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))
        self.w2 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))
        self.se = SqueezeExcite(c)

    def forward(self, s):
        g = torch.sigmoid(self.w1(F.relu(self.pw(s)))) + torch.sigmoid(self.w2(F.relu(self.edge(s))))
        sg = s * g
        return self.se(sg) + sg


class ChannelAttention(nn.Module):
    """Channel self-attention with unit-norm queries/keys, learned temperature and top-k masking (Eq. 6)."""

    def __init__(self, c: int, heads: int = 4, keep: float = 0.8):
        super().__init__()
        self.heads, self.keep = heads, keep
        self.norm = nn.GroupNorm(1, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1, bias=False)
        self.dw = nn.Conv2d(3 * c, 3 * c, 3, padding=1, groups=3 * c, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, z):
        b, c, h, w = z.shape
        q, k, v = self.dw(self.qkv(self.norm(z))).chunk(3, 1)
        shape = (b, self.heads, c // self.heads, h * w)
        q, k, v = (t.reshape(shape) for t in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        n = attn.shape[-1]
        kth = max(1, int(round(self.keep * n)))
        thresh = attn.topk(kth, dim=-1).values[..., -1:]
        attn = attn.masked_fill(attn < thresh, float("-inf")).softmax(-1)
        return z + self.out((attn @ v).reshape(b, c, h, w))


class SelectiveFusionBlock(nn.Module):
    """Two depthwise branches weighted by spatial and channel selection (Eq. 5), then channel attention."""

    def __init__(self, c: int, layers: int = 3, heads: int = 4, r: int = 8):
        super().__init__()
        self.u1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.Conv2d(c, c, 1))
        self.u2 = nn.Sequential(nn.Conv2d(c, c, 5, padding=4, dilation=2, groups=c, bias=False), nn.Conv2d(c, c, 1))
        self.spatial = nn.Conv2d(4, 2, 7, padding=3)
        self.fc = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True), nn.Conv2d(c // r, 2 * c, 1))
        self.attn = nn.Sequential(*[ChannelAttention(c, heads) for _ in range(layers)])

    def forward(self, z):
        u1, u2 = self.u1(z), self.u2(z)
        stats = torch.cat([u1.mean(1, True), u1.amax(1, True), u2.mean(1, True), u2.amax(1, True)], 1)
        s = torch.sigmoid(self.spatial(stats))
        cw = self.fc(F.adaptive_avg_pool2d(u1 + u2, 1)).view(z.shape[0], 2, z.shape[1], 1, 1).softmax(1)
        z = z * torch.sigmoid(s[:, :1] * cw[:, 0] * u1 + s[:, 1:] * cw[:, 1] * u2)
        return self.attn(z)


class GatedDecoder(nn.Module):
    """Bilinear up-sampling and a skip gate bounded to [0.5, 1] (Eq. 7)."""

    def __init__(self, ci: int, cs: int, co: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(ci + cs, cs, 1), nn.GELU(), nn.Conv2d(cs, 1, 1), nn.Sigmoid())
        self.block = ResBlock(ci + cs, co)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x, skip], 1))
        return self.block(torch.cat([x, skip * (0.5 + 0.5 * g)], 1))


class Refinement(nn.Module):
    """Uncertainty-driven refinement unit R with a zero-initialised output (Eq. 8)."""

    def __init__(self, width: int = 16, in_ch: int = 4):
        super().__init__()
        self.encode = ResBlock(in_ch + 2, width)
        self.context = ResBlock(width, 2 * width)
        self.decode = ResBlock(3 * width, width)
        self.delta = _zero(nn.Conv2d(width, 1, 1))

    def forward(self, image, logits):
        p = logits.sigmoid()
        local = self.encode(torch.cat([image, p, 4 * p * (1 - p)], 1))
        ctx = F.interpolate(self.context(F.avg_pool2d(local, 2)), size=local.shape[-2:],
                            mode="bilinear", align_corners=False)
        return logits + self.delta(self.decode(torch.cat([local, ctx], 1)))


class DPRNet(nn.Module):
    def __init__(self, width: int = 16, freq: bool = True, pea: bool = True, sfb: bool = True, passes: int = 2):
        super().__init__()
        c = width
        self.passes = passes
        self.enc = nn.ModuleList([DualDomainBlock(4, c, freq), DualDomainBlock(c, 2 * c, freq),
                                  DualDomainBlock(2 * c, 4 * c, freq), DualDomainBlock(4 * c, 8 * c, freq)])
        self.pea = nn.ModuleList([PinwheelEdgeAttention(k) for k in (c, 2 * c, 4 * c)]) if pea else None
        self.context = nn.Sequential(nn.Conv2d(8 * c, 8 * c, 7, padding=3, groups=8 * c, bias=False),
                                     nn.BatchNorm2d(8 * c), nn.GELU(), nn.Conv2d(8 * c, 8 * c, 1))
        self.sfb = SelectiveFusionBlock(8 * c) if sfb else None
        self.dec = nn.ModuleList([GatedDecoder(8 * c, 4 * c, 4 * c), GatedDecoder(4 * c, 2 * c, 2 * c),
                                  GatedDecoder(2 * c, c, c)])
        self.line_h = nn.Conv2d(c, c, (1, 9), padding=(0, 4), groups=c)
        self.line_v = nn.Conv2d(c, c, (9, 1), padding=(4, 0), groups=c)
        self.line = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU())
        self.fusion = ResBlock(2 * c, c)
        self.head = nn.Conv2d(c, 1, 1)
        self.aux_head = nn.Conv2d(2 * c, 1, 1)
        self.refine = Refinement(c)

    def forward(self, x):
        h0, w0 = x.shape[-2:]
        x = F.pad(x, (0, (-w0) % 8, 0, (-h0) % 8), mode="reflect")
        s0 = self.enc[0](x)
        s1 = self.enc[1](F.avg_pool2d(s0, 2))
        s2 = self.enc[2](F.avg_pool2d(s1, 2))
        z = self.enc[3](F.avg_pool2d(s2, 2))
        z = F.gelu(z + self.context(z))
        if self.sfb is not None:
            z = self.sfb(z)
        skips = [s0, s1, s2]
        if self.pea is not None:
            skips = [m(s) for m, s in zip(self.pea, skips)]
        d2 = self.dec[0](z, skips[2])
        d1 = self.dec[1](d2, skips[1])
        d0 = self.dec[2](d1, skips[0])
        line = self.line(torch.cat([self.line_h(s0), self.line_v(s0)], 1))
        z0 = self.head(self.fusion(torch.cat([d0, line], 1)))
        logits = [z0]
        for _ in range(self.passes):
            logits.append(self.refine(x, logits[-1]))
        crop = (..., slice(0, h0), slice(0, w0))
        aux = F.interpolate(self.aux_head(d1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits[-1][crop], "initial_logits": z0[crop], "aux_logits": aux[crop]}


def build(name: str = "dprnet") -> DPRNet:
    if name == "dprnet":
        return DPRNet()
    if name == "baseline":
        return DPRNet(freq=False, pea=False, sfb=False)
    raise ValueError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
