"""
dynamic_model.py
----------------
Defines DynamicNet: a flexible CNN whose depth and width
are fully controlled by a single list of channel widths.

Example:
    model = DynamicNet([16, 32, 64])  # 3 blocks
    model = DynamicNet([32, 64, 128, 256])  # 4 blocks
"""
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial conv -> Pointwise 1x1 conv -> BN -> ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    """A stage of N depthwise separable blocks. First block downsamples."""
    def __init__(self, in_ch: int, out_ch: int, depth: int):
        super().__init__()
        assert depth >= 1, "Stage depth must be at least 1."
        layers = []
        layers.append(DepthwiseSeparableConv(in_ch, out_ch, stride=2))
        for _ in range(depth - 1):
            layers.append(DepthwiseSeparableConv(out_ch, out_ch, stride=1))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class DynamicNet(nn.Module):
    """
    Architecture controlled by `widths`: a list of output channels per block.
    - Input: 3×96×96 (STL-10)
    - Each block halves spatial resolution (stride=2).
    - Global Average Pooling collapses spatial dims to 1×1.
    - Single Linear head maps last width → 10 classes.
    """
    def __init__(self, widths: list[int], depths: list[int] = None):
        super().__init__()
        assert len(widths) >= 1, "Need at least one width."
        if depths is None:
            depths = [1] * len(widths)
        assert len(widths) == len(depths), "widths and depths must strongly match."

        # A standard initial Conv block mapping 3 -> widths[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_ch = widths[0]
        for i, (out_ch, depth) in enumerate(zip(widths, depths)):
            # Force stride=2 in all stages to gradually compress 96x96 -> ~3x3 or 6x6
            layers.append(Stage(in_ch, out_ch, depth))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ── Utility: compute parameter count and file-size estimate ────────────────────

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module, dtype_bytes: int = 4) -> float:
    """Estimated .pth file size in MB (float32 = 4 bytes per param)."""
    return param_count(model) * dtype_bytes / (1024 ** 2)


# ── Interpolation utility (used by search.py) ───────────────────────────────────

def midpoint(lo: list[int], hi: list[int], step: int = 2) -> list[int]:
    """
    Element-wise midpoint, rounded to nearest `step` (default 2).
    """
    assert len(lo) == len(hi), "lo and hi must have the same length."
    mid = []
    for l, h in zip(lo, hi):
        raw = (l + h) / 2
        rounded = round(raw / step) * step
        mid.append(int(max(step, rounded)))   # ensure at least `step`
    return mid


def midpoint_depth(lo: list[int], hi: list[int]) -> list[int]:
    """Element-wise midpoint for depth (step=1)."""
    assert len(lo) == len(hi), "lo and hi must have the same length."
    return [int(max(1, round((l + h) / 2))) for l, h in zip(lo, hi)]


def configs_converged(lo: list[int], hi: list[int], tol: int = 2) -> bool:
    """True when every dimension of lo and hi is within `tol` of each other."""
    return all(abs(h - l) <= tol for l, h in zip(lo, hi))


if __name__ == "__main__":
    m = DynamicNet([32, 64, 128], [2, 2, 2])
    print(f"Params: {param_count(m):,}  Est. MB (FP32): {size_mb(m):.4f}")
    print(f"Est. MB (FP16): {size_mb(m, dtype_bytes=2):.4f}")
