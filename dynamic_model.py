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


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional stride-2 downsampling."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DynamicNet(nn.Module):
    """
    Architecture controlled by `widths`: a list of output channels per block.
    - Input: 3×96×96 (STL-10)
    - Each block halves spatial resolution (stride=2).
    - Global Average Pooling collapses spatial dims to 1×1.
    - Single Linear head maps last width → 10 classes.
    """
    def __init__(self, widths: list[int]):
        super().__init__()
        assert len(widths) >= 1, "Need at least one width."

        layers = []
        in_ch = 3
        for out_ch in widths:
            layers.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], 10)

    def forward(self, x):
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
    Example: midpoint([8,16,32], [16,16,32]) → [12, 16, 32]
    """
    assert len(lo) == len(hi), "lo and hi must have the same number of layers."
    mid = []
    for l, h in zip(lo, hi):
        raw = (l + h) / 2
        rounded = round(raw / step) * step
        mid.append(max(step, rounded))   # ensure at least `step` channels
    return mid


def configs_converged(lo: list[int], hi: list[int], tol: int = 2) -> bool:
    """True when every dimension of lo and hi is within `tol` of each other."""
    return all(abs(h - l) <= tol for l, h in zip(lo, hi))


if __name__ == "__main__":
    # Quick sanity-check on the interpolation
    lo = [8, 16, 32]
    hi = [16, 16, 32]
    mid = midpoint(lo, hi)
    print(f"lo={lo}  hi={hi}  mid={mid}")  # → [12, 16, 32]

    m = DynamicNet(mid)
    print(f"Params: {param_count(m):,}  Est. MB: {size_mb(m):.4f}")
