import torch
import torch.nn as nn

# [TEMPLATE] Set this to the winning configuration found by the Binary Architecture Search!
WINNING_WIDTHS = [32, 64, 128]
WINNING_DEPTHS = [2, 2, 2]

class DepthwiseSeparableConv(nn.Module):
    """MobileNet/ResNet-style Depthwise Separable Block with Identity Skip."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        
        # Depthwise
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_relu = nn.ReLU(inplace=True)
        
        # Pointwise
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.dw_conv(x)
        out = self.dw_bn(out)
        out = self.dw_relu(out)
        
        out = self.pw_conv(out)
        out = self.pw_bn(out)
        
        if self.use_res_connect:
            out += identity
            
        return self.out_relu(out)


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
    Dynamically constructs a Convolutional Neural Network.
    """
    def __init__(self, widths=WINNING_WIDTHS, depths=WINNING_DEPTHS):
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
