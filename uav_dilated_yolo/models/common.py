import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Calculates padding automatically to maintain spatial dimensions."""
    if d > 1:
        # For dilated conv: effective kernel size = d * (k - 1) + 1
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation (SiLU / Swish)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1):
        super().__init__()
        p = autopad(kernel_size, padding, dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=p,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DilatedConv(nn.Module):
    """
    Atrous (Dilated) Convolution Block.
    Expands the receptive field exponentially without downsampling.
    Crucial for capturing the contextual signature of tiny drones in sky backgrounds.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2):
        super().__init__()
        self.block = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation
        )

    def forward(self, x):
        return self.block(x)


class DilatedResBlock(nn.Module):
    """
    Dilated Residual Block with skip connection.
    Preserves fine spatial resolution while learning multi-scale receptive field features.
    """

    def __init__(self, channels, dilation=2, expansion=0.5):
        super().__init__()
        mid_channels = int(channels * expansion)
        self.cv1 = ConvBNAct(channels, mid_channels, kernel_size=1)
        self.cv2 = ConvBNAct(mid_channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class HDCDilatedStage(nn.Module):
    """
    Hybrid Dilated Convolution (HDC) Stage.
    Chains residual blocks with varying dilation rates (e.g. 1, 2, 4)
    to completely eliminate gridding artifacts and capture multi-scale aerial targets.
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4)):
        super().__init__()
        # Downsample transition (stride 2)
        self.downsample = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2)

        # Sequence of dilated residual blocks with progressive dilation rates
        self.blocks = nn.ModuleList([
            DilatedResBlock(out_channels, dilation=d) for d in dilations
        ])

        # Cross-stage fusion
        self.fuse = ConvBNAct(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return self.fuse(x)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) to aggregate multi-scale context."""

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        mid_channels = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, mid_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNAct(mid_channels * 4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
