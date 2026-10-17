import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ConvBNAct


class DilatedPAFPNNeck(nn.Module):
    """
    Path Aggregation Feature Pyramid Network (PAFPN).
    
    Fuses high-level semantic abstractions from deep layers with fine-grained
    spatial coordinates from shallow layers via bidirectional pathways:
    1. Top-Down FPN pathway: transfers strong semantics to high-resolution maps.
    2. Bottom-Up PANet pathway: transfers precise localization signals up the pyramid.
    Supports 4-scale pyramid with P2 micro-target head or legacy 3-scale pyramid.
    """

    def __init__(self, in_channels=(32, 64, 128, 256), out_channels=128):
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channels

        if self.num_levels == 4:
            c2, c3, c4, c5 = in_channels
            # Lateral projections
            self.lateral_p5 = ConvBNAct(c5, out_channels, kernel_size=1)
            self.lateral_p4 = ConvBNAct(c4, out_channels, kernel_size=1)
            self.lateral_p3 = ConvBNAct(c3, out_channels, kernel_size=1)
            self.lateral_p2 = ConvBNAct(c2, out_channels, kernel_size=1)

            # Top-down fusion smoothers
            self.smooth_p4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
            self.smooth_p3 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
            self.smooth_p2 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

            # Bottom-up downsamplers & fusers
            self.down_n2 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
            self.smooth_n3 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

            self.down_n3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
            self.smooth_n4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

            self.down_n4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
            self.smooth_n5 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
        else:
            c3, c4, c5 = in_channels
            self.lateral_p5 = ConvBNAct(c5, out_channels, kernel_size=1)
            self.lateral_p4 = ConvBNAct(c4, out_channels, kernel_size=1)
            self.lateral_p3 = ConvBNAct(c3, out_channels, kernel_size=1)

            self.smooth_p4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
            self.smooth_p3 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

            self.down_n3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
            self.smooth_n4 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

            self.down_n4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
            self.smooth_n5 = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)

    def forward(self, *feats):
        if self.num_levels == 4:
            p2, p3, p4, p5 = feats
            lat_p5 = self.lateral_p5(p5)
            lat_p4 = self.lateral_p4(p4)
            lat_p3 = self.lateral_p3(p3)
            lat_p2 = self.lateral_p2(p2)

            # Top-Down Pathway
            p5_up = F.interpolate(lat_p5, size=lat_p4.shape[2:], mode="nearest")
            f4 = self.smooth_p4(torch.cat([lat_p4, p5_up], dim=1))

            f4_up = F.interpolate(f4, size=lat_p3.shape[2:], mode="nearest")
            f3 = self.smooth_p3(torch.cat([lat_p3, f4_up], dim=1))

            f3_up = F.interpolate(f3, size=lat_p2.shape[2:], mode="nearest")
            n2 = self.smooth_p2(torch.cat([lat_p2, f3_up], dim=1))  # Stride 4 (P2)

            # Bottom-Up Pathway
            n2_down = self.down_n2(n2)
            n3 = self.smooth_n3(torch.cat([f3, n2_down], dim=1))    # Stride 8 (P3)

            n3_down = self.down_n3(n3)
            n4 = self.smooth_n4(torch.cat([f4, n3_down], dim=1))    # Stride 16 (P4)

            n4_down = self.down_n4(n4)
            n5 = self.smooth_n5(torch.cat([lat_p5, n4_down], dim=1))# Stride 32 (P5)

            return n2, n3, n4, n5
        else:
            p3, p4, p5 = feats
            lat_p5 = self.lateral_p5(p5)
            lat_p4 = self.lateral_p4(p4)
            lat_p3 = self.lateral_p3(p3)

            p5_up = F.interpolate(lat_p5, size=lat_p4.shape[2:], mode="nearest")
            f4 = self.smooth_p4(torch.cat([lat_p4, p5_up], dim=1))

            f4_up = F.interpolate(f4, size=lat_p3.shape[2:], mode="nearest")
            n3 = self.smooth_p3(torch.cat([lat_p3, f4_up], dim=1))

            n3_down = self.down_n3(n3)
            n4 = self.smooth_n4(torch.cat([f4, n3_down], dim=1))

            n4_down = self.down_n4(n4)
            n5 = self.smooth_n5(torch.cat([lat_p5, n4_down], dim=1))

            return n3, n4, n5
