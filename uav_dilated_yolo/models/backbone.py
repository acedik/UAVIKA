import torch
import torch.nn as nn
from .common import ConvBNAct, HDCDilatedStage, SPPF


class DilatedUAVBackbone(nn.Module):
    """
    Dilated UAV Feature Extractor Backbone.
    
    Combines progressive downsampling with Hybrid Dilated Convolutions (HDC)
    to retain critical spatial features of small aerial objects while maintaining
    a broad context field.
    
    Outputs 3 multi-scale pyramid levels:
    - P3: Stride 8  (e.g., 80x80 for 640x640 input) -> High spatial resolution for tiny drones
    - P4: Stride 16 (e.g., 40x40 for 640x640 input) -> Balanced spatial-semantic features
    - P5: Stride 32 (e.g., 20x20 for 640x640 input) -> High semantic context (SPPF enhanced)
    """

    def __init__(self, in_channels=3, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stem: Initial spatial processing producing P2 at Stride 4 (640 -> 320 -> 160)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1 // 2, kernel_size=3, stride=2),  # Stride 2
            ConvBNAct(c1 // 2, c1, kernel_size=3, stride=2),           # Stride 4
            ConvBNAct(c1, c1, kernel_size=3, stride=1)
        )

        # Stage 1 -> P3 (Stride 8: 160 -> 80)
        self.stage_p3 = HDCDilatedStage(
            in_channels=c1,
            out_channels=c2,
            dilations=(1, 2)
        )

        # Stage 2 -> P4 (Stride 16: 80 -> 40)
        self.stage_p4 = HDCDilatedStage(
            in_channels=c2,
            out_channels=c3,
            dilations=(1, 2, 4)
        )

        # Stage 3 -> P5 (Stride 32: 40 -> 20) with SPPF
        self.stage_p5 = nn.Sequential(
            HDCDilatedStage(
                in_channels=c3,
                out_channels=c4,
                dilations=(1, 2, 4)
            ),
            SPPF(in_channels=c4, out_channels=c4, k=5)
        )

    def forward(self, x):
        p2 = self.stem(x)       # (B, c1, H/4, W/4)  - P2 Micro-Target Stride 4
        p3 = self.stage_p3(p2)  # (B, c2, H/8, W/8)  - P3 Small-Target Stride 8
        p4 = self.stage_p4(p3)  # (B, c3, H/16, W/16) - P4 Medium-Target Stride 16
        p5 = self.stage_p5(p4)  # (B, c4, H/32, W/32) - P5 Large-Target Stride 32
        return p2, p3, p4, p5

