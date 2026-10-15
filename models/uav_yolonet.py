import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# CUSTOM YOLO-INSPIRED ARCHITECTURE: UAV-YOLONet
# Built from scratch with:
#  - ConvBlock (Conv2d + BatchNorm2d + SiLU)
#  - CSP (Cross Stage Partial) Darknet residual blocks
#  - Dilated Residual Atrous Convolutions for small aerial targets
#  - SPPF (Spatial Pyramid Pooling - Fast) multi-scale context
#  - Squeeze-and-Excitation (SE) Channel Attention
#  - Multi-Scale Spatial-Semantic Feature Aggregator Neck
# =====================================================================

class ConvBlock(nn.Module):
    """Standard YOLO CBS (Convolution + BatchNorm + SiLU) block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: Global Average Pooling
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y))))
        return x * y.view(b, c, 1, 1)


class Bottleneck(nn.Module):
    """Standard YOLO residual bottleneck: 1x1 conv -> 3x3 conv + shortcut."""
    def __init__(self, channels, expansion=0.5):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.cv1 = ConvBlock(channels, hidden_channels, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class DilatedBottleneck(nn.Module):
    """
    Dilated Residual Bottleneck specifically designed for tiny UAV targets.
    Uses dilation rate 2 without downsampling to capture long-range aerial context.
    """
    def __init__(self, channels, expansion=0.5, dilation=2):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.cv1 = ConvBlock(channels, hidden_channels, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(
            hidden_channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        return x + self.se(self.cv2(self.cv1(x)))


class CSPStage(nn.Module):
    """
    Cross Stage Partial (CSP) block: splits features into two paths,
    processes one path through residual bottlenecks, and concatenates.
    """
    def __init__(self, in_channels, out_channels, num_blocks=1, use_dilation=False):
        super().__init__()
        mid_channels = out_channels // 2
        self.cv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        
        blocks = []
        for _ in range(num_blocks):
            if use_dilation:
                blocks.append(DilatedBottleneck(mid_channels, expansion=1.0, dilation=2))
            else:
                blocks.append(Bottleneck(mid_channels, expansion=1.0))
        self.blocks = nn.Sequential(*blocks)
        
        self.cv3 = ConvBlock(mid_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) module as used in YOLOv5/v8/v11.
    Extracts multi-scale contextual features efficiently.
    """
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBlock(hidden * 4, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class UAV_YOLONet(nn.Module):
    """
    Full Custom Tactical Anti-UAV Neural Network Architecture.
    Combines YOLO CSPDarknet backbone, Dilated Convolutions, SPPF, 
    and Multi-Scale Dual-Stage Pooling Head.
    """
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        
        # 1. Stem: Initial downsampling & feature projection (H/4, W/4)
        self.stem1 = ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.stem2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        
        # 2. Stage 1: Standard CSP stage (H/8, W/8)
        self.down1 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage1 = CSPStage(128, 128, num_blocks=1, use_dilation=False)
        
        # 3. Stage 2: Dilated CSP Stage with SE Attention (H/16, W/16) - ideal for tiny drones
        self.down2 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage2 = CSPStage(256, 256, num_blocks=2, use_dilation=True)
        
        # 4. Stage 3: Deep Semantic CSP Stage + SPPF (H/32, W/32)
        self.down3 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.stage3 = CSPStage(512, 512, num_blocks=1, use_dilation=False)
        self.sppf = SPPF(512, 512, k=5)
        self.se3 = SEBlock(512)
        
        # 5. Multi-Scale Dual Pooling Neck
        # Fuses Stage 2 (fine spatial details) and Stage 3 (contextual SPPF)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Combined channels: 512 (from stage 3) + 256 (from stage 2) = 768
        # Using both AvgPool and MaxPool: 768 * 2 = 1536 features
        total_features = (512 + 256) * 2
        
        # 6. Tactical Precision Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(total_features, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(384, num_classes)
        )
        
        # Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for fast and stable training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Stem
        x = self.stem1(x)
        x = self.stem2(x)
        
        # Stage 1
        x = self.down1(x)
        x = self.stage1(x)
        
        # Stage 2 (Fine features)
        x = self.down2(x)
        feat_fine = self.stage2(x)  # (B, 256, H/16, W/16)
        
        # Stage 3 (Deep features with SPPF)
        x = self.down3(feat_fine)
        x = self.stage3(x)
        feat_deep = self.se3(self.sppf(x))  # (B, 512, H/32, W/32)
        
        # Multi-scale aggregation
        # Pool Stage 2
        f2_avg = self.pool(feat_fine)
        f2_max = self.maxpool(feat_fine)
        
        # Pool Stage 3
        f3_avg = self.pool(feat_deep)
        f3_max = self.maxpool(feat_deep)
        
        # Concatenate multi-scale pooled statistics
        fused = torch.cat([f2_avg, f2_max, f3_avg, f3_max], dim=1)
        
        # Classification prediction
        out = self.classifier(fused)
        return out

if __name__ == "__main__":
    # Smoke test architecture
    dummy_input = torch.randn(2, 3, 160, 160)
    model = UAV_YOLONet(num_classes=4)
    out = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 50)
    print("UAV_YOLONet Initialized Successfully!")
    print(f"Total Trainable Parameters: {num_params:,}")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {out.shape}")
    print("=" * 50)
