import torch
import torch.nn as nn
from .backbone import DilatedUAVBackbone
from .neck import DilatedPAFPNNeck
from .head import DecoupledDetectHead


class DilatedUAVNet(nn.Module):
    """
    Complete Dilated-UAVNet One-Stage Detector.
    
    Architecture Architecture:
    [Input 640x640]
          │
    ┌─────▼──────────────────────────────┐
    │ DilatedUAVBackbone (HDC Dilations) │
    │ Outputs: P3 (Stride 8)             │
    │          P4 (Stride 16)            │
    │          P5 (Stride 32 + SPPF)     │
    └─────┬──────────────────────────────┘
          │
    ┌─────▼──────────────────────────────┐
    │ DilatedPAFPNNeck (Path Fusion)     │
    │ Outputs: N3, N4, N5 (128 channels) │
    └─────┬──────────────────────────────┘
          │
    ┌─────▼──────────────────────────────┐
    │ DecoupledDetectHead (Anchor-Free)  │
    │ Cls + Reg + Obj predictions        │
    └────────────────────────────────────┘
    """

    def __init__(self, in_channels=3, num_classes=1, backbone_channels=(32, 64, 128, 256), neck_channels=128, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # 1. Feature Extractor Backbone with Dilated Blocks
        self.backbone = DilatedUAVBackbone(
            in_channels=in_channels,
            channels=backbone_channels
        )

        # 2. Bidirectional Path Aggregation Neck
        neck_in = backbone_channels if len(strides) == 4 else (backbone_channels[1], backbone_channels[2], backbone_channels[3])
        self.neck = DilatedPAFPNNeck(
            in_channels=neck_in,
            out_channels=neck_channels
        )

        # 3. Decoupled Multi-Scale Detection Head
        self.head = DecoupledDetectHead(
            in_channels=neck_channels,
            num_classes=num_classes,
            strides=strides
        )

        # Initialize all layers from scratch
        self._init_weights()

        # Re-apply specialized detection head priors (-4.5 for objectness and class logits)
        # Prevents initial gradient explosion and false positives at epoch 0
        self.head._initialize_biases()

    def _init_weights(self):
        """Kaiming / He normal initialization for zero-pretrained from-scratch training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_raw=False):
        # 1. Extract multi-scale features
        feats = self.backbone(x)

        # 2. Fuse features across scales
        neck_feats = self.neck(*feats)

        # 3. Predict detections
        # return_raw=True forces the raw logits dict list even in eval mode
        # (needed for loss computation during validation)
        if return_raw or self.training:
            return self.head.forward_raw(list(neck_feats))

        return self.head(list(neck_feats))


def build_model(cfg):
    """Factory function to build DilatedUAVNet from config dict."""
    m_cfg = cfg["model"]
    return DilatedUAVNet(
        in_channels=m_cfg.get("in_channels", 3),
        num_classes=cfg["data"].get("num_classes", 1),
        backbone_channels=tuple(m_cfg.get("backbone_channels", [32, 64, 128, 256])),
        neck_channels=m_cfg.get("neck_channels", 128),
        strides=tuple(m_cfg.get("strides", [8, 16, 32]))
    )
