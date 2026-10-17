import torch
import torch.nn as nn
from .common import ConvBNAct


class DecoupledDetectHead(nn.Module):
    """
    Decoupled Anchor-Free Multi-Scale Detection Head.
    
    Separates the classification task from the bounding box regression task.
    Predicts:
    1. Bounding box coordinates (dx, dy, dw, dh)
    2. Objectness confidence (probability that an object exists in this cell)
    3. Classification logits (drone vs other classes)
    
    Evaluates at 3 distinct strides: [8, 16, 32]
    """

    def __init__(self, in_channels=128, num_classes=1, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # Build decoupled branches for each feature scale
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for _ in strides:
            # Classification branch
            self.cls_convs.append(nn.Sequential(
                ConvBNAct(in_channels, in_channels, kernel_size=3),
                ConvBNAct(in_channels, in_channels, kernel_size=3)
            ))
            self.cls_preds.append(
                nn.Conv2d(in_channels, num_classes, kernel_size=1)
            )

            # Regression & Objectness branch
            self.reg_convs.append(nn.Sequential(
                ConvBNAct(in_channels, in_channels, kernel_size=3),
                ConvBNAct(in_channels, in_channels, kernel_size=3)
            ))
            self.reg_preds.append(
                nn.Conv2d(in_channels, 4, kernel_size=1)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels, 1, kernel_size=1)
            )

        self._initialize_biases()

    def _initialize_biases(self):
        """Initializes classification & objectness biases to prevent early gradient explosion."""
        for obj_conv in self.obj_preds:
            # Set initial objectness prior to ~0.01 (sigmoid(-4.5) ≈ 0.01)
            nn.init.constant_(obj_conv.bias, -4.5)

        for cls_conv in self.cls_preds:
            nn.init.constant_(cls_conv.bias, -4.5)

    def forward(self, feats):
        """
        feats: list of feature maps [N3, N4, N5]
        During training: returns raw logits list for loss calculation
        During inference: returns decoded predictions [B, N, 6] (x1, y1, x2, y2, score, cls)
        """
        raw_outputs = []
        decoded_outputs = []

        for i, (stride, feat) in enumerate(zip(self.strides, feats)):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            cls_out = self.cls_preds[i](cls_feat)  # (B, num_classes, H, W)
            reg_out = self.reg_preds[i](reg_feat)  # (B, 4, H, W)
            obj_out = self.obj_preds[i](reg_feat)  # (B, 1, H, W)

            raw_outputs.append({
                "stride": stride,
                "reg": reg_out,
                "obj": obj_out,
                "cls": cls_out
            })

            if not self.training:
                decoded = self._decode(reg_out, obj_out, cls_out, stride)
                decoded_outputs.append(decoded)

        if self.training:
            return raw_outputs

        # Concatenate all scale predictions: (B, total_anchors, 6)
        return torch.cat(decoded_outputs, dim=1)

    def forward_raw(self, feats):
        """
        Always returns the raw logits dict list (reg/obj/cls per stride),
        regardless of train/eval mode. Used for loss computation during
        validation, where BatchNorm must still use running statistics.
        """
        raw_outputs = []

        for i, (stride, feat) in enumerate(zip(self.strides, feats)):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            raw_outputs.append({
                "stride": stride,
                "reg": self.reg_preds[i](reg_feat),
                "obj": self.obj_preds[i](reg_feat),
                "cls": self.cls_preds[i](cls_feat)
            })

        return raw_outputs

    def _decode(self, reg_out, obj_out, cls_out, stride):
        """Decodes raw grid outputs into pixel bounding boxes (x1, y1, x2, y2)."""
        B, _, H, W = reg_out.shape
        device = reg_out.device

        # Create coordinate grid
        yv, xv = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        grid = torch.stack((xv, yv), dim=0).float()  # (2, H, W)

        # Reg: dx, dy (offset from cell), dw, dh (exp scaled by stride)
        # Using sigmoid for offset center (between 0 and 1 relative to grid cell)
        dx = torch.sigmoid(reg_out[:, 0:1, :, :])
        dy = torch.sigmoid(reg_out[:, 1:2, :, :])
        dw = torch.exp(torch.clamp(reg_out[:, 2:3, :, :], min=-4.0, max=4.0))
        dh = torch.exp(torch.clamp(reg_out[:, 3:4, :, :], min=-4.0, max=4.0))

        cx = (grid[0:1] + dx) * stride
        cy = (grid[1:2] + dy) * stride
        w = dw * stride
        h = dh * stride

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes = torch.cat([x1, y1, x2, y2], dim=1)  # (B, 4, H, W)
        obj_scores = torch.sigmoid(obj_out)          # (B, 1, H, W)
        cls_scores = torch.sigmoid(cls_out)          # (B, num_classes, H, W)

        # Flatten spatial dimensions: (B, H*W, channels)
        boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
        obj_scores = obj_scores.permute(0, 2, 3, 1).reshape(B, -1, 1)
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

        # Combined confidence score = obj_score * cls_score
        scores, class_ids = torch.max(cls_scores, dim=-1, keepdim=True)
        final_scores = obj_scores * scores

        return torch.cat([boxes, final_scores, class_ids.float()], dim=-1)
