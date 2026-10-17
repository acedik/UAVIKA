import torch
import torch.nn as nn
import torch.nn.functional as F
from .boxes import bbox_ciou


class ComputeLoss:
    """
    Scale-Balanced Multi-Task Loss for DilatedUAVNet:
    1. Box Loss: Complete-IoU (CIoU) Loss on all matched drone cells across the feature pyramid.
    2. Objectness Loss: BCEWithLogitsLoss across all grid cells with positive weighting.
    3. Classification Loss: BCEWithLogitsLoss on matched drone targets.
    """

    def __init__(self, model, num_classes=1, lambda_box=5.0, lambda_obj=1.0, lambda_cls=0.5):
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def __call__(self, predictions, targets, img_size=640):
        """
        predictions: list of dicts from forward pass:
          [{'stride': s, 'reg': (B, 4, H, W), 'obj': (B, 1, H, W), 'cls': (B, C, H, W)}, ...]
        targets: tensor of shape (N_targets, 6) -> [batch_idx, class_id, x1, y1, x2, y2] in pixels
        """
        device = predictions[0]["reg"].device

        # Balance factor for sparse positive drone cells in empty sky
        pos_weight = torch.tensor([5.0], device=device)
        bce_obj = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        bce_cls = nn.BCEWithLogitsLoss(reduction="sum")

        loss_obj = torch.tensor(0.0, device=device)

        all_pred_boxes = []
        all_gt_boxes = []
        all_pred_cls = []
        all_gt_cls = []

        total_positives = 0

        for pred in predictions:
            stride = pred["stride"]
            reg = pred["reg"]       # (B, 4, H, W)
            obj = pred["obj"]       # (B, 1, H, W)
            cls_out = pred["cls"]   # (B, num_classes, H, W)

            B, _, H, W = reg.shape
            obj_target = torch.zeros_like(obj)

            if targets.shape[0] > 0:
                for b in range(B):
                    b_mask = targets[:, 0] == b
                    b_targets = targets[b_mask]

                    if b_targets.shape[0] == 0:
                        continue

                    # Targets: [batch_idx, class_id, x1, y1, x2, y2]
                    gt_cls = b_targets[:, 1].long()
                    gt_boxes = b_targets[:, 2:6]

                    # Scale selection criteria: assign based on target size
                    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                    gt_scale = torch.sqrt(torch.clamp(gt_w * gt_h, min=1.0))

                    # Continuous scale coverage with overlap:
                    # - Stride 4  (P2): targets with scale < 48 px   (micro targets: distant flock birds, tiny drones)
                    # - Stride 8  (P3): targets with scale 16 - 128 px (small targets)
                    # - Stride 16 (P4): targets with scale 48 - 288 px (medium targets)
                    # - Stride 32 (P5): targets with scale >= 128 px (large close-up targets)
                    if stride == 4:
                        scale_mask = gt_scale < 48
                    elif stride == 8:
                        scale_mask = (gt_scale >= 16) & (gt_scale < 128) if any(p["stride"] == 4 for p in predictions) else (gt_scale < 128)
                    elif stride == 16:
                        scale_mask = (gt_scale >= 48) & (gt_scale < 288)
                    else:
                        scale_mask = gt_scale >= 128

                    active_gt_boxes = gt_boxes[scale_mask]
                    active_gt_cls = gt_cls[scale_mask]

                    for t_idx in range(active_gt_boxes.shape[0]):
                        box = active_gt_boxes[t_idx]
                        c_id = active_gt_cls[t_idx]

                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0

                        gx = int(cx.item() // stride)
                        gy = int(cy.item() // stride)

                        # Center sampling radius: for small targets r=1 (center cell),
                        # for larger targets (scale >= 64 px), assign 3x3 positive neighborhood
                        radius = 2 if gt_scale[t_idx] >= 64 else 1
                        for dy_g in range(-radius + 1, radius):
                            for dx_g in range(-radius + 1, radius):
                                ny, nx = gy + dy_g, gx + dx_g
                                if 0 <= nx < W and 0 <= ny < H:
                                    obj_target[b, 0, ny, nx] = 1.0

                                    # Decode prediction at this grid cell
                                    dx = torch.sigmoid(reg[b, 0, ny, nx])
                                    dy = torch.sigmoid(reg[b, 1, ny, nx])
                                    dw = torch.exp(torch.clamp(reg[b, 2, ny, nx], min=-4.0, max=4.0))
                                    dh = torch.exp(torch.clamp(reg[b, 3, ny, nx], min=-4.0, max=4.0))

                                    pred_cx = (nx + dx) * stride
                                    pred_cy = (ny + dy) * stride
                                    pred_w = dw * stride
                                    pred_h = dh * stride

                                    p_box = torch.stack([
                                        pred_cx - pred_w / 2.0,
                                        pred_cy - pred_h / 2.0,
                                        pred_cx + pred_w / 2.0,
                                        pred_cy + pred_h / 2.0
                                    ])

                                    all_pred_boxes.append(p_box)
                                    all_gt_boxes.append(box)

                                    all_pred_cls.append(cls_out[b, :, ny, nx])
                                    one_hot = torch.zeros(self.num_classes, device=device)
                                    if c_id < self.num_classes:
                                        one_hot[c_id] = 1.0
                                    all_gt_cls.append(one_hot)

            # Objectness loss for this stride
            loss_obj += bce_obj(obj, obj_target)

        # Average objectness loss across the 3 strides
        loss_obj = loss_obj / len(predictions)

        # Global Box & Classification Loss normalized by total matched positives
        total_positives = len(all_pred_boxes)
        if total_positives > 0:
            p_boxes = torch.stack(all_pred_boxes)
            g_boxes = torch.stack(all_gt_boxes)
            ciou = bbox_ciou(p_boxes, g_boxes)
            loss_box = (1.0 - ciou).sum() / total_positives

            p_cls = torch.stack(all_pred_cls)
            g_cls = torch.stack(all_gt_cls)
            loss_cls = bce_cls(p_cls, g_cls) / total_positives
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)

        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        return total_loss, {
            "loss_box": (self.lambda_box * loss_box).item(),
            "loss_obj": (self.lambda_obj * loss_obj).item(),
            "loss_cls": (self.lambda_cls * loss_cls).item(),
            "total_loss": total_loss.item(),
            "positives": total_positives
        }
