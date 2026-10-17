import math
import torch
import torchvision


def xywh2xyxy(x):
    """Converts boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh(x):
    """Converts boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # xc
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # yc
    y[..., 2] = x[..., 2] - x[..., 0]        # w
    y[..., 3] = x[..., 3] - x[..., 1]        # h
    return y


def bbox_ciou(box1, box2, eps=1e-7):
    """
    Computes Complete IoU (CIoU) between box1 and box2.
    Inputs are expected in [x1, y1, x2, y2] format.
    Returns: CIoU tensor (values in [-1, 1], where 1 is perfect overlap).
    """
    # Coordinates of intersection
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    # Areas
    b1_w = (b1_x2 - b1_x1).clamp(min=eps)
    b1_h = (b1_y2 - b1_y1).clamp(min=eps)
    b2_w = (b2_x2 - b2_x1).clamp(min=eps)
    b2_h = (b2_y2 - b2_y1).clamp(min=eps)
    union_area = b1_w * b1_h + b2_w * b2_h - inter_area + eps

    iou = inter_area / union_area

    # Center distance squared
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio consistency term
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(b2_w / (b2_h + eps)) - torch.atan(b1_w / (b1_h + eps)), 2
    )

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return torch.clamp(ciou, min=-1.0, max=1.0)


def non_max_suppression(predictions, conf_threshold=0.30, iou_threshold=0.45, max_det=50):
    """
    Performs Non-Maximum Suppression (NMS) on batch predictions.
    Now class-aware: suppresses per-class to prevent inner bounding boxes.
    Input: [Batch, N_proposals, 6] -> (x1, y1, x2, y2, score, class_id)
    Returns: list of detections per image, each [Num_kept, 6]
    """
    output = [torch.zeros((0, 6), device=predictions.device) for _ in range(predictions.shape[0])]

    for img_idx, pred in enumerate(predictions):
        # Filter by confidence threshold
        mask = pred[:, 4] >= conf_threshold
        pred_cand = pred[mask]

        if pred_cand.shape[0] == 0:
            continue

        # Group by class ID (column 5) for class-aware NMS
        classes = pred_cand[:, 5].unique()
        keep_per_class = []

        for cls in classes:
            class_mask = pred_cand[:, 5] == cls
            class_pred = pred_cand[class_mask]

            # Sort by confidence descending within this class
            scores = class_pred[:, 4]
            order = scores.argsort(descending=True)
            class_pred = class_pred[order]

            # Use torchvision NMS within this class only
            boxes = class_pred[:, :4]
            scores = class_pred[:, 4]
            keep = torchvision.ops.nms(boxes, scores, iou_threshold)
            keep = keep[:max_det // max(1, len(classes))]  # allocate per-class budget
            keep_per_class.append(class_pred[keep])

        if keep_per_class:
            output[img_idx] = torch.cat(keep_per_class)

    return output


def remove_inner_boxes(detections, iou_threshold=0.5):
    """
    Suppresses duplicate / sub-component detections based on confidence descending.
    Never suppresses smaller legitimate targets (like flock birds) simply because a larger
    box exists in the scene.
    
    Input: detections tensor [N, 6] where columns are (x1, y1, x2, y2, conf, class_id)
    Returns: filtered detections sorted by confidence descending.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence (5th column) descending
    scores = detections[:, 4]
    order = scores.argsort(descending=True)
    dets = detections[order]

    keep = []
    suppressed = set()

    for i in range(len(dets)):
        if i in suppressed:
            continue
        keep.append(dets[i])

        x1_a, y1_a, x2_a, y2_a = dets[i, :4]
        cls_a = int(dets[i, 5].item())
        area_a = max(1.0, (x2_a - x1_a).item() * (y2_a - y1_a).item())

        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue

            x1_b, y1_b, x2_b, y2_b = dets[j, :4]
            cls_b = int(dets[j, 5].item())
            area_b = max(1.0, (x2_b - x1_b).item() * (y2_b - y1_b).item())

            # Intersection
            ix1 = max(x1_a.item(), x1_b.item())
            iy1 = max(y1_a.item(), y1_b.item())
            ix2 = min(x2_a.item(), x2_b.item())
            iy2 = min(y2_a.item(), y2_b.item())

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / (area_a + area_b - inter)
                containment_b = inter / area_b  # fraction of candidate B inside higher-conf A

                # If B is heavily contained in higher-conf A (>80%) of same class or high IoU
                if iou >= iou_threshold or (containment_b >= 0.80 and cls_a == cls_b):
                    suppressed.add(j)

    return torch.stack(keep) if keep else torch.zeros((0, 6), device=detections.device)