# 🛰️ Dilated-UAVNet: Custom Anti-UAV One-Stage Object Detector

**Dilated-UAVNet** is a dedicated aerial target detection framework built **100% from scratch in PyTorch** (0% pre-trained weights). It is specifically engineered to detect tiny drones in multi-spectral (visible & infrared/thermal) video feeds without losing fine spatial resolution.

---

## 🎯 Key Architectural Innovations

### 1. Hybrid Dilated Convolutions (HDC)
Standard CNNs aggressively downsample inputs via pooling layers ($640 \to 320 \to 160 \to 80 \to 40$). A distant drone ($15 \times 15$ pixels) completely vanishes before reaching the classification layers.
* **Our Solution**: Dilated (Atrous) residual blocks with rates $d \in \{1, 2, 4\}$ expand the receptive field exponentially while preserving high spatial coordinate precision and eliminating the gridding artifact.

### 2. Path Aggregation Feature Pyramid (PAFPN Neck)
* **Top-Down FPN Pathway**: Transfers strong semantic context from deep layers down to high-resolution layers.
* **Bottom-Up PANet Pathway**: Propagates crisp spatial localization cues up the pyramid, ensuring drones are accurately bounded across varying flight altitudes.

### 3. Decoupled Anchor-Free Detection Head
* Separates bounding box coordinate regression from classification logits.
* Uses scale-invariant Complete-IoU (CIoU) loss + Binary Cross-Entropy (BCE) with positive objectness gating.

---

## 📁 Project Structure

```
D:\UAVIkA\uav_dilated_yolo\
├── configs\
│   └── config.yaml          # Hyperparameters, paths, resolutions, batch size
├── models\
│   ├── common.py            # ConvBNAct, DilatedConv, DilatedResBlock, HDCDilatedStage, SPPF
│   ├── backbone.py          # DilatedUAVBackbone (Outputs multi-scale P3, P4, P5)
│   ├── neck.py              # DilatedPAFPNNeck (Bidirectional feature fusion)
│   ├── head.py              # DecoupledDetectHead (Anchor-free multi-scale heads)
│   └── detector.py          # DilatedUAVNet (Complete end-to-end model assembler)
├── utils\
│   ├── boxes.py             # Bounding box math, CIoU loss, torchvision NMS
│   ├── loss.py              # Multi-task loss (CIoU + BCE objectness + BCE class)
│   ├── dataset.py           # UAVYoloDataset loader with augmentations & collate_fn
│   └── visualizer.py        # Tactical HUD visualizer with glowing target brackets
├── train.py                 # Mixed-precision (AMP FP16) training pipeline from scratch
├── infer.py                 # Real-time inference engine (webcam, video, batch images)
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Train From Scratch (Zero Pretrained Weights)
To start training on your processed dataset:
```bash
python train.py --epochs 80 --batch-size 4
```
* **Hardware optimization**: Mixed Precision (AMP FP16) is enabled by default to fit comfortably within the **4GB VRAM** of your NVIDIA GeForce RTX 2050 while accelerating forward and backward passes.
* Checkpoints are automatically saved to `weights/best_uav_dilated.pth` and `weights/last_uav_dilated.pth`.

To resume interrupted training:
```bash
python train.py --resume weights/last_uav_dilated.pth
```

### 2. Inference & Tactical Air Defense HUD

#### Run on Test Images:
```bash
python infer.py --source D:\UAVIkA\my_test_images --conf 0.30
```

#### Run on Video Files:
```bash
python infer.py --source D:\path\to\drone_video.mp4 --view
```

#### Run Live Camera / RTSP Feed:
```bash
python infer.py --source 0
```

---

## ⚙️ Configuration (`configs/config.yaml`)
You can adjust image resolution, channels, loss balance, and thresholds directly inside [`configs/config.yaml`](configs/config.yaml):
* `input_size`: Default `640` (can be set to `512` for even faster training).
* `batch_size`: Default `4` (optimal for 4GB RTX 2050).
* `conf_threshold`: Default `0.30`.
* `iou_threshold`: Default `0.45`.
