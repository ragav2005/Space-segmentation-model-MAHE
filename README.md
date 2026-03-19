# Space-segmentation-model-MAHE

## Problem Statement 2: Real-time Drivable Space Segmentation

### Focus
Semantic Perception & Edge Cases

### Problem Statement
Level 4 vehicles must identify "Free Space"; areas where the car can physically move; regardless of whether lane markings exist. This track focuses on segmenting the road vs. everything else (curbs, construction barriers, sidewalks) in complex urban settings.

### Key Focus Areas
Encoder-Decoder architectures (U-Net, DeepLabV3+), Real-time backbones (MobileNet/EfficientNet), and Loss functions for class imbalance.

### Restrictions
**Use of pre-trained models is strictly prohibited.** Models must be trained from scratch.

### Dataset
https://drive.google.com/drive/folders/1g5KgxG0p8-MmTiXkNtCpoYSIkdBQprEm

### Objectives
- Perform pixel-wise semantic segmentation of the drivable area.
- Ensure high-frequency performance (inference speed is critical).
- Handle "boundary" cases like road-to-grass transitions or water puddles.

### Expected Outcomes & Metrics

**Outcome:**
A real-time inference pipeline that outputs a binary or multi-class mask representing "Drivable" vs. "Non-Drivable."

**mIoU:**
The primary accuracy metric for segmentation.

**FPS & Architecture:**
Inference speed (FPS) is critical. Additionally, your model architecture and training epochs will also be evaluated.
