# Phase 2 Architecture Update: DeepLab-Lite

## 1. Objective

Transition the existing drivable area segmentation model to a heavier, more accurate architecture to achieve 85-90% mIoU while maintaining a real-time inference speed of 60-90 FPS on an RTX 3050 Mobile GPU.

## 2. Core Architecture: DeepLab-Lite

- **Backbone:** `MobileNetV3-Large`
  - _Why:_ Extremely efficient mobile-optimized network using hardware-aware Neural Architecture Search (NAS) and Squeeze-and-Excitation mechanisms. Much higher capacity than ShuffleNetV2 while remaining fast.
- **Decoder:** `DeepLabV3+ (ASPP)`
  - _Why:_ Atrous Spatial Pyramid Pooling (ASPP) uses dilated convolutions to map multiple Field of Views (FOV) simultaneously. Essential for capturing fine edge details (curbs) and large unbroken regions (the whole road) in the same pass.
- **Sensor Modalities:** RGB + LiDAR (Dense Depth & Height).
  - _Note:_ Radar data continues to be excluded due to extreme sparsity and multipath inference noise.

## 3. Configuration & Hyperparameters

- **Input Resolution:** `512 x 512`
  - Increased resolution provides the ASPP module with enough spatial dimensions to accurately preserve boundary masks.
- **Loss Function Strategy:** `CombinedLoss(OHEM + Dice)`
  - **OHEM (Online Hard Example Mining):** Focuses gradients heavily on the hardest 30-50% of pixels (boundary edges/shadows).
  - **Dice Loss:** Optimizes for spatial overlap, aggressively pushing the mIoU metric higher.
- **Batch Size:** Scaled down appropriately (e.g., `B=8` or `B=4`) to accommodate the 512x512 resolution and heavier backbone within the RTX 3050 4GB/6GB VRAM limit.

## 4. Execution Steps

### Step 1: Baseline Establishment (Current Task)

- Evaluate the existing `ShuffleNetV2 + FPN` model on the validation set.
- Capture baseline metrics: **mIoU** and **FPS** (Frames Per Second).

### Step 2: Configuration Update

- Modify `config/config.py` to change `IMG_SIZE = (512, 512)`.
- Adjust `BATCH_SIZE` dynamically to prevent OOM errors.

### Step 3: Model Implementation

- Create `models/mobilenet_seg.py`.
- Implement `MobileNetV3` encoder mapped to a custom `DeepLabV3+ ASPP` decoder that accepts our `[B, 5, H, W]` multi-modal input tensor.

### Step 4: Training & Validation

- Train the new DeepLab-Lite model using the existing PyTorch 2.x `autocast` pipeline.
- Log via TensorBoard and benchmark against the Step 1 baseline to confirm we hit the >85% mIoU / >70 FPS target.
