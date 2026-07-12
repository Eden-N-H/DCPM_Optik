# Implementation Plan: Road Quality Pipeline

## Overview

This plan implements the multi-task deep learning pipeline for road quality analysis in dependency order: configuration/utilities first, then data structures, synthetic data generation, CycleGAN, multi-task model components, training infrastructure, and reconstruction. Each stage builds on the previous, with property-based tests validating correctness properties from the design document.

## Tasks

- [x] 1. Set up project structure, configuration, and utilities
  - [x] 1.1 Create project directory structure and package files
    - Create directories: `synth/`, `cyclegan/`, `model/`, `reconstruction/`, `training/`, `utils/`, `tests/properties/`, `tests/unit/`, `tests/integration/`
    - Create `__init__.py` for each package
    - Create `pyproject.toml` or `setup.py` with dependencies (torch, torchvision, numpy, hypothesis, pytest, pyyaml, tensorboard, plyfile, opencv-python)
    - _Requirements: 19.1_

  - [x] 1.2 Implement ConfigLoader with YAML loading, defaults, validation, and CLI overrides
    - Implement `utils/config.py` with `ConfigLoader` class
    - Support nested dot-notation access and CLI override parsing
    - Implement schema validation with clear error messages for invalid/missing params
    - Define documented default values for all optional parameters
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

  - [x] 1.3 Write property tests for configuration (Properties 29, 30, 31, 32)
    - **Property 29: YAML configuration round-trip** — write to YAML and read back produces equivalent dict
    - **Property 30: Configuration defaults for partial configs** — unspecified params get defaults, specified params retained
    - **Property 31: Configuration validation rejects invalid values** — negative LR, empty paths, invalid ranges raise errors
    - **Property 32: CLI dot-notation override application** — overrides set nested keys, other keys unchanged
    - **Validates: Requirements 19.1, 19.2, 19.3, 19.4**

  - [x] 1.4 Implement ExperimentLogger with JSON lines and TensorBoard support
    - Implement `utils/logging.py` with `ExperimentLogger` class
    - Support scalar logging, image logging, diagnostic logging (grad norms, losses)
    - Write structured JSON lines log files and TensorBoard summaries
    - _Requirements: 20.1, 20.2, 20.3, 20.4_

  - [x] 1.5 Implement core data structures and type definitions
    - Implement `utils/data_types.py` with all dataclasses: `DefectSpec`, `DefectInstance`, `CameraConfig`, `RenderOutputs`, `DatasetManifest`, `ModelOutput`, `PointCloudData`, `BEVMap`, `Checkpoint`
    - _Requirements: 1.3, 1.4, 2.5, 13.3, 14.1_

- [x] 2. Implement synthetic data generation
  - [x] 2.1 Implement SceneGenerator for road mesh and defect placement
    - Implement `synth/scene_generator.py` with `SceneGenerator` class
    - Implement `generate_road_mesh()` with configurable lanes (1-4), lane width (3.0-3.75m), road length (50-200m)
    - Implement `place_defects()` with 1-10 defects from {crack, pothole, puddle, patch, manhole} with type-specific dimensions
    - Implement overlap detection and resolution (>25% overlap triggers repositioning)
    - Implement `setup_camera()` for dashcam and drone configurations
    - _Requirements: 1.1, 1.2, 1.6, 1.8_

  - [x] 2.2 Write property test for defect overlap constraint (Property 2)
    - **Property 2: Defect placement respects overlap constraint**
    - Test that for any 1-10 randomly placed defects, after overlap resolution, no pair overlaps >25% of smaller defect area and all remain within road bounds
    - **Validates: Requirements 1.8**

  - [x] 2.3 Implement domain randomization and multi-pass rendering
    - Implement `apply_domain_randomization()` with HDRI selection (20+ maps), vehicle placement (0-5), weather effects (clear, overcast, rain)
    - Implement `render()` to produce 512x512 RGB, 16-bit depth (mm), integer segmentation mask, float32 severity map, and camera params JSON in single pass
    - _Requirements: 1.3, 1.4, 1.5_

  - [x] 2.4 Implement DatasetBuilder for full dataset generation with splits
    - Implement `synth/dataset_builder.py` with `DatasetBuilder` class
    - Generate 16,036 images (±1%) split 80/10/10 train/val/test
    - Ensure 50% ±2% dashcam/drone balance
    - Organize output into `{root}/{split}/{view_type}/{scene_id}/` structure
    - Write dataset manifest JSON with per-sample metadata
    - _Requirements: 1.7, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.5 Write property tests for data generation (Properties 1, 3, 4, 5)
    - **Property 1: Camera parameters serialization round-trip** — serialize intrinsics/extrinsics to JSON and back within 1e-6 tolerance
    - **Property 3: Dataset split and viewpoint balance** — train/val/test within 80/10/10 ±1%, viewpoints within 50% ±2%
    - **Property 4: Depth map normalization round-trip** — normalize to uint16 and back preserves ordering within quantization error
    - **Property 5: Segmentation mask encoding validity** — all pixel values in {0, 1, 2} with integer dtype
    - **Validates: Requirements 1.4, 1.7, 2.2, 2.3, 2.4, 2.5**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement CycleGAN domain adaptation
  - [x] 4.1 Implement ResNetGenerator with 9 residual blocks
    - Implement `cyclegan/generator.py` with `ResNetGenerator` class
    - Input: [B, 4, 256, 256] (3 RGB + 1 defect mask), output: [B, 3, 256, 256]
    - Architecture: 7x7 conv (64 filters) → 2 downsampling convs (128, 256) → 9 residual blocks (256) → 2 upsampling convs (128, 64) → 7x7 conv (3 channels)
    - Instance normalization + ReLU after each conv except final; reflection padding; tanh output
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 4.2 Write property test for generator output (Property 6)
    - **Property 6: Generator output bounded by tanh**
    - For any [B, 4, 256, 256] input with RGB in [-1,1] and mask in {0,1}, output shape is [B, 3, 256, 256] and all values in [-1, 1]
    - **Validates: Requirements 3.4**

  - [x] 4.3 Implement PatchGANDiscriminator
    - Implement `cyclegan/discriminator.py` with `PatchGANDiscriminator` class
    - Input: [B, 3, 256, 256], output: [B, 1, 30, 30]
    - 4 conv layers (kernel 4, stride 2, filters [64, 128, 256, 512]) + 1-channel final conv (stride 1)
    - Instance norm after all layers except first and last; LeakyReLU(0.2)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 4.4 Write property test for discriminator output (Property 7)
    - **Property 7: Discriminator spatial output dimensions**
    - For any [B, 3, 256, 256] input, output is exactly [B, 1, 30, 30]
    - **Validates: Requirements 4.1**

  - [x] 4.5 Implement CycleGANTrainer with all loss components
    - Implement `cyclegan/trainer.py` with `CycleGANTrainer` class
    - LSGAN adversarial loss, cycle consistency (λ=10), identity (λ=0.5), defect preservation (λ=5.0)
    - Adam optimizer (lr=2e-4, β1=0.5, β2=0.999)
    - Linear LR decay after epoch 100 of 200
    - Image history buffer (size 50) for discriminator updates
    - NaN/Inf detection with checkpoint saving
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 4.6 Write property tests for CycleGAN losses and training (Properties 8, 9, 10)
    - **Property 8: CycleGAN loss weighted summation** — total = adversarial + 10×cycle + 0.5×identity + 5×defect
    - **Property 9: Learning rate linear decay schedule** — lr=2e-4 for e<100, lr=2e-4×(200-e)/100 for e≥100
    - **Property 10: Image history buffer capacity** — buffer never exceeds 50, sampling returns only inserted images
    - **Validates: Requirements 5.5, 5.7, 5.8**

- [x] 5. Implement multi-task model encoder and attention modules
  - [x] 5.1 Implement ResNet50DSCEncoder with depthwise separable convolutions
    - Implement `model/encoder.py` with `ResNet50DSCEncoder` class
    - ResNet-50 backbone with standard convs in stages 1-2, DSC in stages 3-4
    - Input: [B, 3, 512, 512], outputs: stage1=[B,256,128,128], stage2=[B,512,64,64], stage3=[B,1024,32,32], stage4=[B,2048,16,16]
    - Support pretrained ImageNet weight loading
    - _Requirements: 6.1, 6.2, 6.4_

  - [x] 5.2 Write property test for encoder output shapes (Property 11)
    - **Property 11: Encoder multi-scale output shapes**
    - For any [B, 3, 512, 512] input, verify all four stage output shapes
    - **Validates: Requirements 6.2**

  - [x] 5.3 Implement ViewEmbedding module
    - Implement `model/view_embedding.py` with `ViewEmbedding` class
    - 32-dim learnable embedding for 2 views, broadcast spatially and concatenated to features
    - Input: [B, 2048, 16, 16] + view_label → output: [B, 2080, 16, 16]
    - _Requirements: 6.3_

  - [x] 5.4 Write property test for view embedding (Property 12)
    - **Property 12: View embedding channel augmentation**
    - For any [B, 2048, H, W] features and view in {0,1}, output is [B, 2080, H, W] and first 2048 channels equal input
    - **Validates: Requirements 6.3**

  - [x] 5.5 Implement SOA (Small-Object Attention) module
    - Implement `model/soa.py` with `SOA` class
    - Channel attention: GAP → FC(reduction=16) → ReLU → FC → Sigmoid
    - Spatial attention: 4 parallel avg pools (k={1,3,5,7}) → concat → 1x1 conv → sigmoid
    - High-pass: subtract 7x7 Gaussian (σ=1.0), scale by α=0.3, add to attention-weighted features
    - Apply in order: channel → spatial → high-pass
    - Preserve input shape
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 5.6 Write property test for SOA (Property 14)
    - **Property 14: SOA shape preservation and attention validity**
    - Output has identical shape to input; channel attention weights in [0,1]; spatial attention in [0,1]
    - **Validates: Requirements 8.5, 8.1, 8.2**

  - [x] 5.7 Implement E-ASPP module
    - Implement `model/easpp.py` with `EASPP` class
    - 4 parallel dilated DSC branches (rates 3, 6, 12, 18) each producing 256 channels + BN + ReLU
    - Global average pooling branch → 1x1 conv (256 ch) → BN + ReLU → bilinear upsample
    - Concatenate all 5 branches (1280 ch) → SOA → 1x1 conv → BN + ReLU → 256 ch output
    - Input: [B, 2080, 16, 16] → output: [B, 256, 16, 16]
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 5.8 Write property test for E-ASPP (Property 13)
    - **Property 13: E-ASPP dimension reduction**
    - For any [B, 2080, 16, 16] input, output is exactly [B, 256, 16, 16]
    - **Validates: Requirements 7.3, 7.4**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement multi-task model decoder and task heads
  - [x] 7.1 Implement LightweightDecoder with skip connections
    - Implement `model/decoder.py` with `LightweightDecoder` class
    - 3 sequential upsample blocks: 1/32→1/16 (concat stage3), 1/16→1/8 (concat stage2), 1/8→1/4 (concat stage1)
    - Each block: 2× 3x3 DSC + BN + ReLU, reducing to 256, 128, 64 channels
    - SOA module after DSC layers in each block
    - Output: [B, 64, 128, 128]
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 7.2 Write property test for decoder output (Property 15)
    - **Property 15: Decoder output dimensions from multi-scale inputs**
    - For E-ASPP output [B, 256, 16, 16] and encoder features at specified resolutions, decoder output is [B, 64, 128, 128]
    - **Validates: Requirements 9.1, 9.4**

  - [x] 7.3 Implement all four task heads (Segmentation, Severity, Depth, Camera)
    - Implement `model/heads.py` with `SegmentationHead`, `SeverityHead`, `DepthHead`, `CameraHead`
    - Segmentation: 2× 3x3 conv (128) + BN + ReLU → 1x1 conv (3 classes) → bilinear upsample to 512x512
    - Severity: 2× 3x3 conv (128) + BN + ReLU → 1x1 conv (1 ch) + sigmoid → upsample to 512x512
    - Depth: 2× 3x3 conv (128) + BN + ReLU → 1x1 conv (1 ch) + sigmoid → upsample to 512x512
    - Camera: GAP → FC(512, ReLU) → FC(256, ReLU) → FC(10): softplus for 4 intrinsics, linear for 6 extrinsics
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 7.4 Write property test for task head outputs (Property 16)
    - **Property 16: Task heads output shapes and value ranges**
    - Segmentation: [B, 3, 512, 512] unbounded; Severity: [B, 1, 512, 512] in [0,1]; Depth: [B, 1, 512, 512] in [0,1]; Camera: intrinsics [B, 4] > 0, extrinsics [B, 6]
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

  - [x] 7.5 Implement GradientReversalLayer and DualDomainAdapter
    - Implement `model/domain_adapter.py` with `GradientReversalLayer`, `DomainDiscriminator`, `DualDomainAdapter`
    - GRL: forward returns x unchanged, backward negates gradient × λ
    - Feature discriminator: 3 conv layers (256, 128, 1), kernel 3, stride 2, LeakyReLU(0.2)
    - Logit discriminator: same architecture on segmentation logits
    - λ_adv = 0.1 for both
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 7.6 Write property test for GRL (Property 17)
    - **Property 17: Gradient Reversal Layer forward-backward semantics**
    - Forward pass returns x unchanged; backward gradient is negated and scaled by -λ
    - **Validates: Requirements 11.2, 11.3**

  - [x] 7.7 Assemble MultiTaskModel combining all components
    - Implement `model/multitask.py` with `MultiTaskModel` class
    - Wire together: encoder → view embedding → E-ASPP → decoder → 4 heads + domain adapter
    - Verify parameter count ≤ 28M
    - Support TorchScript export
    - _Requirements: 15.1, 15.3, 15.4_

- [x] 8. Implement data loading and training infrastructure
  - [x] 8.1 Implement dataset classes and data augmentation pipeline
    - Implement `training/dataset.py` with PyTorch Dataset class
    - Training augmentation: random horizontal flip, random rotation (±10°), random crop (480x480), color jitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
    - Geometric transforms applied consistently to RGB + all label maps
    - Validation: center crop (512x512) + normalization only
    - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    - DataLoader with configurable workers (default 4), pin_memory, prefetch_factor=2
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [x] 8.2 Write property test for augmentation consistency (Property 25)
    - **Property 25: Augmentation geometric consistency**
    - For any image-mask pair, applying training augmentation preserves spatial correspondence between pixels and their labels
    - **Validates: Requirements 16.1**

  - [x] 8.3 Implement MultiTaskTrainer with loss computation and AMP
    - Implement `training/trainer.py` with `MultiTaskTrainer` class
    - Total loss: 1.5×L_seg + 1.0×L_depth + 0.3×L_cam + 0.1×L_adv + 0.1×L_view
    - Segmentation loss: cross-entropy with inverse frequency class weights
    - Depth loss: L1 + SSIM (equal weight)
    - Camera loss: L1 for intrinsics, geodesic distance for rotation
    - Adam optimizer (lr=1e-4, β1=0.9, β2=0.999, weight_decay=1e-5)
    - ReduceLROnPlateau (patience=10, factor=0.5)
    - AMP with gradient scaling, grad clip norm=1.0
    - Early stopping (patience=30), max 200 epochs
    - NaN detection with diagnostic checkpoint
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [x] 8.4 Write property test for multi-task loss (Property 18)
    - **Property 18: Multi-task total loss computation**
    - For any non-negative loss components, total = 1.5×L_seg + 1.0×L_depth + 0.3×L_cam + 0.1×L_adv + 0.1×L_view
    - **Validates: Requirements 12.1**

  - [x] 8.5 Implement checkpointing and reproducibility utilities
    - Implement `training/checkpoint.py` with save/load checkpoint functions
    - Store model weights, optimizer state, scheduler state, epoch, best metric, RNG states (Python, NumPy, PyTorch, CUDA)
    - Implement seed setting function for deterministic training
    - Save best-model checkpoint on new best validation mIoU
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [x] 8.6 Write property tests for checkpointing and determinism (Properties 27, 28)
    - **Property 27: Checkpoint serialization round-trip** — save and load produces identical training state with bit-identical forward passes
    - **Property 28: Seed determinism** — two executions with same seed produce bit-identical results
    - **Validates: Requirements 18.1, 18.4**

  - [x] 8.7 Implement MetricsComputer for all evaluation metrics
    - Implement `training/metrics.py` with `MetricsComputer` class
    - Segmentation: mIoU, per-class IoU, pixel accuracy, mean class accuracy
    - Depth: RMSE, AbsRel, δ<1.25, δ<1.25², δ<1.25³
    - Camera: intrinsic MAE, geodesic rotation error (degrees), translation error (meters)
    - Severity: MAE and Pearson correlation within defect regions only
    - _Requirements: 17.1, 17.2, 17.3, 17.4_

  - [x] 8.8 Write property test for evaluation metrics (Property 26)
    - **Property 26: Evaluation metrics validity and identity**
    - All metrics non-negative; mIoU/accuracy/deltas in [0,1]; when pred=target, mIoU=1.0, RMSE=0.0, all deltas=1.0
    - **Validates: Requirements 17.1, 17.2, 17.3, 17.4**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement 3D reconstruction and BEV mapping
  - [x] 10.1 Implement DepthUnprojector and WorldTransformer
    - Implement `reconstruction/unprojector.py` with `DepthUnprojector` and `WorldTransformer`
    - Pinhole unprojection: X = Z × K_inv × [u, v, 1]^T
    - World transform: apply [R|t] extrinsics to 3D points
    - _Requirements: 13.1, 13.2_

  - [x] 10.2 Write property tests for geometry (Properties 19, 20)
    - **Property 19: Pinhole unprojection round-trip** — project points via K then unproject with same K and depth recovers original points within 1e-5
    - **Property 20: Extrinsics transformation round-trip** — forward + inverse transform recovers points within 1e-5
    - **Validates: Requirements 13.1, 13.2**

  - [x] 10.3 Implement PointCloudAggregator with filtering
    - Implement `reconstruction/aggregator.py` with `PointCloudAggregator`
    - `add_frame()` to concatenate world-space point clouds with per-point attributes (position, class, severity)
    - `filter()` to remove points below confidence threshold (0.5) or outside height range (default -0.5 to 0.5)
    - _Requirements: 13.3, 13.4_

  - [x] 10.4 Write property tests for point cloud operations (Properties 21, 22)
    - **Property 21: Point cloud aggregation preserves all data** — after N frames, total count = sum of per-frame counts, all attributes preserved
    - **Property 22: Point cloud filtering correctness** — all remaining points satisfy criteria, no valid point removed
    - **Validates: Requirements 13.3, 13.4**

  - [x] 10.5 Implement BEVProjector and PLY export
    - Implement `reconstruction/bev.py` with `BEVProjector`
    - Orthographic XY projection at configurable resolution (default 0.02m/pixel)
    - Cell assignment: majority-vote class, maximum severity
    - Export color-coded PNG with class-to-color mapping and intensity encoding severity
    - Implement `export_ply()` in aggregator for PLY file with position, RGB, class, severity per point
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 10.6 Write property tests for BEV and PLY (Properties 23, 24)
    - **Property 23: BEV cell class assignment by majority vote** — cell gets highest-frequency class and maximum severity among contributing points
    - **Property 24: PLY export round-trip** — export and re-read recovers all N points with identical attributes
    - **Validates: Requirements 14.2, 14.3**

- [x] 11. Integration and wiring
  - [x] 11.1 Wire end-to-end training loop with logging and TensorBoard
    - Create `training/train.py` entry point that loads config, builds model, dataloader, trainer
    - Integrate ExperimentLogger for per-epoch metrics (JSON lines), TensorBoard summaries every N steps (default 100)
    - Implement training completion/interruption summary JSON (total time, best metrics, best epochs)
    - _Requirements: 20.1, 20.2, 20.3, 20.4_

  - [x] 11.2 Wire reconstruction pipeline from model predictions to BEV output
    - Create `reconstruction/pipeline.py` that chains: model inference → unproject → transform → aggregate → filter → BEV project → export
    - Handle frame-level error conditions (invalid intrinsics, zero depth, empty point cloud)
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 14.1, 14.2, 14.3, 14.4_

  - [x] 11.3 Create CLI entry point with config loading and command-line overrides
    - Create `main.py` entry point supporting `train`, `evaluate`, `reconstruct` subcommands
    - Support `--config path/to/config.yaml` and dot-notation overrides (e.g., `--training.lr=1e-3`)
    - _Requirements: 19.1, 19.4_

  - [x] 11.4 Write integration tests for full pipeline
    - Test TorchScript export and inference equivalence
    - Test end-to-end training step on minimal synthetic data
    - Test reconstruction pipeline on mock predictions
    - _Requirements: 15.2, 15.3_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 32 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All code uses Python with PyTorch; property tests use `hypothesis` with `hypothesis[numpy]`
- The synthetic data generator requires Blender Python API (bpy) — ensure Blender is available or mock for tests
- Inference performance (≤28M params, ~56 FPS on V100) is validated by parameter count check and benchmark integration test

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.5"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3"] },
    { "id": 4, "tasks": ["2.4"] },
    { "id": 5, "tasks": ["2.5", "4.1", "5.5"] },
    { "id": 6, "tasks": ["4.2", "4.3", "5.1", "5.3", "10.1"] },
    { "id": 7, "tasks": ["4.4", "4.5", "5.2", "5.4", "5.7", "10.2", "10.3"] },
    { "id": 8, "tasks": ["4.6", "5.6", "5.8", "10.4", "10.5"] },
    { "id": 9, "tasks": ["7.1", "10.6"] },
    { "id": 10, "tasks": ["7.2", "7.3", "7.5"] },
    { "id": 11, "tasks": ["7.4", "7.6", "7.7"] },
    { "id": 12, "tasks": ["8.1", "8.5", "8.7"] },
    { "id": 13, "tasks": ["8.2", "8.3", "8.6", "8.8"] },
    { "id": 14, "tasks": ["8.4", "11.1", "11.2"] },
    { "id": 15, "tasks": ["11.3"] },
    { "id": 16, "tasks": ["11.4"] }
  ]
}
```
