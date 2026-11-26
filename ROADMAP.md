# Roadmap

Milestones build on each other. Each one ends with an acceptance check.

## M0 · Task definition, data and evaluation

- [x] Data contract: N cameras × T past frames, intrinsics, camera-to-ego extrinsics, ego poses (`orv/data/sample.py`)
- [x] Virtual camera family: 3–15 m up, 5–20 m back, 15–60° pitch, FOV range, plus top-down (`orv/geometry.py`)
- [ ] Dataset adapters producing `SurroundSample`, each with a per-dataset config (camera set, history length, resolution) and a LiDAR-to-camera reprojection check:
  - [ ] nuScenes
  - [ ] nuPlan / OpenScene (NAVSIM)
  - [ ] Waymo Open Perception
  - [ ] Argoverse 2 Sensor
  - [ ] DDAD
- [ ] Evaluation scripts for all [metrics](#metrics) on real and virtual views, per dataset and leave-one-dataset-out

Acceptance: every adapter passes the reprojection check, and the evaluation runs end to end for a baseline on real and virtual views.

## M1 · Pseudo ground-truth factory

- [ ] Per-clip DriveStudio / OmniRe reconstruction: Waymo first, then nuScenes, nuPlan and Argoverse 2 (supported by DriveStudio) and DDAD (needs a converter)
- [ ] Render random chase views along the ego trajectory: RGB, depth, semantics, instances and per-pixel confidence (number of observing views, alpha)
- [ ] Remove reconstruction artifacts with Difix3D+-style refinement
- [ ] Filter clips by held-out-frame PSNR
- [ ] Paired CARLA data: on-board cameras plus a chase camera

Acceptance: pseudo ground truth meets the threshold on held-out real views; elevated views have no large holes.

## M2 · Geometry-first renderer v0

- [ ] Encoder → temporal fusion → Gaussian head (color, latent features, velocity)
- [ ] Real cameras: photometric and LiDAR depth losses
- [ ] Virtual cameras: pseudo ground-truth supervision, weighted by confidence
- [ ] Multi-dataset training: per-dataset batches interleaved across all adapters

Acceptance: real-view reconstruction reaches the target; virtual-view geometry is correct (blur is acceptable); one model trained on the dataset mix is no worse than per-dataset models.

## M3 · Generative refinement v1

- [ ] Decoder input: low-resolution features, depth and alpha in the virtual camera, plus the previous output warped by pose
- [ ] Losses: L1 + LPIPS on confident pixels, adversarial loss (DINO-feature discriminator) on the whole frame, temporal warp consistency
- [ ] Fine-tune a video-diffusion teacher (Cosmos-Transfer or Wan) on render conditions; DMD-style distillation into the one-step decoder
- [ ] Self-Forcing-style autoregressive rollout training against exposure bias

Acceptance: faithfulness no worse than v0, clearly lower FID / FVD, no visible flicker.

## M4 · Real time

- [ ] Features at ⅛–¼ resolution; INT8 / FP8 with TensorRT
- [ ] Cache static Gaussians in the world frame
- [ ] Compare projection operators: Gaussian rasterization, precomputed BEV / voxel ray gather, LVSM-style cross-attention

Acceptance: ≥ 20 FPS on one GPU, with per-frame FLOPs and a latency breakdown.

## M5 · Planning, intent and interaction

- [ ] Share the backbone with DiffusionDrive / TransFuser; multi-task training on NAVSIM
- [ ] Intent: project the planned trajectory and maneuver points into the virtual camera and draw depth-tested chevrons and parking pins; later, feed intent tokens to the renderer
- [ ] Tap-to-go: (u, v) → depth unprojection → snap to drivable area → goal token → GoalFlow-style planner; train with hindsight goal relabeling and learn to reject unreachable goals

Acceptance: planning score (PDMS) does not drop; report tap-to-go success rate, goal error and collision / off-road rate.

## M6 · Ablations

- [ ] One rendering head vs. explicit heads (Sparse4D / StreamPETR + MapTR + FlashOcc + a simple renderer): FLOPs and latency
- [ ] Does a rendering loss improve planning (NAVSIM PDMS)?
- [ ] Video-pretrained vs. image-pretrained encoders
- [ ] Correlation between objects missing from the render and planning errors

## Metrics

| Group | Metrics |
|---|---|
| Real-view fidelity | PSNR / SSIM / LPIPS on the current frame, a future frame (t + 0.5 s) and held-out cameras |
| Virtual-view quality | PSNR / LPIPS against pseudo ground truth; FID / FVD |
| **Faithfulness** (primary) | Run detection and segmentation on the render and compare with ground truth projected into the virtual camera: per-class recall and precision (pedestrians and cyclists reported separately), hallucination rate, miss rate, NTA-IoU / NTL-IoU |
| Cross-dataset generalization | Leave-one-dataset-out: train on the other datasets, evaluate on the held-out rig and cities |
| Temporal stability | Warp error, flicker |
| Efficiency | ms per frame, GFLOPs, memory, compared with explicit-head baselines |
| Interaction | Tap-to-go success rate, goal error, collision / off-road rate (CARLA closed loop or NAVSIM v2) |

## MVP configuration

- Data: Waymo, 5 cameras × 3–4 frames
- Model: DINOv2-S encoder → STORM-style pixel-aligned Gaussians with a 16-d latent → gsplat at ¼ resolution → small U-Net decoder ([`configs/mvp.yaml`](configs/mvp.yaml))
- Supervision: OmniRe pseudo ground truth, Difix3D+ cleanup, adversarial loss
- Intent: projected overlays
- Acceptance: held-out-view LPIPS, virtual-view NTA-IoU and pedestrian recall, ≥ 20 FPS on one RTX 4090-class GPU
- Next: multi-dataset training on nuScenes, nuPlan, Argoverse 2 and DDAD; NAVSIM + GoalFlow for joint planning and tap-to-go
