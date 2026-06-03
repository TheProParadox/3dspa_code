# 3DSPA: 3D Semantic Point Autoencoder

This repository contains the implementation of **3DSPA** (3D Semantic Point Autoencoder), a framework for evaluating video realism using semantic-aware point trajectories. 3DSPA integrates DINOv2 semantic features with dense point tracking, enabling robust assessments of realism, temporal consistency, and physical plausibility in generated videos.

## Installation

```bash
git clone https://github.com/TheProParadox/3dspa_code.git
cd 3dspa_code

pip install -r requirements.txt
mkdir -p checkpoints

# VideoDepthAnything (required for --use_depth)
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
# Download vitb weights: https://huggingface.co/depth-anything/Video-Depth-Anything-Base
# Place as checkpoints/video_depth_anything_vitb.pth

# CoTracker3: loaded via torch.hub on first run (no clone needed)

# TAPVid-3D evaluation only:
# git clone https://github.com/google-deepmind/tapnet.git
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"
```

### Download 3DSPA Checkpoint

Download the model weights from [Google Drive](https://drive.google.com/file/d/1sd3_MuXDXw6TKbay2rh0EjHg3Pbe9RFr/view?usp=sharing) and place them at `./checkpoints/3dspa_ckpt.npz`, or specify with `--checkpoint_path`.

## Repository Structure

| File | Description |
|------|-------------|
| `track_autoencoder.py` | Point track autoencoder (2D) |
| `track_autoencoder_3d.py` | Point track autoencoder (3D, with depth and semantic features) |
| `attention.py` | Transformer attention modules |
| `train.py` | Training script with WandB integration |
| `inference.py` | Inference pipeline (CoTracker3 + DINOv2 + VideoDepthAnything + 3DSPA) |
| `evaluate_tapvid3d.py` | TAPVid-3D evaluation script |
| `data_loader.py` | Data loading utilities |
| `visualize.py` | Visualization utilities for point tracks |
| `visualizer.py` | CLI tool for visualizing point tracks on video |

## Training

```bash
python train.py \
  --model_type=3dspa \
  --checkpoint_dir=./checkpoints/3dspa \
  --wandb_project=3dspa \
  --wandb_run_name=3dspa_full \
  --batch_size=256 \
  --learning_rate=1e-4 \
  --num_epochs=300 \
  --num_output_frames=150 \
  --use_dino=True \
  --use_depth=True
```

## Inference

```bash
python inference.py \
  --checkpoint_path=./checkpoints/3dspa_ckpt.npz \
  --video_path=path/to/video.mp4 \
  --output_dir=./inference_output \
  --use_dino=True \
  --use_depth=True \
  --num_query_points=512 \
  --num_support_tracks=2048 \
  --tracking_grid_size=64 \
  --vda_model_path=./checkpoints/video_depth_anything_vitb.pth
```

tracks are passed to the checkpoint; DINOv2 and depth outputs are saved in
`predictions.npz` for downstream scoring.

## Visualization

The visualization tool projects point tracks onto video frames with color coding based on scores.

**CLI:**

```bash
python visualizer.py \
  --npz_path=./results/example_result_with_scores.npz \
  --output_dir=./visualizations \
  --trail=5 \
  --point_size=2 \
  --normalize_scores \
  --save_frames
```

**Arguments:**
- `--npz_path`: Path to `.npz` file containing `coords`, `coords_score`, `video`, `intrinsics`, `extrinsics`
- `--output_dir`: Output directory (default: same directory as the npz file)
- `--output_name`: Output video name (default: `{npz_stem}_visualized.mp4`)
- `--trail`: Number of frames for trail (default: 5)
- `--point_size`: Radius of points (default: 2)
- `--resize_height`, `--resize_width`: Dimensions for projection scaling (default: 1024)
- `--fps`: Frames per second for output video (default: 10)
- `--normalize_scores` / `--no_normalize_scores`: Normalize scores to `[0, 1]`
- `--save_frames`: Save individual frames as PNGs

**Python API:**

```python
from visualize import (
    load_visualization_data,
    prepare_video_for_visualization,
    project_all_tracks,
    paint_point_track_with_colors,
)
import cv2
import numpy as np
import imageio

data = load_visualization_data("results/example_result_with_scores.npz")
video_rgb, video_bgr = prepare_video_for_visualization(data['video'])
tracks_2d = project_all_tracks(
    data['coords'], data['intrinsics'], data['extrinsics'],
    original_height=video_rgb.shape[1],
    original_width=video_rgb.shape[2],
)

scores = data['coords_score']
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

video_viz = paint_point_track_with_colors(
    video_bgr, tracks_2d, data['visibs'].T, scores,
    trail=5, point_size=2,
)

video_viz_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_viz])
imageio.mimsave("output.mp4", video_viz_rgb, fps=10)
```

## Evaluation

### TAPVid-3D

Uses the official TAPVid-3D metrics from the [tapnet repository](https://github.com/google-deepmind/tapnet).

```bash
python evaluate_tapvid3d.py \
  --checkpoint_path=./checkpoints/3dspa_ckpt.npz \
  --dataset_path=./data/tapvid3d_dataset \
  --output_dir=./eval_results \
  --use_dino=True \
  --use_depth=True \
  --depth_scalings=median,per_trajectory \
  --data_sources=drivetrack,adt,pstudio \
  --use_minival=True
```

**Metrics:**
- `occlusion_accuracy`
- `pts_within_{1,2,4,8,16}` / `average_pts_within_thresh`
- `jaccard_{1,2,4,8,16}` / `average_jaccard`

### Other Benchmarks

Inference can be run on other video understanding benchmarks:

- **VideoPhy-2**: [Dataset](https://huggingface.co/datasets/videophysics/videophy2_test)
- **EvalCrafter**: [Dataset](https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset)
- **IntPhys2**: [Dataset](https://huggingface.co/datasets/facebook/IntPhys2)

For all benchmarks, run `inference.py` with the appropriate checkpoint and process results with benchmark-specific evaluation scripts.

## Acknowledgments

This work builds on:
- [TRAJAN](https://github.com/google-deepmind/tapnet) — point track autoencoder
- [TAPVid-3D](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) — evaluation benchmark
- [CoTracker](https://github.com/facebookresearch/co-tracker) — dense point tracking
- [DINOv2](https://github.com/facebookresearch/dinov2) — visual features
- [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) — depth estimation
