# 3DSPA: 3D Semantic Point Autoencoder

> ⚠️ **This repository is under construction** 

This repository contains the implementation of **3DSPA** (3D Semantic Point Autoencoder), a framework for evaluating video realism using semantic-aware 3D point trajectories. 3DSPA extends TRAJAN to 3D by integrating DINOv2 semantic features and depth information, enabling robust assessments of realism, temporal consistency, and physical plausibility in generated videos.

## Status

- [x] Training code implemented
- [x] Model weights released
- [x] Inference code released
- [x] TAPVid-3D evaluation implemented
- [x] Visualization tools implemented
- [ ] Colab demo implemented

## Installation

```bash
# Clone the repository
git clone https://github.com/bchandna/3dspa_code.git
cd 3dspa_code

# Install dependencies
pip install -r requirements.txt

# Note: TAPVid-3D evaluation requires tapnet repository
# Clone tapnet and add to PYTHONPATH:
git clone https://github.com/google-deepmind/tapnet.git
export PYTHONPATH="${PYTHONPATH}:$(pwd)/tapnet"

# For inference: 2D point tracking
# CoTracker3 (install from GitHub)
git clone https://github.com/facebookresearch/co-tracker.git
pip install ./co-tracker
```

### Download 3DSPA Checkpoint

Download the 3DSPA model weights from [Google Drive](https://drive.google.com/file/d/1sd3_MuXDXw6TKbay2rh0EjHg3Pbe9RFr/view?usp=sharing). Place it in `./checkpoints/` or specify the path with `--checkpoint_path`.

## Files

- `track_autoencoder.py`: TRAJAN model (2D point track autoencoder)
- `track_autoencoder_3d.py`: 3DSPA model (3D point track autoencoder with semantic features)
- `attention.py`: Transformer attention modules
- `train.py`: Training script with WandB integration
- `inference.py`: Inference script for single videos with DINOv2 and VideoDepthAnything
- `evaluate_tapvid3d.py`: TAPVid-3D evaluation script
- `data_loader.py`: Data loading utilities
- `visualize.py`: Visualization utilities for 3D point tracks
- `visualizer.py`: CLI tool for visualizing point tracks on video

## Training

### 3DSPA Training

```bash
python train.py \
  --model_type=3dspa \
  --checkpoint_dir=./checkpoints/3dspa \
  --wandb_project=3dspa \
  --wandb_run_name=3dspa_full \
  --batch_size=64 \
  --learning_rate=1e-4 \
  --num_epochs=300 \
  --num_output_frames=150 \
  --use_dino=True \
  --use_depth=True
```

## Inference

### Running Inference on a Video

The inference script processes a single video with DINOv2 and VideoDepthAnything:

```bash
python inference.py \
  --checkpoint_path=./checkpoints/3dspa_ckpt.npz \
  --video_path=./data/example_video.mp4 \
  --output_dir=./inference_output \
  --use_dino=True \
  --use_depth=True \
  --num_query_points=512 \
  --num_support_tracks=2048 \
  --tracking_grid_size=64 \
  --vda_model_path=./checkpoints/video_depth_anything_vitb.pth
```

**Features:**
- Dense 2D point tracking using CoTracker3 with configurable grid size
- DINOv2 semantic feature extraction with bilinear interpolation for track sampling
- VideoDepthAnything depth estimation with bilinear interpolation
- Automatic 2D to 3D lifting using depth maps and camera intrinsics
- Full 3DSPA inference pipeline with support/query track splitting
- Checkpoint loading with parameter structure validation

**Output:**
- `predictions.npz`: Contains predicted 3D tracks, visibility logits, and ground truth tracks
- `video_info.txt`: Video metadata (FPS, frame count, etc.)

**Dependencies:**
- **CoTracker3**: For 2D point track extraction
  - Install: `git clone https://github.com/facebookresearch/co-tracker.git` then `pip install ./co-tracker`
- **DINOv2**: For semantic feature extraction
  - Automatically installed via `transformers` package
  - Uses `facebook/dinov2-base` model by default
  - Can specify different model with `--dino_model` flag
- **VideoDepthAnything**: For depth estimation
  - Install: `git clone https://github.com/DepthAnything/Video-Depth-Anything.git`
  - Download model checkpoint (e.g., `video_depth_anything_vitb.pth` from [HuggingFace](https://huggingface.co/depth-anything/Video-Depth-Anything-Base))
  - Specify path with `--vda_model_path` or place in `checkpoints/` directory

## Visualization

### Visualizing Point Tracks on Video

The visualization tool projects 3D point tracks to 2D image coordinates and visualizes them on video frames with color coding based on scores (e.g., `coords_score` metric). Colors range from Red (low scores) → White (0.5) → Blue (high scores).

**Using the CLI tool:**

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
- `--npz_path`: Path to .npz file containing `coords`, `coords_score`, `video`, `intrinsics`, `extrinsics`
- `--output_dir`: Output directory (default: same as npz file directory)
- `--output_name`: Output video name (default: `{npz_stem}_visualized.mp4`)
- `--trail`: Number of frames for trail (default: 5)
- `--point_size`: Radius of points (default: 2)
- `--resize_height`, `--resize_width`: Dimensions for projection scaling (default: 1024)
- `--fps`: Frames per second for output video (default: 10)
- `--normalize_scores`: Normalize scores to [0, 1] range (default: True)
- `--no_normalize_scores`: Use raw scores (must be in [0, 1] range)
- `--save_frames`: Save individual frames as PNG images

**Using the Python module:**

```python
from visualize import (
    load_visualization_data,
    prepare_video_for_visualization,
    project_all_tracks,
    paint_point_track_with_colors
)
import cv2

# Load data
data = load_visualization_data("results/example_result_with_scores.npz")

# Prepare video and project tracks
video_rgb, video_bgr = prepare_video_for_visualization(data['video'])
tracks_2d = project_all_tracks(
    data['coords'], data['intrinsics'], data['extrinsics'],
    original_height=video_rgb.shape[1],
    original_width=video_rgb.shape[2]
)

# Normalize scores
scores = data['coords_score']
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# Visualize
video_viz = paint_point_track_with_colors(
    video_bgr, tracks_2d, data['visibs'].T, scores,
    trail=5, point_size=2
)

# Convert to RGB and save
video_viz_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_viz])
imageio.mimsave("output.mp4", video_viz_rgb, fps=10)
```

**Output:**
- `{output_name}.mp4`: Visualized video with colored point tracks
- `{output_name}/frame_*.png`: Individual frames (if `--save_frames` is used)

**Color Mapping:**
- Red (0): Low scores
- White (0.5): Medium scores
- Blue (1.0): High scores

## Evaluation

### TAPVid-3D Evaluation

The evaluation script uses the official TAPVid-3D metrics from the [tapnet repository](https://github.com/google-deepmind/tapnet).

```bash
python evaluate_tapvid3d.py \
  --checkpoint_path=./checkpoints/3dspa_ckpt.npz \
  --dataset_path=./data/tapvid3d_dataset \
  --output_dir=./eval_results \
  --batch_size=8 \
  --use_dino=True \
  --use_depth=True \
  --depth_scalings=median,per_trajectory \
  --data_sources=drivetrack,adt,pstudio \
  --use_minival=True
```

**Metrics computed:**
- `occlusion_accuracy`: Accuracy of occlusion predictions
- `pts_within_{1,2,4,8,16}`: Fraction of points within pixel thresholds
- `jaccard_{1,2,4,8,16}`: Jaccard metric for each threshold
- `average_jaccard`: Average across all thresholds
- `average_pts_within_thresh`: Average across all thresholds

### Other Evaluations

Inference can be performed on other video understanding benchmarks:

- **VideoPhy-2**: [Dataset Link](https://huggingface.co/datasets/videophysics/videophy2_test) - Video physics understanding benchmark.
- **EvalCrafter**: [Dataset Link](https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset) - Video generation evaluation benchmark. 
- **IntPhys2**: [Dataset Link](https://huggingface.co/datasets/facebook/IntPhys2) - Intuitive physics benchmark. 

For all benchmarks, follow the same inference pipeline:
1. Prepare data in the required format
2. Run `inference.py` with the appropriate checkpoint
3. Process results using the evaluation metrics specific to each benchmark

## Data Format

### Training Data (Kubric3D + TAPVid-3D for 3DSPA)
- `video`: [T, H, W, 3] RGB frames
- `tracks_3d`: [N, T, 3] 3D point tracks (x, y, z)
- `visible`: [N, T, 1] visibility flags
- `dino_features`: [N, T, 768] optional DINOv2 features
- `depth_features`: [N, T, 256] optional depth features

### Evaluation Data (TAPVid-3D Minival)
- `video`: [T, H, W, 3] RGB frames
- `query_points`: [Q, 4] (t, x, y, z) query points
- `query_tracks`: [Q, T, 3] ground truth 3D tracks
- `query_tracks_visible`: [Q, T, 1] visibility flags
- `support_tracks`: [N, T, 3] support 3D tracks
- `support_tracks_visible`: [N, T, 1] support visibility

## Citation

If you use this code, please cite our paper:

<!-- ```bibtex
``` -->


## Acknowledgments

This work builds on 
- [TRAJAN](https://github.com/google-deepmind/tapnet) 
- [TAPVid-3D](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) benchmark.
