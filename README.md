# 3DSPA: 3D Semantic Point Autoencoder

> ⚠️ **This repository is under construction** 

This repository contains the implementation of **3DSPA** (3D Semantic Point Autoencoder), a framework for evaluating video realism using semantic-aware 3D point trajectories. 3DSPA extends TRAJAN to 3D by integrating DINOv2 semantic features and depth information, enabling robust assessments of realism, temporal consistency, and physical plausibility in generated videos.

## Status

- [x] Training code implemented
- [x] Model weights released
- [x] TAPVid-3D evaluation implemented
- [ ] Other evaluations (VideoPhy-2, EvalCrafter, IntPhys2) remaining
- [ ] Visualization tools remaining
- [ ] Colab demo remaining

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
# CoTracker3
pip install cotracker
```

## Files

- `track_autoencoder.py`: TRAJAN model (2D point track autoencoder)
- `track_autoencoder_3d.py`: 3DSPA model (3D point track autoencoder with semantic features)
- `attention.py`: Transformer attention modules
- `train.py`: Training script with WandB integration
- `inference.py`: Inference script for single videos with DINOv2 and VideoDepthAnything
- `evaluate_tapvid3d.py`: TAPVid-3D evaluation script
- `data_loader.py`: Data loading utilities

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
  --checkpoint_path=./checkpoints/3dspa/checkpoint_100000 \
  --video_path=./data/example_video.mp4 \
  --output_dir=./inference_output \
  --use_dino=True \
  --use_depth=True \
  --num_query_points=512 \
  --num_support_tracks=2048 \
  --tracking_grid_size=64 \
  --vda_model_path=./checkpoints/depth_anything_vitb14.pth
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
  - Install: `pip install cotracker`
  - Alternatively, use BootsTAPIR from tapnet repository
- **DINOv2**: For semantic feature extraction
  - Automatically installed via `transformers` package
  - Uses `facebook/dinov2-base` model by default
  - Can specify different model with `--dino_model` flag
- **VideoDepthAnything**: For depth estimation
  - Install: `git clone https://github.com/DepthAnything/VideoDepthAnything.git`
  - Download model checkpoint (e.g., `depth_anything_vitb14.pth`)
  - Specify path with `--vda_model_path` or place in `checkpoints/` directory

## Evaluation

### TAPVid-3D Evaluation

The evaluation script uses the official TAPVid-3D metrics from the [tapnet repository](https://github.com/google-deepmind/tapnet).

```bash
python evaluate_tapvid3d.py \
  --checkpoint_path=./checkpoints/3dspa/checkpoint_100000 \
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
