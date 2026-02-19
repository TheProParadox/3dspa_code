"""Inference script for 3DSPA on a single video with DINOv2 and VideoDepthAnything."""

import os
from typing import Dict, Any, Optional

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flax.training import checkpoints

# DINOv2 imports
try:
    from transformers import AutoImageProcessor, AutoModel
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    logging.warning("transformers not installed. DINO features will not be available.")

# CoTracker3 imports (required for 2D point tracking)
# Install from: git clone https://github.com/facebookresearch/co-tracker.git && pip install ./co-tracker
COTRACKER_AVAILABLE = False
try:
    import cotracker
    COTRACKER_AVAILABLE = True
except ImportError:
    pass

VDA_AVAILABLE = False
try:
    import sys
    # Try to import VideoDepthAnything from common installation paths
    # Clone from: https://github.com/DepthAnything/Video-Depth-Anything
    vda_paths = ['Video-Depth-Anything', '../Video-Depth-Anything', './Video-Depth-Anything']
    for vda_path in vda_paths:
        if os.path.exists(vda_path) and vda_path not in sys.path:
            sys.path.insert(0, vda_path)
    from video_depth_anything.video_depth import VideoDepthAnything
    VDA_AVAILABLE = True
except ImportError:
    pass

import track_autoencoder_3d

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Path to 3DSPA model checkpoint')
flags.DEFINE_string('video_path', None, 'Path to input video file')
flags.DEFINE_string('output_dir', './inference_output', 'Output directory for results')
flags.DEFINE_integer('num_output_frames', 150, 'Number of output frames')
flags.DEFINE_bool('use_dino', True, 'Use DINOv2 features')
flags.DEFINE_bool('use_depth', True, 'Use VideoDepthAnything depth features')
flags.DEFINE_integer('num_query_points', 512, 'Number of query points to predict')
flags.DEFINE_integer('num_support_tracks', 2048, 'Number of support tracks')
flags.DEFINE_integer('tracking_grid_size', 64, 'Grid size for dense tracking')
flags.DEFINE_string('dino_model', 'facebook/dinov2-base', 'DINOv2 model name')
flags.DEFINE_string('vda_model_path', None, 'Path to VideoDepthAnything checkpoint (.pth)')
flags.DEFINE_string('vda_encoder', 'vitb', 'VideoDepthAnything encoder: vits, vitb, or vitl')


def load_video(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Load video frames from file.
    
    Returns:
        frames: [T, H, W, 3] numpy array of RGB frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and len(frames) >= max_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    return np.array(frames), fps


def extract_2d_tracks_cotracker(video: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract 2D point tracks using CoTracker3.
    
    Returns:
        Dictionary with tracks [N, T, 2] and visible [N, T, 1]
    """
    logging.info("Extracting 2D point tracks using CoTracker3...")
    
    if not COTRACKER_AVAILABLE:
        raise RuntimeError(
            "CoTracker3 not available. Install from: "
            "git clone https://github.com/facebookresearch/co-tracker.git && pip install ./co-tracker"
        )
    
    T, H, W = video.shape[:3]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use CoTracker3
    tracker = cotracker.CoTracker()
    tracker = tracker.to(device)
    tracker.eval()
    
    # Convert video to torch tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    # Create dense grid of query points on first frame
    grid_size = FLAGS.tracking_grid_size
    query_points = []
    step_x, step_y = W / grid_size, H / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x = (j + 0.5) * step_x
            y = (i + 0.5) * step_y
            query_points.append([0, y, x])  # [t, y, x] format for CoTracker
    
    query_points = torch.tensor(query_points, device=device).float().unsqueeze(0)
    
    # Track points
    with torch.no_grad():
        pred_tracks, pred_visibility = tracker(
            video=video_tensor,
            queries=query_points,
        )
    
    # Convert to numpy: [1, N, T, 2] -> [N, T, 2]
    tracks = pred_tracks[0].cpu().numpy()  # [N, T, 2] in (x, y) format
    visibility = pred_visibility[0].cpu().numpy()  # [N, T] boolean
    
    # Convert visibility to [N, T, 1]
    visible = visibility[..., np.newaxis].astype(np.float32)
    
    logging.info(f"Tracked {tracks.shape[0]} points over {tracks.shape[1]} frames")
    
    return {
        'tracks': tracks.astype(np.float32),
        'visible': visible,
    }


def extract_dino_features(video: np.ndarray, model=None, processor=None) -> np.ndarray:
    """Extract DINOv2 semantic features for each video frame.
    
    Returns:
        features: [T, H_patches, W_patches, 768] DINOv2 patch features
    """
    if not FLAGS.use_dino:
        return None
    
    if not DINO_AVAILABLE:
        raise RuntimeError("DINOv2 not available. Install: pip install transformers")
    
    logging.info("Extracting DINOv2 features...")
    
    # Load model if not provided
    if model is None:
        processor = AutoImageProcessor.from_pretrained(FLAGS.dino_model)
        model = AutoModel.from_pretrained(FLAGS.dino_model).eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    
    T, H, W = video.shape[:3]
    
    # DINOv2 patch size is 14 for base model
    patch_size = 14
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    # Resize video to be divisible by patch size
    target_H = H_patches * patch_size
    target_W = W_patches * patch_size
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_H, target_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_features = []
    
    with torch.no_grad():
        for t in range(T):
            frame = video[t]
            input_tensor = transform(frame).unsqueeze(0).to(device)
            
            # Extract features
            outputs = model(input_tensor, output_hidden_states=True)
            
            # Get patch features from last hidden state
            # DINOv2 outputs: [batch, num_patches + 1, dim] (includes CLS token)
            patch_features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            patch_features = patch_features.reshape(1, H_patches, W_patches, -1)
            
            all_features.append(patch_features.cpu().numpy()[0])
    
    return np.array(all_features)  # [T, H_patches, W_patches, 768]


def extract_depth_features(video: np.ndarray, vda_model=None, fps: float = 30.0) -> np.ndarray:
    """Extract depth maps using VideoDepthAnything.
    
    Returns:
        depth: [T, H, W, 1] depth maps
    """
    if not FLAGS.use_depth:
        return None
    
    if not VDA_AVAILABLE:
        raise RuntimeError(
            "VideoDepthAnything not available. "
            "Install from: https://github.com/DepthAnything/Video-Depth-Anything"
        )
    
    logging.info("Extracting depth features using VideoDepthAnything...")
    
    # Model configs for Video-Depth-Anything (relative depth)
    VDA_MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    encoder = FLAGS.vda_encoder
    if encoder not in VDA_MODEL_CONFIGS:
        raise ValueError(f"vda_encoder must be one of {list(VDA_MODEL_CONFIGS.keys())}")
    
    # Initialize model if not provided
    if vda_model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = FLAGS.vda_model_path
        if model_path is None:
            # Try default paths (Video-Depth-Anything checkpoint names)
            default_paths = [
                f'checkpoints/video_depth_anything_{encoder}.pth',
                f'Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth',
                './checkpoints/video_depth_anything_vitb.pth',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError(
                    "VideoDepthAnything model not found. Specify path with --vda_model_path. "
                    "Download from: https://huggingface.co/depth-anything/Video-Depth-Anything-Base"
                )
        vda_model = VideoDepthAnything(**VDA_MODEL_CONFIGS[encoder], metric=False)
        vda_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        vda_model = vda_model.to(device).eval()
    
    T, H, W = video.shape[:3]
    
    # VideoDepthAnything.infer_video_depth expects frames as numpy [T, H, W, 3]
    # and returns (depths [T, H, W], fps)
    depths, _ = vda_model.infer_video_depth(
        video.astype(np.float32) / 255.0,
        fps,
        input_size=518,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        fp32=False,
    )
    # depths: [T, H, W] -> [T, H, W, 1]
    depth_array = depths[..., np.newaxis].astype(np.float32)
    logging.info(f"Extracted depth maps: {depth_array.shape}")
    
    return depth_array


def lift_2d_to_3d(tracks_2d: np.ndarray, depth: np.ndarray, intrinsics=None) -> np.ndarray:
    """Lift 2D tracks to 3D using depth maps and camera intrinsics.
    
    Returns:
        tracks_3d: [N, T, 3] 3D point tracks (x, y, z) in camera coordinates
    """
    N, T = tracks_2d.shape[:2]
    tracks_3d = np.zeros((N, T, 3))
    
    # Default intrinsics if not provided
    if intrinsics is None:
        H, W = depth.shape[1:3]
        fx = fy = max(H, W)
        cx, cy = W / 2, H / 2
    else:
        fx, fy, cx, cy = intrinsics
    
    # Bilinear interpolation for depth sampling
    for n in range(N):
        for t in range(T):
            x, y = tracks_2d[n, t]
            
            # Bilinear interpolation for depth
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1
            wx, wy = x - x0, y - y0
            
            # Clamp to valid range
            x0 = np.clip(x0, 0, depth.shape[2] - 1)
            y0 = np.clip(y0, 0, depth.shape[1] - 1)
            x1 = np.clip(x1, 0, depth.shape[2] - 1)
            y1 = np.clip(y1, 0, depth.shape[1] - 1)
            
            # Bilinear interpolation
            z00 = depth[t, y0, x0, 0]
            z01 = depth[t, y0, x1, 0]
            z10 = depth[t, y1, x0, 0]
            z11 = depth[t, y1, x1, 0]
            
            z = (z00 * (1 - wx) * (1 - wy) +
                 z01 * wx * (1 - wy) +
                 z10 * (1 - wx) * wy +
                 z11 * wx * wy)
            
            # Convert to 3D camera coordinates
            tracks_3d[n, t, 0] = (x - cx) * z / fx
            tracks_3d[n, t, 1] = (y - cy) * z / fy
            tracks_3d[n, t, 2] = z
    
    return tracks_3d.astype(np.float32)


def sample_dino_features_for_tracks(dino_features: np.ndarray, tracks_2d: np.ndarray, 
                                     video_shape: tuple) -> np.ndarray:
    """Sample DINOv2 features at 2D track locations using bilinear interpolation.
    
    Args:
        dino_features: [T, H_patches, W_patches, 768] DINO patch features
        tracks_2d: [N, T, 2] 2D track positions in original image coordinates
        video_shape: (T, H, W, 3) original video shape
    
    Returns:
        track_dino_features: [N, T, 768] DINO features per track
    """
    if dino_features is None:
        return None
    
    T, H_patches, W_patches, D = dino_features.shape
    _, H, W, _ = video_shape
    N = tracks_2d.shape[0]
    
    # DINOv2 patch size
    patch_size = 14
    scale_h = H_patches / H
    scale_w = W_patches / W
    
    track_features = np.zeros((N, T, D))
    
    for n in range(N):
        for t in range(T):
            x, y = tracks_2d[n, t]
            
            # Convert to patch coordinates
            patch_x = x * scale_w
            patch_y = y * scale_h
            
            # Bilinear interpolation in patch space
            x0, y0 = int(np.floor(patch_x)), int(np.floor(patch_y))
            x1, y1 = x0 + 1, y0 + 1
            wx, wy = patch_x - x0, patch_y - y0
            
            # Clamp to valid range
            x0 = np.clip(x0, 0, W_patches - 1)
            y0 = np.clip(y0, 0, H_patches - 1)
            x1 = np.clip(x1, 0, W_patches - 1)
            y1 = np.clip(y1, 0, H_patches - 1)
            
            # Bilinear interpolation
            f00 = dino_features[t, y0, x0]
            f01 = dino_features[t, y0, x1]
            f10 = dino_features[t, y1, x0]
            f11 = dino_features[t, y1, x1]
            
            track_features[n, t] = (f00 * (1 - wx) * (1 - wy) +
                                   f01 * wx * (1 - wy) +
                                   f10 * (1 - wx) * wy +
                                   f11 * wx * wy)
    
    return track_features.astype(np.float32)


def sample_depth_features_for_tracks(depth: np.ndarray, tracks_2d: np.ndarray) -> np.ndarray:
    """Extract depth-based features at 2D track locations.
    
    Returns:
        track_depth_features: [N, T, 256] depth features per track
    """
    if depth is None:
        return None
    
    N, T = tracks_2d.shape[:2]
    track_depth = np.zeros((N, T, 256))
    
    for n in range(N):
        for t in range(T):
            x, y = tracks_2d[n, t]
            
            # Bilinear interpolation for depth
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1
            wx, wy = x - x0, y - y0
            
            # Clamp to valid range
            x0 = np.clip(x0, 0, depth.shape[2] - 1)
            y0 = np.clip(y0, 0, depth.shape[1] - 1)
            x1 = np.clip(x1, 0, depth.shape[2] - 1)
            y1 = np.clip(y1, 0, depth.shape[1] - 1)
            
            # Bilinear interpolation
            d00 = depth[t, y0, x0, 0]
            d01 = depth[t, y0, x1, 0]
            d10 = depth[t, y1, x0, 0]
            d11 = depth[t, y1, x1, 0]
            
            d = (d00 * (1 - wx) * (1 - wy) +
                 d01 * wx * (1 - wy) +
                 d10 * (1 - wx) * wy +
                 d11 * wx * wy)
            
            # Create depth features: depth value, normalized depth, depth gradient, etc.
            track_depth[n, t, 0] = d
            track_depth[n, t, 1] = d / 10.0  # Normalized
            
            # Depth gradient (simplified)
            if t > 0:
                d_prev = track_depth[n, t-1, 0]
                track_depth[n, t, 2] = d - d_prev  # Temporal gradient
            
            # Additional features can be added here
    
    return track_depth.astype(np.float32)


def load_checkpoint(checkpoint_path: str, model) -> Any:
    """Load model checkpoint using Flax checkpoints.
    
    Returns:
        params: Model parameters
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to restore checkpoint
    try:
        state_dict = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        if 'params' in state_dict:
            params = state_dict['params']
        elif 'optimizer' in state_dict and 'target' in state_dict['optimizer']:
            params = state_dict['optimizer']['target']
        else:
            # Assume the checkpoint is just params
            params = state_dict
        return params
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        raise


def run_inference(video_path: str, checkpoint_path: str) -> Dict[str, Any]:
    """Run 3DSPA inference: extract features, lift to 3D, predict trajectories.
    
    Returns:
        Dictionary with predictions, video, tracks, and depth maps
    """
    logging.info(f"Loading video from {video_path}")
    video, fps = load_video(video_path, max_frames=FLAGS.num_output_frames)
    T, H, W = video.shape[:3]
    logging.info(f"Loaded video: {T} frames, {H}x{W} resolution, {fps:.2f} fps")
    
    # Extract 2D tracks
    track_data = extract_2d_tracks_cotracker(video)
    tracks_2d = track_data['tracks']
    visible = track_data['visible']
    N = tracks_2d.shape[0]
    logging.info(f"Extracted {N} 2D point tracks")
    
    # Extract DINO features
    dino_features = None
    if FLAGS.use_dino:
        dino_features = extract_dino_features(video)
        logging.info(f"Extracted DINOv2 features: {dino_features.shape}")
    
    # Extract depth
    depth = None
    if FLAGS.use_depth:
        depth = extract_depth_features(video, fps=fps)
        logging.info(f"Extracted depth maps: {depth.shape}")
    
    # Lift 2D tracks to 3D
    if depth is not None:
        tracks_3d = lift_2d_to_3d(tracks_2d, depth)
    else:
        # If no depth, use z=1.0 for all points (assume unit depth)
        tracks_3d = np.concatenate([tracks_2d, np.ones((N, T, 1))], axis=-1)
    
    # Sample features for tracks
    dino_track_features = None
    if dino_features is not None:
        dino_track_features = sample_dino_features_for_tracks(
            dino_features, tracks_2d, video.shape
        )
    
    depth_track_features = None
    if depth is not None:
        depth_track_features = sample_depth_features_for_tracks(depth, tracks_2d)
    
    # Split into support and query tracks
    indices = np.random.permutation(N)
    support_indices = indices[:FLAGS.num_support_tracks]
    query_indices = indices[FLAGS.num_support_tracks:FLAGS.num_support_tracks + FLAGS.num_query_points]
    
    support_tracks = tracks_3d[support_indices]
    support_visible = visible[support_indices]
    query_tracks = tracks_3d[query_indices]
    query_visible = visible[query_indices]
    
    # Prepare query points (sample from query tracks at random frames)
    query_points = []
    for i in range(FLAGS.num_query_points):
        t = np.random.randint(0, T)
        x, y, z = query_tracks[i, t]
        query_points.append([t, x, y, z])
    query_points = np.array(query_points)
    
    # Prepare batch
    batch = {
        'support_tracks': jnp.array(support_tracks[np.newaxis]),
        'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
        'query_points': jnp.array(query_points[np.newaxis]),
        'query_tracks': jnp.array(query_tracks[np.newaxis]),
        'query_tracks_visible': jnp.array(query_visible[np.newaxis]),
        'boundary_frame': jnp.array([T]),
    }
    
    if dino_track_features is not None:
        batch['dino_features'] = jnp.array(dino_track_features[support_indices][np.newaxis])
    if depth_track_features is not None:
        batch['depth_features'] = jnp.array(depth_track_features[support_indices][np.newaxis])
    
    # Load model and checkpoint
    logging.info(f"Loading model from {checkpoint_path}")
    model = track_autoencoder_3d.TrackAutoEncoder3D(
        num_output_frames=FLAGS.num_output_frames,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
    
    # Initialize model to get parameter structure
    rng = jax.random.PRNGKey(42)
    init_params = model.init(rng, batch)['params']
    
    # Load actual checkpoint
    params = load_checkpoint(checkpoint_path, model)
    
    # Verify parameter structure matches
    def check_params_structure(expected, actual, path=""):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k in expected.keys():
                if k not in actual:
                    logging.warning(f"Key {path}.{k} missing in checkpoint")
                else:
                    check_params_structure(expected[k], actual[k], f"{path}.{k}")
        elif hasattr(expected, 'shape') and hasattr(actual, 'shape'):
            if expected.shape != actual.shape:
                logging.warning(f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}")
    
    check_params_structure(init_params, params)
    
    # Run inference
    logging.info("Running 3DSPA inference...")
    predictions = model.apply({'params': params}, batch)
    
    logging.info("Inference completed successfully")
    
    return {
        'predictions': predictions,
        'video': video,
        'tracks_3d': tracks_3d,
        'support_tracks': support_tracks,
        'query_tracks': query_tracks,
        'depth': depth,
        'fps': fps,
    }


def save_results(results: Dict[str, Any], output_dir: str):
    """Save predictions and intermediate results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    pred_tracks = np.array(results['predictions'].tracks[0])  # [Q, T, 3]
    pred_visible = np.array(results['predictions'].visible_logits[0])  # [Q, T, 1]
    
    np.savez(
        os.path.join(output_dir, 'predictions.npz'),
        tracks_3d=pred_tracks,
        visible_logits=pred_visible,
        query_tracks=results['query_tracks'],
        support_tracks=results['support_tracks'],
    )
    
    # Save video info
    with open(os.path.join(output_dir, 'video_info.txt'), 'w') as f:
        f.write(f"FPS: {results['fps']}\n")
        f.write(f"Frames: {pred_tracks.shape[1]}\n")
        f.write(f"Query points: {pred_tracks.shape[0]}\n")
    
    logging.info(f"Results saved to {output_dir}")


def main(argv):
    """Main inference function."""
    del argv
    
    if FLAGS.video_path is None:
        raise ValueError('Must provide video_path')
    if FLAGS.checkpoint_path is None:
        raise ValueError('Must provide checkpoint_path')
    
    # Run inference
    results = run_inference(FLAGS.video_path, FLAGS.checkpoint_path)
    
    # Save results
    save_results(results, FLAGS.output_dir)
    
    logging.info("Inference completed!")


if __name__ == '__main__':
    app.run(main)
