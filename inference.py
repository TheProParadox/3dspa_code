"""
3DSPA inference pipeline.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import torch

import track_autoencoder

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Path to 3DSPA checkpoint (.npz)')
flags.DEFINE_string('video_path', None, 'Path to input video')
flags.DEFINE_string('output_dir', './inference_output', 'Output directory')
flags.DEFINE_integer('num_output_frames', 150,
                     'Model output length (must match checkpoint)')
flags.DEFINE_integer('num_query_points', 512, 'Number of query points')
flags.DEFINE_integer('num_support_tracks', 2048, 'Number of support tracks')
flags.DEFINE_integer('tracking_grid_size', 64,
                     'CoTracker3 grid size (tracks grid_size**2 points)')
flags.DEFINE_bool('use_dino', True, 'Extract DINOv2 features (saved to output)')
flags.DEFINE_string('dino_model', 'facebook/dinov2-base', 'HuggingFace DINOv2 model id')
flags.DEFINE_bool('use_depth', True, 'Extract VDA depth maps (saved to output)')
flags.DEFINE_string('vda_model_path', None,
                    'Path to VideoDepthAnything .pth checkpoint')
flags.DEFINE_string('vda_encoder', 'vitb',
                    'VideoDepthAnything encoder: vits | vitb | vitl')


# ---- Video I/O ---------------------------------------------------------------

def load_video(
    path: str, max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Load video as uint8 RGB array.  Returns (frames [T,H,W,3], fps)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and len(frames) >= max_frames:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames in video: {path}")
    return np.stack(frames, axis=0), fps



def extract_2d_tracks(video: np.ndarray) -> Dict[str, np.ndarray]:

    logging.info("Running CoTracker3 for 2D point tracking ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_offline",
    )
    tracker = tracker.to(device).eval()

    vid = (
        torch.from_numpy(video)
        .permute(0, 3, 1, 2)      
        .float().div_(255.0)
        .unsqueeze(0)  
        .to(device)
    )

    with torch.no_grad():
        pred_tracks, pred_vis = tracker(vid, grid_size=FLAGS.tracking_grid_size)

    tracks = pred_tracks[0].permute(1, 0, 2).cpu().numpy()  

    v = pred_vis[0]
    if v.ndim == 2:                                          
        v = v.permute(1, 0).unsqueeze(-1)
    else:                                                     
        v = v.permute(1, 0, 2)
    vis = v.cpu().numpy()                                   

    logging.info(f"  {tracks.shape[0]} points, {tracks.shape[1]} frames")
    return {
        'tracks':  tracks.astype(np.float32),
        'visible': vis.astype(np.float32),
    }



def extract_dino_features(video: np.ndarray) -> np.ndarray:

    import torchvision.transforms as transforms
    from transformers import AutoImageProcessor, AutoModel

    logging.info("Extracting DINOv2 features ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = AutoImageProcessor.from_pretrained(FLAGS.dino_model)
    model = AutoModel.from_pretrained(FLAGS.dino_model).to(device).eval()

    T, H, W = video.shape[:3]
    patch_size = 14
    H_p, W_p = H // patch_size, W // patch_size
    target_H, target_W = H_p * patch_size, W_p * patch_size

    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_H, target_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats = []
    with torch.no_grad():
        for t in range(T):
            inp = xform(video[t]).unsqueeze(0).to(device)
            out = model(inp, output_hidden_states=True)
            pf = out.last_hidden_state[:, 1:, :] 
            pf = pf.reshape(1, H_p, W_p, -1)
            feats.append(pf.cpu().numpy()[0])

    result = np.stack(feats, axis=0)                 
    logging.info(f"  DINOv2 features: {result.shape}")
    return result.astype(np.float32)



_VDA_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,
             'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128,
             'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256,
             'out_channels': [256, 512, 1024, 1024]},
}


def extract_depth(video: np.ndarray, fps: float) -> np.ndarray:
    vda_dirs = [
        'Video-Depth-Anything', '../Video-Depth-Anything',
        './Video-Depth-Anything',
    ]
    for d in vda_dirs:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)

    from video_depth_anything.video_depth import VideoDepthAnything

    enc = FLAGS.vda_encoder
    if enc not in _VDA_CONFIGS:
        raise ValueError(f"vda_encoder must be one of {list(_VDA_CONFIGS)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = FLAGS.vda_model_path
    if model_path is None:
        candidates = [
            f'checkpoints/video_depth_anything_{enc}.pth',
            f'Video-Depth-Anything/checkpoints/video_depth_anything_{enc}.pth',
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break
    if model_path is None:
        raise FileNotFoundError(
            "VideoDepthAnything checkpoint not found. Use --vda_model_path "
            "or download from https://huggingface.co/depth-anything/"
            "Video-Depth-Anything-Base"
        )

    logging.info(f"Loading VideoDepthAnything ({enc}) from {model_path}")
    model = VideoDepthAnything(**_VDA_CONFIGS[enc], metric=False)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=True),
        strict=True,
    )
    model = model.to(device).eval()

    depths, _ = model.infer_video_depth(
        video, fps, input_size=518, device=device,
    )
    logging.info(f"  Depth maps: {depths.shape}")
    return depths.astype(np.float32)



def _unflatten(flat: Dict[str, np.ndarray]) -> dict:
    tree: dict = {}
    for key, val in flat.items():
        parts = key.split('/')
        d = tree
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = jnp.array(val)
    return tree


def load_checkpoint(path: str) -> dict:
    logging.info(f"Loading checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = np.load(path, allow_pickle=True)

    if 'params' in data.files:
        p = data['params']
        params = p.item() if hasattr(p, 'item') and p.ndim == 0 else dict(p)
        return params

    flat = {k: np.array(data[k]) for k in data.files}
    return _unflatten(flat)


def run_inference(
    video_path: str, checkpoint_path: str,
) -> Dict[str, Any]:
    """
    Run the full 3DSPA inference pipeline.
    """
    num_frames = FLAGS.num_output_frames

    video, fps = load_video(video_path)
    T, H, W = video.shape[:3]
    logging.info(f"Video: {T} frames, {H}x{W}, {fps:.1f} fps")

    td = extract_2d_tracks(video)
    tracks_2d = td['tracks']         
    visible   = td['visible']     
    N = tracks_2d.shape[0]

    dino_features = None
    if FLAGS.use_dino:
        dino_features = extract_dino_features(video)

    depth = None
    if FLAGS.use_depth:
        depth = extract_depth(video, fps)

    T_eff = min(T, num_frames)

    if T < num_frames:
        pad_len = num_frames - T
        tracks_2d = np.concatenate(
            [tracks_2d, np.tile(tracks_2d[:, -1:, :], (1, pad_len, 1))],
            axis=1,
        )
        visible = np.concatenate(
            [visible, np.zeros((N, pad_len, 1), np.float32)], axis=1,
        )
    elif T > num_frames:
        tracks_2d = tracks_2d[:, :num_frames]
        visible   = visible[:, :num_frames]

    boundary_frame = T_eff

    total_needed = FLAGS.num_support_tracks + FLAGS.num_query_points
    if total_needed > N:
        logging.warning(
            f"Need {total_needed} tracks but only have {N}. "
            f"Increase --tracking_grid_size (currently {FLAGS.tracking_grid_size})."
        )
    n_support = min(FLAGS.num_support_tracks, N)
    n_query   = min(FLAGS.num_query_points, N - n_support)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    sup_idx = perm[:n_support]
    qry_idx = perm[n_support:n_support + n_query]

    support_tracks  = tracks_2d[sup_idx]
    support_visible = visible[sup_idx]
    query_tracks    = tracks_2d[qry_idx]

    query_points = np.zeros((n_query, 3), dtype=np.float32)
    for i in range(n_query):
        vis_frames = np.where(visible[qry_idx[i], :T_eff, 0] > 0.5)[0]
        t = int(rng.choice(vis_frames)) if len(vis_frames) > 0 else 0
        query_points[i] = [t, query_tracks[i, t, 0], query_tracks[i, t, 1]]

    batch = {
        'support_tracks':         jnp.array(support_tracks[None]),
        'support_tracks_visible': jnp.array(support_visible[None]),
        'query_points':           jnp.array(query_points[None]),
        'boundary_frame':         jnp.array([boundary_frame]),
    }

    logging.info("Initializing 3DSPA model ...")
    model = track_autoencoder.TrackAutoEncoder(
        num_output_frames=num_frames,
    )

    rng_key = jax.random.PRNGKey(42)
    init_vars = model.init(rng_key, batch)
    params = load_checkpoint(checkpoint_path)

    def _check_params(expected, actual, path=""):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for k in expected:
                if k not in actual:
                    logging.warning(f"Missing in checkpoint: {path}/{k}")
                else:
                    _check_params(expected[k], actual[k], f"{path}/{k}")
        elif hasattr(expected, 'shape') and hasattr(actual, 'shape'):
            if expected.shape != actual.shape:
                logging.warning(
                    f"Shape mismatch {path}: "
                    f"model {expected.shape} vs ckpt {actual.shape}"
                )

    _check_params(init_vars['params'], params)

    logging.info("Running 3DSPA model ...")
    outputs = model.apply({'params': params}, batch)

    pred_tracks_2d = np.array(outputs.tracks[0])[:, :T_eff]
    pred_vis       = np.array(outputs.visible_logits[0])[:, :T_eff]
    pred_certain   = np.array(outputs.certain_logits[0])[:, :T_eff]

    logging.info("3DSPA inference complete.")
    return {
        'pred_tracks_2d':    pred_tracks_2d,
        'pred_visible':      pred_vis,
        'pred_certain':      pred_certain,
        'gt_query_2d':       np.array(query_tracks[:, :T_eff]),
        'support_tracks_2d': np.array(support_tracks[:, :T_eff]),
        'dino_features':     dino_features,
        'depth':             depth,
        'video':             video,
        'fps':               fps,
        'query_points':      query_points,
    }



def save_results(results: Dict[str, Any], output_dir: str):
    """Write predictions.npz and video_info.txt to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    pred_2d = results['pred_tracks_2d']
    T = pred_2d.shape[1]

    save_dict = dict(
        pred_tracks_2d=pred_2d,
        pred_visible=results['pred_visible'],
        pred_certain=results['pred_certain'],
        gt_query_2d=results['gt_query_2d'],
        query_points=results['query_points'],
    )
    if results['dino_features'] is not None:
        save_dict['dino_features'] = results['dino_features']
    if results['depth'] is not None:
        save_dict['depth'] = results['depth']

    np.savez(os.path.join(output_dir, 'predictions.npz'), **save_dict)

    with open(os.path.join(output_dir, 'video_info.txt'), 'w') as f:
        f.write(f"fps: {results['fps']}\n")
        f.write(f"frames: {T}\n")
        f.write(f"query_points: {pred_2d.shape[0]}\n")
        f.write(f"dino_features: {results['dino_features'] is not None}\n")
        f.write(f"depth_maps: {results['depth'] is not None}\n")

    logging.info(f"Results saved to {output_dir}")


# ---- Entry point --------------------------------------------------------------
def main(argv):
    del argv
    if FLAGS.video_path is None:
        raise ValueError('--video_path is required')
    if FLAGS.checkpoint_path is None:
        raise ValueError('--checkpoint_path is required')

    results = run_inference(FLAGS.video_path, FLAGS.checkpoint_path)
    save_results(results, FLAGS.output_dir)
    logging.info("Done.")


if __name__ == '__main__':
    app.run(main)
