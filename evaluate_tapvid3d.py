"""TAPVid-3D evaluation for 3DSPA.
Evaluates on the TAPVid-3D minival (or full_eval) split across
data sources (drivetrack, adt, pstudio) and reports:
  - Occlusion Accuracy (OA)
  - Average Percentage within thresholds in 3D (APD)
  - Average Jaccard in 3D (AJ)
Metrics follow Koppula et al. (2024).
"""

import json
import os
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from tapnet.tapvid3d.evaluation import metrics as tapvid3d_metrics
from tapnet.tapvid3d.splits import tapvid3d_splits

import track_autoencoder_3d


FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Path to model checkpoint')
flags.DEFINE_string('dataset_path', None, 'Path to TAPVid-3D dataset')
flags.DEFINE_string('output_dir', './eval_results', 'Output directory')
flags.DEFINE_integer('num_output_frames', 150, 'Number of output frames')
flags.DEFINE_bool('use_dino', True, 'Use DINOv2 features')
flags.DEFINE_bool('use_depth', True, 'Use depth features')
flags.DEFINE_list('depth_scalings', ['median'],
                  'Depth scaling strategies: median, per_trajectory, none')
flags.DEFINE_list('data_sources', ['drivetrack', 'adt', 'pstudio'],
                  'Data sources to evaluate')
flags.DEFINE_bool('use_minival', True, 'Use minival split')


_ZERO_METRICS = {
    'occlusion_accuracy': 0.0,
    'pts_within_1': 0.0,  'jaccard_1': 0.0,
    'pts_within_2': 0.0,  'jaccard_2': 0.0,
    'pts_within_4': 0.0,  'jaccard_4': 0.0,
    'pts_within_8': 0.0,  'jaccard_8': 0.0,
    'pts_within_16': 0.0, 'jaccard_16': 0.0,
    'average_jaccard': 0.0,
    'average_pts_within_thresh': 0.0,
}


# ---- Checkpoint loading -------------------------------------------------------

def _unflatten(flat: Dict[str, np.ndarray]) -> dict:
  """Convert flat 'a/b/c' keys into nested dicts."""
  tree: dict = {}
  for key, val in flat.items():
    parts = key.split('/')
    d = tree
    for p in parts[:-1]:
      d = d.setdefault(p, {})
    d[parts[-1]] = jnp.array(val)
  return tree


def load_checkpoint(path: str) -> dict:
  """Load a .npz checkpoint into a nested param dict."""
  if not os.path.exists(path):
    raise FileNotFoundError(f'Checkpoint not found: {path}')

  data = np.load(path, allow_pickle=True)

  if 'params' in data.files:
    p = data['params']
    return p.item() if hasattr(p, 'item') and p.ndim == 0 else dict(p)

  flat = {k: np.array(data[k]) for k in data.files}
  return _unflatten(flat)


# ---- Model forward pass -------------------------------------------------------

def predict(params, batch, num_output_frames=150):
  """Run 3DSPA forward pass and return predictions."""
  model = track_autoencoder_3d.TrackAutoEncoder3D(
      num_output_frames=num_output_frames,
      use_dino=FLAGS.use_dino,
      use_depth=FLAGS.use_depth,
  )
  return model.apply({'params': params}, batch)


def predictions_to_tapvid3d(predictions):
  """Convert model output to TAPVid-3D evaluation format.

  Returns:
    pred_tracks:   [T, N, 3] float
    pred_occluded: [T, N] bool
  """
  pred_tracks = np.array(predictions.tracks[0])       
  pred_tracks = np.transpose(pred_tracks, (1, 0, 2))

  vis_logits = np.array(predictions.visible_logits[0, :, :, 0])  
  pred_occluded = np.transpose(vis_logits <= 0.0, (1, 0))   

  return pred_tracks, pred_occluded


# ---- Main evaluation ----------------------------------------------------------

def main(argv):
  del argv

  if FLAGS.checkpoint_path is None:
    raise ValueError('--checkpoint_path is required')
  if FLAGS.dataset_path is None:
    raise ValueError('--dataset_path is required')

  os.makedirs(FLAGS.output_dir, exist_ok=True)

  logging.info(f'Loading checkpoint: {FLAGS.checkpoint_path}')
  params = load_checkpoint(FLAGS.checkpoint_path)

  all_results = {}

  for source in FLAGS.data_sources:
    logging.info(f'Evaluating: {source}')

    if FLAGS.use_minival:
      npz_files = tapvid3d_splits.get_minival_files(subset=source)
    else:
      npz_files = tapvid3d_splits.get_full_eval_files(subset=source)

    source_gt_dir = os.path.join(FLAGS.dataset_path, source)
    source_metrics = []

    for npz_file in tqdm.tqdm(npz_files, desc=source):
      gt_file = os.path.join(source_gt_dir, npz_file)

      with open(gt_file, 'rb') as f:
        gt_data = np.load(f, allow_pickle=True)
        queries_xyt = gt_data['queries_xyt']
        tracks_xyz = gt_data['tracks_XYZ']
        visibles = gt_data['visibility']
        intrinsics_params = gt_data['fx_fy_cx_cy']

      batch = {
          'query_points': jnp.array(queries_xyt[np.newaxis]),
          'query_tracks': jnp.array(tracks_xyz[np.newaxis]),
          'query_tracks_visible': jnp.array(
              visibles[np.newaxis, ..., np.newaxis]),
          'intrinsics': jnp.array(intrinsics_params[np.newaxis]),
          'support_tracks': jnp.array(tracks_xyz[np.newaxis]),
          'support_tracks_visible': jnp.array(
              visibles[np.newaxis, ..., np.newaxis]),
          'boundary_frame': jnp.array([tracks_xyz.shape[1]]),
      }

      predictions = predict(params, batch, FLAGS.num_output_frames)
      pred_tracks, pred_occluded = predictions_to_tapvid3d(predictions)

      gt_tracks_tn = np.transpose(tracks_xyz, (1, 0, 2))
      gt_occluded = np.transpose(~visibles, (1, 0))

      video_metrics = {}
      for scaling in FLAGS.depth_scalings:
        try:
          m = tapvid3d_metrics.compute_tapvid3d_metrics(
              gt_occluded=gt_occluded,
              gt_tracks=gt_tracks_tn,
              pred_occluded=pred_occluded,
              pred_tracks=pred_tracks,
              intrinsics_params=intrinsics_params,
              scaling=scaling,
              query_points=queries_xyt[:, ::-1],
              order='t n',
          )
          video_metrics[scaling] = {
              k: float(v) for k, v in m.items()
          }
        except Exception as e:
          logging.warning(f'{npz_file} scaling={scaling}: {e}')
          video_metrics[scaling] = dict(_ZERO_METRICS)
      source_metrics.append(video_metrics)

    all_results[source] = {}
    for scaling in FLAGS.depth_scalings:
      agg = {}
      for key in source_metrics[0][scaling]:
        vals = [m[scaling][key] for m in source_metrics]
        agg[key] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))
      all_results[source][scaling] = agg

    logging.info(f'Results for {source}:')
    for scaling in FLAGS.depth_scalings:
      logging.info(f'  scaling={scaling}')
      for key, val in all_results[source][scaling].items():
        if not key.endswith('_std'):
          logging.info(f'    {key}: {val:.4f}')

  # Overall average across data sources
  overall = {}
  for scaling in FLAGS.depth_scalings:
    overall[scaling] = {}
    for key in all_results[FLAGS.data_sources[0]][scaling]:
      if not key.endswith('_std'):
        vals = [all_results[ds][scaling][key] for ds in FLAGS.data_sources]
        overall[scaling][key] = float(np.mean(vals))
        overall[scaling][f'{key}_std'] = float(np.std(vals))

  logging.info('Overall (averaged across sources):')
  for scaling in FLAGS.depth_scalings:
    logging.info(f'  scaling={scaling}')
    for key, val in overall[scaling].items():
      if not key.endswith('_std'):
        logging.info(f'    {key}: {val:.4f}')

  results_file = os.path.join(FLAGS.output_dir, 'results.json')
  with open(results_file, 'w') as f:
    json.dump({'per_source': all_results, 'overall': overall}, f, indent=2)

  logging.info(f'Saved to {results_file}')


if __name__ == '__main__':
  app.run(main)
