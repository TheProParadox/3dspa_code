"""TAPVid-3D evaluation script using official metrics from tapnet."""

import functools
import os
from typing import Any, Dict, List, Tuple

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax.training import checkpoints

# Import TAPVid-3D metrics from tapnet repository
from tapnet.tapvid3d.evaluation import metrics as tapvid3d_metrics
from tapnet.tapvid3d.splits import tapvid3d_splits

import track_autoencoder_3d


FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Path to model checkpoint')
flags.DEFINE_string('dataset_path', None, 'Path to TAPVid-3D dataset')
flags.DEFINE_string('output_dir', './eval_results', 'Output directory')
flags.DEFINE_integer('batch_size', 8, 'Batch size for evaluation')
flags.DEFINE_integer('num_output_frames', 150, 'Number of output frames')
flags.DEFINE_bool('use_dino', True, 'Use DINO features')
flags.DEFINE_bool('use_depth', True, 'Use depth features')
flags.DEFINE_list('depth_scalings', ['median'], 
                  'Depth scaling strategies: median, per_trajectory, none, etc.')
flags.DEFINE_list('data_sources', ['drivetrack', 'adt', 'pstudio'],
                  'Data sources to evaluate')
flags.DEFINE_bool('use_minival', True, 'Use minival split (otherwise full_eval)')


def convert_predictions_to_tapvid3d_format(predictions, query_points):
  """Convert 3DSPA predictions from [B, Q, T, 3] to TAPVid-3D format [T, N, 3].
  
  Returns:
    pred_tracks: [T, N, 3] numpy array
    pred_occluded: [T, N] numpy array (True = occluded)
  """
  # Convert JAX arrays to numpy
  pred_tracks = np.array(predictions.tracks)  # [B, Q, T, 3]
  pred_visible_logits = np.array(predictions.visible_logits)  # [B, Q, T, 1]
  
  # Convert to [T, N, 3] format (T=time, N=num_tracks)
  # Remove batch dimension and transpose
  pred_tracks = pred_tracks[0]  # [Q, T, 3]
  pred_tracks = np.transpose(pred_tracks, (1, 0, 2))  # [T, Q, 3]
  
  # Convert visibility logits to occluded (True = occluded)
  pred_occluded = (pred_visible_logits[0, :, :, 0] <= 0.0)  # [Q, T]
  pred_occluded = np.transpose(pred_occluded, (1, 0))  # [T, Q]
  
  return pred_tracks, pred_occluded


@functools.partial(jax.jit, static_argnames=['num_output_frames'])
def evaluate_batch(params, batch, num_output_frames=150, use_dino=True, use_depth=True):
  """Evaluate batch using official TAPVid-3D metrics.
  
  Returns:
    metrics: Dictionary of metric values
    predictions: Model predictions
  """
  model = track_autoencoder_3d.TrackAutoEncoder3D(
      num_output_frames=num_output_frames,
      use_dino=use_dino,
      use_depth=use_depth,
  )
  
  # Forward pass
  predictions = model.apply({'params': params}, batch)
  
  # Convert predictions to TAPVid-3D format
  query_points = np.array(batch['query_points'][0])  # [Q, 4]
  pred_tracks, pred_occluded = convert_predictions_to_tapvid3d_format(
      predictions, batch['query_points']
  )
  
  # Extract ground truth
  gt_tracks = np.array(batch['query_tracks'][0])  # [Q, T, 3]
  gt_tracks = np.transpose(gt_tracks, (1, 0, 2))  # [T, Q, 3]
  gt_visible = np.array(batch['query_tracks_visible'][0])  # [Q, T, 1]
  gt_occluded = np.logical_not(gt_visible[:, :, 0])  # [Q, T]
  gt_occluded = np.transpose(gt_occluded, (1, 0))  # [T, Q]
  
  # Get intrinsics (if available, otherwise use defaults)
  if 'intrinsics' in batch:
    intrinsics = np.array(batch['intrinsics'][0])  # [4] (fx, fy, cx, cy)
  else:
    # Default intrinsics for 256x256 images
    intrinsics = np.array([256.0, 256.0, 128.0, 128.0])
  
  # Compute TAPVid-3D metrics using the official implementation
  metrics_dict = tapvid3d_metrics.compute_tapvid3d_metrics(
      gt_occluded=gt_occluded,
      gt_tracks=gt_tracks,
      pred_occluded=pred_occluded,
      pred_tracks=pred_tracks,
      intrinsics_params=intrinsics,
      scaling='per_trajectory',  # Can be 'median', 'per_trajectory', 'none', etc.
      query_points=query_points[:, ::-1],  # Convert to (t, y, x) format
      order='t n',
  )
  
  # Convert numpy arrays to floats for JSON serialization
  metrics = {k: float(v) if isinstance(v, np.ndarray) else v 
             for k, v in metrics_dict.items()}
  
  return metrics, predictions


def load_tapvid3d_dataset(
    dataset_path: str,
    split: str = 'minival',
    batch_size: int = 8,
):
  """Load TAPVid-3D dataset.
  
  This is a placeholder - you'll need to implement actual loading
  based on the TAPVid-3D dataset format.
  """
  # Expected format:
  # - video: [B, T, H, W, 3] RGB frames
  # - query_points: [B, Q, 4] (t, x, y, z) query points
  # - query_tracks: [B, Q, T, 3] ground truth 3D tracks
  # - query_tracks_visible: [B, Q, T, 1] visibility flags
  # - support_tracks: [B, N, T, 3] support 3D tracks
  # - support_tracks_visible: [B, N, T, 1] support visibility
  # - dino_features: [B, N, T, 768] optional DINO features
  # - depth_features: [B, N, T, 256] optional depth features
  
  # Placeholder - implement based on your dataset loader
  raise NotImplementedError(
      'Implement dataset loading based on TAPVid-3D format'
  )


def evaluate_model(
    params: Any,
    dataset,
    num_output_frames: int = 150,
    use_dino: bool = True,
    use_depth: bool = True,
    depth_scalings: List[str] = ['median'],
) -> Dict[str, Dict[str, float]]:
  """Evaluate model on full dataset using TAPVid-3D metrics.
  
  Args:
    params: Model parameters
    dataset: Dataset iterator
    num_output_frames: Number of output frames
    use_dino: Whether to use DINO features
    use_depth: Whether to use depth features
    depth_scalings: List of depth scaling strategies to evaluate
  
  Returns:
    Dictionary mapping depth scaling strategy to metrics dict
  """
  all_metrics = {scaling: [] for scaling in depth_scalings}
  
  for batch in tqdm.tqdm(dataset, desc='Evaluating'):
    # Get predictions
    _, predictions = evaluate_batch(
        params,
        batch,
        num_output_frames=num_output_frames,
        use_dino=use_dino,
        use_depth=use_depth,
    )
    
    # Convert to TAPVid-3D format
    query_points = np.array(batch['query_points'][0])  # [Q, 4]
    pred_tracks, pred_occluded = convert_predictions_to_tapvid3d_format(
        predictions, batch['query_points']
    )
    
    # Extract ground truth
    gt_tracks = np.array(batch['query_tracks'][0])  # [Q, T, 3]
    gt_tracks = np.transpose(gt_tracks, (1, 0, 2))  # [T, Q, 3]
    gt_visible = np.array(batch['query_tracks_visible'][0])  # [Q, T, 1]
    gt_occluded = np.logical_not(gt_visible[:, :, 0])  # [Q, T]
    gt_occluded = np.transpose(gt_occluded, (1, 0))  # [T, Q]
    
    # Get intrinsics
    if 'intrinsics' in batch:
      intrinsics = np.array(batch['intrinsics'][0])  # [4] (fx, fy, cx, cy)
    else:
      intrinsics = np.array([256.0, 256.0, 128.0, 128.0])
    
    # Compute metrics for each scaling strategy
    for scaling in depth_scalings:
      try:
        metrics_dict = tapvid3d_metrics.compute_tapvid3d_metrics(
            gt_occluded=gt_occluded,
            gt_tracks=gt_tracks,
            pred_occluded=pred_occluded,
            pred_tracks=pred_tracks,
            intrinsics_params=intrinsics,
            scaling=scaling,
            query_points=query_points[:, ::-1],  # Convert to (t, y, x) format
            order='t n',
        )
        # Convert to float
        metrics = {k: float(v) if isinstance(v, np.ndarray) else v 
                   for k, v in metrics_dict.items()}
        all_metrics[scaling].append(metrics)
      except Exception as e:
        logging.warning(f'Failed to compute metrics with scaling {scaling}: {e}')
        # Add zero metrics
        zero_metrics = {
            'occlusion_accuracy': 0.0,
            'pts_within_1': 0.0,
            'jaccard_1': 0.0,
            'pts_within_2': 0.0,
            'jaccard_2': 0.0,
            'pts_within_4': 0.0,
            'jaccard_4': 0.0,
            'pts_within_8': 0.0,
            'jaccard_8': 0.0,
            'pts_within_16': 0.0,
            'jaccard_16': 0.0,
            'average_jaccard': 0.0,
            'average_pts_within_thresh': 0.0,
        }
        all_metrics[scaling].append(zero_metrics)
  
  # Aggregate metrics across videos
  aggregated = {}
  for scaling in depth_scalings:
    aggregated[scaling] = {}
    if len(all_metrics[scaling]) == 0:
      continue
    for key in all_metrics[scaling][0].keys():
      values = [m[key] for m in all_metrics[scaling]]
      aggregated[scaling][key] = np.mean(values)
      aggregated[scaling][f'{key}_std'] = np.std(values)
  
  return aggregated


def load_checkpoint(checkpoint_path):
  """Load model checkpoint using Flax checkpoints."""
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
  state_dict = checkpoints.restore_checkpoint(checkpoint_path, target=None)
  if 'params' in state_dict:
    return state_dict['params']
  if 'optimizer' in state_dict and 'target' in state_dict['optimizer']:
    return state_dict['optimizer']['target']
  return state_dict


def main(argv):
  """Main evaluation function."""
  del argv
  
  if FLAGS.checkpoint_path is None:
    raise ValueError('Must provide checkpoint_path')
  if FLAGS.dataset_path is None:
    raise ValueError('Must provide dataset_path')
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  logging.info(f'Loading checkpoint from {FLAGS.checkpoint_path}')
  params = load_checkpoint(FLAGS.checkpoint_path)
  
  # Evaluate on each data source
  all_metrics = {}
  for data_source in FLAGS.data_sources:
    logging.info(f'Evaluating on {data_source} data source')
    
    # Get file list for this data source
    if FLAGS.use_minival:
      npz_files = tapvid3d_splits.get_minival_files(subset=data_source)
    else:
      npz_files = tapvid3d_splits.get_full_eval_files(subset=data_source)
    
    source_gt_dir = os.path.join(FLAGS.dataset_path, data_source)
    source_metrics = []
    
    # Evaluate each video
    for npz_file in tqdm.tqdm(npz_files, desc=f'Evaluating {data_source}'):
      gt_file = os.path.join(source_gt_dir, npz_file)
      
      # Load ground truth
      with open(gt_file, 'rb') as f:
        gt_data = np.load(f, allow_pickle=True)
        queries_xyt = gt_data['queries_xyt']
        tracks_xyz = gt_data['tracks_XYZ']
        visibles = gt_data['visibility']
        intrinsics_params = gt_data['fx_fy_cx_cy']
      
      # Prepare batch for model
      # Convert to model format: [B, Q, T, 3] for tracks
      batch = {
          'query_points': jnp.array(queries_xyt[np.newaxis]),  # [B, Q, 4] (t, x, y, z)
          'query_tracks': jnp.array(tracks_xyz[np.newaxis]),  # [B, Q, T, 3]
          'query_tracks_visible': jnp.array(visibles[np.newaxis, ..., np.newaxis]),  # [B, Q, T, 1]
          'intrinsics': jnp.array(intrinsics_params[np.newaxis]),  # [B, 4]
          'support_tracks': jnp.array(tracks_xyz[np.newaxis]),  # Use same as query for now
          'support_tracks_visible': jnp.array(visibles[np.newaxis, ..., np.newaxis]),
          'boundary_frame': jnp.array([tracks_xyz.shape[1]]),  # Number of frames
      }
      
      # Get predictions
      _, predictions = evaluate_batch(
          params,
          batch,
          num_output_frames=FLAGS.num_output_frames,
          use_dino=FLAGS.use_dino,
          use_depth=FLAGS.use_depth,
      )
      
      # Convert to TAPVid-3D format
      pred_tracks, pred_occluded = convert_predictions_to_tapvid3d_format(
          predictions, batch['query_points']
      )
      
      # Convert GT to [T, N, 3] format
      gt_tracks_tn = np.transpose(tracks_xyz, (1, 0, 2))  # [T, N, 3]
      gt_occluded = np.logical_not(visibles)  # [N, T]
      gt_occluded = np.transpose(gt_occluded, (1, 0))  # [T, N]
      
      # Compute metrics for each scaling strategy
      video_metrics = {}
      for scaling in FLAGS.depth_scalings:
        try:
          metrics_dict = tapvid3d_metrics.compute_tapvid3d_metrics(
              gt_occluded=gt_occluded,
              gt_tracks=gt_tracks_tn,
              pred_occluded=pred_occluded,
              pred_tracks=pred_tracks,
              intrinsics_params=intrinsics_params,
              scaling=scaling,
              query_points=queries_xyt[:, ::-1],  # Convert to (t, y, x) format
              order='t n',
          )
          # Convert to float
          metrics = {k: float(v) if isinstance(v, np.ndarray) else v 
                     for k, v in metrics_dict.items()}
          video_metrics[scaling] = metrics
        except Exception as e:
          logging.warning(f'Failed to compute metrics for {npz_file} with scaling {scaling}: {e}')
          # Add zero metrics
          video_metrics[scaling] = {
              'occlusion_accuracy': 0.0,
              'pts_within_1': 0.0,
              'jaccard_1': 0.0,
              'pts_within_2': 0.0,
              'jaccard_2': 0.0,
              'pts_within_4': 0.0,
              'jaccard_4': 0.0,
              'pts_within_8': 0.0,
              'jaccard_8': 0.0,
              'pts_within_16': 0.0,
              'jaccard_16': 0.0,
              'average_jaccard': 0.0,
              'average_pts_within_thresh': 0.0,
          }
      source_metrics.append(video_metrics)
    
    # Average metrics across videos for this data source
    all_metrics[data_source] = {}
    for scaling in FLAGS.depth_scalings:
      all_metrics[data_source][scaling] = {}
      if len(source_metrics) == 0:
        continue
      for key in source_metrics[0][scaling].keys():
        values = [m[scaling][key] for m in source_metrics]
        all_metrics[data_source][scaling][key] = np.mean(values)
        all_metrics[data_source][scaling][f'{key}_std'] = np.std(values)
    
    # Print results for this data source
    logging.info(f'Metrics for {data_source}:')
    for scaling in FLAGS.depth_scalings:
      logging.info(f'  Scaling: {scaling}')
      for key, value in all_metrics[data_source][scaling].items():
        if not key.endswith('_std'):
          logging.info(f'    {key}: {value:.4f}')
  
  # Compute overall average across all data sources
  overall_metrics = {}
  for scaling in FLAGS.depth_scalings:
    overall_metrics[scaling] = {}
    for key in all_metrics[FLAGS.data_sources[0]][scaling].keys():
      if not key.endswith('_std'):
        values = [all_metrics[ds][scaling][key] for ds in FLAGS.data_sources]
        overall_metrics[scaling][key] = np.mean(values)
        overall_metrics[scaling][f'{key}_std'] = np.std(values)
  
  logging.info('Overall metrics (averaged across all data sources):')
  for scaling in FLAGS.depth_scalings:
    logging.info(f'  Scaling: {scaling}')
    for key, value in overall_metrics[scaling].items():
      if not key.endswith('_std'):
        logging.info(f'    {key}: {value:.4f}')
  
  # Save results
  import json
  results_file = os.path.join(FLAGS.output_dir, 'results.json')
  with open(results_file, 'w') as f:
    json.dump({
        'per_source': all_metrics,
        'overall': overall_metrics,
    }, f, indent=2)
  
  logging.info(f'Results saved to {results_file}')
  logging.info('Evaluation completed')


if __name__ == '__main__':
  app.run(main)

