"""Data loading utilities for 3DSPA training.

Provides batch preparation for two model types:
  - 2D batches for TRAJAN (tracks [N, T, 2])
  - 3D batches for 3DSPA  (tracks [N, T, 3] + optional DINO / depth)

Dataset loaders must be implemented for your storage format.
"""

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np


def prepare_2d_batch(example, num_support_tracks=2048, num_query_tracks=2048,
                     num_frames=150):
  """Split 2D tracks into support / query sets for TRAJAN training."""
  tracks = example['tracks']     # [N, T, 2]
  visible = example['visible']   # [N, T, 1]

  N = tracks.shape[0]
  indices = np.random.permutation(N)
  sup_idx = indices[:num_support_tracks]
  qry_idx = indices[num_support_tracks:num_support_tracks + num_query_tracks]

  support_tracks = tracks[sup_idx]
  support_visible = visible[sup_idx]
  query_tracks = tracks[qry_idx]
  query_visible = visible[qry_idx]

  query_points = np.zeros((len(qry_idx), 3), dtype=np.float32)
  for i in range(len(qry_idx)):
    t = np.random.randint(0, num_frames)
    query_points[i] = [t, query_tracks[i, t, 0], query_tracks[i, t, 1]]

  return {
      'support_tracks':         jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points':           jnp.array(query_points[np.newaxis]),
      'query_tracks':           jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible':   jnp.array(query_visible[np.newaxis]),
      'boundary_frame':         jnp.array([num_frames]),
  }


def prepare_3d_batch(example, num_support_tracks=2048, num_query_tracks=2048,
                     num_frames=150, use_dino=True, use_depth=True):
  """Split 3D tracks into support / query sets for 3DSPA training."""
  tracks_3d = example['tracks_3d']  # [N, T, 3]
  visible = example['visible']      # [N, T, 1]

  N = tracks_3d.shape[0]
  indices = np.random.permutation(N)
  sup_idx = indices[:num_support_tracks]
  qry_idx = indices[num_support_tracks:num_support_tracks + num_query_tracks]

  support_tracks = tracks_3d[sup_idx]
  support_visible = visible[sup_idx]
  query_tracks = tracks_3d[qry_idx]
  query_visible = visible[qry_idx]

  query_points = np.zeros((len(qry_idx), 4), dtype=np.float32)
  for i in range(len(qry_idx)):
    t = np.random.randint(0, num_frames)
    query_points[i] = [
        t,
        query_tracks[i, t, 0],
        query_tracks[i, t, 1],
        query_tracks[i, t, 2],
    ]

  batch = {
      'support_tracks':         jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points':           jnp.array(query_points[np.newaxis]),
      'query_tracks':           jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible':   jnp.array(query_visible[np.newaxis]),
      'boundary_frame':         jnp.array([num_frames]),
  }

  if use_dino and 'dino_features' in example:
    batch['dino_features'] = jnp.array(
        example['dino_features'][sup_idx][np.newaxis],
    )
  if use_depth and 'depth_features' in example:
    batch['depth_features'] = jnp.array(
        example['depth_features'][sup_idx][np.newaxis],
    )

  return batch


def load_kubric3d_dataset(dataset_path, split='train', batch_size=256,
                          shuffle=True, num_support_tracks=2048,
                          num_query_tracks=2048, num_frames=150,
                          use_dino=True, use_depth=True):
  """Load Kubric3D dataset for 3DSPA training.

  Expected fields per example:
    tracks_3d:      [N, T, 3]
    visible:        [N, T, 1]
    dino_features:  [N, T, 768]  (optional)
    depth_features: [N, T, 256]  (optional)
  """
  raise NotImplementedError(
      'Implement Kubric3D loading based on your dataset format.'
  )


def load_tapvid_dataset(dataset_path, split='train', batch_size=256,
                        shuffle=True, num_support_tracks=2048,
                        num_query_tracks=2048, num_frames=150):
  """Load TAPVid dataset for TRAJAN training.

  Expected fields per example:
    tracks:   [N, T, 2]
    visible:  [N, T, 1]
  """
  raise NotImplementedError(
      'Implement TAPVid loading based on your dataset format.'
  )


def load_tapvid3d_dataset(dataset_path, split='minival', batch_size=8,
                          shuffle=False):
  """Load TAPVid-3D dataset for evaluation.

  Expected fields per example:
    queries_xyt:  [Q, 4]
    tracks_XYZ:   [Q, T, 3]
    visibility:   [Q, T]
    fx_fy_cx_cy:  [4]
  """
  raise NotImplementedError(
      'Implement TAPVid-3D loading based on your dataset format.'
  )
