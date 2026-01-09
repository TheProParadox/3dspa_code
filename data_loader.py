"""Data loading utilities for TRAJAN and 3DSPA training."""

import functools
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


def prepare_2d_batch(example, num_support_tracks=2048, num_query_tracks=2048, num_frames=150):
  """Prepare 2D batch for TRAJAN training from example dict.
  
  Splits tracks into support/query sets and formats for model input.
  """
  tracks = example['tracks']  # [N, T, 2]
  visible = example['visible']  # [N, T, 1]
  
  # Sample support and query tracks
  num_total = tracks.shape[0]
  indices = np.random.permutation(num_total)
  
  support_indices = indices[:num_support_tracks]
  query_indices = indices[num_support_tracks:num_support_tracks + num_query_tracks]
  
  support_tracks = tracks[support_indices]  # [N, T, 2]
  support_visible = visible[support_indices]  # [N, T, 1]
  query_tracks = tracks[query_indices]  # [Q, T, 2]
  query_visible = visible[query_indices]  # [Q, T, 1]
  
  # Sample query points (random frame, random position)
  query_points = []
  for i in range(num_query_tracks):
    # Random frame
    t = np.random.randint(0, num_frames)
    # Position at that frame
    x, y = query_tracks[i, t, 0], query_tracks[i, t, 1]
    query_points.append([t, x, y])
  query_points = np.array(query_points)  # [Q, 3]
  
  # Add batch dimension
  batch = {
      'support_tracks': jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points': jnp.array(query_points[np.newaxis]),
      'query_tracks': jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible': jnp.array(query_visible[np.newaxis]),
      'boundary_frame': jnp.array([num_frames]),
  }
  
  return batch


def prepare_3d_batch(example, num_support_tracks=2048, num_query_tracks=2048, 
                     num_frames=150, use_dino=True, use_depth=True):
  """Prepare 3D batch for 3DSPA training with optional DINO and depth features.
  
  Splits 3D tracks into support/query sets and formats for model input.
  """
  tracks_3d = example['tracks_3d']  # [N, T, 3]
  visible = example['visible']  # [N, T, 1]
  
  # Sample support and query tracks
  num_total = tracks_3d.shape[0]
  indices = np.random.permutation(num_total)
  
  support_indices = indices[:num_support_tracks]
  query_indices = indices[num_support_tracks:num_support_tracks + num_query_tracks]
  
  support_tracks = tracks_3d[support_indices]  # [N, T, 3]
  support_visible = visible[support_indices]  # [N, T, 1]
  query_tracks = tracks_3d[query_indices]  # [Q, T, 3]
  query_visible = visible[query_indices]  # [Q, T, 1]
  
  # Sample query points (random frame, random position)
  query_points = []
  for i in range(num_query_tracks):
    # Random frame
    t = np.random.randint(0, num_frames)
    # Position at that frame
    x, y, z = query_tracks[i, t, 0], query_tracks[i, t, 1], query_tracks[i, t, 2]
    query_points.append([t, x, y, z])
  query_points = np.array(query_points)  # [Q, 4]
  
  # Add batch dimension
  batch = {
      'support_tracks': jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points': jnp.array(query_points[np.newaxis]),
      'query_tracks': jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible': jnp.array(query_visible[np.newaxis]),
      'boundary_frame': jnp.array([num_frames]),
  }
  
  # Add optional features
  if use_dino and 'dino_features' in example:
    dino_features = example['dino_features']  # [N, T, 768]
    batch['dino_features'] = jnp.array(
        dino_features[support_indices][np.newaxis]
    )
  
  if use_depth and 'depth_features' in example:
    depth_features = example['depth_features']  # [N, T, 256]
    batch['depth_features'] = jnp.array(
        depth_features[support_indices][np.newaxis]
    )
  
  return batch


def load_kubric3d_dataset(dataset_path, split='train', batch_size=64, shuffle=True,
                          num_support_tracks=2048, num_query_tracks=2048, 
                          num_frames=150, use_dino=True, use_depth=True):
  """Load Kubric3D dataset for 3DSPA training. Implement based on your dataset format."""
  # Expected format from Kubric3D:
  # - video: [T, H, W, 3] RGB frames
  # - tracks_3d: [N, T, 3] 3D point tracks
  # - visible: [N, T, 1] visibility flags
  # - dino_features: [N, T, 768] optional
  # - depth_features: [N, T, 256] optional
  
  def preprocess_fn(example):
    return prepare_3d_batch(
        example,
        num_support_tracks=num_support_tracks,
        num_query_tracks=num_query_tracks,
        num_frames=num_frames,
        use_dino=use_dino,
        use_depth=use_depth,
    )
  
  # Load dataset
  # ds = tfds.load('kubric3d', split=split, data_dir=dataset_path)
  # ds = ds.map(preprocess_fn)
  # 
  # if shuffle:
  #   ds = ds.shuffle(10000)
  # 
  # ds = ds.batch(batch_size)
  # ds = ds.prefetch(tfds.AUTOTUNE)
  # 
  # return ds
  
  raise NotImplementedError(
      'Implement based on your Kubric3D dataset format'
  )


def load_tapvid_dataset(dataset_path, split='train', batch_size=64, shuffle=True,
                        num_support_tracks=2048, num_query_tracks=2048, num_frames=150):
  """Load TAPVid dataset for TRAJAN training. Implement based on your dataset format."""
  def preprocess_fn(example):
    return prepare_2d_batch(
        example,
        num_support_tracks=num_support_tracks,
        num_query_tracks=num_query_tracks,
        num_frames=num_frames,
    )
  
  # Load dataset
  # ds = tfds.load('tapvid', split=split, data_dir=dataset_path)
  # ds = ds.map(preprocess_fn)
  # 
  # if shuffle:
  #   ds = ds.shuffle(10000)
  # 
  # ds = ds.batch(batch_size)
  # ds = ds.prefetch(tfds.AUTOTUNE)
  # 
  # return ds
  
  raise NotImplementedError(
      'Implement based on your TAPVid dataset format'
  )


def load_tapvid3d_dataset(dataset_path, split='minival', batch_size=8, shuffle=False):
  """Load TAPVid-3D dataset for evaluation. Implement based on TAPVid-3D format."""
  # Expected format from TAPVid-3D:
  # - video: [T, H, W, 3] RGB frames
  # - query_points: [Q, 4] (t, x, y, z) query points
  # - query_tracks: [Q, T, 3] ground truth 3D tracks
  # - query_tracks_visible: [Q, T, 1] visibility flags
  # - support_tracks: [N, T, 3] support 3D tracks
  # - support_tracks_visible: [N, T, 1] support visibility
  # - dino_features: [N, T, 768] optional
  # - depth_features: [N, T, 256] optional
  
  # Load dataset
  # ds = tfds.load('tapvid3d', split=split, data_dir=dataset_path)
  # 
  # if shuffle:
  #   ds = ds.shuffle(1000)
  # 
  # ds = ds.batch(batch_size)
  # ds = ds.prefetch(tfds.AUTOTUNE)
  # 
  # return ds
  
  raise NotImplementedError(
      'Implement based on TAPVid-3D dataset format'
  )

