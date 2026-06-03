import os
from typing import Any, Callable, Dict, Iterator, List

import jax.numpy as jnp
import numpy as np


class IterableBatches:
  def __init__(self, batches: List[Dict[str, Any]], shuffle: bool = False):
    self._batches = batches
    self._shuffle = shuffle

  def __iter__(self) -> Iterator[Dict[str, Any]]:
    order = np.arange(len(self._batches))
    if self._shuffle:
      np.random.shuffle(order)
    for i in order:
      yield self._batches[i]

  def __len__(self) -> int:
    return len(self._batches)

  def take(self, n: int) -> 'IterableBatches':
    return IterableBatches(self._batches[:n], shuffle=False)


def _split_root(dataset_path: str, split: str) -> str:
  candidate = os.path.join(dataset_path, split)
  if os.path.isdir(candidate):
    return candidate
  return dataset_path


def _collect_npz_files(root: str) -> List[str]:
  if not os.path.isdir(root):
    raise FileNotFoundError(f'Dataset directory not found: {root}')
  paths = []
  for dirpath, _, filenames in os.walk(root):
    for name in filenames:
      if name.endswith('.npz'):
        paths.append(os.path.join(dirpath, name))
  paths.sort()
  if not paths:
    raise FileNotFoundError(f'No .npz files under {root}')
  return paths


def _trim_time(example: Dict[str, np.ndarray], num_frames: int,
               keys: List[str]) -> Dict[str, np.ndarray]:
  out = dict(example)
  for key in keys:
    if key not in out:
      continue
    arr = out[key]
    if arr.ndim >= 2 and arr.shape[1] > num_frames:
      out[key] = arr[:, :num_frames]
  return out


def _visible_to_nt1(visible: np.ndarray) -> np.ndarray:
  if visible.ndim == 3:
    return visible.astype(np.float32)
  if visible.ndim == 2:
    return visible[..., np.newaxis].astype(np.float32)
  raise ValueError(f'visibility must be [N,T] or [N,T,1], got {visible.shape}')


def prepare_2d_batch(example, num_support_tracks=2048, num_query_tracks=2048,
                     num_frames=150):
  tracks = example['tracks']
  visible = example['visible']

  N = tracks.shape[0]
  indices = np.random.permutation(N)
  sup_idx = indices[:num_support_tracks]
  qry_idx = indices[num_support_tracks:num_support_tracks + num_query_tracks]

  support_tracks = tracks[sup_idx]
  support_visible = visible[sup_idx]
  query_tracks = tracks[qry_idx]
  query_visible = visible[qry_idx]

  t_max = min(num_frames, query_tracks.shape[1])
  query_points = np.zeros((len(qry_idx), 3), dtype=np.float32)
  for i in range(len(qry_idx)):
    t = np.random.randint(0, t_max)
    query_points[i] = [t, query_tracks[i, t, 0], query_tracks[i, t, 1]]

  return {
      'support_tracks': jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points': jnp.array(query_points[np.newaxis]),
      'query_tracks': jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible': jnp.array(query_visible[np.newaxis]),
      'boundary_frame': jnp.array([t_max]),
  }


def prepare_3d_batch(example, num_support_tracks=2048, num_query_tracks=2048,
                     num_frames=150, use_dino=True, use_depth=True):
  tracks_3d = example['tracks_3d']
  visible = example['visible']

  N = tracks_3d.shape[0]
  indices = np.random.permutation(N)
  sup_idx = indices[:num_support_tracks]
  qry_idx = indices[num_support_tracks:num_support_tracks + num_query_tracks]

  support_tracks = tracks_3d[sup_idx]
  support_visible = visible[sup_idx]
  query_tracks = tracks_3d[qry_idx]
  query_visible = visible[qry_idx]

  t_max = min(num_frames, query_tracks.shape[1])
  query_points = np.zeros((len(qry_idx), 4), dtype=np.float32)
  for i in range(len(qry_idx)):
    t = np.random.randint(0, t_max)
    query_points[i] = [
        t,
        query_tracks[i, t, 0],
        query_tracks[i, t, 1],
        query_tracks[i, t, 2],
    ]

  batch = {
      'support_tracks': jnp.array(support_tracks[np.newaxis]),
      'support_tracks_visible': jnp.array(support_visible[np.newaxis]),
      'query_points': jnp.array(query_points[np.newaxis]),
      'query_tracks': jnp.array(query_tracks[np.newaxis]),
      'query_tracks_visible': jnp.array(query_visible[np.newaxis]),
      'boundary_frame': jnp.array([t_max]),
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


def _load_2d_npz(path: str) -> Dict[str, np.ndarray]:
  data = np.load(path, allow_pickle=True)
  if 'tracks' in data.files:
    tracks = np.array(data['tracks'], dtype=np.float32)
  elif 'target_points' in data.files:
    tracks = np.array(data['target_points'], dtype=np.float32)
  else:
    raise KeyError(f'{path}: expected tracks or target_points')

  if 'visible' in data.files:
    visible = np.array(data['visible'])
  elif 'visibility' in data.files:
    visible = np.array(data['visibility'])
  else:
    raise KeyError(f'{path}: expected visible or visibility')

  return {
      'tracks': tracks,
      'visible': _visible_to_nt1(visible),
  }


def _load_3d_npz(path: str) -> Dict[str, np.ndarray]:
  data = np.load(path, allow_pickle=True)
  if 'tracks_3d' in data.files:
    tracks_3d = np.array(data['tracks_3d'], dtype=np.float32)
  elif 'tracks_XYZ' in data.files:
    tracks_3d = np.array(data['tracks_XYZ'], dtype=np.float32)
  else:
    raise KeyError(f'{path}: expected tracks_3d or tracks_XYZ')

  if 'visible' in data.files:
    visible = np.array(data['visible'])
  elif 'visibility' in data.files:
    visible = np.array(data['visibility'])
  else:
    raise KeyError(f'{path}: expected visible or visibility')

  example = {
      'tracks_3d': tracks_3d,
      'visible': _visible_to_nt1(visible),
  }
  if 'dino_features' in data.files:
    example['dino_features'] = np.array(data['dino_features'], dtype=np.float32)
  if 'depth_features' in data.files:
    example['depth_features'] = np.array(
        data['depth_features'], dtype=np.float32,
    )
  return example


def _build_training_dataset(
    dataset_path: str,
    split: str,
    shuffle: bool,
    load_example: Callable[[str], Dict[str, np.ndarray]],
    prepare_fn: Callable[..., Dict[str, Any]],
    time_keys: List[str],
    prepare_kwargs: Dict[str, Any],
) -> IterableBatches:
  root = _split_root(dataset_path, split)
  num_frames = prepare_kwargs.get('num_frames', 150)
  batches = []
  for path in _collect_npz_files(root):
    example = _trim_time(load_example(path), num_frames, time_keys)
    batches.append(prepare_fn(example, **prepare_kwargs))
  return IterableBatches(batches, shuffle=shuffle)


def load_kubric3d_dataset(dataset_path, split='train', batch_size=256,
                          shuffle=True, num_support_tracks=2048,
                          num_query_tracks=2048, num_frames=150,
                          use_dino=True, use_depth=True):
  del batch_size
  return _build_training_dataset(
      dataset_path=dataset_path,
      split=split,
      shuffle=shuffle,
      load_example=_load_3d_npz,
      prepare_fn=prepare_3d_batch,
      time_keys=['tracks_3d', 'visible', 'dino_features', 'depth_features'],
      prepare_kwargs={
          'num_support_tracks': num_support_tracks,
          'num_query_tracks': num_query_tracks,
          'num_frames': num_frames,
          'use_dino': use_dino,
          'use_depth': use_depth,
      },
  )


def load_tapvid_dataset(dataset_path, split='train', batch_size=256,
                        shuffle=True, num_support_tracks=2048,
                        num_query_tracks=2048, num_frames=150):
  del batch_size
  return _build_training_dataset(
      dataset_path=dataset_path,
      split=split,
      shuffle=shuffle,
      load_example=_load_2d_npz,
      prepare_fn=prepare_2d_batch,
      time_keys=['tracks', 'visible'],
      prepare_kwargs={
          'num_support_tracks': num_support_tracks,
          'num_query_tracks': num_query_tracks,
          'num_frames': num_frames,
      },
  )


def _load_tapvid3d_eval_npz(path: str) -> Dict[str, Any]:
  data = np.load(path, allow_pickle=True)
  queries_xyt = np.array(data['queries_xyt'], dtype=np.float32)
  tracks_xyz = np.array(data['tracks_XYZ'], dtype=np.float32)
  visibles = np.array(data['visibility'])
  intrinsics_params = np.array(data['fx_fy_cx_cy'], dtype=np.float32)
  t_len = tracks_xyz.shape[1]
  return {
      'query_points': jnp.array(queries_xyt[np.newaxis]),
      'query_tracks': jnp.array(tracks_xyz[np.newaxis]),
      'query_tracks_visible': jnp.array(
          visibles[np.newaxis, ..., np.newaxis], dtype=np.float32,
      ),
      'intrinsics': jnp.array(intrinsics_params[np.newaxis]),
      'support_tracks': jnp.array(tracks_xyz[np.newaxis]),
      'support_tracks_visible': jnp.array(
          visibles[np.newaxis, ..., np.newaxis], dtype=np.float32,
      ),
      'boundary_frame': jnp.array([t_len]),
  }


def load_tapvid3d_dataset(dataset_path, split='minival', batch_size=8,
                          shuffle=False, data_sources=None):
  del batch_size
  if data_sources is None:
    data_sources = ['drivetrack', 'adt', 'pstudio']

  try:
    from tapnet.tapvid3d.splits import tapvid3d_splits
    use_splits = True
  except ImportError:
    tapvid3d_splits = None
    use_splits = False

  batches = []
  if use_splits and split in ('minival', 'full_eval'):
    get_files = (
        tapvid3d_splits.get_minival_files
        if split == 'minival'
        else tapvid3d_splits.get_full_eval_files
    )
    for source in data_sources:
      source_dir = os.path.join(dataset_path, source)
      for npz_name in get_files(subset=source):
        path = os.path.join(source_dir, npz_name)
        if os.path.isfile(path):
          batches.append(_load_tapvid3d_eval_npz(path))
  else:
    root = _split_root(dataset_path, split)
    for path in _collect_npz_files(root):
      batches.append(_load_tapvid3d_eval_npz(path))

  if not batches:
    raise FileNotFoundError(
        f'No TAPVid-3D examples found under {dataset_path} (split={split})',
    )
  return IterableBatches(batches, shuffle=shuffle)
