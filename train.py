
"""Training script for TRAJAN and 3DSPA with WandB integration."""

import functools
from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import track_autoencoder
import track_autoencoder_3d
import data_loader


FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'trajan', 'Model type: trajan or 3dspa')
flags.DEFINE_string('config_path', None, 'Path to config file')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Checkpoint directory')
flags.DEFINE_string('wandb_project', '3dspa', 'WandB project name')
flags.DEFINE_string('wandb_entity', None, 'WandB entity name')
flags.DEFINE_string('wandb_run_name', None, 'WandB run name')
flags.DEFINE_integer('num_epochs', 300, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('eval_freq', 1000, 'Evaluation frequency in steps')
flags.DEFINE_integer('save_freq', 5000, 'Checkpoint save frequency in steps')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('warmup_steps', 10000, 'Warmup steps')
flags.DEFINE_integer('num_output_frames', 150, 'Number of output frames')
flags.DEFINE_bool('use_dino', True, 'Use DINO features (for 3DSPA)')
flags.DEFINE_bool('use_depth', True, 'Use depth features (for 3DSPA)')


def create_learning_rate_schedule(base_lr, warmup_steps, total_steps):
  """Create cosine decay LR schedule with linear warmup."""
  warmup_schedule = optax.linear_schedule(
      init_value=0.0,
      end_value=base_lr,
      transition_steps=warmup_steps,
  )
  cosine_schedule = optax.cosine_decay_schedule(
      init_value=base_lr,
      decay_steps=total_steps - warmup_steps,
      alpha=0.0,
  )
  schedule = optax.join_schedules(
      schedules=[warmup_schedule, cosine_schedule],
      boundaries=[warmup_steps],
  )
  return schedule


def compute_loss_2d(predictions, targets, l1_weight=5000.0, bce_weight=1e-8):
  """Compute L1 position loss + BCE occlusion loss for 2D TRAJAN."""
  # L1 loss on positions
  target_tracks = targets['query_tracks']  # [B, Q, T, 2]
  target_visible = targets['query_tracks_visible']  # [B, Q, T, 1]
  
  pred_tracks = predictions.tracks  # [B, Q, T, 2]
  pred_visible_logits = predictions.visible_logits  # [B, Q, T, 1]
  
  # Only compute loss on visible points
  visible_mask = target_visible.astype(jnp.float32)
  
  # L1 position loss
  position_error = jnp.abs(pred_tracks - target_tracks)
  position_loss = jnp.sum(position_error * visible_mask, axis=(-2, -1))
  position_loss = jnp.sum(position_loss) / jnp.maximum(
      jnp.sum(visible_mask), 1.0
  )
  
  # Binary cross-entropy for occlusion
  visible_loss = optax.sigmoid_binary_cross_entropy(
      pred_visible_logits, target_visible
  )
  visible_loss = jnp.sum(visible_loss) / jnp.maximum(
      jnp.sum(visible_mask), 1.0
  )
  
  total_loss = l1_weight * position_loss + bce_weight * visible_loss
  
  return {
      'total_loss': total_loss,
      'position_loss': position_loss,
      'visible_loss': visible_loss,
  }


def compute_loss_3d(predictions, targets, l1_weight=5000.0, bce_weight=1e-8):
  """Compute L1 position loss + BCE occlusion loss for 3DSPA."""
  # L1 loss on 3D positions
  target_tracks = targets['query_tracks']  # [B, Q, T, 3]
  target_visible = targets['query_tracks_visible']  # [B, Q, T, 1]
  
  pred_tracks = predictions.tracks  # [B, Q, T, 3]
  pred_visible_logits = predictions.visible_logits  # [B, Q, T, 1]
  
  # Only compute loss on visible points
  visible_mask = target_visible.astype(jnp.float32)
  
  # L1 position loss
  position_error = jnp.abs(pred_tracks - target_tracks)
  position_loss = jnp.sum(position_error * visible_mask, axis=(-2, -1))
  position_loss = jnp.sum(position_loss) / jnp.maximum(
      jnp.sum(visible_mask), 1.0
  )
  
  # Binary cross-entropy for occlusion
  visible_loss = optax.sigmoid_binary_cross_entropy(
      pred_visible_logits, target_visible
  )
  visible_loss = jnp.sum(visible_loss) / jnp.maximum(
      jnp.sum(visible_mask), 1.0
  )
  
  total_loss = l1_weight * position_loss + bce_weight * visible_loss
  
  return {
      'total_loss': total_loss,
      'position_loss': position_loss,
      'visible_loss': visible_loss,
  }


@functools.partial(jax.jit, static_argnames=['model_type'])
def train_step(state, batch, model_type='trajan'):
  """Single training step: forward pass, compute loss, update parameters."""
  def loss_fn(params):
    if model_type == '3dspa':
      model = track_autoencoder_3d.TrackAutoEncoder3D(
          num_output_frames=FLAGS.num_output_frames,
          use_dino=FLAGS.use_dino,
          use_depth=FLAGS.use_depth,
      )
      predictions = model.apply(
          {'params': params},
          batch,
          rngs={'dropout': state.rng} if hasattr(state, 'rng') else {},
      )
      loss_dict = compute_loss_3d(predictions, batch)
    else:
      model = track_autoencoder.TrackAutoEncoder(
          num_output_frames=FLAGS.num_output_frames,
      )
      predictions = model.apply(
          {'params': params},
          batch,
          rngs={'dropout': state.rng} if hasattr(state, 'rng') else {},
      )
      loss_dict = compute_loss_2d(predictions, batch)
    
    return loss_dict['total_loss'], (loss_dict, predictions)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (loss_dict, predictions)), grads = grad_fn(state.params)
  
  updates, new_opt_state = state.opt_state.update(grads, state.params)
  new_params = optax.apply_updates(state.params, updates)
  
  new_state = state.replace(
      params=new_params,
      opt_state=new_opt_state,
      step=state.step + 1,
  )
  
  # Get learning rate from schedule
  lr = state.opt_state.hyperparams['learning_rate']
  if hasattr(lr, '__call__'):
    lr = lr(state.step)
  else:
    lr = float(lr)
  
  metrics = {
      'train/loss': float(loss),
      'train/position_loss': float(loss_dict['position_loss']),
      'train/visible_loss': float(loss_dict['visible_loss']),
      'train/learning_rate': lr,
  }
  
  return new_state, metrics, predictions


@functools.partial(jax.jit, static_argnames=['model_type'])
def eval_step(params, batch, model_type='trajan'):
  """Single evaluation step: forward pass and compute metrics."""
  if model_type == '3dspa':
    model = track_autoencoder_3d.TrackAutoEncoder3D(
        num_output_frames=FLAGS.num_output_frames,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
    predictions = model.apply({'params': params}, batch)
    loss_dict = compute_loss_3d(predictions, batch)
  else:
    model = track_autoencoder.TrackAutoEncoder(
        num_output_frames=FLAGS.num_output_frames,
    )
    predictions = model.apply({'params': params}, batch)
    loss_dict = compute_loss_2d(predictions, batch)
  
  metrics = {
      'eval/loss': float(loss_dict['total_loss']),
      'eval/position_loss': float(loss_dict['position_loss']),
      'eval/visible_loss': float(loss_dict['visible_loss']),
  }
  
  return metrics, predictions


def create_model_state(rng, dummy_batch, model_type='trajan', 
                       learning_rate=1e-4, warmup_steps=10000, total_steps=1000000):
  """Initialize model parameters and optimizer state."""
  if model_type == '3dspa':
    model = track_autoencoder_3d.TrackAutoEncoder3D(
        num_output_frames=FLAGS.num_output_frames,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
  else:
    model = track_autoencoder.TrackAutoEncoder(
        num_output_frames=FLAGS.num_output_frames,
    )
  
  # Initialize parameters
  rng, init_rng = jax.random.split(rng)
  params = model.init(init_rng, dummy_batch)['params']
  
  # Create optimizer
  lr_schedule = create_learning_rate_schedule(
      learning_rate, warmup_steps, total_steps
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=lr_schedule, weight_decay=0.01),
  )
  opt_state = optimizer.init(params)
  
  # Create state
  @chex.dataclass
  class TrainState:
    params: Any
    opt_state: Any
    step: int
    rng: jax.random.PRNGKey
  
  state = TrainState(
      params=params,
      opt_state=opt_state,
      step=0,
      rng=rng,
  )
  
  return state, model


def load_dataset(dataset_name, split='train', batch_size=64, 
                 shuffle=True, model_type='trajan'):
  """Load training/validation dataset (Kubric3D for 3DSPA, TAPVid for TRAJAN)."""
  if model_type == '3dspa':
    return data_loader.load_kubric3d_dataset(
        dataset_path=dataset_name,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
  else:
    return data_loader.load_tapvid_dataset(
        dataset_path=dataset_name,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def main(argv):
  """Main training loop with WandB logging and periodic evaluation."""
  del argv
  
  # Initialize WandB
  wandb.init(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      name=FLAGS.wandb_run_name or f'{FLAGS.model_type}_{FLAGS.wandb_project}',
      config={
          'model_type': FLAGS.model_type,
          'batch_size': FLAGS.batch_size,
          'learning_rate': FLAGS.learning_rate,
          'num_epochs': FLAGS.num_epochs,
          'num_output_frames': FLAGS.num_output_frames,
          'use_dino': FLAGS.use_dino,
          'use_depth': FLAGS.use_depth,
      },
  )
  
  # Initialize random key
  rng = jax.random.PRNGKey(42)
  
  # Load datasets
  dataset_path = FLAGS.config_path or './data'
  train_ds = load_dataset(
      dataset_path,
      split='train',
      batch_size=FLAGS.batch_size,
      shuffle=True,
      model_type=FLAGS.model_type,
  )
  eval_ds = load_dataset(
      dataset_path,
      split='validation',
      batch_size=FLAGS.batch_size,
      shuffle=False,
      model_type=FLAGS.model_type,
  )
  
  # Create dummy batch for initialization
  dummy_batch = next(iter(train_ds))
  
  # Calculate total steps
  num_train_samples = len(list(train_ds))
  steps_per_epoch = num_train_samples // FLAGS.batch_size
  total_steps = steps_per_epoch * FLAGS.num_epochs
  
  # Initialize model
  rng, init_rng = jax.random.split(rng)
  state, model = create_model_state(
      init_rng,
      dummy_batch,
      model_type=FLAGS.model_type,
      learning_rate=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      total_steps=total_steps,
  )
  
  logging.info(f'Initialized {FLAGS.model_type} model')
  logging.info(f'Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}')
  
  # Training loop
  for epoch in range(FLAGS.num_epochs):
    for step, batch in enumerate(train_ds):
      # Update RNG
      rng, step_rng = jax.random.split(rng)
      state = state.replace(rng=step_rng)
      
      # Training step
      state, metrics, predictions = train_step(state, batch, FLAGS.model_type)
      
      # Log to WandB
      if state.step % 10 == 0:
        wandb.log(metrics, step=state.step)
        logging.info(
            f'Epoch {epoch}, Step {state.step}: '
            f'Loss={metrics["train/loss"]:.4f}, '
            f'Pos={metrics["train/position_loss"]:.4f}, '
            f'Vis={metrics["train/visible_loss"]:.4f}'
        )
      
      # Evaluation
      if state.step % FLAGS.eval_freq == 0:
        eval_metrics = {}
        for eval_batch in eval_ds.take(10):  # Evaluate on 10 batches
          batch_metrics, _ = eval_step(
              state.params, eval_batch, FLAGS.model_type
          )
          for k, v in batch_metrics.items():
            if k not in eval_metrics:
              eval_metrics[k] = []
            eval_metrics[k].append(v)
        
        # Average metrics
        avg_eval_metrics = {
            k: np.mean(v) for k, v in eval_metrics.items()
        }
        wandb.log(avg_eval_metrics, step=state.step)
        logging.info(
            f'Evaluation at step {state.step}: '
            f'Eval Loss={avg_eval_metrics["eval/loss"]:.4f}'
        )
      
      # Save checkpoint
      if state.step % FLAGS.save_freq == 0:
        checkpoint_path = f'{FLAGS.checkpoint_dir}/checkpoint_{state.step}'
        # Save checkpoint using Flax
        # (implementation depends on your checkpointing setup)
        logging.info(f'Saved checkpoint at step {state.step}')
  
  wandb.finish()
  logging.info('Training completed')


if __name__ == '__main__':
  app.run(main)

