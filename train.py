"""3DSPA / TRAJAN training with WandB integration."""

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

flags.DEFINE_string('model_type', '3dspa', 'Model type: 3dspa | trajan')
flags.DEFINE_string('dataset_path', './data', 'Path to training data')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Checkpoint directory')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to pretrained checkpoint for initialization '
                    '(e.g. TRAJAN weights when training 3DSPA)')
flags.DEFINE_string('wandb_project', '3dspa', 'WandB project name')
flags.DEFINE_string('wandb_entity', None, 'WandB entity name')
flags.DEFINE_string('wandb_run_name', None, 'WandB run name')
flags.DEFINE_integer('num_epochs', 300, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('eval_freq', 1000, 'Evaluation frequency (steps)')
flags.DEFINE_integer('save_freq', 5000, 'Checkpoint save frequency (steps)')
flags.DEFINE_float('learning_rate', 1e-4, 'Peak learning rate')
flags.DEFINE_integer('warmup_steps', 10000, 'Linear warmup steps')
flags.DEFINE_integer('num_output_frames', 150, 'Number of output frames')
flags.DEFINE_bool('use_dino', True, 'Use DINOv2 features (3DSPA only)')
flags.DEFINE_bool('use_depth', True, 'Use depth features (3DSPA only)')


# ---- Learning rate schedule ---------------------------------------------------

def create_learning_rate_schedule(base_lr, warmup_steps, total_steps):
  """Linear warmup followed by cosine decay to zero."""
  return optax.join_schedules(
      schedules=[
          optax.linear_schedule(0.0, base_lr, warmup_steps),
          optax.cosine_decay_schedule(base_lr, total_steps - warmup_steps,
                                      alpha=0.0),
      ],
      boundaries=[warmup_steps],
  )


# ---- Loss functions -----------------------------------------------------------

def compute_loss_2d(predictions, targets, l1_weight=5000.0, bce_weight=1e-8):
  """L1 position + BCE occlusion loss for 2D tracks (TRAJAN)."""
  target_tracks = targets['query_tracks']
  target_visible = targets['query_tracks_visible']
  pred_tracks = predictions.tracks
  pred_visible_logits = predictions.visible_logits

  visible_mask = target_visible.astype(jnp.float32)

  position_error = jnp.abs(pred_tracks - target_tracks)
  position_loss = (
      jnp.sum(position_error * visible_mask)
      / jnp.maximum(jnp.sum(visible_mask), 1.0)
  )

  visible_loss = optax.sigmoid_binary_cross_entropy(
      pred_visible_logits, target_visible,
  )
  visible_loss = jnp.sum(visible_loss) / jnp.maximum(jnp.sum(visible_mask), 1.0)

  total_loss = l1_weight * position_loss + bce_weight * visible_loss
  return {
      'total_loss': total_loss,
      'position_loss': position_loss,
      'visible_loss': visible_loss,
  }


def compute_loss_3d(predictions, targets, l1_weight=5000.0, bce_weight=1e-8):
  """L1 position + BCE occlusion loss for 3D tracks (3DSPA)."""
  target_tracks = targets['query_tracks']
  target_visible = targets['query_tracks_visible']
  pred_tracks = predictions.tracks
  pred_visible_logits = predictions.visible_logits

  visible_mask = target_visible.astype(jnp.float32)

  position_error = jnp.abs(pred_tracks - target_tracks)
  position_loss = (
      jnp.sum(position_error * visible_mask)
      / jnp.maximum(jnp.sum(visible_mask), 1.0)
  )

  visible_loss = optax.sigmoid_binary_cross_entropy(
      pred_visible_logits, target_visible,
  )
  visible_loss = jnp.sum(visible_loss) / jnp.maximum(jnp.sum(visible_mask), 1.0)

  total_loss = l1_weight * position_loss + bce_weight * visible_loss
  return {
      'total_loss': total_loss,
      'position_loss': position_loss,
      'visible_loss': visible_loss,
  }


# ---- Train / eval steps ------------------------------------------------------

def _build_model(model_type):
  if model_type == '3dspa':
    return track_autoencoder_3d.TrackAutoEncoder3D(
        num_output_frames=FLAGS.num_output_frames,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
  return track_autoencoder.TrackAutoEncoder(
      num_output_frames=FLAGS.num_output_frames,
  )


@functools.partial(jax.jit, static_argnames=['model_type'])
def train_step(state, batch, model_type='3dspa'):
  """Forward pass, loss computation, and parameter update."""
  compute_loss = compute_loss_3d if model_type == '3dspa' else compute_loss_2d
  model = _build_model(model_type)

  def loss_fn(params):
    predictions = model.apply(
        {'params': params}, batch,
        rngs={'dropout': state.rng} if hasattr(state, 'rng') else {},
    )
    loss_dict = compute_loss(predictions, batch)
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

  metrics = {
      'train/loss': float(loss),
      'train/position_loss': float(loss_dict['position_loss']),
      'train/visible_loss': float(loss_dict['visible_loss']),
  }
  return new_state, metrics, predictions


@functools.partial(jax.jit, static_argnames=['model_type'])
def eval_step(params, batch, model_type='3dspa'):
  """Forward pass and metric computation (no gradient)."""
  compute_loss = compute_loss_3d if model_type == '3dspa' else compute_loss_2d
  model = _build_model(model_type)
  predictions = model.apply({'params': params}, batch)
  loss_dict = compute_loss(predictions, batch)

  metrics = {
      'eval/loss': float(loss_dict['total_loss']),
      'eval/position_loss': float(loss_dict['position_loss']),
      'eval/visible_loss': float(loss_dict['visible_loss']),
  }
  return metrics, predictions


# ---- Model state creation -----------------------------------------------------

def create_model_state(rng, dummy_batch, model_type='3dspa',
                       learning_rate=1e-4, warmup_steps=10000,
                       total_steps=1_000_000):
  """Initialize model parameters and AdamW optimizer with cosine schedule."""
  model = _build_model(model_type)

  rng, init_rng = jax.random.split(rng)
  params = model.init(init_rng, dummy_batch)['params']

  lr_schedule = create_learning_rate_schedule(
      learning_rate, warmup_steps, total_steps,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=lr_schedule, weight_decay=0.01),
  )
  opt_state = optimizer.init(params)

  @chex.dataclass
  class TrainState:
    params: Any
    opt_state: Any
    step: int
    rng: jax.random.PRNGKey

  return TrainState(params=params, opt_state=opt_state, step=0, rng=rng), model


# ---- Dataset loading ----------------------------------------------------------

def load_dataset(dataset_path, split='train', batch_size=256,
                 shuffle=True, model_type='3dspa'):
  """Route to the appropriate dataset loader."""
  if model_type == '3dspa':
    return data_loader.load_kubric3d_dataset(
        dataset_path=dataset_path,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        use_dino=FLAGS.use_dino,
        use_depth=FLAGS.use_depth,
    )
  return data_loader.load_tapvid_dataset(
      dataset_path=dataset_path,
      split=split,
      batch_size=batch_size,
      shuffle=shuffle,
  )


# ---- Main training loop ------------------------------------------------------

def main(argv):
  del argv

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

  rng = jax.random.PRNGKey(42)

  train_ds = load_dataset(
      FLAGS.dataset_path, split='train',
      batch_size=FLAGS.batch_size, shuffle=True,
      model_type=FLAGS.model_type,
  )
  eval_ds = load_dataset(
      FLAGS.dataset_path, split='validation',
      batch_size=FLAGS.batch_size, shuffle=False,
      model_type=FLAGS.model_type,
  )

  if len(train_ds) == 0:
    raise ValueError(f'No training examples found under {FLAGS.dataset_path}')

  dummy_batch = next(iter(train_ds))
  steps_per_epoch = len(train_ds)
  total_steps = steps_per_epoch * FLAGS.num_epochs

  rng, init_rng = jax.random.split(rng)
  state, model = create_model_state(
      init_rng, dummy_batch,
      model_type=FLAGS.model_type,
      learning_rate=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      total_steps=total_steps,
  )

  logging.info(f'Model: {FLAGS.model_type}')
  logging.info(f'Total steps: {total_steps}, steps/epoch: {steps_per_epoch}')

  for epoch in range(FLAGS.num_epochs):
    for step, batch in enumerate(train_ds):
      rng, step_rng = jax.random.split(rng)
      state = state.replace(rng=step_rng)

      state, metrics, _ = train_step(state, batch, FLAGS.model_type)

      if state.step % 10 == 0:
        wandb.log(metrics, step=state.step)
        logging.info(
            f'Epoch {epoch}, Step {state.step}: '
            f'Loss={metrics["train/loss"]:.4f}, '
            f'Pos={metrics["train/position_loss"]:.4f}, '
            f'Vis={metrics["train/visible_loss"]:.4f}'
        )

      if state.step % FLAGS.eval_freq == 0:
        eval_metrics = {}
        for eval_batch in eval_ds.take(10):
          batch_metrics, _ = eval_step(
              state.params, eval_batch, FLAGS.model_type,
          )
          for k, v in batch_metrics.items():
            eval_metrics.setdefault(k, []).append(v)

        avg = {k: np.mean(v) for k, v in eval_metrics.items()}
        wandb.log(avg, step=state.step)
        logging.info(
            f'Eval step {state.step}: Loss={avg["eval/loss"]:.4f}'
        )

      if state.step % FLAGS.save_freq == 0:
        ckpt_path = f'{FLAGS.checkpoint_dir}/checkpoint_{state.step}'
        logging.info(f'Saved checkpoint at step {state.step}')

  wandb.finish()
  logging.info('Training complete.')


if __name__ == '__main__':
  app.run(main)
