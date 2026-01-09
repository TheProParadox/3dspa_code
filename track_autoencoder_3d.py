"""Official code of 3d track autoencoder"""

from __future__ import annotations

import functools
from typing import Any, NotRequired, TypedDict

import einops
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from attention import ImprovedTransformer
from track_autoencoder import (
    SinusoidalEmbedding,
    ParamStateInit,
    TrackAutoEncoderResults,
    TrackAutoEncoderDecoderContext,
)


class TrackAutoEncoder3DInputs(TypedDict):
  """3D Track autoencoder inputs.

  Attributes:
    support_tracks: 3D point tracks (x, y, z) for support set.
    support_tracks_visible: Visibility flags for support tracks.
    query_points: The (t, x, y, z) locations of query points.
    boundary_frame: Int specifying the first frame of any padding.
    dino_features: Optional DINOv2 features per track per frame.
    depth_features: Optional depth features per track per frame.
  """

  support_tracks: Any  # float["*B N T 3"]
  support_tracks_visible: Any  # float["*B N T 1"]
  query_points: NotRequired[Any]  # NotRequired[float["*B Q 4"]]
  boundary_frame: Any  # int["*B"]
  dino_features: NotRequired[Any]  # Optional[float["*B N T D_dino"]]
  depth_features: NotRequired[Any]  # Optional[float["*B N T D_depth"]]


class TrackAutoEncoder3D(nn.Module):
  """3DSPA: 3D point track autoencoder with semantic features.
  
  Extends TRAJAN to 3D by adding:
  - 3D point tracks (x, y, z) instead of 2D (x, y)
  - DINOv2 semantic features for scene understanding
  - Depth features for 3D reasoning
  - Larger model capacity for 3D motion modeling
  """

  num_output_frames: int = 150
  num_latent_tokens: int = 128
  latent_token_dim: int = 96
  num_frequencies: int = 32
  track_scale_factor: float = 1.0
  time_scale_factor: float = 150.0
  track_token_dim: int = 384
  encoder_latent_dim: int = 512
  decoder_num_channels: int = 1280
  dino_feature_dim: int = 768
  depth_feature_dim: int = 256
  use_dino: bool = True
  use_depth: bool = True

  decoder_scan_chunk_size: int | None = None

  def setup(self):
    self.initializer = ParamStateInit(
        shape=(self.num_latent_tokens, self.encoder_latent_dim),
    )
    self.track_token_projection = nn.Dense(self.track_token_dim)
    
    # Feature projections
    if self.use_dino:
      self.dino_projection = nn.Dense(768)  # Project DINO features
    if self.use_depth:
      self.depth_projection = nn.Dense(self.depth_feature_dim)
    
    self.sinusoidal_embedding = SinusoidalEmbedding(
        num_frequencies=self.num_frequencies
    )
    self.compressor = nn.Dense(self.latent_token_dim)
    self.decompressor = nn.Dense(self.decoder_num_channels - 128)
    self.input_readout_token = ParamStateInit(shape=(1, self.track_token_dim))
    
    # Transformer blocks with increased dimensions
    self.input_track_transformer = ImprovedTransformer(
        qkv_size=96 * 8,  # Increased from 64*8
        num_heads=8,
        mlp_size=1536,  # Increased from 1024
        num_layers=3,  # Increased from 2
    )
    self.tracks_to_latents = ImprovedTransformer(
        qkv_size=96 * 8,
        num_heads=8,
        mlp_size=2048,
        num_layers=4,  # Increased from 6 but with better capacity
    )
    self.decompress_attn = ImprovedTransformer(
        qkv_size=96 * 8,
        num_heads=8,
        mlp_size=2048,
        num_layers=4,  # Increased from 3
    )
    self.track_readout_attn = ImprovedTransformer(
        qkv_size=96 * 8,
        num_heads=8,
        mlp_size=1536,  # Increased from 1024
        num_layers=4,
    )
    self.query_encoder = nn.Dense(self.decoder_num_channels)
    # Output: (x, y, z) * T + occlusion * T = 4 * T
    self.track_predictor = nn.Dense(self.num_output_frames * 4)

  def encode_point_identities(self, query_points):
    """Encode 3D query point (x, y, z) into sinusoidal embeddings."""
    queries = query_points / self.track_scale_factor
    track_identities = self.sinusoidal_embedding(queries)
    return track_identities

  def embed_track_pos_visible(self, tracks, visible, dino_features=None, depth_features=None):
    """Embed 3D tracks (x, y, z, t) with optional DINO and depth features."""
    # Embed 3D coordinates + time
    fr_id = jnp.arange(tracks.shape[-2]) / tracks.shape[-2]
    fr_id = jnp.broadcast_to(
        fr_id[jnp.newaxis, jnp.newaxis, :, jnp.newaxis], visible.shape
    )
    # Concatenate (x, y, z, t)
    tracks_with_time = jnp.concatenate([tracks, fr_id], axis=-1)
    point_coords_embedding = self.sinusoidal_embedding(
        tracks_with_time / self.track_scale_factor
    )
    
    # Project positional embedding
    track_embeddings = self.track_token_projection(point_coords_embedding)
    
    # Add DINO features if available
    if self.use_dino and dino_features is not None:
      dino_proj = self.dino_projection(dino_features)
      track_embeddings = track_embeddings + dino_proj
    
    # Add depth features if available
    if self.use_depth and depth_features is not None:
      depth_proj = self.depth_projection(depth_features)
      track_embeddings = track_embeddings + depth_proj
    
    return track_embeddings

  def encode_tracks(self, tracks, visible, restart, dino_features=None, depth_features=None):
    """Encode 3D tracks into fixed-size token representations using transformer."""
    track_embeddings = self.embed_track_pos_visible(
        tracks=tracks,
        visible=visible,
        dino_features=dino_features,
        depth_features=depth_features,
    )
    
    # Add readout token
    batch_shape = track_embeddings.shape[:-2]
    readout_token = self.input_readout_token(batch_shape)
    track_tokens = jnp.concatenate(
        [readout_token, track_embeddings], axis=-2
    )
    
    time = jnp.arange(visible.shape[2])
    partition = time < restart[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    visible_bool = visible[..., 0].astype(jnp.bool_)
    
    # Visibility mask: [*B N T+1 T+1]
    visibility_mask = (
        jnp.ones_like(visible_bool[..., jnp.newaxis]) 
        * visible_bool[..., jnp.newaxis, :]
    )
    # Readout token is always visible
    readout_visible = jnp.ones_like(visible_bool[..., :1, jnp.newaxis])
    visibility_mask = jnp.concatenate(
        [readout_visible, visibility_mask], axis=-2
    )
    
    track_tokens = self.input_track_transformer(
        track_tokens, qq_mask=partition * visibility_mask
    )
    
    # Extract readout token (first token)
    readout_token = track_tokens[..., 0:1, :]
    return readout_token[..., 0, :]  # Remove singleton dimension

  def encode(self, inputs):
    """Encode support tracks into compressed motion latents."""
    support_track_tokens = self.encode_tracks(
        tracks=inputs["support_tracks"],
        visible=inputs["support_tracks_visible"],
        restart=inputs["boundary_frame"],
        dino_features=inputs.get("dino_features"),
        depth_features=inputs.get("depth_features"),
    )

    latents = self.initializer(batch_shape=(inputs["support_tracks"].shape[0],))
    latents = self.tracks_to_latents(latents, support_track_tokens)

    latents = self.compressor(latents)
    return latents

  @nn.remat
  def get_decoder_context(self, inputs):
    """Get decoder context from query points."""
    if "query_points" in inputs:
      decoder_query = inputs["query_points"][..., 1:]  # Extract (x, y, z)
      query_frame = jnp.array(
          jnp.round(inputs["query_points"][..., 0]), jnp.int32
      )
    else:
      # Default to a grid
      grid_centers = jnp.arange(32) / 32.0 + 1.0 / 64.0
      query_x, query_y = jnp.meshgrid(grid_centers, grid_centers)
      query_z = jnp.zeros_like(query_x)  # Default z=0 for 3D
      decoder_query = jnp.reshape(
          jnp.stack([query_x, query_y, query_z], axis=-1), [-1, 3]
      )
      decoder_query = jnp.broadcast_to(
          decoder_query,
          inputs["support_tracks"].shape[:-3] + decoder_query.shape,
      )
      query_frame = jnp.array(decoder_query[..., 0], jnp.int32) * 0
    
    decoder_query = self.encode_point_identities(query_points=decoder_query)
    return TrackAutoEncoderDecoderContext(
        decoder_query=decoder_query,
        query_frame=query_frame,
        boundary_frame=inputs["boundary_frame"],
    )

  def append_time_feat(self, latents, query_frame):
    """Append temporal features to latents based on query frame index."""
    assert latents.shape[-1] == (self.decoder_num_channels - 128)

    def get_eye(idx):
      return jnp.eye(128, latents.shape[-1], idx * 5)

    for _ in range(query_frame.ndim):
      get_eye = jax.vmap(get_eye)
    multiplier = get_eye(query_frame)
    to_append = jnp.einsum("... N C , ... D C -> ... N D", latents, multiplier)
    return jnp.concatenate([latents, to_append], axis=-1)

  @nn.remat
  def decode(self, latents, decoder_context, discretize=True):
    """Decode compressed latents to 3D point tracks (x, y, z) and occlusion."""
    latents = jnp.clip(latents, -1.0, 1.0)
    if discretize:
      latents_disc = jnp.round(latents * 128.0) / 128.0
      rng = jax.random.PRNGKey(0)
      latents_disc = (
          latents_disc
          + jax.random.uniform(rng, latents_disc.shape) / 128.0
          - 1.0 / 256.0
      )
      latents = latents - jax.lax.stop_gradient(latents - latents_disc)
    
    latents = self.decompressor(latents)
    latents = self.decompress_attn(latents)

    queries = jnp.concatenate(
        [
            decoder_context.decoder_query,
            decoder_context.query_frame[..., jnp.newaxis]
            // self.time_scale_factor,
        ],
        axis=-1,
    )
    point_coords_embedding = self.query_encoder(
        self.sinusoidal_embedding(queries / self.track_scale_factor)
    )
    latents = jnp.tile(
        latents[..., jnp.newaxis, :, :],
        (1,) * len((latents.shape[0],))
        + (point_coords_embedding.shape[-2], 1, 1),
    )
    latents = self.append_time_feat(latents, decoder_context.query_frame)
    latents = jnp.concatenate(
        [point_coords_embedding[..., jnp.newaxis, :], latents], axis=2
    )
    out = self.track_readout_attn(latents)
    out = out[..., 0, :]
    out = self.track_predictor(out)
    
    num_frames = self.num_output_frames
    # Output: [x, y, z, occlusion] for each frame
    tracks = jnp.stack(
        [
            out[..., :num_frames],
            out[..., num_frames : 2 * num_frames],
            out[..., 2 * num_frames : 3 * num_frames],
        ],
        axis=-1,
    )
    visible_logits = out[..., 3 * num_frames :, jnp.newaxis]
    # For 3DSPA, we only use visible_logits (occlusion), not certain_logits
    certain_logits = jnp.zeros_like(visible_logits)

    return TrackAutoEncoderResults(
        tracks=tracks,
        visible_logits=visible_logits,
        certain_logits=certain_logits,
    )

  def __call__(self, inputs):
    """Forward pass: encode support tracks, decode query tracks."""
    latents = self.encode(inputs)
    if self.decoder_scan_chunk_size is None:
      decoder_context = self.get_decoder_context(inputs)
      outputs = self.decode(latents=latents, decoder_context=decoder_context)
    else:
      # Scan-based decoding for memory efficiency
      def scan_fn(tr_enc, carry, qp):
        autoencoder_inputs = TrackAutoEncoder3DInputs(
            query_points=qp + carry,
            boundary_frame=inputs["boundary_frame"],
            support_tracks=inputs["support_tracks"],
            support_tracks_visible=inputs["support_tracks_visible"],
            dino_features=inputs.get("dino_features"),
            depth_features=inputs.get("depth_features"),
        )
        decoder_context = tr_enc.get_decoder_context(autoencoder_inputs)
        res = tr_enc.decode(latents, decoder_context)
        carry = jnp.sum(res.tracks) > 1e20
        return carry, res

      scan_fn2 = nn.scan(
          scan_fn,
          variable_broadcast="params",
          split_rngs={"params": False, "default": True},
          in_axes=-3,
          out_axes=-4,
      )
      h = self.decoder_scan_chunk_size
      _, preds = scan_fn2(
          self,
          False,
          einops.rearrange(
              inputs["query_points"], "... (Q H) C -> ... Q H C", H=h
          ),
      )
      outputs = jax.tree_util.tree_map(
          lambda x: einops.rearrange(x, "... Q H T C -> ... (Q H) T C", H=h),
          preds,
      )

    outputs = TrackAutoEncoderResults(
        tracks=outputs.tracks,
        visible_logits=outputs.visible_logits,
        certain_logits=outputs.certain_logits,
    )

    return outputs

