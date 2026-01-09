"""Transformer attention modules for 3DSPA and TRAJAN."""

from __future__ import annotations

from typing import Optional

from flax import linen as nn
import jax.numpy as jnp


class ImprovedTransformer(nn.Module):
  """Transformer with cross-attention support.
  
  Uses LayerNorm on keys/queries and parallel cross/self-attention.
  """

  qkv_size: int
  num_heads: int
  mlp_size: int
  num_layers: int

  @nn.compact
  def __call__(
      self,
      queries,  # float['... d1'],
      inputs_kv=None,  # Optional[float['*b N D']]
      qk_mask=None,  # Optional[bool['...']]
      qq_mask=None,  # Optional[bool['...']]
  ):  # -> float['... d2']

    for i in range(self.num_layers):
      if qk_mask is not None and len(qk_mask.shape) == len(inputs_kv.shape):
        qk_mask = qk_mask[..., jnp.newaxis, :, :]
      if qq_mask is not None and len(qq_mask.shape) == len(queries.shape):
        qq_mask = qq_mask[..., jnp.newaxis, :, :]

      queries = ImprovedTransformerBlock(
          qkv_size=self.qkv_size,
          num_heads=self.num_heads,
          mlp_size=self.mlp_size,
          name=f'layer_{i}',
      )(
          queries,
          inputs_kv=inputs_kv,
          qq_mask=qq_mask,
          qk_mask=qk_mask,
      )

    queries = nn.LayerNorm(use_bias=False, use_scale=True, name='norm_encoder')(
        queries
    )

    return queries


class ImprovedTransformerBlock(nn.Module):
  """Transformer block with RMSNorm and parallel attention paths.
  
  Applies self-attention and optional cross-attention in parallel.
  """

  mlp_size: int
  num_heads: int
  qkv_size: int

  @nn.compact
  def __call__(
      self,
      queries,  # float['*b n d'],
      inputs_kv,  # Optional[float['*b N D']]
      qq_mask=None,  # Optional[bool['...']]
      qk_mask=None,  # Optional[bool['...']]
      remat_attn: bool = False,
  ):  # -> Float['*b n d']
    width = queries.shape[-1]
    normed_queries = nn.LayerNorm(
        use_bias=False, use_scale=True, name='norm_q'
    )(queries)
    attn_out = queries

    # Self-attention.
    self_attn_out = ImprovedMHDPAttention(
        num_heads=self.num_heads, qk_size=self.qkv_size, name='self_att'
    )(
        inputs_q=normed_queries,
        inputs_kv=normed_queries,
        mask=jnp.array(qq_mask, jnp.float32) if qq_mask is not None else None,
    )
    attn_out += self_attn_out

    # Cross-attention.
    if inputs_kv is not None:
      cross_attn_out = ImprovedMHDPAttention(
          num_heads=self.num_heads, qk_size=self.qkv_size, name='cross_att'
      )(
          inputs_q=normed_queries,
          inputs_kv=inputs_kv,
          mask=jnp.array(qk_mask, jnp.float32) if qk_mask is not None else None,
      )
      attn_out += cross_attn_out

    # MLP.
    normed_attn_out = nn.LayerNorm(
        use_bias=False, use_scale=True, name='norm_attn'
    )(attn_out)
    h = nn.gelu(nn.Dense(self.mlp_size, name='MLP_in')(normed_attn_out))
    mlp_out = nn.Dense(width, name='MLP_out')(h)
    return attn_out + mlp_out


class ImprovedMHDPAttention(nn.Module):
  """Multi-head dot-product attention with RMSNorm on keys/queries.
  
  Attributes:
    num_heads: Number of attention heads
    qk_size: Total dimension of keys and queries
    v_size: Total dimension of values (defaults to qk_size)
  """

  num_heads: int
  qk_size: int
  v_size: Optional[int] = None

  @nn.compact
  def __call__(
      self,
      inputs_q,  # float['*b q d1'],
      inputs_kv,  # float['*b k d2'],
      mask=None,  # Optional[float['*b #h #q #k']]
  ):  # -> float['*b q d1']
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input tokens from which queries are computed.
      inputs_kv: Input tokens from which the keys and queries are computed.
      mask: Mask for the attention weights.

    Returns:
      output tokens (linear projection of an attention weighted average of value
      tokens per query).
    """
    v_size = self.qk_size if self.v_size is None else self.v_size

    if self.qk_size % self.num_heads:
      raise ValueError(f'{self.num_heads=} must divide {self.qk_size=}.')
    if v_size % self.num_heads:
      raise ValueError(f'{v_size=} must divide {self.num_heads=}.')

    # Project inputs_q to multi-headed queries and keys.
    # dimensions are then [B..., Q, H, qk_size]
    query = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        name='dense_query',
    )(inputs_q)
    key = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        name='dense_key',
    )(inputs_kv)

    # Normalize keys and queries before attention.
    query = nn.RMSNorm(name='norm_query')(query)
    key = nn.RMSNorm(name='norm_key')(key)

    value = nn.DenseGeneral(
        features=(self.num_heads, v_size // self.num_heads),
        use_bias=False,
        name='dense_value',
    )(inputs_kv)

    x = nn.dot_product_attention(query, key, value, mask=mask)

    # Back to the original input dimensions.
    out = nn.DenseGeneral(
        features=inputs_q.shape[-1],
        axis=(-2, -1),
        use_bias=True,
        name='dense_out',
    )(x)

    return out
