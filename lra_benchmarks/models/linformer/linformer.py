# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Linformer models."""
# from flax.deprecated import nn
import flax.linen as nn
import jax.numpy as jnp
import jax
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.linformer import linformer_attention


class LinformerBlock(nn.Module):
  """Linformer layer (https://arxiv.org/abs/2006.04768)."""

  @nn.compact
  def __call__(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            max_len=512,
            cache=None):
    """Applies LinformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      max_len: int, max sequence length.
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm()(inputs)
    x = linformer_attention.LinformerAttention()(
        x,
        inputs_kv=None,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        # dropout_rng=jax.random.PRNGKey(0),
        dropout_rng=self.make_rng("dropout"),
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        max_len=max_len,
        cache=cache)
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm()(x)
    y = common_layers.MlpBlock()(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class LinformerEncoder(nn.Module):
  """Linformer Model Encoder."""

  @nn.compact
  def __call__(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='CLS',
            num_classes=10,
            tied_weights=False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      tied_weights: bool

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, emb_dim))
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(name='posembed_input')(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        )
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(num_layers):
        x = LinformerBlock(name=f'encoderblock_{lyr}')(
            x,
            qkv_dim=qkv_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dtype=dtype,
            padding_mask=src_padding_mask,
            inputs_segmentation=inputs_segmentation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            max_len=max_len,
            deterministic=not train,
            )
    encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded


