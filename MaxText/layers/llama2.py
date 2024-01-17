"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from jax.sharding import Mesh



from flax import linen as nn


import jax.numpy as jnp
import jax
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import initializers

from layers import models

import common_types

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm


#-----------------------------------------
# The Decoder Layer specific for Llama2
#-----------------------------------------


class LlamaDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))


    lnx_rms = models.RMSNorm(
        dtype=cfg.dtype,
        name='pre_self_attention_layer_norm',
        kernel_axes=('embed',),
        epsilon=cfg.rms_norm_epsilon
        )
    lnx = lnx_rms(inputs)

    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # Self-attention block
    attention_layer = Attention(
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dtype=cfg.dtype,
      dropout_rate=cfg.dropout_rate,
      name='self_attention',
      use_int8=cfg.int8_training)

    attention_lnx = attention_layer(
            lnx,
            lnx,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=model_mode)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.RMSNorm(
        dtype=cfg.dtype, name='post_self_attention_layer_norm', kernel_axes=('embed',),
        epsilon=cfg.rms_norm_epsilon,
        )(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ('activation_batch', 'activation_length', 'activation_embed'))

    if cfg.num_experts > 1:
        gate_logits = linears.DenseGeneral(
            cfg.num_experts,
            dtype=cfg.dtype,
            kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
            kernel_axes=('embed', 'mlp'),
            name="gate",
            use_int8=cfg.int8_training,
        )(hidden_states)
        weights, selected_experts = jax.lax.top_k(gate_logits, cfg.num_experts_per_tok)
        weights = jax.nn.softmax(weights, axis=-1)
        mlp_lnx = jnp.zeros_like(hidden_states)
        mlp_lnx = nn.with_logical_constraint(
                mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
            )
        for k in range(cfg.num_experts):
            #batch_idx, nth_expert = jnp.where(selected_experts == k)
            weights_exp = jnp.sum(jnp.multiply(selected_experts==k, weights), axis=-1)
            mlp_lnx_exp = linears.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name=f'mlp_{k}',
            config=cfg,
            )(hidden_states, deterministic=deterministic)
            mlp_lnx_exp = nn.with_logical_constraint(
                mlp_lnx_exp, ('activation_batch', 'activation_length', 'activation_embed')
            )
            mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
            mlp_lnx += mlp_lnx_exp
    else: 
        # MLP block.
        mlp_lnx = linears.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name='mlp',
            config=cfg,
        )(hidden_states, deterministic=deterministic)
        mlp_lnx = nn.with_logical_constraint(
            mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
        )


    layer_output = mlp_lnx + intermediate_inputs

    layer_output = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output