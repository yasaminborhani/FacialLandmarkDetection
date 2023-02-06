import copy
import functools
import math
import re
import string
import tensorflow as tf
import numpy as np

import common_ops
import attention_utils as attn_utils 

def float32_softmax(x, *args, **kwargs):
  y = tf.cast(tf.nn.softmax(tf.cast(x, tf.float32), *args, **kwargs), x.dtype)
  return y

class TrailDense(tf.keras.layers.Layer):
  """Dense module that projects multiple trailing dimensions."""

  def __init__(self,
               output_trailing_dims,
               begin_axis=-1,
               use_bias=True,
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='dense'):
    super(TrailDense, self).__init__(name=name)

    if isinstance(output_trailing_dims, int):
      self._output_trailing_dims = [output_trailing_dims]
    else:
      assert isinstance(output_trailing_dims, (list, tuple)) and all(
          isinstance(i, int) for i in output_trailing_dims), (
              'Invalid output shape: {}.'.format(output_trailing_dims))
      self._output_trailing_dims = list(output_trailing_dims)
    self.begin_axis = begin_axis
    self.use_bias = use_bias

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

  def build(self, input_shape):
    """Create variables and einsum expression based on input shape."""
    # Create variables
    weight_shape = input_shape[self.begin_axis:] + self._output_trailing_dims
    self.weight = self.add_weight(
        name='weight',
        shape=weight_shape,
        initializer=self.kernel_initializer,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=self._output_trailing_dims,
          initializer=self.bias_initializer,
          trainable=True)

    # Create einsum expression
    input_rank = input_shape.rank
    shared_size = self.begin_axis % input_rank
    i_only_size = input_rank - shared_size
    o_only_size = len(self._output_trailing_dims)

    assert input_rank + o_only_size < len(string.ascii_uppercase), (
        'Cannot use einsum as input rank + output rank > 26.')
    einsum_str = string.ascii_uppercase[:input_rank + o_only_size]

    offset = 0
    shared_str = einsum_str[offset:offset+shared_size]
    offset += shared_size
    i_only_str = einsum_str[offset:offset+i_only_size]
    offset += i_only_size
    o_only_str = einsum_str[offset:offset+o_only_size]

    input_str = '{}{}'.format(shared_str, i_only_str)
    output_str = '{}{}'.format(shared_str, o_only_str)
    weight_str = '{}{}'.format(i_only_str, o_only_str)

    self.einsum_expr = '{},{}->{}'.format(input_str, weight_str, output_str)

  def call(self, inputs):
    output = tf.einsum(self.einsum_expr, inputs, self.weight)
    if self.use_bias:
      output += self.bias
    return output

class Attention(tf.keras.layers.Layer):
  """Multi-headed attention module."""
  def __init__(self,
               hidden_size,
               head_size,
               num_heads=None,
               dropatt=0.0,
               attn_axis=0,
               rel_attn_type='2d_multi_head',
               scale_ratio=None,
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='attention'):
    super(Attention, self).__init__(name=name)
    
    self.hidden_size = hidden_size
    self.head_size = head_size
    self.num_heads = num_heads or hidden_size // head_size
    self.dropatt = dropatt
    self.attn_axis = attn_axis
    self.rel_attn_type = rel_attn_type
    self.scale_ratio = scale_ratio

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self._q_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='q')
    self._k_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='k')
    self._v_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='v')
    self._o_proj = TrailDense(
        output_trailing_dims=self.hidden_size, begin_axis=-2,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='o')

    self.q_scale = self.head_size ** -0.5

  def build(self, query_shape):
    num_attn_dims = query_shape.rank - 2   # -2 to account for bsz, hidden size
    assert num_attn_dims < 6, 'Only support at most 6 attention dims.'
    symbols = ''.join([chr(ord('U') + i) for i in range(num_attn_dims - 1)])
    insert = lambda s, i, c: s[:i] + c + s[i:]
    create_expr = lambda s, prefix='B', suffix='NK': prefix + s + suffix
    self.q_expr = create_expr(insert(symbols, self.attn_axis, 'S'))
    self.k_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.v_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.a_expr = create_expr(symbols, suffix='NST')

    ##### Relative attention
    if self.rel_attn_type in ['2d_multi_head', '2d_single_head']:
      query_shape_list = query_shape.as_list()
      if query_shape.rank == 4:
        height, width = query_shape_list[1:3]
      elif query_shape.rank == 3:
        seq_len = query_shape_list[1]
        height = int(seq_len ** 0.5)
        width = height
        if height * width != seq_len:
          raise ValueError('Does not support 2D relative attentive for '
                           'non-square inputs.')
      else:
        raise ValueError(
            'Does not support relative attention for query shape: %s.'
            % query_shape_list)

      if self.scale_ratio is not None:
        scale_ratio = eval(self.scale_ratio)
        vocab_height = 2 * round(height / scale_ratio) - 1
        vocab_width = 2 * round(width / scale_ratio) - 1
      else:
        vocab_height = 2 * height - 1
        vocab_width = 2 * width - 1

      if self.rel_attn_type == '2d_multi_head':
        h_axis = 1
        rel_bias_shape = [self.num_heads, vocab_height, vocab_width]
      elif self.rel_attn_type == '2d_single_head':
        h_axis = 0
        rel_bias_shape = [vocab_height, vocab_width]
      else:
        raise NotImplementedError('rel_attn_type %s not implemented yet.' %
                                  self.rel_attn_type)

      self.relative_bias = self.add_weight(
          'relative_bias',
          rel_bias_shape,
          initializer=self.kernel_initializer,
          trainable=True)

      if self.scale_ratio is not None:
        src_shape = self.relative_bias.shape.as_list()
        relative_bias = tf.expand_dims(self.relative_bias, axis=-1)
        relative_bias = tf.cast(
            tf.image.resize(relative_bias, [2 * height - 1, 2 * width - 1]),
            self.compute_dtype)
        relative_bias = tf.squeeze(relative_bias, axis=-1)
        tgt_shape = relative_bias.shape.as_list()

      else:
        relative_bias = tf.cast(self.relative_bias, self.compute_dtype)

      self.reindexed_bias = attn_utils.reindex_2d_einsum_lookup(
          relative_bias, height, width, height - 1, width - 1,
          h_axis=h_axis)
    else:
      self.reindexed_bias = None

  def call(self, query, training, context=None, attn_mask=None):
    if context is None:
      context = query

    q_heads = self._q_proj(query)
    k_heads = self._k_proj(context)
    v_heads = self._v_proj(context)
    q_heads *= self.q_scale

    # attention
    attn_logits = tf.einsum(
        f'{self.q_expr},{self.k_expr}->{self.a_expr}',
        q_heads, k_heads)

    if self.reindexed_bias is not None:
      attn_logits += self.reindexed_bias

    if attn_mask is not None:
      attn_logits += (1.0 - attn_mask) * attn_logits.dtype.min

    attn_probs = float32_softmax(attn_logits, axis=-1)
    if self.dropatt:
      attn_probs = tf.keras.layers.Dropout(self.dropatt, 'attn_prob_drop')(
          attn_probs, training=training)

    attn_out = tf.einsum(
        f'{self.a_expr},{self.v_expr}->{self.q_expr}',
        attn_probs, v_heads)
    output = self._o_proj(attn_out)

    return output
