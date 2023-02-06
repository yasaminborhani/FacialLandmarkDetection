import tensorflow as tf 
import numpy as np 
import tensorflow_addons as tfa
import common_ops as ops 
import attention_modules as attm

class MLP(tf.keras.layers.Layer):
    def __init__(self, units=[256, 128], activation=tf.nn.gelu, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.fc_layers = [tf.keras.layers.Dense(units=unit, 
                                                activation=self.activation)
                         for unit in self.units]
    def call(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        return x

class FFN(tf.keras.layers.Layer):
  """Positionwise feed-forward network."""

  def __init__(self,
               hidden_size,
               dropout=0.0,
               expansion_rate=4,
               activation='gelu',
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='ffn'):
    super(FFN, self).__init__(name=name)

    self.hidden_size = hidden_size
    self.expansion_rate = expansion_rate
    self.expanded_size = self.hidden_size * self.expansion_rate
    self.dropout = dropout
    self.activation = activation

    self._expand_dense = attm.TrailDense(
        output_trailing_dims=self.expanded_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='expand_dense')
    self._shrink_dense = attm.TrailDense(
        output_trailing_dims=self.hidden_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='shrink_dense')
    self._activation_fn = ops.get_act_fn(self.activation)

  def call(self, inputs, training):
    output = inputs
    output = self._expand_dense(output)
    output = self._activation_fn(output)
    if self.dropout:
      output = tf.keras.layers.Dropout(self.dropout, name='nonlinearity_drop')(
          output, training=training)
    output = self._shrink_dense(output)

    return output


class BlockSA(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 num_heads=8,
                 head_size=None,
                 window_size=7,
                 dropatt=0.0,
                 rel_attn_type='2d_multi_head',
                 scale_ratio=None,
                 kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                 bias_initializer=tf.zeros_initializer,
                 **kwargs):
        
        super(BlockSA, self).__init__(**kwargs) 
        self.window_size        = window_size 
        self.hidden_size        = hidden_size
        self.num_heads          = num_heads
        self.head_size          = self.hidden_size // self.num_heads
        self.dropatt            = dropatt
        self.rel_attn_type      = rel_attn_type
        self.scale_ratio        = scale_ratio
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer

    def build(self, input_shape):
        self.block_attention = attm.Attention(self.hidden_size,
                                              self.head_size,
                                              num_heads=self.num_heads,
                                              dropatt=self.dropatt,
                                              rel_attn_type=self.rel_attn_type,
                                              scale_ratio=self.scale_ratio,
                                              kernel_initializer=self.kernel_initializer,
                                              bias_initializer=self.bias_initializer,
                                              name='attention')
    
    def window_partition(self, features):
        """Partition the input feature maps into non-overlapping windows.

        Args:
        features: [B, H, W, C] feature maps.

        Returns:
        Partitioned features: [B, nH, nW, wSize, wSize, c].

        Raises:
        ValueError: If the feature map sizes are not divisible by window sizes.
        """
        h, w, c     = features.shape[1], features.shape[2], features.shape[3]
        b           = tf.shape(features)[0]
        window_size = self.window_size

        if h % window_size != 0 or w % window_size != 0:
            raise ValueError(f'Feature map sizes {(h, w)} '
                        f'not divisible by window size ({window_size}).')

        features = tf.reshape(features, (b,
                                        h // window_size, window_size,
                                        w // window_size, window_size, c))
        features = tf.transpose(features, (0, 1, 3, 2, 4, 5))
        features = tf.reshape(features, (-1, window_size, window_size, c))
        return features

    def window_stitch_back(self, features, window_size, h, w):
        """Reverse window_partition."""
        b           = tf.shape(features)[0]

        features = tf.reshape(features, [
            -1, h // window_size, w // window_size, window_size, window_size,
            features.shape[-1]
        ])
        return tf.reshape(
            tf.transpose(features, (0, 1, 3, 2, 4, 5)),
            [-1, h, w, features.shape[-1]])
    
    def call(self, x, training=False, attn_mask=None):
        
        h, w  = x.shape[1], x.shape[2]
        output = self.window_partition(x)
        output = ops.maybe_reshape_to_1d(output)
        output = self.block_attention(output, training, attn_mask=attn_mask)
        output = self.window_stitch_back(output, self.window_size, h, w)
        return output


class GridSA(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 grid_size=7,
                 num_heads=8,
                 head_size=None,
                 dropatt=0.0,
                 rel_attn_type='2d_multi_head',
                 scale_ratio=None,
                 kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                 bias_initializer=tf.zeros_initializer,
                 **kwargs):
        
        super(GridSA, self).__init__(**kwargs) 
        self.grid_size          = grid_size
        self.hidden_size        = hidden_size
        self.num_heads          = num_heads
        self.head_size          = self.hidden_size // self.num_heads
        self.dropatt            = dropatt
        self.rel_attn_type      = rel_attn_type
        self.scale_ratio        = scale_ratio
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
    
    def build(self, input_shape):
        self.grid_attention = attm.Attention(self.hidden_size,
                                              self.head_size,
                                              num_heads=self.num_heads,
                                              dropatt=self.dropatt,
                                              rel_attn_type=self.rel_attn_type,
                                              scale_ratio=self.scale_ratio,
                                              kernel_initializer=self.kernel_initializer,
                                              bias_initializer=self.bias_initializer,
                                              name='attention')
    
    def grid_stitch_back(self, features, grid_size, h, w):
        """Reverse window_partition."""
        b = tf.shape(features)[0]
        features = tf.reshape(features, [
            -1, h // grid_size, w // grid_size, grid_size,
            grid_size, features.shape[-1]
        ])
        return tf.reshape(
            tf.transpose(features, (0, 3, 1, 4, 2, 5)),
            [-1, h, w, features.shape[-1]])
    
    def grid_partition(self, features):
        """Partition the input feature maps into non-overlapping windows.
        Args:
        features: [B, H, W, C] feature maps.
        Returns:
        Partitioned features: [B, nH, nW, wSize, wSize, c].
        Raises:
        ValueError: If the feature map sizes are not divisible by window sizes.
        """
        h, w, c = features.shape[1], features.shape[2], features.shape[3]
        b       = tf.shape(features)[0]
        grid_size = self.grid_size
        if h % grid_size != 0 or w % grid_size != 0:
            raise ValueError(f'Feature map sizes {(h, w)} '
                        f'not divisible by window size ({grid_size}).')
        features = tf.reshape(features, (-1,
                                        grid_size, h // grid_size,
                                        grid_size, w // grid_size, c))
        features = tf.transpose(features, (0, 2, 4, 1, 3, 5))
        features = tf.reshape(features, (-1, grid_size, grid_size, c))
        return features

    def call(self, x, training=False, attn_mask=None):
        # Apply global grid
        h, w = x.shape[1], x.shape[2]
        output = self.grid_partition(x)
        output = ops.maybe_reshape_to_1d(output)
        output = self.grid_attention(output, training, attn_mask=attn_mask)
        output = self.grid_stitch_back(output, self.grid_size, h, w)
        return output



class RVT_BlockTranspose(tf.keras.Model):
    def __init__(self,
                 filters=32,
                 hidden_size=32,
                 kernel_size=(3, 3),
                 strides=(1, 1), 
                 padding='valid',
                 norm=None,
                 num_heads=8,
                 window_size=7,
                 grid_size=7,
                 activation=tf.nn.gelu,
                 **kwargs):
        super(RVT_BlockTranspose, self).__init__()
        self.filters     = filters 
        self.kernel_size = kernel_size
        self.strides     = strides
        self.hidden_size = hidden_size 
        self.padding     = padding
        self.scale       = 1e-5
        self.norm        = None
        self.num_heads   = num_heads
        self.window_size = window_size
        self.grid_size   = grid_size
        self.activation  = activation

    def build(self, input_shape):
        self.conv     = tf.keras.layers.Conv2DTranspose(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding)
        
        self.block_sa = BlockSA(hidden_size=self.hidden_size, window_size=self.window_size)
        self.block_ln = tf.keras.layers.LayerNormalization()
        self.block_sc = tf.keras.layers.Rescaling(self.scale)
        self.mlp1     = FFN(hidden_size=self.hidden_size)
        self.mlp1_ln  = tf.keras.layers.LayerNormalization()
        self.mlp1_sc  = tf.keras.layers.Rescaling(self.scale)
        self.grid_sa  = GridSA(hidden_size=self.hidden_size, grid_size=self.grid_size)
        self.grid_ln  = tf.keras.layers.LayerNormalization()
        self.grid_sc  = tf.keras.layers.Rescaling(self.scale)
        self.mlp2     = FFN(hidden_size=self.hidden_size)
        self.mlp2_ln  = tf.keras.layers.LayerNormalization()
        self.mlp2_sc  = tf.keras.layers.Rescaling(self.scale)
        self.normal   = lambda x: x if self.norm is None else self.norm 

    def call(self, x, training=False):
        # Convolution
        x = self.conv(x)
        x = self.normal(x)
        x = self.activation(x)

        # Block Self-Attention 
        r = x
        x = self.block_ln(x)
        x = self.block_sa(x)
        x = self.block_sc(x)
        x = r + x

        # MLP 1
        r = x
        x = self.mlp1_ln(x)
        x = self.mlp1(x)
        x = self.mlp1_sc(x)
        x = x + r 

        # Grid Self-Attention 
        r = x
        x = self.grid_ln(x)
        x = self.grid_sa(x)
        x = self.grid_sc(x)
        x = r + x

        # MLP 2
        r = x
        x = self.mlp1_ln(x)
        x = self.mlp1(x)
        x = self.mlp1_sc(x)
        x = x + r 

        return x
    
class RVT_Block(tf.keras.Model):
    def __init__(self,
                 filters=32,
                 hidden_size=32,
                 kernel_size=(3, 3),
                 strides=(1, 1), 
                 padding='valid',
                 norm=None,
                 num_heads=8,
                 window_size=7,
                 grid_size=7,
                 activation=tf.nn.gelu,
                 **kwargs):
        super(RVT_Block, self).__init__()
        self.filters     = filters 
        self.kernel_size = kernel_size
        self.strides     = strides
        self.hidden_size = hidden_size 
        self.padding     = padding
        self.scale       = 1e-5
        self.norm        = None
        self.num_heads   = num_heads
        self.window_size = window_size
        self.grid_size   = grid_size
        self.activation  = activation
        
    def build(self, input_shape):
        self.conv     = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding)
        
        self.block_sa = BlockSA(hidden_size=self.hidden_size, window_size=self.window_size)
        self.block_ln = tf.keras.layers.LayerNormalization()
        self.block_sc = tf.keras.layers.Rescaling(self.scale)
        self.mlp1     = FFN(hidden_size=self.hidden_size)
        self.mlp1_ln  = tf.keras.layers.LayerNormalization()
        self.mlp1_sc  = tf.keras.layers.Rescaling(self.scale)
        self.grid_sa  = GridSA(hidden_size=self.hidden_size, grid_size=self.grid_size)
        self.grid_ln  = tf.keras.layers.LayerNormalization()
        self.grid_sc  = tf.keras.layers.Rescaling(self.scale)
        self.mlp2     = FFN(hidden_size=self.hidden_size)
        self.mlp2_ln  = tf.keras.layers.LayerNormalization()
        self.mlp2_sc  = tf.keras.layers.Rescaling(self.scale)
        self.normal   = lambda x: x if self.norm is None else self.norm 

    def call(self, x, training=False):
        # Convolution
        x = self.conv(x)
        x = self.normal(x)
        x = self.activation(x)

        # Block Self-Attention 
        r = x
        x = self.block_ln(x)
        x = self.block_sa(x)
        x = self.block_sc(x)
        x = r + x

        # MLP 1
        r = x
        x = self.mlp1_ln(x)
        x = self.mlp1(x)
        x = self.mlp1_sc(x)
        x = x + r 

        # Grid Self-Attention 
        r = x
        x = self.grid_ln(x)
        x = self.grid_sa(x)
        x = self.grid_sc(x)
        x = r + x

        # MLP 2
        r = x
        x = self.mlp1_ln(x)
        x = self.mlp1(x)
        x = self.mlp1_sc(x)
        x = x + r 

        return x
