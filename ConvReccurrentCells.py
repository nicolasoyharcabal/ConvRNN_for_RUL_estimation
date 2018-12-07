from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import numpy as np
import tensorflow as tf

"""
Adapted code from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/rnn/python/ops/rnn_cell.py

"""

class ConvReccurrentCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
     JANET recurrent neural network
  https://arxiv.org/abs/1804.04849
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               kind,
               t_max = None,
               use_bias=True,
               skip_connection=False,
               name="conv_rnn_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      kind: Kind of ConvRNN like ConvJANET or ConvLSTM.
      t_max: Maximun time in the time series for Chrono initializer.
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvReccurrentCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._t_max = t_max
    self._kind = kind
    self._use_bias = use_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    cell, hidden = state

    if self._kind == "JANET":
        new_hidden = _conv([inputs, hidden], self._kernel_shape,
                           2 * self._output_channels, self._use_bias, self._t_max,self._kind)
        gates = array_ops.split(
            value=new_hidden, num_or_size_splits=2, axis=self._conv_ndims + 1)

        new_input, forget_gate = gates
        new_cell = math_ops.sigmoid(forget_gate) * cell + (1 - math_ops.sigmoid(
            forget_gate)) * math_ops.tanh(new_input / 3)
        output = new_cell

    elif self._kind == "LSTM":
        new_hidden = _conv([inputs, hidden], self._kernel_shape,
                           4 * self._output_channels, self._use_bias, self._t_max,self._kind)
        gates = array_ops.split(
            value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)

        input_gate, new_input, forget_gate, output_gate = gates
        new_cell = math_ops.sigmoid(forget_gate) * cell
        new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input/3)
        output = math_ops.tanh(new_cell/3) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = array_ops.concat([output, inputs], axis=-1)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state




def _conv(args, filter_size, num_features, bias,_t_max,kind):
  """Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]

  shape_length = len(shapes[0])

  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      print("shape_length", shape_length, len(shape))
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]
  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length * [1]

  # Now the computation.
  kernel = vs.get_variable(
      "kernel", filter_size + [total_arg_size_depth, num_features], dtype=dtype,
  initializer=tf.contrib.layers.xavier_initializer_conv2d())


  if len(args) == 1:
    res = conv_op(args[0], kernel, strides, padding="SAME")
  else:
    array = array_ops.concat(axis=shape_length - 1, values=args)
    res = conv_op(
        array,
        kernel,
        strides,
        padding="SAME")
  if not bias:
    return res

  bias_term = vs.get_variable(
      "biases", [num_features],
      dtype=dtype,
      initializer=chrono_init(_t_max,kind))
  return res + bias_term

"""
  Chrono initializer
  https://openreview.net/pdf?id=SJcKhk-Ab
"""

def chrono_init(t_max, kind):
    def _initializer(shape, dtype=tf.float32, partition_info=None):

        if kind == "LSTM":
            num_units = shape[0] // 4
            uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                        maxval=t_max, dtype=dtype,
                                                        seed=42))
            # i, j, o, f
            bias_i = -uni_vals
            j_o = tf.zeros(2*num_units)
            bias_f = uni_vals
            return tf.concat([bias_i, j_o, bias_f], 0)

        elif kind == "JANET":
            num_units = shape[0] // 2
            uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                        maxval=t_max, dtype=dtype,
                                                        seed=42))
            bias_j = tf.zeros(num_units)
            bias_f = uni_vals

            return tf.concat([bias_j, bias_f], 0)

    return _initializer

