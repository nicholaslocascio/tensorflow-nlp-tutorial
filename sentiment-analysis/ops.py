import math
import numpy as np
import tensorflow as tf
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, layer_scope, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    print(shape)

    with tf.variable_scope(layer_scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias