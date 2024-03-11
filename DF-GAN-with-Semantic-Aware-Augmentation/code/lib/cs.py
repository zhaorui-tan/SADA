import tensorflow as tf
import numpy as np


# the code is modified from
# **Linear Algebra**


# **Metrics**

# @tf.function
def cs(y_true, y_predict, x_true,):

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    SS = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_predict, x_true), axis=1))
    return SS * 2.5
