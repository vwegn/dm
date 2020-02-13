import math

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.regularizers import L1L2


# Computation of the original coordinate source image position from subsampled coordinate:
def get_image_coordinate_p(i1, i2, alpha, beta):
    p1 = alpha * i1 + beta
    p2 = alpha * i2 + beta
    return p1, p2

# Definition of the dilated kernel for aggregation:
def agg_kernel_for_dilation():
    kernel = np.ones((2, 2, 1, 1), dtype=np.float32)
    kernel *= 1. / 4
    return kernel


# unused in this file
# def get_image_coordinate_q(k1, k2, i1, i2, radius, alpha, beta, level):
#     p1, p2 = get_image_coordinate_p(i1, i2, alpha, beta)
#     q1 = 2 ** level * (k1 - radius * 2 ** level) + p1
#     q2 = 2 ** level * (k2 - radius * 2 ** level) + p2
#     return q1, q2

#
class PrepareForExtraction(Layer):
    def __init__(self, stride, offset, radius, **kwargs):
        self.offset = offset
        self.stride = stride
        self.radius = radius
        super(PrepareForExtraction, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(PrepareForExtraction, self).build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        p, q = x
        p = K.l2_normalize(p, axis=-1)
        q = K.l2_normalize(q, axis=-1)

        # Subsample source image coordinates:
        p = p[:, self.offset::self.stride, self.offset::self.stride, :]
        return [p, q]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_p, shape_q = input_shape

        shape_p_1 = math.ceil((shape_p[1] - self.offset) / self.stride)
        shape_p_2 = math.ceil((shape_p[2] - self.offset) / self.stride)

        return [[shape_p[0], shape_p_1, shape_p_2, shape_p[3]], [shape_q[0], shape_q[1], shape_q[2], shape_q[3]]]


class BottomLevel(Layer):
    def __init__(self, stride, offset, radius, **kwargs):
        self.offset = offset
        self.stride = stride
        self.radius = radius
        super(BottomLevel, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(BottomLevel, self).build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        p, q = x

        q = tf.image.crop_to_bounding_box(q, self.offset, self.offset, q.shape[1]-self.offset, q.shape[2]-self.offset)
        q = tf.extract_image_patches(
            images=q,
            ksizes=(1, 2 * self.radius + 1, 2 * self.radius + 1, 1),
            strides=(1, self.stride, self.stride, 1),
            rates=(1, 1, 1, 1),
            padding='VALID')

        q = K.reshape(q, (-1, q.shape[1], q.shape[2], 2 * self.radius + 1, 2 * self.radius + 1, 128))
        p = K.reshape(p, (-1, p.shape[1], p.shape[2], 1, 1, 128))
        p = K.repeat_elements(p, q.shape[3], 3)
        p = K.repeat_elements(p, q.shape[4], 4)

        # Compute dot product:
        bottom_level = K.sum(p * q, axis=-1)
        return bottom_level

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_p, shape_q = input_shape
        return [(1, shape_p[1], shape_p[2], 2 * self.radius + 1, 2 * self.radius + 1)]


class Aggregation(Layer):
    def __init__(self, levels, exponent=None, shared=False, initial_exponent=1.4, **kwargs):
        self.levels = levels
        # Initialize exponent or receive existing one:
        if not shared or levels == 0:
            self.exponent = K.variable(name='exponent', value=np.ones((1,))*initial_exponent)
        else:
            self.exponent = exponent
        self.shared = shared
        super(Aggregation, self).__init__(**kwargs)

    def get_exponent(self):
        return self.exponent

    def build(self, input_shape):
        if not self.shared or self.levels == 0:
            self._trainable_weights.append(self.exponent)   # Adds exponent as trainable weight
            # self.add_loss(L1L2(l2=0.001)(self.exponent))  # Here regularization can be applied!
        super(Aggregation, self).build(input_shape)

    def call(self, x, **kwargs):
        layer = x
        pH = layer.shape[1]
        pW = layer.shape[2]
        qH = layer.shape[3]
        qW = layer.shape[4]

        layer = K.reshape(layer, (pH, pW, -1, 1))
        layer = K.permute_dimensions(layer, [2, 0, 1, 3])

        # Compute aggregation by dilated convolution:
        layer = K.conv2d(layer, agg_kernel_for_dilation(), strides=(1, 1), padding='same',
                         dilation_rate=(2 * (2 ** self.levels), 2 * (2 ** self.levels)))

        layer = K.permute_dimensions(layer, [1, 2, 0, 3])
        layer = K.reshape(layer, (-1, pH, pW, qH, qW))
        return layer ** self.exponent

    def compute_output_shape(self, input_shape):
        return input_shape


class MaxPoolingWithArgmax(Layer):
    def __init__(self, **kwargs):
        super(MaxPoolingWithArgmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxPoolingWithArgmax, self).build(input_shape)

    def call(self, x, **kwargs):
        layer = x
        pH = layer.shape[1]
        pW = layer.shape[2]
        qH = layer.shape[3]
        qW = layer.shape[4]

        layer = K.reshape(layer, (-1, qH, qW, 1))
        layer, args = tf.nn.max_pool_with_argmax(layer, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME', Targmax=tf.int64)
        qH = layer.shape[1]
        qW = layer.shape[2]

        layer = K.reshape(layer, (-1, pH, pW, qH, qW))
        # args = K.reshape(args, (-1, pH, pW, qH, qW))  # argmaxima are not reshaped!

        return [layer, args]

    def compute_output_shape(self, input_shape):
        shape = input_shape
        # Compute new shape (dilated convolution):
        o1 = math.floor((shape[3] - 1) / 2) + 1
        o2 = math.floor((shape[4] - 1) / 2) + 1
        return [(shape[0], shape[1], shape[2], o1, o2), (shape[1] * shape[2], o1, o2, shape[0])]


class Disaggregation(Layer):
    def __init__(self, levels, **kwargs):
        self.levels = levels
        super(Disaggregation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Disaggregation, self).build(input_shape)

    def call(self, x, **kwargs):
        layer = x
        pH = layer.shape[1]
        pW = layer.shape[2]
        qH = layer.shape[3]
        qW = layer.shape[4]

        layer = K.reshape(layer, (pH, pW, -1, 1))
        layer = K.permute_dimensions(layer, [2, 0, 1, 3])

        layer = tf.pad(layer, [[0, 0], [2 ** self.levels, 2 ** self.levels], [2 ** self.levels, 2 ** self.levels], [0, 0]])
        layer = tf.nn.pool(layer, (2, 2), "MAX", "VALID", dilation_rate=(2 * (2 ** self.levels), 2 * (2 ** self.levels)))

        layer = K.permute_dimensions(layer, [1, 2, 0, 3])
        layer = K.reshape(layer, (-1, pH, pW, qH, qW))
        return layer

    def compute_output_shape(self, input_shape):
        return input_shape


class Unpooling(Layer):
    def __init__(self, shape, **kwargs):
        self.shape = shape  # shape is given as an argument since it is impossible to compute
        super(Unpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Unpooling, self).build(input_shape)

    def call(self, x, **kwargs):
        layer, indices = x

        output_shape = self.shape
        output_shape = tf.cast(output_shape, tf.int64)

        reshaped_indices = tf.reshape(indices, layer.shape)
        num_segments = tf.reduce_prod(output_shape)

        output = tf.unsorted_segment_max(layer, reshaped_indices, num_segments)
        output = tf.reshape(output, output_shape)

        # unsorted_segment_max adds smallest negative values instead of -inf => change to -inf
        condition = tf.less(output, 0)
        case_true = tf.fill(output.shape, -np.inf)
        case_false = output
        output = tf.where(condition, case_true, case_false)

        return output

    def compute_output_shape(self, input_shape):
        return [self.shape]


def broadcast(tensor, shape):
    return tensor + tf.zeros(shape, dtype=tensor.dtype)


def indicator_function(z, sigma):
    result = tf.exp((-tf.norm(z, axis=-1) ** 2) / (2 * (sigma ** 2)))
    return result

# returns three tensors with coordinates for a grid of shape "ref_shape":
def get_indices(ref_shape):
    indices_x = np.arange(ref_shape[0])
    indices_y = np.arange(ref_shape[1])
    indices_x, indices_y = np.ix_(indices_x, indices_y)
    indices_x = np.broadcast_to(indices_x, ref_shape)
    indices_y = np.broadcast_to(indices_y, ref_shape)
    indices = np.stack([indices_x, indices_y], axis=-1)
    return indices, indices_x, indices_y

# Computation of loss function L_1:
# Scoring map: Tensor (b, p1, p2, q1, q2) subsampled in i-coordinates
# Ground truth: Tensor (b, p1, p2, 2) subsampled in k-coordinates
def compute_loss(scoring_map, ground_truth):
    scoring_map = tf.squeeze(scoring_map, axis=0)
    ground_truth = tf.squeeze(ground_truth, axis=0)


    # i_ones = np.arange(ground_truth.shape.as_list()[0])
    # i_twos = np.arange(ground_truth.shape.as_list()[1])
    # i_ones, i_twos = np.ix_(i_ones, i_twos)
    # i_ones = np.broadcast_to(i_ones, ground_truth.shape[:2])
    # i_twos = np.broadcast_to(i_twos, ground_truth.shape[:2])
    #
    # indices = tf.stack([i_ones, i_twos, ground_truth[:, :, 0], ground_truth[:, :, 1]], axis=-1)

    # TERM 1: S(gamma_0(p)|p):
    indices, indices_x, indices_y = get_indices(ground_truth.shape.as_list()[:2])
    indices = tf.stack([indices_x, indices_y, ground_truth[:, :, 0], ground_truth[:, :, 1]], axis=-1)

    # Replace occlusions and out-of-radius matches with 0:
    condition_a = tf.less(indices, 0)
    case_true = tf.cast(tf.fill(indices.shape, 0), dtype="int32")
    case_false = tf.cast(indices, dtype="int32")
    indices = tf.where(condition_a, case_true, case_false)

    matching_scores = tf.gather_nd(scoring_map, indices)
    matching_scores = tf.reshape(matching_scores, [matching_scores.shape[0], matching_scores.shape[1], 1, 1])

    matching_scores = broadcast(matching_scores, scoring_map.shape)

    # TERM 2: 1 - g_sigma(q - gamma_0(p))
    indices2, indices_x, indices_y = get_indices(scoring_map.shape.as_list()[2:])

    q = broadcast(indices2, scoring_map.shape.as_list() + [2])
    q = tf.cast(q, dtype=scoring_map.dtype)
    truth_broadcasted = tf.reshape(ground_truth,
                                   [ground_truth.shape[0], ground_truth.shape[1], 1, 1, ground_truth.shape[2]])
    truth_broadcasted = broadcast(truth_broadcasted, q.shape)
    truth_broadcasted = tf.cast(truth_broadcasted, dtype=scoring_map.dtype)

    diff = q - truth_broadcasted
    indicated = indicator_function(diff, 3)     # Parameter sigma is hardcoded here!

    # Replace -inf scores by arbitrary negative number:
    condition_b = tf.less(matching_scores, 0)
    case_true = tf.fill(matching_scores.shape, -42.0)
    case_false = matching_scores
    matching_scores = tf.where(condition_b, case_true, case_false)

    # Add the individual terms:
    summed = 1 - indicated + scoring_map - matching_scores
    # Take max(summed, 0):
    summed = tf.clip_by_value(summed, 0, np.inf)
    # Remove contribution of occlusions:
    valid_indices = tf.where(tf.reduce_all(tf.is_finite(ground_truth), axis=-1))
    # Compute mean of contributions of valid indices:
    sumsum = tf.reduce_mean(tf.gather_nd(summed, valid_indices), keepdims=True)
    return sumsum

# Computation of alternative loss function L_2:
def compute_loss_alternative(scoring_map, ground_truth):
    scoring_map = tf.squeeze(scoring_map, axis=0)
    ground_truth = tf.squeeze(ground_truth, axis=0)

    indices2, indices_x, indices_y = get_indices(scoring_map.shape.as_list()[2:])

    q = broadcast(indices2, scoring_map.shape.as_list() + [2])
    q = tf.cast(q, dtype=scoring_map.dtype)
    truth_broadcasted = tf.reshape(ground_truth,
                                   [ground_truth.shape[0], ground_truth.shape[1], 1, 1, ground_truth.shape[2]])
    truth_broadcasted = broadcast(truth_broadcasted, q.shape)
    truth_broadcasted = tf.cast(truth_broadcasted, dtype=scoring_map.dtype)

    diff = q - truth_broadcasted
    indicated = indicator_function(diff, 3)  # TODO: Parameter Sigma

    summed = scoring_map - indicated
    summed = tf.clip_by_value(summed, 0, np.inf)
    summed = summed**2

    # Remove contribution of occlusions:
    valid_indices = tf.where(tf.reduce_all(tf.is_finite(ground_truth), axis=-1))
    sumsum = tf.reduce_mean(tf.gather_nd(summed, valid_indices), keepdims=True)
    return sumsum

# Callback class for tracking the values exponents:
class ExponentHistory(Callback):
    def __init__(self, dir, num_exponents, **kwargs):
        self.num_exponents = num_exponents
        self.exponents = []
        self.dir = dir
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.exponents.append(self.model.get_weights()[-self.num_exponents:])
        np.save(self.dir, np.asarray(self.exponents).T)


# Callback class for looking at the gradients:
# It works but should only be used for debugging!
class GradientPrint(Callback):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(**kwargs)

    def on_batch_end(self, epoch, logs=None):
        data = self.data.__next__()

        data = data[0] + [np.ones(1), np.ones(1)] + data[1]
        inputs = data + [0]
        grads = self.get_gradients(inputs)
        print(grads)

    def on_train_begin(self, logs=None):
        weights = self.model.trainable_weights
        optimizer = self.model.optimizer
        gradients = optimizer.get_gradients(self.model.total_loss, weights)
        input_tensors = self.model.inputs + self.model.sample_weights + self.model.targets + [K.learning_phase()]
        self.get_gradients = K.function(inputs=input_tensors, outputs=gradients)
