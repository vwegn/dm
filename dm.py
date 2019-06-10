import os
# os.environ["CUDA_VISIBLE_DEVICES"]="" # For computation only on CPU activate this.

import math
import time
import tensorflow as tf
import numpy as np
from keras import applications
from keras import Sequential, Model
from keras.layers import Input, Add, Lambda, Reshape, Cropping2D
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

from PIL import Image, ImageDraw
from keras.regularizers import l2

from custom_layers import PrepareForExtraction, Aggregation, BottomLevel, MaxPoolingWithArgmax, Disaggregation, \
    Unpooling, ExponentHistory, GradientPrint, compute_loss, compute_loss_alternative, get_indices

import flow_IO as IO    # script for reading data provided by Sintel
from skimage.util.shape import view_as_windows


def preprocess_image(img):
    ret = image.img_to_array(img)
    ret = preprocess_input(ret, mode="tf")  # Note: Pixel values are divided by 127.5 and then subtracted from 1
    return ret


# def compute_coordinate_i(p1, p2, alpha, beta):
#     i1 = (p1 - beta) / alpha
#     i2 = (p2 - beta) / alpha
#     i1 = np.rint(i1)
#     i2 = np.rint(i2)
#     return i1, i2


def compute_coordinate_k(q1, q2, p1, p2, const_r, const_l):
    k1 = (q1 - p1) / 2 ** const_l + (const_r / 2 ** const_l)
    k2 = (q2 - p2) / 2 ** const_l + (const_r / 2 ** const_l)
    k1 = np.rint(k1)
    k2 = np.rint(k2)
    return k1, k2


def read_training_data():
    path_training_sequences = 'MPI-Sintel/training/clean'

    sequences = {}
    for s in os.listdir(path_training_sequences):
        try:
            num_frames = len(os.listdir(path_training_sequences + "/" + s))
            sequences[s] = num_frames
        except NotADirectoryError:
            continue
    # sequences = {'alley_2':50, 'market_2':50}   # hardcoded fast way for debugging
    # sequences = {'sleeping_2': 50, 'alley_2': 50, 'market_2': 50, 'shaman_2': 50, 'temple_2': 50, 'cave_2': 50, 'bandage_2': 50, 'bamboo_2': 50}
    print(sequences)
    return sequences


def subsample_window(window, alpha, beta, axis):
    if axis == 0:
        window = window[beta::alpha, beta::alpha]
    elif axis == 1:
        window = window[:, beta::alpha, beta::alpha]
    else:
        print("Unsupported!")
    return window


# generator which outputs data with two modes "train" and "match"
def generator(sequences, stride, offset, radius, subsampled_padding, valid_patch_length, total_patch_length, ref_padding=0, mode="match"):
    keys = list(sequences.keys())

    # Loop over Sintel sequences:
    i = 0
    skipped_windows = 0
    total_windows = 0
    while i < len(keys):
        j = 1
        # Loop over frames in sequence:
        while j < sequences[keys[i]]:
            # Define data paths:
            path_ref = "MPI-Sintel/training/clean/" + keys[i] + "/frame_{:04d}".format(j) + ".png"
            path_tar = "MPI-Sintel/training/clean/" + keys[i] + "/frame_{:04d}".format(j + 1) + ".png"
            path_flo = "MPI-Sintel/training/flow/" + keys[i] + "/frame_{:04d}".format(j) + ".flo"
            path_occ = "MPI-Sintel/training/occlusions/" + keys[i] + "/frame_{:04d}".format(j) + ".png"

            # Load data:
            ref_image = image.load_img(path_ref)
            tar_image = image.load_img(path_tar)
            flo_truth = IO.read(path_flo)
            occlusion = image.load_img(path_occ, grayscale=True)

            # Preprocess data:
            ref_image = preprocess_image(ref_image)
            tar_image = preprocess_image(tar_image)
            occlusion = np.asarray(occlusion)

            # Create windows:
            ref_window_size = total_patch_length
            tar_window_size = 2 * radius + total_patch_length

            # Additional padding for matching (so that border pixels are matched):
            if mode == "match":
                # ref_padding needed to match boundary pixels, total_patch_length padding needed to get whole image
                ref_image = np.pad(ref_image, [(ref_padding, total_patch_length - ref_padding), (ref_padding, total_patch_length - ref_padding), (0, 0)], 'reflect')
                tar_image = np.pad(tar_image, [(ref_padding + radius, total_patch_length - ref_padding + radius), (ref_padding + radius, total_patch_length - ref_padding + radius), (0, 0)], 'reflect')
                flo_truth = np.pad(flo_truth, [(ref_padding, total_patch_length - ref_padding), (ref_padding, total_patch_length - ref_padding), (0, 0)], 'constant', constant_values=-np.inf)
                occlusion = np.pad(occlusion, [(ref_padding, total_patch_length - ref_padding), (ref_padding, total_patch_length - ref_padding)], 'constant', constant_values=255)
            elif mode == "train":
                tar_image = np.pad(tar_image, [(radius, radius), (radius, radius), (0, 0)], 'reflect')

            # Extract windows from frames:
            truth_indices, truth_indices_x, truth_indices_y = get_indices((ref_window_size, ref_window_size))
            ref_image_windows = view_as_windows(ref_image, (ref_window_size, ref_window_size, 3), step=valid_patch_length)
            tar_image_windows = view_as_windows(tar_image, (tar_window_size, tar_window_size, 3), step=valid_patch_length)

            flo_truth_windows = view_as_windows(flo_truth, (ref_window_size, ref_window_size, 2), step=valid_patch_length)
            occlusion_windows = view_as_windows(occlusion, (ref_window_size, ref_window_size), step=valid_patch_length)

            # Loop over windows (in two dimensions):
            m = 0
            while m < ref_image_windows.shape[0]:
                n = 0
                while n < ref_image_windows.shape[1]:
                    total_windows += 1
                    # Current window:
                    ref_window = ref_image_windows[m, n, 0]
                    tar_window = tar_image_windows[m, n, 0]
                    flo_window = flo_truth_windows[m, n, 0]
                    occ_window = occlusion_windows[m, n]
                    n += 1

                    # Adding batch_size:
                    ref_window = np.expand_dims(ref_window, 0)
                    tar_window = np.expand_dims(tar_window, 0)

                    # Compute ground truth coordinates from optical flow:
                    flo_window = np.rint(flo_window[:, :, ::-1])  # Note, that x and y coordinates have to be flipped.
                    truth = np.rint(truth_indices + flo_window)

                    # Set truth values for occluded points to -np.inf:
                    truth[occ_window == 255, :] = -np.inf

                    # Set out of radius coordinates to +np.inf:
                    valid = np.all(np.isfinite(flo_window), axis=-1)
                    truth[np.logical_and(np.linalg.norm(flo_window, axis=-1, ord=np.inf) > radius, valid)] = np.inf

                    # Discretization of ground truth via discretization scheme:
                    k_ones, k_twos = compute_coordinate_k(truth[:, :, 0], truth[:, :, 1], truth_indices_x,
                                                          truth_indices_y, radius, 0)
                    discretized_truth = np.stack([k_ones, k_twos], axis=-1)

                    # Subsampling of ground truth for computation of accuracy:
                    subsampled_truth = subsample_window(discretized_truth, stride, offset, axis=0)
                    subsampled_truth = subsampled_truth[subsampled_padding:-subsampled_padding,
                                       subsampled_padding:-subsampled_padding]

                    if mode == "train":
                        # Skip window if it contains out of radius coordinate or is completely occluded:
                        if np.any(subsampled_truth == np.inf) or np.all(subsampled_truth == -np.inf):
                            skipped_windows += 1
                            continue
                    elif mode == "match":
                        count_oor = np.count_nonzero(np.any(subsampled_truth == np.inf, axis=-1))

                    # Adding batch_size:
                    subsampled_truth = np.expand_dims(subsampled_truth, 0)

                    # Different outputs for training and matching are needed:
                    if mode == "train":
                        yield [ref_window, tar_window, subsampled_truth], [np.zeros((1, 1, 1)), subsampled_truth]
                    elif mode == "match":
                        yield [ref_window, tar_window, subsampled_truth, ref_image_windows.shape, count_oor]
                m += 1
            j += 1
            # print("{}/{} frames processed. \n".format(j, sequences[keys[i]]))
        i += 1
        # print("{}/{} sequences processed. \n".format(i, len(keys)))
        # Skip last frame in sequence since it has no target frame:
        if i == len(keys):
            i = 0
            # print("All sequences processed. Skipped windows: " + str(skipped_windows))
            # print("Total windows: " + str(total_windows))
            skipped_windows = 0


def dm_match(stride, offset, radius, levels, valid_patch_length, valid_dilation_steps, share_exponent):
    # Start timer:
    time2 = time.time()

    # Define size of windows:
    valid_patch_length = stride * valid_patch_length
    ref_padding = sum([2**i for i in range(levels - valid_dilation_steps, levels)]) * stride  # e.g. (2^1 + 2^2) * 8 = 48
    total_patch_length = valid_patch_length + 2 * ref_padding  # e.g. 56 + 96 = 152

    # Define size of patch, which are windows without padding:
    p_patch_shape = (total_patch_length, total_patch_length)
    q_patch_shape = (total_patch_length + 2 * radius, total_patch_length + 2 * radius)
    subsampled_ref_padding = int(ref_padding / stride)

    # Create Keras Input layers for two images:
    input_p = Input(shape=p_patch_shape + (3,), batch_shape=(1,) + p_patch_shape + (3,))
    input_q = Input(shape=q_patch_shape + (3,), batch_shape=(1,) + q_patch_shape + (3,))

    # Use pre-trained VGG network to compute image descriptors:
    p, q = calculate_descriptors_by_vgg(input_p, input_q)

    # Build DeepMatching tensorflow graph:
    dm = deep_matching(p, q, stride, offset, radius, levels, share_exponent=share_exponent)

    # Subsample result of DeepMatching:
    dm_shape = dm.shape.as_list()
    dm = Reshape([dm.shape.as_list()[1], dm.shape.as_list()[2], -1])(dm)
    dm = Cropping2D(subsampled_ref_padding)(dm)
    dm = Reshape([dm.shape.as_list()[1], dm.shape.as_list()[2], dm_shape[3], dm_shape[4]])(dm)

    indices, indices_x, indices_y = get_indices(p_patch_shape)

    indices_x = indices_x[offset::stride, offset::stride]
    indices_y = indices_y[offset::stride, offset::stride]

    indices_x = indices_x[subsampled_ref_padding:-subsampled_ref_padding, subsampled_ref_padding:-subsampled_ref_padding]
    indices_y = indices_y[subsampled_ref_padding:-subsampled_ref_padding, subsampled_ref_padding:-subsampled_ref_padding]

    # compute matches from scoring map:
    matches = Lambda(
        lambda x: tf.expand_dims(
            tf.py_func(get_match, [x, indices_x, indices_y, stride, offset, radius, 0], Tout=tf.float32), axis=0),
        output_shape=indices_x.shape + (2,), name="matches")(dm)

    # build Keras model for matching:
    model = Model(inputs=[input_p, input_q], outputs=[matches])

    model.load_weights("weights.h5py")    # initialize weights from training

    # Define different counting variables.
    count_all = 0
    count_cor_2 = 0  # Number of correct matches
    count_cor_10 = 0
    count_occ = 0
    count_oor = 0
    count_cor_5 = 0
    sum_norm = 0

    sequences = read_training_data()
    keys = list(sequences.keys())

    full_image = Image.new('RGB', (1024, 436))    # for saving an image
    gen = generator(sequences, stride, offset, radius, subsampled_ref_padding, valid_patch_length, total_patch_length, ref_padding=ref_padding, mode="match")
    x = gen.__next__()

    for m in range(0, len(keys)):
        for n in range(1, sequences[keys[m]]):
            for j in range(0, x[3][0]):
                for i in range(0, x[3][1]):
                    p = x[0]
                    t = x[2]
                    oor = x[4]

                    result = model.predict(x[:2], batch_size=1)
                    normed = np.linalg.norm(result - t, axis=-1, ord=2)

                    sum_norm += np.sum(normed[np.isfinite(normed)])
                    count_all += t.shape[1] * t.shape[2]
                    count_cor_2 += np.count_nonzero(normed <= 2)
                    count_cor_5 += np.count_nonzero(normed <= 5)
                    count_cor_10 += np.count_nonzero(normed <= 10)
                    count_occ += np.count_nonzero(np.all(t < 0, axis=-1)) - oor
                    count_oor += oor

                    dict = {}
                    dict["Matches"] = count_all
                    dict["CorrectT3"] = count_cor_2
                    dict["CorrectT10"] = count_cor_10
                    dict["Occluded"] = count_occ
                    dict["Out"] = count_oor
                    with open('matches.txt', 'w') as f:
                        print(dict, file=f)

                    temp_img = Image.fromarray(
                        ((np.squeeze(p)[ref_padding:-ref_padding, ref_padding:-ref_padding] + 1) * 127.5).astype('uint8'), 'RGB')
                    temp_img = visualize_correct_matches(np.squeeze(result), np.squeeze(t), temp_img, 2, stride, offset)
                    full_image.paste(temp_img, (i * valid_patch_length, j * valid_patch_length))
                    x = gen.__next__()

    acc_occ_2 = count_cor_2 / (count_all - count_occ)
    acc_occ_oor_2 = count_cor_2 / (count_all - count_occ - count_oor)

    acc_occ_5 = count_cor_5 / (count_all - count_occ)
    acc_occ_oor_5 = count_cor_5 / (count_all - count_occ - count_oor)

    acc_occ_10 = count_cor_10 / (count_all - count_occ)
    acc_occ_oor_10 = count_cor_10 / (count_all - count_occ - count_oor)

    # Print metrics:
    print("AccT3:")
    print(acc_occ_2)  # acc without occlusions
    print(acc_occ_oor_2)  # acc without occlusions and oor
    print("AccT5:")
    print(acc_occ_5)
    print(acc_occ_oor_5)
    print("AccT10:")
    print(acc_occ_10)
    print(acc_occ_oor_10)
    print("epe:")
    print((sum_norm) / (count_all - count_occ))

    # full_image.save("test.png")   # Activate for image saving

    print("--- Runtime without imports: %s seconds ---" % (time.time() - time2))


def dm_train(stride, offset, radius, levels, valid_dilation_steps=2, valid_patch_length=7, share_exponent=True, exponents=None):
    # Start timer:
    time2 = time.time()

    # Define size of windows:
    valid_patch_length = valid_patch_length * stride  # e.g. 7 * 8 = 56
    ref_padding = sum([2**i for i in range(levels - valid_dilation_steps, levels)]) * stride  # e.g. (2^1 + 2^2) * 8 = 48
    total_patch_length = valid_patch_length + 2 * ref_padding  # e.g. 56 + 96 = 152

    p_patch_shape = (total_patch_length, total_patch_length)
    q_patch_shape = (2 * radius + total_patch_length, 2 * radius + total_patch_length)
    subsampled_ref_padding = int(ref_padding / stride)

    p_patch_sub_shape = math.floor((p_patch_shape[0] - offset) / stride) + 1 - subsampled_ref_padding * 2

    # Create Keras Input layers for two images and ground truth:
    input_p = Input(shape=p_patch_shape + (3,), batch_shape=(1,) + p_patch_shape + (3,))
    input_q = Input(shape=q_patch_shape + (3,), batch_shape=(1,) + q_patch_shape + (3,))
    input_truth = Input(shape=(p_patch_sub_shape, p_patch_sub_shape) + (2,),
                        batch_shape=(1,) + (p_patch_sub_shape, p_patch_sub_shape) + (2,))

    # Use pre-trained VGG network to compute image descriptors:
    p, q = calculate_descriptors_by_vgg(input_p, input_q)

    if share_exponent:
        num_exponents = 1
    else:
        num_exponents = levels

    if exponents is None:
        exponents = [1.4] * num_exponents

    # Build DeepMatching tensorflow graph:
    dm = deep_matching(p, q, stride, offset, radius, levels, share_exponent=share_exponent, exponents=exponents)

    # Subsample result of DeepMatching:
    dm = Lambda(lambda x: x[:, subsampled_ref_padding:-subsampled_ref_padding, subsampled_ref_padding:-subsampled_ref_padding, :, :])(dm)

    # Build loss function:
    loss = Lambda(lambda x: compute_loss(x[0], x[1]), output_shape=[1, 1], name="unreg_loss")([dm, input_truth])
    # change this to use "compute_loss_alternative" for L_2

    indices, indices_x, indices_y = get_indices(p_patch_shape)

    indices_x = indices_x[offset::stride, offset::stride]
    indices_y = indices_y[offset::stride, offset::stride]

    indices_x = indices_x[subsampled_ref_padding:-subsampled_ref_padding, subsampled_ref_padding:-subsampled_ref_padding]
    indices_y = indices_y[subsampled_ref_padding:-subsampled_ref_padding, subsampled_ref_padding:-subsampled_ref_padding]

    # extract matches for use in metrics:
    matches = Lambda(
        lambda x: tf.py_func(get_match, [x, indices_x, indices_y, stride, offset, radius, 0], Tout=tf.float32),
        output_shape=indices_x.shape + (2,), name="matches")(dm)

    # Define loss function wrapper for Keras:
    def loss_func(y_true, y_pred):
        return y_pred

    # Define Keras model for loss (including DeepMatching):
    model_loss = Model(inputs=[input_p, input_q, input_truth], outputs=[loss, matches])
    # model_loss.summary()  # Activate this for summary of Keras layers of the model!

    # For testing and debugging:
    # gen = generator(read_training_data(), stride, offset, radius, subsampled_ref_padding, valid_patch_length, total_patch_length, mode="train")
    # while gen.__next__():
    #     pass

    # Choose optimizer:
    # opt = Adam()
    opt = SGD(lr=0.0001)
    #opt = SGD()

    # Compile model:
    model_loss.compile(opt,
                       loss={'unreg_loss': loss_func, 'matches': (lambda y_true, y_pred: tf.zeros(1))},
                       metrics={'matches': [accuracy_metric_2, accuracy_metric_5, accuracy_metric_10, end_point_error_metric]})

    # Training:
    num_true_epochs = 25
    num_epochs = num_true_epochs * 1
    history = model_loss.fit_generator(generator(read_training_data(),
                                                 stride, offset, radius,
                                                 subsampled_ref_padding,
                                                 valid_patch_length,
                                                 total_patch_length,
                                                 mode="train"),
                                       steps_per_epoch=2315, # 2315 für 2_Seq, 8639 für 9_Seq, 8543 für 8_Seq
                                       epochs=num_epochs,
                                       callbacks=[ModelCheckpoint("weights.h5py", save_weights_only=True),
                                                  ExponentHistory("exponents.npy", num_exponents),
                                                  #EarlyStopping(monitor="loss"),
                                                  #GradientPrint(gen),
                                                  TerminateOnNaN(),
                                                  ],
                                       verbose=1)
    np.save("history.npy", history.history)

    print("--- Runtime without imports: %s seconds ---" % (time.time() - time2))


def calculate_descriptors_by_vgg(input_p, input_q):
    # Load pretrained VGG Network:
    vgg = applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_q.shape[1:].as_list())
    # print(vgg.summary())

    # Load VGG Network with Keras Sequantial model:
    vgg_partial = Sequential()
    for i in range(0, 6):
        if i is not 3:
            vgg_layer = vgg.layers[i]
            vgg_layer.kernel_regularizer = l2(0.001)
            vgg_layer.trainable = True
            vgg_partial.add(vgg_layer)
    # print(vgg_partial.summary())

    # Extract image descriptors:
    p = vgg_partial(input_p)
    q = vgg_partial(input_q)
    return p, q

# main DeepMatching architecture:
def deep_matching(p, q, alpha, beta, radius, max_level, share_exponent, exponents=None):
    # STEP 0: Preprocess Input (Descriptor extraction, normalization, patch extraction)
    print("STEP 0: Preprocessing")

    # Normalize inputs and subsample first frame / target image:
    p, q = PrepareForExtraction(alpha, beta, radius)([p, q])

    # Compute bottom level scoring map via inner product:
    bottom_level = BottomLevel(alpha, beta, radius)([p, q])

    # Define layer lists:
    scoring_maps = []
    refined_maps = []
    pooling_args = []

    # Add bottom level (S_0) as first scoring map:
    scoring_maps.append(bottom_level)
    print("S" + str(0) + " " + str(scoring_maps[0].shape))

    # STEP 2: Compute higher level scoring maps (bottom-up):
    print("STEP 2: Computing higher level scoring maps (bottom-up)")
    for i in range(0, max_level):
        # Max Pooling (with argmax):
        with tf.device('/cpu:0'):  # Doesn't work properly without this!
            intermediate_map, argmax = MaxPoolingWithArgmax()(scoring_maps[i])
        # Aggregation:
        if share_exponent:
            if i == 0:
                if exponents is not None:
                    agg_layer = Aggregation(i, shared=True, initial_exponent=exponents[i])
                else:
                    agg_layer = Aggregation(i, shared=True)
                exponent = agg_layer.get_exponent()
                aggregated_map = agg_layer(intermediate_map)
            else:
                aggregated_map = Aggregation(i, exponent, shared=True)(intermediate_map)
        elif exponents is not None:
            aggregated_map = Aggregation(i, shared=False, initial_exponent=exponents[i])(intermediate_map)
        else:
            aggregated_map = Aggregation(i, shared=False)(intermediate_map)
        # Add higher level scoring maps (S_1, S_2, ..., S_L) with L = max_level:
        scoring_maps.append(aggregated_map)
        # Save arguments of max pooling for unpooling in step 3:
        pooling_args.append(argmax)
        print("S" + str(i + 1) + " " + str(scoring_maps[i + 1].shape))
    # STEP 3: Compute refined scoring maps (top-down):
    print("STEP 3: Computing refined scoring maps (top-down)")

    # Set entry level (Q_L):
    refined_maps.append(scoring_maps[-1])
    print("Q" + str(max_level) + " " + str(refined_maps[0].shape))

    for i in range(max_level, 0, -1):
        # Disaggregation:
        disaggregated_map = Disaggregation(i - 1)(refined_maps[-1])
        # Unpooling:
        unpooled_map = Unpooling(scoring_maps[i - 1].shape.as_list())([disaggregated_map, pooling_args[i - 1]])
        # Add refined scoring map to respective scoring map from before:
        refined_maps.append(Add()([unpooled_map, scoring_maps[i - 1]]))
        print("Q" + str(i - 1) + " " + str(refined_maps[-1].shape))
    print("Building complete!")
    return refined_maps[-1]

# Extracts matches from DeepMatching scoring map:
def get_match(dm, p1, p2, alpha, beta, r, l):
    patch = dm[0, :, :, :, :]
    patch_shape = patch.shape
    patch = patch.reshape([patch.shape[0] * patch.shape[1], patch.shape[2] * patch.shape[3]])
    argmax = np.argmax(patch, axis=-1)
    k1, k2 = np.unravel_index(argmax, patch_shape[2:])
    k1 = k1.reshape(p1.shape)
    k2 = k2.reshape(p2.shape)

    return np.stack([k1, k2], axis=-1).astype(np.float32)


def accuracy_metric_2(y_true, y_pred):
    return accuracy_occluded(2, y_pred, y_true)


def accuracy_metric_5(y_true, y_pred):
    return accuracy_occluded(5, y_pred, y_true)


def accuracy_metric_10(y_true, y_pred):
    return accuracy_occluded(10, y_pred, y_true)


def end_point_error_metric(y_true, y_pred):
    norm = tf.norm(y_pred - y_true, axis=-1, ord='euclidean')
    cond = tf.is_finite(norm)
    cond.set_shape(y_true.get_shape())
    return tf.reduce_mean(tf.boolean_mask(norm, cond))


def accuracy_occluded(threshold, correspondence, truth):
    normed = tf.norm(correspondence - truth, axis=-1, ord='euclidean')
    count = tf.count_nonzero(tf.less_equal(normed, threshold), dtype=tf.float64)
    greater_truth = tf.count_nonzero(tf.reduce_all(tf.greater_equal(truth, 0), axis=-1),
                                     dtype=tf.float64)
    return count / greater_truth


def visualize_correct_matches(correspondence, truth, imageA, threshold, alpha, beta):
    img = imageA
    pixels = img.load()

    for i in range(0, correspondence.shape[0]):
        for j in range(0, correspondence.shape[1]):
            difference = np.linalg.norm(correspondence[i, j, :] - truth[i, j, :], ord=np.inf)
            p1 = i * alpha + beta
            p2 = j * alpha + beta
            if difference <= threshold:
                pixels[p2, p1] = (0, 255, 0)
                pixels[p2 + 1, p1] = (0, 255, 0)
                pixels[p2 - 1, p1] = (0, 255, 0)
                pixels[p2, p1 + 1] = (0, 255, 0)
                pixels[p2, p1 - 1] = (0, 255, 0)
            elif np.isfinite(difference):
                pixels[p2, p1] = (255, 0, 0)
                pixels[p2 + 1, p1] = (255, 0, 0)
                pixels[p2 - 1, p1] = (255, 0, 0)
                pixels[p2, p1 + 1] = (255, 0, 0)
                pixels[p2, p1 - 1] = (255, 0, 0)
            elif truth[i, j, 0] == np.inf or truth[i, j, 1] == np.inf:
                pixels[p2, p1] = (0, 255, 255)  # oor
                pixels[p2 + 1, p1] = (0, 255, 255)
                pixels[p2 - 1, p1] = (0, 255, 255)
                pixels[p2, p1 + 1] = (0, 255, 255)
                pixels[p2, p1 - 1] = (0, 255, 255)

    return img
    # img.show()


# Run training:
#dm_train(stride=8, offset=4, radius=40, levels=3, valid_dilation_steps=3, valid_patch_length=13, share_exponent=False)

# Run matching:
dm_match(stride=8, offset=4, radius=40, levels=3, valid_dilation_steps=3, valid_patch_length=13, share_exponent=False)

