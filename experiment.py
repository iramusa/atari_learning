# %% imports
import tensorflow as tf
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt


import architecture


import tensorflow as tf


def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_processed': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_processed'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image


def get_n_images(game, train, n=5000):
    if train is True:
        file_train = "full_images/" + game + "-" + "train" + ".tfrecords"
    else:
        file_train = "full_images/" + game + "-" + "valid" + ".tfrecords"

    # returns symbolic label and image
    image = read_and_decode_single_example(file_train)

    sess = tf.Session()

    # Required. See below for explanation
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # grab examples back.

    ims = []
    #     n = TRAIN_SIZE if train else VALID_SIZE
    for i in range(n):
        im = sess.run([image])
        ims.append(im[0])

    return ims


# x_valid = get_all_images(GAME, train=False)


if __name__ == '__main__':
    IMAGE_FOLDER = 'full_images'
    GAME = 'Freeway'

    x_train = get_n_images(GAME, train=True, n=1000)
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], 210, 160, 3))

    x_valid = get_n_images(GAME, train=True, n=100)
    x_valid = np.array(x_valid)
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 210, 160, 3))

    # print(X.shape)
    # plt.imshow(X[0, ...])
    # plt.show()

    mn = architecture.MultiNetwork()

    # r = mn.autoencoder_gen.predict(x_valid[:2,:,:,:])
    # print(r.shape)
    mn.autoencoder_gen.fit(x_train, x_train, nb_epoch=15, batch_size=32, shuffle=True, validation_data=(x_valid, x_valid))

