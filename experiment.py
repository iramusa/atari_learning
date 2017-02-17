#!/usr/bin/env python3

# %% imports
import architecture
import network_params
from image_generators import ImageGenerator

import tensorflow as tf


def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
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

# for testing
# with tf.device("/cpu:0"):

if __name__ == '__main__':
    IMAGE_FOLDER = 'full_images'
    GAME = 'Freeway'
    BATCH_SIZE = 32

    file_train = "{0}/{1}-{2}.tfrecords".format(IMAGE_FOLDER, GAME, 'train')
    # file_valid = "{0}/{1}-{2}.tfrecords".format(IMAGE_FOLDER, GAME, 'valid')

    train_gen = ImageGenerator(file_train, batch_size=32,
                               im_shape=network_params.INPUT_IMAGE_SHAPE,
                               buffer_size=BATCH_SIZE*200)
    # valid_gen = ImageGenerator(file_train, batch_size=32,
    #                            im_shape=network_params.INPUT_IMAGE_SHAPE,
    #                            buffer_size=160)

    mn = architecture.MultiNetwork()

    # mn.autoencoder_gen.tra
    # with tf.device(" / cpu: 0")
    mn.autoencoder_gen.fit_generator(train_gen.generate_ae(),
                                     samples_per_epoch=30*BATCH_SIZE,
                                     nb_epoch=30,
                                     max_q_size=1,
                                     verbose=2)
                                     # validation_data=valid_gen.generate_ae(),
                                     # nb_val_samples=3)

