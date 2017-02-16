import numpy as np
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




class MyImageGenerator(object):
    def __init__(self, file, batch_size, im_shape):
        self.file = file
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.im_med = np.zeros(self.im_shape)

        self.single_image = read_and_decode_single_example(file)
        self.single_image.set_shape(self.im_shape)
        self.shuffled_batch = tf.train.shuffle_batch([self.single_image],
                                                     batch_size=self.batch_size,
                                                     capacity=10000,
                                                     min_after_dequeue=1000)

        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

        ims = self.get_n_batches_images(n=30, subtract_median=False)
        self.im_med = np.median(ims, axis=0)

    def get_single_image(self):
        """Returns the next image from the file.

        :return: float image with values between 0 and 1 -- np.array of shape im_shape

        """
        im = self.sess.run(self.single_image)
        return im

    def get_n_ordered_images(self, n=1000):
        """Returns n images in the order they are stored in the file.

        :param n: number of images
        :return: float images with values between 0 and 1 -- np.array of shape (n, *im_shape)
        """
        ims = []
        for i in range(n):
            im = self.get_single_image()
            ims.append(im)

        ims = np.array(ims)
        ims = np.reshape(ims, [ims.shape[0]] + list(self.im_shape))
        return ims

    def get_shuffled_batch(self, subtract_median=True):
        batch = self.sess.run(self.shuffled_batch)

        # subtract median image and make sure result is between 0 and 1
        if subtract_median:
            batch -= self.im_med - 1
            batch /= 2

        return batch

    def get_n_batches_images(self, n=10, subtract_median=True):
        ims = []
        for i in range(n):
            im = self.get_shuffled_batch(subtract_median)
            ims.append(im)

        ims = np.concatenate(ims, axis=0)
        ims = np.reshape(ims, [ims.shape[0]] + list(self.im_shape))
        return ims

