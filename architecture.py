#!/usr/bin/env python3
"""
Class for large network with multiple branches, cost functions, training stages.
"""

# imports
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.utils.visualize_util import plot
import tensorflow as tf
import numpy as np

import network_params


MODELS_FOLDER = 'models'


def make_trainable(model, trainable):
    """
    Freeze or unfreeze network weights
    :param model: particular model (layers of neural net)
    :param trainable: if true freeze, else unfreeze
    :return: None
    """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def loss_diff(y_true, y_pred):
    # true gradients
    grad_true_y = y_true[:-1, :] - y_true[1:, :]
    grad_true_x = y_true[:, :-1] - y_true[:, 1:]
    grad_pred_y = y_pred[:-1, :] - y_pred[1:, :]
    grad_pred_x = y_pred[:, :-1] - y_pred[:, 1:]

    grad_diff = tf.abs(grad_true_x - grad_pred_x) + tf.abs(grad_true_y - grad_pred_y)
    grad_cost = tf.reduce_sum(grad_diff)

    return grad_cost


class MultiNetwork(object):
    def __init__(self, **kwargs):
        self.models_folder = kwargs.get('models_folder', MODELS_FOLDER)

        self.structure = kwargs.get('structure', network_params.DEFAULT_STRUCTURE)

        # branches of network
        self.encoder = None
        self.decoder = None
        self.physics_predictor = None
        self.action_mapper = None
        self.action_predictor = None
        self.state_sampler = None

        self.encoder_disc = None
        self.screen_discriminator = None
        self.state_discriminator = None

        # full networks
        self.autoencoder_gen = None
        self.autoencoder_disc = None
        self.autoencoder_gan = None

        self.screen_predictor_g = None
        self.screen_predictor_d = None

        self.state_assigner = None
        self.future_sampler_g = None
        self.future_sampler_d = None

        self.build_branches()
        self.build_networks()

    def build_branches(self):
        self.encoder = self.build_branch(self.structure['encoder'])
        self.decoder = self.build_branch(self.structure['decoder'])

        self.encoder_disc = self.build_branch(self.structure['encoder'])
        self.screen_discriminator = self.build_branch(self.structure['screen_discriminator'])

        # self.physics_predictor = self.build_physics_predictor()
        # self.action_mapper = self.build_action_mapper()
        # self.action_predictor = self.build_action_predictor()
        #
        # self.state_sampler = self.build_state_sampler()
        # self.decoder = self.build_decoder()

    def build_networks(self):
        self.build_autoencoder()
        self.build_ae_gan()

    def build_branch(self, structure):
        input_shape = structure.get('input_shape')
        output_shape = structure.get('output_shape')
        name = structure.get('name')

        layers = structure.get('layers')

        input_layer = Input(shape=input_shape)
        x = input_layer

        for layer in layers:
            layer_constructor = layer.get('type')
            pos_args = layer.get(network_params.POSITIONAL_ARGS, [])
            key_args = layer.get(network_params.KEYWORD_ARGS, {})
            # print('Building: ', layer_constructor, pos_args, key_args)
            x = layer_constructor(*pos_args, **key_args)(x)

        branch = Model(input_layer, x, name=name)
        # branch.summary()
        test_data = np.zeros([1] + list(input_shape))
        res = branch.predict(test_data)

        if not branch.output_shape[1:] == output_shape:
            raise ValueError('Bad output shape! Expected: {0} Actual: {1}'.format(output_shape, branch.output_shape))

        plot(branch, to_file='{0}/{1}.png'.format(self.models_folder, name), show_layer_names=True, show_shapes=True)

        return branch

    def build_autoencoder(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z = self.encoder(input_img)
        screen_recon = self.decoder(z)

        self.autoencoder_gen = Model(input_img, screen_recon)
        # self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss=loss_diff)
        self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss='mse')
        # self.autoencoder_gen.summary()
        plot(self.autoencoder_gen, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gen'), show_layer_names=True,
             show_shapes=True)

    def build_ae_gan(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z_disc = self.encoder_disc(input_img)
        screen_disc = self.screen_discriminator(z_disc)

        self.autoencoder_disc = Model(input_img, screen_disc)
        self.autoencoder_disc.compile(optimizer='adam', loss='binary_crossentropy')
        # self.autoencoder_disc.summary()
        plot(self.autoencoder_disc, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_disc'),
             show_layer_names=True,
             show_shapes=True)

        screen_recon = self.autoencoder_gen(input_img)
        fakeness = self.autoencoder_disc(screen_recon)

        self.autoencoder_gan = Model(input_img, fakeness)
        self.autoencoder_gan.compile(optimizer='adam', loss='binary_crossentropy')
        # self.autoencoder_gan.summary()
        plot(self.autoencoder_gan, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gan'),
             show_layer_names=True,
             show_shapes=True)

    def build_physics_predictor(self):
        return self

    def build_action_predictor(self):
        return self

    def build_state_sampler(self):
        return self

    def train_network(self):
        # stages:
        # 1) encoder/decoder
        return self

    def train_batch_ae_discriminator(self, real_images, test=False):
        if not self.autoencoder_disc.trainable:
            make_trainable(self.autoencoder_disc, True)
            self.autoencoder_disc.compile(optimizer='adam', loss='binary_crossentropy')
            # raise ValueError('Discriminator must be trainable')

        batch_size = 64

        labels = np.zeros((batch_size,))
        labels[:int(batch_size/2)] = 1

        # indices = np.random.randint(0, real_images.shape[0], size=int(batch_size/2))
        # real = real_images[indices, ...]

        # generate fake images
        fake_images = self.autoencoder_gen.predict(real_images)

        train = np.concatenate((real_images, fake_images))

        if test:
            loss = self.autoencoder_disc.test_on_batch(train, labels)
        else:
            loss = self.autoencoder_disc.train_on_batch(train, labels)

        return loss

    def train_batch_ae_gan(self, real_images):
        batch_size = 32

        if self.autoencoder_disc.trainable:
            make_trainable(self.autoencoder_disc, False)
            self.autoencoder_disc.compile(optimizer='adam', loss='binary_crossentropy')
            # raise ValueError('Discriminator must not be trainable')

        labels = np.ones((batch_size,))

        loss = self.autoencoder_gan.train_on_batch(real_images, labels)

        return loss

    def show_reconstruction(self):
        return self

    def show_predictions(self):
        return self


if __name__ == '__main__':
    mn = MultiNetwork()


