#!/usr/bin/env python3
"""
Class for large network with multiple branches, cost functions, training stages.
"""

# imports
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Merge, LSTM
# import keras
from keras.utils.visualize_util import plot
import numpy as np

import structures


FOLDER_MODELS = 'models'

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


class MultiNetwork(object):
    def __init__(self, **kwargs):
        self.structure = kwargs.get('structure', structures.DEFAULT_STRUCTURE)

        # branches of network
        self.encoder = None
        self.decoder = None
        self.physics_predictor = None
        self.action_mapper = None
        self.action_predictor = None
        self.state_sampler = None

        self.screen_discriminator = None
        self.state_discriminator = None

        # full networks
        self.autoencoder_gen = None
        self.autoencoder_critic = None
        self.autoencoder_gan = None

        self.screen_predictor_g = None
        self.screen_predictor_d = None

        self.state_assigner = None
        self.future_sampler_g = None
        self.future_sampler_d = None

        self.build_branches()
        self.build_networks()

    def build_branches(self):
        self.encoder = self.build_branch(structures.ENCODER)
        self.decoder = self.build_branch(structures.DECODER)
        self.screen_discriminator = self.build_branch(structures.SCREEN_DISCRIMINATOR)

        # self.physics_predictor = self.build_physics_predictor()
        # self.action_mapper = self.build_action_mapper()
        # self.action_predictor = self.build_action_predictor()
        #
        # self.state_sampler = self.build_state_sampler()
        # self.decoder = self.build_decoder()

    def build_networks(self):
        self.build_autoencoder()

    def build_branch(self, structure):
        input_shape = structure.get('input_shape')
        output_shape = structure.get('output_shape')
        name = structure.get('name')

        layers = structure.get('layers')

        input_layer = Input(shape=input_shape)
        x = input_layer

        for layer in layers:
            layer_constructor = layer.get('type')
            pos_args = layer.get(structures.POSITIONAL_ARGS, [])
            key_args = layer.get(structures.KEYWORD_ARGS, {})
            # print('Building: ', layer_constructor, pos_args, key_args)
            x = layer_constructor(*pos_args, **key_args)(x)

        branch = Model(input_layer, x, name=name)
        # branch.summary()
        test_data = np.zeros([1] + list(input_shape))
        res = branch.predict(test_data)

        if not branch.output_shape[1:] == output_shape:
            raise ValueError('Bad output shape! Expected: {0} Actual: {1}'.format(output_shape, branch.output_shape))

        plot(branch, to_file='{0}/{1}.png'.format(FOLDER_MODELS, name), show_layer_names=True, show_shapes=True)

        return branch

    def build_autoencoder(self):
        input_img = Input(shape=structures.INPUT_IMAGE_SHAPE)
        z = self.encoder(input_img)
        screen_recon = self.decoder(z)
        screen_disc = self.screen_discriminator(z)

        self.autoencoder_gen = Model(input_img, screen_recon)
        self.autoencoder_gen.compile(optimizer='adam', loss='mse')
        self.autoencoder_gen.summary()
        plot(self.autoencoder_gen, to_file='{0}/{1}.png'.format(FOLDER_MODELS, 'autoencoder_gen'), show_layer_names=True,
             show_shapes=True)

        self.autoencoder_critic = Model(input_img, screen_disc)
        self.autoencoder_critic.summary()
        plot(self.autoencoder_critic, to_file='{0}/{1}.png'.format(FOLDER_MODELS, 'autoencoder_critic'), show_layer_names=True,
             show_shapes=True)

        z = self.encoder(input_img)
        screen_recon = self.decoder(z)
        z = self.encoder(screen_recon)
        screen_disc = self.screen_discriminator(z)

        self.autoencoder_gan = Model(input_img, screen_disc)
        self.autoencoder_gan.summary()
        plot(self.autoencoder_gan, to_file='{0}/{1}.png'.format(FOLDER_MODELS, 'autoencoder_gan'),
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
        #



        return self

    def show_reconstruction(self):
        return self

    def show_predictions(self):
        return self

if __name__ == '__main__':
    mn = MultiNetwork()


