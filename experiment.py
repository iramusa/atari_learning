#!/usr/bin/env python3

# %% imports
import sys
import os

import numpy as np
import pandas as pd
from PIL import Image
# from scipy.misc import imsave

import architecture
import network_params
from image_generators import ImageGenerator

DATA_FOLDER = 'full_images'
RECONSTRUCTIONS_FOLDER = 'reconstructions'
PLOTS_FOLDER = 'plots'
MODELS_FOLDER = 'models'
LOG_FILE = 'log.log'
ERR_FILE = 'err.log'

GAME = 'Freeway'
BATCH_SIZE = 32
BATCHES_PER_EPOCH = 100


class Experiment(object):
    def __init__(self, output_folder='generic_exp', description='', epochs=200, **kwargs):
        self.output_folder = output_folder
        self.reconstructions_folder = '{0}/{1}'.format(self.output_folder, RECONSTRUCTIONS_FOLDER)
        self.plots_folder = '{0}/{1}'.format(self.output_folder, PLOTS_FOLDER)
        self.models_folder = '{0}/{1}'.format(self.output_folder, MODELS_FOLDER)

        self.make_directories()

        # divert prints to log file, print line by line so the file can be read real time
        self.log_file = open('{0}/{1}'.format(self.output_folder, LOG_FILE), 'w', buffering=1)
        self.err_file = open('{0}/{1}'.format(self.output_folder, ERR_FILE), 'w', buffering=1)
        self.stdout = sys.stdout  # copy just in case
        self.stderr = sys.stderr  # copy just in case
        sys.stdout = self.log_file
        sys.stderr = self.log_file

        self.name = kwargs.get('name', 'noname')
        self.game = kwargs.get('game', GAME)
        self.description = description

        print('Initialising the experiment {0} in folder {1} on game {2}.\nDescription: {3}'.format(self.name,
                                                                                                    self.output_folder,
                                                                                                    self.game,
                                                                                                    self.description))

        file_train = "{0}/{1}-{2}.tfrecords".format(DATA_FOLDER, GAME, 'train')
        file_valid = "{0}/{1}-{2}.tfrecords".format(DATA_FOLDER, GAME, 'valid')

        self.epochs = epochs
        self.batch_size = kwargs.get('batch_size', BATCH_SIZE)
        self.train_gen = ImageGenerator(file_train, batch_size=self.batch_size,
                                        im_shape=network_params.INPUT_IMAGE_SHAPE,
                                        buffer_size=200*self.batch_size)
        self.valid_gen = ImageGenerator(file_valid, batch_size=self.batch_size,
                                        im_shape=network_params.INPUT_IMAGE_SHAPE,
                                        buffer_size=5*self.batch_size)

        print('Generators started.')

        self.network = architecture.MultiNetwork(models_folder=self.models_folder)
        print('Networks built.')

        self.losses = {'ae_train': [],
                       'ae_valid': [],
                       # 'ae_disc_train': [],
                       # 'ae_disc_valid': [],
                       # 'ae_gan_train': [],
                       # 'ae_gan_valid': [],
                       }

    def train_ae(self, epochs=5, model_checkpoint=False):
        print('Training autoencoder for {0} epochs.'.format(epochs))
        history = self.network.autoencoder_gen.fit_generator(self.train_gen.generate_ae(),
                                                             samples_per_epoch=BATCHES_PER_EPOCH * BATCH_SIZE,
                                                             nb_epoch=epochs,
                                                             max_q_size=5,
                                                             validation_data=self.valid_gen.generate_ae(),
                                                             nb_val_samples=4 * BATCH_SIZE)

        self.losses['ae_train'] += history.history['loss']
        self.losses['ae_valid'] += history.history['val_loss']

        self.save_losses()
        self.save_ae_recons()

        if model_checkpoint:
            epochs_so_far = len(self.losses['ae_train'])
            print('Model checkpoint reached. Saving the model after {0} epochs.'.format(epochs_so_far))
            fpath = '{0}/ae_gen_{1}.hdf5'.format(self.models_folder, epochs_so_far)
            self.network.autoencoder_gen.save_weights(fpath)

    def save_ae_recons(self):
        N_SAMPLES = 5
        im_med = self.train_gen.im_med
        im_valid = self.valid_gen.get_shuffled_batch(subtract_median=True)
        im_recon = self.network.autoencoder_gen.predict(im_valid)

        pairs = []
        for i in range(N_SAMPLES):
            pairs.append(np.concatenate([im_valid[i, ...] + im_med, im_recon[i, ...] + im_med], axis=1))

        tiled = np.concatenate(pairs, axis=0)

        # return to viewable representation
        tiled *= 255
        tiled = tiled.astype('uint8')

        epochs_so_far = len(self.losses['ae_train'])
        print('Saving new reconstructions after {0} epochs.'.format(epochs_so_far))
        # imsave('{0}/{1}.png'.format(self.reconstructions_folder, epochs_so_far), tiled)
        tiled = Image.fromarray(tiled)
        tiled.save('{0}/{1}.png'.format(self.reconstructions_folder, epochs_so_far))

    def save_losses(self):
        # TODO arrays must be the same lengths to use this constructor
        losses = pd.DataFrame.from_dict(self.losses)
        losses.to_csv('{0}/losses.csv'.format(self.output_folder))

    def save_plots(self):
        print('Generating plots.')

    def save_models(self):
        print('Saving final models')
        fpath = '{0}/encoder_final.hdf5'.format(self.models_folder)
        self.network.encoder.save_weights(fpath)
        fpath = '{0}/decoder_final.hdf5'.format(self.models_folder)
        self.network.decoder.save_weights(fpath)
        fpath = '{0}/screen_discriminator_final.hdf5'.format(self.models_folder)
        self.network.screen_discriminator.save_weights(fpath)

    def finish(self):
        print('Finishing.')
        self.save_plots()
        self.save_models()

    def make_directories(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.reconstructions_folder):
            os.makedirs(self.reconstructions_folder)
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

    def run_experiment(self):
        self.train_ae(epochs=5)
        self.finish()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]
        exp = Experiment(output_folder=output_folder)
    else:
        exp = Experiment()

    exp.run_experiment()
    exp.finish()
