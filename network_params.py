from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D,\
    UpSampling2D, Merge, LSTM, Flatten, ZeroPadding2D, Reshape, BatchNormalization


IMAGE_SIZE_X = 160
IMAGE_SIZE_Y = 210
IMAGE_CHANNELS = 3

INPUT_IMAGE_SHAPE = (IMAGE_SIZE_Y, IMAGE_SIZE_X, IMAGE_CHANNELS)

V_SIZE = 2048
A_MAP_SIZE = 100
S_SIZE = 200

# DENSE_SIZE = 2048

POSITIONAL_ARGS = 'pos_args'
KEYWORD_ARGS = 'key_args'

ENCODER = {
        'name': 'conv4',
        'input_shape': INPUT_IMAGE_SHAPE,
        'output_shape': (V_SIZE,),
        'layers': [
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (0, 1)
                }

            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [32, 8, 8],
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    # 'border_mode': 'same'
                }
            },

            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    # 'border_mode': 'same'
                }
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                KEYWORD_ARGS: {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    # 'border_mode': 'same'
                }
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 4, 4],
                KEYWORD_ARGS: {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                }
            },
            {
                'type': Flatten,
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [V_SIZE],
                KEYWORD_ARGS: {
                    'activation': 'relu',
                    'init': 'uniform',
                }
            },
        ],
    }

DECODER = {
        'name': 'deconv4',
        'input_shape': (V_SIZE,),
        'output_shape': INPUT_IMAGE_SHAPE,
        'layers': [
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [8*11*64],
                'output_dim': 8*11*64,
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                }
            },            {
                'type': Reshape,
                POSITIONAL_ARGS: [(11, 8, 64)],
                'shape': (11, 8, 64),
            },
            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 4, 4],
                KEYWORD_ARGS : {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                    'border_mode': 'same'

                }
            },
            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                    'border_mode': 'same'
                }
            },
            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                    'border_mode': 'same'
                }
            },

            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (3, 2)
                }

            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [3, 8, 8],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                    'border_mode': 'same'
                }
            },
        ],
    }

DECODER_DECONV = {
        'name': 'deconv4',
        'input_shape': (V_SIZE,),
        'output_shape': INPUT_IMAGE_SHAPE,
        'layers': [
            {
                'type': Dense,
                POSITIONAL_ARGS: [8*11*64],
                'output_dim': 8*11*64,
                KEYWORD_ARGS: {
                    'activation': 'relu',
                }
            },
            {
                'type': Reshape,
                POSITIONAL_ARGS: [(11, 8, 64)],
                'shape': (11, 8, 64),
            },
            {
                'type': Deconvolution2D,
                POSITIONAL_ARGS: [64, 4, 4],
                'nbfilter': 32,
                'nbrow': 8,
                'nbcol': 8,
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'output_shape': (-1, 24, 18, 64),
                    # 'border_mode': 'same'
                }
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': Deconvolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                'nbfilter': 64,
                'nbrow': 6,
                'nbcol': 6,
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'output_shape': (-1, 50, 38, 64),
                    # 'border_mode': 'same'
                }
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (1, 1)
                }

            },
            {
                'type': Deconvolution2D,
                POSITIONAL_ARGS: [64, 6, 6],
                'nbfilter': 64,
                'nbrow': 6,
                'nbcol': 6,
                KEYWORD_ARGS: {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'output_shape': (-1, 102, 78, 64),
                    # 'border_mode': 'same'
                }
            },
            {
                'type': ZeroPadding2D,
                KEYWORD_ARGS: {
                    'padding': (0, 1)
                }

            },
            # {
            #     'type': MaxPooling2D,
            #     POSITIONAL_ARGS: [(2, 2)],
            #     'pool_size': (2, 2),
            #     KEYWORD_ARGS: {
            #         'border_mode': 'same'
            #     }
            # },
            {
                'type': Deconvolution2D,
                POSITIONAL_ARGS: [3, 8, 8],
                'nbfilter': 64,
                'nbrow': 4,
                'nbcol': 4,
                KEYWORD_ARGS: {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'output_shape': (-1, 210, 160, 3),
                    # 'border_mode': 'same'
                }
            },
        ],
    }

SCREEN_DISCRIMINATOR = {
        'name': 'scr_disc',
        'input_shape': (V_SIZE,),
        'output_shape': (1,),
        'layers': [
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [1],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'sigmoid',
                }
            },
        ],
    }

DEFAULT_STRUCTURE = {
    'encoder': ENCODER,
    'decoder': DECODER,
}
