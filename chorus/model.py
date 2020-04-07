from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras


TARGETS = [
    "Song Sparrow",
    "Carolina Wren",
    "Northern Cardinal",
    "American Robin",
    "Red Crossbill",
    # "Red-winged Blackbird",
    # "House Wren",
    # "Bewick's Wren",
    # "Dark-eyed Junco",
    # "Blue Jay",
    # "Spotted Towhee",
    # "Tufted Titmouse",
    # "Great Horned Owl",
    # "Northern Saw-whet Owl",
    # "Grey Catbird",
    # "Northern Mockingbird",
    # "Marsh Wren",
    # "American Crow",
    # "Common Yellowthroat",
    # "Northern Raven",
]


class Spectrogram(keras.layers.Layer):
    """
    Compute the power spectral density spectrogram.
    """
    def __init__(self, frame_length: int, frame_step: int, **kwargs):
        """
        Inputs
        ------
        frame_length : int
            Length of each window (in samples).
            Must be a positive power of 2.
        frame_step : int
            Step size between windows. Positive integer.
        **kwargs
            Passed to super().__init__()
        """
        super(Spectrogram, self).__init__(**kwargs)

        self.frame_length = frame_length
        self.frame_step = frame_step
        self.input_spec = keras.layers.InputSpec(ndim=2)

    def call(self, x):
        y = tf.signal.stft(x, self.frame_length, self.frame_step, pad_end=True)
        y = tf.abs(tf.math.conj(y) * y)
        # Handle resampling implicitly by either
        # 1. zero padding higher frequencies for slower signals, or
        # 2. cutting off higher frequencies for faster signals.
        # TODO
        return y

    def get_config(self) -> Dict[str, Any]:
        return {
            'frame_length': self.frame_length, 'frame_step': self.frame_step
        }


def _resnet_blocks(x, n_feats_1, n_feats_2, num_id):
    x0 = x

    x = keras.layers.Conv1D(n_feats_1, 1, strides=2, padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(n_feats_1, 3, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(n_feats_2, 1, strides=1, padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)

    x0 = keras.layers.Conv1D(n_feats_2, 1, strides=2, padding='valid')(x0)
    x0 = keras.layers.BatchNormalization()(x0)

    x = keras.layers.Add()([x, x0])
    x = keras.layers.ReLU()(x)

    for _ in range(num_id):
        x0 = x

        x = keras.layers.Conv1D(n_feats_1, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv1D(n_feats_1, 3, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv1D(n_feats_2, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Add()([x, x0])
        x = keras.layers.ReLU()(x)

    return x


def make_model() -> keras.models.Model:
    """
    Create the bird song model.
    """
    audio = keras.layers.Input(shape=(None,), name='audio')

    # The shape of x will be (None, 257) with these parameters
    x = Spectrogram(512, 448)(audio)  # 448 = 512 - 512 // 8

    x = keras.layers.Dropout(rate=0.25)(x)

    x = keras.layers.Conv1D(128, 7, strides=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(3, strides=2)(x)

    for n_feats_1, n_feats_2 in [(32, 64), (64, 128), (64, 256), (64, 512)]:
        x = keras.layers.Dropout(rate=0.25)(x)
        x = _resnet_blocks(x, n_feats_1, n_feats_2, 3)

    # Average the features over the time series.
    x = keras.layers.GlobalAveragePooling1D()(x)

    for unit in [256, 128]:
        x = keras.layers.Dense(unit, activation='relu')(x)

    probs = keras.layers.Dense(len(TARGETS), activation='sigmoid')(x)

    return keras.models.Model(audio, probs)
