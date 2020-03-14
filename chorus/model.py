from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras


TARGETS = [
    "Song Sparrow",
    "Carolina Wren",
    # "Northern Cardinal",
    # "American Robin",
    # "Red Crossbill",
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


def _is_positive_power_of_2(x: int) -> bool:
    """
    True if x is a positive power of 2, False otherwise.

    Runtime scales logarithmically with x.
    """
    checker = 1
    while checker < x:
        checker *= 2
    if checker == x:
        return True
    return False


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

        if not _is_positive_power_of_2(frame_length):
            # Not strictly necessary, but it's fine in my use case
            # and makes downstream calculations easier/faster.
            raise ValueError(f'{frame_length} must be a postiive power of 2')

        self.frame_length = frame_length
        self.frame_step = frame_step
        self.input_spec = keras.layers.InputSpec(ndim=2)

    def call(self, x):
        y = tf.signal.stft(x, self.frame_length, self.frame_step, pad_end=True)
        return tf.abs(tf.math.conj(y) * y)

    def get_config(self) -> Dict[str, Any]:
        return {
            'frame_length': self.frame_length, 'frame_step': self.frame_step
        }


def make_model() -> keras.models.Model:
    """
    Create the bird song model.
    """
    audio = keras.layers.Input(shape=(None,), name='audio')

    # The shape of x will be (None, 257) with these parameters
    x = Spectrogram(512, 448)(audio)  # 448 = 512 - 512 // 8

    x = keras.layers.Dropout(rate=0.25)(x)

    prev_unit = 257
    for unit in [128, 64, 32, 32, 64]:
        x_original = x

        x = keras.layers.Conv1D(unit, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv1D(unit, 3, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv1D(prev_unit, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Add()([x, x_original])
        x = keras.layers.Activation('relu')(x)
        x_original = x

        x = keras.layers.Conv1D(unit, 1, strides=2, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv1D(unit, 3, strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv1D(unit, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)

        x_original = keras.layers.Conv1D(unit, 1, strides=2, padding='valid')(x_original)
        x_original = keras.layers.BatchNormalization()(x_original)

        x = keras.layers.Add()([x, x_original])
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.25)(x)
        prev_unit = unit

    # Average the features over the time series.
    x = keras.layers.GlobalAveragePooling1D()(x)

    for unit in [64, 32]:
        x = keras.layers.Dense(unit, activation='relu')(x)

    probs = keras.layers.Dense(len(TARGETS), activation='sigmoid')(x)

    return keras.models.Model(audio, probs)
