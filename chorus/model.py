from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras


TARGETS = [
    "Song Sparrow",
    "Carolina Wren",
    "Northern Cardinal",
    "American Robin",
    "Red Crossbill",
    "Red-winged Blackbird",
    "House Wren",
    "Bewick's Wren",
    "Dark-eyed Junco",
    "Blue Jay",
    "Spotted Towhee",
    "Tufted Titmouse",
    "Great Horned Owl",
    "Northern Saw-whet Owl",
    "Grey Catbird",
    "Northern Mockingbird",
    "Marsh Wren",
    "American Crow",
    "Common Yellowthroat",
    "Northern Raven",
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
    def __init__(self, frame_length: int, frame_step: int):
        """
        Inputs
        ------
        frame_length : int
            Length of each window (in samples).
            Must be a positive power of 2.
        frame_step : int
            Step size between windows. Positive integer.
        """
        super(Spectrogram, self).__init__()

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

    x = Spectrogram(512, 448)(audio)  # 448 = 512 - 512 // 8

    for unit in [8]:
        x = keras.layers.Bidirectional(
            keras.layers.GRU(unit, return_sequences=True, activation='relu')
        )(x)

    # Average the features over the time series.
    x = keras.layers.GlobalAveragePooling1D()(x)

    for unit in [8]:
        x = keras.layers.Dense(unit, activation='relu')(x)

    probs = keras.layers.Dense(len(TARGETS), activation='sigmoid')(x)

    return keras.models.Model(audio, probs)
