from typing import Dict, Any
from pathlib import Path

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
TARGET_MAX_FREQ = 15_000  # Should be half the minimum expected sample rate
NUM_FREQS = 257
TARGET_STEP_IN_SECS = 0.003


class Spectrogram(keras.layers.Layer):
    """
    Compute the power spectral density spectrogram.
    """
    def __init__(
        self,
        target_max_freq: float,
        num_freqs: int,
        target_step_in_secs: float,
        **kwargs
    ):
        """
        Inputs
        ------
        target_max_freq : float
            Approximate maximum frequency used in the spectrogram.
        num_freqs : int
            Number of frequencies to return. The frequencies will be approx.
                [n * target_max_freq / (target_num_freqs - 1)
                 for n in range(target_num_freqs)]
        target_step_in_secs : float
            The step size of the spectrogram frames in milliseconds.
        **kwargs
            Passed to super().__init__()
        """
        super(Spectrogram, self).__init__(**kwargs)

        self.target_max_freq = target_max_freq
        self.num_freqs = num_freqs
        self.target_step_in_secs = target_step_in_secs
        self.input_spec = [
            keras.layers.InputSpec(ndim=2), keras.layers.InputSpec(ndim=2)
        ]

    def call(self, x_fs):
        """
        Compute the spectrogram of signal x with sampling rate fs.

        Inputs
        ------
        x_fs : Tuple[tensor, tensor]
            The first element is a 1D signal `x`. The second element is a
            1D tensor `fs` whose length is the sampling rate of `x`.
            This is a bonafied HACK, but it's the only way I could get
            around https://github.com/tensorflow/tensorflow/issues/38296
            It works because with the data I have, sampling rates will
            all be integers anyway...
        """
        x, fs = x_fs
        fs = tf.shape(fs)[1]
        frame_step = tf.cast(
            self.target_step_in_secs * tf.cast(fs, tf.float32) + 1.0,
            tf.int32
        )
        nfft = tf.cast(
            fs * (self.num_freqs - 1) / self.target_max_freq,
            tf.int32
        )

        y = tf.signal.stft(x, nfft, frame_step, nfft, pad_end=True)
        y = tf.sqrt(tf.abs(y))[:, :, :self.num_freqs]
        # TODO: zero pad higher freqs if signal is too slow

        # Due to https://github.com/tensorflow/tensorflow/issues/38296,
        # we have to "reshape" y without actually changing anything
        # so that downstream code knows its shape.
        temporal_length = tf.cast(
            tf.math.ceil(tf.shape(x)[1] / frame_step), tf.int32
        )
        return tf.reshape(y, (-1, temporal_length, self.num_freqs))

    def compute_output_shape(self, input_shapes):
        fs = input_shapes[1][1]
        frame_step = int(fs * self.target_step_in_secs + 1)
        temporal_length = (input_shapes[0][1] + frame_step - 1) // frame_step
        return (input_shapes[0][0], temporal_length, self.num_freqs)

    def get_config(self) -> Dict[str, Any]:
        return {
            'target_max_freq': self.target_max_freq,
            'num_freqs': self.num_freqs,
            'target_step_in_secs': self.target_step_in_secs,
        }


def _resnet_blocks(x, n_feats_1, n_feats_2, num_id):
    x0 = x

    x = keras.layers.Conv1D(n_feats_1, 1, strides=5, padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(n_feats_1, 9, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(n_feats_2, 1, strides=1, padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)

    x0 = keras.layers.Conv1D(n_feats_2, 9, strides=5, padding='same')(x0)
    x0 = keras.layers.BatchNormalization()(x0)

    x = keras.layers.Add()([x, x0])
    x = keras.layers.ReLU()(x)

    for _ in range(num_id):
        x0 = x

        x = keras.layers.Conv1D(n_feats_1, 1, strides=1, padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv1D(n_feats_1, 9, strides=1, padding='same')(x)
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
    fs = keras.layers.Input(shape=(None,), name='fs')

    x = Spectrogram(TARGET_MAX_FREQ, NUM_FREQS, TARGET_STEP_IN_SECS)(
        [audio, fs]
    )

    keras.models.Model([audio, fs], x).summary()

    x = keras.layers.Dropout(rate=0.25)(x)

    x = keras.layers.Conv1D(64, 7, strides=4)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(3, strides=2)(x)

    for n_feats_1, n_feats_2 in [(32, 16), (16, 16), (16, 32)]:
        x = keras.layers.Dropout(rate=0.25)(x)
        x = _resnet_blocks(x, n_feats_1, n_feats_2, 3)

    # Average the features over the time series.
    x = keras.layers.GlobalAveragePooling1D()(x)

    for unit in [64, 32]:
        x = keras.layers.Dense(unit, activation='relu')(x)

    probs = keras.layers.Dense(len(TARGETS), activation='sigmoid')(x)

    return keras.models.Model((audio, fs), probs)


def load_model(filepath: Path):
    return tf.keras.models.load_model(
        filepath, custom_objects={'Spectrogram': Spectrogram}
    )
