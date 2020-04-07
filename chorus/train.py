from typing import NamedTuple, Optional
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
import sklearn.preprocessing

from chorus.data import (
    DATA_FOLDER, load_saved_xeno_canto_meta, scientific_to_en
)
from chorus.model import TARGETS, make_model

_XC_DATA_FOLDER = DATA_FOLDER / 'xeno-canto'
LOGS_FOLDER = Path(__file__).parents[1] / 'logs'
SAVED_MODELS = Path(__file__).parents[1] / 'models'

BATCH = 32
SAMPLE_LEN_SECONDS = 20
SAMPLE_RATE = 30_000
TRAIN_SAMPLES = SAMPLE_RATE * SAMPLE_LEN_SECONDS

# sklearn.preprocessing.MultiLabelBinarizer is very loud.
warnings.filterwarnings('ignore', 'unknown class')


class Data(NamedTuple):
    train: tf.data.Dataset
    train_len: int
    test: tf.data.Dataset
    test_len: int


class ReduceLRBacktrack(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, model_filepath, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.model_filepath = model_filepath
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            tf.get_logger().warning(
                f'ReduceLROnPlateau conditioned on metric `{self.monitor}`'
                ' which is not available. Available metrics are:'
                f' {",".join(list(logs.keys()))}'
            )
        if self.monitor_op(current, self.best):
            self.best_epoch = epoch + 1
        elif not self.in_cooldown() and self.wait + 1 >= self.patience:
            # load best model so far
            print(f"\nBacktracking to epoch {self.best_epoch}")
            self.model.load_weights(
                self.model_filepath.format(epoch=self.best_epoch)
            )

        super().on_epoch_end(epoch, logs)  # actually reduce LR


def _make_dataset(
    df: pd.DataFrame, aug_df: Optional[pd.DataFrame]
) -> tf.data.Dataset:
    rng = np.random.RandomState(seed=20200313)
    sci2en = scientific_to_en(df)

    mlb = sklearn.preprocessing.MultiLabelBinarizer(TARGETS)
    mlb.fit([])  # not needed, we passed in TARGETS as the classes

    def row_to_labels(row):
        return [row['en']] + list(map(sci2en.get, filter(len, row['also'])))

    labels = [row_to_labels(row) for _, row in df.iterrows()]
    y = mlb.transform(labels).astype(int)
    print(f'mean value of labels: {y.mean()}')

    def load(xc_id: int) -> np.ndarray:
        x = np.load(_XC_DATA_FOLDER / 'numpy' / f'{xc_id}.npy')
        if aug_df is not None:
            if x.size < TRAIN_SAMPLES:
                # repeat signal to have length >= TRAIN_SAMPLES
                x = np.tile(x, math.ceil(TRAIN_SAMPLES / x.size))
            start = rng.randint(0, max(x.size - TRAIN_SAMPLES, 1))
            x = x[start:start + TRAIN_SAMPLES]
        else:
            x = x[:SAMPLE_RATE * 360]  # limit to 3 minutes
        return x

    def data_generator():
        while True:
            if aug_df is None:
                yield from zip(map(load, df['id']), y)
            else:
                idx = rng.permutation(y.shape[0])
                for audio, label in zip(map(load, df['id'].iloc[idx]), y[idx]):
                    if rng.random() < 1 / 8:
                        aug_row = aug_df.sample(n=1).iloc[0]
                        label = np.logical_or(
                            label,
                            mlb.transform([row_to_labels(aug_row)]).flatten()
                        ).astype(int)
                        # We'll add in the audio, making sure the original
                        # audio is not less than 1/3rd the strength of the new
                        smoothing = rng.random() + 0.5
                        audio *= smoothing
                        audio += load(aug_row['id']) * (1.5 - smoothing)
                    if rng.random() < 1 / 16:
                        audio += rng.normal(
                            scale=audio.std() * rng.random() * 0.75,
                            size=audio.size,
                        )
                    yield audio, label

    return tf.data.Dataset.from_generator(
        data_generator, (tf.float32, tf.int16), ((None,), (len(TARGETS),))
    )


def get_model_data() -> Data:
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'numpy').glob('*')]
    aug_df = df.loc[df['id'].isin(observed_ids)].copy()
    # Filter out poor quality recordings, and recordings with multiple species
    df = df.loc[
        df['q'].isin(['A', 'B', 'C'])
        & (df['en'].isin(TARGETS))
        & (df['id'].isin(observed_ids))
        & (df['length-seconds'] > 5)
    ]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, stratify=df['en'], random_state=20200310
    )

    return Data(
        train=_make_dataset(train_df, aug_df),
        train_len=train_df.shape[0],
        test=_make_dataset(test_df, None),
        test_len=test_df.shape[0],
    )


def train(name: str):
    model = make_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    train, train_len, test, test_len = get_model_data()
    train = train.repeat().batch(BATCH).prefetch(50)
    test = test.repeat().batch(1).prefetch(50)

    tb = tf.keras.callbacks.TensorBoard(
        str(LOGS_FOLDER / name), histogram_freq=5
    )
    checkpoint_name = str(SAVED_MODELS / name / 'weights-{epoch:04d}.hdf5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_name,
        save_best_only=True,
    )
    lr_reducer = ReduceLRBacktrack(checkpoint_name, factor=0.5, patience=5)

    # Create model/logs folders
    (SAVED_MODELS / name).mkdir(parents=True, exist_ok=True)
    (LOGS_FOLDER / name).mkdir(parents=True, exist_ok=True)

    # Write out the targets names
    with open(LOGS_FOLDER / name / 'target-names.txt', 'w') as f:
        if any('\n' in target for target in TARGETS):
            raise ValueError('newline not allowed in target names')
        f.write('\n'.join(TARGETS))

    model.fit(
        train,
        callbacks=[tb, checkpoint, lr_reducer],
        validation_data=test,
        steps_per_epoch=math.ceil(train_len // BATCH),
        validation_steps=test_len,
        epochs=500
    )
