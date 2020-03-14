from typing import NamedTuple
from pathlib import Path
import math

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
SAMPLE_LEN_SECONDS = 30
SAMPLE_RATE = 30_000
TRAIN_SAMPLES = SAMPLE_RATE * SAMPLE_LEN_SECONDS


class Data(NamedTuple):
    train: tf.data.Dataset
    train_len: int
    test: tf.data.Dataset
    test_len: int


def _make_dataset(df: pd.DataFrame, is_training: bool) -> tf.data.Dataset:
    rng = np.random.RandomState(seed=20200313)
    sci2en = scientific_to_en(df)
    mlb = sklearn.preprocessing.MultiLabelBinarizer(TARGETS)
    mlb.fit([])
    labels = [
        [row['en']] + list(map(sci2en.get, filter(len, row['also'])))
        for _, row in df.iterrows()
    ]
    y = mlb.transform(labels).astype(int)

    def load(xc_id: int) -> np.ndarray:
        x = np.load(_XC_DATA_FOLDER / 'numpy' / f'{xc_id}.npy')
        if is_training:
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
            yield from zip(map(load, df['id']), y)

    return tf.data.Dataset.from_generator(
        data_generator, (tf.float32, tf.int16), ((None,), (len(TARGETS),))
    )


def get_model_data() -> Data:
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'audio').glob('*')]
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
        train=_make_dataset(train_df, True),
        train_len=train_df.shape[0],
        test=_make_dataset(test_df, False),
        test_len=test_df.shape[0],
    )


def train(name: str):
    model = make_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    train, train_len, test, test_len = get_model_data()
    train = train.repeat().shuffle(200).batch(BATCH)
    test = test.repeat().batch(1)

    tb = tf.keras.callbacks.TensorBoard(
        str(LOGS_FOLDER / name), histogram_freq=5
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(SAVED_MODELS / name / 'weights-{epoch:04d}.hdf5'),
        save_best_only=True,
    )

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
        callbacks=[tb, checkpoint],
        validation_data=test,
        steps_per_epoch=math.ceil(train_len // BATCH),
        validation_steps=test_len,
        epochs=100
    )
