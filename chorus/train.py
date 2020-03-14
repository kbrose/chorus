from typing import NamedTuple
from pathlib import Path

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


class Data(NamedTuple):
    train: tf.data.Dataset
    train_len: int
    test: tf.data.Dataset
    test_len: int


def _make_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    sci2en = scientific_to_en(df)
    mlb = sklearn.preprocessing.MultiLabelBinarizer(TARGETS)
    mlb.fit([])
    labels = [
        [row['en']] + [sci2en[bird] for bird in filter(len, row['also'])]
        for _, row in df.iterrows()
    ]
    y = mlb.transform(labels).astype(int)

    def load(xc_id: int) -> np.ndarray:
        x = np.load(_XC_DATA_FOLDER / 'numpy' / f'{xc_id}.npy')[:30_000 * 60]
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
        train=_make_dataset(train_df),
        train_len=train_df.shape[0],
        test=_make_dataset(test_df),
        test_len=test_df.shape[0],
    )


def train(name: str):
    model = make_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    train, train_len, test, test_len = get_model_data()
    train = train.repeat().shuffle(200).batch(1)
    test = test.repeat().batch(1)

    tb = tf.keras.callbacks.TensorBoard(
        str(LOGS_FOLDER / name), histogram_freq=5, write_images=True,
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
        steps_per_epoch=train_len,
        validation_steps=test_len,
        epochs=100
    )
