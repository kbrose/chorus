from typing import NamedTuple
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
import sklearn.preprocessing

from chorus.data import DATA_FOLDER, load_saved_xeno_canto_meta
from chorus.model import TARGETS, make_model

_XC_DATA_FOLDER = DATA_FOLDER / 'xeno-canto'
LOGS_FOLDER = Path(__file__).parents[1] / 'logs'
SAVED_MODELS = Path(__file__).parents[1] / 'models'


class Data(NamedTuple):
    train: tf.data.Dataset
    test: tf.data.Dataset


def _make_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    x = [
        np.load(_XC_DATA_FOLDER / 'numpy' / f'{xc_id}.npy')[:30_000 * 60]
        for xc_id in df['id']
    ]
    ohe = sklearn.preprocessing.OneHotEncoder([TARGETS])
    y = ohe.fit_transform(df['en'].values.reshape(-1, 1)).toarray().astype(int)

    def data_generator():
        while True:
            yield from zip(x, y)

    return tf.data.Dataset.from_generator(
        data_generator, (tf.float32, tf.int16), ((None,), (len(TARGETS),))
    )


def get_model_data() -> Data:
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'audio').glob('*')]
    # Filter out poor quality recordings, and recordings with multiple species
    time = pd.to_timedelta(df['length'].apply(
        lambda x: '0:' + x if x.count(':') == 1 else x  # put in hour if needed
    ))
    df = df.loc[
        df['q'].isin(['A', 'B'])
        & (df['also'].apply(lambda x: x == ['']))
        & (df['en'].isin(TARGETS))
        & (df['id'].isin(observed_ids))
        & (time > timedelta(seconds=2))
    ]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, stratify=df['en'], random_state=20200310
    )

    return Data(train=_make_dataset(train_df), test=_make_dataset(test_df))


def train(name: str):
    model = make_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    train, test = get_model_data()
    train = train.repeat().shuffle(200).batch(1)
    test = test.repeat().batch(1)

    tb = tf.keras.callbacks.TensorBoard(str(LOGS_FOLDER / name))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(SAVED_MODELS / name))

    model.fit(
        train,
        callbacks=[tb, checkpoint],
        validation_data=test,
        steps_per_epoch=2239,
        validation_steps=560,
        epochs=100
    )
