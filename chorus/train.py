from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
import sklearn.preprocessing
import librosa
from tqdm import tqdm

from chorus.data import DATA_FOLDER, load_saved_xeno_canto_meta
from chorus.model import TARGETS, make_model

_XC_DATA_FOLDER = DATA_FOLDER / 'xeno-canto'
LOGS_FOLDER = Path(__file__).parents[1] / 'logs'
SAVED_MODELS = Path(__file__).parents[1] / 'models'



class Data(NamedTuple):
    train: tf.data.Dataset
    test: tf.data.Dataset


def _load_audio_file(xc_id: int) -> np.ndarray:
    xc_file = next((_XC_DATA_FOLDER / 'audio').glob(f'{xc_id}*'))
    return librosa.load(xc_file, sr=30000, duration=10.0)[0]


def _make_dataset(df: pd.DataFrame, kind: str) -> tf.data.Dataset:
    x = tf.data.Dataset.from_tensor_slices([
        _load_audio_file(xc_id) for xc_id in tqdm(df['id'])
    ])

    ohe = sklearn.preprocessing.OneHotEncoder(TARGETS)
    y = tf.data.Dataset.from_tensor_slices(ohe.transform(df['en']))

    return tf.data.Dataset.zip((x, y))


def get_model_data() -> Data:
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'audio').glob('*')]
    # Filter out poor quality recordings, and recordings with multiple species
    df = df.loc[
        df['q'].isin(['A', 'B'])
        & (df['also'].apply(lambda x: x == ['']))
        & (df['en'].isin(TARGETS))
        & (df['id'].isin(observed_ids))
    ]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, stratify=df['en'], random_state=20200310
    )

    return Data(
        train=_make_dataset(train_df, 'train'),
        test=_make_dataset(test_df, 'test')
    )


def train():
    model = make_model()
    model.compile('adam', 'binary_crossentropy')

    train, test = get_model_data()
    train = train.repeat().shuffle(200).batch(1)

    tb = tf.keras.callbacks.Tensorboard(str(LOGS_FOLDER))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(SAVED_MODELS))

    model.fit(
        train,
        callbacks=[tb, checkpoint],
        validation_data=test,
        steps_per_epoch=2281,
    )
