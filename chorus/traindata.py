from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from chorus import metadata
from chorus.config import DATA_FOLDER
from chorus.geo import Presence

AUDIO_FOLDER = DATA_FOLDER / "xeno-canto" / "numpy"


class SongDataset(torch.utils.data.Dataset):
    """
    Create a tf.data.Dataset from the given dataframe and optional
    augmenting dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        aug_df: pd.DataFrame | None,
        targets: list[str],
        train_samples: int,
    ):
        """
        Inputs
        ------
        df : pd.DataFrame
            Dataframe that contains the columns 'id', 'en', and 'also'.
            Likely the result of calling xeno_canto_meta()
        aug_df : pd.DataFrame
            Same format as `df`. If provided, we assume we are creating
            a training dataset and will also randomly add noise and shuffling.
            The audio in this dataframe will sometimes be added
            to the original audio.
        targets : list[str]
            names of each target in output
        train_samples : int
            num samples per example
        """
        super().__init__()
        self.df = df
        self.aug_df = aug_df
        self.is_train = aug_df is not None
        self.targets = np.array(targets)
        self.train_samples = train_samples
        self.background_files = list((DATA_FOLDER / "background").glob("*"))

        self.np_rng = np.random.RandomState(seed=20200313)

    @staticmethod
    def _row_to_labels(row):
        return [row["scientific-name"]] + [x for x in row["also"] if x]

    def load(self, xc_id: int, target_length: int) -> np.ndarray:
        x = np.load(AUDIO_FOLDER / f"{xc_id}.npy", mmap_mode="r")
        if x.size < target_length * 2:
            # repeat signal to have length >= target_length * 2
            x = np.tile(x, math.ceil(target_length * 2 / x.size))
        start = self.np_rng.randint(0, max(x.size - target_length, 1))
        x = x[start : start + target_length].copy()
        return x

    def augment(self, x, y_names):
        if self.np_rng.random() < 1 / 32:
            aug_index = self.np_rng.randint(self.aug_df.shape[0])
            aug_row = self.aug_df.iloc[aug_index]
            y_names = y_names + self._row_to_labels(aug_row)
            smoothing = self.np_rng.random() * 0.75
            x += self.load(aug_row["id"], self.train_samples) * smoothing
        if self.np_rng.random() < 1 / 32:
            x += self.np_rng.normal(
                scale=x.std() * self.np_rng.random() * 0.25,
                size=x.size,
            )
        if self.np_rng.random() < 1 / 32:
            background = np.load(
                self.np_rng.choice(self.background_files), mmap_mode="r"
            ) * (self.np_rng.random() / 2 + 0.05)
            if background.size < x.size:
                # Add fade in / fade out
                background[: background.size // 10] *= np.linspace(
                    0, 1, background.size // 10, endpoint=False
                )
                background[-background.size // 10 :] *= np.linspace(
                    1, 0, background.size // 10, endpoint=False
                )
                start = self.np_rng.randint(x.size - background.size)
                x[start : start + background.size] += background
            else:
                start = self.np_rng.randint(background.size - x.size)
                x += background[start : start + x.size]
        return x, y_names

    def __getitem__(self, index):
        """
        Inputs
        ------
        index : int

        Returns
        -------
        (input, expected_output)
        """
        row = self.df.iloc[index]
        xc_id = row["id"]
        x = self.load(xc_id, self.train_samples)
        y_names = self._row_to_labels(row)
        if self.is_train:
            x, y_names = self.augment(x, y_names)
        y = np.isin(self.targets, y_names).astype(np.float32)
        weights = np.ones_like(self.targets, dtype=np.float32)
        # Set weights of everything to 1.0, except the "also" birds which
        # are set to a weight of 0.1
        weights -= np.isin(self.targets, row["also"]).astype(np.float32) * 0.9

        return torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(weights)

    def __len__(self):
        return self.df.shape[0]


class Data(NamedTuple):
    train: SongDataset
    test: SongDataset


def model_data(
    train_samples: int, targets: Optional[list[str]] = None
) -> tuple[list[str], Data]:
    """
    Get the training and testing data to be used for the model.

    Inputs
    ------
    train_sample : int
        How many samples per example
    targets : Optional[list[str]]
        If specified, this is the list of target species.
        If not specified, this is inferred from the data itself.

    Returns
    -------
    targets, Data
    """
    df = metadata.xeno_canto()
    observed_ids = [f.stem for f in AUDIO_FOLDER.glob("*")]
    aug_df = df.loc[df["id"].isin(observed_ids)].copy()
    aug_df = aug_df.loc[
        (aug_df["length-seconds"] >= 5) & (aug_df["length-seconds"] <= 60)
    ]
    names_in_geo = Presence().scientific_names
    # Filter the data
    df = df.loc[
        df["q"].isin(["A", "B", "C"])  # "high quality" only
        & (df["id"].isin(observed_ids))  # ignore data that failed to download
        & (df["length-seconds"] >= 5)  # not too short...
        & (df["length-seconds"] <= 60)  # or too long
        & (df["scientific-name"].isin(names_in_geo))  # align with geo presence
        & (df[["lat", "lng", "week"]].notnull().all(axis=1))  # geo query-able
    ]
    if targets is None:
        targets = sorted(
            df["scientific-name"]
            .value_counts()[lambda x: x >= 50]
            .index.tolist()
        )
    df = df.loc[df["scientific-name"].isin(targets)]
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["scientific-name"], random_state=2020310
    )

    aug_df.drop(test_df.index, inplace=True)

    return targets, Data(
        train=SongDataset(train_df, aug_df, targets, train_samples),
        test=SongDataset(test_df, None, targets, train_samples),
    )
