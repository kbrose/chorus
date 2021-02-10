from typing import NamedTuple, Optional, List, Dict, Tuple
from pathlib import Path
import math
import warnings
import json

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import sklearn.model_selection
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from prefetch_generator import BackgroundGenerator as BgGenerator

from chorus.data import DATA_FOLDER, load_saved_xeno_canto_meta
from chorus.model import Model
from chorus.range import Presence

_XC_DATA_FOLDER = DATA_FOLDER / 'xeno-canto'
LOGS = Path(__file__).parents[1] / 'logs'
MODELS = Path(__file__).parents[1] / 'models'

BATCH = 32
SAMPLE_LEN_SECONDS = 30
SAMPLE_RATE = 30_000
TRAIN_SAMPLES = SAMPLE_RATE * SAMPLE_LEN_SECONDS
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# sklearn.preprocessing.MultiLabelBinarizer is very loud.
warnings.filterwarnings('ignore', 'unknown class')


class SongDataset(torch.utils.data.Dataset):
    """
    Create a tf.data.Dataset from the given dataframe and optional
    augmenting dataframe.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        targets: List[str],
        aug_df: Optional[pd.DataFrame]=None,
    ):
        """
        Inputs
        ------
        df : pd.DataFrame
            Dataframe that contains the columns 'id', 'en', and 'also'.
            Likely the result of calling load_saved_xeno_canto_meta()
        aug_df : pd.DataFrame
            Same format as `df`. If provided, we assume we are creating
            a training dataset and will also randomly add noise and shuffling.
            The audio in this dataframe will sometimes be added
            to the original audio.
        """
        super().__init__()
        self.df = df
        self.aug_df = aug_df

        self.np_rng = np.random.RandomState(seed=20200313)

        def row_to_labels(row):
            return [row['scientific-name']] + [x for x in row['also'] if x]

        mlb = sklearn.preprocessing.MultiLabelBinarizer(classes=targets)
        mlb.fit([])  # not needed, we passed in targets as the classes

        labels = [row_to_labels(row) for _, row in df.iterrows()]
        self.y = mlb.transform(labels).astype(int)

        if aug_df is not None:
            aug_labels = [row_to_labels(row) for _, row in aug_df.iterrows()]
            self.aug_y = mlb.transform(aug_labels).astype(int)
        else:
            self.aug_y = None

    def load(self, xc_id: int, target_length: int) -> np.ndarray:
        x = np.load(_XC_DATA_FOLDER / 'numpy' / f'{xc_id}.npy', mmap_mode='r')
        if x.size < target_length * 2:
            # repeat signal to have length >= target_length * 2
            x = np.tile(x, math.ceil(target_length * 2 / x.size))
        start = self.np_rng.randint(0, max(x.size - target_length, 1))
        x = x[start:start + target_length].copy()
        return x

    def __getitem__(self, index):
        """
        Inputs
        ------
        index : int

        Returns
        -------
        (input, expected_output)
        """
        xc_id = self.df.iloc[index]['id']
        x = self.load(xc_id, TRAIN_SAMPLES)
        y = self.y[index]
        if self.aug_df is not None:
            if self.np_rng.random() < 1 / 32:
                aug_index = self.np_rng.randint(self.aug_df.shape[0])
                aug_row = self.aug_df.iloc[aug_index]
                y = np.logical_or(self.aug_y[aug_index], y)
                smoothing = self.np_rng.random() * 0.75
                x += self.load(aug_row['id'], TRAIN_SAMPLES) * smoothing
            if self.np_rng.random() < 1 / 64:
                x += self.np_rng.normal(
                    scale=x.std() * self.np_rng.random() * 0.25,
                    size=x.size,
                )
        return (
            (
                torch.as_tensor(x),
                torch.as_tensor(self.df[['lat', 'lng']].iloc[index].values),
                torch.as_tensor(self.df['week'].iloc[index])
            ),
            torch.as_tensor(y.astype(float))
        )

    def __len__(self):
        return self.df.shape[0]


class Data(NamedTuple):
    train: SongDataset
    test: SongDataset


def get_model_data() -> Tuple[List[str], Data]:
    """
    Get the training and testing data to be used for the model.

    Returns
    -------
    Data
    """
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'numpy').glob('*')]
    aug_df = df.loc[df['id'].isin(observed_ids)].copy()
    aug_df = aug_df.loc[
        (aug_df['length-seconds'] >= 5) & (aug_df['length-seconds'] <= 60)
    ]
    # Filter out poor quality recordings, and recordings with multiple species
    df = df.loc[
        df['q'].isin(['A', 'B', 'C'])
        & (df['id'].isin(observed_ids))
        & (df['length-seconds'] >= 5)
        & (df['length-seconds'] <= 60)
    ]
    targets = sorted(
        df['scientific-name'].value_counts()[lambda x: x >= 50].index.tolist()
    )
    df = df.loc[df['scientific-name'].isin(targets)]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, stratify=df['scientific-name'], random_state=2020310
    )

    aug_df.drop(test_df.index, inplace=True)

    return targets, Data(
        train=SongDataset(train_df, targets, aug_df),
        test=SongDataset(test_df, targets, None),
    )


def evaluate(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.utils.data.DataLoader,
    presence: Presence,
    all_probs: List[Dict[str, float]],
    tb_writer: SummaryWriter,
    targets: List[str],
):
    preds_targets = [
        (model(x.to(DEVICE))[0], y) for (x, _, _), y in data
    ]
    valid_loss = torch.tensor(
        [loss_fn(preds, y.to(DEVICE)) for preds, y in preds_targets]
    ).mean().cpu().numpy()
    valid_loss = float(valid_loss)

    tb_writer.add_scalar('valid_loss', valid_loss, epoch)

    sig_fn = nn.Sigmoid()
    yhats = np.stack([sig_fn(pt[0]).cpu().numpy()[0] for pt in preds_targets])
    ys = np.stack([pt[1].cpu().numpy()[0] for pt in preds_targets])

    full_f, full_axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for full_ax in full_axs:
        full_ax.set_aspect('equal')
        full_ax.set_xlim([0, 1])
        full_ax.set_ylim([0, 1])
        full_ax.plot([0, 1], [0, 1], 'k--')
    for yhat, y, label in zip(yhats.T, ys.T, targets):
        y_adjusted = [0, 1]
        yhat_adjusted = [0, 0]
        if label in presence.scientific_names:
            probs = np.stack([p[label] for p in all_probs])
            flt = ~np.isnan(probs)
            if flt.any():
                y_adjusted = y[flt]
                yhat_adjusted = yhat[flt] * (probs[flt] + 0.1) / 1.1
        f, axs = plt.subplots(1, 2, sharex=True, sharey=True)

        fpr, tpr, _ = roc_curve(y, yhat)
        axs[0].plot(fpr, tpr)
        auc = roc_auc_score(y, yhat)
        axs[0].set_title(f'AUC = {auc:.3f}')
        full_axs[0].plot(fpr, tpr, alpha=0.1)

        fpr, tpr, _ = roc_curve(y_adjusted, yhat_adjusted)
        axs[1].plot(fpr, tpr)
        auc = roc_auc_score(y_adjusted, yhat_adjusted)
        axs[1].set_title(f'AUC = {auc:.3f}')
        full_axs[1].plot(fpr, tpr, alpha=0.1)

        f.suptitle(f'{label}')

        for ax in axs:
            ax.set_aspect('equal')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.plot([0, 1], [0, 1], 'k--')
        tb_writer.add_figure(label, f, epoch)
    tb_writer.add_figure('all_species', full_f, epoch)

    return valid_loss


def train(name: str):
    """
    Train a chorus model with the given name.
    """
    # Set up data
    targets, (train, test) = get_model_data()
    print(
        f'Training on {len(train)} samples, testing on {len(test)} samples'
        f' from {len(targets)} distinct species.'
    )
    # We do a weighted sample of the training data. For a given row,
    # we first take the rarest class that shows up in that row, and
    # set its weight inversely proportional to the rareness of that class.
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=(train.y / train.y.sum(axis=0)[np.newaxis, :]).max(axis=1),
        num_samples=len(train),
        replacement=True
    )
    train_dl = torch.utils.data.DataLoader(
        train, BATCH, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(
        test, 1, num_workers=4, pin_memory=True)

    # Set up model and optimizations
    model = Model(targets)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    summary(model, input_size=(TRAIN_SAMPLES,))

    presence = Presence()

    # Set up logging / saving
    (MODELS / name).mkdir(parents=True, exist_ok=True)
    (LOGS / name).mkdir(parents=True, exist_ok=True)
    with open(MODELS / name / 'targets.json', 'w') as f:
        json.dump(model.targets, f)
    tb_writer = SummaryWriter(LOGS / name)
    tb_writer.add_graph(model, torch.rand((1, TRAIN_SAMPLES)).to('cuda'))
    postfix_str = '{train_loss: <6} {valid_loss: <6}{star}'

    model = torch.jit.script(model)

    # Initializing geo presence
    inputs = [
        (lat_lng, week) for (_, lat_lng, week), _ in test_dl
    ]
    lat_lngs = np.stack([i[0].cpu().numpy()[0] for i in inputs])
    weeks = np.stack([i[1].cpu().numpy()[0] for i in inputs])
    all_names = presence.scientific_names
    all_probs = [
        presence(lat=lat_lng[0], lng=lat_lng[1], week=week)
        if (not np.isnan(lat_lng).all() and not np.isnan(week))
        else dict(zip(all_names, np.nan * np.zeros(len(all_names))))
        for lat_lng, week in zip(lat_lngs, weeks)
    ]
    del inputs, lat_lngs, weeks, all_names

    best_ep = 0
    best_valid_loss = float('inf')
    for ep in range(1_000):
        with tqdm(desc=f'{ep: >3}', total=len(train_dl), ncols=80) as pbar:
            model.train()
            losses = []
            for (xb, _, _), yb in BgGenerator(train_dl, 5):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                opt.zero_grad()
                y_hat, _ = model(xb)
                loss = loss_fn(y_hat, yb)
                loss.backward()
                opt.step()

                pbar.update()
                losses.append(float(loss.detach().cpu().numpy()))
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=round(np.mean(losses), 4),
                        valid_loss='',
                        star=' '
                    ),
                    refresh=False
                )
            tb_writer.add_scalar('train_loss', np.mean(losses), ep)
            model.eval()
            with torch.no_grad():
                valid_loss = evaluate(
                    ep,
                    model,
                    loss_fn,
                    BgGenerator(test_dl, 24),
                    presence,
                    all_probs,
                    tb_writer,
                    targets,
                )
                star = ' '
                if valid_loss < best_valid_loss:
                    star = '*'
                    best_valid_loss = valid_loss
                    best_ep = ep
                    torch.jit.save(
                        model, str(MODELS / name / f'{ep:0>4}.pth')
                    )
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=round(np.mean(losses), 4),
                        valid_loss=round(valid_loss, 4),
                        star=star
                    ),
                    refresh=True
                )
        if ((ep + 1 - best_ep) % 25) == 0:
            lr = opt.param_groups[0]['lr']
            model = torch.jit.load(str(MODELS / name / f'{best_ep:0>4}.pth'))
            opt.param_groups[0]['lr'] = lr / 10
            print(
                f'lowering learning rate to {opt.param_groups[0]["lr"]}'
                f' and resetting weights to epoch {best_ep}'
            )
