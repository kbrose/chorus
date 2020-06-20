from typing import NamedTuple, Optional, List
from pathlib import Path
import math
import warnings

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
from prefetch_generator import BackgroundGenerator

from chorus.data import (
    DATA_FOLDER, load_saved_xeno_canto_meta, scientific_to_en
)
from chorus.model import TARGETS, Model

_XC_DATA_FOLDER = DATA_FOLDER / 'xeno-canto'
LOGS_FOLDER = Path(__file__).parents[1] / 'logs'
SAVED_MODELS = Path(__file__).parents[1] / 'models'

BATCH = 32
SAMPLE_LEN_SECONDS = 20
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

        sci2en = scientific_to_en(df)

        def row_to_labels(row):
            return [row['en']] + list(map(sci2en.get, filter(len, row['also'])))

        mlb = sklearn.preprocessing.MultiLabelBinarizer(targets)
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
        if x.size < target_length:
            # repeat signal to have length >= target_length
            x = np.tile(x, math.ceil(target_length / x.size))
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
        return torch.as_tensor(x), torch.as_tensor(y.astype(float))

    def __len__(self):
        return self.df.shape[0]


class Data(NamedTuple):
    train: SongDataset
    test: SongDataset


def get_model_data(targets: List[str]) -> Data:
    """
    Get the training and testing data to be used for the model.

    Returns
    -------
    Data
    """
    df = load_saved_xeno_canto_meta()
    observed_ids = [f.stem for f in (_XC_DATA_FOLDER / 'numpy').glob('*')]
    aug_df = df.loc[df['id'].isin(observed_ids)].copy()
    # Filter out poor quality recordings, and recordings with multiple species
    df = df.loc[
        df['q'].isin(['A', 'B', 'C'])
        & (df['en'].isin(targets))
        & (df['id'].isin(observed_ids))
        & (df['length-seconds'] > 5)
    ]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, stratify=df['en'], random_state=20200310
    )

    aug_df.drop(test_df.index, inplace=True)

    return Data(
        train=SongDataset(train_df, targets, aug_df),
        test=SongDataset(test_df, targets, None),
    )


def evaluate(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.utils.data.DataLoader,
    tb_writer: SummaryWriter,
):
    preds_targets = [(model(x.to(DEVICE)), y) for x, y in data]
    valid_loss = torch.tensor(
        [loss_fn(preds, y.to(DEVICE)) for preds, y in preds_targets]
    ).mean().cpu().numpy()
    valid_loss = float(valid_loss)

    tb_writer.add_scalar('valid_loss', valid_loss, epoch)

    yhats = np.stack([pt[0].cpu().numpy()[0] for pt in preds_targets])
    ys = np.stack([pt[1].cpu().numpy()[0] for pt in preds_targets])

    full_f, full_ax = plt.subplots(1)
    full_ax.set_aspect('equal')
    full_ax.set_xlim([0, 1])
    full_ax.set_ylim([0, 1])
    full_ax.plot([0, 1], [0, 1], 'k--')
    for yhat, y, label in zip(yhats.T, ys.T, TARGETS):
        f, ax = plt.subplots(1)
        fpr, tpr, _ = roc_curve(y, yhat)
        ax.plot(fpr, tpr)
        full_ax.plot(fpr, tpr, alpha=0.1)
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot([0, 1], [0, 1], 'k--')
        auc = roc_auc_score(y, yhat)
        ax.set_title(f'{label} - AUC = {auc:.3f}')
        tb_writer.add_figure(label, f, epoch)
    tb_writer.add_figure('all_species', full_f, epoch)

    return valid_loss


def train(name: str, resume: bool=False):
    """
    Train a chorus model with the given name.
    """
    # Set up model and optimizations
    model = Model()
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    summary(model, input_size=(TRAIN_SAMPLES,))

    # Set up data
    train, test = get_model_data(TARGETS)
    print(f'Training on {len(train)} samples, testing on {len(test)} samples')
    train_dl = torch.utils.data.DataLoader(
        train, BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(
        test, 1, num_workers=4, pin_memory=True)

    # Set up logging / saving
    (SAVED_MODELS / name).mkdir(parents=True, exist_ok=True)
    (LOGS_FOLDER / name).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(LOGS_FOLDER / name)
    postfix_str = '{train_loss: <6} {valid_loss: <6}{star}'

    if resume:
        filepath = max((SAVED_MODELS / name).glob('*.pth'))
        state = torch.load(filepath)
        model.load_state_dict(state['model'])
        opt.load_state_dict((state['optim']))
        is_first = False
        best_ep = int(filepath.stem)
        start_ep = best_ep + 1
    else:
        is_first = True
        best_ep = 0
        start_ep = 0
    best_valid_loss = float('inf')
    for ep in range(start_ep, 1_000):
        is_epoch_first = True
        with tqdm(ascii=True, desc=f'{ep: >3}', total=len(train_dl)) as pbar:
            model.train()
            for i, (xb, yb) in enumerate(BackgroundGenerator(train_dl, 10)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

                pbar.update()
                curr_loss = float(loss.detach().cpu().numpy())
                if is_first:
                    tb_writer.add_graph(model, xb)
                    is_first = False
                if is_epoch_first:
                    averaged_train_loss = curr_loss
                    is_epoch_first = False
                else:
                    averaged_train_loss = (
                        averaged_train_loss * 0.95 + curr_loss * 0.05
                    )
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=round(averaged_train_loss, 4),
                        valid_loss='',
                        star=' '
                    ),
                    refresh=False
                )
            tb_writer.add_scalar('train_loss', averaged_train_loss, ep)
            model.eval()
            with torch.no_grad():
                valid_loss = evaluate(
                    ep,
                    model,
                    loss_fn,
                    BackgroundGenerator(test_dl, 10),
                    tb_writer
                )
                star = ' '
                if valid_loss < best_valid_loss:
                    star = '*'
                    best_valid_loss = valid_loss
                    best_ep = ep
                    state = {
                        'model': model.state_dict(),
                        'optim': opt.state_dict(),
                    }
                    torch.save(state,
                               str(SAVED_MODELS / name / f'{ep:0>4}.pth'))
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=round(averaged_train_loss, 4),
                        valid_loss=round(valid_loss, 4),
                        star=star
                    ),
                    refresh=True
                )
        if ((ep + 1 - best_ep) % 25) == 0:
            state = torch.load(str(SAVED_MODELS / name / f'{best_ep:0>4}.pth'))
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['optim'])
            opt.param_groups[0]['lr'] /= 10
            print(
                f'lowering learning rate to {opt.param_groups[0]["lr"]}'
                f' and resetting weights to epoch {best_ep}'
            )
