from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from chorus.config import DEVICE


def evaluate(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.utils.data.DataLoader,
    all_probs: list[dict[str, float]],
    tb_writer: SummaryWriter,
    targets: list[str],
):
    preds = []
    actuals = []
    losses = []
    weights = []
    for x, y, w in data:
        x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
        y_hat, _ = model(x)
        loss = torch.mean(loss_fn(y_hat, y) * w)

        preds.append(torch.sigmoid(y_hat[0]).cpu().numpy())
        actuals.append(y[0].cpu().numpy())
        losses.append(loss.cpu().numpy())
        weights.append(w[0].cpu().numpy())

    valid_loss = float(np.mean(losses))

    tb_writer.add_scalar("loss/valid", valid_loss, epoch)

    yhats = np.array(preds)
    ys = np.array(actuals)
    yhats_adjusted = []
    full_f, full_axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for full_ax in full_axs:
        full_ax.set_aspect("equal")
        full_ax.set_xlim([0, 1])
        full_ax.set_ylim([0, 1])
        full_ax.plot([0, 1], [0, 1], "k--")
    for yhat, y, label in zip(yhats.T, ys.T, targets):
        probs = np.stack([p[label] for p in all_probs])
        yhat_adjusted = yhat * (probs + 0.1) / 1.1
        yhats_adjusted.append(yhat_adjusted)
        f, ax = plt.subplots(1)

        fpr, tpr, _ = roc_curve(y, yhat)
        auc = roc_auc_score(y, yhat)
        ax.plot(fpr, tpr, label=f"no geo: {auc:.3f}")
        full_axs[0].plot(fpr, tpr, alpha=0.1)

        flt = ~np.isnan(yhat_adjusted)
        fpr, tpr, _ = roc_curve(y[flt], yhat_adjusted[flt])
        auc = roc_auc_score(y[flt], yhat_adjusted[flt])
        ax.plot(fpr, tpr, label=f"w/ geo: {auc:.3f}")
        full_axs[1].plot(fpr, tpr, alpha=0.1)

        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.plot([0, 1], [0, 1], "k--")
        tb_writer.add_figure(f"roc/{label}", f, epoch)

    tb_writer.add_figure("roc/all_species", full_f, epoch)
    pure_ranks = []
    adjusted_ranks = []
    for yhat, yhat_adjusted, w in zip(
        yhats, np.array(yhats_adjusted).T, weights
    ):
        idx = np.where(w == 1)[0][0]
        pure_ranks.append((yhat > yhat[idx]).sum())
        adjusted_ranks.append((yhat_adjusted > yhat_adjusted[idx]).sum())

    f, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs[0].set_title("Unadjusted ranks")
    axs[0].hist(pure_ranks)
    axs[1].set_title("Ranks adjusted by geo prob")
    axs[1].hist(adjusted_ranks)
    tb_writer.add_figure("ranks", f, epoch)

    return valid_loss
