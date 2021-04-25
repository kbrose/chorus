from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from prefetch_generator import BackgroundGenerator as BgGenerator
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from chorus.config import DEVICE, SAMPLE_RATE
from chorus.evaluate import evaluate
from chorus.geo import Presence
from chorus.models import Classifier, Isolator, load_classifier
from chorus.traindata import model_data

BATCH_CLASSIFIER = 128
SAMPLE_LEN_SECONDS = 15
TRAIN_SAMPLES = SAMPLE_RATE * SAMPLE_LEN_SECONDS

LOGS = Path(__file__).parents[1] / "logs"
MODELS = Path(__file__).parents[1] / "models"


def train_classifier(name: str):
    """
    Train a chorus model with the given name.
    """
    # Set up data
    targets, (train, test) = model_data(TRAIN_SAMPLES)
    print(
        f"Training on {len(train)} samples, testing on {len(test)} samples"
        f" from {len(targets)} distinct species."
    )
    train_dl = torch.utils.data.DataLoader(
        train, BATCH_CLASSIFIER, shuffle=True, pin_memory=True
    )
    test_dl = torch.utils.data.DataLoader(test, 1, pin_memory=True)

    # Set up model and optimizations
    model = Classifier(targets).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    summary(model, input_size=(TRAIN_SAMPLES,))

    # Set up logging / saving
    (MODELS / "classifier" / name).mkdir(parents=True, exist_ok=True)
    (LOGS / "classifier" / name).mkdir(parents=True, exist_ok=True)
    with open(MODELS / "classifier" / name / "targets.json", "w") as f:
        json.dump(model.targets, f)
    tb_writer = SummaryWriter(LOGS / "classifier" / name)
    tb_writer.add_graph(model, torch.rand((1, TRAIN_SAMPLES)).to("cuda"))
    postfix_str = "{train_loss: <6.4f} {valid_loss: <6.4f}{star}"

    # Initializing geo presence
    presence = Presence()
    lat_lngs = test.df[["lat", "lng"]].values
    weeks = test.df["week"].values
    all_names = presence.scientific_names
    all_probs = [
        presence(lat=lat_lng[0], lng=lat_lng[1], week=week)
        for lat_lng, week in zip(lat_lngs, weeks)
    ]
    del lat_lngs, weeks, all_names

    # Training loop
    best_ep = 0
    best_valid_metric = float("inf")
    for ep in range(150):
        with tqdm(desc=f"{ep: >3}", total=len(train_dl), ncols=80) as pbar:
            model.train()
            losses = []
            for xb, yb, w in BgGenerator(train_dl, 5):
                xb, yb, w = xb.to(DEVICE), yb.to(DEVICE), w.to(DEVICE)

                opt.zero_grad()
                y_hat, _ = model(xb)
                loss = torch.mean(loss_fn(y_hat, yb) * w)
                loss.backward()
                opt.step()

                losses.append(float(loss.detach().cpu().numpy()))
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=np.mean(losses),
                        valid_loss=float("nan"),
                        star=" ",
                    ),
                    refresh=False,
                )
                pbar.update()
            tb_writer.add_scalar("loss/train", np.mean(losses), ep)
            model.eval()
            with torch.no_grad():
                validation_metric = evaluate(
                    ep,
                    model,
                    loss_fn,
                    BgGenerator(test_dl, 24),
                    all_probs,
                    tb_writer,
                    targets,
                )
                star = " "
                if validation_metric < best_valid_metric:
                    star = "*"
                    best_valid_metric = validation_metric
                    best_ep = ep
                    torch.save(
                        {"model": model.state_dict(), "opt": opt.state_dict()},
                        str(MODELS / name / f"{ep:0>4}.pth"),
                    )
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=np.mean(losses),
                        valid_loss=validation_metric,
                        star=star,
                    ),
                    refresh=True,
                )
        if ((ep + 1 - best_ep) % 5) == 0:
            lr = opt.param_groups[0]["lr"]
            checkpoint = torch.load(str(MODELS / name / f"{best_ep:0>4}.pth"))
            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])
            opt.param_groups[0]["lr"] = lr / 2
            print(
                f'lowering learning rate to {opt.param_groups[0]["lr"]}'
                f" and resetting weights to epoch {best_ep}"
            )


def train_isolator(name: str, classifier_filepath: str):
    # Set up data
    targets, (train, test) = model_data(TRAIN_SAMPLES)
    print(
        f"Training on {len(train)} samples, testing on {len(test)} samples"
        f" from {len(targets)} distinct species."
    )
    train_dl = torch.utils.data.DataLoader(
        train, 1, shuffle=True, pin_memory=True
    )
    test_dl = torch.utils.data.DataLoader(test, 1, pin_memory=True)

    # Set up model and optimizations
    classifier = load_classifier(
        Path(classifier_filepath).parent, Path(classifier_filepath).name
    ).to(DEVICE)
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    isolator = Isolator(targets).to(DEVICE)
    opt = torch.optim.Adam(isolator.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    # summary(isolator, input_size=(TRAIN_SAMPLES,))

    # Set up logging / saving
    (MODELS / "classifier" / name).mkdir(parents=True, exist_ok=True)
    (LOGS / "classifier" / name).mkdir(parents=True, exist_ok=True)
    with open(MODELS / "classifier" / name / "targets.json", "w") as f:
        json.dump(isolator.targets, f)
    tb_writer = SummaryWriter(LOGS / "classifier" / name)
    # tb_writer.add_graph(isolator, torch.rand((1, TRAIN_SAMPLES)).to("cuda"))
    postfix_str = "{train_loss: <6.4f} {valid_loss: <6.4f}{star}"

    torch.autograd.set_detect_anomaly(True)
    best_ep = 0
    best_valid_metric = float("inf")
    for ep in range(150):
        with tqdm(desc=f"{ep: >3}", total=len(train_dl), ncols=80) as pbar:
            isolator.train()
            losses = []
            for xb, yb, _ in BgGenerator(train_dl, 5):
                xb = xb.to(DEVICE)

                opt.zero_grad()
                target_inds = torch.where(yb[0])[0]
                xb_isolated = isolator(xb, target_inds=target_inds)
                y_hat = classifier(xb_isolated[0])[0]
                y = torch.zeros((len(target_inds), yb.shape[1])).to(DEVICE)
                for i, ind in enumerate(target_inds):
                    y[i, ind] = 1.0
                loss = loss_fn(y_hat, y)
                # breakpoint()
                loss.backward()
                opt.step()

                losses.append(float(loss.detach().cpu().numpy()))
                pbar.set_postfix_str(
                    postfix_str.format(
                        train_loss=np.mean(losses),
                        valid_loss=float("nan"),
                        star=" ",
                    ),
                    refresh=False,
                )
                pbar.update()
            tb_writer.add_scalar("loss/train", np.mean(losses), ep)
