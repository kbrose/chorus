from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import librosa
import torch

from chorus.models import load_classifier
from chorus.config import SAMPLE_RATE
from chorus.geo import Presence
from chorus.metadata import get_sci2en


warnings.filterwarnings(
    "ignore", "PySoundFile failed. Trying audioread instead."
)


def run_classifier(
    modelpath: Path,
    audiofile: Path,
    latlng: tuple[float, float] | None = None,
    date: datetime.datetime | None = None,
    region_influence: float = 0.9,
    device: str | None = None,
    scientific: bool = False,
):
    """Run classifier located at MODELPATH on AUDIOFILE"""
    if modelpath.is_dir():
        model, classes = load_classifier(modelpath)
    else:
        model, classes = load_classifier(modelpath.parent, modelpath.name)
    model.eval()

    x = torch.tensor(librosa.load(audiofile, sr=SAMPLE_RATE)[0])
    if device:
        model = model.to(device)
        x = x.to(device)
    with torch.no_grad():
        y_hat = torch.sigmoid(model(x.unsqueeze(0))[0])[0].cpu().numpy()
    results = dict(zip(classes, y_hat))

    if latlng is not None:
        if date is None:
            raise ValueError("date must not be None if latlng is not None")
        presence = Presence()
        geoprobs = presence(
            lat=latlng[0],
            lng=latlng[1],
            week=min(date.isocalendar().week, 52),  # type:ignore
        )
        for key in list(results.keys()):
            results[key] = results[key] * (
                region_influence * geoprobs[key] + (1 - region_influence)
            )

    if not scientific:
        sci2en = get_sci2en()
        results = {sci2en[key]: val for key, val in results.items()}

    return results
