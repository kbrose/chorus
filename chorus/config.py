from pathlib import Path

import torch

DATA_FOLDER = Path(__file__).parents[1] / "data"

SAMPLE_RATE = 30_000

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
