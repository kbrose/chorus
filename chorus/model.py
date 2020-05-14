from typing import Literal
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio


TARGETS = [
    "Song Sparrow",
    "Carolina Wren",
    "Northern Cardinal",
    "American Robin",
    "Red Crossbill",
    # "Red-winged Blackbird",
    # "House Wren",
    # "Bewick's Wren",
    # "Dark-eyed Junco",
    # "Blue Jay",
    # "Spotted Towhee",
    # "Tufted Titmouse",
    # "Great Horned Owl",
    # "Northern Saw-whet Owl",
    # "Grey Catbird",
    # "Northern Mockingbird",
    # "Marsh Wren",
    # "American Crow",
    # "Common Yellowthroat",
    # "Northern Raven",
]
TARGET_MAX_FREQ = 15_000  # Should be half the minimum expected sample rate
NUM_FREQS = 257
TARGET_STEP_IN_SECS = 0.003


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(512, 512, 45)
        self.batch_norm1 = nn.BatchNorm1d(NUM_FREQS)

        channels = [NUM_FREQS, 64, 32, 32, 32, 32, 32, 32, 64, 32]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_out, 7, 2)
                for c_in, c_out in zip(channels[:-1], channels[1:])
            ]
        )

        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)

        self.fc1 = nn.Linear(32, 32)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(32, len(TARGETS))
        nn.init.xavier_uniform_(self.fc2.weight)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, include_top=True):
        x = self.spectrogram(x)
        x = self.batch_norm1(x)

        for conv in self.convs:
            x = F.relu(conv(x), inplace=True)
        x = self.global_avg_pool(x)
        x = x.view(x.size()[0], -1)

        if include_top:
            x = F.relu(self.fc1(x), inplace=True)
            x = self.fc2(x)
        return x


def load_model(filepath: Path, device: Literal['cpu', 'cuda'], inference=True):
    model = Model()
    model.load_state_dict(torch.load(str(filepath)))
    if inference:
        model.eval()
    model.to(device)
    return model
