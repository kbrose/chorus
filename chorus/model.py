from typing import Literal
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchaudio.transforms import Spectrogram

TARGET_MAX_FREQ = 15_000  # Should be half the minimum expected sample rate
NUM_FREQS = 257
TARGET_STEP_IN_SECS = 0.003


class ResLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, 1, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(c_out)
        if c_in != c_out or stride > 1:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm1d(c_out)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, targets):
        super().__init__()

        self.identity = torch.nn.Identity()  # used for better summary

        self.init_conv = nn.Conv1d(1, 8, 7, 5)
        self.init_bn = nn.BatchNorm1d(8)
        self.init_maxpool = nn.MaxPool1d(2, 1)
        self.relu = nn.ReLU(inplace=True)

        channels = [8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        strides = [3, 3, 1, 3, 3, 1, 3, 3, 4, 4, 4]
        assert len(strides) == len(channels) - 1
        convs = []
        for kernel in [5, 7, 9]:
            convs.append(nn.Sequential(
                *[ResLayer(channels[i], channels[i + 1], kernel, strides[i])
                  for i in range(len(strides))]
            ))
        self.convs = nn.ModuleList(convs)

        self.fc = nn.Linear(channels[-1] * len(convs), len(targets))

        self.targets = targets

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.identity(x)

        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.init_maxpool(x)

        ys = [torch.mean(conv(x), dim=2) for conv in self.convs]

        y = torch.cat(ys, dim=1)

        return self.fc(y)


def load_model(filepath: Path, device: Literal['cpu', 'cuda'], inference=True):
    model = Model()
    model.load_state_dict(torch.load(str(filepath))['model'])
    if inference:
        model.eval()
    model.to(device)
    return model
