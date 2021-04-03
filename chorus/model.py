import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, kernel: int, stride: int, dilation: int
    ):
        assert kernel % 2, "kernel must be an odd number"
        super().__init__()
        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel, stride, kernel // 2 * dilation, dilation
        )
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel, 1, kernel // 2 * dilation, dilation
        )
        self.bn2 = nn.BatchNorm1d(c_out)
        if c_in != c_out or stride > 1:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm1d(c_out),
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


class Classifier(nn.Module):
    def __init__(self, targets):
        super().__init__()

        self.identity = torch.nn.Identity()  # used for better summary

        self.init_conv = nn.Conv1d(1, 8, 7, 5)
        self.init_bn = nn.BatchNorm1d(8)
        self.init_maxpool = nn.MaxPool1d(2, 1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

        channels = [8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64]
        strides = [3, 3, 1, 3, 3, 1, 3, 3, 2, 2]
        dilations = [1, 2, 3, 1, 2, 3, 1, 2, 3, 3]
        assert len(strides) == len(channels) - 1 == len(dilations)
        resnets = []
        for kernel in [5, 7, 9]:
            resnets.append(
                nn.Sequential(
                    *[
                        ResLayer(
                            channels[i],
                            channels[i + 1],
                            kernel,
                            strides[i],
                            dilations[i],
                        )
                        for i in range(len(strides))
                    ]
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.classifier = nn.Conv1d(
            channels[-1] * len(resnets), len(targets), 1, 1
        )

        self.targets = targets

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.identity(x)

        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.init_maxpool(x)

        ys = []
        for resnet in self.resnets:
            ys.append(self.dropout(resnet(x)))

        y = torch.cat(ys, dim=1)
        logits = self.classifier(y)

        return torch.mean(logits, dim=2), logits


def load_classifier(folder: Path, filename: str = None) -> nn.Module:
    with open(folder / "targets.json") as f:
        targets = json.load(f)
    classifier = Classifier(targets)
    if filename is None:
        filename = max([f.name for f in folder.glob("*.pth")])
    state_dict = torch.load(folder / filename)
    classifier.load_state_dict(state_dict["model"])
    return classifier


def firwin(n, pass_lo, pass_hi, fs):
    """
    Returns a bandpass filter letting through the specified frequencies.

    >>> import torch
    >>> import numpy as np
    >>> import scipy.signal
    >>> fs = 22500
    >>> x = np.random.default_rng(1234).random(300_000)
    >>> x_torch = torch.from_numpy(x).float()[None, None, :]
    >>> filt_torch = firwin(255, 500, 8000, fs)[None, None, :]
    >>> filtered_torch = torch.nn.functional.conv1d(x_torch, filt_torch)[0, 0]
    >>> filt = scipy.signal.firwin(255, [500, 8000], fs=fs, pass_zero=False)
    >>> filtered = scipy.signal.convolve(x, filt, 'valid', 'direct')
    >>> diff = torch.from_numpy(filtered).float() - filtered_torch
    >>> assert diff.abs().max() < 1e-3
    """
    # Adapated from scipy, this is a simplified version of their firwin().
    if not n % 2 or n <= 10:
        raise ValueError("n must be odd and greater than 10")

    # Build hamming window
    fac = torch.linspace(-math.pi, math.pi, n)
    hamming_alpha = 0.54  # no idea where this comes from
    win = torch.ones(n) * hamming_alpha
    win = win + (1 - hamming_alpha) * torch.cos(fac)

    # Build up the coefficients.
    alpha = 0.5 * (n - 1)
    m = torch.arange(0, n) - alpha
    left = pass_lo * 2 / fs
    right = pass_hi * 2 / fs
    h = right * torch.sinc(right * m) - left * torch.sinc(left * m)

    # Modulate coefficients by the window
    coefficients = h * win
    return coefficients


class Isolator(nn.Module):
    def __init__(self, targets: list[str]):
        pass
