import json
import math
from pathlib import Path

import torch
from torch._C import Value
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


def firwin(n, pass_lo, pass_hi):
    """
    Returns bandpass filters letting through the specified frequencies.

    Inputs
    ------
    n : int
        Order of the filter.
    pass_lo, pass_hi : array[float]
        Both are m-length arrays of floats, defining the low and high ends
        of the pass-through bands. One filter will be returned for each
        element of these arrays.
        Should be normalized: divide the desired frequency by the sample rate.
        YOU must make sure that 0 <= pass_lo[i] < pass_hi[i] < 0.5 for all i

    Returns
    -------
    filters : array[float]
        A 2d array of shape (len(pass_lo), n).
        filters[i] is the filter corresponding to the pass-through band
        (pass_lo[i], pass_hi[i])

    >>> import torch
    >>> import numpy as np
    >>> import scipy.signal
    >>> fs = 22500
    >>> x = np.random.default_rng(1234).random(300_000)
    >>> x_torch = torch.from_numpy(x).float()[None, None, :]
    >>> pass_lo, pass_hi = torch.tensor([500]) / fs, torch.tensor([8000]) / fs
    >>> filt_torch = firwin(255, pass_lo, pass_hi)[0][None, None, :]
    >>> filtered_torch = torch.nn.functional.conv1d(x_torch, filt_torch)[0, 0]
    >>> filt = scipy.signal.firwin(255, [500, 8000], fs=fs, pass_zero=False)
    >>> filtered = scipy.signal.convolve(x, filt, 'valid', 'direct')
    >>> diff = torch.from_numpy(filtered).float() - filtered_torch
    >>> assert diff.abs().max() < 1e-3
    """
    # Adapated from scipy, this is a simplified version of their firwin(),
    # except that it handles arrays of pass_lo and pass_hi
    if not n % 2 or n <= 10:
        raise ValueError("n must be odd and greater than 10")

    # Build hamming window
    fac = torch.linspace(-math.pi, math.pi, n)
    hamming_alpha = 0.54  # no idea where this comes from
    win = torch.ones(n) * hamming_alpha
    win = win + (1 - hamming_alpha) * torch.cos(fac)

    # Build up the coefficients.
    alpha = 0.5 * (n - 1)
    m = (torch.arange(0, n) - alpha)[None, :].to(pass_lo.device)
    lo = (pass_lo * 2)[:, None]
    hi = (pass_hi * 2)[:, None]
    h = hi * torch.sinc(hi @ m) - lo * torch.sinc(lo @ m)

    # Modulate coefficients by the window
    coefficients = h * win[None, :].to(h.device)
    return coefficients


class Isolator(nn.Module):
    def __init__(self, targets: list[str]):
        super().__init__()

        self.identity = torch.nn.Identity()  # used for better summary

        channels = [1, 2, 4, 4, 8, 8, 16, 32]
        strides = [2, 2, 1, 2, 2, 1, 2]
        dilations = [1, 2, 2, 1, 2, 2, 1]
        assert len(strides) == len(channels) - 1 == len(dilations)
        self.resnet = nn.Sequential(
            *[
                ResLayer(
                    channels[i],
                    channels[i + 1],
                    5,
                    strides[i],
                    dilations[i],
                )
                for i in range(len(strides))
            ]
        )
        self.n = len(targets)
        self.regressor = nn.Conv1d(channels[-1], self.n * 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.targets = targets

    def forward(self, x, target_inds=None):
        filter_order = 255

        x_original = x
        x = x.unsqueeze(1)
        x = self.identity(x)

        x = self.resnet(x)
        x = self.sigmoid(self.regressor(x))
        x = x.reshape(x_original.shape[0], -1, self.n, 3)

        if target_inds is None:
            target_inds = torch.arange(self.n)
        y = torch.zeros(
            (x_original.shape[0], len(target_inds), x_original.shape[1]),
            device=x.device,
        )
        for i in target_inds:
            for j in range(x_original.shape[0]):
                bandpass_lo = x[j, :, i, 0] * 0.5
                # In order to ensure bandpass_hi > bandpass_lo, we put it
                # in terms of bandpass_lo + (a value guaranteed to be >= 0).
                bandpass_hi = bandpass_lo + x[j, :, i, 1] * (0.5 - bandpass_lo)
                filters = firwin(filter_order, bandpass_lo, bandpass_hi)
                filters = nn.functional.interpolate(
                    filters.T[None, :, :],
                    size=x_original.shape[1],
                    mode="nearest",
                )[0].T

                buffered_x = torch.nn.functional.pad(
                    x_original[j], (filter_order // 2, filter_order // 2)
                )
                buffered_x = buffered_x.unfold(0, filter_order, 1)
                y[j, i, :] = (buffered_x * filters).sum(dim=1)

                volume = nn.functional.interpolate(
                    x[j, :, i, 2][None, None, :],
                    size=x_original.shape[1],
                    mode="linear",
                    align_corners=False,
                )[0, 0]
                y[j, i, :] *= volume

        return y
