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


class Model(nn.Module):
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
