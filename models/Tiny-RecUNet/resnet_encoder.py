import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet encoder producing multi-scale features: f2 (H/2), f4 (H/4), f8 (H/8), f16 (H/16).

    Supports variants: 'resnet34' (BasicBlock) and 'resnet50' (Bottleneck).
    """

    def __init__(self, name: str = "resnet34", in_channels: int = 3):
        super().__init__()
        if name not in {"resnet34", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {name}")
        self.name = name
        block = BasicBlock if name == "resnet34" else Bottleneck
        layers_cfg = [3, 4, 6, 3]  # standard for 34/50

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_cfg[2], stride=2)
        # layer4 exists in classic resnet, but we stop at H/16 for TransUNet
        # self.layer4 = self._make_layer(block, 512, layers_cfg[3], stride=2)

        if name == "resnet34":
            c2, c4, c8, c16 = 64, 64, 128, 256
        else:  # resnet50
            c2, c4, c8, c16 = 64, 256, 512, 1024
        self.channels = {"c2": c2, "c4": c4, "c8": c8, "c16": c16}

        self._init_weights()

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = planes * block.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        f2 = x
        x = self.maxpool(x)  # /4
        x = self.layer1(x)   # /4
        f4 = x
        x = self.layer2(x)   # /8
        f8 = x
        x = self.layer3(x)   # /16
        f16 = x
        return f16, [f8, f4, f2]
