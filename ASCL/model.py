import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_se=False):
        super(self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class AsclSEResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 256, 3)
        self.layer2 = self._make_layer(256, 128, 512, 4, use_se=True)
        self.layer3 = self._make_layer(512, 256, 1024, 6, use_se=True)
        self.layer4 = self._make_layer(1024, 512, 2048, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_projection = nn.Sequential(nn.Linear(2048, 100), nn.ReLU(inplace=True), nn.Linear(100, 100))

    @staticmethod
    def _make_layer(in_channels, mid_channels, out_channels, blocks, use_se=False):
        layers = [ConvBlock(in_channels, mid_channels, out_channels, use_se)]
        layers.extend(ConvBlock(out_channels, mid_channels, out_channels, use_se) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # 64 → 256
        x = self.layer2(x)  # 256 → 512
        x = self.layer3(x)  # 512 → 1024
        x = self.layer4(x)  # 1024 → 2048
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_projection(x)
        return x


# Example usage:
model = AsclSEResNet(num_classes=100)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)
