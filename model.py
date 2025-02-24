import torch
import torch.nn as nn
import torch.nn.functional as F

from fcanet import MultiSpectralAttentionLayer
from senet import SELayer

# Input layer:
# - Size: 448×448×3

# Head block:
# - Size: 112×112×64

# Anti-aliasing block:
# - Input: 112×112×64
# - Output: 112×112×64

# MAIA block:
# - Input: 112×112×64
# - Output: 112×112×256

# Anti-aliasing block:
# - Input: 56×56×512
# - Output: 56×56×512

# MAIA block:
# - Input: 56×56×512
# - Output: 56×56×512

# Anti-aliasing block:
# - Input: 28×28×1024
# - Output: 28×28×1024

# MAIA block:
# - Input: 28×28×1024
# - Output: 28×28×1024

# MAIA block:
# - Input: 14×14×2048
# - Output: 14×14×2048

# Output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaiaNet(nn.Module):
    def __init__(self, num_classes):
        super(MaiaNet, self).__init__()
        self.head = HeadBlock(3, 64)  # Input: 448×448×3 -> 112×112×64
        self.anti_aliasing_1 = AntiAliasingBlock(64, 64, downsample=False)  # 112×112×64 -> 112×112×64
        self.maia_1 = MaiaBlock(64, 256)  # 112×112×64 -> 112×112×256
        self.anti_aliasing_2 = AntiAliasingBlock(256, 512, downsample=True)  # 112×112×256 -> 56×56×512
        self.maia_2 = MaiaBlock(512, 512)  # 56×56×512 -> 56×56×512
        self.anti_aliasing_3 = AntiAliasingBlock(512, 1024, downsample=True)  # 56×56×512 -> 28×28×1024
        self.maia_3 = MaiaBlock(1024, 1024)  # 28×28×1024 -> 28×28×1024
        self.maia_4 = MaiaBlock(1024, 2048, downsample=True)  # 14×14×2048 -> 14×14×2048
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Converts 14x14x2048 to 1x1x2048
        self.fc = nn.Linear(2048, num_classes)  # Fully connected layer (2048 -> num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, verbose=False):
        if verbose:
            print("Input:", x.shape)
        x = self.head(x)
        if verbose:
            print("Head:", x.shape)
        x = self.anti_aliasing_1(x)
        if verbose:
            print("Anti-aliasing 1:", x.shape)
        x = self.maia_1(x)
        if verbose:
            print("MAIA 1:", x.shape)
        x = self.anti_aliasing_2(x)
        if verbose:
            print("Anti-aliasing 2:", x.shape)
        x = self.maia_2(x)
        if verbose:
            print("MAIA 2:", x.shape)
        x = self.anti_aliasing_3(x)
        if verbose:
            print("Anti-aliasing 3:", x.shape)
        x = self.maia_3(x)
        if verbose:
            print("MAIA 3:", x.shape)
        x = self.maia_4(x)
        if verbose:
            print("MAIA 4:", x.shape)

        x = self.global_pool(x)  # Shape: (batch_size, 2048, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (batch_size, 2048)
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x


class HeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class MultiAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiAttention, self).__init__()

        # https://github.com/hujie-frank/SENet/blob/master/README.md
        self.se = SELayer(in_channels, reduction=16)

        # https://github.com/cfzd/FcaNet/blob/master/model/fcanet.py
        self.fca = MultiSpectralAttentionLayer(in_channels, 7, 7, reduction=16, freq_sel_method="top16")

    def forward(self, x):
        x = self.se(x)
        x = self.fca(x)
        return x


class AntiAliasingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(AntiAliasingBlock, self).__init__()

        self.downsample = downsample

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.down_conversion = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        stride = 2 if self.downsample else 1
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.ma = MultiAttention(out_channels)
        self.ibn = nn.InstanceNorm2d(out_channels)

        self.residual_conv = None
        if in_channels != out_channels or downsample:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.block1(x)
        out = self.down_conversion(out)
        out = self.block2(out)
        out = self.ma(out)
        if self.residual_conv:
            x = self.residual_conv(x)
        out = out + x
        out = self.ibn(out)
        out = F.relu(out)
        return out


class MaiaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(MaiaBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        self.ma = MultiAttention(out_channels)
        self.ibn = nn.InstanceNorm2d(out_channels)

        if in_channels != out_channels or downsample:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.ma(out)

        if self.residual_conv:
            x = self.residual_conv(x)
        out = out + x
        out = self.ibn(out)
        out = F.relu(out)
        return out


if __name__ == "__main__":
    model = MaiaNet(num_classes=5).to(device)
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    print(output.shape)
