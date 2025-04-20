import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_freq_indices(method):
    assert method in ["top1", "top2", "top4", "top8", "top16", "top32", "bot1", "bot2", "bot4", "bot8", "bot16", "bot32", "low1", "low2", "low4", "low8", "low16", "low32"]
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif "low" in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif "bot" in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, "x must been 4 dimensions, but got " + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
        super(self).__init__()
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
        super(self).__init__()
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
        super(self).__init__()

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
