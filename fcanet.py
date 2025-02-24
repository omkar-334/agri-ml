import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_freq_indices(method):
    assert method in ["top1", "top2", "top4", "top8", "top16", "top32", "bot1", "bot2", "bot4", "bot8", "bot16", "bot32", "low1", "low2", "low4", "low8", "low16", "low32"]
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = torch.tensor([0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1], device=device)
        all_top_indices_y = torch.tensor([0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3], device=device)
    elif "low" in method:
        all_top_indices_x = torch.tensor([0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4], device=device)
        all_top_indices_y = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3], device=device)
    elif "bot" in method:
        all_top_indices_x = torch.tensor([6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6], device=device)
        all_top_indices_y = torch.tensor([6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3], device=device)
    else:
        raise NotImplementedError

    return all_top_indices_x[:num_freq], all_top_indices_y[:num_freq]


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel).to(device))

    def forward(self, x):
        return torch.sum(x * self.weight, dim=[2, 3])

    @staticmethod
    def build_filter(pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result * math.sqrt(2) if freq != 0 else result

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y, device=device)
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super().__init__()
        self.reduction = reduction
        self.dct_h, self.dct_w = dct_h, dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [x * (dct_h // 7) for x in mapper_x]
        mapper_y = [y * (dct_w // 7) for y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel).to(device)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        n, c, h, w = x.shape

        # Efficient adaptive pooling for non-matching resolutions
        if h != self.dct_h or w != self.dct_w:
            x = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        y = self.dct_layer(x)
        y = self.fc(y).view(n, c, 1, 1)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False).to(device)
