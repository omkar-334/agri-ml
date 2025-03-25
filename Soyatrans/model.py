import torch
import torch.nn as nn
from torchvision import models

from .swin_transformer import PatchMerging, SwinTransformerBlock


class SoyaTrans(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.vgg = VGG()
        self.inception = InceptionV7(128)
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        x = self.vgg(x)
        x = self.inception(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.classifier(x)
        return x


class Stage1(nn.Module):
    def __init__(self, input_res=(56, 56), dim=512, num_heads=8):
        super().__init__()
        self.linear_embedding = nn.Linear(dim, dim)
        self.swin_block1 = SwinTransformerBlock(dim=dim, input_resolution=input_res, num_heads=num_heads)
        self.swin_block2 = SwinTransformerBlock(dim=dim, input_resolution=input_res, num_heads=num_heads)

        # Downsampling Layer (to reduce 56x56 → 14x14)
        self.downsample = nn.Conv2d(dim, dim, kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to (B, L, C) since SwinTransformer block expects this format
        x = x.flatten(2).transpose(1, 2)

        x = self.linear_embedding(x)
        x = self.swin_block1(x)
        x = self.swin_block2(x)

        # Reshape back to (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.downsample(x)
        return x


class Stage2(nn.Module):
    def __init__(self, input_res=(14, 14), dim=512, num_heads=8, window_size=7):
        super().__init__()

        self.patch_merging = PatchMerging(input_resolution=input_res, dim=dim)  # Merges patches, doubles channels (512 → 1024)

        # Reduce channels back to 512 after merging
        self.channel_reduction = nn.Linear(1024, 512)  # Reduce (B, 49, 1024) → (B, 49, 512)

        self.swin_block1 = SwinTransformerBlock(dim=dim, input_resolution=(7, 7), num_heads=num_heads, window_size=window_size)
        self.swin_block2 = SwinTransformerBlock(dim=dim, input_resolution=(7, 7), num_heads=num_heads, window_size=window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Convert (B, C, H, W) → (B, L, C) → (B, 196, 512)

        x = self.patch_merging(x)  # Output: (B, 49, 1024)
        x = self.channel_reduction(x)  # Reduce back to 512 channels (B, 49, 512)

        x = self.swin_block1(x)  # Swin Block 1
        x = self.swin_block2(x)  # Swin Block 2

        # Reshape back to (B, C, H, W) → (B, 512, 7, 7)
        x = x.transpose(1, 2).view(B, 512, 7, 7)
        return x  # Output shape: (B, 512, 7, 7)


class Classifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=6):
        super().__init__()

        # Global Average Pooling (B, 512, 7, 7) → (B, 512, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification layers
        self.norm = nn.LayerNorm(input_dim)  # Layer Normalization
        self.fc = nn.Linear(input_dim, num_classes)  # Fully Connected Layer
        self.softmax = nn.Softmax(dim=1)  # Softmax Activation for Multi-class

    def forward(self, x):
        B, H, W, C = x.shape  # (B, 7, 7, 512)

        x = x.flatten(2)
        x = self.global_avg_pool(x)
        # Remove last dimension → (B, 512)
        x = x.squeeze(-1)

        # Classification steps
        x = self.norm(x)  # Layer Normalization
        x = self.fc(x)  # Fully Connected Layer (B, 512) → (B, num_classes)
        x = self.softmax(x)  # Softmax Activation

        return x  # Output shape: (B, num_classes)


class VGG(nn.Module):
    """VGG model (upto 10 layers)

    Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    """

    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features[:10]

    def forward(self, x):
        return self.vgg(x)


class InceptionV7(nn.Module):
    """Custom Inception V7 model

    branch1 - maxpool 3x3
    branch2 - conv2d (1x1) -> conv2d (3x1 + 1x3) -> conv2d (3x1 + 1x3)
    branch3 - conv2d (1x1) -> conv2d (3x1 + 1x3)
    branch4 - conv2d (1x1)
    """

    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out
