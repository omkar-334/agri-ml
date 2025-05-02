import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pointwise_conv1 = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x = x + shortcut
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=4),  # 224x224 -> 56x56
            ConvNeXtBlock(128, 128),  # Output shape: [batch_size, 128, 56, 56]
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2),  # 56x56 -> 28x28
            ConvNeXtBlock(256, 256),  # Output shape: [batch_size, 256, 28, 28]
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2),  # 28x28 -> 14x14
            ConvNeXtBlock(512, 512),  # Output shape: [batch_size, 512, 14, 14]
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=2, stride=2),  # 14x14 -> 7x7
            ConvNeXtBlock(1024, 1024),  # Output shape: [batch_size, 1024, 7, 7]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [batch_size, 1024, 1, 1]

    def forward(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        # x = self.pool(x).squeeze()  # Shape: [batch_size, 1024]
        x = self.pool(x).view(x.size(0), -1)  # Ensure shape is always [batch_size, 1024]

        return x, features


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Upsampling blocks to decode the feature maps into an image of shape 3x224x224
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=4, padding=1)  # 1x1 -> 4x4
        self.block4 = ConvNeXtBlock(512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=1)  # 4x4 -> 16x16
        self.block3 = ConvNeXtBlock(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=1)  # 16x16 -> 64x64
        self.block2 = ConvNeXtBlock(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=11, padding=6, output_padding=1)  # 64x64 -> 224x224 (final output)
        # 1x1 convolution to match the channels for the skip connections
        self.conv1x1_3 = nn.Conv2d(1024, 512, kernel_size=1)  # Match channels to 512
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1)  # Match channels to 256
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1)  # Match channels to 128

    def forward(self, latent, features):
        # Ensure latent has a batch dimension
        # if latent.dim() == 1:
        #     latent = latent.unsqueeze(0)  # Add batch dimension if missing

        if latent.dim() == 3:  # Check if the batch dimension is missing
            latent = latent.unsqueeze(0)  # Add batch dimension if missing
        # Reshape the latent vector to [batch_size, 1024, 1, 1]
        x = latent.view(latent.size(0), 1024, 1, 1)  # Batch size, 1024 channels, 1x1 spatial size

        # Upsample and add skip connections
        x = self.upconv4(x)  # [batch_size, 512, 4, 4]
        features_3_resized = F.interpolate(features[3], size=x.shape[2:], mode="bilinear", align_corners=False)
        features_3_resized = self.conv1x1_3(features_3_resized)  # Match channels to 512
        x = x + features_3_resized  # Add skip connection
        x = self.block4(x)

        x = self.upconv3(x)  # [batch_size, 256, 16, 16]
        features_2_resized = F.interpolate(features[2], size=x.shape[2:], mode="bilinear", align_corners=False)
        features_2_resized = self.conv1x1_2(features_2_resized)  # Match channels to 256
        x = x + features_2_resized  # Add skip connection
        x = self.block3(x)

        x = self.upconv2(x)  # [batch_size, 128, 64, 64]
        features_1_resized = F.interpolate(features[1], size=x.shape[2:], mode="bilinear", align_corners=False)
        features_1_resized = self.conv1x1_1(features_1_resized)  # Match channels to 128
        x = x + features_1_resized  # Add skip connection
        x = self.block2(x)

        x = self.upconv1(x)  # [batch_size, 3, 224, 224]

        return torch.sigmoid(x)  # Output image between 0 and 1


# Full Autoencoder Architecture:
class Autoencoder(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(1024, num_classes)  # For supervised classification

    def forward(self, x):
        latent, _ = self.encoder(x)
        classification = self.fc(latent)
        return classification

    def train_forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent.unsqueeze(-1).unsqueeze(-1), features)
        classification = self.fc(latent)
        return reconstructed, classification
