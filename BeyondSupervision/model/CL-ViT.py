import collections.abc
import os
from functools import partial
from itertools import repeat

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from albumentations import CenterCrop, CoarseDropout, Compose, Crop, Cutout, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# SSL data augmentations
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def flip_ver(x):
    return x.flip(1)


def flip_hor(x):
    return x.flip(2)


# Custom dataset for testing images
class CustomDatasetFromImagesForalbumentation(Dataset):
    def __init__(self, csv_path, transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels for plant
        self.label_arr_p = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels for disease
        self.label_arr_d = np.asarray(self.data_info.iloc[:, 2])

        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.data_path = data_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.data_path, single_image_name)
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label_p = self.label_arr_p[index]
        single_image_label_d = self.label_arr_d[index]

        return (img_as_tensor, single_image_label_p, single_image_label_d)

    def __len__(self):
        return self.data_len


# Custom dataset for training images
class CustomDatasetForSLandSSL_SingleA(Dataset):
    def __init__(self, csv_path, transforms1, transforms2):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels for plant
        self.label_arr_p = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels for disease
        self.label_arr_d = np.asarray(self.data_info.iloc[:, 2])

        # variables
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.data_len = len(self.data_info.index)
        self.data_path = data_path
        self.img_size = 224

        # SSL data augmentations
        self.crop4_1 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 0, 112, 112),
            ],
            p=1.0,
        )

        self.crop4_2 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 0, 224, 112),
            ],
            p=1.0,
        )

        self.crop4_3 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 112, 112, 224),
            ],
            p=1.0,
        )

        self.crop4_4 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 112, 224, 224),
            ],
            p=1.0,
        )

        self.crop16_1 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 0, 56, 56),
            ],
            p=1.0,
        )

        self.crop16_2 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(56, 0, 112, 56),
            ],
            p=1.0,
        )

        self.crop16_3 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 0, 168, 56),
            ],
            p=1.0,
        )

        self.crop16_4 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(168, 0, 224, 56),
            ],
            p=1.0,
        )

        self.crop16_5 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 56, 56, 112),
            ],
            p=1.0,
        )

        self.crop16_6 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(56, 56, 112, 112),
            ],
            p=1.0,
        )

        self.crop16_7 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 56, 168, 112),
            ],
            p=1.0,
        )

        self.crop16_8 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(168, 56, 224, 112),
            ],
            p=1.0,
        )

        self.crop16_9 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 112, 56, 168),
            ],
            p=1.0,
        )

        self.crop16_10 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(56, 112, 112, 168),
            ],
            p=1.0,
        )

        self.crop16_11 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 112, 168, 168),
            ],
            p=1.0,
        )

        self.crop16_12 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(168, 112, 224, 168),
            ],
            p=1.0,
        )

        self.crop16_13 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(0, 168, 56, 224),
            ],
            p=1.0,
        )

        self.crop16_14 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(56, 168, 112, 224),
            ],
            p=1.0,
        )

        self.crop16_15 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(112, 168, 168, 224),
            ],
            p=1.0,
        )

        self.crop16_16 = Compose(
            [
                Resize(self.img_size, self.img_size),
                Crop(168, 168, 224, 224),
            ],
            p=1.0,
        )
        self.transformToTensor = Compose(
            [
                Resize(img_size, img_size),
                CenterCrop(img_size, img_size, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
        self.transformToPIL = transforms.ToPILImage()

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.data_path, single_image_name)
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform and normalized image
        transformed = self.transforms1(image=image)
        img_as_tensor = transformed["image"]

        # Transform and normalized image (convert to tensor)
        transformed2 = self.transforms2(image=image)
        img_as_tensor2 = transformed2["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label_p = self.label_arr_p[index]
        single_image_label_d = self.label_arr_d[index]

        # Randomised the selection of SSL data augmentations
        ss_aug = np.random.randint(3)

        # If 1st SSL data augmentation
        if ss_aug == 0:
            # Randomised the selection of 1st SSL data augmentation
            # 0, 90, 180 or 270 degree rotations
            ss1_label = np.random.randint(4)
            ss2_label = 0
            ss3_label = 0

            if ss1_label == 0:
                img_as_tensor = img_as_tensor2
            elif ss1_label == 1:
                img_as_tensor = tensor_rot_90(img_as_tensor2)
            elif ss1_label == 2:
                img_as_tensor = tensor_rot_180(img_as_tensor2)
            elif ss1_label == 3:
                img_as_tensor = tensor_rot_270(img_as_tensor2)

        # If 2nd SSL data augmentation
        elif ss_aug == 1:
            # Randomised the selection of 2nd SSL data augmentation
            # original, 4 or 16 randomized image patches augmentations
            ss2_label = np.random.randint(3)
            ss1_label = 0
            ss3_label = 0

            if ss2_label == 0:
                img_as_tensor = img_as_tensor2

            elif ss2_label == 1:
                image = img_as_tensor
                transformed1 = self.crop4_1(image=image)
                i1 = transformed1["image"]

                transformed2 = self.crop4_2(image=image)
                i2 = transformed2["image"]

                transformed3 = self.crop4_3(image=image)
                i3 = transformed3["image"]

                transformed4 = self.crop4_4(image=image)
                i4 = transformed4["image"]

                n1 = np.concatenate((i4, i3), axis=1)
                n2 = np.concatenate((i2, i1), axis=1)

                image = np.concatenate((n1, n2), axis=0)
                transformed5 = self.transformToTensor(image=image)
                img_as_tensor = transformed5["image"]

            elif ss2_label == 2:
                image = img_as_tensor
                transformed1 = self.crop16_1(image=image)
                i1 = transformed1["image"]

                transformed2 = self.crop16_2(image=image)
                i2 = transformed2["image"]

                transformed3 = self.crop16_3(image=image)
                i3 = transformed3["image"]

                transformed4 = self.crop16_4(image=image)
                i4 = transformed4["image"]

                transformed5 = self.crop16_5(image=image)
                i5 = transformed5["image"]

                transformed6 = self.crop16_6(image=image)
                i6 = transformed6["image"]

                transformed7 = self.crop16_7(image=image)
                i7 = transformed7["image"]

                transformed8 = self.crop16_8(image=image)
                i8 = transformed8["image"]

                transformed9 = self.crop16_9(image=image)
                i9 = transformed9["image"]

                transformed10 = self.crop16_10(image=image)
                i10 = transformed10["image"]

                transformed11 = self.crop16_11(image=image)
                i11 = transformed11["image"]

                transformed12 = self.crop16_12(image=image)
                i12 = transformed12["image"]

                transformed13 = self.crop16_13(image=image)
                i13 = transformed13["image"]

                transformed14 = self.crop16_14(image=image)
                i14 = transformed14["image"]

                transformed15 = self.crop16_15(image=image)
                i15 = transformed15["image"]

                transformed16 = self.crop16_16(image=image)
                i16 = transformed16["image"]

                n1 = np.concatenate((i6, i5, i8, i7), axis=1)
                n2 = np.concatenate((i2, i1, i4, i3), axis=1)
                n3 = np.concatenate((i14, i13, i16, i15), axis=1)
                n4 = np.concatenate((i10, i9, i12, i11), axis=1)

                image = np.concatenate((n1, n2, n3, n4), axis=0)
                transformed17 = self.transformToTensor(image=image)
                img_as_tensor = transformed17["image"]

        # If 3rd SSL data augmentation
        elif ss_aug == 2:
            # Randomised the selection of 3rd SSL data augmentation
            # original, horzontal or vertical flip
            ss3_label = np.random.randint(3)
            ss1_label = 0
            ss2_label = 0

            if ss3_label == 0:
                img_as_tensor = img_as_tensor2
            elif ss3_label == 1:
                img_as_tensor = flip_ver(img_as_tensor2)
            elif ss3_label == 2:
                img_as_tensor = flip_hor(img_as_tensor2)

        return (img_as_tensor, single_image_label_p, single_image_label_d, ss1_label, ss2_label, ss3_label)

    def __len__(self):
        return self.data_len


class Model_SL_SSL(nn.Module):
    def __init__(self, model, num_plant, num_disease, num_rotation, num_crop, num_flip):
        super().__init__()
        self.model = model

        self.dim = 768
        self.dim_a = 768
        self.dim_r = 192
        self.num_heads = 12
        self.mlp_ratio = 4.0
        self.qkv_bias = True
        self.drop = 0.0
        self.attn_drop = 0.0
        self.drop_path = 0.0
        self.act_layer = nn.GELU
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block1a_p = Block(
            dim=self.dim_a,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block1b_p = Block(
            dim=self.dim_r,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block1a_d = Block(
            dim=self.dim_a,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block1b_d = Block(
            dim=self.dim_r,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block2a = Block(
            dim=self.dim_a,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block2b = Block(
            dim=self.dim_r,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )
        self.block3a = Block(
            dim=self.dim_a,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block3b = Block(
            dim=self.dim_r,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )
        self.block4a = Block(
            dim=self.dim_a,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        self.block4b = Block(
            dim=self.dim_r,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
        )

        bias = True
        bias_mlp = to_2tuple(bias)
        self.mlp1p = nn.Linear(self.dim, self.dim_r, bias=bias_mlp[0], device=device)
        self.mlp1d = nn.Linear(self.dim, self.dim_r, bias=bias_mlp[0], device=device)
        self.mlp2 = nn.Linear(self.dim, self.dim_r, bias=bias_mlp[0], device=device)
        self.mlp3 = nn.Linear(self.dim, self.dim_r, bias=bias_mlp[0], device=device)
        self.mlp4 = nn.Linear(self.dim, self.dim_r, bias=bias_mlp[0], device=device)

        self.linearh1p = nn.Linear(self.dim_r, (num_plant), device=device)
        self.linearh1d = nn.Linear(self.dim_r, (num_disease), device=device)
        self.linearh2 = nn.Linear(self.dim_r, num_rotation, device=device)
        self.linearh3 = nn.Linear(self.dim_r, num_crop, device=device)
        self.linearh4 = nn.Linear(self.dim_r, num_flip, device=device)
        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def testp(self, i1):
        self.f = self.model.forward_features(i1)

        # Self-attention layer
        self.h1p = self.block1a_p(self.f)
        self.h2 = self.block2a(self.f)
        self.h3 = self.block3a(self.f)
        self.h4 = self.block4a(self.f)

        # MLP layer
        self.h1p = self.mlp1p(self.h1p)
        self.h2 = self.mlp2(self.h2)
        self.h3 = self.mlp3(self.h3)
        self.h4 = self.mlp4(self.h4)

        # Obtain the CLS Token for each features
        self.h1cp = self.h1p[:, 0]
        self.h2c = self.h2[:, 0]
        self.h3c = self.h3[:, 0]
        self.h4c = self.h4[:, 0]

        # Obtain the local patch features for plant feature
        self.h1pp = self.h1p[:, 1:]

        # Cross attention
        # Concatenate each CLS token with local patch features for plant feature
        self.h1new1p = torch.cat((self.h1cp.unsqueeze(1), self.h1pp), dim=1)
        self.h1new2 = torch.cat((self.h2c.unsqueeze(1), self.h1pp), dim=1)
        self.h1new3 = torch.cat((self.h3c.unsqueeze(1), self.h1pp), dim=1)
        self.h1new4 = torch.cat((self.h4c.unsqueeze(1), self.h1pp), dim=1)

        # Attention block
        self.h1n1p = self.block1b_p(self.h1new1p)
        self.h1n2 = self.block1b_p(self.h1new2)
        self.h1n3 = self.block1b_p(self.h1new3)
        self.h1n4 = self.block1b_p(self.h1new4)

        # MLP classifier
        self.h1n1p = self.linearh1p(self.h1n1p[:, 0])
        self.h1n2 = self.linearh1p(self.h1n2[:, 0])
        self.h1n3 = self.linearh1p(self.h1n3[:, 0])
        self.h1n4 = self.linearh1p(self.h1n4[:, 0])

        # softmax calculation
        self.h1n1p_soft = self.softmax(self.h1n1p)
        self.h1n2_soft = self.softmax(self.h1n2)
        self.h1n3_soft = self.softmax(self.h1n3)
        self.h1n4_soft = self.softmax(self.h1n4)
        self.h1nt_soft = self.h1n1p_soft + self.h1n2_soft + self.h1n3_soft + self.h1n4_soft
        return (self.h1n1p, self.h1n1p_soft, self.h1nt_soft)

    def testd(self, i1):
        self.f = self.model.forward_features(i1)

        # Self-attention layer
        self.h1d = self.block1a_d(self.f)
        self.h2 = self.block2a(self.f)
        self.h3 = self.block3a(self.f)
        self.h4 = self.block4a(self.f)

        # MLP layer
        self.h1d = self.mlp1d(self.h1d)
        self.h2 = self.mlp2(self.h2)
        self.h3 = self.mlp3(self.h3)
        self.h4 = self.mlp4(self.h4)

        # Obtain the CLS Token for each features
        self.h1cd = self.h1d[:, 0]
        self.h2c = self.h2[:, 0]
        self.h3c = self.h3[:, 0]
        self.h4c = self.h4[:, 0]

        # Obtain the local patch features for disease feature
        self.h1pd = self.h1d[:, 1:]

        # Cross attention
        # Concatenate each CLS token with local patch features for disease feature
        self.h1new1d = torch.cat((self.h1cd.unsqueeze(1), self.h1pd), dim=1)
        self.h1new2 = torch.cat((self.h2c.unsqueeze(1), self.h1pd), dim=1)
        self.h1new3 = torch.cat((self.h3c.unsqueeze(1), self.h1pd), dim=1)
        self.h1new4 = torch.cat((self.h4c.unsqueeze(1), self.h1pd), dim=1)

        # Attention block
        self.h1n1d = self.block1b_d(self.h1new1d)
        self.h1n2 = self.block1b_d(self.h1new2)
        self.h1n3 = self.block1b_d(self.h1new3)
        self.h1n4 = self.block1b_d(self.h1new4)

        # MLP classifier
        self.h1n1d = self.linearh1d(self.h1n1d[:, 0])
        self.h1n2 = self.linearh1d(self.h1n2[:, 0])
        self.h1n3 = self.linearh1d(self.h1n3[:, 0])
        self.h1n4 = self.linearh1d(self.h1n4[:, 0])

        # softmax calculation
        self.h1n1d_soft = self.softmax(self.h1n1d)
        self.h1n2_soft = self.softmax(self.h1n2)
        self.h1n3_soft = self.softmax(self.h1n3)
        self.h1n4_soft = self.softmax(self.h1n4)
        self.h1nt_soft = self.h1n1d_soft + self.h1n2_soft + self.h1n3_soft + self.h1n4_soft
        return (self.h1n1d, self.h1n1d_soft, self.h1nt_soft)

    def forward(self, i1):
        self.f = self.model.forward_features(i1)

        return (self.h1cc, self.h1p)

    def seen(self, i1):
        self.f = self.model.forward_features(i1)

        # Self-attention layer
        self.h1p = self.block1a_p(self.f)
        self.h1d = self.block1a_d(self.f)
        self.h2 = self.block2a(self.f)
        self.h3 = self.block3a(self.f)
        self.h4 = self.block4a(self.f)

        # MLP layer
        self.h1p = self.mlp1p(self.h1p)
        self.h1d = self.mlp1d(self.h1d)
        self.h2 = self.mlp2(self.h2)
        self.h3 = self.mlp3(self.h3)
        self.h4 = self.mlp4(self.h4)

        # Obtain the CLS Token for each features
        self.h1cp = self.h1p[:, 0]
        self.h1cd = self.h1d[:, 0]
        self.h2c = self.h2[:, 0]
        self.h3c = self.h3[:, 0]
        self.h4c = self.h4[:, 0]

        # randomised the CLS token for plant and SSL prediction
        self.hcp_new = torch.cat((self.h1cp.unsqueeze(1), self.h2c.unsqueeze(1), self.h3c.unsqueeze(1), self.h4c.unsqueeze(1)), dim=1)
        self.hcp_new = self.hcp_new.squeeze(0)
        self.idx = torch.randperm(self.hcp_new.shape[0])
        self.hcpnew_s = self.hcp_new[self.idx].view(self.hcp_new.size())
        self.hcpnew_s = self.hcpnew_s.unsqueeze(0)

        self.h1ccp = self.hcpnew_s[0][0]
        self.h2cc = self.hcpnew_s[0][1]
        self.h3cc = self.hcpnew_s[0][2]
        self.h4cc = self.hcpnew_s[0][3]

        # randomised the CLS token for disease and SSL prediction
        self.hcd_new = torch.cat((self.h1cd.unsqueeze(1), self.h2c.unsqueeze(1), self.h3c.unsqueeze(1), self.h4c.unsqueeze(1)), dim=1)
        self.hcd_new = self.hcd_new.squeeze(0)
        self.idx = torch.randperm(self.hcd_new.shape[0])
        self.hcdnew_s = self.hcd_new[self.idx].view(self.hcd_new.size())
        self.hcdnew_s = self.hcdnew_s.unsqueeze(0)

        self.h1ccd = self.hcdnew_s[0][0]
        self.h2cc = self.hcdnew_s[0][1]
        self.h3cc = self.hcdnew_s[0][2]
        self.h4cc = self.hcdnew_s[0][3]

        # Obtain the local patch features for each feature
        self.h1pp = self.h1p[:, 1:]
        self.h1pd = self.h1d[:, 1:]
        self.h2p = self.h2[:, 1:]
        self.h3p = self.h3[:, 1:]
        self.h4p = self.h4[:, 1:]

        # Cross attention
        # Concatenate the randomised CLS token with local patch features for each feature for plant and SSL prediction
        self.h1newp = torch.cat((self.h1ccp.unsqueeze(0).unsqueeze(0), self.h1pp), dim=1)
        self.h2newp = torch.cat((self.h2cc.unsqueeze(0).unsqueeze(0), self.h2p), dim=1)
        self.h3newp = torch.cat((self.h3cc.unsqueeze(0).unsqueeze(0), self.h3p), dim=1)
        self.h4newp = torch.cat((self.h4cc.unsqueeze(0).unsqueeze(0), self.h4p), dim=1)

        # Concatenate the randomised CLS token with local patch features for each feature for disease and SSL prediction
        self.h1newd = torch.cat((self.h1ccd.unsqueeze(0).unsqueeze(0), self.h1pd), dim=1)
        self.h2newd = torch.cat((self.h2cc.unsqueeze(0).unsqueeze(0), self.h2p), dim=1)
        self.h3newd = torch.cat((self.h3cc.unsqueeze(0).unsqueeze(0), self.h3p), dim=1)
        self.h4newd = torch.cat((self.h4cc.unsqueeze(0).unsqueeze(0), self.h4p), dim=1)

        # Attention block
        self.h1p = self.block1b_p(self.h1newp)
        self.h1d = self.block1b_d(self.h1newd)
        self.h2p = self.block2b(self.h2newp)
        self.h3p = self.block3b(self.h3newp)
        self.h4p = self.block4b(self.h4newp)
        self.h2d = self.block2b(self.h2newd)
        self.h3d = self.block3b(self.h3newd)
        self.h4d = self.block4b(self.h4newd)

        # MLP classifier
        self.h1p = self.linearh1p(self.h1p[:, 0])
        self.h1d = self.linearh1d(self.h1d[:, 0])
        self.h2p = self.linearh2(self.h2p[:, 0])
        self.h3p = self.linearh3(self.h3p[:, 0])
        self.h4p = self.linearh4(self.h4p[:, 0])
        self.h2d = self.linearh2(self.h2d[:, 0])
        self.h3d = self.linearh3(self.h3d[:, 0])
        self.h4d = self.linearh4(self.h4d[:, 0])
        return (self.h1p, self.h1d, self.h2p, self.h3p, self.h4p, self.h2d, self.h3d, self.h4d)

    def unseen(self, i1):
        self.f = self.model.forward_features(i1)

        # Self-attention layer
        self.h2 = self.block2a(self.f)
        self.h3 = self.block3a(self.f)
        self.h4 = self.block4a(self.f)

        # MLP layer
        self.h2 = self.mlp2(self.h2)
        self.h3 = self.mlp3(self.h3)
        self.h4 = self.mlp4(self.h4)

        # Obtain the CLS Token for each features
        self.h2c = self.h2[:, 0]
        self.h3c = self.h3[:, 0]
        self.h4c = self.h4[:, 0]

        # randomised the CLS token for SSL prediction
        self.hc_new = torch.cat((self.h2c.unsqueeze(1), self.h3c.unsqueeze(1), self.h4c.unsqueeze(1)), dim=1)
        self.hc_new = self.hc_new.squeeze(0)
        self.idx = torch.randperm(self.hc_new.shape[0])
        self.hcnew_s = self.hc_new[self.idx].view(self.hc_new.size())
        self.hcnew_s = self.hcnew_s.unsqueeze(0)

        self.h2cc = self.hcnew_s[0][0]
        self.h3cc = self.hcnew_s[0][1]
        self.h4cc = self.hcnew_s[0][2]

        # Obtain the local patch features for each feature
        self.h2p = self.h2[:, 1:]
        self.h3p = self.h3[:, 1:]
        self.h4p = self.h4[:, 1:]

        # Cross attention
        # Concatenate the randomised CLS token with local patch features for each feature for SSL prediction
        self.h2new = torch.cat((self.h2cc.unsqueeze(0).unsqueeze(0), self.h2p), dim=1)
        self.h3new = torch.cat((self.h3cc.unsqueeze(0).unsqueeze(0), self.h3p), dim=1)
        self.h4new = torch.cat((self.h4cc.unsqueeze(0).unsqueeze(0), self.h4p), dim=1)

        # Attention block
        self.h2 = self.block2b(self.h2new)
        self.h3 = self.block3b(self.h3new)
        self.h4 = self.block4b(self.h4new)

        # MLP classifier
        self.h2 = self.linearh2(self.h2[:, 0])
        self.h3 = self.linearh3(self.h3[:, 0])
        self.h4 = self.linearh4(self.h4[:, 0])
        return (self.h2, self.h3, self.h4)


# Hyperparameters and variables
num_plant = 14
num_disease = 21
num_epochs = 30
batch_size = 1
num_rotation = 4
num_crop = 3
num_flip = 3
img_size = 224
learning_rate_layer = 0.001
momentum = 0.9
weight_decay = 0.00001
pretrained = True

# Default data augmentation
ori_train_transforms_2 = Compose(
    [
        Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
    ],
    p=1.0,
)

ori_train_transforms_3 = Compose(
    [
        Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

ori_valid_transforms = Compose(
    [
        Resize(img_size, img_size),
        CenterCrop(img_size, img_size, p=1.0),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

# Data path for datasets and metadatas
data_path = "C:/Users/User/Desktop/Vision Transformer/plantvillage (mix)"
Save_model_path = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/save model"
train_csv_path = "C:/Users/User/Desktop/Vision Transformer/csv file/TIP/PV train (disease separated) 2L 10x PBS.csv"  # 21
test_csv_path1 = "C:/Users/User/Desktop/Vision Transformer/csv file/TIP/PV test (disease separated) 2L.csv"
test_csv_path2 = "C:/Users/User/Desktop/Vision Transformer/csv file/TIP/PV pepper bacteria spot test 2L.csv"


# Dataset declarations
train_dataset = CustomDatasetForSLandSSL_SingleA(train_csv_path, transforms1=ori_train_transforms_2, transforms2=ori_train_transforms_3)
test_dataset1 = CustomDatasetFromImagesForalbumentation(test_csv_path1, transforms=ori_valid_transforms)
test_dataset2 = CustomDatasetFromImagesForalbumentation(test_csv_path2, transforms=ori_valid_transforms)

# Dataset loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, drop_last=False)

# ViT model from Timm with ImageNet pretrained weight
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)
model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=False).to(device)


model_sl_ss = Model_SL_SSL(model, num_plant, num_disease, num_rotation, num_crop, num_flip)
model_sl_ss.to(device)
# print(model_sl_ss)

if pretrained:
    print("Using Pre-Trained Model")
    MODEL_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/saved model/Pre-trained (official)/trained/jx_vit_base_p16_224-80ecf9dd (rwightman, ImageNet21k+ImageNet2012).pth"
    model_sl_ss.model.load_state_dict(torch.load(MODEL_PATH), strict=True)

# Use this if for checkpoint or fine-tune for model
# if pretrained:
#     print("Using Pre-Trained SL+SSL model")
#     MODEL_PATH = "Your path here"
#     model_sl_ss.load_state_dict(torch.load(MODEL_PATH),strict=True)


parameters = list(model_sl_ss.parameters())
error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(parameters, lr=learning_rate_layer, weight_decay=weight_decay, momentum=momentum)
for g in optimizer.param_groups:
    g["lr"] = learning_rate_layer

# Use this if for checkpoint or fine-tune for optimizer
# if pretrained:
#     print("Using Pre-Trained optimizer")
#     OPTIMIZER_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/TIP v2/ablation/vit_base_patch16_224_4e_99.1973_98.8812_WeakAug_0.5SSL123_v8_B16_192__PV_TIP_WU-opt-S0-1.pth"
#     optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))


# Training counter
count = 0
loss_counter = 0
a1 = 0
a_d = 0

for epoch in range(num_epochs):
    print(f"Start of Epoch {epoch + 1} of {num_epochs}")
    print("Current Learning rate: {}".format(optimizer.param_groups[0]["lr"]))

    # Training counter for calculation or debug
    total_train = 0
    seen_count = 0
    unseen_count = 0
    correct_train_p = 0
    correct_train_d = 0
    ss1_correct_train = 0
    ss2_correct_train = 0
    ss3_correct_train = 0
    ss1_p_correct_train = 0
    ss2_p_correct_train = 0
    ss3_p_correct_train = 0
    ss1_d_correct_train = 0
    ss2_d_correct_train = 0
    ss3_d_correct_train = 0
    total_ss1_loss = 0
    total_ss2_loss = 0
    total_ss3_loss = 0
    total_ss1_p_loss = 0
    total_ss2_p_loss = 0
    total_ss3_p_loss = 0
    total_ss1_d_loss = 0
    total_ss2_d_loss = 0
    total_ss3_d_loss = 0
    total_slp_loss = 0
    total_sld_loss = 0
    total_com_loss = 0
    total_l1_penalty = 0
    model_sl_ss.train()

    # alpha and beta for SL and SSL regulazation
    a = 0.5
    b = 1.0
    # Batch size (using gradient accumulation method)
    iter_batch = 32

    for batch_idx, (images, labels_p, labels_d, ss1_labels, ss2_labels, ss3_labels) in enumerate(tqdm(train_loader)):
        images, labels_p, labels_d = images.to(device), labels_p.to(device), labels_d.to(device)
        ss1_labels = ss1_labels.to(device)
        ss2_labels = ss2_labels.to(device)
        ss3_labels = ss3_labels.to(device)

        batch_com_loss = 0

        # To check the image is seen and unseen class
        for x in range(len(labels_d)):
            if labels_d[x] <= (num_disease - 1):
                # Obtain the output for plant, disease and SSL predictions for seen images
                sl_op, sl_od, ss1_op, ss2_op, ss3_op, ss1_od, ss2_od, ss3_od = model_sl_ss.seen(images[x].unsqueeze(0))

                # Calculation the CrossEntrophyLoss for plant, disease and SSL tasks
                slp_l = error(sl_op, labels_p[x].unsqueeze(0))
                sld_l = error(sl_od, labels_d[x].unsqueeze(0))
                ss1_lp = error(ss1_op, ss1_labels[x].unsqueeze(0))
                ss2_lp = error(ss2_op, ss2_labels[x].unsqueeze(0))
                ss3_lp = error(ss3_op, ss3_labels[x].unsqueeze(0))
                ss1_ld = error(ss1_od, ss1_labels[x].unsqueeze(0))
                ss2_ld = error(ss2_od, ss2_labels[x].unsqueeze(0))
                ss3_ld = error(ss3_od, ss3_labels[x].unsqueeze(0))
                seen_count += 1

                # Obtain the predicted label
                h1p_predictions = torch.max(sl_op, 1)[1].to(device)
                h1d_predictions = torch.max(sl_od, 1)[1].to(device)
                h2p_predictions = torch.max(ss1_op, 1)[1].to(device)
                h3p_predictions = torch.max(ss2_op, 1)[1].to(device)
                h4p_predictions = torch.max(ss3_op, 1)[1].to(device)
                h2d_predictions = torch.max(ss1_od, 1)[1].to(device)
                h3d_predictions = torch.max(ss2_od, 1)[1].to(device)
                h4d_predictions = torch.max(ss3_od, 1)[1].to(device)

                # Calculate accuracy
                correct_train_p += (h1p_predictions == labels_p[x].unsqueeze(0)).sum()
                correct_train_d += (h1d_predictions == labels_d[x].unsqueeze(0)).sum()
                ss1_p_correct_train += (h2p_predictions == ss1_labels[x].unsqueeze(0)).sum()
                ss2_p_correct_train += (h3p_predictions == ss2_labels[x].unsqueeze(0)).sum()
                ss3_p_correct_train += (h4p_predictions == ss3_labels[x].unsqueeze(0)).sum()
                ss1_d_correct_train += (h2d_predictions == ss1_labels[x].unsqueeze(0)).sum()
                ss2_d_correct_train += (h3d_predictions == ss2_labels[x].unsqueeze(0)).sum()
                ss3_d_correct_train += (h4d_predictions == ss3_labels[x].unsqueeze(0)).sum()

                # Compute loss according to batch size
                slp_l /= batch_size
                sld_l /= batch_size
                ss1_lp /= batch_size
                ss2_lp /= batch_size
                ss3_lp /= batch_size
                ss1_ld /= batch_size
                ss2_ld /= batch_size
                ss3_ld /= batch_size
                total_slp_loss += slp_l.item()
                total_sld_loss += sld_l.item()
                total_ss1_p_loss += ss1_lp.item()
                total_ss2_p_loss += ss2_lp.item()
                total_ss3_p_loss += ss3_lp.item()
                total_ss1_d_loss += ss1_ld.item()
                total_ss2_d_loss += ss2_ld.item()
                total_ss3_d_loss += ss3_ld.item()
                com_loss = a * (slp_l + sld_l) + (a) * (ss1_lp + ss2_lp + ss3_lp) + (a) * (ss1_ld + ss2_ld + ss3_ld)
                batch_com_loss += com_loss
                total_com_loss += com_loss.item()

            else:
                # OBtain the SSL prediction for unseen images
                ss1_o, ss2_o, ss3_o = model_sl_ss.unseen(images[x].unsqueeze(0))

                unseen_count += 1

                # Calculation the CrossEntrophyLoss for SSL tasks
                ss1_l = error(ss1_o, ss1_labels[x].unsqueeze(0))
                ss2_l = error(ss2_o, ss2_labels[x].unsqueeze(0))
                ss3_l = error(ss3_o, ss3_labels[x].unsqueeze(0))

                # Obtain the predicted label
                h2_predictions = torch.max(ss1_o, 1)[1].to(device)
                h3_predictions = torch.max(ss2_o, 1)[1].to(device)
                h4_predictions = torch.max(ss3_o, 1)[1].to(device)

                # Calculate accuracy
                ss1_correct_train += (h2_predictions == ss1_labels[x].unsqueeze(0)).sum()
                ss2_correct_train += (h3_predictions == ss2_labels[x].unsqueeze(0)).sum()
                ss3_correct_train += (h4_predictions == ss3_labels[x].unsqueeze(0)).sum()

                # Compute loss according to batch size
                ss1_l /= batch_size
                ss2_l /= batch_size
                ss3_l /= batch_size
                total_ss1_loss += ss1_l.item()
                total_ss2_loss += ss2_l.item()
                total_ss3_loss += ss3_l.item()
                com_loss = b * (ss1_l + ss2_l + ss3_l)
                batch_com_loss += com_loss
                total_com_loss += com_loss.item()

        # Gradient calculation and model update
        batch_com_loss /= iter_batch
        batch_com_loss.backward()
        total_train += len(labels_p)
        if ((batch_idx + 1) % iter_batch == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        count += 1

    # Total accuracy calculations
    accuracy_train_p = correct_train_p * 100 / seen_count
    accuracy_train_d = correct_train_d * 100 / seen_count
    accuracy_ss1_p_train = (ss1_p_correct_train + ss1_correct_train) * 100 / total_train
    accuracy_ss2_p_train = (ss2_p_correct_train + ss2_correct_train) * 100 / total_train
    accuracy_ss3_p_train = (ss3_p_correct_train + ss3_correct_train) * 100 / total_train
    accuracy_ss1_d_train = (ss1_d_correct_train + ss1_correct_train) * 100 / total_train
    accuracy_ss2_d_train = (ss2_d_correct_train + ss2_correct_train) * 100 / total_train
    accuracy_ss3_d_train = (ss3_d_correct_train + ss3_correct_train) * 100 / total_train

    print(f"Total unseen images: {unseen_count}")
    print(f"SL training acc for Plant: {accuracy_train_p:.4f}")
    print(f"SL training acc for Disease: {accuracy_train_d:.4f}")
    print(f"SS1 P training acc: {accuracy_ss1_p_train:.4f}")
    print(f"SS2 P training acc: {accuracy_ss2_p_train:.4f}")
    print(f"SS3 P training acc: {accuracy_ss3_p_train:.4f}")
    print(f"SS1 D training acc: {accuracy_ss1_d_train:.4f}")
    print(f"SS2 D training acc: {accuracy_ss2_d_train:.4f}")
    print(f"SS3 D training acc: {accuracy_ss3_d_train:.4f}")

    # Total losses calcuations
    a_p = total_slp_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    a_d = total_sld_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    b = total_ss1_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    c = total_ss2_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    d = total_ss3_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    e = total_ss1_p_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    f1 = total_ss2_p_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    g = total_ss3_p_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    h = total_ss1_d_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    i = total_ss2_d_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    j = total_ss3_d_loss / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))
    k = total_l1_penalty / ((len(train_dataset) // batch_size) + (len(train_dataset) % batch_size > 0))

    print(f"SL loss for Plant: {a_p}")
    print(f"SL loss for Disease: {a_d}")
    print(f"SS1 loss: {b}")
    print(f"SS2 loss: {c}")
    print(f"SS3 loss: {d}")
    print(f"SS1 P loss: {e}")
    print(f"SS2 P loss: {f1}")
    print(f"SS3 P loss: {g}")
    print(f"SS1 D loss: {h}")
    print(f"SS2 D loss: {i}")
    print(f"SS3 D loss: {j}")
    print(f"Total SL+SSL loss: {a_p + a_d + b + c + d + e + f1 + g + h + i + j}")
    print(f"L1 loss: {k}")
    print(f"Total Loss: {a_p + a_d + b + c + d + e + f1 + g + h + i + j + k}")
    print(f"\nEpoch {epoch + 1} of {num_epochs} Done!")
    print("Current Learning rate: {0}".format(optimizer.param_groups[0]["lr"]))  # noqa: UP030

    # Seen test dataset evaluation
    print("\nSeen Testing")
    model_sl_ss.eval()
    total = 0
    correct_p_o = 0
    correct_p_t = 0
    correct_d_o = 0
    correct_d_t = 0
    total_loss_p_loss = 0
    total_loss_d_loss = 0
    correct_test_o_combine = 0
    correct_test_t_combine = 0

    for images, labels_p, labels_d in tqdm(test_loader1):
        images, labels_p, labels_d = images.to(device), labels_p.to(device), labels_d.to(device)

        # Plant and disease prediction
        outputs_p, outputs_soft_p, outputs_t_soft_p = model_sl_ss.testp(images)
        outputs_d, outputs_soft_d, outputs_t_soft_d = model_sl_ss.testd(images)

        # Calculate loss for testing set
        loss_p = error(outputs_p, labels_p)
        loss_d = error(outputs_d, labels_d)
        total_loss_p_loss += loss_p.item()
        total_loss_d_loss += loss_d.item()

        # Obtain the predicted labels
        h1_prediction_p_o = torch.max(outputs_soft_p, 1)[1].to(device)
        h1_prediction_p_t = torch.max(outputs_t_soft_p, 1)[1].to(device)
        h1_prediction_d_o = torch.max(outputs_soft_d, 1)[1].to(device)
        h1_prediction_d_t = torch.max(outputs_t_soft_d, 1)[1].to(device)

        # Calculate accuracy
        correct_p_o += (h1_prediction_p_o == labels_p).sum()
        correct_p_t += (h1_prediction_p_t == labels_p).sum()
        correct_d_o += (h1_prediction_d_o == labels_d).sum()
        correct_d_t += (h1_prediction_d_t == labels_d).sum()

        # Post-calculation for plant disease identification prediction
        for x in range(len(labels_p)):
            if h1_prediction_p_o[x] == labels_p[x] and h1_prediction_d_o[x] == labels_d[x]:
                correct_test_o_combine += 1

        for x in range(len(labels_p)):
            if h1_prediction_p_t[x] == labels_p[x] and h1_prediction_d_t[x] == labels_d[x]:
                correct_test_t_combine += 1

        total += len(labels_p)

    # Total accuracy and losses calculations
    accuracy_p_o = correct_p_o * 100 / total
    accuracy_p_t = correct_p_t * 100 / total
    accuracy_d_o = correct_d_o * 100 / total
    accuracy_d_t = correct_d_t * 100 / total
    accuracy_o = correct_test_o_combine * 100 / total
    accuracy_t = correct_test_t_combine * 100 / total
    a_p = total_loss_p_loss / ((len(test_dataset1) // batch_size) + (len(test_dataset1) % batch_size > 0))
    a_d = total_loss_d_loss / ((len(test_dataset1) // batch_size) + (len(test_dataset1) % batch_size > 0))

    # NOTE
    # acc 1 is calculaed by softmax score of plant classifer only
    # acc 2 is calculated based on summation of all softmax score from plant and SSL classifier

    print(f"Total testing Loss: {a_p + a_d}")
    print(f"Testing acc for seen Plant 1: {accuracy_p_o:.4f}")
    print(f"Testing acc for seen Disease 1: {accuracy_d_o:.4f}")
    print(f"Testing acc for seen Plant 2: {accuracy_p_t:.4f}")
    print(f"Testing acc for seen Disease 2: {accuracy_d_t:.4f}")
    print(f"Testing acc for total seen PD 1: {accuracy_o:.4f}")
    print(f"Testing acc for total seen PD 2: {accuracy_t:.4f}")
    # Unseen test dataset evaluation
    print("\nUnseen Testing")
    model_sl_ss.eval()
    total = 0
    correct_p_o = 0
    correct_p_t = 0
    correct_d_o = 0
    correct_d_t = 0
    count_unseen_p = 0
    correct_test_o_combine = 0
    correct_test_t_combine = 0

    for images, labels_p, labels_d in tqdm(test_loader2):
        images, labels_p, labels_d = images.to(device), labels_p.to(device), labels_d.to(device)

        # Plant and disease prediction
        outputs_p, outputs_soft_p, outputs_t_soft_p = model_sl_ss.testp(images)
        outputs_d, outputs_soft_d, outputs_t_soft_d = model_sl_ss.testd(images)

        # Obtain the predicted labels
        h1_prediction_p_o = torch.max(outputs_soft_p, 1)[1].to(device)
        h1_prediction_p_t = torch.max(outputs_t_soft_p, 1)[1].to(device)
        h1_prediction_d_o = torch.max(outputs_soft_d, 1)[1].to(device)
        h1_prediction_d_t = torch.max(outputs_t_soft_d, 1)[1].to(device)

        # Calculate the accuracy
        correct_p_o += (h1_prediction_p_o == labels_p).sum()
        correct_p_t += (h1_prediction_p_t == labels_p).sum()
        correct_d_o += (h1_prediction_d_o == labels_d).sum()
        correct_d_t += (h1_prediction_d_t == labels_d).sum()

        # Post-calculation for plant disease identification prediction
        for x in range(len(labels_p)):
            if h1_prediction_p_o[x] == labels_p[x] and h1_prediction_d_o[x] == labels_d[x]:
                correct_test_o_combine += 1

        for x in range(len(labels_p)):
            if h1_prediction_p_t[x] == labels_p[x] and h1_prediction_d_t[x] == labels_d[x]:
                correct_test_t_combine += 1

        total += len(labels_p)

    # Total accuracy and losses calculations
    accuracy_p_o = correct_p_o * 100 / total
    accuracy_p_t = correct_p_t * 100 / total
    accuracy_d_o_1 = correct_d_o * 100 / total
    accuracy_d_t_1 = correct_d_t * 100 / total
    accuracy_o = correct_test_o_combine * 100 / total
    accuracy_t = correct_test_t_combine * 100 / total

    # NOTE
    # acc 1 is calculaed by softmax score of plant classifer only
    # acc 2 is calculated based on summation of all softmax score from plant and SSL classifier

    print(f"Testing acc for PBS Plant 1: {accuracy_p_o:.4f}")
    print(f"Testing acc for PBS Disease 1: {accuracy_d_o_1:.4f}")
    print(f"Testing acc for PBS Plant 2: {accuracy_p_t:.4f}")
    print(f"Testing acc for PBS Disease 2: {accuracy_d_t_1:.4f}")
    print(f"Testing acc for total unseen PD 1: {accuracy_o:.4f}")
    print(f"Testing acc for total unseen PD 2: {accuracy_t:.4f}")

    # Model and optimizer saving
    if (epoch + 1) % 1 == 0:
        print("Saving Model")
        torch.save(model_sl_ss.state_dict(), os.path.join(Save_model_path, f"{model_name}_{epoch + 1}e_{accuracy_train_d:.4f}_{accuracy_d_o:.4f}_model.pth"))
        print("Saving optimizer")
        torch.save(optimizer.state_dict(), os.path.join(Save_model_path, f"{model_name}_{epoch + 1}e_{accuracy_train_d:.4f}_{accuracy_d_o:.4f}_opt.pth"))
        print("Saving done")


print("Training done")
