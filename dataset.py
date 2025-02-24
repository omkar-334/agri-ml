# Dataset - https://www.kaggle.com/competitions/cassava-leaf-disease-classification/
# The original Uganda cassava leaf disease dataset comprises 21,393
# images. However, the initial Uganda cassava dataset lacked balance in
# its distribution across various categories. The most imbalanced categories,
# the CMD and CBB disease datasets, contained 13,158 and 1,086
# images, respectively. An imbalanced data distribution causes various
# degrees of long-tail phenomena in feature extraction and class prediction
# (Chen et al., 2022). To overcome the long-tail obstacle for features
# and converge the neural network quickly, data augmentation was used
# to maintain balance in the training dataset to solve data management issues.
# To maintain balance among the categories, augmentation techniques
# such as Gaussian noise, horizontal flipping, cutout, and vertical flipping
# were employed. The 20,000 color images were randomly combined into
# five balanced categories, and the CMD category was randomly selected
# from 13,158 images in the raw data. Notably, CMD data were selected
# without augmentation. Following preprocessing, the images had a resolution
# of 448 × 448, as outlined in Table 2.


# https://www.kaggle.com/code/omkar334/maianet-cassava-leaf-disease/edit

import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),  # Resize to input size of MaiaNet
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flipping
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flipping
        transforms.ToTensor(),  # Convert to tensor before adding noise
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
        transforms.Lambda(lambda x: transforms.functional.erase(x, i=0, j=0, h=50, w=50, v=0.0)),  # Add cutout
    ]
)

# df = pd.read_csv("train.csv")
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

print(df.label.value_counts())
balanced_df = pd.DataFrame()

for label in df["label"].unique():
    label_df = df[df["label"] == label]
    if len(label_df) > 1000:
        _, sampled_df = train_test_split(label_df, test_size=1000, random_state=42, stratify=label_df["label"])
    balanced_df = pd.concat([balanced_df, sampled_df])


class Dataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]["image_id"]
        label = self.dataframe.iloc[idx]["label"]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])


train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)
val_dataset = Dataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
