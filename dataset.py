import os

from PIL import Image
from sklearn.model_selection import train_test_split
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_transform(is_train):
    if is_train:
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
        )
        return transform

    t = [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    return transforms.Compose(t)


class Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
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


def build_dataset(df, path, transform=build_transform(True), test=False):
    """path = "/workspace/data/images" ,"""

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    if test:
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42, stratify=val_df["label"])
        test_dataset = Dataset(test_df, path, transform)

    else:
        test_dataset = None

    train_dataset = Dataset(train_df, path, transform)
    val_dataset = Dataset(val_df, path, transform)

    return train_dataset, test_dataset, val_dataset


def build_loader(train_dataset, val_dataset, batch_size, test_dataset=None):
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader
