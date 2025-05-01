import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# basic

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# maianet
maianet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to input size of MaiaNet
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flipping
    transforms.RandomVerticalFlip(p=0.5),  # Vertical flipping
    transforms.ToTensor(),  # Convert to tensor before adding noise
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
    transforms.Lambda(lambda x: transforms.functional.erase(x, i=0, j=0, h=50, w=50, v=0.0)),  # Add cutout
])

tswin_train_transform = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    interpolation="bicubic",
)

tswin_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to input size of MaiaNet
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


def prepare_dataloaders(path, transform, batch_size=32, num_workers=2, val_split=0.2, test_split=0.1):
    """
    Prepares train, validation, and test dataloaders from an ImageFolder dataset.

    Args:
        path (str): Root directory path to the dataset.
        transform (callable): Transform to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader or None)
    """
    full_dataset = datasets.ImageFolder(root=path, transform=transform)

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    if test_split > 0:
        train_set, val_set, test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    else:
        train_set, val_set = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        test_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader
