from copy import deepcopy

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.GaussianBlur(3), transforms.RandomAffine(degrees=20)], p=0.5),
    transforms.RandomApply([transforms.RandomErasing(p=0.5)], p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_zero_lbp():
    return torch.zeros(26)  # adjust if needed


class UnlabeledWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform or dataset.dataset.transform

    def __getitem__(self, idx):
        img_path, _ = self.dataset.dataset.samples[self.dataset.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, get_zero_lbp()

    def __len__(self):
        return len(self.dataset)


def get_mean_teacher_dataloaders(labeled_data_dir, labeled_ratio=0.2, batch_size=16):
    base_dataset = ImageFolder(labeled_data_dir, transform=basic_transform)

    total_size = len(base_dataset)
    labeled_size = int(labeled_ratio * total_size)
    unlabeled_size = total_size - labeled_size

    labeled_dataset, unlabeled_dataset = random_split(base_dataset, [labeled_size, unlabeled_size])

    # Split labeled into train/test (you'll use test as val)
    train_size = int(0.8 * labeled_size)
    test_size = labeled_size - train_size
    train_dataset, test_dataset = random_split(labeled_dataset, [train_size, test_size])

    # Augment the labeled training dataset
    train_dataset.dataset.transform = basic_transform
    augmented_train_dataset = deepcopy(train_dataset)
    augmented_train_dataset.dataset.transform = augmented_transform

    combined_train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])

    # Prepare unlabeled datasets for teacher and student
    unlabeled_teacher = deepcopy(unlabeled_dataset)
    unlabeled_teacher.dataset.transform = basic_transform

    unlabeled_student = deepcopy(unlabeled_dataset)
    unlabeled_student.dataset.transform = augmented_transform

    args = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }

    train_loader = DataLoader(combined_train_dataset, drop_last=True, **args)
    test_loader = DataLoader(test_dataset, **args)
    unlabeled_loader = DataLoader(UnlabeledWrapper(unlabeled_teacher), drop_last=True, **args)
    unlabeled_student_loader = DataLoader(UnlabeledWrapper(unlabeled_student), drop_last=True, **args)

    return train_loader, test_loader, unlabeled_loader, unlabeled_student_loader
