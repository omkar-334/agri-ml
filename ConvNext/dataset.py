from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


class UnlabeledWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def get_dataloaders(path, batch_size=32, labeled_ratio=0.2):
    full_dataset = ImageFolder(path, transform=transform)

    total_size = len(full_dataset)

    # 20% labeled, 80% unlabeled
    labeled_size = int(labeled_ratio * total_size)
    unlabeled_size = total_size - labeled_size

    labeled_dataset, unlabeled_dataset = random_split(full_dataset, [labeled_size, unlabeled_size])

    # Further split the labeled data into train and validation (e.g., 80% train, 20% validation)
    labeled_train_size = int(0.8 * labeled_size)
    labeled_val_size = labeled_size - labeled_train_size
    labeled_train_dataset, labeled_val_dataset = random_split(labeled_dataset, [labeled_train_size, labeled_val_size])

    # Create the DataLoaders
    train_loader_labeled = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_labeled = DataLoader(labeled_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    train_loader_unlabeled = DataLoader(UnlabeledWrapper(unlabeled_dataset), batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader_labeled, val_loader_labeled, train_loader_unlabeled
