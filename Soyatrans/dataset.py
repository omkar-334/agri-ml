import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to input size of MaiaNet
    # transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flipping
    # transforms.RandomVerticalFlip(p=0.5),  # Vertical flipping
    transforms.ToTensor(),  # Convert to tensor before adding noise
    # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
    # transforms.Lambda(lambda x: transforms.functional.erase(x, i=0, j=0, h=50, w=50, v=0.0)),  # Add cutout
])


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
