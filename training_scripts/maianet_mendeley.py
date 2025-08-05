import os
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from dataset import maianet_transform, prepare_dataloaders
from MAIAnet.maianet import MaiaNet
from MAIAnet.train import Trainer

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Training configuration
batch_size = 16
lr = 2e-4
num_epochs = 35
num_classes = 11
percentages = [0.1, 0.2, 0.3, 0.4, 0.5]

# Load full dataset
train_loader_full, val_loader, test_loader = prepare_dataloaders(
    "data3/mendeley",
    maianet_transform,
    batch_size=16,
    num_workers=4,
    val_split=0.1,
    test_split=0.1,
)

# Get the full training dataset
train_dataset = train_loader_full.dataset

# Loop over each percentage of labeled data
for pct in percentages:
    pct_int = int(pct * 100)
    log_filename = f"logs/maianet_mendeley_{pct_int}pct.log"

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    print(f"\n=== Training with {pct_int}% of labeled training data ===\n")

    # Get subset of training dataset
    num_samples = int(len(train_dataset) * pct)
    indices = torch.randperm(len(train_dataset))[:num_samples]
    subset = Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Initialize model and trainer
    model = MaiaNet(num_classes)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        lr,
        num_epochs,
        batch_size=batch_size,
    )

    # Train model
    trainer.train()

    # Close log and reset stdout
    sys.stdout.close()
