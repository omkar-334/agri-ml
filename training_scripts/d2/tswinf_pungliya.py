import os
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from dataset import prepare_dataloaders, tswin_train_transform
from Tswin.train import main

torch.manual_seed(42)
np.random.seed(42)

# Create logs directory
os.makedirs("logs", exist_ok=True)

NUM_CLASSES = 3

train_loader_full, val_loader, test_loader = prepare_dataloaders(
    "data2/Dataset",
    tswin_train_transform,
    batch_size=16,
    num_workers=4,
    val_split=0.2,
    test_split=0,
)

train_dataset = train_loader_full.dataset

# Loop over each percentage of labeled data

percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
for pct in percentages:
    pct_int = int(pct * 100)
    log_filename = f"logs/tswinf_pungliya_{pct_int}pct.log"

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    print(f"\n=== Training with {pct_int}% of labeled training data ===\n")

    num_samples = int(len(train_dataset) * pct)
    indices = torch.randperm(len(train_dataset))[:num_samples]
    subset = Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        subset, batch_size=16, shuffle=True, num_workers=4
    )

    model = main(train_loader, val_loader, NUM_CLASSES, "models/tswinf_pungliya.pth")
