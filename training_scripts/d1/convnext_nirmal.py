import os
import sys

import numpy as np
import torch

from ConvNext.convnext import Autoencoder
from ConvNext.dataset import get_dataloaders
from ConvNext.train import train_model, validate_model

torch.manual_seed(42)
np.random.seed(42)

# Create logs directory
os.makedirs("logs", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5

for labeled_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    log_filename = f"logs/convnext_nirmal_{int(labeled_ratio * 100)}pct.log"

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    print(
        f"\n=== Training with {int(labeled_ratio * 100)}% of labeled training data ===\n"
    )

    # Get dataloaders
    train_loader_labeled, val_loader_labeled, train_loader_unlabeled = get_dataloaders(
        "data2/Dataset",
        16,
        labeled_ratio,
    )

    model = Autoencoder(NUM_CLASSES).to(DEVICE)

    model = train_model(
        model,
        train_loader_labeled,
        train_loader_unlabeled,
        lr=1e-4,
        num_epochs=35,
    )

    validate_model(model, val_loader_labeled, NUM_CLASSES)
